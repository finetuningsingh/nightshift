"""Fakes for the GPU side of the Glimmer harness (owner 2): FakeProvider, FakeSsh, FakeTrainer.

* ``FakeProvider`` implements ``GpuProvider`` in memory (optionally persisted to a JSON file so a detached watchdog PROCESS and the test
  see one account). It carries the failure injectors of docs notes section 2.4: a foreign instance of the same shape, list and delete
  failures, ambiguous / rejected / handle-less creates, a sold-out GPU, and rows that stay PENDING for a few listings.
* ``FakeSsh`` puts an ``ssh`` executable first on ``PATH``. The "rental" is the local machine: the remote command runs under
  ``bash -c`` (so tests use absolute paths inside a temp directory as remote paths), ``-N -L`` becomes a real TCP forwarder, and
  RULES intercept commands: a rule answers with canned bytes (a hostile tar stream, for example) or runs a stub script. Every call
  is logged to ``calls.jsonl``.
* ``FakeTrainer`` implements ``Trainer`` for the trial runner's tests: it writes a real adapter under the real layout, a real
  train-data file and a real dev ``ScoreRecord`` (request.json and score.json), and charges the meter like the real trainer.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import itertools
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from nightshift.lab.contracts import (
    AdapterFile,
    AdapterRef,
    BudgetExhausted,
    ContractError,
    CreateAmbiguous,
    CreateReceipt,
    CreateRejected,
    FileRef,
    GenerationParams,
    GpuInstance,
    GpuUsage,
    JobStatus,
    LabLayout,
    OwnershipError,
    ProviderUnavailable,
    Quote,
    Resource,
    ScoreRecord,
    SeedAggregate,
    Shape,
    TrainJob,
    TrainResult,
    Verifier,
    adapter_ref_from_dir,
    owner_tag_of_key,
    sha256_file,
    utc_iso,
    write_record,
    METRIC_PROMPT_ALL,
    METRIC_PROMPT_FRACTION,
)

FAKE_KEY_PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFAKEKEYFORTESTSONLYxxxxxxxxxxxxxxxxxxxxxxxx lab-test"


def fake_key_factory(session_dir: Path) -> str:
    """A key_factory for RentalManager that writes no real key (ssh-keygen not needed): one distinct fake public key per session."""
    session_dir.mkdir(parents=True, exist_ok=True)
    tag = session_dir.name.encode().hex()[:24].ljust(24, "0")
    pub = f"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI{tag}FAKE lab-{session_dir.name}"
    fd = os.open(session_dir / "id_ed25519", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----\n")
    (session_dir / "id_ed25519.pub").write_text(pub + "\n")
    return pub


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> float:
        with self._lock:
            self.now += seconds
            return self.now


def _inst_to_dict(i: GpuInstance) -> dict:
    return {
        "handle": i.handle, "status": i.status, "shape": None if i.shape is None else i.shape.to_dict(),
        "created_epoch": i.created_epoch, "owner_tag": i.owner_tag, "host": i.host, "port": i.port, "user": i.user,
    }  # fmt: skip


def _inst_from_dict(d: dict) -> GpuInstance:
    return GpuInstance(
        d["handle"], d["status"], None if d["shape"] is None else Shape.from_dict(d["shape"]), d["created_epoch"],
        d["owner_tag"], d["host"], d["port"], d["user"],
    )  # fmt: skip


class FakeProvider:
    """In-memory GpuProvider. With ``state_path`` the account lives in a JSON file under a flock (for cross-process tests)."""

    name = "fake"

    def __init__(self, clock: Callable[[], float] = None, *, rate: float = 1.31, state_path: Optional[Path] = None) -> None:
        self.clock = clock or FakeClock()
        self.rate = rate
        self.state_path = None if state_path is None else Path(state_path)
        self.instances: dict[str, GpuInstance] = {}
        self.foreign: set[str] = set()
        self.calls: list[tuple] = []
        self.fail_list_n = 0
        self.fail_delete_n = 0
        self.create_mode = "ok"  # ok | reject | ambiguous_created | ambiguous_not_created | no_handle
        self.sold_out: set[str] = set()
        self.available: Optional[bool] = True
        self.pending_lists = 0  # a freshly created row reports PENDING (no host) for this many listings
        self.fixed_usd = 0.0
        self.on_create: Optional[Callable[[], None]] = None  # called inside create (before the row exists)
        self._n = itertools.count(1)
        self._pending: dict[str, int] = {}
        self._lock = threading.RLock()

    # persistence ---------------------------------------------------------------------------------
    @contextlib.contextmanager
    def _state(self):
        with self._lock:
            if self.state_path is None:
                yield
                return
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(self.state_path) + ".lock", "a+") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                try:
                    if self.state_path.exists():
                        doc = json.loads(self.state_path.read_text())
                        self.instances = {h: _inst_from_dict(d) for h, d in doc["instances"].items()}
                        self.foreign = set(doc["foreign"])
                    yield
                    self.state_path.write_text(
                        json.dumps({"instances": {h: _inst_to_dict(i) for h, i in self.instances.items()}, "foreign": sorted(self.foreign)})
                    )
                finally:
                    fcntl.flock(lk, fcntl.LOCK_UN)

    # helpers for tests ---------------------------------------------------------------------------
    def add_foreign(self, shape: Shape, key_tag: str = "someone-else") -> str:
        with self._state():
            h = f"foreign{next(self._n):03d}"
            self.instances[h] = GpuInstance(h, "RUNNING", shape, self.clock(), key_tag, "10.0.0.1", 22)
            self.foreign.add(h)
            return h

    def handles(self) -> set[str]:
        with self._state():
            return set(self.instances)

    def calls_of(self, kind: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == kind]

    def _make(self, shape: Shape, tag: str) -> str:
        h = f"fake{next(self._n):04d}x"
        self.instances[h] = GpuInstance(h, "RUNNING", shape, self.clock(), tag, "127.0.0.1", 2200)
        if self.pending_lists:
            self._pending[h] = self.pending_lists
        return h

    # GpuProvider ---------------------------------------------------------------------------------
    def quote(self, shape: Shape) -> Quote:
        self.calls.append(("quote", shape.gpu))
        avail = None if self.available is None else (self.available and shape.gpu not in self.sold_out)
        return Quote(self.rate, self.fixed_usd, avail, "fake")

    def list_instances(self) -> list[GpuInstance]:
        with self._state():
            self.calls.append(("list",))
            if self.fail_list_n:
                self.fail_list_n -= 1
                raise ProviderUnavailable("fake list failure")
            out = []
            for i in self.instances.values():
                if self._pending.get(i.handle, 0) > 0:
                    self._pending[i.handle] -= 1
                    out.append(GpuInstance(i.handle, "PENDING", i.shape, i.created_epoch, i.owner_tag, None, None, i.user))
                else:
                    out.append(i)
            return out

    def create(self, shape: Shape, ssh_public_key: str) -> CreateReceipt:
        tag = owner_tag_of_key(ssh_public_key)
        with self._state():
            self.calls.append(("create", shape.gpu, tag))
            if self.on_create:
                self.on_create()
            if shape.gpu in self.sold_out or self.create_mode == "reject":
                raise CreateRejected("no capacity")
            if self.create_mode == "ambiguous_not_created":
                raise CreateAmbiguous("timeout")
            h = self._make(shape, tag)
            if self.create_mode == "ambiguous_created":
                raise CreateAmbiguous("timeout after the request was accepted")
            return CreateReceipt(None if self.create_mode == "no_handle" else h, self.clock())

    def find_created(self, before: set[str], shape: Shape, attempt_epoch: float, owner_tag: str) -> Optional[str]:
        new = [
            i for i in self.list_instances()
            if i.handle not in before and i.owner_tag == owner_tag and i.shape == shape and i.created_epoch >= attempt_epoch - 120
        ]  # fmt: skip
        return new[0].handle if len(new) == 1 else None

    def delete(self, handle: str, expect_tag: Optional[str]) -> None:
        with self._state():
            self.calls.append(("delete", handle))
            inst = self.instances.get(handle)
            if inst is None:
                return
            if expect_tag is not None and inst.owner_tag != expect_tag:
                self.calls.pop()
                raise OwnershipError(f"{handle} is not owned by tag {expect_tag!r}; refusing")
            if self.fail_delete_n:
                self.fail_delete_n -= 1
                raise ProviderUnavailable("fake delete failure")
            del self.instances[handle]

    def exists(self, handle: str) -> Optional[bool]:
        try:
            return handle in {i.handle for i in self.list_instances()}
        except ProviderUnavailable:
            return None


class FakeWatchdogLauncher:
    """A launcher for RentalManager that starts no process: it holds the per-pid advisory lock and writes watchdog.json like the real one."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail = False  # a watchdog that never arms
        self._locks: dict[int, Any] = {}
        self._pids = itertools.count(900_000)

    def __call__(self, session_dir: Path, session: str, ledger: Path, at_epoch: int) -> None:
        self.calls.append((session, int(at_epoch)))
        if self.fail:
            return
        pid = next(self._pids)
        fh = open(session_dir / f"watchdog.{pid}.lock", "a+")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._locks[pid] = fh
        (session_dir / "watchdog.json").write_text(json.dumps({"pid": pid, "delete_at_epoch": int(at_epoch), "session": session}))

    def stop(self, pid: int) -> None:
        fh = self._locks.pop(pid, None)
        if fh is not None:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()

    def close(self) -> None:
        for pid in list(self._locks):
            self.stop(pid)


# ----------------------------------------------------------------------------------------------------
# FakeSsh
# ----------------------------------------------------------------------------------------------------
_FAKE_SSH = r'''#!%(python)s
import base64, json, os, socket, subprocess, sys, threading
CONF = %(conf)r
OPT_ARG = set("-i -p -o -F -l -L -R -J -b -c -D -E -e -I -m -O -Q -S -W -w".split())
argv = sys.argv[1:]
opts, pos, forward, i = [], [], None, 0
while i < len(argv):
    a = argv[i]
    if a in OPT_ARG:
        opts.append((a, argv[i + 1])); i += 2; continue
    if a.startswith("-") and not pos:
        opts.append((a, None)); i += 1; continue
    pos.append(a); i += 1
host, cmd = (pos[0] if pos else ""), " ".join(pos[1:])
conf = json.load(open(CONF))
for flag, val in opts:
    if flag == "-L": forward = val
def log(entry):
    with open(conf["log"], "a") as fh:
        fh.write(json.dumps(entry) + "\n")
if forward is not None:
    log({"forward": forward, "host": host, "opts": [o for o in opts if o[0] != "-L"]})
    if conf.get("forward_fail"):
        sys.stderr.write("bind: Address already in use\n"); sys.exit(255)
    _, lp, _, rp = forward.split(":")
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", int(lp))); srv.listen(16)
    def pump(a, b):
        try:
            while True:
                d = a.recv(65536)
                if not d: break
                b.sendall(d)
        except OSError: pass
        finally:
            for s in (a, b):
                try: s.shutdown(socket.SHUT_RDWR)
                except OSError: pass
    while True:
        c, _ = srv.accept()
        try: r = socket.create_connection(("127.0.0.1", int(rp)))
        except OSError: c.close(); continue
        threading.Thread(target=pump, args=(c, r), daemon=True).start()
        threading.Thread(target=pump, args=(r, c), daemon=True).start()
stdin = b"" if sys.stdin.closed else sys.stdin.buffer.read()
log({"host": host, "cmd": cmd, "stdin_len": len(stdin), "opts": opts})
for rule in conf["rules"]:
    if rule["match"] in cmd:
        if "stdout_b64" in rule:
            sys.stdout.buffer.write(base64.b64decode(rule["stdout_b64"])); sys.stdout.buffer.flush()
            sys.stderr.write(rule.get("stderr", "")); sys.exit(rule.get("rc", 0))
        if "script" in rule:
            p = subprocess.run([rule["script"], cmd], input=stdin, capture_output=True, env=dict(os.environ, FAKE_SSH_CMD=cmd))
            sys.stdout.buffer.write(p.stdout); sys.stderr.buffer.write(p.stderr); sys.exit(p.returncode)
        if "rc" in rule:
            sys.stderr.write(rule.get("stderr", "")); sys.exit(rule["rc"])
p = subprocess.run(["bash", "-c", cmd], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.buffer.write(p.stdout); sys.stderr.buffer.write(p.stderr); sys.exit(p.returncode)
'''


class FakeSsh:
    """``with FakeSsh(tmp) as fs:`` puts a fake ``ssh`` first on PATH (and removes it again). ``fs.rules`` intercept commands."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.bin = self.dir / "fakebin"
        self.conf = self.dir / "fake_ssh.json"
        self.log = self.dir / "calls.jsonl"
        self.rules: list[dict] = []
        self.forward_fail = False
        self._old_path: Optional[str] = None

    def _write(self) -> None:
        self.conf.write_text(json.dumps({"rules": self.rules, "log": str(self.log), "forward_fail": self.forward_fail}))

    def add_rule(self, match: str, *, stdout: Optional[bytes] = None, rc: int = 0, stderr: str = "", script: Optional[str] = None) -> None:
        rule: dict[str, Any] = {"match": match}
        if stdout is not None:
            rule.update(stdout_b64=base64.b64encode(stdout).decode(), rc=rc, stderr=stderr)
        elif script is not None:
            rule["script"] = script
        else:
            rule.update(rc=rc, stderr=stderr)
        self.rules.insert(0, rule)
        self._write()

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def commands(self) -> list[str]:
        return [c["cmd"] for c in self.calls() if "cmd" in c]

    def __enter__(self) -> "FakeSsh":
        self.bin.mkdir(parents=True, exist_ok=True)
        exe = self.bin / "ssh"
        exe.write_text(_FAKE_SSH % {"python": sys.executable, "conf": str(self.conf)})
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        self._write()
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.bin}{os.pathsep}{self._old_path}"
        return self

    def __exit__(self, *exc) -> None:
        os.environ["PATH"] = self._old_path or ""


# ----------------------------------------------------------------------------------------------------
# FakeTrainer
# ----------------------------------------------------------------------------------------------------
class FakeTrainer:
    """Trainer double. ``run_job`` writes a real adapter under ``layout.adapter_dir``, a real train-data file, a real dev ScoreRecord
    (``scores/<id>/request.json`` and ``score.json``) and charges the meter ``gpu_minutes`` like the real trainer's ticks."""

    def __init__(self, layout: LabLayout, *, clock: Callable[[], float], dev_score: float = 0.5, gpu_minutes: float = 10.0, usd_per_hour: float = 1.31) -> None:
        self.layout, self.clock = layout, clock
        self.dev_score, self.gpu_minutes, self.usd_per_hour = dev_score, gpu_minutes, usd_per_hour
        self.jobs: list[TrainJob] = []
        self.fail_with: Optional[JobStatus] = None
        self.scores: dict[str, float] = {}  # job_id -> dev score override
        self._n = itertools.count(1)

    def run_job(self, job: TrainJob, meter) -> TrainResult:
        self.jobs.append(job)
        n = next(self._n)
        minutes = min(self.gpu_minutes, job.max_gpu_minutes)
        usage = GpuUsage(minutes, round(minutes * self.usd_per_hour / 60, 6), (f"rent-{job.job_id}",))
        try:
            meter.charge(Resource.GPU_MINUTES, minutes, f"gpu:rent-{job.job_id}:1", {"fake": True})
            meter.charge(Resource.GPU_USD, usage.usd, f"gpu:rent-{job.job_id}:1:usd", {"fake": True})
        except BudgetExhausted as exc:
            return TrainResult(job.job_id, job.trial_id, JobStatus.BUDGET_EXHAUSTED, None, None, None, usage, {}, None, None, f"budget exhausted: {exc}")
        if self.fail_with is not None:
            return TrainResult(job.job_id, job.trial_id, self.fail_with, None, None, None, usage, {}, None, None, f"fake {self.fail_with.value}")
        adapter_id = "ad-" + job.job_id.split("-", 1)[-1] if job.job_id.startswith("j-") else f"ad-{job.job_id}"
        adir = self.layout.adapter_dir(adapter_id)
        adir.mkdir(parents=True, exist_ok=False)
        (adir / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 32, "fake_job": job.job_id}))
        (adir / "adapter_model.safetensors").write_bytes(f"fake weights {job.job_id} {n}".encode())
        ref = adapter_ref_from_dir(adir, adapter_id=adapter_id)
        write_record(self.layout.adapter_json_path(adapter_id), ref)
        td = self.layout.trial_train_data_path(job.trial_id, job.job_id)
        td.parent.mkdir(parents=True, exist_ok=True)
        td.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": f"fake training row {job.job_id}"}]}) + "\n")
        score = self.scores.get(job.job_id, self.dev_score)
        self.write_dev_score(job, ref, score)
        return TrainResult(
            job.job_id, job.trial_id, JobStatus.OK, adapter_id, ref, job.dev_request.request_id, usage, {"train_s": 1.0}, None,
            self.layout.file_ref(td), None,
        )  # fmt: skip

    def write_dev_score(self, job: TrainJob, ref: AdapterRef, score: float) -> ScoreRecord:
        req = job.dev_request.with_adapter(job.dev_request.request_id, ref)
        sid = req.request_id
        pp = self.layout.per_prompt_path(sid)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text('{"prompt_key":"k1","seed":0,"passed":true,"fraction":1.0,"constraints":[],"truncated":false,"error":false}\n')
        seeds = tuple(SeedAggregate(s, 10, {METRIC_PROMPT_ALL: score, METRIC_PROMPT_FRACTION: score}, 0, 0) for s in req.seeds)
        rec = ScoreRecord(
            sid, sid, req.phase, req.assignment_id, req.split, req.trial_id, req.arm, req.final_request_id, req.base, ref.sha256,
            req.scorer_version, Verifier.IFEVALG, {"ifevalg": "fake"}, req.params, seeds, 10, self.layout.file_ref(pp),
            METRIC_PROMPT_ALL, utc_iso(self.clock()),
        )  # fmt: skip
        write_record(self.layout.score_request_path(sid), req)
        write_record(self.layout.score_record_path(sid), rec)
        return rec


def make_tar_bytes(members: list[tuple]) -> bytes:
    """Build a tar stream for hostile-input tests. members: (name, kind, payload) with kind in file | symlink | hardlink | chardev |
    dir | fifo; payload is bytes for a file, the link target for links, ``(declared_size, bytes)`` for a lying header."""
    import io
    import tarfile

    buf = io.BytesIO()
    if len(members) == 1 and members[0][1] == "bigheader":  # header claims `declared` bytes; the stream carries only `body`
        name, _, (declared, body) = members[0]
        info = tarfile.TarInfo(name)
        info.size = declared
        buf.write(info.tobuf(tarfile.GNU_FORMAT))
        buf.write(body)
        return buf.getvalue()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, kind, payload in members:
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            if kind == "file":
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, payload
                tar.addfile(info)
            elif kind == "hardlink":
                info.type, info.linkname = tarfile.LNKTYPE, payload
                tar.addfile(info)
            elif kind == "chardev":
                info.type, info.devmajor, info.devminor = tarfile.CHRTYPE, 1, 3
                tar.addfile(info)
            elif kind == "cont":  # a "contiguous file" or sparse-style type that is neither a link nor a device
                info.type, info.size = tarfile.CONTTYPE, len(payload)
                tar.addfile(info, io.BytesIO(payload))
            elif kind == "dir":
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                tar.addfile(info)
            else:
                raise ValueError(kind)
    return buf.getvalue()


def file_provider_factory() -> FakeProvider:
    """``--provider-factory tests.support.lab_fake_gpu:file_provider_factory``: a FakeProvider whose account is the JSON file named by
    ``LAB_FAKE_GPU_STATE``, so a detached watchdog PROCESS and the test see the same instances (real clock)."""
    import time

    return FakeProvider(time.time, state_path=Path(os.environ["LAB_FAKE_GPU_STATE"]))


# ----------------------------------------------------------------------------------------------------
# doubles for the trainer / model-server tests
# ----------------------------------------------------------------------------------------------------
class FakeMeter:
    """BudgetMeter double: sums charges per call id (idempotent), raises BudgetExhausted AFTER the charge that crosses a cap."""

    def __init__(self, spec, clock: Callable[[], float] = None) -> None:
        from nightshift.lab.contracts import AXIS_OF, BudgetUsage

        self._spec, self._axis_of, self._usage_cls = spec, AXIS_OF, BudgetUsage
        self.charges: dict = {}
        self._lock = threading.Lock()

    @property
    def spec(self):
        return self._spec

    def usage(self):
        with self._lock:
            tot = {r: 0.0 for r in ("gpu_minutes", "researcher_usd", "gpu_usd", "researcher_tokens")}
            for (resource, amount) in self.charges.values():
                tot[resource.value] = tot.get(resource.value, 0.0) + amount
            return self._usage_cls(int(tot["researcher_tokens"]), tot["gpu_minutes"], tot["researcher_usd"], tot["gpu_usd"], 0.0)

    def _exhausted(self, axis: str) -> bool:
        u = self.usage()
        return axis in u.exhausted_axes(self._spec)

    def check(self, resource) -> None:
        axis = self._axis_of[resource]
        if self._exhausted(axis):
            raise BudgetExhausted(axis, getattr(self._spec, axis), 0.0)

    def charge(self, resource, amount: float, call_id: str, detail=None):
        with self._lock:
            if call_id not in self.charges:
                self.charges[call_id] = (resource, float(amount))
        u = self.usage()
        axis = self._axis_of[resource]
        if axis in u.exhausted_axes(self._spec):
            raise BudgetExhausted(axis, getattr(self._spec, axis), getattr(u, axis, 0.0))
        return u

    def reserve_llm(self, model, prompt_tokens_estimate, max_tokens, call_id) -> None:  # pragma: no cover - unused by the GPU side
        pass

    def charge_llm(self, model, usage, call_id):  # pragma: no cover
        return self.usage()


class FakeScorer:
    """Scorer double: writes a real dev ScoreRecord through the real layout (like FakeTrainer) and records what it was asked."""

    scorer_version = "sv1-fakefakefakefake"

    def __init__(self, layout: LabLayout, clock: Callable[[], float], score: float = 0.5) -> None:
        self.layout, self.clock, self.score_value = layout, clock, score
        self.calls: list = []
        self.error: Optional[BaseException] = None
        self._helper = FakeTrainer(layout, clock=clock, dev_score=score)

    def score(self, request, generator, model):
        self.calls.append((request, generator, model))
        if self.error is not None:
            raise self.error
        job_like = type("J", (), {"dev_request": request})
        return self._helper.write_dev_score(job_like, request.adapter, self.score_value)  # type: ignore[arg-type]


FAKE_VLLM = r'''#!%(python)s
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
models = [opt("--served-model-name", "base")]
if "--lora-modules" in args:
    i = args.index("--lora-modules") + 1
    while i < len(args) and not args[i].startswith("--"):
        models.append(args[i].split("=", 1)[0]); i += 1
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": m} for m in models]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        body = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
open(opt("--port") and sys.argv[0] + ".argv", "w").write(json.dumps(args))
ThreadingHTTPServer(("127.0.0.1", int(opt("--port"))), H).serve_forever()
'''

FAKE_AXOLOTL_PYTHON = r'''#!%(python)s
# stands in for $AXOLOTL_VENV/bin/python running lib/axolotl_train.py: writes a (tiny) adapter the way PEFT would
import json, os, struct, sys
out = os.path.join(os.environ["GLIMMER_OUT"], "adapter")
os.makedirs(os.path.join(out, "checkpoint-3"), exist_ok=True)
mode = os.environ.get("FAKE_ADAPTER_MODE", "good")
cfg = {"peft_type": "LORA", "r": 64 if mode == "bad_rank" else 32, "lora_alpha": 16, "target_modules": "q_proj"}
name = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight" if mode != "not_lora" else "model.embed_tokens.weight"
data = b"\x00" * 64
header = json.dumps({name: {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64 if mode != "oob" else 6400]}}).encode()
open(os.path.join(out, "adapter_config.json"), "w").write(json.dumps(cfg))
open(os.path.join(out, "adapter_model.safetensors"), "wb").write(struct.pack("<Q", len(header)) + header + data)
open(os.path.join(out, "checkpoint-3", "optimizer.pt"), "wb").write(b"junk")
open(os.path.join(out, "README.md"), "w").write("peft readme")
'''


def install_fake_vllm(venv_dir: Path) -> Path:
    exe = Path(venv_dir) / "bin" / "vllm"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(FAKE_VLLM % {"python": sys.executable})
    exe.chmod(0o755)
    return exe


def install_fake_axolotl(venv_dir: Path) -> Path:
    exe = Path(venv_dir) / "bin" / "python"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(FAKE_AXOLOTL_PYTHON % {"python": sys.executable})
    exe.chmod(0o755)
    return exe
