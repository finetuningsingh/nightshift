"""Control API: what a researcher may ask of the lab, over the per-trial token.

Mounted at ``/control`` on the same loopback server as the metering proxy (``metering_proxy.LabHttpServer``), reached from the researcher
host through the SSH reverse tunnel. Every call needs ``Authorization: Bearer <trial token>``. The researcher's whole power is here:

====================================  =====================================================================================
``GET  /control/budget``              what is left of the trial budget
``POST /control/submit``              body = recipe tar.gz; query ``entrypoint`` (default ``bash run.sh``), ``max_gpu_minutes``
``GET  /control/status[?job_id=]``    state of a job (default: the latest)
``GET  /control/dev-score?adapter_id`` the control-host DEV score of an adapter this trial's own jobs produced
``GET  /control/pointer?adapter_id``  recipe hash and dev score for ``glimmer pointer``
====================================  =====================================================================================

A job runs on a rented GPU through ``Trainer.run_job`` (one at a time per trial; later jobs queue). This class appends
``lab.gpu_job_started`` / ``lab.gpu_job_finished`` to the trial ledger, reads back the dev ``ScoreRecord`` the trainer wrote, and calls the
trial runner's ``on_job_finished`` hook (master ``lab.score_recorded``, contamination scan). If the hook returns a reason the job is
shown as ``failed`` and its adapter is withheld: no dev score is served for it and the trial runner never selects it.

Recipe tarballs are validated here with the rules of docs/harness-design.md "File formats": gzip tar; regular files and directories only;
relative paths without ``..``, a leading ``/`` or a ``.git`` component; no links, devices or sparse files; at most 5,000 entries, 50 MB
compressed and 200 MB unpacked (the decompressed stream is counted, so a gzip bomb is cut off). ``run_job.sh`` re-checks before extraction.

Ids made here are trial-scoped so that concurrent trials never collide without a shared allocator: job ``<trial>-j<n>``, dev score
``<trial>-dev<n>``.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import os
import queue
import shlex
import tarfile
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional, Sequence

from nightshift.lab.contracts import (
    BudgetExhausted,
    BudgetMeter,
    Clock,
    ContractError,
    GpuUsage,
    JobStatus,
    LabKind,
    LabLayout,
    LedgerWriter,
    Resource,
    ScoreRecord,
    ScoreRequest,
    Split,
    Trainer,
    TrainJob,
    TrainResult,
    TrialSpec,
    loads_strict,
    read_regular_bytes,
)
from nightshift.lab.metering_proxy import Mount, bearer_ok, read_request_body, send_json

MAX_RECIPE_ENTRIES = 5000
MAX_RECIPE_COMPRESSED_BYTES = 50 * 1024 * 1024
MAX_RECIPE_UNPACKED_BYTES = 200 * 1024 * 1024
DEFAULT_ENTRYPOINT = "bash run.sh"
MAX_FAILURE_CHARS = 500
_INTERPRETERS = frozenset({"bash", "sh", "python", "python3", "python3.11", "python3.12", "python3.13"})
_DEV_SCORE_MAX_BYTES = 1 << 24

HookResult = Optional[str]
JobFinishedHook = Callable[[TrainJob, TrainResult, Optional[ScoreRecord]], HookResult]


class RecipeRejected(ValueError):
    """The recipe tarball or its entrypoint breaks a rule; the message is shown to the researcher."""


class _LimitedReader(io.RawIOBase):
    """Reads a gzip stream and refuses to yield more than ``limit`` decompressed bytes (a gzip bomb stops here)."""

    def __init__(self, raw: io.BufferedIOBase, limit: int):
        self._raw, self._left = raw, limit

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        data = self._raw.read(len(b))
        self._left -= len(data)
        if self._left < 0:
            raise RecipeRejected(f"recipe unpacks to more than {MAX_RECIPE_UNPACKED_BYTES} bytes")
        b[: len(data)] = data
        return len(data)


def entrypoint_script(argv: Sequence[str]) -> str:
    """The path inside the tarball that an entrypoint runs: ``bash run.sh`` -> ``run.sh``; ``./go.sh`` -> ``go.sh``."""
    if not argv:
        raise RecipeRejected("entrypoint is empty")
    if argv[0] in _INTERPRETERS:
        rest = [a for a in argv[1:] if not a.startswith("-")]
        if not rest:
            raise RecipeRejected("entrypoint names no script")
        script = rest[0]
    else:
        script = argv[0]
    path = PurePosixPath(script)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RecipeRejected("entrypoint must be a relative path inside the recipe")
    return PurePosixPath(*[p for p in path.parts if p != "."]).as_posix()


def parse_entrypoint(text: str) -> tuple[str, ...]:
    try:
        argv = tuple(shlex.split(text))
    except ValueError:
        raise RecipeRejected("entrypoint cannot be parsed") from None
    if not argv or len(argv) > 20 or any(len(a.encode("utf-8")) > 500 or not a for a in argv):
        raise RecipeRejected("entrypoint must have 1 to 20 non-empty arguments of at most 500 bytes")
    return argv


def validate_recipe_tarball(data: bytes, entrypoint: Sequence[str]) -> int:
    """Raise ``RecipeRejected`` unless ``data`` is an acceptable recipe tarball whose entrypoint script is inside it. Returns the entry count."""
    if not data:
        raise RecipeRejected("recipe is empty")
    if len(data) > MAX_RECIPE_COMPRESSED_BYTES:
        raise RecipeRejected(f"recipe is larger than {MAX_RECIPE_COMPRESSED_BYTES} bytes compressed")
    script = entrypoint_script(entrypoint)
    files: set[str] = set()
    n = 0
    try:
        gz = gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb")
        limited = io.BufferedReader(_LimitedReader(gz, MAX_RECIPE_UNPACKED_BYTES + MAX_RECIPE_ENTRIES * 1536))
        with tarfile.open(fileobj=limited, mode="r|") as tar:
            unpacked = 0
            for member in tar:
                n += 1
                if n > MAX_RECIPE_ENTRIES:
                    raise RecipeRejected(f"recipe has more than {MAX_RECIPE_ENTRIES} entries")
                name = member.name
                if "\x00" in name or "\\" in name:
                    raise RecipeRejected("recipe has an entry name with a NUL or backslash")
                path = PurePosixPath(name)
                if path.is_absolute() or not path.parts or any(p in ("..", ".git") for p in path.parts):
                    raise RecipeRejected(f"recipe entry {name!r} is not an allowed relative path")
                if member.isdir():
                    continue
                if not member.isreg() or member.issparse():
                    raise RecipeRejected(f"recipe entry {name!r} is not a regular file or directory (links and devices are refused)")
                unpacked += member.size
                if unpacked > MAX_RECIPE_UNPACKED_BYTES:
                    raise RecipeRejected(f"recipe unpacks to more than {MAX_RECIPE_UNPACKED_BYTES} bytes")
                files.add(PurePosixPath(*[p for p in path.parts if p != "."]).as_posix())
    except RecipeRejected:
        raise
    except (tarfile.TarError, OSError, EOFError, ValueError):
        raise RecipeRejected("recipe is not a valid gzip tar archive") from None
    if script not in files:
        raise RecipeRejected(f"entrypoint script {script!r} is not a file in the recipe")
    return n


@dataclasses.dataclass
class _Job:
    train_job: TrainJob
    recipe_sha256: str
    state: str = "queued"
    result: Optional[TrainResult] = None
    failure: Optional[str] = None
    record: Optional[ScoreRecord] = None


class ControlApi(Mount):
    prefix = "/control"

    def __init__(
        self,
        trainer: Trainer,
        meter: BudgetMeter,
        layout: LabLayout,
        spec: TrialSpec,
        token: str,
        ledger: LedgerWriter,
        *,
        clock: Clock,
        on_job_finished: Optional[JobFinishedHook] = None,
    ):
        if not token:
            raise ValueError("token is required")
        self._trainer, self._meter, self._layout, self._spec = trainer, meter, layout, spec
        self._token, self._ledger, self._clock = token, ledger, clock
        self._hook = on_job_finished
        self._actor = f"trial:{spec.trial_id}"
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._order: list[str] = []
        self._adapters: dict[str, _Job] = {}
        self._counter = 0
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._closed = False

    # ---- what the trial runner reads ----------------------------------------------------------------
    def ok_adapters(self) -> dict[str, tuple[TrainResult, ScoreRecord]]:
        """Adapters of this trial's own OK jobs that the finish hook accepted, with their control-host dev records."""
        with self._lock:
            return {aid: (j.result, j.record) for aid, j in self._adapters.items() if j.result is not None and j.record is not None}

    def recipe_sha256(self, adapter_id: str) -> Optional[str]:
        """sha256 of the recipe tarball of the accepted job that produced ``adapter_id`` (authoritative; a researcher's own claim is not)."""
        with self._lock:
            j = self._adapters.get(adapter_id)
            return None if j is None else j.recipe_sha256

    def results(self) -> list[TrainResult]:
        with self._lock:
            return [self._jobs[j].result for j in self._order if self._jobs[j].result is not None]

    def n_jobs(self) -> int:
        with self._lock:
            return len(self._order)

    def close(self, *, wait: bool = True, timeout: Optional[float] = None) -> bool:
        """Refuse new jobs, cancel queued ones and (if ``wait``) wait for the running one. Returns True when no job is still running."""
        with self._lock:
            self._closed = True
            for jid in self._order:
                j = self._jobs[jid]
                if j.state == "queued":
                    j.state, j.failure = "failed", "the trial ended before this job started"
            worker = self._worker
        self._queue.put(None)
        if worker is not None and wait:
            worker.join(timeout)
        return worker is None or not worker.is_alive()

    # ---- HTTP -----------------------------------------------------------------------------------------
    def handle(self, h: BaseHTTPRequestHandler) -> None:
        if not bearer_ok(h, self._token):
            send_json(h, 401, {"error": "invalid token"})
            return
        parts = urllib.parse.urlsplit(h.path)
        route = parts.path[len(self.prefix):].strip("/")
        query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)

        def one(name: str) -> Optional[str]:
            v = query.get(name)
            return v[0] if v else None

        if h.command == "GET" and route == "budget":
            send_json(h, 200, self._budget_left())
        elif h.command == "POST" and route == "submit":
            body = read_request_body(h, MAX_RECIPE_COMPRESSED_BYTES + 1)
            status, doc = self._submit(body, one("entrypoint"), one("max_gpu_minutes"))
            send_json(h, status, doc)
        elif h.command == "GET" and route == "status":
            status, doc = self._status(one("job_id"))
            send_json(h, status, doc)
        elif h.command == "GET" and route == "dev-score":
            status, doc = self._dev_score(one("adapter_id"))
            send_json(h, status, doc)
        elif h.command == "GET" and route == "pointer":
            status, doc = self._pointer(one("adapter_id"))
            send_json(h, status, doc)
        else:
            send_json(h, 404, {"error": "not found"})

    # ---- endpoints ------------------------------------------------------------------------------------
    def _budget_left(self) -> dict:
        b, u = self._spec.budget, self._meter.usage()
        return {
            "researcher_tokens_left": max(0, b.researcher_tokens - u.researcher_tokens),
            "gpu_minutes_left": max(0.0, b.gpu_minutes - u.gpu_minutes),
            "dollars_left": max(0.0, b.dollars - u.dollars),
            "wall_seconds_left": max(0.0, b.wall_seconds - u.wall_seconds),
        }

    def _cap_minutes(self, requested: Optional[float]) -> float:
        """The job's GPU-minute cap: what was asked, never more than the GPU minutes or the wall time left."""
        left = self._budget_left()
        cap = min(left["gpu_minutes_left"], left["wall_seconds_left"] / 60.0)
        return cap if requested is None else min(requested, cap)

    def _refuse(self, reason: str) -> tuple[int, dict]:
        return 422, {"job_id": "", "status": "refused", "reason": reason}

    def _submit(self, body: bytes, entrypoint_text: Optional[str], max_minutes_text: Optional[str]) -> tuple[int, dict]:
        try:
            entrypoint = parse_entrypoint(entrypoint_text or DEFAULT_ENTRYPOINT)
            requested: Optional[float] = None
            if max_minutes_text not in (None, ""):
                try:
                    requested = float(max_minutes_text)  # type: ignore[arg-type]
                except ValueError:
                    raise RecipeRejected("max_gpu_minutes must be a number") from None
                if not (requested > 0) or requested != requested or requested == float("inf"):
                    raise RecipeRejected("max_gpu_minutes must be a positive number")
            validate_recipe_tarball(body, entrypoint)
        except RecipeRejected as exc:
            return self._refuse(str(exc))
        for resource in (Resource.GPU_MINUTES, Resource.GPU_USD, Resource.WALL_SECONDS):
            try:
                self._meter.check(resource)
            except BudgetExhausted as exc:
                return self._refuse(f"budget exhausted: {exc.axis}")
        cap = self._cap_minutes(requested)
        if not cap > 0:
            return self._refuse("budget exhausted: no GPU minutes left")
        sha = hashlib.sha256(body).hexdigest()
        with self._lock:
            if self._closed:
                return self._refuse("the trial is ending; no new jobs")
            self._counter += 1
            n = self._counter
            tid = self._spec.trial_id
            path = self._store_recipe(sha, body)
            recipe = self._layout.file_ref(path)
            s = self._spec
            dev_request = ScoreRequest(
                request_id=f"{tid}-dev{n}",
                phase=s.phase,
                assignment_id=s.assignment_id,
                split=Split.DEV,
                trial_id=tid,
                arm=None,
                base=s.base,
                adapter=None,
                seeds=s.score_seeds,
                params=s.score_params,
                scorer_version=s.scorer_version,
                reason=None,
                final_request_id=s.final_request_id,
            )
            job = TrainJob(f"{tid}-j{n}", tid, recipe, entrypoint, cap, s.shape, dev_request)
            self._jobs[job.job_id] = _Job(job, sha)
            self._order.append(job.job_id)
            self._queue.put(job.job_id)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name=f"jobs-{tid}", daemon=True)
                self._worker.start()
        return 200, {"job_id": job.job_id, "status": "queued"}

    def _store_recipe(self, sha: str, data: bytes) -> Path:
        d = self._layout.trial_recipe_dir(self._spec.trial_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{sha}.tar.gz"
        if path.exists():
            return path
        fd, tmp = tempfile.mkstemp(prefix=sha[:8] + ".", suffix=".tmp", dir=str(d))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def _view(self, j: _Job) -> dict:
        out: dict[str, Any] = {"job_id": j.train_job.job_id, "state": j.state, "gpu_minutes_used": 0.0}
        r = j.result
        if r is not None:
            out["gpu_minutes_used"] = r.gpu.minutes
        if j.state == "ok" and r is not None and r.adapter_id and j.record is not None:
            out["adapter_id"] = r.adapter_id
            out["dev_score"] = j.record.headline_mean()
        elif j.state not in ("queued", "running"):
            text = j.failure or (r.failure if r is not None else None) or "the job did not succeed"
            out["failure"] = text.replace(str(self._layout.root), "<lab>")[:MAX_FAILURE_CHARS]
        return out

    def _status(self, job_id: Optional[str]) -> tuple[int, dict]:
        with self._lock:
            if job_id is None:
                if not self._order:
                    return 200, {"state": "none", "n_jobs": 0}
                doc = self._view(self._jobs[self._order[-1]])
                doc["n_jobs"] = len(self._order)
                return 200, doc
            j = self._jobs.get(job_id)
            if j is None:
                return 404, {"error": "unknown job"}
            return 200, self._view(j)

    def _dev_score(self, adapter_id: Optional[str]) -> tuple[int, dict]:
        with self._lock:
            j = self._adapters.get(adapter_id or "")
            if j is None or j.record is None:
                return 404, {"error": "unknown adapter: dev scores exist only for adapters of this trial's own successful jobs"}
            rec = j.record
            return 200, {"adapter_id": adapter_id, "dev_score": rec.headline_mean(), "n_prompts": rec.n_prompts, "metric": rec.headline_metric}

    def _pointer(self, adapter_id: Optional[str]) -> tuple[int, dict]:
        with self._lock:
            j = self._adapters.get(adapter_id or "")
            if j is None or j.record is None:
                return 404, {"error": "unknown adapter"}
            return 200, {"adapter_id": adapter_id, "recipe_sha256": j.recipe_sha256, "dev_score": j.record.headline_mean()}

    # ---- the job worker -------------------------------------------------------------------------------
    def _work(self) -> None:
        while True:
            jid = self._queue.get()
            if jid is None:
                return
            with self._lock:
                job = self._jobs[jid]
                if job.state != "queued":
                    continue
                job.state = "running"
            try:
                self._run(job)
            except Exception as exc:  # noqa: BLE001 - the worker must survive anything a trainer or hook does
                with self._lock:
                    job.state, job.failure = "failed", f"internal error: {type(exc).__name__}"

    def _synth(self, tj: TrainJob, status: JobStatus, failure: str) -> TrainResult:
        return TrainResult(tj.job_id, tj.trial_id, status, None, None, None, GpuUsage(0.0, 0.0, ()), {}, None, None, failure[:3000])

    def _run(self, job: _Job) -> None:
        tj = job.train_job
        cap = self._cap_minutes(tj.max_gpu_minutes)
        if cap > 0:
            tj = dataclasses.replace(tj, max_gpu_minutes=cap)
        self._ledger.append(LabKind.GPU_JOB_STARTED, self._actor, tj)
        if not cap > 0:
            result = self._synth(tj, JobStatus.BUDGET_EXHAUSTED, "no GPU minutes or wall time left when the job came up")
        else:
            try:
                result = self._trainer.run_job(tj, self._meter)
                if result.job_id != tj.job_id or result.trial_id != tj.trial_id:
                    result = self._synth(tj, JobStatus.FAILED, "trainer returned a result for another job")
            except BudgetExhausted as exc:
                result = self._synth(tj, JobStatus.BUDGET_EXHAUSTED, f"budget exhausted on {exc.axis} during the job")
            except Exception as exc:  # noqa: BLE001
                result = self._synth(tj, JobStatus.FAILED, f"trainer error: {type(exc).__name__}: {exc}")
        self._ledger.append(LabKind.GPU_JOB_FINISHED, self._actor, result)
        record = self._read_dev_record(tj, result)
        rejection: HookResult = None
        if self._hook is not None:
            try:
                rejection = self._hook(tj, result, record)
            except Exception as exc:  # noqa: BLE001
                rejection = f"post-job processing failed: {type(exc).__name__}"
        if rejection is None and result.status is JobStatus.OK and record is None:
            rejection = "the dev score of this job is unavailable"
        with self._lock:
            job.result, job.record = result, record
            if rejection is not None:
                job.state, job.failure = "failed", rejection
            else:
                job.state = result.status.value
                if result.status is JobStatus.OK and result.adapter_id:
                    self._adapters.setdefault(result.adapter_id, job)

    def _read_dev_record(self, tj: TrainJob, result: TrainResult) -> Optional[ScoreRecord]:
        if not result.dev_score_id:
            return None
        try:
            rel = self._layout.score_record_path(result.dev_score_id).relative_to(self._layout.root).as_posix()
            rec = ScoreRecord.from_dict(loads_strict(read_regular_bytes(self._layout.root, rel, max_bytes=_DEV_SCORE_MAX_BYTES)))
        except (ContractError, OSError, ValueError):
            return None
        if (
            rec.score_id != result.dev_score_id
            or rec.trial_id != tj.trial_id
            or rec.split is not Split.DEV
            or (result.adapter is not None and rec.adapter_sha256 != result.adapter.sha256)
        ):
            return None
        return rec
