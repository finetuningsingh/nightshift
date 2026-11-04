"""``Trainer`` for the Glimmer harness: one researcher job per lease, a dev score, the adapter pulled and verified, the rental deleted.

``GlimmerTrainer.run_job(job, meter)`` (blocking; safe to call from a worker thread):

1. checks the recipe tarball against ``job.recipe`` and the meter for GPU room (no room: a ``BUDGET_EXHAUSTED`` result, nothing rented);
2. rents ONE GPU through :func:`model_server.acquire_lease` (cap = the smallest of the job's, the meter's and the dollars' room; a minute
   ticker charges the meter and deletes the rental if a cap is crossed), uploads ``hostkit/``, stages the pinned public files, bootstraps;
3. uploads the recipe, the PUBLIC train and dev data and ``job.json``, runs ``hostkit/run_job.sh`` and waits for ``trial.done``;
4. pulls from the rental ONLY ``adapter_config.json`` and ``adapter_model.safetensors`` (exact names, regular files, through
   :mod:`gpu_remote`'s tar-stream parser), validates ``adapter_config.json`` and the safetensors header, hashes them into an
   :class:`AdapterRef` and installs them under ``artifacts/adapters/<id>/``; also the processed training data (``train_data``), the
   timings and the log tail;
5. serves the adapter on the SAME rental and scores it on dev through ``Scorer.score`` on the control host (``scores/<id>/request.json``
   and ``score.json`` are written by the scorer; this class returns the id). Dev numbers are researcher-influenced: the recipe ran as root
   on this machine. Selection scoring happens on a fresh rental elsewhere;
6. deletes the rental (confirmed absent) on EVERY path, and returns a ``TrainResult``.

A job that fails by itself (recipe error, timeout, no adapter, no helper-written training data, reload check, dev scoring) is a
``TrainResult`` with a status other than ``ok``; infrastructure faults (the provider, ssh, the money rule, a rental that cannot be
deleted) raise ``TrainerError``. ``AccessDenied`` from the scorer propagates after the teardown.

UNVERIFIED: everything that touches a real GPU (see docs/harness-design.md, "Only on a GPU").
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from nightshift.lab import gpu_remote as remote
from nightshift.lab.contracts import (
    ADAPTER_FILES,
    AccessDenied,
    BaseModelRef,
    BudgetExhausted,
    BudgetMeter,
    Clock,
    ContractError,
    GeneratorFactory,
    GpuError,
    GpuUsage,
    JobStatus,
    LabLayout,
    Scorer,
    ScorerError,
    ServeSpec,
    Shape,
    TIMING_KEYS,
    TrainerError,
    TrainJob,
    TrainResult,
    adapter_ref_from_dir,
    loads_strict,
    read_regular_bytes,
    sha256_file,
    sha256_text,
    verify_adapter_dir,
    write_record,
)
from nightshift.lab.gpu_rentals import RentalManager
from nightshift.lab.model_server import DEFAULT_VLLM_PORT, Lease, RemoteLayout, acquire_lease, run_wrapped, serve_on_lease

MAX_RECIPE_BYTES = 50 * 1024 * 1024
MAX_ADAPTER_FILE_BYTES = 2 * 1024**3
MAX_CONFIG_BYTES = 64 * 1024
MAX_TRAIN_DATA_BYTES = 1024**3
MAX_SMALL_JSON_BYTES = 64 * 1024
LOG_TAIL_BYTES = 2_000_000
MAX_LORA_RANK = 32  # serve.sh runs vLLM with --max-lora-rank 32
SAFETENSORS_MAX_HEADER = 64 * 1024 * 1024
IFEVALG_FILES = ("instructions.py", "instructions_registry.py", "instructions_util.py")
IFEVALG_REL = ("open-instruct", "open_instruct", "IFEvalG")
DEV_MAX_MODEL_LEN = 8300

class _JobFailed(Exception):
    """The job itself failed (not the infrastructure): becomes a TrainResult with status 'failed'."""


_TIMING_ALIASES = {
    "download": "download_s", "load": "load_s", "generation": "generation_s", "generate": "generation_s", "gen": "generation_s",
    "train": "train_s", "training": "train_s", "export": "export_s", "eval": "eval_s", "evaluation": "eval_s",
}  # fmt: skip


def map_timings(raw: object) -> dict:
    """Map recipe-side timing names to ``TIMING_KEYS`` (seconds). Unknown or non-numeric entries are dropped."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v < 0 or v == float("inf"):
            continue
        key = str(k)
        base = re.sub(r"(_seconds|_secs|_s|_sec)$", "", key.lower())
        name = key if key in TIMING_KEYS else _TIMING_ALIASES.get(base)
        if name:
            out[name] = out.get(name, 0.0) + float(v)
    return out


def validate_adapter_config(data: bytes) -> dict:
    """``adapter_config.json`` must be a PEFT LoRA config the serving stack can load (rank within ``--max-lora-rank``). Raises TrainerError."""
    try:
        cfg = loads_strict(data)
    except ContractError as exc:
        raise TrainerError(f"adapter_config.json is not valid JSON: {exc}") from None
    if not isinstance(cfg, dict):
        raise TrainerError("adapter_config.json is not an object")
    if cfg.get("peft_type") != "LORA":
        raise TrainerError(f"adapter_config.json: peft_type {cfg.get('peft_type')!r} is not 'LORA'")
    r = cfg.get("r")
    if isinstance(r, bool) or not isinstance(r, int) or not 1 <= r <= MAX_LORA_RANK:
        raise TrainerError(f"adapter_config.json: rank r={r!r} is outside 1..{MAX_LORA_RANK}")
    alpha = cfg.get("lora_alpha")
    if alpha is not None and (isinstance(alpha, bool) or not isinstance(alpha, (int, float))):
        raise TrainerError("adapter_config.json: lora_alpha is not a number")
    if cfg.get("use_dora"):
        raise TrainerError("adapter_config.json: DoRA adapters are not supported by the serving stack")
    if cfg.get("modules_to_save"):
        raise TrainerError("adapter_config.json: modules_to_save adds full weights; the serving stack loads LoRA matrices only")
    tm = cfg.get("target_modules")
    if tm is not None and not isinstance(tm, (str, list)):
        raise TrainerError("adapter_config.json: target_modules has an unexpected type")
    return cfg


def validate_safetensors(path: Path) -> int:
    """Check the safetensors header of an adapter file: parseable JSON, every tensor inside the file, every tensor a LoRA matrix.
    Returns the number of tensors. Raises TrainerError."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise TrainerError("adapter_model.safetensors is too short")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > SAFETENSORS_MAX_HEADER or 8 + n > size:
            raise TrainerError("adapter_model.safetensors has an implausible header length")
        try:
            header = json.loads(fh.read(n))
        except ValueError:
            raise TrainerError("adapter_model.safetensors header is not JSON") from None
    if not isinstance(header, dict):
        raise TrainerError("adapter_model.safetensors header is not an object")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise TrainerError("adapter_model.safetensors holds no tensors")
    data_size = size - 8 - n
    for name, info in tensors.items():
        offs = info.get("data_offsets") if isinstance(info, dict) else None
        if "lora_" not in name:
            raise TrainerError(f"adapter_model.safetensors holds a tensor that is not a LoRA matrix: {name[:80]!r}")
        if not (isinstance(offs, list) and len(offs) == 2 and all(isinstance(o, int) for o in offs) and 0 <= offs[0] <= offs[1] <= data_size):
            raise TrainerError(f"adapter_model.safetensors: tensor {name[:80]!r} points outside the file")
    return len(tensors)


def default_stage_files(layout: LabLayout) -> dict:
    """The public files the rental needs: the three pinned IFEvalG files (from the third-party checkout) and, when it exists, rewards.py."""
    out: dict = {}
    base = layout.third_party_dir.joinpath(*IFEVALG_REL)
    for name in IFEVALG_FILES:
        out[f"ifevalg/{name}"] = base / name
    rewards = Path(__file__).resolve().parent / "rewards.py"
    if rewards.is_file():
        out["rewards.py"] = rewards
    return out


class GlimmerTrainer:
    """See the module docstring. Constructor per docs/harness-design.md; everything after ``*`` is optional tuning with safe defaults."""

    def __init__(
        self,
        manager: RentalManager,
        shape: Shape,
        scorer: Scorer,
        generator_factory: GeneratorFactory,
        layout: LabLayout,
        hostkit_dir: Path,
        base: BaseModelRef,
        *,
        clock: Clock,
        remote_layout: RemoteLayout = RemoteLayout(),
        stage_files: Optional[Mapping[str, Path]] = None,
        stage_pins: Optional[Mapping[str, str]] = None,
        weights_sha256_file: Optional[Path] = None,
        pull_margin_s: float = 300.0,
        eval_reserve_s: float = 900.0,  # LIVE 2026-11-04: adapter reload + Inspect dev score of 150 prompts took 7.9 min on an A100 (79 s + 395 s); 15 min is about twice that. 30 min left researchers too little recipe time inside a 72-minute account
        poll_seconds: float = 5.0,
        tick_interval: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        vllm_port: int = DEFAULT_VLLM_PORT,
        ready_timeout_s: float = 900.0,
        bootstrap_timeout_s: float = 7200.0,
        serve_timeout_s: float = 1800.0,
        max_train_data_bytes: int = MAX_TRAIN_DATA_BYTES,
        min_job_seconds: float = 60.0,
    ) -> None:
        self.manager, self.shape, self.scorer, self.generator_factory = manager, shape, scorer, generator_factory
        self.layout, self.hostkit_dir, self.base, self.clock = layout, Path(hostkit_dir), base, clock
        self.remote_layout = remote_layout
        self.stage_files, self.stage_pins = stage_files, dict(stage_pins or {})
        self.weights_sha256_file = weights_sha256_file
        self.pull_margin_s, self.eval_reserve_s = float(pull_margin_s), float(eval_reserve_s)
        self.poll_seconds, self.tick_interval, self.sleep = poll_seconds, tick_interval, sleep
        self.vllm_port, self.ready_timeout_s, self.bootstrap_timeout_s = vllm_port, ready_timeout_s, bootstrap_timeout_s
        self.serve_timeout_s, self.max_train_data_bytes = serve_timeout_s, max_train_data_bytes
        self.min_job_seconds = float(min_job_seconds)

    # ---- helpers ---------------------------------------------------------------------------------
    def _result(self, job: TrainJob, status: JobStatus, failure: Optional[str], *, usage: Optional[GpuUsage] = None, **kw) -> TrainResult:
        return TrainResult(
            job.job_id, job.trial_id, status, kw.get("adapter_id"), kw.get("adapter"), kw.get("dev_score_id"), usage or GpuUsage(0.0, 0.0, ()),
            kw.get("timings", {}), kw.get("log"), kw.get("train_data"), failure if status is not JobStatus.OK else None,
        )  # fmt: skip

    def _build_stage_bundle(self, into: Path) -> Path:
        files = dict(self.stage_files) if self.stage_files is not None else default_stage_files(self.layout)
        pins: list = []
        for rel, src in sorted(files.items()):
            if not re.fullmatch(r"[A-Za-z0-9_./-]+", rel) or ".." in rel.split("/") or rel.startswith("/"):
                raise TrainerError(f"unsafe stage path {rel!r}")
            src = Path(src)
            try:
                digest = remote.sha256_of_regular(src)
            except (OSError, GpuError) as exc:
                raise TrainerError(f"cannot stage {rel}: {src} is not a readable regular file ({exc}). Run scripts/lab_setup_verifiers.sh.") from None
            want = self.stage_pins.get(rel)
            if want is not None and want != digest:
                raise TrainerError(f"{src} has sha256 {digest}, the pin for {rel} is {want}; refusing to stage it")
            dest = into / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            pins.append(f"{digest}  {rel}")
        if self.weights_sha256_file is not None:
            digest = remote.sha256_of_regular(self.weights_sha256_file)
            shutil.copyfile(self.weights_sha256_file, into / "weights.sha256")
            pins.append(f"{digest}  weights.sha256")
        (into / "pins.sha256").write_text("\n".join(pins) + ("\n" if pins else ""))
        return into

    @staticmethod
    def _adapter_id(job: TrainJob) -> str:
        m = re.fullmatch(r"j-(\d+)", job.job_id)
        return f"ad-{m.group(1)}" if m else f"ad-{job.job_id}"[:64]

    def _session_name(self, job: TrainJob) -> str:
        name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{job.trial_id}-{job.job_id}")
        return name if len(name) <= 64 else name[:52] + "-" + sha256_text(name)[:8]

    def _tail_text(self, lease: Lease, path: str, n: int = 1500) -> str:
        try:
            return remote.read_remote_tail(lease.target, path, max_bytes=n, timeout=60).strip()[-n:]
        except GpuError:
            return ""

    # ---- the job -----------------------------------------------------------------------------------
    def run_job(self, job: TrainJob, meter: BudgetMeter) -> TrainResult:
        layout = self.layout
        try:
            recipe_bytes = read_regular_bytes(layout.root, job.recipe.path, max_bytes=MAX_RECIPE_BYTES)
        except (ContractError, OSError) as exc:
            raise TrainerError(f"recipe {job.recipe.path}: {exc}") from None
        if hashlib.sha256(recipe_bytes).hexdigest() != job.recipe.sha256:
            raise TrainerError("the recipe tarball does not match TrainJob.recipe.sha256")
        assignment = job.dev_request.assignment_id
        train, dev = layout.train_data(assignment), layout.dev_data(assignment)
        for p in (train, dev):
            try:
                remote.sha256_of_regular(p)
            except (OSError, GpuError) as exc:
                raise TrainerError(f"public data file {p} is not a readable regular file: {exc}") from None
        timings: dict = {}
        with tempfile.TemporaryDirectory(prefix="lab-stage-") as td:
            bundle = self._build_stage_bundle(Path(td) / "bundle")
            try:
                lease = acquire_lease(
                    self.manager, job.shape, session=self._session_name(job), trial_id=job.trial_id, item=f"train:{job.job_id}",
                    max_minutes=job.max_gpu_minutes, meter=meter, hostkit_dir=self.hostkit_dir, root=layout.root,
                    layout_remote=self.remote_layout, stage_bundle=bundle, clock=self.clock, sleep=self.sleep,
                    poll_seconds=self.poll_seconds, tick_interval=self.tick_interval, ready_timeout_s=self.ready_timeout_s,
                    bootstrap_timeout_s=self.bootstrap_timeout_s, vllm_port=self.vllm_port, base=self.base,
                )  # fmt: skip
            except BudgetExhausted as exc:
                return self._result(job, JobStatus.BUDGET_EXHAUSTED, f"no GPU budget left before the job started: {exc}")
            except GpuError as exc:
                raise TrainerError(f"could not rent a GPU for {job.job_id}: {exc}") from exc
        payload: dict = {}
        try:
            try:
                self._run_on_lease(job, lease, recipe_bytes, train, dev, timings, payload)
                kind, text = "ok", ""
            except _JobFailed as exc:
                kind, text = "failed", str(exc)
            except TimeoutError as exc:
                kind, text = "timeout", str(exc)
            except remote.FetchRefused as exc:
                kind, text = "failed", f"the rental returned something a fetch refuses: {exc}"
            except BudgetExhausted as exc:
                kind, text = "budget", f"the trial's GPU budget ran out ({exc})"
            except (ScorerError, GpuError, TrainerError) as exc:
                if lease.exhausted is not None:
                    kind, text = "budget", f"the trial's GPU budget ran out ({lease.exhausted})"
                elif self.clock() >= lease.rental.cap_epoch - 1:
                    kind, text = "timeout", f"the rental reached its cap and was deleted ({exc})"
                elif isinstance(exc, ScorerError):
                    kind, text = "failed", f"dev scoring failed: {exc}"
                else:
                    raise TrainerError(f"job {job.job_id}: {exc}") from exc
        finally:
            usage = lease.close("job finished")  # TrainerError if the rental cannot be confirmed deleted
        timings.update({k: v for k, v in lease.timings.items() if k in TIMING_KEYS})
        mapped = map_timings(timings)
        if kind == "ok":
            return TrainResult(job.job_id, job.trial_id, JobStatus.OK, payload["adapter"].adapter_id, payload["adapter"], payload["dev_score_id"], usage,
                               mapped, payload.get("log"), payload["train_data"], None)  # fmt: skip
        status = {"budget": JobStatus.BUDGET_EXHAUSTED, "timeout": JobStatus.TIMEOUT}.get(kind, JobStatus.FAILED)
        keep = {k: payload[k] for k in ("adapter", "adapter_id", "log") if k in payload}
        return self._result(job, status, text[:3500], usage=usage, timings=mapped, **keep)

    def _run_on_lease(self, job: TrainJob, lease: Lease, recipe_bytes: bytes, train: Path, dev: Path, timings: dict, payload: dict) -> None:
        layout, rl, target = self.layout, self.remote_layout, lease.target
        work = rl.work
        # ---- inputs
        with tempfile.TemporaryDirectory(prefix="lab-job-") as td:
            tar = Path(td) / "recipe.tar.gz"
            tar.write_bytes(recipe_bytes)
            remote.upload(target, tar, f"{work}/recipe.tar.gz", timeout=600)
            remote.upload(target, train, f"{work}/data/train.jsonl")
            remote.upload(target, dev, f"{work}/data/dev.jsonl")
            deadline = lease.rental.cap_epoch - self.pull_margin_s - self.eval_reserve_s
            if deadline <= self.clock() + self.min_job_seconds:
                setup_min = max(0.0, (self.clock() - lease.rental.create_attempt_epoch) / 60.0)
                cap_min = (lease.rental.cap_epoch - lease.rental.create_attempt_epoch) / 60.0
                raise _JobFailed(
                    f"the rental cap leaves no time for the recipe: of the {cap_min:.0f} minutes requested, about {setup_min:.0f} were used for setup, "
                    f"{self.pull_margin_s / 60:g} are kept to pull the adapter and {self.eval_reserve_s / 60:g} for the dev score, so the recipe would get "
                    f"{max(0.0, (deadline - self.clock()) / 60):.0f}. Ask for at least {(self.pull_margin_s + self.eval_reserve_s) / 60 + setup_min + 2:.0f} "
                    "minutes plus the minutes your recipe needs (every job pays this overhead; it counts against the trial's GPU minutes)")
            jf = Path(td) / "job.json"
            jf.write_text(json.dumps({"entrypoint": list(job.entrypoint), "deadline_epoch": int(deadline)}))
            remote.upload(target, jf, f"{work}/job.json")
        # ---- run
        run_wrapped(lease, ["bash", f"{rl.hostkit}/run_job.sh"], "job", timeout_s=max(60.0, lease.rental.cap_epoch - self.clock()))
        out = f"{work}/out"
        log_text = self._tail_text(lease, f"{out}/job.log", 3000)
        payload["log"] = self._save_log(job, lease)
        try:
            res = remote.run(target, f"cat {out}/trial.done", timeout=60, check=False)
            done = loads_strict(res.text) if res.returncode == 0 else None
        except (GpuError, ContractError):
            done = None
        if not isinstance(done, dict) or "reason" not in done:
            raise _JobFailed(f"run_job.sh ended without a trial.done marker. Log tail: {log_text[-1200:]}")
        reason = str(done["reason"])
        if reason == "timeout":
            raise TimeoutError(f"the recipe was stopped at its deadline ({max(0.0, (deadline - lease.rental.create_attempt_epoch) / 60):.0f} minutes after the rental was created; "
                               f"the last {(self.pull_margin_s + self.eval_reserve_s) / 60:g} minutes of the cap are kept for the adapter pull and the dev score). Log tail: {log_text[-1000:]}")
        if reason != "ok":
            raise _JobFailed(f"job failed: {reason} (exit {done.get('exit_code')}). Log tail: {log_text[-1200:]}")
        # ---- outputs: adapter (exact names), training data, timings
        timings.update(self._fetch_timings(lease, out))
        adapter = self._fetch_adapter(job, lease, out)
        payload["adapter"], payload["adapter_id"] = adapter, adapter.adapter_id
        payload["train_data"] = self._fetch_train_data(job, lease, out)
        # ---- dev score on the same rental
        t_eval = self.clock()
        spec = ServeSpec(self.base, (adapter,), DEV_MAX_MODEL_LEN, max(1.0, (lease.rental.cap_epoch - self.clock()) / 60.0), "dev score", job.trial_id)
        endpoint = serve_on_lease(lease, spec, remote_adapter_dirs={adapter.sha256: f"{out}/adapter"}, serve_timeout_s=self.serve_timeout_s)
        request = job.dev_request.with_adapter(job.dev_request.request_id, adapter)
        record = self.scorer.score(request, self.generator_factory(endpoint), endpoint.model_for(adapter))
        timings["eval_s"] = self.clock() - t_eval
        payload["dev_score_id"] = record.score_id

    # ---- pulls ---------------------------------------------------------------------------------------
    def _save_log(self, job: TrainJob, lease: Lease):
        try:
            text = remote.read_remote_tail(lease.target, f"{self.remote_layout.work}/out/job.log", max_bytes=LOG_TAIL_BYTES, timeout=120)
        except GpuError:
            return None
        path = self.layout.trial_dir(job.trial_id) / "job_logs" / f"{job.job_id}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return self.layout.file_ref(path)

    def _fetch_timings(self, lease: Lease, out: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="lab-timings-") as td:
            if not remote.remote_exists(lease.target, f"{out}/timings.json"):
                return {}
            try:
                got = remote.fetch_files(lease.target, out, ["timings.json"], td, max_file_bytes=MAX_SMALL_JSON_BYTES)["timings.json"]
                return map_timings(loads_strict(got.path.read_bytes()))
            except (GpuError, ContractError):
                return {}

    def _fetch_adapter(self, job: TrainJob, lease: Lease, out: str):
        """Only the two adapter files, exact names, regular files; validated, hashed, installed under artifacts/adapters/<id>/."""
        layout = self.layout
        adapter_id = self._adapter_id(job)
        final = layout.adapter_dir(adapter_id)
        if final.exists():
            raise TrainerError(f"adapter directory {final} already exists")
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{adapter_id}-", dir=final.parent))
        try:
            src = f"{out}/adapter"
            remote.fetch_files(lease.target, src, ["adapter_config.json"], tmp, max_file_bytes=MAX_CONFIG_BYTES)
            remote.fetch_files(lease.target, src, ["adapter_model.safetensors"], tmp, max_file_bytes=MAX_ADAPTER_FILE_BYTES, timeout=3600)
            try:
                validate_adapter_config((tmp / "adapter_config.json").read_bytes())
                validate_safetensors(tmp / "adapter_model.safetensors")
            except TrainerError as exc:
                raise _JobFailed(f"the adapter is not usable: {exc}") from None
            ref = adapter_ref_from_dir(tmp, adapter_id=adapter_id)
            tmp.rename(final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        write_record(layout.adapter_json_path(adapter_id), ref)
        verify_adapter_dir(ref, root=layout.root)  # re-verified after the move
        return ref

    def _fetch_train_data(self, job: TrainJob, lease: Lease, out: str):
        dest = self.layout.trial_train_data_path(job.trial_id, job.job_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = remote.fetch(lease.target, f"{out}/train_data.jsonl", dest, max_bytes=self.max_train_data_bytes, timeout=3600)
        if got.size_bytes == 0:
            raise _JobFailed("train_data.jsonl is empty")
        return self.layout.file_ref(dest)
