"""Test doubles for owner 5 (runtime, metering proxy, control API, researcher tool, trial runner).

Everything here is real code driving fakes: no network beyond loopback, no GPU, no CORAL. Part 1 (below) holds the doubles every test of
this owner needs: a clock, a thread-safe in-memory ledger, a budget meter with the real reserve/settle semantics, a scripted upstream
LLM server. Part 2 holds the trial-runner doubles (trainer, scorer, model server, pack store, state reader, runtime).

``FakeRuntime`` / ``FakeRuntimeHandle`` are the classes ``tests/support/lab_world.py`` expects. ``FakeRuntime()`` with no arguments plays a
tiny researcher: it submits one recipe through the control API of the trial it is started for, waits for the job, and "commits" a
submission for the adapter. ``FakeRuntime(script=fn)`` runs ``fn(ctx, client, handle)`` instead, with a ``ControlClient`` bound to the
trial's endpoint and token.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from nightshift.lab.contracts import (
    PAYLOAD_TYPES,
    BudgetCharge,
    BudgetExhausted,
    BudgetSpec,
    BudgetUsage,
    ContractError,
    LabKind,
    LedgerAnchor,
    LlmUsage,
    Resource,
    assert_ledger_safe,
    canonical_json,
)


# --------------------------------------------------------------------------------------------------
# part 1: clock, ledger, meter, upstream
# --------------------------------------------------------------------------------------------------
class FakeClock:
    def __init__(self, start: float = 1_793_000_000.0):
        self.now = float(start)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> float:
        with self._lock:
            self.now += seconds
            return self.now


class FakeLedger:
    """In-memory ``LedgerWriter`` with the real one's checks: known kind, payload class, ledger-safe strings. Thread-safe."""

    def __init__(self, name: str = "fake"):
        self.name = name
        self.events: list[tuple[str, str, Any]] = []
        self._lock = threading.Lock()
        self.sealed = False
        self.closed = False

    def append(self, kind: str, actor: str, record: Any) -> Any:
        if kind not in LabKind.all():
            raise ContractError(f"unknown kind {kind!r}")
        cls = PAYLOAD_TYPES[kind]
        if not isinstance(record, cls):
            raise ContractError(f"{kind} needs a {cls.__name__}, got {type(record).__name__}")
        assert_ledger_safe(record.to_dict())
        with self._lock:
            if self.closed:
                raise ContractError("ledger is closed")
            self.events.append((kind, actor, record))
            return len(self.events)

    def anchor(self) -> LedgerAnchor:
        with self._lock:
            digest = hashlib.sha256(canonical_json([[k, a, r.to_dict()] for k, a, r in self.events]).encode()).hexdigest()
            return LedgerAnchor(len(self.events), digest[:16])

    def seal(self) -> None:
        self.sealed = True

    def close(self) -> None:
        self.closed = True

    def of(self, kind: str) -> list:
        with self._lock:
            return [r for k, _a, r in self.events if k == kind]

    def kinds(self) -> list[str]:
        with self._lock:
            return [k for k, _a, _r in self.events]


class FakeMeter:
    """A ``BudgetMeter`` with the real contract: reserve-then-settle, idempotent per call id, thread-safe, wall derived from the clock."""

    def __init__(
        self,
        spec: BudgetSpec,
        ledger: Optional[FakeLedger] = None,
        *,
        trial_id: str = "t-0001",
        clock: Optional[FakeClock] = None,
        usd_per_prompt_token: float = 1e-6,
        usd_per_completion_token: float = 4e-6,
        usd_per_gpu_minute: float = 0.02,
    ):
        self._spec = spec
        self.ledger = ledger
        self.trial_id = trial_id
        self.clock = clock or FakeClock()
        self.started = self.clock()
        self.p_usd, self.c_usd, self.gpu_usd = usd_per_prompt_token, usd_per_completion_token, usd_per_gpu_minute
        self._lock = threading.RLock()
        self._tokens = 0
        self._gpu_minutes = 0.0
        self._researcher_usd = 0.0
        self._gpu_usd = 0.0
        self._reserved: dict[str, tuple[int, float]] = {}
        self._settled: set[str] = set()
        self._charged: set[str] = set()
        self.reserve_calls: list[tuple[str, int, int, str]] = []

    @property
    def spec(self) -> BudgetSpec:
        return self._spec

    def _usage_locked(self) -> BudgetUsage:
        rtok = sum(t for t, _u in self._reserved.values())
        rusd = sum(u for _t, u in self._reserved.values())
        return BudgetUsage(self._tokens + rtok, self._gpu_minutes, self._researcher_usd + rusd, self._gpu_usd, max(0.0, self.clock() - self.started))

    def usage(self) -> BudgetUsage:
        with self._lock:
            return self._usage_locked()

    def reservations(self) -> dict:
        with self._lock:
            return dict(self._reserved)

    def _price(self, prompt: int, completion: int) -> float:
        return prompt * self.p_usd + completion * self.c_usd

    def check(self, resource: Resource) -> None:
        with self._lock:
            u = self._usage_locked()
            if resource is Resource.WALL_SECONDS:
                if u.wall_seconds >= self._spec.wall_seconds:
                    raise BudgetExhausted("wall_seconds", self._spec.wall_seconds, u.wall_seconds)
            elif resource is Resource.RESEARCHER_TOKENS:
                if u.researcher_tokens >= self._spec.researcher_tokens:
                    raise BudgetExhausted("researcher_tokens", self._spec.researcher_tokens, u.researcher_tokens)
            elif resource is Resource.GPU_MINUTES:
                if u.gpu_minutes >= self._spec.gpu_minutes:
                    raise BudgetExhausted("gpu_minutes", self._spec.gpu_minutes, u.gpu_minutes)
            elif u.dollars >= self._spec.dollars:
                raise BudgetExhausted("dollars", self._spec.dollars, u.dollars)

    def charge(self, resource: Resource, amount: float, call_id: str, detail: Optional[dict] = None) -> BudgetUsage:
        with self._lock:
            if call_id not in self._charged:
                self._charged.add(call_id)
                if resource is Resource.GPU_MINUTES:
                    self._gpu_minutes += amount
                elif resource is Resource.GPU_USD:
                    self._gpu_usd += amount
                elif resource is Resource.RESEARCHER_USD:
                    self._researcher_usd += amount
                elif resource is Resource.RESEARCHER_TOKENS:
                    self._tokens += int(amount)
                if self.ledger is not None:
                    self.ledger.append(LabKind.BUDGET_CHARGED, f"trial:{self.trial_id}", BudgetCharge(self.trial_id, resource, amount, call_id, dict(detail or {})))
            u = self._usage_locked()
            over = u.exhausted_axes(self._spec)
            if over:
                raise BudgetExhausted(over[0], getattr(self._spec, over[0]), getattr(u, over[0], 0.0))
            return u

    def reserve_llm(self, model: str, prompt_tokens_estimate: int, max_tokens: int, call_id: str) -> None:
        with self._lock:
            self.reserve_calls.append((model, prompt_tokens_estimate, max_tokens, call_id))
            if call_id in self._reserved or call_id in self._settled:
                return
            tokens = prompt_tokens_estimate + max_tokens
            usd = self._price(prompt_tokens_estimate, max_tokens)
            u = self._usage_locked()
            if u.researcher_tokens + tokens > self._spec.researcher_tokens:
                raise BudgetExhausted("researcher_tokens", self._spec.researcher_tokens, u.researcher_tokens + tokens)
            if u.dollars + usd > self._spec.dollars:
                raise BudgetExhausted("dollars", self._spec.dollars, u.dollars + usd)
            self._reserved[call_id] = (tokens, usd)

    def charge_llm(self, model: str, usage: LlmUsage, call_id: str) -> BudgetUsage:
        with self._lock:
            if call_id not in self._settled:
                self._settled.add(call_id)
                self._reserved.pop(call_id, None)
                self._tokens += usage.total_tokens
                self._researcher_usd += self._price(usage.prompt_tokens, usage.completion_tokens)
            u = self._usage_locked()
            over = u.exhausted_axes(self._spec)
            if over:
                raise BudgetExhausted(over[0], getattr(self._spec, over[0]), getattr(u, over[0], 0.0))
            return u


class Upstream:
    """A scripted OpenAI-compatible upstream on loopback. ``queue`` holds ``Reply`` items consumed one per request; when empty a default
    echo reply is used. Records every request (headers lower-cased, parsed body, raw bytes)."""

    class Reply:
        def __init__(
            self,
            *,
            status: int = 200,
            text: str = "ok",
            usage: Optional[tuple] = (11, 7, 0),
            stream_usage: bool = True,
            raw_body: Optional[bytes] = None,
            headers: Optional[dict] = None,
            drop_after_chunks: Optional[int] = None,
            on_request: Optional[Callable[[dict], None]] = None,
            delay_per_chunk: float = 0.0,
        ):
            self.status, self.text, self.usage, self.stream_usage = status, text, usage, stream_usage
            self.raw_body, self.headers = raw_body, dict(headers or {})
            self.drop_after_chunks = drop_after_chunks
            self.on_request = on_request
            self.delay_per_chunk = delay_per_chunk

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.queue: list[Upstream.Reply] = []
        self._lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                rec = {"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body, "raw": raw}
                with outer._lock:
                    outer.requests.append(rec)
                    reply = outer.queue.pop(0) if outer.queue else Upstream.Reply()
                if reply.on_request:
                    reply.on_request(rec)
                try:
                    outer._respond(self, reply, body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the proxy hung up (a test of exactly that)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def start(self) -> "Upstream":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "Upstream":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @staticmethod
    def _usage_obj(usage: tuple) -> dict:
        p, c, cached = usage
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c, "prompt_tokens_details": {"cached_tokens": cached}}

    def _respond(self, h: BaseHTTPRequestHandler, reply: "Upstream.Reply", body: Any) -> None:
        stream = isinstance(body, dict) and body.get("stream") is True
        if reply.status >= 300 or (reply.raw_body is not None and not stream):
            data = reply.raw_body if reply.raw_body is not None else json.dumps({"error": {"message": "upstream says no"}}).encode()
            h.send_response(reply.status)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(data)))
            for k, v in reply.headers.items():
                h.send_header(k, v)
            h.end_headers()
            h.wfile.write(data)
            return
        if not stream:
            doc: dict = {"id": "cmpl-1", "object": "chat.completion", "model": body.get("model") if isinstance(body, dict) else "m",
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": reply.text}, "finish_reason": "stop"}]}
            if reply.usage is not None:
                doc["usage"] = self._usage_obj(reply.usage)
            data = json.dumps(doc).encode()
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(data)))
            h.end_headers()
            h.wfile.write(data)
            return
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        chunks = []
        for i, word in enumerate(reply.text.split() or ["ok"]):
            chunks.append({"id": "cmpl-1", "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]})
        chunks.append({"id": "cmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        if reply.usage is not None and reply.stream_usage:
            chunks.append({"id": "cmpl-1", "choices": [], "usage": self._usage_obj(reply.usage)})
        for i, c in enumerate(chunks):
            if reply.drop_after_chunks is not None and i >= reply.drop_after_chunks:
                return
            h.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            h.wfile.flush()
            if reply.delay_per_chunk:
                time.sleep(reply.delay_per_chunk)
        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()
        if reply.raw_body:  # echo for scrub tests: a trailing comment line
            h.wfile.write(reply.raw_body)


def post_json(url: str, token: str, body: Any, *, timeout: float = 20.0, raw: Optional[bytes] = None, stream: bool = False) -> tuple[int, dict, bytes]:
    """POST JSON; returns (status, lower-cased headers, body bytes). Never raises on HTTP errors."""
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


# --------------------------------------------------------------------------------------------------
# part 2: trial-runner doubles
# --------------------------------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

from nightshift.lab.contracts import (  # noqa: E402
    ADAPTER_FILES,
    CONTAMINATION_NGRAM,
    METRIC_PROMPT_ALL,
    METRIC_PROMPT_FRACTION,
    METRIC_PROMPT_LOOSE,
    METRIC_PROMPT_STRICT,
    Arm,
    Assignment,
    AssignmentManifest,
    AssignmentRole,
    AssignmentWeight,
    BaseModelRef,
    ComparisonMethod,
    ContaminationReport,
    ExitReason,
    FailedTrialPolicy,
    FinalAnalysis,
    FinalRequest,
    FrozenFile,
    GenerationParams,
    GpuUsage,
    ItemMode,
    JobStatus,
    LabState,
    MixedRowPolicy,
    PackCommitted,
    PackRef,
    PlannedTrial,
    RuntimeResult,
    ScoreRecord,
    ScoreRequest,
    SearchClosed,
    SeedAggregate,
    Shape,
    Split,
    SubmittedAttempt,
    Submission,
    TrainJob,
    TrainResult,
    TrialSpec,
    Verifier,
    adapter_ref_from_dir,
    record_sha256,
    utc_iso,
    write_record,
)

NOW = "2026-11-01T12:00:00+00:00"
BASE_COMMIT = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"


def make_recipe(files: Optional[dict] = None) -> bytes:
    """A deterministic recipe tar.gz (default: a run.sh)."""
    files = files if files is not None else {"run.sh": b"#!/bin/bash\necho train\n", "cfg/config.yaml": b"lr: 1e-4\n"}
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name, data in sorted(files.items()):
                ti = tarfile.TarInfo(name)
                ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
                tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class ControlClient:
    """A minimal HTTP client of the control API (the FakeRuntime's 'researcher')."""

    def __init__(self, endpoint: str, token: str):
        self.endpoint, self.token = endpoint.rstrip("/"), token

    def request(self, method: str, path: str, query: Optional[dict] = None, data: Optional[bytes] = None, token: Optional[str] = None) -> tuple[int, dict]:
        url = self.endpoint + path + ("?" + urllib.parse.urlencode(query) if query else "")
        req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": "Bearer " + (self.token if token is None else token)})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"raw": raw.decode("utf-8", "replace")}

    def submit(self, tar: bytes, entrypoint: Optional[str] = None, max_gpu_minutes: Optional[float] = None) -> tuple[int, dict]:
        q = {}
        if entrypoint is not None:
            q["entrypoint"] = entrypoint
        if max_gpu_minutes is not None:
            q["max_gpu_minutes"] = str(max_gpu_minutes)
        return self.request("POST", "/submit", q, tar)

    def status(self, job_id: Optional[str] = None) -> tuple[int, dict]:
        return self.request("GET", "/status", {"job_id": job_id} if job_id else None)

    def wait(self, job_id: str, timeout: float = 30.0) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            _s, doc = self.status(job_id)
            if doc.get("state") not in ("queued", "running"):
                return doc
            time.sleep(0.02)
        raise TimeoutError(f"job {job_id} did not finish")


def build_score_record(layout, request: ScoreRequest, value: float, clock: FakeClock, *, n_prompts: int = 20, n_truncated: int = 0) -> ScoreRecord:
    """What a real scorer leaves on disk: request.json, per_prompt.jsonl, score.json. Returns the record."""
    d = layout.score_dir(request.request_id)
    d.mkdir(parents=True, exist_ok=False)
    write_record(layout.score_request_path(request.request_id), request)
    layout.per_prompt_path(request.request_id).write_text('{"prompt_key": "k1", "seed": 0, "passed": true}\n')
    test = request.split is Split.TEST
    seeds = []
    for s in request.seeds:
        metrics = {METRIC_PROMPT_STRICT: value, METRIC_PROMPT_LOOSE: min(1.0, value + 0.03)} if test else {METRIC_PROMPT_ALL: value, METRIC_PROMPT_FRACTION: min(1.0, value + 0.05)}
        seeds.append(SeedAggregate(s, n_prompts, metrics, n_truncated, 0))
    record = ScoreRecord(
        score_id=request.request_id,
        request_id=request.request_id,
        phase=request.phase,
        assignment_id=request.assignment_id,
        split=request.split,
        trial_id=request.trial_id,
        arm=request.arm,
        final_request_id=request.final_request_id,
        base=request.base,
        adapter_sha256=None if request.adapter is None else request.adapter.sha256,
        scorer_version=request.scorer_version,
        verifier=Verifier.IFBENCH_UPSTREAM if test else Verifier.IFEVALG,
        verifier_pins={"fake": "1"},
        params=request.params,
        per_seed=tuple(seeds),
        n_prompts=n_prompts,
        per_prompt=layout.file_ref(layout.per_prompt_path(request.request_id)),
        headline_metric=METRIC_PROMPT_STRICT if test else METRIC_PROMPT_ALL,
        created_utc=utc_iso(clock()),
    )
    write_record(layout.score_record_path(request.request_id), record)
    return record


class FakeTrainer:
    """``Trainer`` double. Each job writes a real adapter directory, processed training data and a real dev score through the layout."""

    def __init__(self, layout, *, clock: FakeClock, score_for: Optional[Callable[[int], float]] = None, gpu_minutes: float = 5.0):
        self.layout, self.clock = layout, clock
        self.score_for = score_for or (lambda n: 0.40 + 0.05 * n)
        self.gpu_minutes = gpu_minutes
        self.jobs: list[TrainJob] = []
        self.results: list[TrainResult] = []
        self.train_text = "Write a haiku about rivers. Answer in exactly three lines."
        self.fail_status: Optional[JobStatus] = None
        self.raise_exc: Optional[BaseException] = None
        self.block: Optional[threading.Event] = None
        self.entered = threading.Event()
        self.no_train_data = False
        self.dev_trial_override: Optional[str] = None
        self._lock = threading.Lock()
        self._n = 0
        self.max_parallel = 0
        self._running = 0

    def run_job(self, job: TrainJob, meter: Any) -> TrainResult:
        with self._lock:
            self._n += 1
            n = self._n
            self.jobs.append(job)
            self._running += 1
            self.max_parallel = max(self.max_parallel, self._running)
        try:
            self.entered.set()
            if self.block is not None:
                self.block.wait(30)
            if self.raise_exc is not None:
                raise self.raise_exc
            from nightshift.lab.contracts import BudgetExhausted as _BE

            used = 0.0
            try:
                for i in range(int(self.gpu_minutes)):
                    meter.charge(Resource.GPU_MINUTES, 1.0, f"gpu:{job.job_id}:{i}", {"job": job.job_id})
                    meter.charge(Resource.GPU_USD, 0.02, f"gpu:{job.job_id}:{i}:usd", {"job": job.job_id})
                    used += 1.0
                    if used >= job.max_gpu_minutes:
                        break
            except _BE as exc:
                res = TrainResult(job.job_id, job.trial_id, JobStatus.BUDGET_EXHAUSTED, None, None, None, GpuUsage(used, used * 0.02, ("r-1",)), {}, None, None, f"budget exhausted on {exc.axis}")
                self.results.append(res)
                return res
            gpu = GpuUsage(used, used * 0.02, (f"r-{n}",))
            if self.fail_status is not None:
                res = TrainResult(job.job_id, job.trial_id, self.fail_status, None, None, None, gpu, {}, None, None, f"fake failure in {self.layout.root}")
                self.results.append(res)
                return res
            adapter_id = f"ad-{n:04d}"
            adir = self.layout.adapter_dir(adapter_id)
            adir.mkdir(parents=True)
            (adir / ADAPTER_FILES[0]).write_text(json.dumps({"r": 32, "peft_type": "LORA"}))
            (adir / ADAPTER_FILES[1]).write_bytes(f"weights-{adapter_id}".encode() * 8)
            ref = adapter_ref_from_dir(adir, adapter_id=adapter_id)
            write_record(self.layout.adapter_json_path(adapter_id), ref)
            train_data = None
            if not self.no_train_data:
                p = self.layout.trial_train_data_path(job.trial_id, job.job_id)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"prompt": self.train_text, "response": "ok"}) + "\n")
                train_data = self.layout.file_ref(p)
            req = job.dev_request.with_adapter(job.dev_request.request_id, ref)
            if self.dev_trial_override is not None:
                req = dataclasses.replace(req, trial_id=self.dev_trial_override)
            rec = build_score_record(self.layout, req, self.score_for(n), self.clock)
            res = TrainResult(job.job_id, job.trial_id, JobStatus.OK, adapter_id, ref, rec.score_id, gpu, {"train_s": 12.0, "eval_s": 3.0}, None, train_data, None)
            self.results.append(res)
            return res
        finally:
            with self._lock:
                self._running -= 1


class FakeServeSession:
    def __init__(self, spec: Any, usage: Any):
        self.spec = spec
        self._usage = usage
        self.closed = False
        names = {a.sha256: f"lora-{a.adapter_id}" for a in spec.adapters}
        from nightshift.lab.contracts import ServeEndpoint

        self._endpoint = ServeEndpoint("http://127.0.0.1:9", "glimmer", names)

    @property
    def endpoint(self) -> Any:
        return self._endpoint

    def close(self) -> Any:
        self.closed = True
        return self._usage

    def __enter__(self) -> "FakeServeSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FakeModelServer:
    def __init__(self) -> None:
        self.sessions: list[FakeServeSession] = []

    def serve(self, spec: Any, meter: Any = None) -> FakeServeSession:
        s = FakeServeSession(spec, GpuUsage(3.0, 0.07, (f"s-{len(self.sessions) + 1}",)))
        self.sessions.append(s)
        return s


class FakeScorer:
    """``Scorer`` double: writes real score files; failure injection; optional guard like the real ``ScoreService``."""

    scorer_version = "sv1-fakefakefakefake"

    def __init__(self, layout, *, clock: FakeClock, guard: Optional[Callable[[ScoreRequest], None]] = None, score_for: Optional[Callable[[ScoreRequest], float]] = None):
        self.layout, self.clock, self.guard = layout, clock, guard
        self.score_for = score_for or (lambda r: 0.55)
        self.requests: list[ScoreRequest] = []
        self.fail_next = 0
        self.fail_exc: BaseException = RuntimeError("scorer endpoint died")
        self._lock = threading.Lock()

    def score(self, request: ScoreRequest, generator: Any, model: str) -> ScoreRecord:
        with self._lock:
            self.requests.append(request)
            if self.fail_next > 0:
                self.fail_next -= 1
                raise self.fail_exc
        if self.guard is not None:
            self.guard(request)
        return build_score_record(self.layout, request, self.score_for(request), self.clock)


def fake_score_adapter(request: ScoreRequest, model_server: Any, scorer: Any, generator_factory: Callable, *, meter: Any = None) -> tuple[ScoreRecord, GpuUsage]:
    """The shape of ``score_session.score_adapter``: serve, score, always close."""
    from nightshift.lab.contracts import ServeSpec

    spec = ServeSpec(request.base, () if request.adapter is None else (request.adapter,), 8300, 90.0, "selection score", request.trial_id)
    session = model_server.serve(spec, meter)
    try:
        endpoint = session.endpoint
        record = scorer.score(request, generator_factory(endpoint), endpoint.model_for(request.adapter))
    finally:
        usage = session.close()
    return record, usage


class FakePacks:
    def __init__(self) -> None:
        self.checkouts: list[PackRef] = []
        self.fail = False

    def checkout(self, pack: PackRef, dest: Any) -> None:
        if self.fail:
            raise RuntimeError("checkout failed")
        self.checkouts.append(pack)
        d = Path(dest)
        (d / "seed" / "notes").mkdir(parents=True, exist_ok=True)
        (d / "task.yaml").write_text("task:\n  name: fake\n")
        (d / "seed" / "notes" / "seed.md").write_text("# seed notes\n")

    def tree_sha256(self, pack: PackRef) -> str:
        return hashlib.sha256(pack.commit.encode()).hexdigest()


class FakeScanner:
    """A contamination scan double with the call shape the runner uses: a hit when the processed data (or the recipe) contains a marker."""

    def __init__(self, layout, clock: FakeClock, marker: bytes = b"SELECTION-CANARY"):
        self.layout, self.clock, self.marker = layout, clock, marker
        self.calls: list[dict] = []
        self.raise_exc: Optional[BaseException] = None

    def __call__(self, *, trial_id: str, job_id: str, train_data: Any, recipe: Any) -> ContaminationReport:
        self.calls.append({"trial_id": trial_id, "job_id": job_id, "train_data": train_data, "recipe": recipe})
        if self.raise_exc is not None:
            raise self.raise_exc
        data = (self.layout.root / train_data.path).read_bytes() + (self.layout.root / recipe.path).read_bytes()
        hit = int(self.marker in data)
        return ContaminationReport(trial_id, job_id, train_data, 1, hit, 0, CONTAMINATION_NGRAM, ("c" * 64,), hit == 0, utc_iso(self.clock()))


def make_params() -> GenerationParams:
    return GenerationParams.recommended(8000)


def make_manifest() -> AssignmentManifest:
    revs = lambda sel: {"train_sha256": "1" * 64, "dev_sha256": "2" * 64, "selection_sha256": sel}  # noqa: E731
    return AssignmentManifest(
        assignments=(
            Assignment("a-evo-1", AssignmentRole.EVOLUTION, ("ifeval25",), revs("3" * 64)),
            Assignment("a-held-1", AssignmentRole.HELDOUT, ("copy",), revs("4" * 64)),
        ),
        evolution_families=("ifeval25", "counting"),
        heldout_families=("copy", "firstlast"),
        mixed_row_policy=MixedRowPolicy.DROPPED,
        item_mode=ItemMode.SALTED_DRAW,
        source_revisions={
            "ifrlvr_parquet_sha256": "5" * 64,
            "ifbench_commit": "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d",
            "open_instruct_commit": "b781e782a8fc530577f235aefe282dd046cb8ad2",
            "split_salt_sha256": "6" * 64,
        },
        created_utc=NOW,
    )


def make_final_analysis(budget: Any, per_cell: int = 5) -> FinalAnalysis:
    planned = tuple(PlannedTrial(f"f-{arm.value}-{k}", arm, "a-held-1", 100 + k) for arm in (Arm.V0, Arm.EVOLVED) for k in range(per_cell))
    return FinalAnalysis(
        heldout_metric="prompt_all",
        test_metric="prompt_strict",
        assignments=(AssignmentWeight("a-held-1", 1.0),),
        planned_trials=planned,
        confidence=0.95,
        method=ComparisonMethod.WELCH,
        score_seeds=(0, 1, 2),
        score_params=make_params(),
        failed_trial_policy=FailedTrialPolicy.INVALIDATES_CANDIDATE,
        retry_policy="Retry only after an infrastructure failure that revealed no score.",
        budget_sha256=record_sha256(budget),
        frozen_utc=NOW,
    )


class StateStub:
    """``StateReader`` double: only ``state()`` is needed by the trial runner."""

    def __init__(self, state: LabState):
        self._state = state

    def state(self) -> LabState:
        return self._state


class TrialFixture:
    """A lab root in ``tmp`` with frozen manifest and budget, a committed v0 pack, fakes for every dependency of ``TrialRunner``."""

    def __init__(self, tmp: Any, *, budget: Optional[Any] = None, final: bool = False):
        from nightshift.lab.contracts import BudgetSpec, LabLayout

        self.layout = LabLayout(Path(tmp) / "lab")
        self.layout.root.mkdir(parents=True)
        self.clock = FakeClock()
        self.budget = budget or BudgetSpec(researcher_tokens=2_000_000, gpu_minutes=120.0, dollars=25.0, wall_seconds=7200.0)
        self.manifest = make_manifest()
        self.master = FakeLedger("master")
        self.trial_ledgers: dict[str, FakeLedger] = {}
        self.meters: dict[str, FakeMeter] = {}
        self.pack = PackRef("packs/pack.git", "b" * 40)
        state = LabState()
        state.manifest = self._freeze("assignments", self.layout.manifest_path, self.manifest)
        state.budget = self._freeze("budget", self.layout.budget_path, self.budget)
        state.packs = {"v0": PackCommitted("v0", self.pack, None, None)}
        self.analysis: Optional[FinalAnalysis] = None
        if final:
            self.analysis = make_final_analysis(self.budget)
            state.final_analysis = self._freeze("final_analysis", self.layout.final_analysis_path, self.analysis)
            state.search_closed = SearchClosed("v1", self.pack, 3, 12, "done", NOW)
            state.finals = [
                FinalRequest("f-0001", Arm.EVOLVED, "v0", self.pack, self.analysis.sha256(), (Split.SELECTION, Split.TEST), None, None, NOW),
                FinalRequest("f-0002", Arm.V0, "v0", self.pack, self.analysis.sha256(), (Split.SELECTION, Split.TEST), None, None, NOW),
            ]
        self.state = state
        self.reader = StateStub(state)
        self.packs = FakePacks()
        self.trainer = FakeTrainer(self.layout, clock=self.clock)
        self.scorer = FakeScorer(self.layout, clock=self.clock)
        self.model_server = FakeModelServer()
        self.scanner = FakeScanner(self.layout, self.clock)

    def _freeze(self, name: str, path: Path, record: Any) -> FrozenFile:
        write_record(path, record)
        return FrozenFile(name, self.layout.file_ref(path))

    def new_trial_ledger(self, trial_id: str) -> FakeLedger:
        led = FakeLedger(f"trial-{trial_id}")
        self.trial_ledgers[trial_id] = led
        return led

    def meter_factory(self, spec: TrialSpec, ledger: Any) -> FakeMeter:
        m = FakeMeter(spec.budget, ledger, trial_id=spec.trial_id, clock=self.clock)
        self.meters[spec.trial_id] = m
        return m

    def spec(self, trial_id: str = "t-0001", **kw: Any) -> TrialSpec:
        base: dict[str, Any] = dict(
            trial_id=trial_id,
            phase=__import__("nightshift.lab.contracts", fromlist=["Phase"]).Phase.SEARCH,
            arm=None,
            pack_version="v0",
            pack=self.pack,
            assignment_id="a-evo-1",
            seed=7,
            budget=self.budget,
            shape=Shape("a100", 1, 12, 300),
            base=BaseModelRef("meta-models/Muse-Glimmer-30B", BASE_COMMIT),
            data_revisions=dict(self.manifest.get("a-evo-1").data_revisions),
            scorer_version=self.scorer.scorer_version,
            score_params=make_params(),
            score_seeds=(0, 1, 2),
            final_request_id=None,
        )
        base.update(kw)
        return TrialSpec(**base)

    def runner(self, runtime: Any, **kw: Any) -> Any:
        from nightshift.lab.trial import TrialRunner

        self.registered: list[str] = []
        params: dict[str, Any] = dict(
            clock=self.clock,
            port_allocator=lambda: 0,
            upstream_base_url="http://127.0.0.1:9/v1",
            api_key=lambda: "meta-secret-key-0123456789",
            contamination_scan=self.scanner,
            score_adapter=fake_score_adapter,
            register_secret=self.registered.append,
            poll_seconds=0.01,
        )
        params.update(kw)
        return TrialRunner(
            self.layout, self.master, self.new_trial_ledger, self.meter_factory, runtime, self.trainer, self.scorer, self.model_server,
            lambda endpoint: None, self.packs, self.reader, **params,
        )

    def final_spec(self, trial_id: str = "f-evolved-0", **kw: Any) -> TrialSpec:
        from nightshift.lab.contracts import Phase

        planned = self.analysis.planned(trial_id)
        return self.spec(
            trial_id,
            phase=Phase.FINAL,
            arm=planned.arm,
            assignment_id=planned.assignment_id,
            seed=planned.seed,
            data_revisions=dict(self.manifest.get(planned.assignment_id).data_revisions),
            final_request_id="f-0001" if planned.arm is Arm.EVOLVED else "f-0002",
            **kw,
        )


class FakeRuntimeHandle:
    """``RuntimeHandle`` double. ``poll`` returns the exit reason once the scripted researcher is done or ``stop`` was called."""

    def __init__(self, ctx: Any, layout: Any = None, clock: Optional[FakeClock] = None):
        self.ctx, self.layout, self.clock = ctx, layout, clock or FakeClock()
        self._exit: Optional[ExitReason] = None
        self._lock = threading.Lock()
        self.stopped_with: list[ExitReason] = []
        self.stop_event = threading.Event()
        self.submissions: list[SubmittedAttempt] = []
        self.attempts = 0
        self.closed = False
        self.collect_error: Optional[BaseException] = None
        self.initial_context_sha256 = hashlib.sha256(b"fake initial context").hexdigest()
        self.environment: dict = {"isolation": "fake"}

    def finish(self, reason: ExitReason) -> None:
        with self._lock:
            if self._exit is None:
                self._exit = reason

    def poll(self) -> Optional[ExitReason]:
        with self._lock:
            return self._exit

    def stop(self, reason: ExitReason) -> None:
        with self._lock:
            self.stopped_with.append(reason)
            if self._exit is None:
                self._exit = reason
        self.stop_event.set()

    def commit_submission(self, adapter_id: str, *, recipe_sha256: str = "a" * 64, commit: str = "b" * 40, dev_score: Optional[float] = None, at: Optional[float] = None) -> None:
        self.attempts += 1
        when = utc_iso(self.clock() if at is None else at)
        self.submissions.append(SubmittedAttempt(Submission(adapter_id, recipe_sha256, commit, dev_score), when))

    def collect(self) -> RuntimeResult:
        if self.collect_error is not None:
            raise self.collect_error
        trace = notes = None
        if self.layout is not None:
            d = self.layout.trial_trace_dir(self.ctx.spec.trial_id)
            d.mkdir(parents=True, exist_ok=True)
            (d / "agent_events.jsonl").write_text('{"type": "fake"}\n')
            (d / "notes.md").write_text("fake notes\n")
            trace, notes = self.layout.file_ref(d / "agent_events.jsonl"), self.layout.file_ref(d / "notes.md")
        return RuntimeResult(self._exit or ExitReason.CRASHED, self.attempts, tuple(self.submissions), trace, notes, self.initial_context_sha256, dict(self.environment))

    def close(self) -> None:
        self.closed = True


class FakeRuntime:
    """``ResearcherRuntime`` double. Default script: submit one recipe through the control API, wait for it, commit a submission."""

    def __init__(self, layout: Any = None, script: Optional[Callable[[Any, ControlClient, FakeRuntimeHandle], None]] = None, *, clock: Optional[FakeClock] = None, start_error: Optional[BaseException] = None, finish_with: ExitReason = ExitReason.FINISHED):
        self.layout, self.script, self.clock = layout, script or self.default_script, clock or FakeClock()
        self.start_error, self.finish_with = start_error, finish_with
        self.contexts: list[Any] = []
        self.handles: list[FakeRuntimeHandle] = []
        self.threads: list[threading.Thread] = []

    @staticmethod
    def default_script(ctx: Any, client: ControlClient, handle: FakeRuntimeHandle) -> None:
        _s, doc = client.submit(make_recipe())
        if doc.get("status") != "queued":
            return
        res = client.wait(doc["job_id"])
        if res.get("state") == "ok":
            handle.commit_submission(res["adapter_id"])

    def start(self, context: Any) -> FakeRuntimeHandle:
        if self.start_error is not None:
            raise self.start_error
        self.contexts.append(context)
        handle = FakeRuntimeHandle(context, self.layout, self.clock)
        self.handles.append(handle)
        client = ControlClient(context.control_endpoint, context.llm_token)

        def run() -> None:
            try:
                self.script(context, client, handle)
                handle.finish(self.finish_with)
            except Exception:  # noqa: BLE001
                handle.finish(ExitReason.CRASHED)

        t = threading.Thread(target=run, daemon=True)
        self.threads.append(t)
        t.start()
        return handle


# --------------------------------------------------------------------------------------------------
# part 3: a local stand-in for the researcher host, and a fake `coral`
# --------------------------------------------------------------------------------------------------
import os  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "lab" / "coral"


class _Forwarder:
    """A real TCP forwarder: connections to 127.0.0.1:listen_port are relayed to 127.0.0.1:target_port (an SSH -R, locally)."""

    def __init__(self, listen_port: int, target_port: int):
        self.target = target_port
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", listen_port))
        self._srv.listen(32)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._accept, daemon=True)
        self._t.start()

    def alive(self) -> bool:
        return not self._stop.is_set()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                self._srv.settimeout(0.2)
                c, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._pipe, args=(c,), daemon=True).start()

    def _pipe(self, c: socket.socket) -> None:
        try:
            u = socket.create_connection(("127.0.0.1", self.target))
        except OSError:
            c.close()
            return

        def pump(a: socket.socket, b: socket.socket) -> None:
            try:
                while True:
                    d = a.recv(65536)
                    if not d:
                        break
                    b.sendall(d)
            except OSError:
                pass
            finally:
                for s in (a, b):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        t = threading.Thread(target=pump, args=(c, u), daemon=True)
        t.start()
        pump(u, c)
        t.join(5)
        c.close()
        u.close()

    def close(self) -> None:
        self._stop.set()
        self._t.join(2)
        try:
            self._srv.close()
        except OSError:
            pass


class LocalRemote:
    """``RemoteHost`` that runs commands on THIS machine with a scrubbed environment (nothing of the test process's environment, so no
    secret leaks into the 'researcher host'), and whose reverse tunnel is a local TCP forwarder."""

    def __init__(self, home: Path):
        from nightshift.lab.runtime_coral import run_capped

        self._run = run_capped
        Path(home).mkdir(parents=True, exist_ok=True)
        self.env = {"PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(home), "LANG": "C"}
        self.calls: list[list[str]] = []
        self.tunnels: list[_Forwarder] = []
        self.fail_tunnel: Optional[str] = None
        self.reserved_ports: set[int] = set()

    def run(self, argv: Any, *, input: Optional[bytes] = None, timeout: float = 60.0, max_output: int = 16 << 20) -> Any:
        self.calls.append(list(argv))
        return self._run(list(argv), input=input, timeout=timeout, max_output=max_output, env=self.env)

    def reverse_tunnel(self, remote_port: int, local_port: int) -> _Forwarder:
        from nightshift.lab.runtime_coral import RemoteError

        if self.fail_tunnel:
            raise RemoteError(self.fail_tunnel)
        t = _Forwarder(remote_port, local_port)
        self.tunnels.append(t)
        return t


FAKE_CORAL = '''#!{python}
import json, os, pathlib, shutil, signal, subprocess, sys, time
B = json.loads(pathlib.Path({behavior!r}).read_text())
FIX = pathlib.Path({fixtures!r})
args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("CORAL fake"); sys.exit(0)
assert args[:3] == ["start", "-c", "pack/task.yaml"], args
kv = dict(a.split("=", 1) for a in args[3:])
run = pathlib.Path(kv["workspace.run_dir"]); pub = run / ".coral" / "public"
for d in ("attempts", "logs", "notes"): (pub / d).mkdir(parents=True, exist_ok=True)
(run / "start.json").write_text(json.dumps({{"argv": args, "env": dict(os.environ), "cwd": os.getcwd(), "pwd": os.environ.get("PWD")}}))
(pub / "manager.pid").write_text(str(os.getpid()))
repo = run / "repo"
def git(*a, cwd=repo, check=True):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, capture_output=True, text=True, check=check).stdout.strip()
repo.mkdir(); git("init", "-q"); (repo / "seed.txt").write_text("seed\\n"); git("add", "-A"); git("commit", "-qm", "seed")
def make_attempt(name, files, status="improved", ts="2026-11-01T05:36:22.781753+00:00"):
    for rel, text in files.items():
        p = repo / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    git("add", "-A"); git("commit", "-qm", name, "--allow-empty")
    h = git("rev-parse", "HEAD")
    doc = json.loads((FIX / "attempt_improved.json").read_text()); doc.update(commit_hash=h, status=status, timestamp=ts, title=name)
    (pub / "attempts" / (h + ".json")).write_text(json.dumps(doc))
    return h
for i, a in enumerate(B.get("attempts", [])):
    make_attempt("attempt %d" % i, a.get("files", {{}}), a.get("status", "improved"), a.get("timestamp", "2026-11-01T05:36:2%d.000000+00:00" % i))
for n in B.get("log_names", ["a1.2.log", "a1.10.log", "a1.1.log"]):
    (pub / "logs" / n).write_text((FIX / "opencode_events.log").read_text())
(pub / "notes" / "n1.md").write_text("# note one\\n")
for plant in B.get("plant", []):
    p = pub / plant["path"]; p.parent.mkdir(parents=True, exist_ok=True)
    if plant["kind"] == "symlink": os.symlink(plant["target"], p)
    elif plant["kind"] == "hardlink": os.link(plant["target"], p)
    elif plant["kind"] == "fifo": os.mkfifo(p)
    elif plant["kind"] == "file": p.write_text(plant.get("text", ""))
if B.get("tamper_grader"):
    g = pathlib.Path(os.getcwd()) / "pack" / "grader" / "src" / "glimmer_grader" / "grader.py"
    g.write_text(g.read_text() + "\\n# tampered\\n")
if B.get("straggler"):
    (run / "agents" / "a1").mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(["sleep", "300"], cwd=run / "agents" / "a1", start_new_session=True)
    extra = "".join(str(x) + "\\n" for x in B.get("extra_pids", []))
    (pub / "agent.pids").write_text(str(child.pid) + "\\n99999999\\nnot-a-pid\\n" + extra)
if B.get("detached") == "child":
    # a researcher process that starts its own session and leaves the trial directory, but is still a child of the run
    child = subprocess.Popen(["sleep", "300"], cwd="/", start_new_session=True)
    (run / "detached.pid").write_text(str(child.pid))
elif B.get("detached") == "orphan":
    # the same, but double-forked: the sleep is re-parented to init at once, so only its environment still ties it to the trial
    # (a python process rather than sleep: macOS hides the environment of its protected system binaries from ps)
    subprocess.Popen(["sh", "-c", '"$1" -c "import time; time.sleep(300)" & echo $! > "$0"', str(run / "detached.pid"), sys.executable], cwd="/", start_new_session=True).wait()
mode = B.get("mode", "finish")
if mode == "crash":
    print("boom", file=sys.stderr); sys.exit(3)
if mode == "researcher":
    recipe = run / "recipe"; recipe.mkdir(parents=True, exist_ok=True)
    (recipe / "run.sh").write_text("#!/bin/bash\\necho hi\\n")
    def tool(*a):
        r = subprocess.run(["glimmer", *a], cwd=repo, capture_output=True, text=True)
        return r.returncode, json.loads(r.stdout or "{{}}")
    rc, doc = tool("submit", str(recipe))
    if rc == 0:
        rc, done = tool("wait", doc["job_id"], "--interval", "0.1", "--timeout", "60")
        if rc == 0:
            shutil.copytree(recipe, repo / "recipe")
            git("add", "-A"); git("commit", "-qm", "recipe")
            tool("pointer", done["adapter_id"])
            h = make_attempt("pointer", {{}}, "improved", "2026-11-01T05:40:00.000000+00:00")
            (run / "adapter_id.txt").write_text(done["adapter_id"])
if mode == "hang":
    def bye(*_): sys.exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN if B.get("ignore_term") else bye)
    while True: time.sleep(0.2)
(pub / "auto_stop.json").write_text((FIX / "auto_stop.json").read_text())
sys.exit(0)
'''


def install_fake_coral(coral_home: Path, behavior: dict) -> Path:
    """Write ``<coral_home>/.venv/bin/coral`` (a fake) and its behavior file; returns the behavior path (edit it between runs)."""
    bin_dir = Path(coral_home) / ".venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    beh = Path(coral_home) / "behavior.json"
    beh.write_text(json.dumps(behavior))
    script = bin_dir / "coral"
    script.write_text(FAKE_CORAL.format(python=sys.executable, behavior=str(beh), fixtures=str(FIXTURES)))
    script.chmod(0o755)
    return beh
