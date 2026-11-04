"""``TrialExecutor``: one trial, end to end (docs/harness-design.md "One trial, end to end").

``TrialRunner.run(spec)`` is called in a thread of the coordinator process and trials may run concurrently; the runner keeps no per-trial
state (everything lives in a per-run ``_Run`` object), and every ledger it touches is a thread-safe ``LedgerWriter``.

Sequence (the numbers are the design's steps):

1. Refuse before anything is spent: the frozen files (ledger-checked ``load_frozen``), ``check_trial_against_frozen``, the committed pack,
   the scorer version. Create ``trials/<id>/``, write ``spec.json``, append ``lab.trial_started``, open the trial ledger, build the meter.
2. Check out the pack on the control host. 3. Start ONE loopback server holding the metering proxy (``/v1``) and the control API (``/control``).
4. ``runtime.start`` (the runtime stages the researcher host and opens the tunnel). The runner polls the runtime and the meter and stops the
   runtime on any exhausted axis or wall time: the researcher is never asked to stop itself.
5./6. Jobs run through the control API: ``lab.gpu_job_started/finished`` in the trial ledger; the finish hook appends the dev
   ``lab.score_recorded`` and runs the contamination scan. A hit stops the runtime at once and marks the trial CONTAMINATED.
7. Drain the running job, stop the services. 8. Harvest. The RUNNER picks the submission: among attempts naming an adapter produced by THIS
   trial's own accepted jobs, the highest control-host dev score, ties to the earliest ``committed_utc``; anything the researcher wrote
   into ``submission.json`` beyond the adapter id (dev score, recipe hash) is replaced by the control host's values.
9. Re-verify the adapter. 10. Score selection (and, in the final phase, the test split) through ``score_session.score_adapter``; the
   ``ScoreGuard`` lives inside the scorer, so the runner cannot score outside the plan. A scorer failure that revealed no number is recorded
   (``lab.score_failed``) and retried once with the same seeds. 11. Seal the trial ledger, write ``trial.json``, append ``lab.trial_finished``.

Ids made here are trial-scoped (``<trial>-sel``, ``<trial>-test``, ``<trial>-j<n>``, ``<trial>-dev<n>``), unique without a shared allocator.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Callable, Optional

from nightshift.lab.contracts import (
    TEST_ASSIGNMENT_ID,
    AccessDenied,
    AdapterRef,
    AssignmentManifest,
    BudgetExhausted,
    BudgetMeter,
    BudgetSpec,
    BudgetUsage,
    Clock,
    ContaminationReport,
    ContractError,
    ExitReason,
    FinalAnalysis,
    FinalComparisonRefused,
    GeneratorFactory,
    GpuUsage,
    JobStatus,
    LabKind,
    LabLayout,
    LedgerAnchor,
    LedgerWriter,
    MeterFactory,
    ModelServer,
    PackStore,
    Phase,
    PortAllocator,
    ResearcherRuntime,
    Resource,
    Rule,
    RuntimeFailure,
    RuntimeResult,
    ScoreCost,
    ScoreFailure,
    ScoreRecord,
    Scorer,
    ScoreRequest,
    Split,
    StateReader,
    Submission,
    Trainer,
    TrainJob,
    TrainResult,
    TrialContext,
    TrialLedgerFactory,
    TrialRecord,
    TrialSpec,
    TrialStatus,
    check_trial_against_frozen,
    load_frozen,
    utc_iso,
    verify_adapter_dir,
    write_record,
)
from nightshift.lab.control_api import ControlApi
from nightshift.lab.metering_proxy import LabHttpServer, MeteringProxy

ACTOR_COORDINATOR = "coordinator"
ACTOR_SCORER = "scorer"
MAX_FAILURE_CHARS = 1500
MAX_ENV_VALUE = 480
_BUDGET_STOP_RESOURCES = (Resource.RESEARCHER_TOKENS, Resource.GPU_MINUTES, Resource.RESEARCHER_USD, Resource.WALL_SECONDS)

ContaminationScan = Callable[..., ContaminationReport]  # scan(trial_id=, job_id=, train_data=FileRef, recipe=FileRef) -> ContaminationReport


def _clip(text: str, n: int) -> str:
    raw = text.encode("utf-8")
    return text if len(raw) <= n else raw[:n].decode("utf-8", "ignore")


class TrialRunner:
    """``TrialExecutor``. Beyond the design's constructor it takes keyword-only arguments for what the design leaves implicit:

    * ``upstream_base_url`` / ``api_key``: where the metering proxy forwards and the Meta key (a callable returning it, read when a trial
      starts; ``LabConfig.upstream_base_url`` and the environment variable named in ``LabConfig.secret_env``).
    * ``contamination_scan``: ``scan(trial_id=, job_id=, train_data=FileRef, recipe=FileRef) -> ContaminationReport``; the CLI binds
      ``contamination.scan_job`` to the layout, manifest and clock. Required: the scan is not optional.
    * ``score_adapter``: ``score_session.score_adapter`` (imported lazily when not given).
    * ``register_secret``: ``nightshift.secrets.register`` (imported lazily when not given); ``token_factory``; ``poll_seconds`` and ``sleep``
      (the monitor loop; tests inject a fake); ``job_drain_timeout`` (None: wait for the running job, which its own GPU-minute cap bounds).
    """

    def __init__(
        self,
        layout: LabLayout,
        ledger: LedgerWriter,
        new_trial_ledger: TrialLedgerFactory,
        meter_factory: MeterFactory,
        runtime: ResearcherRuntime,
        trainer: Trainer,
        scorer: Scorer,
        model_server: ModelServer,
        generator_factory: GeneratorFactory,
        packs: PackStore,
        state_reader: StateReader,
        *,
        clock: Clock,
        port_allocator: PortAllocator,
        upstream_base_url: str = "",
        api_key: Optional[Callable[[], str]] = None,
        contamination_scan: Optional[ContaminationScan] = None,
        score_adapter: Optional[Callable[..., tuple]] = None,
        register_secret: Optional[Callable[[str], None]] = None,
        token_factory: Optional[Callable[[], str]] = None,
        poll_seconds: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
        job_drain_timeout: Optional[float] = None,
    ):
        self._layout, self._ledger, self._new_trial_ledger, self._meter_factory = layout, ledger, new_trial_ledger, meter_factory
        self._runtime, self._trainer, self._scorer, self._model_server = runtime, trainer, scorer, model_server
        self._generator_factory, self._packs, self._state = generator_factory, packs, state_reader
        self._clock, self._ports = clock, port_allocator
        self._upstream, self._api_key = upstream_base_url, api_key
        self._scan = contamination_scan
        self._score_adapter = score_adapter
        self._register_secret = register_secret
        self._token_factory = token_factory
        self._poll = float(poll_seconds)
        self._sleep = sleep
        self._drain = job_drain_timeout

    # ---- TrialExecutor ----------------------------------------------------------------------------------
    def run(self, spec: TrialSpec) -> TrialRecord:
        self._require_configured()
        frozen = self._check_before_spending(spec)
        layout, tid = self._layout, spec.trial_id
        layout.trial_dir(tid).mkdir(parents=True, exist_ok=False)  # the collision guard: a trial id is used once
        write_record(layout.trial_spec_path(tid), spec)
        started = utc_iso(self._clock())
        self._ledger.append(LabKind.TRIAL_STARTED, ACTOR_COORDINATOR, spec)
        return _Run(self, spec, started, frozen).execute()

    # ---- before anything is spent -------------------------------------------------------------------------
    def _require_configured(self) -> None:
        missing = [n for n, v in (("upstream_base_url", self._upstream), ("api_key", self._api_key), ("contamination_scan", self._scan)) if not v]
        if missing:
            raise RuntimeFailure(f"TrialRunner is not configured: missing {', '.join(missing)}")

    def _check_before_spending(self, spec: TrialSpec) -> dict:
        state = self._state.state()
        if state.manifest is None or state.budget is None:
            raise ContractError("assignments and budget must be frozen before any trial")
        manifest = load_frozen(self._layout, state.manifest, AssignmentManifest)
        budget = load_frozen(self._layout, state.budget, BudgetSpec)
        rule = load_frozen(self._layout, state.rule, Rule) if state.rule is not None else None
        analysis = load_frozen(self._layout, state.final_analysis, FinalAnalysis) if state.final_analysis is not None else None
        if spec.phase is Phase.FINAL and (analysis is None or state.search_closed is None):
            raise ContractError("a final-phase trial needs lab.search_closed and the frozen final analysis")
        check_trial_against_frozen(spec, manifest=manifest, budget=budget, rule=rule, analysis=analysis)
        if spec.phase is Phase.FINAL:
            # the scorer's final guard refuses an uncovered score; refusing here saves the cost of a whole trial that could never be scored
            req = next((f for f in state.finals if f.request_id == spec.final_request_id), None)
            if req is None or req.arm is not spec.arm or req.pack != spec.pack or req.analysis_sha256 != analysis.sha256() or not ({Split.SELECTION} if getattr(analysis, "exploratory", False) else {Split.SELECTION, Split.TEST}) <= set(req.splits):
                raise ContractError("no matching lab.final_requested covers this final-phase trial")
        committed = state.packs.get(spec.pack_version)
        if committed is None or committed.pack != spec.pack:
            raise ContractError(f"pack version {spec.pack_version!r} is not a committed pack at {spec.pack.commit}")
        if spec.scorer_version != self._scorer.scorer_version:
            raise ContractError("TrialSpec.scorer_version differs from the scorer's")
        return {"manifest": manifest, "budget": budget, "rule": rule, "analysis": analysis}


class _Run:
    """The state of one trial run. Created and used by one ``TrialRunner.run`` call (plus the control API's worker thread via the hook)."""

    def __init__(self, runner: TrialRunner, spec: TrialSpec, started_utc: str, frozen: dict):
        self.r, self.spec, self.started_utc, self.frozen = runner, spec, started_utc, frozen
        self.tid = spec.trial_id
        self.layout = runner._layout
        self.tledger: Optional[LedgerWriter] = None
        self.meter: Optional[BudgetMeter] = None
        self.proxy: Optional[MeteringProxy] = None
        self.api: Optional[ControlApi] = None
        self.server: Optional[LabHttpServer] = None
        self.handle: Any = None
        self.token = ""
        self.secrets: list[str] = []
        self._lock = threading.Lock()
        self._stop_reason: Optional[ExitReason] = None
        self._stop_pending: Optional[ExitReason] = None
        self.contaminated = False
        self.scan_failures = 0
        self.budget_axis: Optional[str] = None
        self.env: dict[str, str] = {}
        self.failure: Optional[str] = None
        self.dev_score_ids: dict[str, str] = {}
        self._quiesced = False
        self.job_still_running = False

    # ---- helpers ------------------------------------------------------------------------------------------
    def now(self) -> str:
        return utc_iso(self.r._clock())

    def scrub(self, text: str) -> str:
        for s in self.secrets:
            if s:
                text = text.replace(s, "[REDACTED]")
        return text

    def fail_text(self, exc: BaseException) -> str:
        return _clip(self.scrub(f"{type(exc).__name__}: {exc}"), MAX_FAILURE_CHARS)

    def set_env(self, key: str, value: Any) -> None:
        self.env[key] = _clip(self.scrub(str(value)), MAX_ENV_VALUE)

    def stop_once(self, reason: ExitReason) -> None:
        """Stop the runtime, once, from any thread. A stop requested before the runtime handle exists is applied as soon as it does."""
        with self._lock:
            if self._stop_reason is not None:
                return
            if self.handle is None:
                self._stop_pending = self._stop_pending or reason
                return
            self._stop_reason = reason
            handle = self.handle
        try:
            handle.stop(reason)
        except Exception as exc:  # noqa: BLE001 - the monitor loop sees the run end (or not) either way
            self.set_env("stop_error", self.fail_text(exc))

    # ---- the whole trial ----------------------------------------------------------------------------------
    def execute(self) -> TrialRecord:
        r = self.r
        try:
            self.tledger = r._new_trial_ledger(self.tid)
            try:
                return self._execute()
            except Exception as exc:  # noqa: BLE001 - a bug must still end in a recorded, cleaned-up trial
                self.failure = "internal error: " + self.fail_text(exc)
                self._teardown(crash=True)
                return self._finish(TrialStatus.FAILED, None, None)
        finally:
            try:
                self._teardown(crash=True)
                # a BaseException (KeyboardInterrupt) skips the handler above: the training job it left running must still finish
                # charging the meter and the trial ledger before that ledger is closed
                try:
                    self._quiesce()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                try:
                    if self.tledger is not None:
                        self.tledger.close()
                except Exception:  # noqa: BLE001
                    pass

    def _execute(self) -> TrialRecord:
        r, spec = self.r, self.spec
        self.meter = r._meter_factory(spec, self.tledger)
        self.token = (r._token_factory or _default_token)()
        api_key = r._api_key()
        self.secrets = [self.token, api_key]
        register = r._register_secret or _default_register
        register(self.token)
        register(api_key)
        # 2. clean workspace
        pack_dir = self.layout.trial_pack_dir(self.tid)
        try:
            pack_dir.mkdir(parents=True, exist_ok=False)
            r._packs.checkout(spec.pack, pack_dir)
            self.set_env("pack_tree_sha256", r._packs.tree_sha256(spec.pack))
        except Exception as exc:  # noqa: BLE001
            self.failure = "pack checkout failed: " + self.fail_text(exc)
            return self._finish(TrialStatus.FAILED, None, None)
        # 3. services
        self.proxy = MeteringProxy(self.meter, r._upstream, api_key, self.token, clock=r._clock)
        self.api = ControlApi(r._trainer, self.meter, self.layout, spec, self.token, self.tledger, clock=r._clock, on_job_finished=self._on_job_finished)
        self.server = LabHttpServer([self.proxy, self.api], port=r._ports()).start()
        port = self.server.port
        ctx = TrialContext(spec, str(self.layout.trial_dir(self.tid)), f"http://127.0.0.1:{port}/v1", self.token, f"http://127.0.0.1:{port}/control")
        # 4. launch
        try:
            handle = r._runtime.start(ctx)
        except Exception as exc:  # noqa: BLE001
            self.failure = "researcher runtime failed to start: " + self.fail_text(exc)
            return self._finish(TrialStatus.FAILED, None, None)
        with self._lock:
            self.handle = handle
            pending = self._stop_pending
        if pending is not None:
            self.stop_once(pending)
        # 4./7. watch
        exit_reason = self._monitor()
        self.set_env("runtime_exit", exit_reason.value)
        # 7. drain
        self._quiesce()
        # a reservation the proxy refused with 429 is a budget stop even when the researcher exited before the next poll saw it
        refused = self.proxy.exhausted if self.proxy is not None else None
        if refused is not None:
            self.budget_axis = self.budget_axis or refused.axis
        # 8. harvest
        result: Optional[RuntimeResult] = None
        try:
            result = self.handle.collect()
        except Exception as exc:  # noqa: BLE001
            self.set_env("collect_error", self.fail_text(exc))
            self.failure = "harvest failed: " + self.fail_text(exc)
        close = getattr(self.handle, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                self.set_env("handle_close_error", self.fail_text(exc))
        if result is not None:
            self.env.update({k: v for k, v in result.environment.items() if k not in self.env})
            exit_reason = result.exit_reason
            self.set_env("runtime_exit", exit_reason.value)
        if self.contaminated:
            self.failure = "the contamination scan found selection, held-out or test text in a job's training data or recipe"
            return self._finish(TrialStatus.CONTAMINATED, result, None)
        if self.job_still_running:
            # the job outlived job_drain_timeout: its GPU minutes and scan are not in the usage and ledger about to be sealed, so no
            # number is scored or reported as a finished trial
            self.failure = "a training job was still running after the job drain timeout, so the trial's usage is incomplete"
            return self._finish(TrialStatus.FAILED, result, None)
        if result is None:
            return self._finish(TrialStatus.FAILED, None, None)
        # 8. the runner picks the submission
        picked = self._pick(result)
        if picked is None:
            if exit_reason is ExitReason.CRASHED and not result.submissions:
                self.failure = "the researcher runtime crashed and nothing was submitted"
                return self._finish(TrialStatus.FAILED, result, None)
            self.failure = self.failure or "no submitted attempt names an adapter produced by this trial's own accepted jobs"
            return self._finish(TrialStatus.NO_SUBMISSION, result, None)
        submission, train_result, dev_record = picked
        # 9. adapter
        adapter = train_result.adapter
        try:
            if adapter is None:
                raise ContractError("the accepted job has no adapter")
            verify_adapter_dir(adapter, root=self.layout.root)
        except (ContractError, OSError) as exc:
            self.failure = "adapter verification failed: " + self.fail_text(exc)
            return self._finish(TrialStatus.FAILED, result, _Picked(submission, None, dev_record.score_id, None))
        # 10. selection (and test) scores
        picked_state = _Picked(submission, adapter, dev_record.score_id, None)
        sel_ok, sel_id = self._score(Split.SELECTION, spec.assignment_id, f"{self.tid}-sel", adapter)
        picked_state.selection_score_id = sel_id if sel_ok else None
        if not sel_ok:
            return self._finish(TrialStatus.FAILED, result, picked_state)
        if spec.phase is Phase.FINAL and not getattr(self.frozen.get("analysis"), "exploratory", False):  # an exploratory analysis plans no test scores
            test_ok, _tid = self._score(Split.TEST, TEST_ASSIGNMENT_ID, f"{self.tid}-test", adapter)
            if not test_ok:
                return self._finish(TrialStatus.FAILED, result, picked_state)
        budget_ended = bool(self.budget_axis) or bool(self.meter.usage().exhausted_axes(spec.budget))
        status = TrialStatus.BUDGET_EXHAUSTED if budget_ended else TrialStatus.COMPLETED
        return self._finish(status, result, picked_state)

    # ---- the watch loop -------------------------------------------------------------------------------------
    def _monitor(self) -> ExitReason:
        r = self.r
        while True:
            reason = self.handle.poll()
            if reason is not None:
                return reason
            self._check_budget()
            r._sleep(r._poll)

    def _check_budget(self) -> None:
        exc = self.proxy.exhausted if self.proxy is not None else None
        if exc is None:
            for resource in _BUDGET_STOP_RESOURCES:
                try:
                    self.meter.check(resource)
                except BudgetExhausted as e:
                    exc = e
                    break
        if exc is not None:
            self.budget_axis = self.budget_axis or exc.axis
            self.stop_once(ExitReason.STOPPED_WALL if exc.axis == "wall_seconds" else ExitReason.STOPPED_BUDGET)

    # ---- the control API's hook, in its worker thread ---------------------------------------------------------
    def _on_job_finished(self, job: TrainJob, result: TrainResult, record: Optional[ScoreRecord]) -> Optional[str]:
        r = self.r
        if result.status is JobStatus.BUDGET_EXHAUSTED:
            self.budget_axis = self.budget_axis or "gpu"
        if record is not None:
            r._ledger.append(LabKind.SCORE_RECORDED, ACTOR_SCORER, record)
        if result.train_data is None:
            return None  # nothing was trained on that the control host can scan (a failed job has no adapter either)
        try:
            report = r._scan(trial_id=self.tid, job_id=job.job_id, train_data=result.train_data, recipe=job.recipe)
        except Exception as exc:  # noqa: BLE001 - an unscanned dataset is never scored
            self.scan_failures += 1
            self.set_env("scan_error", self.fail_text(exc))
            return "the contamination scan could not be run on this job's data, so its adapter is withheld"
        r._ledger.append(LabKind.CONTAMINATION_SCANNED, ACTOR_COORDINATOR, report)
        if not report.passed:
            self.contaminated = True
            self.stop_once(ExitReason.STOPPED_CONTAMINATED)
            return "the contamination scan found selection, held-out or test text in this job's training data or recipe"
        return None

    # ---- selecting the submission --------------------------------------------------------------------------------
    def _pick(self, result: RuntimeResult) -> Optional[tuple[Submission, TrainResult, ScoreRecord]]:
        ok = self.api.ok_adapters()  # this trial's own accepted jobs only: an adapter id from another trial is not in here
        best = None
        for sa in result.submissions:
            hit = ok.get(sa.submission.adapter_id)
            if hit is None:
                continue
            key = (-hit[1].headline_mean(), datetime.fromisoformat(sa.committed_utc).timestamp(), sa.submission.adapter_id)
            if best is None or key < best[0]:
                best = (key, sa, hit)
        if best is None:
            ignored = [sa.submission.adapter_id for sa in result.submissions]
            if ignored:
                self.set_env("ignored_submissions", ",".join(ignored))
            return None
        _key, sa, (train_result, dev_record) = best
        recipe = self.api.recipe_sha256(sa.submission.adapter_id) or sa.submission.recipe_sha256
        if recipe != sa.submission.recipe_sha256:
            self.set_env("submission_recipe_mismatch", "true")
        return Submission(sa.submission.adapter_id, recipe, sa.submission.commit, dev_record.headline_mean()), train_result, dev_record

    # ---- scoring ------------------------------------------------------------------------------------------------
    def _score(self, split: Split, assignment_id: str, base_id: str, adapter: AdapterRef) -> tuple[bool, Optional[str]]:
        """Score ``adapter`` once, with one retry after a failure that revealed no number. Returns (ok, score id of the last attempt)."""
        r, spec = self.r, self.spec
        score_adapter = r._score_adapter or _default_score_adapter()
        last_id: Optional[str] = None
        for attempt in range(2):
            request_id = base_id if attempt == 0 else base_id + "2"
            last_id = request_id
            request = ScoreRequest(
                request_id=request_id, phase=spec.phase, assignment_id=assignment_id, split=split, trial_id=self.tid, arm=spec.arm, base=spec.base, adapter=adapter,
                seeds=spec.score_seeds, params=spec.score_params, scorer_version=spec.scorer_version, reason=None, final_request_id=spec.final_request_id,
            )
            try:
                record, usage = score_adapter(request, r._model_server, r._scorer, r._generator_factory)
            except (AccessDenied, FinalComparisonRefused) as exc:
                self.failure = f"scoring on {split.value} was refused: " + self.fail_text(exc)
                return False, None
            except Exception as exc:  # noqa: BLE001 - infrastructure: no number was produced
                text = self.fail_text(exc)
                usage = getattr(exc, "gpu_usage", None)
                if isinstance(usage, GpuUsage):
                    r._ledger.append(LabKind.SCORE_COST, ACTOR_SCORER, ScoreCost(request_id, usage))
                r._ledger.append(
                    LabKind.SCORE_FAILED,
                    ACTOR_SCORER,
                    ScoreFailure(request_id, split, assignment_id, self.tid, adapter.sha256, text, False, self.now()),
                )
                self.failure = f"{split.value} scoring failed" + (" twice" if attempt else "") + ": " + text
                continue
            r._ledger.append(LabKind.SCORE_RECORDED, ACTOR_SCORER, record)
            r._ledger.append(LabKind.SCORE_COST, ACTOR_SCORER, ScoreCost(record.score_id, usage))
            self.failure = None
            return True, record.score_id
        return False, last_id

    # ---- closing ---------------------------------------------------------------------------------------------------
    def _teardown(self, *, crash: bool) -> None:
        """Idempotent cleanup for every path: runtime stopped and closed, jobs cancelled, server shut."""
        handle, api, server = self.handle, self.api, self.server
        try:
            if handle is not None and crash:
                try:
                    if handle.poll() is None:
                        handle.stop(ExitReason.CRASHED)
                except Exception:  # noqa: BLE001
                    pass
                close = getattr(handle, "close", None)
                if close is not None:
                    close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if api is not None:
                api.close(wait=False)
        except Exception:  # noqa: BLE001
            pass
        try:
            if server is not None:
                server.stop()
        except Exception:  # noqa: BLE001
            pass

    def _quiesce(self) -> None:
        """Wait for the running job and for requests the servers are still answering, once, so nothing charges the meter or the ledger after
        the trial's usage is read and its ledger sealed. Every path to ``_finish`` goes through here, the exception path included: the
        runtime is already stopped by then, but a training job it started keeps running and keeps its GPU minutes."""
        if self._quiesced:
            return
        self._quiesced = True
        if self.api is not None and not self.api.close(wait=True, timeout=self.r._drain):
            self.job_still_running = True
            self.set_env("job_still_running", "true")
        if self.server is not None:
            self.server.stop()
            pending = self.server.active_requests
            if pending:
                self.set_env("requests_in_flight_at_stop", pending)
        if self.proxy is not None:
            self.proxy.close()  # a request still waiting on upstream keeps its reservation: it cannot change the meter once usage is read

    def _finish(self, status: TrialStatus, result: Optional[RuntimeResult], picked: Optional["_Picked"]) -> TrialRecord:
        r, spec = self.r, self.spec
        self._quiesce()
        usage: BudgetUsage = self.meter.usage() if self.meter is not None else BudgetUsage(0, 0.0, 0.0, 0.0, 0.0)
        anchor: Optional[LedgerAnchor] = None
        try:
            self.tledger.seal()
            anchor = self.tledger.anchor()
        except Exception as exc:  # noqa: BLE001
            self.set_env("ledger_anchor_error", self.fail_text(exc))
        context_sha = None
        if result is not None:
            context_sha = result.initial_context_sha256
        elif self.handle is not None:
            context_sha = getattr(self.handle, "initial_context_sha256", None)
        scored = status in (TrialStatus.COMPLETED, TrialStatus.BUDGET_EXHAUSTED)
        failure = None if scored else _clip(self.scrub(self.failure or f"trial ended as {status.value}"), 3500)
        record = TrialRecord(
            spec=spec,
            status=status,
            started_utc=self.started_utc,
            finished_utc=self.now(),
            initial_context_sha256=context_sha,
            budget_used=usage,
            submission=None if picked is None else picked.submission,
            adapter=None if picked is None else picked.adapter,
            dev_score_id=None if picked is None else picked.dev_score_id,
            selection_score_id=None if picked is None else picked.selection_score_id,
            trace=None if result is None else result.trace,
            notes=None if result is None else result.notes,
            trial_ledger_anchor=anchor,
            environment=dict(self.env),
            failure=failure,
        )
        write_record(self.layout.trial_record_path(self.tid), record)
        r._ledger.append(LabKind.TRIAL_FINISHED, ACTOR_COORDINATOR, record)
        return record


class _Picked:
    def __init__(self, submission: Submission, adapter: Optional[AdapterRef], dev_score_id: str, selection_score_id: Optional[str]):
        self.submission, self.adapter, self.dev_score_id, self.selection_score_id = submission, adapter, dev_score_id, selection_score_id


def _default_token() -> str:
    import secrets

    return secrets.token_urlsafe(24)


def _default_register(value: str) -> None:
    from nightshift import secrets

    secrets.register(value)


def _default_score_adapter() -> Callable[..., tuple]:
    from nightshift.lab.score_session import score_adapter

    return score_adapter
