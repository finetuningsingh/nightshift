"""The workflow engine: turns lane requests into ledger events (docs/ARCHITECTURE.md 4.4, 5, 7.1-7.7, 8, 13, A.1-A.4).

Guarantees
----------
* Every ledger write of this module goes through ``WorkflowEngine._append``. Once the crash hook has fired (it counts the
  ``experiment.run_started`` events written here) every further ``_append`` raises ``SimulatedCrash``; ``run_lanes`` joins
  the pool and re-raises it. ``SimulatedCrash`` and ``HaltRequested`` are never swallowed, and ``BudgetExceeded`` is
  turned into a block (before a proposal) or an ``unresolved`` record (mid-lane), never ignored. ``HaltRequested`` first
  records every experiment of the interrupted lanes that is still in flight as ``unresolved`` (note ``stopped by HALT``,
  open runs closed as interrupted), then propagates: a sealed ledger never holds an experiment that is mid-stage.
* ``crash_after_runs`` is exact: the ``run_started`` append and the run counter share one lock, so with parallel lanes exactly
  that many runs ever start.
* An implementer that changes nothing is not evaluated: the experiment is recorded ``unresolved`` (the hypothesis is untested)
  with an ``infra`` lesson, and no ``experiment.implemented`` event is written for it.
* Evidence is an ``experiment.graded`` event produced by ``grading.evaluate``. Agent claims and self-reports are only
  recorded, never read for a decision.
* Sealed-split numbers (replication and replication_baseline grades, sealed deltas and margins) are written to the ledger
  for humans and are never placed in a context object, a lesson, memory text or a ``LaneResult``: a replication reports
  only the boolean ``replicated`` (and the coarse outcome ``improved`` / ``no_gain`` derived from it).
* Everything a model or an experiment produced (hypotheses, plans, findings, stdout, diffs, lessons) is sanitised before
  it is written or handed to a role, and model-chosen strings never become a path, id or key without validation.
* Only ``ModelError`` is ever treated as "the model was unavailable". Any other error in a lane is recorded as an
  ``unresolved`` experiment (note ``engine_error:<type>``) so that ``run_lane`` never raises except for
  ``SimulatedCrash`` and ``HaltRequested``.
* Nothing here reads ``planted_fault``; it is only written (for display and the fault scoreboard).
* Red-team lanes (``--plant``): in a run whose suite carries ``suite.planted``, a lane with a ``scripted_id`` is designed and
  implemented by the planted (scripted) suite, and ONLY those two roles come from it: the reviewer, the auditor, the grader and
  the gates are the run's own. Its ``planted_fault`` is recorded exactly as in a scripted run.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .. import grading, sandbox, secrets
from ..fence import fence, sanitize
from ..task import conditions_hash as _conditions_hash
from ..task import task_hash as _task_hash
from ..task import trusted_hash as _trusted_hash
from ..types import (AgentSuite, BudgetExceeded, BudgetLimits, DesignContext, ExperimentPlan, ExperimentView, ExpStatus,
                     GradeResult, HaltRequested, ImplementContext, ImplementResult, Kind, LaneResult, LaneSpec, LedgerCorrupt,
                     LedgerView, ModelError, NightshiftError, Outcome, ProtectedPathViolation, Purpose, ReflectContext,
                     ReviewContext, ReviewVerdict, SimulatedCrash, TaskSpec, canonical_json, short_hash)

__all__ = ["BaselineFailed", "WorkflowEngine"]

TAIL_BYTES = 4096
WORKER_LOG_BYTES = 256 * 1024
EDITABLE_FILE_BYTES = 20_000
EDITABLE_FILE_COUNT = 6
MAX_FINDINGS = 12
MAX_LESSONS = 6
MAX_PIVOTS = 5
CLAIM_TYPES = ("improved", "no_effect", "fixed", "done", "safe", "retraction", "other")
SEVERITIES = ("block", "warn", "info")
VERDICTS = ("accept", "reject", "repair")
TRIGGERS = ("stall", "manual", "periodic")
VIOLATION_KINDS = ("workspace", "trusted", "runtime", "trusted_probe")
PLAN_SUMMARY_CHARS = 600
HALT_NOTE = "stopped by HALT"
BUDGET_NOTE = "stopped by the budget"
NO_CHANGE_NOTE = "the implementer made no change; the hypothesis is untested"
ATTACKER_NOTE = ("that lane is a scripted red-team attacker added by the operator; it cannot be replicated or promoted, so a "
                 "scripted patch never decides the verdict")

_EXP_ID = re.compile(r"^e\d{3,6}$")
_LABEL_BAD = re.compile(r"[^A-Za-z0-9_-]")
_SLUG_BAD = re.compile(r"[^A-Za-z0-9_.-]")
_SCRIPTED_ID = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
_DUP_OF = re.compile(r"duplicate of (e\d{3,6})")


class BaselineFailed(NightshiftError):
    """The unmodified template could not be graded. ``outcome`` is the outcome written for the baseline experiment."""

    def __init__(self, message: str, outcome: str = Outcome.FAILED):
        super().__init__(message)
        self.outcome = outcome


class _BadLane(Exception):
    """A lane request the engine cannot honour (unusable base). Never escapes this module."""


def _gates():
    """``workflows.gates`` is imported on first use: it is written by another engineer and imports ``grading``."""
    from . import gates
    return gates


# --------------------------------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------------------------------
def _slug(value: Any, limit: int = 40) -> str:
    return _SLUG_BAD.sub("_", str(value or ""))[:limit]


def _label(value: Any) -> str:
    """A lane label made safe for ids and actors. ``baseline`` is reserved for the real baseline experiment."""
    text = _LABEL_BAD.sub("_", str(value or ""))[:16] or "lane"
    return "lane_baseline" if text.lower() == "baseline" else text


def _tail(text: Any, limit: int = TAIL_BYTES) -> str:
    raw = str(text or "").encode("utf-8", errors="replace")
    return raw[-limit:].decode("utf-8", errors="ignore")


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _signed(value: Any) -> str:
    f = _finite(value)
    return "n/a" if f is None else f"{f:+.4g}"


def _vkind(kind: Any) -> str:
    return kind if kind in VIOLATION_KINDS else "workspace"


def _same_model(a: str, b: str) -> bool:
    """Two model ids name the same model: an answer may carry a version suffix (``muse-spark-1.3-0615`` answers for
    ``muse-spark-1.3``), which is still the implementer's model, not an independent one. Same rule as ``agents.llm``."""
    return a == b or a.startswith(b + "-") or b.startswith(a + "-")


def _identity(value: Any) -> str:
    return sanitize(str(value or ""), 80).replace("\n", " ").strip()


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temp file in the same directory and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class _Lane:
    """In-process context of one experiment. Everything durable is in the ledger; this only carries the live handles."""

    exp_id: str
    spec: LaneSpec
    stages: frozenset
    plan: ExperimentPlan
    base_id: Optional[str]
    base_dir: Optional[Path]
    ws: Optional[sandbox.Workspace] = None
    worker: str = ""
    tree_hash: str = ""
    last_diff: str = ""
    attempt: int = 1
    pending: Optional[tuple[str, str]] = None      # (outcome, note) decided before the record phase
    skip_ws_event: bool = False
    hard_done: bool = False
    stopping: bool = False                         # the lane is being closed because the run is stopping: no reflection
    notes: list[str] = field(default_factory=list)


@dataclass
class _Eval:
    result: grading.EvalResult
    attempt: int

    @property
    def failure_class(self) -> Optional[str]:
        return self.result.failure_class


# --------------------------------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------------------------------
class WorkflowEngine:
    """Runs the baseline, explore lanes, replications and reflection against one ledger. Thread-safe."""

    def __init__(self, *, task: TaskSpec, run_dir: Path | str, ledger: Any, gate: Any, memory: Any, suite: AgentSuite,
                 auditor: Any, limits: BudgetLimits, clock: Callable[[], float] = time.time,
                 overrides: Optional[Mapping[str, Any]] = None, crash_after_runs: Optional[int] = None):
        ov = dict(overrides or {})
        knobs = {k: float(ov[k]) for k in ("run_timeout_sec", "grade_timeout_sec", "stall_sec") if ov.get(k) is not None}
        self.task = dataclasses.replace(task, **knobs) if knobs else task
        self.run_dir = Path(run_dir)
        self.ledger = ledger
        self.gate = gate
        self.memory = memory
        self.suite = suite
        self.auditor = auditor            # the orchestrator audits between batches; the engine never calls it
        self.limits = limits
        self.clock = clock
        self._heartbeat_sec = float(ov.get("heartbeat_sec") or 2.0)
        self._isolation = str(ov.get("isolation") or "auto")
        self._crash_after = int(crash_after_runs) if crash_after_runs else None
        self._runs_started = 0
        self._crashed = threading.Event()
        self._crash_lock = threading.Lock()
        self._lock = threading.RLock()              # check-then-write of proposals, idempotency keys, replication slots
        self._promo_lock = threading.RLock()        # compare with the incumbent, record, promote
        self._claim_lock = threading.Lock()
        self._baseline_sealed_lock = threading.Lock()
        self._reflect_lock = threading.RLock()
        self._grader_hash_cache: Optional[str] = None
        self._replicating: set[str] = set()

    # ---- ledger access ------------------------------------------------------------------------
    def _append(self, kind: str, actor: str, data: dict):
        """The only way this module writes the ledger. Raises ``SimulatedCrash`` once the crash hook has fired.

        With a crash hook armed, the ``experiment.run_started`` append and the count of started runs happen under one lock, so
        with parallel lanes exactly ``crash_after_runs`` runs ever start: a lane that loses the race finds the crash already
        fired and raises before it writes anything.
        """
        counted = kind == Kind.EXP_RUN_STARTED and self._crash_after is not None
        trip = False
        with self._crash_lock if counted else contextlib.nullcontext():
            if self._crashed.is_set():
                raise SimulatedCrash("the simulated crash has already fired")
            event = self.ledger.append(kind, actor, data)
            if counted:
                self._runs_started += 1
                trip = self._runs_started >= self._crash_after
                if trip:
                    self._crashed.set()
        if trip:
            try:
                self.ledger.crashed = True      # informational: lets the orchestrator close the ledger
            except AttributeError:              # pragma: no cover - a ledger with __slots__
                pass
            raise SimulatedCrash(f"simulated crash after {self._runs_started} untrusted run(s)")
        return event

    @property
    def crashed(self) -> bool:
        return self._crashed.is_set()

    def _view(self) -> LedgerView:
        return self.ledger.view()

    def _check_crash(self) -> None:
        if self._crashed.is_set():
            raise SimulatedCrash("the simulated crash has already fired")

    def _check_stop(self, *, work: bool = False) -> None:
        """Between stages: the crash flag, the HALT file and (before starting untrusted work) the wall budget."""
        if self._crashed.is_set():
            raise SimulatedCrash("the simulated crash has already fired")
        if (self.run_dir / "HALT").exists():
            raise HaltRequested("HALT file present in the run directory")
        if work:
            self.gate.check("wall")

    def _grader_hash(self) -> str:
        recorded = self._view().run.get("grader_hash")
        if recorded:
            return str(recorded)
        with self._lock:
            if self._grader_hash_cache is None:
                self._grader_hash_cache = _trusted_hash(self.task)
            return self._grader_hash_cache

    def _check_task_unchanged(self) -> None:
        expected = self._view().run.get("task_hash")
        if expected and _task_hash(self.task) != str(expected):
            raise ProtectedPathViolation(["task"], "trusted")

    def _isolation_label(self) -> str:
        """The mode the sandbox applies to the runs: the configured request (and ``NIGHTSHIFT_ISOLATION``) lowered to what the
        probe proved, never the request echoed back."""
        return sandbox.effective_isolation(self._isolation)

    def _exp_dir(self, exp_id: str) -> Path:
        return self.run_dir / "exp" / exp_id

    # =============================================================================================
    # baseline
    # =============================================================================================
    def run_baseline(self) -> LaneResult:
        """Write the baseline experiment end to end (or finish an interrupted one); freeze its tree to ``frozen/``.

        Raises ``BaselineFailed`` after recording ``failed`` / ``infra_error`` / ``invalid`` when it cannot be graded.
        """
        self._check_stop()
        view = self._view()
        exp = view.baseline or next((e for e in view.experiments.values() if e.lane == "baseline"), None)
        if exp is not None and exp.status == ExpStatus.RECORDED:
            if exp.outcome == Outcome.BASELINE:
                return self._lane_result(exp.exp_id)
            raise BaselineFailed(f"the baseline was already recorded as {exp.outcome}", exp.outcome or Outcome.FAILED)
        if exp is None:
            exp_id = self.ledger.next_exp_id()
            self.gate.reserve_experiment(exp_id, Purpose.BASELINE)
            self._append(Kind.EXP_PROPOSED, "engine", {
                "exp_id": exp_id, "lane": "baseline", "hypothesis": "unmodified template", "plan": "", "parent_exp_id": None,
                "base": "template", "hyp_fingerprint": self.memory.fingerprint("unmodified template"),
                "replicate_of": None, "scripted_id": None, "planted_fault": None})
        else:
            exp_id = exp.exp_id
        return self._baseline_flow(exp_id)

    def _baseline_flow(self, exp_id: str) -> LaneResult:
        ctx = _Lane(exp_id=exp_id, spec=LaneSpec(lane="baseline", hypothesis="unmodified template"), stages=frozenset(),
                    plan=ExperimentPlan(hypothesis="unmodified template", approach=""), base_id=None, base_dir=None)
        try:
            exp = self._view().exp(exp_id)
            if exp is not None and exp.workspace_path and (self._exp_dir(exp_id) / "workspace").is_dir():
                ctx.ws = self._load_workspace(exp_id)
            else:
                ctx.skip_ws_event = bool(exp is not None and exp.workspace_path)
                self._phase_workspace(ctx)
            frozen_hash = self._freeze_baseline(exp_id, ctx.ws)
            if frozen_hash != ctx.ws.tree_hash:
                raise ProtectedPathViolation(["frozen"], "workspace")
            ctx.tree_hash = frozen_hash
            exp = self._view().exp(exp_id)
            rec = exp.grade_for(Purpose.BASELINE)
            if rec is None or not rec.grade.valid:
                self._check_stop()
                attempt = self._next_attempt(exp_id, (Purpose.BASELINE,))
                out = self._evaluate(exp_id, ctx.ws, self.task.seeds, Purpose.BASELINE, ctx.tree_hash,
                                     attempt=attempt, retry_infra=attempt == 1)
                if out.failure_class is not None:
                    outcome, note = self._failure_outcome(out.failure_class)
                    self._record_baseline_failure(exp_id, outcome, note)
                rec = self._view().exp(exp_id).grade_for(Purpose.BASELINE)
            grade = rec.grade
            self._append(Kind.EXP_RECORDED, "engine", {
                "exp_id": exp_id, "phase": "lane", "outcome": Outcome.BASELINE, "delta": None, "vs_exp_id": None,
                "noise_margin": self.task.noise_margin, "train_delta": None, "train_margin": None,
                "overfit_suspect": False, "replicated": False, "sigma_prior": None, "n": len(grade.per_seed),
                "s_eff": None, "t": None, "sealed_delta": None, "sealed_margin": None, "note": ""})
        except ProtectedPathViolation as exc:
            self._append(Kind.EXP_PROTECTED, "engine", {"exp_id": exp_id, "paths": [str(p) for p in exc.paths][:50],
                                                       "kind": _vkind(exc.kind)})
            self._record_baseline_failure(exp_id, Outcome.INVALID, "protected path violation")
        except HaltRequested:
            self._stop_baseline(exp_id)
            raise
        self._after_record()
        return self._lane_result(exp_id)

    def _stop_baseline(self, exp_id: str) -> None:
        """HALT arrived while the baseline was in flight: close its open runs and record it ``unresolved`` (``HALT_NOTE``).
        Best effort, like ``_stop_lane``: the ``HaltRequested`` being propagated is never replaced by a bookkeeping error."""
        try:
            exp = self._view().exp(exp_id)
            if exp is None or exp.status == ExpStatus.RECORDED:
                return
            self._interrupt_open_runs([exp_id], "the baseline was stopped before the run ended")
            self._append(Kind.EXP_RECORDED, "engine", {
                "exp_id": exp_id, "phase": "lane", "outcome": Outcome.UNRESOLVED, "delta": None, "vs_exp_id": None,
                "noise_margin": self.task.noise_margin, "train_delta": None, "train_margin": None, "overfit_suspect": False,
                "replicated": False, "sigma_prior": None, "n": 0, "s_eff": None, "t": None, "sealed_delta": None,
                "sealed_margin": None, "note": HALT_NOTE})
        except (SimulatedCrash, LedgerCorrupt):
            raise
        except Exception:  # noqa: BLE001 - see the docstring
            return

    def _freeze_baseline(self, exp_id: str, ws: sandbox.Workspace) -> str:
        """``freeze`` the baseline tree. A half-copied ``frozen/`` left by a kill is only replaceable while nothing has been
        graded from it; after that a differing tree is tampering."""
        frozen = self._exp_dir(exp_id) / "frozen"
        try:
            return sandbox.freeze(ws.path, frozen, self.task)
        except FileExistsError:
            if self._view().exp(exp_id).grades:
                raise ProtectedPathViolation(["frozen"], "workspace") from None
            shutil.rmtree(frozen)
            return sandbox.freeze(ws.path, frozen, self.task)

    def _record_baseline_failure(self, exp_id: str, outcome: str, note: str) -> None:
        self._append(Kind.EXP_RECORDED, "engine", {
            "exp_id": exp_id, "phase": "lane", "outcome": outcome, "delta": None, "vs_exp_id": None,
            "noise_margin": self.task.noise_margin, "train_delta": None, "train_margin": None, "overfit_suspect": False,
            "replicated": False, "sigma_prior": None, "n": 0, "s_eff": None, "t": None, "sealed_delta": None,
            "sealed_margin": None, "note": note})
        raise BaselineFailed(f"the baseline could not be graded ({outcome}: {note})", outcome)

    @staticmethod
    def _failure_outcome(failure_class: str) -> tuple[str, str]:
        """Contract 7.2 step 5: infra -> infra_error; crash / timeout / invalid -> failed."""
        if failure_class == "infra":
            return Outcome.INFRA, "infrastructure failure persisted after one retry"
        if failure_class == "timeout":
            return Outcome.FAILED, "the run timed out or stalled"
        if failure_class == "crash":
            return Outcome.FAILED, "the run crashed"
        return Outcome.FAILED, "the trusted grader found the output invalid"

    # =============================================================================================
    # lanes
    # =============================================================================================
    def run_lane(self, stages: Sequence[str], lane: LaneSpec) -> LaneResult:
        """One explore lane (contract 7.2). Raises only ``SimulatedCrash`` and ``HaltRequested``."""
        if getattr(lane, "replicate_of", None):
            return self.replicate(lane.replicate_of)
        stage_set = frozenset(s for s in (stages or ()) if isinstance(s, str))
        self._check_stop()
        opened = self._propose(lane, stage_set)
        if isinstance(opened, LaneResult):
            return opened
        return self._guarded(opened, lambda: self._start_lane(opened))

    def run_lanes(self, stages: Sequence[str], lanes: Sequence[LaneSpec]) -> list[LaneResult]:
        """Run lanes on a pool of at most ``limits.max_parallel`` threads; results are in input order.

        After a ``SimulatedCrash`` the other lanes stop at their next write, the pool is joined and the crash is re-raised.
        """
        return self._gather([(lambda lane=lane: self.run_lane(stages, lane)) for lane in list(lanes)])

    def _gather(self, calls: Sequence[Callable[[], Any]]) -> list[Any]:
        if len(calls) <= 1:
            return [call() for call in calls]
        workers = max(1, min(int(self.limits.max_parallel), len(calls)))
        results: list[Any] = [None] * len(calls)
        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nightshift-lane") as pool:
            futures = [pool.submit(call) for call in calls]
            for i, future in enumerate(futures):
                try:
                    results[i] = future.result()
                except BaseException as exc:  # noqa: BLE001 - re-raised below in priority order, after the pool is joined
                    errors.append(exc)
        for kind in (SimulatedCrash, HaltRequested):
            for exc in errors:
                if isinstance(exc, kind):
                    raise exc
        if errors:
            raise errors[0]
        return results

    def _start_lane(self, ctx: _Lane) -> LaneResult:
        self._design(ctx)
        return self._drive(ctx, "workspace")

    def _drive(self, ctx: _Lane, phase: str) -> LaneResult:
        """Run phases until one returns a ``LaneResult``. A phase returns the name of the next phase."""
        while True:
            self._check_stop()
            step = getattr(self, f"_phase_{phase}")(ctx)
            if isinstance(step, LaneResult):
                return step
            phase = step

    def _guarded(self, ctx: _Lane, call: Callable[[], LaneResult]) -> LaneResult:
        try:
            return call()
        except HaltRequested:
            self._stop_lane(ctx, HALT_NOTE)
            raise
        except (SimulatedCrash, LedgerCorrupt):
            raise
        except BudgetExceeded:
            return self._abort(ctx, BUDGET_NOTE)
        except ProtectedPathViolation as exc:
            return self._violation(ctx, [(_vkind(exc.kind), [str(p) for p in exc.paths][:50])])
        except ModelError:
            return self._abort(ctx, "a model call failed", outcome=Outcome.INFRA)
        except Exception as exc:  # noqa: BLE001 - a lane must always end in a record, see the module docstring
            return self._abort(ctx, f"engine_error:{type(exc).__name__}")

    def _stop_lane(self, ctx: _Lane, note: str) -> None:
        """The run is being halted while this lane is in flight: close its open runs as interrupted and record the experiment
        as ``unresolved`` (``note``), exactly as a budget stop does, so that a sealed ledger never holds an experiment that is
        still mid-stage. Best effort and silent: nothing here may replace the ``HaltRequested`` being propagated, except a
        ``SimulatedCrash`` or a corrupt ledger, which are more serious."""
        try:
            exp = self._view().exp(ctx.exp_id)
            if exp is None or exp.status in (ExpStatus.RECORDED, ExpStatus.BLOCKED):
                return
            ctx.stopping = True
            self._abort(ctx, note)
        except (SimulatedCrash, LedgerCorrupt):
            raise
        except Exception:  # noqa: BLE001 - see the docstring
            return

    def _abort(self, ctx: _Lane, note: str, outcome: str = Outcome.UNRESOLVED) -> LaneResult:
        exp = self._view().exp(ctx.exp_id)
        if exp is not None and exp.status == ExpStatus.RECORDED:
            return self._lane_result(ctx.exp_id, notes=note)
        self._interrupt_open_runs([ctx.exp_id], "the lane was stopped before the run ended")
        ctx.pending = (outcome, note)
        return self._phase_record(ctx)

    # ---- proposal -----------------------------------------------------------------------------
    def _propose(self, lane: LaneSpec, stages: frozenset) -> "_Lane | LaneResult":
        label = _label(lane.lane)
        hypothesis = sanitize(lane.hypothesis, 1000)
        scripted_id = lane.scripted_id if isinstance(lane.scripted_id, str) and _SCRIPTED_ID.match(lane.scripted_id) else None
        base = lane.base if isinstance(lane.base, str) and lane.base else "incumbent"
        spec = LaneSpec(lane=label, hypothesis=hypothesis, approach=sanitize(lane.approach, 1000), base=base[:64],
                        replicate_of=None, scripted_id=scripted_id)
        fp = self.memory.fingerprint(hypothesis)
        with self._lock:
            self._check_stop()
            view = self._view()
            reasons = _gates().launch_gate(view, spec, self.memory, gate=self.gate, run_dir=self.run_dir, limits=self.limits,
                                           task=self.task)
            if not reasons:
                twin = self._inflight_twin(view, fp)
                if twin:
                    reasons = [f"duplicate of {twin}: the same hypothesis is already in flight"]
            if reasons:
                return self._block_before_proposal(spec, fp, reasons)
            try:
                base_id, base_dir = self._resolve_base(view, base)
                if base_id is not None and not spec.scripted_id and self._is_attacker(view.exp(base_id)):
                    raise _BadLane(ATTACKER_NOTE.replace("it cannot be replicated or promoted", "it cannot be the base of another lane"))
            except _BadLane as exc:
                return self._block_before_proposal(spec, fp, [str(exc)], reason="invalid_composition")
            exp_id = self.ledger.next_exp_id()
            try:
                self.gate.reserve_experiment(exp_id, Purpose.PRIMARY)
            except BudgetExceeded as exc:
                return self._block_before_proposal(spec, fp, [f"budget exhausted: {_slug(exc.kind, 20)}"])
            self._append(Kind.EXP_PROPOSED, f"designer:{label}", {
                "exp_id": exp_id, "lane": label, "hypothesis": hypothesis, "plan": spec.approach or hypothesis,
                "parent_exp_id": base_id, "base": base_id or "template", "hyp_fingerprint": fp, "replicate_of": None,
                "scripted_id": scripted_id, "planted_fault": self._planted_fault(spec)})
        return _Lane(exp_id=exp_id, spec=spec, stages=stages, plan=ExperimentPlan(hypothesis=hypothesis, approach=spec.approach),
                     base_id=base_id, base_dir=base_dir)

    def _is_attacker(self, exp: Optional[ExperimentView]) -> bool:
        """True for an experiment written by the scripted actors of a red-team run (``--plant``): in a run whose suite carries
        ``suite.planted``, a lane with a ``scripted_id`` is theirs. A scripted-backend run has no ``planted`` suite and is not affected."""
        return exp is not None and bool(exp.scripted_id) and getattr(self.suite, "planted", None) is not None

    def _actors(self, spec: LaneSpec) -> Any:
        """The suite whose designer and implementer serve ``spec``: the planted (scripted attacker) suite for a lane that carries
        a ``scripted_id`` when the run has one, else the run's own suite. Reviewer, reflector and judges are never taken from it."""
        planted = getattr(self.suite, "planted", None)
        return planted if planted is not None and spec.scripted_id else self.suite

    def _planted_fault(self, spec: LaneSpec) -> Optional[str]:
        """Display metadata for the fault scoreboard; never read by a check or a gate, and never allowed to break a lane.

        Recorded for every lane of a scripted run and for the planted (red-team) lanes of a meta run; any other lane has none."""
        actors = self._actors(spec)
        if not spec.scripted_id or (getattr(self.suite, "backend", None) != "scripted" and actors is self.suite):
            return None
        try:
            from ..agents.scripted import planted_fault_for
            fault = planted_fault_for(self.task, spec.scripted_id, bool(getattr(actors, "control", False)))
        except Exception:  # noqa: BLE001
            return None
        return _slug(fault, 40) or None

    @staticmethod
    def _inflight_twin(view: LedgerView, fp: str) -> Optional[str]:
        for exp in view.experiments.values():
            if exp.lane == "baseline" or exp.replicate_of or exp.blocked is not None or exp.terminal:
                continue
            if exp.hyp_fingerprint == fp:
                return exp.exp_id
        return None

    def _resolve_base(self, view: LedgerView, base: str) -> tuple[Optional[str], Optional[Path]]:
        """``(base experiment id, its frozen/ dir)``; ``(None, None)`` is the template."""
        if base == "template":
            return None, None
        base_id = view.incumbent_id if base == "incumbent" else base
        if base_id is None:
            return None, None
        if not _EXP_ID.match(base_id) or view.exp(base_id) is None:
            raise _BadLane("the base experiment is unknown")
        frozen = self._exp_dir(base_id) / "frozen"
        if not (frozen.is_dir() and any(frozen.iterdir())) and not self._freeze_named_base(view.exp(base_id)):
            raise _BadLane("the base experiment has no frozen tree")
        return base_id, frozen

    def _freeze_named_base(self, exp: ExperimentView) -> bool:
        """Freeze a recorded experiment's workspace so it can be a base, but only if it is exactly what was graded."""
        try:
            ws = self._load_workspace(exp.exp_id)
            diff = sandbox.diff_workspace(self.task, ws)
            if diff.violations or diff.diff_hash != exp.diff_hash:
                return False
            sandbox.freeze(ws.path, self._exp_dir(exp.exp_id) / "frozen", self.task)
            return True
        except (OSError, ValueError, KeyError, ProtectedPathViolation):
            return False

    def _block_before_proposal(self, spec: LaneSpec, fp: str, reasons: Sequence[str], reason: Optional[str] = None) -> LaneResult:
        if any(r.startswith("HALT file present") for r in reasons):
            raise HaltRequested("HALT file present in the run directory")
        dup = next((m.group(1) for r in reasons if (m := _DUP_OF.search(r))), None)
        if reason is None:
            reason = ("budget" if any(r.startswith("budget exhausted") for r in reasons)
                      else "duplicate" if dup else "invalid_composition")
        detail = "; ".join(reasons)[:500]
        self._append(Kind.EXP_BLOCKED, "engine", {
            "exp_id": None, "lane": spec.lane, "hypothesis": spec.hypothesis, "hyp_fingerprint": fp, "reason": reason,
            "detail": detail, "dup_of": dup if reason == "duplicate" else None})
        return LaneResult(exp_id="", lane=spec.lane, status=ExpStatus.BLOCKED, blocked_reason=reason, notes=detail)

    # ---- design -------------------------------------------------------------------------------
    def _design(self, ctx: _Lane) -> None:
        spec = ctx.spec
        plan = ExperimentPlan(hypothesis=spec.hypothesis, approach=spec.approach)
        if "design" in ctx.stages:
            self._check_stop()
            dctx = DesignContext(task=self.task, lane=spec, memory_digest=self.memory.digest(),
                                 incumbent_summary=self._incumbent_summary(), editable_files=self._editable_files(ctx.base_dir),
                                 exp_id=ctx.exp_id)
            try:
                got = self._actors(spec).designer.design(dctx)
            except ModelError:
                got = None
            if isinstance(got, ExperimentPlan):
                touch = got.files_to_touch if isinstance(got.files_to_touch, (list, tuple)) else []
                plan = ExperimentPlan(hypothesis=spec.hypothesis,        # the designer refines the approach, not the hypothesis
                                      approach=sanitize(got.approach, 1500) or spec.approach,
                                      expected_effect=sanitize(got.expected_effect, 400), risks=sanitize(got.risks, 400),
                                      files_to_touch=[sanitize(str(f), 120) for f in touch[:10]])
        ctx.plan = plan

    def _editable_files(self, base_dir: Optional[Path]) -> dict[str, str]:
        root = Path(base_dir) if base_dir is not None else Path(self.task.template_dir)
        found: dict[str, str] = {}
        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
                for name in sorted(filenames):
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    if not any(fnmatch.fnmatchcase(rel, g) for g in self.task.editable):
                        continue
                    st = os.lstat(full)
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    with open(full, "rb") as fh:
                        found[rel] = fh.read(EDITABLE_FILE_BYTES).decode("utf-8", errors="replace")
                    if len(found) >= EDITABLE_FILE_COUNT:
                        return found
        except OSError:
            pass
        return found

    def _incumbent_summary(self) -> str:
        """Dev numbers only: the incumbent's id, dev score and replicated flag, plus its fenced hypothesis."""
        view = self._view()
        inc = view.incumbent
        grade = inc.primary_grade if inc is not None else None
        if inc is None or grade is None or grade.score is None:
            return "No incumbent yet."
        metric = sanitize(self.task.primary_metric, 60)
        if inc.exp_id == view.baseline_id:
            return f"Incumbent {inc.exp_id}: the unmodified baseline, {metric} {grade.score:.6g} (dev split, std {grade.std:.4g})."
        return (f"Incumbent {inc.exp_id}: {metric} {grade.score:.6g} (dev split, std {grade.std:.4g}), "
                f"{'replicated' if inc.replicated else 'not replicated'}. Its hypothesis:\n"
                + fence(sanitize(inc.hypothesis, 300), f"hyp_{inc.exp_id}", max_chars=300))

    # ---- workspace ----------------------------------------------------------------------------
    def _phase_workspace(self, ctx: _Lane) -> str:
        dest = self._exp_dir(ctx.exp_id) / "workspace"
        if os.path.lexists(dest):
            shutil.rmtree(dest)             # our own earlier, interrupted attempt
        ws = sandbox.create_workspace(self.task, self.run_dir, ctx.exp_id, base=ctx.base_dir)
        ctx.ws = ws
        if not ctx.skip_ws_event:
            self._append(Kind.EXP_WORKSPACE, "engine", {
                "exp_id": ctx.exp_id, "path": f"exp/{ctx.exp_id}/workspace", "base_hash": sandbox.base_hash(ws)})
        return "implement"

    def _load_workspace(self, exp_id: str) -> sandbox.Workspace:
        """Rebuild the ``Workspace`` handle of an existing experiment from its ``base.json``."""
        exp_dir = self._exp_dir(exp_id)
        meta = json.loads((exp_dir / "base.json").read_text(encoding="utf-8"))
        hashes = {str(k): str(v) for k, v in dict(meta["base_hashes"]).items()}
        return sandbox.Workspace(exp_id=exp_id, path=exp_dir / "workspace", base_hashes=hashes, tree_hash=str(meta["tree_hash"]))

    def _verify_workspace(self, exp: ExperimentView) -> tuple[sandbox.Workspace, sandbox.WorkspaceDiff, list[str]]:
        """The workspace as it is on disk, its diff, and the problems against what the ledger recorded (empty = intact).

        ``diff_hash`` commits to the content of every changed file, so equality with the recorded value proves the
        editable tree is what was reviewed and graded.
        """
        ws = self._load_workspace(exp.exp_id)
        diff = sandbox.diff_workspace(self.task, ws)
        problems = list(diff.violations)
        if exp.diff_hash and diff.diff_hash != exp.diff_hash:
            problems.append("workspace_modified")
        return ws, diff, problems

    # ---- implement / repair -------------------------------------------------------------------
    def _call_implementer(self, ctx: _Lane, *, repair_findings: Sequence[dict] = (), prior_diff: str = "") -> tuple[
            ImplementResult, sandbox.WorkspaceDiff, list[tuple[str, list[str]]]]:
        """Run the implementer with ``trusted_hash`` checked before and after; returns its result, the diff and violations."""
        before = _trusted_hash(self.task)
        ictx = ImplementContext(task=self.task, exp_id=ctx.exp_id, plan=ctx.plan, lane=ctx.spec, workspace=ctx.ws.path,
                                memory_digest=self.memory.digest(), repair_findings=[dict(f) for f in repair_findings],
                                prior_diff=prior_diff, emit_claim=self._claim_emitter(ctx))
        result = self._actors(ctx.spec).implementer.implement(ictx)
        result = self._clean_result(result)
        groups: list[tuple[str, list[str]]] = []
        if _trusted_hash(self.task) != before:
            groups.append(("trusted", ["trusted"]))
        diff = sandbox.diff_workspace(self.task, ctx.ws)
        walls = [v for v in diff.violations if not v.startswith("probe:")]
        probes = [v for v in diff.violations if v.startswith("probe:")]
        if walls:
            groups.append(("workspace", walls[:50]))
        if probes:
            groups.append(("trusted_probe", probes[:50]))
        return result, diff, groups

    @staticmethod
    def _clean_result(result: Any) -> ImplementResult:
        """An implementer's result reduced to well-typed, sanitised fields: a model-driven worker may return anything, and a
        bad ``steps`` or ``worker`` must not cost the lane its (already graded-to-be) work."""
        if not isinstance(result, ImplementResult):
            return ImplementResult(summary="")
        steps = result.steps if isinstance(result.steps, int) and not isinstance(result.steps, bool) else 0
        return ImplementResult(summary=sanitize(result.summary, 2000), steps=max(0, min(steps, 100_000)),
                               worker=_identity(result.worker), raw_log=sanitize(result.raw_log, 8000),
                               self_report=sanitize(result.self_report, 1000))

    def _phase_implement(self, ctx: _Lane) -> "str | LaneResult":
        result, diff, groups = self._call_implementer(ctx)
        ctx.worker = (_identity(result.worker) or _identity(getattr(self._actors(ctx.spec).implementer, "identity", ""))
                      or "implementer")
        self._write_artifacts(ctx.exp_id, diff.unified, result, 0)
        if groups:
            return self._violation(ctx, groups)
        if not (diff.changed or diff.added or diff.removed):
            # nothing to evaluate: grading an unchanged tree would only re-measure the incumbent's base and call it a result.
            # No ``implemented`` event is written (every empty change would share one idempotency key and look like a duplicate)
            ctx.pending = (Outcome.UNRESOLVED, NO_CHANGE_NOTE)
            return "record"
        idem = short_hash(f"{sandbox.base_hash(ctx.ws)}|{diff.diff_hash}", 16)
        with self._lock:
            twin = self._idem_twin(self._view(), ctx.exp_id, idem)
            if twin:
                detail = f"identical change to {twin}"
                self._append(Kind.EXP_BLOCKED, "engine", {
                    "exp_id": ctx.exp_id, "lane": ctx.spec.lane, "hypothesis": ctx.spec.hypothesis,
                    "hyp_fingerprint": self.memory.fingerprint(ctx.spec.hypothesis), "reason": "duplicate", "detail": detail,
                    "dup_of": twin})
                return self._lane_result(ctx.exp_id, notes=detail)
            self._append(Kind.EXP_IMPLEMENTED, f"implementer:{ctx.spec.lane}", {
                "exp_id": ctx.exp_id, "worker": ctx.worker, "diff_hash": diff.diff_hash,
                "files_changed": sorted(set(diff.changed + diff.added + diff.removed))[:50],
                "diff_summary": diff.summary, "idem_key": idem, "attempt": 1, "steps": max(0, int(result.steps or 0)),
                "tree_hash": diff.tree_hash, "plan_summary": self._plan_summary(ctx.plan)})
        ctx.tree_hash, ctx.last_diff, ctx.attempt = diff.tree_hash, diff.unified, 1
        return "run"

    @staticmethod
    def _plan_summary(plan: ExperimentPlan) -> str:
        """The plan the implementer and the reviewer work from (approach, then expected effect), sanitised and capped at
        ``PLAN_SUMMARY_CHARS``. It holds only what the designer wrote from dev-split facts; it is recorded so the plan is
        auditable and so a resumed lane continues from it instead of from the supervisor's hint."""
        approach = sanitize(plan.approach, PLAN_SUMMARY_CHARS)
        effect = sanitize(plan.expected_effect, PLAN_SUMMARY_CHARS)
        text = f"{approach} Expected effect: {effect}" if approach and effect else approach or effect
        return sanitize(text, PLAN_SUMMARY_CHARS)

    @staticmethod
    def _idem_twin(view: LedgerView, exp_id: str, idem: str) -> Optional[str]:
        """Another recorded or in-flight experiment with this idempotency key (an infra-failed attempt does not count)."""
        for exp in view.experiments.values():
            if exp.exp_id != exp_id and exp.idem_key == idem and exp.blocked is None and exp.dev_outcome != Outcome.INFRA:
                return exp.exp_id
        return None

    def _phase_repair(self, ctx: _Lane) -> "str | LaneResult":
        self._check_stop(work=True)
        exp = self._view().exp(ctx.exp_id)
        last = exp.last_review
        findings = [dict(f) for f in (last.findings if last else [])]
        result, diff, groups = self._call_implementer(ctx, repair_findings=findings,
                                                      prior_diff=self._safe_diff(ctx.last_diff, ctx.exp_id))
        rnd = len(exp.repairs) + 1
        self._write_artifacts(ctx.exp_id, diff.unified, result, rnd)
        if groups:
            return self._violation(ctx, groups)
        if diff.diff_hash == exp.diff_hash:
            ctx.pending = (Outcome.UNRESOLVED, "the repair changed nothing")
            return "record"
        self._append(Kind.EXP_REPAIRED, f"implementer:{ctx.spec.lane}", {
            "exp_id": ctx.exp_id, "repair_round": rnd, "diff_hash": diff.diff_hash,
            "addressed": [f["id"] for f in findings if f.get("severity") != "info"][:20]})
        ctx.tree_hash, ctx.last_diff, ctx.attempt, ctx.hard_done = diff.tree_hash, diff.unified, 1, False
        return "run"

    def _claim_emitter(self, ctx: _Lane) -> Callable[[str, str], str]:
        def emit(claim_type: str, text: str) -> str:
            kind = claim_type if claim_type in CLAIM_TYPES else "other"
            prefix = f"{ctx.exp_id}.c"
            with self._claim_lock:
                used = [int(k[len(prefix):]) for k in self._view().claims if k.startswith(prefix) and k[len(prefix):].isdigit()]
                claim_id = f"{prefix}{max(used, default=0) + 1}"
                self._append(Kind.CLAIM, f"implementer:{ctx.spec.lane}", {
                    "claim_id": claim_id, "exp_id": ctx.exp_id, "claim_type": kind, "text": sanitize(text, 1000),
                    "cites": [], "alias": None})
            return claim_id
        return emit

    def _write_artifacts(self, exp_id: str, diff_text: str, result: ImplementResult, rnd: int) -> None:
        """``diff.patch`` and ``worker.log`` are written by the engine only, with secrets scrubbed."""
        exp_dir = self._exp_dir(exp_id)
        _atomic_write(exp_dir / "diff.patch", secrets.scrub_text(diff_text))
        log = exp_dir / "worker.log"
        try:
            old = log.read_text(encoding="utf-8", errors="replace") if log.is_file() and not log.is_symlink() else ""
        except OSError:
            old = ""
        entry = (f"== round {rnd}: steps={int(result.steps or 0)}\n{sanitize(result.summary, 2000)}\n"
                 f"{sanitize(result.raw_log, 8000)}\n")
        _atomic_write(log, secrets.scrub_text((old + entry)[-WORKER_LOG_BYTES:]))

    @staticmethod
    def _safe_diff(text: str, exp_id: str) -> str:
        return fence(sanitize(text, 40000), f"diff_{exp_id}", max_chars=40000)

    def _violation(self, ctx: _Lane, groups: Sequence[tuple[str, Sequence[str]]]) -> LaneResult:
        """Write ``experiment.protected_violation`` per kind and end the lane as ``invalid``."""
        for kind, paths in groups:
            self._append(Kind.EXP_PROTECTED, "engine", {"exp_id": ctx.exp_id, "paths": list(paths)[:50], "kind": _vkind(kind)})
        ctx.pending = (Outcome.INVALID, "protected path violation (" + ", ".join(sorted({_vkind(k) for k, _ in groups})) + ")")
        return self._phase_record(ctx)

    # ---- run / grade --------------------------------------------------------------------------
    def _hooks(self, exp_id: str, purpose: str, attempt: int) -> grading.EvalHooks:
        isolation = self._isolation_label()

        def started(run_idx: int, seed: int, pid: Optional[int]) -> None:
            self._append(Kind.EXP_RUN_STARTED, "engine", {
                "exp_id": exp_id, "run_idx": run_idx, "purpose": purpose, "seed": seed, "cmd": list(self.task.run_cmd),
                "pid": pid, "attempt": attempt, "isolation": isolation})

        def beat(run_idx: int, output_bytes: int, cpu: Optional[float]) -> None:
            self._append(Kind.EXP_HEARTBEAT, "engine", {
                "exp_id": exp_id, "run_idx": run_idx, "output_bytes": output_bytes, "cpu_sec": cpu})

        def ended(run_idx: int, seed: int, run: sandbox.RunResult) -> None:
            self._append(Kind.EXP_RUN_FINISHED, "engine", {
                "exp_id": exp_id, "run_idx": run_idx, "purpose": purpose, "seed": seed, "exit_code": run.exit_code,
                "duration_sec": round(float(run.duration_sec), 3), "classification": run.classification,
                "stalled": bool(run.stalled), "stdout_tail": _tail(run.stdout), "stderr_tail": _tail(run.stderr),
                "note": sanitize(run.note, 300)})

        def stalled(run_idx: int, idle: float) -> None:
            self._append(Kind.WATCHDOG_STALLED, "watchdog", {"exp_id": exp_id, "run_idx": run_idx, "idle_sec": round(float(idle), 3)})

        return grading.EvalHooks(on_run_start=started, on_heartbeat=beat, on_run_end=ended, on_stalled=stalled)

    def _evaluate_once(self, exp_id: str, ws: sandbox.Workspace, seeds: Sequence[int], purpose: str, expected_tree: str,
                       attempt: int) -> grading.EvalResult:
        self._check_task_unchanged()
        start = len(self._view().exp(exp_id).runs)          # run_idx is the count of existing run_started events, never reused
        result = grading.evaluate(self.task, ws, seeds, purpose=purpose, run_dir=self.run_dir, expected_tree_hash=expected_tree,
                                  hooks=self._hooks(exp_id, purpose, attempt), run_idx_start=start, attempt=attempt,
                                  isolation=self._isolation, heartbeat_sec=self._heartbeat_sec,
                                  expected_grader_hash=self._grader_hash())
        # a run that edited the template (or the scripted bank) for FUTURE workspaces is caught here, on the evaluation that
        # did it, so the guilty experiment is the one that ends invalid and not whichever lane evaluates next
        self._check_task_unchanged()
        return result

    def _evaluate(self, exp_id: str, ws: sandbox.Workspace, seeds: Sequence[int], purpose: str, expected_tree: str, *,
                  attempt: int = 1, retry_infra: bool = True) -> _Eval:
        """Evaluate once; an infrastructure failure is retried once as a whole with ``attempt + 1``.

        An ``experiment.graded`` event is written for the final result unless it was an infrastructure failure (which says
        nothing about the candidate, so nothing is graded and the experiment stays eligible for another try).
        """
        res = self._evaluate_once(exp_id, ws, seeds, purpose, expected_tree, attempt)
        if res.failure_class == "infra" and retry_infra:
            self._check_stop()
            attempt += 1
            res = self._evaluate_once(exp_id, ws, seeds, purpose, expected_tree, attempt)
        if res.failure_class != "infra":
            grade = res.grade
            self._append(Kind.EXP_GRADED, "engine", {
                "exp_id": exp_id, "purpose": purpose, "split": Purpose.split(purpose), "grade": grade.to_dict(),
                "grader_hash": grade.grader_hash, "seeds": list(grade.seeds), "conditions_hash": res.conditions_hash,
                "outputs_hash": short_hash(canonical_json(res.outputs_hashes))})
        return _Eval(res, attempt)

    def _phase_run(self, ctx: _Lane) -> "str | LaneResult":
        self._check_stop(work=True)
        out = self._evaluate(ctx.exp_id, ctx.ws, self.task.seeds, Purpose.PRIMARY, ctx.tree_hash,
                             attempt=ctx.attempt, retry_infra=ctx.attempt == 1)
        if out.failure_class is not None:
            ctx.pending = self._failure_outcome(out.failure_class)
            return "record"
        return "review"

    # ---- review -------------------------------------------------------------------------------
    def _phase_review(self, ctx: _Lane) -> "str | LaneResult":
        return self._review(ctx, hard=False)

    def _phase_review_hard(self, ctx: _Lane) -> "str | LaneResult":
        return self._review(ctx, hard=True)

    def _review(self, ctx: _Lane, *, hard: bool) -> "str | LaneResult":
        reviewer = self.suite.reviewer_hard if hard else self.suite.reviewer
        verdict, findings = self._review_once(ctx, reviewer, hard)
        if hard and verdict == "accept" and not any(f["severity"] == "block" for f in findings):
            ctx.hard_done = True
        return self._after_review(ctx, verdict, findings, hard=hard)

    def _review_once(self, ctx: _Lane, reviewer: Any, hard: bool) -> tuple[str, list[dict]]:
        view = self._view()
        exp = view.exp(ctx.exp_id)
        grade = exp.grade_for(Purpose.PRIMARY).grade
        prior = [dict(f) for f in (exp.last_review.findings if exp.last_review else [])]
        rctx = ReviewContext(task=self.task, exp_id=ctx.exp_id, plan=ctx.plan,
                             diff_text=self._safe_diff(ctx.last_diff, ctx.exp_id),
                             grade_summary=self._grade_summary(view, grade, None),
                             incumbent_summary=self._incumbent_summary(), run_summary=self._run_summary(ctx.exp_id),
                             implementer_identity=ctx.worker, repair_round=len(exp.repairs), prior_findings=prior)
        own = _identity(getattr(reviewer, "identity", ""))
        unavailable_call = False
        try:
            got = reviewer.review(rctx)
        except ModelError:
            unavailable_call = True
            got = ReviewVerdict(verdict="reject", reviewer=own, findings=[{
                "id": "reviewer_unavailable", "severity": "info", "evidence": "",
                "text": "No review was performed: the model could not be reached."}])
        verdict = got.verdict if isinstance(got, ReviewVerdict) and got.verdict in VERDICTS else None
        findings = self._clean_findings(got.findings if isinstance(got, ReviewVerdict) else [])
        if verdict is None:              # not a usable verdict: the safe reading is "no review happened"
            verdict = "reject"
            findings = [{"id": "reviewer_unavailable", "severity": "info", "text": "The reviewer returned no usable verdict.",
                         "evidence": ""}]
        name = _identity(got.reviewer) if isinstance(got, ReviewVerdict) else ""
        name = name or own
        actors = self._actors(ctx.spec)
        implementer = _identity(getattr(actors.implementer, "identity", ""))
        independent = bool(name) and name not in (ctx.worker, implementer)
        models = getattr(self.suite, "models", None) or {}
        role = "reviewer_hard" if hard else "reviewer"
        # the implementer's model is the one that wrote THIS lane's patch: "scripted" for a red-team lane, never the real one
        impl_models = (getattr(actors, "models", None) or {}) if actors is not self.suite else models
        configured, impl_model = _identity(models.get(role)), _identity(impl_models.get("implementer"))
        # the model that ANSWERED, not the one that was asked: a client with a fallback can answer with another model. A review
        # that never got an answer (reviewer_unavailable) has no answering model, whatever a reused reviewer object remembers
        answered = "" if unavailable_call or any(f["id"] == "reviewer_unavailable" for f in findings) \
            else self._answering_model(reviewer)
        review_model = answered or configured
        if answered:
            basis = "model" if impl_model and not _same_model(answered, impl_model) else "context"
        else:
            basis = getattr(reviewer, "independent_basis", None)
            if basis not in ("model", "context"):
                basis = "model" if review_model and impl_model and review_model != impl_model else "context"
        self._append(Kind.EXP_REVIEWED, f"reviewer:{ctx.spec.lane}", {
            "exp_id": ctx.exp_id, "review_round": len(exp.reviews), "reviewer": name or f"reviewer:{ctx.spec.lane}",
            "reviewer_model": review_model, "implementer_model": impl_model, "independent": independent,
            "independent_basis": basis, "verdict": verdict, "findings": findings})
        return verdict, findings

    @staticmethod
    def _answering_model(reviewer: Any) -> str:
        """``reviewer.answering_model`` (the model of its latest call in this thread), sanitised; "" when absent or unusable."""
        try:
            value = getattr(reviewer, "answering_model", None)
        except NightshiftError:
            raise
        except Exception:  # noqa: BLE001 - a misbehaving property must not cost the lane its review
            return ""
        return _identity(value) if isinstance(value, str) else ""

    @staticmethod
    def _clean_findings(raw: Any) -> list[dict]:
        """Model-written findings reduced to ``{id, severity, text, evidence}`` with safe ids and sanitised text."""
        out: list[dict] = []
        used: set[str] = set()
        for item in list(raw)[:MAX_FINDINGS * 2] if isinstance(raw, (list, tuple)) else []:
            if not isinstance(item, Mapping):
                continue
            fid = _slug(item.get("id")) or f"finding_{len(out) + 1}"
            while fid in used:
                fid += "_"
            used.add(fid)
            severity = str(item.get("severity", "")).strip().lower()
            out.append({"id": fid, "severity": severity if severity in SEVERITIES else "warn",
                        "text": sanitize(item.get("text", ""), 400), "evidence": sanitize(item.get("evidence", ""), 300)})
            if len(out) >= MAX_FINDINGS:
                break
        return out

    def _after_review(self, ctx: _Lane, verdict: str, findings: list[dict], *, hard: bool) -> "str | LaneResult":
        exp = self._view().exp(ctx.exp_id)
        blocking = any(f["severity"] == "block" for f in findings)
        if verdict == "accept" and not blocking:
            if not hard and "review_hard" in ctx.stages and not ctx.hard_done:
                if self.suite.reviewer_hard is None:
                    note = "review_hard skipped: no hard reviewer is configured"
                    if note not in ctx.notes:
                        ctx.notes.append(note)
                else:
                    return "review_hard"
            return "record"
        unavailable = any(f["id"] == "reviewer_unavailable" for f in findings)
        if unavailable:
            ctx.pending = (Outcome.UNRESOLVED, "no review could be performed")
        elif verdict == "reject" and not blocking:
            ctx.pending = (Outcome.UNRESOLVED, "the reviewer rejected the change without asking for a repair")
        elif "repair" not in ctx.stages:
            ctx.pending = (Outcome.UNRESOLVED, "the reviewer asked for a repair but the lane has no repair stage")
        elif len(exp.repairs) >= int(self.limits.max_repairs):
            ctx.pending = (Outcome.UNRESOLVED, "the repair limit was reached")
        elif self._same_findings_after_repair(exp):
            ctx.pending = (Outcome.UNRESOLVED, "the same review findings came back after a repair")
        else:
            return "repair"
        return "record"

    @staticmethod
    def _same_findings_after_repair(exp: ExperimentView) -> bool:
        if len(exp.reviews) < 2:
            return False
        cur = exp.reviews[-1]
        # review and review_hard alternate in one lane: the same findings "twice" means twice from the same reviewer
        prev = next((r for r in reversed(exp.reviews[:-1]) if r.reviewer == cur.reviewer), exp.reviews[-2])
        ids = sorted(str(f.get("id")) for f in cur.findings)
        if not ids or ids != sorted(str(f.get("id")) for f in prev.findings):
            return False
        return any(prev.seq < int(r.get("seq", 0)) < cur.seq for r in exp.repairs)

    # ---- summaries handed to reviewers (dev numbers only) -------------------------------------
    def _sigma_prior(self, view: LedgerView) -> float:
        base = view.baseline
        grade = base.primary_grade if base is not None else None
        std = _finite(grade.std) if grade is not None else None
        return max(0.0, std) if std is not None else 0.0

    def _decide(self, view: LedgerView, grade: Optional[GradeResult], inc: Optional[ExperimentView]):
        """``decide_outcome`` of a dev grade against the incumbent's dev grade, or None when it cannot be computed."""
        if grade is None or not grade.valid or inc is None or inc.primary_grade is None:
            return None
        try:
            return grading.decide_outcome(self.task, grade, inc.primary_grade, sigma_prior=self._sigma_prior(view))
        except (ValueError, TypeError, ArithmeticError):
            return None

    def _grade_summary(self, view: LedgerView, grade: GradeResult, failure_class: Optional[str]) -> dict:
        inc = view.incumbent
        dec = self._decide(view, grade, inc)
        ok = dec is not None and dec.outcome != "invalid"
        inc_grade = inc.primary_grade if inc is not None else None
        return {"valid": grade.valid, "score": grade.score, "std": grade.std, "per_seed": list(grade.per_seed),
                "metrics": dict(grade.metrics), "incumbent_score": inc_grade.score if inc_grade is not None else None,
                "delta": dec.delta if ok else None, "margin": dec.margin if ok else None,
                "train_delta": dec.train_delta if ok else None, "overfit_suspect": bool(dec.overfit_suspect) if ok else False,
                "failure_class": failure_class}

    def _run_summary(self, exp_id: str) -> str:
        events = [e for e in self.ledger.events() if e.data.get("exp_id") == exp_id]
        start = max((i for i, e in enumerate(events) if e.kind in (Kind.EXP_IMPLEMENTED, Kind.EXP_REPAIRED)), default=0)
        done = [e for e in events[start:] if e.kind == Kind.EXP_RUN_FINISHED
                and e.data.get("purpose") in (Purpose.PRIMARY, Purpose.BASELINE)]
        if not done:
            return "No run has finished yet."
        counts: dict[str, int] = {}
        for e in done:
            key = sanitize(str(e.data.get("classification")), 16)
            counts[key] = counts.get(key, 0) + 1
        total = sum(float(e.data.get("duration_sec") or 0) for e in done)
        lines = [f"{len(done)} run(s): " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
                 + f"; {total:.2f}s in total; {sum(1 for e in done if e.data.get('stalled'))} stalled."]
        tail = sanitize(str(done[-1].data.get("stdout_tail") or ""), 400)
        if tail:
            lines.append("Last run output (data):\n" + fence(tail, "run_output", max_chars=400))
        return "\n".join(lines)

    # ---- record -------------------------------------------------------------------------------
    def _phase_record(self, ctx: _Lane) -> LaneResult:
        view = self._view()
        exp = view.exp(ctx.exp_id)
        if exp.status == ExpStatus.RECORDED:
            return self._lane_result(ctx.exp_id)
        rec = exp.grade_for(Purpose.PRIMARY)
        if rec is not None and rec.seq < self._tree_seq(ctx.exp_id):
            rec = None                      # that grade belongs to a tree the last repair replaced
        grade = rec.grade if rec is not None else None
        inc = view.incumbent
        dec = self._decide(view, grade, inc)
        outcome, note = ctx.pending if ctx.pending is not None else (None, "")
        if outcome is None:
            if dec is None:
                outcome, note = Outcome.INVALID, "the grade is not comparable with the incumbent's"
            elif dec.outcome == "invalid":
                outcome, note = Outcome.INVALID, "not comparable with the incumbent: " + sanitize(getattr(dec, "detail", ""), 120)
            else:
                outcome = dec.outcome
        usable = dec is not None and dec.outcome != "invalid"
        note = "; ".join([n for n in [note, *ctx.notes] if n])[:300]
        self._append(Kind.EXP_RECORDED, "engine", {
            "exp_id": ctx.exp_id, "phase": "lane", "outcome": outcome, "delta": dec.delta if usable else None,
            "vs_exp_id": inc.exp_id if usable else None, "noise_margin": dec.margin if usable else None,
            "train_delta": dec.train_delta if usable else None, "train_margin": dec.train_margin if usable else None,
            "overfit_suspect": bool(dec.overfit_suspect) if usable else False, "replicated": False,
            "sigma_prior": self._sigma_prior(view), "n": dec.n if usable else 0, "s_eff": dec.s_eff if usable else None,
            "t": dec.t if usable else None, "sealed_delta": None, "sealed_margin": None, "note": note})
        lesson = self._lesson(ctx, outcome, note, dec if usable else None, inc)
        if lesson is not None:
            self._check_crash()
            self.memory.add_lesson(lesson[0], kind="observation", source_exps=[ctx.exp_id], classification=lesson[1],
                                   actor="engine")
        self._after_record(reflect=not ctx.stopping)
        return self._lane_result(ctx.exp_id, notes="awaiting replication" if outcome == Outcome.IMPROVED else note)

    def _tree_seq(self, exp_id: str) -> int:
        """Sequence number of the event that last set this experiment's tree (``implemented`` or ``repaired``)."""
        for ev in reversed(self.ledger.events()):
            if ev.kind in (Kind.EXP_IMPLEMENTED, Kind.EXP_REPAIRED) and ev.data.get("exp_id") == exp_id:
                return ev.seq
        return 0

    def _lesson(self, ctx: _Lane, outcome: str, note: str, dec: Any, inc: Optional[ExperimentView]) -> Optional[tuple[str, str]]:
        """Text and classification of the lesson for a non-improvement. Dev numbers only; never sealed numbers."""
        eid = ctx.exp_id
        if outcome == Outcome.IMPROVED:
            return None
        if outcome == Outcome.INFRA:
            return f"{eid} hit an environment failure; its hypothesis was not tested.", "infra"
        if note.startswith((BUDGET_NOTE, HALT_NOTE, "engine_error")):
            return None
        if note.startswith(NO_CHANGE_NOTE):
            return (f"{eid}: the idea could not be turned into a change (the implementer edited nothing), so the hypothesis "
                    "was not tested."), "infra"
        if outcome in (Outcome.NO_GAIN, Outcome.WORSE) and dec is not None:
            text = (f"{eid} ended {'worse than' if outcome == Outcome.WORSE else 'without a gain over'} the incumbent "
                    f"{inc.exp_id if inc else 'n/a'} on the dev split (dev delta {_signed(dec.delta)}, margin {_finite(dec.margin) or 0:.4g}).")
            if dec.train_delta is not None:
                text += f" Train delta {_signed(dec.train_delta)}."
            if dec.overfit_suspect:
                text += " The training error improved without a held-out gain: possible overfitting."
            return text, "model"
        if outcome == Outcome.FAILED:
            return f"{eid} did not produce a gradable result ({note}); fix the code before retrying the idea.", "model"
        if outcome == Outcome.INVALID:
            return (f"{eid} was rejected as invalid ({note}); its result was discarded and only editable files may change."), "model"
        if outcome == Outcome.UNRESOLVED:
            return f"{eid} was left unresolved ({note}).", "model"
        return None

    def _after_record(self, *, reflect: bool = True) -> None:
        """After every recorded experiment: memory.md, wall time, and (once per streak) reflection.

        ``reflect=False`` is used while the run is being stopped: a halted run does not spend model calls on reflection."""
        self._check_crash()
        self.regenerate_memory()
        self.gate.charge_wall()
        if reflect:
            self._maybe_auto_reflect()

    def regenerate_memory(self) -> None:
        """Atomically rewrite ``run_dir/memory.md`` from the ledger (dev numbers only)."""
        self.memory.write_markdown(self.run_dir / "memory.md")

    def _lane_result(self, exp_id: str, notes: str = "") -> LaneResult:
        """A ``LaneResult`` from the ledger. Only dev numbers: ``dev_outcome`` and ``delta`` are the lane-phase values."""
        exp = self._view().exp(exp_id)
        grade = exp.primary_grade
        review = exp.last_review
        blocked = exp.blocked.get("reason") if exp.blocked else None
        scored = grade is not None and grade.valid and exp.dev_outcome not in (Outcome.FAILED, Outcome.INVALID, Outcome.INFRA)
        return LaneResult(exp_id=exp.exp_id, lane=exp.lane, status=exp.status, outcome=exp.dev_outcome, delta=exp.delta,
                          score=grade.score if scored else None, replicated=False,
                          review_verdict=review.verdict if review else None, blocked_reason=blocked, notes=notes[:300])

    # =============================================================================================
    # replication and promotion (sealed split)
    # =============================================================================================
    def replicate(self, exp_id: str) -> LaneResult:
        """Re-grade an improved experiment on the sealed split and promote it if it replicates (contract 7.3).

        The result has ``score`` and ``delta`` None and the boolean ``replicated``; a refusal has ``replicated`` None.
        """
        self._check_stop()
        target = exp_id if isinstance(exp_id, str) else ""
        spec = LaneSpec(lane="replicate", hypothesis=f"replicate {_slug(target, 20) or 'unknown'}", replicate_of=target or "invalid")
        with self._lock:
            reasons = _gates().launch_gate(self._view(), spec, self.memory, gate=self.gate, run_dir=self.run_dir,
                                           limits=self.limits, task=self.task)
            if not self.task.sealed:
                reasons.append("this task has no independent re-check")
            if self._is_attacker(self._view().exp(target)):
                reasons.append(ATTACKER_NOTE)       # a scripted attacker's patch must never become the incumbent or the verdict
            if target in self._replicating:
                reasons.append(f"{_slug(target, 20)} is already being replicated")
            if not reasons:
                try:
                    self.gate.reserve_replication(target)
                except BudgetExceeded:
                    reasons.append("budget exhausted: replication")
            if reasons:
                return self._block_replication(spec, target, reasons)
            self._replicating.add(target)
        try:
            return self._replicate_core(target)
        finally:
            with self._lock:
                self._replicating.discard(target)

    def _block_replication(self, spec: LaneSpec, target: str, reasons: Sequence[str]) -> LaneResult:
        if any(r.startswith("HALT file present") for r in reasons):
            raise HaltRequested("HALT file present in the run directory")
        reason = "budget" if any(r.startswith("budget exhausted") for r in reasons) else "not_replicable"
        detail = (f"target {_slug(target, 20) or 'unknown'}: " + "; ".join(reasons))[:500]
        self._append(Kind.EXP_BLOCKED, "engine", {
            "exp_id": None, "lane": "replicate", "hypothesis": spec.hypothesis,
            "hyp_fingerprint": self.memory.fingerprint(spec.hypothesis), "reason": reason, "detail": detail, "dup_of": None})
        return LaneResult(exp_id=target if _EXP_ID.match(target) else "", lane="replicate", status=ExpStatus.BLOCKED,
                          replicated=None, blocked_reason=reason, notes=detail)  # type: ignore[arg-type]

    def _replicate_core(self, exp_id: str) -> LaneResult:
        try:
            return self._replicate_steps(exp_id)
        except HaltRequested:
            try:
                self._interrupt_open_runs([exp_id], "the replication was stopped before the run ended")
            except (SimulatedCrash, LedgerCorrupt):
                raise
            except Exception:  # noqa: BLE001 - bookkeeping must not replace the halt
                pass
            raise
        except (SimulatedCrash, BudgetExceeded):
            raise
        except ProtectedPathViolation as exc:
            if self._replication_settled(exp_id):
                return self._settled_result(exp_id, "replicated, not promoted: protected path violation")
            self._append(Kind.EXP_PROTECTED, "engine", {"exp_id": exp_id, "paths": [str(p) for p in exc.paths][:50],
                                                       "kind": _vkind(exc.kind)})
            self._void_replication(exp_id, "protected path violation")
            return self._finish_replication(exp_id, Outcome.INVALID, False, "protected path violation", lesson="failed")
        except Exception as exc:  # noqa: BLE001 - a replication must always end in a record
            if self._replication_settled(exp_id):      # the result is recorded; only what came after it (promotion) failed
                return self._settled_result(exp_id, f"replicated, not promoted: engine_error:{type(exc).__name__}")
            return self._finish_replication(exp_id, Outcome.INFRA, False, f"engine_error:{type(exc).__name__}", lesson="infra")

    def _replication_settled(self, exp_id: str) -> bool:
        """True when the latest replication of ``exp_id`` already has its (non-infrastructure) ``recorded`` event."""
        for ev in reversed(self.ledger.events()):
            if ev.data.get("exp_id") != exp_id:
                continue
            if ev.kind == Kind.EXP_RECORDED and ev.data.get("phase") == "replication":
                return ev.data.get("outcome") != Outcome.INFRA
            if ev.kind == Kind.EXP_GRADED and ev.data.get("purpose") == Purpose.REPLICATION:
                return False
        return False

    def _settled_result(self, exp_id: str, note: str) -> LaneResult:
        exp = self._view().exp(exp_id)
        replicated = bool(exp is not None and exp.replicated)
        return LaneResult(exp_id=exp_id, lane=exp.lane if exp else "", status=ExpStatus.RECORDED,
                          outcome=Outcome.IMPROVED if replicated else Outcome.NO_GAIN, delta=None, score=None,
                          replicated=replicated, notes=note[:300])

    def _void_replication(self, exp_id: str, reason: str) -> None:
        """Make a replication that can never finish (a protected-path violation) terminal.

        Without a sealed grade the candidate would still be ``awaiting_replication`` and could be replicated again and again.
        An invalid sealed grade carries no number and ends the wait, exactly as a crash on the sealed seeds does.
        """
        exp = self._view().exp(exp_id)
        if exp is None or exp.grade_for(Purpose.REPLICATION) is not None:
            return
        seeds = [int(x) for x in self.task.replication_seeds]
        grader_hash = self._grader_hash()
        grade = GradeResult(valid=False, score=None, std=0.0, per_seed=[], metrics={}, per_seed_metrics=[],
                            grader_hash=grader_hash, notes="", error=sanitize(reason, 200), split="sealed", n_eval=0, seeds=seeds)
        self._append(Kind.EXP_GRADED, "engine", {
            "exp_id": exp_id, "purpose": Purpose.REPLICATION, "split": "sealed", "grade": grade.to_dict(),
            "grader_hash": grader_hash, "seeds": seeds, "outputs_hash": "",
            "conditions_hash": _conditions_hash(self.task, seeds, "sealed", grader_hash=grader_hash)})

    def _replicate_steps(self, exp_id: str) -> LaneResult:
        view = self._view()
        exp, base = view.exp(exp_id), view.baseline
        if exp is None or base is None:
            return self._finish_replication(exp_id, Outcome.INFRA, False, "the baseline or the candidate is missing", lesson="infra")
        start_incumbent = view.incumbent_id

        if self._ensure_baseline_sealed(base.exp_id) is not None:
            return self._finish_replication(exp_id, Outcome.INFRA, False, "the baseline's sealed grade could not be computed",
                                            lesson="infra")
        self._check_stop()
        exp = self._view().exp(exp_id)
        if exp.grade_for(Purpose.REPLICATION) is None:
            ws, diff, problems = self._verify_workspace(exp)
            if problems:
                self._append(Kind.EXP_PROTECTED, "engine", {"exp_id": exp_id, "paths": problems[:50], "kind": "workspace"})
                self._void_replication(exp_id, "the workspace no longer matches the graded tree")
                return self._finish_replication(exp_id, Outcome.INVALID, False, "the workspace no longer matches the graded tree",
                                                lesson="failed")
            attempt = self._next_attempt(exp_id, (Purpose.REPLICATION,))
            out = self._evaluate(exp_id, ws, self.task.replication_seeds, Purpose.REPLICATION, diff.tree_hash,
                                 attempt=attempt, retry_infra=attempt == 1)
            if out.failure_class == "infra":
                return self._finish_replication(exp_id, Outcome.INFRA, False, "infrastructure failure during replication",
                                                lesson="infra")
        else:
            ws = self._load_workspace(exp_id)
        if not self._view().exp(exp_id).grade_for(Purpose.REPLICATION).grade.valid:
            return self._finish_replication(exp_id, Outcome.FAILED, False, "the sealed evaluation did not complete",
                                            lesson="failed")

        with self._promo_lock:
            view = self._view()
            vs = self._comparison_base(view)
            dec = _gates().sealed_decision(self.task, view, exp_id, vs)
            if dec is None or dec.outcome == "invalid":
                result = self._finish_replication(exp_id, Outcome.INVALID, False, "the sealed grades are not comparable",
                                                  lesson="failed", vs=vs, after=False)
            else:
                replicated = dec.outcome == Outcome.IMPROVED
                note = "superseded" if not replicated and view.incumbent_id != start_incumbent else ""
                result = self._finish_replication(exp_id, Outcome.IMPROVED if replicated else Outcome.NO_GAIN, replicated, note,
                                                  dec=dec, vs=vs, lesson=None if replicated else "failed", after=False)
                if replicated:
                    result = self._promote(exp_id, ws, result)
        self._after_record()            # outside the promotion lock: it may call the reflector
        return result

    def _comparison_base(self, view: LedgerView) -> str:
        """The experiment whose sealed grade the candidate is compared with: the incumbent now, else the baseline."""
        inc = view.incumbent
        if inc is not None and inc.exp_id != view.baseline_id:
            rec = inc.grade_for(Purpose.REPLICATION)
            if rec is not None and rec.grade.valid:
                return inc.exp_id
        return view.baseline_id

    def _promote(self, exp_id: str, ws: sandbox.Workspace, result: LaneResult) -> LaneResult:
        view = self._view()
        reasons = _gates().promotion_gate(view, exp_id)
        if reasons:
            result.notes = ("replicated, not promoted: " + "; ".join(reasons))[:300]
            return result
        exp = view.exp(exp_id)
        tree = sandbox.freeze(ws.path, self._exp_dir(exp_id) / "frozen", self.task)
        sealed = exp.replication_grade
        self._append(Kind.INCUMBENT_PROMOTED, "engine", {
            "exp_id": exp_id, "score": exp.primary_grade.score if exp.primary_grade else None,
            "prev_exp_id": view.incumbent_id, "replicated": True, "replication_exp_ids": [exp_id], "tree_hash": tree,
            "sealed_score": sealed.score if sealed else None, "sealed_delta": exp.sealed_delta,
            "sealed_margin": exp.sealed_margin})
        result.notes = "replicated and promoted"
        return result

    def _ensure_baseline_sealed(self, base_id: str) -> Optional[str]:
        """Grade the baseline's ``frozen/`` tree on the replication seeds if that has not happened. None = available."""
        with self._baseline_sealed_lock:
            base = self._view().exp(base_id)
            rec = base.grade_for(Purpose.REPLICATION_BASELINE)
            if rec is not None and rec.grade.valid:
                return None
            frozen = self._exp_dir(base_id) / "frozen"
            meta = json.loads((self._exp_dir(base_id) / "base.json").read_text(encoding="utf-8"))
            tree = sandbox.tree_hash(frozen, self.task)
            if tree != str(meta["tree_hash"]):
                raise ProtectedPathViolation(["frozen"], "runtime")
            ws = sandbox.Workspace(exp_id=base_id, path=frozen, base_hashes={}, tree_hash=tree)
            attempt = self._next_attempt(base_id, (Purpose.REPLICATION_BASELINE,))
            out = self._evaluate(base_id, ws, self.task.replication_seeds, Purpose.REPLICATION_BASELINE, tree,
                                 attempt=attempt, retry_infra=attempt == 1)
            return out.failure_class

    def _finish_replication(self, exp_id: str, outcome: str, replicated: bool, note: str, *, dec: Any = None,
                            vs: Optional[str] = None, lesson: Optional[str] = None, after: bool = True) -> LaneResult:
        """Write ``experiment.recorded(phase=replication)``, the number-free lesson, and build the ``LaneResult``."""
        view = self._view()
        usable = dec is not None and dec.outcome != "invalid"
        self._append(Kind.EXP_RECORDED, "engine", {
            "exp_id": exp_id, "phase": "replication", "outcome": outcome, "delta": None, "vs_exp_id": vs,
            "noise_margin": None, "train_delta": None, "train_margin": None, "overfit_suspect": False,
            "replicated": replicated, "sigma_prior": self._sigma_prior(view), "n": dec.n if usable else 0,
            "s_eff": dec.s_eff if usable else None, "t": dec.t if usable else None,
            "sealed_delta": dec.delta if usable else None, "sealed_margin": dec.margin if usable else None, "note": note})
        if lesson == "infra":
            self._lesson_text(f"{exp_id}: the replication hit an environment failure and was not completed; the hypothesis "
                              "was not tested on the independent re-check.", exp_id, "infra")
        elif lesson == "failed":
            self._lesson_text(f"{exp_id} improved on the dev split but did not replicate on the independent re-check; possible "
                              "overfitting to the evaluation it was tuned on.", exp_id, "model")
        if after:
            self._after_record()
        exp = self._view().exp(exp_id)
        shown = Outcome.INFRA if outcome == Outcome.INFRA else (Outcome.IMPROVED if replicated else Outcome.NO_GAIN)
        if outcome == Outcome.INVALID:
            shown = Outcome.INVALID
        return LaneResult(exp_id=exp_id, lane=exp.lane if exp else "", status=ExpStatus.RECORDED, outcome=shown, delta=None,
                          score=None, replicated=replicated, notes="" if replicated else
                          ("superseded by a newer incumbent" if note == "superseded" else "did not replicate")[:300])

    def _lesson_text(self, text: str, exp_id: str, classification: str) -> None:
        self._check_crash()
        self.memory.add_lesson(text, kind="observation", source_exps=[exp_id], classification=classification, actor="engine")

    # =============================================================================================
    # reflection
    # =============================================================================================
    def reflect(self, trigger: str = "manual") -> dict:
        """Ask the reflector to consolidate lessons and propose pivots; returns ``{"lessons": [...], "pivots": [...]}``.

        Catches only ``ModelError`` (then returns empty lists and writes nothing).
        """
        trig = trigger if trigger in TRIGGERS else "manual"
        with self._reflect_lock:
            view = self._view()
            ctx = ReflectContext(task=self.task, trigger=trig, experiments_table=self.memory.tried_table(),
                                 lessons=[dict(x) for x in view.lessons[-20:]], incumbent_summary=self._incumbent_summary())
            try:
                got = self.suite.reflector.reflect(ctx)
            except ModelError:
                return {"lessons": [], "pivots": []}
            raw = list(got.lessons)[:MAX_LESSONS] if isinstance(getattr(got, "lessons", None), (list, tuple)) else []
            texts: list[str] = []
            ids: list[str] = []
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                text = sanitize(item.get("text", ""), 400)
                if not text:
                    continue
                src = item.get("source_exps")
                sources = [s for s in (src if isinstance(src, (list, tuple)) else [])[:10]
                           if isinstance(s, str) and _EXP_ID.match(s) and s in view.experiments]
                infra = bool(sources) and all(view.experiments[s].dev_outcome == Outcome.INFRA or _gates().never_tested(view.experiments[s])
                                              for s in sources)
                self._check_crash()
                ids.append(self.memory.add_lesson(text, kind=str(item.get("kind", "observation")), source_exps=sources,
                                                  classification="infra" if infra else "model", actor="reflector"))
                texts.append(text)
            piv = got.pivots if isinstance(getattr(got, "pivots", None), (list, tuple)) else []
            pivots = [p for p in (sanitize(x, 200) for x in piv[:MAX_PIVOTS] if isinstance(x, str)) if p]
            self._append(Kind.REFLECTION, "reflector", {"trigger": trig, "consolidated": ids, "pivots": pivots})
            self.regenerate_memory()
            return {"lessons": texts, "pivots": pivots}

    def _maybe_auto_reflect(self) -> None:
        """Reflect once per streak of ``stall_after`` consecutive non-improving experiments. Best effort: a spent budget
        or an unavailable reflector must not undo the experiment that was just recorded."""
        need = int(self.limits.stall_after)
        if need <= 0:
            return
        with self._reflect_lock:            # check and reflect as one step: two lanes ending the same streak reflect once
            streak = _gates().stall_streak(self._view())      # like the memory's, but untested hypotheses do not count
            if streak < need or self._streak_already_reflected(streak):
                return
            try:
                self.reflect("stall")
            except BudgetExceeded:
                return

    def _streak_already_reflected(self, streak: int) -> bool:
        view = self._view()
        lane_exps = sorted((e for e in view.experiments.values() if e.status == ExpStatus.RECORDED and e.lane != "baseline"
                            and not e.replicate_of and e.dev_outcome != Outcome.INFRA and not _gates().never_tested(e)),
                           key=lambda e: e.recorded_seq or 0)
        if streak > len(lane_exps):
            return False
        start = lane_exps[len(lane_exps) - streak]
        return any(r.get("trigger") == "stall" and int(r.get("seq", 0)) > (start.recorded_seq or 0) for r in view.reflections)

    # =============================================================================================
    # reconcile / resume
    # =============================================================================================
    def reconcile(self) -> list[str]:
        """Close every started run that has no ``run_finished`` as ``interrupted``. Idempotent; returns the experiment ids."""
        return self._interrupt_open_runs(list(self._view().order), "interrupted: the process was cut short by a restart")

    def _interrupt_open_runs(self, exp_ids: Sequence[str], note: str) -> list[str]:
        """Write ``run_finished(classification=interrupted)`` for every started run of ``exp_ids`` that never finished."""
        touched: list[str] = []
        view = self._view()
        for exp_id in exp_ids:
            exp = view.experiments.get(exp_id)
            for run in (exp.runs if exp is not None else []):
                if run.finished_seq is not None:
                    continue
                self._append(Kind.EXP_RUN_FINISHED, "engine", {
                    "exp_id": exp_id, "run_idx": run.run_idx, "purpose": run.purpose, "seed": run.seed, "exit_code": None,
                    "duration_sec": round(max(0.0, float(self.clock()) - run.started_ts), 3),
                    "classification": "interrupted", "stalled": False, "stdout_tail": "", "stderr_tail": "", "note": note})
                if exp_id not in touched:
                    touched.append(exp_id)
        return touched

    def resume_pending(self) -> list[LaneResult]:
        """Continue every non-terminal experiment from its last completed stage; never re-proposes or re-charges."""
        self.reconcile()
        results: list[LaneResult] = []
        view = self._view()
        if any(e.lane == "baseline" and not e.terminal for e in view.experiments.values()):
            results.append(self.run_baseline())
            view = self._view()
        pending = [e.exp_id for e in view.in_flight() if e.lane != "baseline" and not e.replicate_of]
        pending += [e.exp_id for e in view.experiments.values()
                    if e.status == ExpStatus.VIOLATION and e.recorded_seq is None and e.lane != "baseline"]
        results += self._gather([(lambda i=i: self._resume_lane(i)) for i in dict.fromkeys(pending)])
        for exp_id in self._replications_in_progress():
            results.append(self._replicate_core(exp_id))
        results += self._recover_promotions()
        return results

    def _recover_promotions(self) -> list[LaneResult]:
        """Finish a promotion that a crash cut short: the replication is recorded as replicated but ``incumbent.promoted`` is
        missing. The gates decide again against the incumbent of the moment; nothing is re-run and nothing is re-charged."""
        view = self._view()
        promoted = {str(p.get("exp_id")) for p in view.promotions}
        todo = [e.exp_id for e in view.experiments.values()
                if e.replicated and e.sealed_outcome == Outcome.IMPROVED and e.dev_outcome == Outcome.IMPROVED
                and e.exp_id != view.incumbent_id and e.exp_id not in promoted]
        results: list[LaneResult] = []
        for exp_id in todo:
            try:
                with self._promo_lock:
                    view = self._view()
                    exp = view.exp(exp_id)
                    dec = _gates().sealed_decision(self.task, view, exp_id, self._comparison_base(view))
                    if dec is None or dec.outcome != Outcome.IMPROVED:
                        continue                                    # a better incumbent has been promoted since
                    ws, _, problems = self._verify_workspace(exp)
                    if problems:
                        continue                                    # never freeze a tree that is not the one that was graded
                    result = self._promote(exp_id, ws, self._settled_result(exp_id, ""))
                if result.notes == "replicated and promoted":
                    results.append(result)
            except (SimulatedCrash, HaltRequested):
                raise
            except Exception:  # noqa: BLE001 - recovery is best effort; the candidate simply stays replicated and unpromoted
                continue
        return results

    def _replications_in_progress(self) -> list[str]:
        charged: dict[str, int] = {}
        started: dict[str, int] = {}
        recorded: dict[str, int] = {}
        for ev in self.ledger.events():
            exp_id = ev.data.get("exp_id")
            if not isinstance(exp_id, str):
                continue
            if ev.kind == Kind.BUDGET_CHARGED and ev.data.get("kind") == "replication":
                charged[exp_id] = ev.seq
            elif ev.kind == Kind.EXP_RUN_STARTED and ev.data.get("purpose") == Purpose.REPLICATION:
                started[exp_id] = ev.seq
            elif ev.kind == Kind.EXP_RECORDED and ev.data.get("phase") == "replication":
                recorded[exp_id] = ev.seq
        view = self._view()
        todo = {e: max(charged.get(e, 0), started.get(e, 0)) for e in {*charged, *started}
                if max(charged.get(e, 0), started.get(e, 0)) > recorded.get(e, 0) and view.exp(e) is not None}
        return sorted(todo, key=lambda e: todo[e])

    def _next_attempt(self, exp_id: str, purposes: Sequence[str]) -> int:
        """``NIGHTSHIFT_ATTEMPT`` for the next evaluation of ``exp_id``: one more than the attempts already made since the
        tree last changed (implemented / repaired) or the purpose was last graded."""
        attempts: list[int] = []
        for ev in reversed(self.ledger.events()):
            if ev.data.get("exp_id") != exp_id:
                continue
            if ev.kind in (Kind.EXP_IMPLEMENTED, Kind.EXP_REPAIRED):
                break
            if ev.kind == Kind.EXP_GRADED and ev.data.get("purpose") in purposes:
                break
            if ev.kind == Kind.EXP_RUN_STARTED and ev.data.get("purpose") in purposes:
                attempts.append(int(ev.data.get("attempt") or 1))
        return max(attempts, default=0) + 1

    def _resume_lane(self, exp_id: str) -> LaneResult:
        exp = self._view().exp(exp_id)
        base_id = None if exp.base in ("", "template") else exp.base
        try:
            from .templates import DEFAULT_STAGES
            stages = frozenset(DEFAULT_STAGES)
        except ImportError:                                                  # pragma: no cover
            stages = frozenset(("design", "implement", "run", "grade", "review", "repair", "record"))
        # the plan the implementer and reviewer worked from (``implemented.plan_summary``), not the supervisor's hint
        approach = self._recorded_plan(exp_id) or exp.plan
        ctx = _Lane(exp_id=exp_id, spec=LaneSpec(lane=exp.lane, hypothesis=exp.hypothesis, approach=exp.plan,
                                                 base=exp.base or "template", scripted_id=exp.scripted_id),
                    stages=stages, plan=ExperimentPlan(hypothesis=exp.hypothesis, approach=approach), base_id=base_id,
                    base_dir=self._exp_dir(base_id) / "frozen" if base_id else None, worker=exp.worker)
        return self._guarded(ctx, lambda: self._resume_steps(ctx))

    def _recorded_plan(self, exp_id: str) -> str:
        """``plan_summary`` of the latest ``experiment.implemented`` of ``exp_id`` ("" if none: a ledger from before the key
        existed, or an experiment that was never implemented)."""
        for ev in reversed(self.ledger.events()):
            if ev.kind == Kind.EXP_IMPLEMENTED and ev.data.get("exp_id") == exp_id:
                value = ev.data.get("plan_summary")
                return sanitize(value, PLAN_SUMMARY_CHARS) if isinstance(value, str) else ""
        return ""

    def _resume_steps(self, ctx: _Lane) -> LaneResult:
        exp = self._view().exp(ctx.exp_id)
        last = next((e for e in reversed(self.ledger.events()) if e.data.get("exp_id") == ctx.exp_id
                     and e.kind in (Kind.EXP_PROPOSED, Kind.EXP_WORKSPACE, Kind.EXP_IMPLEMENTED, Kind.EXP_PROTECTED,
                                    Kind.EXP_RUN_STARTED, Kind.EXP_RUN_FINISHED, Kind.EXP_GRADED, Kind.EXP_REVIEWED,
                                    Kind.EXP_REPAIRED, Kind.EXP_RECORDED)), None)
        kind = last.kind if last else Kind.EXP_PROPOSED
        if kind == Kind.EXP_RECORDED:
            return self._lane_result(ctx.exp_id)
        if kind == Kind.EXP_PROTECTED:
            ctx.pending = (Outcome.INVALID, "protected path violation (recorded before an interruption)")
            return self._phase_record(ctx)
        if kind == Kind.EXP_PROPOSED:
            return self._drive(ctx, "workspace")
        if kind == Kind.EXP_WORKSPACE:
            ctx.skip_ws_event = True
            return self._drive(ctx, "workspace")      # rebuild the workspace; its event is already in the ledger
        # the workspace exists: prove it is what the ledger says before doing anything with it
        ctx.ws = self._load_workspace(ctx.exp_id)
        if kind == Kind.EXP_REVIEWED and self._wants_repair(exp):
            ctx.last_diff = sandbox.diff_workspace(self.task, ctx.ws).unified    # a half-done repair may be on disk
            ctx.tree_hash = sandbox.tree_hash(ctx.ws.path, self.task)
            return self._drive(ctx, "repair")
        ws, diff, problems = self._verify_workspace(exp)
        if problems:
            return self._violation(ctx, [("workspace", problems[:50])])
        ctx.ws, ctx.tree_hash, ctx.last_diff = ws, diff.tree_hash, diff.unified
        if kind in (Kind.EXP_IMPLEMENTED, Kind.EXP_REPAIRED):
            ctx.attempt = 1
            return self._drive(ctx, "run")
        if kind in (Kind.EXP_RUN_STARTED, Kind.EXP_RUN_FINISHED):
            ctx.attempt = self._next_attempt(ctx.exp_id, (Purpose.PRIMARY,))
            return self._drive(ctx, "run")
        if kind == Kind.EXP_GRADED:
            rec = exp.grade_for(Purpose.PRIMARY)
            if rec is not None and rec.grade.valid:
                return self._drive(ctx, "review")
            ctx.pending = self._failure_outcome(self._failure_class_from_runs(exp))
            return self._phase_record(ctx)
        # reviewed and nothing more to do but record
        review = exp.last_review
        findings = [dict(f) for f in (review.findings if review else [])]
        step = self._after_review(ctx, review.verdict if review else "reject", findings, hard=True)
        return self._drive(ctx, step) if isinstance(step, str) else step

    def _wants_repair(self, exp: ExperimentView) -> bool:
        review = exp.last_review
        if review is None or any(f.get("id") == "reviewer_unavailable" for f in review.findings):
            return False
        blocking = any(f.get("severity") == "block" for f in review.findings)
        return ((review.verdict == "repair" or blocking) and len(exp.repairs) < int(self.limits.max_repairs)
                and not self._same_findings_after_repair(exp))

    @staticmethod
    def _failure_class_from_runs(exp: ExperimentView) -> str:
        classes = {r.classification for r in exp.runs if r.purpose == Purpose.PRIMARY}
        for name in ("infra", "timeout", "crash"):
            if name in classes:
                return name
        return "invalid"
