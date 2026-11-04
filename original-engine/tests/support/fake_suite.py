"""Deterministic fake agents and an engine harness for tests that run the REAL sandbox and grader on ``tests/fixtures/mini_task``.

Nothing here talks to a model. The fake roles do exactly what a test tells them to, so every engine path can be driven:
improving, worsening, no-gain, overfitting, tampering, probing, duplicate, crashing, hanging, infra-flaky and defective
(reviewer-catchable) implementations, plus reviewers that repair, reject, are unavailable or are not independent.

Quick start::

    with tempfile.TemporaryDirectory() as tmp:
        h = EngineHarness(Path(tmp))                     # ledger + run.started + gate + memory + fake suite + engine
        h.engine.run_baseline()                          # writes e000, freezes it, records it as the baseline
        res = h.engine.run_lane(STAGES_FULL, h.agents.lane("improve"))
        res.outcome                                      # "improved" (dev), awaiting replication
        h.engine.replicate(res.exp_id).replicated        # True: the sealed split agrees
        h.close()

Pieces
------
``FakeAgents``      the controller. ``.suite(backend)`` builds an ``AgentSuite``; ``.lane(name)`` builds a ``LaneSpec`` whose
                    ``scripted_id`` is a patch name (the fake implementer applies ``PATCHES[name]``); ``.contexts`` records
                    EVERY context object any fake role received, in order, and ``.context_text()`` joins their ``str()``
                    (the sealed-canary tests grep it); ``.on_implement[name]`` / ``.designer_hook`` / ``.review_script`` /
                    ``.reflect_hook`` let a test raise ``SimulatedCrash`` / ``BudgetExceeded`` / ``ModelError``, create a
                    HALT file, tamper with the trusted directory, or return a scripted ``ReviewVerdict``.
``PATCHES``         name -> ``Patch(files, repair_files, claim)``; see ``patch_names()``. All are small ``model.py`` bodies.
``EngineHarness``   everything the engine needs, rooted in a temp directory. ``restart()`` builds a fresh gate / memory /
                    engine over the SAME ledger object, which is how tests simulate a process restart after a crash.
``baseline_snapshot`` a run directory with a recorded baseline; ``EngineHarness(root, snapshot=...)`` copies it (fast start).
``copy_mini_task``  a private copy of the fixture task whose grader can make the SEALED score distinctive (``offset``: every
                    sealed score is shifted by 0.314159265) or uninformative (``flat``: every model gets the same sealed score).
                    Copies are what tests use when they need to modify the trusted directory or the template.

The fake roles never read ``planted_fault`` and never see sealed numbers unless the engine puts them in a context, which is
precisely what the canary tests check.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from nightshift.budget import BudgetGate
from nightshift.ledger import Ledger
from nightshift.memory import Memory
from nightshift.task import load_task, task_hash, trusted_hash
from nightshift.types import (AgentSuite, BudgetLimits, DesignContext, ExperimentPlan, ImplementContext, ImplementResult, Kind,
                              LaneSpec, ReflectContext, ReflectionResult, ReviewContext, ReviewVerdict, TaskSpec)
from tests.support.synthetic import FakeClock

MINI_TASK_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mini_task"
STAGES_FULL = ["design", "implement", "run", "grade", "review", "repair", "record"]
STAGES_MIN = ["implement", "run", "grade", "review", "record"]
DEFECT_MARKER = "BUG(fake): "
SEALED_SENTINEL = 0.314159265

_TEMPLATE_MODEL = (MINI_TASK_DIR / "template" / "model.py").read_text(encoding="utf-8")
_TEMPLATE_RUN = (MINI_TASK_DIR / "template" / "run.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# model.py bodies
# --------------------------------------------------------------------------------------------------
def binned_model(bins: int, *, extra: str = "") -> str:
    """Piecewise-constant fit with ``bins`` bins. On the fixture: 5 bins ~0.195, 10 ~0.143, 20 ~0.128 (baseline ~0.291)."""
    return f'''{extra}
def fit(xs, ys):
    bins = {bins}
    sums, counts = [0.0] * bins, [0] * bins
    for x, y in zip(xs, ys):
        b = min(int(x * bins), bins - 1)
        sums[b] += y
        counts[b] += 1
    overall = sum(ys) / len(ys)
    means = [sums[i] / counts[i] if counts[i] else overall for i in range(bins)]
    return lambda x: means[min(max(int(x * bins), 0), bins - 1)]
'''


CONSTANT_PLUS = '''
def fit(xs, ys):
    mean = sum(ys) / len(ys) + 0.5
    return lambda x: mean
'''
MEMORISER = '''
def fit(xs, ys):
    table = dict(zip(xs, ys))
    mean = sum(ys) / len(ys)
    return lambda x: table.get(x, mean)
'''
CRASHING = '''
def fit(xs, ys):
    raise RuntimeError("boom")
'''
HANGING = '''
import time


def fit(xs, ys):
    time.sleep(60)
'''
FLAKY_PREFIX = '''import os
import sys

if os.environ.get("NIGHTSHIFT_ATTEMPT", "1") == "1":
    sys.stderr.write("MemoryError: simulated infrastructure failure\\n")
    sys.exit(1)
'''
NAN_OUTPUT = '''
def fit(xs, ys):
    return lambda x: float("nan")
'''
ALWAYS_INFRA = '''import sys

sys.stderr.write("MemoryError: simulated infrastructure failure\\n")
sys.exit(1)
'''


@dataclass(frozen=True)
class Patch:
    """What the fake implementer writes for one scripted id."""

    files: dict                                   # workspace-relative path -> full new text
    repair_files: Optional[dict] = None           # written instead of ``files`` when asked to repair (default: files + a comment)
    claim: Optional[tuple[str, str]] = None       # (claim_type, text) emitted through ``ctx.emit_claim`` on the first pass
    note: str = ""                                # documentation for readers of the tests


PATCHES: dict[str, Patch] = {
    "improve": Patch({"model.py": binned_model(5)}, note="dev ~0.195 vs 0.291, sealed agrees: improves and replicates"),
    "improve_again": Patch({"model.py": binned_model(5)}, note="byte-identical to improve: the same idempotency key"),
    "improve_mid": Patch({"model.py": binned_model(10)}, note="better than improve"),
    "improve_best": Patch({"model.py": binned_model(20)}, note="better than improve_mid"),
    "worse": Patch({"model.py": CONSTANT_PLUS}, note="much worse than the baseline"),
    "noop": Patch({"model.py": _TEMPLATE_MODEL + "\n# no functional change\n"}, note="identical predictions: no_gain"),
    "noop2": Patch({"model.py": _TEMPLATE_MODEL + "\n# another comment\n"}, note="another no_gain with a different diff"),
    "overfit": Patch({"model.py": MEMORISER}, note="train error ~0, held-out equals the baseline: overfit_suspect"),
    "crash": Patch({"model.py": CRASHING}, note="every run exits 1 with a traceback: failed"),
    "hang": Patch({"model.py": HANGING}, note="needs overrides stall_sec / run_timeout_sec: failed, watchdog.stalled"),
    "infra_flaky": Patch({"model.py": FLAKY_PREFIX + binned_model(5)}, note="MemoryError on attempt 1, fine on attempt 2"),
    "infra_always": Patch({"model.py": ALWAYS_INFRA}, note="MemoryError on every attempt: infra_error"),
    "nan_output": Patch({"model.py": NAN_OUTPUT}, note="the run succeeds but the grader finds the predictions invalid: failed"),
    "tamper": Patch({"run.py": _TEMPLATE_RUN + "\n# tampered\n"}, note="edits a read-only file: protected_violation(workspace)"),
    "probe": Patch({"model.py": _TEMPLATE_MODEL + "\n# peek at open('../../trusted/labels.csv')\n"},
                   note="mentions the trusted directory: protected_violation(trusted_probe)"),
    "defective": Patch({"model.py": binned_model(5, extra=f"# {DEFECT_MARKER}off-by-one in the bin index\n")},
                       repair_files={"model.py": binned_model(5)}, note="the default fake reviewer asks for a repair; the repair fixes it"),
    "defective_stubborn": Patch({"model.py": binned_model(5, extra=f"# {DEFECT_MARKER}never fixed\n")},
                                note="every repair still contains the marker (but a different diff each round)"),
    "claims_win": Patch({"model.py": _TEMPLATE_MODEL + "\n# claims a win it does not have\n"},
                        claim=("improved", "a large improvement on held-out data"), note="claim_without_run: outcome no_gain"),
}


def patch_names() -> list[str]:
    return sorted(PATCHES)


# --------------------------------------------------------------------------------------------------
# fake roles
# --------------------------------------------------------------------------------------------------
class _NullSupervisor:
    identity = "fake-supervisor"

    def run(self, env: Any) -> None:       # pragma: no cover - the engine never runs the supervisor
        return None


class FakeDesigner:
    identity = "fake-designer"

    def __init__(self, agents: "FakeAgents"):
        self.agents = agents

    def design(self, ctx: DesignContext) -> ExperimentPlan:
        self.agents.record(ctx)
        self.agents.design_calls += 1
        if self.agents.designer_hook:
            self.agents.designer_hook(ctx)
        return ExperimentPlan(hypothesis=ctx.lane.hypothesis, approach=f"fake plan for {ctx.lane.scripted_id}",
                              expected_effect="fake", risks="fake", files_to_touch=["model.py"])


class FakeImplementer:
    identity = "fake-implementer"

    def __init__(self, agents: "FakeAgents"):
        self.agents = agents

    def implement(self, ctx: ImplementContext) -> ImplementResult:
        a = self.agents
        a.record(ctx)
        name = ctx.lane.scripted_id or "noop"
        with a._lock:
            a._active += 1
            a.peak_concurrency = max(a.peak_concurrency, a._active)
        try:
            if a.implement_delay:
                time.sleep(a.implement_delay)
            hook = a.on_implement.get(name)
            if hook:
                hook(ctx)
            patch = a.patches[name]
            repairing = bool(ctx.repair_findings)
            if repairing:
                with a._lock:
                    a.repair_counts[ctx.exp_id] = a.repair_counts.get(ctx.exp_id, 0) + 1
                    n = a.repair_counts[ctx.exp_id]
                files = patch.repair_files if patch.repair_files is not None else {
                    k: v + f"\n# repair round {n}\n" for k, v in patch.files.items()}
            else:
                files = patch.files
            for rel, text in files.items():
                path = Path(ctx.workspace) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, path)
            if patch.claim and not repairing and ctx.emit_claim is not None:
                ctx.emit_claim(*patch.claim)
            return ImplementResult(summary=f"fake {'repair' if repairing else 'implementation'} of {name}", steps=len(files),
                                   worker=a.implementer_identity, raw_log="wrote: " + ", ".join(sorted(files)),
                                   self_report="fake: everything improved a lot (never evidence)")
        finally:
            with a._lock:
                a._active -= 1


class FakeReviewer:
    """Accepts, unless the diff contains ``DEFECT_MARKER`` (then it asks for a repair with a block finding) or a test installed
    ``agents.review_script`` (a callable ``ReviewContext -> ReviewVerdict``) or a queue in ``agents.review_queue``."""

    def __init__(self, agents: "FakeAgents", identity: str, queue_attr: str):
        self.agents, self.identity, self._queue_attr = agents, identity, queue_attr

    def review(self, ctx: ReviewContext) -> ReviewVerdict:
        a = self.agents
        a.record(ctx)
        a.review_calls.append(self.identity)
        queue: deque = getattr(a, self._queue_attr)
        if queue:
            item = queue.popleft()
            return item(ctx) if callable(item) else item
        if a.review_script is not None and self._queue_attr == "review_queue":
            return a.review_script(ctx)
        if DEFECT_MARKER in ctx.diff_text:
            fid = f"defect-{ctx.repair_round}" if a.vary_finding_ids else "defect"
            return ReviewVerdict(verdict="repair", reviewer=self.identity, summary="fake: defect marker found", findings=[
                {"id": fid, "severity": "block", "text": "the diff contains the defect marker", "evidence": DEFECT_MARKER}])
        return ReviewVerdict(verdict="accept", reviewer=self.identity, summary="fake: looks fine", findings=[])


class FakeReflector:
    identity = "fake-reflector"

    def __init__(self, agents: "FakeAgents"):
        self.agents = agents

    def reflect(self, ctx: ReflectContext) -> ReflectionResult:
        a = self.agents
        a.record(ctx)
        a.reflect_calls += 1
        if a.reflect_hook:
            a.reflect_hook(ctx)
        if a.reflection is not None:
            return a.reflection
        ids = re.findall(r"\be\d{3,}\b", ctx.experiments_table)[-2:]
        return ReflectionResult(lessons=[{"text": f"Fake reflection {a.reflect_calls}: do not repeat ideas that gave no gain.",
                                          "kind": "strategy", "source_exps": ids, "classification": "unknown"}],
                                pivots=["try a different model family"], summary="fake")


class FakeAgents:
    """Controller and recorder for the fake roles. One instance per test; thread-safe where lanes run in parallel."""

    def __init__(self, task: Optional[TaskSpec] = None):
        self.task = task
        self.patches: dict[str, Patch] = dict(PATCHES)
        self.contexts: list[Any] = []
        self.on_implement: dict[str, Callable[[ImplementContext], None]] = {}
        self.designer_hook: Optional[Callable[[DesignContext], None]] = None
        self.reflect_hook: Optional[Callable[[ReflectContext], None]] = None
        self.review_script: Optional[Callable[[ReviewContext], ReviewVerdict]] = None
        self.review_queue: deque = deque()          # verdicts (or callables) returned by the next review() calls
        self.hard_queue: deque = deque()            # the same for the review_hard reviewer
        self.reflection: Optional[ReflectionResult] = None
        self.vary_finding_ids = False               # defect findings get a new id each repair round
        self.implement_delay = 0.0
        self.implementer_identity = FakeImplementer.identity
        self.reviewer_identity = "fake-reviewer"
        self.hard_reviewer_identity = "fake-reviewer-hard"
        self.design_calls = 0
        self.review_calls: list[str] = []
        self.reflect_calls = 0
        self.repair_counts: dict[str, int] = {}
        self.peak_concurrency = 0
        self._active = 0
        self._lane_n = 0
        self._lock = threading.Lock()

    # ---- recording ----------------------------------------------------------------------------
    def record(self, ctx: Any) -> None:
        with self._lock:
            self.contexts.append(ctx)

    def context_text(self) -> str:
        """``str()`` of every context any role received: the haystack for the sealed-canary assertions."""
        with self._lock:
            return "\n".join(str(c) for c in self.contexts)

    def contexts_of(self, cls: type) -> list[Any]:
        with self._lock:
            return [c for c in self.contexts if isinstance(c, cls)]

    # ---- building -----------------------------------------------------------------------------
    def lane(self, name: str, label: Optional[str] = None, *, base: str = "incumbent", hypothesis: Optional[str] = None,
             approach: str = "") -> LaneSpec:
        """A lane whose fake implementer applies ``PATCHES[name]``; each call gets a distinct hypothesis unless one is given."""
        with self._lock:
            self._lane_n += 1
            n = self._lane_n
        return LaneSpec(lane=label or chr(ord("A") + (n - 1) % 26), hypothesis=hypothesis or f"Fake experiment {n}: apply the {name} change",
                        approach=approach, base=base, scripted_id=name)

    def suite(self, backend: str = "fake", *, reviewer_hard: bool = False, models: Optional[dict] = None) -> AgentSuite:
        suite = AgentSuite(
            backend=backend, supervisor=_NullSupervisor(), designer=FakeDesigner(self), implementer=FakeImplementer(self),
            reviewer=FakeReviewer(self, self.reviewer_identity, "review_queue"), reflector=FakeReflector(self),
            reviewer_hard=FakeReviewer(self, self.hard_reviewer_identity, "hard_queue") if reviewer_hard else None,
            models=models if models is not None else {
                "supervisor": "fake-supervisor-model", "designer": "fake-model", "implementer": "fake-model",
                "reviewer": "fake-model", "reviewer_hard": "fake-hard-model", "reflector": "fake-model"})
        suite.fakes = self  # type: ignore[attr-defined]
        return suite


# --------------------------------------------------------------------------------------------------
# a private copy of the fixture task
# --------------------------------------------------------------------------------------------------
_HELDOUT_LINE = "    heldout = rmse([(preds[i], labels[i][1]) for i in chosen])\n"


def copy_mini_task(dest: Path, *, sealed: str = "plain", template_model: Optional[str] = None) -> TaskSpec:
    """Copy ``tests/fixtures/mini_task`` to ``dest`` and load it.

    ``sealed="plain"``   the unmodified grader.
    ``sealed="offset"``  sealed scores are the real score + 0.314159265 (deltas unchanged): every sealed number is recognisable.
    ``sealed="flat"``    sealed scores ignore the predictions (0.314159265 + a seed-only term): no model can replicate.
    ``template_model``   replaces ``template/model.py`` (e.g. a baseline that crashes).
    """
    dest = Path(dest)
    shutil.copytree(MINI_TASK_DIR, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    grade = dest / "trusted" / "grade.py"
    text = grade.read_text(encoding="utf-8")
    if _HELDOUT_LINE not in text:
        raise RuntimeError("the fixture grader changed; update copy_mini_task")
    if sealed == "offset":
        text = text.replace(_HELDOUT_LINE, _HELDOUT_LINE + f'    if args.split == "sealed":\n        heldout += {SEALED_SENTINEL}\n')
    elif sealed == "flat":
        text = text.replace(_HELDOUT_LINE, _HELDOUT_LINE + f'    if args.split == "sealed":\n'
                                                           f'        heldout = {SEALED_SENTINEL} + 0.0001 * (args.seed % 7)\n')
    elif sealed != "plain":
        raise ValueError(f"unknown sealed mode {sealed!r}")
    grade.write_text(text, encoding="utf-8")
    if template_model is not None:
        (dest / "template" / "model.py").write_text(template_model, encoding="utf-8")
    return load_task(dest)


# --------------------------------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------------------------------
class EngineHarness:
    """A ledger with ``run.started``, a budget gate, memory, a fake suite and a ``WorkflowEngine``, rooted in ``root``.

    ``overrides`` defaults to ``{"isolation": "none"}`` (fast; the sandbox's own tests cover isolation). Pass ``{}`` for
    ``auto``. The clock is a ``FakeClock`` shared by the ledger and the gate, so wall time never runs out by accident.
    """

    def __init__(self, root: Path, *, task: Optional[TaskSpec] = None, agents: Optional[FakeAgents] = None,
                 limits: Optional[BudgetLimits] = None, overrides: Optional[dict] = None, crash_after_runs: Optional[int] = None,
                 backend: str = "fake", write_run_started: bool = True, reviewer_hard: bool = False,
                 snapshot: Optional[Path] = None):
        self.root = Path(root)
        self.run_dir = self.root / "run"
        self.task = task or load_task(MINI_TASK_DIR)
        self.agents = agents or FakeAgents(self.task)
        self.limits = limits or BudgetLimits(max_experiments=16, max_wall_sec=100000.0, max_parallel=3, max_replications=6,
                                             max_repairs=2, stall_after=3)
        self.overrides = {"isolation": "none"} if overrides is None else dict(overrides)
        self.backend, self.reviewer_hard = backend, reviewer_hard
        if snapshot is not None:
            # a run directory that already holds run.started and a recorded baseline (see ``baseline_snapshot``): the copy is
            # byte for byte what ``baseline()`` would have written, minus the cost of three real runs
            shutil.copytree(snapshot, self.run_dir, ignore=shutil.ignore_patterns("ledger.jsonl.lock"))
            last_ts = Ledger.read_only(self.run_dir).events()[-1].ts
            self.clock = FakeClock(start=last_ts, step=0.001)
            self.ledger = Ledger.open(self.run_dir, clock=self.clock, fsync=False)
            write_run_started = False
        else:
            self.clock = FakeClock(step=0.001)
            self.ledger = Ledger.create(self.run_dir, clock=self.clock, fsync=False)
        if write_run_started:
            self.ledger.append(Kind.RUN_STARTED, "engine", {
                "task": self.task.name, "backend": backend, "models": {"supervisor": "fake"}, "limits": self.limits.to_dict(),
                "task_hash": task_hash(self.task), "grader_hash": trusted_hash(self.task), "objective": self.task.objective,
                "primary_metric": self.task.primary_metric, "train_metric": self.task.train_metric, "isolation": "none",
                "config": {}})
        self.engine = None
        self.restart(crash_after_runs=crash_after_runs)

    def restart(self, *, crash_after_runs: Optional[int] = None):
        """Fresh gate, memory, suite and engine over the SAME ledger: the in-process equivalent of a restart after a crash."""
        from nightshift.workflows.engine import WorkflowEngine
        self.gate = BudgetGate(self.limits, self.ledger, clock=self.clock)
        self.memory = Memory(self.ledger, self.task)
        self.suite = self.agents.suite(self.backend, reviewer_hard=self.reviewer_hard)
        self.engine = WorkflowEngine(task=self.task, run_dir=self.run_dir, ledger=self.ledger, gate=self.gate, memory=self.memory,
                                     suite=self.suite, auditor=None, limits=self.limits, clock=self.clock,
                                     overrides=self.overrides, crash_after_runs=crash_after_runs)
        return self.engine

    @classmethod
    def baseline_snapshot(cls, dest: Path, **kw) -> Path:
        """Run the baseline once in ``dest`` and return its run directory. Pass it as ``snapshot=`` to later harnesses (same
        task and default limits) to start from a recorded baseline without re-running it."""
        h = cls(Path(dest), **kw)
        h.baseline()
        h.close()
        return h.run_dir

    # ---- shortcuts ----------------------------------------------------------------------------
    def baseline(self):
        return self.engine.run_baseline()

    def lane(self, name: str, stages: Sequence[str] = STAGES_FULL, **kw):
        return self.engine.run_lane(list(stages), self.agents.lane(name, **kw))

    def lanes(self, names: Sequence[str], stages: Sequence[str] = STAGES_FULL):
        return self.engine.run_lanes(list(stages), [self.agents.lane(n) for n in names])

    def view(self):
        return self.ledger.view()

    def events(self, *kinds: str, exp_id: Optional[str] = None) -> list:
        return [e for e in self.ledger.events() if (not kinds or e.kind in kinds)
                and (exp_id is None or e.data.get("exp_id") == exp_id)]

    def kinds(self, exp_id: Optional[str] = None) -> list[str]:
        return [e.kind for e in self.events(exp_id=exp_id)]

    def halt(self) -> None:
        (self.run_dir / "HALT").write_text("stop\n", encoding="utf-8")

    def close(self) -> None:
        self.ledger.close()
