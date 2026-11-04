"""A stand-in for ``WorkflowEngine`` (docs/ARCHITECTURE.md 6.7, API_B6) for orchestrator tests, plus small suite helpers.

``FakeEngine`` implements the engine API the orchestrator calls (``run_baseline``, ``run_lane``, ``run_lanes``,
``replicate``, ``reflect``, ``reconcile``, ``resume_pending``, ``regenerate_memory``) and writes the same ledger events the
real engine writes, with outcomes taken from a ``plan`` instead of from running anything. It uses the REAL gates for
launch/replication refusals, so duplicate, HALT and budget behaviour is the real one.

    factory = fake_engine_factory(plan={"good": {"score": 0.8, "delta": 0.2, "outcome": "improved"}})
    orch = Orchestrator(run_dir, task, suite, limits, ledger=..., gate=..., auditor=..., backend="scripted",
                        engine_factory=factory)

``plan`` maps a lower-case substring of a hypothesis to ``{score, delta, outcome, review, protected, sealed_pass}``; a
hypothesis matching nothing is a no-gain. ``sealed_sentinel`` makes every sealed number in the ledger equal that sentinel
(the canary of docs section 13); a replication result then also carries the sentinel in its notes and outcome class, to prove
the orchestrator does not forward them.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from nightshift.task import conditions_hash
from nightshift.types import (AgentSuite, ExpStatus, Kind, LaneResult, LaneSpec, NightshiftError, Outcome, SimulatedCrash,
                              sha256_hex)
from tests.support.synthetic import RunBuilder, grade_dict

try:  # use the real class when the engine exists so isinstance checks in callers hold
    from nightshift.workflows.engine import BaselineFailed
except ImportError:  # pragma: no cover - while engine.py is still being written
    class BaselineFailed(NightshiftError):   # type: ignore[no-redef]
        """The baseline experiment could not be graded."""


class FakeEngine:
    def __init__(self, *, task, run_dir, ledger, gate, memory, suite, auditor, limits, clock=time.time, overrides=None,
                 crash_after_runs: Optional[int] = None, plan: Optional[dict] = None, baseline_score: float = 1.0,
                 fail_baseline: bool = False, sealed_sentinel: Optional[float] = None, reflect_output: Optional[dict] = None,
                 replicate_note: str = ""):
        self.task, self.run_dir, self.ledger, self.gate, self.memory = task, run_dir, ledger, gate, memory
        self.suite, self.auditor, self.limits, self.overrides = suite, auditor, limits, overrides
        self.crash_after_runs = crash_after_runs
        self.plan = {k.lower(): v for k, v in (plan or {}).items()}
        self.baseline_score = baseline_score
        self.fail_baseline = fail_baseline
        self.sealed_sentinel = sealed_sentinel
        self.reflect_output = reflect_output
        self.replicate_note = replicate_note
        self.calls: list[tuple] = []
        self.runs_started = 0
        self.crashed = False
        self._lock = threading.RLock()
        self._rb = RunBuilder(ledger, seeds=list(task.seeds), replication_seeds=list(task.replication_seeds))

    # ---- helpers ---------------------------------------------------------------------------------
    def _spec_for(self, hypothesis: str) -> dict:
        text = (hypothesis or "").lower()
        for key, spec in self.plan.items():
            if key in text:
                return spec
        return {"score": self.baseline_score, "delta": 0.0, "outcome": Outcome.NO_GAIN}

    def _gh(self) -> str:
        """The run's grader hash, so the audit's hash checks see consistent grades."""
        return str(self.ledger.view().run.get("grader_hash") or "g" * 12)

    def _cond(self, seeds: Sequence[int], split: str) -> str:
        return conditions_hash(self.task, list(seeds), split, grader_hash=self._gh())

    def _grade_event(self, exp_id: str, purpose: str, score: float, train: Optional[float], seeds: Sequence[int], split: str) -> None:
        self._append(Kind.EXP_GRADED, "engine", {
            "exp_id": exp_id, "purpose": purpose, "split": split,
            "grade": grade_dict(score, train, seeds=list(seeds), split=split, grader_hash=self._gh()),
            "grader_hash": self._gh(), "seeds": list(seeds), "conditions_hash": self._cond(seeds, split)})

    def _sealed_pair(self, ok: bool) -> tuple[float, float]:
        """(sealed_delta, sealed_margin) as recorded. With a sentinel both carry its digits yet stay consistent with ``ok``."""
        if self.sealed_sentinel is None:
            return (0.05, 0.02) if ok else (-0.05, 0.02)
        big, small = self.sealed_sentinel, self.sealed_sentinel / 10
        return (big, small) if ok else (small, big)

    def _tick(self) -> None:
        """Count one untrusted run started; the crash hook fires once the count reaches ``crash_after_runs``."""
        with self._lock:
            self.runs_started += 1
            if self.crash_after_runs is not None and self.runs_started >= self.crash_after_runs:
                self.crashed = True
                raise SimulatedCrash(f"crash hook after {self.runs_started} runs")

    def _append(self, kind: str, actor: str, data: dict):
        if self.crashed:
            raise SimulatedCrash("the ledger is crashed")
        return self.ledger.append(kind, actor, data)

    def _run_events(self, exp_id: str, purpose: str, seeds: Sequence[int], first_idx: int) -> None:
        for offset, seed in enumerate(seeds):
            idx = first_idx + offset
            self._append(Kind.EXP_RUN_STARTED, "engine", {"exp_id": exp_id, "run_idx": idx, "purpose": purpose, "seed": seed,
                                                          "cmd": ["python", "run.py"], "pid": 1, "attempt": 1, "isolation": "none"})
            self._tick()
            self._append(Kind.EXP_RUN_FINISHED, "engine", {"exp_id": exp_id, "run_idx": idx, "purpose": purpose, "seed": seed,
                                                           "exit_code": 0, "duration_sec": 0.1, "classification": "ok",
                                                           "stdout_tail": "", "stderr_tail": ""})

    def _finish_lane(self, exp_id: str, lane: str, hypothesis: str, first_idx: int = 0) -> LaneResult:
        """Run (events only), grade, review and record an experiment according to the plan."""
        spec = self._spec_for(hypothesis)
        view = self.ledger.view()
        self._run_events(exp_id, "primary", self.task.seeds, first_idx)
        self._grade_event(exp_id, "primary", spec["score"], spec.get("train"), self.task.seeds, "dev")
        self._append(Kind.EXP_REVIEWED, f"reviewer:{lane}", {
            "exp_id": exp_id, "review_round": 0, "reviewer": "reviewer-y", "verdict": spec.get("review", "accept"),
            "findings": [], "independent": True})
        outcome = spec.get("outcome", Outcome.NO_GAIN)
        self._append(Kind.EXP_RECORDED, "engine", {
            "exp_id": exp_id, "phase": "lane", "outcome": outcome, "delta": spec.get("delta", 0.0),
            "vs_exp_id": view.incumbent_id, "noise_margin": self.task.noise_margin, "train_delta": spec.get("train_delta"),
            "replicated": False})
        self.regenerate_memory()
        return LaneResult(exp_id=exp_id, lane=lane, status=ExpStatus.RECORDED, outcome=outcome, delta=spec.get("delta", 0.0),
                          score=spec["score"], review_verdict=spec.get("review", "accept"), notes=spec.get("notes", ""))

    # ---- engine API ------------------------------------------------------------------------------
    def regenerate_memory(self) -> None:
        self.memory.write_markdown(self.run_dir / "memory.md")

    def _complete_baseline(self, exp_id: str, first_idx: int = 0) -> LaneResult:
        self._run_events(exp_id, "baseline", self.task.seeds, first_idx)
        self._grade_event(exp_id, "baseline", self.baseline_score, 0.9, self.task.seeds, "dev")
        self._append(Kind.EXP_RECORDED, "engine", {
            "exp_id": exp_id, "phase": "lane", "outcome": Outcome.BASELINE, "delta": None, "vs_exp_id": None,
            "noise_margin": self.task.noise_margin, "train_delta": None, "replicated": False})
        self.regenerate_memory()
        return LaneResult(exp_id=exp_id, lane="baseline", status=ExpStatus.RECORDED, outcome=Outcome.BASELINE,
                          score=self.baseline_score)

    def run_baseline(self) -> LaneResult:
        self.calls.append(("run_baseline",))
        if self.fail_baseline:
            raise BaselineFailed("the baseline could not be graded")
        with self._lock:
            exp_id = self.ledger.next_exp_id()
        self.gate.reserve_experiment(exp_id, "baseline")
        self._append(Kind.EXP_PROPOSED, "engine", {
            "exp_id": exp_id, "lane": "baseline", "hypothesis": "unmodified template", "plan": "", "parent_exp_id": None,
            "base": "template", "hyp_fingerprint": "baseline", "replicate_of": None})
        self._append(Kind.EXP_WORKSPACE, "engine", {"exp_id": exp_id, "path": f"exp/{exp_id}/workspace", "base_hash": "b" * 12})
        return self._complete_baseline(exp_id)

    def _block(self, lane: LaneSpec, reasons: list[str]) -> LaneResult:
        text = " ".join(reasons)
        reason = ("halt" if "HALT" in text else "budget" if "budget" in text else "duplicate" if "duplicate" in text
                  else "not_replicable" if lane.replicate_of or "replicat" in text else "protected")
        self._append(Kind.EXP_BLOCKED, "engine", {
            "exp_id": None, "lane": lane.lane, "hypothesis": lane.hypothesis, "hyp_fingerprint": self.memory.fingerprint(lane.hypothesis),
            "reason": reason, "detail": text[:200], "dup_of": None})
        return LaneResult(exp_id="", lane=lane.lane, status=ExpStatus.BLOCKED, blocked_reason=reason, notes=text[:200])

    def run_lane(self, stages: list[str], lane: LaneSpec) -> LaneResult:
        from nightshift.workflows import gates
        self.calls.append(("run_lane", list(stages), lane))
        with self._lock:                               # check-then-propose is atomic, as in the real engine
            reasons = gates.launch_gate(self.ledger.view(), lane, self.memory, gate=self.gate, run_dir=self.run_dir,
                                        limits=self.limits, task=self.task)
            if reasons:
                return self._block(lane, reasons)
            exp_id = self.ledger.next_exp_id()
            self.gate.reserve_experiment(exp_id)
            self._append(Kind.EXP_PROPOSED, f"designer:{lane.lane}", {
                "exp_id": exp_id, "lane": lane.lane, "hypothesis": lane.hypothesis, "plan": lane.approach,
                "parent_exp_id": None, "base": lane.base, "hyp_fingerprint": self.memory.fingerprint(lane.hypothesis),
                "replicate_of": None, "scripted_id": lane.scripted_id, "planted_fault": None})
            self._append(Kind.EXP_WORKSPACE, "engine", {"exp_id": exp_id, "path": f"exp/{exp_id}/workspace", "base_hash": "b" * 12})
            spec = self._spec_for(lane.hypothesis)
            self._append(Kind.EXP_IMPLEMENTED, f"implementer:{lane.lane}", {
                "exp_id": exp_id, "worker": "worker-x", "diff_hash": sha256_hex(exp_id)[:12], "files_changed": ["model.py"],
                "diff_summary": "edit", "idem_key": sha256_hex(exp_id)[:16], "attempt": 1, "steps": 2,
                "tree_hash": sha256_hex("tree" + exp_id)[:12]})
            if spec.get("protected"):
                self._append(Kind.EXP_PROTECTED, "engine", {"exp_id": exp_id, "paths": list(spec["protected"]), "kind": "workspace"})
                return LaneResult(exp_id=exp_id, lane=lane.lane, status=ExpStatus.VIOLATION, outcome=Outcome.INVALID)
        return self._finish_lane(exp_id, lane.lane, lane.hypothesis)

    def run_lanes(self, stages: list[str], lanes: list[LaneSpec]) -> list[LaneResult]:
        self.calls.append(("run_lanes", list(stages), list(lanes)))
        if not lanes:
            return []
        with ThreadPoolExecutor(max_workers=max(1, int(self.limits.max_parallel))) as pool:
            futures = [pool.submit(self.run_lane, stages, lane) for lane in lanes]
            results, crash = [], None
            for f in futures:
                try:
                    results.append(f.result())
                except SimulatedCrash as exc:
                    crash = crash or exc
        if crash is not None:
            raise crash
        return results

    def replicate(self, exp_id: str) -> LaneResult:
        from nightshift.workflows import gates
        self.calls.append(("replicate", exp_id))
        lane = LaneSpec(lane="", hypothesis="", replicate_of=exp_id)
        with self._lock:
            reasons = gates.launch_gate(self.ledger.view(), lane, self.memory, gate=self.gate, run_dir=self.run_dir,
                                        limits=self.limits, task=self.task)
            if reasons:
                return self._block(lane, reasons)
            self.gate.reserve_replication(exp_id)
            view = self.ledger.view()
            exp, base_id = view.experiments[exp_id], view.baseline_id
            spec = self._spec_for(exp.hypothesis)
            ok = bool(spec.get("sealed_pass", True))
            sentinel = self.sealed_sentinel
            dev_score = exp.primary_grade.score
            base_sealed = 0.9 if sentinel is not None else self.baseline_score
            cand_sealed = (sentinel if ok else base_sealed + 0.05) if sentinel is not None else (
                self.baseline_score - 0.05 if ok else self.baseline_score + 0.05)
            reps = list(self.task.replication_seeds)
            if view.baseline.grade_for("replication_baseline") is None:
                self._run_events(base_id, "replication_baseline", reps[:1], len(view.baseline.runs))
                self._grade_event(base_id, "replication_baseline", base_sealed, None, reps, "sealed")
            self._run_events(exp_id, "replication", reps[:1], len(exp.runs))
            self._grade_event(exp_id, "replication", cand_sealed, None, reps, "sealed")
            self._append(Kind.EXP_RECORDED, "engine", {
                "exp_id": exp_id, "phase": "replication", "outcome": Outcome.IMPROVED if ok else Outcome.NO_GAIN,
                "sealed_delta": self._sealed_pair(ok)[0], "sealed_margin": self._sealed_pair(ok)[1], "replicated": ok,
                "note": "" if ok else "did not replicate on the independent re-check"})
            if ok:
                self._append(Kind.INCUMBENT_PROMOTED, "engine", {
                    "exp_id": exp_id, "score": dev_score, "prev_exp_id": view.incumbent_id, "replicated": True,
                    "replication_exp_ids": [exp_id], "tree_hash": exp.tree_hash,
                    "sealed_score": cand_sealed, "sealed_delta": self._sealed_pair(True)[0],
                    "sealed_margin": self._sealed_pair(True)[1]})
        note = self.replicate_note
        outcome = Outcome.IMPROVED if ok else Outcome.NO_GAIN          # the sealed class: a leak if forwarded
        if sentinel is not None:
            note = f"{note} sealed delta {sentinel} margin {sentinel}".strip()
        return LaneResult(exp_id=exp_id, lane=exp.lane, status=ExpStatus.RECORDED, outcome=outcome, delta=None, score=None,
                          replicated=ok, notes=note)

    def reflect(self, trigger: str) -> dict:
        self.calls.append(("reflect", trigger))
        out = self.reflect_output or {"lessons": ["quadratic terms help"], "pivots": ["try damping"]}
        for text in out.get("lessons", []):
            self.memory.add_lesson(text, kind="strategy", classification="model")
        self._append(Kind.REFLECTION, "reflector", {"trigger": trigger, "consolidated": [], "pivots": list(out.get("pivots", []))})
        return {"lessons": list(out.get("lessons", [])), "pivots": list(out.get("pivots", []))}

    def reconcile(self) -> list[str]:
        self.calls.append(("reconcile",))
        touched: list[str] = []
        for exp in self.ledger.view().experiments.values():
            open_runs = [r for r in exp.runs if r.finished_seq is None]
            for run in open_runs:
                self._append(Kind.EXP_RUN_FINISHED, "engine", {
                    "exp_id": exp.exp_id, "run_idx": run.run_idx, "purpose": run.purpose, "seed": run.seed, "exit_code": None,
                    "duration_sec": 0.0, "classification": "interrupted", "stalled": False, "stdout_tail": "",
                    "stderr_tail": "", "note": "interrupted by a crash"})
            if open_runs and exp.exp_id not in touched:
                touched.append(exp.exp_id)
        return touched

    def resume_pending(self) -> list[LaneResult]:
        self.calls.append(("resume_pending",))
        results = []
        for exp in self.ledger.view().in_flight():
            if exp.lane == "baseline":
                results.append(self._complete_baseline(exp.exp_id, first_idx=len(exp.runs)))
                continue
            results.append(self._finish_lane(exp.exp_id, exp.lane, exp.hypothesis, first_idx=len(exp.runs)))
        return results


def fake_engine_factory(**options: Any) -> Callable[..., FakeEngine]:
    """An ``engine_factory`` for ``Orchestrator`` that builds a ``FakeEngine`` with ``options``; ``.instances`` lists them."""
    instances: list[FakeEngine] = []

    def factory(**kw: Any) -> FakeEngine:
        engine = FakeEngine(**kw, **options)
        instances.append(engine)
        return engine

    factory.instances = instances          # type: ignore[attr-defined]
    return factory


class _Dummy:
    identity = "dummy"


class SerialImplementer(_Dummy):
    """An implementer that declares ``requires_serial`` (as the Muse Code worker does): untrusted runs must not overlap."""

    identity = "serial-implementer"
    requires_serial = True


class RecordingSupervisor:
    """A supervisor whose behaviour is a callable ``fn(env)``; ``.runs`` counts how often it was started."""

    identity = "recording-supervisor"

    def __init__(self, fn: Optional[Callable[[Any], None]] = None):
        self.fn = fn
        self.runs = 0

    def run(self, env: Any) -> None:
        self.runs += 1
        if self.fn is not None:
            self.fn(env)


def make_suite(supervisor: Any = None, *, judge: Any = None, final_judge: Any = None, backend: str = "scripted",
               models: Optional[dict] = None, requires_serial: bool = False) -> AgentSuite:
    """An ``AgentSuite`` whose non-supervisor roles are inert placeholders (the FakeEngine never calls them).

    ``requires_serial=True`` gives it an implementer that declares the attribute, like the Muse Code worker."""
    return AgentSuite(backend=backend, supervisor=supervisor or RecordingSupervisor(), designer=_Dummy(),
                      implementer=SerialImplementer() if requires_serial else _Dummy(),
                      reviewer=_Dummy(), reflector=_Dummy(), judge=judge, final_judge=final_judge,
                      models=models if models is not None else {"supervisor": "fake-model"})
