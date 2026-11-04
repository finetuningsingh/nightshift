"""Hard gates (docs/ARCHITECTURE.md 7.2, 7.3, 7.8): pure functions of a ``LedgerView`` that return refusal reasons.

Guarantees
----------
* A gate returns a list of reasons; an empty list means allowed. Gates never raise on malformed ledger content or
  malformed arguments (they return a reason instead). Only a ``NightshiftError`` from a collaborator (budget gate,
  memory) other than ``BudgetExceeded`` propagates.
* Reasons are shown to the supervisor, so they are templates plus validated ids (experiment, claim and finding ids
  matching ``[A-Za-z0-9_.@:-]``). They contain no model-written text and no number derived from a sealed-split grade:
  the sealed split's outcome is reported only as "not replicated" or "the improvement over the baseline is not confirmed".
  Reasons never name the sealed split: the agents' prompts call it "the independent re-check".
* Nothing here reads the fault-injection labels of the demo; gates judge only recorded evidence.
* The only I/O is the existence check of ``run_dir/HALT`` in ``launch_gate``.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from .. import grading
from ..types import (BudgetExceeded, BudgetLimits, ExpStatus, GradeResult, LedgerView, NightshiftError, Outcome, Purpose,
                     TaskSpec)

MAX_CLAIMS = 50          # a finish citing more claims than this is refused outright
MAX_LISTED = 8           # ids named in one reason before "and N more"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.@:-]{1,60}$")
_BASES = ("incumbent", "template")
# Findings that say only "no replication yet" (I09 replicate-before-final, I14 summary-matches-ledger): a finish that cites an
# unreplicated experiment raises them. Promotion already demands a recorded sealed replication, so they cannot block it.
_PENDING_REPLICATION_INVARIANTS = frozenset({"I09", "I14"})


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _ids(values: Sequence[Any]) -> str:
    """Comma-separated ids for a reason. Ids that do not look like ledger ids are replaced, never echoed."""
    shown = [v if isinstance(v, str) and _SAFE_ID.match(v) else "<malformed id>" for v in list(values)[:MAX_LISTED]]
    more = len(values) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


def _sorted_ids(values: Any) -> list:
    """Sort ids of any JSON type (a malformed finding may carry a non-string id) without ever comparing across types."""
    return sorted(values, key=lambda v: (not isinstance(v, str), str(v)))


def _one(value: Any) -> str:
    return _ids([value])


def _says_only_no_replication_yet(finding: Any) -> bool:
    """True for a deterministic I09 or I14 finding. An LLM-written I14 finding (hidden failure, a scripted result presented
    as a model result) says more than "no replication yet", and a malformed invariant id (unhashable, not a string) is not
    one of them either: both keep blocking."""
    invariant = getattr(finding, "invariant_id", None)
    return (isinstance(invariant, str) and invariant in _PENDING_REPLICATION_INVARIANTS
            and getattr(finding, "source", None) == "deterministic")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:       # an int too large for a float, e.g. 10**400 in a hand-edited ledger
        return None
    return number if math.isfinite(number) else None


def _limits_of(view: LedgerView, limits: Optional[BudgetLimits]) -> BudgetLimits:
    """``limits`` if given, else the run's recorded limits; any field that is not a finite number falls back to the default."""
    defaults = BudgetLimits()
    source: Any = limits if isinstance(limits, BudgetLimits) else None
    if source is None:
        raw = view.run.get("limits") if isinstance(view.run, dict) else None
        if not isinstance(raw, dict):
            return defaults
        try:
            source = BudgetLimits.from_dict(raw)
        except Exception:  # noqa: BLE001 - a malformed run.started must not stop a gate
            return defaults
    fields = {name: getattr(source, name, None) for name in vars(defaults)}
    return BudgetLimits(**{name: value if _number(value) is not None else getattr(defaults, name)
                           for name, value in fields.items()})


def _grade(view: LedgerView, exp_id: Any, purpose: str) -> Optional[GradeResult]:
    if not isinstance(exp_id, str):
        return None
    exp = view.experiments.get(exp_id)
    rec = exp.grade_for(purpose) if exp is not None else None
    grade = rec.grade if rec is not None else None
    return grade if isinstance(grade, GradeResult) else None


def _baseline_sigma(view: LedgerView) -> float:
    """The baseline's per-seed std on the dev split (the statistical prior for every comparison)."""
    base = view.baseline
    grade = base.primary_grade if base is not None else None
    std = _number(grade.std) if grade is not None else None
    return max(0.0, std) if std is not None else 0.0


# --------------------------------------------------------------------------------------------------
# sealed comparison
# --------------------------------------------------------------------------------------------------
def sealed_decision(task: TaskSpec, view: LedgerView, cand_id: str, base_id: str) -> Optional[grading.Decision]:
    """Compare ``cand_id``'s sealed ``replication`` grade with ``base_id``'s sealed grade (two-sided 95%).

    The base grade is ``base_id``'s own ``replication`` grade, or its ``replication_baseline`` grade when ``base_id`` is the
    baseline experiment. ``sigma_prior`` is the baseline's dev per-seed std. Returns None when either grade is missing or
    the comparison cannot be computed. The result carries sealed numbers: it is for the engine and the gates, never a model.
    """
    cand = _grade(view, cand_id, Purpose.REPLICATION)
    base = _grade(view, base_id, Purpose.REPLICATION_BASELINE if base_id == view.baseline_id else Purpose.REPLICATION)
    if cand is None or base is None:
        return None
    try:
        return grading.decide_outcome(task, cand, base, k_sigma=grading.T975, sigma_prior=_baseline_sigma(view))
    except (ValueError, TypeError, ArithmeticError, AttributeError):
        return None


# --------------------------------------------------------------------------------------------------
# dev comparisons (dev numbers only)
# --------------------------------------------------------------------------------------------------
def dev_superseded(task: TaskSpec, view: LedgerView, exp_id: str) -> bool:
    """True when ``exp_id`` no longer beats the CURRENT incumbent on the dev split, so a replication would be wasted.

    A candidate that improved on the incumbent of its time stays "awaiting replication" after a better result has been
    promoted. Its dev grade is compared with the current incumbent's dev grade through ``grading.decide_outcome`` with the
    sigma prior the engine uses (the baseline's dev per-seed std); superseded means the outcome is not ``improved``.
    Dev numbers only: no sealed grade is read. False when the answer cannot be computed (unknown experiment, no dev grade,
    no incumbent, not comparable, malformed ledger) and for the incumbent itself: only a computed "does not beat it" refuses.
    """
    exp = view.experiments.get(exp_id) if isinstance(exp_id, str) else None
    inc = view.incumbent
    if exp is None or inc is None or exp.exp_id == inc.exp_id:
        return False
    cand, base = exp.primary_grade, inc.primary_grade
    if not isinstance(cand, GradeResult) or not isinstance(base, GradeResult):
        return False
    try:
        decision = grading.decide_outcome(task, cand, base, sigma_prior=_baseline_sigma(view))
    except (ValueError, TypeError, ArithmeticError, AttributeError):
        return False
    return decision.outcome in (Outcome.NO_GAIN, Outcome.WORSE)


def never_tested(exp: Any) -> bool:
    """True for a recorded experiment that ended ``unresolved`` before any untrusted run started (the implementer made no
    change, or the run was stopped first): its hypothesis was not tested, so it must not count as a failed one."""
    return (getattr(exp, "dev_outcome", None) == Outcome.UNRESOLVED and not getattr(exp, "runs", None)
            and not getattr(exp, "grades", None))


def stall_streak(view: LedgerView) -> int:
    """Trailing recorded lane experiments (record order) that did not improve, skipping infrastructure failures and
    experiments whose hypothesis was never tested (``never_tested``). A candidate that is awaiting replication or replicated
    ends the streak; one whose replication failed counts as non-improving. Same walk as ``Memory.consecutive_non_improving``,
    with untested experiments neither counted nor breaking the streak."""
    lane_exps = sorted((e for e in view.experiments.values() if e.status == ExpStatus.RECORDED and e.lane != "baseline"
                        and e.outcome != Outcome.BASELINE and not e.replicate_of), key=lambda e: (e.recorded_seq or 0, e.proposed_seq))
    count = 0
    for exp in reversed(lane_exps):
        final = exp.dev_outcome if exp.sealed_outcome == Outcome.INFRA else exp.outcome
        if final == Outcome.INFRA or exp.dev_outcome == Outcome.INFRA or never_tested(exp):
            continue
        if final == Outcome.IMPROVED:
            break
        count += 1
    return count


# --------------------------------------------------------------------------------------------------
# promotion
# --------------------------------------------------------------------------------------------------
def promotion_gate(view: LedgerView, exp_id: str) -> list[str]:
    """Reasons to refuse promoting ``exp_id`` to incumbent; [] means allowed.

    Allowed only for an experiment that was recorded ``improved`` on dev, has just replicated on the sealed split, was never
    promoted before (a superseded incumbent cannot come back), is not already the incumbent, has no open ``block`` finding
    about it and no protected-path violation. An experiment that still awaits replication is refused ("not replicated"):
    every promotion needs a sealed grade (invariant I09). Open deterministic I09 and I14 findings about the experiment itself
    are ignored (an LLM-written I14 finding is not): they only record that a finish cited it before it was replicated, which the recorded sealed grade now answers.
    Every other open ``block`` finding about it still refuses.
    """
    exp = view.experiments.get(exp_id) if isinstance(exp_id, str) else None
    if exp is None:
        return ["unknown experiment " + _one(exp_id)]
    reasons: list[str] = []
    if exp_id == view.incumbent_id:
        reasons.append("already the incumbent")
    elif any(isinstance(p, dict) and p.get("exp_id") == exp_id for p in view.promotions):
        reasons.append("already promoted once and since superseded")
    if exp.dev_outcome != Outcome.IMPROVED:
        reasons.append("not recorded as improved on the dev split")
    elif exp.grade_for(Purpose.REPLICATION) is None:
        reasons.append("not replicated yet: it must pass the independent re-check before promotion")
    elif not exp.replicated or exp.sealed_outcome != Outcome.IMPROVED:
        reasons.append("not replicated: it did not replicate on the independent re-check")
    else:
        delta, margin = _number(exp.sealed_delta), _number(exp.sealed_margin)
        if delta is not None and margin is not None and not delta > margin:
            reasons.append("not replicated: the independent re-check does not show an improvement")
    if exp.protected_violations:
        reasons.append("it has protected-path violations")
    blocking = _sorted_ids(f.finding_id for f in view.open_findings("block")
                           if f.exp_id == exp_id and not _says_only_no_replication_yet(f))
    if blocking:
        reasons.append("open block findings about it: " + _ids(blocking))
    return reasons


# --------------------------------------------------------------------------------------------------
# launch
# --------------------------------------------------------------------------------------------------
def _budget_reasons(gate: Any, action: str) -> list[str]:
    try:
        gate.check(action)
    except BudgetExceeded as exc:
        kind = exc.kind if isinstance(exc.kind, str) and _SAFE_ID.match(exc.kind) else action
        return [f"budget exhausted: {kind}"]
    return []


def _replication_reasons(view: LedgerView, target: Any, limits: BudgetLimits, task: Optional[TaskSpec] = None) -> list[str]:
    exp = view.experiments.get(target) if isinstance(target, str) else None
    if exp is None:
        return ["replication target does not exist"]
    reasons: list[str] = []
    if not exp.awaiting_replication:
        reasons.append(f"{_one(target)} is not awaiting replication")
    elif task is not None and dev_superseded(task, view, target):
        reasons.append("superseded on dev: a better result has been promoted since")
    used = _number(view.budget_used.get("replications")) or 0.0
    if used >= limits.max_replications:
        reasons.append("budget exhausted: replication")
    return reasons


def _tried_twin(view: LedgerView, memory: Any, hypothesis: str, first_hit: Any) -> Any:
    """``first_hit`` (the memory's earliest recorded twin of ``hypothesis``) unless it was never tested, in which case the
    earliest twin that really was tried, else None: a hypothesis whose implementer changed nothing may be proposed again."""
    first = view.experiments.get(first_hit) if isinstance(first_hit, str) else None
    if first is None or not never_tested(first):
        return first_hit
    fp = memory.fingerprint(hypothesis)
    twins = [e for e in view.experiments.values()
             if e.status == ExpStatus.RECORDED and e.lane != "baseline" and e.outcome != Outcome.BASELINE and not e.replicate_of
             and e.dev_outcome != Outcome.INFRA and not never_tested(e) and fp in (e.hyp_fingerprint, memory.fingerprint(e.hypothesis))]
    return min(twins, key=lambda e: e.proposed_seq).exp_id if twins else None


def _explore_reasons(view: LedgerView, lane: Any, memory: Any) -> list[str]:
    reasons: list[str] = []
    hypothesis = getattr(lane, "hypothesis", None)
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        reasons.append("the hypothesis is empty")
    else:
        try:
            normalised = memory.fingerprint(hypothesis)
            empty = normalised == memory.fingerprint("")
            dup = None if empty else _tried_twin(view, memory, hypothesis, memory.seen(hypothesis))
        except NightshiftError:
            raise
        except Exception:  # noqa: BLE001 - a malformed ledger must produce a reason, not a crash
            reasons.append("the duplicate check could not be completed")
        else:
            if empty:
                reasons.append("the hypothesis is empty")
            elif dup:
                reasons.append(f"duplicate of {_one(dup)}: the same hypothesis was already tried")
    base = getattr(lane, "base", None)
    if not isinstance(base, str) or not base:
        reasons.append("base must be 'incumbent', 'template' or a recorded experiment id")
    elif base not in _BASES:
        target = view.experiments.get(base)
        if target is None or target.status != "recorded":
            reasons.append("base must be 'incumbent', 'template' or a recorded experiment id")
    return reasons


def launch_gate(view: LedgerView, lane: Any, memory: Any, *, gate: Any = None, run_dir: Any = None,
                limits: Optional[BudgetLimits] = None, task: Optional[TaskSpec] = None) -> list[str]:
    """Reasons to refuse launching ``lane``; [] means allowed.

    Checks, in order: a HALT file in ``run_dir``; the budget (``gate.check`` for wall time and for an experiment or a
    replication); then, for a replication lane (``lane.replicate_of``), that the target exists, awaits replication and the
    replication limit is not reached and, when ``task`` is given, that the candidate still beats the current incumbent on dev
    (``dev_superseded``: a replication of a dominated candidate would waste one of the few sealed queries); for an explore lane, a non-empty hypothesis, no recorded duplicate (``memory.seen``)
    and a ``base`` that is ``incumbent``, ``template`` or a recorded experiment id.
    """
    reasons: list[str] = []
    if run_dir is not None:
        try:
            if (Path(run_dir) / "HALT").exists():
                reasons.append("HALT file present: the run was asked to stop")
        except (OSError, TypeError, ValueError):
            reasons.append("run directory cannot be checked for HALT")
    replicate_of = getattr(lane, "replicate_of", None)
    if gate is not None:
        reasons += _budget_reasons(gate, "wall")
        reasons += _budget_reasons(gate, "replication" if replicate_of else "experiment")
    if replicate_of:
        reasons += _replication_reasons(view, replicate_of, _limits_of(view, limits), task)
    else:
        reasons += _explore_reasons(view, lane, memory)
    return list(dict.fromkeys(reasons))


# --------------------------------------------------------------------------------------------------
# finish
# --------------------------------------------------------------------------------------------------
def _claim_reasons(task: TaskSpec, view: LedgerView, claim_id: str) -> list[str]:
    claim = view.claims.get(claim_id)
    if not isinstance(claim, dict):
        return [f"unknown claim {_one(claim_id)}"]
    ctype = claim.get("claim_type")
    cid = _one(claim_id)
    baseline_is_incumbent = view.incumbent_id is not None and view.incumbent_id == view.baseline_id
    if ctype != "improved":
        if baseline_is_incumbent and ctype != "no_effect":
            return [f"claim {cid}: while the baseline is the incumbent only no_effect claims can be cited"]
        return []
    exp_id = claim.get("exp_id")
    if not isinstance(exp_id, str) or exp_id not in view.experiments:
        return [f"improved claim {cid} does not refer to a known experiment"]
    if baseline_is_incumbent:
        return [f"improved claim {cid}: nothing has beaten the baseline, so only no_effect claims can be cited"]
    if exp_id != view.incumbent_id:
        return [f"improved claim {cid} refers to an experiment that is not the incumbent"]
    exp = view.experiments[exp_id]
    if not exp.replicated:
        return [f"improved claim {cid}: {_one(exp_id)} is not replicated"]
    if view.baseline_id is None:
        return [f"improved claim {cid}: there is no baseline to compare with"]
    decision = sealed_decision(task, view, exp_id, view.baseline_id)
    if decision is None or decision.outcome != Outcome.IMPROVED:
        # one wording for "no comparison" and "no improvement": the refusal must not reveal which of them it was
        return [f"improved claim {cid}: the improvement over the baseline is not confirmed"]
    return []


def finish_gate(view: LedgerView, task: TaskSpec, claim_ids: Any, *, limits: Optional[BudgetLimits] = None) -> list[str]:
    """Reasons to refuse ``finish``; [] means accepted (contract 7.8).

    Refused unless the baseline is recorded, nothing is in flight (an experiment awaiting replication is terminal and
    does not count), no ``block`` finding is open, and every cited claim exists and is acceptable: a cited ``improved``
    claim must be about the incumbent, which must be replicated and beat the baseline on the sealed split; while the
    baseline is the incumbent only ``no_effect`` claims may be cited. An honest finish with no improvement claim passes.
    ``limits`` is accepted for symmetry with ``launch_gate`` and is not needed by the finish rules.
    """
    reasons: list[str] = []
    if view.baseline_id is None or view.baseline is None or view.baseline.status != "recorded":
        reasons.append("the baseline has not been recorded")
    in_flight = [e.exp_id for e in view.in_flight() if not e.awaiting_replication]
    if in_flight:
        reasons.append("experiments still in flight: " + _ids(in_flight))
    blocking = _sorted_ids(f.finding_id for f in view.open_findings("block"))
    if blocking:
        reasons.append("open block findings: " + _ids(blocking))

    if claim_ids is None:
        claim_ids = []
    if isinstance(claim_ids, (str, bytes)) or not isinstance(claim_ids, (list, tuple)):
        reasons.append("claim_ids must be a list of claim ids")
        return reasons
    if len(claim_ids) > MAX_CLAIMS:
        reasons.append("too many claims cited")
        return reasons
    for claim_id in dict.fromkeys(c for c in claim_ids if isinstance(c, str)):
        reasons += _claim_reasons(task, view, claim_id)
    if any(not isinstance(c, str) for c in claim_ids):
        reasons.append("claim_ids must be strings")
    return reasons
