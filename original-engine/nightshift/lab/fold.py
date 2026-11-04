"""Read model of the lab: master-ledger events -> ``LabState``, and the queries the other modules ask (``FoldReader`` is the ``StateReader``).

``fold_events`` never raises for bad data. Anything it cannot use is listed in ``LabState.malformed`` as ``(seq, reason)`` and skipped, and
``verify`` fails on a non-empty list. The cases (docs/harness-design.md, "Ledger side"):

* a non-lab or unknown kind, a trial-ledger kind in the master ledger, a payload that is not its contract record;
* a second event of a ``LabKind.FREEZE_ONCE`` kind (the first wins), a second ``lab.initialized``;
* a version label that maps to two commits; a duplicate trial id, score id, candidate id, final request id or decision for a candidate;
* a duplicate ``(phase, arm, pack commit, assignment, seed)`` trial (the arm is part of the key because a final-phase v0 and evolved trial
  of the same commit may share a planned seed when no candidate was accepted);
* a second selection or test score for the same ``(phase, split, adapter_sha256, assignment)``: the first completed record is the one used;
* a score whose trial has no ``lab.trial_started``; a ``lab.trial_finished`` without ``lab.trial_started``;
* a decision before the rule or without its candidate; a second final request for the same ``(arm, pack commit)`` without a valid
  ``repeat_of`` (excluded, so a guard that looks the request up by id refuses it).

``fold_events`` also leaves ``state.event_seq`` (an attribute, not a dataclass field): ``{(kind, key): seq}`` for the first event of each
kind-and-key, so a verifier can compare orders (for example the rule's sequence number with the first candidate's).

``FoldReader.selection_trial_scores`` is the one place that decides which trials count for a pack version and which score of a trial is
used. ``FoldReader.next_id`` allocates ids: it never hands out an id twice in this process.
"""

from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from typing import Iterable, Optional, Sequence

from nightshift.lab.contracts import (
    PAYLOAD_TYPES,
    AssignmentManifest,
    BudgetSpec,
    ContractError,
    CostBreakdown,
    FinalAnalysis,
    FinalRequest,
    Gate,
    GateOutcome,
    LabKind,
    LabLayout,
    LabState,
    LedgerAnchor,
    PackRef,
    Phase,
    Rule,
    Split,
    TrialScore,
    TrialStatus,
    Verifier,
    FailedTrialPolicy,
    comparability_key,
    load_frozen,
    payload_from_dict,
)
from nightshift.lab.ledger import LabLedger
from nightshift.types import Event

__all__ = ["fold_events", "fold_ledger", "FoldReader", "ID_PREFIXES"]

_FROZEN_ATTR = {
    LabKind.ASSIGNMENTS_FROZEN: "manifest",
    LabKind.BUDGET_FROZEN: "budget",
    LabKind.SPREAD_PLANNED: "spread_plan",
    LabKind.SPREAD_MEASURED: "spread",
    LabKind.RULE_FROZEN: "rule",
    LabKind.SEARCH_CLOSED: "search_closed",
    LabKind.FINAL_ANALYSIS_FROZEN: "final_analysis",
    LabKind.FINAL_REVEALED: "reveal",
}
_SCORED = (TrialStatus.COMPLETED, TrialStatus.BUDGET_EXHAUSTED)


class _Folder:
    def __init__(self) -> None:
        self.state = LabState()
        self.seq_of: dict[tuple[str, str], int] = {}
        self.trial_keys: dict[tuple, str] = {}
        self.score_keys: dict[tuple, str] = {}
        self.pack_commits: dict[str, str] = {}
        self.final_pairs: set[tuple] = set()
        self.final_ids: set[str] = set()
        self.seen_init = False

    def bad(self, seq: int, reason: str) -> None:
        self.state.malformed.append((seq, reason))

    def mark(self, kind: str, key: str, seq: int) -> None:
        self.seq_of.setdefault((kind, key), seq)

    def apply(self, ev: Event) -> None:
        kind, seq, st = ev.kind, ev.seq, self.state
        if not isinstance(kind, str) or kind not in LabKind.all():
            self.bad(seq, f"unknown or non-lab event kind {kind!r}")
            return
        if kind in LabKind.TRIAL:
            self.bad(seq, f"{kind} belongs in a trial ledger, not the master ledger")
            return
        try:
            rec = payload_from_dict(kind, ev.data)
        except (ContractError, TypeError, ValueError, KeyError, AttributeError) as exc:
            self.bad(seq, f"{kind}: bad payload ({exc})")
            return
        if not isinstance(rec, PAYLOAD_TYPES[kind]):  # pragma: no cover - payload_from_dict guarantees it
            self.bad(seq, f"{kind}: payload is not a {PAYLOAD_TYPES[kind].__name__}")
            return

        if kind in _FROZEN_ATTR:
            attr = _FROZEN_ATTR[kind]
            if getattr(st, attr) is not None:
                self.bad(seq, f"second {kind} (the first one wins)")
                return
            setattr(st, attr, rec)
            self.mark(kind, "", seq)
        elif kind == LabKind.INITIALIZED:
            if self.seen_init:
                self.bad(seq, "second lab.initialized (the first one wins)")
                return
            self.seen_init = True
            st.initialized = rec
        elif kind == LabKind.GATE_RECORDED:
            st.gates.append(rec)
        elif kind == LabKind.PACK_COMMITTED:
            old = self.pack_commits.get(rec.version_label)
            if old is not None and old != rec.pack.commit:
                self.bad(seq, f"version label {rec.version_label!r} maps to two commits")
                return
            self.pack_commits[rec.version_label] = rec.pack.commit
            st.packs.setdefault(rec.version_label, rec)
            self.mark(kind, rec.version_label, seq)
        elif kind == LabKind.TRIAL_STARTED:
            self._trial_started(seq, rec)
        elif kind == LabKind.TRIAL_FINISHED:
            tid = rec.spec.trial_id
            if tid not in st.trials_started:
                self.bad(seq, f"lab.trial_finished for {tid!r} without lab.trial_started")
            elif tid in st.trials:
                self.bad(seq, f"duplicate lab.trial_finished for {tid!r}")
            elif rec.spec.to_dict() != st.trials_started[tid].to_dict():
                self.bad(seq, f"lab.trial_finished for {tid!r} carries a spec that differs from its lab.trial_started")
            else:
                st.trials[tid] = rec
                self.mark(kind, tid, seq)
        elif kind == LabKind.SCORE_RECORDED:
            self._score(seq, rec)
        elif kind == LabKind.SCORE_COST:
            if rec.score_id in st.score_costs:
                self.bad(seq, f"duplicate lab.score_cost for {rec.score_id!r}")
            else:
                st.score_costs[rec.score_id] = rec
        elif kind == LabKind.SCORE_FAILED:
            st.score_failures.append(rec)
        elif kind == LabKind.CONTAMINATION_SCANNED:
            old = st.contamination.get(rec.trial_id)
            if old is None or old.passed:  # a hit is sticky: a later clean scan of another job must not hide it
                st.contamination[rec.trial_id] = rec
            self.mark(kind, rec.trial_id, seq)
        elif kind == LabKind.CANDIDATE_PROPOSED:
            if rec.candidate_id in st.candidates:
                self.bad(seq, f"duplicate lab.candidate_proposed for {rec.candidate_id!r}")
            else:
                st.candidates[rec.candidate_id] = rec
                self.mark(kind, rec.candidate_id, seq)
        elif kind == LabKind.CANDIDATE_DECIDED:
            if st.rule is None:
                self.bad(seq, f"decision for {rec.candidate_id!r} before lab.rule_frozen")
            elif rec.candidate_id not in st.candidates:
                self.bad(seq, f"decision for {rec.candidate_id!r}, which has no lab.candidate_proposed")
            elif rec.candidate_id in st.decisions:
                self.bad(seq, f"second decision for {rec.candidate_id!r} (the first one wins)")
            else:
                st.decisions[rec.candidate_id] = rec
                self.mark(kind, rec.candidate_id, seq)
        elif kind == LabKind.OPERATOR_CHANGE:
            st.operator_changes.append(rec)
        elif kind == LabKind.FINAL_REQUESTED:
            self._final(seq, rec)

    def _trial_started(self, seq: int, spec) -> None:
        st = self.state
        if spec.trial_id in st.trials_started:
            self.bad(seq, f"duplicate lab.trial_started for {spec.trial_id!r}")
            return
        key = (spec.phase.value, None if spec.arm is None else spec.arm.value, spec.pack.commit, spec.assignment_id, spec.seed)
        if key in self.trial_keys:
            self.bad(seq, f"trial {spec.trial_id!r} repeats (phase, arm, pack commit, assignment, seed) of {self.trial_keys[key]!r}")
            return
        self.trial_keys[key] = spec.trial_id
        st.trials_started[spec.trial_id] = spec
        self.mark(LabKind.TRIAL_STARTED, spec.trial_id, seq)

    def _score(self, seq: int, rec) -> None:
        st = self.state
        if rec.score_id in st.scores:
            self.bad(seq, f"duplicate lab.score_recorded for {rec.score_id!r}")
            return
        if rec.trial_id is not None and rec.trial_id not in st.trials_started:
            self.bad(seq, f"score {rec.score_id!r} belongs to trial {rec.trial_id!r}, which has no lab.trial_started")
            return
        if rec.split in (Split.SELECTION, Split.TEST):
            key = (rec.phase.value, rec.split.value, rec.adapter_sha256, rec.assignment_id)
            if key in self.score_keys:
                self.bad(seq, f"second {rec.split.value} score for the same adapter and assignment as {self.score_keys[key]!r} (the first one is used)")
                return
            self.score_keys[key] = rec.score_id
        st.scores[rec.score_id] = rec
        self.mark(LabKind.SCORE_RECORDED, rec.score_id, seq)

    def _final(self, seq: int, rec: FinalRequest) -> None:
        st = self.state
        if rec.request_id in self.final_ids:
            self.bad(seq, f"duplicate lab.final_requested for {rec.request_id!r}")
            return
        pair = (rec.arm.value, None if rec.pack is None else rec.pack.commit)
        if pair in self.final_pairs:
            if rec.repeat_of is None:
                self.bad(seq, f"second final request for {pair[0]} / {pair[1]} without repeat_of")
                return
            if rec.repeat_of not in self.final_ids:
                self.bad(seq, f"final request {rec.request_id!r} repeats unknown request {rec.repeat_of!r}")
                return
        self.final_ids.add(rec.request_id)
        self.final_pairs.add(pair)
        st.finals.append(rec)
        self.mark(LabKind.FINAL_REQUESTED, rec.request_id, seq)


def fold_events(events: Iterable[Event]) -> LabState:
    """Fold master-ledger events (in ledger order) into a ``LabState``. Malformed events are listed, never fatal."""
    folder = _Folder()
    last: Optional[Event] = None
    for ev in events:
        folder.apply(ev)
        last = ev
    st = folder.state
    if last is not None:
        try:
            st.head = LedgerAnchor(last.seq, last.hash[:16])
        except ContractError:  # a hand-built event with a non-hex hash; real ledger events always have one
            st.head = None
    st.event_seq = folder.seq_of  # type: ignore[attr-defined]  # not a dataclass field; see the module docstring
    return st


def fold_ledger(dir: Path | str) -> LabState:
    """Fold the master ledger in ``dir`` (opened read-only: no lock, so it works while the coordinator is writing)."""
    led = LabLedger.read_only(dir)
    try:
        return fold_events(led.events())
    finally:
        led.close()


ID_PREFIXES = ("t", "cand", "ad", "a", "j", "sc", "f")
_ID_WIDTH = {"cand": 3}  # cand-001; every other prefix uses four digits


class FoldReader:
    """Implements ``StateReader`` over the master ledger of ``layout``.

    ``state()`` folds the file each time (read-only, no lock). Pass the coordinator's own ``LabLedger`` as ``ledger`` to fold its in-memory
    events instead, which is faster and sees an append the instant it returns.
    """

    def __init__(self, layout: LabLayout, *, ledger: Optional[LabLedger] = None) -> None:
        self.layout = layout
        self._ledger = ledger
        self._issued: set[str] = set()
        self._id_lock = threading.Lock()

    # ---- state ----------------------------------------------------------------------------------
    def state(self) -> LabState:
        if self._ledger is not None:
            return fold_events(self._ledger.events())
        return fold_ledger(self.layout.master_ledger_dir)

    def incumbent_version(self, state: LabState) -> str:
        """The version label of the latest accepted candidate (ledger order), else ``v0``."""
        accepted = [d.candidate_version for d in state.decisions.values() if d.accepted]
        return accepted[-1] if accepted else "v0"

    def has_gate(self, state: LabState, gate: Gate) -> bool:
        """True when the owner has recorded a decision on ``gate`` (either outcome). It lets gated work start; it does not say the gate is closed."""
        return any(g.gate == gate for g in state.gates)

    def gate_status(self, state: LabState, gate: Gate) -> str:
        """``closed`` only after a ``CLARIFIED`` record; ``open_waived`` when the only record is ``PROCEED_WITH_RISK`` (the spec: accepting the risk
        does not close the gate); ``unrecorded`` otherwise."""
        outcomes = {g.outcome for g in state.gates if g.gate == gate}
        if GateOutcome.CLARIFIED in outcomes:
            return "closed"
        return "open_waived" if GateOutcome.PROCEED_WITH_RISK in outcomes else "unrecorded"

    def final_requests(self, state: LabState, pack: Optional[PackRef]) -> list[FinalRequest]:
        """Final requests for ``pack`` (the same repository and commit); ``None`` lists the arm-1 (stock) requests."""
        return [r for r in state.finals if r.pack == pack]

    # ---- frozen files ---------------------------------------------------------------------------
    def load_frozen_all(self, state: LabState) -> tuple[Rule, AssignmentManifest, BudgetSpec]:
        """Parse the rule, the manifest and the budget, re-hashing each file against the ledger's ``FileRef`` first. ``ContractError`` when
        one is not frozen, was rewritten or is not a regular file."""
        for name, frozen in (("rule", state.rule), ("assignments", state.manifest), ("budget", state.budget)):
            if frozen is None:
                raise ContractError(f"{name} is not frozen in the ledger")
        return (
            load_frozen(self.layout, state.rule, Rule),
            load_frozen(self.layout, state.manifest, AssignmentManifest),
            load_frozen(self.layout, state.budget, BudgetSpec),
        )

    def load_final_analysis(self, state: LabState) -> FinalAnalysis:
        if state.final_analysis is None:
            raise ContractError("the final analysis is not frozen in the ledger")
        return load_frozen(self.layout, state.final_analysis, FinalAnalysis)

    # ---- the statistics input -------------------------------------------------------------------
    def selection_trial_scores(self, state: LabState, rule: Rule, pack_version: str) -> list[TrialScore]:
        """One ``TrialScore`` per finished search-phase trial of ``pack_version`` on ``rule.assignments``, in ledger order.

        A trial with status COMPLETED or BUDGET_EXHAUSTED counts normally with ``rule.metric`` of its selection score (mean over seeds).
        Every other finished trial, a contaminated one, and one whose score differs from the rule (metric, scorer version, decoding,
        seeds, verifier, split, assignment, adapter; the mismatch is added to ``state.malformed``) follows ``rule.failed_trial_policy``:
        INVALIDATES_CANDIDATE returns ``[]`` (the whole set is unusable), SCORED_ZERO adds a 0.0 score with ``score_id`` None. A trial that
        has not finished yet is not counted.
        """
        out: list[TrialScore] = []
        for trial_id, spec in state.trials_started.items():
            if spec.phase is not Phase.SEARCH or spec.pack_version != pack_version or spec.assignment_id not in rule.assignments:
                continue
            rec = state.trials.get(trial_id)
            if rec is None:
                continue
            ts = self._trial_score(state, rule, trial_id, spec, rec)
            if ts is None:
                if rule.failed_trial_policy is FailedTrialPolicy.INVALIDATES_CANDIDATE:
                    return []
                ts = TrialScore(trial_id, spec.assignment_id, None, rule.metric, rule.scorer_version, comparability_key(spec, rule.metric), 0.0)
            out.append(ts)
        return out

    def _trial_score(self, state, rule: Rule, trial_id: str, spec, rec) -> Optional[TrialScore]:
        if rec.status not in _SCORED or rec.selection_score_id is None or rec.adapter is None:
            return None
        hit = state.contamination.get(trial_id)
        if hit is not None and not hit.passed:
            return None
        score = state.scores.get(rec.selection_score_id)
        if score is None:
            self._flag(state, 0, f"trial {trial_id!r} names selection score {rec.selection_score_id!r}, which is not in the ledger")
            return None
        problems = []
        if score.split is not Split.SELECTION or score.phase is not Phase.SEARCH:
            problems.append("split or phase")
        if score.trial_id != trial_id or score.assignment_id != spec.assignment_id:
            problems.append("trial or assignment")
        if score.base != spec.base:
            problems.append("base checkpoint")
        if score.verifier is not Verifier.IFEVALG:
            problems.append("verifier")
        if score.scorer_version != rule.scorer_version:
            problems.append("scorer version")
        if score.params != rule.score_params:
            problems.append("decoding")
        if {s.seed for s in score.per_seed} != set(rule.score_seeds):
            problems.append("seeds")
        if any(rule.metric not in s.metrics for s in score.per_seed):
            problems.append("metric")
        if score.adapter_sha256 != rec.adapter.sha256:
            problems.append("adapter")
        if problems:
            seq = getattr(state, "event_seq", {}).get((LabKind.SCORE_RECORDED, score.score_id), 0)
            self._flag(state, seq, f"selection score {score.score_id!r} differs from the rule: {', '.join(problems)}")
            return None
        values = score.metric_by_seed(rule.metric)
        return TrialScore(trial_id, spec.assignment_id, score.score_id, rule.metric, rule.scorer_version, comparability_key(spec, rule.metric), math.fsum(values) / len(values))

    @staticmethod
    def _flag(state: LabState, seq: int, reason: str) -> None:
        entry = (seq, reason)
        if entry not in state.malformed:
            state.malformed.append(entry)

    # ---- costs ----------------------------------------------------------------------------------
    def cost_breakdown(self, state: LabState, trial_ids: Sequence[str], candidate_id: Optional[str]) -> CostBreakdown:
        """The four cost centres for these trials: the candidate's propose cost, the trials' booked researcher and GPU dollars (the
        meter's ticks, from ``TrialRecord.budget_used``) and the GPU billed for scoring their scores (``lab.score_cost``). A trial that
        has not finished contributes nothing; money reporting also reads the rental ledger."""
        wanted = set(trial_ids)
        propose = state.candidates[candidate_id].propose_usd if candidate_id is not None and candidate_id in state.candidates else 0.0
        researcher = math.fsum(state.trials[t].budget_used.researcher_usd for t in wanted if t in state.trials)
        gpu = math.fsum(state.trials[t].budget_used.gpu_usd for t in wanted if t in state.trials)
        failed_owner = {f.request_id: f.trial_id for f in state.score_failures}  # a failed attempt is billed too
        scorer = math.fsum(
            cost.gpu.usd for sid, cost in state.score_costs.items()
            if (state.scores[sid].trial_id if sid in state.scores else failed_owner.get(sid)) in wanted
        )
        return CostBreakdown(propose, researcher, gpu, scorer)

    # ---- ids ------------------------------------------------------------------------------------
    def next_id(self, state: LabState, prefix: str, *, taken: Iterable[str] = (), reserve: bool = True) -> str:
        """The next unused ``<prefix>-<NNNN>`` (``cand-NNN``). Used ids come from the ledger state, from the lab directory on disk
        (trial, score, adapter and candidate directories, train-data files), from ``taken`` and, when ``reserve`` is true, from every
        id this reader has already issued, so two calls (or two threads) never get the same id. Prefixes: t, cand, ad, a, j, sc, f."""
        if prefix not in ID_PREFIXES:
            raise ContractError(f"next_id: prefix {prefix!r} is not one of {ID_PREFIXES}")
        pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
        used = set(taken) | self._ids_in_state(state, prefix) | self._ids_on_disk(prefix)
        with self._id_lock:
            used |= self._issued
            numbers = [int(m.group(1)) for m in (pattern.match(u) for u in used) if m]
            n = (max(numbers) + 1) if numbers else 1
            new = f"{prefix}-{n:0{_ID_WIDTH.get(prefix, 4)}d}"
            if reserve:
                self._issued.add(new)
            return new

    def _ids_in_state(self, state: LabState, prefix: str) -> set[str]:
        ids: set[str] = set()
        if prefix == "t":
            ids |= set(state.trials_started) | set(state.trials)
            ids |= {s.trial_id for s in state.scores.values() if s.trial_id}
            ids |= {s.trial_id for s in state.score_failures if s.trial_id}
            if state.spread is not None:
                ids |= {t.trial_id for t in state.spread.trials}
            try:
                ids |= {p.trial_id for p in self.load_final_analysis(state).planned_trials}
            except (ContractError, OSError):
                pass
        elif prefix == "cand":
            ids |= set(state.candidates) | set(state.decisions)
            ids |= {p.candidate_id for p in state.packs.values() if p.candidate_id}
        elif prefix == "ad":
            ids |= {t.adapter.adapter_id for t in state.trials.values() if t.adapter is not None}
        elif prefix == "a":
            ids |= {s.assignment_id for s in state.trials_started.values()}
            ids |= {s.assignment_id for s in state.scores.values()}
            try:
                if state.manifest is not None:
                    ids |= {a.assignment_id for a in load_frozen(self.layout, state.manifest, AssignmentManifest).assignments}
            except (ContractError, OSError):
                pass
        elif prefix == "j":
            ids |= {c.job_id for c in state.contamination.values()}
        elif prefix == "sc":
            ids |= set(state.scores) | set(state.score_costs) | {f.request_id for f in state.score_failures}
            for t in state.trials.values():
                ids |= {i for i in (t.dev_score_id, t.selection_score_id) if i}
        elif prefix == "f":
            ids |= {r.request_id for r in state.finals}
            ids |= {s.final_request_id for s in state.trials_started.values() if s.final_request_id}
        return ids

    def _ids_on_disk(self, prefix: str) -> set[str]:
        root = self.layout.root
        names: set[str] = set()

        def dirs(p: Path) -> None:
            try:
                names.update(c.name for c in p.iterdir())
            except OSError:
                pass

        if prefix == "t":
            dirs(root / "trials")
        elif prefix == "cand":
            dirs(root / "candidates")
        elif prefix == "sc":
            dirs(root / "scores")
        elif prefix == "ad":
            dirs(self.layout.artifacts_dir / "adapters")
        elif prefix == "a":
            dirs(self.layout.public_data_dir / "assignments")
        elif prefix == "j":
            try:
                for trial in (root / "trials").iterdir():
                    names.update(p.stem for p in (trial / "train_data").glob("*.jsonl"))
            except OSError:
                pass
        return names
