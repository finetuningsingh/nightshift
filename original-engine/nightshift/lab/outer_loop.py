"""The outer-loop controller (owner 4): propose, screen, confirm, decide, accept.

One ``run_round`` = one screening round (up to ``rule.screen_batch_size`` candidates, never more than 5) plus at most one confirmation.
The CONTROLLER decides, from the frozen rule and the control-host numbers; the proposer only proposes (docs/harness-design.md,
"Controller"). Everything the proposer sees goes through ``outer_context.build_context`` and ``assert_no_forbidden``.

Facts the design leaves open, decided here:

* Trials run one after the other (a second GPU rental is blocked while one is unsettled anyway); the confirmation interleaves candidate
  and incumbent trials, alternating which goes first.
* Trial seeds are fresh per (pack commit, assignment): the next integer above every seed the ledger holds, so a confirmation never
  repeats a screening trial and a restarted controller never repeats a seed.
* ``shape`` and ``base`` of a trial are taken from the incumbent's existing search trials (they are part of the comparability key, so a
  candidate must use exactly what the incumbent used). Pass them explicitly to override.
* A round is not resumed. At the start of a round every candidate that has no decision (the process died mid-round) gets an INVALID
  SCREEN decision "interrupted"; it still counts toward ``rule.max_candidates``. A search trial that started and never finished must be
  closed with ``cli abort-trial`` first; the controller refuses to start while one exists.
* The ledger is the authority: the accepting decision is appended BEFORE ``PackStore.accept`` moves ``main``; the next round reads the
  incumbent from the ledger, and ``accept`` is idempotent, so a crash between the two is repaired by calling ``accept`` again.
* ``stop_after_rejections`` counts consecutive ROUNDS (decisions share a ``round_index``) in which no candidate was accepted.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from nightshift import secrets
from nightshift.lab.contracts import (
    BaseModelRef,
    BudgetExhausted,
    BudgetSpec,
    CandidateRecord,
    Clock,
    ContractError,
    Decision,
    DecisionNumbers,
    DecisionRule,
    DecisionStage,
    EditOp,
    Gate,
    GateNotRecorded,
    JobStatus,
    LabError,
    LabKind,
    LabLayout,
    LabState,
    LedgerWriter,
    Lever,
    Outcome,
    PackCommitted,
    PackEdit,
    PackRef,
    PackStore,
    Phase,
    Proposal,
    ProposalInvalid,
    ProposalSource,
    Rule,
    Shape,
    StateReader,
    TrainResult,
    TrialExecutor,
    TrialRecord,
    TrialScore,
    TrialSpec,
    Split,
    check_trial_against_frozen,
    loads_strict,
    read_regular_bytes,
    utc_iso,
    write_record,
)
from nightshift.lab.outer_context import (
    CandidateFact,
    ContextLimits,
    ProposalFacts,
    TrialEvidence,
    VersionDev,
    assert_no_forbidden,
    build_context,
    contract_text,
    default_fence,
    failure_signatures,
    forbidden_values,
    summarize_trace,
)
from nightshift.lab.packs import DEFAULT_EDITABLE_PATHS, default_lint_ranges
from nightshift.lab.proposer import ProposerUnavailable

ACTOR = "outer"
MAX_READ_BYTES = 16 * 1024 * 1024


class OuterRefused(LabError):
    """The controller will not start (a precondition is missing). Nothing was spent."""


@dataclass
class _Cand:
    """A candidate proposed in the current round."""

    label: str
    record: CandidateRecord
    pack: Optional[PackRef]
    runnable: bool
    screen_ids: list = field(default_factory=list)
    confirm_ids: list = field(default_factory=list)
    decided: Optional[Decision] = None


class OuterController:
    def __init__(
        self,
        layout: LabLayout,
        ledger: LedgerWriter,
        state_reader: StateReader,
        packs: PackStore,
        proposer: ProposalSource,
        executor: TrialExecutor,
        decisions: DecisionRule,
        *,
        clock: Clock,
        lab_cap: Any,
        hosts: Sequence[str] = (),
        editable_paths: Sequence[str] = DEFAULT_EDITABLE_PATHS,
        fence: Callable[..., str] = default_fence,
        limits: ContextLimits = ContextLimits(),
        proposal_usd_cap: float = 1.0,
        trial_llm_usd_cap: Optional[float] = None,
        scoring_usd_per_trial: float = 0.0,
        deadline_epoch: Optional[float] = None,
        shape: Optional[Shape] = None,
        base: Optional[BaseModelRef] = None,
        lint_ranges: Optional[Mapping[str, Any]] = None,
        after_decision: Optional[Callable[[Decision], None]] = None,
    ):
        self.layout, self.ledger, self.reader, self.packs = layout, ledger, state_reader, packs
        self.proposer, self.executor, self.rules = proposer, executor, decisions
        self.clock, self.lab_cap, self.hosts = clock, lab_cap, tuple(hosts)
        self.editable_paths = tuple(editable_paths)
        self.fence, self.limits = fence, limits
        self.proposal_usd_cap = float(proposal_usd_cap)
        self.trial_llm_usd_cap = trial_llm_usd_cap
        self.scoring_usd_per_trial = float(scoring_usd_per_trial)
        self.deadline_epoch = deadline_epoch
        self._shape, self._base = shape, base
        self._contract = contract_text(lint_ranges if lint_ranges is not None else default_lint_ranges())
        self._after_decision = after_decision
        self._issued_seeds: dict[tuple, int] = {}
        self._confirm_inc_ids: list[str] = []

    # ------------------------------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------------------------------
    def run(self, max_rounds: int) -> list[Decision]:
        out: list[Decision] = []
        for _ in range(max_rounds):
            ds = self.run_round()
            if not ds:
                break
            out.extend(ds)
        return out

    def run_round(self) -> list[Decision]:
        """One screening round plus at most one confirmation. Returns the decisions appended (empty: nothing to do, e.g. the candidate
        limit is reached). Raises ``OuterRefused`` / ``GateNotRecorded`` / ``BudgetExhausted`` when it must not start."""
        state = self.reader.state()
        rule, manifest, budget = self._preconditions(state)
        self._reconcile_main(state)
        out: list[Decision] = self._close_orphans(state, rule)
        state = self.reader.state()
        slots = rule.max_candidates - len(state.candidates)
        if slots <= 0 or self._stopped_by_rejections(state, rule) or self._past_deadline() or self._confirmations_exhausted(state, rule):
            return out
        batch_size = min(rule.screen_batch_size, slots)
        self._check_lab_room(rule, budget, batch_size)
        round_index = self._next_round_index(state)
        incumbent_label = self.reader.incumbent_version(state)
        incumbent_pack = self._pack_of(state, incumbent_label)
        template = self._template(state, incumbent_label)

        batch = self._propose_batch(rule, manifest, budget, round_index, incumbent_label, incumbent_pack, batch_size)
        if not batch:
            return out
        for c in batch:
            if not c.runnable:
                out.append(self._decide_invalid(c, rule, round_index, incumbent_label, f"failed lint or apply: {'; '.join(c.record.lint_messages)}"[:3900]))
        runnable = [c for c in batch if c.runnable]
        stopped = None
        try:
            self._run_screen_trials(runnable, rule, manifest, budget, template, incumbent_label)
        except BudgetExhausted as exc:
            stopped = str(exc)
        if stopped is not None:
            for c in runnable:
                out.append(self._decide_invalid(c, rule, round_index, incumbent_label, f"the round stopped early: {stopped}"[:3900]))
            return out
        if not runnable:
            return out

        state = self.reader.state()
        inc_scores = self.reader.selection_trial_scores(state, rule, incumbent_label)
        cand_scores = {c.label: self.reader.selection_trial_scores(state, rule, c.label) for c in runnable}
        deltas = self.rules.screen(rule, inc_scores, cand_scores)
        best = self._screen_winner(runnable, deltas)
        for c in runnable:
            if c is best:
                continue
            d = deltas.get(c.label)
            if d is None:
                out.append(self._decide_invalid(c, rule, round_index, incumbent_label, "its trials could not be compared with the incumbent's", scores=(inc_scores, cand_scores[c.label])))
            else:
                out.append(
                    self._append_decision(
                        c, rule, round_index, DecisionStage.SCREEN, Outcome.REJECTED, incumbent_label,
                        "did not have the largest positive screening delta of its round" if d > 0 else "no gain over the incumbent in the screening trials",
                        inc_scores, cand_scores[c.label], d, None,
                    )
                )
        if best is None:
            return out
        try:
            self._run_confirm_trials(best, rule, manifest, budget, template, incumbent_label, round_index)
        except BudgetExhausted as exc:
            out.append(self._decide_invalid(best, rule, round_index, incumbent_label, f"the confirmation stopped early: {exc}"[:3900]))
            return out
        out.append(self._confirm(best, rule, round_index, incumbent_label, deltas[best.label]))
        return out

    # ------------------------------------------------------------------------------------------
    # preconditions and bookkeeping
    # ------------------------------------------------------------------------------------------
    def _preconditions(self, state: LabState) -> tuple:
        if state.search_closed is not None:
            raise OuterRefused("the search is closed (lab.search_closed exists)")
        if state.rule is None:
            raise OuterRefused("the keep-or-reject rule is not frozen")
        if not self.reader.has_gate(state, Gate.TERMS_10_1_IX):
            raise GateNotRecorded("the terms gate (terms_10_1_ix) has no recorded outcome; run `record-gate` first")
        rule, manifest, budget = self.reader.load_frozen_all(state)
        if state.spread is None:
            raise OuterRefused("the v0 spread has not been measured (lab.spread_measured is missing)")
        pending = sorted(t for t, s in state.trials_started.items() if s.phase is Phase.SEARCH and t not in state.trials)
        if pending:
            raise OuterRefused(f"search trial(s) {pending} started and never finished; close them with `abort-trial` first")
        return rule, manifest, budget

    def _reconcile_main(self, state: LabState) -> None:
        """Repair a crash between the ACCEPTED decision (appended first) and ``PackStore.accept``: the ledger's incumbent is the authority,
        so move ``main`` to it again (``accept`` is idempotent). Runs before any early return, so a restart at the candidate limit repairs too."""
        label = self.reader.incumbent_version(state)
        if label != "v0":
            self.packs.accept(self._pack_of(state, label), label)

    def _close_orphans(self, state: LabState, rule: Rule) -> list[Decision]:
        out: list[Decision] = []
        orphans = [cid for cid in state.candidates if cid not in state.decisions]
        if not orphans:
            return out
        next_round = self._next_round_index(state)
        for cid in orphans:
            rec = state.candidates[cid]
            c = _Cand(cid, rec, rec.pack, False)
            round_index = self._orphan_round(state, cid, next_round)
            out.append(self._decide_invalid(c, rule, round_index, rec.parent_version, "interrupted: the round that proposed it did not finish and is not resumed"))
        return out

    @staticmethod
    def _orphan_round(state: LabState, cid: str, next_round: int) -> int:
        """The round an undecided candidate was proposed in. A round proposes its whole batch before it decides any of it, so the first decision
        appended after the candidate's proposal belongs to the candidate's own round (a crash after some of the batch was decided). With no
        such decision, every candidate of its round is undecided and the round is the next one. Giving a half-decided round's leftover a NEW
        round index would count that one round as two batches toward ``stop_after_rejections``."""
        seq = state.event_seq.get((LabKind.CANDIDATE_PROPOSED, cid))
        if seq is None:
            return next_round
        later = [(state.event_seq[(LabKind.CANDIDATE_DECIDED, d)], dec.round_index) for d, dec in state.decisions.items() if state.event_seq.get((LabKind.CANDIDATE_DECIDED, d), -1) > seq]
        return min(later)[1] if later else next_round

    def _next_round_index(self, state: LabState) -> int:
        return max((d.round_index for d in state.decisions.values()), default=-1) + 1

    def _stopped_by_rejections(self, state: LabState, rule: Rule) -> bool:
        if rule.stop_after_rejections is None:
            return False
        by_round: dict[int, bool] = {}
        for d in state.decisions.values():
            by_round[d.round_index] = by_round.get(d.round_index, False) or d.accepted
        streak = 0
        for idx in sorted(by_round, reverse=True):
            if by_round[idx]:
                break
            streak += 1
        return streak >= rule.stop_after_rejections

    @staticmethod
    def _confirmations_exhausted(state: LabState, rule: Rule) -> bool:
        """The false-accept bound the report prints assumes at most R = ceil(max_candidates / screen_batch_size) confirmation tests. A round
        whose proposals were cut short (the provider failed after some of them) still gets its own confirmation, so without this cap many
        small rounds would run more tests than the bound counts. INVALID confirmations ran no test and are not counted."""
        rounds = -(-rule.max_candidates // rule.screen_batch_size)
        tests = sum(1 for d in state.decisions.values() if d.stage is DecisionStage.CONFIRM and d.outcome is not Outcome.INVALID)
        return tests >= rounds

    def _past_deadline(self) -> bool:
        return self.deadline_epoch is not None and self.clock() >= self.deadline_epoch

    def _lab_remaining(self) -> Optional[float]:
        cap = self.lab_cap
        if cap is None:
            return None
        rem = getattr(cap, "remaining", None)
        if callable(rem):
            return float(rem())
        if rem is not None:
            return float(rem)
        return float(cap.cap_usd) - float(cap.spent_usd)

    def _trial_usd(self, budget: BudgetSpec) -> float:
        return (self.trial_llm_usd_cap if self.trial_llm_usd_cap is not None else budget.dollars) + self.scoring_usd_per_trial

    def _check_lab_room(self, rule: Rule, budget: BudgetSpec, batch_size: int) -> None:
        rem = self._lab_remaining()
        if rem is None:
            return
        a = len(rule.assignments)
        trials = batch_size * rule.screen_trials_per_candidate * a + (rule.confirm_trials_per_candidate + rule.confirm_trials_per_incumbent) * a
        need = batch_size * self.proposal_usd_cap + trials * self._trial_usd(budget)
        if need > rem:
            raise BudgetExhausted("lab_dollars", max(rem, 0.0), need)

    def _pack_of(self, state: LabState, label: str) -> PackRef:
        pc = state.packs.get(label)
        if pc is None:
            raise OuterRefused(f"version {label!r} has no lab.pack_committed event")
        return pc.pack

    def _template(self, state: LabState, incumbent_label: str) -> tuple:
        if self._shape is not None and self._base is not None:
            return self._shape, self._base
        specs = [s for s in state.trials_started.values() if s.phase is Phase.SEARCH and s.pack_version == incumbent_label]
        specs = specs or [s for s in state.trials_started.values() if s.phase is Phase.SEARCH]
        if not specs:
            raise OuterRefused("no search trial exists to take the GPU shape and base model from; pass shape and base explicitly")
        return self._shape or specs[0].shape, self._base or specs[0].base

    # ------------------------------------------------------------------------------------------
    # proposing
    # ------------------------------------------------------------------------------------------
    def _propose_batch(self, rule, manifest, budget, round_index, incumbent_label, incumbent_pack, n) -> list[_Cand]:
        batch: list[_Cand] = []
        for _ in range(n):
            state = self.reader.state()
            cid = self.reader.next_id(state, "cand")
            facts = self._facts(state, rule, budget, incumbent_label, incumbent_pack, cid)
            ctx = build_context(facts, fence=self.fence, limits=self.limits)
            forbidden = forbidden_values(state, self.layout, self.hosts)
            if state.rule is not None and not forbidden:
                raise OuterRefused("the denylist is empty although a rule exists; refusing to send a prompt")
            prompt_text = ctx.system_prompt + "\n\n" + ctx.user_prompt
            assert_no_forbidden(prompt_text, forbidden)
            try:
                result = self.proposer.propose(ctx)
            except (BudgetExhausted, ProposerUnavailable):
                if batch:  # run what was proposed; the next round meets the same wall again and raises
                    break
                raise
            except ProposalInvalid as exc:
                batch.append(self._record_unparseable(cid, ctx, prompt_text, exc, incumbent_pack, incumbent_label))
                continue
            self._add_spent(result.usd)
            batch.append(self._record_candidate(cid, ctx, prompt_text, result, incumbent_pack, incumbent_label))
        return batch

    def _add_spent(self, usd: float) -> None:
        add = getattr(self.lab_cap, "add_spent", None)
        if callable(add) and usd > 0:
            add(float(usd))

    def _write_text(self, path: Path, text: str) -> Any:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.scrub_text(text), encoding="utf-8")
        return self.layout.file_ref(path)

    def _candidate_dir(self, cid: str) -> Path:
        d = self.layout.candidate_dir(cid)
        try:
            d.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise OuterRefused(f"candidate directory {d.name} already exists without a ledger record; move it aside and retry") from None
        return d

    def _record_candidate(self, cid, ctx, prompt_text, result, incumbent_pack, incumbent_label) -> _Cand:
        proposal = result.proposal
        self._candidate_dir(cid)
        prompt_ref = self._write_text(self.layout.candidate_prompt_path(cid), prompt_text)
        response_ref = self._write_text(self.layout.candidate_response_path(cid), result.raw_response)
        write_record(self.layout.candidate_proposal_path(cid), proposal)
        proposal_ref = self.layout.file_ref(self.layout.candidate_proposal_path(cid))
        pack: Optional[PackRef] = None
        diff_ref = None
        lint_passed, messages = False, ()
        try:
            pack, diff = self.packs.apply(proposal, incumbent_pack, cid)
            diff_ref = self._write_text(self.layout.candidate_diff_path(cid), diff)
            lint = self.packs.lint(pack)
            lint_passed, messages = lint.passed, tuple(lint.messages)
            write_record(self.layout.candidate_lint_path(cid), {"passed": lint.passed, "messages": list(lint.messages), "changed_paths": list(lint.changed_paths)})
        except ProposalInvalid as exc:
            messages = (str(exc)[:500],)
            pack, diff_ref = None, None
        rec = CandidateRecord(
            candidate_id=cid, version_label=cid, parent=incumbent_pack, parent_version=incumbent_label,
            pack=pack, proposal=proposal_ref, diff=diff_ref, prompt=prompt_ref, response=response_ref,
            lever=proposal.lever, hypothesis=proposal.hypothesis, reason=_reason(proposal),
            lint_passed=lint_passed, lint_messages=messages,
            proposer_model=result.model, proposer_served_model=result.served_model, fallback_used=result.fallback_used,
            propose_usd=result.usd, created_utc=utc_iso(self.clock()),
        )
        self.ledger.append(LabKind.CANDIDATE_PROPOSED, ACTOR, rec)
        if lint_passed:
            self.ledger.append(LabKind.PACK_COMMITTED, ACTOR, PackCommitted(cid, pack, incumbent_pack, cid))
        return _Cand(cid, rec, pack, lint_passed)

    def _record_unparseable(self, cid, ctx, prompt_text, exc, incumbent_pack, incumbent_label) -> _Cand:
        """A reply that is no Proposal: INVALID, no GPU spent. A placeholder proposal.json keeps the file contract (see handoff)."""
        reason = str(getattr(exc, "reason", None) or exc)[:900]
        usd = float(getattr(exc, "usd", 0.0) or 0.0)
        self._add_spent(usd)
        self._candidate_dir(cid)
        prompt_ref = self._write_text(self.layout.candidate_prompt_path(cid), prompt_text)
        response_ref = self._write_text(self.layout.candidate_response_path(cid), getattr(exc, "raw_response", "") or "(no reply)")
        placeholder = Proposal(
            hypothesis=f"INVALID: no proposal could be read from the reply ({reason})",
            lever=Lever.OTHER,
            change_summary="none (placeholder for an unreadable reply)",
            expected_effect="none",
            risk="none",
            edits=(PackEdit("notes/INVALID_PLACEHOLDER.md", EditOp.REPLACE_FILE, "not a proposal; never apply this record\n", ()),),
        )
        write_record(self.layout.candidate_proposal_path(cid), placeholder)
        rec = CandidateRecord(
            candidate_id=cid, version_label=cid, parent=incumbent_pack, parent_version=incumbent_label, pack=None,
            proposal=self.layout.file_ref(self.layout.candidate_proposal_path(cid)), diff=None, prompt=prompt_ref, response=response_ref,
            lever=Lever.OTHER, hypothesis=placeholder.hypothesis[:3900], reason=placeholder.hypothesis[:3900], lint_passed=False,
            lint_messages=(reason[:500],),
            proposer_model=(getattr(exc, "model", "") or "unknown"), proposer_served_model=(getattr(exc, "served_model", "") or "unknown"),
            fallback_used=bool(getattr(exc, "fallback_used", False)), propose_usd=usd, created_utc=utc_iso(self.clock()),
        )
        self.ledger.append(LabKind.CANDIDATE_PROPOSED, ACTOR, rec)
        return _Cand(cid, rec, None, False)

    # ------------------------------------------------------------------------------------------
    # facts for the proposer
    # ------------------------------------------------------------------------------------------
    def _facts(self, state, rule, budget, incumbent_label, incumbent_pack, cid) -> ProposalFacts:
        history = []
        for k, rec in state.candidates.items():
            dec = state.decisions.get(k)
            history.append(
                CandidateFact(
                    k, rec.lever.value, rec.hypothesis, self._change_summary(rec), self._read_text(rec.diff.path if rec.diff else None, 200_000),
                    dec.outcome if dec else None, tuple(rec.lint_messages) if not rec.lint_passed else (),
                )
            )
        trials = [r for r in state.trials.values() if r.spec.phase is Phase.SEARCH and r.spec.pack_version == incumbent_label]
        evidence = tuple(self._evidence(state, r) for r in trials[-self.limits.max_evidence:])
        return ProposalFacts(
            candidate_id=cid, incumbent=incumbent_pack, incumbent_version=incumbent_label, budget=budget,
            editable_paths=self.editable_paths, pack_files=self._pack_files(incumbent_pack), history=tuple(history),
            evidence=evidence, version_dev=self._version_dev(state), contract=self._contract,
        )

    def _pack_files(self, pack: PackRef) -> dict:
        files: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="outer-pack-") as td:
            dest = Path(td) / "pack"
            self.packs.checkout(pack, dest)
            for p in sorted(dest.rglob("*")):
                if p.is_file() and not p.is_symlink():
                    try:
                        files[p.relative_to(dest).as_posix()] = p.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue
        return files

    def _read_text(self, rel: Optional[str], cap: int) -> str:
        if rel is None:
            return ""
        try:
            return read_regular_bytes(self.layout.root, rel, max_bytes=MAX_READ_BYTES).decode("utf-8", "replace")[:cap]
        except (ContractError, OSError):
            return ""

    def _change_summary(self, rec: CandidateRecord) -> str:
        try:
            raw = read_regular_bytes(self.layout.root, rec.proposal.path, max_bytes=MAX_READ_BYTES).decode("utf-8")
            return Proposal.from_dict(loads_strict(raw)).change_summary
        except (ContractError, OSError, UnicodeDecodeError):
            return rec.reason[:400]

    def _dev_of(self, state: LabState, score_id: Optional[str]) -> Optional[float]:
        rec = state.scores.get(score_id) if score_id else None
        return rec.headline_mean() if rec is not None and rec.split is Split.DEV else None

    def _version_dev(self, state: LabState) -> tuple:
        by: dict[str, list] = {}
        for r in state.trials.values():
            if r.spec.phase is Phase.SEARCH:
                by.setdefault(r.spec.pack_version, [])
                v = self._dev_of(state, r.dev_score_id)
                if v is not None:
                    by[r.spec.pack_version].append(v)
        return tuple(VersionDev(k, len(v), (sum(v) / len(v)) if v else None) for k, v in by.items())

    def _evidence(self, state: LabState, rec: TrialRecord) -> TrialEvidence:
        tid = rec.spec.trial_id
        results = self._job_results(tid)
        traj = tuple(x for x in (self._dev_of(state, r.dev_score_id) for r in results) if x is not None)
        notes = self._read_text(rec.notes.path if rec.notes else None, 400_000)
        trace = self._read_text(rec.trace.path if rec.trace else None, 8_000_000)
        summary = summarize_trace(trace) if trace else None
        wall = rec.budget_used.wall_seconds
        gpu_job_minutes = sum(r.gpu.minutes for r in results)
        busy = min(1.0, gpu_job_minutes * 60.0 / wall) if wall > 0 else None
        texts = [rec.failure or "", notes[:50_000], *(r.failure or "" for r in results)]
        return TrialEvidence(
            trial_id=tid, status=rec.status.value, dev_trajectory=traj, n_jobs=len(results),
            n_jobs_ok=sum(1 for r in results if r.status is JobStatus.OK),
            gpu_minutes_used=rec.budget_used.gpu_minutes, gpu_minutes_budget=rec.spec.budget.gpu_minutes,
            tokens_used=rec.budget_used.researcher_tokens, tokens_budget=rec.spec.budget.researcher_tokens,
            gpu_busy_fraction=busy, failure_signatures=failure_signatures(texts),
            tool_counts=summary.tool_counts if summary else (), failure_text=rec.failure or "",
            notes_excerpt=notes[-self.limits.notes_chars:] if notes else "", trace_excerpt=summary.excerpt if summary else "",
        )

    def _job_results(self, trial_id: str) -> list[TrainResult]:
        try:
            raw = read_regular_bytes(self.layout.root, (self.layout.trial_ledger_dir(trial_id) / "ledger.jsonl").relative_to(self.layout.root).as_posix(), max_bytes=MAX_READ_BYTES)
        except (ContractError, OSError):
            return []
        out: list[TrainResult] = []
        for line in raw.decode("utf-8", "replace").splitlines():
            if LabKind.GPU_JOB_FINISHED not in line:
                continue
            try:
                ev = loads_strict(line)
                if ev.get("kind") == LabKind.GPU_JOB_FINISHED:
                    out.append(TrainResult.from_dict(ev["data"]))
            except (ContractError, KeyError, AttributeError):
                continue
        return out

    # ------------------------------------------------------------------------------------------
    # trials
    # ------------------------------------------------------------------------------------------
    def _next_seed(self, state: LabState, pack: PackRef, assignment: str) -> int:
        key = (pack.commit, assignment)
        used = {s.seed for s in state.trials_started.values() if s.phase is Phase.SEARCH and s.pack.commit == pack.commit and s.assignment_id == assignment}
        if key in self._issued_seeds:
            used.add(self._issued_seeds[key])
        seed = (max(used) + 1) if used else 0
        self._issued_seeds[key] = seed
        return seed

    def _run_trial(self, label, pack, assignment, rule, manifest, budget, template) -> TrialRecord:
        rem = self._lab_remaining()
        if rem is not None and rem < self._trial_usd(budget):
            raise BudgetExhausted("lab_dollars", max(rem, 0.0), self._trial_usd(budget))
        state = self.reader.state()
        shape, base = template
        spec = TrialSpec(
            trial_id=self.reader.next_id(state, "t"), phase=Phase.SEARCH, arm=None, pack_version=label, pack=pack,
            assignment_id=assignment, seed=self._next_seed(state, pack, assignment), budget=budget, shape=shape, base=base,
            data_revisions=dict(manifest.get(assignment).data_revisions), scorer_version=rule.scorer_version,
            score_params=rule.score_params, score_seeds=rule.score_seeds, final_request_id=None,
        )
        check_trial_against_frozen(spec, manifest=manifest, budget=budget, rule=rule)
        rec = self.executor.run(spec)
        if rec.spec.trial_id != spec.trial_id:
            raise LabError(f"the executor returned trial {rec.spec.trial_id!r} for {spec.trial_id!r}")
        return rec

    def _run_screen_trials(self, runnable, rule, manifest, budget, template, incumbent_label) -> None:
        # candidates one after the other; the incumbent's existing trials are the comparison (no fresh incumbent trials at screening)
        for c in runnable:
            for a in rule.assignments:
                for _ in range(rule.screen_trials_per_candidate):
                    c.screen_ids.append(self._run_trial(c.label, c.pack, a, rule, manifest, budget, template).spec.trial_id)

    def _run_confirm_trials(self, best, rule, manifest, budget, template, incumbent_label, round_index) -> None:
        inc_pack = best.record.parent
        self._confirm_inc_ids = []
        for a in rule.assignments:
            n = max(rule.confirm_trials_per_candidate, rule.confirm_trials_per_incumbent)
            for i in range(n):
                order = ("cand", "inc") if (i + round_index) % 2 == 0 else ("inc", "cand")
                for who in order:
                    if who == "cand" and i < rule.confirm_trials_per_candidate:
                        best.confirm_ids.append(self._run_trial(best.label, best.pack, a, rule, manifest, budget, template).spec.trial_id)
                    elif who == "inc" and i < rule.confirm_trials_per_incumbent:
                        self._confirm_inc_ids.append(self._run_trial(incumbent_label, inc_pack, a, rule, manifest, budget, template).spec.trial_id)

    # ------------------------------------------------------------------------------------------
    # deciding
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _screen_winner(runnable: Sequence[_Cand], deltas: Mapping[str, Optional[float]]) -> Optional[_Cand]:
        best: Optional[_Cand] = None
        for c in sorted(runnable, key=lambda c: c.label):
            d = deltas.get(c.label)
            if d is not None and d > 0 and (best is None or d > deltas[best.label]):
                best = c
        return best

    def _confirm(self, best: _Cand, rule: Rule, round_index: int, incumbent_label: str, screen_delta: float) -> Decision:
        state = self.reader.state()
        inc_ids, cand_ids = set(self._confirm_inc_ids), set(best.confirm_ids)
        inc = [s for s in self.reader.selection_trial_scores(state, rule, incumbent_label) if s.trial_id in inc_ids]
        cand = [s for s in self.reader.selection_trial_scores(state, rule, best.label) if s.trial_id in cand_ids]
        nums: DecisionNumbers = self.rules.decide(rule, inc, cand)
        d = self._append_decision(
            best, rule, round_index, DecisionStage.CONFIRM, nums.outcome, incumbent_label, nums.reason, inc, cand, screen_delta,
            None if nums.outcome is Outcome.INVALID else nums, extra_trials=self._confirm_inc_ids,
        )
        if d.accepted:
            self.packs.accept(best.pack, best.label)
        return d

    def _decide_invalid(self, c: _Cand, rule, round_index, incumbent_label, reason, scores: tuple = ((), ())) -> Decision:
        return self._append_decision(c, rule, round_index, DecisionStage.SCREEN, Outcome.INVALID, incumbent_label, reason, scores[0], scores[1], None, None)

    def _append_decision(self, c: _Cand, rule: Rule, round_index: int, stage: DecisionStage, outcome: Outcome, incumbent_label: str,
                         reason: str, inc_scores: Sequence[TrialScore], cand_scores: Sequence[TrialScore],
                         screen_delta: Optional[float], nums: Optional[DecisionNumbers], extra_trials: Sequence[str] = ()) -> Decision:
        state = self.reader.state()
        trial_ids = [*c.screen_ids, *c.confirm_ids, *extra_trials]
        cost = self.reader.cost_breakdown(state, trial_ids, c.label)
        d = Decision(
            candidate_id=c.label, rule_sha256=rule.sha256(), stage=stage, round_index=round_index, outcome=outcome, reason=reason[:3900],
            incumbent_version=incumbent_label, candidate_version=c.label, incumbent_scores=tuple(inc_scores), candidate_scores=tuple(cand_scores),
            screen_delta=screen_delta, delta=nums.delta if nums else None, margin=nums.margin if nums else None,
            df=nums.df if nums else None, sigma_used=nums.sigma_used if nums else None, cost=cost, decided_utc=utc_iso(self.clock()),
        )
        self.ledger.append(LabKind.CANDIDATE_DECIDED, ACTOR, d)
        c.decided = d
        if self._after_decision is not None:
            self._after_decision(d)
        return d


def _reason(p: Proposal) -> str:
    text = f"{p.change_summary}\n\nExpected effect: {p.expected_effect}"
    return text.encode("utf-8")[:3900].decode("utf-8", "ignore")
