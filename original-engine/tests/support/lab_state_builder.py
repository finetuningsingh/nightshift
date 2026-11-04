"""Builds a ``LabState`` by hand, with real files behind it, for the statistics, report and (via owner 6) integration tests.

``LabStateBuilder`` makes the same records the ledger would hold (``TrialSpec``, ``TrialRecord``, ``ScoreRecord``, ``Decision``, ...)
and writes the files they reference under a temporary lab root: the frozen records (``frozen/*.json``, hashed into ``FrozenFile``), the
per-prompt files of every score (``scores/<id>/per_prompt.jsonl``), and, optionally, a minimal trial ledger per trial. Decisions are
computed with the real ``nightshift.lab.stats`` so they agree with a recomputation. Only ``contracts`` and ``stats`` are imported.

Quick use::

    b = LabStateBuilder.standard(tmpdir, reveal=True)   # a whole experiment: spread, rule, candidates, search closed, final, reveal
    state, layout = b.build(), b.layout

or step by step (every method returns the builder, or the thing it made)::

    b = LabStateBuilder(tmpdir).freeze_manifest().freeze_budget()
    b.measure_spread([0.60, 0.62, 0.61, 0.63, 0.64]); rule = b.freeze_rule()
    cand = b.propose("cand-001", hypothesis="shorter notes")
    b.judge_screen_loss("cand-001", [0.60, 0.61])

Scores are multiples of ``1 / n_prompts`` (default 200 for dev and selection, 300 for the IFBench test split); a requested score is rounded
to the nearest such multiple, and the three generation seeds of a record sit at -1, 0 and +1 prompt around it so that the mean over
seeds is the requested value.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Optional, Sequence

from nightshift.lab import stats
from nightshift.lab.contracts import (
    TEST_ASSIGNMENT_ID,
    AdapterFile,
    AdapterRef,
    Arm,
    Assignment,
    AssignmentManifest,
    AssignmentRole,
    AssignmentWeight,
    BaseModelRef,
    BudgetCharge,
    BudgetSpec,
    BudgetUsage,
    CandidateRecord,
    ComparisonMethod,
    ContaminationReport,
    CostBreakdown,
    Decision,
    DecisionStage,
    FailedTrialPolicy,
    FileRef,
    FinalAnalysis,
    FinalRequest,
    FinalReveal,
    FrozenFile,
    GateRecord,
    Gate,
    GateOutcome,
    GenerationParams,
    GpuUsage,
    ItemMode,
    JobStatus,
    LabInit,
    LabLayout,
    LabState,
    LedgerAnchor,
    Lever,
    MixedRowPolicy,
    OperatorChange,
    Outcome,
    PackCommitted,
    PackRef,
    Phase,
    PlannedTrial,
    Resource,
    Rule,
    ScoreCost,
    ScoreRecord,
    SearchClosed,
    SeedAggregate,
    Shape,
    SpreadMeasured,
    SpreadPlan,
    Split,
    Submission,
    TrainResult,
    TrialRecord,
    TrialScore,
    TrialSpec,
    TrialStatus,
    Verifier,
    comparability_key,
    record_sha256,
    sha256_text,
    utc_iso,
    write_record,
)

BASE = BaseModelRef("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68")
SCORER_VERSION = "sv1-0123456789abcdef"
SEEDS = (0, 1, 2)
PACK_REPO = "packs/pack.git"
T0 = 1_793_000_000.0  # 2026-10-26, a fixed epoch so every record time is deterministic


def _commit(label: str) -> str:
    return sha256_text("pack:" + label)[:40]


class LabStateBuilder:
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        n_evolution: int = 1,
        n_heldout: int = 1,
        item_mode: ItemMode = ItemMode.SALTED_DRAW,
        n_prompts: int = 200,
        test_prompts: int = 300,
    ) -> None:
        self.layout = LabLayout(Path(root))
        self.state = LabState()
        self.n_evolution, self.n_heldout = n_evolution, n_heldout
        self.item_mode = item_mode
        self.n_prompts, self.test_prompts = n_prompts, test_prompts
        self.params = GenerationParams.recommended(8000)
        self.shape = Shape("a100", 1, 12, 300)
        self.budget = BudgetSpec(researcher_tokens=2_000_000, gpu_minutes=120.0, dollars=25.0, wall_seconds=7200.0)
        self.manifest: Optional[AssignmentManifest] = None
        self.rule: Optional[Rule] = None
        self.analysis: Optional[FinalAnalysis] = None
        self.packs: dict = {}
        self.incumbent_version = "v0"
        self.incumbent_trials: list = []  # TrialScore of the incumbent's trials on the rule assignment
        self.trial_scores: dict = {}  # trial_id -> TrialScore (selection split, search or final)
        self.final_request_ids: dict = {}  # arm -> request id
        self._n_trial = self._n_score = self._n_adapter = self._n_seed = 0
        self._tick = 0
        self._rank = list(range(max(n_prompts, test_prompts)))
        random.Random(1).shuffle(self._rank)
        self.state.initialized = LabInit("lab-test", self.utc(), "ax43")

    # ------------------------------------------------------------------------------------------
    def utc(self) -> str:
        self._tick += 1
        return utc_iso(T0 + 60 * self._tick)

    def _frozen(self, name: str, path: Path, record) -> FrozenFile:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_record(path, record)
        return FrozenFile(name, self.layout.file_ref(path))

    def _new_ids(self):
        self._n_trial += 1
        return f"t-{self._n_trial:04d}"

    # ------------------------------------------------------------------------------------------
    # frozen inputs
    # ------------------------------------------------------------------------------------------
    def freeze_manifest(self) -> "LabStateBuilder":
        def revs(aid: str) -> dict:
            return {"train_sha256": sha256_text(aid + ":train"), "dev_sha256": sha256_text(aid + ":dev"), "selection_sha256": sha256_text(aid + ":selection")}

        assignments = tuple(Assignment(f"a-evo-{i + 1}", AssignmentRole.EVOLUTION, ("ifeval25",), revs(f"a-evo-{i + 1}")) for i in range(self.n_evolution)) + tuple(
            Assignment(f"a-held-{i + 1}", AssignmentRole.HELDOUT, ("copy",), revs(f"a-held-{i + 1}")) for i in range(self.n_heldout)
        )
        self.manifest = AssignmentManifest(
            assignments=assignments,
            evolution_families=("ifeval25", "counting"),
            heldout_families=("copy", "firstlast"),
            mixed_row_policy=MixedRowPolicy.DROPPED,
            item_mode=self.item_mode,
            source_revisions={
                "ifrlvr_parquet_sha256": "5" * 64,
                "ifbench_commit": "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d",
                "open_instruct_commit": "b781e782a8fc530577f235aefe282dd046cb8ad2",
                "split_salt_sha256": "6" * 64,
            },
            created_utc=self.utc(),
        )
        self.state.manifest = self._frozen("assignments", self.layout.manifest_path, self.manifest)
        return self

    def freeze_budget(self) -> "LabStateBuilder":
        self.state.budget = self._frozen("budget", self.layout.budget_path, self.budget)
        return self

    def record_gate(self) -> "LabStateBuilder":
        self.state.gates.append(GateRecord(Gate.TERMS_10_1_IX, GateOutcome.PROCEED_WITH_RISK, "Proceeding with the risk stated.", "operator", self.utc()))
        return self

    def add_pack(self, version_label: str, parent_label: Optional[str] = None, candidate_id: Optional[str] = None) -> PackRef:
        ref = PackRef(PACK_REPO, _commit(version_label))
        self.packs[version_label] = ref
        parent = self.packs.get(parent_label) if parent_label else None
        self.state.packs[version_label] = PackCommitted(version_label, ref, parent, candidate_id)
        return ref

    # ------------------------------------------------------------------------------------------
    # scores and trials
    # ------------------------------------------------------------------------------------------
    def _counts(self, score: float, n: int) -> list:
        k = max(1, min(n - 1, round(score * n)))
        return [k - 1, k, k + 1]  # three seeds around the requested value

    def _write_score(self, *, split: Split, phase: Phase, assignment_id: str, trial_id: Optional[str], arm: Optional[Arm], final_request_id: Optional[str], adapter_sha: Optional[str], score: float, n: Optional[int] = None, verifier: Optional[Verifier] = None, truncated: int = 0, errors: int = 0) -> ScoreRecord:
        n = n or (self.test_prompts if split is Split.TEST else self.n_prompts)
        self._n_score += 1
        sid = f"sc-{self._n_score:04d}"
        ks = self._counts(score, n)
        rows, aggs = [], []
        for seed, k in zip(SEEDS, ks):
            for i in range(n):
                ok = self._rank[i] < k
                if split is Split.TEST:
                    rows.append({"prompt_key": f"p{i:03d}", "seed": seed, "strict_prompt": ok, "loose_prompt": ok, "strict_instructions": [ok], "loose_instructions": [ok], "truncated": False, "error": False})
                else:
                    rows.append({"prompt_key": f"p{i:03d}", "seed": seed, "passed": ok, "fraction": 1.0 if ok else 0.0, "constraints": [{"id": "copy:copy", "passed": ok}], "truncated": False, "error": False})
            p = k / n
            m = {"prompt_strict": p, "prompt_loose": min(1.0, p + 0.01), "instruction_strict": p, "instruction_loose": p} if split is Split.TEST else {"prompt_all": p, "prompt_fraction": min(1.0, p + 0.05)}
            aggs.append(SeedAggregate(seed, n, m, truncated, errors))
        path = self.layout.per_prompt_path(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        verifier = verifier or (Verifier.IFBENCH_UPSTREAM if split is Split.TEST else Verifier.IFEVALG)
        rec = ScoreRecord(
            score_id=sid,
            request_id=sid,
            phase=phase,
            assignment_id=assignment_id,
            split=split,
            trial_id=trial_id,
            arm=arm,
            final_request_id=final_request_id,
            base=BASE,
            adapter_sha256=adapter_sha,
            scorer_version=SCORER_VERSION,
            verifier=verifier,
            verifier_pins={"verifier": "pinned"},
            params=self.params,
            per_seed=tuple(aggs),
            n_prompts=n,
            per_prompt=self.layout.file_ref(path),
            headline_metric="prompt_strict" if split is Split.TEST else "prompt_all",
            created_utc=self.utc(),
        )
        self.state.scores[sid] = rec
        return rec

    def add_trial(
        self,
        *,
        pack_version: str = "v0",
        assignment: str = "a-evo-1",
        score: Optional[float] = None,
        test_score: Optional[float] = None,
        status: TrialStatus = TrialStatus.COMPLETED,
        phase: Phase = Phase.SEARCH,
        arm: Optional[Arm] = None,
        final_request_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        isolation: str = "per-trial unix user",
        gpu_minutes: float = 40.0,
        gpu_usd: float = 0.80,
        researcher_usd: float = 3.0,
        score_gpu_usd: float = 0.30,
        n_prompts: Optional[int] = None,
        test_n_prompts: Optional[int] = None,
        truncated: int = 0,
        errors: int = 0,
    ) -> str:
        """One finished trial with its adapter and selection score (and a test score when ``test_score`` is given, final phase only)."""
        tid = trial_id or self._new_ids()
        self._n_seed += 1
        pack = self.packs[pack_version]
        a = self.manifest.get(assignment)
        spec = TrialSpec(
            trial_id=tid,
            phase=phase,
            arm=arm,
            pack_version=pack_version,
            pack=pack,
            assignment_id=assignment,
            seed=1000 + self._n_seed,
            budget=self.budget,
            shape=self.shape,
            base=BASE,
            data_revisions=dict(a.data_revisions),
            scorer_version=SCORER_VERSION,
            score_params=self.params,
            score_seeds=SEEDS,
            final_request_id=final_request_id,
        )
        self.state.trials_started[tid] = spec
        used = BudgetUsage(1_200_000, gpu_minutes, researcher_usd, gpu_usd, 3000.0)
        started = self.utc()
        if score is None:
            self.state.trials[tid] = TrialRecord(spec, status, started, self.utc(), None, used, None, None, None, None, None, None, None, {"isolation": isolation}, "no submission")
            return tid
        self._n_adapter += 1
        aid = f"ad-{self._n_adapter:04d}"
        adapter = AdapterRef.from_files(aid, [AdapterFile("adapter_config.json", sha256_text(aid + "c"), 1000), AdapterFile("adapter_model.safetensors", sha256_text(aid + "m"), 7_000_000)])
        sel = self._write_score(split=Split.SELECTION, phase=phase, assignment_id=assignment, trial_id=tid, arm=arm, final_request_id=final_request_id, adapter_sha=adapter.sha256, score=score, n=n_prompts, truncated=truncated, errors=errors)
        self.state.score_costs[sel.score_id] = ScoreCost(sel.score_id, GpuUsage(20.0, score_gpu_usd, (f"rent-{sel.score_id}",)))
        if test_score is not None:
            self._write_score(split=Split.TEST, phase=phase, assignment_id=TEST_ASSIGNMENT_ID, trial_id=tid, arm=arm, final_request_id=final_request_id, adapter_sha=adapter.sha256, score=test_score, n=test_n_prompts)
        sub = Submission(aid, sha256_text(tid + "recipe"), _commit(tid), sel.headline_mean())
        self.state.trials[tid] = TrialRecord(
            spec, status, started, self.utc(), sha256_text(tid + "ctx"), used, sub, adapter, f"dev-{tid}", sel.score_id, None, None, LedgerAnchor(5, "c" * 16), {"isolation": isolation}, None
        )
        self.trial_scores[tid] = TrialScore(tid, assignment, sel.score_id, "prompt_all", SCORER_VERSION, comparability_key(spec, "prompt_all"), sel.headline_mean())
        return tid

    def write_trial_ledger(self, trial_id: str, *, charges: Sequence[BudgetCharge] = (), gpu_usd: float = 0.90, gpu_minutes: float = 41.0) -> None:
        """A minimal ``trials/<id>/ledger.jsonl`` (events with kind and data) holding the given charges and one finished GPU job."""
        rec = self.state.trials[trial_id]
        events = [(c_kind, c.to_dict()) for c_kind, c in (("lab.budget_charged", ch) for ch in charges)]
        if rec.adapter is not None:
            job = TrainResult("j-" + trial_id, trial_id, JobStatus.OK, rec.adapter.adapter_id, rec.adapter, rec.dev_score_id, GpuUsage(gpu_minutes, gpu_usd, ("rent-" + trial_id,)), {"train_s": 1200.0}, None, FileRef(f"trials/{trial_id}/train_data/j.jsonl", "d" * 64, 10), None)
            events.append(("lab.gpu_job_finished", job.to_dict()))
        d = self.layout.trial_dir(trial_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "ledger.jsonl").write_text("".join(json.dumps({"seq": i + 1, "ts": T0, "run_id": "r", "kind": k, "actor": "t", "data": data, "prev_hash": "0" * 64, "hash": "0" * 64}, sort_keys=True) + "\n" for i, (k, data) in enumerate(events)), encoding="utf-8")

    def add_contamination(self, trial_id: str, *, hits: int = 0) -> None:
        ref = FileRef(f"trials/{trial_id}/train_data/j.jsonl", "d" * 64, 10)
        self.state.contamination[trial_id] = ContaminationReport(trial_id, "j-" + trial_id, ref, 120, hits, 0, 13, ("e" * 64,), hits == 0, self.utc())

    def add_operator_change(self, target: str = "proposer_prompt", detail: str = "Shortened the lever list.") -> None:
        self.state.operator_changes.append(OperatorChange(target, detail, len(self.state.candidates), self.utc()))

    # ------------------------------------------------------------------------------------------
    # spread, rule
    # ------------------------------------------------------------------------------------------
    def measure_spread(self, scores: Sequence[float], assignment: str = "a-evo-1", **trial_kw) -> SpreadMeasured:
        if "v0" not in self.packs:
            self.add_pack("v0")
        ids = [self.add_trial(pack_version="v0", assignment=assignment, score=s, **trial_kw) for s in scores]
        ts = [self.trial_scores[i] for i in ids]
        self.state.spread_plan = SpreadPlan("prompt_all", SCORER_VERSION, (assignment,), len(ts), self.utc())
        est = stats.trial_spread(ts)
        self.state.spread = SpreadMeasured("prompt_all", SCORER_VERSION, len(ts), 0, tuple(ts), est)
        self.incumbent_trials = list(ts)
        return self.state.spread

    def freeze_rule(self, **kw) -> Rule:
        sp = self.state.spread
        est = sp.estimate
        base = dict(
            metric="prompt_all", split=Split.SELECTION, scorer_version=SCORER_VERSION, assignments=("a-evo-1",), spread=est, alpha=0.05, min_gain=0.0,
            score_seeds=SEEDS, score_params=self.params, screen_batch_size=3, screen_trials_per_candidate=2, confirm_trials_per_candidate=3,
            confirm_trials_per_incumbent=3, variance_guard_ratio=1.5, failed_trial_policy=FailedTrialPolicy.INVALIDATES_CANDIDATE, degenerate_override=None,
            max_candidates=5, stop_after_rejections=None, repeated_decisions_note="Screen then confirm; selection results are an adaptive search trace, not evidence of a gain.",
            frozen_utc=self.utc(),
        )
        base.update(kw)
        if "minimum_detectable_gain" not in base:
            nc, ni = base["confirm_trials_per_candidate"], base["confirm_trials_per_incumbent"]
            df = est.df + (nc - 1) + (ni - 1)
            base["minimum_detectable_gain"] = stats.minimum_detectable_gain(max(est.sigma, 1e-9), df, nc, ni, len(base["assignments"]), base["alpha"], min_gain=base["min_gain"])
        base.setdefault("assumed_gain", base["minimum_detectable_gain"] * 1.5)
        self.rule = Rule(**base)
        self.state.rule = self._frozen("rule", self.layout.rule_path, self.rule)
        return self.rule

    # ------------------------------------------------------------------------------------------
    # candidates and decisions
    # ------------------------------------------------------------------------------------------
    def propose(self, cid: str, *, lever: Lever = Lever.CONTEXT_NOTES, hypothesis: str = "Researchers re-read the whole log every turn.", lint_passed: bool = True, propose_usd: float = 0.30) -> CandidateRecord:
        parent = self.packs[self.incumbent_version]
        pack = self.add_pack(cid, self.incumbent_version, cid) if lint_passed else None
        ref = lambda name: FileRef(f"candidates/{cid}/{name}", "d" * 64, 5)  # noqa: E731
        rec = CandidateRecord(
            cid, cid, parent, self.incumbent_version, pack, ref("proposal.json"), ref("diff.patch") if lint_passed else None, ref("prompt.txt"), ref("response.txt"),
            lever, hypothesis, "Bound the notes summary.", lint_passed, () if lint_passed else ("task.yaml: key 'grader.timeout' changed",), "muse-wonder-1.0", "muse-wonder-1.0", False, propose_usd, self.utc(),
        )
        self.state.candidates[cid] = rec
        return rec

    def _trials_for(self, pack_version: str, scores: Sequence[float]) -> list:
        return [self.trial_scores[self.add_trial(pack_version=pack_version, score=s)] for s in scores]

    def _cost(self, cid: str, trial_ids: Sequence[str]) -> CostBreakdown:
        t = [self.state.trials[i] for i in trial_ids]
        scoring = math.fsum(self.state.score_costs[x.selection_score_id].gpu.usd for x in t if x.selection_score_id)
        return CostBreakdown(self.state.candidates[cid].propose_usd, math.fsum(x.budget_used.researcher_usd for x in t), math.fsum(x.budget_used.gpu_usd for x in t), scoring)

    def _decision(self, cid, stage, outcome, reason, inc, cand, screen_delta, nums, trial_ids, round_index) -> Decision:
        d = Decision(
            candidate_id=cid, rule_sha256=self.rule.sha256(), stage=stage, round_index=round_index, outcome=outcome, reason=reason, incumbent_version=self.incumbent_version,
            candidate_version=cid, incumbent_scores=tuple(inc), candidate_scores=tuple(cand), screen_delta=screen_delta,
            delta=None if nums is None else nums.delta, margin=None if nums is None else nums.margin, df=None if nums is None else nums.df,
            sigma_used=None if nums is None else nums.sigma_used, cost=self._cost(cid, trial_ids), decided_utc=self.utc(),
        )
        self.state.decisions[cid] = d
        return d

    def judge_invalid(self, cid: str, reason: str = "the pack failed lint") -> Decision:
        return self._decision(cid, DecisionStage.SCREEN, Outcome.INVALID, reason, (), (), None, None, (), 0)

    def judge_screen_loss(self, cid: str, scores: Sequence[float], round_index: int = 0) -> Decision:
        screen = self._trials_for(cid, scores)
        delta = stats.screen(self.rule, self.incumbent_trials, {cid: screen})[cid]
        return self._decision(cid, DecisionStage.SCREEN, Outcome.REJECTED, "lost its screening round", self.incumbent_trials, screen, delta, None, [t.trial_id for t in screen], round_index)

    def judge_confirm(self, cid: str, screen_scores: Sequence[float], cand_scores: Sequence[float], inc_scores: Sequence[float], round_index: int = 0) -> Decision:
        screen = self._trials_for(cid, screen_scores)
        screen_delta = stats.screen(self.rule, self.incumbent_trials, {cid: screen})[cid]
        fresh_inc = self._trials_for(self.incumbent_version, inc_scores)
        cand = self._trials_for(cid, cand_scores)
        nums = stats.decide(self.rule, fresh_inc, cand)
        ids = [t.trial_id for t in screen + fresh_inc + cand]
        d = self._decision(cid, DecisionStage.CONFIRM, nums.outcome, nums.reason, fresh_inc, cand, screen_delta if nums.outcome is not Outcome.INVALID else None, nums, ids, round_index)
        self.incumbent_trials += fresh_inc
        if nums.outcome is Outcome.ACCEPTED:
            self.incumbent_version, self.incumbent_trials = cid, list(cand)
        return d

    def close_search(self) -> SearchClosed:
        self.state.search_closed = SearchClosed(self.incumbent_version, self.packs[self.incumbent_version], len(self.state.candidates), sum(1 for t in self.state.trials.values() if t.spec.phase is Phase.SEARCH), "max_candidates reached", self.utc())
        return self.state.search_closed

    # ------------------------------------------------------------------------------------------
    # final phase
    # ------------------------------------------------------------------------------------------
    def freeze_final_analysis(self, per_cell: int = 5, **kw) -> FinalAnalysis:
        ids = [f"a-held-{i + 1}" for i in range(self.n_heldout)]
        planned = tuple(PlannedTrial(f"t-f-{arm.value}-{a[-1]}-{k}", arm, a, 500 + k) for arm in (Arm.V0, Arm.EVOLVED) for a in ids for k in range(per_cell))
        base = dict(
            heldout_metric="prompt_all", test_metric="prompt_strict", assignments=tuple(AssignmentWeight(a, 1.0 / len(ids)) for a in ids), planned_trials=planned, confidence=0.95,
            method=ComparisonMethod.WELCH if len(ids) == 1 else ComparisonMethod.BLOCKED_T, score_seeds=SEEDS, score_params=self.params,
            failed_trial_policy=FailedTrialPolicy.INVALIDATES_CANDIDATE, retry_policy="Retry only after an infrastructure failure that revealed no score.",
            budget_sha256=record_sha256(self.budget), frozen_utc=self.utc(),
        )
        base.update(kw)
        self.analysis = FinalAnalysis(**base)
        self.state.final_analysis = self._frozen("final_analysis", self.layout.final_analysis_path, self.analysis)
        return self.analysis

    def request_final(self) -> "LabStateBuilder":
        """One FinalRequest per arm (stock covers the test split only), as the cli ``final`` command appends them."""
        sha = self.analysis.sha256()
        for i, (arm, ver, splits) in enumerate(((Arm.STOCK, None, (Split.TEST,)), (Arm.V0, "v0", (Split.SELECTION, Split.TEST)), (Arm.EVOLVED, self.incumbent_version, (Split.SELECTION, Split.TEST))), 1):
            rid = f"f-{i:04d}"
            self.final_request_ids[arm] = rid
            self.state.finals.append(FinalRequest(rid, arm, ver, self.packs[ver] if ver else None, sha, splits, None, None, self.utc()))
        return self

    def add_final_trials(self, v0: dict, evolved: dict, v0_test: dict, evolved_test: dict, **trial_kw) -> None:
        """Scores per held-out assignment id, in planned order: ``{"a-held-1": [..]}`` for held-out and for test."""
        for arm, ver, held, test in ((Arm.V0, "v0", v0, v0_test), (Arm.EVOLVED, self.incumbent_version, evolved, evolved_test)):
            for aid in held:
                plan = [p for p in self.analysis.planned_trials if p.arm is arm and p.assignment_id == aid]
                for p, s, ts_ in zip(plan, held[aid], test[aid]):
                    self.add_trial(pack_version=ver, assignment=aid, score=s, test_score=ts_, phase=Phase.FINAL, arm=arm, final_request_id=self.final_request_ids[arm], trial_id=p.trial_id, **trial_kw)

    def add_stock_score(self, test_score: float, n_prompts: Optional[int] = None) -> ScoreRecord:
        return self._write_score(split=Split.TEST, phase=Phase.FINAL, assignment_id=TEST_ASSIGNMENT_ID, trial_id=None, arm=Arm.STOCK, final_request_id=self.final_request_ids[Arm.STOCK], adapter_sha=None, score=test_score, n=n_prompts)

    def reveal(self) -> FinalReveal:
        n = sum(1 for r in self.state.scores.values() if r.phase is Phase.FINAL)
        self.state.reveal = FinalReveal(self.analysis.sha256(), n, self.utc())
        return self.state.reveal

    # ------------------------------------------------------------------------------------------
    def build(self) -> LabState:
        n_events = 2 + len(self.state.scores) + len(self.state.trials_started) * 2 + len(self.state.decisions) * 2
        self.state.head = LedgerAnchor(n_events, sha256_text(f"head{n_events}")[:16])
        return self.state

    @classmethod
    def standard(
        cls,
        root: str | os.PathLike,
        *,
        close: bool = True,
        final: bool = True,
        reveal: bool = True,
        n_heldout: int = 1,
        ledgers: bool = True,
        final_values: Optional[dict] = None,
        **kw,
    ) -> "LabStateBuilder":
        """A complete small experiment. Search scores are near 0.6 to 0.7 (n = 200 prompts); final held-out scores near 0.80 and 0.83 and
        test scores near 0.40 to 0.45, so a final score never coincides with a search number in the printed report."""
        b = cls(root, n_heldout=n_heldout, **kw)
        b.freeze_manifest().freeze_budget()
        b.add_pack("v0")
        b.measure_spread([0.605, 0.620, 0.615, 0.630, 0.640])
        b.freeze_rule()
        b.record_gate()
        # round 0: cand-001 loses its screen, cand-002 is confirmed and accepted, cand-003 fails lint
        b.propose("cand-001", lever=Lever.SEARCH_POLICY, hypothesis="Try fewer, longer training runs.")
        b.propose("cand-002", lever=Lever.CONTEXT_NOTES, hypothesis="Keep a bounded notes file instead of re-reading the whole log.")
        b.propose("cand-003", lever=Lever.TOOLING, hypothesis="Add a helper that rewrites the grader.", lint_passed=False)
        b.judge_screen_loss("cand-001", [0.600, 0.610], 0)
        b.judge_confirm("cand-002", [0.665, 0.675], [0.665, 0.680, 0.690], [0.610, 0.625, 0.620], 0)
        b.judge_invalid("cand-003", "the pack failed lint")
        # round 1 against the new incumbent: cand-004 loses its screen, cand-005 is confirmed and rejected
        b.propose("cand-004", lever=Lever.BUDGET_USE, hypothesis="Spend the first third of the budget on a smoke test.")
        b.propose("cand-005", lever=Lever.VERIFICATION_DISCIPLINE, hypothesis="Require a dev score before every submit.")
        b.judge_screen_loss("cand-004", [0.670, 0.675], 1)
        b.judge_confirm("cand-005", [0.700, 0.705], [0.685, 0.690, 0.695], [0.680, 0.675, 0.690], 1)
        b.add_operator_change()
        b.add_contamination("t-0001", hits=0)
        if close:
            b.close_search()
        if final:
            b.freeze_final_analysis(per_cell=5 if n_heldout == 1 else 3)
            b.request_final()
            cell = 5 if n_heldout == 1 else 3
            fv = {
                "heldout_v0": [0.775, 0.785, 0.780, 0.790, 0.770],
                "heldout_ev": [0.830, 0.840, 0.825, 0.835, 0.845],
                "test_v0": [0.4033, 0.4100, 0.4000, 0.4133, 0.4067],
                "test_ev": [0.4400, 0.4467, 0.4333, 0.4500, 0.4433],
            }
            fv.update(final_values or {})

            def per_assignment(vals):  # a second held-out assignment is shifted a little so that assignments differ
                return {f"a-held-{i + 1}": [round(v + 0.01 * i, 4) for v in vals[:cell]] for i in range(n_heldout)}

            heldout_v0, heldout_ev = per_assignment(fv["heldout_v0"]), per_assignment(fv["heldout_ev"])
            test_v0, test_ev = per_assignment(fv["test_v0"]), per_assignment(fv["test_ev"])
            b.add_final_trials(heldout_v0, heldout_ev, test_v0, test_ev)
            b.add_stock_score(0.3733)
            if reveal:
                b.reveal()
        if ledgers:
            for tid in b.state.trials:
                b.write_trial_ledger(tid)
        return b
