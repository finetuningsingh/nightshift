"""Tests for nightshift.lab.fold (the read model and the queries).

The ledgers here are written with the reused ``nightshift.ledger.Ledger`` directly, because ``LabLedger`` (rightly) refuses the malformed
events the fold has to survive.

Run: PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_fold
"""

import dataclasses
import os
import tempfile
import threading
import unittest
from pathlib import Path

from nightshift.ledger import Ledger
from nightshift.lab import contracts as c
from nightshift.lab.contracts import (
    AdapterFile,
    AdapterRef,
    Arm,
    ContaminationReport,
    ContractError,
    FileRef,
    FinalRequest,
    FrozenFile,
    LabKind,
    LabLayout,
    LabState,
    PackCommitted,
    PackRef,
    PlannedTrial,
    ScoreCost,
    Split,
    Submission,
    TrialStatus,
    write_record,
)
from nightshift.lab.fold import FoldReader, fold_events, fold_ledger
from nightshift.lab.ledger import LabLedger
from tests.test_lab_contracts import (
    H40,
    H64,
    NOW,
    PACK_REPO,
    PERSISTED,
    budget,
    candidate,
    decision,
    final_analysis,
    manifest,
    rule,
    score_record,
    screen_decision,
    seed_agg,
    spec,
    trial_record,
)

C2 = "c" * 40  # a second pack commit
C3 = "d" * 40


def adapter_n(n: int) -> AdapterRef:
    """Adapters with different file hashes, so different ``sha256`` (``adapter()`` of the contracts tests is always the same bytes)."""
    return AdapterRef.from_files(
        f"ad-{n:04d}",
        [AdapterFile("adapter_config.json", f"{n:064x}", 1000), AdapterFile("adapter_model.safetensors", "2" * 64, 766_880_464)],
    )


class Clock:
    def __init__(self) -> None:
        self.t = 1_793_534_400.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


class Lab:
    """A master ledger built with raw appends, plus the frozen files on disk."""

    def __init__(self, test: unittest.TestCase) -> None:
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.layout = LabLayout(Path(tmp.name) / "lab")
        self.layout.root.mkdir()
        self.raw = Ledger.create(self.layout.master_ledger_dir, clock=Clock(), fsync=False)
        test.addCleanup(self.raw.close)
        self.n_adapter = 0
        self.n_seed = 0

    # ---- raw appends --------------------------------------------------------------------------
    def add(self, kind, rec, actor="coordinator") -> int:
        data = rec.to_dict() if hasattr(rec, "to_dict") else rec
        return self.raw.append(kind, actor, data).seq

    def events(self):
        return self.raw.events()

    def state(self) -> LabState:
        return fold_events(self.raw.events())

    def reader(self) -> FoldReader:
        return FoldReader(self.layout)

    # ---- frozen files -------------------------------------------------------------------------
    def freeze(self, kind, name, path: Path, record) -> FrozenFile:
        write_record(path, record)
        ff = FrozenFile(name, self.layout.file_ref(path))
        self.add(kind, ff)
        return ff

    def freeze_basics(self, the_rule=None) -> None:
        self.freeze(LabKind.ASSIGNMENTS_FROZEN, "assignments", self.layout.manifest_path, manifest())
        self.freeze(LabKind.BUDGET_FROZEN, "budget", self.layout.budget_path, budget())
        self.freeze(LabKind.RULE_FROZEN, "rule", self.layout.rule_path, the_rule or rule(score_seeds=(0, 1)))

    # ---- trials -------------------------------------------------------------------------------
    def trial(
        self,
        tid: str,
        *,
        pack_version="v0",
        commit=H40,
        assignment="a-evo-1",
        seed=None,
        p=0.7,
        status=TrialStatus.COMPLETED,
        finish=True,
        score_kw=None,
        record_kw=None,
    ):
        if seed is None:
            self.n_seed += 1
            seed = 1000 + self.n_seed
        sp = spec(trial_id=tid, pack_version=pack_version, pack=PackRef(PACK_REPO, commit), assignment_id=assignment, seed=seed, score_seeds=(0, 1))
        self.add(LabKind.TRIAL_STARTED, sp)
        if not finish:
            return sp, None, None
        if status in (TrialStatus.COMPLETED, TrialStatus.BUDGET_EXHAUSTED):
            self.n_adapter += 1
            ad = adapter_n(self.n_adapter)
            sk = dict(
                score_id=f"sc-{tid}",
                request_id=f"sc-{tid}",
                trial_id=tid,
                assignment_id=assignment,
                adapter_sha256=ad.sha256,
                per_seed=(seed_agg(0, p), seed_agg(1, p)),
            )
            sk.update(score_kw or {})
            sc = score_record(**sk)
            self.add(LabKind.SCORE_RECORDED, sc)
            kw = dict(
                spec=sp,
                status=status,
                submission=Submission(ad.adapter_id, H64, H40, 0.7),
                adapter=ad,
                dev_score_id=f"dev-{tid}",
                selection_score_id=sc.score_id,
            )
        else:
            sc = None
            kw = dict(spec=sp, status=status, submission=None, adapter=None, dev_score_id=None, selection_score_id=None, failure="it broke")
        kw.update(record_kw or {})
        rec = trial_record(**kw)
        self.add(LabKind.TRIAL_FINISHED, rec)
        return sp, sc, rec


class FoldBasicsTests(unittest.TestCase):
    def test_empty_ledger_is_an_empty_state(self):
        st = fold_events([])
        self.assertEqual(st, LabState())
        self.assertIsNone(st.head)

    def test_a_full_ledger_folds_into_every_field(self):
        lab = Lab(self)
        lab.add(LabKind.INITIALIZED, PERSISTED["LabInit"]())
        lab.freeze_basics()
        lab.add(LabKind.SPREAD_PLANNED, PERSISTED["SpreadPlan"]())
        lab.add(LabKind.SPREAD_MEASURED, PERSISTED["SpreadMeasured"]())
        lab.add(LabKind.GATE_RECORDED, PERSISTED["GateRecord"]())
        lab.add(LabKind.PACK_COMMITTED, PERSISTED["PackCommitted"]())
        sp, sc, rec = lab.trial("t-0001")
        lab.add(LabKind.SCORE_COST, ScoreCost("sc-t-0001", c.GpuUsage(20.0, 0.5, ("r9",))))
        lab.add(LabKind.SCORE_FAILED, PERSISTED["ScoreFailure"]())
        lab.add(LabKind.CONTAMINATION_SCANNED, PERSISTED["ContaminationReport"]())
        lab.add(LabKind.CANDIDATE_PROPOSED, candidate())
        lab.add(LabKind.CANDIDATE_DECIDED, decision())
        lab.add(LabKind.OPERATOR_CHANGE, PERSISTED["OperatorChange"]())
        lab.add(LabKind.SEARCH_CLOSED, PERSISTED["SearchClosed"]())
        lab.add(LabKind.FINAL_ANALYSIS_FROZEN, PERSISTED["FrozenFile"]())
        lab.add(LabKind.FINAL_REQUESTED, PERSISTED["FinalRequest"]())
        lab.add(LabKind.FINAL_REVEALED, PERSISTED["FinalReveal"]())
        st = lab.state()
        self.assertEqual(st.malformed, [])
        self.assertEqual(st.initialized, PERSISTED["LabInit"]())
        self.assertEqual((st.manifest.name, st.budget.name, st.rule.name), ("assignments", "budget", "rule"))
        self.assertEqual(st.spread_plan, PERSISTED["SpreadPlan"]())
        self.assertEqual(st.spread, PERSISTED["SpreadMeasured"]())
        self.assertEqual(st.gates, [PERSISTED["GateRecord"]()])
        self.assertEqual(st.packs, {"v0": PERSISTED["PackCommitted"]()})
        self.assertEqual(st.trials_started, {"t-0001": sp})
        self.assertEqual(st.trials, {"t-0001": rec})
        self.assertEqual(st.scores, {sc.score_id: sc})
        self.assertEqual(list(st.score_costs), ["sc-t-0001"])
        self.assertEqual(st.score_failures, [PERSISTED["ScoreFailure"]()])
        self.assertEqual(list(st.contamination), ["t001"])
        self.assertEqual(list(st.candidates), ["cand-001"])
        self.assertEqual(list(st.decisions), ["cand-001"])
        self.assertEqual(st.operator_changes, [PERSISTED["OperatorChange"]()])
        self.assertEqual(st.search_closed, PERSISTED["SearchClosed"]())
        self.assertEqual(st.final_analysis, PERSISTED["FrozenFile"]())
        self.assertEqual(st.finals, [PERSISTED["FinalRequest"]()])
        self.assertEqual(st.reveal, PERSISTED["FinalReveal"]())
        last = lab.events()[-1]
        self.assertEqual(st.head, c.LedgerAnchor(last.seq, last.hash[:16]))

    def test_fold_ledger_reads_the_file_and_agrees_with_fold_events(self):
        lab = Lab(self)
        lab.freeze_basics()
        lab.trial("t-0001")
        self.assertEqual(fold_ledger(lab.layout.master_ledger_dir), lab.state())

    def test_event_seq_records_the_first_event_of_each_kind_and_key(self):
        lab = Lab(self)
        s_rule = lab.add(LabKind.RULE_FROZEN, PERSISTED["FrozenFile"]())
        s_cand = lab.add(LabKind.CANDIDATE_PROPOSED, candidate())
        st = lab.state()
        self.assertEqual(st.event_seq[(LabKind.RULE_FROZEN, "")], s_rule)
        self.assertEqual(st.event_seq[(LabKind.CANDIDATE_PROPOSED, "cand-001")], s_cand)


class MalformedTests(unittest.TestCase):
    def reasons(self, st):
        return [(s, r) for s, r in st.malformed]

    def only(self, st, seq, text):
        self.assertEqual(len(st.malformed), 1, st.malformed)
        self.assertEqual(st.malformed[0][0], seq)
        self.assertIn(text, st.malformed[0][1])

    def test_unknown_foreign_and_trial_kinds(self):
        lab = Lab(self)
        a = lab.add("lab.nope", {"x": 1})
        b = lab.add("budget.charged", {"x": 1})
        d = lab.add(LabKind.BUDGET_CHARGED, PERSISTED["BudgetCharge"]())
        e = lab.add(LabKind.GPU_JOB_STARTED, PERSISTED["TrainJob"]())
        st = lab.state()
        self.assertEqual([s for s, _ in st.malformed], [a, b, d, e])
        self.assertIn("unknown or non-lab", st.malformed[0][1])
        self.assertIn("trial ledger", st.malformed[2][1])
        self.assertEqual(st.trials_started, {})

    def test_bad_payloads_are_listed_not_raised(self):
        lab = Lab(self)
        a = lab.add(LabKind.TRIAL_STARTED, {"trial_id": "t1"})
        b = lab.add(LabKind.OPERATOR_CHANGE, {"target": "x", "detail": "y", "candidates_so_far": -1, "recorded_utc": NOW, "schema_version": 1})
        d = lab.add(LabKind.RULE_FROZEN, {"schema_version": 99, "name": "rule", "file": {"path": "a", "sha256": H64, "size_bytes": 1}})
        e = lab.add(LabKind.GATE_RECORDED, "not even a dict")
        st = lab.state()
        self.assertEqual([s for s, _ in st.malformed], [a, b, d, e])
        self.assertTrue(all("bad payload" in r for _, r in st.malformed))
        self.assertIsNone(st.rule)

    def test_second_event_of_every_freeze_once_kind_is_listed_and_the_first_wins(self):
        other = {
            LabKind.SPREAD_PLANNED: dataclasses.replace(PERSISTED["SpreadPlan"](), planned_trials=6),
            LabKind.SPREAD_MEASURED: None,
            LabKind.SEARCH_CLOSED: dataclasses.replace(PERSISTED["SearchClosed"](), reason="another reason"),
            LabKind.FINAL_REVEALED: dataclasses.replace(PERSISTED["FinalReveal"](), n_scores=99),
        }
        attr = {
            LabKind.ASSIGNMENTS_FROZEN: "manifest", LabKind.BUDGET_FROZEN: "budget", LabKind.SPREAD_PLANNED: "spread_plan",
            LabKind.SPREAD_MEASURED: "spread", LabKind.RULE_FROZEN: "rule", LabKind.SEARCH_CLOSED: "search_closed",
            LabKind.FINAL_ANALYSIS_FROZEN: "final_analysis", LabKind.FINAL_REVEALED: "reveal",
        }
        self.assertEqual(set(attr), set(LabKind.FREEZE_ONCE))
        for kind in LabKind.FREEZE_ONCE:
            with self.subTest(kind):
                lab = Lab(self)
                first = PERSISTED[c.PAYLOAD_TYPES[kind].__name__]()
                second = other.get(kind)
                if second is None:
                    second = FrozenFile("other", FileRef("frozen/other.json", "9" * 64, 3)) if isinstance(first, FrozenFile) else first
                lab.add(kind, first)
                s2 = lab.add(kind, second)
                st = lab.state()
                self.only(st, s2, "the first one wins")
                self.assertEqual(getattr(st, attr[kind]), first)

    def test_version_label_mapping_to_two_commits(self):
        lab = Lab(self)
        lab.add(LabKind.PACK_COMMITTED, PackCommitted("v0", PackRef(PACK_REPO, H40), None, None))
        lab.add(LabKind.PACK_COMMITTED, PackCommitted("v0", PackRef(PACK_REPO, H40), None, None))  # the same commit again is harmless
        bad = lab.add(LabKind.PACK_COMMITTED, PackCommitted("v0", PackRef(PACK_REPO, C2), None, None))
        st = lab.state()
        self.only(st, bad, "two commits")
        self.assertEqual(st.packs["v0"].pack.commit, H40)

    def test_duplicate_trial_id_and_duplicate_phase_commit_assignment_seed(self):
        lab = Lab(self)
        lab.trial("t-0001", seed=5, finish=False)
        s_dup_id = lab.add(LabKind.TRIAL_STARTED, spec(trial_id="t-0001", pack=PackRef(PACK_REPO, H40), seed=6))
        s_dup_key = lab.add(LabKind.TRIAL_STARTED, spec(trial_id="t-0002", pack=PackRef(PACK_REPO, H40), seed=5))
        # same seed on another assignment, another commit or another phase is a different trial
        lab.add(LabKind.TRIAL_STARTED, spec(trial_id="t-0003", pack=PackRef(PACK_REPO, H40), seed=5, assignment_id="a-evo-2"))
        lab.add(LabKind.TRIAL_STARTED, spec(trial_id="t-0004", pack=PackRef(PACK_REPO, C2), seed=5))
        st = lab.state()
        self.assertEqual([s for s, _ in st.malformed], [s_dup_id, s_dup_key])
        self.assertIn("duplicate lab.trial_started", st.malformed[0][1])
        self.assertIn("repeats", st.malformed[1][1])
        self.assertEqual(list(st.trials_started), ["t-0001", "t-0003", "t-0004"])

    def test_final_phase_trials_of_the_same_commit_may_share_a_seed_across_arms(self):
        lab = Lab(self)
        lab.add(LabKind.TRIAL_STARTED, spec(trial_id="f-1", phase=c.Phase.FINAL, arm=Arm.V0, assignment_id="a-held-1", final_request_id="f-0001", seed=100))
        lab.add(LabKind.TRIAL_STARTED, spec(trial_id="f-2", phase=c.Phase.FINAL, arm=Arm.EVOLVED, assignment_id="a-held-1", final_request_id="f-0001", seed=100))
        bad = lab.add(LabKind.TRIAL_STARTED, spec(trial_id="f-3", phase=c.Phase.FINAL, arm=Arm.EVOLVED, assignment_id="a-held-1", final_request_id="f-0001", seed=100))
        st = lab.state()
        self.only(st, bad, "repeats")
        self.assertEqual(list(st.trials_started), ["f-1", "f-2"])

    def test_duplicate_trial_finished_and_finished_without_started(self):
        lab = Lab(self)
        sp, sc, rec = lab.trial("t-0001")
        dup = lab.add(LabKind.TRIAL_FINISHED, rec)
        other = trial_record(spec=spec(trial_id="t-0009", seed=77))
        orphan = lab.add(LabKind.TRIAL_FINISHED, other)
        st = lab.state()
        self.assertEqual([s for s, _ in st.malformed], [dup, orphan])
        self.assertIn("duplicate lab.trial_finished", st.malformed[0][1])
        self.assertIn("without lab.trial_started", st.malformed[1][1])
        self.assertEqual(list(st.trials), ["t-0001"])

    def test_a_trial_finished_whose_spec_differs_from_the_started_one_is_listed_and_not_counted(self):
        lab = Lab(self)
        started = spec(trial_id="t-0001", pack_version="v0", seed=3)
        lab.add(LabKind.TRIAL_STARTED, started)
        swapped = lab.add(LabKind.TRIAL_FINISHED, trial_record(spec=dataclasses.replace(started, pack_version="cand-999", seed=4)))
        st = lab.state()
        self.only(st, swapped, "differs from its lab.trial_started")
        self.assertEqual(st.trials, {})
        # the identical spec is accepted afterwards
        lab.add(LabKind.TRIAL_FINISHED, trial_record(spec=started))
        self.assertEqual(list(lab.state().trials), ["t-0001"])

    def test_score_problems(self):
        lab = Lab(self)
        sp, sc, rec = lab.trial("t-0001")
        dup_id = lab.add(LabKind.SCORE_RECORDED, sc)
        # a second selection score for the same adapter, assignment and phase (new id): the first is used
        again = score_record(score_id="sc-again", request_id="sc-again", trial_id="t-0001", adapter_sha256=sc.adapter_sha256, per_seed=(seed_agg(0, 0.1), seed_agg(1, 0.1)))
        second = lab.add(LabKind.SCORE_RECORDED, again)
        # a score of a trial that never started
        orphan = lab.add(LabKind.SCORE_RECORDED, score_record(score_id="sc-orphan", request_id="sc-orphan", trial_id="t-9999", adapter_sha256="9" * 64))
        # dev scores of the same adapter are not subject to the one-score rule
        lab.add(LabKind.SCORE_RECORDED, score_record(score_id="sc-dev-1", request_id="sc-dev-1", trial_id="t-0001", split=Split.DEV, adapter_sha256=sc.adapter_sha256))
        lab.add(LabKind.SCORE_RECORDED, score_record(score_id="sc-dev-2", request_id="sc-dev-2", trial_id="t-0001", split=Split.DEV, adapter_sha256=sc.adapter_sha256))
        st = lab.state()
        self.assertEqual([s for s, _ in st.malformed], [dup_id, second, orphan])
        self.assertIn("duplicate lab.score_recorded", st.malformed[0][1])
        self.assertIn("the first one is used", st.malformed[1][1])
        self.assertIn("no lab.trial_started", st.malformed[2][1])
        self.assertEqual(sorted(st.scores), ["sc-dev-1", "sc-dev-2", sc.score_id])
        self.assertEqual(st.scores[sc.score_id], sc)

    def test_second_test_score_for_the_same_adapter_is_listed(self):
        lab = Lab(self)
        lab.add(LabKind.TRIAL_STARTED, spec(trial_id="f-1", phase=c.Phase.FINAL, arm=Arm.EVOLVED, assignment_id="a-held-1", final_request_id="f-0001"))
        from tests.test_lab_contracts import ifbench_record

        first = ifbench_record(score_id="sc-t1", request_id="sc-t1", trial_id="f-1", adapter_sha256="7" * 64)
        lab.add(LabKind.SCORE_RECORDED, first)
        second = lab.add(LabKind.SCORE_RECORDED, ifbench_record(score_id="sc-t2", request_id="sc-t2", trial_id="f-1", adapter_sha256="7" * 64))
        stock_a = lab.add(LabKind.SCORE_RECORDED, ifbench_record(score_id="sc-s1", request_id="sc-s1", trial_id=None, arm=Arm.STOCK, adapter_sha256=None))
        stock_b = lab.add(LabKind.SCORE_RECORDED, ifbench_record(score_id="sc-s2", request_id="sc-s2", trial_id=None, arm=Arm.STOCK, adapter_sha256=None))
        st = lab.state()
        self.assertEqual([s for s, _ in st.malformed], [second, stock_b])
        self.assertEqual(sorted(st.scores), ["sc-s1", "sc-t1"])
        self.assertLess(stock_a, stock_b)

    def test_duplicate_score_cost_candidate_and_decision_problems(self):
        lab = Lab(self)
        lab.add(LabKind.SCORE_COST, ScoreCost("sc1", c.GpuUsage(1.0, 1.0, ())))
        cost2 = lab.add(LabKind.SCORE_COST, ScoreCost("sc1", c.GpuUsage(2.0, 2.0, ())))
        early = lab.add(LabKind.CANDIDATE_DECIDED, decision())  # no rule yet
        lab.add(LabKind.RULE_FROZEN, PERSISTED["FrozenFile"]())
        no_cand = lab.add(LabKind.CANDIDATE_DECIDED, decision())
        lab.add(LabKind.CANDIDATE_PROPOSED, candidate())
        dup_cand = lab.add(LabKind.CANDIDATE_PROPOSED, candidate())
        lab.add(LabKind.CANDIDATE_DECIDED, decision())
        dup_dec = lab.add(LabKind.CANDIDATE_DECIDED, screen_decision())
        st = lab.state()
        self.assertEqual([s for s, _ in st.malformed], [cost2, early, no_cand, dup_cand, dup_dec])
        self.assertIn("before lab.rule_frozen", st.malformed[1][1])
        self.assertIn("no lab.candidate_proposed", st.malformed[2][1])
        self.assertTrue(st.decisions["cand-001"].accepted)  # the first decision wins
        self.assertEqual(st.score_costs["sc1"].gpu.usd, 1.0)

    def test_final_request_problems(self):
        lab = Lab(self)
        pack = PackRef(PACK_REPO, H40)

        def req(rid, arm=Arm.V0, repeat_of=None, reason=None, pk=pack):
            version = None if arm is Arm.STOCK else "v0"
            return FinalRequest(rid, arm, version, None if arm is Arm.STOCK else pk, H64, (Split.SELECTION, Split.TEST), repeat_of, reason, NOW)

        lab.add(LabKind.FINAL_REQUESTED, req("f-0001"))
        dup_id = lab.add(LabKind.FINAL_REQUESTED, req("f-0001"))
        no_repeat = lab.add(LabKind.FINAL_REQUESTED, req("f-0002"))
        unknown = lab.add(LabKind.FINAL_REQUESTED, req("f-0003", repeat_of="f-9999", reason="scorer crashed"))
        lab.add(LabKind.FINAL_REQUESTED, req("f-0004", repeat_of="f-0001", reason="scorer crashed"))
        lab.add(LabKind.FINAL_REQUESTED, req("f-0005", Arm.STOCK))
        lab.add(LabKind.FINAL_REQUESTED, req("f-0006", Arm.EVOLVED, pk=PackRef(PACK_REPO, C2)))
        stock_again = lab.add(LabKind.FINAL_REQUESTED, req("f-0007", Arm.STOCK))
        st = lab.state()
        self.assertEqual([s for s, _ in st.malformed], [dup_id, no_repeat, unknown, stock_again])
        self.assertIn("without repeat_of", st.malformed[1][1])
        self.assertIn("unknown request", st.malformed[2][1])
        self.assertEqual([r.request_id for r in st.finals], ["f-0001", "f-0004", "f-0005", "f-0006"])

    def test_second_initialized(self):
        lab = Lab(self)
        lab.add(LabKind.INITIALIZED, PERSISTED["LabInit"]())
        s2 = lab.add(LabKind.INITIALIZED, c.LabInit("lab2", NOW, "elsewhere"))
        st = lab.state()
        self.only(st, s2, "first one wins")
        self.assertEqual(st.initialized.lab_id, "lab1")

    def test_a_contamination_hit_is_sticky(self):
        lab = Lab(self)

        def rep(job, hits):
            return ContaminationReport("t-0001", job, FileRef(f"trials/t-0001/train_data/{job}.jsonl", H64, 5), 10, hits, 0, 13, (H64,), hits == 0, NOW)

        lab.add(LabKind.CONTAMINATION_SCANNED, rep("j-0001", 0))
        lab.add(LabKind.CONTAMINATION_SCANNED, rep("j-0002", 2))
        lab.add(LabKind.CONTAMINATION_SCANNED, rep("j-0003", 0))
        st = lab.state()
        self.assertFalse(st.contamination["t-0001"].passed)
        self.assertEqual(st.contamination["t-0001"].job_id, "j-0002")

    def test_junk_events_never_raise(self):
        from nightshift.types import Event

        junk = [
            Event(1, 1.0, "r", None, "a", {}, "0" * 64, "h" * 64),
            Event(2, 1.0, "r", 5, "a", [], "0" * 64, "h" * 64),
            Event(3, 1.0, "r", LabKind.RULE_FROZEN, "a", None, "0" * 64, "h" * 64),
            Event(4, 1.0, "r", LabKind.TRIAL_STARTED, "a", {"schema_version": "x"}, "0" * 64, "h" * 64),
        ]
        st = fold_events(junk)
        self.assertEqual([s for s, _ in st.malformed], [1, 2, 3, 4])


class FoldReaderBasicsTests(unittest.TestCase):
    def test_state_reads_the_file_or_a_live_ledger(self):
        lab = Lab(self)
        lab.trial("t-0001", finish=False)
        reader = lab.reader()
        self.assertEqual(list(reader.state().trials_started), ["t-0001"])
        live = FoldReader(lab.layout, ledger=_FakeLive(lab.events()))
        self.assertEqual(list(live.state().trials_started), ["t-0001"])

    def test_incumbent_version_is_the_latest_accepted_candidate(self):
        lab = Lab(self)
        r = lab.reader()
        lab.add(LabKind.RULE_FROZEN, PERSISTED["FrozenFile"]())
        self.assertEqual(r.incumbent_version(lab.state()), "v0")
        for cid in ("cand-001", "cand-002", "cand-003"):
            lab.add(LabKind.CANDIDATE_PROPOSED, candidate(candidate_id=cid, version_label=cid))
        lab.add(LabKind.CANDIDATE_DECIDED, screen_decision(candidate_id="cand-001", candidate_version="cand-001"))
        self.assertEqual(r.incumbent_version(lab.state()), "v0")
        lab.add(LabKind.CANDIDATE_DECIDED, decision(candidate_id="cand-002", candidate_version="cand-002"))
        self.assertEqual(r.incumbent_version(lab.state()), "cand-002")
        lab.add(LabKind.CANDIDATE_DECIDED, decision(candidate_id="cand-003", candidate_version="cand-003", outcome=c.Outcome.REJECTED, delta=0.01, margin=0.03))
        self.assertEqual(r.incumbent_version(lab.state()), "cand-002")

    def test_has_gate(self):
        lab = Lab(self)
        r = lab.reader()
        self.assertFalse(r.has_gate(lab.state(), c.Gate.TERMS_10_1_IX))
        lab.add(LabKind.GATE_RECORDED, PERSISTED["GateRecord"]())
        self.assertTrue(r.has_gate(lab.state(), c.Gate.TERMS_10_1_IX))

    def test_gate_status_is_closed_only_by_a_clarification(self):
        lab = Lab(self)
        r = lab.reader()
        gate = c.Gate.TERMS_10_1_IX
        self.assertEqual(r.gate_status(lab.state(), gate), "unrecorded")
        lab.add(LabKind.GATE_RECORDED, c.GateRecord(gate, c.GateOutcome.PROCEED_WITH_RISK, "waiver", "owner", "2026-11-02T12:00:00+00:00"))
        self.assertEqual(r.gate_status(lab.state(), gate), "open_waived")
        self.assertTrue(r.has_gate(lab.state(), gate))  # a waiver still lets gated work start
        lab.add(LabKind.GATE_RECORDED, c.GateRecord(gate, c.GateOutcome.CLARIFIED, "Meta answered", "owner", "2026-11-02T13:00:00+00:00"))
        self.assertEqual(r.gate_status(lab.state(), gate), "closed")

    def test_final_requests_by_pack(self):
        lab = Lab(self)
        r = lab.reader()
        v0, ev = PackRef(PACK_REPO, H40), PackRef(PACK_REPO, C2)
        mk = lambda rid, arm, pk: FinalRequest(rid, arm, None if pk is None else "v", pk, H64, (Split.TEST,), None, None, NOW)  # noqa: E731
        lab.add(LabKind.FINAL_REQUESTED, mk("f-0001", Arm.STOCK, None))
        lab.add(LabKind.FINAL_REQUESTED, mk("f-0002", Arm.V0, v0))
        lab.add(LabKind.FINAL_REQUESTED, mk("f-0003", Arm.EVOLVED, ev))
        st = lab.state()
        self.assertEqual([x.request_id for x in r.final_requests(st, None)], ["f-0001"])
        self.assertEqual([x.request_id for x in r.final_requests(st, v0)], ["f-0002"])
        self.assertEqual([x.request_id for x in r.final_requests(st, ev)], ["f-0003"])
        self.assertEqual(r.final_requests(st, PackRef(PACK_REPO, C3)), [])


class _FakeLive:
    def __init__(self, events):
        self._events = events

    def events(self, since_seq=0):
        return list(self._events)


class LoadFrozenTests(unittest.TestCase):
    def test_loads_the_three_frozen_files(self):
        lab = Lab(self)
        lab.freeze_basics()
        got_rule, got_manifest, got_budget = lab.reader().load_frozen_all(lab.state())
        self.assertEqual((got_rule, got_manifest, got_budget), (rule(score_seeds=(0, 1)), manifest(), budget()))

    def test_something_not_frozen_is_an_error(self):
        lab = Lab(self)
        lab.freeze(LabKind.BUDGET_FROZEN, "budget", lab.layout.budget_path, budget())
        with self.assertRaises(ContractError):
            lab.reader().load_frozen_all(lab.state())

    def test_a_rewritten_file_is_rejected(self):
        for name in ("rule", "assignments", "budget"):
            with self.subTest(name):
                lab = Lab(self)
                lab.freeze_basics()
                path = {"rule": lab.layout.rule_path, "assignments": lab.layout.manifest_path, "budget": lab.layout.budget_path}[name]
                os.chmod(path, 0o644)
                if name == "rule":
                    write_record(path, rule(score_seeds=(0, 1), alpha=0.5))  # still a valid Rule, different bytes
                elif name == "budget":
                    write_record(path, c.BudgetSpec(2_000_000, 120.0, 99.0, 7200.0))
                else:
                    write_record(path, manifest(created_utc="2026-11-02T12:00:00+00:00"))
                with self.assertRaises(ContractError) as cm:
                    lab.reader().load_frozen_all(lab.state())
                self.assertIn("does not match", str(cm.exception))

    def test_a_same_size_edit_and_a_symlink_are_rejected(self):
        lab = Lab(self)
        lab.freeze_basics()
        raw = lab.layout.budget_path.read_bytes()
        lab.layout.budget_path.write_bytes(raw.replace(b"25.0", b"26.0"))
        self.assertEqual(len(raw), lab.layout.budget_path.stat().st_size)
        with self.assertRaises(ContractError):
            lab.reader().load_frozen_all(lab.state())
        lab.layout.budget_path.write_bytes(raw)
        lab.reader().load_frozen_all(lab.state())  # restored bytes load again
        elsewhere = lab.layout.root / "elsewhere.json"
        elsewhere.write_bytes(raw)
        lab.layout.budget_path.unlink()
        lab.layout.budget_path.symlink_to(elsewhere)
        with self.assertRaises(ContractError):
            lab.reader().load_frozen_all(lab.state())

    def test_final_analysis_loads_and_is_hash_checked(self):
        lab = Lab(self)
        analysis = final_analysis()
        lab.freeze(LabKind.FINAL_ANALYSIS_FROZEN, "final_analysis", lab.layout.final_analysis_path, analysis)
        self.assertEqual(lab.reader().load_final_analysis(lab.state()), analysis)
        with self.assertRaises(ContractError):
            FoldReader(lab.layout).load_final_analysis(LabState())
        lab.layout.final_analysis_path.write_text(lab.layout.final_analysis_path.read_text() + " ")
        with self.assertRaises(ContractError):
            lab.reader().load_final_analysis(lab.state())


class SelectionScoresTests(unittest.TestCase):
    def setUp(self):
        self.lab = Lab(self)
        self.rule = rule(score_seeds=(0, 1))
        self.reader = self.lab.reader()

    def scores(self, version="v0", the_rule=None):
        return self.reader.selection_trial_scores(self.lab.state(), the_rule or self.rule, version)

    def test_scored_trials_become_trial_scores_with_the_mean_over_seeds(self):
        self.lab.trial("t-0001", p=0.70)
        self.lab.trial("t-0002", p=0.60)
        out = self.scores()
        self.assertEqual([(s.trial_id, s.assignment_id, s.score_id, s.metric, s.scorer_version) for s in out],
                         [("t-0001", "a-evo-1", "sc-t-0001", "prompt_all", "s1"), ("t-0002", "a-evo-1", "sc-t-0002", "prompt_all", "s1")])
        self.assertEqual([s.score for s in out], [0.70, 0.60])
        st = self.lab.state()
        self.assertEqual(out[0].comparability_sha256, c.comparability_key(st.trials_started["t-0001"], "prompt_all"))

    def test_mean_over_seeds_of_the_rule_metric(self):
        self.lab.trial("t-0001", score_kw=dict(per_seed=(seed_agg(0, 0.5), seed_agg(1, 0.7))))
        (s,) = self.scores()
        self.assertAlmostEqual(s.score, 0.6)
        (f,) = self.scores(the_rule=rule(score_seeds=(0, 1), metric="prompt_fraction"))
        self.assertAlmostEqual(f.score, 0.65)
        self.assertEqual(f.metric, "prompt_fraction")

    def test_only_search_phase_trials_of_the_pack_version_on_the_rule_assignments(self):
        self.lab.trial("t-0001", pack_version="v0")
        self.lab.trial("t-0002", pack_version="cand-001", commit=C2)
        self.lab.trial("t-0003", pack_version="v0", assignment="a-other")
        self.lab.add(LabKind.TRIAL_STARTED, spec(trial_id="f-0001", phase=c.Phase.FINAL, arm=Arm.V0, assignment_id="a-held-1", final_request_id="f-0001", pack_version="v0"))
        self.assertEqual([s.trial_id for s in self.scores("v0")], ["t-0001"])
        self.assertEqual([s.trial_id for s in self.scores("cand-001")], ["t-0002"])
        self.assertEqual(self.scores("cand-002"), [])

    def test_a_trial_that_has_not_finished_is_not_counted(self):
        self.lab.trial("t-0001")
        self.lab.trial("t-0002", finish=False)
        self.assertEqual([s.trial_id for s in self.scores()], ["t-0001"])

    def test_budget_exhausted_trial_with_a_scored_adapter_counts_normally(self):
        self.lab.trial("t-0001", status=TrialStatus.BUDGET_EXHAUSTED, p=0.4)
        (s,) = self.scores()
        self.assertEqual((s.trial_id, s.score_id, s.score), ("t-0001", "sc-t-0001", 0.4))

    def test_other_non_scored_trials_follow_the_failed_trial_policy(self):
        for status in (TrialStatus.FAILED, TrialStatus.NO_SUBMISSION, TrialStatus.ABORTED, TrialStatus.CONTAMINATED):
            with self.subTest(status=status.value):
                self.setUp()
                self.lab.trial("t-0001", p=0.7)
                self.lab.trial("t-0002", status=status)
                self.lab.trial("t-0003", p=0.8)
                self.assertEqual(self.scores(), [])  # INVALIDATES_CANDIDATE: the whole set is unusable
                zero = rule(score_seeds=(0, 1), failed_trial_policy=c.FailedTrialPolicy.SCORED_ZERO)
                out = self.scores(the_rule=zero)
                self.assertEqual([(s.trial_id, s.score_id, s.score) for s in out], [("t-0001", "sc-t-0001", 0.7), ("t-0002", None, 0.0), ("t-0003", "sc-t-0003", 0.8)])
                st = self.lab.state()
                self.assertEqual(out[1].comparability_sha256, c.comparability_key(st.trials_started["t-0002"], "prompt_all"))

    def test_a_failing_contamination_report_voids_even_a_completed_trial(self):
        self.lab.trial("t-0001", p=0.9)
        bad = ContaminationReport("t-0001", "j-0001", FileRef("trials/t-0001/train_data/j-0001.jsonl", H64, 5), 10, 1, 0, 13, (H64,), False, NOW)
        self.lab.add(LabKind.CONTAMINATION_SCANNED, bad)
        self.assertEqual(self.scores(), [])
        zero = rule(score_seeds=(0, 1), failed_trial_policy=c.FailedTrialPolicy.SCORED_ZERO)
        (s,) = self.scores(the_rule=zero)
        self.assertEqual((s.score_id, s.score), (None, 0.0))
        # a clean report does not void it
        self.setUp()
        self.lab.trial("t-0001", p=0.5)
        clean = dataclasses.replace(bad, n_exact_hits=0, passed=True)
        self.lab.add(LabKind.CONTAMINATION_SCANNED, clean)
        self.assertEqual([s.score for s in self.scores()], [0.5])

    def test_a_score_that_differs_from_the_rule_is_not_used_and_is_listed(self):
        cases = {
            "scorer version": dict(scorer_version="s2"),
            "decoding": dict(params=c.GenerationParams.recommended(4000)),
            "seeds": dict(per_seed=(seed_agg(0, 0.7), seed_agg(7, 0.7))),
            "adapter": dict(adapter_sha256="e" * 64),
            "base checkpoint": dict(base=c.BaseModelRef("org/base", "b" * 40)),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                self.setUp()
                self.lab.trial("t-0001", score_kw=kw)
                self.assertEqual(self.scores(), [])
                st = self.lab.state()
                self.reader.selection_trial_scores(st, self.rule, "v0")
                self.reader.selection_trial_scores(st, self.rule, "v0")  # asking twice lists it once
                self.assertEqual(len(st.malformed), 1)
                seq, reason = st.malformed[0]
                self.assertIn("differs from the rule", reason)
                self.assertIn(name, reason)
                self.assertEqual(seq, st.event_seq[(LabKind.SCORE_RECORDED, "sc-t-0001")])

    def test_a_missing_metric_is_a_mismatch(self):
        self.lab.trial("t-0001", p=0.7)
        zero = rule(score_seeds=(0, 1), failed_trial_policy=c.FailedTrialPolicy.SCORED_ZERO, metric="prompt_fraction")
        self.assertEqual(len(self.scores(the_rule=zero)), 1)  # prompt_fraction exists on selection scores
        st = self.lab.state()
        st.scores["sc-t-0001"] = dataclasses.replace(
            st.scores["sc-t-0001"], per_seed=tuple(dataclasses.replace(s, metrics={"prompt_all": 0.7}) for s in st.scores["sc-t-0001"].per_seed)
        )
        self.assertEqual(self.reader.selection_trial_scores(st, zero, "v0")[0].score_id, None)

    def test_a_trial_record_that_names_a_missing_score_is_a_failed_trial(self):
        self.lab.trial("t-0001", record_kw=dict(selection_score_id="sc-gone"))
        self.lab.trial("t-0002")
        self.assertEqual(self.scores(), [])
        st = self.lab.state()
        self.reader.selection_trial_scores(st, self.rule, "v0")
        self.assertIn("not in the ledger", st.malformed[0][1])

    def test_a_second_score_of_the_same_adapter_is_not_the_one_a_trial_names(self):
        sp, sc, rec = self.lab.trial("t-0001", p=0.7)
        again = score_record(score_id="sc-again", request_id="sc-again", trial_id="t-0001", adapter_sha256=sc.adapter_sha256, per_seed=(seed_agg(0, 0.2), seed_agg(1, 0.2)))
        self.lab.add(LabKind.SCORE_RECORDED, again)
        (s,) = self.scores()
        self.assertEqual((s.score_id, s.score), ("sc-t-0001", 0.7))  # the first completed record is the one used


class CostBreakdownTests(unittest.TestCase):
    def test_a_failed_scoring_attempt_that_was_billed_counts_as_scorer_cost(self):
        lab = Lab(self)
        lab.trial("t-0001")
        lab.trial("t-0002")
        lab.add(LabKind.SCORE_FAILED, c.ScoreFailure("sc-fail", c.Split.SELECTION, "a-evo-1", "t-0001", "a" * 64, "endpoint died", False, NOW))
        lab.add(LabKind.SCORE_COST, ScoreCost("sc-fail", c.GpuUsage(20.0, 0.26, ("r1",))))
        lab.add(LabKind.SCORE_COST, ScoreCost("sc-t-0001", c.GpuUsage(20.0, 0.5, ("r2",))))
        self.assertAlmostEqual(lab.reader().cost_breakdown(lab.state(), ["t-0001"], None).scorer_usd, 0.76)
        self.assertAlmostEqual(lab.reader().cost_breakdown(lab.state(), ["t-0002"], None).scorer_usd, 0.0)

    def test_the_four_cost_centres(self):
        lab = Lab(self)
        lab.add(LabKind.CANDIDATE_PROPOSED, candidate(propose_usd=0.31))
        lab.trial("t-0001", record_kw=dict(budget_used=c.BudgetUsage(10, 1.0, 2.0, 3.0, 4.0)))
        lab.trial("t-0002", record_kw=dict(budget_used=c.BudgetUsage(10, 1.0, 0.5, 0.25, 4.0)))
        lab.trial("t-0003", record_kw=dict(budget_used=c.BudgetUsage(10, 1.0, 100.0, 100.0, 4.0)))  # not asked for
        lab.trial("t-0004", finish=False)
        lab.add(LabKind.SCORE_COST, ScoreCost("sc-t-0001", c.GpuUsage(20.0, 0.7, ("r1",))))
        lab.add(LabKind.SCORE_COST, ScoreCost("sc-t-0002", c.GpuUsage(20.0, 0.1, ("r2",))))
        lab.add(LabKind.SCORE_COST, ScoreCost("sc-t-0003", c.GpuUsage(20.0, 50.0, ("r3",))))
        lab.add(LabKind.SCORE_COST, ScoreCost("sc-stray", c.GpuUsage(20.0, 50.0, ("r4",))))
        cb = lab.reader().cost_breakdown(lab.state(), ["t-0001", "t-0002", "t-0004", "t-unknown"], "cand-001")
        self.assertAlmostEqual(cb.outer_propose_usd, 0.31)
        self.assertAlmostEqual(cb.candidate_researcher_usd, 2.5)
        self.assertAlmostEqual(cb.candidate_gpu_usd, 3.25)
        self.assertAlmostEqual(cb.scorer_usd, 0.8)
        none = lab.reader().cost_breakdown(lab.state(), [], None)
        self.assertEqual(none.total_usd, 0.0)
        self.assertEqual(lab.reader().cost_breakdown(lab.state(), [], "cand-404").outer_propose_usd, 0.0)


class NextIdTests(unittest.TestCase):
    def test_sequences_and_widths(self):
        lab = Lab(self)
        r = lab.reader()
        st = lab.state()
        self.assertEqual(r.next_id(st, "t"), "t-0001")
        self.assertEqual(r.next_id(st, "cand"), "cand-001")
        self.assertEqual(r.next_id(st, "ad"), "ad-0001")
        self.assertEqual(r.next_id(st, "a"), "a-0001")
        self.assertEqual(r.next_id(st, "j"), "j-0001")
        self.assertEqual(r.next_id(st, "sc"), "sc-0001")
        self.assertEqual(r.next_id(st, "f"), "f-0001")
        with self.assertRaises(ContractError):
            r.next_id(st, "zz")

    def test_ids_in_the_ledger_are_skipped_after_a_restart(self):
        lab = Lab(self)
        lab.trial("t-0001")
        lab.trial("t-0007", finish=False)
        lab.add(LabKind.CANDIDATE_PROPOSED, candidate(candidate_id="cand-004", version_label="cand-004"))
        lab.add(LabKind.FINAL_REQUESTED, FinalRequest("f-0003", Arm.STOCK, None, None, H64, (Split.TEST,), None, None, NOW))
        lab.add(LabKind.CONTAMINATION_SCANNED, ContaminationReport("t-0001", "j-0012", FileRef("trials/t-0001/train_data/j-0012.jsonl", H64, 5), 10, 0, 0, 13, (H64,), True, NOW))
        st = lab.state()
        fresh = lab.reader()  # a restarted process has issued nothing
        self.assertEqual(fresh.next_id(st, "t"), "t-0008")
        self.assertEqual(fresh.next_id(st, "cand"), "cand-005")
        self.assertEqual(fresh.next_id(st, "f"), "f-0004")
        self.assertEqual(fresh.next_id(st, "j"), "j-0013")
        self.assertEqual(fresh.next_id(st, "sc"), "sc-0001")  # scores are named sc-t-NNNN here, so none match the pattern

    def test_ids_on_disk_are_skipped(self):
        lab = Lab(self)
        for d in ("trials/t-0005", "scores/sc-0009", "artifacts/adapters/ad-0003", "candidates/cand-002", "data/assignments/a-0002", "trials/t-0001/train_data"):
            (lab.layout.root / d).mkdir(parents=True)
        (lab.layout.root / "trials/t-0001/train_data/j-0004.jsonl").write_text("{}\n")
        r, st = lab.reader(), lab.state()
        self.assertEqual(r.next_id(st, "t"), "t-0006")
        self.assertEqual(r.next_id(st, "sc"), "sc-0010")
        self.assertEqual(r.next_id(st, "ad"), "ad-0004")
        self.assertEqual(r.next_id(st, "cand"), "cand-003")
        self.assertEqual(r.next_id(st, "a"), "a-0003")
        self.assertEqual(r.next_id(st, "j"), "j-0005")

    def test_the_final_plans_trial_ids_are_taken(self):
        lab = Lab(self)
        planned = tuple(PlannedTrial(f"t-{50 + i:04d}", arm, "a-held-1", 100 + k) for i, (arm, k) in enumerate((a, k) for a in (Arm.V0, Arm.EVOLVED) for k in range(5)))
        lab.freeze(LabKind.FINAL_ANALYSIS_FROZEN, "final_analysis", lab.layout.final_analysis_path, final_analysis(planned_trials=planned))
        self.assertEqual(lab.reader().next_id(lab.state(), "t"), "t-0060")

    def test_an_id_is_never_issued_twice_by_one_reader(self):
        lab = Lab(self)
        r, st = lab.reader(), lab.state()
        got = [r.next_id(st, "t") for _ in range(3)]
        self.assertEqual(got, ["t-0001", "t-0002", "t-0003"])
        self.assertEqual(r.next_id(st, "t", reserve=False), "t-0004")
        self.assertEqual(r.next_id(st, "t", reserve=False), "t-0004")  # a peek reserves nothing
        self.assertEqual(r.next_id(st, "t", taken=["t-0004", "t-0010"]), "t-0011")

    def test_concurrent_callers_get_distinct_ids(self):
        lab = Lab(self)
        r, st = lab.reader(), lab.state()
        got, lock = [], threading.Lock()

        def go():
            for _ in range(25):
                new = r.next_id(st, "j")
                with lock:
                    got.append(new)

        threads = [threading.Thread(target=go) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(got), 200)
        self.assertEqual(len(set(got)), 200)

    def test_ids_do_not_cross_prefixes(self):
        lab = Lab(self)
        lab.add(LabKind.CANDIDATE_PROPOSED, candidate(candidate_id="cand-009", version_label="cand-009"))
        r, st = lab.reader(), lab.state()
        self.assertEqual(r.next_id(st, "ad"), "ad-0001")
        self.assertEqual(r.next_id(st, "a"), "a-0001")


class LiveLedgerIntegrationTests(unittest.TestCase):
    def test_fold_reader_over_a_labledger_sees_appends_at_once(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        layout = LabLayout(Path(tmp.name) / "lab")
        led = LabLedger.create(layout.master_ledger_dir, clock=Clock(), fsync=False)
        self.addCleanup(led.close)
        reader = FoldReader(layout, ledger=led)
        self.assertEqual(reader.state(), LabState())
        led.append(LabKind.SPREAD_PLANNED, "coordinator", PERSISTED["SpreadPlan"]())
        st = reader.state()
        self.assertEqual(st.spread_plan, PERSISTED["SpreadPlan"]())
        self.assertEqual(st.head, led.anchor())
        # and the same file read cold by a second reader
        self.assertEqual(FoldReader(layout).state().spread_plan, PERSISTED["SpreadPlan"]())


if __name__ == "__main__":
    unittest.main()
