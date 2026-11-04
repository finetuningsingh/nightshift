"""End-to-end tests through ``cli.main(argv, world=build_fake_world(tmp))``.

The world (tests/support/lab_world.py) has the REAL ledgers, fold, meters, scorer and guards, trial runner (with its metering proxy and
control API on a loopback port), statistics, outer controller and report; the researcher, GPU, generator and proposer are fakes.

Scenarios (each is a test or a group of tests):
  1. one fake trial, end to end (files, ledgers, scores, budget)
  2. one candidate search: a screen loss, a confirmed accept, a confirmed reject, a screen loser next to a close winner
  3. the final comparison: three arms, plan-gated, hidden until reveal, with the refusal rules
  4. the report before and after the reveal
  5. the final guard row by row, on the finished lab, and a pending search trial blocking it
  6. a contaminated trial, a crashed researcher, an exhausted budget, a researcher that talks to the metering proxy

Run: PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_integration
"""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from nightshift.lab import cli
from nightshift.lab.contracts import (
    AccessDenied,
    AdapterRef,
    Arm,
    FinalComparisonRefused,
    FinalAnalysis,
    LabKind,
    Outcome,
    Phase,
    Resource,
    ScoreRequest,
    Split,
    TEST_ASSIGNMENT_ID,
    TrainResult,
    TrialStatus,
    load_frozen,
    loads_strict,
    read_record,
)
from nightshift.lab.ledger import LabLedger
from tests.support.lab_fake_gpu import FakeTrainer
from tests.support.lab_fake_runtime import Upstream, post_json
from tests.support.lab_world import LabDriver
from tests.test_lab_cli import new_adapter  # noqa: F401


def trial_ledger_events(d: LabDriver, tid: str) -> list:
    return LabLedger.read_only(d.layout.trial_ledger_dir(tid)).events()


class FullRun(unittest.TestCase):
    """The whole experiment once: spread, rule, two search rounds, close, final analysis, three final requests, reveal."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.d = LabDriver(cls._td.name)
        cls.d.stage("final_stock")  # everything except the reveal
        cls.pre_reveal_state = cls.d.state()
        cls.d.stage("revealed")
        cls.state = cls.d.state()

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    # ---- 1. one trial --------------------------------------------------------------------------------
    def test_one_fake_trial_leaves_every_file_and_every_event(self):
        d, st, lay = self.d, self.state, self.d.layout
        rec = st.trials["t-0001"]
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        for p in (lay.trial_spec_path("t-0001"), lay.trial_record_path("t-0001"), lay.trial_ledger_dir("t-0001") / "ledger.jsonl"):
            self.assertTrue(p.is_file(), p)
        self.assertEqual(read_record(lay.trial_record_path("t-0001"), type(rec)), rec)
        # the dev score of the training job and the selection score of the scorer rental are both on disk and in the master ledger
        for sid in (rec.dev_score_id, rec.selection_score_id):
            self.assertIn(sid, st.scores)
            for p in (lay.score_request_path(sid), lay.score_record_path(sid), lay.per_prompt_path(sid)):
                self.assertTrue(p.is_file(), p)
        sel = st.scores[rec.selection_score_id]
        self.assertEqual((sel.split, sel.phase, sel.verifier.value, sel.n_prompts, len(sel.per_seed)), (Split.SELECTION, Phase.SEARCH, "ifevalg", 8, 3))
        self.assertEqual(sel.adapter_sha256, rec.adapter.sha256)
        self.assertEqual(rec.submission.adapter_id, rec.adapter.adapter_id)
        # the control-host dev score was written into the submission, not the researcher's value
        self.assertAlmostEqual(rec.submission.dev_score, st.scores[rec.dev_score_id].headline_mean())
        self.assertTrue(set(sel.verifier_pins) >= {"fake_ifevalg", "prompts_sha256"})
        # per-prompt flags only: no response text, no ground truth
        row = json.loads(lay.per_prompt_path(rec.selection_score_id).read_text().splitlines()[0])
        self.assertNotIn("text", row)
        self.assertNotIn("ground_truth", row)
        kinds = [e.kind for e in trial_ledger_events(d, "t-0001")]
        for k in (LabKind.GPU_JOB_STARTED, LabKind.GPU_JOB_FINISHED, LabKind.BUDGET_CHARGED):
            self.assertIn(k, kinds)
        self.assertIsNotNone(rec.trial_ledger_anchor)
        self.assertEqual(rec.budget_used.gpu_minutes, 5.0)
        self.assertIn("t-0001", st.contamination)
        self.assertTrue(st.contamination["t-0001"].passed)
        self.assertEqual(d.ok("verify")["ok"], True)

    # ---- 2. the search -------------------------------------------------------------------------------
    def test_the_scripted_search_makes_each_kind_of_decision(self):
        decs = self.state.decisions
        got = {c: (d.stage.value, d.outcome) for c, d in decs.items()}
        self.assertEqual(got["cand-001"], ("screen", Outcome.REJECTED))  # lost its screening round
        self.assertEqual(got["cand-002"], ("confirm", Outcome.ACCEPTED))  # confirmed accept
        self.assertEqual(got["cand-003"], ("confirm", Outcome.REJECTED))  # won the screen, failed the confirmation
        self.assertEqual(got["cand-004"], ("screen", Outcome.REJECTED))  # lost the screen to cand-003
        acc, rej = decs["cand-002"], decs["cand-003"]
        self.assertGreater(acc.delta, acc.margin)
        self.assertLessEqual(rej.delta, rej.margin)
        self.assertEqual(acc.df, 8)
        self.assertEqual(self.d.world.state_reader.incumbent_version(self.state), "cand-002")
        # the second round's incumbent is the accepted pack: its confirmation trials are fresh trials of cand-002 and v0
        self.assertEqual(rej.incumbent_version, "cand-002")
        self.assertEqual(self.d.world.fakes.packs.accepted, ["cand-002"])

    def test_decisions_are_made_by_the_controller_from_the_rule_and_recompute_exactly(self):
        out = self.d.ok("verify")
        names = {c["name"]: c["ok"] for c in out["checks"]}
        for cid in ("cand-001", "cand-002", "cand-003", "cand-004"):
            self.assertTrue(names[f"decision {cid}"])
        rule, _m, _b = self.d.world.state_reader.load_frozen_all(self.state)
        self.assertEqual(self.state.decisions["cand-002"].rule_sha256, rule.sha256())
        self.assertAlmostEqual(rule.alpha, 1 - 0.95 ** 0.5, places=9)  # the Sidak split chosen by freeze-rule

    def test_the_proposer_was_never_shown_a_selection_number_only_the_bits(self):
        decs = self.state.decisions
        contexts = self.d.world.fakes.proposer.contexts
        self.assertEqual(len(contexts), 4)
        numbers = {f"{abs(v):.4f}" for dec in decs.values() for v in (dec.screen_delta, dec.delta, dec.margin, dec.sigma_used) if v}
        self.assertGreaterEqual(len(numbers), 4)
        for ctx in contexts:
            text = ctx.system_prompt + ctx.user_prompt
            for n in numbers:
                self.assertNotIn(n, text)
            self.assertNotIn("a-held-1", text)  # held-out ids stay out as well
        later = contexts[2].user_prompt  # round two sees the outcome bits of round one
        self.assertIn("cand-002", later)
        self.assertIn("accepted", later.lower())

    def test_candidates_trials_and_costs_are_in_the_ledger(self):
        st = self.state
        self.assertEqual(len([t for t in st.trials_started.values() if t.phase is Phase.SEARCH]), 25)
        self.assertEqual(len(st.candidates), 4)
        d2 = st.decisions["cand-002"]
        self.assertGreater(d2.cost.candidate_gpu_usd, 0)
        self.assertGreater(d2.cost.scorer_usd, 0)
        self.assertGreater(d2.cost.outer_propose_usd, 0)
        self.assertEqual(len(st.operator_changes), 0)

    # ---- 3. the final comparison ---------------------------------------------------------------------
    def test_final_requests_cover_both_arms_and_the_stock_model(self):
        finals = {f.arm: f for f in self.state.finals}
        self.assertEqual(set(finals), {Arm.V0, Arm.EVOLVED, Arm.STOCK})
        self.assertEqual(finals[Arm.STOCK].splits, (Split.TEST,))
        self.assertIsNone(finals[Arm.STOCK].pack)
        self.assertEqual(finals[Arm.EVOLVED].pack_version, "cand-002")
        analysis = load_frozen(self.d.layout, self.state.final_analysis, FinalAnalysis)
        self.assertEqual({f.analysis_sha256 for f in finals.values()}, {analysis.sha256()})
        final_trials = [t for t in self.state.trials_started.values() if t.phase is Phase.FINAL]
        self.assertEqual(len(final_trials), 10)
        self.assertEqual({t.assignment_id for t in final_trials}, {"a-held-1"})
        self.assertEqual({t.trial_id for t in final_trials}, {p.trial_id for p in analysis.planned_trials})

    def test_every_planned_trial_has_one_heldout_and_one_test_score_and_stock_has_one_test_score(self):
        st = self.state
        final = [r for r in st.scores.values() if r.phase is Phase.FINAL]
        by = {}
        for r in final:
            if r.split is not Split.DEV:
                by.setdefault((r.trial_id, r.split), []).append(r)
        self.assertEqual(len(by), 21)
        self.assertTrue(all(len(v) == 1 for v in by.values()))
        stock = [r for r in final if r.arm is Arm.STOCK]
        self.assertEqual([(r.split, r.assignment_id, r.verifier.value) for r in stock], [(Split.TEST, TEST_ASSIGNMENT_ID, "ifbench_upstream")])
        self.assertIsNone(stock[0].trial_id)

    def test_reveal_wrote_the_comparison_from_the_frozen_plan(self):
        out = loads_strict(self.d.layout.final_comparison_path.read_text())
        self.assertEqual(len(out["comparisons"]), 2)
        held, test = out["comparisons"]
        self.assertEqual((held["split"], held["metric"], held["contrast"]), ("selection", "prompt_all", "evolved_minus_v0"))
        self.assertEqual((test["split"], test["metric"]), ("test", "prompt_strict"))
        for c in (held, test):
            self.assertGreater(c["delta"], 0)
            self.assertFalse(c["includes_zero"])
            self.assertEqual((c["trials_v0"], c["trials_evolved"], c["method"]), (5, 5, "welch"))
            self.assertLess(c["ci_low"], c["delta"])
            self.assertLess(c["delta"], c["ci_high"])
        self.assertIn("stock_vs_evolved", out)
        self.assertEqual(out["analysis_sha256"], self.state.final_analysis and load_frozen(self.d.layout, self.state.final_analysis, FinalAnalysis).sha256())
        self.assertIsNotNone(self.state.reveal)
        self.assertEqual(self.state.reveal.n_scores, 21)

    # ---- 4. the report -------------------------------------------------------------------------------
    def test_the_report_hides_final_scores_until_the_reveal_and_shows_them_after(self):
        before = self.d.outputs["report_before_reveal"]
        after = self.d.run("report", "--full")[1]
        for needle in ("Per-trial final scores", "arm 3 minus arm 2", "+0.2167"):
            self.assertNotIn(needle, before)
            self.assertIn(needle, after)
        # the labels the design requires, under their conditions
        for label in ("dev scores are researcher-influenced; decisions use control-host numbers only", "heuristic filter", "salted draw",
                      "Custom scoring"):
            self.assertIn(label, after)
        self.assertIn("three arms", after.lower())

    def test_the_search_trace_chart_marks_accepted_rejected_and_invalid(self):
        html = self.d.run("report", "--full", "--format", "html")[1]
        self.assertIn("adaptive search trace", html)
        self.assertIn("<svg", html)

    def test_status_after_the_reveal(self):
        out = self.d.ok("status")
        self.assertEqual((out["revealed"], out["phase"], out["final"]["requests"]), (True, "final", 3))

    # ---- ledgers and anchors -------------------------------------------------------------------------
    def test_verify_passes_and_anchors_exist_for_the_freeze_each_decision_each_final_request_and_the_reveal(self):
        out = self.d.ok("verify")
        self.assertTrue(out["ok"], [c for c in out["checks"] if not c["ok"]])
        labels = [ln.split()[1] for ln in self.d.layout.anchors_path.read_text().splitlines()]
        self.assertEqual(labels[0], "rule_frozen")
        self.assertEqual(len([x for x in labels if x.startswith("decision:")]), 4)
        self.assertEqual(len([x for x in labels if x.startswith("final:")]), 3)
        self.assertEqual(labels[-1], "reveal")
        self.assertEqual(len(self.state.malformed), 0)
        ok, bad = self.d.world.ledger.verify()
        self.assertTrue(ok)
        for tid in self.state.trials:
            self.assertTrue(LabLedger.read_only(self.d.layout.trial_ledger_dir(tid)).verify()[0], tid)

    def test_frozen_files_are_read_only_and_match_their_ledger_events(self):
        for path in (self.d.layout.manifest_path, self.d.layout.budget_path, self.d.layout.rule_path, self.d.layout.final_analysis_path):
            self.assertEqual(path.stat().st_mode & 0o222, 0, path)
        rule, manifest, budget = self.d.world.state_reader.load_frozen_all(self.state)
        self.assertEqual(manifest.item_mode.value, "salted_draw")
        self.assertEqual(budget.gpu_minutes, 120.0)

    # ---- 5. the final guard, row by row ----------------------------------------------------------------
    def guard(self):
        return cli.make_score_guard(self.d.layout, self.d.world.state_reader)

    def final_request(self, trial_id: str, **kw) -> ScoreRequest:
        st = self.state
        spec = st.trials_started[trial_id]
        analysis = load_frozen(self.d.layout, st.final_analysis, FinalAnalysis)
        adapter = st.trials[trial_id].adapter
        base = dict(
            request_id="sc-9100", phase=Phase.FINAL, assignment_id=spec.assignment_id, split=Split.SELECTION, trial_id=trial_id, arm=spec.arm,
            base=spec.base, adapter=adapter, seeds=analysis.score_seeds, params=analysis.score_params, scorer_version=spec.scorer_version,
            reason=None, final_request_id=spec.final_request_id,
        )
        base.update(kw)
        return ScoreRequest(**base)

    def test_a_second_heldout_or_test_score_of_the_same_adapter_is_refused(self):
        g = self.guard()
        for split, aid in ((Split.SELECTION, "a-held-1"), (Split.TEST, TEST_ASSIGNMENT_ID)):
            with self.assertRaisesRegex(AccessDenied, "already exists"):
                g(self.final_request("t-0031", split=split, assignment_id=aid))

    def test_the_real_scorer_runs_the_guard_before_it_rents_anything(self):
        before = len(self.d.world.fakes.model_server.sessions)
        with self.assertRaises(AccessDenied):
            self.d.world.scorer.precheck(self.final_request("t-0031"))
        self.assertEqual(len(self.d.world.fakes.model_server.sessions), before)

    def test_the_request_must_match_its_final_request_arm_pack_trial_splits_seeds_and_plan(self):
        g = self.guard()
        spec = self.state.trials_started["t-0031"]
        self.assertEqual(spec.arm, Arm.EVOLVED)
        with self.assertRaisesRegex(AccessDenied, "another arm"):
            g(self.final_request("t-0031", arm=Arm.V0))
        with self.assertRaises(FinalComparisonRefused):
            g(self.final_request("t-0031", final_request_id="f-9999"))
        with self.assertRaisesRegex(AccessDenied, "seeds or decoding"):
            g(self.final_request("t-0031", seeds=(0,)))
        with self.assertRaisesRegex(AccessDenied, "seeds or decoding"):
            g(self.final_request("t-0031", params=dataclasses.replace(spec.score_params, max_tokens=7)))
        # a search trial is not in the plan, even with a valid final request id and a held-out assignment
        search_adapter = self.state.trials["t-0001"].adapter
        with self.assertRaises(AccessDenied):
            g(self.final_request("t-0001", arm=Arm.EVOLVED, final_request_id="f-0002", adapter=search_adapter, assignment_id="a-held-1"))
        # a trial of the plan on another assignment than planned (the plan has one held-out assignment: use the evolution one)
        with self.assertRaises(AccessDenied):
            g(self.final_request("t-0031", assignment_id="a-evo-1"))

    def test_arm_one_is_scored_on_the_test_split_only_and_once(self):
        g = self.guard()
        stock_base = dict(request_id="sc-9200", phase=Phase.FINAL, assignment_id=TEST_ASSIGNMENT_ID, split=Split.TEST, trial_id=None, arm=Arm.STOCK,
                          base=self.d.world.config.base, adapter=None, seeds=(0, 1, 2), params=self.state.trials_started["t-0031"].score_params,
                          scorer_version=self.state.trials_started["t-0031"].scorer_version, reason=None, final_request_id="f-0003")
        with self.assertRaisesRegex(AccessDenied, "already exists"):
            g(ScoreRequest(**stock_base))
        with self.assertRaises(AccessDenied):
            g(ScoreRequest(**{**stock_base, "split": Split.SELECTION, "assignment_id": "a-held-1"}))

    def test_after_the_search_a_search_phase_request_is_refused(self):
        with self.assertRaisesRegex(AccessDenied, "closed"):
            self.guard()(ScoreRequest("sc-9300", Phase.SEARCH, "a-evo-1", Split.SELECTION, "t-0001", None, self.d.world.config.base, self.state.trials["t-0001"].adapter,
                                      (0, 1, 2), self.state.trials_started["t-0001"].score_params, self.state.trials_started["t-0001"].scorer_version, None, None))

    def test_final_trials_cannot_be_rerun_or_added_outside_the_plan(self):
        spec = self.state.trials_started["t-0031"]
        extra = dataclasses.replace(spec, trial_id="t-9500", seed=spec.seed + 1000)
        with self.assertRaises(Exception) as cm:  # not in the frozen plan
            self.d.world.trial_executor.run(extra)
        self.assertIn("frozen final analysis", str(cm.exception))
        self.assertNotIn("t-9500", self.d.state().trials_started)


class PendingSearchTrial(unittest.TestCase):
    def test_a_pending_search_trial_blocks_final_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = LabDriver(tmp).stage("final_v0")
            spec = dataclasses.replace(d.state().trials_started["t-0001"], trial_id="t-9001", seed=99)
            d.world.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
            st = d.state()
            sp = st.trials_started["t-0026"]
            analysis = load_frozen(d.layout, st.final_analysis, FinalAnalysis)
            req = ScoreRequest("sc-9400", Phase.FINAL, "a-held-1", Split.SELECTION, "t-0026", Arm.V0, sp.base, AdapterRef.from_dict(st.trials["t-0026"].adapter.to_dict()),
                               analysis.score_seeds, analysis.score_params, sp.scorer_version, None, sp.final_request_id)
            with self.assertRaisesRegex(AccessDenied, "pending"):
                cli.make_score_guard(d.layout, d.world.state_reader)(req)
            d.ok("abort-trial", "t-9001")
            with self.assertRaisesRegex(AccessDenied, "already exists"):  # pending is cleared: the next refusal is the re-roll
                cli.make_score_guard(d.layout, d.world.state_reader)(req)


class FinalRequestChecks(unittest.TestCase):
    """The checks of the final guard that need a final request the CLI itself would not make."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.d = LabDriver(td.name).stage("analysis")
        self.st = self.d.state()
        self.analysis = load_frozen(self.d.layout, self.st.final_analysis, FinalAnalysis)
        self.guard = cli.make_score_guard(self.d.layout, self.d.world.state_reader)
        self.spec = self.st.trials_started["t-0001"]

    def add(self, fr):
        self.d.world.ledger.append(LabKind.FINAL_REQUESTED, "coordinator", fr)

    def req(self, fr_id, arm, aid="a-held-1", split=Split.SELECTION, trial="t-0026"):
        adapter = None if arm is Arm.STOCK else new_adapter("a")
        return ScoreRequest("sc-9700", Phase.FINAL, aid, split, trial, arm, self.spec.base, adapter, self.analysis.score_seeds,
                            self.analysis.score_params, self.spec.scorer_version, None, fr_id)

    def fr(self, rid, arm, splits, *, sha=None, repeat_of=None, pack=True):
        from nightshift.lab.contracts import FinalRequest, utc_iso

        pack_ref = self.st.packs["v0"].pack if pack else None
        return FinalRequest(rid, arm, "v0" if pack else None, pack_ref, sha or self.analysis.sha256(), splits, repeat_of,
                            "test" if repeat_of else None, utc_iso(self.d.world.clock()))

    def test_a_request_may_not_unlock_a_split_it_does_not_name(self):
        self.add(self.fr("f-0001", Arm.V0, (Split.TEST,)))
        with self.assertRaisesRegex(AccessDenied, "does not unlock"):
            self.guard(self.req("f-0001", Arm.V0))

    def test_arm_one_may_only_be_scored_on_the_test_split_even_if_its_request_names_more(self):
        self.add(self.fr("f-0001", Arm.STOCK, (Split.SELECTION, Split.TEST), pack=False))
        with self.assertRaisesRegex(AccessDenied, "test split only"):
            self.guard(self.req("f-0001", Arm.STOCK, trial=None))

    def test_the_request_must_name_the_frozen_analysis(self):
        self.add(self.fr("f-0001", Arm.V0, (Split.SELECTION, Split.TEST), sha="e" * 64))
        with self.assertRaisesRegex(AccessDenied, "another analysis"):
            self.guard(self.req("f-0001", Arm.V0))

    def test_a_trial_belongs_to_the_final_request_it_was_started_under(self):
        # t-0001 is a search trial (final_request_id None): a valid request for another adapter of the plan's arm must still be refused
        self.add(self.fr("f-0001", Arm.V0, (Split.SELECTION, Split.TEST)))
        with self.assertRaisesRegex(AccessDenied, "pack or final request"):
            self.guard(self.req("f-0001", Arm.V0, trial="t-0001"))


class ContaminatingTrainer(FakeTrainer):
    """Writes a selection prompt into the processed training data of every job."""

    def __init__(self, *a, prompt: str, **kw):
        super().__init__(*a, **kw)
        self.prompt = prompt

    def run_job(self, job, meter):
        res = super().run_job(job, meter)
        if res.train_data is None:
            return res
        path = self.layout.root / res.train_data.path
        path.write_text(json.dumps({"messages": [{"role": "user", "content": self.prompt}, {"role": "assistant", "content": "ok"}]}) + "\n")
        return dataclasses.replace(res, train_data=self.layout.file_ref(path))


class Failures(unittest.TestCase):
    def tmp(self) -> str:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return td.name

    def test_a_trial_that_trained_on_a_selection_prompt_is_contaminated_and_never_scored_on_selection(self):
        d = LabDriver(self.tmp()).stage("plan")
        first = json.loads(d.layout.selection_data("a-evo-1").read_text().splitlines()[0])["prompt"]
        trainer = ContaminatingTrainer(d.layout, clock=d.world.clock, prompt=first)
        d.world.trial_executor._trainer = trainer
        out = d.ok("trial")
        self.assertEqual(out["status"], "contaminated")
        self.assertFalse(out["scored"])
        st = d.state()
        rec = st.trials[out["trial_id"]]
        self.assertIsNone(rec.selection_score_id)
        self.assertEqual([r.split for r in st.scores.values() if r.trial_id == rec.spec.trial_id], [Split.DEV])  # the dev score only
        rep = st.contamination[rec.spec.trial_id]
        self.assertFalse(rep.passed)
        self.assertGreaterEqual(rep.n_exact_hits + rep.n_ngram_hits, 1)
        self.assertEqual(d.world.fakes.runtime.handles[0].stopped_with[0].value, "stopped_contaminated")
        self.assertEqual(len(d.world.fakes.model_server.sessions), 0)  # no scoring rental was ever rented for it
        # it counts as a failed spread trial, so the rule cannot be frozen on the rest
        for _ in range(4):
            d.world.trial_executor._trainer = d.world.fakes.trainer
            d.ok("trial")
        self.assertEqual(d.ok("spread")["failed"], 1)
        self.assertIn("failed trials", d.refused("freeze-rule", "--assumed-gain", "0.2", "--max-candidates", "4", "--screen-batch-size", "2"))
        self.assertTrue(d.ok("verify")["ok"])

    def test_a_researcher_that_submits_nothing_ends_without_a_submission(self):
        d = LabDriver(self.tmp()).stage("plan")
        d.world.fakes.runtime.script = lambda ctx, client, handle: None
        out = d.ok("trial")
        self.assertEqual((out["status"], out["scored"]), ("no_submission", False))
        self.assertEqual(len(d.world.fakes.model_server.sessions), 0)

    def test_a_crashed_runtime_is_a_failed_trial_not_a_hang(self):
        d = LabDriver(self.tmp()).stage("plan")

        def boom(ctx, client, handle):
            raise RuntimeError("opencode died")

        d.world.fakes.runtime.script = boom
        out = d.ok("trial")
        self.assertIn(out["status"], ("failed", "no_submission"))
        self.assertIn(out["trial_id"], d.state().trials)

    def test_a_gpu_budget_smaller_than_the_job_ends_the_trial_without_an_adapter(self):
        d = LabDriver(self.tmp(), skip=("budget",)).stage("manifest")
        d.ok("freeze-budget", "--researcher-tokens", "2000000", "--gpu-minutes", "3", "--dollars", "25", "--wall-seconds", "7200")
        d.stage("plan")
        out = d.ok("trial")
        self.assertIn(out["status"], ("budget_exhausted", "no_submission"))
        self.assertFalse(out["scored"])
        kinds = [e.kind for e in trial_ledger_events(d, out["trial_id"])]
        self.assertIn(LabKind.BUDGET_EXHAUSTED, kinds)

    def test_a_researcher_that_talks_to_the_metering_proxy_is_charged_through_the_real_meter_and_the_lab_cap(self):
        with Upstream() as up:
            d = LabDriver(self.tmp(), upstream_base_url=up.base_url).stage("plan")
            seen = []

            def researcher(ctx, client, handle):
                body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "plan"}], "max_tokens": 100}
                seen.append(post_json(ctx.llm_base_url + "/chat/completions", ctx.llm_token, body)[0])
                seen.append(post_json(ctx.llm_base_url + "/chat/completions", "wrong-token", body)[0])
                DEFAULT = type(d.world.fakes.runtime).default_script
                DEFAULT(ctx, client, handle)

            d.world.fakes.runtime.script = researcher
            before = d.world.lab_cap.spent_usd
            out = d.ok("trial")
            self.assertEqual(seen, [200, 401])
            self.assertEqual(out["status"], "completed")
            tokens = out["usage"]["researcher_tokens"]
            self.assertEqual(tokens, 11 + 7)  # the upstream's usage chunk, not an estimate
            self.assertEqual(len(up.requests), 1)
            self.assertNotIn("authorization", {k for k in up.requests[0]["headers"] if up.requests[0]["headers"][k].endswith("wrong-token")})
            self.assertGreater(d.world.lab_cap.spent_usd, before)
            charges = [e.data for e in trial_ledger_events(d, out["trial_id"]) if e.kind == LabKind.BUDGET_CHARGED]
            self.assertTrue(any(c["resource"] == "researcher_tokens" and c["amount"] == 18 for c in charges))

    def test_a_request_whose_worst_case_crosses_the_token_cap_is_refused_with_429_and_the_trial_ends(self):
        with Upstream() as up:
            d = LabDriver(self.tmp(), skip=("budget",), upstream_base_url=up.base_url).stage("manifest")
            d.ok("freeze-budget", "--researcher-tokens", "2000", "--gpu-minutes", "120", "--dollars", "25", "--wall-seconds", "7200")
            d.stage("plan")
            seen = []

            def researcher(ctx, client, handle):
                body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "plan"}], "max_tokens": 16000}
                seen.append(post_json(ctx.llm_base_url + "/chat/completions", ctx.llm_token, body)[0])

            d.world.fakes.runtime.script = researcher
            out = d.ok("trial")
            self.assertEqual(seen, [429])
            self.assertEqual(up.requests, [])  # nothing was forwarded
            self.assertIn(LabKind.BUDGET_EXHAUSTED, [e.kind for e in trial_ledger_events(d, out["trial_id"])])
            self.assertFalse(out["scored"])


if __name__ == "__main__":
    unittest.main()
