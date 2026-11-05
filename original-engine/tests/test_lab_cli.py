"""Tests for nightshift.lab.cli: the commands, every order rule, the score guards and ``verify``.

Almost everything runs through ``cli.main(argv, world=...)`` with the world of fakes from ``tests/support/lab_world.py``: the ledgers,
fold, budget meters, scorer, guards, trial runner, statistics and controller are the real ones; the researcher, the GPU, the generator and
the proposer are fakes. Run: PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_cli
"""

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nightshift.lab import cli
from nightshift.lab.contracts import (
    AccessDenied,
    AdapterFile,
    AdapterRef,
    Arm,
    CostBreakdown,
    Decision,
    DecisionStage,
    FinalComparisonRefused,
    GateNotRecorded,
    LabKind,
    LabLayout,
    Outcome,
    Phase,
    Resource,
    ScoreFailure,
    ScoreRequest,
    Split,
    TrialStatus,
    write_record,
)
from tests.support.lab_world import BUDGET_ARGS, RULE_ARGS, LabDriver, build_fake_world, fixture_args

PROJECT = Path(__file__).resolve().parent.parent


class Base(unittest.TestCase):
    def tmp(self) -> str:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return td.name

    def driver(self, stage: str = "init", **kw) -> LabDriver:
        d = LabDriver(self.tmp(), **kw)
        return d.stage(stage)


def search_request(d: LabDriver, adapter: AdapterRef, **kw) -> ScoreRequest:
    state = d.state()
    rule = cli.load_frozen(d.layout, state.rule, cli.Rule) if state.rule is not None else None
    base = dict(
        request_id="sc-9001", phase=Phase.SEARCH, assignment_id="a-evo-1", split=Split.SELECTION, trial_id="t-0001", arm=None,
        base=d.world.config.base, adapter=adapter, seeds=rule.score_seeds if rule else (0, 1, 2), params=rule.score_params if rule else None,
        scorer_version=rule.scorer_version if rule else d.world.scorer.scorer_version, reason=None, final_request_id=None,
    )
    if base["params"] is None:
        base["params"] = state.trials_started["t-0001"].score_params
    base.update(kw)
    return ScoreRequest(**base)


def new_adapter(tag: str = "x") -> AdapterRef:
    return AdapterRef.from_files(f"ad-{tag}", [AdapterFile("adapter_config.json", "1" * 64, 10), AdapterFile("adapter_model.safetensors", (tag * 64)[:64], 20)])


class UsageAndProduction(Base):
    def test_usage_errors_exit_2(self):
        import io

        d = self.driver()
        for argv in ([], [d.root], [d.root, "no-such-command"], [d.root, "freeze-budget"], [d.root, "trial", "--seed", "x"]):
            self.assertEqual(cli.main(argv, world=d.world, out=io.StringIO(), err=io.StringIO()), 2, argv)

    def test_a_world_for_another_root_is_a_usage_error(self):
        import io

        d = self.driver()
        self.assertEqual(cli.main([self.tmp(), "status"], world=d.world, out=io.StringIO(), err=io.StringIO()), 2)

    def test_the_design_commands_all_exist(self):
        design = {"init", "freeze-manifest", "freeze-budget", "record-gate", "commit-pack", "plan-spread", "trial", "spread", "freeze-rule", "outer",
                  "close-search", "freeze-final-analysis", "final", "reveal", "score", "abort-trial", "operator-change", "report", "status", "verify",
                  "verify-isolation", "rentals", "anchor"}
        self.assertEqual(set(cli.COMMANDS), design)
        self.assertEqual(set(cli.COMMAND_FUNCS), design)

    def test_production_path_without_a_world_runs_the_light_commands(self):
        d = self.driver()  # only for a valid LabConfig
        cfg = Path(self.tmp()) / "cfg.json"
        cfg.write_text(json.dumps(d.world.config.to_dict()))
        root = str(Path(self.tmp()) / "prod")

        def run(*argv):
            import io

            out, err = io.StringIO(), io.StringIO()
            return cli.main([root, *argv], out=out, err=err), out.getvalue(), err.getvalue()

        rc, out, err = run("status")
        self.assertEqual(rc, 1)  # not an initialized lab
        self.assertIn("init", err)
        rc, out, err = run("init", "--config", str(cfg))
        self.assertEqual(rc, 0, err)
        lay = LabLayout(root)
        self.assertTrue(lay.lab_config_path.exists())
        self.assertEqual(oct(lay.split_salt_path.stat().st_mode & 0o777), "0o600")
        self.assertEqual(oct(lay.private_dir.stat().st_mode & 0o777), "0o700")
        self.assertEqual(run("init")[0], 1)  # already initialized
        rc, out, err = run("status")
        self.assertEqual((rc, json.loads(out)["initialized"]), (0, True))
        rc, out, _ = run("verify")
        self.assertEqual((rc, json.loads(out)["ok"]), (0, True))
        self.assertEqual(run("anchor")[0], 0)
        self.assertTrue(lay.anchors_path.read_text().startswith("1:"))
        rc, out, _ = run("report")
        self.assertEqual(rc, 0)
        self.assertIn("bits and costs", out)
        self.assertEqual(run("report", "--full")[0], 1)  # refused before lab.search_closed

    def test_init_without_a_config_is_refused(self):
        import io

        err = io.StringIO()
        self.assertEqual(cli.main([str(Path(self.tmp()) / "nolab"), "init"], out=io.StringIO(), err=err), 1)
        self.assertIn("lab.json", err.getvalue())

    def test_python_dash_m_entry_point(self):
        r = subprocess.run([sys.executable, "-m", "nightshift.lab", str(Path(self.tmp()) / "x"), "status"], cwd=PROJECT, capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("not an initialized lab", r.stderr)
        r = subprocess.run([sys.executable, "-m", "nightshift.lab"], cwd=PROJECT, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(r.returncode, 2)

    def test_build_world_constructs_the_real_objects_without_touching_anything(self):
        from nightshift.lab.contracts import LabWorld
        from nightshift.lab.ledger import LabLedger
        from nightshift.lab import stats as stats_mod

        d = self.driver()
        cfg = Path(self.tmp()) / "cfg.json"
        cfg.write_text(json.dumps(d.world.config.to_dict()))
        root = Path(self.tmp()) / "prod"
        import io

        self.assertEqual(cli.main([str(root), "init", "--config", str(cfg)], out=io.StringIO(), err=io.StringIO()), 0)
        led = LabLedger.open(root)
        self.addCleanup(led.close)
        w = cli.build_world(LabLayout(root), d.world.config, ledger=led)
        self.assertIsInstance(w, LabWorld)
        names = {k: type(getattr(w, k)).__name__ for k in ("scorer", "trainer", "model_server", "runtime", "packs", "trial_executor", "proposal_source", "state_reader")}
        self.assertEqual(names, {"scorer": "LazyScorer", "trainer": "GlimmerTrainer", "model_server": "RentalModelServer", "runtime": "CoralRuntime",
                                 "packs": "PackRepo", "trial_executor": "TrialRunner", "proposal_source": "_LazyProposalSource", "state_reader": "FoldReader"})
        self.assertIs(w.decision_rule, stats_mod)
        self.assertEqual(w.lab_cap.cap_usd, d.world.config.lab_budget_usd)
        # nothing was built that needs a frozen manifest, a checkout or a key
        with self.assertRaises(Exception):
            w.scorer.scorer_version


class WonderGate(Base):
    def _gate_limit(self, source) -> float:
        key = "WONDER_GATE_TEST_KEY"
        source._config = dataclasses.replace(source._config, secret_env={**source._config.secret_env, "meta_api_key": key})
        self.addCleanup(source.close)
        with mock.patch.dict(os.environ, {key: "test-key-1234"}), mock.patch("nightshift.lab.proposer.WonderProposalSource") as wonder:
            wonder.return_value.propose.return_value = "ok"
            self.assertEqual(source.propose(None), "ok")
        (client,) = wonder.call_args.args
        return client._gate.limits.max_llm_usd

    def _source(self):
        w = build_fake_world(self.tmp())
        return cli._LazyProposalSource(w.layout, w.config, lambda: 1000.0)

    def test_the_wonder_gate_follows_the_per_call_cap_instead_of_the_five_dollar_default(self):
        src = self._source()
        src.set_call_cap(0.01, 4)
        self.assertAlmostEqual(self._gate_limit(src), 0.04)
        self.assertEqual(self._gate_limit(self._source()), 5.0)  # untouched: the old default

    def test_the_cap_cannot_change_once_the_gate_exists(self):
        src = self._source()
        self._gate_limit(src)
        with self.assertRaises(cli.LabError):
            src.set_call_cap(1.0, 1)


class OrderRules(Base):
    def test_the_early_commands_need_init_and_each_other(self):
        d = self.driver("init", skip=())
        self.assertIn("already initialized", d.refused("init"))
        e = LabDriver(self.tmp())
        e.refused("freeze-budget", *BUDGET_ARGS)  # before init
        e.refused("freeze-manifest", *fixture_args(e.world))
        e.stage("budget")
        self.assertIn("frozen", e.refused("freeze-budget", *BUDGET_ARGS))  # freeze once
        e.refused("plan-spread", "--trials", "5")  # no manifest
        e.stage("manifest")
        self.assertIn("commit v0", e.refused("plan-spread", "--trials", "5"))
        e.stage("pack")
        self.assertIn("v0", e.refused("commit-pack"))
        e.stage("plan")
        e.refused("plan-spread", "--trials", "5")  # one plan

    def test_a_spread_trial_needs_a_declared_plan_and_the_spread_needs_the_whole_plan(self):
        d = self.driver("pack")
        self.assertIn("plan-spread precedes", d.refused("trial"))
        d.stage("plan")
        self.assertIn("not committed", d.refused("trial", "--pack-version", "cand-001"))
        for _ in range(3):
            d.ok("trial")
        msg = d.refused("spread")
        self.assertIn("5", msg)
        self.assertIn("3", msg)  # the plan has 5 trials; 3 were started
        for _ in range(2):
            d.ok("trial")
        self.assertEqual(d.ok("spread")["completed"], 5)
        self.assertIn("measured", d.refused("spread"))

    def test_a_refused_trial_spends_nothing(self):
        d = self.driver("plan")
        before = len(d.world.fakes.trainer.jobs)
        with mock.patch("nightshift.lab.cli.check_trial_against_frozen", side_effect=cli.ContractError("TrialSpec.budget is not the frozen budget")):
            self.assertIn("frozen budget", d.refused("trial"))
        self.assertEqual((len(d.world.fakes.trainer.jobs), len(d.state().trials_started)), (before, 0))

    def test_freeze_rule_refusals(self):
        d = self.driver("plan")
        self.assertIn("spread", d.refused("freeze-rule", *RULE_ARGS))  # no spread yet
        d.stage("spread")
        msg = d.refused("freeze-rule", "--assumed-gain", "0.01", "--max-candidates", "4", "--screen-batch-size", "2")
        self.assertIn("cannot detect the gain", msg)  # minimum detectable gain exceeds the assumed gain
        msg = d.refused("freeze-rule", "--assumed-gain", "0.2", "--max-candidates", "40", "--screen-batch-size", "2")
        self.assertIn("half", msg)  # more than half of the affordable trials
        d.refused("freeze-rule", *RULE_ARGS, "--assignments", "a-held-1")  # a held-out assignment cannot be a rule assignment
        d.refused("freeze-rule", *RULE_ARGS, "--confirm-trials", "1")  # too few confirmation trials
        self.assertIsNone(d.state().rule)
        d.stage("rule")
        self.assertIn("already", d.refused("freeze-rule", *RULE_ARGS))
        self.assertEqual(os.stat(d.layout.rule_path).st_mode & 0o222, 0)  # read-only after the freeze

    def test_alpha_is_split_over_the_rounds_unless_given(self):
        d = self.driver("spread")
        out = d.ok("freeze-rule", *RULE_ARGS)  # 2 rounds: family alpha 0.05 split
        self.assertAlmostEqual(out["alpha"], 1 - 0.95 ** 0.5, places=9)
        self.assertLess(out["family_false_accept_bound"], 0.0501)
        e = self.driver("spread")
        self.assertEqual(e.ok("freeze-rule", "--assumed-gain", "0.2", "--max-candidates", "2", "--screen-batch-size", "2")["alpha"], 0.05)  # one round
        f = self.driver("spread")
        self.assertEqual(f.ok("freeze-rule", *RULE_ARGS, "--alpha", "0.05")["alpha"], 0.05)  # explicit wins

    def test_a_failed_spread_trial_blocks_the_rule(self):
        d = self.driver("plan")
        for _ in range(4):
            d.ok("trial")
        d.world.fakes.runtime.script = lambda ctx, client, handle: None  # a researcher that submits nothing
        rec = d.ok("trial")
        self.assertEqual(rec["status"], "no_submission")
        out = d.ok("spread")
        self.assertEqual((out["completed"], out["failed"]), (4, 1))
        self.assertIn("failed trials", d.refused("freeze-rule", *RULE_ARGS))

    def test_a_failed_spread_trial_may_count_as_zero_and_then_the_rule_must_score_failures_as_zero(self):
        d = self.driver("plan")
        for _ in range(4):
            d.ok("trial")
        d.world.fakes.runtime.script = lambda ctx, client, handle: None  # a researcher that submits nothing
        self.assertEqual(d.ok("trial")["status"], "no_submission")
        out = d.ok("spread", "--failed-as-zero")
        self.assertEqual((out["completed"], out["failed"]), (5, 0))
        meas = d.state().spread
        zeros = [t for t in meas.trials if t.score_id is None]
        self.assertEqual([t.score for t in zeros], [0.0])
        self.assertIn("scored_zero", d.refused("freeze-rule", *RULE_ARGS, "--failed-trial-policy", "invalidates_candidate"))
        msg = d.refused("freeze-rule", *RULE_ARGS, "--failed-trial-policy", "scored_zero")
        self.assertNotIn("scored_zero", msg)  # the policy guard is satisfied; a zero widens the spread, so power is what refuses here
        self.assertIn("minimum detectable gain", msg)

    def test_outer_needs_the_rule_the_terms_gate_and_lab_dollars(self):
        d = self.driver("spread")
        self.assertIn("rule", d.refused("outer"))
        e = self.driver("rule", skip=("gate",))
        self.assertEqual(e.state().gates, [])
        msg = e.refused("outer", "--trial-llm-usd", "2")
        self.assertIn("terms", msg.lower())
        self.assertEqual(e.state().candidates, {})
        out = e.ok("record-gate", "--outcome", "proceed_with_risk", "--statement", "operator accepts the risk", "--decided-by", "owner")
        self.assertEqual(out["gate_status"], "open_waived")  # the answer to the record itself says the gate is still open
        self.assertEqual(e.ok("status")["terms_gate_status"], "open_waived")  # a waiver unblocks the work and the gate stays open
        e.world.fakes.proposer._script.append(AssertionError("must not be reached"))
        # the whole-trial dollar cap (25) times the trials of one round exceeds the 200 dollar lab cap: refused before any proposal
        self.assertIn("lab", e.refused("outer").lower())
        self.assertEqual((e.state().candidates, len(e.world.fakes.proposer.contexts)), ({}, 0))

    def test_outer_hands_wonder_usd_to_the_production_proposal_source(self):
        d = self.driver("search")
        calls = []
        d.world.fakes.proposer.set_call_cap = lambda usd, n: calls.append((usd, n))
        d.ok("outer", "--wonder-usd", "0.01", "--trial-llm-usd", "2")
        rule = cli.load_frozen(d.layout, d.state().rule, cli.Rule)
        self.assertEqual(calls, [(0.01, rule.max_candidates)])

    def test_close_search_refusals(self):
        d = self.driver("plan")
        self.assertIn("rule", d.refused("close-search", "--reason", "x"))
        e = self.driver("rule")
        spec = dataclasses.replace(e.state().trials_started["t-0001"], trial_id="t-9001", seed=99)
        e.world.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
        self.assertIn("t-9001", e.refused("close-search", "--reason", "x"))
        e.ok("abort-trial", "t-9001")
        e.ok("close-search", "--reason", "x")
        self.assertIn("closed", e.refused("close-search", "--reason", "again"))

    def test_an_exploratory_final_pack_needs_a_recorded_operator_change(self):
        d = self.driver("search")
        d.ok("outer", "--rounds", "1", "--trial-llm-usd", "2")
        cands = [c for c in d.state().candidates]
        self.assertTrue(cands)
        cand = cands[0]
        if cand not in d.state().packs:
            self.skipTest("the scripted candidate was not committed")
        if any(t not in d.state().trials for t in d.state().trials_started):
            self.skipTest("a scripted search trial is still pending")
        self.assertIn("operator-change", d.refused("close-search", "--reason", "deadline", "--final-pack", cand))
        self.assertIn("not committed", d.refused("close-search", "--reason", "deadline", "--final-pack", "cand-999"))
        d.ok("operator-change", "--target", "final_pack", "--detail", f"owner-approved exploratory held-out run of {cand}")
        out = d.ok("close-search", "--reason", "deadline", "--final-pack", cand)
        self.assertEqual(out["incumbent_version"], cand)
        self.assertIn("exploratory final pack", d.state().search_closed.reason)

    def test_final_analysis_and_final_order_rules(self):
        d = self.driver("search")
        self.assertIn("close-search", d.refused("freeze-final-analysis", "--trials-per-pack", "5"))
        self.assertIn("lab.search_closed", d.refused("final", "--arm", "v0"))
        d.stage("closed")
        d.refused("final", "--arm", "v0")  # closed but no frozen analysis
        self.assertIn("at least", d.refused("freeze-final-analysis", "--trials-per-pack", "4"))
        d.refused("freeze-final-analysis", "--trials-per-pack", "5", "--assignments", "a-evo-1")  # an evolution assignment is not held out
        self.assertIsNone(d.state().final_analysis)
        d.stage("analysis")
        self.assertIn("already", d.refused("freeze-final-analysis", "--trials-per-pack", "5"))

    def test_planned_search_plus_final_trials_may_not_exceed_the_affordable_trials(self):
        d = self.driver("closed", affordable_trials=52)  # the rule plans 25 (<= 26); 25 + 2 x 14 = 53 > 52
        self.assertIn("affordable", d.refused("freeze-final-analysis", "--trials-per-pack", "14"))
        d.ok("freeze-final-analysis", "--trials-per-pack", "13")  # 25 + 26 = 51

    def test_final_refuses_a_second_request_for_the_same_pack_commit(self):
        d = self.driver("final_v0")
        self.assertIn("exists", d.refused("final", "--arm", "v0"))
        self.assertIn("--reason", d.refused("final", "--arm", "v0", "--repeat-of", "f-0001"))
        self.assertIn("unknown", d.refused("final", "--arm", "v0", "--repeat-of", "f-7777", "--reason", "why"))
        self.assertEqual(len([f for f in d.state().finals if f.arm is Arm.V0]), 1)
        d.ok("final", "--arm", "stock")
        d.refused("final", "--arm", "stock")

    def test_final_resume_runs_only_the_planned_trials_that_have_not_started(self):
        d = self.driver("analysis")
        real = d.world.trial_executor.run
        calls = {"n": 0}

        def flaky(spec):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("the coordinator died")
            return real(spec)

        with mock.patch.object(d.world.trial_executor, "run", side_effect=flaky):
            rc, _out, err = d.run("final", "--arm", "v0")
        self.assertEqual(rc, 3)
        self.assertIn("coordinator died", err)
        started = [t for t, s in d.state().trials_started.items() if s.phase is Phase.FINAL]
        self.assertEqual(len(started), 2)
        out = d.ok("final", "--arm", "v0", "--resume", "f-0001")
        self.assertEqual(len(out["trials"]), 3)
        started = [t for t, s in d.state().trials_started.items() if s.phase is Phase.FINAL]
        self.assertEqual(len(started), 5)
        self.assertEqual(len([f for f in d.state().finals if f.arm is Arm.V0]), 1)
        self.assertEqual(d.run("final", "--arm", "evolved", "--resume", "f-0001")[0], 2)  # a request of another arm

    def test_final_resume_refuses_a_started_trial_that_never_finished_instead_of_skipping_it(self):
        d = self.driver("analysis")
        real = d.world.trial_executor.run
        calls = {"n": 0}

        def dies_midway(spec):
            calls["n"] += 1
            if calls["n"] == 3:  # the coordinator dies after lab.trial_started, before lab.trial_finished
                d.world.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
                raise RuntimeError("the coordinator died")
            return real(spec)

        with mock.patch.object(d.world.trial_executor, "run", side_effect=dies_midway):
            self.assertEqual(d.run("final", "--arm", "v0")[0], 3)
        st = d.state()
        stuck = [t for t in st.trials_started if t not in st.trials]
        self.assertEqual(len(stuck), 1)
        with mock.patch.object(d.world.trial_executor, "run", side_effect=AssertionError("must not run")) as ex:
            err = d.refused("final", "--arm", "v0", "--resume", "f-0001")
        self.assertIn(stuck[0], err)
        self.assertIn("abort-trial", err)
        ex.assert_not_called()
        d.ok("abort-trial", stuck[0])
        d.ok("final", "--arm", "v0", "--resume", "f-0001")  # now the remaining planned trials run

    def test_final_repeat_is_refused_when_its_planned_trial_ids_were_used(self):
        d = self.driver("final_v0")
        with mock.patch.object(d.world.trial_executor, "run", side_effect=AssertionError("must not run")):
            self.assertIn("used once", d.refused("final", "--arm", "v0", "--repeat-of", "f-0001", "--reason", "scorer crashed"))
        self.assertEqual(len([f for f in d.state().finals if f.arm is Arm.V0]), 1)  # no second request was recorded

    def test_reveal_refuses_until_every_planned_trial_is_scored_and_only_once(self):
        d = self.driver("final_v0")
        self.assertIn("missing", d.refused("reveal"))
        self.assertFalse(d.layout.final_comparison_path.exists())
        self.assertIsNone(d.state().reveal)
        d.stage("final_stock")
        d.ok("reveal")
        self.assertTrue(d.layout.final_comparison_path.exists())
        self.assertIn("already", d.refused("reveal"))

    def test_an_exploratory_final_analysis_is_a_recorded_two_per_pack_held_out_only_comparison(self):
        d = self.driver("search")
        d.stage("closed")
        self.assertIn("operator-change", d.refused("freeze-final-analysis", "--trials-per-pack", "2", "--exploratory"))
        self.assertIn("at least 5", d.refused("freeze-final-analysis", "--trials-per-pack", "2"))  # the confirmatory floor is unchanged
        d.ok("operator-change", "--target", "final_analysis_exploratory", "--detail", "owner-approved Tier B 2 + 2 on held-out, exploratory")
        d.ok("freeze-final-analysis", "--trials-per-pack", "2", "--exploratory")
        self.assertTrue(d.state().final_analysis is not None)
        self.assertIn("no stock", d.refused("final", "--arm", "stock"))
        d.ok("final", "--arm", "v0")
        self.assertIn("missing", d.refused("reveal"))
        d.ok("final", "--arm", "evolved")
        d.ok("reveal")
        finals = [s for s in d.state().trials_started.values() if s.phase is Phase.FINAL]
        self.assertEqual(len(finals), 4)
        test_scores = [r for r in d.state().scores.values() if r.phase is Phase.FINAL and r.split is Split.TEST]
        self.assertEqual(test_scores, [])

    def test_the_diagnostic_score_command_carries_the_trials_arm_into_the_final_guard(self):
        d = self.driver("final_v0")
        st = d.state()
        tid, spec = next((t, s) for t, s in st.trials_started.items() if s.phase is Phase.FINAL)
        self.assertIs(spec.arm, Arm.V0)
        adapter = st.trials[tid].adapter.adapter_id
        err = d.refused("score", "--adapter", adapter, "--trial", tid, "--final-request", spec.final_request_id, "--split", "test")
        self.assertNotIn("another arm", err)  # it gets past the arm check and stops at the no-reroll rule: this score already exists
        self.assertIn("already exists", err)

    def test_the_final_guard_binds_the_adapter_to_the_planned_trials_own_output(self):
        d = self.driver("final_v0")
        st = d.state()
        tid, spec = next((t, s) for t, s in st.trials_started.items() if s.phase is Phase.FINAL)
        analysis = cli.load_frozen(d.layout, st.final_analysis, cli.FinalAnalysis)
        guard = cli.make_score_guard(d.layout, d.world.state_reader)
        own = st.trials[tid].adapter
        req = ScoreRequest("sc-9001", Phase.FINAL, cli.TEST_ASSIGNMENT_ID, Split.TEST, tid, Arm.V0, d.world.config.base, new_adapter("7"), analysis.score_seeds,
                           analysis.score_params, spec.scorer_version, None, spec.final_request_id)
        with self.assertRaisesRegex(AccessDenied, "not the output of trial"):  # a new adapter hash would otherwise be a fresh no-reroll key
            guard(req)
        with self.assertRaisesRegex(AccessDenied, "already exists"):  # the trial's own adapter passes the binding and meets the no-reroll rule
            guard(dataclasses.replace(req, adapter=own))
        # LIVE 2026-11-04: a running final trial's job adapter is dev-scored before the trial names an adapter; DEV is not bound
        dev = dataclasses.replace(req, request_id="sc-9002", split=Split.DEV, assignment_id=spec.assignment_id, arm=None)
        guard(dev)  # refused before the fix with "the adapter is not the output of trial"


        self.assertIn("search_closed", self.driver("search").refused("reveal"))

    def test_abort_trial_rebuilds_usage_from_the_trial_ledger(self):
        d = self.driver("rule")
        spec = dataclasses.replace(d.state().trials_started["t-0001"], trial_id="t-9001", seed=99)
        d.world.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
        tled = d.world.new_trial_ledger("t-9001")
        meter = d.world.meter_factory(spec, tled)
        meter.charge(Resource.GPU_MINUTES, 7.0, "gpu:r:1")
        meter.charge(Resource.GPU_USD, 0.5, "gpu:r:1")
        tled.close()
        ctx = "ab" * 32
        (d.layout.trial_dir("t-9001") / "initial_context.sha256").write_text(ctx + "\n")
        out = d.ok("abort-trial", "t-9001")
        self.assertEqual(out["usage"]["gpu_minutes"], 7.0)
        rec = d.state().trials["t-9001"]
        self.assertEqual((rec.status, rec.initial_context_sha256, rec.budget_used.gpu_usd), (TrialStatus.ABORTED, ctx, 0.5))
        self.assertIn("finished", d.refused("abort-trial", "t-9001"))
        self.assertIn("no lab.trial_started", d.refused("abort-trial", "t-9999"))
        self.assertIn("finished", d.refused("abort-trial", "t-0001"))  # a finished trial cannot be aborted

    def test_abort_without_a_context_file_records_null(self):
        d = self.driver("rule")
        spec = dataclasses.replace(d.state().trials_started["t-0001"], trial_id="t-9002", seed=98)
        d.world.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
        d.ok("abort-trial", "t-9002")
        self.assertIsNone(d.state().trials["t-9002"].initial_context_sha256)

    def test_status_lists_a_trial_that_has_no_lab_trial_finished(self):
        d = self.driver("rule")
        self.assertEqual(d.ok("status")["trials"]["pending"], [])
        spec = dataclasses.replace(d.state().trials_started["t-0001"], trial_id="t-9003", seed=97)
        d.world.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
        out = d.ok("status")
        self.assertEqual(out["trials"]["pending"], ["t-9003"])
        self.assertIn("abort-trial", out["note"])

    def test_selection_numbers_are_never_printed_by_the_diagnostic_score_command(self):
        from unittest import mock

        from nightshift.lab.contracts import GpuUsage

        d = self.driver("rule")
        rec = d.state().scores[d.state().trials["t-0001"].selection_score_id]
        fake = dataclasses.replace(rec, score_id="sc-8000", request_id="sc-8000")
        with mock.patch("nightshift.lab.score_session.score_adapter", return_value=(fake, GpuUsage(1.0, 0.02, ("r",)))):
            rc, out, err = d.run("score", "--adapter", d.state().trials["t-0002"].adapter.adapter_id, "--split", "selection", "--assignment", "a-evo-1", "--trial", "t-0002")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("headline_mean", out)
        self.assertNotIn(f"{rec.headline_mean():.3f}"[:5], out)
        self.assertEqual(json.loads(out)["split"], "selection")

    def test_a_frozen_file_is_never_rewritten_even_when_the_second_freeze_is_refused(self):
        d = self.driver("manifest")
        import io

        env = cli._Env(d.layout, d.world, io.StringIO(), io.StringIO(), "freeze-budget")
        spec = cli.BudgetSpec(1, 1.0, 1.0, 1.0)
        env.freeze(LabKind.BUDGET_FROZEN, "budget", d.layout.budget_path, spec)
        before = d.layout.budget_path.read_bytes()
        with self.assertRaises(cli.LedgerError):
            env.freeze(LabKind.BUDGET_FROZEN, "budget", d.layout.budget_path, cli.BudgetSpec(2, 2.0, 2.0, 2.0))
        self.assertEqual(d.layout.budget_path.read_bytes(), before)

    def test_a_file_written_before_a_crash_but_never_named_by_the_ledger_can_be_frozen_again(self):
        d = self.driver("manifest")
        import io

        write_record(d.layout.budget_path, cli.BudgetSpec(9, 9.0, 9.0, 9.0))  # the crash happened after this write and before lab.budget_frozen
        self.assertIsNone(d.state().budget)
        spec = cli.BudgetSpec(1, 1.0, 1.0, 1.0)
        env = cli._Env(d.layout, d.world, io.StringIO(), io.StringIO(), "freeze-budget")
        env.freeze(LabKind.BUDGET_FROZEN, "budget", d.layout.budget_path, spec)
        self.assertIsNotNone(d.state().budget)
        self.assertEqual(cli.load_frozen(d.layout, d.state().budget, cli.BudgetSpec), spec)

    def test_rentals_and_anchor(self):
        d = self.driver("init")
        self.assertIsNone(d.ok("rentals", "cost")["hard_stop_usd"])
        self.assertEqual(d.run("rentals", "init")[0], 2)  # needs --hard-stop-usd and --why
        d.ok("rentals", "init", "--hard-stop-usd", "40", "--why", "unit test")
        self.assertEqual(d.ok("rentals", "cost")["hard_stop_usd"], 40)
        a1 = d.ok("anchor", "--label", "one")["anchor"]
        a2 = d.ok("anchor", "--label", "two")["anchor"]
        lines = d.layout.anchors_path.read_text().splitlines()
        self.assertEqual([ln.split()[0] for ln in lines], [a1, a2])
        self.assertEqual(a1, a2)  # no event in between: the head is the same
        d.ok("record-gate", "--outcome", "clarified", "--statement", "s", "--decided-by", "o")
        self.assertNotEqual(d.ok("anchor")["anchor"], a1)

    def test_anchor_push_commits_only_the_anchors_file(self):
        d = self.driver("init")
        remote = Path(self.tmp()) / "remote.git"
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        run = lambda *a, cwd: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, env=env)  # noqa: E731
        run("init", "--bare", "-b", "main", str(remote), cwd=self.tmp())
        a = d.layout.anchors_path.parent
        a.mkdir(parents=True, exist_ok=True)
        run("init", "-b", "main", cwd=a)
        run("remote", "add", "origin", str(remote), cwd=a)
        (a / "other.txt").write_text("not mine")
        with mock.patch.dict(os.environ, env):
            out = d.ok("anchor", "--push")
        self.assertTrue(out.get("pushed"), out)
        shown = subprocess.run(["git", "--git-dir", str(remote), "show", "--stat", "--format=%s", "main"], capture_output=True, text=True).stdout
        self.assertIn("master.anchors", shown)
        self.assertNotIn("other.txt", shown)


    def _git_anchor_repo(self, d):
        remote = Path(self.tmp()) / "remote.git"
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        run = lambda *a, cwd: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, env=env)  # noqa: E731
        run("init", "--bare", "-b", "main", str(remote), cwd=self.tmp())
        a = d.layout.anchors_path.parent
        a.mkdir(parents=True, exist_ok=True)
        run("init", "-b", "main", cwd=a)
        run("remote", "add", "origin", str(remote), cwd=a)
        return remote, env

    def test_the_rule_freeze_milestone_commits_and_pushes_its_anchor(self):
        d = self.driver("spread")
        remote, env = self._git_anchor_repo(d)
        with mock.patch.dict(os.environ, env):
            out = d.ok("freeze-rule", *RULE_ARGS)
        log = subprocess.run(["git", "--git-dir", str(remote), "show", "--format=%s", "main:master.anchors"], capture_output=True, text=True)
        self.assertEqual(log.returncode, 0, log.stderr)
        self.assertIn(out["anchor"] + " rule_frozen", log.stdout)

    def test_a_milestone_whose_push_fails_still_succeeds_and_says_so(self):
        d = self.driver("spread")  # the anchors directory is not a git work tree
        rc, out, err = d.run("freeze-rule", *RULE_ARGS)
        self.assertEqual(rc, 0, err)
        self.assertIsNotNone(d.state().rule)
        self.assertIn("only local, not pushed", err)
        self.assertIn(json.loads(out)["anchor"], err)


class AfterSearch(Base):
    """One scripted search (cand-001 loses its screening, cand-002 is confirmed, cand-003 fails its confirmation, cand-004 loses the screen),
    shared by the tests that only read it."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.d = LabDriver(cls._td.name).stage("search")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def driver(self, stage="init", **kw):
        return self.d

    def test_the_rule_is_frozen_before_candidates_and_budget_and_manifest_are_frozen_before_them(self):
        d = self.driver("search")
        self.assertEqual(len(d.state().candidates), 4)
        for argv in (["freeze-budget", *BUDGET_ARGS], ["freeze-manifest", *fixture_args(d.world)]):
            d.refused(*argv)
        d.refused("freeze-rule", *RULE_ARGS)

    def test_candidates_are_counted_from_the_ledger_so_outer_cannot_exceed_the_maximum(self):
        d = self.driver("search")
        out = d.ok("outer", "--rounds", "3", "--trial-llm-usd", "2")
        self.assertEqual(out["decisions"], [])
        self.assertEqual(len(d.state().candidates), 4)

    def test_report_full_is_refused_during_the_search_and_the_default_view_shows_no_numbers(self):
        d = self.driver("search")
        d.refused("report", "--full")
        status, report = d.search_outputs
        decisions = d.state().decisions
        numbers = {f"{abs(v):.4f}" for dec in decisions.values() for v in (dec.screen_delta, dec.delta, dec.margin, dec.sigma_used) if v}
        self.assertTrue(numbers)
        text = json.dumps(status) + report
        for n in numbers:
            self.assertNotIn(n, text)
        for word in ("sigma", "margin", "delta"):
            self.assertNotIn(word, json.dumps(status))
        self.assertEqual({c["outcome"] for c in status["candidates"]}, {"accepted", "rejected"})
        self.assertEqual({c["id"]: c["outcome"] for c in status["candidates"]},
                         {"cand-001": "rejected", "cand-002": "accepted", "cand-003": "rejected", "cand-004": "rejected"})

    def test_operator_change_counts_the_candidates_so_far(self):
        d = self.driver("search")
        d.ok("operator-change", "--target", "lint_ranges", "--detail", "widened the heartbeat range")
        [oc] = d.state().operator_changes
        self.assertEqual((oc.target, oc.candidates_so_far), ("lint_ranges", 4))

    def test_verify_passes_on_a_clean_lab_and_names_every_check(self):
        d = self.driver("search")
        out = d.ok("verify")
        self.assertTrue(out["ok"])
        names = " ".join(c["name"] for c in out["checks"])
        for needle in ("master ledger hash chain", "trial ledger t-0001", "trial t-0001 anchor", "decision cand-002", "anchors match the ledger",
                       "rule frozen before the first candidate", "frozen files match the ledger", "no malformed events"):
            self.assertIn(needle, names)

    def test_status_shows_the_incumbent_and_no_pending_trials(self):
        out = self.d.ok("status")
        self.assertEqual((out["incumbent"], out["phase"], out["trials"]["pending"]), ("cand-002", "search", []))

class AfterClose(Base):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.d = LabDriver(cls._td.name).stage("closed")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_the_full_report_shows_what_the_default_view_hid(self):
        decisions = self.d.state().decisions
        numbers = {f"{abs(v):.4f}" for dec in decisions.values() for v in (dec.screen_delta, dec.delta, dec.margin, dec.sigma_used) if v}
        rc, full, err = self.d.run("report", "--full")
        self.assertEqual(rc, 0, err)
        for n in numbers:
            self.assertIn(n, full)
        default = self.d.run("report")[1]
        self.assertNotIn("0.0880", default)  # the default view stays bit-only even after the search is closed
        self.assertNotIn("Per-trial final scores", full)  # and nothing final is shown before the reveal

    def test_the_html_report_and_writing_under_reports(self):
        out = self.d.ok("report", "--full", "--format", "html", "--out", "full.html")
        self.assertEqual(out["written"], "reports/full.html")
        self.assertIn("<svg", self.d.layout.reports_dir.joinpath("full.html").read_text())
        self.assertEqual(self.d.run("report", "--out", "../escape.md")[0], 2)

    def test_searching_is_over_so_a_search_phase_score_is_refused_whatever_the_request_says(self):
        d = self.d
        adapter = d.state().trials["t-0001"].adapter
        guard = cli.make_score_guard(d.layout, d.world.state_reader)
        with self.assertRaisesRegex(AccessDenied, "closed"):
            guard(search_request(d, new_adapter("9")))
        rc, _o, err = d.run("score", "--adapter", adapter.adapter_id, "--split", "dev", "--assignment", "a-evo-1")
        self.assertEqual(rc, 2)  # after the close a score needs --final-request


class ScoreGuards(Base):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.d = LabDriver(cls._td.name).stage("rule")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def guard(self, req):
        return cli.make_score_guard(self.d.layout, self.d.world.state_reader)(req)

    def test_a_selection_score_cannot_be_rerolled(self):
        adapter = self.d.state().trials["t-0001"].adapter
        with self.assertRaisesRegex(AccessDenied, "already exists"):
            self.guard(search_request(self.d, adapter))
        rc, _o, err = self.d.run("score", "--adapter", adapter.adapter_id, "--split", "selection", "--assignment", "a-evo-1", "--trial", "t-0001")
        self.assertEqual(rc, 1)
        self.assertIn("already exists", err)

    def test_a_new_adapter_with_the_rules_seeds_and_decoding_passes(self):
        self.guard(search_request(self.d, new_adapter("a")))

    def test_seeds_decoding_version_and_assignment_must_be_the_rules(self):
        a = new_adapter("b")
        rule = cli.load_frozen(self.d.layout, self.d.state().rule, cli.Rule)
        for kw in (dict(seeds=(0,)), dict(params=dataclasses.replace(rule.score_params, temperature=0.5)), dict(scorer_version="sv1-other"),
                   dict(assignment_id="a-held-1")):
            with self.assertRaises(AccessDenied, msg=str(kw)):
                self.guard(search_request(self.d, a, **kw))

    def test_the_phase_comes_from_the_ledger_not_from_the_request(self):
        with self.assertRaisesRegex(AccessDenied, "before lab.search_closed"):
            self.guard(search_request(self.d, new_adapter("c"), phase=Phase.FINAL, final_request_id="f-0001"))

    def test_a_retry_is_allowed_after_one_failure_that_revealed_nothing_and_not_after_two_or_a_revealed_one(self):
        d = self.driver("rule")
        a = new_adapter("d")
        fail = lambda rid, revealed=False: ScoreFailure(rid, Split.SELECTION, "a-evo-1", "t-0001", a.sha256, "endpoint died", revealed, d.world.config and cli.utc_iso(d.world.clock()))  # noqa: E731
        guard = cli.make_score_guard(d.layout, d.world.state_reader)
        guard(search_request(d, a))
        d.world.ledger.append(LabKind.SCORE_FAILED, "scorer", fail("sc-9001"))
        guard(search_request(d, a, request_id="sc-9002"))  # the one retry
        d.world.ledger.append(LabKind.SCORE_FAILED, "scorer", fail("sc-9002"))
        with self.assertRaisesRegex(AccessDenied, "retry"):
            guard(search_request(d, a, request_id="sc-9003"))
        b = new_adapter("e")
        d.world.ledger.append(LabKind.SCORE_FAILED, "scorer", ScoreFailure("sc-9004", Split.SELECTION, "a-evo-1", "t-0001", b.sha256, "partial", True, cli.utc_iso(d.world.clock())))
        with self.assertRaisesRegex(AccessDenied, "revealed"):
            guard(search_request(d, b, request_id="sc-9005"))

    def test_before_the_rule_selection_scoring_needs_a_declared_plan(self):
        d = self.driver("pack")
        guard = cli.make_score_guard(d.layout, d.world.state_reader)
        req = ScoreRequest("sc-9001", Phase.SEARCH, "a-evo-1", Split.SELECTION, "t-0001", None, d.world.config.base, new_adapter("f"), (0, 1, 2),
                           cli.GenerationParams.recommended(8000), d.world.scorer.scorer_version, None, None)
        with self.assertRaisesRegex(AccessDenied, "spread plan"):
            guard(req)
        d.stage("plan")
        guard(req)
        with self.assertRaises(AccessDenied):
            guard(dataclasses.replace(req, scorer_version="sv1-other"))

    def test_the_diagnostic_score_command_goes_through_the_same_scorer(self):
        out = self.d.ok("score", "--adapter", self.d.state().trials["t-0002"].adapter.adapter_id, "--split", "dev", "--assignment", "a-evo-1", "--trial", "t-0002")
        self.assertIn("headline_mean", out)  # dev numbers are public to researchers
        self.assertNotIn("headline_mean", self.d.ok("status"))
        # held-out data is unreachable in the search phase, dev included; the test split too
        adapter = self.d.state().trials["t-0003"].adapter.adapter_id
        for split, aid in (("dev", "a-held-1"), ("selection", "a-held-1")):
            self.assertIn("", self.d.refused("score", "--adapter", adapter, "--split", split, "--assignment", aid, "--trial", "t-0003"))
        rc, _o, err = self.d.run("score", "--adapter", adapter, "--split", "test", "--trial", "t-0003")
        self.assertEqual(rc, 1, err)


class VerifyTests(Base):
    def test_verify_recomputes_decisions_and_catches_a_forged_one(self):
        d = self.driver("search")
        st = d.state()
        real = st.decisions["cand-003"]  # a rejected confirmation
        self.assertIs(real.outcome, Outcome.REJECTED)
        # a self-consistent forged decision for a new candidate: real scores, but numbers that say "accepted"
        cand = dataclasses.replace(st.candidates["cand-003"], candidate_id="cand-099", version_label="cand-099", pack=None, diff=None, lint_passed=False,
                                   lint_messages=("forged",))
        d.world.ledger.append(LabKind.CANDIDATE_PROPOSED, "outer", cand)
        forged = Decision("cand-099", real.rule_sha256, DecisionStage.CONFIRM, 9, Outcome.ACCEPTED, "delta 0.50 exceeds margin 0.05", real.incumbent_version,
                          "cand-099", real.incumbent_scores, real.candidate_scores, 0.5, 0.5, 0.05, real.df, real.sigma_used, CostBreakdown(0, 0, 0, 0), real.decided_utc)
        d.world.ledger.append(LabKind.CANDIDATE_DECIDED, "outer", forged)
        rc, out, _err = d.run("verify")
        self.assertEqual(rc, 1)
        bad = [c for c in json.loads(out)["checks"] if not c["ok"]]
        self.assertEqual([c["name"] for c in bad], ["decision cand-099"])
        self.assertIn("recomputed", bad[0]["detail"])

    def test_verify_refuses_a_decision_built_from_invented_trials_with_no_score_ids(self):
        d = self.driver("search")
        st = d.state()
        real = st.decisions["cand-003"]
        rule = cli.load_frozen(d.layout, st.rule, cli.Rule)
        fake = lambda ts, i, score: dataclasses.replace(ts, trial_id=f"t-9{i:03d}", score_id=None, score=score)  # noqa: E731
        inc = tuple(fake(ts, i, 0.1) for i, ts in enumerate(real.incumbent_scores))
        cand = tuple(fake(ts, 500 + i, 0.9) for i, ts in enumerate(real.candidate_scores))
        nums = cli.stats.decide(rule, inc, cand)  # numbers that agree with the invented scores
        self.assertIs(nums.outcome, Outcome.ACCEPTED)
        c = dataclasses.replace(st.candidates["cand-003"], candidate_id="cand-098", version_label="cand-098", pack=None, diff=None, lint_passed=False,
                                lint_messages=("forged",))
        d.world.ledger.append(LabKind.CANDIDATE_PROPOSED, "outer", c)
        forged = Decision("cand-098", real.rule_sha256, DecisionStage.CONFIRM, 9, nums.outcome, "forged", real.incumbent_version, "cand-098", inc, cand,
                          nums.delta, nums.delta, nums.margin, nums.df, nums.sigma_used, CostBreakdown(0, 0, 0, 0), real.decided_utc)
        d.world.ledger.append(LabKind.CANDIDATE_DECIDED, "outer", forged)
        rc, out, _err = d.run("verify")
        self.assertEqual(rc, 1)
        bad = [x for x in json.loads(out)["checks"] if not x["ok"]]
        self.assertEqual([x["name"] for x in bad], ["decision cand-098"])
        self.assertIn("without a score id", bad[0]["detail"])

    def test_verify_refuses_a_decision_whose_scores_belong_to_another_pack(self):
        d = self.driver("search")
        st = d.state()
        real = st.decisions["cand-003"]
        rule = cli.load_frozen(d.layout, st.rule, cli.Rule)
        self.assertEqual(cli._recompute_decision(st, rule, real, d.world.state_reader), (True, ""))
        # incumbent trials presented as the candidate's confirmation scores, numbers recomputed to agree with them
        inc_pack = st.trials_started[real.incumbent_scores[0].trial_id].pack_version
        self.assertEqual(inc_pack, real.incumbent_version)
        nums = cli.stats.decide(rule, real.incumbent_scores, real.incumbent_scores)
        forged = dataclasses.replace(real, candidate_scores=real.incumbent_scores, outcome=nums.outcome, delta=nums.delta, margin=nums.margin,
                                     df=nums.df, sigma_used=nums.sigma_used)
        ok, detail = cli._recompute_decision(st, rule, forged, d.world.state_reader)
        self.assertFalse(ok)
        self.assertIn("listed as a candidate score", detail)
        swapped = dataclasses.replace(real, incumbent_scores=real.candidate_scores)
        self.assertIn("listed as a incumbent score", cli._recompute_decision(st, rule, swapped, d.world.state_reader)[1])

    def test_scores_of_trials_that_are_not_finished_and_eligible_are_not_confirmation_evidence(self):
        import copy

        d = self.driver("search")
        st = d.state()
        real = st.decisions["cand-003"]
        rule = cli.load_frozen(d.layout, st.rule, cli.Rule)
        self.assertEqual(cli._recompute_decision(st, rule, real, d.world.state_reader), (True, ""))
        ids = {ts.trial_id for ts in real.incumbent_scores + real.candidate_scores}
        # the same score records, but no lab.trial_finished for their trials
        unfinished = copy.copy(st)
        unfinished.trials = {k: v for k, v in st.trials.items() if k not in ids}
        for reader in (d.world.state_reader, None):
            ok, detail = cli._recompute_decision(unfinished, rule, real, reader)
            self.assertFalse(ok)
            self.assertIn("finished, eligible search trial", detail)
        # a finished trial that the fold would not count (failed status) is no evidence either, with or without a reader
        victim = real.candidate_scores[0].trial_id
        failed = copy.copy(st)
        failed.trials = dict(st.trials)
        failed.trials[victim] = dataclasses.replace(st.trials[victim], status=TrialStatus.FAILED, failure="it broke")
        for reader in (d.world.state_reader, None):
            ok, detail = cli._recompute_decision(failed, rule, real, reader)
            self.assertFalse(ok)
            self.assertIn(victim, detail)

    def test_a_scored_zero_trial_is_accepted_only_when_the_fold_counts_it_as_failed(self):
        d = self.driver("search")
        st = d.state()
        rule = dataclasses.replace(cli.load_frozen(d.layout, st.rule, cli.Rule), failed_trial_policy=cli.FailedTrialPolicy.SCORED_ZERO)
        real = st.decisions["cand-003"]
        ts = real.incumbent_scores[0]
        spec = st.trials_started[ts.trial_id]
        zero = dataclasses.replace(ts, score_id=None, score=0.0)
        dec = dataclasses.replace(real, rule_sha256=rule.sha256(), incumbent_scores=(zero,), candidate_scores=())

        class Reader:
            def __init__(self, failed):
                self.failed = failed

            def selection_trial_scores(self, state, rule, pack_version):
                return [dataclasses.replace(zero)] if self.failed and pack_version == spec.pack_version else []

        marker = "without a score id"
        self.assertNotIn(marker, cli._recompute_decision(st, rule, dec, Reader(True))[1])
        self.assertIn(marker, cli._recompute_decision(st, rule, dec, Reader(False))[1])
        self.assertIn(marker, cli._recompute_decision(st, rule, dec, None)[1])
        strict = dataclasses.replace(rule, failed_trial_policy=cli.FailedTrialPolicy.INVALIDATES_CANDIDATE)
        self.assertIn(marker, cli._recompute_decision(st, strict, dec, Reader(True))[1])

    def test_verify_fails_on_a_malformed_score_that_only_the_decision_recheck_finds(self):
        d = self.driver("search")
        self.assertEqual(d.run("verify")[0], 0)
        real = cli._recompute_decision

        def flagging(state, rule, dec, reader=None):
            # what the fold does when a decision's trial has a score whose scorer version differs from the rule: it flags the record
            # while the scores are read, which happens only now, not while the ledger is folded
            state.malformed.append((7, "selection score 'sc-1' differs from the rule: scorer version"))
            return real(state, rule, dec, reader)

        with mock.patch.object(cli, "_recompute_decision", flagging):
            rc, out, _ = d.run("verify")
        self.assertEqual(rc, 1)
        res = json.loads(out)
        self.assertFalse(res["ok"])
        self.assertEqual([c["name"] for c in res["checks"] if not c["ok"]], ["no malformed events"])

    def test_verify_catches_a_deleted_trial_ledger_that_the_master_ledger_anchors(self):
        import io
        import shutil

        d = self.driver("rule")
        anchored = [t for t, r in d.state().trials.items() if r.trial_ledger_anchor is not None]
        self.assertTrue(anchored)
        self.assertEqual(d.run("verify")[0], 0)
        d.world.ledger.close()
        shutil.rmtree(d.layout.trial_ledger_dir(anchored[0]) / "ledger.jsonl", ignore_errors=True)
        (d.layout.trial_ledger_dir(anchored[0]) / "ledger.jsonl").unlink()
        out = io.StringIO()
        self.assertEqual(cli.main([d.root, "verify"], out=out, err=io.StringIO()), 1)
        self.assertIn(f"trial ledger {anchored[0]}", [c["name"] for c in json.loads(out.getvalue())["checks"] if not c["ok"]])

    def test_verify_catches_a_rewritten_frozen_file(self):
        d = self.driver("rule")
        path = d.layout.rule_path
        os.chmod(path, 0o644)
        data = json.loads(path.read_text())
        data["min_gain"] = 0.001
        path.write_text(json.dumps(data))
        rc, out, _ = d.run("verify")
        self.assertEqual(rc, 1)
        self.assertIn("frozen files match the ledger", [c["name"] for c in json.loads(out)["checks"] if not c["ok"]])

    def test_verify_checks_each_frozen_file_before_all_three_are_frozen(self):
        d = self.driver("plan")  # budget and manifest are frozen, the rule is not
        self.assertIsNone(d.state().rule)
        self.assertEqual(d.run("verify")[0], 0)
        for path in (d.layout.budget_path, d.layout.manifest_path):
            with self.subTest(path.name):
                saved = path.read_bytes()
                os.chmod(path, 0o644)
                path.unlink()  # a missing frozen file
                rc, out, _ = d.run("verify")
                self.assertEqual(rc, 1)
                self.assertIn("frozen files match the ledger", [c["name"] for c in json.loads(out)["checks"] if not c["ok"]])
                path.write_bytes(saved)
                self.assertEqual(d.run("verify")[0], 0)

    def test_verify_catches_an_anchor_that_the_ledger_does_not_have(self):
        d = self.driver("rule")
        lines = d.layout.anchors_path.read_text().splitlines()
        seq = lines[0].split(":")[0]
        d.layout.anchors_path.write_text(f"{seq}:{'0' * 16} rule_frozen 2026-11-01T00:00:00+00:00\n")
        rc, out, _ = d.run("verify")
        self.assertEqual(rc, 1)
        self.assertIn("anchors match the ledger", [c["name"] for c in json.loads(out)["checks"] if not c["ok"]])
        d.layout.anchors_path.write_text(f"9999:{'0' * 16} later 2026-11-01T00:00:00+00:00\n")  # a head beyond the ledger: it was truncated
        self.assertEqual(d.run("verify")[0], 1)

    def test_verify_catches_a_tampered_master_ledger_file_and_a_tampered_trial_ledger(self):
        d = self.driver("rule")
        d.world.ledger.close()
        text = d.layout.master_ledger_dir.joinpath("ledger.jsonl").read_text()
        import io

        # untouched file: the production path (no world) verifies it
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(cli.main([d.root, "verify"], out=out, err=err), 0, err.getvalue())
        tampered = text.replace('"researcher_tokens": 2000000', '"researcher_tokens": 3000000', 1)
        self.assertNotEqual(tampered, text)
        d.layout.master_ledger_dir.joinpath("ledger.jsonl").write_text(tampered)
        out = io.StringIO()
        self.assertEqual(cli.main([d.root, "verify"], out=out, err=io.StringIO()), 1)
        self.assertFalse(json.loads(out.getvalue())["ok"])
        d.layout.master_ledger_dir.joinpath("ledger.jsonl").write_text(text)
        tl = d.layout.trial_ledger_dir("t-0001") / "ledger.jsonl"
        os.chmod(tl, 0o644)  # a finished trial ledger is sealed read-only
        t = tl.read_text()
        self.assertIn('"amount"', t)
        tl.write_text(t.replace('"amount": 5.0', '"amount": 1.0', 1))
        out = io.StringIO()
        self.assertEqual(cli.main([d.root, "verify"], out=out, err=io.StringIO()), 1)
        self.assertIn("trial ledger t-0001", [c["name"] for c in json.loads(out.getvalue())["checks"] if not c["ok"]])

    def test_verify_fails_when_the_rule_was_frozen_after_a_candidate(self):
        from tests.test_lab_contracts import candidate, rule as make_rule

        d = self.driver("spread")
        st = d.state()
        # the order rule refuses this through the CLI ...
        d.world.ledger.append(LabKind.CANDIDATE_PROPOSED, "outer",
                              dataclasses.replace(candidate(), parent=st.packs["v0"].pack, pack=None, diff=None, lint_passed=False, lint_messages=("x",)))
        self.assertIn("first candidate", d.refused("freeze-rule", *RULE_ARGS))
        # ... so write the rule behind its back: the fold accepts the event, verify must not
        spec = st.trials_started["t-0001"]
        late = make_rule(spread=st.spread.estimate, scorer_version=st.spread_plan.scorer_version, score_seeds=spec.score_seeds, score_params=spec.score_params)
        path = d.layout.rule_path
        write_record(path, late)
        d.world.ledger.append(LabKind.RULE_FROZEN, "coordinator", cli.FrozenFile("rule", d.layout.file_ref(path)))
        rc, out, _ = d.run("verify")
        self.assertEqual(rc, 1)
        self.assertEqual([c["name"] for c in json.loads(out)["checks"] if not c["ok"]], ["rule frozen before the first candidate"])


class IsolationChecks(Base):
    def test_link_checks_pass_here(self):
        d = self.driver()
        out = d.ok("verify-isolation")
        self.assertTrue(out["ok"])
        names = [c["name"] for c in out["checks"]]
        for needle in ("read_regular_bytes refuses link.txt", "read_regular_bytes refuses hard.txt", "file_ref refuses link.txt", "file_ref refuses hard.txt"):
            self.assertIn(needle, names)

    def test_a_reader_that_follows_links_fails_the_check(self):
        d = self.driver()

        def naive(root, rel, *, max_bytes):
            return (Path(root) / rel).read_bytes()

        with mock.patch("nightshift.lab.cli.read_regular_bytes", naive):
            rc, out, _ = d.run("verify-isolation")
        self.assertEqual(rc, 1)
        failed = [c["name"] for c in json.loads(out)["checks"] if not c["ok"]]
        self.assertIn("read_regular_bytes refuses link.txt", failed)

    def test_a_file_ref_that_follows_links_fails_the_check(self):
        d = self.driver()
        with mock.patch("nightshift.lab.contracts.file_ref", lambda root, path: object()):
            rc, out, _ = d.run("verify-isolation")
        self.assertEqual(rc, 1)

    def test_researcher_host_mode_checks_tokens_and_the_lab_root(self):
        d = self.driver()
        with mock.patch.dict(os.environ, {"FAKE_META_KEY": "leaked"}):
            rc, out, _ = d.run("verify-isolation", "--researcher-host", "--skip-ssh")
        self.assertEqual(rc, 1)
        self.assertIn("no lab secret in this environment", [c["name"] for c in json.loads(out)["checks"] if not c["ok"]])
        with mock.patch.dict(os.environ, {"HOME": self.tmp()}, clear=False):
            os.environ.pop("FAKE_META_KEY", None)
            (Path(os.environ["HOME"]) / ".thunder").write_text("token")
            rc, out, _ = d.run("verify-isolation", "--researcher-host", "--skip-ssh")
            self.assertIn("no Thunder token file readable", [c["name"] for c in json.loads(out)["checks"] if not c["ok"]])


class SidakAndHelpers(unittest.TestCase):
    def test_sidak_alpha(self):
        self.assertEqual(cli.sidak_alpha(0.05, 1), 0.05)
        a = cli.sidak_alpha(0.05, 2)
        self.assertAlmostEqual(1 - (1 - a) ** 2, 0.05, places=12)
        self.assertLess(cli.sidak_alpha(0.05, 4), cli.sidak_alpha(0.05, 2))

    def test_researcher_model_is_selected_by_the_environment_and_keeps_its_default(self):
        self.assertEqual(cli.researcher_model({}), "muse-spark-1.3")  # old roots: unchanged
        self.assertEqual(cli.researcher_model({"LAB_RESEARCHER_MODEL": ""}), "muse-spark-1.3")
        self.assertEqual(cli.researcher_model({"LAB_RESEARCHER_MODEL": "muse-wonder-full-1.0"}), "muse-wonder-full-1.0")
        self.assertEqual(cli.outer_challenge_enabled({"LAB_PROPOSER": "muse_cli"}), True)
        self.assertEqual(cli.outer_challenge_enabled({"LAB_PROPOSER": "muse_cli", "LAB_OUTER_CHALLENGE": "off"}), False)
        self.assertEqual(cli.outer_challenge_enabled({}), False)  # the API proposer path is unchanged


if __name__ == "__main__":
    unittest.main()
