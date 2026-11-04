"""Tests for nightshift.lab.outer_loop (owner 4): scripted proposer, scripted trial executor, REAL ledger, fold, stats and budget cap.

Run: PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_outer_loop
"""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from nightshift.lab.contracts import (
    AccessDenied,
    BudgetExhausted,
    DecisionStage,
    GateNotRecorded,
    LabKind,
    Outcome,
    SearchClosed,
    read_record,
    Proposal,
)
from tests.support.lab_fake_proposer import (
    EVO,
    InMemoryPackStore,
    OuterLab,
    ScriptedProposalSource,
    make_proposal,
    make_rule,
)

try:
    from nightshift.lab import stats
    from nightshift.lab.budget import LabCap
    from nightshift.lab.outer_loop import OuterController, OuterRefused
    from nightshift.lab.proposer import UnparseableProposal
except ImportError as exc:  # pragma: no cover - owners 3 and 6 modules are not in the tree
    raise unittest.SkipTest(f"needs the stats, budget, fold and ledger modules: {exc}")


def prop(n: int):
    """A distinct valid proposal: it appends a unique line to the notes (so every candidate is a different pack)."""
    return make_proposal("notes/seed.md", "## Log", f"## Log\n- idea {n}", summary=f"idea {n}")


def values(table: dict, *, noise: float = 0.004):
    """value_fn: the selection score of a trial = table[pack_version] (default 0.60) plus a small seed-dependent wobble."""

    def fn(spec):
        base = table.get(spec.pack_version, 0.60)
        if callable(base):
            return base(spec)
        return base + noise * ((spec.seed * 7) % 5 - 2)

    return fn


class Base(unittest.TestCase):
    def build(self, *, value_fn=None, script=(), rule=None, cap: float = 1000.0, store=None, **lab_kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.lab = OuterLab(tmp.name, value_fn=value_fn, rule=rule, store=store, **lab_kw)
        self.addCleanup(self.lab.ledger.close)
        self.cap = LabCap(cap)
        self.proposer = ScriptedProposalSource(list(script))
        self.hooked = []
        self.ctl = self.controller()
        return self.lab

    def controller(self, executor=None, **kw):
        return OuterController(
            self.lab.layout, self.lab.ledger, self.lab.reader, self.lab.store, self.proposer, executor or self.lab, stats,
            clock=self.lab.clock, lab_cap=self.cap, hosts=("ax43.example", "ax41.example"), after_decision=self.hooked.append, **kw,
        )

    def state(self):
        return self.lab.reader.state()


class RoundTests(Base):
    def test_a_screen_loss_a_confirmed_accept_and_the_trial_plan(self):
        """3 candidates: cand-002 gains 0.15, the others do not. Two lose the screen, one is confirmed and accepted."""
        self.build(value_fn=values({"cand-002": 0.75}), script=[prop(1), prop(2), prop(3)])
        ds = self.ctl.run_round()
        by_id = {d.candidate_id: d for d in ds}
        self.assertEqual(sorted(by_id), ["cand-001", "cand-002", "cand-003"])
        for cid in ("cand-001", "cand-003"):
            self.assertEqual((by_id[cid].stage, by_id[cid].outcome), (DecisionStage.SCREEN, Outcome.REJECTED))
            self.assertIsNotNone(by_id[cid].screen_delta)
            self.assertIsNone(by_id[cid].delta)
        win = by_id["cand-002"]
        self.assertEqual((win.stage, win.outcome), (DecisionStage.CONFIRM, Outcome.ACCEPTED))
        self.assertGreater(win.delta, win.margin)
        self.assertGreater(win.screen_delta, 0)
        self.assertEqual(win.rule_sha256, self.lab.rule.sha256())
        self.assertEqual(self.lab.store.accepted, ["cand-002"])
        st = self.state()
        self.assertEqual(self.lab.reader.incumbent_version(st), "cand-002")
        self.assertEqual(st.malformed, [])
        self.assertEqual(self.lab.ledger.verify(), (True, None))
        # the plan: 5 spread trials, then 3 candidates x 2 screen trials, then 3 + 3 fresh confirmation trials
        run = [s for s in self.lab.executed][5:]
        self.assertEqual(len(run), 6 + 6)
        screen, confirm = run[:6], run[6:]
        self.assertEqual([s.pack_version for s in screen], ["cand-001", "cand-001", "cand-002", "cand-002", "cand-003", "cand-003"])
        self.assertEqual(sorted(s.pack_version for s in confirm), ["cand-002"] * 3 + ["v0"] * 3)
        # interleaved and counterbalanced (AB, BA, AB): time drift cannot favour one side
        self.assertEqual([s.pack_version for s in confirm], ["cand-002", "v0", "v0", "cand-002", "cand-002", "v0"])
        # fresh seeds: a confirmation never repeats a screening trial of the same pack
        seeds = [s.seed for s in confirm if s.pack_version == "cand-002"]
        self.assertTrue(set(seeds).isdisjoint({s.seed for s in screen if s.pack_version == "cand-002"}))
        keys = [(s.pack.commit, s.assignment_id, s.seed) for s in self.lab.executed]
        self.assertEqual(len(keys), len(set(keys)))
        # the confirmation compares fresh trials only
        screen_ids = {s.trial_id for s in screen}
        self.assertTrue({t.trial_id for t in win.candidate_scores}.isdisjoint(screen_ids))
        self.assertEqual(len(win.candidate_scores), 3)
        self.assertEqual(len(win.incumbent_scores), 3)
        self.assertTrue(all(t.trial_id in {s.trial_id for s in confirm} for t in win.incumbent_scores))
        # cost centres: the proposal is booked on the candidate; trials on the cost breakdown
        self.assertAlmostEqual(win.cost.outer_propose_usd, 0.05)
        self.assertGreater(win.cost.candidate_researcher_usd, 0)
        self.assertEqual(len(self.hooked), 3)
        # the Wonder spend reached the lab-wide cap
        self.assertAlmostEqual(self.cap.spent_usd, 0.15)

    def test_a_confirmed_reject_when_the_screen_gain_was_luck(self):
        lucky = lambda spec: (0.72 if spec.seed < 2 else 0.60) + 0.004 * ((spec.seed * 7) % 5 - 2)
        self.build(value_fn=values({"cand-001": lucky}), script=[prop(1), prop(2), prop(3)])
        ds = self.ctl.run_round()
        win = next(d for d in ds if d.candidate_id == "cand-001")
        self.assertEqual((win.stage, win.outcome), (DecisionStage.CONFIRM, Outcome.REJECTED))
        self.assertLessEqual(win.delta, win.margin)
        self.assertGreater(win.screen_delta, 0.05, "it did look good at the screen")
        self.assertEqual(self.lab.store.accepted, [])
        self.assertEqual(self.lab.reader.incumbent_version(self.state()), "v0")
        self.assertEqual(self.state().malformed, [])

    def test_no_positive_screen_delta_means_no_confirmation(self):
        self.build(value_fn=values({}), script=[prop(1), prop(2), prop(3)])
        ds = self.ctl.run_round()
        self.assertEqual({d.stage for d in ds}, {DecisionStage.SCREEN})
        self.assertEqual({d.outcome for d in ds}, {Outcome.REJECTED})
        self.assertEqual(len(self.lab.executed), 5 + 6, "no confirmation trial was spent")

    def test_candidates_are_counted_from_the_ledger_so_the_limit_holds_across_invocations(self):
        rule = make_rule(max_candidates=4, screen_batch_size=3)
        self.build(rule=rule, value_fn=values({}), script=[prop(i) for i in range(1, 8)])
        self.assertEqual(len(self.ctl.run_round()), 3)
        self.assertEqual(len(self.ctl.run_round()), 1, "only one slot was left")
        self.assertEqual(self.ctl.run_round(), [])
        again = self.controller()  # a second `outer` invocation / a restart
        self.assertEqual(again.run_round(), [])
        self.assertEqual(again.run(5), [])
        self.assertEqual(len(self.state().candidates), 4)
        self.assertEqual(len(self.proposer.contexts), 4)

    def test_run_stops_at_the_limit_and_accumulates_rounds(self):
        rule = make_rule(max_candidates=4, screen_batch_size=2, stop_after_rejections=None)
        self.build(rule=rule, value_fn=values({}), script=[prop(i) for i in range(1, 8)])
        ds = self.ctl.run(10)
        self.assertEqual(len(ds), 4)
        self.assertEqual(sorted({d.round_index for d in ds}), [0, 1])

    def test_stop_after_rejections_counts_rounds_without_an_accept(self):
        rule = make_rule(max_candidates=6, screen_batch_size=2, stop_after_rejections=1)
        self.build(rule=rule, value_fn=values({}), script=[prop(i) for i in range(1, 8)])
        self.assertEqual(len(self.ctl.run_round()), 2)
        self.assertEqual(self.ctl.run_round(), [])

    def test_a_deadline_in_the_past_proposes_nothing(self):
        self.build(value_fn=values({}), script=[prop(1)])
        ctl = self.controller(deadline_epoch=1.0)
        self.assertEqual(ctl.run_round(), [])
        self.assertEqual(self.proposer.contexts, [])

    def test_a_crash_between_the_accepting_decision_and_accept_is_repaired_on_restart_even_at_the_candidate_limit(self):
        rule = make_rule(max_candidates=3, screen_batch_size=3)
        self.build(value_fn=values({"cand-002": 0.75}), script=[prop(i) for i in range(1, 4)], rule=rule)
        store, real_accept, calls = self.lab.store, self.lab.store.accept, []

        def dying_accept(pack, label):
            calls.append(label)
            raise RuntimeError("the process died before main moved")

        store.accept = dying_accept
        with self.assertRaises(RuntimeError):
            self.ctl.run_round()
        store.accept = real_accept
        self.assertEqual(self.lab.reader.incumbent_version(self.state()), "cand-002", "the ledger already holds the acceptance")
        self.assertEqual(store.accepted, [])
        self.assertEqual(self.controller().run_round(), [], "the candidate limit is reached: nothing new is proposed")
        self.assertEqual(store.accepted, ["cand-002"], "the restart moved main to the ledger's incumbent")
        self.assertEqual(store.main, self.state().packs["cand-002"].pack.commit)

    def test_the_second_round_builds_on_the_accepted_incumbent(self):
        self.build(value_fn=values({"cand-001": 0.75, "cand-004": 0.60}), script=[prop(i) for i in range(1, 7)], rule=make_rule(screen_batch_size=3))
        self.ctl.run_round()
        self.assertEqual(self.lab.reader.incumbent_version(self.state()), "cand-001")
        self.ctl.run_round()
        second = [s for s in self.lab.executed if s.pack_version == "cand-004"]
        self.assertEqual(len(second), 2)
        rec = self.state().candidates["cand-004"]
        self.assertEqual(rec.parent_version, "cand-001")
        self.assertEqual(rec.parent, self.state().packs["cand-001"].pack)
        # screening compared with the new incumbent's trials
        d = self.state().decisions["cand-004"]
        self.assertEqual(d.incumbent_version, "cand-001")


class InvalidCandidateTests(Base):
    def test_an_unparseable_reply_is_an_invalid_candidate_that_costs_only_wonder_tokens(self):
        bad = UnparseableProposal("the reply is not a JSON object", raw_response="I would rather not.", usd=0.07, model="muse-wonder-1.0", served_model="muse-wonder-1.0")
        self.build(value_fn=values({}), script=[bad, prop(2), prop(3)])
        ds = self.ctl.run_round()
        st = self.state()
        rec = st.candidates["cand-001"]
        self.assertIsNone(rec.pack)
        self.assertFalse(rec.lint_passed)
        self.assertAlmostEqual(rec.propose_usd, 0.07)
        inv = next(d for d in ds if d.candidate_id == "cand-001")
        self.assertEqual((inv.stage, inv.outcome), (DecisionStage.SCREEN, Outcome.INVALID))
        self.assertIsNone(inv.delta)
        self.assertTrue(all(s.pack_version != "cand-001" for s in self.lab.executed), "no trial for an invalid candidate")
        # the files exist and the placeholder proposal is a readable Proposal
        lay = self.lab.layout
        self.assertEqual(lay.candidate_response_path("cand-001").read_text(), "I would rather not.")
        self.assertIn("INVALID", read_record(lay.candidate_proposal_path("cand-001"), Proposal).hypothesis)
        self.assertEqual(st.malformed, [])

    def test_a_proposal_whose_block_does_not_match_is_invalid_and_spends_no_gpu(self):
        bad = make_proposal("notes/seed.md", "this text is not in the file", "x")
        self.build(value_fn=values({}), script=[bad, prop(2), prop(3)])
        ds = self.ctl.run_round()
        inv = next(d for d in ds if d.candidate_id == "cand-001")
        self.assertEqual(inv.outcome, Outcome.INVALID)
        rec = self.state().candidates["cand-001"]
        self.assertIn("matches 0 times", rec.lint_messages[0])
        self.assertTrue(read_record(self.lab.layout.candidate_proposal_path("cand-001"), Proposal).edits, "the real proposal is kept")
        self.assertTrue(all(s.pack_version != "cand-001" for s in self.lab.executed))

    def test_the_proposer_sees_the_pack_contract_and_the_editable_paths(self):
        self.build(value_fn=values({}), script=[prop(1), prop(2), prop(3)])
        self.ctl.run_round()
        u = self.proposer.contexts[0].user_prompt
        self.assertIn("## Pack contract", u)
        self.assertIn("agents.count: 1 to 4", u)
        self.assertIn("Editable paths: task.yaml, notes/, skills/, roles/, seed/", u)

    def test_the_provider_failing_midway_runs_what_was_proposed_and_failing_first_raises(self):
        from nightshift.lab.proposer import ProposerUnavailable

        self.build(value_fn=values({}), script=[prop(1), ProposerUnavailable("503")])
        ds = self.ctl.run_round()
        self.assertEqual([d.candidate_id for d in ds], ["cand-001"])
        self.build(value_fn=values({}), script=[ProposerUnavailable("503")])
        with self.assertRaises(ProposerUnavailable):
            self.ctl.run_round()
        self.assertEqual(self.state().candidates, {})

    def test_partial_batches_cannot_run_more_confirmations_than_the_reported_bound_counts(self):
        """max_candidates 4 and batch 2 give R = 2 rounds in the false-accept bound. A provider that fails after each proposal makes every
        candidate its own round; the controller must stop after R confirmation tests, not run one per candidate."""
        from nightshift.lab.proposer import ProposerUnavailable

        script = []
        for i in range(1, 5):
            script += [prop(i), ProposerUnavailable("503")]
        self.build(value_fn=values({f"cand-00{i}": 0.60 + 0.05 * i for i in range(1, 5)}), script=script, rule=make_rule(max_candidates=4, screen_batch_size=2))
        for _ in range(4):
            self.ctl.run_round()
        confirms = [d for d in self.state().decisions.values() if d.stage is DecisionStage.CONFIRM]
        self.assertEqual(len(confirms), 2, [(d.candidate_id, d.stage, d.outcome) for d in self.state().decisions.values()])
        self.assertEqual(len(self.state().candidates), 2, "no further proposal is bought once the confirmation budget is spent")

    def test_a_forbidden_path_is_invalid(self):
        bad = make_proposal("grader/grader.py", "a", "b")
        self.build(value_fn=values({}), script=[bad, prop(2), prop(3)])
        ds = self.ctl.run_round()
        self.assertEqual(next(d for d in ds if d.candidate_id == "cand-001").outcome, Outcome.INVALID)

    def test_a_lint_failure_is_invalid_with_the_lint_messages(self):
        self.build(value_fn=values({}), script=[prop(1), prop(2), prop(3)])
        self.lab.store.lint_messages["cand-002"] = ("agents.model is not editable: parent 'a', candidate 'b'",)
        ds = self.ctl.run_round()
        inv = next(d for d in ds if d.candidate_id == "cand-002")
        self.assertEqual(inv.outcome, Outcome.INVALID)
        rec = self.state().candidates["cand-002"]
        self.assertFalse(rec.lint_passed)
        self.assertEqual(rec.lint_messages, ("agents.model is not editable: parent 'a', candidate 'b'",))
        self.assertIsNotNone(rec.pack, "the commit exists and stays tagged")
        self.assertTrue(all(s.pack_version != "cand-002" for s in self.lab.executed))

    def test_a_failed_trial_invalidates_the_candidate_under_the_frozen_policy(self):
        fail_first = lambda spec: None if spec.seed == 0 else 0.75
        self.build(value_fn=values({"cand-001": fail_first}), script=[prop(1), prop(2), prop(3)])
        ds = self.ctl.run_round()
        d = next(d for d in ds if d.candidate_id == "cand-001")
        self.assertEqual((d.stage, d.outcome), (DecisionStage.SCREEN, Outcome.INVALID))

    def test_invalid_candidates_count_toward_the_limit(self):
        rule = make_rule(max_candidates=2, screen_batch_size=2)
        bad = UnparseableProposal("no", raw_response="x")
        self.build(rule=rule, value_fn=values({}), script=[bad, bad, prop(3)])
        self.ctl.run_round()
        self.assertEqual(len(self.state().candidates), 2)
        self.assertEqual(self.ctl.run_round(), [])

    def test_the_provider_reporting_a_fallback_is_recorded_on_the_candidate(self):
        from nightshift.lab.contracts import ProposalResult

        p = prop(1)
        res = ProposalResult(p, "muse-wonder-1.0", "muse-spark-1.3", True, 0.02, 10, 10, "{}")
        self.build(value_fn=values({}), script=[res, prop(2)], rule=make_rule(screen_batch_size=2))
        self.ctl.run_round()
        rec = self.state().candidates["cand-001"]
        self.assertTrue(rec.fallback_used)
        self.assertEqual((rec.proposer_model, rec.proposer_served_model), ("muse-wonder-1.0", "muse-spark-1.3"))


class PreconditionTests(Base):
    def test_no_terms_gate_no_round(self):
        self.build(with_gate=False, script=[prop(1)])
        with self.assertRaises(GateNotRecorded):
            self.ctl.run_round()
        self.assertEqual(self.proposer.contexts, [])

    def test_no_spread_no_round(self):
        self.build(with_spread=False, script=[prop(1)])
        with self.assertRaisesRegex(OuterRefused, "spread"):
            self.ctl.run_round()

    def test_a_closed_search_refuses(self):
        self.build(value_fn=values({}), script=[prop(1)])
        st = self.state()
        self.lab.ledger.append(LabKind.SEARCH_CLOSED, "coordinator", SearchClosed("v0", st.packs["v0"].pack, 0, 5, "done", "2026-11-01T12:00:00+00:00"))
        with self.assertRaisesRegex(OuterRefused, "closed"):
            self.ctl.run_round()

    def test_a_started_but_unfinished_trial_blocks_the_round(self):
        self.build(value_fn=values({}), script=[prop(1)])
        spec = self.lab.make_spec("v0", self.lab.v0, EVO, 99)
        self.lab.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
        with self.assertRaisesRegex(OuterRefused, spec.trial_id):
            self.ctl.run_round()

    def test_not_enough_lab_dollars_for_the_worst_case_refuses_before_any_call(self):
        self.build(value_fn=values({}), script=[prop(1)], cap=100.0)
        with self.assertRaises(BudgetExhausted) as cm:
            self.ctl.run_round()
        self.assertEqual(cm.exception.axis, "lab_dollars")
        self.assertEqual(self.proposer.contexts, [])
        self.assertEqual(len(self.lab.executed), 5)

    def test_the_trial_llm_share_sizes_the_worst_case(self):
        self.build(value_fn=values({}), script=[prop(1), prop(2), prop(3)], cap=100.0)
        ctl = self.controller(trial_llm_usd_cap=5.0)
        self.assertEqual(len(ctl.run_round()), 3)

    def test_running_out_of_lab_dollars_mid_round_closes_the_open_candidates_invalid(self):
        self.build(value_fn=values({}), script=[prop(1), prop(2), prop(3)], cap=80.0)
        ctl = self.controller(trial_llm_usd_cap=5.0, scoring_usd_per_trial=0.0)
        # the cap check passes at the start; then the money disappears after the first screening trial
        orig = self.lab.run_trial
        calls = []

        def spend(spec):
            calls.append(spec)
            rec = orig(spec)
            if len(calls) == 1:
                self.cap.add_spent(79.9)
            return rec

        class Ex:
            run = staticmethod(spend)

        ctl.executor = Ex()
        ds = ctl.run_round()
        self.assertEqual({d.outcome for d in ds}, {Outcome.INVALID})
        self.assertEqual(len(ds), 3)
        self.assertTrue(all("stopped early" in d.reason for d in ds))
        self.assertEqual(len(calls), 1)

    def test_an_empty_denylist_with_a_rule_refuses(self):
        self.build(value_fn=values({}), script=[prop(1)])
        import nightshift.lab.outer_loop as ol

        orig = ol.forbidden_values
        ol.forbidden_values = lambda *a, **k: frozenset()
        self.addCleanup(setattr, ol, "forbidden_values", orig)
        with self.assertRaisesRegex(OuterRefused, "denylist is empty"):
            self.ctl.run_round()
        self.assertEqual(self.proposer.contexts, [])


class CrashTests(Base):
    def test_a_round_that_died_is_closed_invalid_and_the_next_round_proceeds(self):
        self.build(value_fn=values({}), script=[prop(1), prop(2), prop(3), prop(4), prop(5), prop(6)])

        class Dies:
            def run(self, spec):
                raise RuntimeError("the machine went away")

        crashing = self.controller(executor=Dies())
        with self.assertRaises(RuntimeError):
            crashing.run_round()
        st = self.state()
        self.assertEqual(sorted(st.candidates), ["cand-001", "cand-002", "cand-003"])
        self.assertEqual(st.decisions, {})
        fresh = self.controller()
        ds = fresh.run_round()
        closed = [d for d in ds if "interrupted" in d.reason]
        self.assertEqual(sorted(d.candidate_id for d in closed), ["cand-001", "cand-002", "cand-003"])
        self.assertTrue(all(d.outcome is Outcome.INVALID and d.stage is DecisionStage.SCREEN for d in closed))
        self.assertEqual(len(self.state().candidates), 6, "the interrupted ones count, and a fresh batch followed")
        self.assertEqual(self.state().malformed, [])

    def test_a_half_decided_round_keeps_its_round_index_when_its_leftover_is_closed(self):
        """Two screening losers are decided (round 0), then the machine dies during the winner's confirmation. The winner is the same batch:
        closing it must not make a second 'rejected round' (stop_after_rejections=2 would then end the search after one real batch)."""
        rule = make_rule(max_candidates=9, screen_batch_size=3, stop_after_rejections=2)
        self.build(value_fn=values({"cand-002": 0.75}), script=[prop(i) for i in range(1, 8)], rule=rule)
        lab = self.lab

        class DiesInConfirmation:
            def run(self, spec):
                if len(lab.executed) >= 5 + 6:  # the five spread trials and the six screening trials are done
                    raise RuntimeError("the machine went away")
                return lab.run(spec)

        with self.assertRaises(RuntimeError):
            self.controller(executor=DiesInConfirmation()).run_round()
        st = self.state()
        self.assertEqual(sorted(st.decisions), ["cand-001", "cand-003"])
        self.assertEqual({d.round_index for d in st.decisions.values()}, {0})
        ds = self.controller().run_round()
        closed = [d for d in ds if "interrupted" in d.reason]
        self.assertEqual([(d.candidate_id, d.round_index) for d in closed], [("cand-002", 0)])
        self.assertEqual(len(self.state().candidates), 6, "the search was not stopped: a second real batch was proposed")
        self.assertEqual(self.state().malformed, [])

    def test_an_executor_that_dies_after_trial_start_leaves_a_trial_the_operator_must_abort(self):
        self.build(value_fn=values({}), script=[prop(1), prop(2), prop(3)])
        lab = self.lab

        class DiesAfterStart:
            def run(self, spec):
                lab.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
                raise RuntimeError("died")

        with self.assertRaises(RuntimeError):
            self.controller(executor=DiesAfterStart()).run_round()
        with self.assertRaisesRegex(OuterRefused, "abort-trial"):
            self.controller().run_round()


class RedactionInTheLoopTests(Base):
    def test_the_proposer_prompt_has_the_outcome_bits_and_none_of_the_numbers(self):
        self.build(value_fn=values({"cand-001": 0.7312}), script=[prop(i) for i in range(1, 7)], rule=make_rule(screen_batch_size=3))
        self.ctl.run_round()
        self.ctl.run_round()  # these proposals are built after the round-1 decisions exist
        text = self.proposer.contexts[-1].user_prompt + self.proposer.contexts[-1].system_prompt
        from nightshift.lab.outer_context import assert_no_forbidden, forbidden_values

        forb = forbidden_values(self.state(), self.lab.layout, ("ax43.example",))
        self.assertTrue(forb)
        assert_no_forbidden(text, forb)  # does not raise
        self.assertIn("outcome ACCEPTED", text)
        self.assertIn("outcome REJECTED", text)
        d = self.state().decisions["cand-001"]
        for secret in (d.delta, d.margin, d.sigma_used, d.screen_delta):
            if secret is not None:
                self.assertNotIn(f"{secret:.4f}", text)
        for word in ("margin", "sigma", "delta", "degrees of freedom"):
            self.assertNotIn(word, self.proposer.contexts[-1].user_prompt.lower())
        self.assertIn("PENDING", self.proposer.contexts[-1].user_prompt, "candidates of the same round are shown as pending")
        for host in ("ax43", "ax41", str(self.lab.layout.root)):
            self.assertNotIn(host, text)

    def test_a_selection_number_in_the_incumbent_pack_stops_the_call(self):
        # the v0 selection scores are 0.6 +/- the wobble; put one of them (to 4 decimals) into the incumbent's notes
        store = InMemoryPackStore({"task.yaml": "task:\n  name: x\n", "notes/seed.md": "# notes\n\n## Log\nlast time we saw 0.5920 on the private set\n"})
        self.build(value_fn=values({}), script=[prop(1)], store=store)
        leaked = {f"{s.score:.4f}" for s in [*self.state().spread.trials]}
        self.assertIn("0.5920", leaked)
        with self.assertRaises(AccessDenied) as cm:
            self.ctl.run_round()
        self.assertNotIn("0.5920", str(cm.exception), "the error must not print the number")
        self.assertEqual(self.proposer.contexts, [], "nothing was sent")

    def test_a_held_out_assignment_id_in_the_pack_stops_the_call(self):
        store = InMemoryPackStore({"task.yaml": "task:\n  name: x\n", "notes/seed.md": "# notes\n\n## Log\nsee assignment a-held-1\n"})
        self.build(value_fn=values({}), script=[prop(1)], store=store)
        with self.assertRaises(AccessDenied):
            self.ctl.run_round()
        self.assertEqual(self.proposer.contexts, [])

    def test_dev_scores_and_budget_are_shown_at_two_decimals(self):
        self.build(value_fn=values({}), script=[prop(i) for i in range(1, 7)], rule=make_rule(screen_batch_size=3))
        self.ctl.run_round()
        self.ctl.run_round()
        text = self.proposer.contexts[-1].user_prompt
        self.assertIn("v0: mean best dev score 0.70 over 5 trials", text)
        self.assertIn("researcher model tokens: 2000000", text)


class EvidenceTests(Base):
    def test_trial_evidence_comes_from_the_trial_ledger_notes_and_trace(self):
        from nightshift.lab.contracts import FileRef, GpuUsage, JobStatus, TrainResult

        self.build(value_fn=values({}), script=[prop(1)])
        st = self.state()
        tid = next(iter(st.trials))
        rec = st.trials[tid]
        lay = self.lab.layout
        d = lay.trial_dir(tid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "trace").mkdir(exist_ok=True)
        (d / "trace" / "agent.log").write_text(
            "\n".join(json.dumps({"type": "tool_use", "part": {"type": "tool", "tool": t}}) for t in ["bash", "bash", "read", "bash"]) + "\nnot json\n",
            encoding="utf-8",
        )
        (d / "notes.md").write_text("job 1 failed: CUDA error: out of memory\nthe adapter did not load\n", encoding="utf-8")
        ok = TrainResult("j-0001", tid, JobStatus.OK, rec.adapter.adapter_id, rec.adapter, rec.dev_score_id, GpuUsage(10.0, 1.0, ("r1",)), {}, None, FileRef("trials/x/td.jsonl", "a" * 64, 5), None)
        bad = TrainResult("j-0002", tid, JobStatus.FAILED, None, None, None, GpuUsage(5.0, 0.3, ("r2",)), {}, None, None, "CUDA error: out of memory")
        with open(d / "ledger.jsonl", "w", encoding="utf-8") as fh:
            for r in (ok, bad):
                fh.write(json.dumps({"seq": 1, "kind": LabKind.GPU_JOB_FINISHED, "data": r.to_dict()}) + "\n")
        rec2 = dataclasses.replace(rec, trace=lay.file_ref(d / "trace" / "agent.log"), notes=lay.file_ref(d / "notes.md"))
        ev = self.ctl._evidence(st, rec2)
        self.assertEqual((ev.n_jobs, ev.n_jobs_ok), (2, 1))
        self.assertEqual(ev.tool_counts, (("bash", 3), ("read", 1)))
        self.assertIn("out of memory", ev.failure_signatures)
        self.assertIn("adapter did not load", ev.failure_signatures)
        self.assertEqual(len(ev.dev_trajectory), 1)
        self.assertAlmostEqual(ev.gpu_busy_fraction, 15 * 60 / 1800.0)
        self.assertIn("did not load", ev.notes_excerpt)
        # a planted symlink in the trial directory is never followed
        link = d / "notes_link.md"
        link.symlink_to(Path(self.lab.layout.root) / "ledger.jsonl")
        with self.assertRaises(Exception):
            lay.file_ref(link)
        ev2 = self.ctl._evidence(st, dataclasses.replace(rec, notes=FileRef("trials/%s/notes_link.md" % tid, "a" * 64, 5)))
        self.assertEqual(ev2.notes_excerpt, "")


class RealPackRepoTests(Base):
    """The controller over the real git-backed PackRepo and the real v0 template (strict apply, real lint, accept = fast-forward)."""

    def setUp(self):
        import shutil

        if shutil.which("git") is None:
            self.skipTest("git is not installed")

    def build_real(self, script, value_fn):
        from nightshift.lab.packs import PackRepo

        self.build(value_fn=value_fn, script=script, store_factory=lambda lay: PackRepo(lay, lint_ranges_path=lay.root / "lint_ranges.json", clock=lambda: 1_793_000_000.0))
        self.git = lambda *a: __import__("subprocess").run(["git", "--git-dir", str(self.lab.layout.pack_repo), *a], capture_output=True, text=True, check=True).stdout.strip()

    def test_a_good_edit_wins_a_bad_edit_fails_lint_and_main_follows_the_ledger(self):
        from nightshift.lab.contracts import EditOp, Lever, PackEdit, Proposal, SearchReplace

        good = make_proposal("notes/seed.md", "## Log", "## Log (newest first)", summary="newest first")
        model = Proposal("h", Lever.BUDGET_USE, "use a bigger model", "better", "cost",
                         (PackEdit("task.yaml", EditOp.SEARCH_REPLACE, None, (SearchReplace('model: "muse/muse-spark-1.3"', 'model: "meta/muse-wonder-1.0"'),)),))
        other = make_proposal("seed/recipe/run.sh", 'SAMPLES="${SAMPLES:-4}"', 'SAMPLES="${SAMPLES:-6}"', summary="six samples")
        self.build_real([good, model, other], values({"cand-001": 0.75}))
        ds = {d.candidate_id: d for d in self.ctl.run_round()}
        self.assertEqual(ds["cand-001"].outcome, Outcome.ACCEPTED)
        self.assertEqual(ds["cand-002"].outcome, Outcome.INVALID)
        self.assertEqual(ds["cand-003"].outcome, Outcome.REJECTED)
        st = self.state()
        self.assertTrue(any("agents.model" in m and "not editable" in m for m in st.candidates["cand-002"].lint_messages), st.candidates["cand-002"].lint_messages)
        c1 = st.candidates["cand-001"]
        self.assertTrue(c1.lint_passed)
        self.assertEqual(self.git("rev-parse", "main"), c1.pack.commit, "main follows the accepted candidate")
        self.assertEqual(self.git("rev-parse", "pack/cand-002"), st.candidates["cand-002"].pack.commit, "an invalid candidate stays tagged")
        self.assertEqual(self.git("rev-parse", "pack/v0"), self.lab.v0.commit)
        diff = (self.lab.layout.root / c1.diff.path).read_text()
        self.assertIn("+## Log (newest first)", diff)
        self.assertEqual(st.packs["cand-001"].pack, c1.pack)
        self.assertEqual(st.malformed, [])
        # the next round starts from the accepted pack: the proposer sees its text
        self.proposer._script.extend([prop(4), prop(5), prop(6)])
        self.ctl.run_round()
        self.assertIn("## Log (newest first)", self.proposer.contexts[-1].user_prompt)
        self.assertIn("version cand-001", self.proposer.contexts[-1].user_prompt)


if __name__ == "__main__":
    unittest.main()
