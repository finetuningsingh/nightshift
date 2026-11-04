"""Tests for nightshift.lab.stats (owner 3).

Hand-computed expectations are written out from the formulas in docs/harness-design.md, "Statistics", with the t values taken from the
reused ``nightshift.grading`` tables (T95 / T975), never from the module under test. The false-accept simulations drive the REAL
``trial_spread`` / ``screen`` / ``select_screen_winner`` / ``decide`` over simulated trials; only the trial scores are random.

Run:  PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_stats
Print the simulation numbers at higher precision:  PYTHONDONTWRITEBYTECODE=1 python3.13 -m tests.test_lab_stats --simulate [reps]
"""

import dataclasses
import itertools
import math
import random
import sys
import unittest

from nightshift import grading
from nightshift.lab import contracts as c
from nightshift.lab import stats
from nightshift.lab.contracts import (
    Arm,
    AssignmentWeight,
    ContractError,
    DecisionRule,
    FinalAnalysis,
    GenerationParams,
    Outcome,
    PlannedTrial,
    Rule,
    SpreadEstimate,
    Split,
    TrialScore,
)

NOW = "2026-11-01T12:00:00+00:00"
KEY = "a" * 64
KEY2 = "b" * 64


def ts(trial, assignment, score, key=KEY, metric="prompt_all", scorer="s1"):
    return TrialScore(trial, assignment, None, metric, scorer, key, score)


def make_rule(**kw):
    base = dict(
        metric="prompt_all",
        split=Split.SELECTION,
        scorer_version="s1",
        assignments=("a-evo-1",),
        spread=SpreadEstimate(0.02, 4, ("v1", "v2", "v3", "v4", "v5"), 1),
        alpha=0.05,
        min_gain=0.0,
        score_seeds=(0, 1, 2),
        score_params=GenerationParams.recommended(8000),
        screen_batch_size=5,
        screen_trials_per_candidate=2,
        confirm_trials_per_candidate=3,
        confirm_trials_per_incumbent=3,
        variance_guard_ratio=1.5,
        failed_trial_policy=c.FailedTrialPolicy.INVALIDATES_CANDIDATE,
        assumed_gain=0.06,
        minimum_detectable_gain=0.05,
        degenerate_override=None,
        max_candidates=10,
        stop_after_rejections=None,
        repeated_decisions_note="Screen then confirm; a heuristic filter, not evidence of a gain.",
        frozen_utc=NOW,
    )
    base.update(kw)
    return Rule(**base)


def two_assignment_rule(**kw):
    base = dict(
        assignments=("a-evo-1", "a-evo-2"),
        spread=SpreadEstimate(0.02, 4, ("v1", "v2", "v3", "v4", "v5", "v6"), 2),
    )
    base.update(kw)
    return make_rule(**base)


# t values from the reused nightshift.grading tables, so they are independent of stats.t_quantile
T95_8 = grading.T95[8]  # 1.860 (3 decimals in the table)
T95_12 = grading.T95[12]  # 1.7823


# --------------------------------------------------------------------------------------------------
class TQuantileTests(unittest.TestCase):
    def test_matches_the_reused_tables_at_their_tabulated_df(self):
        for df in sorted(set(grading.T95)):
            tol = 6e-4 if df <= 10 else 6e-5
            with self.subTest(table="T95", df=df):
                self.assertAlmostEqual(stats.t_quantile(0.05, df), grading.T95[df], delta=tol)
        for df in sorted(set(grading.T975)):
            tol = 6e-4 if df <= 10 else 6e-5
            with self.subTest(table="T975", df=df):
                self.assertAlmostEqual(stats.t_quantile(0.025, df), grading.T975[df], delta=tol)

    def test_known_exact_values_and_symmetry(self):
        self.assertAlmostEqual(stats.t_quantile(0.025, 1), 12.7062047, places=5)
        self.assertAlmostEqual(stats.t_quantile(0.005, 10), 3.169273, places=5)
        self.assertAlmostEqual(stats.t_quantile(0.10, 4), 1.533206, places=5)
        self.assertAlmostEqual(stats.t_quantile(0.05, 1e7), 1.644854, places=4)  # normal limit
        self.assertEqual(stats.t_quantile(0.5, 7), 0.0)
        self.assertAlmostEqual(stats.t_quantile(0.95, 8), -stats.t_quantile(0.05, 8), places=12)

    def test_non_integer_df_is_between_its_neighbours(self):
        lo, mid, hi = stats.t_quantile(0.05, 4), stats.t_quantile(0.05, 4.5), stats.t_quantile(0.05, 5)
        self.assertGreater(lo, mid)
        self.assertGreater(mid, hi)

    def test_rejects_bad_arguments(self):
        for alpha, df in ((0.0, 5), (1.0, 5), (-0.1, 5), (float("nan"), 5), (0.05, 0), (0.05, -3), (0.05, float("inf")), (True, 5)):
            with self.subTest(alpha=alpha, df=df):
                with self.assertRaises(ContractError):
                    stats.t_quantile(alpha, df)


# --------------------------------------------------------------------------------------------------
class TrialSpreadTests(unittest.TestCase):
    def test_pooled_within_cell_variance_and_df(self):
        # cell A: 0.70 0.72 0.74 (ss 0.0008, n 3); cell B: 0.60 0.64 (ss 0.0008, n 2) -> df 3, sigma^2 = 0.0016/3
        scores = [ts("t1", "A", 0.70), ts("t2", "A", 0.72), ts("t3", "A", 0.74), ts("t4", "B", 0.60), ts("t5", "B", 0.64)]
        est = stats.trial_spread(scores)
        self.assertEqual(est.df, 3)
        self.assertEqual(est.n_assignments, 2)
        self.assertAlmostEqual(est.sigma, math.sqrt(0.0016 / 3), places=12)
        self.assertEqual(set(est.trial_ids), {"t1", "t2", "t3", "t4", "t5"})

    def test_a_cell_with_one_trial_is_excluded(self):
        scores = [ts("t1", "A", 0.70), ts("t2", "A", 0.74), ts("t3", "B", 0.10)]
        est = stats.trial_spread(scores)
        self.assertEqual(est.trial_ids, ("t1", "t2"))
        self.assertEqual((est.df, est.n_assignments), (1, 1))
        self.assertAlmostEqual(est.sigma, 0.04 / math.sqrt(2), places=12)  # sd of {0.70, 0.74}
        self.assertEqual(est.df, len(est.trial_ids) - est.n_assignments)

    def test_too_few_trials(self):
        for scores in ([], [ts("t1", "A", 0.5)], [ts("t1", "A", 0.5), ts("t2", "B", 0.6)]):
            with self.subTest(n=len(scores)):
                with self.assertRaisesRegex(ContractError, "too few trials to estimate spread"):
                    stats.trial_spread(scores)

    def test_duplicate_trial_ids_are_refused(self):
        with self.assertRaises(ContractError):
            stats.trial_spread([ts("t1", "A", 0.5), ts("t1", "A", 0.6)])

    def test_identical_trials_give_sigma_zero(self):
        self.assertEqual(stats.trial_spread([ts("t1", "A", 0.5), ts("t2", "A", 0.5)]).sigma, 0.0)

    def test_input_order_does_not_matter(self):
        scores = [ts(f"t{i}", "A" if i % 2 else "B", 0.5 + 0.013 * ((i * 7) % 5)) for i in range(1, 9)]
        a = stats.trial_spread(scores)
        b = stats.trial_spread(list(reversed(scores)))
        self.assertEqual(a, b)


# --------------------------------------------------------------------------------------------------
class DecideTests(unittest.TestCase):
    def setUp(self):
        self.rule = make_rule()
        self.inc = [ts("i1", "a-evo-1", 0.60), ts("i2", "a-evo-1", 0.62), ts("i3", "a-evo-1", 0.64)]
        self.cand = [ts("c1", "a-evo-1", 0.70), ts("c2", "a-evo-1", 0.72), ts("c3", "a-evo-1", 0.74)]

    def test_margin_df_8_hand_computed(self):
        # ss_c = ss_i = 0.0008; sigma_used^2 = (4*0.02^2 + 0.0008 + 0.0008)/8 = 0.0004; df = 4 + 2 + 2 = 8
        d = stats.decide(self.rule, self.inc, self.cand)
        self.assertEqual(d.outcome, Outcome.ACCEPTED)
        self.assertEqual(d.df, 8)
        self.assertAlmostEqual(d.sigma_used, 0.02, places=12)
        self.assertAlmostEqual(d.delta, 0.10, places=12)
        se_factor = math.sqrt(1 / 3 + 1 / 3)
        self.assertAlmostEqual(d.margin, T95_8 * 0.02 * se_factor, delta=0.02 * se_factor * 6e-4)  # table has 3 decimals
        self.assertAlmostEqual(d.margin, 1.8595480 * 0.02 * se_factor, places=6)  # exact t(0.95, 8)
        self.assertGreater(d.delta, d.margin)

    def test_min_gain_is_added_to_the_margin(self):
        base = stats.decide(self.rule, self.inc, self.cand)
        with_gain = stats.decide(make_rule(min_gain=0.05), self.inc, self.cand)
        self.assertAlmostEqual(with_gain.margin, base.margin + 0.05, places=12)
        self.assertEqual(with_gain.outcome, Outcome.ACCEPTED)  # 0.10 > 0.0804
        self.assertEqual(stats.decide(make_rule(min_gain=0.075), self.inc, self.cand).outcome, Outcome.REJECTED)

    def test_reject_when_delta_is_below_the_margin(self):
        cand = [ts("c1", "a-evo-1", 0.61), ts("c2", "a-evo-1", 0.63), ts("c3", "a-evo-1", 0.65)]
        d = stats.decide(self.rule, self.inc, cand)
        self.assertEqual(d.outcome, Outcome.REJECTED)
        self.assertAlmostEqual(d.delta, 0.01, places=12)
        self.assertLess(d.delta, d.margin)

    def test_the_boundary_is_strict(self):
        d0 = stats.decide(self.rule, self.inc, self.cand)
        # shift the candidate so that delta equals the margin as closely as the floats allow, then one ulp-ish either side
        shift = d0.margin - d0.delta
        just_above = [dataclasses.replace(t, score=t.score + shift + 1e-6) for t in self.cand]
        just_below = [dataclasses.replace(t, score=t.score + shift - 1e-6) for t in self.cand]
        self.assertEqual(stats.decide(self.rule, self.inc, just_above).outcome, Outcome.ACCEPTED)
        self.assertEqual(stats.decide(self.rule, self.inc, just_below).outcome, Outcome.REJECTED)

    def test_an_exact_tie_is_not_an_accept(self):
        # sigma 0 and every trial identical: margin = min_gain = 0.25 and delta = 0.75 - 0.5 = 0.25 exactly (binary fractions)
        rule = make_rule(spread=SpreadEstimate(0.0, 4, ("v1", "v2", "v3", "v4", "v5"), 1), min_gain=0.25, minimum_detectable_gain=0.25, assumed_gain=0.3)
        inc = [ts(f"i{k}", "a-evo-1", 0.5) for k in range(3)]
        cand = [ts(f"c{k}", "a-evo-1", 0.75) for k in range(3)]
        d = stats.decide(rule, inc, cand)
        self.assertEqual((d.delta, d.margin), (0.25, 0.25))
        self.assertEqual(d.outcome, Outcome.REJECTED)
        cand = [ts(f"c{k}", "a-evo-1", 0.76) for k in range(3)]
        self.assertEqual(stats.decide(rule, inc, cand).outcome, Outcome.ACCEPTED)

    def test_the_pooled_sigma_grows_with_the_confirmation_variance(self):
        mild = [ts("c1", "a-evo-1", 0.705), ts("c2", "a-evo-1", 0.72), ts("c3", "a-evo-1", 0.735)]  # sd 0.015, inside the guard
        d = stats.decide(self.rule, self.inc, mild)
        ss_c = 2 * 0.015**2
        expect = math.sqrt((4 * 0.02**2 + ss_c + 0.0008) / 8)
        self.assertAlmostEqual(d.sigma_used, expect, places=12)

    def test_larger_pooled_df_is_hand_computed(self):
        # 13 v0 trials (df 12) with 4 + 4 confirmation trials: df = 12 + 3 + 3 = 18, t(0.95, 18) = 1.7341 from the reused table
        rule = make_rule(spread=SpreadEstimate(0.02, 12, tuple(f"v{k}" for k in range(1, 14)), 1), confirm_trials_per_candidate=4, confirm_trials_per_incumbent=4)
        inc = [ts(f"i{k}", "a-evo-1", v) for k, v in enumerate([0.58, 0.60, 0.62, 0.64])]  # ss = 0.0020
        cand = [ts(f"c{k}", "a-evo-1", v) for k, v in enumerate([0.68, 0.70, 0.72, 0.74])]  # ss = 0.0020, sd 0.0258 < 0.03
        d = stats.decide(rule, inc, cand)
        self.assertEqual(d.df, 18)
        sigma_used = math.sqrt((12 * 0.02**2 + 0.0020 + 0.0020) / 18)
        self.assertAlmostEqual(d.sigma_used, sigma_used, places=12)
        se_factor = math.sqrt(1 / 4 + 1 / 4)
        self.assertAlmostEqual(d.margin, grading.T95[18] * sigma_used * se_factor, delta=sigma_used * se_factor * 6e-5)
        self.assertAlmostEqual(d.delta, 0.10, places=12)

    def test_two_assignments_df_and_se_factor(self):
        rule = two_assignment_rule()
        inc = [ts(f"i{a}{k}", f"a-evo-{a}", 0.60 + 0.02 * k, key=KEY if a == 1 else KEY2) for a in (1, 2) for k in range(3)]
        cand = [ts(f"c{a}{k}", f"a-evo-{a}", 0.70 + 0.02 * k, key=KEY if a == 1 else KEY2) for a in (1, 2) for k in range(3)]
        d = stats.decide(rule, inc, cand)
        # df = 4 + 2 cells * 2 + 2 cells * 2 = 12; each cell ss = 0.0008; sigma_used^2 = (4*0.0004 + 4*0.0008)/12 = 0.0004
        self.assertEqual(d.df, 12)
        self.assertAlmostEqual(d.sigma_used, 0.02, places=12)
        se_factor = 0.5 * math.sqrt(2 * (1 / 3 + 1 / 3))
        self.assertAlmostEqual(d.margin, T95_12 * 0.02 * se_factor, delta=0.02 * se_factor * 6e-5)
        self.assertAlmostEqual(d.delta, 0.10, places=12)

    def test_delta_is_the_unweighted_mean_over_assignments(self):
        rule = two_assignment_rule()
        inc = [ts(f"i{a}{k}", f"a-evo-{a}", 0.60) for a in (1, 2) for k in range(3)]
        cand = [ts(f"c1{k}", "a-evo-1", 0.70) for k in range(3)] + [ts(f"c2{k}", "a-evo-2", 0.62) for k in range(3)]
        self.assertAlmostEqual(stats.decide(rule, inc, cand).delta, (0.10 + 0.02) / 2, places=12)

    def test_input_order_does_not_change_the_numbers(self):
        a = stats.decide(self.rule, self.inc, self.cand)
        b = stats.decide(self.rule, list(reversed(self.inc)), list(reversed(self.cand)))
        self.assertEqual((a.delta, a.margin, a.sigma_used, a.df), (b.delta, b.margin, b.sigma_used, b.df))

    def test_decision_numbers_round_trip_through_the_decision_contract(self):
        # the contract refuses a Decision whose outcome contradicts delta > margin; build one from the numbers
        d = stats.decide(self.rule, self.inc, self.cand)
        c.Decision(
            candidate_id="cand-001",
            rule_sha256=self.rule.sha256(),
            stage=c.DecisionStage.CONFIRM,
            round_index=0,
            outcome=d.outcome,
            reason=d.reason,
            incumbent_version="v0",
            candidate_version="cand-001",
            incumbent_scores=tuple(self.inc),
            candidate_scores=tuple(self.cand),
            screen_delta=0.05,
            delta=d.delta,
            margin=d.margin,
            df=d.df,
            sigma_used=d.sigma_used,
            cost=c.CostBreakdown(0.1, 1.0, 1.0, 0.1),
            decided_utc=NOW,
        )

    def assertInvalid(self, d, text):
        self.assertEqual(d.outcome, Outcome.INVALID)
        self.assertIsNone(d.delta)
        self.assertIsNone(d.margin)
        self.assertIsNone(d.df)
        self.assertIsNone(d.sigma_used)
        self.assertIn(text, d.reason)

    def test_invalid_comparability_mismatch_within_an_assignment(self):
        cand = [ts("c1", "a-evo-1", 0.7), ts("c2", "a-evo-1", 0.72, key=KEY2), ts("c3", "a-evo-1", 0.74)]
        self.assertInvalid(stats.decide(self.rule, self.inc, cand), "comparability")
        inc = [ts("i1", "a-evo-1", 0.6, key=KEY2)] + self.inc[1:]
        self.assertInvalid(stats.decide(self.rule, inc, self.cand), "comparability")

    def test_comparability_keys_may_differ_between_assignments(self):
        rule = two_assignment_rule()
        inc = [ts(f"i{a}{k}", f"a-evo-{a}", 0.60 + 0.01 * k, key=KEY if a == 1 else KEY2) for a in (1, 2) for k in range(3)]
        cand = [ts(f"c{a}{k}", f"a-evo-{a}", 0.70 + 0.01 * k, key=KEY if a == 1 else KEY2) for a in (1, 2) for k in range(3)]
        self.assertNotEqual(stats.decide(rule, inc, cand).outcome, Outcome.INVALID)

    def test_invalid_trial_counts(self):
        self.assertInvalid(stats.decide(self.rule, self.inc, self.cand[:2]), "candidate trials")
        self.assertInvalid(stats.decide(self.rule, self.inc[:2], self.cand), "incumbent trials")
        extra = self.cand + [ts("c4", "a-evo-1", 0.7)]
        self.assertInvalid(stats.decide(self.rule, self.inc, extra), "candidate trials")

    def test_invalid_missing_assignment_on_either_side(self):
        rule = two_assignment_rule()
        inc = [ts(f"i1{k}", "a-evo-1", 0.60 + 0.01 * k) for k in range(3)]
        cand = [ts(f"c1{k}", "a-evo-1", 0.70 + 0.01 * k) for k in range(3)] + [ts(f"c2{k}", "a-evo-2", 0.7) for k in range(3)]
        self.assertInvalid(stats.decide(rule, inc, cand), "a-evo-2 has no incumbent")
        self.assertInvalid(stats.decide(rule, cand, inc), "a-evo-2 has no candidate")

    def test_invalid_empty_sides_and_foreign_assignment(self):
        self.assertInvalid(stats.decide(self.rule, [], self.cand), "no incumbent")
        self.assertInvalid(stats.decide(self.rule, self.inc, []), "no candidate")
        stray = self.cand[:2] + [ts("c3", "a-held-1", 0.74)]
        self.assertInvalid(stats.decide(self.rule, self.inc, stray), "not a rule assignment")

    def test_invalid_metric_and_scorer_version(self):
        cand = [dataclasses.replace(t, metric="prompt_fraction") for t in self.cand]
        self.assertInvalid(stats.decide(self.rule, self.inc, cand), "metric")
        cand = [dataclasses.replace(t, scorer_version="s2") for t in self.cand]
        self.assertInvalid(stats.decide(self.rule, self.inc, cand), "scorer version")

    def test_invalid_when_a_trial_is_used_twice(self):
        cand = [self.inc[0]] + self.cand[1:]
        self.assertInvalid(stats.decide(self.rule, self.inc, cand), "twice")

    def test_variance_guard(self):
        # frozen sigma 0.02 x ratio 1.5 = 0.03: a candidate trial sd of 0.04 is INVALID however large its delta
        wild = [ts("c1", "a-evo-1", 0.80), ts("c2", "a-evo-1", 0.88), ts("c3", "a-evo-1", 0.96)]
        self.assertInvalid(stats.decide(self.rule, self.inc, wild), "variance guard")
        edge_ok = [ts("c1", "a-evo-1", 0.70), ts("c2", "a-evo-1", 0.725), ts("c3", "a-evo-1", 0.75)]  # sd 0.025 < 0.03
        self.assertNotEqual(stats.decide(self.rule, self.inc, edge_ok).outcome, Outcome.INVALID)
        # the incumbent's own variance is not guarded (its sd feeds the pooled sigma instead)
        noisy_inc = [ts("i1", "a-evo-1", 0.50), ts("i2", "a-evo-1", 0.60), ts("i3", "a-evo-1", 0.70)]
        self.assertNotEqual(stats.decide(self.rule, noisy_inc, self.cand).outcome, Outcome.INVALID)

    def test_a_looser_ratio_lets_the_noisy_candidate_through(self):
        wild = [ts("c1", "a-evo-1", 0.80), ts("c2", "a-evo-1", 0.88), ts("c3", "a-evo-1", 0.96)]
        self.assertNotEqual(stats.decide(make_rule(variance_guard_ratio=5.0), self.inc, wild).outcome, Outcome.INVALID)

    def test_never_raises_for_data_problems(self):
        weird = [
            ([], []),
            ([ts("i", "zzz", 0.1)], [ts("c", "zzz", 0.2)]),
            (self.inc, self.inc),
            (self.inc * 2, self.cand),
        ]
        for inc, cand in weird:
            with self.subTest(n=(len(inc), len(cand))):
                self.assertEqual(stats.decide(self.rule, inc, cand).outcome, Outcome.INVALID)


# --------------------------------------------------------------------------------------------------
class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.rule = make_rule(screen_trials_per_candidate=2)
        self.inc = [ts("i1", "a-evo-1", 0.60), ts("i2", "a-evo-1", 0.62), ts("i3", "a-evo-1", 0.64), ts("i4", "a-evo-1", 0.58)]

    def test_delta_is_candidate_mean_minus_incumbent_mean(self):
        cands = {
            "cand-001": [ts("c1", "a-evo-1", 0.70), ts("c2", "a-evo-1", 0.72)],
            "cand-002": [ts("c3", "a-evo-1", 0.50), ts("c4", "a-evo-1", 0.52)],
        }
        out = stats.screen(self.rule, self.inc, cands)
        self.assertAlmostEqual(out["cand-001"], 0.71 - 0.61, places=12)
        self.assertAlmostEqual(out["cand-002"], 0.51 - 0.61, places=12)  # negative deltas are returned, there is no threshold
        self.assertEqual(list(out), ["cand-001", "cand-002"])

    def test_two_assignments_average_the_per_assignment_deltas(self):
        rule = two_assignment_rule(screen_trials_per_candidate=2)
        inc = [ts("i11", "a-evo-1", 0.60), ts("i12", "a-evo-1", 0.60), ts("i21", "a-evo-2", 0.50), ts("i22", "a-evo-2", 0.50)]
        cand = [ts("c11", "a-evo-1", 0.70), ts("c12", "a-evo-1", 0.70), ts("c21", "a-evo-2", 0.50), ts("c22", "a-evo-2", 0.50)]
        self.assertAlmostEqual(stats.screen(rule, inc, {"k": cand})["k"], 0.05, places=12)

    def test_none_when_not_comparable(self):
        good = [ts("c1", "a-evo-1", 0.7), ts("c2", "a-evo-1", 0.7)]
        cands = {
            "wrong-count": good[:1],
            "too-many": good + [ts("c9", "a-evo-1", 0.7)],
            "key-mismatch": [ts("c3", "a-evo-1", 0.7), ts("c4", "a-evo-1", 0.7, key=KEY2)],
            "other-assignment": [ts("c5", "a-held-1", 0.7), ts("c6", "a-held-1", 0.7)],
            "metric": [ts("c7", "a-evo-1", 0.7, metric="prompt_fraction"), ts("c8", "a-evo-1", 0.7, metric="prompt_fraction")],
            "good": good,
        }
        out = stats.screen(self.rule, self.inc, cands)
        self.assertIsNotNone(out["good"])
        for k in cands:
            if k != "good":
                self.assertIsNone(out[k], k)

    def test_none_for_everyone_when_the_incumbent_is_missing_an_assignment(self):
        rule = two_assignment_rule(screen_trials_per_candidate=2)
        cand = [ts("c11", "a-evo-1", 0.7), ts("c12", "a-evo-1", 0.7), ts("c21", "a-evo-2", 0.7), ts("c22", "a-evo-2", 0.7)]
        self.assertIsNone(stats.screen(rule, self.inc, {"k": cand})["k"])

    def test_the_incumbent_may_have_any_number_of_trials(self):
        for n in (1, 2, 7):
            inc = [ts(f"i{k}", "a-evo-1", 0.6) for k in range(n)]
            self.assertIsNotNone(stats.screen(self.rule, inc, {"k": [ts("c1", "a-evo-1", 0.7), ts("c2", "a-evo-1", 0.7)]})["k"])

    def test_select_screen_winner(self):
        sel = stats.select_screen_winner
        self.assertEqual(sel({"cand-001": 0.01, "cand-002": 0.05, "cand-003": -0.2}), "cand-002")
        self.assertEqual(sel({"cand-002": 0.05, "cand-001": 0.05}), "cand-001")  # ties go to the smaller id
        self.assertIsNone(sel({"cand-001": -0.01, "cand-002": 0.0}))  # delta must be > 0
        self.assertEqual(sel({"cand-001": None, "cand-002": 0.001}), "cand-002")
        self.assertIsNone(sel({"cand-001": None}))
        self.assertIsNone(sel({}))


# --------------------------------------------------------------------------------------------------
class MinimumDetectableGainTests(unittest.TestCase):
    def test_large_df_matches_the_textbook_normal_formula(self):
        got = stats.minimum_detectable_gain(0.02, 10_000_000, 3, 3, 1, 0.05)
        want = (1.6448536 + 0.8416212) * 0.02 * math.sqrt(2 / 3)
        self.assertAlmostEqual(got, want, places=5)

    def test_small_df_is_larger_than_the_normal_formula(self):
        got = stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05)
        normal = (1.6448536 + 0.8416212) * 0.02 * math.sqrt(2 / 3)
        self.assertGreater(got, normal * 1.05)
        self.assertAlmostEqual(got, 0.04453, places=4)  # df 8: ncp 2.727, computed independently by simulation below

    def test_matches_a_monte_carlo_power(self):
        rng = random.Random(5)
        sigma, df0, nc, ni = 0.02, 4, 3, 3
        gain = stats.minimum_detectable_gain(sigma, df0 + (nc - 1) + (ni - 1), nc, ni, 1, 0.05)
        tq = grading.T95[8]
        acc = 0
        reps = 40_000
        for _ in range(reps):
            s0 = sum(rng.gauss(0, sigma) ** 2 for _ in range(df0))  # frozen part: sigma^2 chi2_4
            cand = [rng.gauss(gain, sigma) for _ in range(nc)]
            inc = [rng.gauss(0, sigma) for _ in range(ni)]
            ss = s0 + sum((x - sum(cand) / nc) ** 2 for x in cand) + sum((x - sum(inc) / ni) ** 2 for x in inc)
            sig = math.sqrt(ss / 8)
            acc += (sum(cand) / nc - sum(inc) / ni) > tq * sig * math.sqrt(1 / nc + 1 / ni)
        self.assertAlmostEqual(acc / reps, 0.8, delta=0.012)

    def test_more_trials_or_assignments_detect_smaller_gains(self):
        base = stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05)
        self.assertLess(stats.minimum_detectable_gain(0.02, 8, 5, 5, 1, 0.05), base)
        self.assertLess(stats.minimum_detectable_gain(0.02, 8, 3, 3, 2, 0.05), base)
        self.assertAlmostEqual(stats.minimum_detectable_gain(0.04, 8, 3, 3, 1, 0.05), 2 * base, places=9)  # linear in sigma

    def test_min_gain_shifts_it_and_zero_sigma_returns_min_gain(self):
        base = stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05)
        self.assertAlmostEqual(stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05, min_gain=0.01), base + 0.01, places=12)
        self.assertEqual(stats.minimum_detectable_gain(0.0, 8, 3, 3, 1, 0.05), 0.0)

    def test_higher_power_needs_a_larger_gain(self):
        self.assertGreater(stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05, 0.9), stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05, 0.8))

    def test_from_se_agrees_with_the_search_form(self):
        se = 0.02 * math.sqrt(2 / 3)
        self.assertAlmostEqual(stats.minimum_detectable_gain_from_se(se, 8, 0.05), stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05), places=12)

    def test_bad_arguments(self):
        for args in ((0.02, 8, 0, 3, 1, 0.05), (0.02, 8, 3, 3, 0, 0.05), (-1.0, 8, 3, 3, 1, 0.05), (0.02, 8, 3, 3, 1, 1.5), (0.02, 8, 3.0, 3, 1, 0.05)):
            with self.subTest(args=args):
                with self.assertRaises(ContractError):
                    stats.minimum_detectable_gain(*args)
        with self.assertRaises(ContractError):
            stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05, 1.0)

    def test_the_rule_contract_accepts_the_computed_value(self):
        mdg = stats.minimum_detectable_gain(0.02, 8, 3, 3, 1, 0.05)
        make_rule(minimum_detectable_gain=mdg, assumed_gain=mdg * 1.01)
        with self.assertRaises(ContractError):
            make_rule(minimum_detectable_gain=mdg, assumed_gain=mdg * 0.99)


# --------------------------------------------------------------------------------------------------
def analysis(n_assignments=1, per_cell=5, weights=None, **kw):
    ids = ["a-held-1", "a-held-2", "a-held-3"][:n_assignments]
    planned = tuple(PlannedTrial(f"f-{arm.value}-{aid}-{k}", arm, aid, 100 + k) for arm in (Arm.V0, Arm.EVOLVED) for aid in ids for k in range(per_cell))
    weights = weights or [1.0 / n_assignments] * n_assignments
    base = dict(
        heldout_metric="prompt_all",
        test_metric="prompt_strict",
        assignments=tuple(AssignmentWeight(i, w) for i, w in zip(ids, weights)),
        planned_trials=planned,
        confidence=0.95,
        method=c.ComparisonMethod.WELCH if n_assignments == 1 else c.ComparisonMethod.BLOCKED_T,
        score_seeds=(0, 1, 2),
        score_params=GenerationParams.recommended(8000),
        failed_trial_policy=c.FailedTrialPolicy.INVALIDATES_CANDIDATE,
        retry_policy="Retry only after an infrastructure failure that revealed no score.",
        budget_sha256="c" * 64,
        frozen_utc=NOW,
    )
    base.update(kw)
    return FinalAnalysis(**base)


def final_scores(an, values, metric=None, key_for=lambda aid: KEY):
    """values: {(arm, assignment): [scores in planned order]} -> (v0 scores, evolved scores)"""
    metric = metric or an.heldout_metric
    out = {Arm.V0: [], Arm.EVOLVED: []}
    for arm in (Arm.V0, Arm.EVOLVED):
        for aid in [w.assignment_id for w in an.assignments]:
            planned = [t for t in an.planned_trials if t.arm is arm and t.assignment_id == aid]
            for t, v in zip(planned, values[(arm, aid)]):
                out[arm].append(ts(t.trial_id, aid, v, key=key_for(aid), metric=metric))
    return out[Arm.V0], out[Arm.EVOLVED]


class CompareFinalTests(unittest.TestCase):
    def test_welch_for_one_assignment_hand_computed(self):
        an = analysis(1, per_cell=5)
        v0v = [0.60, 0.62, 0.64, 0.66, 0.68]  # mean 0.64, var 0.001
        evv = [0.66, 0.68, 0.70, 0.72, 0.74]  # mean 0.70, var 0.001
        v0, ev = final_scores(an, {(Arm.V0, "a-held-1"): v0v, (Arm.EVOLVED, "a-held-1"): evv})
        r = stats.compare_final(an, v0, ev, Split.SELECTION)
        self.assertEqual(r.method, c.ComparisonMethod.WELCH)
        self.assertAlmostEqual(r.delta, 0.06, places=12)
        se = math.sqrt(0.001 / 5 + 0.001 / 5)
        self.assertEqual(r.df, 8.0)  # Welch-Satterthwaite with equal variances and n: 2n-2 = 8
        self.assertAlmostEqual(r.ci_low, 0.06 - grading.T975[8] * se, delta=se * 6e-4)
        self.assertAlmostEqual(r.ci_high, 0.06 + grading.T975[8] * se, delta=se * 6e-4)
        self.assertAlmostEqual(r.mean_v0, 0.64, places=12)
        self.assertAlmostEqual(r.mean_evolved, 0.70, places=12)
        self.assertEqual((r.trials_v0, r.trials_evolved, r.n_assignments), (5, 5, 1))
        self.assertEqual(r.contrast, "evolved_minus_v0")
        self.assertEqual(r.level, 0.95)
        self.assertFalse(r.includes_zero)
        self.assertIn("does not generalise across assignments", r.note)
        self.assertEqual(r.analysis_sha256, an.sha256())
        self.assertEqual(r.metric, "prompt_all")
        self.assertEqual(r.split, Split.SELECTION)

    def test_welch_unequal_variance_floors_the_df(self):
        an = analysis(1, per_cell=5)
        v0v = [0.60, 0.62, 0.64, 0.66, 0.68]
        evv = [0.50, 0.70, 0.60, 0.80, 0.65]
        v0, ev = final_scores(an, {(Arm.V0, "a-held-1"): v0v, (Arm.EVOLVED, "a-held-1"): evv})
        r = stats.compare_final(an, v0, ev, Split.SELECTION)
        m_e, m_0 = sum(evv) / 5, sum(v0v) / 5
        s_e = sum((x - m_e) ** 2 for x in evv) / 4 / 5
        s_0 = sum((x - m_0) ** 2 for x in v0v) / 4 / 5
        wsdf = (s_e + s_0) ** 2 / (s_e**2 / 4 + s_0**2 / 4)
        self.assertEqual(r.df, float(math.floor(wsdf)))
        self.assertNotEqual(wsdf, r.df)  # it really was floored
        half = grading.T975[int(r.df)] * math.sqrt(s_e + s_0)
        self.assertAlmostEqual(r.ci_high - r.ci_low, 2 * half, delta=0.003)

    def test_blocked_t_for_two_assignments_hand_computed(self):
        an = analysis(2, per_cell=3)
        values = {
            (Arm.V0, "a-held-1"): [0.60, 0.62, 0.64],  # mean 0.62
            (Arm.EVOLVED, "a-held-1"): [0.66, 0.68, 0.70],  # mean 0.68 -> d1 = 0.06
            (Arm.V0, "a-held-2"): [0.50, 0.52, 0.54],  # mean 0.52
            (Arm.EVOLVED, "a-held-2"): [0.52, 0.54, 0.56],  # mean 0.54 -> d2 = 0.02
        }
        v0, ev = final_scores(an, values, key_for=lambda aid: KEY if aid == "a-held-1" else KEY2)
        r = stats.compare_final(an, v0, ev, Split.SELECTION)
        self.assertEqual(r.method, c.ComparisonMethod.BLOCKED_T)
        self.assertAlmostEqual(r.delta, 0.04, places=12)
        se = math.sqrt(((0.06 - 0.04) ** 2 + (0.02 - 0.04) ** 2) / 1 / 2)  # s/sqrt(H) with s^2 = 0.0008, H = 2 -> 0.02
        self.assertAlmostEqual(se, 0.02, places=12)
        self.assertEqual(r.df, 1.0)
        self.assertAlmostEqual(r.ci_low, 0.04 - grading.T975[1] * 0.02, places=3)
        self.assertAlmostEqual(r.ci_high, 0.04 + grading.T975[1] * 0.02, places=3)
        self.assertAlmostEqual(r.mean_v0, 0.57, places=12)
        self.assertAlmostEqual(r.mean_evolved, 0.61, places=12)
        self.assertEqual((r.n_assignments, r.trials_v0, r.trials_evolved), (2, 6, 6))
        self.assertTrue(r.includes_zero)  # 0.04 +- 12.7 * 0.02
        self.assertIn("2 held-out assignments", r.note)

    def test_blocked_t_three_assignments_df_two(self):
        an = analysis(3, per_cell=2)
        diffs = {"a-held-1": 0.06, "a-held-2": 0.02, "a-held-3": 0.10}
        values = {}
        for aid, d in diffs.items():
            values[(Arm.V0, aid)] = [0.50, 0.54]
            values[(Arm.EVOLVED, aid)] = [0.50 + d, 0.54 + d]
        v0, ev = final_scores(an, values, key_for=lambda aid: aid[-1] * 64)
        r = stats.compare_final(an, v0, ev, Split.SELECTION)
        mean_d = 0.06
        s = math.sqrt(sum((d - mean_d) ** 2 for d in diffs.values()) / 2)
        self.assertEqual(r.df, 2.0)
        self.assertAlmostEqual(r.delta, mean_d, places=12)
        self.assertAlmostEqual(r.ci_high - r.delta, grading.T975[2] * s / math.sqrt(3), places=3)

    def test_unequal_weights_use_the_reliability_weighted_standard_error(self):
        an = analysis(2, per_cell=3, weights=[0.75, 0.25])
        values = {
            (Arm.V0, "a-held-1"): [0.60, 0.62, 0.64],
            (Arm.EVOLVED, "a-held-1"): [0.66, 0.68, 0.70],
            (Arm.V0, "a-held-2"): [0.50, 0.52, 0.54],
            (Arm.EVOLVED, "a-held-2"): [0.52, 0.54, 0.56],
        }
        v0, ev = final_scores(an, values, key_for=lambda aid: KEY if aid == "a-held-1" else KEY2)
        r = stats.compare_final(an, v0, ev, Split.SELECTION)
        delta = 0.75 * 0.06 + 0.25 * 0.02
        self.assertAlmostEqual(r.delta, delta, places=12)
        sw2 = 0.75**2 + 0.25**2
        var_w = (0.75 * (0.06 - delta) ** 2 + 0.25 * (0.02 - delta) ** 2) / (1 - sw2)
        self.assertAlmostEqual((r.ci_high - r.ci_low) / 2, grading.T975[1] * math.sqrt(var_w * sw2), delta=0.003)
        self.assertAlmostEqual(r.mean_v0, 0.75 * 0.62 + 0.25 * 0.52, places=12)

    def test_test_split_uses_the_test_metric_and_groups_by_planned_assignment(self):
        an = analysis(2, per_cell=3)
        values = {(Arm.V0, a): [0.30, 0.31, 0.32] for a in ("a-held-1", "a-held-2")}
        values.update({(Arm.EVOLVED, a): [0.33, 0.34, 0.35] for a in ("a-held-1", "a-held-2")})
        v0, ev = final_scores(an, values, metric="prompt_strict", key_for=lambda aid: KEY if aid == "a-held-1" else KEY2)
        # the TrialScore.assignment_id of a test score is not checked against the plan for the test split
        v0 = [dataclasses.replace(t, assignment_id=c.TEST_ASSIGNMENT_ID) for t in v0]
        ev = [dataclasses.replace(t, assignment_id=c.TEST_ASSIGNMENT_ID) for t in ev]
        r = stats.compare_final(an, v0, ev, Split.TEST)
        self.assertEqual(r.metric, "prompt_strict")
        self.assertEqual(r.split, Split.TEST)
        self.assertAlmostEqual(r.delta, 0.03, places=12)
        self.assertEqual(r.n_assignments, 2)

    def test_the_metric_must_match_the_split(self):
        an = analysis(1)
        v0, ev = final_scores(an, {(Arm.V0, "a-held-1"): [0.5] * 5, (Arm.EVOLVED, "a-held-1"): [0.6] * 5})
        with self.assertRaisesRegex(ContractError, "metric"):
            stats.compare_final(an, v0, ev, Split.TEST)

    def test_dev_is_refused(self):
        an = analysis(1)
        with self.assertRaises(ContractError):
            stats.compare_final(an, [], [], Split.DEV)

    def test_missing_extra_or_duplicate_trials_are_refused(self):
        an = analysis(1)
        v0, ev = final_scores(an, {(Arm.V0, "a-held-1"): [0.5, 0.51, 0.52, 0.53, 0.54], (Arm.EVOLVED, "a-held-1"): [0.6, 0.61, 0.62, 0.63, 0.64]})
        with self.assertRaisesRegex(ContractError, "no score"):
            stats.compare_final(an, v0[:4], ev, Split.SELECTION)
        with self.assertRaisesRegex(ContractError, "not in the"):
            stats.compare_final(an, v0 + [ts("f-rogue", "a-held-1", 0.9)], ev, Split.SELECTION)
        with self.assertRaisesRegex(ContractError, "twice"):
            stats.compare_final(an, v0 + [v0[0]], ev, Split.SELECTION)
        with self.assertRaisesRegex(ContractError, "not in the"):
            stats.compare_final(an, ev, v0, Split.SELECTION)  # arms swapped

    def test_comparability_mismatch_between_arms_is_refused(self):
        an = analysis(1)
        v0, ev = final_scores(an, {(Arm.V0, "a-held-1"): [0.5] * 5, (Arm.EVOLVED, "a-held-1"): [0.6] * 5})
        ev = [dataclasses.replace(ev[0], comparability_sha256=KEY2)] + ev[1:]
        with self.assertRaisesRegex(ContractError, "comparability"):
            stats.compare_final(an, v0, ev, Split.SELECTION)

    def test_wrong_assignment_on_a_selection_score_is_refused(self):
        an = analysis(2, per_cell=3)
        values = {(arm, a): [0.5, 0.52, 0.54] for arm in (Arm.V0, Arm.EVOLVED) for a in ("a-held-1", "a-held-2")}
        v0, ev = final_scores(an, values)
        v0 = [dataclasses.replace(v0[0], assignment_id="a-held-2")] + v0[1:]
        with self.assertRaisesRegex(ContractError, "scored on assignment"):
            stats.compare_final(an, v0, ev, Split.SELECTION)

    def test_identical_trials_give_a_zero_width_interval(self):
        an = analysis(1)
        v0, ev = final_scores(an, {(Arm.V0, "a-held-1"): [0.5] * 5, (Arm.EVOLVED, "a-held-1"): [0.6] * 5})
        r = stats.compare_final(an, v0, ev, Split.SELECTION)
        self.assertAlmostEqual(r.ci_low, 0.1, places=12)
        self.assertAlmostEqual(r.ci_high, 0.1, places=12)
        self.assertEqual(r.df, 8.0)

    def test_result_survives_the_contract_round_trip(self):
        an = analysis(1)
        v0, ev = final_scores(an, {(Arm.V0, "a-held-1"): [0.5, 0.52, 0.54, 0.56, 0.58], (Arm.EVOLVED, "a-held-1"): [0.5, 0.53, 0.55, 0.6, 0.57]})
        r = stats.compare_final(an, v0, ev, Split.SELECTION)
        self.assertEqual(c.FinalComparison.from_dict(c.loads_strict(c.dumps_record(r))), r)


# --------------------------------------------------------------------------------------------------
def score_record(per_seed_values, metric="prompt_strict", split=Split.TEST):
    aggs = tuple(c.SeedAggregate(i, 300, {metric: v, "prompt_loose": min(1.0, v + 0.02), "instruction_strict": v, "instruction_loose": v}, 0, 0) for i, v in enumerate(per_seed_values))
    return c.ScoreRecord(
        score_id="sc-stock",
        request_id="sc-stock",
        phase=c.Phase.FINAL,
        assignment_id=c.TEST_ASSIGNMENT_ID,
        split=split,
        trial_id=None,
        arm=Arm.STOCK,
        final_request_id="f-1",
        base=c.BaseModelRef("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"),
        adapter_sha256=None,
        scorer_version="s1",
        verifier=c.Verifier.IFBENCH_UPSTREAM,
        verifier_pins={"ifbench": "1c40f0c1"},
        params=GenerationParams.recommended(8000),
        per_seed=aggs,
        n_prompts=300,
        per_prompt=c.FileRef("scores/sc-stock/per_prompt.jsonl", "d" * 64, 10),
        headline_metric=metric,
        created_utc=NOW,
    )


class CompareToStockTests(unittest.TestCase):
    def test_one_sample_interval_around_the_stock_mean(self):
        rec = score_record([0.30, 0.32, 0.34])
        trials = [ts(f"t{i}", c.TEST_ASSIGNMENT_ID, v, metric="prompt_strict") for i, v in enumerate([0.40, 0.42, 0.44, 0.46, 0.48])]
        r = stats.compare_to_stock(trials, rec)
        self.assertAlmostEqual(r.stock_mean, 0.32, places=12)
        self.assertAlmostEqual(r.arm_mean, 0.44, places=12)
        self.assertAlmostEqual(r.delta, 0.12, places=12)
        sd = math.sqrt(0.001)
        self.assertAlmostEqual(r.arm_trial_sd, sd, places=12)
        self.assertAlmostEqual(r.ci_high - r.delta, grading.T975[4] * sd / math.sqrt(5), delta=0.0005)
        self.assertEqual((r.n_trials, r.df), (5, 4))
        self.assertAlmostEqual(r.stock_seed_sd, 0.02, places=12)
        self.assertFalse(r.includes_zero)

    def test_includes_zero_and_validation(self):
        rec = score_record([0.44])
        trials = [ts(f"t{i}", c.TEST_ASSIGNMENT_ID, v, metric="prompt_strict") for i, v in enumerate([0.40, 0.48, 0.44, 0.52, 0.36])]
        r = stats.compare_to_stock(trials, rec)
        self.assertTrue(r.includes_zero)
        self.assertIsNone(r.stock_seed_sd)
        with self.assertRaises(ContractError):
            stats.compare_to_stock(trials[:1], rec)
        with self.assertRaisesRegex(ContractError, "metric"):
            stats.compare_to_stock([dataclasses.replace(t, metric="prompt_all") for t in trials], rec)
        with self.assertRaises(ContractError):
            stats.compare_to_stock(trials, rec, level=1.0)

    def test_the_comparison_can_be_made_on_a_metric_other_than_the_records_headline(self):
        rec = score_record([0.30, 0.32, 0.34])  # prompt_loose = prompt_strict + 0.02
        trials = [ts(f"t{i}", c.TEST_ASSIGNMENT_ID, v, metric="prompt_loose") for i, v in enumerate([0.40, 0.42, 0.44])]
        with self.assertRaisesRegex(ContractError, "metric"):
            stats.compare_to_stock(trials, rec)
        r = stats.compare_to_stock(trials, rec, metric="prompt_loose")
        self.assertEqual(r.metric, "prompt_loose")
        self.assertAlmostEqual(r.stock_mean, 0.34, places=12)
        with self.assertRaisesRegex(ContractError, "no prompt_all"):
            stats.compare_to_stock([dataclasses.replace(t, metric="prompt_all") for t in trials], rec, metric="prompt_all")


# --------------------------------------------------------------------------------------------------
class DiagnosticsTests(unittest.TestCase):
    def test_seed_spread(self):
        rec = score_record([0.30, 0.32, 0.34])
        s = stats.seed_spread(rec)
        self.assertEqual((s["n_seeds"], s["metric"]), (3, "prompt_strict"))
        self.assertAlmostEqual(s["mean"], 0.32, places=12)
        self.assertAlmostEqual(s["sd"], 0.02, places=12)
        self.assertEqual((s["min"], s["max"]), (0.30, 0.34))
        self.assertIsNone(stats.seed_spread(score_record([0.3]))["sd"])
        self.assertAlmostEqual(stats.seed_spread(rec, "prompt_loose")["mean"], 0.34, places=12)

    def test_assignment_spread(self):
        scores = [ts("t1", "A", 0.60), ts("t2", "A", 0.62), ts("t3", "B", 0.70), ts("t4", "B", 0.72), ts("t5", "C", 0.50)]
        s = stats.assignment_spread(scores)
        self.assertEqual(s["n_assignments"], 3)
        means = [0.61, 0.71, 0.50]
        m = sum(means) / 3
        self.assertAlmostEqual(s["sd"], math.sqrt(sum((x - m) ** 2 for x in means) / 2), places=12)
        self.assertIsNone(stats.assignment_spread([ts("t1", "A", 0.6), ts("t2", "A", 0.7)]))
        self.assertIsNone(stats.assignment_spread([]))

    def test_prompt_outcomes(self):
        rows = [
            {"prompt_key": "p1", "seed": 0, "passed": True, "error": False},
            {"prompt_key": "p1", "seed": 1, "passed": False, "error": False},
            {"prompt_key": "p2", "seed": 0, "passed": True, "error": True},  # an errored row counts as a failure
            {"prompt_key": "p2", "seed": 1, "passed": True, "error": False},
        ]
        self.assertEqual(stats.prompt_outcomes(rows), {"p1": 0.5, "p2": 0.5})
        self.assertEqual(stats.prompt_outcomes(rows, per_seed=True), {("p1", 0): True, ("p1", 1): False, ("p2", 0): False, ("p2", 1): True})
        test_rows = [{"prompt_key": "p1", "seed": 0, "strict_prompt": True, "loose_prompt": False}]
        self.assertEqual(stats.prompt_outcomes(test_rows, field="strict_prompt"), {"p1": 1.0})
        with self.assertRaises(ContractError):
            stats.prompt_outcomes(rows + rows[:1], per_seed=True)

    def test_bootstrap_is_seeded_paired_and_centred_on_the_difference(self):
        rng = random.Random(1)
        a = {f"p{i}": float(rng.random() < 0.5) for i in range(200)}
        b = {k: float(v or rng.random() < 0.15) for k, v in a.items()}  # b is a with some extra passes
        r1 = stats.paired_bootstrap_prompts(a, b, resamples=500, seed=7)
        r2 = stats.paired_bootstrap_prompts(a, b, resamples=500, seed=7)
        r3 = stats.paired_bootstrap_prompts(a, b, resamples=500, seed=8)
        self.assertEqual(r1, r2)
        self.assertNotEqual((r1["ci_low"], r1["ci_high"]), (r3["ci_low"], r3["ci_high"]))
        self.assertAlmostEqual(r1["delta"], sum(b.values()) / 200 - sum(a.values()) / 200, places=12)
        self.assertLess(r1["ci_low"], r1["delta"])
        self.assertGreater(r1["ci_high"], r1["delta"])
        self.assertGreater(r1["ci_low"], 0.0)  # the added passes are well outside the noise
        self.assertEqual(r1["resamples"], 500)
        self.assertIn("never decides a keep", r1["label"])
        self.assertEqual(stats.paired_bootstrap_prompts(a, b, seed=1)["resamples"], stats.DEFAULT_BOOTSTRAP_RESAMPLES)

    def test_bootstrap_identical_checkpoints_have_a_zero_interval(self):
        a = {f"p{i}": float(i % 2) for i in range(20)}
        r = stats.paired_bootstrap_prompts(a, dict(a), resamples=50, seed=3)
        self.assertEqual((r["delta"], r["ci_low"], r["ci_high"]), (0.0, 0.0, 0.0))

    def test_bootstrap_refuses_different_prompt_sets(self):
        with self.assertRaises(ContractError):
            stats.paired_bootstrap_prompts({"p1": 1.0}, {"p2": 1.0}, resamples=10, seed=1)
        with self.assertRaises(ContractError):
            stats.paired_bootstrap_prompts({}, {}, resamples=10, seed=1)
        with self.assertRaises(ContractError):
            stats.paired_bootstrap_prompts({"p": 1.0}, {"p": 1.0}, resamples=0, seed=1)

    def test_mcnemar_exact(self):
        # 10 discordant pairs: a-only 1, b-only 9 -> two-sided exact p = 2 * P(X <= 1 | n=10) = 2 * 11/1024
        a = {i: (i == 0) or i >= 10 for i in range(20)}
        b = {i: (i >= 1) for i in range(20)}
        r = stats.mcnemar(a, b)
        self.assertEqual((r["a_only"], r["b_only"], r["n_discordant"]), (1, 9, 10))
        self.assertAlmostEqual(r["p_value"], 22 / 1024, places=12)

    def test_mcnemar_no_discordant_pairs_and_balanced(self):
        a = {i: True for i in range(5)}
        self.assertEqual(stats.mcnemar(a, dict(a))["p_value"], 1.0)
        self.assertEqual(stats.mcnemar({1: True, 2: False}, {1: False, 2: True})["p_value"], 1.0)

    def test_mcnemar_does_not_overflow_with_a_thousand_or_more_discordant_prompts(self):
        for n in (1024, 1500):
            keys = range(n)
            r = stats.mcnemar({k: False for k in keys}, {k: True for k in keys})
            self.assertEqual((r["b_only"], r["a_only"], r["n_discordant"]), (n, 0, n))
            self.assertTrue(0.0 <= r["p_value"] < 1e-300, r["p_value"])  # 2 * 2**-n is tiny (or underflows to 0.0), not OverflowError
        half = range(1100)
        r = stats.mcnemar({k: k < 550 for k in half}, {k: k >= 550 for k in half})
        self.assertEqual(r["p_value"], 1.0)

    def test_mcnemar_needs_booleans_and_identical_prompts(self):
        with self.assertRaises(ContractError):
            stats.mcnemar({"p": 0.5}, {"p": 1})
        with self.assertRaises(ContractError):
            stats.mcnemar({"p": True}, {"q": True})


class ProtocolTests(unittest.TestCase):
    def test_the_module_is_a_decision_rule(self):
        self.assertIsInstance(stats, DecisionRule)

    def test_false_accept_bound(self):
        self.assertAlmostEqual(stats.false_accept_bound(0.05, 2), 0.0975, places=12)
        self.assertAlmostEqual(stats.false_accept_bound(0.05, 1), 0.05, places=12)
        with self.assertRaises(ContractError):
            stats.false_accept_bound(0.05, 0)


# --------------------------------------------------------------------------------------------------
# The seeded false-accept simulation (acceptance test of docs/harness-design.md, "Statistics")
# --------------------------------------------------------------------------------------------------
MU, SIGMA = 0.5, 0.02  # trial score = MU + noise; inside [0, 1] for any sd used below


class _Counter:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return f"t{self.n}"


def simulate_search(rng, *, K, n0, n_screen, nc, ni, alpha, candidate_sd_ratio=1.0, n_heavy=None, n_assignments=1, batch=5, guard=1.5, min_gain=0.0, gain=0.0):
    """One search under a true null (every candidate equals the incumbent unless ``gain``): returns (number accepted, number of
    confirmations run, number INVALID). Uses the real stats functions; only the trial scores are simulated.

    Controller flow (docs "Statistics"): each round screens up to ``batch`` candidates against the incumbent's existing trials, the
    best candidate with delta > 0 is confirmed with FRESH candidate and incumbent trials; an accepted candidate becomes the incumbent
    and its confirmation trials are its trials.
    """
    ids = _Counter()
    assignments = tuple(f"a-evo-{i + 1}" for i in range(n_assignments))

    def draw(mean, n, aid, sd_ratio=1.0):
        return [TrialScore(ids.next(), aid, None, "prompt_all", "s1", KEY, mean + rng.gauss(0.0, SIGMA * sd_ratio)) for _ in range(n)]

    def ratio_of(index):  # candidate number 1..K: the first ``n_heavy`` candidates (all of them by default) have the larger sd
        return candidate_sd_ratio if (n_heavy is None or index <= n_heavy) else 1.0

    v0 = [t for aid in assignments for t in draw(MU, n0, aid)]
    spread = stats.trial_spread(v0)
    df = spread.df + n_assignments * ((nc - 1) + (ni - 1))
    mdg = stats.minimum_detectable_gain(max(spread.sigma, 1e-9), df, nc, ni, n_assignments, alpha, min_gain=min_gain)
    rule = make_rule(
        assignments=assignments,
        spread=spread,
        alpha=alpha,
        min_gain=min_gain,
        screen_batch_size=batch,
        screen_trials_per_candidate=n_screen,
        confirm_trials_per_candidate=nc,
        confirm_trials_per_incumbent=ni,
        variance_guard_ratio=guard,
        assumed_gain=mdg * 1.5,
        minimum_detectable_gain=mdg,
        max_candidates=K,
    )
    incumbent_mean = MU
    incumbent = list(v0)
    accepted = confirmations = invalid = 0
    done = 0
    while done < K:
        k = min(batch, K - done)
        done += k
        screens = {
            f"cand-{done - k + j + 1:03d}": [t for aid in assignments for t in draw(incumbent_mean + gain, n_screen, aid, ratio_of(done - k + j + 1))]
            for j in range(k)
        }
        winner = stats.select_screen_winner(stats.screen(rule, incumbent, screens))
        if winner is None:
            continue
        fresh_inc = [t for aid in assignments for t in draw(incumbent_mean, ni, aid)]
        fresh_cand = [t for aid in assignments for t in draw(incumbent_mean + gain, nc, aid, ratio_of(int(winner.split("-")[1])))]
        confirmations += 1
        d = stats.decide(rule, fresh_inc, fresh_cand)
        incumbent += fresh_inc
        if d.outcome is Outcome.INVALID:
            invalid += 1
        elif d.outcome is Outcome.ACCEPTED:
            accepted += 1
            incumbent = fresh_cand
            incumbent_mean += gain
    return accepted, confirmations, invalid


def false_accept_rate(reps, seed, **kw):
    """(P(at least one accept), mean confirmations, P(a confirmation was INVALID)) over ``reps`` seeded searches."""
    rng = random.Random(seed)
    any_acc = conf = inv = 0
    for _ in range(reps):
        a, n_conf, n_inv = simulate_search(rng, **kw)
        any_acc += a > 0
        conf += n_conf
        inv += n_inv > 0
    return any_acc / reps, conf / reps, inv / reps


# the configurations the critics simulated: 5 v0 trials (df 4), 3 + 3 confirmation trials -> pooled df 8; 2 screen trials per candidate
BASE = dict(n0=5, n_screen=2, nc=3, ni=3, alpha=0.05, n_assignments=1)
REPS = 15000
FAR_BOUND = 0.08


class FalseAcceptSimulationTests(unittest.TestCase):
    """The acceptance test: the false-accept rate of screen-then-confirm, all candidates truly equal to the incumbent, stays below
    0.08 (the critics' 0.05 plus Monte-Carlo slack). If one of these fails, stop and report instead of loosening the bound.

    Numbers from ``python3.13 -m tests.test_lab_stats --simulate 30000`` are in the handoff. The one configuration outside the bound (K = 10,
    every candidate noisy, alpha 0.05 per round) is the unsplit one; the default ``freeze-rule`` alpha (Sidak split over the rounds) is gated
    below, and the unsplit case is pinned as outside the bound (not hidden with ``expectedFailure``; the bound is NOT relaxed)."""

    def far(self, seed, **kw):
        far, conf, inv = false_accept_rate(REPS, seed, **kw)
        return far

    def test_simulation_flow_actually_runs_confirmations(self):
        rng = random.Random(0)
        n_conf = sum(simulate_search(rng, K=5, **BASE)[1] for _ in range(200))
        self.assertGreater(n_conf, 150)  # a confirmation is run whenever some candidate screens above the incumbent

    def test_k5_one_round_false_accept_rate_below_bound(self):
        far = self.far(20261101, K=5, **BASE)
        self.assertLess(far, FAR_BOUND, f"false-accept rate {far:.4f} (K=5, {REPS} searches)")

    def test_k10_two_rounds_false_accept_rate_below_bound(self):
        far = self.far(20261102, K=10, **BASE)
        self.assertLess(far, FAR_BOUND, f"false-accept rate {far:.4f} (K=10, {REPS} searches)")

    def test_a_candidate_with_twice_the_sd_is_held_by_the_variance_guard_k5(self):
        far = self.far(20261103, K=5, candidate_sd_ratio=2.0, n_heavy=1, **BASE)
        self.assertLess(far, FAR_BOUND, f"false-accept rate {far:.4f} (K=5, one candidate with sd x2, {REPS} searches)")

    def test_a_candidate_with_twice_the_sd_is_held_by_the_variance_guard_k10(self):
        far = self.far(20261104, K=10, candidate_sd_ratio=2.0, n_heavy=1, **BASE)
        self.assertLess(far, FAR_BOUND, f"false-accept rate {far:.4f} (K=10, one candidate with sd x2, {REPS} searches)")

    def test_every_candidate_with_twice_the_sd_k5(self):
        far = self.far(20261105, K=5, candidate_sd_ratio=2.0, **BASE)
        self.assertLess(far, FAR_BOUND, f"false-accept rate {far:.4f} (K=5, every candidate sd x2, {REPS} searches)")

    def test_a_per_round_alpha_of_0_05_over_two_rounds_does_not_hold_the_bound_when_every_candidate_is_noisy(self):
        # Measured 0.0846 (30000 searches). Two rounds at alpha 0.05 allow up to 1-(1-0.05)^2 = 0.0975 by themselves, and the variance guard
        # removes only part of that for candidates noisier than the margin assumes. This is why ``freeze-rule`` splits the family alpha over
        # the rounds by default (next test, which gates the bound); this one pins that the unsplit configuration really is outside it, so the
        # split is not an unneeded ceremony. It asserts the failure instead of hiding it behind ``expectedFailure``.
        far = self.far(20261106, K=10, candidate_sd_ratio=2.0, **BASE)
        self.assertGreater(far, FAR_BOUND, f"false-accept rate {far:.4f} (K=10, every candidate sd x2, alpha 0.05 per round, {REPS} searches)")

    def test_the_default_per_round_alpha_of_the_cli_holds_the_bound_for_every_candidate_noisy(self):
        from nightshift.lab.cli import sidak_alpha

        alpha = sidak_alpha(0.05, 2)  # K = 10 candidates in batches of 5 = 2 rounds
        self.assertAlmostEqual(alpha, 0.025320565519103, places=12)
        kw = {**BASE, "alpha": alpha}
        far = self.far(20261112, K=10, candidate_sd_ratio=2.0, **kw)
        self.assertLess(far, FAR_BOUND, f"false-accept rate {far:.4f} (K=10, every candidate sd x2, alpha {alpha:.4f} per round, {REPS} searches)")

    def test_splitting_alpha_over_the_rounds_holds_the_bound_in_every_case(self):
        # R = 2 rounds at alpha 0.025 each (Sidak: 1-(1-0.05)^(1/2) = 0.0253): nominal family bound 0.049; measured 0.037 (equal sd), 0.054 (every candidate sd x2)
        kw = {**BASE, "alpha": 0.025}
        self.assertLess(self.far(20261107, K=10, **kw), FAR_BOUND)
        self.assertLess(self.far(20261108, K=10, candidate_sd_ratio=2.0, **kw), FAR_BOUND)
        self.assertLess(self.far(20261109, K=10, candidate_sd_ratio=2.0, n_heavy=1, **kw), FAR_BOUND)

    def test_the_guard_is_what_holds_the_noisy_candidate(self):
        # same scenario with the guard switched off (ratio 100): the rate must be clearly worse, or the guard tests above prove nothing
        with_guard = self.far(20261110, K=10, candidate_sd_ratio=2.0, n_heavy=1, **BASE)
        without = self.far(20261110, K=10, candidate_sd_ratio=2.0, n_heavy=1, guard=100.0, **BASE)
        self.assertGreater(without, with_guard + 0.015)

    def test_the_rule_does_detect_a_real_gain(self):
        # power sanity: a candidate that is truly 3 sigma better is usually kept (so the low false-accept rate is not just "never accept")
        self.assertGreater(self.far(20261111, K=1, gain=3 * SIGMA, **BASE), 0.6)


def _simulate_main(reps):
    print(f"false-accept simulation: {reps} seeded searches per row; all candidates truly equal to the incumbent unless noted")
    print("columns: P(>=1 false accept) | mean confirmations per search | P(some confirmation INVALID by the variance guard)")
    rows = [
        ("K=5 (1 round), n0=5 nc=ni=3 alpha=0.05", dict(K=5, **BASE)),
        ("K=10 (2 rounds), same", dict(K=10, **BASE)),
        ("K=20 (4 rounds), same", dict(K=20, **BASE)),
        ("K=5, candidate sd x2", dict(K=5, candidate_sd_ratio=2.0, **BASE)),
        ("K=10, candidate sd x2", dict(K=10, candidate_sd_ratio=2.0, **BASE)),
        ("K=10, candidate sd x2, guard OFF (ratio 100)", dict(K=10, candidate_sd_ratio=2.0, guard=100.0, **BASE)),
        ("K=5, ONE candidate with sd x2 (the others equal)", dict(K=5, candidate_sd_ratio=2.0, n_heavy=1, **BASE)),
        ("K=10, ONE candidate with sd x2 (the others equal)", dict(K=10, candidate_sd_ratio=2.0, n_heavy=1, **BASE)),
        ("K=10, ONE candidate with sd x2, guard OFF", dict(K=10, candidate_sd_ratio=2.0, n_heavy=1, guard=100.0, **BASE)),
        ("K=10, alpha 0.025", dict(K=10, **{**BASE, "alpha": 0.025})),
        ("K=10, alpha 0.025, candidate sd x2", dict(K=10, candidate_sd_ratio=2.0, **{**BASE, "alpha": 0.025})),
        ("K=10, n0=4 nc=ni=4 (pooled df 9)", dict(K=10, **{**BASE, "n0": 4, "nc": 4, "ni": 4})),
        ("K=10, 2 assignments (n0=3 per cell, nc=ni=3)", dict(K=10, **{**BASE, "n0": 3, "n_assignments": 2})),
        ("K=1, true gain 3 sigma (power)", dict(K=1, gain=3 * SIGMA, **BASE)),
    ]
    for i, (name, kw) in enumerate(rows):
        far, conf, inv = false_accept_rate(reps, 1000 + i, **kw)
        print(f"{name:70s} {far:7.4f} | {conf:5.2f} | {inv:6.4f}   (bound 1-(1-alpha)^R = {stats.false_accept_bound(kw.get('alpha', 0.05), -(-kw['K'] // 5)):.4f})")


if __name__ == "__main__":
    if "--simulate" in sys.argv:
        i = sys.argv.index("--simulate")
        _simulate_main(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 20000)
    else:
        unittest.main()
