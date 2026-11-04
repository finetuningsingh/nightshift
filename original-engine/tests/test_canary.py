"""Unit tests of the sealed-number canary generator (``tests/support/canary.py``).

The simulated-Muse end-to-end test and the scripted golden demo both search what the agents saw for the needles this generator
makes, so the generator itself must be right: it must catch the ways a sealed number is really written (rounded to 2 to 6
decimals, as a percentage, in scientific notation, under a label, a value below 1e-3), must not fire on legitimate dev numbers
or on text that merely looks similar, and must not produce needles so short that they match everything.

Run: ``python3 -m unittest tests.test_canary -v``
"""
from __future__ import annotations

import unittest

from nightshift.types import Event, Kind
from tests.support import canary
from tests.support.canary import (MIN_SIGNIFICANT_DIGITS, SHORT_NEEDLE_DIGITS, acceptable, canary_needles, dev_values, find_leaks,
                                  labelled_forms, number_forms, occurs_in, run_needles, sealed_floats, sealed_values,
                                  significant_digits)

SEALED = 1.5625820215618353       # a sealed delta: the number of the three leaks that the old five-decimal needle missed
# the three proven gaps of the old generator: a sealed delta written 1.5626, 1.56 or +156.3%
GAP_LEAKS = ("sealed delta 1.5626 on the held-out data",
             "improvement of 1.56 on the independent re-check",
             "the candidate is +156.3% vs baseline")


def old_needles(value: float) -> set[str]:
    """The generator this module replaces (five decimals rounded and truncated; values under 1e-3 skipped)."""
    out = {f"{abs(value):.5f}"}
    text = repr(abs(value))
    if "e" not in text and "." in text:
        whole, frac = text.split(".")
        if len(frac) >= 5:
            out.add(f"{whole}.{frac[:5]}")
    return {n for n in out if abs(value) >= 1e-3 and not n.endswith("0000")}


def event(seq: int, kind: str, **data) -> Event:
    return Event(seq=seq, ts=float(seq), run_id="r-test", kind=kind, actor="engine", data=data, prev_hash="0" * 64, hash=f"{seq:064d}")


def graded(seq: int, split: str, purpose: str, score: float, n_eval: int, per_seed: list[float]) -> Event:
    grade = {"score": score, "std": 0.011, "per_seed": per_seed, "metrics": {"heldout_rmse": score, "train_rmse": score / 7},
             "per_seed_metrics": [{"heldout_rmse": v, "train_rmse": v / 7} for v in per_seed], "n_eval": n_eval}
    return event(seq, Kind.EXP_GRADED, exp_id="e001", split=split, purpose=purpose, grade=grade)


def ledger_with_a_replication() -> list[Event]:
    return [
        graded(1, "dev", "primary", 0.8991803454457031, 120, [0.91234567, 0.88345678, 0.90456789]),
        event(2, Kind.EXP_RECORDED, exp_id="e001", phase="lane", delta=0.7291, noise_margin=0.035, s_eff=0.0412, t=2.776,
              sigma_prior=0.0100116, replicated=False),
        graded(3, "sealed", "replication", 0.9450522356237963, 210, [0.9580233033149254, 0.8568986218545067, 0.9745053798145629]),
        event(4, Kind.EXP_RECORDED, exp_id="e001", phase="replication", sealed_delta=0.6832842455649646,
              sealed_margin=0.10735692691147965, s_eff=0.08647600375704073, t=2.776, sigma_prior=0.0100116, n=5, replicated=True),
        event(5, Kind.INCUMBENT_PROMOTED, exp_id="e001", score=0.8991803454457031, sealed_score=0.9450522356237963,
              sealed_delta=0.6832842455649646, sealed_margin=0.10735692691147965),
    ]


class TestTheOldGeneratorHadGaps(unittest.TestCase):
    def test_the_old_needles_caught_a_verbatim_leak_only(self):
        old = old_needles(SEALED)
        self.assertEqual(old, {"1.56258"})
        self.assertTrue(find_leaks(["delta=1.56258"], old))
        for leak in GAP_LEAKS:
            self.assertEqual(find_leaks([leak], old), [], f"the old needle already caught {leak!r}")

    def test_a_value_under_a_thousandth_had_no_needle_at_all(self):
        self.assertEqual(old_needles(0.000123456), set())


class TestTheNewGeneratorCatchesTheGaps(unittest.TestCase):
    def test_each_proven_gap_is_caught(self):
        needles = sorted(number_forms(SEALED))
        for leak in GAP_LEAKS:
            self.assertTrue(find_leaks([leak], needles), f"missed {leak!r}")

    def test_the_old_leak_and_the_full_precision_leak_are_still_caught(self):
        needles = sorted(number_forms(SEALED))
        for leak in ("delta=1.56258", f"delta={SEALED!r}", f'{{"sealed_delta": {SEALED}}}', "delta -1.5626", f"x{SEALED:.6f}y",
                     f"{SEALED:.3e}", f"{SEALED:.2E}", "1.5625e0"):
            self.assertTrue(find_leaks([leak], needles), f"missed {leak!r}")

    def test_every_rounding_and_truncation_of_a_value_is_a_needle(self):
        forms = number_forms(SEALED)
        for places in (2, 3, 4, 5, 6):
            self.assertIn(f"{SEALED:.{places}f}", forms)
            self.assertIn(repr(SEALED)[:2 + places], forms, f"truncation at {places} decimals")
        self.assertIn("1.56", forms)                    # 1.5625... rounds to 1.56 and truncates to 1.56
        self.assertIn("1.563", forms)                   # rounded and not only truncated

    def test_percentages_and_per_mille_are_needles(self):
        forms = number_forms(SEALED)
        for shown in ("156.3", "156.26", "156%", "156 %", "1562.6", "1563‰", "1563 per mille"):
            self.assertIn(shown, forms)

    def test_scientific_notation_is_a_needle(self):
        forms = number_forms(SEALED)
        for shown in ("1.56e+00", "1.563e+00", "1.5626e+00", "1.56E+00", "1.56e0", "1.562582e+00"):
            self.assertIn(shown, forms)
        self.assertEqual(f"{SEALED:e}", "1.562582e+00", "the default %e / format(x, 'e') has 6 decimals in the mantissa")
        self.assertTrue(find_leaks([f"delta {SEALED:e}", "delta %e" % SEALED], sorted(forms)))

    def test_a_value_under_a_thousandth_is_covered(self):
        tiny = 0.000123456
        forms = number_forms(tiny)
        self.assertTrue(forms, "a tiny sealed number must still have needles")
        for leak in ("margin 0.000123", "margin 0.0001235", "margin 1.23e-04", "margin 1.2346e-04", f"margin {tiny!r}", "margin .000123"):
            self.assertTrue(find_leaks([leak], sorted(forms)), f"missed {leak!r}")
        # the full repr of a value that Python prints in scientific notation contains a mantissa prefix
        smaller = 1.5625820215618352e-05
        self.assertEqual(repr(smaller), "1.5625820215618352e-05")
        self.assertTrue(find_leaks([f"std {smaller!r}", "std 1.56258e-05", "std 0.0000156"], sorted(number_forms(smaller))))

    def test_a_per_mille_sign_is_found_in_a_json_body_that_escapes_it(self):
        """A request body is JSON with ASCII escapes: ``\u2030``. The per-mille needle must still be found in it."""
        import json
        needles = sorted(number_forms(SEALED))
        self.assertIn("1563\u2030", needles)
        body = json.dumps({"messages": [{"role": "assistant", "content": "up 1563\u2030 on the baseline"}]})
        self.assertIn("\\u2030", body)
        self.assertTrue(find_leaks([body], needles))
        self.assertEqual(find_leaks([json.dumps({"content": "up 99 on the baseline \u2030"})], needles), [])

    def test_the_sign_is_ignored(self):
        self.assertEqual(number_forms(-SEALED), number_forms(SEALED))
        self.assertTrue(find_leaks(["the delta was -1.5626"], sorted(number_forms(SEALED))))

    def test_non_finite_and_zero_have_no_needles(self):
        for v in (0.0, float("nan"), float("inf"), -0.0):
            self.assertEqual(number_forms(v), set(), v)

    def test_huge_values_do_not_raise(self):
        self.assertIsInstance(number_forms(1.5e300), set)
        self.assertIsInstance(number_forms(2.5e-300), set)


class TestTheThreshold(unittest.TestCase):
    def test_the_threshold_is_three_significant_digits_and_six_for_a_free_prefix(self):
        self.assertEqual((MIN_SIGNIFICANT_DIGITS, SHORT_NEEDLE_DIGITS), (3, 6))

    def test_significant_digits_ignore_zeros_at_both_ends(self):
        for text, n in (("1.5", 2), ("0.0", 0), ("1.56", 3), ("0.0123", 3), ("2.00", 1), ("1.500", 2), ("10.5", 3), (".5625", 4),
                        ("1.56e+00", 3), ("156%", 3), ("0.000123", 3), ("abc", 0)):
            self.assertEqual(significant_digits(text), n, text)

    def test_needles_that_match_innocuous_text_are_refused(self):
        for text in ("1.5", "0.0", "0.05", "2.00", "1.50", "0.10", "10", "156", "15626", "0", "100%", "50%", "1e+00"):
            self.assertFalse(acceptable(text), text)

    def test_needles_with_enough_information_are_kept(self):
        for text in ("1.56", "0.945", "156%", "156.3", "1.56e+00", ".5625", "0.000123", "1562‰"):
            self.assertTrue(acceptable(text), text)

    def test_no_generated_needle_is_below_the_threshold(self):
        for value in (SEALED, 0.9450522356237963, 0.10735692691147965, 2.776, 0.000123456, 1.5, 0.5, 2.0, 12345.678):
            for needle in number_forms(value):
                self.assertTrue(acceptable(needle), (value, needle))
                self.assertNotRegex(needle, r"^\d+$")

    def test_a_round_number_produces_no_needle(self):
        for value in (1.5, 0.5, 2.0, 0.05, 0.25, 100.0, 0.1):
            self.assertEqual(number_forms(value), set(), value)

    def test_innocuous_text_matches_no_needle_of_a_typical_sealed_number(self):
        needles = sorted(canary_needles([SEALED, 0.9450522356237963, 0.10735692691147965]))
        innocuous = ["score 1.5 and 0.0 and 2.00; 10 seeds, 100% of 156 runs, seq 1562", "budget 2.0 usd, noise margin 0.035, p<0.05",
                     "the sealed split is held back"]
        self.assertEqual(find_leaks(innocuous, needles), [])


class TestMatchingIsBoundaryAware(unittest.TestCase):
    def test_a_short_needle_does_not_fire_inside_a_longer_number(self):
        self.assertFalse(occurs_in("1.56", "dev score 1.5601"))
        self.assertFalse(occurs_in("1.56", "x 21.56 y"))
        self.assertFalse(occurs_in(".5625", "x 1.5625 y"))
        self.assertTrue(occurs_in("1.56", "improvement of 1.56 on"))
        self.assertTrue(occurs_in("1.56", "v1.56"))
        self.assertTrue(occurs_in("1.56", "1.56."))
        self.assertTrue(occurs_in("1.56", "(1.56)"))

    def test_a_long_prefix_is_found_inside_the_full_precision_repr(self):
        self.assertTrue(occurs_in("1.56258", "delta=1.5625820215618353"))
        self.assertFalse(occurs_in("1.5625", "delta=1.5625820215618353"), "a 5 digit needle is not a free prefix")

    def test_a_prefix_with_zeros_in_it_is_still_a_prefix(self):
        """0.2865001893964 printed to 7 decimals is 0.2865001. Its 6 decimals 0.286500 have only 4 significant digits, but they are
        6 digits long, so they are found at the start of the longer rendering (and not inside another number)."""
        needles = sorted(number_forms(0.2865001893964))
        self.assertIn("0.286500", needles)
        self.assertTrue(occurs_in("0.286500", "value 0.2865001 here"))
        self.assertTrue(find_leaks(["value 0.2865001 here", "value 0.28650018 here"], needles))
        self.assertFalse(occurs_in("0.286500", "value 10.286500 here"))
        self.assertFalse(occurs_in("0.2865", "value 0.28650018 here"), "a 4 digit needle is not a free prefix")

    def test_a_leading_dot_needle_matches_a_number_written_without_the_zero(self):
        self.assertTrue(occurs_in(".5625", "delta .5625 and more"))
        self.assertTrue(occurs_in(".5625", "(.5625)"))

    def test_units_and_labels_work_as_part_of_the_needle(self):
        self.assertTrue(occurs_in("156%", "up 156% on baseline"))
        self.assertFalse(occurs_in("156%", "up 1156% on baseline"))
        self.assertTrue(occurs_in("n_eval=210", 'n_eval=210, "x"'))
        self.assertFalse(occurs_in("n_eval=21", "n_eval=210"))
        self.assertFalse(occurs_in("t = 2.776", "cat = 2.776"), "a label starts at a word boundary")

    def test_find_leaks_searches_every_text_and_keeps_the_needle_order(self):
        self.assertEqual(find_leaks(["nothing", "x 1.5626 y", "z 156.3% z"], ["156.3", "9.9999", "1.5626"]), ["156.3", "1.5626"])
        self.assertEqual(find_leaks([], ["1.5626"]), [])
        self.assertEqual(find_leaks(["x"], []), [])


class TestBudgetAndClockCountersAreNotSearched(unittest.TestCase):
    """Spend and clock time differ in every run, so a counter that renders like a coarse form of a sealed number would be a flaky
    false alarm: the s_eff 0.07633516 of a real run and a running spend of 0.0763 USD."""

    def needles(self):
        return canary_needles([("s_eff", 0.07633516088153562)])

    def test_a_counter_that_looks_like_a_sealed_number_is_not_a_leak(self):
        for text in ('{"budget_used": {"usd": 0.0763, "wall_sec": 5.5464}}', '{\\"usd\\": 0.0763}', "spent usd 0.0763 so far",
                     "wall_sec: 0.0763", "remaining=0.0763"):
            self.assertEqual(find_leaks([text], self.needles()), [], text)

    def test_a_duration_or_an_amount_of_money_with_its_unit_is_not_a_leak(self):
        for text in ("3 run(s): ok x3; 0.0763s in total", "took 0.0763 s", "0.0763 seconds", "cost $0.0763", "0.0763usd", "0.0763 ms"):
            self.assertEqual(find_leaks([text], self.needles()), [], text)
        for text in ("0.0763 of the spread", "0.0763 sigma", "0.0763 sealed"):
            self.assertTrue(find_leaks([text], self.needles()), text)

    def test_the_same_digits_under_any_other_key_are_a_leak(self):
        for text in ('{"delta": 0.0763}', "the spread was 0.0763", '{"score": 0.0763}', "s_eff=0.0763"):
            self.assertTrue(find_leaks([text], self.needles()), text)

    def test_the_exemption_can_be_switched_off(self):
        self.assertTrue(find_leaks(['{"usd": 0.0763}'], self.needles(), ignore_keys=()))

    def test_only_the_number_right_after_the_key_is_exempt(self):
        self.assertTrue(find_leaks(['{"usd": 0.5, "delta": 0.0763}'], self.needles()))
        self.assertTrue(find_leaks(["usd 0.5 and then 0.0763"], self.needles()))


class TestNoFalseAlarmsOnLegitimateNumbers(unittest.TestCase):
    def test_a_needle_that_a_dev_number_also_makes_is_dropped(self):
        sealed, dev = 1.5625820215618353, 1.5601234
        self.assertIn("1.56", number_forms(sealed))
        self.assertIn("1.56", number_forms(dev))
        needles = canary_needles([sealed], [dev])
        self.assertNotIn("1.56", needles)
        self.assertIn("1.5626", needles)
        self.assertEqual(find_leaks(["dev score 1.56 and 1.5601234"], needles), [])
        self.assertTrue(find_leaks(["sealed delta 1.5626"], needles))

    def test_the_dev_number_itself_is_never_flagged_whatever_its_rendering(self):
        sealed = [0.9450522356237963, 0.10735692691147965, 0.08647600375704073]
        dev = [0.8991803454457031, 0.7291, 0.0412, 0.0100116]
        needles = canary_needles(sealed, dev)
        for value in dev:
            for rendering in (repr(value), f"{value:.2f}", f"{value:.3f}", f"{value:.4f}", f"{value:.6f}", f"{value:.3e}", f"{value*100:.1f}%"):
                self.assertEqual(find_leaks([f"dev value {rendering} here"], needles), [], rendering)

    def test_a_sealed_number_the_dev_split_does_not_share_still_leaves_needles(self):
        needles = canary_needles([SEALED], [0.8991803454457031])
        self.assertGreaterEqual(len(needles), 20)

    def test_a_named_integer_equal_to_a_dev_value_is_dropped(self):
        self.assertEqual(canary_needles([("n_eval", 210)], [("n_eval", 210)]), [])
        self.assertTrue(canary_needles([("n_eval", 210)], [("n_eval", 120)]))

    def test_an_unnamed_integer_has_no_needle(self):
        self.assertEqual(canary_needles([210, 5, True], []), [])


class TestNamedQuantities(unittest.TestCase):
    def test_labelled_forms_cover_the_common_layouts(self):
        forms = labelled_forms("n_eval", 210)
        for shown in ("n_eval=210", "n_eval = 210", "n_eval: 210", '"n_eval": 210', "n_eval 210"):
            self.assertIn(shown, forms)
        self.assertEqual(labelled_forms("flag", True), set())

    def test_s_eff_and_t_are_searched_under_their_labels_and_as_numbers(self):
        needles = canary_needles([("s_eff", 0.08647600375704073), ("t", 3.182)])
        for leak in ("s_eff=0.0865", "the effective spread s_eff: 0.08647600375704073", "t = 3.182", "0.0865"):
            self.assertTrue(find_leaks([leak], needles), leak)

    def test_the_n_eval_of_the_sealed_split_is_found_when_the_dev_split_has_another(self):
        needles = canary_needles([("n_eval", 210)], [("n_eval", 120)])
        self.assertTrue(find_leaks(["evaluated on n_eval=210 rows"], needles))
        self.assertEqual(find_leaks(["evaluated on n_eval=120 rows", "210 rows"], needles), [])


class TestExtractionFromTheLedger(unittest.TestCase):
    def setUp(self) -> None:
        self.events = ledger_with_a_replication()

    def test_sealed_values_hold_the_grade_the_statistics_and_the_promotion(self):
        values = sealed_values(self.events)
        plain = {v for v in values if not isinstance(v, tuple)}
        named = {v for v in values if isinstance(v, tuple)}
        for expected in (0.9450522356237963, 0.9580233033149254, 0.6832842455649646, 0.10735692691147965):
            self.assertIn(expected, plain)
        self.assertIn(("n_eval", 210), named)
        self.assertIn(("s_eff", 0.08647600375704073), named)
        self.assertIn(("t", 2.776), named)
        self.assertNotIn(0.8991803454457031, plain, "a dev score is not sealed")
        self.assertNotIn(("n_eval", 120), named)

    def test_dev_values_hold_the_dev_grade_and_the_lane_record_but_nothing_sealed(self):
        values = dev_values(self.events)
        plain = {v for v in values if not isinstance(v, tuple)}
        named = {v for v in values if isinstance(v, tuple)}
        self.assertIn(0.91234567, plain)
        self.assertIn(0.8991803454457031, plain)
        self.assertIn(("t", 2.776), named)
        self.assertIn(("sigma_prior", 0.0100116), named)
        self.assertIn(("n_eval", 120), named)
        self.assertNotIn(0.9450522356237963, plain)
        self.assertNotIn(("s_eff", 0.08647600375704073), named)

    def test_a_sealed_field_of_any_event_counts(self):
        extra = event(9, Kind.SUP_NUDGE, text="x", detail={"sealed_note": [0.123456, {"sealed_x": 0.654321}], "dev_note": 0.5})
        floats = sealed_floats([extra])
        self.assertEqual(sorted(floats), [0.123456, 0.654321])

    def test_the_run_needles_are_the_sealed_ones_minus_the_dev_ones(self):
        needles = run_needles(self.events)
        self.assertGreaterEqual(len(needles), 100)
        # t is 2.776 in both splits: it must not be a needle (it would fire on any dev statistics the supervisor sees)
        self.assertNotIn("t = 2.776", needles)
        self.assertNotIn("2.776", needles)
        self.assertNotIn("n_eval=120", needles, "a dev quantity is never a needle")
        for leak in ("replicated, sealed margin 0.1074", "delta +68.33% vs baseline", "sealed score 0.945", "s_eff=0.0865",
                     "n_eval=210", "0.68328"):
            self.assertTrue(find_leaks([leak], needles), f"missed {leak!r}")
        for legit in ("dev score 0.8991803454457031", "dev delta 0.7291", "baseline spread 0.0412", "n_eval=120", "t = 2.776",
                      "dev 0.899", "dev 89.9%"):
            self.assertEqual(find_leaks([legit], needles), [], f"false alarm on {legit!r}")

    def test_a_run_without_a_replication_has_no_needles(self):
        self.assertEqual(run_needles(self.events[:2]), [])

    def test_the_module_exports_what_the_end_to_end_tests_import(self):
        for name in ("run_needles", "sealed_values", "dev_values", "find_leaks", "canary_needles", "number_forms", "sealed_floats"):
            self.assertTrue(callable(getattr(canary, name)), name)


if __name__ == "__main__":
    unittest.main()
