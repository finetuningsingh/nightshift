"""Tests for nightshift.lab.budget (per-trial hard stop and the lab-wide cap).

Run: PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_budget
"""

import tempfile
import threading
import unittest
from pathlib import Path

from nightshift.lab import contracts as c
from nightshift.lab.budget import LabCap, TrialBudgetMeter, lab_spent_usd, outer_usd_spent, usage_from_events
from nightshift.lab.contracts import (
    BudgetCharge,
    BudgetExhausted,
    BudgetExhaustedEvent,
    BudgetMeter,
    BudgetSpec,
    BudgetUsage,
    ContractError,
    LabKind,
    LabLayout,
    LabState,
    LlmUsage,
    Resource,
)
from nightshift.lab.ledger import LabLedger
from nightshift.models.pricing import usd_for as price
from nightshift.types import Usage
from tests.test_lab_contracts import candidate, trial_record

TRIAL = "t-0001"
MODEL = "muse-spark-1.3"


class Clock:
    """A manual clock: only the test moves it."""

    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def llm_usd(prompt: int, completion: int, cached: int = 0) -> float:
    return price(MODEL, Usage(prompt, completion, cached, prompt + completion))


class MeterCase(unittest.TestCase):
    spec = BudgetSpec(researcher_tokens=10_000, gpu_minutes=10.0, dollars=0.05, wall_seconds=100.0)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.clock = Clock()
        self.ledger = LabLedger.create(self.root / "t", clock=self.clock, fsync=False)
        self.addCleanup(self.ledger.close)

    def meter(self, spec=None, ledger=None, lab_cap=None, trial_id=TRIAL) -> TrialBudgetMeter:
        return TrialBudgetMeter(spec or self.spec, ledger or self.ledger, trial_id, clock=self.clock, lab_cap=lab_cap)

    def kinds(self):
        return [e.kind for e in self.ledger.events()]

    def charges(self):
        return [BudgetCharge.from_dict(e.data) for e in self.ledger.events() if e.kind == LabKind.BUDGET_CHARGED]

    def exhausted_events(self):
        return [BudgetExhaustedEvent.from_dict(e.data) for e in self.ledger.events() if e.kind == LabKind.BUDGET_EXHAUSTED]


class ChargeTests(MeterCase):
    def test_implements_the_budget_meter_protocol(self):
        m = self.meter()
        self.assertIsInstance(m, BudgetMeter)
        self.assertEqual(m.spec, self.spec)

    def test_a_trial_spec_may_stand_for_its_budget(self):
        from tests.test_lab_contracts import spec as trial_spec

        sp = trial_spec(trial_id=TRIAL)
        m = TrialBudgetMeter(sp, self.ledger, sp.trial_id, clock=self.clock)
        self.assertEqual(m.spec, sp.budget)
        with self.assertRaises(ContractError):
            TrialBudgetMeter({"researcher_tokens": 1}, self.ledger, TRIAL, clock=self.clock)

    def test_charge_books_one_event_and_returns_usage(self):
        m = self.meter()
        u = m.charge(Resource.GPU_MINUTES, 3.0, "gpu:r1:1", {"rate": 1.2})
        self.assertEqual((u.gpu_minutes, u.researcher_tokens, u.dollars), (3.0, 0, 0.0))
        (ch,) = self.charges()
        self.assertEqual((ch.trial_id, ch.resource, ch.amount, ch.call_id, ch.detail), (TRIAL, Resource.GPU_MINUTES, 3.0, "gpu:r1:1", {"rate": 1.2}))
        self.assertEqual(self.ledger.events()[0].actor, f"trial:{TRIAL}")

    def test_charge_is_idempotent_per_call_id_and_resource(self):
        m = self.meter()
        for _ in range(3):
            u = m.charge(Resource.GPU_MINUTES, 1.0, "gpu:r1:1")
        self.assertEqual(u.gpu_minutes, 1.0)
        self.assertEqual(len(self.charges()), 1)

    def test_one_call_id_may_book_minutes_and_dollars(self):
        # a GPU tick books GPU_MINUTES and GPU_USD under the same call id; the second must not be dropped as a duplicate
        m = self.meter()
        m.charge(Resource.GPU_MINUTES, 1.0, "gpu:r1:7")
        u = m.charge(Resource.GPU_USD, 0.02, "gpu:r1:7")
        self.assertEqual((u.gpu_minutes, u.gpu_usd), (1.0, 0.02))
        self.assertEqual(len(self.charges()), 2)

    def test_wall_seconds_cannot_be_charged(self):
        m = self.meter()
        with self.assertRaises(ContractError):
            m.charge(Resource.WALL_SECONDS, 5.0, "w1")
        self.assertEqual(self.charges(), [])

    def test_invalid_amounts_are_refused_and_nothing_is_booked(self):
        m = self.meter()
        for bad in (-1.0, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ContractError):
                    m.charge(Resource.GPU_MINUTES, bad, f"c{bad}")
        self.assertEqual(m.usage().gpu_minutes, 0.0)
        self.assertEqual(self.charges(), [])

    def test_dollars_axis_is_shared_by_researcher_and_gpu_spend(self):
        m = self.meter()
        m.charge(Resource.RESEARCHER_USD, 0.03, "r1")
        with self.assertRaises(BudgetExhausted) as cm:
            m.charge(Resource.GPU_USD, 0.02, "g1")  # 0.03 + 0.02 reaches the 0.05 cap
        self.assertEqual(cm.exception.axis, "dollars")
        self.assertAlmostEqual(m.usage().dollars, 0.05)

    def test_a_charge_that_crosses_a_cap_is_booked_raises_and_logs_once(self):
        m = self.meter()
        m.charge(Resource.GPU_MINUTES, 9.0, "g1")
        with self.assertRaises(BudgetExhausted) as cm:
            m.charge(Resource.GPU_MINUTES, 2.0, "g2")
        self.assertEqual((cm.exception.axis, cm.exception.limit, cm.exception.used), ("gpu_minutes", 10.0, 11.0))
        self.assertEqual(m.usage().gpu_minutes, 11.0)  # booked, not dropped
        (ev,) = self.exhausted_events()
        self.assertEqual((ev.trial_id, ev.axis, ev.limit, ev.used), (TRIAL, "gpu_minutes", 10.0, 11.0))
        with self.assertRaises(BudgetExhausted):
            m.charge(Resource.GPU_MINUTES, 1.0, "g3")
        self.assertEqual(len(self.exhausted_events()), 1)  # once per axis
        self.assertEqual(self.ledger.verify(), (True, None))

    def test_exhausting_one_axis_leaves_the_others_usable(self):
        m = self.meter()
        with self.assertRaises(BudgetExhausted):
            m.charge(Resource.GPU_MINUTES, 10.0, "g1")  # usage >= cap counts as used up
        m.check(Resource.RESEARCHER_TOKENS)
        m.check(Resource.WALL_SECONDS)
        with self.assertRaises(BudgetExhausted):
            m.check(Resource.GPU_MINUTES)


class CheckTests(MeterCase):
    def test_check_passes_below_the_cap_and_raises_at_it(self):
        m = self.meter()
        m.charge(Resource.RESEARCHER_TOKENS, 9_999, "a")
        m.check(Resource.RESEARCHER_TOKENS)
        with self.assertRaises(BudgetExhausted):
            m.charge(Resource.RESEARCHER_TOKENS, 1, "b")
        with self.assertRaises(BudgetExhausted):
            m.check(Resource.RESEARCHER_TOKENS)

    def test_a_zero_cap_is_used_up_at_once(self):
        m = self.meter(BudgetSpec(0, 10.0, 1.0, 100.0))
        with self.assertRaises(BudgetExhausted):
            m.check(Resource.RESEARCHER_TOKENS)

    def test_wall_time_comes_from_the_clock_since_start(self):
        m = self.meter()
        self.clock.t += 99.0
        m.check(Resource.WALL_SECONDS)
        self.assertEqual(m.usage().wall_seconds, 99.0)
        self.clock.t += 1.0
        with self.assertRaises(BudgetExhausted) as cm:
            m.check(Resource.WALL_SECONDS)
        self.assertEqual(cm.exception.axis, "wall_seconds")
        (ev,) = self.exhausted_events()
        self.assertEqual(ev.axis, "wall_seconds")

    def test_check_uses_the_axis_of_the_resource(self):
        m = self.meter()
        m.charge(Resource.RESEARCHER_USD, 0.04, "r")
        m.check(Resource.RESEARCHER_USD)
        m.charge(Resource.GPU_USD, 0.005, "g")
        self.assertAlmostEqual(m.usage().dollars, 0.045)
        with self.assertRaises(BudgetExhausted):
            m.charge(Resource.GPU_USD, 0.005, "g2")
        for res in (Resource.RESEARCHER_USD, Resource.GPU_USD):
            with self.assertRaises(BudgetExhausted):
                m.check(res)


class LlmTests(MeterCase):
    def test_reserve_books_the_worst_case_and_usage_includes_it(self):
        m = self.meter()
        m.reserve_llm(MODEL, 3_000, 4_000, "call1")
        u = m.usage()
        self.assertEqual(u.researcher_tokens, 7_000)
        self.assertAlmostEqual(u.researcher_usd, llm_usd(3_000, 4_000))
        self.assertEqual(self.charges(), [])  # a reservation is not a ledger charge

    def test_reserve_that_would_cross_the_token_cap_is_refused_and_books_nothing(self):
        m = self.meter()
        m.reserve_llm(MODEL, 3_000, 4_000, "call1")
        with self.assertRaises(BudgetExhausted) as cm:
            m.reserve_llm(MODEL, 3_000, 4_000, "call2")  # 14000 > 10000
        self.assertEqual(cm.exception.axis, "researcher_tokens")
        self.assertEqual(m.usage().researcher_tokens, 7_000)
        (ev,) = self.exhausted_events()
        self.assertEqual(ev.axis, "researcher_tokens")

    def test_reserve_that_would_cross_the_dollar_cap_is_refused(self):
        m = self.meter(BudgetSpec(10_000_000, 10.0, 0.05, 100.0))
        m.reserve_llm(MODEL, 3_000, 4_000, "call1")  # 0.02075
        m.reserve_llm(MODEL, 3_000, 4_000, "call2")  # 0.0415
        with self.assertRaises(BudgetExhausted) as cm:
            m.reserve_llm(MODEL, 3_000, 4_000, "call3")
        self.assertEqual(cm.exception.axis, "dollars")

    def test_a_single_call_whose_prompt_alone_crosses_the_cap_is_refused(self):
        m = self.meter()
        with self.assertRaises(BudgetExhausted):
            m.reserve_llm(MODEL, 20_000, 1, "big")
        self.assertEqual(m.usage().researcher_tokens, 0)

    def test_reserve_is_idempotent_per_call_id(self):
        m = self.meter()
        for _ in range(3):
            m.reserve_llm(MODEL, 1_000, 1_000, "call1")
        self.assertEqual(m.usage().researcher_tokens, 2_000)

    def test_charge_llm_replaces_the_reservation_with_the_actual_usage_in_two_events(self):
        m = self.meter()
        m.reserve_llm(MODEL, 3_000, 4_000, "call1")
        u = m.charge_llm(MODEL, LlmUsage(prompt_tokens=2_000, completion_tokens=500, cached_tokens=400), "call1")
        self.assertEqual(u.researcher_tokens, 2_500)
        self.assertAlmostEqual(u.researcher_usd, llm_usd(2_000, 500, 400))
        by_id = {ch.call_id: ch for ch in self.charges()}
        self.assertEqual(set(by_id), {"call1:tokens", "call1:usd"})
        self.assertEqual((by_id["call1:tokens"].resource, by_id["call1:tokens"].amount), (Resource.RESEARCHER_TOKENS, 2_500))
        self.assertEqual(by_id["call1:usd"].resource, Resource.RESEARCHER_USD)
        self.assertAlmostEqual(by_id["call1:usd"].amount, llm_usd(2_000, 500, 400))
        self.assertEqual(by_id["call1:tokens"].detail["cached_tokens"], 400)
        # the room the reservation held is free again
        m.reserve_llm(MODEL, 3_000, 4_000, "call2")

    def test_a_settlement_rejected_for_its_suffixed_id_keeps_the_reservation_booked(self):
        m = self.meter(BudgetSpec(researcher_tokens=20_000, gpu_minutes=10.0, dollars=1.0, wall_seconds=100.0))
        long_id = "c" * 200  # valid as a reservation; "<id>:tokens" is 207 bytes and is refused
        m.reserve_llm(MODEL, 10_000, 10_000, long_id)
        with self.assertRaises(ContractError):
            m.charge_llm(MODEL, LlmUsage(100, 50, 0), long_id)
        self.assertEqual(m.usage().researcher_tokens, 20_000)  # the worst case is still held, not erased
        with self.assertRaises(BudgetExhausted):
            m.reserve_llm(MODEL, 10_000, 10_000, "another")

    def test_charge_llm_without_a_reservation_just_charges(self):
        m = self.meter()
        u = m.charge_llm(MODEL, LlmUsage(100, 50, 0), "direct")
        self.assertEqual(u.researcher_tokens, 150)

    def test_charge_llm_is_idempotent(self):
        m = self.meter()
        m.reserve_llm(MODEL, 100, 100, "call1")
        for _ in range(3):
            u = m.charge_llm(MODEL, LlmUsage(100, 50, 0), "call1")
        self.assertEqual(u.researcher_tokens, 150)
        self.assertEqual(len(self.charges()), 2)

    def test_both_events_are_booked_before_budget_exhausted_is_raised(self):
        # tokens cap 10000 is crossed by the tokens half; the dollar half must still be booked
        m = self.meter()
        with self.assertRaises(BudgetExhausted) as cm:
            m.charge_llm(MODEL, LlmUsage(10_000, 100, 0), "big")
        self.assertEqual(cm.exception.axis, "researcher_tokens")
        ids = {ch.call_id for ch in self.charges()}
        self.assertEqual(ids, {"big:tokens", "big:usd"})
        self.assertAlmostEqual(m.usage().researcher_usd, llm_usd(10_000, 100))

    def test_dollar_half_crossing_still_raises_after_both_are_booked(self):
        m = self.meter(BudgetSpec(10_000_000, 10.0, 0.01, 100.0))
        with self.assertRaises(BudgetExhausted) as cm:
            m.charge_llm(MODEL, LlmUsage(1_000, 5_000, 0), "big")  # 0.02250 dollars
        self.assertEqual(cm.exception.axis, "dollars")
        self.assertEqual({ch.call_id for ch in self.charges()}, {"big:tokens", "big:usd"})

    def test_unsettled_reservation_stays_at_its_worst_case_and_blocks_room(self):
        m = self.meter()
        m.reserve_llm(MODEL, 3_000, 4_000, "lost")  # the stream never reported usage
        self.assertEqual(m.usage().researcher_tokens, 7_000)
        with self.assertRaises(BudgetExhausted):
            m.reserve_llm(MODEL, 3_000, 4_000, "next")
        self.assertEqual(m.usage().researcher_tokens, 7_000)  # still only the worst case, never lower

    def test_unknown_model_is_priced_like_spark_not_free(self):
        m = self.meter()
        u = m.charge_llm("partner/unknown-model", LlmUsage(1_000, 1_000, 0), "x")
        self.assertAlmostEqual(u.researcher_usd, llm_usd(1_000, 1_000))
        self.assertGreater(u.researcher_usd, 0)

    def test_detail_is_added_to_both_events(self):
        m = self.meter()
        m.charge_llm(MODEL, LlmUsage(10, 5, 0), "est", {"usage_estimated": True})
        self.assertTrue(all(ch.detail["usage_estimated"] is True for ch in self.charges()))

    def test_bad_arguments(self):
        m = self.meter()
        with self.assertRaises(ContractError):
            m.charge_llm(MODEL, {"prompt_tokens": 1}, "x")
        with self.assertRaises(ContractError):
            m.reserve_llm(MODEL, -1, 10, "x")
        with self.assertRaises(ContractError):
            m.reserve_llm(MODEL, 10, True, "x")


class ReopenTests(MeterCase):
    def test_a_new_meter_on_a_reopened_ledger_starts_from_what_was_booked(self):
        m = self.meter()
        m.charge(Resource.GPU_MINUTES, 4.0, "g1")
        m.charge_llm(MODEL, LlmUsage(1_000, 500, 0), "c1")
        with self.assertRaises(BudgetExhausted):
            m.charge(Resource.GPU_MINUTES, 7.0, "g2")
        before = m.usage()
        self.ledger.close()
        led = LabLedger.open(self.root / "t", clock=self.clock, fsync=False)
        self.addCleanup(led.close)
        n_events = len(led.events())
        m2 = TrialBudgetMeter(self.spec, led, TRIAL, clock=self.clock)
        after = m2.usage()
        self.assertEqual((after.researcher_tokens, after.gpu_minutes), (before.researcher_tokens, before.gpu_minutes))
        self.assertAlmostEqual(after.researcher_usd, before.researcher_usd)
        # idempotent across the reopen: the same call ids add nothing and write nothing
        m2.charge_llm(MODEL, LlmUsage(1_000, 500, 0), "c1")
        self.assertEqual(len(led.events()), n_events)
        self.assertEqual(m2.usage().researcher_tokens, before.researcher_tokens)
        # the exhausted axis stays closed, and its event is not written a second time
        with self.assertRaises(BudgetExhausted):
            m2.check(Resource.GPU_MINUTES)
        self.assertEqual(len([e for e in led.events() if e.kind == LabKind.BUDGET_EXHAUSTED]), 1)

    def test_an_unsettled_reservation_survives_a_crash_and_a_settled_one_does_not(self):
        spec = BudgetSpec(researcher_tokens=10_000, gpu_minutes=10.0, dollars=1.0, wall_seconds=100.0)
        m = self.meter(spec)
        m.reserve_llm(MODEL, 3_000, 4_000, "inflight")  # forwarded, then the coordinator dies before settlement
        m.reserve_llm(MODEL, 100, 100, "done")
        m.charge_llm(MODEL, LlmUsage(100, 50, 0), "done")
        self.ledger.close()
        led = LabLedger.open(self.root / "t", clock=self.clock, fsync=False)
        self.addCleanup(led.close)
        m2 = TrialBudgetMeter(spec, led, TRIAL, clock=self.clock)
        u = m2.usage()
        self.assertEqual(u.researcher_tokens, 7_000 + 150)
        self.assertAlmostEqual(u.researcher_usd, llm_usd(3_000, 4_000) + llm_usd(100, 50))
        with self.assertRaises(BudgetExhausted):  # the 7,000 in flight still count against the 10,000-token allowance
            m2.reserve_llm(MODEL, 3_000, 4_000, "next")
        # the offline rebuild (abort-trial, lab_spent_usd) agrees with the meter
        got = usage_from_events(led.events(), TRIAL)
        self.assertEqual(got.researcher_tokens, u.researcher_tokens)
        self.assertAlmostEqual(got.researcher_usd, u.researcher_usd)
        # settling the in-flight call after the reopen replaces the worst case with the real usage
        m2.charge_llm(MODEL, LlmUsage(200, 100, 0), "inflight")
        self.assertEqual(m2.usage().researcher_tokens, 150 + 300)

    def test_a_crash_between_the_token_and_dollar_settlement_keeps_the_dollars_reserved(self):
        spec = BudgetSpec(researcher_tokens=10_000, gpu_minutes=10.0, dollars=1.0, wall_seconds=100.0)
        m = self.meter(spec)
        m.reserve_llm(MODEL, 1_000, 1_000, "call1")
        # settlement books tokens first; the process dies before the dollars are written
        self.ledger.append(LabKind.BUDGET_CHARGED, "trial:" + TRIAL, BudgetCharge(TRIAL, Resource.RESEARCHER_TOKENS, 1_500.0, "call1:tokens", {}))
        self.ledger.close()
        led = LabLedger.open(self.root / "t", clock=self.clock, fsync=False)
        self.addCleanup(led.close)
        worst = llm_usd(1_000, 1_000)
        self.assertGreater(worst, 0.0)
        u = TrialBudgetMeter(spec, led, TRIAL, clock=self.clock).usage()
        self.assertEqual(u.researcher_tokens, 1_500)
        self.assertAlmostEqual(u.researcher_usd, worst)
        got = usage_from_events(led.events(), TRIAL)
        self.assertEqual(got.researcher_tokens, 1_500)
        self.assertAlmostEqual(got.researcher_usd, worst)
        # a retried settlement replaces the held dollars with the real figure and does not double count the tokens
        m2 = TrialBudgetMeter(spec, led, TRIAL, clock=self.clock)
        m2.charge_llm(MODEL, LlmUsage(1_000, 500, 0), "call1")
        self.assertAlmostEqual(m2.usage().researcher_usd, llm_usd(1_000, 500))
        self.assertEqual(m2.usage().researcher_tokens, 1_500)

    def test_other_trials_charges_in_the_ledger_are_ignored(self):
        m = self.meter(trial_id="t-0002")
        m.charge(Resource.GPU_MINUTES, 4.0, "g1")
        m1 = self.meter(trial_id=TRIAL)
        self.assertEqual(m1.usage().gpu_minutes, 0.0)

    def test_wall_clock_restarts_from_the_first_event_of_the_ledger(self):
        m = self.meter()
        m.charge(Resource.GPU_MINUTES, 1.0, "g1")
        self.clock.t += 60.0
        m2 = self.meter()
        self.assertEqual(m2.usage().wall_seconds, 60.0)

    def test_usage_from_events_matches_the_meter(self):
        m = self.meter()
        m.charge(Resource.GPU_MINUTES, 2.0, "g1")
        m.charge(Resource.GPU_USD, 0.01, "g1")
        m.charge_llm(MODEL, LlmUsage(1_000, 500, 100), "c1")
        self.clock.t += 30.0
        m.charge(Resource.GPU_MINUTES, 1.0, "g2")
        got = usage_from_events(self.ledger.events(), TRIAL)
        want = m.usage()
        self.assertEqual((got.researcher_tokens, got.gpu_minutes), (want.researcher_tokens, want.gpu_minutes))
        self.assertAlmostEqual(got.researcher_usd, want.researcher_usd)
        self.assertAlmostEqual(got.gpu_usd, want.gpu_usd)
        self.assertEqual(got.wall_seconds, 30.0)
        self.assertEqual(usage_from_events([], TRIAL), BudgetUsage(0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(usage_from_events(self.ledger.events(), "t-9999").gpu_minutes, 0.0)


class FailClosedTests(MeterCase):
    class BrokenLedger:
        """Accepts the first ``ok`` appends, then fails."""

        def __init__(self, ok: int) -> None:
            self.ok, self.events_written = ok, []

        def append(self, kind, actor, record):
            if len(self.events_written) >= self.ok:
                raise OSError("disk full")
            self.events_written.append((kind, record))

        def anchor(self):  # pragma: no cover
            raise NotImplementedError

        def seal(self):  # pragma: no cover
            pass

        def close(self):  # pragma: no cover
            pass

    def test_a_charge_that_cannot_be_recorded_closes_the_axis_and_raises(self):
        led = self.BrokenLedger(ok=1)
        m = self.meter(ledger=led)
        m.charge(Resource.GPU_MINUTES, 1.0, "g1")
        with self.assertRaises(BudgetExhausted) as cm:
            m.charge(Resource.GPU_MINUTES, 1.0, "g2")  # far below the cap, but not recorded
        self.assertEqual(cm.exception.axis, "gpu_minutes")
        self.assertEqual(m.usage().gpu_minutes, 2.0)  # still counted in memory
        with self.assertRaises(BudgetExhausted):
            m.check(Resource.GPU_MINUTES)
        m.check(Resource.RESEARCHER_TOKENS)  # other axes are untouched

    def test_a_retry_of_the_failed_call_id_does_not_double_count(self):
        led = self.BrokenLedger(ok=0)
        m = self.meter(ledger=led)
        for _ in range(2):
            with self.assertRaises((BudgetExhausted,)):
                m.charge(Resource.GPU_MINUTES, 1.0, "g1")
        self.assertLessEqual(m.usage().gpu_minutes, 1.0)

    def test_a_reservation_that_cannot_be_recorded_is_refused_and_books_nothing(self):
        led = self.BrokenLedger(ok=0)
        cap = LabCap(1.0, clock=self.clock)
        m = self.meter(ledger=led, lab_cap=cap)
        with self.assertRaises(BudgetExhausted):
            m.reserve_llm(MODEL, 10, 10, "c1")
        self.assertEqual(m.usage().researcher_tokens, 0)
        self.assertEqual(cap.reserved_usd, 0.0)
        with self.assertRaises(BudgetExhausted):
            m.reserve_llm(MODEL, 10, 10, "c2")

    def test_reserve_is_refused_on_a_closed_axis(self):
        led = self.BrokenLedger(ok=0)
        m = self.meter(ledger=led)
        with self.assertRaises(BudgetExhausted):
            m.charge_llm(MODEL, LlmUsage(10, 10, 0), "c1")
        with self.assertRaises(BudgetExhausted):
            m.reserve_llm(MODEL, 10, 10, "c2")

    def test_exhausted_event_failure_does_not_hide_the_refusal(self):
        led = self.BrokenLedger(ok=0)
        m = self.meter(BudgetSpec(0, 10.0, 1.0, 100.0), ledger=led)
        with self.assertRaises(BudgetExhausted):
            m.check(Resource.RESEARCHER_TOKENS)


class LabCapTests(MeterCase):
    def make_cap(self, cap=0.05, spent=0.0) -> LabCap:
        return LabCap(cap, spent, clock=self.clock)

    def test_cap_arithmetic(self):
        cap = self.make_cap(0.10, 0.02)
        cap.reserve("a", 0.05)
        self.assertAlmostEqual(cap.remaining(), 0.03)
        with self.assertRaises(BudgetExhausted) as cm:
            cap.reserve("b", 0.04)
        self.assertEqual(cm.exception.axis, "lab_dollars")
        cap.reserve("a", 9.0)  # idempotent per key
        self.assertAlmostEqual(cap.remaining(), 0.03)
        cap.settle("a", 0.01)
        self.assertAlmostEqual(cap.spent_usd, 0.03)
        self.assertAlmostEqual(cap.remaining(), 0.07)
        cap.add_spent(0.07)
        self.assertTrue(cap.exhausted())
        with self.assertRaises(ContractError):
            LabCap(-1.0, 0.0)
        with self.assertRaises(ContractError):
            cap.reserve("x", float("nan"))

    def test_the_lab_cap_is_shared_across_trial_meters(self):
        cap = self.make_cap(0.03)
        other = LabLedger.create(self.root / "t2", clock=self.clock, fsync=False)
        self.addCleanup(other.close)
        big = BudgetSpec(10_000_000, 10.0, 100.0, 100.0)
        m1 = self.meter(big, lab_cap=cap, trial_id="t-0001")
        m2 = self.meter(big, ledger=other, lab_cap=cap, trial_id="t-0002")
        m1.reserve_llm(MODEL, 3_000, 4_000, "a")  # 0.02075 of the 0.03 lab cap
        with self.assertRaises(BudgetExhausted) as cm:
            m2.reserve_llm(MODEL, 3_000, 4_000, "b")
        self.assertEqual(cm.exception.axis, "lab_dollars")
        self.assertEqual(m2.usage().researcher_tokens, 0)  # nothing booked on the refused side
        events = [BudgetExhaustedEvent.from_dict(e.data) for e in other.events() if e.kind == LabKind.BUDGET_EXHAUSTED]
        self.assertEqual([e.axis for e in events], ["lab_dollars"])
        # settling m1 at its small real usage frees the room for m2
        m1.charge_llm(MODEL, LlmUsage(100, 50, 0), "a")
        m2.reserve_llm(MODEL, 3_000, 4_000, "b")

    def test_charge_llm_settles_the_lab_cap_exactly_once(self):
        cap = self.make_cap(1.0)
        m = self.meter(BudgetSpec(10_000_000, 10.0, 100.0, 100.0), lab_cap=cap)
        m.reserve_llm(MODEL, 3_000, 4_000, "a")
        self.assertAlmostEqual(cap.reserved_usd, llm_usd(3_000, 4_000))
        m.charge_llm(MODEL, LlmUsage(1_000, 500, 0), "a")
        m.charge_llm(MODEL, LlmUsage(1_000, 500, 0), "a")  # a repeat adds nothing
        self.assertAlmostEqual(cap.spent_usd, llm_usd(1_000, 500))
        self.assertEqual(cap.reserved_usd, 0.0)
        m.charge_llm(MODEL, LlmUsage(1_000, 500, 0), "unreserved")
        self.assertAlmostEqual(cap.spent_usd, 2 * llm_usd(1_000, 500))

    def test_settling_a_recovered_reservation_replaces_its_seeded_dollars(self):
        spec = BudgetSpec(researcher_tokens=10_000, gpu_minutes=10.0, dollars=1.0, wall_seconds=100.0)
        m = self.meter(spec)
        m.reserve_llm(MODEL, 1_000, 1_000, "inflight")  # forwarded, then the coordinator dies
        worst = llm_usd(1_000, 1_000)
        self.ledger.close()
        led = LabLedger.open(self.root / "t", clock=self.clock, fsync=False)
        self.addCleanup(led.close)
        # a restarted coordinator seeds the cap from the ledgers: the in-flight worst case is already counted as spent
        cap = self.make_cap(1.0, usage_from_events(led.events(), TRIAL).researcher_usd)
        self.assertAlmostEqual(cap.spent_usd, worst)
        m2 = TrialBudgetMeter(spec, led, TRIAL, clock=self.clock, lab_cap=cap)
        self.assertAlmostEqual(cap.projected(), worst)  # adopting it does not count it twice
        m2.charge_llm(MODEL, LlmUsage(100, 50, 0), "inflight")
        self.assertAlmostEqual(m2.usage().researcher_usd, llm_usd(100, 50))
        self.assertAlmostEqual(cap.projected(), llm_usd(100, 50))  # the lab cap agrees with the trial
        self.assertEqual(cap.reserved_usd, 0.0)

    def test_a_direct_researcher_usd_charge_counts_but_gpu_spend_does_not(self):
        cap = self.make_cap(1.0)
        m = self.meter(lab_cap=cap)
        m.charge(Resource.RESEARCHER_USD, 0.01, "r")
        m.charge(Resource.RESEARCHER_USD, 0.01, "r")  # idempotent
        m.charge(Resource.GPU_USD, 0.02, "g")
        self.assertAlmostEqual(cap.spent_usd, 0.01)

    def test_check_raises_once_the_lab_cap_is_used_up(self):
        cap = self.make_cap(0.01)
        m = self.meter(BudgetSpec(10_000_000, 10.0, 100.0, 100.0), lab_cap=cap)
        m.check(Resource.RESEARCHER_USD)
        cap.add_spent(0.01)
        for res in (Resource.RESEARCHER_USD, Resource.RESEARCHER_TOKENS):
            with self.assertRaises(BudgetExhausted) as cm:
                m.check(res)
            self.assertEqual(cm.exception.axis, "lab_dollars")
        m.check(Resource.GPU_MINUTES)  # GPU rental is not under the Meta credit

    def test_concurrent_reservations_never_exceed_the_lab_cap(self):
        one = llm_usd(1_000, 1_000)  # 0.0055
        cap = self.make_cap(one * 10.5)
        big = BudgetSpec(10_000_000, 10.0, 100.0, 100.0)
        meters = []
        for k in range(4):
            led = LabLedger.create(self.root / f"c{k}", clock=self.clock, fsync=False)
            self.addCleanup(led.close)
            meters.append(self.meter(big, ledger=led, lab_cap=cap, trial_id=f"t-{k + 10:04d}"))
        ok, refused = [], []

        def go(k: int) -> None:
            for i in range(10):
                try:
                    meters[k].reserve_llm(MODEL, 1_000, 1_000, f"c{i}")
                    ok.append(1)
                except BudgetExhausted:
                    refused.append(1)

        threads = [threading.Thread(target=go, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(ok), 10)
        self.assertEqual(len(refused), 30)
        self.assertLessEqual(cap.reserved_usd, cap.cap_usd)


class ConcurrencyTests(MeterCase):
    def test_concurrent_charges_are_all_booked_and_the_chain_verifies(self):
        m = self.meter(BudgetSpec(10**9, 10**6, 10**6, 10**6))

        def go(k: int) -> None:
            for i in range(20):
                m.charge(Resource.GPU_MINUTES, 1.0, f"gpu:r{k}:{i}")
                m.charge(Resource.GPU_MINUTES, 1.0, f"gpu:r{k}:{i}")  # duplicate

        threads = [threading.Thread(target=go, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(m.usage().gpu_minutes, 160.0)
        self.assertEqual(len(self.charges()), 160)
        self.assertEqual(self.ledger.verify(), (True, None))

    def test_concurrent_reservations_on_one_meter_never_exceed_the_cap(self):
        m = self.meter(BudgetSpec(10_000, 10.0, 100.0, 100.0))
        ok = []

        def go(k: int) -> None:
            for i in range(10):
                try:
                    m.reserve_llm(MODEL, 400, 100, f"c{k}-{i}")
                    ok.append(1)
                except BudgetExhausted:
                    pass

        threads = [threading.Thread(target=go, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(ok), 20)  # 20 x 500 = 10000, and the cap is "no more than"
        self.assertLessEqual(m.usage().researcher_tokens, 10_000)


class LabSpentTests(MeterCase):
    def test_lab_spent_usd_sums_finished_trials_and_candidate_propose_cost(self):
        rec = trial_record()  # budget_used.researcher_usd == 6.5
        state = LabState(trials={"t001": rec}, candidates={"cand-001": candidate()})
        self.assertAlmostEqual(lab_spent_usd(state), 6.5 + 0.31)

    def test_with_a_layout_it_adds_unfinished_trials_and_the_outer_ledger(self):
        layout = LabLayout(self.root / "lab")
        t = LabLedger.create(layout.trial_ledger_dir("t-0007"), clock=self.clock, fsync=False)
        self.addCleanup(t.close)
        m = TrialBudgetMeter(self.spec, t, "t-0007", clock=self.clock)
        m.charge(Resource.RESEARCHER_USD, 0.02, "r1")
        m.charge(Resource.GPU_USD, 0.01, "g1")
        from nightshift.ledger import Ledger

        outer = Ledger.create(layout.outer_dir, clock=self.clock, fsync=False)
        self.addCleanup(outer.close)
        outer.append("budget.charged", "engine", {"kind": "llm", "amount": 0.40, "unit": "usd", "call_id": "w1", "model": "muse-wonder-1.0"})
        outer.append("budget.charged", "engine", {"kind": "experiment", "amount": 1, "unit": "count"})
        state = LabState(trials={"t001": trial_record()}, trials_started={"t-0007": trial_record().spec, "t001": trial_record().spec})
        self.assertAlmostEqual(outer_usd_spent(layout.outer_dir), 0.40)
        self.assertAlmostEqual(lab_spent_usd(state, layout), 6.5 + 0.02 + 0.40)
        self.assertEqual(outer_usd_spent(self.root / "nowhere"), 0.0)


if __name__ == "__main__":
    unittest.main()
