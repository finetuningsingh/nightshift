import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nightshift import budget
from nightshift.budget import BudgetGate, MeteredClient, remaining_from_view
from nightshift.ledger import Ledger
from nightshift.types import (BudgetExceeded, BudgetLimits, ChatMessage, ChatResponse, Kind, ModelError, Usage)
from tests.support.synthetic import FakeClock, RunBuilder

# muse-spark-1.3: $1.25 in, $0.15 cached in, $4.25 out per 1M tokens
SPARK = "muse-spark-1.3"


def charged(led, kind=None):
    return [e for e in led.events() if e.kind == Kind.BUDGET_CHARGED and (kind is None or e.data["kind"] == kind)]


def exhausted(led):
    return [e for e in led.events() if e.kind == Kind.BUDGET_EXHAUSTED]


class Base(unittest.TestCase):
    limits = BudgetLimits(max_experiments=3, max_replications=2, max_llm_usd=1.0, max_supervisor_turns=4,
                          max_wall_sec=100.0)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.led = Ledger.create(Path(self.tmp.name), clock=FakeClock(), fsync=False)
        self.addCleanup(self.led.close)
        self.rb = RunBuilder(self.led)
        self.rb.start(limits=self.limits)
        self.wall = FakeClock(start=1000.0, step=0.0)   # the gate's clock moves only when a test advances it
        self.gate = BudgetGate(self.limits, self.led, clock=self.wall)


class CheckTests(Base):
    def test_check_passes_while_budget_remains(self):
        for action in budget.ACTIONS:
            self.gate.check(action)
        self.assertEqual(exhausted(self.led), [])

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ValueError):
            self.gate.check("coffee")

    def test_experiment_limit_raises_with_kind_limit_used(self):
        for i in range(3):
            self.gate.reserve_experiment(f"e{i + 1:03d}")
        with self.assertRaises(BudgetExceeded) as cm:
            self.gate.check("experiment")
        self.assertEqual((cm.exception.kind, cm.exception.limit, cm.exception.used), ("experiment", 3, 3))

    def test_exhausted_event_written_once_per_kind(self):
        for i in range(3):
            self.gate.reserve_experiment(f"e{i + 1:03d}")
        for _ in range(4):
            with self.assertRaises(BudgetExceeded):
                self.gate.check("experiment")
        with self.assertRaises(BudgetExceeded):
            self.gate.reserve_experiment("e009")
        evs = exhausted(self.led)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].data, {"kind": "experiment", "limit": 3, "used": 3})
        # a different kind gets its own, single, event
        self.rb.turn(4)
        for _ in range(2):
            with self.assertRaises(BudgetExceeded):
                self.gate.check("supervisor_turn")
        self.assertEqual(sorted(e.data["kind"] for e in exhausted(self.led)), ["experiment", "supervisor_turn"])

    def test_exhausted_not_rewritten_by_a_second_gate_after_resume(self):
        self.rb.turn(4)
        with self.assertRaises(BudgetExceeded):
            self.gate.check("supervisor_turn")
        again = BudgetGate(self.limits, self.led, clock=self.wall)
        with self.assertRaises(BudgetExceeded):
            again.check("supervisor_turn")
        self.assertEqual(len(exhausted(self.led)), 1)

    def test_llm_budget_uses_ledger_usd(self):
        self.rb.charge_llm(0.6, "c1")
        self.gate.check("llm")
        self.rb.charge_llm(0.5, "c2")
        with self.assertRaises(BudgetExceeded) as cm:
            self.gate.check("llm")
        self.assertEqual(cm.exception.kind, "llm")
        self.assertAlmostEqual(cm.exception.used, 1.1)
        self.assertEqual(cm.exception.limit, 1.0)

    def test_supervisor_turns_use_view_counter(self):
        self.rb.turn(3)
        self.gate.check("supervisor_turn")
        self.rb.turn(1)
        with self.assertRaises(BudgetExceeded) as cm:
            self.gate.check("supervisor_turn")
        self.assertEqual((cm.exception.limit, cm.exception.used), (4, 4))

    def test_wall_clock_counts_uncharged_time_and_charged_time(self):
        self.wall.advance(60)
        self.gate.check("wall")
        self.gate.charge_wall()            # 60 s now in the ledger, none uncharged
        self.wall.advance(39)
        self.gate.check("wall")            # 60 + 39 < 100
        self.wall.advance(2)
        with self.assertRaises(BudgetExceeded) as cm:
            self.gate.check("wall")
        self.assertEqual(cm.exception.kind, "wall")
        self.assertAlmostEqual(cm.exception.used, 101.0)

    def test_wall_survives_a_new_process(self):
        self.wall.advance(95)
        self.gate.charge_wall()
        later = BudgetGate(self.limits, self.led, clock=self.wall)    # a resumed process
        later_wall = later.snapshot()["used"]["wall_sec"]
        self.assertAlmostEqual(later_wall, 95.0)
        self.wall.advance(6)
        with self.assertRaises(BudgetExceeded):
            later.check("wall")

    def test_replication_limit(self):
        self.gate.reserve_replication("e001")
        self.gate.reserve_replication("e002")
        with self.assertRaises(BudgetExceeded) as cm:
            self.gate.reserve_replication("e003")
        self.assertEqual((cm.exception.kind, cm.exception.limit, cm.exception.used), ("replication", 2, 2))
        self.assertEqual(len(charged(self.led, "replication")), 2)


class ReserveTests(Base):
    def test_reserve_experiment_writes_charge_with_exp_and_purpose(self):
        self.gate.reserve_experiment("e001", "primary")
        (ev,) = charged(self.led, "experiment")
        self.assertEqual(ev.actor, "engine")
        d = ev.data
        self.assertEqual((d["amount"], d["unit"], d["exp_id"], d["purpose"]), (1, "count", "e001", "primary"))
        self.assertEqual(self.led.view().budget_used["experiments"], 1)

    def test_default_purpose_is_primary(self):
        self.gate.reserve_experiment("e001")
        self.assertEqual(charged(self.led, "experiment")[0].data["purpose"], "primary")

    def test_baseline_charges_zero_and_takes_no_slot(self):
        for i in range(3):
            self.gate.reserve_experiment(f"e{i + 1:03d}")
        self.gate.reserve_experiment("e000", "baseline")        # even with every slot used
        base = [e for e in charged(self.led, "experiment") if e.data["exp_id"] == "e000"]
        self.assertEqual(len(base), 1)
        self.assertEqual((base[0].data["amount"], base[0].data["purpose"]), (0, "baseline"))
        self.assertEqual(self.led.view().budget_used["experiments"], 3)

    def test_baseline_still_refused_when_wall_time_is_gone(self):
        self.wall.advance(500)
        with self.assertRaises(BudgetExceeded) as cm:
            self.gate.reserve_experiment("e000", "baseline")
        self.assertEqual(cm.exception.kind, "wall")
        self.assertEqual(charged(self.led), [])

    def test_double_reserve_is_a_noop(self):
        self.gate.reserve_experiment("e001")
        n = len(self.led.events())
        self.gate.reserve_experiment("e001")
        self.assertEqual(len(self.led.events()), n)
        self.assertEqual(self.led.view().budget_used["experiments"], 1)

    def test_same_exp_different_purpose_is_a_separate_charge(self):
        self.gate.reserve_experiment("e001", "primary")
        self.gate.reserve_experiment("e001", "retry")
        self.assertEqual(self.led.view().budget_used["experiments"], 2)

    def test_idempotent_across_a_new_gate_even_when_exhausted(self):
        for i in range(3):
            self.gate.reserve_experiment(f"e{i + 1:03d}")
        resumed = BudgetGate(self.limits, self.led, clock=self.wall)
        resumed.reserve_experiment("e002")                    # already charged: resume must not be blocked or re-charge
        self.assertEqual(len(charged(self.led, "experiment")), 3)
        self.assertEqual(exhausted(self.led), [])

    def test_refused_reserve_charges_nothing(self):
        for i in range(3):
            self.gate.reserve_experiment(f"e{i + 1:03d}")
        with self.assertRaises(BudgetExceeded):
            self.gate.reserve_experiment("e004")
        self.assertEqual(len(charged(self.led, "experiment")), 3)
        with self.assertRaises(BudgetExceeded):          # and it is not remembered as charged
            self.gate.reserve_experiment("e004")

    def test_empty_exp_id_rejected(self):
        with self.assertRaises(ValueError):
            self.gate.reserve_experiment("")

    def test_reserve_replication_payload_and_idempotence(self):
        self.gate.reserve_replication("e001")
        self.gate.reserve_replication("e001")
        (ev,) = charged(self.led, "replication")
        d = ev.data
        self.assertEqual((d["amount"], d["unit"], d["exp_id"], d["purpose"]), (1, "count", "e001", "replication"))
        self.assertEqual(self.led.view().budget_used["replications"], 1)
        self.assertEqual(self.led.view().budget_used.get("experiments", 0), 0)


class ChargeLLMTests(Base):
    def test_cost_matches_published_price(self):
        usd = self.gate.charge_llm(SPARK, Usage(prompt_tokens=1_000_000, completion_tokens=100_000), "c1", "supervisor")
        self.assertAlmostEqual(usd, 1.25 + 0.425)

    def test_cached_tokens_cost_less(self):
        full = self.gate.charge_llm(SPARK, Usage(prompt_tokens=100_000), "c1", "supervisor")
        part = self.gate.charge_llm(SPARK, Usage(prompt_tokens=100_000, cached_tokens=100_000), "c2", "supervisor")
        self.assertLess(part, full)

    def test_pricing_module_is_used_for_cost(self):
        with mock.patch.object(budget, "_usd_for", lambda model, usage: 0.42):
            self.assertEqual(self.gate.charge_llm("any-model", Usage(prompt_tokens=1), "c1", "supervisor"), 0.42)
        self.assertEqual(self.led.view().budget_used["usd"], 0.42)

    def test_payload_has_every_usage_key(self):
        self.gate.charge_llm(SPARK, Usage(prompt_tokens=50, completion_tokens=7, cached_tokens=10), "c1", "designer:A",
                             reasoning_effort="high", latency_ms=812.5, retries=2, schema_ok=True, stream_fallback=True,
                             cached_known=True, requested_model="muse-wonder-1.0", exp_id="e001")
        (ev,) = charged(self.led, "llm")
        self.assertEqual(ev.actor, "designer:A")
        d = ev.data
        self.assertEqual((d["kind"], d["unit"], d["call_id"], d["model"], d["exp_id"]), ("llm", "usd", "c1", SPARK, "e001"))
        self.assertGreater(d["amount"], 0)
        self.assertEqual(d["usage"], {
            "prompt_tokens": 50, "completion_tokens": 7, "cached_tokens": 10, "cached_known": True,
            "reasoning_effort": "high", "latency_ms": 812.5, "retries": 2, "schema_ok": True, "stream_fallback": True,
            "requested_model": "muse-wonder-1.0"})

    def test_usage_defaults_when_detail_is_absent(self):
        self.gate.charge_llm(SPARK, Usage(prompt_tokens=5), "c1", "supervisor")
        u = charged(self.led, "llm")[0].data["usage"]
        self.assertEqual(set(u), set(budget.USAGE_KEYS))
        self.assertIs(u["cached_known"], False)
        self.assertEqual((u["retries"], u["stream_fallback"], u["requested_model"]), (0, False, SPARK))
        self.assertIsNone(u["reasoning_effort"])

    def test_cached_tokens_imply_cached_known(self):
        self.gate.charge_llm(SPARK, Usage(prompt_tokens=5, cached_tokens=3), "c1", "supervisor")
        self.assertIs(charged(self.led, "llm")[0].data["usage"]["cached_known"], True)

    def test_second_charge_with_same_call_id_is_free_and_unrecorded(self):
        first = self.gate.charge_llm(SPARK, Usage(prompt_tokens=1000), "dup", "supervisor")
        second = self.gate.charge_llm(SPARK, Usage(prompt_tokens=1000), "dup", "supervisor")
        self.assertGreater(first, 0)
        self.assertEqual(second, 0.0)
        self.assertEqual(len(charged(self.led, "llm")), 1)

    def test_call_id_already_in_ledger_is_not_charged_again(self):
        self.rb.charge_llm(0.1, "from-before-a-crash")
        fresh_gate = BudgetGate(self.limits, self.led, clock=self.wall)
        self.assertEqual(fresh_gate.charge_llm(SPARK, Usage(prompt_tokens=1000), "from-before-a-crash", "x"), 0.0)
        self.assertEqual(self.led.view().budget_used["usd"], 0.1)

    def test_missing_call_id_is_an_error(self):
        with self.assertRaises(ValueError):
            self.gate.charge_llm(SPARK, Usage(), "", "x")

    def test_unknown_model_still_gets_a_price(self):
        self.assertGreater(self.gate.charge_llm("some-new-model", Usage(prompt_tokens=1_000_000), "c1", "x"), 0)

    def test_llm_charge_does_not_check_limits(self):
        self.gate.charge_llm(SPARK, Usage(prompt_tokens=10_000_000), "big", "x")   # ~$12.5 > the $1 limit; the money is spent
        self.assertGreater(self.led.view().budget_used["usd"], self.limits.max_llm_usd)
        with self.assertRaises(BudgetExceeded):
            self.gate.check("llm")


class ChargeWallTests(Base):
    def test_first_charge_measures_from_construction(self):
        self.wall.advance(12.5)
        self.assertEqual(self.gate.charge_wall(), 12.5)
        (ev,) = charged(self.led, "wall")
        self.assertEqual((ev.data["amount"], ev.data["unit"]), (12.5, "sec"))

    def test_later_charges_are_deltas(self):
        self.wall.advance(10)
        self.gate.charge_wall()
        self.wall.advance(4)
        self.assertEqual(self.gate.charge_wall(), 4)
        self.assertEqual(self.led.view().budget_used["wall_sec"], 14)

    def test_zero_delta_writes_nothing(self):
        self.assertEqual(self.gate.charge_wall(), 0.0)
        self.assertEqual(charged(self.led, "wall"), [])


class RemainingTests(Base):
    def test_remaining_and_snapshot(self):
        self.gate.reserve_experiment("e001")
        self.gate.reserve_replication("e001")
        self.gate.charge_llm(SPARK, Usage(prompt_tokens=100_000), "c1", "x")
        self.rb.turn(2)
        self.wall.advance(30)
        snap = self.gate.snapshot()
        self.assertEqual(snap["limits"], self.limits.to_dict())
        self.assertEqual(snap["used"]["experiments"], 1)
        self.assertEqual(snap["used"]["replications"], 1)
        self.assertEqual(snap["used"]["supervisor_turns"], 2)
        self.assertEqual(snap["used"]["llm_calls"], 1)
        self.assertAlmostEqual(snap["used"]["llm_usd"], 0.125)
        self.assertAlmostEqual(snap["used"]["wall_sec"], 30.0)
        rem = self.gate.remaining()
        self.assertEqual(rem, snap["remaining"])
        self.assertEqual(rem["experiments"], 2)
        self.assertEqual(rem["replications"], 1)
        self.assertEqual(rem["supervisor_turns"], 2)
        self.assertAlmostEqual(rem["llm_usd"], 0.875)
        self.assertAlmostEqual(rem["wall_sec"], 70.0)
        self.assertEqual(snap["exhausted"], [])

    def test_remaining_never_negative_and_lists_exhausted(self):
        self.rb.charge_llm(5.0, "c1")
        self.wall.advance(1000)
        snap = self.gate.snapshot()
        self.assertEqual(snap["remaining"]["llm_usd"], 0.0)
        self.assertEqual(snap["remaining"]["wall_sec"], 0.0)
        self.assertEqual(snap["exhausted"], ["llm_usd", "wall_sec"])

    def test_remaining_from_view_is_pure(self):
        self.rb.baseline(1.0)
        self.rb.charge_llm(0.25, "c1")
        self.rb.turn(1)
        view = self.led.view()
        before = len(self.led.events())
        a = remaining_from_view(self.limits, view, now=view.first_ts + 10)
        b = remaining_from_view(self.limits, view, now=view.first_ts + 10)
        self.assertEqual(a, b)
        self.assertEqual(len(self.led.events()), before)
        self.assertEqual(a["remaining"]["supervisor_turns"], 3)
        self.assertAlmostEqual(a["remaining"]["llm_usd"], 0.75)
        # without a wall mark the dashboard estimate is the elapsed time since the first event
        self.assertAlmostEqual(a["used"]["wall_sec"], 10.0)

    def test_remaining_from_view_exact_wall_with_mark(self):
        view = self.led.view()
        r = remaining_from_view(self.limits, view, now=500.0, wall_mark=470.0)
        self.assertAlmostEqual(r["used"]["wall_sec"], 30.0)

    def test_finished_run_wall_estimate_stops_growing(self):
        self.rb.finish()
        view = self.led.view()
        a = remaining_from_view(self.limits, view, now=view.last_ts + 1)
        b = remaining_from_view(self.limits, view, now=view.last_ts + 100_000)
        self.assertEqual(a["used"]["wall_sec"], b["used"]["wall_sec"])


class FakeInner:
    """A ModelClient that returns a canned response or raises."""

    def __init__(self, model="muse-spark-1.3", answers_as=None, usage=None, text="{}", raises=None, raw=None):
        self.model = model
        self.answers_as = answers_as or model
        self.usage = usage or Usage(prompt_tokens=1000, completion_tokens=100, cached_tokens=0, total_tokens=1100)
        self.text = text
        self.raises = raises
        self.raw = raw or {}
        self.calls = []

    def chat(self, messages, *, tools=None, response_schema=None, schema_name="response", reasoning_effort=None,
             max_completion_tokens=None, temperature=None, cache_key=None, timeout=None):
        self.calls.append({"messages": list(messages), "reasoning_effort": reasoning_effort, "cache_key": cache_key,
                           "response_schema": response_schema, "timeout": timeout})
        if self.raises:
            raise self.raises
        return ChatResponse(message=ChatMessage(role="assistant", content=self.text), finish_reason="stop",
                            usage=self.usage, model=self.answers_as, raw=self.raw)


class MeteredClientTests(Base):
    msgs = [ChatMessage(role="user", content="hi")]

    def test_exposes_inner_model(self):
        self.assertEqual(MeteredClient(FakeInner("m-1"), self.gate, "supervisor").model, "m-1")

    def test_charges_after_call_from_response_usage_and_model(self):
        inner = FakeInner(model="muse-wonder-1.0", answers_as="muse-spark-1.3")
        client = MeteredClient(inner, self.gate, "reviewer:A")
        resp = client.chat(self.msgs, reasoning_effort="xhigh")
        self.assertEqual(resp.text, "{}")
        (ev,) = charged(self.led, "llm")
        self.assertEqual(ev.actor, "reviewer:A")
        self.assertEqual(ev.data["model"], "muse-spark-1.3")                     # the model that answered
        self.assertEqual(ev.data["usage"]["requested_model"], "muse-wonder-1.0")  # the one asked for
        self.assertEqual(ev.data["usage"]["reasoning_effort"], "xhigh")
        self.assertEqual((ev.data["usage"]["prompt_tokens"], ev.data["usage"]["completion_tokens"]), (1000, 100))
        self.assertAlmostEqual(ev.data["amount"], (1000 * 1.25 + 100 * 4.25) / 1e6)

    def test_latency_is_measured_and_forwarded(self):
        ticks = iter([10.0, 10.25])
        client = MeteredClient(FakeInner(), self.gate, "x", timer=lambda: next(ticks))
        client.chat(self.msgs)
        self.assertEqual(charged(self.led, "llm")[0].data["usage"]["latency_ms"], 250.0)

    def test_arguments_are_forwarded_to_the_inner_client(self):
        inner = FakeInner()
        MeteredClient(inner, self.gate, "x").chat(self.msgs, reasoning_effort="low", cache_key="k", timeout=9.0)
        call = inner.calls[0]
        self.assertEqual((call["reasoning_effort"], call["cache_key"], call["timeout"]), ("low", "k", 9.0))

    def test_every_call_gets_a_fresh_call_id(self):
        client = MeteredClient(FakeInner(), self.gate, "x")
        for _ in range(5):
            client.chat(self.msgs)
        ids = [e.data["call_id"] for e in charged(self.led, "llm")]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)

    def test_model_error_passes_through_uncharged(self):
        err = ModelError("rate limited", status=429, retryable=True)
        client = MeteredClient(FakeInner(raises=err), self.gate, "x")
        with self.assertRaises(ModelError) as cm:
            client.chat(self.msgs)
        self.assertIs(cm.exception, err)
        self.assertEqual(charged(self.led, "llm"), [])

    def test_checks_before_calling_and_never_calls_when_exhausted(self):
        self.rb.charge_llm(2.0, "spent")
        inner = FakeInner()
        client = MeteredClient(inner, self.gate, "x")
        with self.assertRaises(BudgetExceeded) as cm:
            client.chat(self.msgs)
        self.assertEqual(cm.exception.kind, "llm")
        self.assertEqual(inner.calls, [])
        self.assertEqual(len(exhausted(self.led)), 1)

    def test_schema_ok_reflects_whether_the_reply_parsed(self):
        schema = {"type": "object"}
        MeteredClient(FakeInner(text='{"a": 1}'), self.gate, "x").chat(self.msgs, response_schema=schema)
        MeteredClient(FakeInner(text="not json"), self.gate, "x").chat(self.msgs, response_schema=schema)
        MeteredClient(FakeInner(text="plain"), self.gate, "x").chat(self.msgs)
        flags = [e.data["usage"]["schema_ok"] for e in charged(self.led, "llm")]
        self.assertEqual(flags, [True, False, None])

    def test_optional_client_meta_is_forwarded(self):
        raw = {"nightshift": {"retries": 3, "stream_fallback": True, "cached_known": True}}
        MeteredClient(FakeInner(raw=raw), self.gate, "x").chat(self.msgs)
        u = charged(self.led, "llm")[0].data["usage"]
        self.assertEqual((u["retries"], u["stream_fallback"], u["cached_known"]), (3, True, True))

    def test_response_without_usage_still_charges_zero_tokens(self):
        inner = FakeInner()
        inner.usage = None
        MeteredClient(inner, self.gate, "x").chat(self.msgs)
        (ev,) = charged(self.led, "llm")
        self.assertEqual(ev.data["amount"], 0.0)


class ConcurrencyTests(Base):
    def _run(self, n, fn):
        errors = []
        start = threading.Barrier(n)

        def work(i):
            start.wait()
            try:
                fn(i)
            except BudgetExceeded as exc:
                errors.append(exc)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in ts))
        return errors

    def test_with_one_slot_left_exactly_one_reserve_wins(self):
        self.gate.reserve_experiment("e001")
        self.gate.reserve_experiment("e002")
        errors = self._run(8, lambda i: self.gate.reserve_experiment(f"e1{i:02d}"))
        self.assertEqual(len(errors), 7)
        self.assertEqual(self.led.view().budget_used["experiments"], 3)
        self.assertEqual(len(charged(self.led, "experiment")), 3)
        self.assertEqual(len(exhausted(self.led)), 1)
        self.assertTrue(all(e.kind == "experiment" for e in errors))

    def test_same_experiment_reserved_by_eight_threads_charges_once(self):
        errors = self._run(8, lambda i: self.gate.reserve_experiment("e001"))
        self.assertEqual(errors, [])
        self.assertEqual(len(charged(self.led, "experiment")), 1)

    def test_with_one_replication_left_exactly_one_wins(self):
        self.gate.reserve_replication("e001")
        errors = self._run(8, lambda i: self.gate.reserve_replication(f"e1{i:02d}"))
        self.assertEqual(len(errors), 7)
        self.assertEqual(self.led.view().budget_used["replications"], 2)

    def test_duplicate_call_ids_are_charged_once_each(self):
        totals = []
        lock = threading.Lock()

        def work(i):
            for k in range(25):
                usd = self.gate.charge_llm(SPARK, Usage(prompt_tokens=1000, completion_tokens=100), f"call-{k}", "x")
                with lock:
                    totals.append(usd)

        self._run(8, work)                                 # 200 charge_llm calls, 25 distinct ids
        llm = charged(self.led, "llm")
        self.assertEqual(len(llm), 25)
        self.assertEqual(len({e.data["call_id"] for e in llm}), 25)
        self.assertEqual(sum(1 for t in totals if t > 0), 25)
        self.assertEqual(len(totals), 200)
        self.assertAlmostEqual(self.led.view().budget_used["usd"], 25 * (1000 * 1.25 + 100 * 4.25) / 1e6)
        self.assertEqual(self.led.verify_chain(), (True, None))


class AdversarialReviewTests(Base):
    """Regressions and extra coverage found by the adversarial review."""

    msgs = [ChatMessage(role="user", content="hi")]

    def test_purpose_none_is_the_primary_slot_and_idempotent_across_gates(self):
        self.gate.reserve_experiment("e001", None)          # type: ignore[arg-type]
        BudgetGate(self.limits, self.led, clock=self.wall).reserve_experiment("e001")
        self.gate.reserve_experiment("e001", "primary")
        (ev,) = charged(self.led, "experiment")
        self.assertEqual(ev.data["purpose"], "primary")

    def test_two_gates_on_one_ledger_do_not_both_take_the_last_slot(self):
        self.gate.reserve_experiment("e001")
        self.gate.reserve_experiment("e002")
        other = BudgetGate(self.limits, self.led, clock=self.wall)
        gates = [self.gate, other]
        wins, losses = [], []
        start = threading.Barrier(8)

        def work(i):
            start.wait()
            try:
                gates[i % 2].reserve_experiment(f"e1{i:02d}")
                wins.append(i)
            except BudgetExceeded:
                losses.append(i)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        self.assertEqual((len(wins), len(losses)), (1, 7))
        self.assertEqual(self.led.view().budget_used["experiments"], 3)

    def test_two_gates_charge_a_shared_call_id_once(self):
        other = BudgetGate(self.limits, self.led, clock=self.wall)
        a = self.gate.charge_llm(SPARK, Usage(prompt_tokens=1000), "same", "x")
        b = other.charge_llm(SPARK, Usage(prompt_tokens=1000), "same", "x")
        self.assertGreater(a, 0)
        self.assertEqual(b, 0.0)
        self.assertEqual(len(charged(self.led, "llm")), 1)

    def test_metered_client_records_the_effort_the_client_actually_sent(self):
        raw = {"nightshift": {"reasoning_effort": "medium"}}
        MeteredClient(FakeInner(raw=raw), self.gate, "x").chat(self.msgs)                       # default effort inside the client
        # The request said "high" but the client reports that it put "medium" on the wire (a fallback model has its own
        # effort): the charge records what was sent. (This test used to expect the requested "high".)
        MeteredClient(FakeInner(raw=raw), self.gate, "x").chat(self.msgs, reasoning_effort="high")
        efforts = [e.data["usage"]["reasoning_effort"] for e in charged(self.led, "llm")]
        self.assertEqual(efforts, ["medium", "medium"])

    def test_metered_client_is_a_model_client(self):
        from nightshift.types import ModelClient
        self.assertIsInstance(MeteredClient(FakeInner(), self.gate, "x"), ModelClient)

    def test_metered_client_from_many_threads_charges_every_call_once(self):
        client = MeteredClient(FakeInner(), self.gate, "x")
        big = BudgetGate(BudgetLimits(max_llm_usd=1000.0), self.led, clock=self.wall)
        client = MeteredClient(FakeInner(), big, "x")
        start = threading.Barrier(8)

        def work(_):
            start.wait()
            for _ in range(10):
                client.chat(self.msgs)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        ids = [e.data["call_id"] for e in charged(self.led, "llm")]
        self.assertEqual((len(ids), len(set(ids))), (80, 80))
        self.assertEqual(self.led.verify_chain(), (True, None))

    def test_budget_error_from_the_inner_client_is_not_swallowed_or_charged(self):
        client = MeteredClient(FakeInner(raises=BudgetExceeded("inner", kind="llm")), self.gate, "x")
        with self.assertRaises(BudgetExceeded):
            client.chat(self.msgs)
        self.assertEqual(charged(self.led, "llm"), [])

    def test_check_replication_and_wall_write_one_exhausted_event_each(self):
        self.gate.reserve_replication("e001")
        self.gate.reserve_replication("e002")
        self.wall.advance(500)
        for action in ("replication", "replication", "wall", "wall"):
            with self.assertRaises(BudgetExceeded) as cm:
                self.gate.check(action)
            self.assertEqual(cm.exception.kind, action)
        self.assertEqual(sorted(e.data["kind"] for e in exhausted(self.led)), ["replication", "wall"])

    def test_baseline_reserve_is_idempotent(self):
        self.gate.reserve_experiment("e000", "baseline")
        self.gate.reserve_experiment("e000", "baseline")
        self.assertEqual(len(charged(self.led, "experiment")), 1)

    def test_experiment_and_replication_slots_are_independent(self):
        for i in range(3):
            self.gate.reserve_experiment(f"e{i + 1:03d}")
        self.gate.reserve_replication("e001")           # experiments exhausted, replication budget untouched
        self.assertEqual(self.gate.remaining()["replications"], 1)
        self.assertEqual(self.gate.remaining()["experiments"], 0)

    def test_snapshot_has_the_contract_keys(self):
        snap = self.gate.snapshot()
        self.assertEqual(set(snap["remaining"]), {"experiments", "replications", "llm_usd", "supervisor_turns", "wall_sec"})
        self.assertEqual(set(snap), {"limits", "used", "remaining", "exhausted"})

    def test_remaining_from_view_accepts_the_dashboard_call_shape(self):
        view = self.led.view()
        out = remaining_from_view(self.limits, view, view.first_ts + 5)       # positional (limits, view, now)
        self.assertEqual(out["remaining"]["experiments"], 3)

    def test_nonfinite_and_negative_costs_are_recorded_as_zero(self):
        with mock.patch.object(budget, "_usd_for", lambda m, u: float("nan")):
            self.assertEqual(self.gate.charge_llm("m", Usage(prompt_tokens=5), "n1", "x"), 0.0)
        with mock.patch.object(budget, "_usd_for", lambda m, u: -3.0):
            self.assertEqual(self.gate.charge_llm("m", Usage(prompt_tokens=5), "n2", "x"), 0.0)
        self.assertEqual(self.led.view().budget_used["usd"], 0.0)

    def test_real_pricing_module_is_wired_in(self):
        from nightshift.models import pricing
        self.assertIs(budget._usd_for, pricing.usd_for)
        u = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        self.assertAlmostEqual(self.gate.charge_llm("muse-wonder-1.0", u, "w", "x"), pricing.usd_for("muse-wonder-1.0", u))



class ChargeContextTests(Base):
    """LLM spend is attributable per experiment and per lane without a client of its own per lane."""
    msgs = [ChatMessage(role="user", content="hi")]

    def llm(self):
        return charged(self.led, "llm")

    def test_exp_id_purpose_and_a_lane_suffix_are_written_into_the_charge(self):
        client = MeteredClient(FakeInner(), self.gate, "designer")
        client.chat(self.msgs, charge_context={"exp_id": "e007", "purpose": "implement", "actor": "designer:A"})
        (ev,) = self.llm()
        self.assertEqual((ev.actor, ev.data["exp_id"], ev.data["purpose"]), ("designer:A", "e007", "implement"))
        self.assertTrue(ev.data["call_id"].startswith("designer:"))           # call ids stay keyed by the role

    def test_a_bare_lane_label_and_a_full_actor_give_the_same_actor(self):
        client = MeteredClient(FakeInner(), self.gate, "reviewer_hard")
        for given in ("B", "reviewer:B", "reviewer_hard:B"):
            client.chat(self.msgs, charge_context={"actor": given})
        self.assertEqual([e.actor for e in self.llm()], ["reviewer_hard:B"] * 3)   # the client's own role always leads

    def test_no_actor_or_the_bare_role_leaves_the_actor_alone(self):
        client = MeteredClient(FakeInner(), self.gate, "reflector")
        client.chat(self.msgs, charge_context={"purpose": "reflect"})
        client.chat(self.msgs, charge_context={"actor": "reflector"})
        client.chat(self.msgs)
        self.assertEqual([e.actor for e in self.llm()], ["reflector"] * 3)
        self.assertEqual([(e.data["exp_id"], e.data["purpose"]) for e in self.llm()], [(None, "reflect"), (None, None), (None, None)])

    def test_the_context_is_never_forwarded_to_the_inner_client(self):
        class Strict:                                   # the ModelClient signature: an unexpected keyword is a TypeError
            model = SPARK

            def chat(self, messages, *, tools=None, response_schema=None, schema_name="response", reasoning_effort=None,
                     max_completion_tokens=None, temperature=None, cache_key=None, timeout=None):
                return ChatResponse(message=ChatMessage(role="assistant", content="{}"), finish_reason="stop",
                                    usage=Usage(prompt_tokens=10, completion_tokens=1), model=SPARK)
        MeteredClient(Strict(), self.gate, "x").chat(self.msgs, charge_context={"exp_id": "e001", "purpose": "p", "actor": "A"})
        self.assertEqual(len(self.llm()), 1)

    def test_labels_are_reduced_to_safe_characters_and_capped(self):
        client = MeteredClient(FakeInner(), self.gate, "designer")
        client.chat(self.msgs, charge_context={"exp_id": "e1\n<b>x</b>" + "z" * 100, "purpose": "a b/c", "actor": "A B\n"})
        (ev,) = self.llm()
        self.assertRegex(ev.data["exp_id"], r"^[A-Za-z0-9_.-]{1,40}$")
        self.assertEqual(ev.data["purpose"], "a_b_c")
        self.assertRegex(ev.actor, r"^designer:[A-Za-z0-9_.-]+$")

    def test_a_context_that_is_not_a_mapping_or_has_junk_values_is_ignored(self):
        client = MeteredClient(FakeInner(), self.gate, "x")
        for ctx in ("exp_id=e1", ["e1"], 5, {}, {"exp_id": 7, "purpose": None, "actor": 3}):
            client.chat(self.msgs, charge_context=ctx)
        self.assertEqual([(e.actor, e.data["exp_id"], e.data["purpose"]) for e in self.llm()], [("x", None, None)] * 5)

    def test_the_context_does_not_change_the_budget_check_or_idempotence(self):
        self.rb.charge_llm(2.0, "spent")
        inner = FakeInner()
        with self.assertRaises(BudgetExceeded):
            MeteredClient(inner, self.gate, "x").chat(self.msgs, charge_context={"exp_id": "e001"})
        self.assertEqual(inner.calls, [])

    def test_metered_client_advertises_support_and_scripted_clients_do_not(self):
        from nightshift.models.scripted import ScriptedClient
        self.assertTrue(MeteredClient.supports_charge_context)
        self.assertFalse(getattr(ScriptedClient(), "supports_charge_context", False))

    def test_a_shared_client_attributes_concurrent_lanes_separately(self):
        client = MeteredClient(FakeInner(), BudgetGate(BudgetLimits(max_llm_usd=1000.0), self.led, clock=self.wall), "implementer")
        start = threading.Barrier(4)

        def work(lane):
            start.wait()
            for _ in range(5):
                client.chat(self.msgs, charge_context={"exp_id": f"e00{lane}", "purpose": "implement", "actor": f"implementer:{lane}"})
        ts = [threading.Thread(target=work, args=(i,)) for i in range(1, 5)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        pairs = sorted({(e.actor, e.data["exp_id"]) for e in self.llm()})
        self.assertEqual(pairs, [(f"implementer:{i}", f"e00{i}") for i in range(1, 5)])
        self.assertEqual(len(self.llm()), 20)


class RecordedEffortTests(Base):
    """``usage.reasoning_effort`` is what went on the wire, not what was requested."""
    msgs = [ChatMessage(role="user", content="hi")]

    def effort(self, raw, requested):
        MeteredClient(FakeInner(raw=raw), self.gate, "x").chat(self.msgs, reasoning_effort=requested)
        return charged(self.led, "llm")[-1].data["usage"]["reasoning_effort"]

    def test_a_fallback_models_own_effort_is_recorded(self):
        raw = {"_nightshift": {"reasoning_effort": "xhigh", "fallback_used": True}}
        self.assertEqual(self.effort(raw, "high"), "xhigh")

    def test_the_underscore_spelling_wins_over_the_plain_one(self):
        raw = {"_nightshift": {"reasoning_effort": "xhigh"}, "nightshift": {"reasoning_effort": "low"}}
        self.assertEqual(self.effort(raw, "high"), "xhigh")

    def test_the_plain_spelling_is_still_read(self):
        self.assertEqual(self.effort({"nightshift": {"reasoning_effort": "minimal"}}, "high"), "minimal")

    def test_none_sent_is_recorded_as_none_not_as_the_request(self):
        # the host rejected the parameter and the client stripped it: nothing was sent, so nothing is claimed
        self.assertIsNone(self.effort({"_nightshift": {"reasoning_effort": None}}, "high"))

    def test_a_client_that_reports_nothing_gets_the_requested_effort(self):
        self.assertEqual(self.effort({}, "low"), "low")
        self.assertEqual(self.effort({"_nightshift": {"retries": 1}}, "medium"), "medium")
        self.assertIsNone(self.effort({}, None))

    def test_a_response_whose_raw_is_not_a_dict_is_tolerated(self):
        self.assertEqual(self.effort(["not", "a", "dict"], "low"), "low")


if __name__ == "__main__":
    unittest.main()
