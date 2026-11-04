"""Tests for nightshift.lab.ledger (the write side of the lab ledgers).

Run: PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_ledger
"""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nightshift import secrets
from nightshift.lab import contracts as c
from nightshift.lab.contracts import LabKind, LedgerError, LedgerWriter, OperatorChange, PAYLOAD_TYPES
from nightshift.lab.ledger import LabLedger, LedgerPayloadError, register_secrets
from tests.test_lab_contracts import NOW, PERSISTED


class FakeClock:
    def __init__(self, t: float = 1_793_534_400.0) -> None:
        self.t = t

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def op(detail: str = "a change") -> OperatorChange:
    return OperatorChange("proposer_prompt", detail, 0, NOW)


class LedgerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "ledger"
        self.clock = FakeClock()
        secrets.clear()
        self.addCleanup(secrets.clear)

    def make(self) -> LabLedger:
        led = LabLedger.create(self.dir, clock=self.clock, fsync=False)
        self.addCleanup(led.close)
        return led


class KindAndPayloadTests(LedgerTestCase):
    def test_every_lab_kind_accepts_its_contract_record_and_the_chain_verifies(self):
        led = self.make()
        self.assertEqual(set(PAYLOAD_TYPES), set(LabKind.all()))
        for kind, cls in PAYLOAD_TYPES.items():
            with self.subTest(kind):
                ev = led.append(kind, "coordinator", PERSISTED[cls.__name__]())
                self.assertEqual(ev.kind, kind)
                self.assertEqual(c.payload_from_dict(kind, ev.data), PERSISTED[cls.__name__]())
        self.assertEqual(led.verify(), (True, None))
        self.assertEqual(len(led.events()), len(PAYLOAD_TYPES))

    def test_unknown_or_foreign_kind_is_refused_and_nothing_is_written(self):
        led = self.make()
        for kind in ("experiment.proposed", "budget.charged", "lab.nope", "", None, 7):
            with self.subTest(kind=kind):
                with self.assertRaises(LedgerError):
                    led.append(kind, "coordinator", op())
        self.assertEqual(led.events(), [])

    def test_wrong_payload_class_is_refused(self):
        led = self.make()
        with self.assertRaises(LedgerPayloadError):
            led.append(LabKind.OPERATOR_CHANGE, "coordinator", PERSISTED["BudgetSpec"]())
        with self.assertRaises(LedgerPayloadError):
            led.append(LabKind.OPERATOR_CHANGE, "coordinator", op().to_dict())  # a dict is not a contract record
        with self.assertRaises(LedgerPayloadError):
            led.append(LabKind.TRIAL_STARTED, "coordinator", PERSISTED["TrialRecord"]())
        self.assertEqual(led.events(), [])

    def test_payload_error_is_both_a_ledger_error_and_a_contract_error(self):
        led = self.make()
        with self.assertRaises(c.ContractError):
            led.append(LabKind.OPERATOR_CHANGE, "coordinator", object())
        with self.assertRaises(LedgerError):
            led.append(LabKind.OPERATOR_CHANGE, "coordinator", object())

    def test_overlong_string_and_non_finite_number_are_refused_before_any_write(self):
        class Big(OperatorChange):
            def to_dict(self):
                d = super().to_dict()
                d["detail"] = "x" * 9000
                return d

        class Nan(OperatorChange):
            def to_dict(self):
                d = super().to_dict()
                d["candidates_so_far"] = float("nan")
                return d

        led = self.make()
        for cls in (Big, Nan):
            with self.subTest(cls.__name__):
                with self.assertRaises(LedgerPayloadError):
                    led.append(LabKind.OPERATOR_CHANGE, "coordinator", cls("t", "d", 0, NOW))
        self.assertEqual(led.events(), [])

    def test_actor_must_be_a_lab_actor(self):
        led = self.make()
        for actor in ("coordinator", "scorer", "outer", "trial:t-0001"):
            led.append(LabKind.OPERATOR_CHANGE, actor, op(actor))
        for actor in ("engine", "trial:", "trial:../x", "Coordinator", "", None):
            with self.subTest(actor=actor):
                with self.assertRaises(LedgerError):
                    led.append(LabKind.OPERATOR_CHANGE, actor, op())
        self.assertEqual(len(led.events()), 4)

    def test_implements_the_ledger_writer_protocol(self):
        self.assertIsInstance(self.make(), LedgerWriter)


class FreezeOnceTests(LedgerTestCase):
    def test_second_event_of_every_freeze_once_kind_is_refused(self):
        led = self.make()
        for kind in LabKind.FREEZE_ONCE:
            with self.subTest(kind):
                rec = PERSISTED[PAYLOAD_TYPES[kind].__name__]()
                led.append(kind, "coordinator", rec)
                with self.assertRaises(LedgerError):
                    led.append(kind, "coordinator", rec)
        self.assertEqual(len(led.events()), len(LabKind.FREEZE_ONCE))
        self.assertEqual(led.verify(), (True, None))

    def test_freeze_once_survives_a_reopen(self):
        led = self.make()
        led.append(LabKind.RULE_FROZEN, "coordinator", PERSISTED["FrozenFile"]())
        led.close()
        again = LabLedger.open(self.dir, clock=self.clock, fsync=False)
        self.addCleanup(again.close)
        with self.assertRaises(LedgerError):
            again.append(LabKind.RULE_FROZEN, "coordinator", PERSISTED["FrozenFile"]())
        # a different frozen kind that carries the same payload class is still allowed once
        again.append(LabKind.BUDGET_FROZEN, "coordinator", PERSISTED["FrozenFile"]())

    def test_kinds_that_may_repeat_do_repeat(self):
        led = self.make()
        for i in range(3):
            led.append(LabKind.OPERATOR_CHANGE, "coordinator", op(str(i)))
            led.append(LabKind.PACK_COMMITTED, "coordinator", PERSISTED["PackCommitted"]())
        self.assertEqual(len(led.events()), 6)

    def test_freeze_once_holds_under_concurrent_appends(self):
        # a clock that sleeps widens the window between the freeze-once check and the write (the reused Ledger reads the clock inside append)
        slow = lambda: (time.sleep(0.002), self.clock())[1]  # noqa: E731
        led = LabLedger.create(self.dir, clock=slow, fsync=False)
        self.addCleanup(led.close)
        results = []

        def go():
            try:
                led.append(LabKind.SEARCH_CLOSED, "coordinator", PERSISTED["SearchClosed"]())
                results.append("ok")
            except LedgerError:
                results.append("refused")

        threads = [threading.Thread(target=go) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count("ok"), 1)
        self.assertEqual(len(led.events()), 1)


class ConcurrencyTests(LedgerTestCase):
    def test_many_threads_append_and_the_chain_still_verifies(self):
        led = self.make()
        n_threads, per_thread = 12, 25

        def go(k: int) -> None:
            for i in range(per_thread):
                led.append(LabKind.OPERATOR_CHANGE, f"trial:t-{k:04d}", op(f"{k}-{i}"))

        threads = [threading.Thread(target=go, args=(k,)) for k in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        events = led.events()
        self.assertEqual(len(events), n_threads * per_thread)
        self.assertEqual([e.seq for e in events], list(range(1, n_threads * per_thread + 1)))
        self.assertEqual(led.verify(), (True, None))
        # every thread's appends stayed in order
        for k in range(n_threads):
            mine = [e.data["detail"] for e in events if e.actor == f"trial:t-{k:04d}"]
            self.assertEqual(mine, [f"{k}-{i}" for i in range(per_thread)])
        # and the file on disk agrees with memory
        ro = LabLedger.read_only(self.dir)
        self.addCleanup(ro.close)
        self.assertEqual(len(ro.events()), n_threads * per_thread)
        self.assertEqual(ro.verify(), (True, None))


class OpenCloseTests(LedgerTestCase):
    def test_create_refuses_an_existing_ledger(self):
        led = self.make()
        led.append(LabKind.OPERATOR_CHANGE, "coordinator", op())
        led.close()
        with self.assertRaises(LedgerError):
            LabLedger.create(self.dir, clock=self.clock, fsync=False)

    def test_second_writer_is_refused_while_the_first_is_open(self):
        self.make()
        with self.assertRaises(LedgerError):
            LabLedger.open(self.dir, clock=self.clock, fsync=False)

    def test_reopen_continues_the_chain(self):
        led = self.make()
        led.append(LabKind.OPERATOR_CHANGE, "coordinator", op("one"))
        before = led.anchor()
        led.close()
        again = LabLedger.open(self.dir, clock=self.clock, fsync=False)
        self.addCleanup(again.close)
        self.assertEqual(again.anchor(), before)
        again.append(LabKind.OPERATOR_CHANGE, "coordinator", op("two"))
        self.assertEqual(again.verify(), (True, None))
        self.assertEqual([e.data["detail"] for e in again.events()], ["one", "two"])

    def test_torn_final_line_is_recovered_on_open(self):
        led = self.make()
        led.append(LabKind.OPERATOR_CHANGE, "coordinator", op("kept"))
        led.close()
        with open(self.dir / "ledger.jsonl", "ab") as fh:
            fh.write(b'{"seq": 2, "ts": 1.0, "run_id"')
        again = LabLedger.open(self.dir, clock=self.clock, fsync=False)
        self.addCleanup(again.close)
        self.assertEqual(len(again.events()), 1)
        again.append(LabKind.OPERATOR_CHANGE, "coordinator", op("after"))
        self.assertEqual(again.verify(), (True, None))

    def test_read_only_cannot_append_but_follows_the_writer(self):
        led = self.make()
        led.append(LabKind.OPERATOR_CHANGE, "coordinator", op("one"))
        ro = LabLedger.read_only(self.dir)
        self.addCleanup(ro.close)
        self.assertFalse(ro.writable)
        with self.assertRaises(LedgerError):
            ro.append(LabKind.OPERATOR_CHANGE, "coordinator", op("nope"))
        led.append(LabKind.OPERATOR_CHANGE, "coordinator", op("two"))
        self.assertEqual(len(ro.events()), 1)
        self.assertEqual(ro.refresh(), 1)
        self.assertEqual(len(ro.events()), 2)

    def test_closed_and_sealed_ledgers_refuse_appends(self):
        led = self.make()
        led.append(LabKind.OPERATOR_CHANGE, "coordinator", op())
        led.seal()
        self.assertEqual(oct((self.dir / "ledger.jsonl").stat().st_mode & 0o777), oct(0o444))
        with self.assertRaises(LedgerError):
            led.append(LabKind.OPERATOR_CHANGE, "coordinator", op("late"))
        led.close()  # idempotent
        self.assertEqual(led.verify(), (True, None))

    def test_anchor_is_the_head(self):
        led = self.make()
        self.assertEqual(led.anchor(), c.LedgerAnchor(0, "0" * 16))
        ev = led.append(LabKind.OPERATOR_CHANGE, "coordinator", op())
        self.assertEqual(led.anchor(), c.LedgerAnchor(1, ev.hash[:16]))


class TamperTests(LedgerTestCase):
    def test_editing_an_event_breaks_verify(self):
        led = self.make()
        for i in range(3):
            led.append(LabKind.OPERATOR_CHANGE, "coordinator", op(f"change {i}"))
        led.close()
        path = self.dir / "ledger.jsonl"
        lines = path.read_text().splitlines()
        row = json.loads(lines[1])
        row["data"]["detail"] = "rewritten"
        lines[1] = json.dumps(row, sort_keys=True, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n")
        ro = LabLedger.read_only(self.dir)
        self.addCleanup(ro.close)
        self.assertEqual(ro.verify(), (False, 2))


class SecretTests(LedgerTestCase):
    def test_registered_secret_never_reaches_the_file(self):
        secret = "sk-test-0123456789abcdef"
        os.environ["LAB_TEST_SECRET"] = secret
        self.addCleanup(os.environ.pop, "LAB_TEST_SECRET", None)
        register_secrets("LAB_TEST_SECRET")
        led = self.make()
        led.append(LabKind.OPERATOR_CHANGE, "coordinator", op(f"the key is {secret} ok"))
        led.close()
        self.assertNotIn(secret, (self.dir / "ledger.jsonl").read_text())
        self.assertIn("[REDACTED]", (self.dir / "ledger.jsonl").read_text())

    def test_unregistered_value_is_written_as_is(self):
        # documents the limit: only registered values are scrubbed (register BEFORE the first append)
        led = self.make()
        led.append(LabKind.OPERATOR_CHANGE, "coordinator", op("plain-value-not-registered"))
        led.close()
        self.assertIn("plain-value-not-registered", (self.dir / "ledger.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
