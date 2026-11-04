import json
import tempfile
import threading
import unittest
from pathlib import Path

from nightshift import secrets
from nightshift.ledger import Ledger, LedgerLocked, cap_strings, verify_events
from nightshift.types import ExpStatus, Kind, LedgerCorrupt, Outcome, Verdict
from tests.support.synthetic import FakeClock, RunBuilder


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_chain_roundtrip_and_tamper_detection(self):
        led = Ledger.create(self.dir, clock=FakeClock(), fsync=False)
        led.append(Kind.RUN_STARTED, "engine", {"task": "t"})
        led.append(Kind.SUP_TURN, "supervisor", {"turn": 1})
        self.assertEqual(led.verify_chain(), (True, None))
        led.close()
        lines = (self.dir / "ledger.jsonl").read_text().splitlines()
        obj = json.loads(lines[0])
        obj["data"]["task"] = "tampered"
        lines[0] = json.dumps(obj, sort_keys=True)
        (self.dir / "ledger.jsonl").write_text("\n".join(lines) + "\n")
        re = Ledger.read_only(self.dir)
        self.assertEqual(re.verify_chain(), (False, 1))

    def test_single_writer_lock_and_reopen(self):
        led = Ledger.create(self.dir, fsync=False)
        led.append(Kind.RUN_STARTED, "engine", {})
        with self.assertRaises(LedgerLocked):
            Ledger.open(self.dir)
        led.close()
        again = Ledger.open(self.dir, fsync=False)
        self.assertEqual(again.append(Kind.SUP_TURN, "s", {}).seq, 2)
        self.assertEqual(again.run_id, led.run_id)

    def test_torn_tail_is_dropped(self):
        led = Ledger.create(self.dir, fsync=False)
        led.append(Kind.RUN_STARTED, "engine", {})
        led.close()
        with open(self.dir / "ledger.jsonl", "ab") as fh:
            fh.write(b'{"seq": 2, "partial')
        again = Ledger.open(self.dir, fsync=False)
        self.assertTrue(again.recovered_torn_tail)
        self.assertEqual(len(again.events()), 1)
        again.append(Kind.SUP_TURN, "s", {})
        self.assertEqual(again.verify_chain(), (True, None))

    def test_concurrent_appends_keep_chain_valid(self):
        led = Ledger.create(self.dir, fsync=False)
        def work(i):
            for j in range(50):
                led.append(Kind.SUP_TURN, f"t{i}", {"j": j})
        ts = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(led.events()), 300)
        self.assertEqual(led.verify_chain(), (True, None))
        self.assertEqual(len({e.seq for e in led.events()}), 300)

    def test_exp_ids_unique_across_threads_and_reopen(self):
        led = Ledger.create(self.dir, fsync=False)
        ids = []
        def work():
            for _ in range(20):
                ids.append(led.next_exp_id())
        ts = [threading.Thread(target=work) for _ in range(5)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(set(ids)), 100)
        self.assertEqual(sorted(ids)[0], "e000")

    def test_read_only_follows_writer(self):
        led = Ledger.create(self.dir, fsync=False)
        led.append(Kind.RUN_STARTED, "engine", {"task": "t"})
        ro = Ledger.read_only(self.dir)
        self.assertEqual(len(ro.events()), 1)
        led.append(Kind.SUP_TURN, "s", {})
        self.assertEqual(ro.refresh(), 1)
        self.assertEqual(ro.view().supervisor_turns, 1)
        with self.assertRaises(PermissionError):
            ro.append(Kind.SUP_TURN, "s", {})


class LedgerHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(secrets.clear)

    def test_corrupt_middle_line_raises_and_never_truncates(self):
        led = Ledger.create(self.dir, fsync=False)
        for i in range(5):
            led.append(Kind.SUP_TURN, "s", {"i": i})
        led.close()
        path = self.dir / "ledger.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        lines[2] = b"this is not json\n"
        path.write_bytes(b"".join(lines))
        size_before = path.stat().st_size
        with self.assertRaises(LedgerCorrupt) as cm:
            Ledger.open(self.dir, fsync=False)
        self.assertEqual(cm.exception.first_bad_line, 3)
        self.assertEqual(path.stat().st_size, size_before, "a corrupt ledger must never be truncated")
        with self.assertRaises(LedgerCorrupt):
            Ledger.read_only(self.dir)

    def test_valid_json_of_wrong_shape_is_corrupt_not_a_crash(self):
        led = Ledger.create(self.dir, fsync=False)
        led.append(Kind.SUP_TURN, "s", {})
        led.append(Kind.SUP_TURN, "s", {})
        led.close()
        path = self.dir / "ledger.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        lines[0] = b"[1, 2, 3]\n"
        path.write_bytes(b"".join(lines))
        with self.assertRaises(LedgerCorrupt):
            Ledger.open(self.dir, fsync=False)

    def test_torn_tail_makes_a_backup_before_truncating(self):
        led = Ledger.create(self.dir, fsync=False)
        led.append(Kind.SUP_TURN, "s", {})
        led.close()
        with open(self.dir / "ledger.jsonl", "ab") as fh:
            fh.write(b'{"seq": 2, "par')
        Ledger.open(self.dir, fsync=False).close()
        self.assertTrue(list(self.dir.glob("ledger.jsonl.torn-*")))

    def test_secrets_are_scrubbed_and_long_strings_capped(self):
        secrets.register("sk-live-0123456789abcdef")
        led = Ledger.create(self.dir, fsync=False)
        ev = led.append(Kind.SUP_TURN, "s", {"text": "key is sk-live-0123456789abcdef ok", "hdr": "Bearer abcdefghijkl123456",
                                              "big": "x" * 20000, "nested": {"k": ["sk-live-0123456789abcdef"]}})
        raw = (self.dir / "ledger.jsonl").read_text()
        self.assertNotIn("sk-live-0123456789abcdef", raw)
        self.assertNotIn("abcdefghijkl123456", raw)
        self.assertIn("[REDACTED]", ev.data["text"])
        self.assertIn("truncated", ev.data["big"])
        self.assertLess(len(ev.data["big"]), 9000)

    def test_lone_surrogate_in_model_text_does_not_crash_append(self):
        led = Ledger.create(self.dir, fsync=False)
        ev = led.append(Kind.CLAIM, "implementer:A", {"claim_id": "c", "text": "bad \ud800 text", "nested": ["\udfff"]})
        self.assertIn("bad", ev.data["text"])
        ev.data["text"].encode("utf-8")
        led.close()
        again = Ledger.open(self.dir, fsync=False)       # and the file round-trips
        self.assertEqual(again.verify_chain(), (True, None))

    def test_cap_strings_is_utf8_safe(self):
        out = cap_strings("é" * 10000, limit=100)
        out.encode("utf-8")

    def test_head_anchor_and_seal(self):
        led = Ledger.create(self.dir, fsync=False)
        self.assertEqual(led.head_anchor()[0], 0)
        ev = led.append(Kind.RUN_STARTED, "engine", {})
        self.assertEqual(led.head_anchor(), (1, ev.hash[:16]))
        led.seal()
        self.assertEqual((self.dir / "ledger.jsonl").stat().st_mode & 0o222, 0)

    def test_read_only_picks_up_run_id_after_writer_starts(self):
        ro = Ledger.read_only(self.dir)             # opened before the run exists
        led = Ledger.create(self.dir, fsync=False)
        led.append(Kind.RUN_STARTED, "engine", {"task": "t"})
        self.assertEqual(ro.refresh(), 1)
        self.assertEqual(ro.run_id, led.run_id)
        self.assertEqual(ro.view().run["run_id"], led.run_id)

    def test_run_id_mixing_is_detected(self):
        led = Ledger.create(self.dir, fsync=False)
        led.append(Kind.SUP_TURN, "s", {})
        led.append(Kind.SUP_TURN, "s", {})
        evs = led.events()
        import dataclasses
        forged = dataclasses.replace(evs[1], run_id="other")
        self.assertFalse(verify_events([evs[0], forged])[0])


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.led = Ledger.create(self.tmp.name, clock=FakeClock(), fsync=False)
        self.rb = RunBuilder(self.led)
        self.rb.start()

    def test_baseline_experiment_promotion_flow(self):
        b = self.rb.baseline(1.0, train=0.9)
        e1 = self.rb.experiment("A", "quadratic", score=0.8, train=0.7, delta=0.2, vs=b)
        self.rb.replicate(e1, 0.82)
        self.rb.promote(e1)
        v = self.led.view()
        self.assertEqual(v.baseline_id, b)
        self.assertEqual(v.incumbent_id, e1)
        self.assertEqual(v.exp(e1).status, ExpStatus.RECORDED)
        self.assertEqual(v.exp(e1).outcome, Outcome.IMPROVED)
        self.assertTrue(v.exp(e1).replicated)
        self.assertAlmostEqual(v.exp(e1).primary_grade.score, 0.8)
        self.assertAlmostEqual(v.exp(e1).replication_grade.score, 0.82)

    def test_findings_and_acks(self):
        b = self.rb.baseline()
        fid = self.rb.finding("I01", "c001", severity="block", exp_id=b)
        self.assertEqual([f.finding_id for f in self.led.view().open_findings("block")], [fid])
        self.rb.ack(fid, "addressed")
        self.assertEqual(self.led.view().open_findings(), [])
        self.assertEqual(self.led.view().findings[fid].status, "addressed")
        self.assertEqual(self.led.view().findings[fid].exp_id, b)

    def test_ack_written_by_the_supervisor_does_not_close_a_finding(self):
        self.rb.baseline()
        fid = self.rb.finding("I03", "e001", severity="block")
        self.rb.ack(fid, "dismissed", "i would like this to go away please", actor="supervisor")
        self.assertEqual(self.led.view().findings[fid].status, "open")
        self.assertEqual(len(self.led.view().acks), 1)

    def test_failed_replication_keeps_dev_numbers(self):
        b = self.rb.baseline(1.0, train=0.9)
        e1 = self.rb.experiment("A", "x", score=0.8, train=0.7, delta=0.2, vs=b, noise_margin=0.05)
        v = self.led.view()
        self.assertTrue(v.exp(e1).awaiting_replication)
        self.rb.replicate(e1, 0.99, success=False)
        v = self.led.view()
        e = v.exp(e1)
        self.assertAlmostEqual(e.delta, 0.2)            # dev delta untouched by the sealed result
        self.assertAlmostEqual(e.noise_margin, 0.05)
        self.assertEqual(e.outcome, Outcome.NO_GAIN)    # but the final outcome follows the sealed result
        self.assertEqual(e.dev_outcome, Outcome.IMPROVED)
        self.assertIsNotNone(e.sealed_delta)
        self.assertFalse(e.replicated)
        self.assertFalse(e.awaiting_replication)
        self.assertEqual(e.status, ExpStatus.RECORDED)

    def test_successful_replication_and_promotion(self):
        b = self.rb.baseline(1.0)
        self.rb.replication_baseline(b, 1.02)
        e1 = self.rb.experiment("A", "x", score=0.8, delta=0.2, vs=b)
        self.rb.replicate(e1, 0.82)
        self.rb.promote(e1, prev=b)
        v = self.led.view()
        self.assertEqual(v.incumbent_id, e1)
        self.assertTrue(v.exp(e1).replicated)
        self.assertEqual(v.exp(e1).replication_grade.split, "sealed")
        self.assertEqual(v.exp(b).grade_for("replication_baseline").grade.split, "sealed")

    def test_malformed_payloads_are_isolated_not_fatal(self):
        b = self.rb.baseline()
        self.led.append(Kind.CLAIM, "supervisor", {"claim_id": "x", "exp_id": ["not", "hashable"], "claim_type": "improved"})
        self.led.append(Kind.LESSON, "reflector", "this payload is a string, not a dict")
        self.led.append(Kind.EXP_RUN_STARTED, "engine", {"purpose": "primary"})     # missing exp_id
        e = self.rb.experiment("A", "still works afterwards", score=0.9, delta=0.1, vs=b)
        v = self.led.view()
        self.assertEqual(v.exp(e).status, ExpStatus.RECORDED)
        self.assertGreaterEqual(len(v.malformed), 2)
        self.assertTrue(all("seq" in m and "error" in m for m in v.malformed))

    def test_supervisor_turn_details_are_kept(self):
        self.led.append(Kind.SUP_TURN, "supervisor", {"turn": 1, "model": "m", "action_summary": "a", "say": "hello",
                                                      "tool_calls": [{"name": "get_state", "args_summary": ""}],
                                                      "reasoning_effort": "high", "latency_ms": 12.5, "source": "supervisor"})
        t = self.led.view().turns[0]
        self.assertEqual(t["tool_calls"][0]["name"], "get_state")
        self.assertEqual(t["reasoning_effort"], "high")

    def test_budget_and_turn_counters(self):
        self.rb.baseline()
        self.rb.charge_llm(0.25, "call-1")
        self.rb.charge_llm(0.25, "call-2")
        self.rb.turn(3)
        v = self.led.view()
        self.assertAlmostEqual(v.budget_used["usd"], 0.5)
        self.assertEqual(v.charged_call_ids, {"call-1", "call-2"})
        self.assertEqual(v.turns_since_experiment, 3)
        self.rb.experiment("A", "x")
        self.assertEqual(self.led.view().turns_since_experiment, 0)

    def test_protected_violation_is_terminal(self):
        self.rb.baseline()
        e = self.rb.experiment("A", "cheat", protected=["run.py"], recorded=False)
        self.assertEqual(self.led.view().exp(e).status, ExpStatus.VIOLATION)

    def test_verify_events_detects_reorder(self):
        self.rb.baseline()
        evs = self.led.events()
        evs[2], evs[3] = evs[3], evs[2]
        self.assertFalse(verify_events(evs)[0])


if __name__ == "__main__":
    unittest.main()
