import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from nightshift.lab.contracts import (
    CreateAmbiguous,
    CreateRejected,
    GpuError,
    GpuProvider,
    OwnershipError,
    ProviderUnavailable,
    Shape,
    owner_tag_of_key,
)
from nightshift.lab.gpu_remote import SshTarget
from nightshift.lab.gpu_rentals import RentalManager, RentalRefused, alive_watchdogs, billed_minutes, ssh_keygen_factory
from tests.support.lab_fake_gpu import FakeClock, FakeProvider, FakeWatchdogLauncher, fake_key_factory

SHAPE = Shape("a100", 1, 12, 300)
RATE = 1.31


class Base(unittest.TestCase):
    hard_stop = 20.0

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.clock = FakeClock()
        self.provider = FakeProvider(self.clock, rate=RATE)
        self.launcher = FakeWatchdogLauncher()
        self.addCleanup(self.launcher.close)
        self.mgr = self.manager()
        if self.hard_stop:
            self.mgr.init(self.hard_stop, "test-machine", "unit test")

    def manager(self, **kw) -> RentalManager:
        args = dict(
            clock=self.clock, launcher=self.launcher, key_factory=fake_key_factory, sleep=lambda s: self.clock.advance(s), stopper=self.launcher.stop
        )
        args.update(kw)
        return RentalManager(self.tmp / "gpu" / "rentals.json", self.provider, **args)

    def create(self, session="s1", hours=1.0, trial="t-0001", item="train", mgr=None):
        return (mgr or self.mgr).create(session, SHAPE, hours, trial, item)


class StateMachine(Base):
    def test_provider_protocol(self):
        self.assertIsInstance(self.provider, GpuProvider)

    def test_create_goes_live_with_a_watchdog_and_a_tag(self):
        r = self.create()
        self.assertEqual(r.state, "live")
        self.assertEqual(r.handle, "fake0001x")
        row = self.mgr.read()["rentals"][0]
        pub = (self.tmp / "gpu" / "sessions" / "s1" / "id_ed25519.pub").read_text().strip()
        self.assertEqual(row["owner_tag"], owner_tag_of_key(pub))
        self.assertEqual(self.provider.instances[r.handle].owner_tag, row["owner_tag"])
        self.assertEqual(row["cap_epoch"], self.clock() + 3600)
        self.assertEqual(self.launcher.calls, [("s1", int(self.clock() + 3600))])
        self.assertEqual(self.mgr.read()["rentals"][0]["usd_per_hour"], RATE)

    def test_row_and_watchdog_exist_before_the_provider_is_called(self):
        seen = {}

        def probe():
            doc = self.mgr.read()
            seen["state"] = doc["rentals"][0]["state"]
            seen["watchdogs"] = len(self.launcher.calls)

        self.provider.on_create = probe
        self.create()
        self.assertEqual(seen, {"state": "creating", "watchdogs": 1})

    def test_unarmed_watchdog_refuses_the_create_before_anything_is_rented(self):
        self.launcher.fail = True
        with self.assertRaisesRegex(GpuError, "did not arm"):
            self.create()
        self.assertEqual(self.provider.calls_of("create"), [])
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "not_created")
        self.launcher.fail = False
        self.create("s2")  # a refused create does not block the next one

    def test_a_rejected_create_settles_as_not_created_and_bills_nothing(self):
        self.provider.create_mode = "reject"
        with self.assertRaises(CreateRejected):
            self.create()
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "not_created")
        self.assertEqual(self.mgr.usage_for("t-0001").minutes, 0)
        self.provider.create_mode = "ok"
        self.create("s2")

    def test_listing_failure_before_the_create_creates_nothing(self):
        self.provider.fail_list_n = 1
        with self.assertRaises(ProviderUnavailable):
            self.create()
        self.assertEqual(self.provider.calls_of("create"), [])
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "not_created")

    def test_sold_out_shape_is_refused_at_quote_time(self):
        self.provider.sold_out.add("a100")
        with self.assertRaisesRegex(RentalRefused, "not available"):
            self.create()
        self.assertEqual(self.provider.calls_of("create"), [])
        self.assertEqual(self.mgr.read()["rentals"], [])

    def test_ambiguous_create_that_made_an_instance_is_recovered_by_tag(self):
        self.provider.create_mode = "ambiguous_created"
        r = self.create()
        self.assertEqual(r.state, "live")
        self.assertEqual(r.handle, "fake0001x")

    def test_ambiguous_create_with_a_foreign_same_shape_instance_never_adopts_it(self):
        foreign = self.provider.add_foreign(SHAPE)
        self.provider.create_mode = "ambiguous_not_created"
        with self.assertRaisesRegex(CreateAmbiguous, "Do NOT create again"):
            self.create()
        row = self.mgr.read()["rentals"][0]
        self.assertEqual(row["state"], "uuid_unknown")
        self.assertIsNone(row["handle"])
        self.assertNotEqual(row["handle"], foreign)

    def test_second_create_is_blocked_while_unsettled_and_recover_settles_it(self):
        self.provider.create_mode = "ambiguous_not_created"
        with self.assertRaises(CreateAmbiguous):
            self.create()
        self.provider.create_mode = "ok"
        with self.assertRaisesRegex(RentalRefused, "unsettled"):
            self.create("s2")
        self.assertEqual(self.provider.calls_of("create").__len__(), 1)
        with self.assertRaises(CreateAmbiguous):
            self.mgr.recover("s1")  # nothing carries the tag: a human must read the console
        r = self.mgr.recover("s1", confirm_none=True)
        self.assertEqual(r.state, "not_created")
        self.create("s2")

    def test_receipt_without_a_handle_is_recovered_by_the_tag(self):
        self.provider.create_mode = "no_handle"
        r = self.create()
        self.assertEqual(r.state, "live")
        self.assertTrue(r.handle)

    def test_concurrent_creates_cannot_both_run(self):
        gate, entered = threading.Event(), threading.Event()
        self.provider.on_create = lambda: (entered.set(), gate.wait(5))
        out = {}

        def first():
            out["r"] = self.create("s1")

        t = threading.Thread(target=first)
        t.start()
        self.assertTrue(entered.wait(5))
        with self.assertRaisesRegex(RentalRefused, "unsettled"):
            self.create("s2")
        gate.set()
        t.join(5)
        self.assertEqual(out["r"].state, "live")
        self.assertEqual(len(self.provider.calls_of("create")), 1)

    def test_a_live_rental_does_not_block_another_create(self):
        self.create("s1")
        self.create("s2")
        self.assertEqual(len(self.mgr.read()["rentals"]), 2)

    def test_session_names_must_be_unique_and_safe(self):
        self.create("s1")
        with self.assertRaises(RentalRefused):
            self.create("s1")
        for bad in ("../x", "a b", "", "a/b"):
            with self.assertRaises(GpuError):
                self.create(bad)

    def test_cap_hours_bounds(self):
        for bad in (0, -1, 13, True, "2"):
            with self.assertRaises(RentalRefused):
                self.create("sx", hours=bad)

    def test_machine_binding(self):
        m = self.manager(machine="other-box")
        with self.assertRaisesRegex(RentalRefused, "belongs to machine"):
            self.create(mgr=m)

    def test_real_ssh_keygen_makes_a_private_key_and_a_taggable_public_key(self):
        if shutil.which("ssh-keygen") is None:
            self.skipTest("ssh-keygen not installed")
        d = self.tmp / "keys" / "k1"
        pub = ssh_keygen_factory(d)
        self.assertTrue(pub.startswith("ssh-ed25519 "))
        self.assertEqual(oct((d / "id_ed25519").stat().st_mode & 0o777), "0o600")
        self.assertEqual(len(owner_tag_of_key(pub)), 16)


class MoneyRule(Base):
    def test_no_default_hard_stop_exists(self):
        self.hard_stop = 0
        mgr = RentalManager(self.tmp / "other" / "r.json", self.provider, clock=self.clock, launcher=self.launcher, key_factory=fake_key_factory)
        with self.assertRaisesRegex(RentalRefused, "no hard stop"):
            mgr.create("s1", SHAPE, 1, "t", "train")
        self.assertEqual(self.provider.calls_of("create"), [])

    def test_init_validates_and_refuses_to_overwrite(self):
        with self.assertRaises(GpuError):
            self.mgr.init(10, "m", "again")
        m2 = RentalManager(self.tmp / "o" / "r.json", self.provider, clock=self.clock)
        for args in ((0, "m", "w"), (-5, "m", "w"), (float("nan"), "m", "w"), (True, "m", "w"), (10, "", "w"), (10, "m", " ")):
            with self.assertRaises(GpuError):
                m2.init(*args)

    def test_cap_that_would_cross_the_stop_is_refused_and_creates_nothing(self):
        self.mgr.set_hard_stop(5.0, "lower for the test")
        with self.assertRaises(RentalRefused) as cm:
            self.create(hours=4)  # 4 h = 5.24 > 5.0
        self.assertEqual(self.provider.calls_of("create"), [])
        self.assertAlmostEqual(cm.exception.check["new_cap_usd"], 5.24, places=2)

    def test_exactly_at_the_stop_is_allowed(self):
        self.mgr.set_hard_stop(round(60 * RATE / 60, 6), "exactly one hour")
        self.create(hours=1)

    def test_a_fractional_second_start_never_adds_a_billable_minute_beyond_the_priced_cap(self):
        from nightshift.lab.gpu_rentals import cap_minutes

        self.clock.now = 1000.25  # rounding the armed deadline UP would make the cap 61 billed minutes, priced as 60
        self.mgr.set_hard_stop(round(60 * RATE / 60, 6), "exactly one hour")
        self.create(hours=1)
        row = self.mgr.read()["rentals"][0]
        self.assertEqual(cap_minutes(row), 60)
        self.assertLessEqual(row["cap_epoch"], 1000.25 + 3600)
        self.assertEqual(row["cap_epoch"], int(row["cap_epoch"]))
        self.assertEqual(self.launcher.calls, [("s1", int(row["cap_epoch"]))])

    def test_other_live_rentals_count_up_to_their_caps(self):
        self.create("s1", hours=6)  # 7.86 committed
        self.create("s2", hours=6)  # 15.72 committed in total
        with self.assertRaisesRegex(RentalRefused, "hard-stop rule"):
            self.create("s3", hours=4)  # 5.24 more would be 20.96 > 20
        self.create("s3", hours=3)  # 19.65 <= 20

    def test_billed_minutes_round_up_and_free_up_room_after_delete(self):
        self.create("s1", hours=10)  # 13.10 committed
        self.clock.advance(61)  # 2 billed minutes
        r = self.mgr.rental("s1")
        self.mgr.delete(r, "done")
        usage = self.mgr.usage_for("t-0001")
        self.assertEqual(usage.minutes, 2.0)
        self.assertAlmostEqual(usage.usd, 2 * RATE / 60, places=6)
        self.assertEqual(usage.rental_ids, ("s1",))
        self.create("s2", hours=12)  # the 10 h commitment is gone: only 2 billed minutes remain

    def test_reserve_counts(self):
        with self.mgr._locked() as doc:
            doc["reserve_usd"] = 19.0
        with self.assertRaises(RentalRefused):
            self.create(hours=1)

    def test_cost_report(self):
        self.create("s1", hours=2)
        self.clock.advance(300)
        c = self.mgr.cost()
        self.assertEqual(c["hard_stop_usd"], 20.0)
        self.assertAlmostEqual(c["spent_usd"], 5 * RATE / 60, places=6)
        self.assertAlmostEqual(c["committed_usd"], (120 - 5) * RATE / 60, places=4)
        self.assertTrue(c["rentals"][0]["watchdog_alive"])
        self.assertLess(c["remaining_usd"], 20.0)

    def test_billed_minutes_helper_is_ceil(self):
        row = {"create_attempt_epoch": 0.0, "deleted_epoch": 60.0}
        self.assertEqual(billed_minutes(row, 1e9), 1)
        row["deleted_epoch"] = 60.01
        self.assertEqual(billed_minutes(row, 1e9), 2)
        row["deleted_epoch"] = None
        self.assertEqual(billed_minutes(row, 119.0), 2)


class Delete(Base):
    def test_delete_confirms_absence_then_records(self):
        r = self.create()
        self.clock.advance(125)
        self.mgr.delete(r, "finished")
        row = self.mgr.read()["rentals"][0]
        self.assertEqual(row["state"], "deleted")
        self.assertEqual(row["delete_reason"], "finished")
        self.assertEqual(row["deleted_epoch"], self.clock())
        self.assertNotIn(r.handle, self.provider.handles())
        sd = self.tmp / "gpu" / "sessions" / "s1"
        self.assertTrue((sd / "deleted-utc.txt").exists())
        self.assertFalse((sd / "id_ed25519").exists(), "the per-rental private key is deleted at teardown")

    def test_delete_is_idempotent(self):
        r = self.create()
        self.mgr.delete(r, "a")
        n = len(self.provider.calls_of("delete"))
        self.mgr.delete(r, "b")
        self.assertEqual(len(self.provider.calls_of("delete")), n)
        self.assertEqual(self.mgr.read()["rentals"][0]["delete_reason"], "a")

    def test_delete_survives_a_failing_provider_and_can_be_retried(self):
        r = self.create()
        self.provider.fail_delete_n = 1
        with self.assertRaises(ProviderUnavailable):
            self.mgr.delete(r, "x")
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "live")
        self.mgr.delete(r, "x")
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "deleted")

    def test_listing_failure_sends_no_delete_and_records_nothing(self):
        r = self.create()
        self.provider.fail_list_n = 1
        with self.assertRaises(ProviderUnavailable):
            self.mgr.delete(r, "x")
        self.assertEqual(self.provider.calls_of("delete"), [])
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "live")

    def test_an_unknown_exists_answer_is_never_recorded_as_deleted(self):
        r = self.create()
        real_exists = self.provider.exists
        self.provider.exists = lambda h: None  # the provider cannot tell
        with self.assertRaisesRegex(GpuError, "not confirmed"):
            self.mgr.delete(r, "x", confirm_tries=3)
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "live")
        self.provider.exists = real_exists
        self.mgr.delete(r, "x")

    def test_still_present_after_delete_is_not_recorded_as_deleted(self):
        r = self.create()
        self.provider.delete = lambda handle, tag: None  # the provider says OK but the instance remains
        with self.assertRaisesRegex(GpuError, "not confirmed"):
            self.mgr.delete(r, "x", confirm_tries=3)
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "live")

    def test_delete_of_a_foreign_handle_raises_and_sends_nothing(self):
        r = self.create()
        foreign = self.provider.add_foreign(SHAPE)
        with self.mgr._locked() as doc:  # a corrupted ledger that points at someone else's instance
            doc["rentals"][0]["handle"] = foreign
        with self.assertRaises(OwnershipError):
            self.mgr.delete(r, "x")
        self.assertIn(foreign, self.provider.handles())
        self.assertEqual(self.provider.calls_of("delete"), [])

    def test_delete_only_touches_sessions_in_the_ledger(self):
        with self.assertRaisesRegex(GpuError, "not in the rental ledger"):
            self.mgr.delete("never-created", "x")

    def test_other_instances_vanishing_during_the_delete_are_logged(self):
        r = self.create()
        other = self.provider.add_foreign(SHAPE)
        real_delete = self.provider.delete

        def racy(handle, tag):
            real_delete(handle, tag)
            del self.provider.instances[other]  # another lane deleted its own instance at the same time

        self.provider.delete = racy
        self.mgr.delete(r, "x")
        self.assertEqual(self.mgr.read()["rentals"][0]["anomalies"][0]["other_instances_vanished_during_delete"], [other])

    def test_delete_of_an_unsettled_rental_finds_it_by_tag(self):
        self.provider.create_mode = "ambiguous_created"
        self.provider.fail_list_n = 0
        orig = self.provider.find_created
        self.provider.find_created = lambda *a, **k: None  # recovery during create fails
        with self.assertRaises(CreateAmbiguous):
            self.create()
        self.provider.find_created = orig
        self.mgr.delete("s1", "cleanup")  # the manager searches by tag, finds the instance and deletes it
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "deleted")
        self.assertEqual(self.provider.handles(), set())


class WaitAndEndpoint(Base):
    def test_wait_ready_polls_until_running_with_an_address(self):
        self.provider.pending_lists = 3
        r = self.create()
        inst = self.mgr.wait_ready(r, timeout_s=600)
        self.assertEqual((inst.status, inst.host), ("RUNNING", "127.0.0.1"))

    def test_wait_ready_timeout_does_not_delete_and_says_it_is_billing(self):
        self.provider.pending_lists = 10**6
        r = self.create()
        with self.assertRaisesRegex(GpuError, "STILL BILLING"):
            self.mgr.wait_ready(r, timeout_s=30)
        self.assertIn(r.handle, self.provider.handles())
        self.assertEqual(self.provider.calls_of("delete"), [])

    def test_endpoint_reads_the_current_address_and_writes_a_private_ssh_config(self):
        r = self.create()
        t = self.mgr.endpoint(r)
        self.assertIsInstance(t, SshTarget)
        self.assertEqual((t.host, t.port, t.user), ("127.0.0.1", 2200, "ubuntu"))
        self.assertTrue(t.key_path.endswith("sessions/s1/id_ed25519"))
        cfg = self.tmp / "gpu" / "sessions" / "s1" / "ssh_config"
        self.assertEqual(oct(cfg.stat().st_mode & 0o777), "0o600")
        self.provider.instances[r.handle] = type(self.provider.instances[r.handle])(
            r.handle, "RUNNING", SHAPE, self.clock(), "x", "10.9.9.9", 4444
        )
        self.assertEqual(self.mgr.endpoint(r).host, "10.9.9.9")  # dynamic addresses

    def test_endpoint_refuses_a_deleted_rental(self):
        r = self.create()
        self.mgr.delete(r, "x")
        with self.assertRaises(GpuError):
            self.mgr.endpoint(r)


class Rearm(Base):
    def test_rearm_moves_the_cap_arms_the_new_watchdog_first_and_stops_the_old(self):
        r = self.create(hours=1)
        old_pids = alive_watchdogs(r.session_dir)
        self.assertEqual(len(old_pids), 1)
        new = self.mgr.rearm(r, hours_from_now=2)
        self.assertEqual(new.cap_epoch, self.clock() + 7200)
        self.assertEqual(len(self.launcher.calls), 2)
        now_alive = alive_watchdogs(r.session_dir)
        self.assertEqual(len(now_alive), 1)
        self.assertNotIn(old_pids[0], now_alive)
        self.assertEqual(self.mgr.read()["rentals"][0]["rearms"][0]["to_cap_epoch"], new.cap_epoch)

    def test_rearm_is_checked_against_the_money_rule(self):
        r = self.create(hours=1)
        self.mgr.set_hard_stop(3.0, "tight")
        with self.assertRaisesRegex(RentalRefused, "rearm"):
            self.mgr.rearm(r, hours_from_now=3)  # 3.93 > 3.0
        self.assertEqual(self.mgr.rental("s1").cap_epoch, r.cap_epoch)
        self.mgr.rearm(r, hours_from_now=2)  # 2.62 fits

    def test_shortening_is_always_allowed_even_over_the_stop(self):
        r = self.create(hours=2)
        self.mgr.set_hard_stop(0.5, "already over")
        self.mgr.rearm(r, hours_from_now=0.25)
        self.assertEqual(self.mgr.rental("s1").cap_epoch, self.clock() + 900)

    def test_rearm_bounds(self):
        r = self.create()
        with self.assertRaises(RentalRefused):
            self.mgr.rearm(r, at_epoch=self.clock() - 5)
        with self.assertRaisesRegex(RentalRefused, "hours ahead"):
            self.mgr.rearm(r, hours_from_now=25)
        with self.assertRaises(GpuError):
            self.mgr.rearm(r)
        with self.assertRaises(GpuError):
            self.mgr.rearm(r, at_epoch=1, hours_from_now=1)

    def test_failed_arming_restores_the_old_cap(self):
        r = self.create(hours=1)
        self.launcher.fail = True
        with self.assertRaisesRegex(GpuError, "did not arm"):
            self.mgr.rearm(r, hours_from_now=2)
        self.assertEqual(self.mgr.rental("s1").cap_epoch, r.cap_epoch)

    def test_shortening_keeps_the_old_commitment_until_the_new_watchdog_is_armed(self):
        r = self.create(hours=12)  # 12 * 1.31 = 15.72 of the 20.00 stop
        outcome = {}
        inner = self.launcher

        def launcher(session_dir, session, ledger, at_epoch):
            if session == "s1" and "ran" not in outcome:
                outcome["ran"] = True
                try:
                    outcome["created"] = self.mgr.create("s2", SHAPE, 5, "t-0002", "train")  # 6.55 fits only if the 12 h cap were released
                except RentalRefused as exc:
                    outcome["refused"] = exc
            return inner(session_dir, session, ledger, at_epoch)

        self.mgr.launcher = launcher
        self.mgr.rearm(r, hours_from_now=1)
        self.assertIn("refused", outcome, "a concurrent create spent money the old cap still commits")
        self.assertEqual(self.mgr.rental("s1").cap_epoch, self.clock() + 3600)

    def test_failed_arming_of_a_shorter_cap_leaves_the_old_cap(self):
        r = self.create(hours=12)
        self.launcher.fail = True
        with self.assertRaisesRegex(GpuError, "did not arm"):
            self.mgr.rearm(r, hours_from_now=1)
        self.assertEqual(self.mgr.rental("s1").cap_epoch, r.cap_epoch)
        self.assertEqual(self.mgr.read()["rentals"][0]["rearms"], [])

    def test_rearm_of_a_deleted_rental_is_refused(self):
        r = self.create()
        self.mgr.delete(r, "x")
        with self.assertRaises(RentalRefused):
            self.mgr.rearm(r, hours_from_now=1)


class LedgerFile(Base):
    def test_two_processes_worth_of_writers_do_not_lose_updates(self):
        self.create("s1", hours=1)
        errors = []

        def worker(i):
            try:
                for k in range(20):
                    self.mgr._update("s1", **{f"w{i}": k})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        row = self.mgr.read()["rentals"][0]
        self.assertEqual([row[f"w{i}"] for i in range(4)], [19] * 4)

    def test_ledger_is_valid_json_and_has_no_private_key_material(self):
        self.create()
        text = (self.tmp / "gpu" / "rentals.json").read_text()
        json.loads(text)
        self.assertNotIn("PRIVATE KEY", text)

    def test_wrong_schema_is_refused(self):
        (self.tmp / "gpu" / "rentals.json").write_text(json.dumps({"schema": "other"}))
        with self.assertRaises(GpuError):
            self.mgr.read()


if __name__ == "__main__":
    unittest.main()
