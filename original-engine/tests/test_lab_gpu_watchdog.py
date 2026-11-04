import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from nightshift.lab import gpu_watchdog as wd
from nightshift.lab.contracts import Shape
from nightshift.lab.gpu_rentals import RentalManager, alive_watchdogs
from tests.support.lab_fake_gpu import FakeClock, FakeProvider, FakeWatchdogLauncher, fake_key_factory

SHAPE = Shape("a100", 1, 12, 300)
REPO = Path(__file__).resolve().parents[1]


class StopLoop(Exception):
    pass


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.clock = FakeClock()
        self.provider = FakeProvider(self.clock)
        self.launcher = FakeWatchdogLauncher()
        self.addCleanup(self.launcher.close)
        self.ledger = self.tmp / "gpu" / "rentals.json"
        self.mgr = RentalManager(
            self.ledger, self.provider, clock=self.clock, launcher=self.launcher, key_factory=fake_key_factory,
            sleep=lambda s: self.clock.advance(s), stopper=self.launcher.stop,
        )  # fmt: skip
        self.mgr.init(50.0, "box", "test")
        self.logs = []

    def sleeper(self, limit=10_000):
        calls = {"n": 0}

        def sleep(s):
            calls["n"] += 1
            if calls["n"] > limit:
                raise StopLoop()
            self.clock.advance(s)

        sleep.calls = calls
        return sleep

    def run_watchdog(self, session="s1", sleep=None, **kw):
        return wd.watchdog(session, self.ledger, self.provider, clock=self.clock, sleep=sleep or self.sleeper(), log=self.logs.append, **kw)

    def create(self, session="s1", hours=1.0):
        return self.mgr.create(session, SHAPE, hours, "t-0001", "train")


class Watchdog(Base):
    def test_deletes_at_the_cap_and_not_before(self):
        r = self.create(hours=1)
        seen = []

        def sleep(s):
            seen.append((self.clock(), self.provider.handles()))
            self.clock.advance(s)

        self.assertEqual(self.run_watchdog(sleep=sleep), 0)
        before_cap = [h for t, h in seen if t < r.cap_epoch]
        self.assertTrue(before_cap and all(r.handle in h for h in before_cap), "never deletes early")
        self.assertNotIn(r.handle, self.provider.handles())
        self.assertGreaterEqual(self.clock(), r.cap_epoch)
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "deleted")
        self.assertIn("watchdog", self.mgr.read()["rentals"][0]["delete_reason"])

    def test_holds_the_liveness_lock_while_it_runs_and_releases_it(self):
        r = self.create()
        alive_during = []

        def sleep(s):
            alive_during.append(len(alive_watchdogs(r.session_dir)))
            self.clock.advance(s)

        self.run_watchdog(sleep=sleep)
        self.assertIn(2, alive_during)  # the fake launcher's lock plus the real watchdog's own
        self.assertEqual(len(alive_watchdogs(r.session_dir)), 1)  # only the fake launcher's lock remains
        info = json.loads((r.session_dir / "watchdog.json").read_text())
        self.assertEqual(info["pid"], os.getpid())
        self.assertEqual(info["delete_at_epoch"], int(r.cap_epoch))

    def test_refuses_an_epoch_later_than_the_ledger_cap(self):
        r = self.create()
        self.assertEqual(self.run_watchdog(at_epoch=int(r.cap_epoch) + 1), 2)
        self.assertIn(r.handle, self.provider.handles())
        self.assertTrue(any("past the ledger cap" in m for m in self.logs))

    def test_refuses_a_session_that_is_not_in_the_ledger(self):
        self.assertEqual(self.run_watchdog(session="nope"), 2)

    def test_an_earlier_epoch_is_allowed(self):
        r = self.create(hours=1)
        self.assertEqual(self.run_watchdog(at_epoch=int(self.clock() + 100)), 0)
        self.assertNotIn(r.handle, self.provider.handles())
        self.assertLess(self.clock(), r.cap_epoch)

    def test_exits_quietly_when_the_rental_was_deleted_first(self):
        r = self.create()
        self.mgr.delete(r, "finished early")
        self.assertEqual(self.run_watchdog(), 0)
        self.assertEqual(self.provider.calls_of("delete").__len__(), 1)

    def test_exits_without_deleting_when_a_rearm_moved_the_cap(self):
        r = self.create(hours=1)
        rearmed = []

        def sleep(s):
            if not rearmed:  # the operator rearms while this (old) watchdog sleeps
                rearmed.append(self.mgr.rearm(r, hours_from_now=3))
            self.clock.advance(s)

        self.assertEqual(self.run_watchdog(sleep=sleep), 0)
        self.assertGreaterEqual(self.clock(), r.cap_epoch)
        self.assertIn(r.handle, self.provider.handles(), "the old watchdog must not delete after a rearm")
        self.assertTrue(any("cap moved" in m for m in self.logs))

    def test_deletes_at_its_own_deadline_when_a_rearm_died_before_arming_its_replacement(self):
        r = self.create(hours=1)
        self.launcher.stop(alive_watchdogs(r.session_dir)[0])  # the real watchdog under test takes the place of the fake launcher's
        moved = []

        def sleep(s):
            if not moved:  # rearm publishes the later cap, then the coordinator dies before arming the replacement
                moved.append(self.mgr._update("s1", cap_epoch=r.cap_epoch + 23 * 3600))
            self.clock.advance(s)

        self.assertEqual(self.run_watchdog(sleep=sleep), 0)
        self.assertNotIn(r.handle, self.provider.handles(), "no replacement watchdog exists, so the old one must still delete")
        self.assertTrue(any("no other watchdog is alive" in m for m in self.logs))

    def test_a_default_cap_with_a_fractional_clock_arms_a_watchdog_the_real_watchdog_accepts(self):
        self.clock.now = 1000.25
        r = self.create(hours=1 / 60)
        epoch = self.launcher.calls[-1][1]
        self.assertLessEqual(epoch, r.cap_epoch)
        self.assertEqual(self.run_watchdog(at_epoch=epoch), 0)
        self.assertNotIn(r.handle, self.provider.handles())

    def test_retries_until_absence_is_confirmed(self):
        r = self.create()
        self.provider.fail_delete_n = 3
        sleep = self.sleeper()
        self.assertEqual(self.run_watchdog(sleep=sleep), 0)
        self.assertNotIn(r.handle, self.provider.handles())
        self.assertGreaterEqual(len([m for m in self.logs if "retrying" in m]), 3)

    def test_retries_through_provider_outages(self):
        r = self.create()
        self.clock.advance(3601)
        self.provider.fail_list_n = 4
        self.assertEqual(self.run_watchdog(), 0)
        self.assertNotIn(r.handle, self.provider.handles())

    def test_never_deletes_an_instance_without_our_tag(self):
        r = self.create()
        foreign = self.provider.add_foreign(SHAPE)
        with self.mgr._locked() as doc:
            doc["rentals"][0]["handle"] = foreign
        self.clock.advance(3601)
        with self.assertRaises(StopLoop):
            self.run_watchdog(sleep=self.sleeper(limit=5))
        self.assertIn(foreign, self.provider.handles())
        self.assertEqual(self.provider.calls_of("delete"), [])
        self.assertTrue((r.session_dir / "ALARM").exists())

    def test_unsettled_rental_is_searched_by_tag_and_an_unfindable_one_alarms_without_deleting_anything(self):
        self.provider.create_mode = "ambiguous_not_created"
        with self.assertRaises(Exception):
            self.create()
        foreign = self.provider.add_foreign(SHAPE)
        sd = self.mgr.sessions_dir / "s1"
        self.clock.advance(3601)
        with self.assertRaises(StopLoop):
            self.run_watchdog(sleep=self.sleeper(limit=4))
        self.assertTrue((sd / "ALARM").exists())
        self.assertEqual(self.provider.handles(), {foreign})
        # the instance appears (the create was slow): the watchdog finds it by tag and deletes it
        row = self.mgr.read()["rentals"][0]
        pub = (sd / "id_ed25519.pub").read_text().strip()
        self.provider.create_mode = "ok"
        self.provider.create(SHAPE, pub)
        self.provider.instances["fake0002x"] = type(self.provider.instances["fake0002x"])(
            "fake0002x", "RUNNING", SHAPE, self.clock() - 10, row["owner_tag"], "127.0.0.1", 22
        )
        self.assertEqual(self.run_watchdog(sleep=self.sleeper()), 0)
        self.assertEqual(self.provider.handles(), {foreign})


class Reap(Base):
    def test_reaps_only_rentals_far_past_their_cap(self):
        a = self.create("s1", hours=1)
        self.clock.advance(3600 + 599)
        self.assertEqual(wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper()), [])
        self.assertIn(a.handle, self.provider.handles())
        self.clock.advance(2)
        out = wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper())
        self.assertEqual([(o["session"], o["action"]) for o in out], [("s1", "deleted")])
        self.assertNotIn(a.handle, self.provider.handles())

    def test_is_idempotent(self):
        self.create("s1", hours=1)
        self.clock.advance(5000)
        first = wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper())
        second = wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper())
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

    def test_never_touches_instances_that_are_not_in_the_ledger(self):
        foreign = self.provider.add_foreign(SHAPE)
        self.create("s1", hours=1)
        self.clock.advance(10_000)
        wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper())
        self.assertEqual(self.provider.handles(), {foreign})

    def test_reports_failures_and_keeps_going(self):
        self.create("s1", hours=1)
        self.create("s2", hours=1)
        self.clock.advance(10_000)
        self.provider.fail_list_n = 10
        out = wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper())
        self.assertEqual([o["action"] for o in out], ["failed", "failed"])
        self.provider.fail_list_n = 0
        out = wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper())
        self.assertEqual([o["action"] for o in out], ["deleted", "deleted"])

    def test_no_record_mode_deletes_but_never_writes_the_ledger(self):
        r = self.create("s1", hours=1)
        self.clock.advance(10_000)
        before = self.ledger.read_bytes()
        out = wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper(), record=False)
        self.assertEqual(out[0]["action"], "deleted")
        self.assertNotIn(r.handle, self.provider.handles())
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_an_expired_unsettled_rental_is_reaped_by_tag(self):
        # the case-C hazard: the coordinator died after the provider create and before the handle was recorded
        self.provider.create_mode = "ambiguous_created"
        orig = self.provider.find_created
        self.provider.find_created = lambda *a, **k: None
        with self.assertRaises(Exception):
            self.create("s1", hours=1)
        self.provider.find_created = orig
        self.assertEqual(self.mgr.read()["rentals"][0]["state"], "uuid_unknown")
        self.assertEqual(len(self.provider.handles()), 1)
        self.clock.advance(5000)
        out = wd.reap(self.ledger, self.provider, clock=self.clock, sleep=self.sleeper())
        self.assertEqual(out[0]["action"], "deleted")
        self.assertEqual(self.provider.handles(), set())


class CommandLine(Base):
    def run_main(self, argv, provider=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = wd.main(argv, provider=provider, clock=self.clock, sleep=self.sleeper())
        return code, out.getvalue(), err.getvalue()

    def test_reap_entry_point(self):
        self.create("s1", hours=1)
        self.clock.advance(5000)
        code, out, _ = self.run_main(["reap", "--ledger", str(self.ledger)], provider=self.provider)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.splitlines()[0])["action"], "deleted")

    def test_reap_exit_status_is_nonzero_when_a_delete_fails(self):
        self.create("s1", hours=1)
        self.clock.advance(5000)
        self.provider.fail_list_n = 10
        code, out, _ = self.run_main(["reap", "--ledger", str(self.ledger)], provider=self.provider)
        self.assertEqual(code, 1)

    def test_canary(self):
        code, out, _ = self.run_main(["canary"], provider=self.provider)
        self.assertEqual((code, json.loads(out)["ok"]), (0, True))
        self.provider.fail_list_n = 1
        code, _, err = self.run_main(["canary"], provider=self.provider)
        self.assertEqual(code, 1)
        self.assertIn("CANARY FAILED", err)

    def test_watchdog_entry_point(self):
        r = self.create()
        code, _, _ = self.run_main(["watchdog", "s1", "--ledger", str(self.ledger)], provider=self.provider)
        self.assertEqual(code, 0)
        self.assertNotIn(r.handle, self.provider.handles())

    def test_missing_token_gives_a_clean_failure_for_canary(self):
        env = {k: v for k, v in os.environ.items() if k != "THUNDER_API_KEY"}
        p = subprocess.run(
            [sys.executable, "-m", "nightshift.lab.gpu_watchdog", "canary", "--base-url", "http://127.0.0.1:1"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
        )  # fmt: skip
        self.assertEqual(p.returncode, 1)
        self.assertIn("CANARY FAILED", p.stderr)

    def test_help_runs(self):
        p = subprocess.run([sys.executable, "-m", "nightshift.lab.gpu_watchdog", "--help"], cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0)
        self.assertIn("reap", p.stdout)


class DetachedProcess(unittest.TestCase):
    """A real watchdog PROCESS, started by a parent that then dies (the coordinator crash), deletes the instance at its deadline."""

    def test_watchdog_survives_its_parent_and_deletes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            state = tmp / "account.json"
            os.environ["LAB_FAKE_GPU_STATE"] = str(state)
            self.addCleanup(os.environ.pop, "LAB_FAKE_GPU_STATE", None)
            provider = FakeProvider(time.time, state_path=state)
            launcher = FakeWatchdogLauncher()
            self.addCleanup(launcher.close)
            ledger = tmp / "gpu" / "rentals.json"
            mgr = RentalManager(ledger, provider, clock=time.time, launcher=launcher, key_factory=fake_key_factory, sleep=lambda s: None)
            mgr.init(50.0, "box", "test")
            r = mgr.create("s1", SHAPE, 1.0, "t-0001", "train")
            self.assertIn(r.handle, provider.handles())
            deadline = int(time.time()) + 3
            parent = (
                "import os, sys\n"
                "from pathlib import Path\n"
                "from nightshift.lab.gpu_rentals import spawn_watchdog_process\n"
                "launch = spawn_watchdog_process(['--provider-factory', 'tests.support.lab_fake_gpu:file_provider_factory'])\n"
                f"launch(Path({str(r.session_dir)!r}), 's1', Path({str(ledger)!r}), {deadline})\n"
                "os._exit(0)\n"  # the coordinator dies right after spawning
            )
            env = dict(os.environ, PYTHONPATH=str(REPO))
            subprocess.run([sys.executable, "-c", parent], cwd=REPO, env=env, check=True, timeout=60)
            info_path = r.session_dir / "watchdog.json"
            for _ in range(100):  # armed: lock held and watchdog.json carries the child's pid
                if info_path.exists() and json.loads(info_path.read_text()).get("delete_at_epoch") == deadline:
                    break
                time.sleep(0.1)
            info = json.loads(info_path.read_text())
            self.assertNotEqual(info["pid"], os.getpid())
            self.assertIn(info["pid"], alive_watchdogs(r.session_dir))
            self.assertEqual(os.getsid(info["pid"]), info["pid"], "the watchdog leads its own session")
            self.assertIn(r.handle, provider.handles(), "not deleted before the deadline")
            for _ in range(200):
                if r.handle not in provider.handles():
                    break
                time.sleep(0.1)
            self.assertNotIn(r.handle, provider.handles())
            for _ in range(50):
                if info["pid"] not in alive_watchdogs(r.session_dir):
                    break
                time.sleep(0.1)
            self.assertNotIn(info["pid"], alive_watchdogs(r.session_dir))
            self.assertEqual(mgr.read()["rentals"][0]["state"], "deleted")


if __name__ == "__main__":
    unittest.main()
