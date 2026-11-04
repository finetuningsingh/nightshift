"""``scripts/simulate_live.py``: the rehearsal of a live run against the loopback simulator.

What is pinned here: argument validation happens before anything starts (a bad ``--plant`` is exit 2 with the CLI's own wording
and an existing ``--out`` is never touched); a run that fails before ``run.started`` is not verified, quotes its own error and
leaves no empty run directory behind (a directory that holds events is never removed); a normal run ends in ``verify`` and the
summary with the endpoint recorded as loopback (a SIMULATION, never a Meta run); Ctrl-C while ``--serve`` is active stops the
child, still verifies the ledger and exits 130. Everything runs on loopback; there is no network and no API key.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from nightshift.cli import PLANTABLE
from nightshift.ledger import Ledger
from tests.support.synthetic import FakeClock, RunBuilder

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "simulate_live.py"

_spec = importlib.util.spec_from_file_location("simulate_live_under_test", SCRIPT)
sim_live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim_live)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_script(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for k in ("MODEL_API_KEY", "META_API_KEY"):
        env.pop(k, None)
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def events_of(run_dir: Path) -> list:
    return Ledger.read_only(run_dir).events()


class CaseWithTmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()


class ArgumentValidationTests(CaseWithTmp):
    def main(self, *argv: str):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = sim_live.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_an_unknown_plant_name_is_exit_2_with_the_clis_wording_and_list_and_nothing_is_created(self):
        out_dir = self.tmp / "sim"
        code, out, err = self.main("--plant", "bogus", "--out", str(out_dir))
        self.assertEqual(code, 2)
        self.assertIn("--plant: unknown behavior 'bogus'; choose from " + ", ".join(PLANTABLE), err)
        self.assertNotIn("SIMULATED MODEL", out)            # nothing started, no simulator, no child
        self.assertFalse(out_dir.exists())
        self.assertFalse(out_dir.parent.joinpath("sim.lock").exists())

    def test_a_bad_name_among_good_ones_and_malformed_lists_are_refused_up_front(self):
        for plant, fragment in (("tamper,bogus", "unknown behavior 'bogus'"), ("tamper,,overfit", "separated by commas"),
                                ("tamper,tamper", "listed more than once")):
            code, out, err = self.main("--plant", plant, "--out", str(self.tmp / "sim"))
            self.assertEqual(code, 2, plant)
            self.assertIn(fragment, err)
            self.assertNotIn("SIMULATED MODEL", out)
            self.assertFalse((self.tmp / "sim").exists())

    def test_the_names_accepted_are_exactly_the_ones_the_cli_plants(self):
        from nightshift.agents.scripted import PLANTABLE as scripted_plantable
        self.assertEqual(tuple(PLANTABLE), tuple(scripted_plantable))

    def test_an_existing_out_directory_is_refused_and_left_exactly_as_it_was(self):
        out_dir = self.tmp / "sim"
        out_dir.mkdir()
        (out_dir / "keep.txt").write_text("mine")
        code, out, err = self.main("--out", str(out_dir))
        self.assertEqual(code, 2)
        self.assertIn("already exists; choose another --out", err)
        self.assertNotIn("SIMULATED MODEL", out)
        self.assertEqual({p.name for p in out_dir.iterdir()}, {"keep.txt"})
        self.assertEqual((out_dir / "keep.txt").read_text(), "mine")

    def test_an_out_path_below_a_file_is_a_usage_error_and_never_a_traceback(self):
        blocker = self.tmp / "afile"
        blocker.write_text("a file")
        code, out, err = self.main("--out", str(blocker / "sim"))
        self.assertEqual(code, 2)
        self.assertIn("cannot create the parent of", err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("SIMULATED MODEL", out)
        self.assertEqual(blocker.read_text(), "a file")

    def test_an_existing_out_file_is_refused_too(self):
        target = self.tmp / "sim"
        target.write_text("a file")
        self.assertEqual(self.main("--out", str(target))[0], 2)
        self.assertEqual(target.read_text(), "a file")

    def test_the_existing_directory_error_comes_through_the_real_script_too(self):
        out_dir = self.tmp / "sim"
        out_dir.mkdir()
        proc = run_script("--out", str(out_dir), timeout=60)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("already exists", proc.stderr)
        self.assertNotIn("run directory:", proc.stdout)

    def test_the_docstring_states_the_run_time_and_the_measured_plant_list(self):
        doc = sim_live.__doc__
        self.assertIn("5 to 10 seconds", doc)
        self.assertNotIn("about a minute", doc)
        for name in ("tamper", "injection", "hang", "infra_flake", "duplicate", "claim_without_run", "overfit"):
            self.assertIn(name, doc)
        self.assertIn("did NOT run", doc)
        example = [ln for ln in doc.splitlines() if "--plant" in ln and "python3" in ln]
        self.assertTrue(example)
        for ln in example:         # the module's own example only uses plants that DO run
            plants = ln.split("--plant")[1].split()[0].split(",")
            self.assertFalse({"claim_without_run", "overfit"} & set(plants), ln)


class FailedStartTests(CaseWithTmp):
    def test_a_bad_task_exits_nonzero_skips_verify_and_quotes_the_runs_own_error(self):
        out_dir = self.tmp / "sim"
        proc = run_script("--task", "no-such-task", "--out", str(out_dir), timeout=60)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("no task found for 'no-such-task'", proc.stderr)
        self.assertIn("failed before it started (exit 2)", proc.stderr)
        self.assertNotIn("events:", proc.stdout)                    # verify never ran
        self.assertNotIn("has no ledger.jsonl", proc.stdout + proc.stderr)
        self.assertNotIn("run directory:", proc.stdout)
        self.assertFalse(out_dir.exists())
        # the same --out is usable again: the next attempt fails for the same honest reason, not with 'already exists'
        again = run_script("--task", "no-such-task", "--out", str(out_dir), timeout=60)
        self.assertEqual(again.returncode, 2)
        self.assertNotIn("already exists", again.stderr)

    def test_the_printed_command_is_shell_quoted_so_it_can_be_pasted(self):
        out_dir = self.tmp / "sim"
        proc = run_script("--task", "no such task", "--out", str(out_dir), timeout=60)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("--task 'no such task'", proc.stdout)

    def test_a_busy_port_leaves_no_empty_run_directory_behind(self):
        out_dir = self.tmp / "sim"
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            proc = run_script("--serve", "--port", str(busy.getsockname()[1]), "--out", str(out_dir), timeout=90)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("cannot start the dashboard", proc.stderr)
        self.assertIn("failed before it started", proc.stderr)
        self.assertIn("removed the empty run directory", proc.stderr)
        self.assertNotIn("events:", proc.stdout)                    # no verify of the failed run
        self.assertNotIn("run directory:", proc.stdout)
        self.assertFalse(out_dir.exists(), "the empty run directory must be gone so --out can be reused")

    def test_a_failed_start_does_not_touch_a_directory_this_invocation_did_not_create(self):
        out_dir = self.tmp / "sim"
        out_dir.mkdir()
        proc = run_script("--task", "no-such-task", "--out", str(out_dir), timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("already exists", proc.stderr)
        self.assertTrue(out_dir.is_dir())


class CleanupHelperTests(CaseWithTmp):
    def empty_run(self, name: str = "run") -> Path:
        d = self.tmp / name
        led = Ledger.create(d, clock=FakeClock(), fsync=False)
        led.close()
        return d

    def test_an_empty_ledger_directory_is_removed(self):
        d = self.empty_run()
        self.assertEqual(sim_live._ledger_state(d), (0, False))
        self.assertTrue(sim_live._remove_if_empty(d))
        self.assertFalse(d.exists())

    def test_a_directory_that_holds_events_is_never_removed(self):
        d = self.tmp / "run"
        led = Ledger.create(d, clock=FakeClock(), fsync=False)
        RunBuilder(led).start("fake-task", backend="meta")
        led.close()
        events, started = sim_live._ledger_state(d)
        self.assertEqual((events > 0, started), (True, True))
        self.assertFalse(sim_live._remove_if_empty(d))
        self.assertTrue((d / "ledger.jsonl").is_file())

    def test_a_directory_with_other_files_is_kept(self):
        d = self.empty_run()
        (d / "notes.txt").write_text("not ours to delete")
        self.assertFalse(sim_live._remove_if_empty(d))
        self.assertTrue((d / "notes.txt").is_file())

    def test_a_missing_directory_and_an_unreadable_ledger_are_handled(self):
        self.assertEqual(sim_live._ledger_state(self.tmp / "nope"), (0, False))
        self.assertFalse(sim_live._remove_if_empty(self.tmp / "nope"))
        d = self.tmp / "bad"
        d.mkdir()
        (d / "ledger.jsonl").write_text("{not json\n{also not\n")
        events, started = sim_live._ledger_state(d)
        self.assertEqual((events, started), (None, True))           # unknown: kept and still verified
        self.assertFalse(sim_live._remove_if_empty(d))

    def test_the_runs_own_error_is_the_last_error_line_else_the_last_line(self):
        self.assertEqual(sim_live._own_error(["task x", "error: first", "", "error: cannot start the dashboard: busy", "bye"]),
                         "cannot start the dashboard: busy")
        self.assertEqual(sim_live._own_error(["a", "Traceback ...", "ValueError: boom"]), "ValueError: boom")
        self.assertIn("no error message", sim_live._own_error([]))


class ExitStatusTests(unittest.TestCase):
    def test_a_child_killed_by_a_signal_becomes_the_shells_128_plus_signal(self):
        self.assertEqual(sim_live._exit_status(-signal.SIGKILL), 128 + signal.SIGKILL)
        self.assertEqual(sim_live._exit_status(-signal.SIGTERM), 128 + signal.SIGTERM)
        self.assertEqual((sim_live._exit_status(0), sim_live._exit_status(2)), (0, 2))


class NormalRunTests(CaseWithTmp):
    def test_a_short_run_verifies_prints_the_summary_and_records_a_loopback_endpoint(self):
        out_dir = self.tmp / "sim"
        t0 = time.monotonic()
        proc = run_script("--plant", "tamper,injection", "--out", str(out_dir))
        elapsed = time.monotonic() - t0
        self.assertEqual(proc.returncode, 0, proc.stdout[-1500:] + proc.stderr[-1500:])
        self.assertLess(elapsed, 60.0)
        self.assertIn("SIMULATED MODEL on loopback; nothing here is Meta model output.", proc.stdout)
        self.assertIn("chain consistent", proc.stdout)
        self.assertIn(f"run directory: {out_dir}", proc.stdout)
        self.assertRegex(proc.stdout, r"requests served by the simulator: [1-9]\d*")
        events = events_of(out_dir)
        started = [e for e in events if e.kind == "run.started"][0].data
        self.assertEqual(started["config"]["endpoint"]["kind"], "loopback")        # a simulation, never "meta-api"
        self.assertEqual([r["behavior"] for r in started["config"]["red_team"]], ["tamper", "injection"])
        lanes = {e.data.get("lane") for e in events if e.kind == "experiment.proposed"}
        self.assertTrue({"RT-tamper", "RT-injection"} <= lanes, lanes)
        self.assertFalse(any(e.kind == "budget.charged" and "meta-api" in json.dumps(e.data) for e in events))


class CtrlCTests(CaseWithTmp):
    def wait_for_health(self, port: int, proc: subprocess.Popen, timeout: float = 90.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self.fail(f"the script exited early with {proc.returncode}: {proc.stdout.read()[-1500:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.1)
        self.fail("the dashboard never answered /api/health")

    def serve_then_signal(self, signum: int) -> tuple[int, str, int]:
        """Start the script with --serve on a free port, wait for the dashboard, send ``signum``; (exit code, output, port)."""
        out_dir = self.tmp / "sim"
        port = free_port()
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        for k in ("MODEL_API_KEY", "META_API_KEY"):
            env.pop(k, None)
        # SIGINT is reset to its default in the child: a test runner started from a background job inherits SIGINT as ignored, and
        # Python then never turns it into KeyboardInterrupt. Its own session, so a failure can kill the script AND its run together.
        proc = subprocess.Popen([sys.executable, str(SCRIPT), "--serve", "--port", str(port), "--out", str(out_dir)], cwd=ROOT,
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))

        def kill_group() -> None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass

        self.addCleanup(kill_group)
        self.addCleanup(proc.stdout.close)
        self.wait_for_health(port, proc)
        proc.send_signal(signum)
        try:
            out, _ = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            kill_group()
            out, _ = proc.communicate(timeout=20)
            self.fail("the script did not stop after the signal:\n" + out[-2000:])
        return proc.returncode, out, port

    def assert_stopped_cleanly(self, code: int, out: str, port: int) -> None:
        out_dir = self.tmp / "sim"
        self.assertEqual(code, 130, out[-2000:])
        self.assertNotIn("Traceback", out)
        self.assertNotIn("KeyboardInterrupt", out)
        self.assertIn("interrupted", out)
        # verify still ran on the interrupted ledger. Its verdict is not asserted: an interrupt that lands inside Ledger.append
        # (between the write and the in-memory update) makes the next append reuse a seq and breaks the chain; that defect is in
        # nightshift/ledger.py (reported to its owner), and is not this script's to hide or cause
        self.assertRegex(out, r"events: \d+")
        self.assertRegex(out, r"chain (consistent|BROKEN at seq \d+)")
        self.assertIn(f"run directory: {out_dir}", out)
        self.assertRegex(out, r"requests served by the simulator: \d+")
        self.assertEqual(events_of(out_dir)[0].kind, "run.started")
        with self.assertRaises((urllib.error.URLError, OSError)):    # the child (and its dashboard) is gone
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2)

    def test_sigint_while_serving_stops_the_child_verifies_the_ledger_and_exits_130(self):
        self.assert_stopped_cleanly(*self.serve_then_signal(signal.SIGINT))

    def test_sigterm_takes_the_same_path_and_never_leaves_the_run_and_its_port_behind(self):
        self.assert_stopped_cleanly(*self.serve_then_signal(signal.SIGTERM))

    def test_a_closed_run_is_not_waited_for_in_full(self):
        """After SIGINT a --serve child only holds its dashboard open; once run.finished is in the ledger the script moves on."""
        d = self.tmp / "closed"
        led = Ledger.create(d, clock=FakeClock(), fsync=False)
        led.append("run.finished", "orchestrator", {})
        led.close()
        self.assertTrue(sim_live._run_finished(d))
        self.assertFalse(sim_live._run_finished(self.tmp / "nope"))
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        t0 = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            sim_live._wait_slices(proc, 30.0, d, stop_when_finished=True)
        self.assertLess(time.monotonic() - t0, 10.0)
        # without the flag (or without a run directory) the full patience applies
        with self.assertRaises(subprocess.TimeoutExpired):
            sim_live._wait_slices(proc, 1.0, d, stop_when_finished=False)

    def test_the_wait_helper_escalates_and_never_leaves_the_child_running(self):
        """A child that ignores SIGINT is terminated after the patience runs out (patience shortened for the test)."""
        code = ("import signal,sys,time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nprint('up', flush=True)\n"
                "time.sleep(60)\n")
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        self.addCleanup(proc.stdout.close)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        self.assertEqual(proc.stdout.readline().strip(), "up")
        real_wait, calls = proc.wait, []

        def interrupting_wait(timeout=None):
            if not calls:
                calls.append("ctrl-c")
                raise KeyboardInterrupt
            return real_wait(timeout)

        proc.wait = interrupting_wait
        saved = (sim_live.CHILD_GRACE_SEC, sim_live.CHILD_INT_SEC, sim_live.CHILD_TERM_SEC)
        sim_live.CHILD_GRACE_SEC = sim_live.CHILD_INT_SEC = sim_live.CHILD_TERM_SEC = 0.5
        self.addCleanup(lambda: setattr(sim_live, "CHILD_GRACE_SEC", saved[0]))
        self.addCleanup(lambda: setattr(sim_live, "CHILD_INT_SEC", saved[1]))
        self.addCleanup(lambda: setattr(sim_live, "CHILD_TERM_SEC", saved[2]))
        with redirect_stderr(io.StringIO()):
            code_, interrupted = sim_live._wait(proc)
        self.assertTrue(interrupted)
        self.assertEqual(code_, -signal.SIGTERM)
        self.assertIsNotNone(proc.poll())


if __name__ == "__main__":
    unittest.main()
