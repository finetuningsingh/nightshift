"""Tests for nightshift.cli. Commands that need the orchestrator run through a monkeypatched ``cli._api`` seam; every other
command is exercised with real files."""
from __future__ import annotations

import inspect
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nightshift import cli, secrets
from nightshift.dashboard.state_api import format_anchor
from nightshift.ledger import Ledger
from nightshift.orchestrator import ResumeRefused
from nightshift.types import BudgetLimits, Kind, NightshiftError, SimulatedCrash
from tests.support.synthetic import FakeClock, RunBuilder

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mini_task"
KEY_CANARY = "sk-canary-0123456789abcdef"


def build_clean(run_dir, backend: str = "scripted", *, variant: str = "clean", task: str = "fake-task") -> Ledger:
    """A finished run: baseline, one replicated promotion. Variants add the conditions the exit code depends on."""
    led = Ledger.create(run_dir, clock=FakeClock(), fsync=False)
    rb = RunBuilder(led)
    rb.start(task, backend=backend)
    base = rb.baseline(1.0, 0.9)
    if variant != "no_incumbent":
        e = rb.experiment("A", "quadratic", score=0.8, train=0.7, delta=0.2, noise_margin=0.01)
        rb.replication_baseline(base, 1.0)
        rb.replicate(e, 0.8)
        rb.promote(e, score=0.8)
        if variant == "open_block":
            rb.finding("I03", e, exp_id=e)
    reason, verdict = "supervisor_finish", "success"
    if variant == "no_incumbent":
        reason, verdict = "budget_exhausted", "missing_authority"
    if variant == "defects":
        verdict = "unresolved_defects"
    if variant == "error_stop":       # the model API refused the key: reason error, verdict missing_authority
        reason, verdict = "error", "missing_authority"
    led.append(Kind.RUN_FINISHED, "engine", {"reason": reason, "summary": "done", "verdict_class": verdict, "final_claims": []})
    led.close()
    return led


class CliCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()     # a symlinked TMPDIR (macOS /var -> /private/var) must not matter
        env = {k: v for k, v in os.environ.items() if k not in cli.KEY_ENVS}
        p = mock.patch.dict(os.environ, env, clear=True)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(secrets.clear)

    def cli(self, *argv: str):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def run_dir(self, name: str = "run", **kw) -> Path:
        d = self.tmp / name
        build_clean(d, **kw)
        return d

    def tampered_dir(self, name: str = "tampered", line: int = 7) -> Path:
        """A copy of a small finished run with one event edited in place: the hash chain breaks at seq ``line + 1``."""
        d = self.run_dir(name)
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)           # a finished run's ledger is 0444
        lines = path.read_text().splitlines()
        row = json.loads(lines[line])
        row["data"]["x"] = 1
        lines[line] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        return d


# --------------------------------------------------------------------------------------------------
# commands that need no orchestrator
# --------------------------------------------------------------------------------------------------
class ParserTests(CliCase):
    def test_usage_errors_exit_2(self):
        for argv in ([], ["bogus"], ["run"], ["run", "--task", "x", "--backend", "gpt"], ["verify"], ["serve"],
                     ["run", "--task", "x", "--max-usd", "abc"], ["demo", "--replay-speed", "-1"],
                     ["serve", "x", "--replay", "5", "--replay-duration", "5"]):
            code, _, err = self.cli(*argv)
            self.assertEqual(code, 2, argv)

    def test_help_exits_0_and_hides_the_crash_hook(self):
        code, out, _ = self.cli("--help")
        self.assertEqual(code, 0)
        for name in ("run", "resume", "serve", "report", "audit", "verify", "doctor", "demo", "tasks"):
            self.assertIn(name, out)
        code, out, _ = self.cli("run", "--help")
        self.assertEqual(code, 0)
        self.assertNotIn("crash-after-runs", out)
        for flag in ("--task", "--backend", "--run-dir", "--max-experiments", "--max-usd", "--max-wall", "--parallel", "--config",
                     "--implementer", "--control", "--serve", "--port"):
            self.assertIn(flag, out)

    def test_hidden_crash_flag_is_still_accepted(self):
        args = cli.build_parser().parse_args(["run", "--task", "t", "--crash-after-runs", "4"])
        self.assertEqual(args.crash_after_runs, 4)

    def test_python_dash_m_nightshift_works_from_any_cwd(self):
        d = self.run_dir()
        env = {**os.environ, "PYTHONPATH": str(REPO)}
        proc = subprocess.run([sys.executable, "-m", "nightshift", "verify", str(d)], cwd=self.tmp, env=env, capture_output=True,
                              text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("chain consistent", proc.stdout)
        bad = subprocess.run([sys.executable, "-m", "nightshift", "nonsense"], cwd=self.tmp, env=env, capture_output=True, text=True,
                             timeout=30)
        self.assertEqual(bad.returncode, 2)


class VerifyTests(CliCase):
    def test_verify_prints_the_anchor_and_succeeds(self):
        d = self.run_dir()
        code, out, _ = self.cli("verify", str(d))
        events = Ledger.read_only(d).events()
        self.assertEqual(code, 0)
        self.assertIn(f"head anchor: {format_anchor(events)}", out)
        self.assertIn("chain consistent", out)
        self.assertIn(f"events: {len(events)}", out)

    def test_expect_head_matches_anchor_or_hash_prefix(self):
        d = self.run_dir()
        events = Ledger.read_only(d).events()
        for good in (format_anchor(events), events[-1].hash[:16], events[-1].hash[:8], events[-1].hash.upper()[:12]):
            code, out, _ = self.cli("verify", str(d), "--expect-head", good)
            self.assertEqual(code, 0, good)
            self.assertIn("matches", out)
        for bad in ("deadbeefdeadbeef", f"{events[-1].seq}:deadbeefdeadbeef", f"{events[-1].seq - 1}:{events[-1].hash[:16]}", "", events[-1].hash[:7]):
            code, out, _ = self.cli("verify", str(d), "--expect-head", bad)
            self.assertEqual(code, 1, bad)
            self.assertIn("MISMATCH", out)

    def test_truncation_is_caught_only_by_the_recorded_anchor(self):
        d = self.run_dir()
        anchor = format_anchor(Ledger.read_only(d).events())
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-3]))
        code, out, _ = self.cli("verify", str(d))
        self.assertEqual(code, 0)                       # still a consistent prefix: that is the documented limit
        code, out, _ = self.cli("verify", str(d), "--expect-head", anchor)
        self.assertEqual(code, 1)
        self.assertIn("truncated", out)

    def test_an_edited_event_is_reported_with_its_seq(self):
        d = self.run_dir()
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)
        lines = path.read_text().splitlines()
        row = json.loads(lines[7])
        row["data"]["x"] = 1
        lines[7] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        code, out, _ = self.cli("verify", str(d))
        self.assertEqual(code, 1)
        self.assertIn("chain BROKEN at seq 8", out)
        code, out, _ = self.cli("verify", str(d), "--expect-head", "anything12345")
        self.assertEqual(code, 1)

    def test_an_unparseable_middle_line_is_a_broken_chain(self):
        d = self.run_dir()
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)
        lines = path.read_text().splitlines()
        lines[4] = "{broken"
        path.write_text("\n".join(lines) + "\n")
        code, out, _ = self.cli("verify", str(d))
        self.assertEqual(code, 1)
        self.assertIn("chain BROKEN", out)

    def test_torn_final_line_is_noted_not_fatal(self):
        d = self.run_dir()
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)
        with open(path, "ab") as fh:
            fh.write(b'{"seq": 99, "ts":')
        code, out, _ = self.cli("verify", str(d))
        self.assertEqual(code, 0)
        self.assertIn("incomplete", out)

    def test_a_ledger_with_no_events_is_not_a_consistent_run(self):
        d = self.tmp / "empty"
        d.mkdir()
        (d / "ledger.jsonl").write_text("")
        for extra in ([], ["--expect-head", "0:0000000000000000"]):
            code, out, _ = self.cli("verify", str(d), *extra)
            self.assertEqual(code, 1, extra)
            self.assertIn("empty ledger", out)
            self.assertNotIn("chain consistent", out)
            self.assertIn("events: 0", out)

    def test_missing_run_dir_is_a_usage_error(self):
        code, _, err = self.cli("verify", str(self.tmp / "nope"))
        self.assertEqual(code, 2)
        self.assertIn("ledger.jsonl", err)

    def test_anchor_matches_helper(self):
        d = self.run_dir()
        events = Ledger.read_only(d).events()
        self.assertFalse(cli.anchor_matches("x", []))
        self.assertTrue(cli.anchor_matches(f"  {format_anchor(events)}  ", events))


class ReportCommandTests(CliCase):
    def test_report_writes_both_files(self):
        d = self.run_dir()
        code, out, _ = self.cli("report", str(d))
        self.assertEqual(code, 0)
        self.assertTrue((d / "report.md").is_file() and (d / "report.html").is_file())
        self.assertIn("report.html", out)
        out_dir = self.tmp / "elsewhere"
        code, _, _ = self.cli("report", str(d), "--out", str(out_dir))
        self.assertEqual(code, 0)
        self.assertTrue((out_dir / "report.md").is_file())
        self.assertNotIn("<script", (out_dir / "report.html").read_text().lower())

    def test_report_errors(self):
        self.assertEqual(self.cli("report", str(self.tmp / "nope"))[0], 2)
        d = self.run_dir()
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)
        lines = path.read_text().splitlines()
        lines[3] = "garbage"
        path.write_text("\n".join(lines) + "\n")
        code, _, err = self.cli("report", str(d))
        self.assertEqual(code, 1)
        self.assertIn("corrupt", err)


    def test_a_clean_chain_prints_no_warning(self):
        code, out, err = self.cli("report", str(self.run_dir()))
        self.assertEqual((code, err), (0, ""))

    def test_a_broken_chain_still_writes_the_report_but_warns_on_stderr_and_exits_1(self):
        d = self.tampered_dir()
        code, out, err = self.cli("report", str(d))
        self.assertEqual(code, 1)
        self.assertTrue((d / "report.md").is_file() and (d / "report.html").is_file())     # written first
        self.assertIn("report.md", out)
        self.assertIn("BROKEN at seq 8", err)
        self.assertIn("warning", err)

    def test_a_broken_chain_warns_with_a_custom_out_dir_too(self):
        d = self.tampered_dir()
        out_dir = self.tmp / "elsewhere"
        code, _, err = self.cli("report", str(d), "--out", str(out_dir))
        self.assertEqual(code, 1)
        self.assertTrue((out_dir / "report.md").is_file())
        self.assertIn("BROKEN at seq 8", err)


class ServeCommandTests(CliCase):
    def serve(self, *argv: str):
        seen = {}

        def fake_blocking(server):
            seen["server"] = server
            seen["replay"] = server.replay
            seen["speed"] = server.replay_speed
            seen["host"] = server.server_address[0]

        with mock.patch.object(cli, "_serve_blocking", fake_blocking):
            code, out, err = self.cli("serve", *argv)
        return code, out, err, seen

    def test_serve_live_binds_loopback_and_prints_the_url(self):
        d = self.run_dir()
        code, out, _, seen = self.serve(str(d), "--port", "0")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"dashboard: http://127\.0\.0\.1:\d+/")
        self.assertEqual((seen["replay"], seen["host"]), (False, "127.0.0.1"))

    def test_serve_replay_options(self):
        d = self.run_dir()
        code, out, _, seen = self.serve(str(d), "--port", "0", "--replay", "25")
        self.assertEqual((code, seen["replay"], seen["speed"]), (0, True, 25.0))
        self.assertIn("(replay)", out)
        code, _, _, seen = self.serve(str(d), "--port", "0", "--replay-duration", "3")
        self.assertEqual((code, seen["replay"]), (0, True))

    def test_serve_rejects_bad_input(self):
        d = self.run_dir()
        self.assertEqual(self.serve(str(self.tmp / "nope"))[0], 2)
        self.assertEqual(self.serve(str(d), "--port", "0", "--replay", "500")[0], 2)
        code, _, err, _ = self.serve(str(d), "--port", "0", "--host", "0.0.0.0")
        self.assertEqual(code, 2)
        self.assertIn("loopback", err)

    def test_allow_remote_binds_wide_with_a_warning(self):
        d = self.run_dir()
        code, out, err, seen = self.serve(str(d), "--port", "0", "--allow-remote")
        self.assertEqual((code, seen["host"]), (0, "0.0.0.0"))
        self.assertIn("WARNING", err)

    def test_replay_of_a_broken_ledger_is_refused_with_exit_1(self):
        d = self.run_dir()
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)
        lines = path.read_text().splitlines()
        row = json.loads(lines[5])
        row["data"]["x"] = 1
        lines[5] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        code, _, err, _ = self.serve(str(d), "--port", "0", "--replay", "10")
        self.assertEqual(code, 1)
        self.assertIn("broken", err)

    def test_busy_port_is_a_usage_error(self):
        import socket
        d = self.run_dir()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            code, _, err, _ = self.serve(str(d), "--port", str(s.getsockname()[1]))
        self.assertEqual(code, 2)
        self.assertIn("cannot start", err)

    def test_live_serve_of_a_broken_chain_warns_at_start_and_still_serves(self):
        d = self.tampered_dir()
        code, out, err, seen = self.serve(str(d), "--port", "0")
        self.assertEqual(code, 0)
        self.assertIn("BROKEN at seq 8", err)
        self.assertRegex(out, r"dashboard: http://127\.0\.0\.1:\d+/")
        self.assertEqual(seen["replay"], False)

    def test_the_chain_warning_never_raises_on_an_unparseable_or_unreadable_ledger(self):
        d = self.tampered_dir()
        (d / "ledger.jsonl").write_text("{not json\n{also not\n")
        self.assertIn("is corrupt", cli._chain_warning(d))
        self.assertIsNone(cli._chain_warning(self.run_dir("clean")))
        with mock.patch.object(cli.Ledger, "read_only", side_effect=PermissionError(13, "Permission denied")):
            self.assertIn("could not be read", cli._chain_warning(d))

    def test_live_serve_of_a_clean_chain_prints_no_warning(self):
        code, _, err, _ = self.serve(str(self.run_dir()), "--port", "0")
        self.assertEqual((code, err), (0, ""))


class DoctorTests(CliCase):
    def fake_api(self, **over):
        base = dict(available_isolation=lambda: "userns", run_smoke=None, build_client_factory=None, summarize=None,
                    ModelConfigError=ValueError)
        base.update(over)
        return SimpleNamespace(**base)

    def doctor(self, api, *argv):
        with mock.patch.object(cli, "_api", lambda: api):
            return self.cli("doctor", *argv)

    def test_reports_python_numpy_isolation_muse_and_keys(self):
        code, out, _ = self.doctor(self.fake_api())
        self.assertEqual(code, 0)
        self.assertIn(f"python: {sys.version.split()[0]}", out)
        self.assertRegex(out, r"numpy: (present|missing)")
        self.assertIn("isolation probe: userns", out)
        self.assertIn("MODEL_API_KEY: not set", out)
        self.assertIn("META_API_KEY: not set", out)

    def test_without_a_key_the_last_line_says_what_that_means(self):
        code, out, _ = self.doctor(self.fake_api())
        self.assertEqual(code, 0)
        self.assertEqual(out.rstrip("\n").splitlines()[-1],
                         "No API key: the offline demo works (python3 -m nightshift demo); live runs need MODEL_API_KEY, "
                         "see docs/RUNNING_LIVE.md.")

    def test_with_a_key_the_no_key_line_is_absent(self):
        for name in cli.KEY_ENVS:
            with mock.patch.dict(os.environ, {name: KEY_CANARY}):
                code, out, err = self.doctor(self.fake_api())
            self.assertEqual(code, 0)
            self.assertNotIn("No API key", out)
            self.assertNotIn(KEY_CANARY, out + err)

    def test_the_key_is_reported_present_but_never_printed(self):
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            code, out, err = self.doctor(self.fake_api())
        self.assertIn("MODEL_API_KEY: set", out)
        self.assertNotIn(KEY_CANARY, out + err)
        self.assertNotIn("sk-canary", out + err)

    def test_numpy_line_does_not_depend_on_the_environment(self):
        """``find_spec`` is mocked both ways: the result is the same on a machine with and without numpy."""
        with mock.patch.object(cli.importlib.util, "find_spec", return_value=None):
            code, out, _ = self.doctor(self.fake_api())
        self.assertEqual(code, 0)
        self.assertIn("numpy: missing (only the optional spiral task needs it)", out)
        with mock.patch.object(cli.importlib.util, "find_spec", return_value=object()), \
                mock.patch("importlib.metadata.version", return_value="9.8.7"):
            code, out, _ = self.doctor(self.fake_api())
        self.assertIn("numpy: present (9.8.7)", out)
        with mock.patch.object(cli.importlib.util, "find_spec", return_value=object()), \
                mock.patch("importlib.metadata.version", side_effect=RuntimeError("no metadata")):
            self.assertIn("numpy: present\n", self.doctor(self.fake_api())[1])

    def test_an_invalid_isolation_override_is_printed_with_the_allowed_values_and_exits_2(self):
        probed = []
        with mock.patch.dict(os.environ, {"NIGHTSHIFT_ISOLATION": "bogus"}):
            code, out, err = self.doctor(self.fake_api(available_isolation=lambda: probed.append(1) or "userns"), "--live")
        self.assertEqual(code, 2)
        self.assertIn("isolation probe: not run:", out)
        for word in ("NIGHTSHIFT_ISOLATION", "bogus", "auto", "userns", "unshare-net", "none"):
            self.assertIn(word, out)
        self.assertNotIn("probe failed", out)
        self.assertEqual(probed, [])                       # nothing was probed under a configuration that cannot run
        self.assertIn("MODEL_API_KEY", out)                # the rest of the report is still there
        self.assertNotIn("live smoke", out)

    def test_probe_failure_does_not_stop_the_report(self):
        def boom():
            raise RuntimeError("probe exploded")
        code, out, _ = self.doctor(self.fake_api(available_isolation=boom))
        self.assertEqual(code, 0)
        self.assertIn("probe failed (RuntimeError)", out)
        self.assertIn("MODEL_API_KEY", out)

    def make_muse(self, body: str) -> Path:
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(exist_ok=True)
        muse = bin_dir / "muse"
        muse.write_text(f"#!/bin/sh\n{body}\n")
        muse.chmod(muse.stat().st_mode | stat.S_IXUSR)
        return bin_dir

    def test_muse_on_path_is_run_with_version(self):
        bin_dir = self.make_muse('echo "muse 9.9.9"')
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}):
            code, out, _ = self.doctor(self.fake_api())
        self.assertIn("muse 9.9.9", out)
        self.assertIn("exit 0", out)

    def test_muse_version_timeout_is_handled(self):
        bin_dir = self.make_muse("sleep 30")
        real_run = subprocess.run

        def quick(*a, **kw):
            kw["timeout"] = 0.3
            return real_run(*a, **kw)
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}), \
                mock.patch.object(cli.subprocess, "run", quick):
            code, out, _ = self.doctor(self.fake_api())
        self.assertEqual(code, 0)
        self.assertIn("timed out", out)

    def test_muse_absent(self):
        with mock.patch.dict(os.environ, {"PATH": str(self.tmp)}):
            code, out, _ = self.doctor(self.fake_api())
        self.assertIn("muse binary: not on PATH", out)

    def test_live_runs_the_smoke_probe_and_maps_its_result_to_the_exit_code(self):
        calls = []

        def run_smoke(factory, out_path=None):
            calls.append((factory, out_path))
            return {"ok": calls[0][0] == "factory-ok", "probes": []}
        api = self.fake_api(run_smoke=run_smoke, build_client_factory=lambda: "factory-ok",
                            summarize=lambda r: ["PASS  tool_roundtrip", "overall: OK"])
        code, out, _ = self.doctor(api, "--live", "--live-out", "x.json")
        self.assertEqual(code, 0)
        self.assertIn("overall: OK", out)
        self.assertEqual(calls, [("factory-ok", "x.json")])
        calls.clear()
        api2 = self.fake_api(run_smoke=run_smoke, build_client_factory=lambda: "factory-bad", summarize=lambda r: ["overall: FAILED"])
        self.assertEqual(self.doctor(api2, "--live")[0], 1)

    @staticmethod
    def writing_smoke(report=None):
        """A ``run_smoke`` that, like the real one, writes its report to ``out_path`` and returns it."""
        report = {"ok": True, "probes": []} if report is None else report

        def run_smoke(factory, out_path=None):
            if out_path:
                Path(out_path).write_text(json.dumps(report))
            return dict(report)
        return run_smoke

    def live_api(self, report=None, **over):
        report = {"ok": True, "probes": []} if report is None else report
        return self.fake_api(run_smoke=over.pop("run_smoke", lambda f, o=None: dict(report)), build_client_factory=lambda: "f",
                             summarize=lambda r: ["overall: OK"], **over)

    def test_live_probes_the_muse_code_worker_and_records_it_in_the_probe_file(self):
        bin_dir = self.make_muse('echo "muse 1.2.3"')
        probe_file = self.tmp / "demo" / "live_probe.json"
        probe_file.parent.mkdir()
        seen = []

        def run_smoke(factory, out_path=None):
            Path(out_path).write_text(json.dumps({"ok": True, "probes": []}))       # what the real smoke run writes
            return {"ok": True, "probes": []}

        def probe_exec(timeout_sec=90.0):
            seen.append(timeout_sec)
            return {"ok": True, "exit_code": 0, "stderr_tail": "", "found_trust_stop": False, "usage_line_seen": True,
                    "model_reported": "muse-spark-1.3", "detail": "hello.txt written"}
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}), \
                mock.patch.object(cli, "_opt", lambda mod, name: probe_exec if name == "probe_exec" else None):
            code, out, _ = self.doctor(self.live_api(run_smoke=run_smoke), "--live", "--live-out", str(probe_file))
        self.assertEqual((code, seen), (0, [90.0]))
        self.assertLess(out.index("overall: OK"), out.index("muse code probe: PASS"))      # after the API probes
        self.assertIn("model reported muse-spark-1.3", out)
        self.assertIn("usage line seen yes", out)
        self.assertIn("trust stop no", out)
        saved = json.loads(probe_file.read_text())
        self.assertTrue(saved["ok"])
        self.assertEqual(saved["muse_code"]["ok"], True)
        self.assertEqual(saved["muse_code"]["model_reported"], "muse-spark-1.3")
        self.assertTrue(saved["muse_code"]["probed"])

    def test_a_failing_muse_code_probe_is_printed_and_recorded_but_does_not_change_the_api_verdict(self):
        bin_dir = self.make_muse('echo "muse 1.2.3"')
        probe = lambda timeout_sec=90.0: {"ok": False, "exit_code": 3, "stderr_tail": f"boom {KEY_CANARY}", "found_trust_stop": True,  # noqa: E731
                                          "usage_line_seen": False, "model_reported": None, "detail": "muse stopped on the trust message"}
        out_file = self.tmp / "probe.json"
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "MODEL_API_KEY": KEY_CANARY}), \
                mock.patch.object(cli, "_opt", lambda mod, name: probe if name == "probe_exec" else None):
            code, out, err = self.doctor(self.live_api(run_smoke=self.writing_smoke()), "--live", "--live-out", str(out_file))
        self.assertEqual(code, 0)
        self.assertIn("muse code probe: FAIL (exit 3", out)
        self.assertIn("trust stop yes", out)
        self.assertIn("trust message", out)
        self.assertNotIn(KEY_CANARY, out + err + out_file.read_text())
        self.assertFalse(json.loads(out_file.read_text())["muse_code"]["ok"])

    def test_a_probe_file_the_smoke_run_did_not_write_is_not_created_by_the_muse_section(self):
        bin_dir = self.make_muse('echo "muse 1.2.3"')
        target = self.tmp / "never-written.json"
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}), \
                mock.patch.object(cli, "_opt", lambda mod, name: None):
            code, _, _ = self.doctor(self.live_api(), "--live", "--live-out", str(target))
        self.assertEqual(code, 0)
        self.assertFalse(target.exists())

    def test_live_without_a_muse_binary_says_so_and_does_not_probe(self):
        called = []
        with mock.patch.dict(os.environ, {"PATH": str(self.tmp)}), \
                mock.patch.object(cli, "_opt", lambda mod, name: (lambda **kw: called.append(kw)) if name == "probe_exec" else None):
            code, out, _ = self.doctor(self.live_api(), "--live")
        self.assertEqual((code, called), (0, []))
        self.assertIn("muse binary: not on PATH, Muse Code worker not probed", out)

    def test_an_old_muse_code_module_without_probe_exec_only_prints_a_note(self):
        bin_dir = self.make_muse('echo "muse 1.2.3"')
        out_file = self.tmp / "probe.json"
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}), \
                mock.patch.object(cli, "_opt", lambda mod, name: None):
            code, out, _ = self.doctor(self.live_api(run_smoke=self.writing_smoke()), "--live", "--live-out", str(out_file))
        self.assertEqual(code, 0)
        self.assertIn("has no probe_exec", out)
        self.assertEqual(json.loads(out_file.read_text())["muse_code"], {"probed": False,
                                                                           "reason": "nightshift.agents.muse_code has no probe_exec"})

    def test_a_muse_code_probe_that_raises_is_reported_not_propagated(self):
        bin_dir = self.make_muse('echo "muse 1.2.3"')

        def probe(timeout_sec=90.0):
            raise RuntimeError("kaboom")
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}), \
                mock.patch.object(cli, "_opt", lambda mod, name: probe if name == "probe_exec" else None):
            code, out, _ = self.doctor(self.live_api(), "--live")
        self.assertEqual(code, 0)
        self.assertIn("probe raised RuntimeError", out)

    def test_live_without_a_key_is_a_setup_error(self):
        class NoKey(Exception):
            pass

        def factory():
            raise NoKey("no API key found")
        api = self.fake_api(run_smoke=lambda f, o=None: {"ok": True}, build_client_factory=factory, ModelConfigError=NoKey)
        code, out, _ = self.doctor(api, "--live")
        self.assertEqual(code, 2)
        self.assertIn("no API key", out)

    def test_live_with_the_module_missing(self):
        code, out, _ = self.doctor(self.fake_api(), "--live")
        self.assertEqual(code, 1)
        self.assertIn("not available", out)

    def test_doctor_with_the_real_api_does_not_crash(self):
        code, out, _ = self.cli("doctor")
        self.assertEqual(code, 0)
        self.assertIn("isolation probe:", out)


class TasksCommandTests(CliCase):
    def test_lists_and_validates_with_calibration(self):
        seen = []

        def fake_list(root, on_error=None):
            on_error(Path(root) / "broken", ValueError("bad toml"))
            return [SimpleNamespace(name="good", description="g"), SimpleNamespace(name="weak", description="w")]

        def fake_validate(task, calibrate=False):
            seen.append((task.name, calibrate))
            return [] if task.name == "good" else ["seeds_vacuous: all seeds score the same"]
        api = SimpleNamespace(list_tasks=fake_list, validate_task=fake_validate)
        with mock.patch.object(cli, "_api", lambda: api):
            code, out, _ = self.cli("tasks", "--root", str(self.tmp))
        self.assertEqual(code, 1)
        self.assertEqual(seen, [("good", True), ("weak", True)])
        self.assertIn("OK   good", out)
        self.assertIn("FAIL weak", out)
        self.assertIn("seeds_vacuous", out)
        self.assertIn("FAIL broken: bad toml", out)

    def test_all_good_is_exit_0_and_empty_is_exit_1(self):
        api = SimpleNamespace(list_tasks=lambda root, on_error=None: [SimpleNamespace(name="good", description="g")],
                              validate_task=lambda t, calibrate=False: [])
        with mock.patch.object(cli, "_api", lambda: api):
            self.assertEqual(self.cli("tasks", "--root", str(self.tmp))[0], 0)
        empty = SimpleNamespace(list_tasks=lambda root, on_error=None: [], validate_task=lambda t, calibrate=False: [])
        with mock.patch.object(cli, "_api", lambda: empty):
            code, out, _ = self.cli("tasks", "--root", str(self.tmp))
        self.assertEqual(code, 1)
        self.assertIn("no tasks found", out)

    def test_a_missing_optional_requirement_is_a_skip_not_a_failure(self):
        problem = "missing_requirements: numpy (calibration skipped)"
        api = SimpleNamespace(list_tasks=lambda root, on_error=None: [SimpleNamespace(name="spiral", description="s"),
                                                                      SimpleNamespace(name="good", description="g")],
                              validate_task=lambda t, calibrate=False: [problem] if t.name == "spiral" else [])
        with mock.patch.object(cli, "_api", lambda: api):
            code, out, _ = self.cli("tasks", "--root", str(self.tmp))
        self.assertEqual(code, 0, out)
        self.assertIn("SKIP spiral", out)
        self.assertIn(problem, out)
        self.assertIn("OK   good", out)
        self.assertNotIn("FAIL", out)

    def test_a_structural_problem_next_to_a_skip_still_fails(self):
        api = SimpleNamespace(list_tasks=lambda root, on_error=None: [SimpleNamespace(name="both", description="b")],
                              validate_task=lambda t, calibrate=False: ["missing_requirements: numpy (calibration skipped)",
                                                                        "bad_objective: 'sideways'"])
        with mock.patch.object(cli, "_api", lambda: api):
            code, out, _ = self.cli("tasks", "--root", str(self.tmp))
        self.assertEqual(code, 1)
        self.assertIn("FAIL both", out)
        self.assertIn("bad_objective", out)

    def test_real_loader_a_task_requiring_an_absent_module_is_skipped(self):
        root = self.tmp / "tasks"
        shutil.copytree(FIXTURE, root / "mini-curve", ignore=shutil.ignore_patterns("__pycache__"))
        toml = root / "mini-curve" / "task.toml"
        toml.write_text(toml.read_text().replace("requires = []", 'requires = ["no_such_module_xyz"]'))
        code, out, _ = self.cli("tasks", "--root", str(root))
        self.assertEqual(code, 0, out)
        self.assertIn("SKIP mini-curve", out)
        self.assertIn("no_such_module_xyz", out)

    def test_real_loader_a_structural_error_in_a_task_with_an_absent_module_still_fails(self):
        root = self.tmp / "tasks"
        shutil.copytree(FIXTURE, root / "mini-curve", ignore=shutil.ignore_patterns("__pycache__"))
        toml = root / "mini-curve" / "task.toml"
        toml.write_text(toml.read_text().replace("requires = []", 'requires = ["no_such_module_xyz"]')
                        .replace('objective = "minimize"', 'objective = "sideways"'))
        code, out, _ = self.cli("tasks", "--root", str(root))
        self.assertEqual(code, 1, out)

    def test_real_task_loader_and_calibration_on_the_fixture(self):
        root = self.tmp / "tasks"
        shutil.copytree(FIXTURE, root / "mini-curve", ignore=shutil.ignore_patterns("__pycache__"))
        code, out, _ = self.cli("tasks", "--root", str(root))
        self.assertEqual(code, 0, out)
        self.assertIn("OK   mini-curve", out)

    def test_a_task_that_fails_validation_is_reported_by_the_real_loader(self):
        root = self.tmp / "tasks"
        shutil.copytree(FIXTURE, root / "mini-curve", ignore=shutil.ignore_patterns("__pycache__"))
        toml = root / "mini-curve" / "task.toml"
        toml.write_text(toml.read_text().replace("seeds = [0, 1, 2]", "seeds = [0, 1]"))
        code, out, _ = self.cli("tasks", "--root", str(root))
        self.assertEqual(code, 1)
        self.assertIn("FAIL mini-curve", out)

    def test_missing_module_is_a_clear_error(self):
        with mock.patch.object(cli, "_api", lambda: SimpleNamespace(list_tasks=None, validate_task=None)):
            code, _, err = self.cli("tasks")
        self.assertEqual(code, 1)
        self.assertIn("list_tasks is not available", err)


class AuditCommandTests(CliCase):
    def lying_run(self, name: str = "run") -> Path:
        d = self.tmp / name
        led = Ledger.create(d, clock=FakeClock(), fsync=False)
        rb = RunBuilder(led)
        rb.start("mini-curve")
        rb.baseline(1.0)
        e = rb.experiment("A", "does nothing", score=1.0, delta=0.0, noise_margin=0.01)
        rb.claim(e, "improved", "big gain")                     # contradicted by the recorded no_gain: I01
        rb.promote(e, replicated=False)                          # and it is the incumbent, so the finding cannot be contained
        rb.finish()
        led.close()
        return d

    def test_audit_runs_on_a_copy_and_leaves_the_original_untouched(self):
        d = self.lying_run()
        before = (d / "ledger.jsonl").read_bytes()
        listing = sorted(p.name for p in d.iterdir())
        code, out, _ = self.cli("audit", str(d), "--task", str(FIXTURE))
        self.assertEqual(code, 1)
        self.assertIn("I01@e001.c1", out)
        self.assertIn("temporary copy", out)
        self.assertRegex(out, r"findings: \d+ total, \d+ new, [1-9]\d* open block")
        self.assertEqual((d / "ledger.jsonl").read_bytes(), before)
        self.assertEqual(sorted(p.name for p in d.iterdir()), listing)

    def test_audit_write_appends_to_the_real_ledger_and_restores_the_seal(self):
        d = self.lying_run()
        os.chmod(d / "ledger.jsonl", 0o444)
        before = len(Ledger.read_only(d).events())
        code, out, _ = self.cli("audit", str(d), "--task", str(FIXTURE), "--write")
        self.assertEqual(code, 1)
        self.assertIn("written to the ledger", out)
        view = Ledger.read_only(d).view()
        self.assertGreater(len(Ledger.read_only(d).events()), before)
        self.assertIn("I01@e001.c1", view.findings)
        self.assertEqual(stat.S_IMODE((d / "ledger.jsonl").stat().st_mode), 0o444)
        ok, bad = Ledger.read_only(d).verify_chain()
        self.assertTrue(ok, bad)

    def test_clean_run_has_zero_open_block_findings(self):
        d = self.tmp / "clean"
        led = Ledger.create(d, clock=FakeClock(), fsync=False)
        rb = RunBuilder(led)
        rb.start("mini-curve")
        rb.baseline(1.0)
        rb.experiment("A", "honest null", score=1.0, delta=0.0, noise_margin=0.01)
        rb.finish()
        led.close()
        code, out, _ = self.cli("audit", str(d), "--task", str(FIXTURE))
        self.assertEqual(code, 0, out)
        self.assertIn("0 open block", out)

    def test_audit_errors(self):
        self.assertEqual(self.cli("audit", str(self.tmp / "nope"))[0], 2)
        d = self.lying_run()
        code, _, err = self.cli("audit", str(d), "--task", str(self.tmp / "no-such-task"))
        self.assertEqual(code, 2)
        code, _, err = self.cli("audit", str(d), "--task", str(FIXTURE), "--judge", "meta")
        self.assertEqual(code, 2)
        self.assertIn("MODEL_API_KEY", err)

    def test_audit_write_refuses_a_ledger_that_a_run_is_still_writing(self):
        d = self.lying_run()
        holder = Ledger.open(d, fsync=False)
        self.addCleanup(holder.close)
        code, _, err = self.cli("audit", str(d), "--task", str(FIXTURE), "--write")
        self.assertEqual(code, 1)
        self.assertIn("still writing", err)

    def test_judge_meta_passes_a_judge_to_the_auditor(self):
        d = self.lying_run()
        captured = {}

        class FakeAuditor:
            def __init__(self, ledger, task, *, judge=None, **kw):
                captured["judge"] = judge

            def audit_window(self):
                return []

            def final_sweep(self):
                return []
        api = SimpleNamespace(load_task=lambda p: SimpleNamespace(name="mini-curve"), Auditor=FakeAuditor,
                              LLMJudge=lambda client: ("judge", client), load_routing=lambda path: {"r": 1},
                              client_for=lambda role, routing: (role, routing), ModelConfigError=ValueError)
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}), mock.patch.object(cli, "_api", lambda: api):
            code, out, err = self.cli("audit", str(d), "--task", str(FIXTURE), "--judge", "meta")
        self.assertEqual(code, 0)
        self.assertEqual(captured["judge"], ("judge", ("auditor_final", {"r": 1})))
        self.assertNotIn(KEY_CANARY, out + err)


class HelperTests(CliCase):
    def test_repo_root_from_a_subdirectory_and_from_outside(self):
        self.assertEqual(cli._repo_root(REPO / "nightshift" / "dashboard"), REPO)
        self.assertEqual(cli._repo_root(REPO), REPO)
        self.assertEqual(cli._repo_root(self.tmp), REPO)         # outside any checkout: the one this package lives in

    def test_repo_root_uses_the_cwd_by_default(self):
        old = os.getcwd()
        self.addCleanup(os.chdir, old)
        os.chdir(REPO / "tests")
        self.assertEqual(cli._repo_root(), REPO)

    def test_find_task_dir(self):
        repo = self.tmp / "repo"
        (repo / "tasks" / "alpha").mkdir(parents=True)
        (repo / "tasks" / "alpha" / "task.toml").write_text("x")
        self.assertEqual(cli._find_task_dir("alpha", repo), repo / "tasks" / "alpha")
        self.assertEqual(cli._find_task_dir("tasks/alpha", repo), repo / "tasks" / "alpha")
        with self.assertRaises(cli.CliError) as cm:
            cli._find_task_dir("beta", repo)
        self.assertEqual(cm.exception.code, 2)

    def test_unique_dir(self):
        d = self.tmp / "demo"
        self.assertEqual(cli._unique_dir(d), d)
        build_clean(d)
        self.assertEqual(cli._unique_dir(d), self.tmp / "demo-2")
        build_clean(self.tmp / "demo-2")
        self.assertEqual(cli._unique_dir(d), self.tmp / "demo-3")


# --------------------------------------------------------------------------------------------------
# commands that go through the orchestrator seam
# --------------------------------------------------------------------------------------------------
class SeamCase(CliCase):
    def setUp(self) -> None:
        super().setUp()
        self.calls: list = []
        self.repo = self.tmp / "repo"
        (self.repo / "tasks" / "law-discovery").mkdir(parents=True)
        (self.repo / "tasks" / "law-discovery" / "task.toml").write_text("name='law-discovery'")
        (self.repo / "tasks" / "fake-task").mkdir(parents=True)
        (self.repo / "tasks" / "fake-task" / "task.toml").write_text("name='fake-task'")

    def make_api(self, *, variant: str = "clean", result=None, on_run=None, crash=False, run_exc=None, create_exc=None,
                 resume_exc=None, no_orchestrator=False, control_variant: str | None = None):
        calls = self.calls
        case = self

        class Orch:
            def __init__(self, run_dir):
                self.run_dir = Path(run_dir)

            @classmethod
            def create(cls, run_dir, task, **kw):
                calls.append(("create", Path(run_dir), task, kw))
                if create_exc:
                    raise create_exc
                v = control_variant if (kw.get("control") and control_variant) else variant
                build_clean(run_dir, kw.get("backend", "scripted"), variant=v, task=task.name)
                return cls(run_dir)

            @classmethod
            def resume(cls, run_dir, task, **kw):
                calls.append(("resume", Path(run_dir), task, kw))
                if resume_exc:
                    raise resume_exc
                return cls(run_dir)

            def run(self):
                calls.append(("run", self.run_dir))
                if on_run:
                    on_run()
                if crash:
                    raise SimulatedCrash("boom")
                if run_exc:
                    raise run_exc
                return result if result is not None else {"run_id": "r1", "reason": "supervisor_finish", "verdict_class": "success",
                                                          "incumbent": "e001", "counts": {}, "head_anchor": "x"}

        def load_task(path):
            calls.append(("load_task", Path(path)))
            toml = Path(path) / "task.toml"
            if not toml.is_file():
                raise ValueError(f"{toml}: cannot read task.toml")
            return SimpleNamespace(name=Path(path).name)

        return SimpleNamespace(
            load_task=load_task, harden_parent=lambda: calls.append(("harden",)), Orchestrator=None if no_orchestrator else Orch,
            load_routing=lambda path: calls.append(("routing", path)) or {"routing": path}, ModelConfigError=ValueError,
            list_tasks=None, validate_task=None, available_isolation=None)

    def invoke(self, api, *argv: str, cwd: Path | None = None):
        patches = [mock.patch.object(cli, "_api", lambda: api), mock.patch.object(cli, "_repo_root", lambda start=None: self.repo)]
        old = os.getcwd()
        if cwd:
            os.chdir(cwd)
            self.addCleanup(os.chdir, old)
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return self.cli(*argv)

    def task_dir(self, name: str = "fake-task") -> str:
        return str(self.repo / "tasks" / name)

    def names(self):
        return [c[0] for c in self.calls]


class RunCommandTests(SeamCase):
    def test_harden_comes_first_then_create_then_run_and_the_anchor_is_printed(self):
        api = self.make_api()
        rd = self.tmp / "r1"
        code, out, _ = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(rd))
        self.assertEqual(code, 0)
        self.assertEqual([n for n in self.names() if n != "load_task"], ["harden", "create", "run"])
        anchor = format_anchor(Ledger.read_only(rd).events())
        self.assertIn(f"head anchor: {anchor}", out)
        self.assertIn("verdict=success", out)

    def test_options_reach_orchestrator_create(self):
        api = self.make_api()
        rd = self.tmp / "r1"
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):      # muse_code is a meta-backend implementer
            code, _, _ = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(rd), "--backend", "meta",
                                     "--max-experiments", "5", "--max-usd", "0.5", "--max-wall", "90", "--parallel", "2",
                                     "--implementer", "muse_code", "--control", "--config", "n.toml")
        self.assertEqual(code, 0)
        create = [c for c in self.calls if c[0] == "create"][0]
        _, run_dir, task, kw = create
        self.assertEqual(run_dir, rd)
        self.assertEqual(task.name, "fake-task")
        lim = kw["limits"]
        self.assertIsInstance(lim, BudgetLimits)
        self.assertEqual((lim.max_experiments, lim.max_llm_usd, lim.max_wall_sec, lim.max_parallel), (5, 0.5, 90.0, 2))
        self.assertEqual((kw["backend"], kw["implementer"], kw["control"]), ("meta", "muse_code", True))
        self.assertEqual(kw["routing"], {"routing": "n.toml"})
        self.assertNotIn("crash_after_runs", kw)

    def test_the_scripted_backend_refuses_the_muse_code_implementer_before_anything_happens(self):
        """The scripted suite has its own implementer and would ignore the flag while the ledger said muse_code."""
        api = self.make_api()
        for argv in (("--implementer", "muse_code"), ("--backend", "scripted", "--implementer", "muse_code")):
            with self.subTest(argv=argv):
                code, out, err = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"), *argv)
                self.assertEqual(code, 2)
                self.assertIn("muse_code", err)
                self.assertIn("--backend meta", err)
                self.assertEqual(out, "")
        self.assertEqual([n for n in self.names() if n in ("create", "run", "harden")], [])
        self.assertFalse((self.tmp / "r1").exists())

    def test_the_patch_implementer_is_fine_on_either_backend(self):
        api = self.make_api()
        code, _, _ = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"), "--implementer", "patch")
        self.assertEqual(code, 0)
        self.assertEqual([c for c in self.calls if c[0] == "create"][0][3]["implementer"], "patch")

    def test_defaults_do_not_override_the_budget_limits(self):
        api = self.make_api()
        self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"))
        kw = [c for c in self.calls if c[0] == "create"][0][3]
        self.assertEqual(kw["limits"].to_dict(), BudgetLimits().to_dict())
        self.assertEqual((kw["implementer"], kw["control"], kw["routing"]), ("patch", False, None))

    def test_hidden_crash_after_runs_is_passed_through(self):
        api = self.make_api()
        self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"), "--crash-after-runs", "3")
        self.assertEqual([c for c in self.calls if c[0] == "create"][0][3]["crash_after_runs"], 3)

    def test_nonpositive_limits_are_usage_errors(self):
        api = self.make_api()
        for flag, value in (("--max-experiments", "0"), ("--max-usd", "-1"), ("--max-wall", "0"), ("--parallel", "0")):
            code, _, err = self.invoke(api, "run", "--task", self.task_dir(), flag, value)
            self.assertEqual(code, 2, flag)
            self.assertIn("positive", err)
        self.assertNotIn("create", self.names())

    def test_api_keys_are_registered_and_never_printed(self):
        api = self.make_api()
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            code, out, err = self.invoke(api, "run", "--task", self.task_dir(), "--backend", "meta", "--run-dir", str(self.tmp / "r1"))
        self.assertEqual(code, 0)
        self.assertEqual(secrets.scrub_text(f"x {KEY_CANARY} y"), "x [REDACTED] y")
        self.assertNotIn(KEY_CANARY, out + err)

    def test_meta_backend_without_a_key_is_a_usage_error_and_creates_nothing(self):
        api = self.make_api()
        code, _, err = self.invoke(api, "run", "--task", self.task_dir(), "--backend", "meta")
        self.assertEqual(code, 2)
        self.assertIn("MODEL_API_KEY", err)
        self.assertEqual(self.names().count("create"), 0)
        self.assertEqual(self.names()[0], "harden")

    def test_bad_task_is_a_usage_error(self):
        api = self.make_api()
        code, _, err = self.invoke(api, "run", "--task", "no/such/task")
        self.assertEqual(code, 2)
        self.assertIn("no task found", err)

    def test_missing_orchestrator_is_a_clear_error(self):
        api = self.make_api(no_orchestrator=True)
        code, _, err = self.invoke(api, "run", "--task", self.task_dir())
        self.assertEqual(code, 1)
        self.assertIn("Orchestrator is not available", err)

    def test_existing_run_dir_points_to_resume(self):
        api = self.make_api(create_exc=FileExistsError("x"))
        code, _, err = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"))
        self.assertEqual(code, 2)
        self.assertIn("resume", err)

    def test_default_run_dir_is_under_runs_in_the_cwd(self):
        api = self.make_api()
        work = self.tmp / "work"
        work.mkdir()
        code, _, _ = self.invoke(api, "run", "--task", self.task_dir(), cwd=work)
        self.assertEqual(code, 0)
        run_dir = [c for c in self.calls if c[0] == "create"][0][1]
        self.assertEqual(run_dir.parent, Path("runs"))
        self.assertTrue(run_dir.name.startswith("fake-task-"))

    def test_exit_status_rules(self):
        for variant, expected in (("clean", 0), ("open_block", 1), ("no_incumbent", 1), ("defects", 1)):
            self.calls.clear()
            api = self.make_api(variant=variant, result={"run_id": "r", "reason": "x", "verdict_class": None, "incumbent": None})
            code, _, _ = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / f"r-{variant}"))
            self.assertEqual(code, expected, variant)

    def test_a_run_stopped_by_an_error_exits_nonzero_even_when_the_verdict_is_missing_authority(self):
        # a refused API key ends the run with reason 'error' and verdict missing_authority; a script must not read that as success
        api = self.make_api(variant="error_stop", result={"run_id": "r", "reason": "error", "verdict_class": "missing_authority",
                                                          "incumbent": "e000"})
        code, _, _ = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r-err"))
        self.assertEqual(code, 1)

    def test_budget_exhausted_with_an_incumbent_is_still_success(self):
        d = self.tmp / "r1"

        def finish_budget():
            pass
        api = self.make_api(variant="clean", result={"run_id": "r", "reason": "budget_exhausted", "verdict_class": "missing_authority",
                                                    "incumbent": "e001"})
        code, _, _ = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(d))
        self.assertEqual(code, 0)

    def test_simulated_crash_kills_the_process_with_137_and_cleans_up_nothing(self):
        api = self.make_api(crash=True)

        class Exited(BaseException):
            pass

        def fake_exit(code):
            raise Exited(code)
        with mock.patch.object(cli.os, "_exit", fake_exit):
            with self.assertRaises(Exited) as cm:
                self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"))
        self.assertEqual(cm.exception.args[0], 137)

    def test_a_real_process_dies_with_exit_137_and_keeps_what_was_written(self):
        run_dir = self.tmp / "crashrun"
        script = textwrap.dedent(f"""
            import sys
            from pathlib import Path
            from types import SimpleNamespace
            sys.path.insert(0, {str(REPO)!r})
            from nightshift import cli
            from nightshift.types import SimulatedCrash
            from tests.test_cli import build_clean
            class Orch:
                @classmethod
                def create(cls, run_dir, task, **kw):
                    assert kw.get("crash_after_runs") == 2
                    build_clean(run_dir, "scripted", variant="no_incumbent")
                    return cls()
                def run(self):
                    raise SimulatedCrash("kill -9")
            api = SimpleNamespace(load_task=lambda p: SimpleNamespace(name="fake-task"), harden_parent=None, Orchestrator=Orch,
                                  ModelConfigError=ValueError)
            cli._api = lambda: api
            sys.exit(cli.main(["run", "--task", {self.task_dir()!r}, "--run-dir", {str(run_dir)!r}, "--crash-after-runs", "2"]))
        """)
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, cwd=self.tmp)
        self.assertEqual(proc.returncode, 137, proc.stderr)
        ok, _ = Ledger.read_only(run_dir).verify_chain()
        self.assertTrue(ok)

    def test_keyboard_interrupt_does_not_promise_a_resume_that_a_finished_run_would_refuse(self):
        api = self.make_api(run_exc=KeyboardInterrupt())
        code, _, err = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r0"))
        self.assertEqual(code, 130)
        self.assertNotIn("nightshift resume", err)
        self.assertIn("start a new run", err)
        self.assertIn(f"nightshift report {self.tmp / 'r0'}", err)
        self.assertIn("cannot be resumed", err)

    def test_keyboard_interrupt_and_run_failure(self):
        api = self.make_api(run_exc=KeyboardInterrupt())
        self.assertEqual(self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"))[0], 130)
        api = self.make_api(run_exc=NightshiftError("lane exploded"))
        code, _, err = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r2"))
        self.assertEqual(code, 1)
        self.assertIn("lane exploded", err)

    def test_serve_flag_serves_the_live_ledger_during_the_run_and_holds_afterwards(self):
        held = []
        fetched = {}

        def during_run():
            import http.client
            import re
            url = re.search(r"dashboard: http://127\.0\.0\.1:(\d+)/", sys.stdout.getvalue()).group(1)
            conn = http.client.HTTPConnection("127.0.0.1", int(url), timeout=5)
            conn.request("GET", "/api/health")
            fetched["status"] = conn.getresponse().status
            conn.close()
        api = self.make_api(on_run=during_run)
        with mock.patch.object(cli, "_hold", lambda server: held.append(server)):
            code, out, _ = self.invoke(api, "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"), "--serve", "--port", "0")
        self.assertEqual(code, 0)
        self.assertEqual(fetched["status"], 200)
        self.assertEqual(len(held), 1)
        with self.assertRaises(OSError):                       # the server was shut down on the way out
            import socket
            socket.create_connection(("127.0.0.1", held[0].server_address[1]), timeout=1).close()


class ResumeCommandTests(SeamCase):
    def test_resume_finds_the_task_from_the_ledger_and_defaults_the_backend(self):
        d = self.tmp / "r1"
        build_clean(d, "scripted", variant="clean")
        api = self.make_api()
        code, out, _ = self.invoke(api, "resume", str(d))
        self.assertEqual(code, 0)
        self.assertEqual(self.names()[0], "harden")
        _, run_dir, task, kw = [c for c in self.calls if c[0] == "resume"][0]
        self.assertEqual((run_dir, task.name), (d, "fake-task"))
        self.assertEqual((kw["backend"], kw["accept_broken_chain"]), ("scripted", False))
        self.assertNotIn("accept_changed_task", kw)
        self.assertNotIn("crash_after_runs", kw)
        self.assertIn("head anchor:", out)
        self.assertEqual(self.names().count("create"), 0)

    def test_flags_are_forwarded(self):
        d = self.tmp / "r1"
        build_clean(d, "meta")
        api = self.make_api()
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            code, _, _ = self.invoke(api, "resume", str(d), "--accept-broken-chain", "--task", self.task_dir())
        self.assertEqual(code, 0)
        kw = [c for c in self.calls if c[0] == "resume"][0][3]
        self.assertEqual((kw["backend"], kw["accept_broken_chain"]), ("meta", True))
        self.assertNotIn("accept_changed_task", kw)

    def test_resuming_a_scripted_run_with_muse_code_is_refused(self):
        d = self.tmp / "r1"
        build_clean(d, "scripted")
        api = self.make_api()
        code, _, err = self.invoke(api, "resume", str(d), "--implementer", "muse_code", "--task", self.task_dir())
        self.assertEqual(code, 2)
        self.assertIn("--backend meta", err)
        self.assertEqual(self.names().count("resume"), 0)

    def test_resuming_a_meta_run_with_muse_code_is_allowed(self):
        d = self.tmp / "r1"
        build_clean(d, "meta")
        api = self.make_api()
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            code, _, _ = self.invoke(api, "resume", str(d), "--implementer", "muse_code", "--task", self.task_dir())
        self.assertEqual(code, 0)
        self.assertEqual([c for c in self.calls if c[0] == "resume"][0][3]["implementer"], "muse_code")

    def test_accepting_a_changed_task_is_not_an_option(self):
        d = self.tmp / "r1"
        build_clean(d, "scripted")
        api = self.make_api()
        code, _, err = self.invoke(api, "resume", str(d), "--accept-changed-task")
        self.assertEqual(code, 2)
        self.assertIn("unrecognized arguments", err)
        self.assertEqual(self.names(), [])                       # nothing ran
        for argv in (["resume", "--help"], ["--help"]):
            _, out, _ = self.cli(*argv)
            self.assertNotIn("changed-task", out)
            self.assertNotIn("changed task", out)
        _, out, _ = self.cli("resume", "--help")
        self.assertIn("--accept-broken-chain", out)
        self.assertNotIn("accept_changed_task", inspect.getsource(cli))
        self.assertNotIn("accept-changed-task", inspect.getsource(cli))

    def test_a_task_that_changed_is_refused_with_a_message_that_says_to_start_a_new_run(self):
        d = self.tmp / "r1"
        build_clean(d, "scripted")
        api = self.make_api(resume_exc=ResumeRefused("the task or grader changed since this run started; start a new run"))
        code, _, err = self.invoke(api, "resume", str(d))
        self.assertEqual(code, 1)
        self.assertIn("start a new run", err)
        self.assertNotIn("accept", err)

    def test_resume_errors(self):
        api = self.make_api()
        self.assertEqual(self.invoke(api, "resume", str(self.tmp / "nope"))[0], 2)
        d = self.tmp / "r1"
        build_clean(d, task="a-task-nobody-has")
        code, _, err = self.invoke(api, "resume", str(d))
        self.assertEqual(code, 2)
        self.assertIn("a-task-nobody-has", err)
        d2 = self.tmp / "r2"
        build_clean(d2)
        api2 = self.make_api(resume_exc=NightshiftError("the hash chain is broken; pass --accept-broken-chain"))
        code, _, err = self.invoke(api2, "resume", str(d2))
        self.assertEqual(code, 1)
        self.assertIn("accept-broken-chain", err)


class DemoCommandTests(SeamCase):
    def demo(self, api, *argv, **kw):
        held = []
        with mock.patch.object(cli, "_hold", lambda server: held.append(server)):
            result = self.invoke(api, "demo", *argv, **kw)
        return (*result, held)

    def test_default_demo_runs_law_discovery_scripted_from_any_cwd_and_writes_reports(self):
        api = self.make_api()
        elsewhere = self.tmp / "somewhere" / "deep"
        elsewhere.mkdir(parents=True)
        code, out, _, held = self.demo(api, cwd=elsewhere)
        self.assertEqual(code, 0, out)
        _, run_dir, task, kw = [c for c in self.calls if c[0] == "create"][0]
        self.assertEqual(task.name, "law-discovery")
        self.assertEqual(run_dir, elsewhere / "runs" / "demo")      # the caller's cwd; the repo only locates the tasks
        self.assertEqual((kw["backend"], kw["control"]), ("scripted", False))
        self.assertTrue((run_dir / "report.md").is_file() and (run_dir / "report.html").is_file())
        self.assertIn("scripted backend (no language model)", out)
        self.assertIn("head anchor:", out)
        self.assertEqual(held, [])
        self.assertFalse((self.repo / "runs").exists())

    def test_existing_demo_dir_is_not_overwritten(self):
        build_clean(self.repo / "runs" / "demo")
        api = self.make_api()
        code, out, _, _ = self.demo(api, cwd=self.repo)
        self.assertEqual(code, 0)
        self.assertEqual([c for c in self.calls if c[0] == "create"][0][1], self.repo / "runs" / "demo-2")
        self.assertIn(f"the directory {self.repo / 'runs' / 'demo'} already existed, using {self.repo / 'runs' / 'demo-2'}", out)

    def test_a_fresh_demo_dir_prints_no_already_existed_message(self):
        code, out, _, _ = self.demo(self.make_api(), cwd=self.repo)
        self.assertEqual(code, 0)
        self.assertNotIn("already existed", out)

    def test_the_head_anchor_is_printed_once_per_run(self):
        code, out, _, _ = self.demo(self.make_api(), cwd=self.repo)
        self.assertEqual(out.count("head anchor:"), 1, out)
        code, out, _, _ = self.demo(self.make_api(), "--control", "--out", str(self.tmp / "twice"))
        self.assertEqual(out.count("head anchor:"), 2, out)       # the demo run and the control run, one each

    def test_task_and_out_options(self):
        api = self.make_api()
        code, _, _, _ = self.demo(api, "--task", "fake-task", "--out", str(self.tmp / "mine"))
        self.assertEqual(code, 0)
        _, run_dir, task, _ = [c for c in self.calls if c[0] == "create"][0]
        self.assertEqual((run_dir, task.name), (self.tmp / "mine", "fake-task"))
        self.assertEqual(self.demo(api, "--task", "nonexistent")[0], 2)

    def test_serve_starts_the_dashboard_and_holds(self):
        api = self.make_api()
        code, out, _, held = self.demo(api, "--serve", "--port", "0")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"dashboard: http://127\.0\.0\.1:\d+/")
        self.assertEqual(len(held), 1)
        code, out, _, held = self.demo(api, "--serve", "--no-serve", "--port", "0", "--out", str(self.tmp / "other"))
        self.assertNotIn("dashboard:", out)
        self.assertEqual(held, [])

    def test_control_runs_a_second_honest_run_that_must_have_zero_findings(self):
        api = self.make_api()
        code, out, _, _ = self.demo(api, "--control", cwd=self.repo)
        self.assertEqual(code, 0, out)
        creates = [c for c in self.calls if c[0] == "create"]
        self.assertEqual([c[3]["control"] for c in creates], [False, True])
        self.assertEqual(creates[1][1], self.repo / "runs" / "demo-control")
        self.assertIn(f"negative control run dir: {self.repo / 'runs' / 'demo-control'}", out)
        self.assertNotIn("already existed", out)
        self.assertIn("negative control findings: 0 (expected 0)", out)
        self.assertTrue((creates[1][1] / "report.md").is_file())

    def test_an_existing_control_dir_is_not_overwritten_and_the_message_says_so(self):
        build_clean(self.repo / "runs" / "demo-control")
        code, out, _, _ = self.demo(self.make_api(), "--control", cwd=self.repo)
        self.assertEqual(code, 0, out)
        creates = [c for c in self.calls if c[0] == "create"]
        self.assertEqual(creates[1][1], self.repo / "runs" / "demo-control-2")
        self.assertIn(f"the directory {self.repo / 'runs' / 'demo-control'} already existed, using "
                      f"{self.repo / 'runs' / 'demo-control-2'}", out)

    def test_a_control_run_with_findings_fails_the_demo(self):
        api = self.make_api(control_variant="open_block")
        code, out, _, _ = self.demo(api, "--control")
        self.assertEqual(code, 1)
        self.assertIn("negative control findings: 1 (expected 0)", out)

    def test_with_crash_runs_a_child_that_dies_then_resumes_in_a_second_process(self):
        children = []

        def fake_child(argv, cwd):
            children.append((list(argv), cwd))
            return 137 if len(children) == 1 else 0
        api = self.make_api()
        with mock.patch.object(cli, "_run_child", fake_child):
            code, out, _, _ = self.demo(api, "--with-crash", cwd=self.repo)
        self.assertEqual(code, 0)
        self.assertEqual(len(children), 2)
        first, second = children[0][0], children[1][0]
        task_dir = str(self.repo / "tasks" / "law-discovery")
        out_dir = str(self.repo / "runs" / "demo")
        self.assertEqual(first, ["run", "--task", task_dir, "--backend", "scripted", "--run-dir", out_dir,
                                 "--crash-after-runs", str(cli.DEMO_CRASH_AFTER_RUNS)])
        self.assertEqual(second, ["resume", out_dir, "--task", task_dir])
        self.assertEqual(children[0][1], self.repo)
        self.assertNotIn("create", self.names())                  # the parent never runs the orchestrator itself
        self.assertIn("killed mid-experiment (exit 137)", out)

    def test_with_crash_where_the_first_process_does_not_crash_skips_the_resume(self):
        children = []
        api = self.make_api()
        with mock.patch.object(cli, "_run_child", lambda argv, cwd: children.append(argv) or 1):
            code, out, _, _ = self.demo(api, "--with-crash")
        self.assertEqual((code, len(children)), (1, 1))
        self.assertIn("instead of crashing", out)

    def test_run_child_launches_python_dash_m_nightshift_from_the_repo(self):
        seen = {}

        def fake_run(cmd, cwd=None, env=None, check=None):
            seen.update(cmd=cmd, cwd=cwd, env=env)
            return SimpleNamespace(returncode=137)
        with mock.patch.object(cli.subprocess, "run", fake_run):
            rc = cli._run_child(["run", "--task", "t"], self.repo)
        self.assertEqual(rc, 137)
        self.assertEqual(seen["cmd"], [sys.executable, "-m", "nightshift", "run", "--task", "t"])
        self.assertEqual(seen["cwd"], self.repo)
        self.assertTrue(seen["env"]["PYTHONPATH"].startswith(str(self.repo)))

    def test_replay_serves_an_existing_ledger_read_only(self):
        d = self.tmp / "old"
        build_clean(d)
        before = (d / "ledger.jsonl").read_bytes()
        states = []

        def hold(server):
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            conn.request("GET", "/api/state")
            states.append(json.loads(conn.getresponse().read()))
            conn.close()
        api = self.make_api()
        with mock.patch.object(cli, "_hold", hold):
            code, out, _ = self.invoke(api, "demo", "--replay", str(d), "--port", "0", "--replay-speed", "50")
        self.assertEqual(code, 0)
        self.assertEqual(states[0]["run"]["mode"], "replay")
        self.assertEqual(states[0]["replay"]["speed"], 50.0)
        self.assertEqual(self.names(), [])                          # nothing was run
        self.assertEqual((d / "ledger.jsonl").read_bytes(), before)
        self.assertIn("replaying", out)
        self.assertEqual(self.invoke(api, "demo", "--replay", str(self.tmp / "nope"))[0], 2)

    def test_replay_of_a_broken_ledger_is_refused(self):
        d = self.tmp / "old"
        build_clean(d)
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)
        lines = path.read_text().splitlines()
        row = json.loads(lines[4])
        row["data"]["x"] = 1
        lines[4] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        api = self.make_api()
        with mock.patch.object(cli, "_hold", lambda s: None):
            code, _, err = self.invoke(api, "demo", "--replay", str(d), "--port", "0")
        self.assertEqual(code, 1)
        self.assertIn("broken", err)


class AdversarialCliTests(CliCase):
    def corrupt_run(self) -> Path:
        d = self.run_dir()
        path = d / "ledger.jsonl"
        os.chmod(path, 0o644)
        lines = path.read_text().splitlines()
        lines[3] = "{garbage"
        path.write_text("\n".join(lines) + "\n")
        return d

    def test_a_corrupt_ledger_is_a_message_and_exit_1_for_every_command(self):
        d = self.corrupt_run()
        api = SimpleNamespace(load_task=lambda p: SimpleNamespace(name="fake-task"), harden_parent=None, Orchestrator=None,
                              Auditor=object, ModelConfigError=ValueError)
        with mock.patch.object(cli, "_api", lambda: api), \
                mock.patch.object(cli, "_repo_root", lambda start=None: self.tmp):
            (self.tmp / "tasks" / "fake-task").mkdir(parents=True)
            (self.tmp / "tasks" / "fake-task" / "task.toml").write_text("x")
            for argv in (["audit", str(d), "--task", str(FIXTURE)], ["resume", str(d)], ["verify", str(d)], ["report", str(d)],
                         ["serve", str(d), "--port", "0"], ["serve", str(d), "--port", "0", "--replay", "5"],
                         ["demo", "--replay", str(d), "--port", "0"]):
                code, out, err = self.cli(*argv)      # a traceback would escape as an exception here
                self.assertEqual(code, 1, argv)
                self.assertNotIn("Traceback", out + err)
                self.assertTrue((out + err).strip(), argv)

    def test_a_nightshift_error_escaping_a_command_is_reported_not_raised(self):
        d = self.run_dir()

        def boom(args):
            raise NightshiftError("the thing the library refused")
        parser_default = cli.build_parser().parse_args(["verify", str(d)]).func
        self.assertIs(parser_default, cli.cmd_verify)
        with mock.patch.object(cli, "cmd_verify", boom):
            code, _, err = self.cli("verify", str(d))
        self.assertEqual(code, 1)
        self.assertIn("the thing the library refused", err)
        self.assertNotIn("Traceback", err)

    def test_a_ledger_locked_by_a_writer_is_a_message_for_audit_write(self):
        d = self.run_dir()
        holder = Ledger.open(d, fsync=False)
        self.addCleanup(holder.close)
        code, _, err = self.cli("audit", str(d), "--task", str(FIXTURE), "--write")
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)


class PlantAndContextTests(SeamCase):
    """``run --plant`` (scripted red-team lanes in a meta run) and ``run --context`` (background material)."""

    LAW = str(REPO / "tasks" / "law-discovery")
    SPIRAL = str(REPO / "tasks" / "spiral-classifier")

    def real_api(self):
        from nightshift.task import load_task
        api = self.make_api()
        api.load_task = load_task                       # the real loader: --plant reads the task's scripted bank
        return api

    def run_meta(self, api, *argv: str, rd: str = "r1"):
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            return self.invoke(api, "run", "--backend", "meta", "--run-dir", str(self.tmp / rd), *argv)

    def create_kw(self):
        return [c for c in self.calls if c[0] == "create"][0][3]

    def write(self, name: str, text: str | bytes) -> str:
        path = self.tmp / name
        path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
        return str(path)

    def assert_refused_before_anything(self, code, err, *needles, hardened=False):
        """Exit 2 with every needle in the message, and no run created or started. ``hardened``: the check needs the loaded task,
        so the process was hardened first (as it is before any task is read)."""
        self.assertEqual(code, 2, err)
        for needle in needles:
            self.assertIn(needle, err)
        self.assertEqual([n for n in self.names() if n in ("harden", "create", "run")], ["harden"] if hardened else [])

    # ---- --plant ------------------------------------------------------------------------------
    def test_plant_with_the_scripted_backend_is_a_usage_error(self):
        for argv in (("--plant", "tamper"), ("--backend", "scripted", "--plant", "tamper,overfit")):
            with self.subTest(argv=argv):
                code, out, err = self.invoke(self.real_api(), "run", "--task", self.LAW, "--run-dir", str(self.tmp / "r1"), *argv)
                self.assert_refused_before_anything(code, err, "--plant", "--backend meta")
        self.assertFalse((self.tmp / "r1").exists())

    def test_bad_plant_names_are_usage_errors(self):
        cases = {"unknown": ("nonsense", "nonsense"), "not a fault": ("normal", "normal"), "empty name": ("tamper,,overfit", "comma"),
                 "empty": ("", "comma"), "repeated": ("tamper,tamper", "more than once"),
                 "repeated across flags": None}
        for label, case in cases.items():
            with self.subTest(label):
                argv = ["--plant", "tamper", "--plant", "tamper"] if case is None else ["--plant", case[0]]
                needle = "more than once" if case is None else case[1]
                code, _, err = self.run_meta(self.real_api(), "--task", self.LAW, *argv)
                self.assert_refused_before_anything(code, err, needle)

    def test_a_behavior_the_tasks_bank_lacks_is_a_usage_error_naming_what_it_has(self):
        code, _, err = self.run_meta(self.real_api(), "--task", self.SPIRAL, "--plant", "tamper,hang")
        self.assert_refused_before_anything(code, err, "hang", "tamper", hardened=True)

    def test_plant_reaches_orchestrator_create_in_the_order_given_and_is_announced(self):
        code, out, _ = self.run_meta(self.real_api(), "--task", self.LAW, "--plant", "tamper,overfit", "--plant", "claim_without_run")
        self.assertEqual(code, 0)
        self.assertEqual(self.create_kw()["plant"], ["tamper", "overfit", "claim_without_run"])
        self.assertNotIn("background", self.create_kw())
        self.assertIn("red team: tamper (tamper), overfit (poly9), claim_without_run (claim_without_run)", out)
        self.assertIn("scripted attacker lanes", out)
        self.assertIn("real model calls", out)

    def test_a_plain_run_passes_neither_plant_nor_background(self):
        self.run_meta(self.real_api(), "--task", self.LAW)
        self.assertEqual(set(self.create_kw()) & {"plant", "background", "context"}, set())

    def test_plant_has_help_text_and_is_not_a_resume_option(self):
        _, out, _ = self.cli("run", "--help")
        for flag in ("--plant", "--context"):
            self.assertIn(flag, out)
        self.assertIn("tamper", out)
        d = self.run_dir()
        code, _, err = self.invoke(self.make_api(), "resume", str(d), "--plant", "tamper")
        self.assertEqual(code, 2)
        self.assertIn("unrecognized arguments", err)
        self.assertIn("--context", self.cli("resume", "--help")[1])        # --context IS a resume option (same files, checked by sha256)

    # ---- --context ----------------------------------------------------------------------------
    def test_context_with_the_scripted_backend_is_a_usage_error(self):
        f = self.write("notes.md", "related work")
        code, out, err = self.invoke(self.make_api(), "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"), "--context", f)
        self.assert_refused_before_anything(code, err, "--context", "--backend meta")
        self.assertEqual(out, "")

    def test_unusable_context_files_are_usage_errors_with_a_clear_message(self):
        big = "x" * (cli.CONTEXT_FILE_MAX_CHARS + 1)
        cases = {
            "missing": (str(self.tmp / "nope.md"), "does not exist"),
            "directory": (str(self.tmp), "not a file"),
            "binary": (self.write("a.bin", b"\x89PNG\r\n\x1a\n\x00\x00\xff\xfe"), "not UTF-8"),
            "nul": (self.write("nul.txt", "text with a \x00 inside"), "control characters"),
            "escape": (self.write("esc.txt", "colour \x1b[31mred"), "control characters"),
            "empty": (self.write("empty.md", ""), "empty"),
            "blank": (self.write("blank.md", " \n\t\n"), "empty"),
            "too long": (self.write("big.txt", big), "at most"),
            "far too long": (self.write("huge.txt", b"y" * (4 * cli.CONTEXT_FILE_MAX_CHARS + 1)), "larger than"),
        }
        for label, (path, needle) in cases.items():
            with self.subTest(label):
                code, _, err = self.run_meta(self.make_api(), "--task", self.task_dir(), "--context", path)
                self.assert_refused_before_anything(code, err, "--context", needle)

    def test_paths_the_filesystem_cannot_resolve_are_usage_errors_not_tracebacks(self):
        """Verifier regression: ``~nosuchuser/x`` made ``expanduser`` raise RuntimeError, and a name longer than the filesystem
        allows made ``exists`` raise OSError (ENAMETOOLONG); both escaped as a traceback instead of exit 2."""
        loop_a, loop_b = self.tmp / "loop_a", self.tmp / "loop_b"
        loop_a.symlink_to(loop_b)
        loop_b.symlink_to(loop_a)
        cases = {"unknown home": ("~no-such-user-xyz/notes.md", "home directory"), "name too long": ("n" * 400 + ".md", "cannot use"),
                 "symlink loop": (str(loop_a), "--context"), "file as a directory": (self.write("plain.md", "text") + "/inner.md", "does not exist")}
        for label, (path, needle) in cases.items():
            with self.subTest(label):
                code, _, err = self.run_meta(self.make_api(), "--task", self.task_dir(), "--context", path)
                self.assert_refused_before_anything(code, err, "--context", needle)
                self.assertNotIn("Traceback", err)

    def test_a_file_that_is_listed_twice_and_a_total_over_the_cap_are_refused(self):
        f = self.write("a.md", "some notes")
        code, _, err = self.run_meta(self.make_api(), "--task", self.task_dir(), "--context", f, "--context", f)
        self.assert_refused_before_anything(code, err, "more than once")
        a, b, c = (self.write(n, "z" * cli.CONTEXT_FILE_MAX_CHARS) for n in ("p.md", "q.md", "r.md"))
        code, _, err = self.run_meta(self.make_api(), "--task", self.task_dir(), "--context", a, b)       # exactly the cap: fine
        self.assertEqual(code, 0)
        self.calls.clear()
        code, _, err = self.run_meta(self.make_api(), "--task", self.task_dir(), "--context", a, b, c, rd="r2")
        self.assert_refused_before_anything(code, err, "together exceed")

    def test_context_files_become_one_background_text_and_ledger_records_only_hashes(self):
        notes = "Damped oscillators: see Smith 2020.\nUse a sinusoid with decay.\n"
        paper = "Abstract. We show that extrapolation needs the right functional form.\n"
        f1, f2, f3 = self.write("notes.md", notes), self.write("paper.txt", paper), self.write("more.md", "\ufeffA note with a BOM and é\n")
        sub = self.tmp / "sub"
        sub.mkdir()
        (sub / "deep.md").write_text("in a sub directory")
        code, out, err = self.run_meta(self.make_api(), "--task", self.task_dir(), "--context", f1, f2, "--context", f3,
                                       str(sub / "deep.md"))
        self.assertEqual(code, 0, err)
        kw = self.create_kw()
        import hashlib
        self.assertEqual([c["name"] for c in kw["context"]], ["notes.md", "paper.txt", "more.md", "deep.md"])
        self.assertEqual(kw["context"][0], {"name": "notes.md", "sha256": hashlib.sha256(notes.encode()).hexdigest(), "chars": len(notes)})
        self.assertEqual(kw["context"][2]["chars"], len("A note with a BOM and é\n"))          # the BOM is not text
        self.assertEqual(kw["context"][2]["sha256"], hashlib.sha256(Path(f3).read_bytes()).hexdigest())   # the file's own bytes
        for text in (notes.strip(), paper.strip(), "A note with a BOM and é", "in a sub directory"):
            self.assertIn(text, kw["background"])
        self.assertLess(kw["background"].index("notes.md"), kw["background"].index("paper.txt"))      # in the order given
        self.assertNotIn(str(self.tmp), json.dumps(kw["context"]))                                   # basenames only
        self.assertIn("context: 4 file(s): notes.md", out)

    def test_a_background_longer_than_the_roles_use_is_said_so_in_plain_words(self):
        sup, des = 60_000, 8_000
        cases = {"fits": ("a" * 500, None), "designer only": ("b" * (des + 500), "designer reads only"),
                 "both": ("c" * (sup + 500), "only its first 60,000 characters reach the supervisor")}
        for label, (text, needle) in cases.items():
            with self.subTest(label):
                self.calls.clear()
                code, out, _ = self.run_meta(self.make_api(), "--task", self.task_dir(), "--context", self.write(label.replace(" ", "_") + ".md", text),
                                             rd="r-" + label.replace(" ", "_"))
                self.assertEqual(code, 0)
                notes = [l for l in out.splitlines() if l.startswith("note:")]
                if needle is None:
                    self.assertEqual(notes, [])
                else:
                    self.assertEqual(len(notes), 1)
                    self.assertIn(needle, notes[0])
                self.assertIn("characters;", out)

    def test_the_text_reaches_build_suite_through_the_real_orchestrator_and_only_hashes_reach_the_ledger(self):
        """A fake build_suite captures what the factory is asked to build; the rest (ledger, orchestrator, CLI) is real."""
        from nightshift.orchestrator import Orchestrator
        from nightshift.task import load_task
        from tests.support.fake_engine import RecordingSupervisor, fake_engine_factory, make_suite
        captured: dict = {}

        def fake_build(backend, task, run_dir, routing, gate, ledger, **kw):
            captured.update(kw, backend=backend)
            suite = make_suite(RecordingSupervisor(lambda env: env.finish("done")), backend="meta")
            if kw.get("plant"):
                suite.planted = make_suite(backend="scripted")
            return suite

        api = self.make_api()
        api.Orchestrator, api.load_task = Orchestrator, load_task
        secret_line = "SECRET-RELATED-WORK-LINE-42"
        f = self.write("notes.md", f"Background.\n{secret_line}\n")
        rd = self.tmp / "e2e"
        from nightshift import orchestrator as orch_mod
        with mock.patch("nightshift.agents.factory.build_suite", side_effect=fake_build), \
                mock.patch("nightshift.workflows.engine.WorkflowEngine", fake_engine_factory()), \
                mock.patch.object(orch_mod.sandbox, "harden_parent"), \
                mock.patch.object(orch_mod.sandbox, "effective_isolation", return_value="none"), \
                mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            code, out, err = self.invoke(api, "run", "--task", self.LAW, "--backend", "meta", "--run-dir", str(rd), "--context", f,
                                         "--plant", "tamper")
        self.assertEqual(code, 0, err)
        self.assertEqual((captured["backend"], captured["plant"]), ("meta", ["tamper"]))
        self.assertIn(secret_line, captured["background"])
        started = [e for e in Ledger.read_only(rd).events() if e.kind == Kind.RUN_STARTED][0].data["config"]
        self.assertEqual(started["red_team"], [{"behavior": "tamper", "scripted_id": "tamper"}])
        self.assertEqual([(c["name"], c["chars"]) for c in started["context"]], [("notes.md", len(f"Background.\n{secret_line}\n"))])
        self.assertNotIn(secret_line, (rd / "ledger.jsonl").read_text(encoding="utf-8"))
        self.assertNotIn(KEY_CANARY, out + err)


def write_started(run_dir, backend: str, context: list[dict], task: str = "fake-task") -> None:
    """An unfinished run whose ``run.started.config.context`` holds ``context``."""
    led = Ledger.create(run_dir, clock=FakeClock(), fsync=False)
    led.append(Kind.RUN_STARTED, "engine", {
        "task": task, "backend": backend, "models": {"supervisor": backend}, "limits": BudgetLimits().to_dict(),
        "task_hash": "t" * 12, "grader_hash": "g" * 12, "objective": "minimize", "primary_metric": "heldout",
        "config": {"context": context} if context else {}})
    led.close()


class IsolationEnvAndTasksDirTests(SeamCase):
    """``NIGHTSHIFT_ISOLATION`` is validated before anything is created; ``NIGHTSHIFT_TASKS_DIR`` and the missing-tasks hint."""

    ALLOWED = ("auto", "userns", "unshare-net", "none")

    def bad_isolation(self):
        return mock.patch.dict(os.environ, {"NIGHTSHIFT_ISOLATION": "bogus"})

    def assert_names_allowed_values(self, err):
        self.assertIn("NIGHTSHIFT_ISOLATION", err)
        self.assertIn("bogus", err)
        for word in self.ALLOWED:
            self.assertIn(word, err)

    def test_run_refuses_an_invalid_override_before_creating_anything(self):
        rd = self.tmp / "r1"
        with self.bad_isolation():
            code, out, err = self.invoke(self.make_api(), "run", "--task", self.task_dir(), "--run-dir", str(rd))
        self.assertEqual(code, 2)
        self.assert_names_allowed_values(err)
        self.assertEqual((out, self.names(), rd.exists()), ("", [], False))

    def test_resume_refuses_an_invalid_override(self):
        d = self.run_dir()
        with self.bad_isolation():
            code, out, err = self.invoke(self.make_api(), "resume", str(d))
        self.assertEqual(code, 2)
        self.assert_names_allowed_values(err)
        self.assertEqual(self.names(), [])

    def test_demo_refuses_an_invalid_override_and_leaves_no_run_directory(self):
        with self.bad_isolation():
            code, out, err = self.invoke(self.make_api(), "demo", cwd=self.repo)
        self.assertEqual(code, 2)
        self.assert_names_allowed_values(err)
        self.assertEqual((out, self.names(), (self.repo / "runs").exists()), ("", [], False))

    def test_a_valid_override_is_accepted_in_any_case(self):
        with mock.patch.dict(os.environ, {"NIGHTSHIFT_ISOLATION": " None "}):
            code, _, err = self.invoke(self.make_api(), "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"))
        self.assertEqual(code, 0, err)

    # ---- the tasks directory --------------------------------------------------------------------
    def empty_tasks_dir(self) -> Path:
        d = self.tmp / "no-tasks"
        d.mkdir()
        return d

    def test_every_command_that_needs_the_bundled_tasks_says_where_they_went(self):
        empty = self.empty_tasks_dir()
        d = self.run_dir()
        real = SimpleNamespace(list_tasks=lambda root, on_error=None: [], validate_task=lambda t, calibrate=False: [])
        for argv in (("demo",), ("run", "--task", "law-discovery"), ("resume", str(d)), ("audit", str(d)), ("tasks",)):
            with self.subTest(argv=argv), mock.patch.dict(os.environ, {"NIGHTSHIFT_TASKS_DIR": str(empty)}):
                api = self.make_api()
                api.list_tasks, api.validate_task = real.list_tasks, real.validate_task
                api.Auditor = object
                code, out, err = self.invoke(api, *argv, cwd=self.tmp)
                self.assertEqual(code, 2, (out, err))
                self.assertIn("bundled tasks not found next to the nightshift package", err)
                self.assertIn("pip install -e .", err)
                self.assertIn("--task /path/to/task", err)
                self.assertEqual(len(err.strip().splitlines()), 1)         # one line
        self.assertEqual([n for n in self.names() if n in ("create", "run", "resume")], [])

    def test_an_explicit_task_path_works_without_bundled_tasks(self):
        empty = self.empty_tasks_dir()
        with mock.patch.dict(os.environ, {"NIGHTSHIFT_TASKS_DIR": str(empty)}):
            code, _, err = self.invoke(self.make_api(), "run", "--task", self.task_dir(), "--run-dir", str(self.tmp / "r1"))
        self.assertEqual(code, 0, err)

    def test_the_tasks_dir_variable_replaces_the_checkouts_tasks(self):
        other = self.tmp / "other-tasks"
        (other / "elsewhere").mkdir(parents=True)
        (other / "elsewhere" / "task.toml").write_text("name='elsewhere'")
        with mock.patch.dict(os.environ, {"NIGHTSHIFT_TASKS_DIR": str(other)}):
            code, _, err = self.invoke(self.make_api(), "demo", "--task", "elsewhere", "--out", str(self.tmp / "o"))
            self.assertEqual(code, 0, err)
            self.assertEqual([c for c in self.calls if c[0] == "create"][0][2].name, "elsewhere")
            self.assertEqual(self.cli("demo", "--task", "law-discovery", "--out", str(self.tmp / "o2"))[0], 2)

    def test_the_tasks_command_uses_the_variable_and_lists_what_is_there(self):
        other = self.tmp / "other-tasks"
        shutil.copytree(FIXTURE, other / "mini-curve", ignore=shutil.ignore_patterns("__pycache__"))
        with mock.patch.dict(os.environ, {"NIGHTSHIFT_TASKS_DIR": str(other)}):
            code, out, _ = self.cli("tasks")
        self.assertEqual(code, 0, out)
        self.assertIn("OK   mini-curve", out)

    def test_tasks_dir_helper_and_repo_root_fallback(self):
        with mock.patch.dict(os.environ, {"NIGHTSHIFT_TASKS_DIR": ""}):
            self.assertEqual(cli._tasks_dir(self.repo), self.repo / "tasks")
            self.assertIsNone(cli._tasks_dir(self.tmp / "nowhere"))
        with mock.patch.dict(os.environ, {"NIGHTSHIFT_TASKS_DIR": str(self.repo / "tasks")}):
            self.assertEqual(cli._tasks_dir(self.tmp / "nowhere"), self.repo / "tasks")

    def test_the_demo_output_directory_is_relative_to_the_callers_cwd_not_the_repository(self):
        where = self.tmp / "caller"
        where.mkdir()
        code, out, err = self.invoke(self.make_api(), "demo", "--out", "mine/run", cwd=where)
        self.assertEqual(code, 0, err)
        self.assertEqual([c for c in self.calls if c[0] == "create"][0][1], where.resolve() / "mine" / "run")
        self.assertFalse((self.repo / "mine").exists())

    def test_an_out_path_with_an_unknown_home_is_a_usage_error_not_a_traceback(self):
        """Regression: expanding ``~`` in ``--out`` raised RuntimeError for ``~nosuchuser/...``."""
        code, out, err = self.invoke(self.make_api(), "demo", "--out", "~no-such-user-xyz/runs", cwd=self.tmp)
        self.assertEqual(code, 2, (out, err))
        self.assertIn("--out", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual([c for c in self.calls if c[0] == "create"], [])

    def test_the_crash_demo_hands_its_children_absolute_paths(self):
        children = []
        where = self.tmp / "caller"
        where.mkdir()
        with mock.patch.object(cli, "_run_child", lambda argv, cwd: children.append((argv, cwd)) or (137 if len(children) == 1 else 0)):
            code, _, _ = self.invoke(self.make_api(), "demo", "--with-crash", "--out", "rel/out", cwd=where)
        self.assertEqual(code, 0)
        want = str(where.resolve() / "rel" / "out")
        first, second = children[0][0], children[1][0]
        self.assertEqual(first[first.index("--run-dir") + 1], want)
        self.assertEqual(second[1], want)
        for argv in (first, second):
            self.assertTrue(Path(argv[argv.index("--task") + 1]).is_absolute(), argv)


class ResumeContextTests(SeamCase):
    """``resume --context FILE...``: the same background as at the start, checked by sha256."""

    def setUp(self) -> None:
        super().setUp()
        self.notes = self.tmp / "notes.md"
        self.notes.write_text("Related work: the SECRET-BACKGROUND-LINE.\n", encoding="utf-8")
        self.other = self.tmp / "other.md"
        self.other.write_text("Second file.\n", encoding="utf-8")
        _, self.records = cli._read_context([[str(self.notes), str(self.other)]], "meta")
        self.d = self.tmp / "r1"
        write_started(self.d, "meta", self.records)

    def resume(self, *argv: str, api=None):
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            return self.invoke(api or self.make_api(), "resume", str(self.d), "--task", self.task_dir(), *argv)

    def resume_kw(self):
        return [c for c in self.calls if c[0] == "resume"][0][3]

    def test_the_same_files_are_passed_to_the_orchestrator_in_any_order(self):
        for order in ((self.notes, self.other), (self.other, self.notes)):
            self.calls.clear()
            code, out, err = self.resume("--context", *map(str, order))
            self.assertEqual(code, 0, err)
            kw = self.resume_kw()
            self.assertIn("SECRET-BACKGROUND-LINE", kw["background"])
            self.assertEqual({(c["name"], c["sha256"]) for c in kw["context"]}, {(c["name"], c["sha256"]) for c in self.records})
            self.assertIn("2 file(s), the same as at the start", out)
            self.assertNotIn("warning", err)
            self.assertNotIn("SECRET-BACKGROUND-LINE", out + err)

    def test_the_flag_can_be_repeated(self):
        code, _, err = self.resume("--context", str(self.notes), "--context", str(self.other))
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.resume_kw()["context"]), 2)

    def test_a_changed_file_is_a_usage_error_naming_it(self):
        self.notes.write_text("edited since the run started\n", encoding="utf-8")
        code, out, err = self.resume("--context", str(self.notes), str(self.other))
        self.assertEqual(code, 2)
        self.assertIn("notes.md", err)
        self.assertIn("changed", err)
        self.assertEqual([n for n in self.names() if n == "resume"], [])

    def test_a_file_that_was_not_part_of_the_run_is_a_usage_error_naming_it(self):
        extra = self.tmp / "extra.md"
        extra.write_text("not recorded\n", encoding="utf-8")
        code, _, err = self.resume("--context", str(self.notes), str(self.other), str(extra))
        self.assertEqual(code, 2)
        self.assertIn("extra.md", err)
        self.assertEqual([n for n in self.names() if n == "resume"], [])

    def test_a_missing_file_of_the_original_set_is_a_usage_error_naming_it(self):
        code, _, err = self.resume("--context", str(self.notes))
        self.assertEqual(code, 2)
        self.assertIn("other.md", err)
        self.assertEqual([n for n in self.names() if n == "resume"], [])

    def test_two_background_files_with_the_same_basename_are_each_matched_on_their_own(self):
        """Regression: the check keyed the recorded files by name, so with a/notes.md and b/notes.md one of them could be
        dropped (or swapped) without the usage error naming it; only the orchestrator's refusal (exit 1) caught it."""
        a, b = self.tmp / "a" / "notes.md", self.tmp / "b" / "notes.md"
        for path, text in ((a, "first notes\n"), (b, "second notes\n")):
            path.parent.mkdir()
            path.write_text(text, encoding="utf-8")
        _, records = cli._read_context([[str(a), str(b)]], "meta")
        d2 = self.tmp / "twins"
        write_started(d2, "meta", records)

        def resume(*files):
            self.calls.clear()
            with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
                return self.invoke(self.make_api(), "resume", str(d2), "--task", self.task_dir(), "--context", *map(str, files))
        for files in ((b,), (a,), (a, a)):
            with self.subTest(files=[str(f) for f in files]):
                code, _, err = resume(*files)
                self.assertEqual(code, 2, err)
                self.assertIn("notes.md", err)
                self.assertEqual([n for n in self.names() if n == "resume"], [])
        for files in ((a, b), (b, a)):
            code, _, err = resume(*files)
            self.assertEqual(code, 0, err)
        a.write_text("edited\n", encoding="utf-8")
        code, _, err = resume(a, b)
        self.assertEqual(code, 2)
        self.assertIn("changed", err)

    def test_context_for_a_run_that_started_without_any_is_a_usage_error(self):
        d2 = self.tmp / "plain"
        write_started(d2, "meta", [])
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            code, _, err = self.invoke(self.make_api(), "resume", str(d2), "--task", self.task_dir(), "--context", str(self.notes))
        self.assertEqual(code, 2)
        self.assertIn("started without background files", err)

    def test_an_unreadable_file_uses_the_same_rules_as_run(self):
        code, _, err = self.resume("--context", str(self.tmp / "gone.md"))
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)

    def test_without_context_the_run_resumes_with_a_clear_warning(self):
        code, out, err = self.resume()
        self.assertEqual(code, 0, err)
        self.assertIn("warning", err)
        self.assertIn("WITHOUT the background", err)
        self.assertNotIn("background", self.resume_kw())
        self.assertNotIn("context", self.resume_kw())

    def test_a_run_without_background_resumes_silently(self):
        d2 = self.tmp / "plain"
        write_started(d2, "meta", [])
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": KEY_CANARY}):
            code, _, err = self.invoke(self.make_api(), "resume", str(d2), "--task", self.task_dir())
        self.assertEqual((code, err), (0, ""))

    def test_the_scripted_backend_has_no_context_to_reload(self):
        d2 = self.tmp / "scripted"
        write_started(d2, "scripted", [])
        code, _, err = self.invoke(self.make_api(), "resume", str(d2), "--task", self.task_dir(), "--context", str(self.notes))
        self.assertEqual(code, 2)
        self.assertIn("--backend meta", err)
        self.assertEqual([n for n in self.names() if n in ("create", "resume", "run")], [])


if __name__ == "__main__":
    unittest.main()
