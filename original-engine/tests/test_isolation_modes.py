"""The whole pipeline in every isolation mode, and what each mode actually prevents versus merely detects.

Hackathon judges and CI runners use laptops and containers where user namespaces may be unavailable, so Nightshift must
work, and say honestly what it guarantees, in all three modes of ``nightshift.sandbox`` (see its module docstring for the
table this file pins):

* ``userns``       the trusted directory, the ledger and sibling experiments are hidden, the network is empty, every
                   process dies with the namespace.
* ``unshare-net``  the network is empty; the run can still READ the trusted directory and the ledger (not even detected);
                   a CHANGE to the trusted directory is detected after the run.
* ``none``         as ``unshare-net`` and the network is reachable.

For each mode this machine supports (the others are skipped with the probe's reason) the tests run a small end-to-end
evaluation through ``grading.evaluate`` over ``tests/fixtures/mini_task``, a hostile run, and check that the mode applied is
reported on ``RunResult.isolation`` and in the ``experiment.run_started`` event. A second group simulates machines without
``unshare`` or ``/proc`` (and a macOS-style platform) and shows that an honest run stays ``ok``; the scripted demo is run
to completion with ``NIGHTSHIFT_ISOLATION=none`` as a subprocess (about 20 s).

Run: ``python3 -m unittest tests.test_isolation_modes -v``
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from nightshift import grading, sandbox
from nightshift.ledger import Ledger
from nightshift.task import trusted_manifest
from nightshift.types import Kind, ProtectedPathViolation, TaskSpec
from nightshift.view import build_view
from nightshift.types import Outcome
from tests.support.fake_suite import EngineHarness, Patch, binned_model, copy_mini_task
from tests.test_demo_e2e import EXPECTED_FINDINGS, EXPECTED_OUTCOMES, normalised_findings, outcomes

ROOT = Path(__file__).resolve().parents[1]
HAS_PROC = os.path.isdir("/proc/self") and sys.platform.startswith("linux")
LEDGER_TEXT = b'{"seq": 1, "secret": "the ledger the run must not read"}\n'
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------------------------------
# which modes this machine supports
# --------------------------------------------------------------------------------------------------
def _without_override():
    env = {k: v for k, v in os.environ.items() if k != sandbox.ISOLATION_ENV}
    return mock.patch.dict(os.environ, env, clear=True)


def _supported() -> tuple[set[str], dict[str, str]]:
    """(modes the probe proves here, mode -> why not). Independent of any NIGHTSHIFT_ISOLATION set by the caller."""
    with _without_override():
        ok = {m for m in sandbox.ISOLATION_LEVELS if sandbox.effective_isolation(m) == m}
        reason = str(sandbox._probed().get("reason") or "the probe did not prove it")
    return ok, {m: reason for m in sandbox.ISOLATION_LEVELS if m not in ok}


SUPPORTED, WHY_NOT = _supported()
BEST = max(SUPPORTED, key=sandbox._RANK.__getitem__)


def _net_prefix_hides_processes() -> bool:
    """Did the probe prove the root-only pid-namespace variant of ``unshare-net`` here? Then nothing outlives the run."""
    if "unshare-net" not in SUPPORTED:
        return False
    with _without_override():
        info = sandbox._effective_isolation("unshare-net")[1]
    return sandbox.prefix_isolates_pids(info.get("net_prefix"))      # type: ignore[arg-type]


PIDNS_NET = _net_prefix_hides_processes()

# What a hostile run achieves in each mode. This is the executable form of the table in the sandbox module docstring.
#   read_*       "read": the bytes were readable; "hidden": masked or absent        network   "reachable" | "blocked"
#   write_trusted  the run's attempt to append to trusted/labels.csv                 trusted_change  "detected" | "none"
#   write_outside  a file outside the workspace and outside trusted/ (never stopped, never noticed, in any mode)
#   daemon_*     "killed" | "survives" (checked from outside, by command line)
EXPECTED = {
    "userns": dict(read_trusted="hidden", read_ledger="hidden", network="blocked", write_trusted="denied",
                   trusted_change="none", write_outside="written", daemon_plain="killed", daemon_evasive="killed"),
    # with the pid-namespace variant (root only, probe-proven) the evasive daemon dies with the namespace init as well
    "unshare-net": dict(read_trusted="read", read_ledger="read", network="blocked", write_trusted="written",
                        trusted_change="detected", write_outside="written", daemon_plain="killed",
                        daemon_evasive="killed" if PIDNS_NET else "survives"),
    "none": dict(read_trusted="read", read_ledger="read", network="reachable", write_trusted="written",
                 trusted_change="detected", write_outside="written", daemon_plain="killed", daemon_evasive="survives"),
}

HOSTILE = r'''
import json, os, socket, sys
TRUSTED_LABELS, LEDGER, OUTSIDE, PORT = %(trusted)r, %(ledger)r, %(outside)r, %(port)d
PLAIN, EVASIVE = %(plain)r, %(evasive)r
REPORT = {}


def attempt(name, fn):
    try:
        REPORT[name] = fn()
    except Exception as exc:
        REPORT[name] = "ERR " + type(exc).__name__


def read_len(path):
    with open(path, "rb") as fh:
        return len(fh.read())


def append(path, data):
    with open(path, "ab") as fh:
        fh.write(data)
    return "written"


def connect(port):
    socket.create_connection(("127.0.0.1", port), timeout=2).close()
    return "connected"


def daemon(marker, evasive):
    code = "# " + marker + "\nimport time\ntime.sleep(25)\n"
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            if evasive:                                  # classic double fork, stdio closed, working directory elsewhere
                if os.fork() > 0:
                    os._exit(0)
                os.chdir("/")
                null = os.open(os.devnull, os.O_RDWR)
                for fd in (0, 1, 2):
                    os.dup2(null, fd)
            os.execv(sys.executable, [sys.executable, "-c", code])
        finally:
            os._exit(1)
    if evasive:
        os.waitpid(pid, 0)
    return "spawned"


attempt("read_trusted", lambda: read_len(TRUSTED_LABELS))
attempt("read_ledger", lambda: read_len(LEDGER))
attempt("write_trusted", lambda: append(TRUSTED_LABELS, b"tampered\n"))
attempt("network", lambda: connect(PORT))
attempt("write_outside", lambda: append(OUTSIDE, b"x"))
attempt("daemon_plain", lambda: daemon(PLAIN, False))
attempt("daemon_evasive", lambda: daemon(EVASIVE, True))
print("HOSTILE_REPORT " + json.dumps(REPORT), flush=True)
'''


def marked_pids(token: str) -> list[int]:
    """Live processes whose command line contains ``token`` (zombies have an empty command line)."""
    needle, found = token.encode(), []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/cmdline", "rb") as fh:
                    if needle in fh.read():
                        found.append(int(name))
            except OSError:
                continue
    return found


def wait_gone(token: str, limit: float = 3.0) -> list[int]:
    end = time.monotonic() + limit
    while True:
        left = marked_pids(token)
        if not left or time.monotonic() >= end:
            return left
        time.sleep(0.05)


def kill_marked(token: str) -> None:
    for pid in marked_pids(token):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


_REFERENCE: dict[str, float] = {}


def reference_score() -> float:
    """The mini-task score of the unmodified template with isolation ``none``: every mode must reproduce it exactly."""
    if "score" not in _REFERENCE:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {sandbox.ISOLATION_ENV: "none"}):
            task = copy_mini_task(Path(tmp) / "task")
            ws = sandbox.create_workspace(task, Path(tmp) / "run", "e001")
            res = grading.evaluate(task, ws, SEEDS, purpose="primary", run_dir=Path(tmp) / "run", expected_tree_hash=ws.tree_hash)
            assert res.grade.valid and res.grade.score is not None, res.grade.error
            _REFERENCE["score"] = res.grade.score
    return _REFERENCE["score"]


class Fixture:
    """A private copy of the mini task, a run directory with a ledger file, and a workspace made from the template."""

    def __init__(self, root: Path):
        self.root = root
        self.task: TaskSpec = copy_mini_task(root / "task")
        self.run_dir = root / "run"
        self.ws = sandbox.create_workspace(self.task, self.run_dir, "e001")
        (self.run_dir / "ledger.jsonl").write_bytes(LEDGER_TEXT)

    def evaluate(self, seeds=SEEDS, *, ws=None, tree=None, hooks=None, isolation="auto"):
        ws = ws or self.ws
        return grading.evaluate(self.task, ws, seeds, purpose="primary", run_dir=self.run_dir,
                                expected_tree_hash=tree or ws.tree_hash, hooks=hooks, isolation=isolation)


# --------------------------------------------------------------------------------------------------
# one test class per mode
# --------------------------------------------------------------------------------------------------
class ModeTests:
    """Mixin: ``MODE`` is forced through ``NIGHTSHIFT_ISOLATION`` (the path a judge would use), applied with ``auto``."""

    MODE = ""

    @classmethod
    def setUpClass(cls):
        if cls.MODE not in SUPPORTED:
            raise unittest.SkipTest(f"isolation mode {cls.MODE!r} is not available on this machine: {WHY_NOT[cls.MODE]}")
        patcher = mock.patch.dict(os.environ, {sandbox.ISOLATION_ENV: cls.MODE})
        patcher.start()
        cls.addClassCleanup(patcher.stop)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fx = Fixture(Path(tmp.name))
        self.tmp = Path(tmp.name)

    def expected_note_part(self) -> str:
        """What ``RunResult.note`` must say about the mode: nothing when the probe's best mode was used."""
        if self.MODE == "none":
            return f"{sandbox.ISOLATION_ENV}=none"
        if self.MODE != BEST:
            return f"isolation lowered from {BEST} to {self.MODE}"
        return ""

    # ---- an honest end-to-end evaluation --------------------------------------------------------
    def test_honest_evaluation_end_to_end(self):
        res = self.fx.evaluate()
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertIsNone(res.failure_class)
        self.assertEqual([r.classification for r in res.runs], ["ok"] * 3)
        self.assertEqual({r.isolation for r in res.runs}, {self.MODE})
        self.assertEqual(res.grade.score, reference_score(), "the score does not depend on the isolation mode")
        self.assertEqual(len(res.outputs_hashes), 3)
        self.assertTrue(all(res.outputs_hashes))
        want = self.expected_note_part()
        for run in res.runs:
            if want:
                self.assertIn(want, run.note)
            else:
                self.assertEqual(run.note, "")
        self.assertEqual(sandbox.available_isolation(), self.MODE)

    def test_the_same_mode_is_applied_by_an_explicit_argument_without_the_variable(self):
        with _without_override():
            res = self.fx.evaluate([0], isolation=self.MODE)
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual(res.runs[0].isolation, self.MODE)
        self.assertEqual(res.runs[0].note, "")

    # ---- reporting -------------------------------------------------------------------------------
    def test_the_mode_is_reported_on_the_run_result_and_in_the_run_started_event(self):
        h = EngineHarness(self.tmp / "engine", overrides={})          # isolation "auto": the engine asks the sandbox
        self.addCleanup(h.close)
        h.baseline()
        started, finished = h.events(Kind.EXP_RUN_STARTED), h.events(Kind.EXP_RUN_FINISHED)
        self.assertEqual(len(started), 3)
        self.assertEqual({e.data["isolation"] for e in started}, {self.MODE})
        self.assertEqual({e.data["classification"] for e in finished}, {"ok"})
        want = self.expected_note_part()
        for e in finished:
            self.assertIn(want, e.data["note"]) if want else self.assertEqual(e.data["note"], "")
        res = self.fx.evaluate([0])
        self.assertEqual(res.runs[0].isolation, started[0].data["isolation"])

    # ---- a hostile run: what is prevented, what is only detected ----------------------------------
    def hostile_run(self) -> dict:
        fx, task = self.fx, self.fx.task
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(5)
        token = uuid.uuid4().hex[:12]
        plain, evasive = f"nsiso-plain-{token}", f"nsiso-evasive-{token}"
        outside = self.tmp / "outside.txt"
        script = HOSTILE % dict(trusted=str(task.trusted_dir / "labels.csv"), ledger=str(fx.run_dir / "ledger.jsonl"),
                                outside=str(outside), port=listener.getsockname()[1], plain=plain, evasive=evasive)
        run_py = fx.ws.path / "run.py"
        run_py.write_text(script + "\n" + run_py.read_text())
        tree = sandbox.tree_hash(fx.ws.path, task)
        before = trusted_manifest(task)
        if HAS_PROC:
            self.addCleanup(kill_marked, token)
        seen: list[sandbox.RunResult] = []
        violation, result = None, None
        try:
            result = fx.evaluate([0], tree=tree, hooks=grading.EvalHooks(on_run_end=lambda i, s, run: seen.append(run)))
        except ProtectedPathViolation as exc:
            violation = exc
        self.assertEqual(len(seen), 1, "the hostile run itself must complete")
        run = seen[0]
        self.assertEqual((run.classification, run.isolation), ("ok", self.MODE), run.stderr)
        lines = [ln for ln in run.stdout.splitlines() if ln.startswith("HOSTILE_REPORT ")]
        self.assertEqual(len(lines), 1, run.stdout)
        report = json.loads(lines[0].split(" ", 1)[1])
        facts = {
            "read_trusted": "read" if isinstance(report["read_trusted"], int) and report["read_trusted"] > 0 else "hidden",
            "read_ledger": "read" if report["read_ledger"] == len(LEDGER_TEXT) else "hidden",
            "network": "reachable" if report["network"] == "connected" else "blocked",
            "write_trusted": "written" if report["write_trusted"] == "written" else "denied",
            "trusted_change": "detected" if violation is not None and violation.kind == "trusted" else "none",
            "write_outside": "written" if outside.exists() and outside.read_bytes() == b"x" else "blocked",
        }
        if HAS_PROC:
            for name, marker in (("daemon_plain", plain), ("daemon_evasive", evasive)):
                facts[name] = "survives" if wait_gone(marker) else "killed"
        self.hostile = dict(report=report, violation=violation, result=result, before=before)
        return facts

    def test_hostile_run_prevents_and_detects_exactly_what_the_mode_promises(self):
        facts = self.hostile_run()
        want = {k: v for k, v in EXPECTED[self.MODE].items() if k in facts}
        self.assertEqual(facts, want)

    def test_hostile_run_consequences(self):
        self.hostile_run()
        task, h = self.fx.task, self.hostile
        if self.MODE == "userns":
            # prevented: nothing to detect, the grader runs, the trusted tree and the ledger are untouched
            self.assertIsNone(h["violation"])
            self.assertTrue(h["result"].grade.valid, h["result"].grade.error)
            self.assertEqual(trusted_manifest(task), h["before"])
            self.assertEqual((self.fx.run_dir / "ledger.jsonl").read_bytes(), LEDGER_TEXT)
        else:
            # not prevented, but the change is detected and named; the evaluation is refused
            self.assertIsNotNone(h["violation"])
            self.assertEqual(h["violation"].kind, "trusted")
            self.assertEqual(h["violation"].paths, ["trusted/labels.csv"])
            self.assertNotEqual(trusted_manifest(task), h["before"])


    # ---- write, then put the bytes back before exiting -------------------------------------------
    def restoring(self, labels: Path) -> str:
        """Module-level code of a model: rewrite the first bytes of ``labels``, put the original bytes AND mtime back, report.

        The path is assembled at run time from short pieces, as a hostile model would, so the diff tripwire (which flags text
        that names a trusted path) does not end the experiment before the run: this is about what happens DURING the run."""
        pieces = [str(labels)[i:i + 3] for i in range(0, len(str(labels)), 3)]
        return textwrap.dedent(f"""
            import json, os
            LABELS = "".join({pieces!r})
            report = {{}}
            try:
                st = os.stat(LABELS)
                orig = open(LABELS, "rb").read()
                with open(LABELS, "r+b") as fh:
                    fh.write(b"X" * 8)
                with open(LABELS, "r+b") as fh:
                    fh.write(orig[:8])
                os.utime(LABELS, ns=(st.st_atime_ns, st.st_mtime_ns))
                report["restore"] = "restored"
            except Exception as exc:
                report["restore"] = "ERR " + type(exc).__name__
            print("RESTORE_REPORT " + json.dumps(report), flush=True)
        """)

    def test_write_then_restore_is_prevented_by_userns_and_detected_by_the_weaker_modes(self):
        task = self.fx.task
        labels = task.trusted_dir / "labels.csv"
        original, mtime = labels.read_bytes(), labels.stat().st_mtime_ns
        manifest = trusted_manifest(task)
        run_py = self.fx.ws.path / "run.py"
        run_py.write_text(self.restoring(labels) + "\n" + run_py.read_text())
        tree = sandbox.tree_hash(self.fx.ws.path, task)
        seen: list[sandbox.RunResult] = []
        violation, result = None, None
        try:
            result = self.fx.evaluate([0], tree=tree, hooks=grading.EvalHooks(on_run_end=lambda i, s, run: seen.append(run)))
        except ProtectedPathViolation as exc:
            violation = exc
        self.assertEqual(len(seen), 1)
        run = seen[0]
        self.assertEqual((run.classification, run.isolation), ("ok", self.MODE), run.stderr)
        report = json.loads(next(ln for ln in run.stdout.splitlines() if ln.startswith("RESTORE_REPORT ")).split(" ", 1)[1])
        # whatever the mode, the file ends up byte for byte, mtime for mtime, what it was: the content hash cannot tell
        self.assertEqual((labels.read_bytes(), labels.stat().st_mtime_ns), (original, mtime))
        self.assertEqual(trusted_manifest(task), manifest)
        if self.MODE == "userns":
            self.assertTrue(report["restore"].startswith("ERR"), "the file is not even visible inside the namespace")
            self.assertIn("FileNotFoundError", report["restore"])
            self.assertIsNone(violation)
            self.assertTrue(result.grade.valid, result.grade.error)
            self.assertEqual(run.trusted_changed, [])
        else:
            self.assertEqual(report["restore"], "restored")
            self.assertIsNotNone(violation, "the restored change must still be detected")
            self.assertEqual((violation.kind, violation.paths), ("trusted", ["trusted/labels.csv"]))
            self.assertEqual(run.trusted_changed, ["trusted/labels.csv"])
            self.assertFalse(run.trusted_contested)

    def test_write_then_restore_ends_the_experiment_invalid_in_the_engine(self):
        task = copy_mini_task(self.tmp / "engine-task")
        h = EngineHarness(self.tmp / "engine-restore", task=task, overrides={})
        self.addCleanup(h.close)
        h.baseline()
        h.agents.patches["restore"] = Patch({"model.py": binned_model(5, extra=self.restoring(task.trusted_dir / "labels.csv"))})
        res = h.lane("restore")
        protected = h.events(Kind.EXP_PROTECTED, exp_id="e001")
        if self.MODE == "userns":
            self.assertEqual(protected, [])
            self.assertNotEqual(res.outcome, Outcome.INVALID)
        else:
            self.assertEqual(res.outcome, Outcome.INVALID)
            self.assertEqual([(e.data["kind"], e.data["paths"]) for e in protected], [("trusted", ["trusted/labels.csv"])])
            self.assertEqual(h.events(Kind.EXP_GRADED, exp_id="e001"), [], "an experiment that touched trusted/ is never graded")
            self.assertEqual({e.data["isolation"] for e in h.events(Kind.EXP_RUN_STARTED, exp_id="e001")}, {self.MODE})


for _mode in sandbox.ISOLATION_LEVELS:
    _name = "Mode" + "".join(part.capitalize() for part in _mode.split("-")) + "Tests"
    globals()[_name] = type(_name, (ModeTests, unittest.TestCase), {"MODE": _mode, "__module__": __name__})
del _mode, _name


# --------------------------------------------------------------------------------------------------
# facts that do not depend on the mode
# --------------------------------------------------------------------------------------------------
class TripwireTests(unittest.TestCase):
    """A read of the trusted directory is invisible under ``unshare-net`` and ``none``; the only trace is the diff tripwire."""

    def test_text_naming_a_trusted_path_is_flagged_in_the_diff_but_a_plain_read_leaves_no_other_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            (fx.ws.path / "model.py").write_text(
                "def fit(xs, ys):\n    labels = open('../../../../task/trusted/labels.csv').read()\n    return lambda x: 0.0\n")
            diff = sandbox.diff_workspace(fx.task, fx.ws)
            self.assertIn("probe:model.py", diff.violations)
            (fx.ws.path / "model.py").write_text("def fit(xs, ys):\n    return lambda x: 0.0\n")
            self.assertEqual([v for v in sandbox.diff_workspace(fx.task, fx.ws).violations if v.startswith("probe:")], [])


class OverrideTests(unittest.TestCase):
    def test_the_override_is_validated_when_the_pipeline_first_uses_it(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {sandbox.ISOLATION_ENV: "sandbox9000"}):
            fx = Fixture(Path(tmp))
            with self.assertRaises(ValueError) as caught:
                fx.evaluate([0])
        for allowed in ("auto", "userns", "unshare-net", "none"):
            self.assertIn(allowed, str(caught.exception))

    def test_forcing_a_mode_the_machine_lacks_runs_with_what_is_available_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {sandbox.ISOLATION_ENV: "userns"}), \
                mock.patch.dict(sandbox._iso_info, {"level": "none", "reason": "user namespaces unavailable (probe failed)"}, clear=True):
            res = Fixture(Path(tmp)).evaluate([0])
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual(res.runs[0].isolation, "none")
        self.assertIn("NIGHTSHIFT_ISOLATION=userns not available (user namespaces unavailable (probe failed)); applied none",
                      res.runs[0].note)


# --------------------------------------------------------------------------------------------------
# machines without namespaces, /proc or setrlimit(RLIMIT_AS)
# --------------------------------------------------------------------------------------------------
class NoNamespaceMachineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        patcher = mock.patch.dict(sandbox._iso_info, clear=True)         # a fresh probe, undone afterwards
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_honest_run_is_ok(self, res: grading.EvalResult, isolation: str = "none") -> None:
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertIsNone(res.failure_class)
        self.assertEqual([r.classification for r in res.runs], ["ok"] * len(res.runs))
        self.assertEqual({r.isolation for r in res.runs}, {isolation})
        self.assertEqual(res.grade.score, reference_score())

    def test_path_without_unshare(self):
        empty = self.tmp / "emptybin"
        empty.mkdir()
        with _without_override(), mock.patch.dict(os.environ, {"PATH": str(empty)}):
            self.assertIsNone(__import__("shutil").which("unshare"))
            self.assertEqual(sandbox.available_isolation(refresh=True), "none")
            res = Fixture(self.tmp / "fx").evaluate()
        self.assert_honest_run_is_ok(res)
        self.assertEqual([r.note for r in res.runs], [""] * 3, "nothing was asked for and nothing was lost: no note")

    def test_a_probe_that_fails_everywhere(self):
        with _without_override(), mock.patch.object(sandbox, "_probe_run", return_value=(None, b"")):
            self.assertEqual(sandbox.available_isolation(refresh=True), "none")
            res = Fixture(self.tmp / "fx").evaluate()
        self.assert_honest_run_is_ok(res)

    def test_a_forced_mode_on_such_a_machine_degrades_with_the_reason_in_the_note(self):
        with mock.patch.dict(os.environ, {sandbox.ISOLATION_ENV: "unshare-net"}), \
                mock.patch.object(sandbox.shutil, "which", return_value=None):
            res = Fixture(self.tmp / "fx").evaluate()
        self.assert_honest_run_is_ok(res)
        self.assertIn("unshare or sh not found on PATH", res.runs[0].note)
        self.assertIn("applied none", res.runs[0].note)


class DegradedPlatformTests(unittest.TestCase):
    """macOS-style degradation: no /proc, no setrlimit(RLIMIT_AS), no os.waitid. An honest run is still ``ok``."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        patcher = mock.patch.dict(sandbox._iso_info, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != sandbox.ISOLATION_ENV}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def evaluate(self, seeds=SEEDS, **kw):
        return Fixture(self.tmp / "fx").evaluate(seeds, **kw)

    def test_forcing_the_linux_only_paths_off(self):
        # "proc_table" would raise if anything still looked at /proc; os.waitid is missing on macOS
        with mock.patch.object(sandbox, "_LINUX", False), \
                mock.patch.object(sandbox, "_proc_table", side_effect=AssertionError("/proc used off Linux")), \
                mock.patch.object(sandbox, "_task_count_for_uid", side_effect=AssertionError("/proc used off Linux")), \
                mock.patch.object(sandbox.os, "waitid", side_effect=AttributeError("no waitid")):
            self.assertEqual(sandbox.available_isolation(refresh=True), "none")
            plan = {name for name, _, _ in sandbox._limit_plan(Fixture(self.tmp / "p").task, 10.0)}
            self.assertEqual(plan, {"RLIMIT_CORE", "RLIMIT_CPU", "RLIMIT_FSIZE"})
            self.assertIsNone(sandbox._Tree(os.getpid(), None, set()).poll())
            res = self.evaluate()
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual([r.classification for r in res.runs], ["ok"] * 3)
        self.assertEqual({r.isolation for r in res.runs}, {"none"})
        self.assertEqual(res.grade.score, reference_score())
        self.assertEqual({r.note for r in res.runs}, {""})

    def test_off_linux_a_forced_namespace_mode_degrades_and_says_not_linux(self):
        with mock.patch.object(sandbox, "_LINUX", False), mock.patch.dict(os.environ, {sandbox.ISOLATION_ENV: "userns"}):
            res = self.evaluate([0])
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual(res.runs[0].isolation, "none")
        self.assertIn("NIGHTSHIFT_ISOLATION=userns not available (not Linux); applied none", res.runs[0].note)

    def test_linux_without_a_readable_proc(self):
        with mock.patch.object(sandbox, "_proc_table", side_effect=FileNotFoundError("/proc")), \
                mock.patch.object(sandbox, "_task_count_for_uid", return_value=None):
            res = self.evaluate()
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual([r.classification for r in res.runs], ["ok"] * 3)
        self.assertEqual(res.grade.score, reference_score())

    def test_setrlimit_failures_do_not_fail_an_honest_run(self):
        # macOS rejects RLIMIT_AS / RLIMIT_NPROC values that Linux accepts; the shim applies every limit on its own
        plan = [("RLIMIT_CORE", 0, 0), ("RLIMIT_AS", 1 << 30, 1 << 20), ("RLIMIT_NPROC", -1, -2),
                ("RLIMIT_NOT_THERE", 1, 1), ("RLIMIT_CPU", 100, 120)]
        with mock.patch.object(sandbox, "_limit_plan", return_value=plan):
            res = self.evaluate()
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual([r.classification for r in res.runs], ["ok"] * 3)
        self.assertEqual(res.grade.score, reference_score())

    def test_without_any_limit_support_the_run_is_still_ok(self):
        with mock.patch.object(sandbox, "_limit_plan", return_value=[]):
            res = self.evaluate([0])
        self.assertTrue(res.grade.valid, res.grade.error)

    @unittest.skipUnless(HAS_PROC, "needs /proc to look for leftovers")
    def test_off_linux_the_process_group_is_still_killed_without_the_proc_sweep(self):
        token = uuid.uuid4().hex[:12]
        self.addCleanup(kill_marked, token)
        fx = Fixture(self.tmp / "fx")
        run_py = fx.ws.path / "run.py"
        run_py.write_text(textwrap.dedent(f"""
            import subprocess, sys
            subprocess.Popen([sys.executable, "-c", "# nsiso-pg-{token}\\nimport time; time.sleep(25)"])
        """) + run_py.read_text())
        with mock.patch.object(sandbox, "_LINUX", False):
            res = fx.evaluate([0], tree=sandbox.tree_hash(fx.ws.path, fx.task))
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual(wait_gone(f"nsiso-pg-{token}"), [], "a same-group child must die with the run even without /proc")


# --------------------------------------------------------------------------------------------------
# the scripted demo on a machine without namespaces
# --------------------------------------------------------------------------------------------------
class DemoWithoutNamespacesTests(unittest.TestCase):
    """``python3 -m nightshift demo --no-serve`` with NIGHTSHIFT_ISOLATION=none: success and no false findings (~20 s)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nightshift-iso-demo-"))
        cls.addClassCleanup(lambda: __import__("shutil").rmtree(cls.tmp, True))
        cls.out = cls.tmp / "demo"
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(PYTHONPATH=str(ROOT), **{sandbox.ISOLATION_ENV: "none"})
        cls.proc = subprocess.run([sys.executable, "-m", "nightshift", "demo", "--no-serve", "--out", str(cls.out)], cwd=ROOT,
                                  env=env, capture_output=True, text=True, timeout=300, check=False)
        cls.events = list(Ledger.read_only(cls.out).events()) if (cls.out / "ledger.jsonl").is_file() else []
        cls.view = build_view(cls.events) if cls.events else None

    def detail(self) -> str:
        return f"\nstdout:\n{self.proc.stdout[-1500:]}\nstderr:\n{self.proc.stderr[-1500:]}"

    def test_completes_with_success(self):
        self.assertEqual(self.proc.returncode, 0, self.detail())
        self.assertIn("verdict=success", self.proc.stdout)
        self.assertEqual(self.view.finished["verdict_class"], "success")
        self.assertEqual(self.view.malformed, [])

    def test_no_false_findings_and_the_same_outcomes_as_the_namespaced_golden_run(self):
        self.assertEqual(normalised_findings(self.view), EXPECTED_FINDINGS, "no false and no missed finding")
        self.assertEqual([f.finding_id for f in self.view.open_findings()], [])
        self.assertEqual(outcomes(self.view), EXPECTED_OUTCOMES)
        report = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertRegex(report, r"False positives: 0\b")

    def test_every_untrusted_run_says_it_ran_without_isolation(self):
        started = [e for e in self.events if e.kind == Kind.EXP_RUN_STARTED]
        self.assertGreater(len(started), 20)
        self.assertEqual({e.data["isolation"] for e in started}, {"none"})
        self.assertEqual(self.view.run["isolation"], "none")
        finished = [e for e in self.events if e.kind == Kind.EXP_RUN_FINISHED]
        self.assertTrue(all(f"{sandbox.ISOLATION_ENV}=none" in e.data["note"] for e in finished))
        report = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertIn("Without isolation", report)

    def test_the_ledger_verifies(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-m", "nightshift", "verify", str(self.out)], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("chain consistent", proc.stdout)


if __name__ == "__main__":
    unittest.main()
