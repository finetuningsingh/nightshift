import io
import json
import os
import socket
import subprocess
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

from nightshift.lab import control_api, researcher_tool as rt
from nightshift.lab.contracts import BudgetExhausted, JobStatus
from nightshift.lab.control_api import ControlApi, validate_recipe_tarball
from nightshift.lab.metering_proxy import LabHttpServer
from tests.support.lab_fake_runtime import FakeLedger, FakeMeter, TrialFixture

TOKEN = "tool-token-0123456789abcdef"


class ToolCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.fx = TrialFixture(self.tmp)
        self.spec = self.fx.spec("t-0001")
        self.meter = FakeMeter(self.spec.budget, FakeLedger(), trial_id="t-0001", clock=self.fx.clock)
        self.api = ControlApi(self.fx.trainer, self.meter, self.fx.layout, self.spec, TOKEN, FakeLedger(), clock=self.fx.clock)
        self.server = LabHttpServer([self.api]).start()
        self.addCleanup(self.server.stop)
        self.addCleanup(lambda: self.api.close(wait=False))
        self.work = self.tmp / "work" / "agents" / "a1"  # a researcher worktree below the trial dir that holds the token
        self.work.mkdir(parents=True)
        (self.tmp / "work" / ".glimmer_token").write_text(TOKEN + "\n")
        self.env = {"GLIMMER_CONTROL": f"http://127.0.0.1:{self.server.port}/control"}
        self.recipe = self.work / "recipe"
        self.recipe.mkdir()
        (self.recipe / "run.sh").write_text("#!/bin/bash\necho hi\n")
        os.chmod(self.recipe / "run.sh", 0o755)
        (self.recipe / "cfg").mkdir()
        (self.recipe / "cfg" / "c.yaml").write_text("a: 1\n")

    def run_tool(self, *argv, env=None, cwd=None, **kw):
        out, err = io.StringIO(), io.StringIO()
        code = rt.main(list(argv), env=self.env if env is None else env, cwd=str(cwd or self.work), out=out, err=err, **kw)
        text = out.getvalue()
        doc = None
        if text:
            self.assertEqual(text.count("\n"), 1, text)  # exactly one JSON object on stdout
            doc = json.loads(text)
        return code, doc, err.getvalue()

    def git_init(self):
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "init", "-q", str(self.work)], check=True, env=env)
        subprocess.run(["git", "-C", str(self.work), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(self.work), "commit", "-qm", "recipe"], check=True, env=env)
        return subprocess.run(["git", "-C", str(self.work), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


class CommandTests(ToolCase):
    def test_budget(self):
        code, doc, _ = self.run_tool("budget")
        self.assertEqual(code, 0)
        self.assertEqual(doc["gpu_minutes_left"], 120.0)
        self.assertEqual(set(doc), {"researcher_tokens_left", "gpu_minutes_left", "dollars_left", "wall_seconds_left", "every_job_costs"})
        from nightshift.lab import researcher_tool as rt
        from nightshift.lab.trainer import GlimmerTrainer
        import inspect

        params = inspect.signature(GlimmerTrainer.__init__).parameters
        self.assertEqual(rt.JOB_OVERHEAD["adapter_pull_minutes"] * 60, params["pull_margin_s"].default)
        self.assertEqual(rt.JOB_OVERHEAD["dev_score_reserve_minutes"] * 60, params["eval_reserve_s"].default)

    def test_submit_wait_dev_score_status_pointer_roundtrip(self):
        code, doc, _ = self.run_tool("submit", "recipe", "--max-gpu-minutes", "30")
        self.assertEqual((code, doc["status"]), (0, "queued"))
        job_id = doc["job_id"]
        code, done, _ = self.run_tool("wait", job_id, "--interval", "0.05")
        self.assertEqual((code, done["state"]), (0, "ok"))
        self.assertEqual(self.fx.trainer.jobs[0].max_gpu_minutes, 30.0)
        aid = done["adapter_id"]
        code, ds, _ = self.run_tool("dev-score", aid)
        self.assertEqual((code, ds["adapter_id"], ds["metric"], ds["n_prompts"]), (0, aid, "prompt_all", 20))
        self.assertAlmostEqual(ds["dev_score"], done["dev_score"])
        code, st, _ = self.run_tool("status")
        self.assertEqual((code, st["state"], st["job_id"]), (0, "ok", job_id))
        commit = self.git_init()
        code, ptr, _ = self.run_tool("pointer", aid)
        self.assertEqual(code, 0)
        written = json.loads((self.work / "submission.json").read_text())
        self.assertEqual(written, ptr)
        self.assertEqual(set(written), {"adapter_id", "recipe_sha256", "commit", "dev_score"})
        self.assertEqual((written["adapter_id"], written["commit"]), (aid, commit))
        self.assertEqual(len(written["recipe_sha256"]), 64)
        # the file is a valid Submission as the contracts define it
        from nightshift.lab.contracts import Submission

        Submission(written["adapter_id"], written["recipe_sha256"], written["commit"], written["dev_score"])

    def test_pointer_outside_git_is_refused_and_writes_nothing(self):
        _c, doc, _ = self.run_tool("submit", "recipe")
        _c, done, _ = self.run_tool("wait", doc["job_id"], "--interval", "0.05")
        code, doc, _ = self.run_tool("pointer", done["adapter_id"])
        self.assertEqual(code, 1)
        self.assertIn("git", doc["error"])
        self.assertFalse((self.work / "submission.json").exists())

    def test_unknown_adapter_and_job_are_refused(self):
        self.assertEqual(self.run_tool("dev-score", "ad-0042")[0], 1)
        self.assertEqual(self.run_tool("pointer", "ad-0042")[0], 1)
        self.assertEqual(self.run_tool("status", "t-0001-j9")[0], 1)
        self.assertEqual(self.run_tool("wait", "t-0001-j9")[0], 1)

    def test_failed_job_makes_wait_exit_1(self):
        self.fx.trainer.fail_status = JobStatus.FAILED
        _c, doc, _ = self.run_tool("submit", "recipe")
        code, done, _ = self.run_tool("wait", doc["job_id"], "--interval", "0.05")
        self.assertEqual((code, done["state"]), (1, "failed"))

    def test_wait_times_out_with_exit_4(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.fx.trainer.block = gate
        _c, doc, _ = self.run_tool("submit", "recipe")
        ticks = iter(range(0, 1000))
        slept = []
        code, st, _ = self.run_tool("wait", doc["job_id"], "--timeout", "3", "--interval", "1", sleep=slept.append, monotonic=lambda: float(next(ticks)))
        self.assertEqual(code, 4)
        self.assertIn(st["state"], ("queued", "running"))
        self.assertTrue(st["timed_out"])
        self.assertTrue(slept)

    def test_budget_refusal_exits_1_with_the_reason(self):
        from nightshift.lab.contracts import Resource

        with self.assertRaises(BudgetExhausted):
            self.meter.charge(Resource.GPU_MINUTES, 120.0, "all")  # crosses the cap
        code, doc, _ = self.run_tool("submit", "recipe")
        self.assertEqual((code, doc["status"]), (1, "refused"))
        self.assertIn("budget exhausted", doc["reason"])
        self.assertEqual(self.fx.trainer.jobs, [])


class FailureModeTests(ToolCase):
    def test_usage_errors_exit_2(self):
        for argv in ([], ["nonsense"], ["submit"], ["wait"], ["dev-score"], ["submit", "recipe", "--bogus"], ["wait", "j", "--timeout", "abc"]):
            code, doc, err = self.run_tool(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIsNone(doc)

    def test_unreachable_control_api_exits_3(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        code, doc, _ = self.run_tool("budget", env={"GLIMMER_CONTROL": f"http://127.0.0.1:{port}/control"})
        self.assertEqual(code, 3)
        self.assertIn("unreachable", doc["error"])

    def test_missing_configuration_is_a_refusal_that_never_leaks_the_token(self):
        code, doc, _ = self.run_tool("budget", env={})
        self.assertEqual(code, 1)
        code, doc, _ = self.run_tool("budget", cwd=self.tmp)  # no .glimmer_token at or above the cwd
        self.assertEqual(code, 1)
        self.assertIn("token", doc["error"])

    def test_wrong_token_is_refused_and_not_echoed(self):
        (self.tmp / "work" / ".glimmer_token").write_text("not-the-token-0123456789\n")
        out, err = io.StringIO(), io.StringIO()
        code = rt.main(["budget"], env=self.env, cwd=str(self.work), out=out, err=err)
        self.assertEqual(code, 1)
        self.assertNotIn("not-the-token", out.getvalue() + err.getvalue())

    def test_token_file_env_overrides_the_search(self):
        other = self.tmp / "elsewhere.token"
        other.write_text(TOKEN)
        (self.tmp / "work" / ".glimmer_token").write_text("garbage-garbage-garbage\n")
        code, doc, _ = self.run_tool("budget", env={**self.env, "GLIMMER_TOKEN_FILE": str(other)})
        self.assertEqual(code, 0)

    def test_server_side_refusal_of_recipe_reaches_the_researcher(self):
        # a recipe the tool would pack but the control API refuses because of a tampered limit
        orig = control_api.MAX_RECIPE_ENTRIES
        control_api.MAX_RECIPE_ENTRIES = 1
        self.addCleanup(setattr, control_api, "MAX_RECIPE_ENTRIES", orig)
        code, doc, _ = self.run_tool("submit", "recipe")
        self.assertEqual((code, doc["status"]), (1, "refused"))
        self.assertIn("entries", doc["reason"])


class PackingTests(ToolCase):
    def members(self, data):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            return {m.name: m for m in tar.getmembers()}

    def test_deterministic_bytes_executable_bit_and_no_git(self):
        (self.recipe / ".git").mkdir()
        (self.recipe / ".git" / "config").write_text("[core]")
        a = rt.pack_recipe(str(self.recipe))
        os.utime(self.recipe / "cfg" / "c.yaml", (1, 1))  # timestamps do not matter
        b = rt.pack_recipe(str(self.recipe))
        self.assertEqual(a, b)
        self.assertEqual(a[4:8], b"\x00\x00\x00\x00")  # no timestamp in the gzip header
        m = self.members(a)
        self.assertEqual(sorted(m), ["cfg", "cfg/c.yaml", "run.sh"])
        self.assertEqual(m["run.sh"].mode, 0o755)
        self.assertEqual(m["cfg/c.yaml"].mode, 0o644)
        self.assertEqual((m["run.sh"].uid, m["run.sh"].mtime), (0, 0))
        # what the tool packs, the control API accepts
        validate_recipe_tarball(a, ("bash", "run.sh"))

    def test_symlink_special_file_and_missing_entrypoint_are_errors(self):
        os.symlink("/etc/passwd", self.recipe / "link")
        with self.assertRaises(rt.ToolError) as cm:
            rt.pack_recipe(str(self.recipe))
        self.assertIn("link", str(cm.exception))
        os.unlink(self.recipe / "link")
        os.mkfifo(self.recipe / "fifo")
        with self.assertRaises(rt.ToolError):
            rt.pack_recipe(str(self.recipe))
        os.unlink(self.recipe / "fifo")
        with self.assertRaises(rt.ToolError):
            rt.pack_recipe(str(self.recipe), "bash nothere.sh")
        with self.assertRaises(rt.ToolError):
            rt.pack_recipe(str(self.recipe), "bash ../run.sh")
        with self.assertRaises(rt.ToolError):
            rt.pack_recipe(str(self.work / "no-such-dir"))

    def test_symlinked_recipe_dir_entries_never_reach_a_submission(self):
        os.symlink("/etc/passwd", self.recipe / "sneaky")
        code, doc, _ = self.run_tool("submit", "recipe")
        self.assertEqual(code, 1)
        self.assertIn("symbolic link", doc["error"])
        self.assertEqual(self.fx.trainer.jobs, [])

    def test_limits_match_the_control_api(self):
        for name in ("MAX_RECIPE_ENTRIES", "MAX_RECIPE_COMPRESSED_BYTES", "MAX_RECIPE_UNPACKED_BYTES", "DEFAULT_ENTRYPOINT"):
            self.assertEqual(getattr(rt, name), getattr(control_api, name), name)
        self.assertEqual(rt._INTERPRETERS, control_api._INTERPRETERS)

    def test_entrypoint_script_agrees_with_the_server(self):
        for text in ("bash run.sh", "python3 -u ./cfg/train.py --epochs 3", "./go.sh", "sh x/y.sh"):
            self.assertEqual(rt.entrypoint_script(text), control_api.entrypoint_script(control_api.parse_entrypoint(text)))

    def test_module_uses_only_the_standard_library(self):
        src = Path(rt.__file__).read_text()
        self.assertNotIn("nightshift", src.replace("nightshift.lab.control_api", ""))  # only mentioned in comments/docstrings


if __name__ == "__main__":
    unittest.main()
