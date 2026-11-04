import gzip
import importlib.util
import io
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from nightshift.lab import runtime_coral as rc
from nightshift.lab.contracts import ExitReason, ResearcherRuntime, RuntimeFailure, RuntimeHandle, TrialContext
from nightshift.lab.control_api import ControlApi
from nightshift.lab.metering_proxy import LabHttpServer
from tests.support.lab_fake_runtime import FIXTURES, ControlClient, FakeLedger, FakeMeter, LocalRemote, TrialFixture, install_fake_coral, make_recipe

TOKEN = "trial-token-0123456789abcdef-XYZ"
SECRET = "META-KEY-should-never-reach-the-researcher-host"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def eventually(fn, timeout=20.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(interval)
    raise AssertionError("condition not reached in time")


def process_gone(pid):
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)  # children of this test process are zombies until reaped
    except ChildProcessError:
        reaped = 0  # not our child: only a failed kill(0) proves it is gone
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return bool(reaped)


# --------------------------------------------------------------------------------------------------
class ConfigRenderingTests(unittest.TestCase):
    def test_opencode_json_equals_the_probe_evidence(self):
        golden = json.loads((FIXTURES / "opencode.json").read_text())
        self.assertEqual(rc.render_opencode_json(4310), golden)
        doc = rc.render_opencode_json(5555)
        p = doc["provider"]["muse"]
        self.assertEqual(p["options"]["baseURL"], "http://127.0.0.1:5555/v1")
        self.assertEqual(p["models"]["muse-spark-1.3"]["limit"], {"context": 1048576, "output": 16000})
        self.assertNotIn("meta", doc["provider"])  # "meta" collides with a built-in opencode provider
        self.assertNotIn("permission", doc)  # CORAL writes its own permission block

    def test_litellm_config_routes_to_the_proxy_with_the_trial_token(self):
        text = rc.render_litellm_config(40123, TOKEN, "high")
        self.assertIn('api_base: "http://127.0.0.1:40123/v1"', text)
        self.assertIn(f'api_key: "{TOKEN}"', text)
        self.assertIn('model: "openai/muse-spark-1.3"', text)
        self.assertIn('      extra_body:\n        reasoning_effort: "high"', text)  # inside extra_body, not top level
        self.assertEqual(text.count("reasoning_effort"), 1)
        self.assertIn("drop_params: true", text)
        self.assertNotIn("api.meta.ai", text)

    def test_hostile_token_cannot_inject_yaml(self):
        text = rc.render_litellm_config(1, 'x"\nmodel_list: []\n  - evil: "1', "high")
        self.assertEqual(text.count("\nmodel_list"), 0)
        self.assertEqual([l for l in text.splitlines() if l.startswith("model_list")], ["model_list:"])

    def test_entrypoint_json_names_a_real_grader(self):
        spec = rc.load_entrypoint_spec()
        self.assertEqual(set(spec), {"entrypoint", "setup", "timeout"})
        self.assertRegex(spec["entrypoint"], r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
        self.assertTrue(all(isinstance(c, str) for c in spec["setup"]))
        self.assertGreater(spec["timeout"], 0)
        mod, cls = spec["entrypoint"].split(":")
        self.assertTrue((rc.GRADER_DIR / "src" / mod.replace(".", "/")).with_suffix(".py").is_file())
        self.assertIn(f"class {cls}", (rc.GRADER_DIR / "src" / mod.replace(".", "/")).with_suffix(".py").read_text())
        self.assertIn("./grader", spec["setup"][0])  # the runtime stages the package at <pack>/grader

    def test_overrides(self):
        spec = rc.load_entrypoint_spec()
        o = rc.coral_overrides("/srv/lab/trials/t-0001", 41000, max_real_attempts=7)
        d = dict(x.split("=", 1) for x in o)
        self.assertEqual(d["run.session"], "local")
        self.assertEqual(d["agents.runtime"], "opencode")
        self.assertEqual(d["agents.model"], "muse/muse-spark-1.3")
        self.assertEqual(d["agents.gateway.port"], "41000")
        self.assertEqual(d["agents.gateway.config"], "/srv/lab/trials/t-0001/runtime/litellm_config.yaml")
        self.assertEqual((d["grader.entrypoint"], json.loads(d["grader.setup"]), int(d["grader.timeout"])), (spec["entrypoint"], spec["setup"], spec["timeout"]))
        self.assertEqual(d["run.stop.max_real_attempts"], "7")
        self.assertEqual(d["workspace.run_dir"], "/srv/lab/trials/t-0001/run")
        self.assertEqual(json.loads(d["agents.runtime_options.mounts"]), {"/srv/lab/trials/t-0001/runtime/opencode.json": "opencode.json"})
        for key in ("workspace.results_dir", "workspace.repo_path", "workspace.run_dir"):
            self.assertTrue(d[key].startswith("/"), key)  # absolute: workspace paths resolve against coral's cwd
        self.assertNotIn("run.stop.max_real_attempts", dict(x.split("=", 1) for x in rc.coral_overrides("/r", 1)))

    def test_scripts_parse(self):
        launch = rc.render_launch_script("/srv/lab/trials/t-1", "/opt/coral", rc.coral_overrides("/srv/lab/trials/t-1", 41000), control_url="http://127.0.0.1:9/control")
        for name, text in (("launch", launch), ("poll", rc.POLL_SCRIPT), ("stop", rc.STOP_SCRIPT), ("remove", rc.REMOVE_SCRIPT), ("harvest", rc.HARVEST_SCRIPT.replace("%PATHS%", "a b")), ("listing", rc.LISTING_SCRIPT)):
            r = subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True)
            self.assertEqual(r.returncode, 0, (name, r.stderr))

    def test_launch_environment_is_built_from_scratch(self):
        launch = rc.render_launch_script("/srv/lab/trials/t-1", "/opt/coral", [], control_url="http://127.0.0.1:9/control")
        self.assertIn(' PATH="$ROOT/bin:/opt/coral/.venv/bin:', launch)  # the opencode PWD shim comes first
        for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
            self.assertIn(f'{var}="$ROOT/xdg/', launch)
        self.assertIn("env -i ", launch)  # nothing of the remote shell's environment is inherited
        for forbidden in ("MODEL_API_KEY", "META", "THUNDER", "API_KEY"):
            self.assertNotIn(forbidden, launch)
        evil = rc.render_launch_script("/r", "/c", ["a=$(touch /tmp/pwned)", "b=;rm -rf ~"], control_url="http://x/y'; id #")
        r = subprocess.run(["bash", "-n"], input=evil, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0)
        self.assertNotIn("$(touch /tmp/pwned)\n", evil.replace("'a=$(touch /tmp/pwned)'", ""))  # only ever inside quotes


class ParserTests(unittest.TestCase):
    def test_attempts_from_the_probe(self):
        a = rc.parse_attempt((FIXTURES / "attempt_improved.json").read_bytes())
        self.assertEqual(a["commit"], "8bc454ec26423751aa2d928ca32142f50ea20d84")
        self.assertEqual((a["status"], a["agent_id"]), ("improved", "captain-nemo"))
        self.assertTrue(a["timestamp"].startswith("2026-11-01T05:36:22"))
        p = rc.parse_attempt((FIXTURES / "attempt_pending.json").read_bytes())
        self.assertEqual(p["status"], "pending")
        for bad in (b"[]", b"nope", b'{"commit_hash": "xyz", "timestamp": "2026-11-01T00:00:00+00:00"}', b'{"commit_hash": "%s"}' % (b"a" * 40), b'{"commit_hash": "%s", "timestamp": "yesterday"}' % (b"a" * 40),
                    b'{"commit_hash": "%s", "timestamp": NaN}' % (b"a" * 40)):
            self.assertIsNone(rc.parse_attempt(bad), bad)

    def test_timestamps(self):
        self.assertEqual(rc.normalize_timestamp("2026-11-01T05:36:22.781753+00:00"), "2026-11-01T05:36:22.781753+00:00")
        self.assertEqual(rc.normalize_timestamp("2026-11-01T05:36:22"), "2026-11-01T05:36:22+00:00")
        self.assertIsNone(rc.normalize_timestamp(5))
        self.assertIsNone(rc.normalize_timestamp("not a time"))

    def test_submission_parser_ignores_the_researchers_dev_score(self):
        good = {"adapter_id": "ad-0001", "recipe_sha256": "a" * 64, "commit": "b" * 40, "dev_score": 0.99}
        self.assertEqual(rc.parse_submission(json.dumps(good).encode()), ("ad-0001", "a" * 64, "b" * 40))
        good["dev_score"] = 12345  # not even a valid score: still accepted, and still ignored
        self.assertEqual(rc.parse_submission(json.dumps(good).encode())[0], "ad-0001")
        for bad in ({}, {"adapter_id": "../x", "recipe_sha256": "a" * 64, "commit": "b" * 40}, {"adapter_id": "ad", "recipe_sha256": "short", "commit": "b" * 40},
                    {"adapter_id": "ad", "recipe_sha256": "a" * 64, "commit": "zz"}, {"adapter_id": 5, "recipe_sha256": "a" * 64, "commit": "b" * 40}):
            self.assertIsNone(rc.parse_submission(json.dumps(bad).encode()), bad)
        self.assertIsNone(rc.parse_submission(b"[1]"))
        self.assertIsNone(rc.parse_submission(b'{"adapter_id": "a", "adapter_id": "b"}'))  # duplicate keys

    def test_logs_are_concatenated_by_agent_and_numeric_restart_order(self):
        with tempfile.TemporaryDirectory() as t:
            logs = Path(t) / "logs"
            logs.mkdir()
            for name in ("zed.1.log", "a1.10.log", "a1.2.log", "a1.1.log", "notalog.txt"):
                (logs / name).write_text(json.dumps({"from": name}))  # no trailing newline on purpose
            os.symlink("/etc/hosts", logs / "a1.3.log")
            out = rc.concat_logs(Path(t)).decode().splitlines()
            marks = [json.loads(l) for l in out if '"log_file"' in l]
            self.assertEqual([(m["agent"], m["restart"]) for m in marks], [("a1", 1), ("a1", 2), ("a1", 10), ("zed", 1)])
            for line in out:
                json.loads(line)  # still JSONL
            self.assertNotIn("localhost", "\n".join(out))  # the symlink was not followed

    def test_listing_hash_depends_on_path_and_content(self):
        base = {"a/b": b"1", "c": b"2"}
        h = rc.listing_sha256(base)
        self.assertEqual(h, rc.listing_sha256(dict(reversed(list(base.items())))))
        self.assertNotEqual(h, rc.listing_sha256({"a/b": b"1", "c": b"3"}))
        self.assertNotEqual(h, rc.listing_sha256({"a/b": b"1", "d": b"2"}))
        self.assertNotEqual(h, rc.listing_sha256({**base, "e": b""}))


class SafeExtractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.canary = self.tmp / "private" / "canary.txt"
        self.canary.parent.mkdir()
        self.canary.write_text("CANARY-PRIVATE-DATA-8c1f2e")

    def tar(self, members):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for ti, data in members:
                t.addfile(ti, io.BytesIO(data) if data is not None else None)
        return buf.getvalue()

    @staticmethod
    def reg(name, data=b"x"):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        return ti, data

    @staticmethod
    def special(name, type_, link=""):
        ti = tarfile.TarInfo(name)
        ti.type, ti.linkname = type_, link
        return ti, None

    def files_under(self, d):
        return sorted(p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file())

    def test_only_regular_files_are_extracted_and_links_never_followed(self):
        data = self.tar([
            self.reg("attempts/a.json", b"{}"),
            self.special("attempts/evil.json", tarfile.SYMTYPE, str(self.canary)),
            self.special("notes/hl.md", tarfile.LNKTYPE, "attempts/a.json"),
            self.special("notes/dev", tarfile.CHRTYPE),
            self.special("notes/fifo", tarfile.FIFOTYPE),
            self.reg("../escape.txt"),
            self.reg("/abs.txt"),
            self.reg("a/../../escape2.txt"),
            self.reg("notes/ok.md", b"fine"),
        ])
        dest = self.tmp / "out"
        st = rc.safe_extract(data, dest)
        self.assertEqual(self.files_under(dest), ["attempts/a.json", "notes/ok.md"])
        self.assertFalse((self.tmp / "escape.txt").exists() or (self.tmp.parent / "escape.txt").exists() or Path("/abs.txt").exists() or (self.tmp / "escape2.txt").exists())
        self.assertEqual(st.files, 2)
        self.assertEqual(st.n_skipped, 7)
        for p in Path(dest).rglob("*"):
            self.assertFalse(p.is_symlink())
            self.assertNotIn("CANARY", p.read_text() if p.is_file() else "")

    def test_limits(self):
        data = self.tar([self.reg("big.bin", b"z" * 5000), self.reg("a", b"1"), self.reg("b", b"2"), self.reg("c", b"3")])
        st = rc.safe_extract(data, self.tmp / "o1", max_file_bytes=1000)
        self.assertEqual((st.files, st.n_skipped), (3, 1))
        st = rc.safe_extract(data, self.tmp / "o2", max_files=2, max_file_bytes=1000)
        self.assertEqual((st.files, st.n_skipped), (2, 2))
        st = rc.safe_extract(data, self.tmp / "o3", max_total=3, max_file_bytes=1000)
        self.assertEqual(st.files, 3)  # the three 1-byte files fit

    def test_existing_files_are_not_overwritten_or_followed(self):
        dest = self.tmp / "out"
        dest.mkdir()
        os.symlink(self.canary, dest / "a")
        data = self.tar([self.reg("a", b"overwrite attempt")])
        st = rc.safe_extract(data, dest)
        self.assertEqual(self.canary.read_text(), "CANARY-PRIVATE-DATA-8c1f2e")
        self.assertEqual(st.files, 0)

    def test_gzip_bomb_stops_and_garbage_is_refused(self):
        with mock.patch.object(rc, "MAX_HARVEST_BYTES", 1 << 20):
            big = tarfile.TarInfo("zeros.bin")
            big.size = 40 * 1024 * 1024
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as t:
                class Zeros(io.RawIOBase):
                    left = big.size
                    def readable(self): return True
                    def readinto(self, b):
                        n = min(len(b), self.left)
                        b[:n] = bytes(n)
                        self.left -= n
                        return n
                t.addfile(big, io.BufferedReader(Zeros()))
            st = rc.safe_extract(buf.getvalue(), self.tmp / "bomb", max_total=1 << 20)
            self.assertEqual(st.files, 0)
            self.assertGreaterEqual(st.n_skipped, 1)
        with self.assertRaises(RuntimeFailure):
            rc.safe_extract(b"this is not gzip", self.tmp / "g")

    def test_truncated_stream_keeps_what_arrived(self):
        data = self.tar([self.reg("a", b"1" * 100), self.reg("b", b"2" * 100)])
        raw = gzip.decompress(data)
        cut = gzip.compress(raw[: 512 + 512 + 600])
        st = rc.safe_extract(cut, self.tmp / "t")
        self.assertGreaterEqual(st.files, 1)
        self.assertGreaterEqual(st.n_skipped, 1)


# --------------------------------------------------------------------------------------------------
class AssetTests(unittest.TestCase):
    def test_shim_execs_the_next_opencode_with_the_real_cwd_in_pwd(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t).resolve()
            bin_dir, real_dir, work, stale = t / "bin", t / "real", t / "work", t / "stale"
            for d in (bin_dir, real_dir, work, stale):
                d.mkdir()
            shim = bin_dir / "opencode"
            shim.write_text(rc.SHIM_PATH.read_text())
            shim.chmod(0o755)
            fake = real_dir / "opencode"
            fake.write_text('#!/bin/bash\necho "pwd=$PWD args=$*"\n')
            fake.chmod(0o755)
            env = {"PATH": f"{bin_dir}:{real_dir}:/usr/bin:/bin", "PWD": str(stale), "HOME": str(t)}
            out = subprocess.run([str(shim), "run", "--model", "muse/x"], cwd=work, env=env, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.strip(), f"pwd={work} args=run --model muse/x")  # the stale PWD was repaired, the shim itself was skipped
            out = subprocess.run([str(shim)], cwd=work, env={**env, "PATH": f"{bin_dir}:/usr/bin:/bin"}, capture_output=True, text=True)
            self.assertEqual(out.returncode, 127)

    def test_staged_shim_is_the_asset(self):
        self.assertTrue(os.access(rc.SHIM_PATH, os.X_OK))
        self.assertTrue(rc.SHIM_PATH.read_text().startswith("#!/bin/bash"))


def load_grader_module(with_base: bool):
    spec = importlib.util.spec_from_file_location("glimmer_grader_under_test", rc.GRADER_DIR / "src" / "glimmer_grader" / "grader.py")
    mod = importlib.util.module_from_spec(spec)
    if with_base:
        class TaskGrader:  # the shape of coral.grader.TaskGrader that the grader relies on
            def __init__(self, codebase_path):
                self.codebase_path = codebase_path

            def score(self, value, explanation=""):
                return ("score", value, explanation)

            def fail(self, explanation=""):
                return ("fail", None, explanation)

        fake_coral = type(sys)("coral")
        fake_grader = type(sys)("coral.grader")
        fake_grader.TaskGrader = TaskGrader
        with mock.patch.dict(sys.modules, {"coral": fake_coral, "coral.grader": fake_grader}):
            spec.loader.exec_module(mod)
    else:
        spec.loader.exec_module(mod)
    return mod


class GraderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.fx = TrialFixture(self.tmp)
        spec = self.fx.spec("t-0001")
        self.api = ControlApi(self.fx.trainer, FakeMeter(spec.budget, clock=self.fx.clock), self.fx.layout, spec, TOKEN, FakeLedger(), clock=self.fx.clock)
        self.server = LabHttpServer([self.api]).start()
        self.addCleanup(self.server.stop)
        self.addCleanup(lambda: self.api.close(wait=False))
        self.url = f"http://127.0.0.1:{self.server.port}/control"
        self.client = ControlClient(self.url, TOKEN)
        self.mod = load_grader_module(with_base=False)
        self.tree = self.tmp / "checkout"
        self.tree.mkdir()

    def run_job(self):
        _s, doc = self.client.submit(make_recipe())
        return self.client.wait(doc["job_id"])

    def test_returns_the_control_hosts_dev_score_and_nothing_else(self):
        done = self.run_job()
        (self.tree / "submission.json").write_text(json.dumps({"adapter_id": done["adapter_id"], "recipe_sha256": "a" * 64, "commit": "b" * 40, "dev_score": 0.999}))
        score, expl = self.mod.evaluate_submission(str(self.tree), self.url, TOKEN)
        self.assertAlmostEqual(score, done["dev_score"])  # not the 0.999 the researcher wrote
        self.assertIn("researcher-influenced", expl)
        self.assertIn(done["adapter_id"], expl)
        self.assertNotIn("selection", expl.replace("not a selection or test score", ""))
        self.assertNotIn(str(self.fx.layout.root), expl)

    def test_failures_are_explained_and_score_none(self):
        s, e = self.mod.evaluate_submission(str(self.tree), self.url, TOKEN)
        self.assertIsNone(s)
        self.assertIn("submission.json", e)
        (self.tree / "submission.json").write_text("not json")
        self.assertIsNone(self.mod.evaluate_submission(str(self.tree), self.url, TOKEN)[0])
        (self.tree / "submission.json").write_text(json.dumps({"adapter_id": "ad-9999"}))
        s, e = self.mod.evaluate_submission(str(self.tree), self.url, TOKEN)
        self.assertIsNone(s)
        self.assertIn("unknown adapter", e)
        done = self.run_job()
        (self.tree / "submission.json").write_text(json.dumps({"adapter_id": done["adapter_id"]}))
        s, e = self.mod.evaluate_submission(str(self.tree), self.url, "wrong-token")
        self.assertIsNone(s)
        self.assertNotIn("wrong-token", e)
        port = free_port()
        s, e = self.mod.evaluate_submission(str(self.tree), f"http://127.0.0.1:{port}/control", TOKEN)
        self.assertIsNone(s)
        self.assertIn("unreachable", e)

    def test_submission_must_be_a_regular_small_file(self):
        canary = self.tmp / "canary.json"
        canary.write_text(json.dumps({"adapter_id": "ad-0001"}))
        os.symlink(canary, self.tree / "submission.json")
        with self.assertRaises(ValueError):
            self.mod.read_submission(str(self.tree))
        os.unlink(self.tree / "submission.json")
        (self.tree / "submission.json").write_text(" " * (self.mod.MAX_SUBMISSION_BYTES + 1))
        with self.assertRaises(ValueError):
            self.mod.read_submission(str(self.tree))
        (self.tree / "submission.json").write_text(json.dumps({"adapter_id": "../../etc/passwd"}))
        with self.assertRaises(ValueError):
            self.mod.read_submission(str(self.tree))

    def test_grader_class_reads_environment_and_wraps_with_the_coral_base(self):
        mod = load_grader_module(with_base=True)
        done = self.run_job()
        (self.tree / "submission.json").write_text(json.dumps({"adapter_id": done["adapter_id"]}))
        tokfile = self.tmp / "tok"
        tokfile.write_text(TOKEN + "\n")
        g = mod.Grader(str(self.tree))
        with mock.patch.dict(os.environ, {"GLIMMER_CONTROL": self.url, "GLIMMER_TOKEN_FILE": str(tokfile)}):
            kind, value, expl = g.evaluate()
        self.assertEqual(kind, "score")
        self.assertAlmostEqual(value, done["dev_score"])
        with mock.patch.dict(os.environ, {"GLIMMER_CONTROL": "", "GLIMMER_TOKEN_FILE": ""}):
            self.assertEqual(g.evaluate()[0], "fail")
        with mock.patch.dict(os.environ, {"GLIMMER_CONTROL": self.url, "GLIMMER_TOKEN_FILE": str(self.tmp / "missing")}):
            self.assertEqual(g.evaluate()[0], "fail")


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------------------------------
# the runtime against a local stand-in for the researcher host and a fake `coral`
# --------------------------------------------------------------------------------------------------
class RuntimeCase(unittest.TestCase):
    behavior = {"mode": "finish"}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        self.fx = TrialFixture(self.tmp)
        self.layout = self.fx.layout
        self.canary = self.tmp / "private" / "canary.txt"
        self.canary.parent.mkdir()
        self.canary.write_text("CANARY-PRIVATE-DATA-8c1f2e")
        self.trials_root = self.tmp / "srv" / "trials"
        self.trials_root.mkdir(parents=True)
        self.coral_home = self.tmp / "coral"
        self.behavior_path = install_fake_coral(self.coral_home, self.behavior)
        self.remote = LocalRemote(self.tmp / "home")
        self.runtime = rc.CoralRuntime(
            self.layout, self.remote, str(self.coral_home), {"coral_commit": "0123dfb939b35228cf2c1fde224cd0561e727408", "opencode_version": "1.19.3"}, free_port,
            clock=self.fx.clock, remote_trials_root=str(self.trials_root), max_real_attempts=5,
        )
        for aid in ("a-evo-1",):
            self.layout.train_data(aid).parent.mkdir(parents=True, exist_ok=True)
            self.layout.train_data(aid).write_text('{"key": "t1", "prompt": "p"}\n')
            self.layout.dev_data(aid).write_text('{"key": "d1", "prompt": "q"}\n')
        self.handles = []
        self.addCleanup(self.cleanup_handles)

    def cleanup_handles(self):
        for h in self.handles:
            h.close()
        for t in self.remote.tunnels:
            t.close()

    def set_behavior(self, **kw):
        self.behavior_path.write_text(json.dumps(kw))

    def context(self, trial_id="t-0001", token=TOKEN, pack_files=None):
        spec = self.fx.spec(trial_id)
        pack_dir = self.layout.trial_pack_dir(trial_id)
        pack_dir.mkdir(parents=True)
        self.fx.packs.checkout(spec.pack, pack_dir)
        for name, text in (pack_files or {}).items():
            (pack_dir / name).write_text(text)
        port = free_port()
        return TrialContext(spec, str(self.layout.trial_dir(trial_id)), f"http://127.0.0.1:{port}/v1", token, f"http://127.0.0.1:{port}/control")

    def start(self, **kw):
        ctx = self.context(**kw)
        h = self.runtime.start(ctx)
        self.handles.append(h)
        return ctx, h

    def wait_exit(self, h, timeout=30.0):
        return eventually(h.poll, timeout)

    def root_of(self, ctx):
        return self.trials_root / ctx.spec.trial_id

    def start_json(self, ctx):
        return json.loads(eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists() and (self.root_of(ctx) / "run" / "start.json").read_text()))


class StartTests(RuntimeCase):
    def test_protocols(self):
        self.assertIsInstance(self.runtime, ResearcherRuntime)
        ctx, h = self.start()
        self.assertIsInstance(h, RuntimeHandle)

    def test_staged_trial_directory_and_launch_environment(self):
        os.environ["META_API_KEY"] = SECRET
        self.addCleanup(os.environ.pop, "META_API_KEY", None)
        ctx, h = self.start()
        root = self.root_of(ctx)
        info = self.start_json(ctx)
        # what is on the researcher host
        self.assertEqual((root / ".glimmer_token").read_text(), TOKEN)
        self.assertEqual(oct((root / ".glimmer_token").stat().st_mode & 0o777), "0o600")
        self.assertEqual((root / "data" / "train.jsonl").read_bytes(), self.layout.train_data("a-evo-1").read_bytes())
        self.assertEqual((root / "data" / "dev.jsonl").read_bytes(), self.layout.dev_data("a-evo-1").read_bytes())
        self.assertTrue((root / "pack" / "task.yaml").is_file())
        self.assertTrue((root / "pack" / "grader" / "pyproject.toml").is_file())
        self.assertTrue((root / "pack" / "grader" / "src" / "glimmer_grader" / "grader.py").is_file())
        self.assertFalse((root / "pack" / "grader" / "ENTRYPOINT.json").exists())
        self.assertTrue(os.access(root / "bin" / "opencode", os.X_OK) and os.access(root / "bin" / "glimmer", os.X_OK))
        self.assertEqual((root / "tools" / "researcher_tool.py").read_bytes(), rc.TOOL_PATH.read_bytes())
        self.assertEqual((root / "runtime" / "opencode.json").read_bytes(), (root / "pack" / "seed" / "opencode.json").read_bytes())
        port_gw = json.loads((root / "runtime" / "opencode.json").read_text())["provider"]["muse"]["options"]["baseURL"].split(":")[-1].split("/")[0]
        lite = (root / "runtime" / "litellm_config.yaml").read_text()
        remote_port = lite.split("127.0.0.1:")[1].split("/")[0]
        self.assertIn(f'api_key: "{TOKEN}"', lite)
        self.assertNotEqual(port_gw, remote_port)
        self.assertEqual(oct((root / "runtime" / "litellm_config.yaml").stat().st_mode & 0o777), "0o600")
        # the command line coral was started with
        argv = info["argv"]
        self.assertEqual(argv[:3], ["start", "-c", "pack/task.yaml"])
        d = dict(a.split("=", 1) for a in argv[3:])
        self.assertEqual(d["agents.gateway.port"], port_gw)
        self.assertEqual(d["agents.gateway.config"], str(root / "runtime" / "litellm_config.yaml"))
        self.assertEqual(d["run.stop.max_real_attempts"], "5")
        self.assertEqual(Path(info["cwd"]).resolve(), root)
        # the environment it was started with: from scratch
        env = info["env"]
        self.assertEqual(env["PATH"].split(":")[0], str(root / "bin"))
        self.assertEqual(env["PATH"].split(":")[1], str(self.coral_home / ".venv" / "bin"))
        for v in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
            self.assertTrue(env[v].startswith(str(root / "xdg")), v)
        self.assertEqual(env["HOME"], str(root / "home"))  # a fresh per-trial home, not the shared user's persistent one
        self.assertNotEqual(env["HOME"], os.environ.get("HOME"))
        self.assertEqual(env["GLIMMER_CONTROL"], f"http://127.0.0.1:{remote_port}/control")
        self.assertEqual(env["GLIMMER_TOKEN_FILE"], str(root / ".glimmer_token"))
        self.assertEqual(env["GLIMMER_DATA_DIR"], str(root / "data"))
        blob = json.dumps(env)
        self.assertNotIn(SECRET, blob)
        self.assertNotIn("META_API_KEY", blob)
        self.assertNotIn(TOKEN, blob)  # the token is in a 0600 file, not in the environment
        # the tunnel: remote port -> the local port of the trial server
        self.assertEqual(len(self.remote.tunnels), 1)
        self.assertEqual(self.remote.tunnels[0].target, int(ctx.llm_base_url.split(":")[-1].split("/")[0]))

    def test_initial_context_hash_covers_pack_data_and_grader_but_not_per_trial_secrets(self):
        ctx1, h1 = self.start(trial_id="t-0001", token="token-one-0123456789abcdef")
        eventually(h1.poll)
        h1.close()
        ctx2, h2 = self.start(trial_id="t-0002", token="token-two-0123456789abcdef")
        self.assertEqual(h1.initial_context_sha256, h2.initial_context_sha256)  # other trial id, token and ports: same context
        eventually(h2.poll)
        h2.close()  # shared_user_serial: one trial at a time
        ctx3, h3 = self.start(trial_id="t-0003", pack_files={"seed_notes.md": "carried over from a previous trial"})
        self.assertNotEqual(h1.initial_context_sha256, h3.initial_context_sha256)
        h3.close()
        self.layout.dev_data("a-evo-1").write_text('{"key": "d1", "prompt": "changed"}\n')
        ctx4, h4 = self.start(trial_id="t-0004")
        self.assertNotEqual(h1.initial_context_sha256, h4.initial_context_sha256)

    def test_failures_to_start_leave_nothing_behind(self):
        self.remote.fail_tunnel = "ssh: connect refused"
        ctx = self.context()
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(ctx)
        self.assertFalse(self.root_of(ctx).exists())  # the staged directory was removed
        self.assertFalse((self.coral_home / "started").exists())
        self.remote.fail_tunnel = None
        ctx = self.context("t-0002")
        self.layout.dev_data("a-evo-1").unlink()
        calls_before = len(self.remote.calls)
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(ctx)
        self.assertEqual(len(self.remote.calls), calls_before)  # refused before anything was sent
        self.layout.dev_data("a-evo-1").write_text("{}\n")
        ctx = self.context("t-0003")
        os.symlink(self.canary, self.layout.trial_pack_dir("t-0003") / "link")
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(ctx)

    def test_endpoints_must_be_loopback_and_on_one_port(self):
        ctx = self.context()
        bad = TrialContext(ctx.spec, ctx.trial_dir, "http://10.0.0.5:9999/v1", ctx.llm_token, ctx.control_endpoint)
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(bad)
        bad = TrialContext(ctx.spec, ctx.trial_dir, ctx.llm_base_url, ctx.llm_token, "http://127.0.0.1:1/control")
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(bad)

    def test_constructor_accepts_a_ssh_target_string_and_validates(self):
        r = rc.CoralRuntime(self.layout, "researcher@ax41", "/opt/coral", {}, free_port, clock=self.fx.clock)
        self.assertIsInstance(r._remote, rc.SshRemote)
        with self.assertRaises(TypeError):
            rc.CoralRuntime(self.layout, object(), "/opt/coral", {}, free_port, clock=self.fx.clock)
        with self.assertRaises(ValueError):
            rc.CoralRuntime(self.layout, self.remote, "/opt/coral", {}, free_port, clock=self.fx.clock, remote_trials_root="relative/path")
        for bad in ('/opt/co"ral', "/opt/$(x)", "relative", "/opt/a b", "/opt/co'ral"):
            with self.assertRaises(ValueError, msg=bad):
                rc.CoralRuntime(self.layout, self.remote, bad, {}, free_port, clock=self.fx.clock)
        with self.assertRaises(ValueError):
            rc.CoralRuntime(self.layout, self.remote, "/opt/coral", {}, free_port, clock=self.fx.clock, remote_trials_root="/srv/$(x)")
        with self.assertRaises(ValueError):
            rc.SshRemote("-oProxyCommand=evil")


class FinishAndHarvestTests(RuntimeCase):
    behavior = {
        "mode": "finish",
        "attempts": [
            {"files": {"submission.json": json.dumps({"adapter_id": "ad-0001", "recipe_sha256": "a" * 64, "commit": "c" * 40, "dev_score": 0.99})}, "timestamp": "2026-11-01T05:36:20.000000+00:00"},
            {"files": {"submission.json": json.dumps({"adapter_id": "ad-0002", "recipe_sha256": "d" * 64, "commit": "e" * 40, "dev_score": None})}, "timestamp": "2026-11-01T05:36:21.500000+00:00"},
            {"files": {"submission.json": "this is not json"}, "timestamp": "2026-11-01T05:36:22.000000+00:00"},
            {"files": {"submission.json": json.dumps({"adapter_id": "../evil", "recipe_sha256": "a" * 64, "commit": "c" * 40})}, "timestamp": "2026-11-01T05:36:23.000000+00:00"},
            {"files": {}, "timestamp": "2026-11-01T05:36:24.000000+00:00", "status": "pending"},
        ],
    }

    def test_finished_run_is_harvested(self):
        ctx, h = self.start()
        self.assertEqual(self.wait_exit(h), ExitReason.FINISHED)
        res = h.collect()
        self.assertIs(res, h.collect())  # idempotent
        self.assertEqual(res.exit_reason, ExitReason.FINISHED)
        self.assertEqual(res.attempts, 4)  # the pending one is not counted
        # submissions are read from the attempts' own commits: the two valid ones, in commit-time order
        # (the fake commits submission.json into the repo, so later attempts also contain the earlier file: only the changed content counts)
        ids = [s.submission.adapter_id for s in res.submissions]
        self.assertEqual(ids[:2], ["ad-0001", "ad-0002"])
        for s in res.submissions:
            self.assertIsNone(s.submission.dev_score)  # the researcher's dev_score is discarded
        self.assertEqual(res.submissions[0].committed_utc, "2026-11-01T05:36:20+00:00")
        self.assertEqual(res.initial_context_sha256, h.initial_context_sha256)
        # trace: event logs concatenated by restart number
        trace = self.layout.root / res.trace.path
        marks = [json.loads(l) for l in trace.read_text().splitlines() if '"log_file"' in l]
        self.assertEqual([m["restart"] for m in marks], [1, 2, 10])
        self.assertEqual(self.layout.file_ref(trace), res.trace)
        self.assertIn("note one", (self.layout.root / res.notes.path).read_text())
        self.assertTrue((self.layout.trial_trace_dir("t-0001") / "public" / "attempts").is_dir())
        # environment labels
        env = res.environment
        self.assertEqual(env["isolation"], "shared_user_serial")
        self.assertEqual((env["coral_commit"], env["opencode_version"]), ("0123dfb939b35228cf2c1fde224cd0561e727408", "1.19.3"))
        self.assertEqual(env["coral_exit"], "0")
        self.assertEqual(env["grader_modified"], "false")
        self.assertEqual(env["harvest_skipped"], "0")
        for v in env.values():
            self.assertLessEqual(len(v.encode()), 500)

    def test_submission_comes_from_the_commit_not_the_live_worktree(self):
        ctx, h = self.start()
        self.wait_exit(h)
        repo = self.root_of(ctx) / "run" / "repo"
        # the researcher keeps editing after the last attempt; none of it was committed by an attempt
        (repo / "submission.json").write_text(json.dumps({"adapter_id": "ad-live", "recipe_sha256": "f" * 64, "commit": "f" * 40}))
        res = h.collect()
        self.assertNotIn("ad-live", [s.submission.adapter_id for s in res.submissions])

    def test_collect_before_exit_is_refused(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        self.assertIsNone(h.poll())
        with self.assertRaises(RuntimeFailure):
            h.collect()
        h.stop(ExitReason.STOPPED_WALL)

    def test_collect_with_nothing_to_harvest(self):
        ctx, h = self.start()
        self.wait_exit(h)
        h.close()  # the remote directory is gone
        with self.assertRaises(RuntimeFailure):
            h.collect()

    def test_close_removes_the_remote_tree_but_only_inside_the_trials_root(self):
        ctx, h = self.start()
        self.wait_exit(h)
        h.collect()
        root = self.root_of(ctx)
        self.assertTrue(root.exists())
        h.close()
        self.assertFalse(root.exists())
        h.close()  # idempotent
        # a handle pointed outside the trials root cannot delete anything
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep")
        rogue = rc.CoralHandle(remote=self.remote, layout=self.layout, root=str(outside), base=str(self.trials_root), tunnel=None, trial_id="t-9999", initial_context_sha256="0" * 64, grader_hashes={}, environment={}, clock=self.fx.clock)
        rogue.close()
        self.assertEqual((outside / "keep.txt").read_text(), "keep")


class PlantedLinkTests(RuntimeCase):
    def test_links_and_special_files_in_harvest_paths_never_reach_the_control_host(self):
        hard = self.tmp / "hardtarget.txt"
        hard.write_text("CANARY-PRIVATE-DATA-8c1f2e")
        self.set_behavior(
            mode="finish",
            attempts=[{"files": {"submission.json": json.dumps({"adapter_id": "ad-0001", "recipe_sha256": "a" * 64, "commit": "c" * 40})}}],
            plant=[
                {"kind": "symlink", "path": "attempts/evil.json", "target": str(self.canary)},
                {"kind": "symlink", "path": "notes/leak.md", "target": str(self.canary)},
                {"kind": "symlink", "path": "logs/a1.5.log", "target": str(self.canary)},
                {"kind": "symlink", "path": "notes/dir", "target": str(self.canary.parent)},
                {"kind": "hardlink", "path": "notes/hard.md", "target": str(hard)},
                {"kind": "fifo", "path": "notes/pipe"},
                {"kind": "symlink", "path": "sessions.json", "target": str(self.canary)},
            ],
        )
        ctx, h = self.start()
        self.wait_exit(h)
        res = h.collect()
        for p in Path(self.layout.root).rglob("*"):
            if p.is_file():
                self.assertNotIn(b"CANARY-PRIVATE-DATA", p.read_bytes(), str(p))
            self.assertFalse(p.is_symlink(), str(p))
        pub = self.layout.trial_trace_dir("t-0001") / "public"
        for planted in ("attempts/evil.json", "notes/leak.md", "logs/a1.5.log", "notes/dir", "notes/hard.md", "notes/pipe", "sessions.json"):
            self.assertFalse((pub / planted).exists(), planted)  # filtered on the remote side: never in the stream
        self.assertTrue((pub / "notes" / "n1.md").is_file())
        self.assertEqual([s.submission.adapter_id for s in res.submissions], ["ad-0001"])  # the real attempt still came through


class StopAndCrashTests(RuntimeCase):
    def test_stop_kills_the_run_and_stragglers_inside_the_trial_directory_only(self):
        decoy = subprocess.Popen(["sleep", "300"], cwd=str(self.tmp), start_new_session=True)
        self.addCleanup(decoy.kill)
        self.set_behavior(mode="hang", straggler=True, extra_pids=[decoy.pid])
        ctx, h = self.start()
        pids = eventually(lambda: (self.root_of(ctx) / "run" / ".coral" / "public" / "agent.pids").exists() and (self.root_of(ctx) / "run" / ".coral" / "public" / "agent.pids").read_text())
        straggler = int(pids.split()[0])
        self.assertIsNone(h.poll())
        main_pid = int((self.root_of(ctx) / "coral.pid").read_text())
        os.kill(straggler, 0)
        h.stop(ExitReason.STOPPED_BUDGET)
        self.assertEqual(h.poll(), ExitReason.STOPPED_BUDGET)
        h.stop(ExitReason.STOPPED_WALL)  # idempotent: the first reason stands
        self.assertEqual(h.poll(), ExitReason.STOPPED_BUDGET)

        eventually(lambda: process_gone(main_pid), 10)
        eventually(lambda: process_gone(straggler), 10)
        self.assertIsNone(decoy.poll())  # a pid listed in agent.pids that lives outside the trial directory is left alone
        res = h.collect()
        self.assertEqual(res.exit_reason, ExitReason.STOPPED_BUDGET)

    def test_processes_that_start_their_own_session_and_leave_the_trial_directory_are_killed(self):
        for kind in ("child", "orphan"):
            with self.subTest(kind=kind):
                self.set_behavior(mode="hang", detached=kind)
                ctx, h = self.start(trial_id="t-" + kind)
                pidfile = self.root_of(ctx) / "run" / "detached.pid"
                pid = int(eventually(lambda: pidfile.exists() and pidfile.read_text().strip()))
                self.addCleanup(lambda p=pid: os.kill(p, signal.SIGKILL) if not process_gone(p) else None)
                os.kill(pid, 0)
                self.assertNotIn(str(self.root_of(ctx)), os.readlink("/proc/%d/cwd" % pid) if os.path.exists("/proc/%d/cwd" % pid) else "/")
                h.stop(ExitReason.STOPPED_BUDGET)
                eventually(lambda: process_gone(pid), 10)
                h.close()

    def test_a_run_that_ignores_sigterm_is_killed(self):
        self.set_behavior(mode="hang", ignore_term=True)
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        main_pid = int((self.root_of(ctx) / "coral.pid").read_text())
        h.stop(ExitReason.STOPPED_WALL)
        eventually(lambda: process_gone(main_pid), 10)

    def test_stop_after_the_run_ended_on_its_own_keeps_the_natural_reason(self):
        ctx, h = self.start()
        self.assertEqual(self.wait_exit(h), ExitReason.FINISHED)
        h.stop(ExitReason.STOPPED_BUDGET)
        self.assertEqual(h.poll(), ExitReason.FINISHED)

    def test_crash_is_reported_with_the_exit_status(self):
        self.set_behavior(mode="crash")
        ctx, h = self.start()
        self.assertEqual(self.wait_exit(h), ExitReason.CRASHED)
        res = h.collect()
        self.assertEqual(res.exit_reason, ExitReason.CRASHED)
        self.assertEqual(res.environment["coral_exit"], "3")
        self.assertEqual(res.submissions, ())

    def test_a_vanished_process_without_exit_status_is_a_crash(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        os.killpg(int((self.root_of(ctx) / "coral.pid").read_text()), signal.SIGKILL)
        self.assertEqual(self.wait_exit(h), ExitReason.CRASHED)
        self.assertEqual(h.collect().environment["coral_exit"], "vanished")

    def test_a_dead_tunnel_stops_the_run_and_reports_a_crash(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        main_pid = int((self.root_of(ctx) / "coral.pid").read_text())
        self.remote.tunnels[0].close()
        self.assertEqual(h.poll(), ExitReason.CRASHED)
        eventually(lambda: process_gone(main_pid), 10)  # without a tunnel the run is useless: it was stopped, not just reported
        self.assertEqual(h.collect().environment["tunnel"], "died")

    def test_unreachable_host_eventually_counts_as_a_crash(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        real_run = self.remote.run

        def broken(argv, **kw):
            raise rc.RemoteError("ssh: no route")

        self.remote.run = broken
        for _ in range(9):
            self.assertIsNone(h.poll())
        self.assertEqual(h.poll(), ExitReason.CRASHED)
        self.remote.run = real_run
        h.stop(ExitReason.STOPPED_WALL)


class SerialIsolationTests(RuntimeCase):
    def test_a_second_trial_cannot_start_while_the_first_holds_the_shared_user(self):
        ctx1, h1 = self.start(trial_id="t-0001", token="token-one-0123456789abcdef")
        ctx2 = self.context("t-0002", token="token-two-0123456789abcdef")
        with self.assertRaisesRegex(RuntimeFailure, "one trial at a time"):
            self.runtime.start(ctx2)
        self.assertFalse(self.root_of(ctx2).exists())  # nothing of the second trial reached the researcher host
        self.assertEqual((self.root_of(ctx1) / ".glimmer_token").read_text(), "token-one-0123456789abcdef")
        h1.close()
        ctx3, h3 = self.start(trial_id="t-0003")  # the host is free again once the first trial's processes are gone
        self.assertIsNone(h3.poll())

    def _second_runtime(self):
        return rc.CoralRuntime(self.layout, self.remote, str(self.coral_home), {}, free_port, clock=self.fx.clock, remote_trials_root=str(self.trials_root))

    def test_a_restarted_coordinator_cannot_start_a_trial_beside_a_surviving_one(self):
        ctx1, h1 = self.start(trial_id="t-0001", token="token-one-0123456789abcdef")
        restarted = self._second_runtime()  # the first coordinator is gone; its detached trial is not
        ctx2 = self.context("t-0002", token="token-two-0123456789abcdef")
        with self.assertRaisesRegex(RuntimeFailure, r"t-0001.*one trial at a time|one trial at a time.*t-0001"):
            restarted.start(ctx2)
        self.assertFalse(self.root_of(ctx2).exists())  # nothing of the second trial reached the host
        self.assertEqual((self.root_of(ctx1) / ".glimmer_token").read_text(), "token-one-0123456789abcdef")
        # the refusal did not give the survivor's claim away
        with self.assertRaisesRegex(RuntimeFailure, "one trial at a time"):
            self._second_runtime().start(self.context("t-0003"))
        h1.close()  # a clean close frees the host for any coordinator
        h3 = self._second_runtime().start(self.context("t-0004"))
        self.handles.append(h3)
        self.assertIsNone(h3.poll())

    def test_a_retry_of_the_same_trial_id_after_a_crash_is_refused_and_keeps_the_claim(self):
        ctx1, h1 = self.start(trial_id="t-0001")
        with self.assertRaisesRegex(RuntimeFailure, "one trial at a time"):
            self._second_runtime().start(ctx1)
        with self.assertRaisesRegex(RuntimeFailure, "one trial at a time"):
            self._second_runtime().start(self.context("t-0002"))
        h1.close()

    def test_a_failed_start_gives_the_host_back(self):
        self.remote.fail_tunnel = "ssh: connect refused"
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(self.context("t-0001"))
        self.remote.fail_tunnel = None
        ctx, h = self.start(trial_id="t-0002")
        self.assertIsNone(h.poll())

    def test_other_isolation_labels_are_the_callers_assertion_and_are_not_serialised(self):
        runtime = rc.CoralRuntime(
            self.layout, self.remote, str(self.coral_home), {}, free_port, clock=self.fx.clock, remote_trials_root=str(self.trials_root), isolation="per_trial_user",
        )
        h1 = runtime.start(self.context("t-0001"))
        h2 = runtime.start(self.context("t-0002"))
        self.handles += [h1, h2]
        self.assertEqual(h1._env["isolation"], "per_trial_user")

    def test_a_trial_whose_processes_could_not_be_confirmed_stopped_keeps_the_host_and_its_pid_files(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start(trial_id="t-0001")
        eventually(lambda: (self.root_of(ctx) / "coral.pid").exists())
        real_run = self.remote.run

        def no_stop(argv, **kw):
            if len(argv) > 2 and argv[2] == rc.STOP_SCRIPT:
                raise rc.RemoteError("ssh: no route")
            return real_run(argv, **kw)

        self.remote.run = no_stop
        h.close()
        self.assertTrue((self.root_of(ctx) / "coral.pid").exists())  # not deleted: a later cleanup needs it
        with self.assertRaisesRegex(RuntimeFailure, "could not be confirmed stopped"):
            self.runtime.start(self.context("t-0002"))
        self.remote.run = real_run
        self.assertTrue(rc._stop_remote(self.remote, str(self.root_of(ctx))))  # cleanup by hand


class StopScriptOwnershipTests(unittest.TestCase):
    def test_a_pid_file_naming_a_foreign_process_group_is_not_signalled(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "trial").resolve()
            root.mkdir()
            other = Path(d, "elsewhere").resolve()
            other.mkdir()
            code = "import os,subprocess,sys,time;os.setsid();subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);print(os.getpid(),flush=True);time.sleep(60)"
            foreign = subprocess.Popen([sys.executable, "-c", code], cwd=other, stdout=subprocess.PIPE, text=True, env={"PATH": os.environ["PATH"]})
            try:
                fpid = int(foreign.stdout.readline())
                (root / "coral.pid").write_text(str(fpid))  # tampering or pid reuse
                r = subprocess.run(["bash", "-c", rc.STOP_SCRIPT, "_", str(root)], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0, r.stderr)
                time.sleep(0.5)
                self.assertIsNone(foreign.poll(), "the foreign process group leader was signalled")
                kids = subprocess.run(["pgrep", "-g", str(fpid)], capture_output=True, text=True).stdout.split()
                self.assertGreaterEqual(len(kids), 2, "a member of the foreign process group was signalled")
            finally:
                try:
                    os.killpg(foreign.pid, 9)
                except OSError:
                    pass
                foreign.wait()


class LaunchHygieneTests(RuntimeCase):
    def test_nothing_of_the_remote_shells_environment_reaches_coral(self):
        self.remote.env["PRIVATE_CANARY_TOKEN"] = "canary-value-7f3a"
        self.remote.env["SSH_AUTH_SOCK"] = "/tmp/agent.sock"
        ctx, h = self.start()
        env = self.start_json(ctx)["env"]
        self.assertNotIn("PRIVATE_CANARY_TOKEN", env)
        self.assertNotIn("canary-value-7f3a", json.dumps(env))
        self.assertNotIn("SSH_AUTH_SOCK", env)
        self.assertEqual(env["GLIMMER_TRIAL_ROOT"], str(self.root_of(ctx)))


class LostLaunchAcknowledgmentTests(RuntimeCase):
    def test_coral_that_started_but_was_never_acknowledged_is_stopped_before_its_directory_is_deleted(self):
        self.set_behavior(mode="hang")
        real_run = self.remote.run
        seen = {}
        calls = []

        def lose_the_ack(argv, **kw):
            calls.append(argv[2] if len(argv) > 2 else "")
            res = real_run(argv, **kw)
            if len(argv) > 2 and argv[2].startswith("#!/bin/bash"):
                pidfile = Path(argv[2].split("export ROOT=")[1].split("\n")[0]) / "coral.pid"
                seen["pid"] = int(pidfile.read_text())
                raise rc.RemoteError("command timed out after 120s")
            return res

        self.remote.run = lose_the_ack
        ctx = self.context()
        with self.assertRaises(rc.RemoteError):
            self.runtime.start(ctx)
        self.remote.run = real_run
        eventually(lambda: process_gone(seen["pid"]), 10)  # it was running when the ack was lost
        self.assertFalse(self.root_of(ctx).exists())
        stops = [i for i, c in enumerate(calls) if c == rc.STOP_SCRIPT]
        removes = [i for i, c in enumerate(calls) if c == rc.REMOVE_SCRIPT]
        self.assertTrue(stops and removes and stops[0] < removes[0], (stops, removes))  # stopped first, deleted second
        ctx2, h2 = self.start(trial_id="t-0002")  # the host was handed back
        self.assertIsNone(h2.poll())


class TamperTests(RuntimeCase):
    def test_a_modified_grader_is_reported(self):
        self.set_behavior(mode="finish", tamper_grader=True)
        ctx, h = self.start()
        self.wait_exit(h)
        res = h.collect()
        self.assertEqual(res.environment["grader_modified"], "true")
        self.assertGreaterEqual(int(res.environment["grader_changed_files"]), 1)

    def test_a_deleted_grader_is_reported(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        import shutil

        shutil.rmtree(self.root_of(ctx) / "pack" / "grader")
        h.stop(ExitReason.STOPPED_WALL)
        self.assertEqual(h.collect().environment["grader_modified"], "true")


class RemoteHelperTests(unittest.TestCase):
    def test_run_capped_enforces_timeout_and_output_cap_and_passes_stdin(self):
        r = rc.run_capped(["cat"], input=b"hello", timeout=10)
        self.assertEqual((r.returncode, r.stdout), (0, b"hello"))
        with self.assertRaises(rc.RemoteError):
            rc.run_capped(["sleep", "30"], timeout=0.3)
        with self.assertRaises(rc.RemoteError):
            rc.run_capped([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 5_000_000)"], timeout=20, max_output=1000)
        with self.assertRaises(rc.RemoteError):
            rc.run_capped(["/no/such/binary"], timeout=5)

    def test_run_capped_timeout_covers_a_descendant_that_keeps_the_output_pipes_open(self):
        # the shell exits at once; its background child inherits stdout/stderr and lives for 4s
        started = time.monotonic()
        with self.assertRaises(rc.RemoteError) as cm:
            rc.run_capped(["sh", "-c", "sleep 4 & exit 0"], timeout=0.3)
        self.assertIn("timed out", str(cm.exception))
        self.assertLess(time.monotonic() - started, 2.5)

    def test_ssh_remote_builds_a_batch_mode_command(self):
        seen = {}
        with mock.patch.object(rc, "run_capped", lambda cmd, **kw: seen.update(cmd=cmd, kw=kw) or rc.RemoteResult(0, b"", b"")):
            rc.SshRemote("researcher@ax41", port=2222, identity="/k").run(["bash", "-c", "echo $1", "_", "a b"], timeout=5)
        cmd = seen["cmd"]
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("BatchMode=yes", cmd)
        self.assertEqual(cmd[cmd.index("--") + 1], "researcher@ax41")
        self.assertEqual(shlex.split(cmd[-1]), ["bash", "-c", "echo $1", "_", "a b"])  # the remote shell sees exactly the argv
        self.assertIn("2222", cmd)


if __name__ == "__main__":
    unittest.main()
