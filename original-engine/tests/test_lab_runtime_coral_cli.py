"""Tests of the CLI-backed researcher runtime (nightshift/lab/runtime_coral_cli.py) and its CORAL custom runtime for Muse Code
(nightshift/lab/assets/coral_ext/glimmer_muse_runtime.py).

There is no docker and no Muse on the test machine: a small fake ``docker`` (a python script in a scratch directory) runs the container's command
as an ordinary process with exactly the environment the ``-e`` flags name, records what it was asked, and answers ``inspect`` / ``ps`` / ``stop`` /
``kill`` / ``rm``; a fake ``muse`` records its argv and environment. The real container and the real Muse are exercised only by the live probe.
Every test input is data handed to a function or written to a file; nothing hostile is given to a shell that is not the code under test.
"""

import dataclasses
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

from nightshift.lab import runtime_coral as rc
from nightshift.lab import runtime_coral_cli as cli
from nightshift.lab.contracts import ExitReason, ResearcherRuntime, RuntimeFailure, RuntimeHandle
from tests.support.lab_fake_runtime import FIXTURES, LocalRemote, TrialFixture, install_fake_coral
from tests.test_lab_runtime_coral import TOKEN, RuntimeCase, eventually, free_port

# the content of the (never parsed) Muse login file: token-like strings of several shapes
SECRET_A = "mc_live_0123456789ABCDEFghijklmnopqrstuvwxyz_QRSTUV-9876543210"
SECRET_B = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature-0123456789abcdefghij"
AUTH_BYTES = json.dumps({"access": SECRET_A, "refresh": SECRET_B, "who": "test", "n": 7}, indent=1).encode() + b"\n"
OTHER = "mc_live_OTHERACCOUNT0123456789abcdefghijklmnopqrstuvwx"


FAKE_DOCKER = r'''#!{python}
import json, os, pathlib, signal, subprocess, sys, time
STATE = pathlib.Path({state!r}); STATE.mkdir(exist_ok=True)
args = sys.argv[1:]
def rec(): return STATE / "calls.jsonl"
with open(rec(), "a") as fh: fh.write(json.dumps(args) + "\n")
def alive(pid):
    try: os.kill(pid, 0)
    except OSError: return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0]: return False
    except ChildProcessError: pass
    return True
def info(name):
    p = STATE / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None
def kill(name, sig):
    i = info(name)
    if i:
        try: os.killpg(i["pid"], sig)
        except (ProcessLookupError, PermissionError): pass
def running(name):
    i = info(name)
    return bool(i) and alive(i["pid"])
cmd = args[0] if args else ""
if (STATE / "daemon_down").exists() and cmd != "image":
    sys.stderr.write("Cannot connect to the Docker daemon\n"); sys.exit(1)
if cmd == "version": sys.exit(0)
if cmd == "image" and args[1] == "inspect":
    sys.exit(1 if (STATE / "missing_image").exists() else 0)
if cmd == "rm":
    name = args[-1]; kill(name, signal.SIGKILL); time.sleep(0.05)
    if not (STATE / "rm_fails").exists():
        (STATE / (name + ".json")).unlink(missing_ok=True)
    sys.exit(0)
if cmd in ("stop", "kill"):
    name = args[-1]
    if cmd == "stop" and not (STATE / "stop_ignored").exists():
        kill(name, signal.SIGTERM)
        for _ in range(40):
            if not running(name): break
            time.sleep(0.05)
        if running(name): kill(name, signal.SIGKILL)
    elif cmd == "kill":
        kill(name, signal.SIGKILL)
    sys.exit(0)
if cmd == "ps":
    allc = "-a" in args or "-aq" in args
    flt = [a for a in args if a.startswith("name=")][0][len("name=^/"):-1]
    i = info(flt)
    if i and (allc or running(flt)): print("cid-" + flt)
    sys.exit(0)
if cmd == "inspect":
    fmt, name = args[2], args[3]
    i = info(name)
    if not i: sys.exit(1)
    st = "running" if running(name) else "exited"
    ef = pathlib.Path(i["exitfile"])
    code = ef.read_text().strip() if ef.exists() else "137"
    print(st if "Status" in fmt else code); sys.exit(0)
if cmd == "run":
    opts = {{"v": [], "e": []}}; i = 1; image = None
    flags_with_value = {{"--name", "--user", "--network", "--cap-drop", "--security-opt", "--pids-limit", "--memory", "--memory-swap", "--cpus", "--tmpfs", "-v", "-e", "-w"}}
    while i < len(args):
        a = args[i]
        if a in flags_with_value:
            if a == "-v": opts["v"].append(args[i + 1])
            elif a == "-e": opts["e"].append(args[i + 1])
            else: opts[a] = args[i + 1]
            i += 2
        elif a.startswith("-"): opts[a] = True; i += 1
        else: image = a; i += 1; break
    command = args[i:]
    name = opts["--name"]
    env = dict(e.split("=", 1) for e in opts["e"])
    fakebin = {fakebin!r}
    env["PATH"] = fakebin + ":" + env["PATH"]
    env["FAKE_STATE"] = str(STATE)
    exitfile = STATE / (name + ".exit")
    exitfile.unlink(missing_ok=True)
    wrapper = 'inner="$1"; shift; bash -c "$inner"; echo $? > "$EXITFILE"'
    env["EXITFILE"] = str(exitfile)
    p = subprocess.Popen(["bash", "-c", 'bash -c "$0"; echo $? > "$EXITFILE"', command[-1]], env=env, cwd=opts["-w"], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (STATE / (name + ".json")).write_text(json.dumps({{"pid": p.pid, "exitfile": str(exitfile), "opts": opts, "image": image, "command": command}}))
    print("cid-" + name); sys.exit(0)
sys.stderr.write("fake docker: unsupported " + repr(args) + "\n"); sys.exit(64)
'''

FAKE_TIMEOUT = r'''#!/bin/bash
# records its arguments, then runs the command (the macOS test host has no coreutils timeout)
echo "$@" >> "$FAKE_STATE/timeout.args"
while [ "$#" -gt 0 ]; do case "$1" in -s|-k) shift 2;; [0-9]*) shift; break;; *) break;; esac; done
exec "$@"
'''




def write_muse_home(home: Path, auth: bytes = AUTH_BYTES, version: str = "1.4.14-R7309.1") -> Path:
    """A scratch ``muse-home``: bin/ (a fake launcher), auth/auth.json, and a cfg/ the container must never see."""
    (home / "bin").mkdir(parents=True, exist_ok=True)
    (home / "auth").mkdir(exist_ok=True)
    (home / "cfg").mkdir(exist_ok=True)
    launcher = home / "bin" / "muse"
    launcher.write_text("#!/bin/bash\necho fake muse\n")
    launcher.chmod(0o755)
    (home / "bin" / ".muse-version").write_text(version + "\n")
    (home / "auth" / "auth.json").write_bytes(auth)
    (home / "auth" / "auth.json").chmod(0o600)
    (home / "cfg" / "private.txt").write_text("PRIVATE-CFG-9d2c")
    return home


def tree_contains(root: Path, needle: bytes) -> list[str]:
    hits = []
    for p in Path(root).rglob("*"):
        if p.is_file() and not p.is_symlink() and needle in p.read_bytes():
            hits.append(str(p))
    return hits


def shlex_split(script: str) -> list[str]:
    import shlex

    out = []
    for line in script.splitlines():
        if line.startswith("docker run"):
            toks = shlex.split(line)
            out += [toks[i + 1] for i, t in enumerate(toks) if t == "-v" and "$UV" not in toks[i + 1]]
    return out


# --------------------------------------------------------------------------------------------------
class PureTests(unittest.TestCase):
    def test_overrides_select_the_muse_runtime_model_and_no_gateway(self):
        spec = rc.load_entrypoint_spec()
        o = cli.coral_cli_overrides("/srv/lab/trials/t-0001", model="muse-spark-1.3", max_real_attempts=3, reasoning_effort="medium")
        d = dict(x.split("=", 1) for x in o)
        self.assertEqual((d["agents.runtime"], d["agents.model"], d["agents.gateway.enabled"]), ("glimmer_muse_runtime:MuseCodeRuntime", "muse-spark-1.3", "false"))
        self.assertEqual(d["agents.runtime_options.reasoning_effort"], "medium")
        self.assertEqual(sum(1 for x in o if x.startswith("agents.runtime=")), 1)
        for k in d:
            self.assertFalse(k.startswith("agents.gateway.") and k != "agents.gateway.enabled", k)  # no port, no config, no key
        self.assertNotIn("agents.runtime_options.mounts", d)  # opencode-only
        self.assertEqual((d["grader.entrypoint"], json.loads(d["grader.setup"]), int(d["grader.timeout"])), (spec["entrypoint"], spec["setup"], spec["timeout"]))
        self.assertEqual(d["run.session"], "local")  # not CORAL's own docker session, which mounts the whole ~/.claude
        self.assertEqual(d["run.stop.max_real_attempts"], "3")
        for key in ("workspace.results_dir", "workspace.repo_path", "workspace.run_dir"):
            self.assertTrue(d[key].startswith("/srv/lab/trials/t-0001/"), key)
        w = dict(x.split("=", 1) for x in cli.coral_cli_overrides("/srv/lab/trials/t-0001", model="muse-wonder-full-1.0"))
        self.assertEqual(w["agents.model"], "muse-wonder-full-1.0")
        self.assertEqual(cli.DEFAULT_MODEL, "muse-spark-1.3")  # old roots keep their default
        plain = dict(x.split("=", 1) for x in cli.coral_cli_overrides("/r"))
        self.assertNotIn("run.stop.max_real_attempts", plain)
        self.assertNotIn("agents.runtime_options.reasoning_effort", plain)
        for bad in ("x; rm -rf ~", "", "$(id)", "a b"):
            with self.assertRaises(ValueError):
                cli.coral_cli_overrides("/r", model=bad)
        with self.assertRaises(ValueError):
            cli.coral_cli_overrides("/r", reasoning_effort="maximum overdrive")

    def test_scripts_parse(self):
        launch = cli.render_container_launch_script("/srv/lab/trials/t-1", "/opt/coral", cli.coral_cli_overrides("/srv/lab/trials/t-1"), control_url="http://127.0.0.1:9/control", name="glimmer-t-1", muse_bin_dir="/srv/muse/bin")
        install = cli.render_install_script("/srv/work/coral")
        for name, text in (("launch", launch), ("poll", cli.CLI_POLL_SCRIPT), ("stop", cli.CLI_STOP_SCRIPT), ("remove", cli.CLI_REMOVE_SCRIPT), ("cmp", cli.AUTH_CMP_SCRIPT), ("install", install)):
            r = subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True)
            self.assertEqual(r.returncode, 0, (name, r.stderr))
        for name, text in (("auth", cli._AUTH_SCRIPT), ("scan", cli._SCAN_SCRIPT)):
            compile(text, name, "exec")

    def test_launch_script_is_a_contained_container_with_exactly_the_planned_mounts(self):
        s = cli.render_container_launch_script("/srv/lab/trials/t-1", "/opt/coral", ["a=b"], control_url="http://127.0.0.1:9/control", name="glimmer-t-1", muse_bin_dir="/srv/muse/bin", wall_seconds=600, memory="8g", cpus="4")
        for flag in ("--init", "--cap-drop ALL", "--security-opt no-new-privileges", "--pids-limit 4096", "--memory 8g", "--memory-swap 8g", "--cpus 4", '--user "$(id -u):$(id -g)"', "--network host"):
            self.assertIn(flag, s)
        self.assertNotIn("--privileged", s)
        self.assertNotIn("--rm", s)  # the exit status must stay readable until close() removes the container
        self.assertEqual(sorted(shlex_split(s)), sorted(["/srv/lab/trials/t-1:/srv/lab/trials/t-1:rw", "/opt/coral:/opt/coral:ro", "/srv/muse/bin:/opt/muse:ro"]))
        self.assertIn('-v "$UV:/usr/local/bin/uv:ro"', s)
        self.assertNotIn("auth.json:", s)  # the login is reached through the trial directory copy, not mounted from the home
        self.assertIn("timeout -s TERM -k 30 720 coral start", s)  # wall limit plus the 120 s grace
        self.assertIn("coral.exit", s)

    def test_container_environment_is_built_from_scratch_and_holds_no_secret(self):
        env = cli.container_environment("/srv/lab/trials/t-1", "/opt/coral", "http://127.0.0.1:41000/control")
        self.assertEqual(env["PATH"].split(":")[:2], ["/srv/lab/trials/t-1/bin", "/opt/coral/.venv/bin"])
        self.assertEqual(env["MUSE_AUTH_PATH"], "/srv/lab/trials/t-1/xdg/config/muse/auth.json")
        # the NATIVE CLI reads $XDG_CONFIG_HOME/muse/auth.json (the launcher alone honours MUSE_AUTH_PATH): the staged copy must be exactly there
        self.assertEqual(env["MUSE_AUTH_PATH"], env["XDG_CONFIG_HOME"] + "/muse/auth.json")
        self.assertEqual((env["MUSE_NO_AUTO_UPDATE"], env["GLIMMER_MUSE_CMD"], env["PYTHONPATH"]), ("1", "/opt/muse/muse", "/srv/lab/trials/t-1/coral_ext"))
        self.assertEqual(env["HOME"], "/srv/lab/trials/t-1/home")
        for forbidden in cli.FORBIDDEN_ENV:
            self.assertNotIn(forbidden, env)
        blob = json.dumps(env).upper()
        for forbidden in ("API_KEY", "SECRET", "ANTHROPIC", "THUNDER", "BASETEN", "META_"):
            self.assertNotIn(forbidden, blob)
        script = cli.render_container_launch_script("/r/t", "/c", [], control_url="http://x/y", name="glimmer-t", muse_bin_dir="/m/bin")
        for forbidden in cli.FORBIDDEN_ENV:
            self.assertNotIn(forbidden, script)

    def test_hostile_values_cannot_inject_into_the_launch_script(self):
        evil = ["a=$(touch /tmp/pwned)", "b=;rm -rf ~", "c='\"`id`"]
        s = cli.render_container_launch_script("/r/t", "/c", evil, control_url="http://x/y'; id #", name="glimmer-t", muse_bin_dir="/m/bin")
        self.assertEqual(subprocess.run(["bash", "-n"], input=s, text=True, capture_output=True).returncode, 0)
        for bad in (dict(name="a b"), dict(name="x;y"), dict(memory="8g; id"), dict(cpus="1;2"), dict(image="Bad Image"), dict(network="none; x"), dict(uv_path="/a b/uv"), dict(muse_bin_dir="/a:b")):
            with self.assertRaises(ValueError, msg=str(bad)):
                cli.render_container_launch_script("/r/t", "/c", [], control_url="http://x/y", **{"name": "glimmer-t", "muse_bin_dir": "/m/bin", **bad})
        with self.assertRaises(ValueError):
            cli.render_container_launch_script("/r/t:rw", "/c", [], control_url="http://x/y", name="glimmer-t", muse_bin_dir="/m/bin")
        with self.assertRaises(RuntimeFailure):
            cli.container_name("t-1; id")


# --------------------------------------------------------------------------------------------------
def ev(payload_type, payload, stream_kind="session", stream_id="01a1a403-03ab-7550-8cf4-d7aa26fcd193"):
    return {"schema_version": 1, "stream": {"kind": stream_kind, "id": stream_id}, "payload_type": payload_type, "payload": payload}


def usage_ev(command_id, models, window=0, weekly=0):
    return ev("run.usage", {"kind": "run_usage", "command_id": command_id, "models": models, "calls": 2,
                            "allowance": {"tier": None, "window": {"used_percent": window}, "weekly": {"used_percent": weekly}}}, "exec.usage")


def model_ev(model_id):
    return ev("run.model.configured", {"kind": "run_model_configured", "model_id": model_id, "provider_id": "meta"})


class MuseUsageTests(unittest.TestCase):
    def logs(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pub = Path(tmp.name)
        (pub / "logs").mkdir()
        for name, lines in files.items():
            (pub / "logs" / name).write_text("".join((l if isinstance(l, str) else json.dumps(l)) + "\n" for l in lines))
        return pub

    def test_usage_events_add_up_per_model_and_the_last_event_of_a_run_wins(self):
        m = lambda i, c, o, r, n, model="muse-spark-1.3": {"model": model, "input_tokens": i, "cached_tokens": c, "output_tokens": o, "reasoning_tokens": r, "calls": n}
        a = [model_ev("muse-spark-1.3"), usage_ev("run-1", [m(10, 0, 5, 1, 1)]), usage_ev("run-1", [m(58061, 100, 119, 25, 2)], window=3, weekly=1),
             usage_ev("run-2", [m(7, 0, 3, 0, 1), m(1, 0, 1, 0, 1, "muse-wonder-full-1.0")], window=4, weekly=1)]
        b = [model_ev("muse-spark-1.3"), usage_ev("run-3", [m(2, 0, 2, 0, 1)])]
        u = cli.muse_usage_from_logs(self.logs({"a1.0.log": a, "a1.1.log": b, "a2.0.log": [{"type": "coral", "subtype": "prompt"}, "not json"]}))
        self.assertEqual(u["totals"], {"input_tokens": 58071, "cached_tokens": 100, "output_tokens": 125, "reasoning_tokens": 25, "calls": 5})
        self.assertEqual(set(u["models"]), {"muse-spark-1.3", "muse-wonder-full-1.0"})
        self.assertEqual((u["runs"], u["logs_with_usage"], u["logs_without_usage"]), (3, 2, 1))
        self.assertEqual(u["configured_models"], ["muse-spark-1.3"])
        self.assertEqual((u["allowance_window_used_percent"], u["allowance_weekly_used_percent"]), (4, 1))
        env = cli.usage_environment(u, u["configured_models"])
        self.assertEqual(env["researcher_tokens"], "unmetered, wall-limited")
        self.assertEqual(env["researcher_model_configured"], "muse-spark-1.3")
        self.assertIn("input=58071 cached=100 output=125 reasoning=25 calls=5 runs=3 logs=2/3", env["researcher_usage_reported"])
        self.assertIn("not metered", env["researcher_usage_reported"])
        self.assertEqual(env["researcher_allowance_used_percent"], "window=4 weekly=1")

    def test_nothing_is_invented(self):
        pub = self.logs({"a1.0.log": [{"type": "system"}, "garbage", '{"payload_type": "run.usage", "payload": ', ev("run.usage", {"command_id": "r", "models": "not a list"}),
                                       usage_ev("r2", [{"model": "m", "input_tokens": -4, "output_tokens": True, "calls": "7"}])]})
        u = cli.muse_usage_from_logs(pub)  # a run whose numbers are all unusable still counts as a run with zeros: nothing is made up
        self.assertEqual(u["totals"], {k: 0 for k in cli._USAGE_FIELDS})
        self.assertIsNone(cli.muse_usage_from_logs(self.logs({"a1.0.log": [{"type": "coral"}]})))
        self.assertIsNone(cli.muse_usage_from_logs(Path("/nonexistent/for/sure")))
        env = cli.usage_environment(None)
        self.assertEqual(env["researcher_tokens"], "unmetered, wall-limited")
        self.assertTrue(env["researcher_usage_reported"].startswith("none"))
        self.assertEqual(env["researcher_model_configured"], "not reported")
        pub2 = self.logs({"a1.0.log": [model_ev("muse-spark-1.3"), {"type": "coral"}]})  # a model but no usage
        self.assertIsNone(cli.muse_usage_from_logs(pub2))
        self.assertEqual(cli.configured_models_from_logs(pub2), ["muse-spark-1.3"])

    def test_served_and_configured_models_are_compared_with_the_requested_one(self):
        m = lambda model: {"model": model, "input_tokens": 1, "cached_tokens": 0, "output_tokens": 1, "reasoning_tokens": 0, "calls": 1}
        wonder = "muse-wonder-full-1.0"
        u = cli.muse_usage_from_logs(self.logs({"a1.0.log": [model_ev(wonder), usage_ev("r", [m(wonder)])]}))
        self.assertEqual(cli.served_models(u), [wonder])
        self.assertEqual(cli.usage_environment(u, u["configured_models"])["researcher_model_served"], wonder)
        self.assertEqual(cli.model_verdict(wonder, cli.served_models(u), u["configured_models"]), (f"ok: Muse reported only {wonder}", []))
        # a silent fallback: requested Wonder, served Spark (in one run of one agent is enough)
        u2 = cli.muse_usage_from_logs(self.logs({"a1.0.log": [model_ev(wonder), usage_ev("r", [m(wonder)])], "a2.0.log": [usage_ev("r2", [m("muse-spark-1.3")])]}))
        verdict, others = cli.model_verdict(wonder, cli.served_models(u2), u2["configured_models"])
        self.assertEqual(others, ["muse-spark-1.3"])
        self.assertTrue(verdict.startswith("MISMATCH: requested muse-wonder-full-1.0"))
        # a configured id that differs counts too, even when no usage was reported
        self.assertEqual(cli.model_verdict(wonder, [], ["muse-spark-1.3"])[1], ["muse-spark-1.3"])
        # nothing reported is unverified, never "ok"
        self.assertEqual(cli.model_verdict(wonder, [], []), ("unverified: Muse reported no model", []))
        self.assertEqual(cli.usage_environment(None)["researcher_model_served"], "not reported")
        # a usage entry without a model name is not an id
        u3 = cli.muse_usage_from_logs(self.logs({"a1.0.log": [usage_ev("r", [{"input_tokens": 3}])]}))
        self.assertEqual(cli.served_models(u3), [])

    def test_symlinked_logs_are_ignored(self):
        pub = self.logs({"a1.0.log": [usage_ev("r", [{"model": "m", "input_tokens": 3}])]})
        secret = pub / "secret.txt"
        secret.write_text(json.dumps(usage_ev("r", [{"model": "m", "input_tokens": 999}])))
        os.symlink(secret, pub / "logs" / "evil.0.log")
        self.assertEqual(cli.muse_usage_from_logs(pub)["totals"]["input_tokens"], 3)


# --------------------------------------------------------------------------------------------------
class AuthScriptTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.muse = write_muse_home(self.tmp / "muse-home")
        self.dest = self.tmp / "trial" / "xdg" / "config" / "muse"
        self.dest.parent.mkdir(parents=True)

    def run_script(self, src=None, dest=None, bindir=None):
        return subprocess.run([sys.executable, "-I", "-c", cli._AUTH_SCRIPT, str(src or self.muse / "auth" / "auth.json"), str(dest or self.dest), str(bindir or self.muse / "bin")],
                              capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})

    def test_copies_exactly_the_login_file_bytes_and_nothing_else(self):
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stderr)
        for secret in (SECRET_A, SECRET_B):
            self.assertNotIn(secret, r.stdout + r.stderr)
        info = json.loads(r.stdout)
        self.assertEqual((info["auth_bytes"], info["muse_version"]), (len(AUTH_BYTES), "1.4.14-R7309.1"))
        self.assertEqual(sorted(p.name for p in self.dest.iterdir()), ["auth.json"])
        self.assertEqual((self.dest / "auth.json").read_bytes(), AUTH_BYTES)
        self.assertEqual(stat.S_IMODE((self.dest / "auth.json").stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.dest.stat().st_mode), 0o700)
        self.assertEqual(tree_contains(self.dest, b"PRIVATE-CFG-9d2c"), [])  # the rest of the muse home is not copied

    def test_the_login_need_not_be_json(self):
        write_muse_home(self.muse, auth=b"\x00\x01 opaque \xff bytes")
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual((self.dest / "auth.json").read_bytes(), b"\x00\x01 opaque \xff bytes")

    def test_refuses_unusable_sources_and_leaves_nothing(self):
        real = self.tmp / "elsewhere.json"
        real.write_bytes(AUTH_BYTES)
        link = self.tmp / "link.json"
        os.symlink(real, link)
        big = self.tmp / "big.json"
        big.write_bytes(b"x" * ((1 << 20) + 1))
        empty = self.tmp / "empty.json"
        empty.write_bytes(b"")
        for src in (self.tmp / "missing.json", link, big, empty, self.tmp):
            r = self.run_script(src=src)
            self.assertEqual(r.returncode, 3, (src, r.stderr))
            self.assertFalse(self.dest.exists(), src)
            self.assertNotIn(SECRET_A, r.stdout + r.stderr)
        (self.muse / "bin" / "muse").unlink()
        r = self.run_script()
        self.assertEqual(r.returncode, 3)
        self.assertIn("launcher", r.stderr)

    def test_never_overwrites_an_existing_destination(self):
        self.dest.mkdir()
        (self.dest / "keep").write_text("keep")
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertEqual((self.dest / "keep").read_text(), "keep")

    def scan(self, root):
        return subprocess.run([sys.executable, "-I", "-c", cli._SCAN_SCRIPT, str(root)], capture_output=True, text=True)

    def test_scan_finds_the_login_or_a_slice_of_it_and_prints_no_secret(self):
        root = self.tmp / "scan"
        (root / "xdg" / "config" / "muse").mkdir(parents=True)
        (root / "xdg" / "config" / "muse" / "auth.json").write_bytes(AUTH_BYTES)
        pub = root / "run" / ".coral" / "public"
        (pub / "notes").mkdir(parents=True)
        (pub / "notes" / "ok.md").write_text("nothing here " + OTHER + " who test n 7")
        r = self.scan(root)
        self.assertEqual((r.returncode, r.stdout.split()[:2]), (0, ["hits=0", "scanned=1"]))
        (pub / "notes" / "short.md").write_text("x" * 5000 + SECRET_A[:20] + "y")  # too short to be recognised: the scan is not a promise about fragments
        self.assertEqual(self.scan(root).returncode, 0)
        (pub / "notes" / "short.md").unlink()
        for lo in (0, 5, 13, 30):  # a slice of 31 characters from anywhere in a token
            (pub / "notes" / "slice.md").write_text("x" * 5000 + SECRET_A[lo:lo + 31] + "y")
            r = self.scan(root)
            self.assertEqual(r.returncode, 7, lo)
            self.assertNotIn(SECRET_A[lo:lo + 31], r.stdout + r.stderr)
        (pub / "notes" / "slice.md").unlink()
        (pub / "notes" / "b.md").write_text("token: " + SECRET_B)
        self.assertEqual(self.scan(root).returncode, 7)
        # a secret split across the 1 MiB read boundary is still found
        (pub / "notes" / "b.md").write_bytes(b"z" * ((1 << 20) - 20) + SECRET_B.encode() + b"\n")
        self.assertEqual(self.scan(root).returncode, 7)
        (root / "xdg" / "config" / "muse" / "auth.json").unlink()
        self.assertEqual((self.scan(root).returncode, self.scan(root).stdout.strip()), (0, "unavailable"))
        (root / "xdg" / "config" / "muse" / "auth.json").write_bytes(b"{}")  # nothing token-like to look for
        self.assertEqual(self.scan(root).stdout.strip(), "unavailable")


# --------------------------------------------------------------------------------------------------
STUB_RUNTIME_PY = '''
import json, subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

@dataclass
class AgentHandle:
    agent_id: str
    process: Any
    worktree_path: Path
    log_path: Path
    session_id: Any = None
    _log_file: Any = None
    err_file: Any = None
    err_path: Any = None

def write_coral_log_entry(log_file, prompt, source, agent_id, session_id=None, task_name=None, task_description=None):
    entry = {"type": "coral", "subtype": "prompt", "source": source, "agent_id": agent_id, "prompt": prompt, "timestamp": datetime.now(UTC).isoformat()}
    if session_id: entry["session_id"] = session_id
    if task_name: entry["task_name"] = task_name
    log_file.write(json.dumps(entry) + "\\n"); log_file.flush()
'''
STUB_PROCESS_PY = '''
def open_agent_stderr_for_log_dir(log_dir, agent_id):
    return None
'''
STUB_CLASSIFIER_PY = '''
def classify_by_uptime(exit_code, uptime_seconds, min_clean_runtime_seconds):
    return "clean" if exit_code == 0 and uptime_seconds is not None and uptime_seconds >= min_clean_runtime_seconds else "no_result"
'''
FAKE_MUSE = r'''#!/bin/bash
# a stand-in for the Muse launcher: records argv and environment, prints Muse-like JSONL events, exits like a finished run
out="{out}"
printf '%s\n' "$@" > "$out.argv"
env | sort > "$out.env"
pwd > "$out.cwd"
cat >> "$out.count" <<< x
sid=""; prev=""
for a in "$@"; do [ "$prev" = "--session-id" ] && sid="$a"; prev="$a"; done
echo '{"stream": {"kind": "session", "id": "'"$sid"'"}, "payload_type": "run.model.configured", "payload": {"model_id": "muse-spark-1.3"}}'
echo '{"stream": {"kind": "exec.usage", "id": "'"$sid"'"}, "payload_type": "run.usage", "payload": {"command_id": "c1", "models": [{"model": "muse-spark-1.3", "input_tokens": 11, "cached_tokens": 0, "output_tokens": 2, "reasoning_tokens": 0, "calls": 1}]}}'
echo '{"stream": {"kind": "session", "id": "'"$sid"'"}, "payload_type": "run.terminal.completed", "payload": {}}'
exit "${FAKE_MUSE_EXIT:-0}"
'''


class MuseRuntimeModuleTests(unittest.TestCase):
    """Runs the staged CORAL runtime class against a stub ``coral`` package and a fake ``muse``."""

    @classmethod
    def setUpClass(cls):
        cls._stub = tempfile.TemporaryDirectory()
        base = Path(cls._stub.name)
        for rel, text in (("coral/__init__.py", ""), ("coral/agent/__init__.py", ""), ("coral/agent/runtime.py", STUB_RUNTIME_PY), ("coral/agent/process.py", STUB_PROCESS_PY), ("coral/agent/exit_classifier.py", STUB_CLASSIFIER_PY)):
            (base / rel).parent.mkdir(parents=True, exist_ok=True)
            (base / rel).write_text(text)
        sys.path.insert(0, str(base))
        cls._saved = {k: v for k, v in sys.modules.items() if k == "coral" or k.startswith("coral.")}
        for k in cls._saved:
            del sys.modules[k]
        spec = importlib.util.spec_from_file_location("glimmer_muse_runtime_under_test", cli.EXT_MODULE)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls._stub.name)
        for k in [k for k in sys.modules if k == "coral" or k.startswith("coral.")]:
            del sys.modules[k]
        sys.modules.update(cls._saved)
        cls._stub.cleanup()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        self.wt = self.tmp / "agents" / "agent-1"
        self.wt.mkdir(parents=True)
        for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"], ["commit", "-q", "--allow-empty", "-m", "seed"]):
            subprocess.run(["git", *cmd], cwd=self.wt, check=True, capture_output=True)
        (self.wt / ".coral_agent_id").write_text("agent-1")
        self.logs = self.tmp / ".coral" / "public" / "logs"
        self.fake = self.tmp / "muse"
        self.out = self.tmp / "rec"
        self.fake.write_text(FAKE_MUSE.replace("{out}", str(self.out)))
        self.fake.chmod(0o755)
        self.env = {"GLIMMER_MUSE_CMD": str(self.fake), "PATH": os.environ["PATH"], "HOME": str(self.tmp / "home"), "MUSE_AUTH_PATH": str(self.tmp / "auth.json"),
                    "MUSE_NO_AUTO_UPDATE": "1", "GLIMMER_CONTROL": "http://127.0.0.1:1/control", "XDG_CONFIG_HOME": str(self.tmp / "xdg")}

    def start(self, env=None, **kw):
        from unittest import mock

        with mock.patch.dict(os.environ, env if env is not None else self.env, clear=True):
            h = self.mod.MuseCodeRuntime().start(self.wt, self.wt / "AGENTS.md", model=kw.pop("model", "muse-spark-1.3"), log_dir=self.logs, **kw)
        self.addCleanup(lambda: h.process.poll() is None and h.process.kill())
        h.process.wait(timeout=30)
        h._log_file.close()
        return h

    def argv(self):
        return (Path(str(self.out) + ".argv")).read_text().splitlines()

    def test_protocol_and_names(self):
        rt = self.mod.MuseCodeRuntime()
        self.assertEqual((rt.instruction_filename, rt.shared_dir_name), ("AGENTS.md", ".agents"))
        for attr in ("start", "extract_session_id", "classify_exit"):
            self.assertTrue(callable(getattr(rt, attr)))

    def test_command_line(self):
        h = self.start(prompt="do the thing; $(id) `x` 'q' \"d\"", task_name="t")
        a = self.argv()
        self.assertEqual(a[:2], ["exec", "--json"])
        for flag in ("--yolo", "--no-foreign-personal-context", "--disable-request-user-input", "--disable-web-tools"):
            self.assertIn(flag, a)
        self.assertEqual(a[a.index("--model") + 1], "muse-spark-1.3")
        self.assertEqual(a[a.index("--workspace") + 1], str(self.wt))
        sid = a[a.index("--session-id") + 1]
        self.assertRegex(sid, r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        pf = Path(a[a.index("--prompt-file") + 1])
        self.assertEqual(pf.read_text(), "do the thing; $(id) `x` 'q' \"d\"")  # the prompt travels as a file: never an argv word, never a shell word
        self.assertNotIn("do the thing", " ".join(a))
        self.assertNotIn("--reasoning-effort", a)
        self.assertEqual(Path(Path(str(self.out) + ".cwd").read_text().strip()).resolve(), self.wt)
        self.assertEqual(h.log_path.name, "agent-1.0.log")
        self.assertEqual(h.session_id, sid)

    def test_the_requested_model_reaches_muse_exec(self):
        self.start(model="muse-wonder-full-1.0")
        a = self.argv()
        self.assertEqual(a.count("--model"), 1)
        self.assertEqual(a[a.index("--model") + 1], "muse-wonder-full-1.0")

    def test_options(self):
        self.start(runtime_options={"reasoning_effort": "medium", "web_tools": True}, max_turns=40)
        a = self.argv()
        self.assertEqual(a[a.index("--reasoning-effort") + 1], "medium")
        self.assertEqual(a[a.index("--max-model-steps") + 1], "40")
        self.assertNotIn("--disable-web-tools", a)
        with self.assertRaises(ValueError):
            self.start(runtime_options={"reasoning_effort": "turbo"})

    def test_environment_is_an_allowlist_and_the_api_keys_never_reach_the_child(self):
        poisoned = {**self.env, "MUSE_API_KEY": "k1", "BASETEN_API_KEY": "k2", "META_API_KEY": "k3", "ANTHROPIC_API_KEY": "k4", "OPENAI_API_KEY": "k5", "THUNDER_API_KEY": "k6",
                    "AWS_SECRET_ACCESS_KEY": "k7", "SSH_AUTH_SOCK": "/tmp/agent.sock", "GITHUB_TOKEN": "k8", "HF_TOKEN": "k9", "MUSE_CUSTOM_HEADERS": "h", "MUSE_BASE_URL": "http://evil"}
        self.start(env=poisoned)
        child = dict(l.split("=", 1) for l in Path(str(self.out) + ".env").read_text().splitlines() if "=" in l)
        blob = json.dumps(child)
        for name in ("MUSE_API_KEY", "BASETEN_API_KEY", "META_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "THUNDER_API_KEY", "AWS_SECRET_ACCESS_KEY", "SSH_AUTH_SOCK", "GITHUB_TOKEN", "HF_TOKEN", "MUSE_CUSTOM_HEADERS", "MUSE_BASE_URL"):
            self.assertNotIn(name, child)
        for value in ("k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "evil"):
            self.assertNotIn(f'"{value}"', blob)
        self.assertEqual(child["MUSE_AUTH_PATH"], str(self.tmp / "auth.json"))
        self.assertEqual(child["MUSE_NO_AUTO_UPDATE"], "1")
        self.assertEqual(child["VIRTUAL_ENV"], str(self.wt / ".venv"))
        self.assertTrue(child["PATH"].startswith(str(self.wt / ".venv") + "/bin:"))
        # even a permissive allowlist would be stopped by the explicit never-list
        for name in self.mod.ENV_NEVER:
            self.assertNotIn(name, self.mod.ENV_ALLOW)
        env = self.mod.build_agent_env({"META_API_KEY": "x", "HOME": "/h", "PATH": "/p"}, self.wt)
        self.assertNotIn("META_API_KEY", env)

    def test_the_shared_dir_is_excluded_from_commits(self):
        self.start()
        self.start()  # idempotent
        exclude = (self.wt / ".git" / "info" / "exclude").read_text().splitlines()
        self.assertEqual(exclude.count(".agents/"), 1)
        self.assertEqual(exclude.count(".muse/"), 1)
        (self.wt / ".agents").mkdir()
        os.symlink(self.tmp, self.wt / ".agents" / "notes")
        (self.wt / "solution.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=self.wt, check=True)
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=self.wt, capture_output=True, text=True).stdout.split()
        self.assertIn("solution.py", staged)
        self.assertFalse([f for f in staged if f.startswith(".agents")])  # coral eval's `git add -A` would otherwise commit the symlinks

    def test_log_session_id_and_restart_prompts(self):
        h = self.start()
        log = h.log_path.read_text().splitlines()
        first = json.loads(log[0])
        self.assertEqual((first["type"], first["source"], first["session_id"]), ("coral", "start", h.session_id))
        rt = self.mod.MuseCodeRuntime()
        self.assertEqual(rt.extract_session_id(h.log_path), h.session_id)
        self.assertIsNone(rt.extract_session_id(self.tmp / "missing.log"))
        # a restart: a NEW session by default, with a prompt that says the conversation is gone
        h2 = self.start(resume_session_id=h.session_id)
        self.assertNotEqual(h2.session_id, h.session_id)
        self.assertIn("conversation is gone", Path(self.argv()[self.argv().index("--prompt-file") + 1]).read_text())
        self.assertEqual(json.loads(h2.log_path.read_text().splitlines()[0])["source"], "restart")
        self.assertEqual(h2.log_path.name, "agent-1.1.log")  # numbered, never overwritten
        # unless the operator opts in to reusing the session id
        h3 = self.start(resume_session_id=h.session_id, runtime_options={"reuse_session": True})
        self.assertEqual(h3.session_id, h.session_id)
        h4 = self.start(resume_session_id="not-a-uuid; rm -rf ~", runtime_options={"reuse_session": True})
        self.assertNotEqual(h4.session_id, "not-a-uuid; rm -rf ~")

    def test_classify_exit(self):
        h = self.start()
        rt = self.mod.MuseCodeRuntime()
        self.assertEqual(rt.classify_exit(h.log_path, 0, 1.0), "clean")  # run.terminal.completed present
        self.assertEqual(rt.classify_exit(h.log_path, 1, 500.0), "no_result")
        bare = self.tmp / "bare.log"
        bare.write_text('{"type": "coral"}\n')
        self.assertEqual(rt.classify_exit(bare, 0, 1.0), "no_result")  # too short and no marker
        self.assertEqual(rt.classify_exit(bare, 0, 500.0), "clean")
        self.assertEqual(rt.classify_exit(self.tmp / "missing.log", 0, 1.0), "no_result")

    def test_unsupported_modes_are_refused(self):
        rt = self.mod.MuseCodeRuntime()
        for kw in ({"gateway_url": "http://x"}, {"gateway_api_key": "k"}, {"run_as_user": {"uid": 1}}, {"sandbox": object()}):
            with self.assertRaises(RuntimeError):
                rt.start(self.wt, self.wt / "AGENTS.md", log_dir=self.logs, **kw)

    def test_the_staged_file_is_what_the_runtime_ships(self):
        self.assertTrue(cli.EXT_MODULE.is_file())
        self.assertIn("GLIMMER_MUSE_CMD", cli.EXT_MODULE.read_text())
        self.assertEqual(cli.RUNTIME_ENTRYPOINT, "glimmer_muse_runtime:MuseCodeRuntime")


# --------------------------------------------------------------------------------------------------
class CliCase(RuntimeCase):
    """RuntimeCase with the Muse CLI runtime, a fake docker and a scratch muse home."""

    behavior = {"mode": "finish"}
    wall_minutes = 72

    def setUp(self):
        super().setUp()
        self.fake_bin = self.tmp / "fakebin"
        self.fake_bin.mkdir()
        self.dstate = self.tmp / "docker-state"
        for name, text in (("docker", FAKE_DOCKER.format(python=sys.executable, state=str(self.dstate), fakebin=str(self.fake_bin))), ("timeout", FAKE_TIMEOUT)):
            (self.fake_bin / name).write_text(text)
            (self.fake_bin / name).chmod(0o755)
        self.dstate.mkdir()
        self.remote.env["PATH"] = f"{self.fake_bin}:{self.remote.env['PATH']}"
        self.remote.env["FAKE_STATE"] = str(self.dstate)
        self.muse_home = write_muse_home(self.tmp / "muse-home")
        self.uv = self.tmp / "uvbin" / "uv"
        self.uv.parent.mkdir()
        self.uv.write_text("#!/bin/sh\n")
        self.uv.chmod(0o755)
        self.runtime = self.make_runtime()

    def make_runtime(self, **kw):
        args = dict(muse_home=str(self.muse_home), clock=self.fx.clock, remote_trials_root=str(self.trials_root), max_real_attempts=5, uv_path=str(self.uv), wall_limit_minutes=self.wall_minutes)
        args.update(kw)
        return cli.CoralMuseRuntime(self.layout, self.remote, str(self.coral_home), {"coral_commit": "0123dfb939b35228cf2c1fde224cd0561e727408"}, free_port, **args)

    def calls(self):
        p = self.dstate / "calls.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def container(self, ctx):
        return json.loads((self.dstate / f"glimmer-{ctx.spec.trial_id}.json").read_text())

    def container_alive(self, ctx):
        r = self.remote.run(["docker", "ps", "-q", "-f", f"name=^/glimmer-{ctx.spec.trial_id}$"])
        return bool(r.stdout.strip())

    def container_exists(self, ctx):
        r = self.remote.run(["docker", "ps", "-aq", "-f", f"name=^/glimmer-{ctx.spec.trial_id}$"])
        return bool(r.stdout.strip())


class StartTests(CliCase):
    def test_protocols_and_signature(self):
        self.assertIsInstance(self.runtime, ResearcherRuntime)
        ctx, h = self.start()
        self.assertIsInstance(h, RuntimeHandle)
        with self.assertRaises(TypeError):
            cli.CoralMuseRuntime(self.layout, self.remote, str(self.coral_home), {}, free_port, clock=self.fx.clock)  # the muse home is a required choice
        self.assertIs(cli.CoralCliRuntime, cli.CoralMuseRuntime)

    def test_constructor_validates(self):
        for kw in (dict(muse_home="relative/dir"), dict(muse_home="/a b"), dict(model="x; id"), dict(reasoning_effort="warp"), dict(login_label="a b"), dict(wall_limit_minutes=0), dict(wall_limit_minutes=True),
                   dict(wall_limit_minutes=5000), dict(memory="all"), dict(image="Bad Image")):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.make_runtime(**kw)
        self.assertEqual(cli.DEFAULT_WALL_MINUTES, 72)
        self.assertEqual(self.make_runtime()._wall_seconds, 72 * 60)
        self.assertEqual(self.make_runtime()._model, "muse-spark-1.3")

    def test_staged_trial_directory_container_and_login(self):
        os.environ["META_API_KEY"] = "META-KEY-must-not-reach-the-researchers"
        self.addCleanup(os.environ.pop, "META_API_KEY", None)
        ctx, h = self.start()
        root = self.root_of(ctx)
        info = self.start_json(ctx)
        # no opencode, no gateway, no provider config, no metering route; the Muse runtime class is staged
        self.assertFalse((root / "bin" / "opencode").exists())
        self.assertFalse((root / "runtime").exists())
        self.assertFalse((root / "pack" / "seed" / "opencode.json").exists())
        self.assertEqual((root / "coral_ext" / "glimmer_muse_runtime.py").read_bytes(), cli.EXT_MODULE.read_bytes())
        self.assertTrue(os.access(root / "bin" / "glimmer", os.X_OK))
        self.assertEqual((root / ".glimmer_token").read_text(), TOKEN)
        self.assertEqual((root / "tools" / "researcher_tool.py").read_bytes(), rc.TOOL_PATH.read_bytes())
        # the login: exactly the one file, byte for byte, 0600, inside the trial directory only
        auth = root / "xdg" / "config" / "muse" / "auth.json"
        self.assertEqual(auth.read_bytes(), AUTH_BYTES)
        self.assertEqual(stat.S_IMODE(auth.stat().st_mode), 0o600)
        self.assertEqual(sorted(p.name for p in (root / "xdg" / "config" / "muse").iterdir()), ["auth.json"])
        self.assertEqual(tree_contains(self.dstate, SECRET_A.encode()), [])  # the login is in no docker argument
        self.assertEqual(tree_contains(root, b"PRIVATE-CFG-9d2c"), [])  # nothing else of the muse home
        # what docker was asked
        self.assertEqual(len([c for c in self.calls() if c[0] == "run"]), 1)
        c = self.container(ctx)
        o = c["opts"]
        self.assertEqual(c["image"], cli.DEFAULT_IMAGE)
        self.assertEqual((o["--name"], o["--network"], o["--cap-drop"], o["--security-opt"]), (f"glimmer-{ctx.spec.trial_id}", "host", "ALL", "no-new-privileges"))
        self.assertEqual(o["--user"], f"{os.getuid()}:{os.getgid()}")
        self.assertEqual(sorted(o["v"]), sorted([f"{root}:{root}:rw", f"{self.coral_home}:{self.coral_home}:ro", f"{self.muse_home}/bin:/opt/muse:ro", f"{self.uv}:/usr/local/bin/uv:ro"]))
        self.assertTrue(all(not v.startswith(str(self.muse_home / "auth")) and not v.startswith(str(self.muse_home / "cfg")) for v in o["v"]))
        self.assertNotIn("--privileged", o)
        self.assertNotIn("--rm", o)
        # the environment and command line CORAL got
        env = info["env"]
        self.assertEqual(env["MUSE_AUTH_PATH"], str(auth))
        self.assertEqual((env["MUSE_NO_AUTO_UPDATE"], env["GLIMMER_MUSE_CMD"], env["PYTHONPATH"]), ("1", "/opt/muse/muse", str(root / "coral_ext")))
        self.assertEqual(env["HOME"], str(root / "home"))
        self.assertEqual(env["PATH"].split(":")[1], str(root / "bin"))  # the fake docker prepends its own stand-in for `timeout`
        self.assertEqual(env["GLIMMER_TOKEN_FILE"], str(root / ".glimmer_token"))
        for name in cli.FORBIDDEN_ENV:
            self.assertNotIn(name, env)
        blob = json.dumps(env)
        self.assertNotIn("META-KEY-must-not-reach", blob)
        self.assertNotIn(SECRET_A, blob)
        self.assertNotIn(TOKEN, blob)
        self.assertEqual(self.remote.tunnels[0].target, int(ctx.control_endpoint.split(":")[-1].split("/")[0]))
        argv = info["argv"]
        self.assertEqual(argv[:3], ["start", "-c", "pack/task.yaml"])
        d = dict(a.split("=", 1) for a in argv[3:])
        self.assertEqual((d["agents.runtime"], d["agents.model"], d["agents.gateway.enabled"], d["run.stop.max_real_attempts"]), ("glimmer_muse_runtime:MuseCodeRuntime", "muse-spark-1.3", "false", "5"))
        self.assertNotIn("agents.gateway.port", d)
        self.assertEqual(Path(info["cwd"]).resolve(), root)
        self.assertIn("-k 30 4440", (self.dstate / "timeout.args").read_text())  # the in-container timeout is the wall limit plus grace

    def test_the_llm_endpoint_is_ignored(self):
        ctx = self.context()
        ctx = dataclasses.replace(ctx, llm_base_url="http://127.0.0.1:1/v1")  # nothing listens there and it is not the control port
        h = self.runtime.start(ctx)
        self.handles.append(h)
        self.assertEqual(self.wait_exit(h), ExitReason.FINISHED)

    def test_a_missing_login_refuses_the_start_and_leaves_nothing(self):
        (self.muse_home / "auth" / "auth.json").unlink()
        ctx = self.context()
        with self.assertRaises(RuntimeFailure) as cm:
            self.runtime.start(ctx)
        self.assertIn("Muse login", str(cm.exception))
        self.assertFalse(self.root_of(ctx).exists())
        self.assertFalse(self.container_exists(ctx))
        self.assertFalse((self.trials_root / ".holder").exists())

    def test_a_missing_image_refuses_the_start_and_removes_the_login(self):
        (self.dstate / "missing_image").write_text("")
        ctx = self.context()
        with self.assertRaises(RuntimeFailure) as cm:
            self.runtime.start(ctx)
        self.assertIn("image", str(cm.exception))
        self.assertFalse(self.root_of(ctx).exists())
        self.assertFalse((self.trials_root / ".holder").exists())
        (self.dstate / "missing_image").unlink()
        ctx2 = self.context(trial_id="t-0002")
        h = self.runtime.start(ctx2)  # the host is free again
        self.handles.append(h)

    def test_a_second_trial_cannot_start_beside_the_first(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(self.context(trial_id="t-0002"))
        h.stop(ExitReason.STOPPED_BUDGET)


class FinishAndHarvestTests(CliCase):
    behavior = {
        "mode": "finish",
        "attempts": [
            {"files": {"submission.json": json.dumps({"adapter_id": "ad-0001", "recipe_sha256": "a" * 64, "commit": "c" * 40, "dev_score": 0.99})}, "timestamp": "2026-11-01T05:36:20.000000+00:00"},
            {"files": {}, "timestamp": "2026-11-01T05:36:24.000000+00:00", "status": "pending"},
        ],
    }

    def test_finished_run_is_harvested_and_the_environment_is_honest(self):
        ctx, h = self.start()
        self.assertEqual(self.wait_exit(h), ExitReason.FINISHED)
        res = h.collect()
        self.assertIs(res, h.collect())
        self.assertEqual(res.attempts, 1)
        self.assertEqual([s.submission.adapter_id for s in res.submissions], ["ad-0001"])
        self.assertIsNone(res.submissions[0].submission.dev_score)
        env = res.environment
        self.assertEqual(env["runtime"], "coral+muse_code+docker")
        self.assertEqual((env["model"], env["researcher_model_requested"], env["researcher_tokens"], env["researcher_login"]), ("muse-spark-1.3", "muse-spark-1.3", "unmetered, wall-limited", "muse-login"))
        self.assertEqual(env["researcher_wall_limit_s"], "4320")
        self.assertTrue(env["researcher_usage_reported"].startswith("none"))  # the fake's logs are not Muse logs: nothing is invented
        self.assertEqual(env["researcher_model_configured"], "not reported")
        self.assertEqual((env["container_network"], env["coral_exit"], env["muse_version"]), ("host", "0", "1.4.14-R7309.1"))
        self.assertEqual(env["login_rewritten_in_container"], "no")
        self.assertTrue(env["login_scan"].startswith("clean"))
        self.assertEqual(env["grader_modified"], "false")
        blob = json.dumps(env)
        self.assertNotIn(SECRET_A, blob)
        self.assertNotIn(SECRET_B, blob)
        # no login reached the control host's side of the trial
        self.assertEqual(tree_contains(self.layout.trial_dir("t-0001"), SECRET_A.encode()), [])
        self.assertEqual(tree_contains(self.layout.trial_trace_dir("t-0001"), SECRET_B.encode()), [])
        self.assertFalse((self.layout.trial_trace_dir("t-0001") / "public" / "xdg").exists())

    def test_usage_and_models_reported_by_muse_are_recorded_when_present(self):
        self.set_behavior(mode="finish", log_names=["a1.0.log", "a1.1.log"])
        ctx, h = self.start()
        self.wait_exit(h)
        pub = self.root_of(ctx) / "run" / ".coral" / "public" / "logs"
        for n, (i, o) in (("a1.0.log", (100, 20)), ("a1.1.log", (7, 3))):
            (pub / n).write_text(json.dumps(model_ev("muse-spark-1.3")) + "\n" + json.dumps(usage_ev(f"r-{n}", [{"model": "muse-spark-1.3", "input_tokens": i, "cached_tokens": 5, "output_tokens": o, "reasoning_tokens": 1, "calls": 2}], 2, 1)) + "\n")
        res = h.collect()
        self.assertIn("input=107 cached=10 output=23 reasoning=2 calls=4", res.environment["researcher_usage_reported"])
        self.assertEqual(res.environment["researcher_model_configured"], "muse-spark-1.3")
        self.assertEqual(res.environment["researcher_allowance_used_percent"], "window=2 weekly=1")
        self.assertEqual(res.environment["researcher_tokens"], "unmetered, wall-limited")  # still not a meter

    def _wonder_trial(self, logs):
        """A trial that requested muse-wonder-full-1.0; ``logs`` maps a log name to the models its events report."""
        self.runtime = self.make_runtime(model="muse-wonder-full-1.0")
        self.set_behavior(mode="finish", log_names=sorted(logs))
        ctx, h = self.start()
        self.wait_exit(h)
        self.assertEqual(dict(a.split("=", 1) for a in self.start_json(ctx)["argv"][3:])["agents.model"], "muse-wonder-full-1.0")
        pub = self.root_of(ctx) / "run" / ".coral" / "public" / "logs"
        for n, (configured, served) in logs.items():
            (pub / n).write_text(json.dumps(model_ev(configured)) + "\n" + json.dumps(usage_ev(f"r-{n}", [{"model": served, "input_tokens": 3, "output_tokens": 1, "calls": 1}])) + "\n")
        return ctx, h

    def test_the_requested_configured_and_served_wonder_model_are_recorded(self):
        w = "muse-wonder-full-1.0"
        ctx, h = self._wonder_trial({"a1.0.log": (w, w), "a2.0.log": (w, w)})
        env = h.collect().environment
        self.assertEqual((env["model"], env["researcher_model_requested"], env["researcher_model_configured"], env["researcher_model_served"]), (w, w, w, w))
        self.assertEqual(env["researcher_model_check"], f"ok: Muse reported only {w}")

    def test_a_silent_fallback_to_another_model_fails_the_trial_loudly(self):
        w = "muse-wonder-full-1.0"
        ctx, h = self._wonder_trial({"a1.0.log": (w, w), "a2.0.log": (w, "muse-spark-1.3")})
        with self.assertRaises(cli.ResearcherModelMismatch) as cm:
            h.collect()
        self.assertIsInstance(cm.exception, RuntimeFailure)  # the trial runner records a failed trial
        self.assertIn("requested muse-wonder-full-1.0", str(cm.exception))
        self.assertIn("muse-spark-1.3", str(cm.exception))
        with self.assertRaises(cli.ResearcherModelMismatch):  # asking again gives the same verdict, never a result
            h.collect()
        self.assertTrue((self.layout.trial_trace_dir("t-0001") / "public" / "logs").exists(), "the evidence is harvested")

    def test_a_configured_model_other_than_the_requested_one_fails_the_trial(self):
        ctx, h = self._wonder_trial({"a1.0.log": ("muse-spark-1.3", "muse-wonder-full-1.0")})
        with self.assertRaises(cli.ResearcherModelMismatch):
            h.collect()

    def test_a_refreshed_login_is_reported(self):
        ctx, h = self.start()
        self.wait_exit(h)
        (self.root_of(ctx) / "xdg" / "config" / "muse" / "auth.json").write_bytes(AUTH_BYTES + b" ")  # what a token refresh by Muse would do to its copy
        self.assertEqual(h.collect().environment["login_rewritten_in_container"], "yes")

    def test_a_harvest_that_contains_the_login_is_refused_and_nothing_is_copied(self):
        ctx, h = self.start()
        self.wait_exit(h)
        leak = self.root_of(ctx) / "run" / ".coral" / "public" / "notes" / "leak.md"
        leak.write_text("my token is " + SECRET_A)
        with self.assertRaises(RuntimeFailure) as cm:
            h.collect()
        self.assertIn("Muse login", str(cm.exception))
        self.assertNotIn(SECRET_A, str(cm.exception))
        self.assertFalse((self.layout.trial_trace_dir("t-0001") / "public").exists())
        self.assertEqual(tree_contains(self.layout.trial_trace_dir("t-0001"), SECRET_A.encode()), [])
        leak.unlink()
        self.assertEqual(h.collect().exit_reason, ExitReason.FINISHED)  # a clean tree can be harvested afterwards

    def test_close_removes_the_container_the_login_and_the_trial_directory(self):
        ctx, h = self.start()
        self.wait_exit(h)
        h.collect()
        root = self.root_of(ctx)
        self.assertTrue(self.container_exists(ctx))
        h.close()
        self.assertFalse(self.container_exists(ctx))
        self.assertFalse(root.exists())
        self.assertEqual(tree_contains(self.trials_root, SECRET_A.encode()), [])
        self.assertFalse((self.trials_root / ".holder").exists())
        h.close()  # idempotent
        self.assertIn(["rm", "-f", f"glimmer-{ctx.spec.trial_id}"], self.calls())

    def test_a_container_that_cannot_be_removed_keeps_the_directory_and_the_host(self):
        ctx, h = self.start()
        self.wait_exit(h)
        (self.dstate / "rm_fails").write_text("")
        h.close()
        self.assertTrue(self.root_of(ctx).exists())  # unconfirmed: the operator cleans up
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(self.context(trial_id="t-0002"))
        (self.dstate / "rm_fails").unlink()


class StopWallAndCrashTests(CliCase):
    wall_minutes = 1

    def test_stop_kills_the_container_and_keeps_the_reason(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        self.assertIsNone(h.poll())
        self.assertTrue(self.container_alive(ctx))
        h.stop(ExitReason.STOPPED_BUDGET)
        self.assertFalse(self.container_alive(ctx))
        self.assertEqual(h.poll(), ExitReason.STOPPED_BUDGET)
        h.stop(ExitReason.STOPPED_WALL)  # idempotent, first reason wins
        self.assertEqual(h.poll(), ExitReason.STOPPED_BUDGET)
        self.assertEqual(h.collect().exit_reason, ExitReason.STOPPED_BUDGET)

    def test_a_run_that_ignores_term_is_killed(self):
        self.set_behavior(mode="hang", ignore_term=True)
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        h.stop(ExitReason.STOPPED_CONTAMINATED)
        self.assertFalse(self.container_alive(ctx))

    def test_the_wall_limit_stops_the_team(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        self.assertIsNone(h.poll())
        self.fx.clock.advance(59)
        self.assertIsNone(h.poll())
        self.assertTrue(self.container_alive(ctx))
        self.fx.clock.advance(2)
        self.assertEqual(h.poll(), ExitReason.STOPPED_WALL)
        self.assertFalse(self.container_alive(ctx))
        res = h.collect()
        self.assertEqual(res.exit_reason, ExitReason.STOPPED_WALL)
        self.assertEqual(res.environment["wall_limit"], "reached")

    def test_the_in_container_timeout_is_the_wall_limit_not_a_crash(self):
        self.set_behavior(mode="crash")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "coral.exit").exists())
        (self.root_of(ctx) / "coral.exit").write_text("124\n")  # what coreutils timeout exits with
        self.assertEqual(h.poll(), ExitReason.STOPPED_WALL)
        self.assertIn("in-container", h.collect().environment["wall_limit"])

    def test_crash_is_reported_with_the_exit_status(self):
        self.set_behavior(mode="crash")
        ctx, h = self.start()
        self.assertEqual(self.wait_exit(h), ExitReason.CRASHED)
        self.assertEqual(h.collect().environment["coral_exit"], "3")

    def test_a_container_killed_from_outside_is_a_crash(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        self.remote.run(["docker", "kill", f"glimmer-{ctx.spec.trial_id}"])
        self.assertEqual(self.wait_exit(h), ExitReason.CRASHED)
        self.assertEqual(h.collect().environment["coral_exit"], "137")

    def test_an_unreachable_docker_daemon_is_not_a_crash_until_it_stays_gone(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        (self.dstate / "daemon_down").write_text("")
        for _ in range(9):
            self.assertIsNone(h.poll())
        self.assertEqual(h.poll(), ExitReason.CRASHED)
        (self.dstate / "daemon_down").unlink()
        h.close()

    def test_a_stop_that_docker_cannot_confirm_is_not_reported_stopped(self):
        self.set_behavior(mode="hang")
        ctx, h = self.start()
        eventually(lambda: (self.root_of(ctx) / "run" / "start.json").exists())
        (self.dstate / "daemon_down").write_text("")
        h.close()
        self.assertTrue(self.root_of(ctx).exists())  # nothing deleted while the container's state is unknown
        (self.dstate / "daemon_down").unlink()
        with self.assertRaises(RuntimeFailure):
            self.runtime.start(self.context(trial_id="t-0002"))  # and the host stays claimed
        self.remote.run(["docker", "rm", "-f", f"glimmer-{ctx.spec.trial_id}"])


class BackwardCompatibilityTests(unittest.TestCase):
    def test_the_opencode_runtime_still_uses_its_own_scripts(self):
        # the hooks added to runtime_coral keep their defaults (the 57 tests of test_lab_runtime_coral cover the behaviour)
        h = rc.CoralHandle(remote=object(), layout=None, root="/b/t-1", base="/b", tunnel=None, trial_id="t-1", initial_context_sha256="0" * 64, grader_hashes={}, environment={}, clock=lambda: 0.0)
        self.assertEqual(h._poll_argv(), ["bash", "-c", rc.POLL_SCRIPT, "_", "/b/t-1"])
        self.assertEqual(h._stop_argv(), ["bash", "-c", rc.STOP_SCRIPT, "_", "/b/t-1"])
        self.assertTrue(h._remove_resources())
        self.assertTrue(rc.CoralRuntime._uses_llm_endpoint)
        self.assertFalse(cli.CoralCliRuntime._uses_llm_endpoint)


if __name__ == "__main__":
    unittest.main()
