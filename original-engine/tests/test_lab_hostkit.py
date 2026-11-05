import hashlib
import http.server
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nightshift.lab.contracts import AdapterFile, AdapterRef, adapter_ref_from_dir, canonical_listing_sha256

ROOT = Path(__file__).resolve().parents[1]
HOSTKIT = ROOT / "nightshift" / "lab" / "hostkit"
LIB = HOSTKIT / "lib"
RFT = HOSTKIT / "recipes" / "reference_rft"
PY = sys.executable


def sh_files():
    return sorted(p for p in HOSTKIT.rglob("*") if p.is_file() and (p.suffix == ".sh" or p.parent.name == "bin"))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Static(unittest.TestCase):
    """Cheap structural checks (shellcheck is not required)."""

    def test_the_expected_files_exist(self):
        for rel in (
            "bootstrap.sh", "stage.sh", "run_job.sh", "serve.sh", "glimmer-qlora-text.yaml", "bin/glimmer-axolotl-train", "bin/glimmer-vllm-up",
            "lib/env.sh", "lib/check_recipe_tar.py", "lib/run_with_timeout.py", "lib/job_entry.py", "lib/glimmer_data.py", "lib/axolotl_train.py",
            "lib/reload_check.py", "lib/adapter_listing.py", "lib/verify_sha256.py", "recipes/reference_rft/run.sh",
            "recipes/reference_rft/rft_pipeline.py", "recipes/reference_rft/config.yaml",
        ):  # fmt: skip
            self.assertTrue((HOSTKIT / rel).is_file(), rel)

    def test_every_shell_script_passes_bash_n(self):
        files = sh_files()
        self.assertGreaterEqual(len(files), 7)
        for f in files:
            with self.subTest(f=f.name):
                r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_scripts_are_executable_and_have_a_shebang(self):
        for f in sh_files():
            if f.name == "env.sh":  # sourced, never executed
                continue
            with self.subTest(f=f.name):
                self.assertTrue(os.access(f, os.X_OK))
                self.assertTrue(f.read_text().startswith("#!/usr/bin/env bash"))

    def test_every_python_file_compiles(self):
        for f in HOSTKIT.rglob("*.py"):
            with self.subTest(f=f.name):
                compile(f.read_text(), str(f), "exec")

    def test_every_path_a_script_references_exists_in_the_hostkit_tree(self):
        ref = re.compile(r"\$\{?HOSTKIT\}?/([A-Za-z0-9_./-]+)")
        checked = 0
        for f in list(sh_files()) + list(HOSTKIT.rglob("*.py")) + list(HOSTKIT.rglob("*.md")):
            for m in ref.finditer(f.read_text()):
                rel = m.group(1).rstrip(".,;:)")
                checked += 1
                with self.subTest(script=f.name, ref=rel):
                    self.assertTrue((HOSTKIT / rel).exists(), f"{f.name} references $HOSTKIT/{rel}")
        self.assertGreater(checked, 15)
        # the relative references inside recipes
        here_refs = re.findall(r"\$HERE/([A-Za-z0-9_./-]+)", (RFT / "run.sh").read_text())
        self.assertGreaterEqual(len(here_refs), 2)
        for rel in here_refs:
            self.assertTrue((RFT / rel.rstrip(".,;:)")).exists(), rel)

    def test_every_script_sources_the_shared_environment_first(self):
        for f in sh_files():
            if f.name == "env.sh":
                continue
            with self.subTest(f=f.name):
                self.assertIn('source "$HOSTKIT/lib/env.sh"', f.read_text())

    def test_no_script_names_selection_heldout_or_test_data(self):
        banned = re.compile(r"selection\.jsonl|IFBench_test|ifbench_test|held-?out\.jsonl|/private/", re.I)
        for f in HOSTKIT.rglob("*"):
            if f.is_file() and f.suffix in (".sh", ".py", ".yaml", "") and "selection" not in f.name:
                for m in banned.finditer(f.read_text(errors="ignore")):
                    # documentation comments may say that these are NEVER staged; code must not open them
                    line = f.read_text(errors="ignore")[: m.start()].count("\n")
                    text = f.read_text(errors="ignore").splitlines()[line]
                    self.assertTrue(text.lstrip().startswith(("#", '"""', "Only", "never", "-")) or "never" in text, f"{f.name}:{line + 1}: {text}")

    def test_the_pins_in_the_scripts_match_the_design(self):
        env = (LIB / "env.sh").read_text() + (HOSTKIT / "bootstrap.sh").read_text() + (HOSTKIT / "serve.sh").read_text()
        for pin in ("a4e59da52a7bc87ae7251dd5545c0dd437c44b68", "axolotl==0.19.0", "torch==2.12.1", "flash-attn==2.8.3", "vllm==0.32.0",
                    "4dfa52212d0552805aec5a6a2176c0abbea277d2", "--max-lora-rank 32", "--enable-lora", "--reasoning-parser muse_glimmer"):  # fmt: skip
            self.assertIn(pin, env)


def run_script(args, env=None, cwd=None, timeout=120, input_text=None):
    return subprocess.run(args, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout, input=input_text)


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name).resolve()
        self.opt = self.tmp / "opt"
        self.work = self.tmp / "work"
        self.models = self.tmp / "models"
        for d in (self.opt / "bin", self.work, self.models):
            d.mkdir(parents=True)
        self.env = dict(
            os.environ, HOSTKIT=str(HOSTKIT), LAB_OPT=str(self.opt), LAB_WORK=str(self.work), LAB_MODELS=str(self.models),
            LAB_PYTHON=PY, LAB_SKIP_RELOAD_CHECK="1", PYTHONDONTWRITEBYTECODE="1",
        )  # fmt: skip
        self.out = self.work / "out"

    def stub(self, name, body, directory=None):
        p = (directory or self.opt / "bin") / name
        p.write_text("#!/usr/bin/env bash\n" + body)
        p.chmod(0o755)
        return p


def make_recipe_tar(path: Path, members: dict) -> Path:
    """members: name -> bytes (file) | ('symlink', target) | ('hardlink', target) | ('dir',) | ('exec', bytes)"""
    with tarfile.open(path, "w:gz") as tar:
        for name, v in members.items():
            ti = tarfile.TarInfo(name)
            if isinstance(v, bytes):
                ti.size, ti.mode = len(v), 0o644
                tar.addfile(ti, io.BytesIO(v))
            elif v[0] == "exec":
                ti.size, ti.mode = len(v[1]), 0o755
                tar.addfile(ti, io.BytesIO(v[1]))
            elif v[0] == "symlink":
                ti.type, ti.linkname = tarfile.SYMTYPE, v[1]
                tar.addfile(ti)
            elif v[0] == "hardlink":
                ti.type, ti.linkname = tarfile.LNKTYPE, v[1]
                tar.addfile(ti)
            elif v[0] == "dir":
                ti.type, ti.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(ti)
            elif v[0] == "chardev":
                ti.type, ti.devmajor, ti.devminor = tarfile.CHRTYPE, 1, 3
                tar.addfile(ti)
    return path


class RecipeTar(Base):
    def check(self, members, *extra, **kw):
        tar = make_recipe_tar(self.tmp / "r.tar.gz", members)
        return run_script([PY, "-I", str(LIB / "check_recipe_tar.py"), str(tar), *extra], **kw)

    def test_valid_recipe_extracts_with_modes(self):
        dest = self.tmp / "x"
        r = self.check({"run.sh": ("exec", b"echo hi\n"), "sub/": ("dir",), "sub/a.txt": b"a", "config.yaml": b"k: v\n"}, "--extract", str(dest), "--entrypoint", "run.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.access(dest / "run.sh", os.X_OK))
        self.assertEqual((dest / "sub" / "a.txt").read_text(), "a")

    def test_hostile_archives_are_refused_and_extract_nothing(self):
        cases = {
            "symlink": {"a": ("symlink", "/etc/passwd")},
            "hardlink": {"a": b"x", "b": ("hardlink", "a")},
            "device": {"a": ("chardev",)},
            "parent traversal": {"../evil": b"x"},
            "nested traversal": {"a/../../evil": b"x"},
            "absolute": {"/etc/evil": b"x"},
            ".git component": {".git/config": b"x"},
            "nested .git": {"sub/.git/hooks/pre-commit": b"x"},
            "dot": {".": ("dir",)},
            "backslash": {"a\\b": b"x"},
        }
        for name, members in cases.items():
            dest = self.tmp / f"x-{re.sub(r'[^a-z]', '', name)}"
            with self.subTest(name):
                r = self.check(members, "--extract", str(dest))
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("refused", r.stderr)
                self.assertFalse(dest.exists() and any(dest.iterdir()), "nothing may be extracted from a refused archive")
        self.assertFalse((self.tmp.parent / "evil").exists())

    def test_limits(self):
        many = {f"f{i}": b"x" for i in range(30)}
        self.assertEqual(self.check(many, "--max-entries", "10").returncode, 1)
        self.assertEqual(self.check({"a": b"x" * 100}, "--max-unpacked", "50").returncode, 1)
        self.assertEqual(self.check({"a": b"x" * 100}, "--max-unpacked", "500").returncode, 0)

    def test_entrypoint_must_be_a_regular_file_in_the_archive(self):
        self.assertEqual(self.check({"a": b"x"}, "--entrypoint", "run.sh").returncode, 1)

    def test_not_a_tarball_and_a_symlinked_tarball(self):
        junk = self.tmp / "junk.tar.gz"
        junk.write_bytes(b"not a tar")
        self.assertEqual(run_script([PY, "-I", str(LIB / "check_recipe_tar.py"), str(junk)]).returncode, 1)
        real = make_recipe_tar(self.tmp / "real.tar.gz", {"a": b"x"})
        link = self.tmp / "link.tar.gz"
        link.symlink_to(real)
        self.assertEqual(run_script([PY, "-I", str(LIB / "check_recipe_tar.py"), str(link)]).returncode, 1)

    def test_extract_never_writes_through_a_preplanted_symlink(self):
        dest = self.tmp / "x"
        dest.mkdir()
        canary = self.tmp / "canary.txt"
        canary.write_text("untouched")
        (dest / "a").symlink_to(canary)
        r = self.check({"a": b"overwrite"}, "--extract", str(dest))
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(canary.read_text(), "untouched")


class Timeout(Base):
    def run_to(self, deadline, *cmd, grace=1):
        return run_script([PY, "-I", str(LIB / "run_with_timeout.py"), "--deadline-epoch", str(deadline), "--grace", str(grace), "--", *cmd])

    def test_passes_through_the_exit_status(self):
        self.assertEqual(self.run_to(time.time() + 30, "bash", "-c", "exit 3").returncode, 3)
        self.assertEqual(self.run_to(time.time() + 30, "bash", "-c", "exit 0").returncode, 0)

    def test_kills_at_the_deadline(self):
        t0 = time.time()
        r = self.run_to(time.time() + 1, "bash", "-c", "sleep 30")
        self.assertEqual(r.returncode, 124)
        self.assertLess(time.time() - t0, 10)

    def test_kills_a_term_ignoring_command_after_the_grace_period(self):
        r = self.run_to(time.time() + 1, "bash", "-c", "trap '' TERM; sleep 30")
        self.assertEqual(r.returncode, 124)

    def test_kills_background_leftovers_when_the_command_exits(self):
        marker = self.tmp / "left.txt"
        r = self.run_to(time.time() + 30, "bash", "-c", f"(sleep 2; echo alive > {marker}) & exit 0")
        self.assertEqual(r.returncode, 0)
        time.sleep(3)
        self.assertFalse(marker.exists(), "a forgotten background process must not outlive the job")

    def test_missing_command(self):
        self.assertEqual(self.run_to(time.time() + 5, "/nonexistent/binary").returncode, 127)


class DataAndHashes(Base):
    def run_reg(self, lines):
        d = self.tmp / "d.jsonl"
        d.write_text("\n".join(lines) + "\n")
        return run_script([PY, "-I", str(LIB / "glimmer_data.py"), "register", str(d), str(self.out)])

    def test_register_appends_valid_rows(self):
        row = json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
        self.assertEqual(self.run_reg([row, row]).returncode, 0)
        self.assertEqual(self.run_reg([row]).returncode, 0)
        self.assertEqual(len((self.out / "train_data.jsonl").read_text().splitlines()), 3)

    def test_register_rejects_bad_rows_and_writes_nothing(self):
        good = json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
        bad = {
            "not json": "{nope",
            "no messages": json.dumps({"text": "x"}),
            "images": json.dumps({"messages": [{"role": "assistant", "content": "a"}], "images": ["x"]}),
            "no assistant": json.dumps({"messages": [{"role": "user", "content": "q"}]}),
            "bad role": json.dumps({"messages": [{"role": "root", "content": "q"}]}),
        }
        for name, line in bad.items():
            with self.subTest(name):
                self.assertNotEqual(self.run_reg([good, line]).returncode, 0)
        self.assertFalse((self.out / "train_data.jsonl").exists())

    def test_register_rejects_a_symlinked_input(self):
        real = self.tmp / "real.jsonl"
        real.write_text(json.dumps({"messages": [{"role": "assistant", "content": "a"}]}) + "\n")
        link = self.tmp / "link.jsonl"
        link.symlink_to(real)
        r = run_script([PY, "-I", str(LIB / "glimmer_data.py"), "register", str(link), str(self.out)])
        self.assertNotEqual(r.returncode, 0)

    def test_adapter_listing_equals_the_contracts_hash(self):
        d = self.tmp / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 32}')
        (d / "adapter_model.safetensors").write_bytes(os.urandom(2048))
        (d / "checkpoint-9").mkdir()
        expect = adapter_ref_from_dir(d, adapter_id="ad1").sha256
        r = run_script([PY, "-I", str(LIB / "adapter_listing.py"), str(d)])
        self.assertEqual(r.stdout.strip(), expect)
        (d / "adapter_model.safetensors").unlink()
        self.assertNotEqual(run_script([PY, "-I", str(LIB / "adapter_listing.py"), str(d)]).returncode, 0)

    def test_adapter_listing_refuses_links(self):
        d = self.tmp / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        real = self.tmp / "w.bin"
        real.write_bytes(b"w")
        (d / "adapter_model.safetensors").symlink_to(real)
        self.assertNotEqual(run_script([PY, "-I", str(LIB / "adapter_listing.py"), str(d)]).returncode, 0)

    def test_verify_sha256(self):
        root = self.tmp / "w"
        (root / "sub").mkdir(parents=True)
        (root / "a.bin").write_bytes(b"AAA")
        (root / "sub" / "b.bin").write_bytes(b"BBB")
        pins = self.tmp / "pins"
        pins.write_text(f"{sha(b'AAA')}  a.bin\n{sha(b'BBB')}  sub/b.bin\n")
        v = [PY, "-I", str(LIB / "verify_sha256.py"), str(root), str(pins)]
        self.assertEqual(run_script(v).returncode, 0)
        (root / "a.bin").write_bytes(b"tampered")
        r = run_script(v)
        self.assertEqual(r.returncode, 1)
        self.assertIn("a.bin", r.stderr)
        for evil in (f"{sha(b'x')}  ../outside", f"{sha(b'x')}  /etc/passwd", "short  a.bin"):
            pins.write_text(evil + "\n")
            self.assertEqual(run_script(v).returncode, 1)


class Stage(Base):
    def bundle(self, files: dict, pins=None):
        b = self.tmp / "bundle"
        shutil.rmtree(b, ignore_errors=True)
        for rel, data in files.items():
            (b / rel).parent.mkdir(parents=True, exist_ok=True)
            (b / rel).write_bytes(data)
        lines = pins if pins is not None else [f"{sha(d)}  {rel}" for rel, d in files.items()]
        (b / "pins.sha256").write_text("\n".join(lines) + "\n")
        return b

    def stage(self, b):
        return run_script(["bash", str(HOSTKIT / "stage.sh"), str(b)], env=self.env)

    def test_stage_installs_pinned_files_and_launchers(self):
        b = self.bundle({"ifevalg/instructions.py": b"I = 1\n", "ifevalg/instructions_registry.py": b"R = 1\n", "rewards.py": b"def reward(): pass\n", "weights.sha256": b"\n"})
        r = self.stage(b)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.opt / "ifevalg" / "instructions.py").read_bytes(), b"I = 1\n")
        self.assertEqual((self.opt / "rewards.py").read_bytes(), b"def reward(): pass\n")
        launcher = self.opt / "bin" / "glimmer-vllm-up"
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertIn(str(HOSTKIT), launcher.read_text())

    def test_a_wrong_hash_installs_nothing(self):
        b = self.bundle({"rewards.py": b"good", "ifevalg/instructions.py": b"x"}, pins=[f"{sha(b'good')}  rewards.py", f"{sha(b'other')}  ifevalg/instructions.py"])
        r = self.stage(b)
        self.assertEqual(r.returncode, 1)
        self.assertFalse((self.opt / "rewards.py").exists())
        self.assertFalse((self.opt / "ifevalg" / "instructions.py").exists())

    def test_an_unpinned_file_in_the_bundle_installs_nothing(self):
        b = self.bundle({"rewards.py": b"good", "selection.jsonl": b'{"ground_truth": "secret"}'}, pins=[f"{sha(b'good')}  rewards.py"])
        r = self.stage(b)
        self.assertEqual(r.returncode, 1)
        self.assertIn("not pinned", r.stderr)
        self.assertFalse((self.opt / "rewards.py").exists())

    def test_missing_pins_file(self):
        b = self.tmp / "b2"
        b.mkdir()
        (b / "rewards.py").write_text("x")
        self.assertEqual(self.stage(b).returncode, 1)


class Serve(Base):
    def make_adapter(self, name="ad1"):
        d = self.tmp / name
        d.mkdir()
        (d / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 32}')
        (d / "adapter_model.safetensors").write_bytes(b"weights-" + name.encode())
        return d, adapter_ref_from_dir(d, adapter_id=name).sha256

    def serve(self, *args, **env):
        e = dict(self.env, LAB_DRY_RUN="1", **env)
        return run_script(["bash", str(HOSTKIT / "serve.sh"), *args], env=e)

    def test_base_only_command(self):
        r = self.serve("--port", "8123")
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = r.stdout.split("\n")
        self.assertIn("--served-model-name", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "8123")
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "8300")
        self.assertIn("--enforce-eager", argv)
        self.assertIn("muse_glimmer", argv)
        self.assertNotIn("--enable-lora", argv)
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1", "the server is reachable only through the ssh forward")

    def test_adapters_are_served_by_name_after_a_hash_check(self):
        d1, s1 = self.make_adapter("ad1")
        d2, s2 = self.make_adapter("ad2")
        r = self.serve("--adapter", f"lora-one={d1}={s1}", "--adapter", f"lora-two={d2}={s2}")
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = r.stdout.split("\n")
        self.assertIn("--enable-lora", argv)
        self.assertEqual(argv[argv.index("--max-lora-rank") + 1], "32")
        self.assertEqual(argv[argv.index("--max-loras") + 1], "2")
        i = argv.index("--lora-modules")
        self.assertEqual(argv[i + 1 : i + 3], [f"lora-one={d1}", f"lora-two={d2}"])

    def test_an_adapter_whose_hash_differs_is_never_served(self):
        d1, s1 = self.make_adapter("ad1")
        (d1 / "adapter_model.safetensors").write_bytes(b"swapped after the control host verified it")
        r = self.serve("--adapter", f"lora-one={d1}={s1}")
        self.assertEqual(r.returncode, 3)
        self.assertIn("refusing to serve", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_weights_pin_mismatch_stops_the_server(self):
        (self.models / "glimmer").mkdir()
        (self.models / "glimmer" / "w.safetensors").write_bytes(b"real")
        (self.opt / "weights.sha256").write_text(f"{sha(b'other')}  w.safetensors\n")
        r = self.serve()
        self.assertEqual(r.returncode, 4)
        (self.opt / "weights.sha256").write_text(f"{sha(b'real')}  w.safetensors\n")
        self.assertEqual(self.serve().returncode, 0)

    def test_bad_arguments(self):
        for args in (["--port", "80; rm -rf /"], ["--served-name", "a b"], ["--adapter", "x=y"], ["--bogus"], ["--adapter", "n=/d=nothex"]):
            with self.subTest(args=args):
                self.assertEqual(self.serve(*args).returncode, 2)


class FakeVllm:
    """A loopback OpenAI-compatible server. ``replies(prompt, n)`` returns a list of (content, finish_reason, reasoning)."""

    def __init__(self, replies=None):
        self.requests = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = b'{"data": [{"id": "muse-glimmer"}]}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(req)
                prompt = req["messages"][-1]["content"]
                choices = []
                for i, (text, finish, reasoning) in enumerate((replies or (lambda p, n: [(f"answer {i} to {p}", "stop", "thinking") for i in range(n)]))(prompt, req.get("n", 1))):
                    msg = {"role": "assistant", "content": text}
                    if reasoning:
                        msg["reasoning"] = reasoning
                    choices.append({"index": i, "message": msg, "finish_reason": finish})
                body = json.dumps({"choices": choices, "usage": {"completion_tokens": 7 * len(choices)}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class VllmUp(Base):
    def test_start_waits_for_readiness_and_stop_kills_it(self):
        fake = FakeVllm()
        self.addCleanup(fake.close)
        # a stand-in for serve.sh that just stays alive; the readiness probe talks to the fake server on the same port
        stub = self.stub("fake_serve.sh", "while true; do sleep 1; done\n", directory=self.tmp)
        env = dict(self.env, LAB_SERVE_SH=str(stub))
        r = run_script(["bash", str(HOSTKIT / "bin" / "glimmer-vllm-up"), "--port", str(fake.port), "--wait", "20"], env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), f"http://127.0.0.1:{fake.port}")
        pid = int((self.work / "vllm.pid").read_text())
        os.kill(pid, 0)  # alive
        r = run_script(["bash", str(HOSTKIT / "bin" / "glimmer-vllm-up"), "--stop"], env=env)
        self.assertEqual(r.returncode, 0)
        with self.assertRaises(ProcessLookupError):
            for _ in range(50):
                os.kill(pid, 0)
                time.sleep(0.1)
            os.kill(pid, 0)
        self.assertFalse((self.work / "vllm.pid").exists())

    def test_a_server_that_dies_is_reported(self):
        stub = self.stub("dies.sh", "echo boom >&2; exit 1\n", directory=self.tmp)
        env = dict(self.env, LAB_SERVE_SH=str(stub))
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        r = run_script(["bash", str(HOSTKIT / "bin" / "glimmer-vllm-up"), "--port", str(port), "--wait", "20"], env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("boom", r.stderr)

    def test_stop_without_a_server_is_fine(self):
        self.assertEqual(run_script(["bash", str(HOSTKIT / "bin" / "glimmer-vllm-up"), "--stop"], env=self.env).returncode, 0)

    def test_bad_argument(self):
        self.assertEqual(run_script(["bash", str(HOSTKIT / "bin" / "glimmer-vllm-up"), "--nope"], env=self.env).returncode, 2)


class RunJob(Base):
    """The job contract, end to end, with the heavy helpers replaced by stubs on PATH."""

    def setUp(self):
        super().setUp()
        (self.work / "data").mkdir()
        (self.work / "data" / "train.jsonl").write_text('{"key": "k1", "prompt": "p"}\n')
        # the stand-in for glimmer-axolotl-train: really registers the data (the real helper script), then writes an adapter
        self.stub(
            "glimmer-axolotl-train",
            f'set -e\n"{PY}" -I "{LIB / "glimmer_data.py"}" register "$3" "$GLIMMER_OUT"\n'
            'mkdir -p "$GLIMMER_OUT/adapter"\n'
            'echo \'{"peft_type": "LORA", "r": 32}\' > "$GLIMMER_OUT/adapter/adapter_config.json"\n'
            'echo weights > "$GLIMMER_OUT/adapter/adapter_model.safetensors"\n'
            'mkdir -p "$GLIMMER_OUT/adapter/checkpoint-3"; echo junk > "$GLIMMER_OUT/adapter/checkpoint-3/x"\n',
        )

    def submit(self, members, entrypoint=("bash", "run.sh"), deadline_in=60):
        make_recipe_tar(self.work / "recipe.tar.gz", members)
        (self.work / "job.json").write_text(json.dumps({"entrypoint": list(entrypoint), "deadline_epoch": int(time.time()) + deadline_in}))
        r = run_script(["bash", str(HOSTKIT / "run_job.sh")], env=self.env, timeout=120)
        done = json.loads((self.out / "trial.done").read_text())
        return r, done

    TRAIN_SH = b'echo \'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\' > sft.jsonl\nglimmer-axolotl-train config.yaml --data sft.jsonl\n'

    def test_a_good_recipe_ends_ok_with_the_adapter_and_the_training_data(self):
        r, done = self.submit({"run.sh": ("exec", self.TRAIN_SH), "config.yaml": b"x: 1\n"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((done["reason"], done["exit_code"]), ("ok", 0))
        self.assertTrue((self.out / "adapter" / "adapter_model.safetensors").exists())
        self.assertEqual(len((self.out / "train_data.jsonl").read_text().splitlines()), 1)
        self.assertIn("starting the entrypoint", (self.out / "job.log").read_text())

    def test_trial_done_is_written_last(self):
        self.submit({"run.sh": ("exec", self.TRAIN_SH), "config.yaml": b"x: 1\n"})
        self.assertGreaterEqual((self.out / "trial.done").stat().st_mtime_ns, (self.out / "adapter" / "adapter_model.safetensors").stat().st_mtime_ns)

    def test_a_failing_recipe(self):
        _, done = self.submit({"run.sh": ("exec", b"exit 7\n")})
        self.assertEqual((done["reason"], done["exit_code"]), ("recipe_failed", 7))

    def test_a_recipe_past_its_deadline_is_killed(self):
        t0 = time.time()
        _, done = self.submit({"run.sh": ("exec", b"sleep 60\n")}, deadline_in=2)
        self.assertEqual(done["reason"], "timeout")
        self.assertLess(time.time() - t0, 40)

    def test_a_recipe_without_an_adapter(self):
        _, done = self.submit({"run.sh": ("exec", b"echo hi\n")})
        self.assertEqual(done["reason"], "no_adapter")

    def test_a_symlinked_adapter_file_does_not_count(self):
        body = b'mkdir -p "$GLIMMER_OUT/adapter"; echo c > "$GLIMMER_OUT/adapter/adapter_config.json"; ln -s /etc/hosts "$GLIMMER_OUT/adapter/adapter_model.safetensors"\n'
        _, done = self.submit({"run.sh": ("exec", body)})
        self.assertEqual(done["reason"], "no_adapter")

    def test_a_job_that_never_called_a_helper_has_no_training_data_and_fails(self):
        body = b'mkdir -p "$GLIMMER_OUT/adapter"; echo c > "$GLIMMER_OUT/adapter/adapter_config.json"; echo w > "$GLIMMER_OUT/adapter/adapter_model.safetensors"\n'
        _, done = self.submit({"run.sh": ("exec", body)})
        self.assertEqual(done["reason"], "no_train_data")

    def test_a_hostile_tarball_is_refused_before_anything_runs(self):
        marker = self.tmp / "ran.txt"
        for members in ({"run.sh": ("exec", f"touch {marker}\n".encode()), "evil": ("symlink", "/etc/passwd")}, {"../escape.sh": b"x", "run.sh": b"x"}):
            r, done = self.submit(members)
            self.assertEqual(done["reason"], "bad_recipe")
        self.assertFalse(marker.exists())
        self.assertFalse((self.work / "recipe").exists() and any((self.work / "recipe").iterdir()))

    def test_malformed_job_json(self):
        make_recipe_tar(self.work / "recipe.tar.gz", {"run.sh": b"x"})
        (self.work / "job.json").write_text('{"entrypoint": "bash run.sh", "deadline_epoch": 1}')
        run_script(["bash", str(HOSTKIT / "run_job.sh")], env=self.env)
        self.assertEqual(json.loads((self.out / "trial.done").read_text())["reason"], "bad_recipe")

    def test_the_entrypoint_is_never_parsed_by_a_shell(self):
        marker = self.tmp / "injected.txt"
        _, done = self.submit({"run.sh": ("exec", b'echo "$1" > arg.txt\n' + self.TRAIN_SH), "config.yaml": b"x: 1\n"}, entrypoint=("bash", "run.sh", f"; touch {marker} #"))
        self.assertEqual(done["reason"], "ok")
        self.assertFalse(marker.exists())
        self.assertEqual((self.work / "recipe" / "arg.txt").read_text().strip(), f"; touch {marker} #")

    def test_the_recipe_sees_the_documented_environment(self):
        body = b'env | grep -E "^(GLIMMER_BASE|GLIMMER_OUT|GLIMMER_CONFIG_TEMPLATE|IFEVALG_PATH|REWARDS_PY)=" > env.txt\ncommand -v glimmer-axolotl-train >> env.txt\n' + self.TRAIN_SH
        self.submit({"run.sh": ("exec", body), "config.yaml": b"x: 1\n"})
        text = (self.work / "recipe" / "env.txt").read_text()
        for name in ("GLIMMER_BASE", "GLIMMER_OUT", "GLIMMER_CONFIG_TEMPLATE", "IFEVALG_PATH", "REWARDS_PY"):
            self.assertIn(name + "=", text)
        self.assertIn(str(self.opt / "bin" / "glimmer-axolotl-train"), text)

    def test_a_forgotten_background_process_does_not_outlive_the_job(self):
        marker = self.tmp / "bg.txt"
        body = f"(sleep 2; echo alive > {marker}) &\n".encode() + self.TRAIN_SH
        self.submit({"run.sh": ("exec", body), "config.yaml": b"x: 1\n"})
        time.sleep(3)
        self.assertFalse(marker.exists())

    def test_a_stale_result_from_an_earlier_run_is_cleared(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "train_data.jsonl").write_text('{"messages": []}\n')
        (self.out / "adapter").mkdir()
        (self.out / "adapter" / "adapter_config.json").write_text("{}")
        (self.out / "adapter" / "adapter_model.safetensors").write_text("old")
        _, done = self.submit({"run.sh": ("exec", b"echo nothing\n")})
        self.assertEqual(done["reason"], "no_adapter")


FAKE_REWARDS = (
    "def preflight():\n    pass\n"
    "def verify_constraints(response, ground_truth):\n    return [ground_truth in response]\n"
)  # a stand-in verifier: the constraint is "the answer contains the ground-truth string"


class OriginFixtures(Base):
    """A job directory with approved train rows, a recorded sample for each, and a stand-in verifier (the real one needs NLTK data)."""

    def setUp(self):
        super().setUp()
        (self.work / "data").mkdir()
        self.train_rows = [{"key": "k1", "prompt": "Write about cats.", "ground_truth": "meow"}, {"key": "k2", "prompt": "Write about dogs.", "ground_truth": "woof"}]
        (self.work / "data" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.train_rows))
        (self.opt / "rewards.py").write_text(FAKE_REWARDS)
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples = [
            {"key": "k1", "model": "muse-glimmer", "answer": "meow meow", "reasoning": "think cat", "finish_reason": "stop"},
            {"key": "k2", "model": "muse-glimmer", "answer": "woof", "reasoning": None, "finish_reason": "stop"},
        ]
        self.write_samples(self.samples)

    def write_samples(self, samples):
        (self.out / "glimmer_samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))

    def data_file(self, rows, name="d.jsonl"):
        p = self.tmp / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    @staticmethod
    def row(prompt, answer, reasoning=None, system=None):
        a = {"role": "assistant", "content": answer}
        if reasoning is not None:
            a["reasoning_content"] = reasoning
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}, a]
        return {"messages": msgs}

    def check(self, path):
        return run_script([PY, "-I", str(LIB / "data_origin.py"), "check", str(path)], env=self.env)


class OriginCheck(OriginFixtures):
    def test_samples_of_the_local_model_on_approved_prompts_pass_and_are_recorded(self):
        d = self.data_file([self.row("Write about cats.", "meow meow", "think cat", system="s"), self.row("Write about dogs.", "woof")])
        r = self.check(d)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads((self.out / "origin_report.json").read_text())
        self.assertEqual(report[0]["rows"], 2)
        self.assertEqual(report[0]["data_sha256"], hashlib.sha256(d.read_bytes()).hexdigest())

    def test_a_hand_written_or_other_model_answer_stops_the_job(self):
        r = self.check(self.data_file([self.row("Write about cats.", "meow, written by someone else")]))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("do not come from the local Glimmer model", r.stderr)
        self.assertFalse((self.out / "origin_report.json").exists())

    def test_a_prompt_that_is_not_an_approved_training_prompt_stops_the_job(self):
        self.assertNotEqual(self.check(self.data_file([self.row("A selection prompt", "meow meow")])).returncode, 0)

    def test_a_sample_from_another_model_or_a_truncated_one_does_not_count(self):
        for patch in ({"model": "muse-spark-1.3"}, {"finish_reason": "length"}):
            self.write_samples([dict(self.samples[0], **patch)])
            self.assertNotEqual(self.check(self.data_file([self.row("Write about cats.", "meow meow")])).returncode, 0, patch)

    def test_a_sample_the_verifier_rejects_does_not_count(self):
        self.write_samples([dict(self.samples[0], answer="purr")])
        self.assertNotEqual(self.check(self.data_file([self.row("Write about cats.", "purr")])).returncode, 0)

    def test_the_wrong_reasoning_or_the_wrong_prompt_pairing_stops_the_job(self):
        self.assertNotEqual(self.check(self.data_file([self.row("Write about cats.", "meow meow", "invented reasoning")])).returncode, 0)
        self.assertNotEqual(self.check(self.data_file([self.row("Write about dogs.", "meow meow")])).returncode, 0)  # a k1 answer under the k2 prompt

    def test_unexpected_roles_and_extra_turns_are_refused(self):
        tool = {"messages": [{"role": "user", "content": "Write about cats."}, {"role": "tool", "content": "x"}, {"role": "assistant", "content": "meow meow"}]}
        twice = {"messages": [{"role": "user", "content": "Write about cats."}, {"role": "assistant", "content": "meow meow"}, {"role": "assistant", "content": "meow meow"}]}
        for bad in (tool, twice):
            self.assertNotEqual(self.check(self.data_file([bad])).returncode, 0)

    def test_without_recorded_samples_or_approved_prompts_the_origin_cannot_be_established(self):
        d = self.data_file([self.row("Write about cats.", "meow meow")])
        (self.out / "glimmer_samples.jsonl").unlink()
        r = self.check(d)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot be established", r.stderr)
        self.write_samples(self.samples)
        (self.work / "data" / "train.jsonl").unlink()
        self.assertNotEqual(self.check(d).returncode, 0)

    def test_an_empty_data_file_is_refused(self):
        self.assertNotEqual(self.check(self.data_file([])).returncode, 0)


class AxolotlHelper(OriginFixtures):
    def test_usage_errors(self):
        self.assertEqual(run_script(["bash", str(HOSTKIT / "bin" / "glimmer-axolotl-train")], env=self.env).returncode, 2)
        self.assertEqual(run_script(["bash", str(HOSTKIT / "bin" / "glimmer-axolotl-train"), "missing.yaml", "--data", "d.jsonl"], env=self.env).returncode, 2)

    def test_origin_and_registration_happen_before_training_and_bad_data_stops_it(self):
        cfg = self.tmp / "c.yaml"
        cfg.write_text("a: 1\n")
        good = self.data_file([self.row("Write about cats.", "meow meow", "think cat")], "g.jsonl")
        (self.opt / "venv-axolotl" / "bin").mkdir(parents=True)
        py = self.opt / "venv-axolotl" / "bin" / "python"
        py.write_text(f'#!/usr/bin/env bash\necho "$@" > "{self.tmp}/trainer_args.txt"\n')
        py.chmod(0o755)
        r = run_script(["bash", str(HOSTKIT / "bin" / "glimmer-axolotl-train"), str(cfg), "--data", str(good)], env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len((self.out / "train_data.jsonl").read_text().splitlines()), 1)
        self.assertIn("axolotl_train.py", (self.tmp / "trainer_args.txt").read_text())
        self.assertIn("origin check passed", r.stdout)
        for name, text in (("b.jsonl", "not json\n"), ("h.jsonl", json.dumps(self.row("Write about cats.", "hand-written")) + "\n")):
            bad = self.tmp / name
            bad.write_text(text)
            (self.tmp / "trainer_args.txt").unlink(missing_ok=True)
            r = run_script(["bash", str(HOSTKIT / "bin" / "glimmer-axolotl-train"), str(cfg), "--data", str(bad)], env=self.env)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((self.tmp / "trainer_args.txt").exists(), "training must not start on data of unknown origin")

    def test_train_command_finds_axolotl_in_the_venv_without_it_being_on_path(self):
        spec = importlib.util.spec_from_file_location("axolotl_train", LIB / "axolotl_train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        venv_bin = "/opt/lab/venv-axolotl/bin"
        argv, env = mod.train_command("/work/out/harness-config.yaml", venv_bin + "/python", {"PATH": "/opt/lab/bin:/usr/bin"})
        self.assertEqual(argv[:3], [venv_bin + "/axolotl", "train", "/work/out/harness-config.yaml"])
        self.assertEqual(env["PATH"], venv_bin + os.pathsep + "/opt/lab/bin:/usr/bin", "torchrun is found in the venv too")
        self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["WANDB_MODE"]), ("0", "disabled"))

    def test_the_dose_is_computed_from_the_exact_rows_and_recorded_not_refused(self):
        spec = importlib.util.spec_from_file_location("axolotl_train", LIB / "axolotl_train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        by_epochs = mod.dose({"micro_batch_size": 1, "gradient_accumulation_steps": 4, "num_epochs": 2}, 321)
        self.assertEqual((by_epochs["steps_per_epoch"], by_epochs["steps"], by_epochs["examples_seen"], by_epochs["epochs"]), (81, 162, 648, 2.0))
        self.assertEqual(by_epochs["checks"], [])
        by_steps = mod.dose({"micro_batch_size": 1, "gradient_accumulation_steps": 4, "num_epochs": 2, "max_steps": 300}, 320)
        self.assertEqual((by_steps["steps"], by_steps["examples_seen"]), (300, 1200))
        self.assertAlmostEqual(by_steps["epochs"], 1200 / 320)
        self.assertIn("max_steps=300 overrides num_epochs=2", by_steps["checks"])
        short = mod.dose({"gradient_accumulation_steps": 4, "max_steps": 10}, 320)
        self.assertTrue(any(c.startswith("less than one epoch: 40 examples seen of 320 rows") for c in short["checks"]))
        self.assertIn("no training rows", mod.dose({"num_epochs": 1}, 0)["checks"])
        data = self.tmp / "rows.jsonl"
        data.write_text('{"messages": []}\n\n{"messages": []}\n')
        self.assertEqual(mod.count_rows(str(data)), 2)
        out = self.tmp / "dose-out"
        mod.record_dose(str(out), by_steps)
        mod.record_dose(str(out), by_epochs)  # one line per training call (rounds)
        lines = [json.loads(l) for l in (out / "dose.jsonl").read_text().splitlines()]
        self.assertEqual([l["steps"] for l in lines], [300, 162])

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML not installed (the Axolotl environment has it)")
    def test_overlay_forces_the_harness_settings(self):
        spec = importlib.util.spec_from_file_location("axolotl_train", LIB / "axolotl_train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import yaml

        cfg = yaml.safe_load((HOSTKIT / "glimmer-qlora-text.yaml").read_text())
        cfg.update(wandb_project="mine", model_revision="x", hub_model_id="me/leak", push_to_hub=True, val_set_size=0.5, learning_rate=1e-4)
        out = mod.overlay(cfg, "/work/out/sft.jsonl", "/models/glimmer", "/work/out/adapter", "/work/cache/prepared")
        self.assertEqual(out["datasets"][0]["path"], "/work/out/sft.jsonl")
        self.assertEqual(out["output_dir"], "/work/out/adapter")
        self.assertEqual(out["val_set_size"], 0.0)
        self.assertEqual(out["learning_rate"], 1e-4, "the researcher's hyperparameters survive")
        for gone in ("wandb_project", "model_revision", "hub_model_id", "push_to_hub"):
            self.assertNotIn(gone, out)
        self.assertEqual(out["eot_tokens"], ["<|eot|>", "<|eom|>"])
        with self.assertRaises(SystemExit):
            mod.overlay({"datasets": [{}, {}]}, "d", "b", "o", "c")


STUB_REWARDS = '''
def preflight():
    pass
def error_counts():
    return {}
def verify_constraints(response, ground_truth):
    # a row passes iff the response contains the word the ground truth asks for
    return [ground_truth in response]
'''


class RftPipeline(Base):
    def setUp(self):
        super().setUp()
        rows = [{"key": f"k{i}", "prompt": f"prompt {i}", "ground_truth": f"word{i}", "instruction_ids": ["a:b"], "family": "f"} for i in range(6)]
        self.prompts = self.tmp / "train.jsonl"
        self.prompts.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.rewards = self.tmp / "rewards.py"
        self.rewards.write_text(STUB_REWARDS)
        self.env["GLIMMER_OUT"] = str(self.out)

    def pipe(self, *args):
        return run_script([PY, "-I", str(RFT / "rft_pipeline.py"), *args], env=self.env)

    def test_generate_then_filter(self):
        def replies(prompt, n):
            i = prompt.split()[-1]
            out = [(f"here is word{i}", "stop", "thought"), (f"here is word{i}", "stop", "thought"), ("no match", "stop", ""), (f"word{i} but cut", "length", "")]
            return out[:n]

        fake = FakeVllm(replies)
        self.addCleanup(fake.close)
        r = self.pipe("generate", "--prompts", str(self.prompts), "--out", str(self.tmp / "samples.jsonl"), "--base-url", f"http://127.0.0.1:{fake.port}", "--n-prompts", "4", "--samples", "4")
        self.assertEqual(r.returncode, 0, r.stderr)
        req = fake.requests[0]
        self.assertEqual(req["messages"][0], {"role": "system", "content": "You are a helpful AI assistant."}, "an explicit system message in every request")
        self.assertEqual((req["n"], req["temperature"], req["top_p"], req["top_k"]), (4, 1.0, 0.95, 64))
        self.assertEqual(req["chat_template_kwargs"], {"reasoning_strength": "high"})
        samples = [json.loads(line) for line in (self.tmp / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(len(samples), 16)
        self.assertEqual(len({s["key"] for s in samples}), 4)
        self.assertIn("generation_s", json.loads((self.out / "timings.json").read_text()))
        r = self.pipe("filter", "--samples", str(self.tmp / "samples.jsonl"), "--prompts", str(self.prompts), "--rewards", str(self.rewards),
                      "--out", str(self.tmp / "sft.jsonl"), "--stats", str(self.tmp / "stats.json"), "--keep-per-prompt", "2")  # fmt: skip
        self.assertEqual(r.returncode, 0, r.stderr)
        stats = json.loads((self.tmp / "stats.json").read_text())
        self.assertEqual((stats["n_samples"], stats["n_pass"], stats["n_truncated"], stats["n_prompts_with_pass"]), (16, 8, 4, 4))
        self.assertAlmostEqual(stats["yield_fraction"], 0.5)
        sft = [json.loads(line) for line in (self.tmp / "sft.jsonl").read_text().splitlines()]
        self.assertEqual(len(sft), 4, "duplicates are kept once: two identical passing answers per prompt give one row")
        for row in sft:
            self.assertEqual(list(row), ["messages"])
            self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
            self.assertEqual(row["messages"][2]["reasoning_content"], "thought")
            self.assertNotIn("cut", row["messages"][2]["content"], "truncated generations are never trained on")

    def test_filter_without_reasoning_and_with_nothing_passing(self):
        samples = self.tmp / "s.jsonl"
        samples.write_text(json.dumps({"key": "k1", "sample": 0, "text": "word1 ok", "reasoning": "r", "finish_reason": "stop", "completion_tokens": 3, "error": None}) + "\n")
        r = self.pipe("filter", "--samples", str(samples), "--prompts", str(self.prompts), "--rewards", str(self.rewards), "--out", str(self.tmp / "o.jsonl"),
                      "--stats", str(self.tmp / "st.json"), "--include-reasoning", "0")  # fmt: skip
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("reasoning_content", (self.tmp / "o.jsonl").read_text())
        samples.write_text(json.dumps({"key": "k1", "sample": 0, "text": "wrong", "reasoning": "", "finish_reason": "stop", "completion_tokens": 3, "error": None}) + "\n")
        r = self.pipe("filter", "--samples", str(samples), "--prompts", str(self.prompts), "--rewards", str(self.rewards), "--out", str(self.tmp / "o2.jsonl"), "--stats", str(self.tmp / "st2.json"))  # fmt: skip
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nothing to train on", r.stderr)

    def test_errors_and_unknown_keys_are_not_trained_on(self):
        samples = self.tmp / "s.jsonl"
        rows = [
            {"key": "k1", "sample": 0, "text": "word1", "reasoning": "", "finish_reason": "stop", "completion_tokens": 1, "error": "timeout"},
            {"key": "zzz", "sample": 0, "text": "word1", "reasoning": "", "finish_reason": "stop", "completion_tokens": 1, "error": None},
            {"key": "k2", "sample": 0, "text": "  ", "reasoning": "", "finish_reason": "stop", "completion_tokens": 1, "error": None},
            {"key": "k3", "sample": 0, "text": "word3", "reasoning": "", "finish_reason": "stop", "completion_tokens": 1, "error": None},
        ]
        samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        r = self.pipe("filter", "--samples", str(samples), "--prompts", str(self.prompts), "--rewards", str(self.rewards), "--out", str(self.tmp / "o.jsonl"), "--stats", str(self.tmp / "st.json"))  # fmt: skip
        self.assertEqual(r.returncode, 0, r.stderr)
        stats = json.loads((self.tmp / "st.json").read_text())
        self.assertEqual((stats["n_errors"], stats["n_empty"], stats["n_kept"]), (1, 1, 1))

    def test_generate_fails_loudly_when_every_request_fails(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        r = self.pipe("generate", "--prompts", str(self.prompts), "--out", str(self.tmp / "s.jsonl"), "--base-url", f"http://127.0.0.1:{port}", "--n-prompts", "1", "--timeout", "2")
        self.assertNotEqual(r.returncode, 0)

    def test_reference_run_sh_wires_the_steps_in_order(self):
        fake = FakeVllm(lambda p, n: [(f"answer word{p.split()[-1]}", "stop", "")] * n)
        self.addCleanup(fake.close)
        log = self.tmp / "calls.log"
        self.stub("glimmer-vllm-up", f'echo "vllm-up $@" >> "{log}"\n')
        self.stub("glimmer-axolotl-train", f'echo "train $@" >> "{log}"; test -s "{self.out}/sft.jsonl" && echo sft-present >> "{log}"\nmkdir -p "{self.out}/adapter"\n')
        self.env.update(LAB_WORK=str(self.work), RFT_TRAIN=str(self.prompts), REWARDS_PY=str(self.rewards), RFT_BASE_URL=f"http://127.0.0.1:{fake.port}",
                        RFT_N_PROMPTS="3", RFT_SAMPLES="2", RFT_PORT="8321")  # fmt: skip
        r = run_script(["bash", str(RFT / "run.sh")], env=self.env, cwd=self.tmp, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        calls = log.read_text().splitlines()
        self.assertEqual(calls[0], "vllm-up --port 8321")
        self.assertEqual(calls[1], "vllm-up --stop", "generation and training never share the GPU")
        self.assertTrue(calls[2].startswith("train ") and calls[2].endswith(f"--data {self.out}/sft.jsonl"))
        self.assertEqual(calls[3], "sft-present")
        timings = json.loads((self.out / "timings.json").read_text())
        self.assertIn("generation_s", timings)
        self.assertIn("train_s", timings)
        self.assertEqual(json.loads((self.out / "rft_stats.json").read_text())["n_samples"], 6)
        self.assertTrue(set(timings) <= {"download_s", "load_s", "generation_s", "train_s", "export_s", "eval_s"})


if __name__ == "__main__":
    unittest.main()
