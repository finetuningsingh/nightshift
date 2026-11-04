import hashlib
import io
import os
import socket
import sys
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.request import urlopen

from nightshift.lab import gpu_remote as gr
from nightshift.lab.contracts import GpuError
from tests.support.lab_fake_gpu import FakeSsh, make_tar_bytes

TARGET = gr.SshTarget("127.0.0.1", 22, "ubuntu", "/nonexistent/key")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name).resolve()
        self.fs = FakeSsh(self.tmp / "fake")
        self.fs.__enter__()
        self.addCleanup(self.fs.__exit__)
        self.addCleanup(self._td.cleanup)
        self.remote = self.tmp / "remote"
        self.remote.mkdir()
        self.local = self.tmp / "local"
        self.local.mkdir()


class FetchHostileInput(Base):
    """Everything a rental returns is untrusted: each refusal below is a planted attack, and nothing may be left behind."""

    def fetch(self, names=("a.bin",), **kw):
        kw.setdefault("max_file_bytes", 1000)
        return gr.fetch_files(TARGET, str(self.remote), list(names), self.local, **kw)

    def serve(self, tar_bytes: bytes):
        self.fs.add_rule("tar -C", stdout=tar_bytes)

    def assert_clean(self):
        self.assertEqual(sorted(p.name for p in self.local.iterdir()), [])

    def test_regular_file_is_fetched_and_hashed(self):
        (self.remote / "a.bin").write_bytes(b"hello adapter")
        got = self.fetch()
        self.assertEqual((self.local / "a.bin").read_bytes(), b"hello adapter")
        self.assertEqual(got["a.bin"].sha256, sha(b"hello adapter"))

    def test_real_remote_symlink_is_refused(self):
        secret = self.tmp / "secret.txt"
        secret.write_text("control host secret")
        (self.remote / "a.bin").symlink_to(secret)
        with self.assertRaisesRegex(GpuError, "symbolic link"):
            self.fetch()
        self.assert_clean()

    def test_real_remote_hard_link_is_refused(self):
        (self.remote / "a.bin").write_bytes(b"x")
        os.link(self.remote / "a.bin", self.remote / "b.bin")
        with self.assertRaisesRegex(GpuError, "hard link"):
            self.fetch(["a.bin", "b.bin"])
        self.assert_clean()

    def test_symlink_member_in_a_hostile_stream(self):
        self.serve(make_tar_bytes([("a.bin", "symlink", "/etc/passwd")]))
        with self.assertRaisesRegex(GpuError, "symbolic link"):
            self.fetch()
        self.assert_clean()

    def test_device_member(self):
        self.serve(make_tar_bytes([("a.bin", "chardev", None)]))
        with self.assertRaisesRegex(GpuError, "device"):
            self.fetch()

    def test_fifo_and_directory_members(self):
        self.serve(make_tar_bytes([("a.bin", "fifo", None)]))
        with self.assertRaises(GpuError):
            self.fetch()
        self.serve(make_tar_bytes([("a.bin", "dir", None)]))
        with self.assertRaisesRegex(GpuError, "directory"):
            self.fetch()
        self.assert_clean()

    def test_path_escape_members(self):
        for name in ("../escape.bin", "/etc/cron.d/x", "sub/a.bin", "./a.bin"):
            self.serve(make_tar_bytes([(name, "file", b"pwn")]))
            with self.assertRaisesRegex(GpuError, "unexpected member"):
                self.fetch()
        self.assertFalse((self.tmp / "escape.bin").exists())
        self.assert_clean()

    def test_unrequested_extra_member(self):
        self.serve(make_tar_bytes([("a.bin", "file", b"ok"), ("evil.sh", "file", b"#!/bin/sh")]))
        with self.assertRaisesRegex(GpuError, "unexpected member"):
            self.fetch()
        self.assert_clean()

    def test_duplicate_member(self):
        self.serve(make_tar_bytes([("a.bin", "file", b"1"), ("a.bin", "file", b"2")]))
        with self.assertRaisesRegex(GpuError, "duplicate"):
            self.fetch()
        self.assert_clean()

    def test_missing_member(self):
        self.serve(make_tar_bytes([("a.bin", "file", b"1")]))
        with self.assertRaisesRegex(GpuError, "did not return"):
            self.fetch(["a.bin", "b.bin"])
        self.assert_clean()

    def test_declared_oversize_is_refused_before_reading(self):
        self.serve(make_tar_bytes([("a.bin", "bigheader", (10_000_000, b"tiny"))]))
        with self.assertRaisesRegex(GpuError, "is 10000000 bytes, above the 1000-byte limit"):
            self.fetch()
        self.assert_clean()

    def test_only_plain_regular_file_types_are_accepted(self):
        self.serve(make_tar_bytes([("a.bin", "cont", b"x")]))
        with self.assertRaisesRegex(GpuError, "tar type"):
            self.fetch()
        self.assert_clean()

    def test_real_oversize_file(self):
        (self.remote / "a.bin").write_bytes(b"x" * 1001)
        with self.assertRaisesRegex(GpuError, "limit"):
            self.fetch()
        self.assert_clean()

    def test_total_size_cap(self):
        (self.remote / "a.bin").write_bytes(b"x" * 600)
        (self.remote / "b.bin").write_bytes(b"y" * 600)
        with self.assertRaisesRegex(GpuError, "total size"):
            self.fetch(["a.bin", "b.bin"], max_total_bytes=1000)
        self.assert_clean()

    def test_endless_stream_is_capped(self):
        self.fs.add_rule("tar -C", script=str(self._endless_script()))
        with self.assertRaises(GpuError):
            self.fetch(max_file_bytes=100)
        self.assert_clean()

    def _endless_script(self) -> Path:
        s = self.tmp / "endless.sh"
        s.write_text("#!/bin/bash\nhead -c 3000000 /dev/zero\n")
        s.chmod(0o755)
        return s

    def test_sha_mismatch_deletes_the_files(self):
        (self.remote / "a.bin").write_bytes(b"data")
        with self.assertRaisesRegex(GpuError, "sha256"):
            self.fetch(expected_sha256={"a.bin": "0" * 64})
        self.assert_clean()

    def test_sha_match_passes(self):
        (self.remote / "a.bin").write_bytes(b"data")
        self.fetch(expected_sha256={"a.bin": sha(b"data")})
        self.assertTrue((self.local / "a.bin").exists())

    def test_names_must_be_plain(self):
        for bad in ("../x", "a/b", "", ".", "..", "a b", "-rf"):
            with self.assertRaises(GpuError):
                self.fetch([bad])

    def test_remote_dir_must_be_absolute_and_normal(self):
        for bad in ("relative", "/a/../b", "/a//b/", "/x\ny"):
            with self.assertRaises(GpuError):
                gr.fetch_files(TARGET, bad, ["a.bin"], self.local, max_file_bytes=10)

    def test_destination_symlink_is_refused(self):
        real = self.tmp / "real"
        real.mkdir()
        link = self.tmp / "linkdir"
        link.symlink_to(real)
        (self.remote / "a.bin").write_bytes(b"x")
        with self.assertRaises(GpuError):
            gr.fetch_files(TARGET, str(self.remote), ["a.bin"], link, max_file_bytes=10)

    def test_a_planted_local_symlink_at_the_destination_is_not_followed(self):
        canary = self.tmp / "canary.txt"
        canary.write_text("DO NOT TOUCH")
        (self.local / ".a.bin.fetch").symlink_to(canary)  # the temp name a fetch would use
        (self.remote / "a.bin").write_bytes(b"new")
        with self.assertRaises(Exception):
            self.fetch()
        self.assertEqual(canary.read_text(), "DO NOT TOUCH")

    def test_endless_remote_stderr_is_never_buffered_on_the_control_host_disk(self):
        from unittest import mock

        self.fs.add_rule("tar -C", stdout=make_tar_bytes([("a.bin", "file", b"ok")]), rc=3, stderr="x" * 3_000_000 + "FINAL-ERROR")
        with mock.patch.object(gr.tempfile, "TemporaryFile", side_effect=AssertionError("stderr must not be spooled to a file")):
            with self.assertRaisesRegex(GpuError, "exited 3.*FINAL-ERROR"):  # the tail is kept, the rest dropped
                self.fetch()
        self.assert_clean()

    def test_remote_tar_failure_status(self):
        with self.assertRaises(GpuError):
            self.fetch(["missing.bin"])

    def test_fetch_single_file_with_rename(self):
        (self.remote / "a.bin").write_bytes(b"one")
        got = gr.fetch(TARGET, str(self.remote / "a.bin"), self.local / "renamed.bin", sha256=sha(b"one"), max_bytes=100)
        self.assertEqual((self.local / "renamed.bin").read_bytes(), b"one")
        self.assertEqual(got.size_bytes, 3)
        self.assertEqual([p.name for p in self.local.iterdir()], ["renamed.bin"])

    def test_remote_path_with_shell_metacharacters_is_quoted(self):
        weird = self.remote / "dir with $(touch pwned) ; x"
        weird.mkdir()
        (weird / "a.bin").write_bytes(b"q")
        gr.fetch_files(TARGET, str(weird), ["a.bin"], self.local, max_file_bytes=10)
        self.assertFalse(Path("pwned").exists())
        self.assertEqual((self.local / "a.bin").read_bytes(), b"q")


class Upload(Base):
    def test_upload_verifies_and_moves_into_place(self):
        src = self.tmp / "payload.bin"
        src.write_bytes(os.urandom(5000))
        dest = self.remote / "deep" / "payload.bin"
        digest = gr.upload(TARGET, src, str(dest))
        self.assertEqual(digest, sha(src.read_bytes()))
        self.assertEqual(dest.read_bytes(), src.read_bytes())
        self.assertFalse((self.remote / "deep" / "payload.bin.partial").exists())

    def test_upload_refuses_local_symlink_hardlink_and_special_files(self):
        real = self.tmp / "real.bin"
        real.write_bytes(b"x")
        link = self.tmp / "link.bin"
        link.symlink_to(real)
        hard = self.tmp / "hard.bin"
        os.link(real, hard)
        for bad in (link, hard, self.tmp):
            with self.assertRaises(GpuError):
                gr.upload(TARGET, bad, str(self.remote / "x"))
        self.assertEqual(list(self.remote.iterdir()), [])

    def test_upload_detects_corruption_on_the_rental(self):
        src = self.tmp / "p.bin"
        src.write_bytes(b"clean bytes")
        # a rental that corrupts what it stores: the hash check runs on the rental and the file is not moved into place
        self.fs.add_rule("cat >", script=str(self._corrupting_script()))
        with self.assertRaisesRegex(GpuError, "verification"):
            gr.upload(TARGET, src, str(self.remote / "p.bin"))
        self.assertFalse((self.remote / "p.bin").exists())

    def _corrupting_script(self) -> Path:
        s = self.tmp / "corrupt.py"
        s.write_text(
            f"#!{sys.executable}\n"
            "import os, re, shlex, subprocess, sys\n"
            "cmd = os.environ['FAKE_SSH_CMD']\n"
            "data = sys.stdin.buffer.read()\n"
            "p = subprocess.run(['bash', '-c', cmd], input=data, capture_output=True)\n"
            "path = shlex.split(re.search(r'cat > (.*)$', cmd).group(1))[0]\n"
            "open(path, 'ab').write(b'corrupt')\n"
            "sys.exit(p.returncode)\n"
        )
        s.chmod(0o755)
        return s

    def test_remote_destination_must_be_absolute(self):
        src = self.tmp / "p.bin"
        src.write_bytes(b"x")
        with self.assertRaises(GpuError):
            gr.upload(TARGET, src, "relative/p.bin")
        with self.assertRaises(GpuError):
            gr.upload(TARGET, src, "/a/../etc/p.bin")

    def test_upload_tree_copies_and_verifies(self):
        tree = self.tmp / "tree"
        (tree / "bin").mkdir(parents=True)
        (tree / "bin" / "run.sh").write_text("#!/bin/bash\necho hi\n")
        (tree / "bin" / "run.sh").chmod(0o755)
        (tree / "conf.yaml").write_text("a: 1\n")
        (tree / "__pycache__").mkdir()
        (tree / "__pycache__" / "x.pyc").write_bytes(b"junk")
        manifest = gr.upload_tree(TARGET, tree, str(self.remote / "hostkit"))
        self.assertEqual(sorted(manifest), ["bin/run.sh", "conf.yaml"])
        self.assertEqual((self.remote / "hostkit" / "conf.yaml").read_text(), "a: 1\n")
        self.assertTrue(os.access(self.remote / "hostkit" / "bin" / "run.sh", os.X_OK))
        self.assertFalse((self.remote / "hostkit" / "__pycache__").exists())

    def test_upload_tree_refuses_links(self):
        tree = self.tmp / "tree"
        tree.mkdir()
        (tree / "ok.txt").write_text("ok")
        (tree / "evil").symlink_to("/etc/passwd")
        with self.assertRaises(GpuError):
            gr.upload_tree(TARGET, tree, str(self.remote / "hk"))
        sub = self.tmp / "tree2"
        (sub / "d").mkdir(parents=True)
        (sub / "dirlink").symlink_to(sub / "d")
        with self.assertRaises(GpuError):
            gr.upload_tree(TARGET, sub, str(self.remote / "hk2"))
        self.assertFalse((self.remote / "hk").exists())


class RunAndDetach(Base):
    def test_run_returns_output_and_checks_status(self):
        self.assertEqual(gr.run(TARGET, "echo hi").text.strip(), "hi")
        with self.assertRaises(GpuError):
            gr.run(TARGET, "exit 4")
        self.assertEqual(gr.run(TARGET, "exit 4", check=False).returncode, 4)

    def test_run_timeout(self):
        with self.assertRaisesRegex(GpuError, "timed out"):
            gr.run(TARGET, "sleep 5", timeout=0.3)

    def test_run_detached_returns_a_pid_and_the_job_outlives_the_call(self):
        marker = self.remote / "done.txt"
        log = self.remote / "logs" / "job.log"
        pid = gr.run_detached(TARGET, ["bash", "-c", f"sleep 0.4; echo finished > {marker}; echo out"], str(log), env={"LAB_X": "a b"})
        self.assertTrue(pid.isdigit())
        for _ in range(50):
            if marker.exists():
                break
            threading.Event().wait(0.1)
        self.assertTrue(marker.exists())
        self.assertIn("out", log.read_text())

    def test_run_detached_quotes_arguments(self):
        out = self.remote / "o.txt"
        gr.run_detached(TARGET, ["bash", "-c", 'printf "%s" "$0" > "$1"', "a;b $(id) 'q'", str(out)], str(self.remote / "l.log"))
        for _ in range(50):
            if out.exists() and out.read_text():
                break
            threading.Event().wait(0.1)
        self.assertEqual(out.read_text(), "a;b $(id) 'q'")

    def test_run_detached_rejects_bad_env_names(self):
        with self.assertRaises(GpuError):
            gr.run_detached(TARGET, ["true"], str(self.remote / "l.log"), env={"bad name": "x"})

    def test_read_remote_tail(self):
        (self.remote / "l.log").write_text("x" * 100 + "END")
        self.assertTrue(gr.read_remote_tail(TARGET, str(self.remote / "l.log"), max_bytes=10).endswith("END"))
        self.assertEqual(gr.read_remote_tail(TARGET, str(self.remote / "none.log")), "")

    def test_remote_output_is_bounded_locally(self):
        with self.assertRaisesRegex(GpuError, "more than 1000 bytes"):
            gr.run(TARGET, "head -c 5000000 /dev/zero", max_output=1000, check=False)
        res = gr.run(TARGET, "head -c 5000000 /dev/zero", max_output=1000, truncate=True, check=False)
        self.assertEqual(len(res.stdout), 1000)
        self.assertEqual(gr.run(TARGET, "printf abc", max_output=1000).stdout, b"abc")

    def test_read_remote_tail_does_not_trust_the_remote_tail_command(self):
        bin_dir = self.tmp / "evilbin"
        bin_dir.mkdir()
        evil = bin_dir / "tail"
        evil.write_text("#!/bin/sh\nhead -c 1048576 /dev/zero | tr '\\0' x\n")
        evil.chmod(0o755)
        (self.remote / "l.log").write_text("x" * 100 + "END")
        with mock.patch.dict(os.environ, {"PATH": f"{bin_dir}:{os.environ['PATH']}"}):
            text = gr.read_remote_tail(TARGET, str(self.remote / "l.log"), max_bytes=10)
        self.assertLessEqual(len(text), 10)


class Echo(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Forward(Base):
    def test_forward_port_reaches_the_remote_loopback_port(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
        self.addCleanup(srv.server_close)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        lp = gr.free_loopback_port()
        with gr.forward_port(TARGET, lp, srv.server_address[1]) as fwd:
            fwd.wait_ready(10)
            self.assertEqual(urlopen(f"http://127.0.0.1:{lp}/", timeout=5).read(), b'{"ok": true}')
        self.assertFalse(fwd.alive())
        self.assertTrue(any("forward" in c for c in self.fs.calls()))
        self.assertTrue(any("ExitOnForwardFailure=yes" in str(c["opts"]) for c in self.fs.calls() if "forward" in c))

    def test_forward_failure_is_reported(self):
        self.fs.forward_fail = True
        self.fs._write()
        fwd = gr.forward_port(TARGET, gr.free_loopback_port(), 9)
        with self.assertRaisesRegex(GpuError, "exited"):
            fwd.wait_ready(5)
        fwd.close()

    def test_bad_ports(self):
        for bad in (0, 70000, True):
            with self.assertRaises(GpuError):
                gr.forward_port(TARGET, bad, 22)


class Target(unittest.TestCase):
    def test_target_validation(self):
        for host in ("a b", "host;rm", "", "-oProxyCommand=x y"):
            with self.assertRaises(GpuError):
                gr.SshTarget(host, 22, "ubuntu", "/k")
        with self.assertRaises(GpuError):
            gr.SshTarget("h", 22, "root; x", "/k")
        args = TARGET.args("-N")
        self.assertEqual(args[0], "ssh")
        self.assertIn("BatchMode=yes", args)
        self.assertEqual(args[-1], "ubuntu@127.0.0.1")

    def test_ssh_config_is_private(self):
        with tempfile.TemporaryDirectory() as td:
            gr.write_ssh_config(Path(td) / "cfg", "alias", TARGET)
            self.assertEqual(oct((Path(td) / "cfg").stat().st_mode & 0o777), "0o600")


if __name__ == "__main__":
    unittest.main()
