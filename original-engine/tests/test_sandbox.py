import dataclasses
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from nightshift import sandbox, secrets
from nightshift.types import ProtectedPathViolation, TaskSpec

HAS_FIFO = hasattr(os, "mkfifo")
LINUX = sys.platform.startswith("linux")
OK_RUN = textwrap.dedent("""
    import os
    print("hello", flush=True)
    os.makedirs("outputs", exist_ok=True)
    open("outputs/p.txt", "w").write(os.environ["NIGHTSHIFT_SEED"])
""")


def make_task(root: Path, run_py: str = OK_RUN, **overrides) -> TaskSpec:
    """A minimal task on disk: template/{run.py,model.py,readme.txt}, trusted/{grade.py,secret.txt}."""
    (root / "template").mkdir(parents=True)
    (root / "trusted").mkdir()
    (root / "template" / "run.py").write_text(run_py)
    (root / "template" / "model.py").write_text("def f():\n    return 1\n")
    (root / "template" / "readme.txt").write_text("read only\n")
    (root / "trusted" / "grade.py").write_text("print('grader')\n")
    (root / "trusted" / "secret.txt").write_text("labels live here\n")
    fields = dict(
        name="t", description="", brief="b", task_dir=root, template_dir=root / "template", trusted_dir=root / "trusted",
        objective="minimize", primary_metric="m", train_metric=None, editable=["model.py"], scratch=["outputs/**"],
        run_cmd=["python3", "run.py"], run_timeout_sec=10.0,
        grade_cmd=["python3", "{task_dir}/trusted/grade.py", "--workspace", "{workspace}", "--seed", "{seed}",
                   "--split", "{split}"],
        grade_timeout_sec=10.0, seeds=[0, 1, 2], replication_seeds=[100, 101, 102], noise_margin=0.01, requires=[],
        stall_sec=5.0)
    fields.update(overrides)
    return TaskSpec(**fields)


def alive(pid: int) -> bool:
    """Is ``pid`` a live (non-zombie) process?"""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            raw = fh.read()
        return raw[raw.rindex(b")") + 2:].split()[0] != b"Z"
    except FileNotFoundError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class SandboxCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.task = make_task(self.dir / "task")
        self.run_dir = self.dir / "run"

    def workspace(self, exp_id: str = "e001", task: TaskSpec | None = None) -> sandbox.Workspace:
        return sandbox.create_workspace(task or self.task, self.run_dir, exp_id)


class ScanAndCopyTests(SandboxCase):
    def src(self) -> Path:
        d = self.dir / "src"
        (d / "sub").mkdir(parents=True)
        (d / "a.py").write_text("a")
        (d / "sub" / "b.txt").write_text("b")
        (d / "run.sh").write_text("#!/bin/sh\n")
        (d / "run.sh").chmod(0o755)
        return d

    def violation_of(self, src: Path, kind: str = "workspace") -> ProtectedPathViolation:
        dst = self.dir / "dst"
        with self.assertRaises(ProtectedPathViolation) as cm:
            sandbox.copy_regular(src, dst, self.task, kind=kind)
        self.assertFalse(dst.exists(), "nothing may be written when the source has a special file")
        self.assertEqual(cm.exception.kind, kind)
        return cm.exception

    def test_copies_nested_files_and_keeps_exec_bit(self):
        src, dst = self.src(), self.dir / "dst"
        sandbox.copy_regular(src, dst, self.task)
        self.assertEqual((dst / "sub" / "b.txt").read_text(), "b")
        self.assertTrue(os.access(dst / "run.sh", os.X_OK))
        self.assertFalse(os.access(dst / "a.py", os.X_OK))

    def test_never_uses_copytree(self):
        with mock.patch("shutil.copytree", side_effect=AssertionError("copytree used")):
            sandbox.copy_regular(self.src(), self.dir / "dst", self.task)
            self.workspace()

    def test_symlink_to_file_is_special(self):
        src = self.src()
        os.symlink("/etc/hostname", src / "link")
        self.assertEqual(self.violation_of(src).paths, ["special:link"])

    def test_symlink_to_directory_named_outputs_is_special(self):
        src = self.src()
        os.symlink(self.dir / "task" / "trusted", src / "outputs")
        self.assertEqual(self.violation_of(src, kind="runtime").paths, ["special:outputs"])

    def test_dangling_symlink_is_special(self):
        src = self.src()
        os.symlink("/nonexistent/target", src / "dangling")
        self.assertEqual(self.violation_of(src).paths, ["special:dangling"])

    @unittest.skipUnless(HAS_FIFO, "no mkfifo")
    def test_fifo_is_special_and_does_not_block(self):
        src = self.src()
        os.mkfifo(src / "pipe")
        t0 = time.monotonic()
        self.assertEqual(self.violation_of(src).paths, ["special:pipe"])
        self.assertLess(time.monotonic() - t0, 2)

    def test_hardlink_is_special(self):
        src = self.src()
        os.link(src / "a.py", src / "a_copy.py")
        paths = self.violation_of(src).paths
        self.assertEqual(sorted(paths), ["special:a.py", "special:a_copy.py"])

    def test_hardlink_to_a_file_outside_the_tree_is_special(self):
        src = self.src()
        outside = self.dir / "outside.txt"
        outside.write_text("x")
        os.link(outside, src / "inside.txt")
        self.assertIn("special:inside.txt", self.violation_of(src).paths)

    def test_newline_in_filename_is_special_and_escaped(self):
        src = self.src()
        (src / "evil\nname.py").write_text("x")
        paths = self.violation_of(src).paths
        self.assertEqual(paths, ["special:evil\\nname.py"])
        self.assertNotIn("\n", paths[0])

    def test_trailing_newline_is_not_accepted_by_the_name_check(self):
        src = self.src()
        (src / "ok.py\n").write_text("x")
        self.assertEqual(len(self.violation_of(src).paths), 1)

    def test_unsafe_names_are_special(self):
        for name in ("has space.py", "semi;colon.py", "dollar$.py", "unié.py", "a" * 129, "dots..py"):
            with self.subTest(name=name[:20]):
                src = self.dir / f"s{abs(hash(name))}"
                src.mkdir()
                (src / name).write_text("x")
                self.assertEqual(len(self.violation_of(src).paths), 1)
                shutil.rmtree(self.dir / "dst", ignore_errors=True)

    def test_unsafe_directory_name_is_flagged_once(self):
        src = self.src()
        (src / "bad dir").mkdir()
        for i in range(5):
            (src / "bad dir" / f"f{i}").write_text("x")
        self.assertEqual(self.violation_of(src).paths, ["special:bad dir"])

    def test_file_count_cap(self):
        src = self.dir / "many"
        src.mkdir()
        for i in range(8):
            (src / f"f{i}").write_text("x")
        with mock.patch.object(sandbox, "MAX_FILES", 5):
            self.assertEqual(self.violation_of(src).paths, ["special:too_many_files"])

    def test_a_directory_bomb_is_capped_too(self):
        src = self.dir / "bomb"
        src.mkdir()
        for i in range(12):
            (src / f"d{i}").mkdir()
        with mock.patch.object(sandbox, "MAX_FILES", 3), mock.patch.object(sandbox, "MAX_DIRS", 4):
            self.assertEqual(self.violation_of(src).paths, ["special:too_many_files"])
        with mock.patch.object(sandbox, "MAX_FILES", 3), mock.patch.object(sandbox, "MAX_DIRS", 50):
            sandbox.copy_regular(src, self.dir / "ok", self.task)          # under the allowance: fine

    def test_default_caps_are_the_contract_numbers(self):
        self.assertEqual((sandbox.MAX_FILES, sandbox.MAX_TREE_BYTES), (2000, 64 * 1024 * 1024))

    def test_byte_cap(self):
        src = self.dir / "big"
        src.mkdir()
        (src / "a").write_bytes(b"x" * 600)
        (src / "b").write_bytes(b"x" * 600)
        with mock.patch.object(sandbox, "MAX_TREE_BYTES", 1000):
            self.assertEqual(self.violation_of(src).paths, ["special:too_large"])

    def test_source_that_is_a_symlink_is_refused(self):
        real = self.src()
        link = self.dir / "linked"
        os.symlink(real, link)
        self.assertEqual(self.violation_of(link).paths, ["special:."])

    def test_destination_must_be_a_fresh_directory(self):
        src, dst = self.src(), self.dir / "dst"
        dst.mkdir()
        (dst / "old").write_text("x")
        with self.assertRaises(FileExistsError):
            sandbox.copy_regular(src, dst, self.task)
        self.assertEqual((dst / "old").read_text(), "x")          # a refused copy must not delete what was there
        shutil.rmtree(dst)
        os.symlink(self.dir, dst)
        with self.assertRaises(FileExistsError):
            sandbox.copy_regular(src, dst, self.task)

    def test_empty_existing_destination_is_accepted(self):
        dst = self.dir / "dst"
        dst.mkdir()
        sandbox.copy_regular(self.src(), dst, self.task)
        self.assertTrue((dst / "a.py").exists())

    def test_file_that_becomes_a_symlink_between_scan_and_copy_is_not_followed(self):
        src = self.src()
        secret = self.dir / "secret.txt"
        secret.write_text("TOP SECRET")
        real_scan = sandbox._scan

        def swapping_scan(root, **kw):
            scan = real_scan(root, **kw)
            os.unlink(src / "a.py")
            os.symlink(secret, src / "a.py")
            return scan

        with mock.patch.object(sandbox, "_scan", swapping_scan):
            with self.assertRaises(ProtectedPathViolation) as cm:
                sandbox.copy_regular(src, self.dir / "dst", self.task)
        self.assertEqual(cm.exception.paths, ["special:a.py"])
        self.assertFalse((self.dir / "dst").exists())


class WorkspaceTests(SandboxCase):
    def test_create_from_template(self):
        ws = self.workspace()
        self.assertEqual(ws.path, self.run_dir / "exp" / "e001" / "workspace")
        self.assertEqual(sorted(ws.base_hashes), ["model.py", "readme.txt", "run.py"])
        self.assertEqual(ws.tree_hash, sandbox.tree_hash(ws.path, self.task))
        self.assertRegex(ws.tree_hash, r"^[0-9a-f]{12}$")
        meta = json.loads((self.run_dir / "exp" / "e001" / "base.json").read_text())
        self.assertEqual(meta, {"base_ref": "template", "base_hashes": ws.base_hashes, "tree_hash": ws.tree_hash})
        self.assertEqual(sandbox.base_hash(ws), sandbox.base_hash(self.workspace("e002")))

    def test_base_hash_matches_the_contract_formula(self):
        from nightshift.types import canonical_json, short_hash
        ws = self.workspace()
        self.assertEqual(sandbox.base_hash(ws), short_hash(canonical_json(sorted(ws.base_hashes.items()))))

    def test_scratch_files_in_the_template_do_not_count(self):
        (self.task.template_dir / "outputs").mkdir()
        (self.task.template_dir / "outputs" / "x.csv").write_text("x")
        ws = self.workspace()
        self.assertNotIn("outputs/x.csv", ws.base_hashes)
        self.assertTrue((ws.path / "outputs" / "x.csv").exists())

    def test_create_from_frozen_base_recomputes_the_tree_hash(self):
        first = self.workspace("e001")
        (first.path / "model.py").write_text("def f():\n    return 2\n")
        frozen = self.run_dir / "exp" / "e001" / "frozen"
        want = sandbox.freeze(first.path, frozen, self.task)
        second = sandbox.create_workspace(self.task, self.run_dir, "e002", base=frozen)
        self.assertEqual(second.tree_hash, want)
        self.assertEqual((second.path / "model.py").read_text(), "def f():\n    return 2\n")
        self.assertTrue(os.access(second.path / "model.py", os.W_OK) or os.geteuid() == 0)
        meta = json.loads((self.run_dir / "exp" / "e002" / "base.json").read_text())
        self.assertEqual(meta["base_ref"], "e001")

    def test_refuses_existing_workspace_and_unsafe_ids(self):
        self.workspace()
        with self.assertRaises(FileExistsError):
            self.workspace()
        for bad in ("../x", "a/b", "", ".", "e 1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.workspace(bad)

    def test_template_with_a_symlink_is_refused(self):
        os.symlink("/etc/hostname", self.task.template_dir / "link")
        with self.assertRaises(ProtectedPathViolation):
            self.workspace()
        self.assertFalse((self.run_dir / "exp" / "e001" / "workspace").exists())


class TreeHashTests(SandboxCase):
    def test_ignores_scratch_but_not_other_files(self):
        ws = self.workspace()
        h = sandbox.tree_hash(ws.path, self.task)
        (ws.path / "outputs").mkdir()
        (ws.path / "outputs" / "x").write_text("scratch")
        self.assertEqual(sandbox.tree_hash(ws.path, self.task), h)
        (ws.path / "extra.py").write_text("x")
        self.assertNotEqual(sandbox.tree_hash(ws.path, self.task), h)

    def test_sensitive_to_content_and_names_not_to_mtime(self):
        ws = self.workspace()
        h = sandbox.tree_hash(ws.path, self.task)
        os.utime(ws.path / "model.py", (1, 1))
        self.assertEqual(sandbox.tree_hash(ws.path, self.task), h)
        (ws.path / "model.py").rename(ws.path / "model2.py")
        self.assertNotEqual(sandbox.tree_hash(ws.path, self.task), h)

    def test_special_files_do_not_enter_the_hash(self):
        ws = self.workspace()
        h = sandbox.tree_hash(ws.path, self.task)
        os.symlink("/etc/hostname", ws.path / "link")
        self.assertEqual(sandbox.tree_hash(ws.path, self.task), h)


class DiffTests(SandboxCase):
    def test_no_changes(self):
        ws = self.workspace()
        d = sandbox.diff_workspace(self.task, ws)
        self.assertEqual((d.changed, d.added, d.removed, d.unified, d.violations), ([], [], [], "", []))
        self.assertEqual(d.summary, "no changes")
        self.assertEqual(d.tree_hash, ws.tree_hash)
        self.assertRegex(d.diff_hash, r"^[0-9a-f]{12}$")

    def test_edit_to_an_editable_file(self):
        ws = self.workspace()
        (ws.path / "model.py").write_text("def f():\n    return 42\n")
        d = sandbox.diff_workspace(self.task, ws)
        self.assertEqual(d.changed, ["model.py"])
        self.assertEqual(d.violations, [])
        self.assertIn("--- a/model.py", d.unified)
        self.assertIn("-    return 1", d.unified)
        self.assertIn("+    return 42", d.unified)
        self.assertIn("M model.py", d.summary)
        self.assertNotEqual(d.tree_hash, ws.tree_hash)

    def test_scratch_changes_are_invisible(self):
        ws = self.workspace()
        (ws.path / "outputs").mkdir()
        (ws.path / "outputs" / "p.csv").write_text("trusted labels.csv ledger.jsonl")     # tripwire text, but scratch
        d = sandbox.diff_workspace(self.task, ws)
        self.assertEqual((d.changed, d.added, d.violations), ([], [], []))

    def test_changing_a_readonly_file_adds_or_removes_is_protected(self):
        ws = self.workspace()
        (ws.path / "readme.txt").write_text("edited\n")
        (ws.path / "new.py").write_text("x = 1\n")
        (ws.path / "run.py").unlink()
        d = sandbox.diff_workspace(self.task, ws)
        self.assertEqual((d.changed, d.added, d.removed), (["readme.txt"], ["new.py"], ["run.py"]))
        self.assertEqual(sorted(d.violations), ["protected:new.py", "protected:readme.txt", "protected:run.py"])
        self.assertIn("/dev/null", d.unified)

    def test_special_files_are_violations(self):
        ws = self.workspace()
        os.symlink("/etc/hostname", ws.path / "link")
        os.link(ws.path / "model.py", ws.path / "model_hard.py")
        (ws.path / "bad name.py").write_text("x")
        if HAS_FIFO:
            os.mkfifo(ws.path / "fifo")
        v = sandbox.diff_workspace(self.task, ws).violations
        for name in ("link", "model.py", "model_hard.py", "bad name.py") + (("fifo",) if HAS_FIFO else ()):
            self.assertIn(f"special:{name}", v)

    def test_violation_strings_never_contain_raw_control_characters(self):
        ws = self.workspace()
        (ws.path / "x\ny\r\x1b[2J.py").write_text("x")
        for v in sandbox.diff_workspace(self.task, ws).violations:
            self.assertRegex(v, r"^[\x20-\x7e]+$")

    def test_binary_file_is_reported_not_decoded(self):
        ws = self.workspace()
        (ws.path / "model.py").write_bytes(b"\x00\x01\x02\xff binary")
        d = sandbox.diff_workspace(self.task, ws)
        self.assertIn("binary file changed", d.unified)
        self.assertNotIn("\x00", d.unified)
        (ws.path / "model.py").write_bytes(b"caf\xe9\n")                  # not UTF-8
        self.assertIn("binary file changed", sandbox.diff_workspace(self.task, ws).unified)

    def test_unified_is_capped_at_200_kb(self):
        task = dataclasses.replace(self.task, editable=["*.py"])
        ws = self.workspace(task=task)
        for i in range(4):
            (ws.path / f"gen{i}.py").write_text("".join(f"x{j} = {j}\n" for j in range(15000)))
        d = sandbox.diff_workspace(task, ws)
        self.assertLessEqual(len(d.unified.encode()), 200 * 1024 + 100)
        self.assertIn("diff truncated", d.unified)
        self.assertEqual(len(d.added), 4)

    def test_cap_does_not_change_the_diff_hash(self):
        task = dataclasses.replace(self.task, editable=["*.py"])
        a, b = self.workspace("e001", task), self.workspace("e002", task)
        for ws, tail in ((a, "A"), (b, "B")):
            (ws.path / "big.py").write_text("".join(f"x{j} = {j}\n" for j in range(30000)) + f"# {tail}\n")
        self.assertNotEqual(sandbox.diff_workspace(task, a).diff_hash, sandbox.diff_workspace(task, b).diff_hash)

    def test_large_file_is_summarised(self):
        ws = self.workspace()
        (ws.path / "model.py").write_text("x = 1\n" * 100000)
        d = sandbox.diff_workspace(self.task, ws)
        self.assertIn("large file changed: model.py", d.unified)

    def test_diff_hash_depends_on_content_only(self):
        a, b, c = self.workspace("e001"), self.workspace("e002"), self.workspace("e003")
        for ws in (a, b):
            (ws.path / "model.py").write_text("def f():\n    return 7\n")
        (c.path / "model.py").write_text("def f():\n    return 8\n")
        ha, hb, hc = (sandbox.diff_workspace(self.task, w).diff_hash for w in (a, b, c))
        self.assertEqual(ha, hb)
        self.assertNotEqual(ha, hc)
        self.assertNotEqual(ha, sandbox.diff_workspace(self.task, self.workspace("e004")).diff_hash)

    def test_tripwire_tokens_in_added_text(self):
        for token in ("trusted", "labels.", "ledger.jsonl", "replication_seeds", "tasks/"):
            with self.subTest(token=token):
                ws = self.workspace(f"e_{abs(hash(token))}")
                (ws.path / "model.py").write_text(f"def f():\n    open('../../{token}x')\n    return 1\n")
                d = sandbox.diff_workspace(self.task, ws)
                self.assertEqual(d.violations, ["probe:model.py"])

    def test_tripwire_ignores_unchanged_and_removed_lines_and_binary_free_edits(self):
        (self.task.template_dir / "model.py").write_text("# this is trusted by the maintainers\ndef f():\n    return 1\n")
        ws = self.workspace()
        (ws.path / "model.py").write_text("# this is trusted by the maintainers\ndef f():\n    return 2\n")
        self.assertEqual(sandbox.diff_workspace(self.task, ws).violations, [])
        (ws.path / "model.py").write_text("def f():\n    return 2\n")              # removing the word is fine
        self.assertEqual(sandbox.diff_workspace(self.task, ws).violations, [])
        (ws.path / "model.py").write_text("# this is trusted by the maintainers!\ndef f():\n    return 2\n")   # changed line
        self.assertEqual(sandbox.diff_workspace(self.task, ws).violations, ["probe:model.py"])

    def test_tripwire_sees_binary_and_large_files(self):
        ws = self.workspace()
        (ws.path / "model.py").write_bytes(b"\x00\x00ledger.jsonl\x00")
        self.assertEqual(sandbox.diff_workspace(self.task, ws).violations, ["probe:model.py"])
        (ws.path / "model.py").write_text("x = 1\n" * 100000 + "# replication_seeds\n")
        self.assertEqual(sandbox.diff_workspace(self.task, ws).violations, ["probe:model.py"])

    def test_tripwire_on_a_non_editable_file_reports_both(self):
        ws = self.workspace()
        (ws.path / "readme.txt").write_text("see tasks/secret\n")
        self.assertEqual(sorted(sandbox.diff_workspace(self.task, ws).violations),
                         ["probe:readme.txt", "protected:readme.txt"])

    def test_base_is_found_through_base_json_when_the_workspace_object_is_rebuilt(self):
        ws = self.workspace()
        (ws.path / "model.py").write_text("def f():\n    return 5\n")
        rebuilt = sandbox.Workspace(ws.exp_id, ws.path, ws.base_hashes, ws.tree_hash)
        self.assertIn("-    return 1", sandbox.diff_workspace(self.task, rebuilt).unified)

    def test_base_from_a_frozen_parent_is_found_through_base_json(self):
        first = self.workspace("e001")
        (first.path / "model.py").write_text("def f():\n    return 2\n")
        sandbox.freeze(first.path, self.run_dir / "exp" / "e001" / "frozen", self.task)
        second = sandbox.create_workspace(self.task, self.run_dir, "e002", base=self.run_dir / "exp" / "e001" / "frozen")
        (second.path / "model.py").write_text("def f():\n    return 3\n")
        rebuilt = sandbox.Workspace(second.exp_id, second.path, second.base_hashes, second.tree_hash)
        d = sandbox.diff_workspace(self.task, rebuilt)
        self.assertIn("-    return 2", d.unified)
        self.assertIn("+    return 3", d.unified)

    def test_unavailable_base_degrades_to_a_full_file_with_a_note(self):
        ws = self.workspace()
        (ws.path / "model.py").write_text("def f():\n    return 9\n")
        (self.run_dir / "exp" / "e001" / "base.json").unlink()
        rebuilt = sandbox.Workspace(ws.exp_id, ws.path, ws.base_hashes, ws.tree_hash)
        d = sandbox.diff_workspace(self.task, rebuilt)
        self.assertEqual(d.changed, ["model.py"])
        self.assertIn("base unavailable", d.unified)

    def test_template_edited_after_workspace_creation_is_detected_as_unavailable_base(self):
        ws = self.workspace()
        (ws.path / "model.py").write_text("def f():\n    return 9\n")
        (self.task.template_dir / "model.py").write_text("# the template moved on\n")
        d = sandbox.diff_workspace(self.task, ws)
        self.assertIn("base unavailable", d.unified)
        self.assertNotIn("moved on", d.unified)

    def test_whitespace_only_change_is_still_a_change(self):
        ws = self.workspace()
        (ws.path / "model.py").write_text("def f():\n    return 1")             # trailing newline removed
        d = sandbox.diff_workspace(self.task, ws)
        self.assertEqual(d.changed, ["model.py"])
        self.assertIn("whitespace or line-ending change only", d.unified)


class FreezeTests(SandboxCase):
    def test_freeze_copies_non_scratch_files_read_only(self):
        ws = self.workspace()
        (ws.path / "outputs").mkdir()
        (ws.path / "outputs" / "x.csv").write_text("scratch")
        (ws.path / "model.py").write_text("def f():\n    return 3\n")
        dest = self.dir / "frozen"
        h = sandbox.freeze(ws.path, dest, self.task)
        self.assertEqual(h, sandbox.tree_hash(ws.path, self.task))
        self.assertEqual(h, sandbox.tree_hash(dest, self.task))
        self.assertFalse((dest / "outputs").exists())
        files = [p for p in dest.rglob("*") if p.is_file()]
        self.assertTrue(files)
        for p in files:
            self.assertEqual(stat.S_IMODE(p.lstat().st_mode) & 0o222, 0, p)
        shutil.rmtree(dest)          # a frozen tree must stay removable for non-root users
        self.assertFalse(dest.exists())

    def test_freeze_is_idempotent_and_refuses_a_different_tree(self):
        ws = self.workspace()
        dest = self.dir / "frozen"
        h = sandbox.freeze(ws.path, dest, self.task)
        self.assertEqual(sandbox.freeze(ws.path, dest, self.task), h)
        (ws.path / "model.py").write_text("def f():\n    return 3\n")
        with self.assertRaises(FileExistsError):
            sandbox.freeze(ws.path, dest, self.task)

    def test_freeze_refuses_special_files(self):
        ws = self.workspace()
        os.symlink("/etc/hostname", ws.path / "link")
        with self.assertRaises(ProtectedPathViolation) as cm:
            sandbox.freeze(ws.path, self.dir / "frozen", self.task)
        self.assertEqual(cm.exception.paths, ["special:link"])
        self.assertFalse((self.dir / "frozen").exists())


class StageOutputsTests(SandboxCase):
    def run_ws(self) -> Path:
        d = self.dir / "runws"
        (d / "outputs").mkdir(parents=True)
        return d

    def stage(self, run_ws: Path, name: str = "staged", task: TaskSpec | None = None) -> tuple[str, Path]:
        dest = self.dir / name / "outputs"
        return sandbox.stage_outputs(run_ws, dest, task or self.task), dest

    def test_copies_regular_files_and_hash_is_stable_and_content_sensitive(self):
        ws = self.run_ws()
        (ws / "outputs" / "p.csv").write_text("1,2\n")
        (ws / "outputs" / "sub").mkdir()
        (ws / "outputs" / "sub" / "q.json").write_text("{}")
        h1, dest = self.stage(ws)
        self.assertRegex(h1, r"^[0-9a-f]{12}$")
        self.assertEqual((dest / "p.csv").read_text(), "1,2\n")
        self.assertEqual((dest / "sub" / "q.json").read_text(), "{}")
        h2, _ = self.stage(ws, "again")
        self.assertEqual(h1, h2)
        (ws / "outputs" / "p.csv").write_text("1,3\n")
        self.assertNotEqual(h1, self.stage(ws, "third")[0])

    def test_missing_outputs_directory_stages_nothing(self):
        ws = self.dir / "empty"
        ws.mkdir()
        h, dest = self.stage(ws)
        self.assertEqual(list(dest.iterdir()), [])
        self.assertEqual(h, self.stage(self.run_ws(), "other")[0])

    def test_outputs_symlink_to_a_directory_is_never_followed(self):
        ws = self.dir / "runws"
        ws.mkdir()
        os.symlink(self.task.trusted_dir, ws / "outputs")
        h, dest = self.stage(ws)
        self.assertEqual(list(dest.iterdir()), [])
        _, skipped = sandbox.stage_outputs_report(ws, self.dir / "s2" / "outputs", self.task)
        self.assertEqual(skipped, ["special:outputs"])
        fresh = self.dir / "fresh"
        (fresh / "outputs").mkdir(parents=True)
        self.assertEqual(h, self.stage(fresh, "empty")[0])

    def test_symlinks_to_files_and_directories_inside_outputs_are_skipped(self):
        ws = self.run_ws()
        (ws / "outputs" / "ok.txt").write_text("ok")
        os.symlink(self.task.trusted_dir / "secret.txt", ws / "outputs" / "stolen.txt")
        os.symlink(self.task.trusted_dir, ws / "outputs" / "dirlink")
        _, dest = self.stage(ws)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["ok.txt"])

    @unittest.skipUnless(HAS_FIFO, "no mkfifo")
    def test_fifo_is_skipped_without_blocking(self):
        ws = self.run_ws()
        os.mkfifo(ws / "outputs" / "pipe")
        (ws / "outputs" / "ok.txt").write_text("ok")
        t0 = time.monotonic()
        _, dest = self.stage(ws)
        self.assertLess(time.monotonic() - t0, 2)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["ok.txt"])

    def test_hardlinks_are_skipped(self):
        ws = self.run_ws()
        (ws / "outputs" / "a.txt").write_text("a")
        os.link(ws / "outputs" / "a.txt", ws / "outputs" / "b.txt")
        outside = self.task.trusted_dir / "secret.txt"
        os.link(outside, ws / "outputs" / "leak.txt")
        _, dest = self.stage(ws)
        self.assertEqual(list(dest.iterdir()), [])

    def test_bad_names_are_skipped(self):
        ws = self.run_ws()
        for name in ("has space", "new\nline", "semi;colon", "x" * 65, "unié"):
            (ws / "outputs" / name).write_text("x")
        (ws / "outputs" / "good-name_1.csv").write_text("x")
        _, dest = self.stage(ws)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["good-name_1.csv"])

    def test_depth_is_limited_to_two(self):
        ws = self.run_ws()
        (ws / "outputs" / "a" / "b").mkdir(parents=True)
        (ws / "outputs" / "top.txt").write_text("1")
        (ws / "outputs" / "a" / "mid.txt").write_text("2")
        (ws / "outputs" / "a" / "b" / "deep.txt").write_text("3")
        _, dest = self.stage(ws)
        self.assertEqual(sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()), ["a/mid.txt", "top.txt"])

    def test_at_most_32_files(self):
        ws = self.run_ws()
        for i in range(40):
            (ws / "outputs" / f"f{i:02d}.txt").write_text(str(i))
        _, dest = self.stage(ws)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), [f"f{i:02d}.txt" for i in range(32)])

    def test_oversized_files_are_skipped(self):
        task = dataclasses.replace(self.task, max_output_bytes=100)
        ws = self.run_ws()
        (ws / "outputs" / "small.txt").write_bytes(b"x" * 100)
        (ws / "outputs" / "big.txt").write_bytes(b"x" * 101)
        _, dest = self.stage(ws, task=task)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["small.txt"])

    def test_an_unbounded_outputs_directory_is_not_listed_whole(self):
        ws = self.run_ws()
        for i in range(30):
            (ws / "outputs" / f"f{i}.txt").write_text("x")
        with mock.patch.object(sandbox, "MAX_FILES", 5), mock.patch.object(sandbox, "MAX_DIRS", 5):
            _, skipped = sandbox.stage_outputs_report(ws, self.dir / "s" / "outputs", self.task)
        self.assertEqual(skipped, ["toomany:outputs"])
        self.assertEqual(list((self.dir / "s" / "outputs").iterdir()), [])

    def test_refuses_a_non_empty_destination(self):
        ws = self.run_ws()
        dest = self.dir / "staged" / "outputs"
        dest.mkdir(parents=True)
        (dest / "old").write_text("x")
        with self.assertRaises(FileExistsError):
            sandbox.stage_outputs(ws, dest, self.task)


class RunUntrustedTests(SandboxCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(secrets.clear)

    def run_script(self, source: str, *, seed: int = 1, task: TaskSpec | None = None, **kw) -> sandbox.RunResult:
        task = task or self.task
        ws = sandbox.create_workspace(task, self.run_dir, f"e{len(list((self.run_dir / 'exp').glob('*'))) if (self.run_dir / 'exp').exists() else 0:03d}")
        (ws.path / "run.py").write_text(textwrap.dedent(source))
        self.last_ws = ws.path
        kw.setdefault("isolation", "none")
        return sandbox.run_untrusted(task, ws.path, seed, **kw)

    def test_successful_run(self):
        res = self.run_script(OK_RUN, seed=7)
        self.assertEqual((res.classification, res.exit_code, res.timed_out, res.stalled), ("ok", 0, False, False))
        self.assertEqual(res.stdout, "hello\n")
        self.assertEqual((self.last_ws / "outputs" / "p.txt").read_text(), "7")
        self.assertGreater(res.duration_sec, 0)
        self.assertEqual(res.output_bytes, len("hello\n"))
        self.assertEqual(res.isolation, "none")
        self.assertIsInstance(res.pid, int)

    def test_python3_means_the_interpreter_running_nightshift(self):
        res = self.run_script("import sys; print(sys.executable)")
        self.assertEqual(res.stdout.strip(), sys.executable)

    def test_environment_is_exactly_the_contract_and_secrets_never_reach_the_child(self):
        parent_env = {"MODEL_API_KEY": "sk-model-secret-123456", "META_API_KEY": "meta-secret-654321",
                      "HOME": "/parent/home/dir", "AWS_SECRET_ACCESS_KEY": "aws-secret-000", "LD_PRELOAD": "",
                      "PYTHONPATH": "/parent/pythonpath", "PYTHONSTARTUP": "/parent/startup.py"}
        with mock.patch.dict(os.environ, parent_env):
            res = self.run_script("import json, os; print(json.dumps(dict(os.environ)))", seed=3, attempt=2)
        env = json.loads(res.stdout)
        required = {"NIGHTSHIFT_SEED", "NIGHTSHIFT_WORKSPACE", "NIGHTSHIFT_ATTEMPT", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
                    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HOME", "PATH"}
        self.assertTrue(required <= set(env), required - set(env))
        self.assertLessEqual(set(env) - required, {"LC_CTYPE"})            # CPython's own C-locale coercion
        self.assertEqual((env["NIGHTSHIFT_SEED"], env["NIGHTSHIFT_ATTEMPT"]), ("3", "2"))
        self.assertEqual((env["PYTHONHASHSEED"], env["PYTHONDONTWRITEBYTECODE"]), ("0", "1"))
        self.assertEqual({env[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}, {"1"})
        self.assertEqual(env["HOME"], str(self.last_ws))
        self.assertEqual(env["NIGHTSHIFT_WORKSPACE"], str(self.last_ws))
        self.assertNotIn("NIGHTSHIFT_SHIM_CONFIG", env)
        blob = res.stdout
        for secret in ("sk-model-secret", "meta-secret", "/parent/home", "aws-secret", "/parent/pythonpath", "startup.py"):
            self.assertNotIn(secret, blob)

    def test_pythonpath_is_the_parent_of_each_required_module(self):
        modules = self.dir / "mods"
        (modules / "fakepkg").mkdir(parents=True)
        (modules / "fakepkg" / "__init__.py").write_text("VALUE = 41\n")
        task = dataclasses.replace(self.task, requires=["fakepkg"])
        with mock.patch.object(sys, "path", [str(modules), *sys.path]):
            res = self.run_script("import fakepkg, os; print(fakepkg.VALUE, os.environ['PYTHONPATH'])", task=task)
        self.assertEqual(res.stdout.split(), ["41", str(modules)])

    def test_rlimits_are_really_applied(self):
        res = self.run_script("import resource, json; print(json.dumps([resource.getrlimit(resource.RLIMIT_CORE), "
                              "resource.getrlimit(getattr(resource, 'RLIMIT_NOFILE'))]))")
        core, nofile = json.loads(res.stdout)
        self.assertEqual(core, [0, 0])
        if LINUX:
            self.assertEqual(nofile, [256, 256])

    def test_a_failing_setrlimit_still_yields_ok(self):
        plan = [("RLIMIT_NOFILE", 100, 10),            # soft > hard: setrlimit raises ValueError
                ("RLIMIT_DOES_NOT_EXIST", 1, 1),         # getattr raises
                ("RLIMIT_CORE", 0),                      # malformed entry
                ("RLIMIT_AS", -5, -5),                   # invalid values
                ("RLIMIT_CPU", 100, 200)]                # a good one after the bad ones still applies
        with mock.patch.object(sandbox, "_limit_plan", return_value=plan):
            res = self.run_script("import resource; print('fine', resource.getrlimit(resource.RLIMIT_CPU))")
        self.assertEqual(res.classification, "ok")
        self.assertIn("fine (100, 200)", res.stdout)

    def test_spawns_through_the_shim_without_preexec_fn(self):
        real = subprocess.Popen
        seen = {}

        def spy(args, **kw):
            seen["args"], seen["kw"] = args, kw
            return real(args, **kw)

        with mock.patch.object(sandbox.subprocess, "Popen", spy):
            self.run_script(OK_RUN)
        self.assertEqual(seen["args"][:4], [sys.executable, "-I", "-c", sandbox.SHIM])
        self.assertEqual(seen["args"][4:], [sys.executable, "run.py"])
        self.assertNotIn("preexec_fn", seen["kw"])
        self.assertTrue(seen["kw"]["start_new_session"])
        self.assertTrue(seen["kw"]["close_fds"])

    def test_nonzero_exit_is_a_crash_with_stderr_captured(self):
        res = self.run_script("import sys; sys.stderr.write('boom\\n'); sys.exit(3)")
        self.assertEqual((res.classification, res.exit_code), ("crash", 3))
        self.assertEqual(res.stderr, "boom\n")

    def test_unrunnable_command_is_a_crash_not_an_exception(self):
        task = dataclasses.replace(self.task, run_cmd=["no-such-binary-xyz"])
        res = self.run_script("pass", task=task)
        self.assertEqual((res.classification, res.exit_code), ("crash", 127))
        self.assertIn("cannot execute", res.stderr)

    def test_spawn_failure_is_infra(self):
        with mock.patch.object(sandbox.subprocess, "Popen", side_effect=OSError("no fork")):
            res = self.run_script(OK_RUN)
        self.assertEqual(res.classification, "infra")
        self.assertIn("spawn_failed", res.note)
        self.assertIsNone(res.exit_code)

    def test_missing_required_module_is_infra(self):
        task = dataclasses.replace(self.task, requires=["no_such_module_xyz"])
        res = self.run_script("import no_such_module_xyz", task=task)
        self.assertEqual(res.classification, "infra")

    def test_wall_timeout_kills_the_group_quickly(self):
        t0 = time.monotonic()
        res = self.run_script("import time, os; open('pid','w').write(str(os.getpid())); time.sleep(30)", timeout_sec=1.0)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual((res.classification, res.timed_out, res.stalled), ("timeout", True, False))
        self.assertIn("timeout after", res.note)
        self.assertFalse(alive(int((self.last_ws / "pid").read_text())))

    def test_stall_watchdog_calls_hook_once_then_kills(self):
        task = dataclasses.replace(self.task, stall_sec=0.5)
        idle = []
        t0 = time.monotonic()
        res = self.run_script("import time; print('start', flush=True); time.sleep(30)", task=task, on_stalled=idle.append)
        self.assertLess(time.monotonic() - t0, 6)
        self.assertEqual((res.classification, res.stalled, res.timed_out), ("timeout", True, False))
        self.assertEqual(len(idle), 1)
        self.assertGreaterEqual(idle[0], 0.5)
        self.assertIn("stalled", res.note)

    def test_a_run_that_keeps_printing_is_not_stalled(self):
        task = dataclasses.replace(self.task, stall_sec=0.7)
        res = self.run_script("import time\nfor i in range(15):\n    print(i, flush=True)\n    time.sleep(0.08)\n", task=task)
        self.assertEqual((res.classification, res.stalled), ("ok", False))

    def test_heartbeats_report_output_bytes_and_cpu(self):
        beats = []
        res = self.run_script("import time\nfor i in range(10):\n    print('x' * 9, flush=True)\n    time.sleep(0.1)\n",
                              on_heartbeat=lambda b, c: beats.append((b, c)), heartbeat_sec=0.1)
        self.assertEqual(res.classification, "ok")
        self.assertGreaterEqual(len(beats), 3)
        self.assertEqual([b for b, _ in beats], sorted(b for b, _ in beats))
        self.assertGreater(beats[-1][0], 0)
        if LINUX:
            self.assertTrue(all(isinstance(c, float) for _, c in beats))

    def test_on_start_receives_the_pid(self):
        pids = []
        res = self.run_script(OK_RUN, on_start=pids.append)
        self.assertEqual(pids, [res.pid])

    def test_output_flood_is_capped_and_killed(self):
        t0 = time.monotonic()
        res = self.run_script("import sys\nchunk = 'x' * 65536\nfor _ in range(2000):\n    sys.stdout.write(chunk)\n"
                              "    sys.stdout.flush()\n", log_dir=self.dir / "logs", log_prefix="run_0")
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual(res.classification, "crash")
        self.assertIn("output_limit", res.note)
        log = self.dir / "logs" / "run_0.stdout"
        self.assertEqual(log.stat().st_size, 8 * 1024 * 1024)
        self.assertLessEqual(len(res.stdout.encode()), 64 * 1024)
        self.assertGreater(res.output_bytes, 8 * 1024 * 1024)

    def test_stderr_flood_is_capped_too(self):
        res = self.run_script("import sys\nchunk = 'e' * 65536\nfor _ in range(2000):\n    sys.stderr.write(chunk)\n"
                              "    sys.stderr.flush()\n", log_dir=self.dir / "logs", log_prefix="r")
        self.assertEqual((res.classification, res.note), ("crash", "output_limit"))
        self.assertEqual((self.dir / "logs" / "r.stderr").stat().st_size, 8 * 1024 * 1024)

    def test_result_streams_are_the_last_64_kib_and_logs_hold_everything(self):
        res = self.run_script("for i in range(40000):\n    print('line', i)\n", log_dir=self.dir / "logs", log_prefix="r")
        self.assertEqual(res.classification, "ok")
        self.assertLessEqual(len(res.stdout.encode()), 64 * 1024)
        self.assertTrue(res.stdout.endswith("line 39999\n"))
        full = (self.dir / "logs" / "r.stdout").read_text()
        self.assertTrue(full.startswith("line 0\n"))
        self.assertTrue(full.endswith("line 39999\n"))

    def test_secrets_are_scrubbed_from_tails_and_log_files(self):
        secrets.register("tok-super-secret-value")
        res = self.run_script("print('key=tok-super-secret-value')", log_dir=self.dir / "logs", log_prefix="s")
        self.assertNotIn("tok-super-secret-value", res.stdout)
        self.assertIn("[REDACTED]", res.stdout)
        self.assertNotIn("tok-super-secret-value", (self.dir / "logs" / "s.stdout").read_text())

    def test_detached_daemon_is_killed_and_cannot_rewrite_outputs_after_exit(self):
        script = """
            import os, subprocess, sys
            os.makedirs("outputs", exist_ok=True)
            open("outputs/p.txt", "w").write("honest")
            daemon = subprocess.Popen([sys.executable, "-c", (
                "import os, time\\ntime.sleep(0.7)\\nopen('outputs/p.txt', 'w').write('tampered')\\n"
                "open('outputs/late.txt', 'w').write('late')\\n")], start_new_session=True)
            open("daemon.pid", "w").write(str(daemon.pid))
        """
        res = self.run_script(script)
        pid = int((self.last_ws / "daemon.pid").read_text())
        self.assertEqual(res.classification, "ok")
        self.assertFalse(alive(pid), "the daemon must be dead when run_untrusted returns")
        time.sleep(1.0)
        self.assertEqual((self.last_ws / "outputs" / "p.txt").read_text(), "honest")
        self.assertFalse((self.last_ws / "outputs" / "late.txt").exists())

    @unittest.skipUnless(LINUX, "straggler sweep uses /proc")
    def test_daemon_that_closed_its_stdio_is_found_by_its_working_directory(self):
        script = """
            import os, subprocess, sys
            daemon = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
                                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            open("daemon.pid", "w").write(str(daemon.pid))
        """
        res = self.run_script(script)
        self.assertFalse(alive(int((self.last_ws / "daemon.pid").read_text())))
        self.assertIn("killed_stragglers", res.note)

    @unittest.skipUnless(LINUX, "straggler sweep uses /proc")
    def test_daemon_that_changed_directory_and_closed_stdio_is_found_by_its_parent_chain(self):
        script = """
            import os, subprocess, sys, time
            daemon = subprocess.Popen([sys.executable, "-c", "import os, time; os.chdir('/'); time.sleep(30)"],
                                      start_new_session=True, cwd="/", stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            open("daemon.pid", "w").write(str(daemon.pid))
            time.sleep(1.0)
        """
        self.run_script(script, heartbeat_sec=0.05)
        self.assertFalse(alive(int((self.last_ws / "daemon.pid").read_text())))

    def test_group_is_killed_when_a_hook_raises(self):
        seen = []

        def boom(pid):
            seen.append(pid)
            raise RuntimeError("hook failure")

        with self.assertRaisesRegex(RuntimeError, "hook failure"):
            self.run_script("import time; time.sleep(30)", on_start=boom)
        self.assertFalse(alive(seen[0]))

    def test_group_is_killed_when_the_heartbeat_hook_raises(self):
        pids = []

        def boom(output_bytes, cpu):
            raise RuntimeError("heartbeat failure")

        with self.assertRaisesRegex(RuntimeError, "heartbeat failure"):
            self.run_script("import time; print('x', flush=True); time.sleep(30)", on_start=pids.append,
                            on_heartbeat=boom, heartbeat_sec=0.05)
        self.assertFalse(alive(pids[0]))

    def test_logs_go_to_the_requested_files(self):
        res = self.run_script("import sys; print('out'); print('err', file=sys.stderr)", log_dir=self.dir / "lg", log_prefix="run_5")
        self.assertEqual((self.dir / "lg" / "run_5.stdout").read_text(), "out\n")
        self.assertEqual((self.dir / "lg" / "run_5.stderr").read_text(), "err\n")
        self.assertEqual(res.output_bytes, 8)

    def test_no_log_dir_leaves_nothing_behind(self):
        before = set(Path(tempfile.gettempdir()).glob("nightshift-logs-*"))
        self.run_script(OK_RUN)
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("nightshift-logs-*")), before)

    def test_bad_arguments(self):
        ws = self.workspace()
        with self.assertRaises(ValueError):
            sandbox.run_untrusted(self.task, ws.path, 0, timeout_sec=0)
        with self.assertRaises(ValueError):
            sandbox.run_untrusted(self.task, ws.path, 0, isolation="chroot")


class LimitPlanTests(SandboxCase):
    def plan(self, **patches) -> dict[str, tuple[int, int]]:
        return {name: (soft, hard) for name, soft, hard in sandbox._limit_plan(self.task, 10.0)}

    @unittest.skipUnless(LINUX, "linux limits")
    def test_linux_limits(self):
        with mock.patch.object(os, "geteuid", return_value=1000), mock.patch.object(sandbox, "_task_count_for_uid", return_value=40):
            plan = self.plan()
        self.assertEqual(plan["RLIMIT_CORE"], (0, 0))
        self.assertEqual(plan["RLIMIT_NOFILE"], (256, 256))
        self.assertEqual(plan["RLIMIT_AS"], (4 << 30, 4 << 30))
        self.assertEqual(plan["RLIMIT_NPROC"], (104, 104))             # current count + 64
        self.assertEqual(plan["RLIMIT_FSIZE"], (self.task.max_output_bytes,) * 2)
        soft, hard = plan["RLIMIT_CPU"]
        self.assertGreater(soft, 10)
        self.assertGreater(hard, soft, "SIGXCPU must come before SIGKILL so the death is classified as a crash")

    @unittest.skipUnless(LINUX, "linux limits")
    def test_nproc_is_skipped_for_root_and_when_the_count_is_unknown(self):
        with mock.patch.object(os, "geteuid", return_value=0):
            self.assertNotIn("RLIMIT_NPROC", self.plan())
        with mock.patch.object(os, "geteuid", return_value=1000), mock.patch.object(sandbox, "_task_count_for_uid", return_value=None):
            self.assertNotIn("RLIMIT_NPROC", self.plan())

    def test_only_cpu_fsize_and_core_apply_off_linux(self):
        with mock.patch.object(sandbox, "_LINUX", False):
            self.assertEqual(sorted(self.plan()), ["RLIMIT_CORE", "RLIMIT_CPU", "RLIMIT_FSIZE"])

    def test_the_trusted_plan_has_no_memory_or_file_size_cap(self):
        names = {n for n, _, _ in sandbox._limit_plan(self.task, 10.0, trusted=True)}
        self.assertTrue({"RLIMIT_CORE", "RLIMIT_CPU"} <= names)
        self.assertFalse({"RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NPROC"} & names)

    @unittest.skipUnless(LINUX, "needs /proc")
    def test_task_count_counts_threads_of_this_user(self):
        import threading
        before = sandbox._task_count_for_uid()
        stop = threading.Event()
        # the per-uid count is process-global and other threads (this suite's own workers) come and go while we
        # measure, so start many and allow slack instead of asserting an exact delta
        threads = [threading.Thread(target=stop.wait) for _ in range(40)]
        for t in threads:
            t.start()
        try:
            self.assertGreaterEqual(sandbox._task_count_for_uid(), before + 25)
        finally:
            stop.set()
            for t in threads:
                t.join()


class ClassifyFailureTests(SandboxCase):
    def setUp(self):
        super().setUp()
        self.t = dataclasses.replace(self.task, requires=["numpy"])

    def test_table(self):
        c = lambda code, err="", to=False: sandbox.classify_failure(self.t, code, err, to)  # noqa: E731
        self.assertEqual(c(0), "ok")
        self.assertEqual(c(0, "MemoryError mentioned but exit 0"), "ok")
        self.assertEqual(c(1, to=True), "timeout")
        self.assertEqual(c(-9, to=True), "timeout")
        self.assertEqual(c(1), "crash")
        self.assertEqual(c(2, "Traceback ... ValueError: nope"), "crash")
        self.assertEqual(c(1, "ModuleNotFoundError: No module named 'numpy'"), "infra")
        self.assertEqual(c(1, "ModuleNotFoundError: No module named 'numpy.core'"), "infra")
        self.assertEqual(c(1, "ImportError: cannot import name 'x' from 'numpy'"), "infra")
        self.assertEqual(c(1, "ModuleNotFoundError: No module named 'pandas'"), "crash")
        self.assertEqual(c(1, "ModuleNotFoundError: No module named 'mynumpy_helpers'"), "crash")
        self.assertEqual(c(1, "MemoryError"), "infra")
        self.assertEqual(c(1, "OSError: [Errno 28] No space left on device"), "infra")
        self.assertEqual(c(1, "Cannot allocate memory"), "infra")
        self.assertEqual(c(1, "socket.gaierror: Temporary failure in name resolution"), "infra")
        self.assertEqual(c(1, "BlockingIOError: Resource temporarily unavailable"), "infra")
        self.assertEqual(c(-9), "infra")
        self.assertEqual(c(137), "infra")
        self.assertEqual(c(-11), "crash")
        self.assertEqual(c(None), "infra")

    def test_a_run_killed_from_outside_is_infra(self):
        ws = self.workspace()
        (ws.path / "run.py").write_text("import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
        res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="none")
        self.assertEqual((res.exit_code, res.classification), (-9, "infra"))


class IsolationTests(SandboxCase):
    def test_reports_a_known_level_and_is_stable(self):
        level = sandbox.available_isolation()
        self.assertIn(level, ("userns", "unshare-net", "none"))
        self.assertEqual(sandbox.available_isolation(), level)
        self.assertEqual(sandbox.available_isolation(refresh=True), level)

    def test_none_when_unshare_is_missing_or_off_linux(self):
        self.addCleanup(lambda: sandbox.available_isolation(refresh=True))
        with mock.patch.object(sandbox.shutil, "which", return_value=None):
            self.assertEqual(sandbox.available_isolation(refresh=True), "none")
        sandbox.available_isolation(refresh=True)
        with mock.patch.object(sandbox, "_LINUX", False):
            self.assertEqual(sandbox.available_isolation(refresh=True), "none")

    def test_a_failing_probe_is_not_reported(self):
        self.addCleanup(lambda: sandbox.available_isolation(refresh=True))
        with mock.patch.object(sandbox, "_probe_run", return_value=(1, b"")):
            self.assertEqual(sandbox.available_isolation(refresh=True), "none")
        with mock.patch.object(sandbox, "_probe_run", return_value=(0, b'{"secret": ["s.txt"]}\n')):
            self.assertEqual(sandbox.available_isolation(refresh=True), "none")
        with mock.patch.object(sandbox, "_probe_run", return_value=(0, b"not json")):
            self.assertEqual(sandbox.available_isolation(refresh=True), "none")

    def test_requesting_more_than_is_available_falls_back(self):
        self.addCleanup(lambda: sandbox.available_isolation(refresh=True))
        with mock.patch.dict(sandbox._iso_info, {"level": "none"}):
            ws = self.workspace()
            res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="userns")
        self.assertEqual((res.classification, res.isolation), ("ok", "none"))

    def test_auto_applies_the_probed_level_and_reports_it(self):
        ws = self.workspace()
        res = sandbox.run_untrusted(self.task, ws.path, 0)
        self.assertEqual(res.classification, "ok")
        self.assertEqual(res.isolation, sandbox.available_isolation())

    @unittest.skipUnless(sys.platform.startswith("linux"), "namespaces are Linux only")
    def test_userns_masks_trusted_ledger_siblings_and_network(self):
        if sandbox.available_isolation() != "userns":
            self.skipTest("user namespaces unavailable here")
        ledger = self.run_dir / "ledger.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ledger.write_text("secret ledger\n")
        sibling = self.run_dir / "exp" / "e009"
        sibling.mkdir(parents=True)
        (sibling / "workspace.txt").write_text("someone else's work")
        ws = self.workspace("e001")
        probe = textwrap.dedent(f"""
            import json, os, socket, subprocess
            def listing(p):
                try:
                    return sorted(os.listdir(p))
                except OSError as e:
                    return "ERR " + type(e).__name__
            out = dict(trusted=listing({str(self.task.trusted_dir)!r}), sibling=listing({str(sibling)!r}),
                       ledger=os.path.getsize({str(ledger)!r}), own=listing({str(ws.path.parent)!r}),
                       pid=os.getpid(), ifaces=sorted(n for _, n in socket.if_nameindex()),
                       umount=subprocess.run(["umount", {str(self.task.trusted_dir)!r}], capture_output=True).returncode)
            print(json.dumps(out))
        """)
        (ws.path / "run.py").write_text(probe)
        res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="userns",
                                    mask_paths=[ledger, sibling, self.run_dir / "no-such-file"])
        self.assertEqual(res.classification, "ok", res.stderr)
        self.assertEqual(res.isolation, "userns")
        seen = json.loads(res.stdout)
        self.assertEqual(seen["trusted"], [])
        self.assertEqual(seen["sibling"], [])
        self.assertEqual(seen["ledger"], 0)
        self.assertIn("workspace", seen["own"])
        self.assertEqual(seen["pid"], 1)
        self.assertEqual(seen["ifaces"], ["lo"])
        self.assertNotEqual(seen["umount"], 0, "the run must not be able to unmask the trusted directory")
        # outside the namespace nothing changed
        self.assertEqual(ledger.read_text(), "secret ledger\n")
        self.assertTrue((self.task.trusted_dir / "grade.py").exists())

    def test_without_isolation_the_trusted_directory_is_readable_by_path(self):
        ws = self.workspace()
        (ws.path / "run.py").write_text(f"import os; print(sorted(os.listdir({str(self.task.trusted_dir)!r})))")
        res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="none")
        self.assertIn("grade.py", res.stdout)      # honest labelling: with "none" protection rests on hash checks only

    @unittest.skipUnless(sys.platform.startswith("linux"), "namespaces are Linux only")
    def test_userns_kills_everything_including_daemons(self):
        if sandbox.available_isolation() != "userns":
            self.skipTest("user namespaces unavailable here")
        ws = self.workspace()
        (ws.path / "run.py").write_text(textwrap.dedent("""
            import subprocess, sys
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="/")
        """))
        before = _count_sleepers()
        sandbox.run_untrusted(self.task, ws.path, 0, isolation="userns")
        time.sleep(0.1)
        self.assertEqual(_count_sleepers(), before)

    def test_mask_failure_is_infra_not_a_crash(self):
        if sandbox.available_isolation() != "userns":
            self.skipTest("user namespaces unavailable here")
        ws = self.workspace()
        with mock.patch.object(sandbox, "_mask_script",
                               return_value="false || { echo NIGHTSHIFT_MASK_FAILED >&2; exit 97; }\n"):
            res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="userns")
        self.assertEqual((res.classification, res.note), ("infra", "isolation_mask_failed"))


class IsolationOverrideTests(SandboxCase):
    """``NIGHTSHIFT_ISOLATION`` may only lower the isolation the probe proved, never raise it."""

    def env(self, value: str | None):
        env = {k: v for k, v in os.environ.items() if k != sandbox.ISOLATION_ENV}
        if value is not None:
            env[sandbox.ISOLATION_ENV] = value
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def proven(self, level: str, **extra):
        info = {"level": level, "unshare": "/usr/bin/unshare", "sh": "/bin/sh", "net_prefix": None, "reason": ""}
        info.update(extra)
        patcher = mock.patch.dict(sandbox._iso_info, info, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_allowed_values(self):
        self.assertEqual(sandbox.ISOLATION_CHOICES, ("auto", "userns", "unshare-net", "none"))
        self.assertEqual(sandbox.ISOLATION_ENV, "NIGHTSHIFT_ISOLATION")

    def test_unset_blank_and_mixed_case_values(self):
        for value, want in ((None, "auto"), ("", "auto"), ("   ", "auto"), ("auto", "auto"), (" None ", "none"),
                            ("USERNS", "userns"), ("Unshare-Net", "unshare-net")):
            self.env(value)
            self.assertEqual(sandbox.isolation_override(), want, value)

    def test_an_invalid_value_is_a_value_error_naming_the_allowed_values_at_first_use(self):
        self.env("chroot")
        self.proven("none")
        ws = self.workspace()
        for call in (sandbox.available_isolation, sandbox.effective_isolation,
                     lambda: sandbox.run_untrusted(self.task, ws.path, 0, isolation="none")):
            with self.assertRaises(ValueError) as caught:
                call()
            message = str(caught.exception)
            self.assertIn("NIGHTSHIFT_ISOLATION", message)
            for allowed in ("auto", "userns", "unshare-net", "none"):
                self.assertIn(allowed, message)
        self.env("userns")
        self.assertEqual(sandbox.isolation_override(), "userns")

    def test_the_explicit_argument_error_names_the_allowed_values_too(self):
        self.env(None)
        with self.assertRaises(ValueError) as caught:
            sandbox.effective_isolation("chroot")
        self.assertIn("userns", str(caught.exception))
        self.assertIn("unshare-net", str(caught.exception))

    def test_none_is_applied_without_running_the_probe(self):
        self.env("none")
        with mock.patch.dict(sandbox._iso_info, clear=True), \
                mock.patch.object(sandbox, "_probe_isolation", side_effect=AssertionError("probed")):
            self.assertEqual(sandbox.available_isolation(), "none")
            ws = self.workspace()
            res = sandbox.run_untrusted(self.task, ws.path, 0)
        self.assertEqual((res.classification, res.isolation), ("ok", "none"))
        self.assertIn("NIGHTSHIFT_ISOLATION=none", res.note)

    def test_an_explicit_none_needs_no_probe_and_adds_no_note(self):
        self.env(None)
        with mock.patch.dict(sandbox._iso_info, clear=True), \
                mock.patch.object(sandbox, "_probe_isolation", side_effect=AssertionError("probed")):
            ws = self.workspace()
            res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="none")
        self.assertEqual((res.isolation, res.note), ("none", ""))

    def test_an_explicit_none_is_not_blamed_on_the_variable_when_it_is_also_none(self):
        # regression: the whole existing suite failed under NIGHTSHIFT_ISOLATION=none because callers that pass
        # isolation="none" themselves got a note naming the variable
        self.env("none")
        with mock.patch.dict(sandbox._iso_info, clear=True), \
                mock.patch.object(sandbox, "_probe_isolation", side_effect=AssertionError("probed")):
            level, _info, notes = sandbox._effective_isolation("none")
            ws = self.workspace()
            res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="none")
        self.assertEqual((level, notes), ("none", []))
        self.assertEqual((res.isolation, res.note), ("none", ""))
        self.assertEqual(sandbox._effective_isolation("userns")[2], ["isolation set to none by NIGHTSHIFT_ISOLATION=none"])
        self.assertEqual(sandbox._effective_isolation("auto")[2], ["isolation set to none by NIGHTSHIFT_ISOLATION=none"])

    def test_the_override_lowers_what_the_probe_proved(self):
        self.proven("userns")
        for value, level in (("auto", "userns"), ("userns", "userns"), ("none", "none")):
            self.env(value)
            self.assertEqual(sandbox.available_isolation(), level, value)
        with mock.patch.object(sandbox, "_probe_net_prefix", return_value=["/usr/bin/unshare", "--net"]):
            self.env("unshare-net")
            level, info, notes = sandbox._effective_isolation("auto")
        self.assertEqual(level, "unshare-net")
        self.assertEqual(info["net_prefix"], ["/usr/bin/unshare", "--net"])
        self.assertEqual(notes, ["isolation lowered from userns to unshare-net by NIGHTSHIFT_ISOLATION=unshare-net"])

    def test_the_override_never_raises_the_isolation(self):
        self.proven("none", reason="user namespaces unavailable (probe failed)")
        for value in ("userns", "unshare-net"):
            self.env(value)
            with mock.patch.object(sandbox, "_probe_net_prefix", return_value=None):
                level, _info, notes = sandbox._effective_isolation("auto")
            self.assertEqual(level, "none", value)
            self.assertEqual(len(notes), 1, notes)
            self.assertIn(f"NIGHTSHIFT_ISOLATION={value} not available", notes[0])
            self.assertIn("user namespaces unavailable", notes[0])
            self.assertIn("applied none", notes[0])

    def test_forcing_a_stronger_mode_than_available_is_recorded_in_the_run_note(self):
        self.proven("none", reason="unshare or sh not found on PATH")
        self.env("userns")
        ws = self.workspace()
        res = sandbox.run_untrusted(self.task, ws.path, 0)
        self.assertEqual((res.classification, res.isolation), ("ok", "none"))
        self.assertIn("NIGHTSHIFT_ISOLATION=userns not available (unshare or sh not found on PATH); applied none", res.note)

    def test_an_explicit_request_above_the_probe_is_recorded_too(self):
        self.env(None)
        self.proven("none", reason="not Linux")
        ws = self.workspace()
        res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="userns")
        self.assertEqual(res.isolation, "none")
        self.assertIn("isolation=userns not available (not Linux); applied none", res.note)

    def test_the_explicit_argument_and_the_override_both_only_lower(self):
        self.proven("userns")
        self.env("none")
        self.assertEqual(sandbox.effective_isolation("userns"), "none")
        self.env("userns")
        self.assertEqual(sandbox.effective_isolation("none"), "none")
        self.assertEqual(sandbox.effective_isolation("userns"), "userns")

    def test_unshare_net_on_a_userns_machine_is_proven_before_it_is_used_and_cached(self):
        self.proven("userns")
        self.env("unshare-net")
        with mock.patch.object(sandbox, "_probe_net_prefix", return_value=["/usr/bin/unshare", "--net"]) as probe:
            self.assertEqual(sandbox.available_isolation(), "unshare-net")
            self.assertEqual(sandbox.available_isolation(), "unshare-net")
        self.assertEqual(probe.call_count, 1)
        with mock.patch.object(sandbox, "_probe_net_prefix", return_value=None) as probe:
            self.proven("userns")
            self.assertEqual(sandbox.available_isolation(), "none")      # not proven -> never assumed
            self.assertEqual(sandbox.available_isolation(), "none")
        self.assertEqual(probe.call_count, 1, "a negative answer is cached as well")

    def test_concurrent_first_uses_of_unshare_net_prove_it_once(self):
        import threading
        self.proven("userns")
        self.env("unshare-net")

        def slow(*_a):
            time.sleep(0.2)
            return ["/usr/bin/unshare", "--net"]

        results = []
        with mock.patch.object(sandbox, "_probe_net_prefix", side_effect=slow) as probe:
            threads = [threading.Thread(target=lambda: results.append(sandbox.effective_isolation())) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(results, ["unshare-net"] * 6)
        self.assertEqual(probe.call_count, 1)

    def test_the_cached_probe_does_not_depend_on_the_override_so_changing_it_takes_effect_at_once(self):
        self.env(None)
        calls = []
        real = sandbox._probe_isolation

        def counting():
            calls.append(1)
            return real()

        with mock.patch.dict(sandbox._iso_info, clear=True), mock.patch.object(sandbox, "_probe_isolation", counting):
            best = sandbox.available_isolation()
            for value in ("none", "auto", "userns", "none", "auto"):
                self.env(value)
                want = "none" if value == "none" else best
                self.assertEqual(sandbox.available_isolation(), want, value)
            self.assertEqual(len(calls), 1)
            sandbox.available_isolation(refresh=True)
            self.assertEqual(len(calls), 2)

    def test_the_default_behaviour_is_unchanged_without_the_variable(self):
        self.env(None)
        self.proven("unshare-net", net_prefix=["/usr/bin/unshare", "--net"])
        level, _info, notes = sandbox._effective_isolation("auto")
        self.assertEqual((level, notes), ("unshare-net", []))

    def test_without_unshare_the_whole_run_still_works_and_says_why_a_forced_mode_was_not_used(self):
        self.env("userns")
        with mock.patch.dict(sandbox._iso_info, clear=True), mock.patch.object(sandbox.shutil, "which", return_value=None):
            self.assertEqual(sandbox.available_isolation(), "none")
            ws = self.workspace()
            res = sandbox.run_untrusted(self.task, ws.path, 0)
        self.assertEqual((res.classification, res.isolation), ("ok", "none"))
        self.assertIn("unshare or sh not found on PATH", res.note)


class TrustedStatTests(SandboxCase):
    """``trusted_stat`` / ``trusted_stat_diff``: a stat-only fingerprint that sees a restored file and ignores reads."""

    def snap(self) -> dict:
        return sandbox.trusted_stat(self.task.trusted_dir)

    def test_covers_every_file_and_directory_and_never_reads_content(self):
        (self.task.trusted_dir / "sub").mkdir()
        (self.task.trusted_dir / "sub" / "x.csv").write_text("a,b\n")
        with mock.patch.object(sandbox, "_open_regular", side_effect=AssertionError("content read")), \
                mock.patch("builtins.open", side_effect=AssertionError("content read")):
            snap = self.snap()
        self.assertEqual(sorted(snap), [".", "grade.py", "secret.txt", "sub", "sub/x.csv"])
        self.assertEqual(snap["."][8], ("grade.py", "secret.txt", "sub"))
        self.assertIsNone(snap["grade.py"][8])
        self.assertEqual(sandbox.trusted_stat_diff(snap, self.snap()), [])

    def test_reading_a_file_is_not_a_change(self):
        before = self.snap()
        for _ in range(3):
            (self.task.trusted_dir / "secret.txt").read_bytes()
            os.listdir(self.task.trusted_dir)
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap()), [])

    def test_write_then_restore_is_visible_although_bytes_and_mtime_are_back(self):
        path = self.task.trusted_dir / "secret.txt"
        original, st = path.read_bytes(), path.stat()
        before = self.snap()
        time.sleep(0.02)                      # a different kernel timestamp tick than the file's creation
        with open(path, "r+b") as fh:
            fh.write(b"X" * 4)
        with open(path, "r+b") as fh:
            fh.write(original[:4])
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        after = path.stat()
        self.assertEqual((path.read_bytes(), after.st_mtime_ns, after.st_size), (original, st.st_mtime_ns, st.st_size))
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap()), ["trusted/secret.txt"])

    def test_added_removed_replaced_renamed_and_chmodded_entries_are_all_named(self):
        t = self.task.trusted_dir
        before = self.snap()
        (t / "new.csv").write_text("n")
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap()), ["trusted", "trusted/new.csv"])
        before = self.snap()
        os.unlink(t / "new.csv")
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap()), ["trusted", "trusted/new.csv"])
        before = self.snap()
        os.replace(t / "secret.txt", t / "moved.txt")
        os.replace(t / "moved.txt", t / "secret.txt")                 # same name and bytes again, new ctime
        self.assertIn("trusted/secret.txt", sandbox.trusted_stat_diff(before, self.snap()))
        before = self.snap()
        os.chmod(t / "grade.py", 0o600)
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap()), ["trusted/grade.py"])

    def test_missing_and_unreadable_trees_differ_from_a_normal_one(self):
        normal = self.snap()
        gone = sandbox.trusted_stat(self.dir / "does-not-exist")
        self.assertEqual(gone, {".": ("missing",)})
        self.assertEqual(sandbox.trusted_stat_diff(normal, gone)[0], "trusted")
        with mock.patch.object(sandbox, "MAX_TRUSTED_ENTRIES", 1):
            self.assertEqual(sandbox.trusted_stat(self.task.trusted_dir)["."], ("too_many",))

    def test_symlinks_are_not_followed(self):
        t = self.task.trusted_dir
        outside = self.dir / "outside"
        outside.mkdir()
        (outside / "f").write_text("1")
        os.symlink(outside, t / "ln")
        snap = self.snap()
        self.assertNotIn("ln/f", snap)
        (outside / "f").write_text("changed")                          # a change behind a link is not a change of trusted/
        self.assertEqual(sandbox.trusted_stat_diff(snap, self.snap()), [])

    def test_names_from_the_run_are_rendered_log_safe(self):
        before = self.snap()
        (self.task.trusted_dir / "a\nb\x1b[31m").write_text("x")
        names = sandbox.trusted_stat_diff(before, self.snap())
        self.assertTrue(all("\n" not in n and "\x1b" not in n for n in names), names)

    def test_bytecode_is_listed_for_the_run_check_and_ignorable_for_the_grader_check(self):
        t = self.task.trusted_dir
        before = self.snap()
        (t / "__pycache__").mkdir()
        (t / "__pycache__" / "grade.cpython-313.pyc").write_bytes(b"x")
        (t / "random.pyc").write_bytes(b"x")
        after = self.snap()
        self.assertEqual(sandbox.trusted_stat_diff(before, after),
                         ["trusted", "trusted/__pycache__", "trusted/__pycache__/grade.cpython-313.pyc", "trusted/random.pyc"])
        self.assertEqual(sandbox.trusted_stat_diff(before, after, ignore_bytecode=True), [])
        (t / "grade.py").write_text("print('changed')\n")
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap(), ignore_bytecode=True), ["trusted/grade.py"])


class TrustedStatAdversarialTests(SandboxCase):
    """Ways to hide a change from the snapshot that the first version of the check missed or could have missed."""

    def snap(self) -> dict:
        return sandbox.trusted_stat(self.task.trusted_dir)

    def test_a_trusted_directory_that_is_a_link_is_watched_through_the_link(self):
        real = self.dir / "task" / "trusted_real"
        os.rename(self.task.trusted_dir, real)
        os.symlink(real, self.task.trusted_dir)
        before = self.snap()
        self.assertIn("secret.txt", before, "the tree behind the link is what is watched, not just the link")
        path = real / "secret.txt"
        original, st = path.read_bytes(), path.stat()
        time.sleep(0.02)
        with open(path, "r+b") as fh:
            fh.write(b"X" * 4)
        with open(path, "r+b") as fh:
            fh.write(original[:4])
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap()), ["trusted/secret.txt"])

    def test_swapping_the_link_for_a_copy_of_the_tree_is_a_change(self):
        real = self.dir / "task" / "trusted_real"
        os.rename(self.task.trusted_dir, real)
        os.symlink(real, self.task.trusted_dir)
        before = self.snap()
        copy = self.dir / "task" / "trusted_copy"
        shutil.copytree(real, copy, symlinks=True)
        os.unlink(self.task.trusted_dir)
        os.symlink(copy, self.task.trusted_dir)
        self.assertTrue(sandbox.trusted_stat_diff(before, self.snap()))

    def test_a_dangling_link_is_missing_not_an_error(self):
        os.symlink(self.dir / "nowhere", self.dir / "dangling")
        self.assertEqual(sandbox.trusted_stat(self.dir / "dangling"), {".": ("missing",)})

    def test_a_file_that_comes_and_goes_is_still_seen_after_the_grader_when_no_bytecode_is_involved(self):
        t = self.task.trusted_dir
        before = self.snap()
        (t / "random.py").write_text("import os\n")              # planted for the grader's import, removed afterwards
        os.unlink(t / "random.py")
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap(), ignore_bytecode=True), ["trusted"])

    def test_bytecode_next_to_a_transient_file_is_excused_only_for_the_bytecode(self):
        t = self.task.trusted_dir
        before = self.snap()
        (t / "__pycache__").mkdir()
        (t / "__pycache__" / "grade.cpython-313.pyc").write_bytes(b"x")
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap(), ignore_bytecode=True), [])
        before = self.snap()                                      # __pycache__ exists now; rewriting inside it is excused
        (t / "__pycache__" / "grade.cpython-313.pyc").write_bytes(b"yy")
        (t / "__pycache__" / "other.cpython-313.pyc").write_bytes(b"z")
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap(), ignore_bytecode=True), [])
        before = self.snap()
        (t / "x.py").write_text("1")
        os.unlink(t / "x.py")
        self.assertEqual(sandbox.trusted_stat_diff(before, self.snap(), ignore_bytecode=True), ["trusted"])


class StrayBytecodeTests(SandboxCase):
    """A source-less ``*.pyc`` next to the grader shadows the standard library and is invisible to the content hash."""

    def stray(self) -> list[str]:
        return sandbox.stray_bytecode(sandbox.trusted_stat(self.task.trusted_dir))

    def test_a_clean_trusted_tree_has_none(self):
        self.assertEqual(self.stray(), [])

    def test_bytecode_inside_pycache_is_not_stray_but_a_top_level_or_nested_pyc_is(self):
        t = self.task.trusted_dir
        (t / "__pycache__").mkdir()
        (t / "__pycache__" / "grade.cpython-313.pyc").write_bytes(b"x")
        self.assertEqual(self.stray(), [])
        (t / "random.pyc").write_bytes(b"x")
        (t / "sub").mkdir()
        (t / "sub" / "json.pyc").write_bytes(b"x")
        (t / "sub" / "__pycache__").mkdir()
        (t / "sub" / "__pycache__" / "m.cpython-313.pyc").write_bytes(b"x")
        self.assertEqual(self.stray(), ["trusted/random.pyc", "trusted/sub/json.pyc"])

    def test_names_are_log_safe(self):
        (self.task.trusted_dir / "a\nb.pyc").write_bytes(b"x")
        self.assertTrue(all("\n" not in n for n in self.stray()))

    def test_python_really_imports_such_a_file_ahead_of_the_standard_library(self):
        """The reason for the check: a planted ``colorsys.pyc`` runs when the grader imports ``colorsys``."""
        import py_compile
        t = self.task.trusted_dir
        src = self.dir / "evil.py"
        src.write_text("print('PLANTED CODE RAN')\n")
        py_compile.compile(str(src), cfile=str(t / "colorsys.pyc"))
        (t / "grade_probe.py").write_text("import colorsys\n")
        res = subprocess.run([sys.executable, "-B", str(t / "grade_probe.py")], capture_output=True, text=True, cwd=self.dir)
        self.assertIn("PLANTED CODE RAN", res.stdout)
        self.assertEqual(self.stray(), ["trusted/colorsys.pyc"])


class TrustedWatchTests(SandboxCase):
    """The window gate: when is a difference attributable, when is it ambiguous, and what ``solo`` guarantees."""

    def watch(self, **kw) -> sandbox.TrustedWatch:
        return sandbox.TrustedWatch(self.task.trusted_dir, **kw)

    def touch(self) -> None:
        with open(self.task.trusted_dir / "secret.txt", "ab") as fh:
            fh.write(b"x")

    def test_a_lone_window_is_never_contested(self):
        with self.watch() as w:
            self.assertEqual(w.check(), [])
            self.touch()
            self.assertEqual(w.check(), ["trusted/secret.txt"])
            self.assertFalse(w.contested)
            self.assertEqual(w.check(), ["trusted/secret.txt"], "check() never re-baselines")

    def test_a_window_that_overlaps_another_is_contested_whichever_order_they_start_and_end_in(self):
        with self.watch() as a:
            with self.watch() as b:
                self.assertTrue(a.contested, "b started inside a's window")
                self.assertTrue(b.contested, "a was active when b started")
            self.assertTrue(a.contested, "b ended, but it did overlap")
        with self.watch() as c:
            self.assertFalse(c.contested, "windows that do not overlap in time do not contest each other")

    def test_arm_takes_the_baseline_again(self):
        with self.watch() as w:
            self.touch()
            self.assertTrue(w.check())
            w.arm()
            self.assertEqual(w.check(), [])

    def test_the_gate_is_released_on_every_exit_path(self):
        with self.assertRaises(RuntimeError):
            with self.watch():
                raise RuntimeError("boom")
        with mock.patch.object(sandbox, "trusted_stat", side_effect=[OSError("stat"), {}]):
            with self.assertRaises(OSError):
                self.watch().__enter__()
        with self.watch(solo=True) as w:
            self.assertTrue(w.solo)

    def test_solo_waits_for_the_others_and_holds_new_ones_back(self):
        import threading
        order: list[str] = []
        first_in, release_first = threading.Event(), threading.Event()

        def shared_one():
            with self.watch():
                order.append("shared-in")
                first_in.set()
                release_first.wait(5)
                order.append("shared-out")

        def solo_one():
            first_in.wait(5)
            with self.watch(solo=True) as w:
                order.append("solo-in")
                self.assertTrue(w.solo)
                self.assertFalse(w.contested)
                time.sleep(0.1)
                order.append("solo-out")

        def late_shared():
            time.sleep(0.2)                      # the solo request is waiting by now
            with self.watch():
                order.append("late-in")

        threads = [threading.Thread(target=f) for f in (shared_one, solo_one, late_shared)]
        for t in threads:
            t.start()
        time.sleep(0.4)
        self.assertEqual(order, ["shared-in"], "the solo window waits and the late window is held back behind it")
        release_first.set()
        for t in threads:
            t.join(10)
        self.assertEqual(order, ["shared-in", "shared-out", "solo-in", "solo-out", "late-in"])

    def test_a_solo_window_that_is_broken_into_by_a_waiter_that_gave_up_is_contested(self):
        """After SOLO_WAIT_SEC a shared request enters anyway; the solo run must then not be charged with that run's changes."""
        import threading
        with mock.patch.object(sandbox, "SOLO_WAIT_SEC", 0.05):
            with self.watch(solo=True) as solo:
                self.assertTrue(solo.solo)
                self.assertFalse(solo.contested)
                entered = threading.Event()

                def intruder():
                    with self.watch():
                        entered.set()

                t = threading.Thread(target=intruder)
                t.start()
                t.join(10)
                self.assertTrue(entered.is_set(), "the intruder gave up waiting and entered")
                self.assertTrue(solo.contested, "the solo guarantee was broken, so a difference is ambiguous")

    def test_random_interleavings_never_hide_an_overlap_and_a_granted_solo_window_is_never_shared(self):
        import random
        import threading
        records: list[tuple[float, float, bool, bool]] = []
        lock = threading.Lock()

        def worker(seed: int) -> None:
            rnd = random.Random(seed)
            for _ in range(12):
                w = self.watch(solo=rnd.random() < 0.3)
                with w:
                    t0 = time.monotonic()
                    time.sleep(rnd.random() * 0.01)
                    t1 = time.monotonic()
                    contested = w.contested
                with lock:
                    records.append((t0, t1, w.solo, contested))

        with mock.patch.object(sandbox, "SOLO_WAIT_SEC", 0.3):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
        self.assertEqual(len(records), 72)
        for a in records:
            overlapping = [b for b in records if b is not a and b[0] < a[1] and a[0] < b[1]]
            if overlapping:
                self.assertTrue(a[3], "a window that overlapped another must say so")
            if a[2]:
                self.assertEqual(overlapping, [], "a granted solo window never shares the process with another window")

    def test_a_solo_request_that_cannot_get_a_quiet_process_falls_back_to_a_shared_window(self):
        with self.watch():
            gate = sandbox._GATE
            others, starts, granted = gate.enter(solo=True, wait_sec=0.05)
            self.addCleanup(gate.leave, solo=False)
            self.assertFalse(granted)
            self.assertEqual(others, 1)


class TrustedWatchInRunUntrustedTests(SandboxCase):
    """``run_untrusted`` reports a change to ``trusted/`` after the process group is reaped, in every mode."""

    WRITE_RESTORE = textwrap.dedent("""
        import os
        path = {path!r}
        st = os.stat(path)
        orig = open(path, "rb").read()
        with open(path, "r+b") as fh:
            fh.write(b"X" * 4)
        with open(path, "r+b") as fh:
            fh.write(orig[:4])
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        print("done")
    """)

    def run_script(self, source: str, **kw) -> sandbox.RunResult:
        ws = sandbox.create_workspace(self.task, self.run_dir, f"e{len(list((self.run_dir / 'exp').glob('*'))) if (self.run_dir / 'exp').exists() else 0:03d}")
        (ws.path / "run.py").write_text(source)
        kw.setdefault("isolation", "none")
        return sandbox.run_untrusted(self.task, ws.path, 1, **kw)

    def test_an_honest_run_reports_nothing(self):
        res = self.run_script(OK_RUN)
        self.assertEqual((res.classification, res.trusted_changed, res.trusted_contested), ("ok", [], False))
        self.assertNotIn("trusted_tree_changed", res.note)

    def test_reading_trusted_is_not_a_change(self):
        res = self.run_script(f"import os\nprint(len(open({str(self.task.trusted_dir / 'secret.txt')!r}).read()), os.listdir({str(self.task.trusted_dir)!r}))\n")
        self.assertEqual((res.classification, res.trusted_changed), ("ok", []))

    def test_write_then_restore_is_reported_with_the_same_path_shape_as_a_hash_mismatch(self):
        path = self.task.trusted_dir / "secret.txt"
        before = path.read_bytes()
        time.sleep(0.02)
        res = self.run_script(self.WRITE_RESTORE.format(path=str(path)))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual((res.classification, res.trusted_changed, res.trusted_contested), ("ok", ["trusted/secret.txt"], False))
        self.assertIn("trusted_tree_changed: trusted/secret.txt", res.note)

    def test_a_change_made_by_a_survivor_after_the_run_is_not_attributed_to_the_run_window_but_the_watch_sees_it(self):
        with self.assertRaises(RuntimeError):
            with sandbox.TrustedWatch(self.task.trusted_dir) as w:
                self.run_script(OK_RUN, watch=w)
                with open(self.task.trusted_dir / "secret.txt", "ab") as fh:
                    fh.write(b"late")
                self.assertEqual(w.check(), ["trusted/secret.txt"])
                raise RuntimeError

    def test_overlapping_runs_make_a_difference_ambiguous_so_the_run_is_infra_not_tampering(self):
        import threading
        marker = self.dir / "other-started"
        path = self.task.trusted_dir / "secret.txt"
        results: dict[str, sandbox.RunResult] = {}
        slow = f"import os, time\nopen({str(marker)!r}, 'w').write('x')\ntime.sleep(1.5)\nprint('ok')\n"
        hostile = (f"import os, time\nwhile not os.path.exists({str(marker)!r}):\n    time.sleep(0.01)\n"
                   + self.WRITE_RESTORE.format(path=str(path)))
        time.sleep(0.02)

        def go(name, source):
            results[name] = self.run_script(source)
        # make the two workspaces first: run_script names them from the directory listing
        threads = [threading.Thread(target=go, args=("honest", slow)), threading.Thread(target=go, args=("hostile", hostile))]
        threads[0].start()
        time.sleep(0.3)
        threads[1].start()
        for t in threads:
            t.join(30)
        for name in ("honest", "hostile"):
            res = results[name]
            self.assertEqual((res.classification, res.trusted_contested), ("infra", True), (name, res.note))
            self.assertEqual(res.trusted_changed, ["trusted/secret.txt"])
            self.assertIn("may come from a concurrent lane", res.note)
            self.assertIn("trusted_tree_changed_during_run", res.note)
        # run alone (the retry), the hostile run is caught for what it is and the honest one is clean
        self.assertEqual(self.run_script(OK_RUN, solo=True).trusted_changed, [])
        again = self.run_script(self.WRITE_RESTORE.format(path=str(path)), solo=True)
        self.assertEqual((again.classification, again.trusted_changed, again.trusted_contested), ("ok", ["trusted/secret.txt"], False))

    def test_userns_masks_the_tree_so_there_is_nothing_to_change_or_report(self):
        if sandbox.effective_isolation("userns") != "userns":
            self.skipTest("user namespaces are not available here")
        path = self.task.trusted_dir / "secret.txt"
        res = self.run_script(self.WRITE_RESTORE.format(path=str(path)), isolation="userns")
        self.assertEqual(res.isolation, "userns")
        self.assertEqual(res.trusted_changed, [])
        self.assertNotEqual(res.exit_code, 0, "the path is not even visible inside the namespace")
        self.assertIn("FileNotFoundError", res.stderr)


class PidNamespaceNetVariantTests(SandboxCase):
    """``unshare-net`` as root also gets a pid namespace and a private /proc, but only when a probe proves it works."""

    def setUp(self):
        super().setUp()
        self.addCleanup(lambda: sandbox.available_isolation(refresh=True))

    def test_prefix_helper(self):
        self.assertTrue(sandbox.prefix_isolates_pids(["/usr/bin/unshare", "--net", "--pid", "--fork", "--mount-proc"]))
        self.assertFalse(sandbox.prefix_isolates_pids(["/usr/bin/unshare", "--net"]))
        self.assertFalse(sandbox.prefix_isolates_pids(None))

    def test_candidates_are_tried_strongest_first_and_each_must_be_proven(self):
        hardened = ["/u", "--net", "--pid", "--fork", "--mount-proc"]
        with mock.patch.object(sandbox.os, "geteuid", return_value=0), \
                mock.patch.object(sandbox, "_probe_pidns_prefix", return_value=True) as pidns, \
                mock.patch.object(sandbox, "_probe_run") as probe_run:
            self.assertEqual(sandbox._probe_net_prefix("/u", "/bin/sh"), hardened)
        pidns.assert_called_once_with(hardened, "/bin/sh")
        probe_run.assert_not_called()
        with mock.patch.object(sandbox.os, "geteuid", return_value=0), \
                mock.patch.object(sandbox, "_probe_pidns_prefix", return_value=False), \
                mock.patch.object(sandbox, "_probe_run", return_value=(0, b'["lo"]\n')):
            self.assertEqual(sandbox._probe_net_prefix("/u", "/bin/sh"), ["/u", "--net"])
        with mock.patch.object(sandbox.os, "geteuid", return_value=1000), \
                mock.patch.object(sandbox, "_probe_pidns_prefix", side_effect=AssertionError("non-root cannot unshare --pid")), \
                mock.patch.object(sandbox, "_probe_run", side_effect=[(1, b""), (0, b'["lo"]\n')]):
            self.assertEqual(sandbox._probe_net_prefix("/u", "/bin/sh"), ["/u", "--user", "--map-root-user", "--net"])

    def test_the_pid_namespace_probe_rejects_anything_short_of_the_full_effect(self):
        good = {"ifaces": ["lo"], "pid": 1, "procs": 2, "capeff": 0, "umount_ok": False}
        runs = lambda seen, status=7: [(0, json.dumps(seen).encode()), (status, b"")]      # noqa: E731
        prefix = ["/u", "--net", "--pid", "--fork", "--mount-proc"]
        with mock.patch.object(sandbox, "_probe_run", side_effect=runs(good)):
            self.assertTrue(sandbox._probe_pidns_prefix(prefix, "/bin/sh"))
        for field_name, bad in (("ifaces", ["eth0", "lo"]), ("pid", 4242), ("procs", 400), ("capeff", 1 << 21), ("umount_ok", True)):
            with self.subTest(field=field_name), mock.patch.object(sandbox, "_probe_run", side_effect=runs({**good, field_name: bad})):
                self.assertFalse(sandbox._probe_pidns_prefix(prefix, "/bin/sh"))
        with mock.patch.object(sandbox, "_probe_run", side_effect=runs(good, status=1)):
            self.assertFalse(sandbox._probe_pidns_prefix(prefix, "/bin/sh"), "the command's exit status must come back unchanged")
        with mock.patch.object(sandbox, "_probe_run", return_value=(1, b"")):
            self.assertFalse(sandbox._probe_pidns_prefix(prefix, "/bin/sh"))
        with mock.patch.object(sandbox, "_probe_run", return_value=(0, b"not json")):
            self.assertFalse(sandbox._probe_pidns_prefix(prefix, "/bin/sh"))

    def test_capabilities_are_dropped_exactly_when_a_namespace_keeps_the_run_away_from_the_host(self):
        seen: list[dict] = []

        def fake_execute(argv, *, env, **kw):
            seen.append(json.loads(env["NIGHTSHIFT_SHIM_CONFIG"]))
            return sandbox._Outcome(0, False, False, False, 0.0, 1, 0, None)

        cases = [("unshare-net", ["/u", "--net", "--pid", "--fork", "--mount-proc"], True),
                 ("unshare-net", ["/u", "--net"], False), ("userns", None, True), ("none", None, False)]
        for level, prefix, want in cases:
            info = {"level": level, "unshare": "/u", "sh": "/bin/sh", "net_prefix": prefix, "reason": ""}
            ws = sandbox.create_workspace(self.task, self.run_dir, f"e{len(seen):03d}")
            with self.subTest(level=level, prefix=prefix), mock.patch.object(sandbox, "_execute", fake_execute), \
                    mock.patch.object(sandbox, "_effective_isolation", return_value=(level, info, [])):
                sandbox.run_untrusted(self.task, ws.path, 1, isolation=level)
            self.assertEqual(seen[-1]["drop_caps"], want)

    @unittest.skipUnless(LINUX and os.geteuid() == 0, "needs Linux and root")
    def test_as_root_the_run_cannot_read_the_orchestrators_environ_when_the_variant_is_proven(self):
        with mock.patch.dict(os.environ, {sandbox.ISOLATION_ENV: "unshare-net"}):
            sandbox.available_isolation(refresh=True)
            prefix = sandbox._effective_isolation("unshare-net")[1].get("net_prefix")
            if not sandbox.prefix_isolates_pids(prefix):
                self.skipTest(f"the pid-namespace variant is not proven here (net prefix {prefix})")
            peek = f"import os\ntry:\n    open('/proc/{os.getpid()}/environ', 'rb').read(); print('READ')\nexcept OSError as e:\n    print('DENIED', type(e).__name__)\n"
            ws = sandbox.create_workspace(self.task, self.run_dir, "e900")
            (ws.path / "run.py").write_text(peek)
            hardened = sandbox.run_untrusted(self.task, ws.path, 1, isolation="unshare-net")
            ws2 = sandbox.create_workspace(self.task, self.run_dir, "e901")
            (ws2.path / "run.py").write_text(peek)
            control = sandbox.run_untrusted(self.task, ws2.path, 1, isolation="none")
        self.assertEqual(hardened.isolation, "unshare-net")
        self.assertTrue(hardened.stdout.startswith("DENIED"), hardened.stdout + hardened.stderr)
        self.assertEqual(control.stdout.strip(), "READ", "positive control: as root, without a pid namespace, it is readable")

    @unittest.skipUnless(LINUX and os.geteuid() == 0, "needs Linux and root")
    def test_the_run_cannot_undo_the_private_proc_and_the_exit_status_survives(self):
        with mock.patch.dict(os.environ, {sandbox.ISOLATION_ENV: "unshare-net"}):
            sandbox.available_isolation(refresh=True)
            if not sandbox.prefix_isolates_pids(sandbox._effective_isolation("unshare-net")[1].get("net_prefix")):
                self.skipTest("the pid-namespace variant is not proven here")
            ws = sandbox.create_workspace(self.task, self.run_dir, "e902")
            (ws.path / "run.py").write_text("import subprocess, sys\nr = subprocess.run(['umount', '/proc'], capture_output=True)\n"
                                            "print('umount rc', r.returncode)\nsys.exit(3)\n")
            res = sandbox.run_untrusted(self.task, ws.path, 1, isolation="unshare-net")
        self.assertEqual((res.exit_code, res.classification), (3, "crash"))
        self.assertNotIn("umount rc 0", res.stdout)


def _count_sleepers() -> int:
    n = 0
    if not LINUX:
        return n
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/cmdline", "rb") as fh:
                    if b"time.sleep(30)" in fh.read():
                        n += 1
            except OSError:
                continue
    return n


class HardenParentTests(unittest.TestCase):
    ROOT = str(Path(__file__).resolve().parents[1])
    CHILD_READS_PARENT_ENVIRON = (
        "import subprocess, sys\n"
        "from nightshift import sandbox\n"
        "if sys.argv[1] == 'harden':\n    sandbox.harden_parent()\n"
        "code = ('import os\\ntry:\\n    open(\"/proc/%d/environ\" % os.getppid()).read(); print(\"READ\")\\n'\n"
        "        'except PermissionError:\\n    print(\"DENIED\")')\n"
        "print(subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.strip())\n")

    def python(self, code: str, *args: str, user: int | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=20, user=user,
                              group=user, env={**os.environ, "PYTHONPATH": self.ROOT, "PYTHONDONTWRITEBYTECODE": "1"})

    @unittest.skipUnless(LINUX, "prctl is Linux only")
    def test_process_becomes_non_dumpable(self):
        res = self.python("import ctypes\nfrom nightshift import sandbox\nlibc = ctypes.CDLL(None)\n"
                          "before = libc.prctl(3, 0, 0, 0, 0)\nsandbox.harden_parent()\nprint(before, libc.prctl(3, 0, 0, 0, 0))")
        self.assertEqual(res.stdout.split(), ["1", "0"], res.stderr)

    @unittest.skipUnless(LINUX, "prctl is Linux only")
    def test_a_same_uid_child_cannot_read_the_parents_environ(self):
        user = 65534 if os.geteuid() == 0 else None          # root can read everything, so drop to an unprivileged uid
        try:
            control = self.python(self.CHILD_READS_PARENT_ENVIRON, "plain", user=user)
        except (PermissionError, OSError) as exc:
            self.skipTest(f"cannot run an unprivileged python here: {exc}")
        if control.stdout.strip() != "READ":
            self.skipTest(f"positive control did not read the environ ({control.stdout!r} {control.stderr[-200:]!r})")
        hardened = self.python(self.CHILD_READS_PARENT_ENVIRON, "harden", user=user)
        self.assertEqual(hardened.stdout.strip(), "DENIED", hardened.stderr)

    @unittest.skipUnless(LINUX and os.geteuid() == 0, "needs root: the limit of the hardening is that root is not stopped")
    def test_root_is_not_stopped_by_it(self):
        """The docstring says so; this keeps the claim honest: a child of a root orchestrator reads the environ anyway."""
        res = self.python(self.CHILD_READS_PARENT_ENVIRON, "harden")
        self.assertEqual(res.stdout.strip(), "READ", res.stderr)

    def test_the_docstring_states_what_it_protects_and_what_it_does_not(self):
        doc = sandbox.harden_parent.__doc__ or ""
        for phrase in ("NON-ROOT", "root", "CAP_SYS_PTRACE", "macOS"):
            self.assertIn(phrase, doc)

    def test_never_raises(self):
        with mock.patch("ctypes.CDLL", side_effect=OSError("no libc")):
            self.assertIsNone(sandbox.harden_parent())
        with mock.patch.object(sandbox, "_LINUX", False):
            self.assertIsNone(sandbox.harden_parent())


if __name__ == "__main__":
    unittest.main()
