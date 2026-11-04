"""Tests for nightshift.lab.packs, assets/lint_pack.py and the v0 pack template (owner 4).

Real git on temporary repositories; the task.yaml comparison runs over JSON fixtures; no network, no CORAL, no PyYAML.
Run: PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_packs
"""

import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nightshift.lab import packs
from nightshift.lab.contracts import (
    ContractError,
    EditOp,
    LabLayout,
    Lever,
    PackEdit,
    PackRef,
    Proposal,
    ProposalInvalid,
    SearchReplace,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lab" / "pack_json"
V0_DIR = packs.PACK_V0_TEMPLATE
NOW = 1_793_000_000.0


def clock():
    return NOW


def v0_json() -> dict:
    return json.loads((FIXTURES / "v0_task.json").read_text(encoding="utf-8"))


def ranges() -> dict:
    return packs.default_lint_ranges()


def sr(path: str, *pairs: tuple) -> PackEdit:
    return PackEdit(path, EditOp.SEARCH_REPLACE, None, tuple(SearchReplace(a, b) for a, b in pairs))


def rf(path: str, content: str) -> PackEdit:
    return PackEdit(path, EditOp.REPLACE_FILE, content, ())


def proposal(*edits: PackEdit, summary: str = "bound the notes") -> Proposal:
    return Proposal(
        hypothesis="Researchers re-read the whole log.",
        lever=Lever.CONTEXT_NOTES,
        change_summary=summary,
        expected_effect="Less context per turn.",
        risk="Loses detail.",
        edits=edits,
    )


class StrictReaderTests(unittest.TestCase):
    def test_v0_task_yaml_equals_the_committed_json_dump(self):
        """The fixture is what the lint compares; it cannot drift from the template."""
        parsed = packs.parse_task_yaml((V0_DIR / "task.yaml").read_text(encoding="utf-8"))
        self.assertEqual(parsed, v0_json())

    def test_literal_blocks_clip_and_strip(self):
        doc = packs.parse_task_yaml("a: |\n  one\n  two\n\nb: |-\n  x\nc: 1\n")
        self.assertEqual(doc, {"a": "one\ntwo\n", "b": "x", "c": 1})

    def test_scalars_and_nesting(self):
        doc = packs.parse_task_yaml(
            'k: "q: #x"\ns: \'it\'\'s\'\nn: ~\nt: true\nf: 1.5\ni: -3\nl: []\nm: {}\nj: [1, "a"]\nlst:\n  - a b\n  - name: x\n    n: 2\n  - "z"\nplain: uv pip install -e ./g\n'
        )
        self.assertEqual(doc["k"], "q: #x")
        self.assertEqual(doc["s"], "it's")
        self.assertIsNone(doc["n"])
        self.assertIs(doc["t"], True)
        self.assertEqual((doc["f"], doc["i"], doc["l"], doc["m"], doc["j"]), (1.5, -3, [], {}, [1, "a"]))
        self.assertEqual(doc["lst"], ["a b", {"name": "x", "n": 2}, "z"])
        self.assertEqual(doc["plain"], "uv pip install -e ./g")

    def test_refusals(self):
        bad = {
            "tab": "a:\n\tb: 1\n",
            "duplicate key": "a: 1\na: 2\n",
            "bool-like plain scalar": "a: yes\n",
            "trailing comment": "a: 1 # c\n",
            "anchor": "a: &x 1\n",
            "alias": "a: *x\n",
            "tag": "a: !!str 1\n",
            "folded scalar": "a: >\n  text\n",
            "sequence at the key's indent": "a:\n- 1\n",
            "date-like": "a: 2026-11-01\n",
            "exponent": "a: 1e3\n",
            "hex": "a: 0x10\n",
            "colon in plain scalar": "a: b: c\n",
            "non-JSON flow": "a: [x, y]\n",
            "document marker": "---\na: 1\n",
            "quoted text after string": 'a: "x" y\n',
            "empty literal": "a: |\nb: 1\n",
            "carriage return": "a: 1\r\n",
            "top-level list": "- 1\n",
            "bad key": "a b: 1\n",
        }
        for why, text in bad.items():
            with self.subTest(why):
                with self.assertRaises(ValueError):
                    packs.parse_task_yaml(text)

    def test_pyyaml_agrees_on_the_v0_file_when_it_is_installed(self):
        try:
            import yaml  # type: ignore
        except ImportError:
            self.skipTest("PyYAML is not installed here; the lead-run check on the researcher host compares the two readers")
        text = (V0_DIR / "task.yaml").read_text(encoding="utf-8")
        self.assertEqual(yaml.safe_load(text), packs.parse_task_yaml(text))


    def test_ruby_psych_agrees_on_the_accepted_subset_when_ruby_is_installed(self):
        """A second YAML 1.1 implementation (libyaml, like PyYAML) reads everything the strict reader accepts in the same way."""
        if shutil.which("ruby") is None:
            self.skipTest("ruby is not installed")
        docs = [
            (V0_DIR / "task.yaml").read_text(encoding="utf-8"),
            'k: "q: #x"\ns: \'it\'\'s\'\nn: ~\nt: true\nf: 1.5\ni: -3\nl: []\nm: {}\nj: [1, "a"]\nlst:\n  - a b\n  - name: x\n    n: 2\n  - "z"\nplain: uv pip install -e ./g\n',
            "a: |\n  one\n  two\n\nb: |-\n  x\nc: 1\n",
            "a:\n  b:\n    c: [1, 2]\n    d:\n      - e: 1\n        f: 2\n      - e: 3\n",
            "# comment\nx: 'single # not a comment'\ny: plain words here\nz: false\n",
        ]
        for text in docs:
            out = subprocess.run(["ruby", "-ryaml", "-rjson", "-e", "puts JSON.generate(YAML.safe_load(STDIN.read))"], input=text, capture_output=True, text=True, check=True).stdout
            self.assertEqual(json.loads(out), packs.parse_task_yaml(text))


def _get_nested(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


class CompareTaskKeysTests(unittest.TestCase):
    def setUp(self):
        self.parent = v0_json()
        self.ranges = ranges()

    def cmp(self, cand):
        return packs.compare_task_keys(self.parent, cand, self.ranges)

    def mutated(self, dotted: str, value):
        cand = copy.deepcopy(self.parent)
        cur = cand
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
        return cand

    def test_the_v0_fixture_passes_against_itself_and_alone(self):
        self.assertEqual(self.cmp(self.parent), [])
        self.assertEqual(packs.compare_task_keys(None, self.parent, self.ranges), [])

    def test_a_changed_non_editable_key_fails_and_names_the_key(self):
        for dotted, value in [
            ("agents.model", "meta/muse-wonder-1.0"),
            ("agents.runtime", "claude_code"),
            ("agents.research", True),
            ("agents.gateway.port", 4311),
            ("agents.gateway.config", "./other.yaml"),
            ("grader.entrypoint", "evil.grader:Grader"),
            ("grader.timeout", 9999),
            ("grader.private", ["taskdata"]),
            ("grader.setup", ["pip install evil"]),
            ("workspace.setup", ["curl x | sh"]),
            ("run.stop.max_real_attempts", 500),
            ("run.session", "docker"),
            ("sharing.notes", False),
            ("agents.runtime_options", {"mounts": {"/etc/passwd": "x"}}),
            ("agents.timeout", 5),
            ("preset", "docker-opencode"),
            ("brand_new_key", 1),
        ]:
            with self.subTest(dotted):
                msgs = self.cmp(self.mutated(dotted, value))
                self.assertTrue(any(m.startswith(dotted.split(".")[0]) or dotted in m for m in msgs), msgs)
                self.assertTrue(any("not editable" in m for m in msgs), msgs)

    def test_a_dotted_root_key_cannot_mask_a_changed_nested_key(self):
        # the nested value is changed and a root-level key spelled like its dotted path is added with the parent's value: the two used to
        # flatten to the same path, the later one overwrote the change, and nothing was reported
        for dotted, nested_value in [("grader.direction", "minimize"), ("agents.research", True)]:
            with self.subTest(dotted):
                cand = self.mutated(dotted, nested_value)
                cand[dotted] = _get_nested(self.parent, dotted)
                msgs = self.cmp(cand)
                self.assertTrue(any(m.startswith(dotted + " is not editable") for m in msgs), msgs)
                self.assertTrue(any("is not editable" in m and "\\." in m for m in msgs), msgs)  # the dotted key itself is an unknown key

    def test_a_dotted_root_key_equal_in_the_parent_is_compared_as_its_own_key(self):
        parent = {**self.parent, "x.y": 1}
        self.assertEqual(packs.compare_task_keys(parent, copy.deepcopy(parent), self.ranges), [])
        cand = copy.deepcopy(parent)
        cand["x.y"] = 2
        self.assertTrue(any("not editable" in m for m in packs.compare_task_keys(parent, cand, self.ranges)))

    def test_a_removed_non_editable_key_fails(self):
        cand = copy.deepcopy(self.parent)
        del cand["grader"]["direction"]
        self.assertTrue(any("grader.direction" in m for m in self.cmp(cand)))

    def test_int_versus_bool_is_a_difference(self):
        cand = self.mutated("run.stop.max_real_attempts", True)
        self.assertTrue(self.cmp(cand))

    def test_editable_keys_inside_the_ranges_pass(self):
        cand = copy.deepcopy(self.parent)
        cand["task"]["description"] = "New instructions."
        cand["task"]["tips"] = "A tip."
        cand["agents"]["count"] = 3
        cand["agents"]["heartbeat"][0]["every"] = 7
        cand["agents"]["heartbeat"][2]["prompt"] = "Pivot now."
        cand["agents"]["skills"] = ["skills/notes-hygiene"]
        cand["agents"]["assignments"] = [
            {"count": 1, "runtime_options": {"role_file": "roles/lead.md"}},
            {"count": 2, "runtime": "opencode", "model": "muse/muse-spark-1.3"},
        ]
        self.assertEqual(self.cmp(cand), [])

    def test_editable_keys_outside_the_ranges_fail(self):
        cases = {
            "agents.count": 9,
            "agents.count ": 0,
            "agents.count  ": True,
            "task.description": "",
            "task.description ": "x" * 20001,
            "agents.heartbeat": [{"name": "reflect", "every": 0}],
            "agents.heartbeat ": [{"name": "reflect", "every": 51}],
            "agents.heartbeat  ": [{"name": "exfiltrate", "every": 5}],
            "agents.heartbeat   ": [{"name": "reflect", "every": 5, "command": "rm -rf"}],
            "agents.heartbeat    ": [{"name": "reflect", "every": 5, "trigger": "cron"}],
            "agents.heartbeat     ": [{"name": "pivot", "every": 5, "options": {"url": "x"}}],
            "agents.heartbeat      ": [{"name": "reflect", "every": 5}] * 9,
            "agents.assignments": [{"count": 1, "model": "meta/muse-wonder-1.0"}],
            "agents.assignments ": [{"count": 1, "runtime": "claude_code"}],
            "agents.assignments  ": [{"count": 1, "runtime_options": {"mounts": {"a": "b"}}}],
            "agents.assignments   ": [{"count": 1, "runtime_options": {"role_file": "/etc/passwd"}}],
            "agents.assignments    ": [{"count": 1, "runtime_options": {"role_file": "roles/../x"}}],
            "agents.assignments     ": [{"count": 4}, {"count": 4}],
            "agents.assignments      ": [{"count": 1, "isolate_user": "root"}],
            "agents.skills": ["/etc"],
            "agents.skills ": ["skills/../../x"],
            "agents.skills  ": ["other/dir"],
        }
        for key, value in cases.items():
            with self.subTest(key, value=str(value)[:40]):
                msgs = self.cmp(self.mutated(key.strip(), value))
                self.assertTrue(msgs, "expected a message")

    def test_messages_fit_the_contract_limit(self):
        cand = self.mutated("grader.entrypoint", "x" * 5000)
        for m in self.cmp(cand):
            self.assertLessEqual(len(m.encode("utf-8")), 500)

    def test_grader_values_must_equal_the_runtimes(self):
        ep = {"entrypoint": "glimmer_grader.grader:Grader", "setup": ["uv pip install -e ./grader"], "timeout": 120}
        self.assertEqual(packs.compare_grader_values(self.parent, ep), [])
        msgs = packs.compare_grader_values(self.mutated("grader.timeout", 121), ep)
        self.assertTrue(any("grader.timeout" in m for m in msgs))
        self.assertTrue(packs.compare_grader_values(self.mutated("grader.entrypoint", "x:Y"), ep))

    def test_the_v0_template_uses_the_runtimes_grader_values(self):
        if not packs.ENTRYPOINT_JSON.is_file():
            self.skipTest("owner 5's ENTRYPOINT.json is not in the tree yet")
        ep = json.loads(packs.ENTRYPOINT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(packs.compare_grader_values(self.parent, ep), [])


class ApplyEditsTests(unittest.TestCase):
    def setUp(self):
        self.files = {
            "task.yaml": "task:\n  name: x\n",
            "notes/seed.md": "alpha\nbeta\ngamma\n",
            "seed/recipe/run.sh": "echo hi\n",
        }
        self.editable = packs.make_is_editable(packs.DEFAULT_EDITABLE_PATHS)

    def apply(self, *edits, **kw):
        return packs.apply_proposal_edits(self.files, proposal(*edits), is_editable=self.editable, **kw)

    def test_search_replace_and_replace_file(self):
        out = self.apply(sr("notes/seed.md", ("beta", "BETA"), ("gamma\n", "")), rf("notes/new.md", "fresh\n"))
        self.assertEqual(out["notes/seed.md"], "alpha\nBETA\n")
        self.assertEqual(out["notes/new.md"], "fresh\n")
        self.assertEqual(out["task.yaml"], self.files["task.yaml"])

    def test_an_unmatched_block_is_rejected_not_skipped(self):
        with self.assertRaisesRegex(ProposalInvalid, "matches 0 times"):
            self.apply(sr("notes/seed.md", ("beta", "B"), ("not there", "x")))

    def test_an_ambiguous_block_is_rejected(self):
        self.files["notes/seed.md"] = "a\na\n"
        with self.assertRaisesRegex(ProposalInvalid, "matches 2 times"):
            self.apply(sr("notes/seed.md", ("a", "b")))

    def test_overlapping_blocks_are_rejected(self):
        with self.assertRaisesRegex(ProposalInvalid, "overlap"):
            self.apply(sr("notes/seed.md", ("alpha\nbeta", "x"), ("beta\ngamma", "y")))

    def test_search_replace_on_a_missing_file_is_rejected(self):
        with self.assertRaisesRegex(ProposalInvalid, "does not exist"):
            self.apply(sr("notes/none.md", ("a", "b")))

    def test_forbidden_paths_are_rejected(self):
        for path in [
            "grader/grader.py",
            "litellm_config.yaml",
            "workspace/x",
            "seed/opencode.json",
            "seed/opencode.jsonc",
            "seed/sub/opencode.json",
            "seed/sub/opencode.jsonc",
            "seed/.opencode/config.json",
            "seed/AGENTS.md",
            "notes/.coral/x",
            "seed/.claude/x",
            "run.sh",
            "submission.json",
            ".gitignore",
        ]:
            with self.subTest(path):
                with self.assertRaisesRegex(ProposalInvalid, "not an editable"):
                    self.apply(rf(path, "x"))

    def test_the_contract_already_refuses_traversal_and_dot_git(self):
        for path in ["../x", "/abs", "notes/../../x", "notes/./x", ".git/config", "notes//x"]:
            with self.subTest(path):
                with self.assertRaises(ContractError):
                    rf(path, "x")

    def test_size_caps(self):
        with self.assertRaisesRegex(ProposalInvalid, "file limit"):
            self.apply(rf("notes/big.md", "x" * 100), max_file_bytes=50)
        with self.assertRaisesRegex(ProposalInvalid, "bytes \\(limit"):
            self.apply(rf("notes/a.md", "x" * 40), rf("notes/b.md", "y" * 40), max_pack_bytes=100)
        with self.assertRaisesRegex(ProposalInvalid, "files"):
            self.apply(rf("notes/a.md", "a"), max_pack_files=3)

    def test_a_proposal_that_changes_nothing_is_rejected(self):
        with self.assertRaisesRegex(ProposalInvalid, "changes nothing"):
            self.apply(sr("notes/seed.md", ("beta", "beta")))

    def test_file_directory_clash_is_rejected(self):
        with self.assertRaisesRegex(ProposalInvalid, "also be a directory"):
            self.apply(rf("notes/seed.md/child.md", "x"))

    def test_editable_entries_are_prefixes_only_with_a_trailing_slash(self):
        ed = packs.make_is_editable(("task.yaml", "notes/"))
        self.assertTrue(ed("task.yaml") and ed("notes/a/b.md"))
        self.assertFalse(ed("task.yaml.bak") or ed("notes") or ed("notesx/a") or ed("seed/a"))
        with self.assertRaises(ContractError):
            packs.make_is_editable(("/abs",))


class RepoCase(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "lab"
        self.root.mkdir()
        self.layout = LabLayout(self.root)
        self.ranges_path = self.root / "lint_ranges.json"
        self.repo = packs.PackRepo(self.layout, lint_ranges_path=self.ranges_path, clock=clock)

    def refs(self) -> dict:
        out = subprocess.run(
            ["git", "--git-dir", str(self.layout.pack_repo), "for-each-ref", "--format=%(refname) %(objectname)"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return dict(line.split(" ") for line in out.splitlines())

    def git(self, *args, input=None):
        return subprocess.run(
            ["git", "--git-dir", str(self.layout.pack_repo), *args], capture_output=True, text=True, input=input, check=True, env={
                **os.environ,
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@localhost",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@localhost",
            },
        ).stdout.strip()


class RepoTests(RepoCase):
    def test_init_v0_creates_main_and_the_v0_tag(self):
        v0 = self.repo.init_v0()
        self.assertEqual(v0.repo_path, "packs/pack.git")
        self.assertEqual(len(v0.commit), 40)
        refs = self.refs()
        self.assertEqual(refs["refs/heads/main"], v0.commit)
        self.assertEqual(refs["refs/tags/pack/v0"], v0.commit)
        self.assertEqual(self.repo.main_pack(), v0)
        self.assertEqual(self.repo.tagged("v0"), v0)
        self.assertTrue(self.ranges_path.is_file(), "the pack contract is written at init")
        self.assertEqual(json.loads(self.ranges_path.read_text()), ranges())

    def test_init_v0_twice_refuses(self):
        self.repo.init_v0()
        with self.assertRaisesRegex(ContractError, "already has a v0"):
            self.repo.init_v0()

    def test_init_v0_is_deterministic_for_content_and_clock(self):
        other_root = Path(self.tmp.name) / "lab2"
        other_root.mkdir()
        other = packs.PackRepo(LabLayout(other_root), lint_ranges_path=other_root / "r.json", clock=clock)
        self.assertEqual(self.repo.init_v0().commit, other.init_v0().commit)

    def test_the_v0_template_holds_no_protected_or_binary_files(self):
        v0 = self.repo.init_v0()
        files = self.repo.read_files(v0)
        self.assertIn("task.yaml", files)
        self.assertIn("notes/seed.md", files)
        self.assertIn("seed/recipe/run.sh", files)
        self.assertFalse([p for p in files if packs.is_protected(p)])
        self.assertTrue(all(packs.make_is_editable(packs.DEFAULT_EDITABLE_PATHS)(p) for p in files), "every v0 file is editable by the outer loop")

    def test_init_refuses_a_template_with_a_symlink_or_protected_file(self):
        t = Path(self.tmp.name) / "tpl"
        t.mkdir()
        (t / "task.yaml").write_text("task: {}\n")
        os.symlink("/etc/passwd", t / "link")
        with self.assertRaisesRegex(ContractError, "regular files"):
            self.repo.init_v0(t)
        os.unlink(t / "link")
        (t / "opencode.json").write_text("{}")
        with self.assertRaisesRegex(ContractError, "protected"):
            self.repo.init_v0(t)

    def test_checkout_is_faithful_clean_and_matches_the_tree_hash(self):
        v0 = self.repo.init_v0()
        dest = self.root / "co"
        self.repo.checkout(v0, dest)
        self.assertFalse((dest / ".git").exists())
        self.assertEqual(packs.dir_listing_sha256(dest), self.repo.tree_sha256(v0))
        self.assertEqual((dest / "task.yaml").read_text(), (V0_DIR / "task.yaml").read_text())
        with self.assertRaisesRegex(ContractError, "empty directory"):
            self.repo.checkout(v0, dest)

    def test_checkout_of_a_foreign_repo_or_commit_is_refused(self):
        v0 = self.repo.init_v0()
        with self.assertRaisesRegex(ContractError, "not this lab"):
            self.repo.checkout(PackRef("packs/other.git", v0.commit), self.root / "x")
        with self.assertRaisesRegex(ContractError, "not in the pack repository"):
            self.repo.checkout(PackRef(v0.repo_path, "1" * 40), self.root / "x")

    def test_apply_makes_a_tagged_child_commit_and_a_diff(self):
        v0 = self.repo.init_v0()
        before = self.refs()
        cand, diff = self.repo.apply(proposal(sr("notes/seed.md", ("## Log", "## Log (newest first)"))), v0, "cand-001")
        self.assertNotEqual(cand.commit, v0.commit)
        self.assertEqual(self.git("rev-parse", cand.commit + "^"), v0.commit)
        self.assertEqual(self.refs()["refs/tags/pack/cand-001"], cand.commit)
        self.assertEqual(self.refs()["refs/heads/main"], v0.commit, "apply never moves main")
        self.assertIn("+## Log (newest first)", diff)
        self.assertIn("notes/seed.md", diff)
        self.assertEqual({k: v for k, v in self.refs().items() if k != "refs/tags/pack/cand-001"}, before)
        self.assertIn("cand-001: bound the notes", self.git("log", "-1", "--format=%B", cand.commit))
        self.assertEqual(self.repo.read_files(cand)["notes/seed.md"].count("newest first"), 1)

    def test_a_rejected_proposal_changes_nothing(self):
        v0 = self.repo.init_v0()
        before = self.refs()
        objects = self.git("count-objects", "-v")
        with self.assertRaises(ProposalInvalid):
            self.repo.apply(proposal(sr("notes/seed.md", ("no such text", "x"))), v0, "cand-001")
        with self.assertRaises(ProposalInvalid):
            self.repo.apply(proposal(rf("grader/evil.py", "x")), v0, "cand-001")
        self.assertEqual(self.refs(), before)
        self.assertEqual(self.git("count-objects", "-v").split("\n")[0], objects.split("\n")[0], "no loose object was written")

    def test_a_version_label_is_never_reused_or_unsafe(self):
        v0 = self.repo.init_v0()
        self.repo.apply(proposal(rf("notes/a.md", "a")), v0, "cand-001")
        with self.assertRaisesRegex(ContractError, "already exists"):
            self.repo.apply(proposal(rf("notes/b.md", "b")), v0, "cand-001")
        for label in ["a..b", "x.lock", "-bad", "has space", "a/b", ""]:
            with self.subTest(label):
                with self.assertRaises(ContractError):
                    self.repo.apply(proposal(rf("notes/c.md", "c")), v0, label)

    def test_accept_fast_forwards_main_and_is_idempotent(self):
        v0 = self.repo.init_v0()
        c1, _ = self.repo.apply(proposal(rf("notes/a.md", "a")), v0, "cand-001")
        self.repo.accept(c1, "cand-001")
        self.assertEqual(self.refs()["refs/heads/main"], c1.commit)
        self.repo.accept(c1, "cand-001")
        c2, _ = self.repo.apply(proposal(rf("notes/b.md", "b")), c1, "cand-002")
        self.repo.accept(c2, "cand-002")
        self.assertEqual(self.refs()["refs/heads/main"], c2.commit)
        self.assertEqual(self.git("rev-list", "--count", "main"), "3", "linear history of accepted versions")

    def test_accept_refuses_a_non_fast_forward_and_a_retagged_commit(self):
        v0 = self.repo.init_v0()
        c1, _ = self.repo.apply(proposal(rf("notes/a.md", "a")), v0, "cand-001")
        c2, _ = self.repo.apply(proposal(rf("notes/b.md", "b")), v0, "cand-002")
        self.repo.accept(c1, "cand-001")
        with self.assertRaisesRegex(ContractError, "not a fast-forward"):
            self.repo.accept(c2, "cand-002")
        with self.assertRaisesRegex(ContractError, "points at"):
            self.repo.accept(c2, "cand-001")
        self.assertEqual(self.refs()["refs/heads/main"], c1.commit)

    def test_refs_only_ever_grow_and_nothing_is_deleted(self):
        v0 = self.repo.init_v0()
        seen = set(self.refs())
        c1, _ = self.repo.apply(proposal(rf("notes/a.md", "a")), v0, "cand-001")
        self.repo.apply(proposal(rf("notes/b.md", "b")), v0, "cand-002")
        self.repo.accept(c1, "cand-001")
        self.assertTrue(seen <= set(self.refs()))
        self.assertEqual({"refs/tags/pack/cand-001", "refs/tags/pack/cand-002"} <= set(self.refs()), True)

    def test_concurrent_applies_with_distinct_labels_all_succeed(self):
        v0 = self.repo.init_v0()
        errors = []

        def work(i):
            try:
                self.repo.apply(proposal(rf(f"notes/n{i}.md", str(i))), v0, f"cand-{i:03d}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(1, 6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(len([r for r in self.refs() if r.startswith("refs/tags/pack/cand-")]), 5)

    def test_size_caps_apply_to_a_checkout(self):
        small = packs.PackRepo(self.layout, lint_ranges_path=self.ranges_path, clock=clock, max_file_bytes=10)
        with self.assertRaisesRegex(ContractError, "larger than"):
            small.init_v0()

    def craft_commit(self, parent: PackRef, mutate) -> PackRef:
        """A commit made outside PackRepo (plumbing), as an attacker or a bug could: ``mutate(entries)`` edits {path: (mode, sha)}."""
        ls = self.git("ls-tree", "-r", parent.commit).splitlines()
        entries = {}
        for line in ls:
            meta, path = line.split("\t")
            mode, typ, sha = meta.split()
            entries[path] = (mode, sha)
        mutate(entries)
        lines = "".join(f"{m} blob {s}\t{p}\0" for p, (m, s) in entries.items())
        # a flat mktree only works for top-level names, so build the nested trees with the repo's own writer
        tree = self.repo._write_tree(entries)
        commit = self.git("commit-tree", tree, "-p", parent.commit, "-m", "crafted")
        return PackRef(parent.repo_path, commit)

    def test_a_symlink_or_submodule_in_a_tree_is_refused_by_checkout_and_lint(self):
        v0 = self.repo.init_v0()
        blob = self.git("hash-object", "-w", "--stdin", input="/etc/passwd")

        def link(entries):
            entries["notes/link"] = ("120000", blob)

        crafted = self.craft_commit(v0, link)
        with self.assertRaisesRegex(ContractError, "regular files"):
            self.repo.checkout(crafted, self.root / "x")
        res = self.repo.lint(crafted)
        self.assertFalse(res.passed)
        self.assertIn("regular files", res.messages[0])

    def test_a_protected_file_smuggled_into_a_commit_fails_lint(self):
        v0 = self.repo.init_v0()
        blob = self.git("hash-object", "-w", "--stdin", input="{}")
        crafted = self.craft_commit(v0, lambda e: e.__setitem__("seed/opencode.json", ("100644", blob)))
        res = self.repo.lint(crafted)
        self.assertFalse(res.passed)
        self.assertTrue(any("protected" in m for m in res.messages))


class LintTests(RepoCase):
    def lint_edit(self, *edits, label="cand-001"):
        v0 = self.repo.init_v0() if not (self.layout.pack_repo / "HEAD").exists() else self.repo.main_pack()
        cand, _ = self.repo.apply(proposal(*edits), v0, label)
        return self.repo.lint(cand), cand

    def test_v0_lints_clean(self):
        v0 = self.repo.init_v0()
        res = self.repo.lint(v0)
        self.assertTrue(res.passed, res.messages)
        self.assertIn("task.yaml", res.changed_paths)

    def test_a_text_edit_to_notes_and_recipe_passes_and_lists_changed_paths(self):
        res, _ = self.lint_edit(sr("notes/seed.md", ("## Log", "## Log (newest first)")), sr("seed/recipe/run.sh", ('SAMPLES="${SAMPLES:-4}"', 'SAMPLES="${SAMPLES:-6}"')))
        self.assertTrue(res.passed, res.messages)
        self.assertEqual(res.changed_paths, ("notes/seed.md", "seed/recipe/run.sh"))

    def test_editable_task_keys_pass(self):
        res, _ = self.lint_edit(sr("task.yaml", ("  count: 2\n", "  count: 3\n"), ("      every: 5\n", "      every: 3\n")))
        self.assertTrue(res.passed, res.messages)

    def test_a_changed_model_fails_lint_and_names_the_key(self):
        res, _ = self.lint_edit(sr("task.yaml", ('model: "muse/muse-spark-1.3"', 'model: "meta/muse-wonder-1.0"')))
        self.assertFalse(res.passed)
        self.assertTrue(any("agents.model" in m and "not editable" in m for m in res.messages), res.messages)

    def test_a_changed_grader_or_gateway_or_stop_fails_lint(self):
        for old, new, key in [
            ("  timeout: 120\n", "  timeout: 99999\n", "grader.timeout"),
            ("    port: 4310\n", "    port: 4311\n", "agents.gateway.port"),
            ("    max_real_attempts: 8\n", "    max_real_attempts: 80\n", "run.stop.max_real_attempts"),
            ("  research: false\n", "  research: true\n", "agents.research"),
        ]:
            with self.subTest(key):
                self.setUp()
                res, _ = self.lint_edit(sr("task.yaml", (old, new)))
                self.assertFalse(res.passed)
                self.assertTrue(any(key in m for m in res.messages), res.messages)

    def test_heartbeat_and_count_outside_their_ranges_fail(self):
        res, _ = self.lint_edit(sr("task.yaml", ("  count: 2\n", "  count: 12\n")))
        self.assertTrue(any("agents.count" in m for m in res.messages), res.messages)
        self.setUp()
        res, _ = self.lint_edit(sr("task.yaml", ("      every: 5\n", "      every: 500\n")))
        self.assertTrue(any("every" in m for m in res.messages), res.messages)

    def test_a_task_yaml_outside_the_strict_subset_fails_lint(self):
        res, _ = self.lint_edit(rf("task.yaml", "task: &a\n  name: x\n"))
        self.assertFalse(res.passed)
        self.assertTrue(any("task.yaml" in m for m in res.messages), res.messages)

    def test_a_missing_grader_value_against_entrypoint_json_fails(self):
        ep = self.root / "ENTRYPOINT.json"
        ep.write_text(json.dumps({"entrypoint": "other.mod:Grader", "setup": ["uv pip install -e ./grader"], "timeout": 120}))
        repo = packs.PackRepo(self.layout, lint_ranges_path=self.ranges_path, clock=clock, entrypoint_json=ep)
        v0 = repo.init_v0()
        res = repo.lint(v0)
        self.assertFalse(res.passed)
        self.assertTrue(any("grader.entrypoint" in m for m in res.messages), res.messages)

    def test_missing_referenced_files_fail_lint(self):
        res, _ = self.lint_edit(sr("task.yaml", ("agents:\n  count: 2\n", 'agents:\n  skills:\n    - "skills/nope"\n  count: 2\n')))
        self.assertTrue(any("does not exist in the pack" in m for m in res.messages), res.messages)
        self.setUp()
        res, _ = self.lint_edit(rf("skills/hygiene/SKILL.md", "# skill\n"), sr("task.yaml", ("agents:\n  count: 2\n", 'agents:\n  skills:\n    - "skills/hygiene"\n  count: 2\n')))
        self.assertTrue(res.passed, res.messages)

    def test_a_missing_ranges_file_is_an_error_not_a_pass(self):
        v0 = self.repo.init_v0()
        self.ranges_path.unlink()
        with self.assertRaisesRegex(ContractError, "lint ranges"):
            self.repo.lint(v0)

    def test_the_subprocess_dumper_agrees_with_the_in_process_one(self):
        v0 = self.repo.init_v0()
        task = self.root / "t.yaml"
        task.write_text(self.repo.read_files(v0)["task.yaml"], encoding="utf-8")
        a = packs.strict_dumper(task)
        b = packs.SubprocessDumper([sys.executable])(task)
        self.assertEqual(a["config"], b["config"])
        self.assertIs(b["coral_validated"], False)
        repo2 = packs.PackRepo(self.layout, lint_ranges_path=self.ranges_path, clock=clock, task_dumper=packs.SubprocessDumper([sys.executable]))
        self.assertTrue(repo2.lint(v0).passed)

    def test_the_subprocess_dumper_reports_a_failed_script(self):
        bad = self.root / "bad.yaml"
        bad.write_text("a: 1\na: 2\n")
        with self.assertRaisesRegex(ContractError, "duplicate key"):
            packs.SubprocessDumper([sys.executable])(bad)

    def test_the_note_that_coral_did_not_validate_is_advisory(self):
        v0 = self.repo.init_v0()
        res = self.repo.lint(v0)
        self.assertTrue(res.passed)
        self.assertTrue(all(m.startswith("note:") for m in res.messages))


def load_script(path: Path, name: str):
    """Execute a recipe script from source (no bytecode cache next to the template)."""
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    return mod


class RecipeTests(unittest.TestCase):
    """The starting recipe's scripts: syntax, and the pure logic that can run without a GPU."""

    recipe = V0_DIR / "seed" / "recipe"

    def test_run_sh_parses_and_reads_only_documented_names(self):
        if shutil.which("bash") is None:
            self.skipTest("bash is not installed")
        subprocess.run(["bash", "-n", str(self.recipe / "run.sh")], check=True)
        text = (self.recipe / "run.sh").read_text()
        for needed in ("glimmer-vllm-up", "glimmer-axolotl-train", "GLIMMER_CONFIG_TEMPLATE", "GLIMMER_OUT"):
            self.assertIn(needed, text)

    def test_python_scripts_compile(self):
        for name in ("generate.py", "build_sft.py"):
            compile((self.recipe / name).read_text(), name, "exec")

    def test_selection_keeps_the_shortest_passing_answers_per_prompt(self):
        mod = load_script(self.recipe / "build_sft.py", "recipe_build_sft")
        samples = [
            {"key": "a", "prompt": "pa", "ground_truth": "g", "response": "long answer ok", "reasoning": ""},
            {"key": "a", "prompt": "pa", "ground_truth": "g", "response": "ok", "reasoning": "think"},
            {"key": "a", "prompt": "pa", "ground_truth": "g", "response": "bad", "reasoning": ""},
            {"key": "b", "prompt": "pb", "ground_truth": "g", "response": "bad", "reasoning": ""},
        ]
        kept, stats = mod.select(samples, lambda s: "ok" in s["response"], 1)
        self.assertEqual([s["response"] for s in kept], ["ok"])
        self.assertEqual(stats, {"prompts": 2, "prompts_solved": 1, "kept": 1, "samples": 4})
        row = mod.to_chat(kept[0])
        self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
        self.assertEqual(row["messages"][2]["reasoning_content"], "think")
        self.assertNotIn("reasoning_content", mod.to_chat(samples[0])["messages"][2])

    def test_build_sft_main_uses_the_rewards_module_it_is_pointed_at(self):
        mod = load_script(self.recipe / "build_sft.py", "recipe_build_sft2")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "rewards.py").write_text(
                "def preflight(): pass\ndef error_counts(): return {}\ndef reward(response, ground_truth, mode='all'): return 1.0 if response.startswith('PASS') else 0.0\n"
            )
            rows = [
                {"key": "a", "prompt": "p", "ground_truth": "g", "response": "PASS one", "reasoning": ""},
                {"key": "b", "prompt": "q", "ground_truth": "g", "response": "FAIL", "reasoning": ""},
            ]
            (td / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = mod.main(["--samples", str(td / "s.jsonl"), "--out", str(td / "o.jsonl"), "--rewards", str(td / "rewards.py")])
            self.assertEqual(rc, 0)
            out = [json.loads(line) for line in (td / "o.jsonl").read_text().splitlines()]
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["messages"][2]["content"], "PASS one")

    def test_generate_is_deterministic_and_keeps_only_complete_answers(self):
        gen = load_script(self.recipe / "generate.py", "recipe_generate")
        rows = [{"key": f"k{i}", "prompt": f"p{i}", "ground_truth": "g"} for i in range(10)]
        self.assertEqual([r["key"] for r in gen.pick(rows, 4)], [r["key"] for r in gen.pick(list(reversed(rows)), 4)])
        seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                body = json.dumps({"data": [{"id": "m"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                n = int(self.headers["Content-Length"])
                req = json.loads(self.rfile.read(n))
                seen.append(req)
                cut = req["messages"][1]["content"] == "p1"
                body = json.dumps({"choices": [{"message": {"content": "answer", "reasoning_content": "why"}, "finish_reason": "length" if cut else "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "train.jsonl"
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            out = Path(td) / "samples.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gen.main(["--base-url", base, "--wait-only", "--wait-seconds", "5"]), 0)
                rc = gen.main(["--base-url", base, "--data", str(data), "--out", str(out), "--n-prompts", "10", "--samples", "2", "--workers", "4"])
            self.assertEqual(rc, 0)
            got = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(got), 18, "10 prompts x 2 samples, minus the 2 answers of p1 that were cut off")
        self.assertTrue(all(r["finish_reason"] == "stop" and r["reasoning"] == "why" for r in got))
        self.assertTrue(all(q["messages"][0] == {"role": "system", "content": "You are a helpful AI assistant."} for q in seen))
        self.assertTrue(all(q["temperature"] == 1.0 and q["top_k"] == 64 for q in seen))


if __name__ == "__main__":
    unittest.main()
