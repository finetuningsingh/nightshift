"""Verifier adapters (owner 1). The third-party packages are not in the repository: the adapter logic is tested against stand-in
checkouts written by the test (same file layout and API as upstream, hashes passed through ``expected_sha256``), and against the real
packages only when ``LAB_THIRD_PARTY`` points at ``<dir>/IFBench`` and ``<dir>/open-instruct`` with NLTK data available."""

import hashlib
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from nightshift.lab import rewards, verifiers
from nightshift.lab.contracts import ScorerError, Verifier
from nightshift.lab.verifiers import IfbenchUpstream, IfEvalG, PromptResult, aggregate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result(key, verifier, ids, strict, loose=None, error=False):
    return PromptResult(key, verifier, tuple(ids), tuple(strict), None if loose is None else tuple(loose), error, int(error))


class AggregateTests(unittest.TestCase):
    def test_ifevalg_metrics_hand_computed(self):
        rs = [
            result("a", Verifier.IFEVALG, ["x", "y"], [True, True]),
            result("b", Verifier.IFEVALG, ["x", "y"], [True, False]),
            result("c", Verifier.IFEVALG, ["x", "y", "z", "w"], [False, False, False, False]),
            result("d", Verifier.IFEVALG, ["x"], [True]),
        ]
        m = aggregate(rs)
        self.assertEqual(set(m), {"prompt_all", "prompt_fraction"})
        self.assertAlmostEqual(m["prompt_all"], 2 / 4)  # a and d pass every constraint
        self.assertAlmostEqual(m["prompt_fraction"], (1.0 + 0.5 + 0.0 + 1.0) / 4)

    def test_ifbench_metrics_hand_computed(self):
        rs = [
            result("a", Verifier.IFBENCH_UPSTREAM, ["x", "y"], [True, False], [True, True]),
            result("b", Verifier.IFBENCH_UPSTREAM, ["x"], [True], [True]),
            result("c", Verifier.IFBENCH_UPSTREAM, ["x", "y"], [False, False], [False, True]),
        ]
        m = aggregate(rs)
        self.assertEqual(set(m), {"prompt_strict", "prompt_loose", "instruction_strict", "instruction_loose"})
        self.assertAlmostEqual(m["prompt_strict"], 1 / 3)
        self.assertAlmostEqual(m["prompt_loose"], 2 / 3)
        self.assertAlmostEqual(m["instruction_strict"], 2 / 5)
        self.assertAlmostEqual(m["instruction_loose"], 4 / 5)

    def test_strict_only_ifbench_has_no_loose_metrics(self):
        m = aggregate([result("a", Verifier.IFBENCH_UPSTREAM, ["x"], [True])])
        self.assertEqual(set(m), {"prompt_strict", "instruction_strict"})

    def test_refuses_empty_and_mixed(self):
        with self.assertRaises(ScorerError):
            aggregate([])
        with self.assertRaises(ScorerError):
            aggregate([result("a", Verifier.IFEVALG, ["x"], [True]), result("b", Verifier.IFBENCH_UPSTREAM, ["x"], [True], [True])])

    def test_prompt_result_validates_shape(self):
        with self.assertRaises(ScorerError):
            result("a", Verifier.IFEVALG, [], [])
        with self.assertRaises(ScorerError):
            result("a", Verifier.IFEVALG, ["x", "y"], [True])
        with self.assertRaises(ScorerError):
            result("a", Verifier.IFBENCH_UPSTREAM, ["x"], [True], [True, False])
        with self.assertRaises(ScorerError):
            result("a", Verifier.IFEVALG, ["x"], [True]).passed_loose


# ---------------------------------------------------------------------------------------------------------------------
# a stand-in IFEvalG directory: the three file names, the same import line, the same API
# ---------------------------------------------------------------------------------------------------------------------
STAND_IN_IFEVALG = {
    "instructions_util.py": "def count_words(text):\n    return len(text.split())\n",
    "instructions.py": textwrap.dedent('''
        from open_instruct.IFEvalG import instructions_util

        class Base:
            def __init__(self, key):
                self.key = key
            def build_description(self, **kw):
                self.kw = kw
                return self.key

        class MinWords(Base):
            def check_following(self, value):
                return instructions_util.count_words(value) >= self.kw["n"]

        class NoComma(Base):
            def check_following(self, value):
                return "," not in value

        class Explodes(Base):
            def check_following(self, value):
                raise RuntimeError("boom")

        class NeedsKw(Base):
            def build_description(self, *, n):
                self.n = n
            def check_following(self, value):
                return len(value) > self.n
    '''),
    "instructions_registry.py": textwrap.dedent('''
        from open_instruct.IFEvalG import instructions

        INSTRUCTION_DICT = {
            "t:min_words": instructions.MinWords,
            "t:no_comma": instructions.NoComma,
            "t:explodes": instructions.Explodes,
            "t:needs_n": instructions.NeedsKw,
        }
    '''),
}


def write_stand_in_ifevalg(root: Path) -> dict:
    d = root / "open_instruct" / "IFEvalG"
    d.mkdir(parents=True)
    for name, text in STAND_IN_IFEVALG.items():
        (d / name).write_text(text)
    return {name: sha(d / name) for name in STAND_IN_IFEVALG}


def gt(ids, kwargs):
    return repr([{"instruction_id": ids, "kwargs": kwargs}])


class IfEvalGAdapterTests(unittest.TestCase):
    def setUp(self):
        rewards._reset_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.hashes = write_stand_in_ifevalg(self.root)
        self.addCleanup(rewards._reset_for_tests)
        self.addCleanup(self.tmp.cleanup)

    def backend(self):
        return IfEvalG(self.root, expected_sha256=self.hashes)

    def test_scores_constraints_and_counts_checker_errors(self):
        b = self.backend()
        rows = [
            {"key": "r1", "prompt": "p1", "ground_truth": gt(["t:min_words", "t:no_comma"], [{"n": 3}, None]), "instruction_ids": ["t:min_words", "t:no_comma"]},
            {"key": "r2", "prompt": "p2", "ground_truth": gt(["t:min_words", "t:explodes"], [{"n": 1}, None]), "instruction_ids": ["t:min_words", "t:explodes"]},
            {"key": "r3", "prompt": "p3", "ground_truth": gt(["t:min_words", "t:no_comma"], [{"n": 3}, None]), "instruction_ids": ["t:min_words", "t:no_comma"]},
        ]
        out = b.score(rows, ["one two three", "hello there", "one, two, three"])
        self.assertEqual([r.strict for r in out], [(True, True), (True, False), (True, False)])
        self.assertEqual([r.error for r in out], [False, True, False])
        self.assertEqual(out[1].n_checker_errors, 1)
        self.assertIsNone(out[0].loose)
        self.assertEqual(out[0].verifier, Verifier.IFEVALG)
        self.assertAlmostEqual(aggregate(out)["prompt_all"], 1 / 3)

    def test_none_kwargs_are_dropped_and_unknown_ids_are_errors_not_crashes(self):
        b = self.backend()
        rows = [
            {"key": "r1", "prompt": "p", "ground_truth": gt(["t:needs_n", "t:nope"], [{"n": 2, "other": None}, None])},
            {"key": "r2", "prompt": "p", "ground_truth": gt(["t:needs_n"], [{"n": None}])},  # None dropped -> missing keyword -> counted error
        ]
        out = b.score(rows, ["abcdef", "abcdef"])
        self.assertEqual(out[0].strict, (True, False))
        self.assertEqual(out[0].n_checker_errors, 1)
        self.assertEqual(out[1].strict, (False,))
        self.assertTrue(out[1].error)

    def test_empty_response_and_thinking_section(self):
        b = self.backend()
        row = {"key": "r1", "prompt": "p", "ground_truth": gt(["t:min_words"], [{"n": 2}])}
        out = b.score([row, row, row, row], ["", "<think>lots of words in here</think>", "<think>x</think>two words", "  \n"])
        self.assertEqual([r.strict for r in out], [(False,), (False,), (True,), (False,)])
        self.assertFalse(any(r.error for r in out))

    def test_missing_responses_are_refused_not_scored_as_failures(self):
        b = self.backend()
        row = {"key": "r1", "prompt": "p", "ground_truth": gt(["t:no_comma"], [None])}
        with self.assertRaises(ScorerError):
            b.score([row, row], ["fine"])
        with self.assertRaises(ScorerError):
            b.score([row], [None])
        with self.assertRaises(ScorerError):
            b.score([row], ["fine"], mode="loose")
        with self.assertRaises(ScorerError):
            b.score([{"key": "r1", "prompt": "p"}], ["fine"])

    def test_declared_ids_must_match_ground_truth(self):
        b = self.backend()
        row = {"key": "r1", "prompt": "p", "ground_truth": gt(["t:no_comma"], [None]), "instruction_ids": ["t:min_words"]}
        with self.assertRaises(ScorerError):
            b.score([row], ["fine"])

    def test_pins_hold_the_three_file_hashes_and_the_commit(self):
        pins = self.backend().pins()
        self.assertEqual(pins["open_instruct_commit"], rewards.OPEN_INSTRUCT_COMMIT)
        for name in STAND_IN_IFEVALG:
            self.assertEqual(pins["ifevalg_" + name.replace(".", "_")], self.hashes[name])
        self.assertTrue(all(isinstance(v, str) and v for v in pins.values()))

    def test_a_modified_file_is_refused_before_import(self):
        d = self.root / "open_instruct" / "IFEvalG"
        (d / "instructions.py").write_text("raise SystemExit('executed')\n")
        with self.assertRaises(ScorerError):
            self.backend()
        self.assertNotIn("open_instruct.IFEvalG.instructions", sys.modules)

    def test_modified_after_construction_is_still_refused_at_import(self):
        b = self.backend()
        d = self.root / "open_instruct" / "IFEvalG"
        (d / "instructions_registry.py").write_text("raise SystemExit('executed')\n")
        with self.assertRaises(ScorerError):
            b.score([{"key": "r", "prompt": "p", "ground_truth": gt(["t:no_comma"], [None])}], ["x"])

    def test_missing_file_symlink_and_wrong_pins_are_refused(self):
        d = self.root / "open_instruct" / "IFEvalG"
        with self.assertRaises(ScorerError):
            IfEvalG(self.root, expected_sha256={**self.hashes, "instructions.py": "0" * 64})
        real = d / "instructions_util.py"
        moved = self.root / "elsewhere.py"
        real.rename(moved)
        with self.assertRaises(ScorerError):
            self.backend()  # missing
        real.symlink_to(moved)
        with self.assertRaises(ScorerError):
            self.backend()  # a symbolic link is not a pinned file
        with self.assertRaises(ScorerError):
            IfEvalG(self.root / "nope")

    def test_default_pins_are_the_real_open_instruct_pins(self):
        with self.assertRaises(ScorerError):  # the stand-in is not the pinned upstream
            IfEvalG(self.root)

    def test_flat_directory_layout_as_on_a_rental(self):
        flat = self.root / "flat"
        flat.mkdir()
        for name, text in STAND_IN_IFEVALG.items():
            (flat / name).write_text(text)
        b = IfEvalG(flat, expected_sha256=self.hashes)
        out = b.score([{"key": "r", "prompt": "p", "ground_truth": gt(["t:no_comma"], [None])}], ["ok"])
        self.assertEqual(out[0].strict, (True,))

    def test_only_one_ifevalg_directory_per_process(self):
        self.backend().score([{"key": "r", "prompt": "p", "ground_truth": gt(["t:no_comma"], [None])}], ["ok"])
        other = self.root / "other"
        other.mkdir()
        hashes = write_stand_in_ifevalg(other)
        with self.assertRaises(ScorerError):
            IfEvalG(other, expected_sha256=hashes).score([{"key": "r", "prompt": "p", "ground_truth": gt(["t:no_comma"], [None])}], ["ok"])

    def test_pinned_constants_agree_with_lab_pins_json(self):
        pins = json.loads((Path(__file__).resolve().parents[1] / "scripts" / "lab_pins.json").read_text())
        self.assertEqual(pins["open_instruct"]["commit"], rewards.OPEN_INSTRUCT_COMMIT)
        self.assertEqual(pins["open_instruct"]["ifevalg_sha256"], rewards.IFEVALG_SHA256)
        self.assertEqual(pins["ifbench"]["commit"], verifiers.IFBENCH_COMMIT)
        self.assertEqual(pins["ifbench"]["file_sha256"], verifiers.IFBENCH_FILE_SHA256)


# ---------------------------------------------------------------------------------------------------------------------
# a stand-in IFBench checkout: it reproduces the two upstream hazards the adapter exists to handle
# ---------------------------------------------------------------------------------------------------------------------
STAND_IN_IFBENCH = {
    "ifbench/__init__.py": "from ifbench import instructions_registry  # noqa\n",
    "ifbench/instructions_registry.py": textwrap.dedent('''
        class Contains:
            def __init__(self, key):
                self.key = key
                self.kw = {}
            def build_description(self, **kw):
                if any(v is None for v in kw.values()):
                    raise TypeError("unexpected None keyword")  # loose upstream does not filter None
                self.kw = kw
            def get_instruction_args(self):
                return None
            def check_following(self, value):
                return self.kw.get("word", "x") in value

        class Boom(Contains):
            def check_following(self, value):
                raise RuntimeError("checker bug")

        INSTRUCTION_DICT = {"t:contains": Contains, "t:boom": Boom}
    '''),
    "evaluation_lib.py": textwrap.dedent('''
        import dataclasses
        from ifbench import instructions_registry

        @dataclasses.dataclass
        class InputExample:
            key: int
            instruction_id_list: list
            prompt: str
            kwargs: list

        @dataclasses.dataclass
        class OutputExample:
            instruction_id_list: list
            prompt: str
            response: str
            follow_all_instructions: bool
            follow_instruction_list: list

        def get_response_for_prompt(inp, p2r):
            return p2r.get(inp.prompt, p2r.get(inp.prompt.strip()))

        def _empty(inp):
            return OutputExample(inp.instruction_id_list, inp.prompt, "", False, [False] * len(inp.instruction_id_list))

        def test_instruction_following_strict(inp, prompt_to_response):
            response = get_response_for_prompt(inp, prompt_to_response)
            if response is None:
                return _empty(inp)
            flags = []
            for i, iid in enumerate(inp.instruction_id_list):
                inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
                inp.kwargs[i] = {k: v for k, v in inp.kwargs[i].items() if v is not None}  # in place, like upstream strict
                inst.build_description(**inp.kwargs[i])
                flags.append(bool(response and response.strip() and inst.check_following(response)))
            return OutputExample(inp.instruction_id_list, inp.prompt, response, all(flags), flags)

        def test_instruction_following_loose(inp, prompt_to_response):
            response = get_response_for_prompt(inp, prompt_to_response)
            if response is None:
                return _empty(inp)
            variants = [response, response.replace("*", "")]
            flags = []
            for i, iid in enumerate(inp.instruction_id_list):
                inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
                inst.build_description(**inp.kwargs[i])  # no None filtering: raises TypeError when strict has not run first
                flags.append(any(v.strip() and inst.check_following(v) for v in variants))
            return OutputExample(inp.instruction_id_list, inp.prompt, response, all(flags), flags)
    '''),
}


def write_stand_in_ifbench(root: Path) -> dict:
    for rel, text in STAND_IN_IFBENCH.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return {rel: sha(root / rel) for rel in STAND_IN_IFBENCH}


def ib_row(key, prompt, ids, kwargs):
    return {"key": key, "prompt": prompt, "instruction_id_list": ids, "kwargs": kwargs}


class IfbenchAdapterTests(unittest.TestCase):
    def setUp(self):
        verifiers._reset_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "IFBench"
        self.root.mkdir()
        self.hashes = write_stand_in_ifbench(self.root)
        self.addCleanup(verifiers._reset_for_tests)
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(verifiers, "check_nltk_data", lambda *a, **k: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def backend(self, **kw):
        return IfbenchUpstream(self.root, expected_sha256=self.hashes, expected_commit=None, **kw)

    def test_none_kwargs_are_filtered_so_loose_works_on_its_own_objects(self):
        b = self.backend()
        rows = [ib_row("0", "say apple", ["t:contains"], [{"word": "apple", "other": None}])]
        out = b.score(rows, ["I like apple"])
        self.assertEqual(out[0].strict, (True,))
        self.assertEqual(out[0].loose, (True,))
        self.assertFalse(out[0].error)
        # the hazard is real in the stand-in: calling upstream loose directly on the raw kwargs raises
        import evaluation_lib as ev  # imported by the adapter
        inp = ev.InputExample("0", ["t:contains"], "p", [{"word": "apple", "other": None}])
        with self.assertRaises(TypeError):
            ev.test_instruction_following_loose(inp, {"p": "apple"})

    def test_loose_can_be_more_lenient_than_strict(self):
        b = self.backend()
        row = ib_row("0", "p", ["t:contains"], [{"word": "apple"}])
        out = b.score([row], ["ap*ple"])  # the stand-in loose variant strips asterisks; strict does not
        self.assertEqual((out[0].strict, out[0].loose), ((False,), (True,)))
        self.assertFalse(out[0].passed_strict)
        self.assertTrue(out[0].passed_loose)
        out = b.score([row], ["*apple*"])  # both accept it
        self.assertEqual((out[0].strict, out[0].loose), ((True,), (True,)))
        out = b.score([ib_row("0", "p", ["t:contains"], [{"word": "ap*ple"}])], ["ap*ple"])
        self.assertEqual((out[0].strict, out[0].loose), ((True,), (True,)))

    def test_identical_prompts_do_not_collide(self):
        b = self.backend()
        rows = [ib_row("a", "same prompt", ["t:contains"], [{"word": "red"}]), ib_row("b", "same prompt", ["t:contains"], [{"word": "red"}])]
        out = b.score(rows, ["red", "blue"])
        self.assertEqual([r.strict for r in out], [(True,), (False,)])
        self.assertEqual([r.key for r in out], ["a", "b"])

    def test_a_checker_exception_is_counted_and_fails_the_prompt(self):
        b = self.backend()
        rows = [ib_row("0", "p", ["t:contains", "t:boom"], [{"word": "a"}, {}]), ib_row("1", "q", ["t:contains"], [{"word": "a"}])]
        out = b.score(rows, ["a", "a"])
        self.assertTrue(out[0].error)
        self.assertEqual(out[0].n_checker_errors, 1)
        self.assertEqual(out[0].strict, (False, False))
        self.assertEqual(out[0].loose, (False, False))
        self.assertFalse(out[1].error)
        self.assertEqual(out[1].strict, (True,))

    def test_empty_response_fails_every_constraint_without_an_error(self):
        out = self.backend().score([ib_row("0", "p", ["t:contains"], [{"word": "a"}])], [""])
        self.assertEqual((out[0].strict, out[0].loose, out[0].error), ((False,), (False,), False))

    def test_strict_mode_skips_loose(self):
        out = self.backend().score([ib_row("0", "p", ["t:contains"], [{"word": "a"}])], ["a"], mode="strict")
        self.assertIsNone(out[0].loose)
        self.assertEqual(out[0].strict, (True,))

    def test_thinking_is_stripped_by_default_and_the_flag_is_recorded(self):
        row = ib_row("0", "p", ["t:contains"], [{"word": "secret"}])
        text = "<think>secret plan</think>final answer"
        self.assertEqual(self.backend().score([row], [text])[0].strict, (False,))
        self.assertEqual(self.backend(strip_thinking=False).score([row], [text])[0].strict, (True,))
        self.assertEqual(self.backend().pins()["strip_thinking"], "true")
        self.assertEqual(self.backend(strip_thinking=False).pins()["strip_thinking"], "false")

    def test_bad_rows_and_missing_responses_are_refused(self):
        b = self.backend()
        row = ib_row("0", "p", ["t:contains"], [{"word": "a"}])
        with self.assertRaises(ScorerError):
            b.score([row, row], ["a"])
        with self.assertRaises(ScorerError):
            b.score([row], [None])
        with self.assertRaises(ScorerError):
            b.score([{"key": "0", "prompt": "p"}], ["a"])
        with self.assertRaises(ScorerError):
            b.score([ib_row("0", "p", ["t:contains"], [])], ["a"])
        with self.assertRaises(ScorerError):
            b.score([ib_row("0", "p", [], [])], ["a"])

    def test_modified_file_missing_checkout_and_wrong_commit_are_refused(self):
        (self.root / "ifbench" / "instructions_registry.py").write_text("raise SystemExit('executed')\n")
        with self.assertRaises(ScorerError):
            self.backend()
        self.assertNotIn("ifbench", sys.modules)
        with self.assertRaises(ScorerError):
            IfbenchUpstream(self.root.parent / "absent", expected_sha256=self.hashes)

    def test_commit_in_git_head_must_equal_the_pin(self):
        git = self.root / ".git"
        git.mkdir()
        (git / "HEAD").write_text("a" * 40 + "\n")
        with self.assertRaises(ScorerError):
            IfbenchUpstream(self.root, expected_sha256=self.hashes, expected_commit="b" * 40)
        b = IfbenchUpstream(self.root, expected_sha256=self.hashes, expected_commit="a" * 40)
        self.assertEqual(b.pins()["ifbench_commit"], "a" * 40)

    def test_pins_cover_every_executed_file(self):
        pins = self.backend().pins()
        self.assertEqual(pins["ifbench_evaluation_lib_py"], self.hashes["evaluation_lib.py"])
        self.assertEqual(pins["ifbench_ifbench_instructions_registry_py"], self.hashes["ifbench/instructions_registry.py"])

    def test_one_checkout_per_process(self):
        row = ib_row("0", "p", ["t:contains"], [{"word": "a"}])
        self.backend().score([row], ["a"])
        other = Path(self.tmp.name) / "IFBench2"
        other.mkdir()
        hashes = write_stand_in_ifbench(other)
        with self.assertRaises(ScorerError):
            IfbenchUpstream(other, expected_sha256=hashes, expected_commit=None).score([row], ["a"])

    def test_modules_already_imported_from_elsewhere_are_refused_not_reused(self):
        import types
        foreign = Path(self.tmp.name) / "elsewhere"
        foreign.mkdir()
        fake = types.ModuleType("evaluation_lib")
        fake.__file__ = str(foreign / "evaluation_lib.py")
        self.addCleanup(sys.modules.pop, "evaluation_lib", None)
        sys.modules["evaluation_lib"] = fake
        b = self.backend()
        with self.assertRaises(ScorerError) as cm:
            b.preflight()
        self.assertIn("evaluation_lib", str(cm.exception))
        self.assertIs(sys.modules["evaluation_lib"], fake)  # left alone: it is not ours to drop
        self.assertEqual(verifiers._IFBENCH_LOADED, {})
        row = ib_row("0", "p", ["t:contains"], [{"word": "a"}])
        with self.assertRaises(ScorerError):
            b.score([row], ["a"])
        # a module that really comes from this checkout is fine
        verifiers._reset_for_tests()
        self.assertTrue(self.backend().score([row], ["a"])[0].strict[0])
        self.assertTrue(Path(sys.modules["evaluation_lib"].__file__).resolve().is_relative_to(self.root.resolve()))

    def test_missing_nltk_data_is_a_scorer_error(self):
        with mock.patch.object(verifiers, "check_nltk_data", side_effect=ScorerError("NLTK resource punkt is missing")):
            with self.assertRaises(ScorerError):
                self.backend().preflight()
        self.assertNotIn("ifbench", sys.modules)  # nothing was imported before the data check


class PreflightTests(unittest.TestCase):
    def test_check_nltk_data_is_loud_when_a_resource_is_missing(self):
        try:
            import nltk  # noqa: F401
        except ImportError:
            with self.assertRaises(ScorerError):
                verifiers.check_nltk_data()
            return
        with self.assertRaises(ScorerError) as cm:
            verifiers.check_nltk_data(("tokenizers/definitely_not_a_real_resource",))
        self.assertIn("definitely_not_a_real_resource", str(cm.exception))

    def test_module_preflight_calls_each_backend_that_has_one(self):
        class A:
            called = 0
            def preflight(self):
                A.called += 1
        verifiers.preflight(A(), object(), A())
        self.assertEqual(A.called, 2)


class RewardsTests(unittest.TestCase):
    """rewards.py is staged to the rental on its own: stdlib only, no nightshift. Logic is tested on the stand-in IFEvalG directory."""

    def setUp(self):
        rewards._reset_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.hashes = write_stand_in_ifevalg(self.root)
        self.checker = rewards.Checker(self.root / "open_instruct" / "IFEvalG", expected=self.hashes)
        saved = rewards._DEFAULT
        rewards._DEFAULT = self.checker
        def restore():
            rewards._DEFAULT = saved
            rewards._reset_for_tests()
            self.tmp.cleanup()
        self.addCleanup(restore)

    GT = gt(["t:min_words", "t:no_comma"], [{"n": 3}, None])

    def test_modes(self):
        good, half, bad = "one two three", "one, two, three", ""
        self.assertEqual([rewards.reward(t, self.GT) for t in (good, half, bad)], [1.0, 0.5, 0.0])
        self.assertEqual([rewards.reward(t, self.GT, "all") for t in (good, half, bad)], [1.0, 0.0, 0.0])
        self.assertEqual([rewards.reward(t, self.GT, "hybrid") for t in (good, half, bad)], [1.0, 0.25, 0.0])
        with self.assertRaises(ValueError):
            rewards.reward(good, self.GT, "nope")

    def test_batch_and_verify_constraints(self):
        self.assertEqual(rewards.reward_batch(["one two three", "x"], [self.GT, self.GT]), [1.0, 0.5])
        self.assertEqual(rewards.verify_constraints("one two three", self.GT), [True, True])
        with self.assertRaises(ValueError):
            rewards.reward_batch(["a"], [self.GT, self.GT])

    def test_checker_exceptions_score_false_and_are_counted(self):
        g = gt(["t:explodes", "t:nope", "t:no_comma"], [None, None, None])
        self.assertEqual(rewards.verify_constraints("fine", g), [False, False, True])
        self.assertEqual(rewards.error_counts(), {"t:explodes": 1, "t:nope": 1})
        rewards.verify_constraints("fine", g)
        self.assertEqual(rewards.error_counts()["t:explodes"], 2)

    def test_empty_and_thinking_only_responses_score_zero_without_errors(self):
        for text in ("", "   ", "<think>only thoughts</think>", "<think>a</think>  "):
            self.assertEqual(rewards.reward(text, self.GT), 0.0, text)
        self.assertEqual(rewards.error_counts(), {})

    def test_remove_thinking_section_matches_open_instruct(self):
        f = rewards.remove_thinking_section
        self.assertEqual(f("<|assistant|> hi "), "hi")
        self.assertEqual(f("a</think>b</think> c "), "c")
        self.assertEqual(f("<answer>42</answer>"), "42")

    def test_ground_truth_parsing(self):
        ids = ["t:no_comma", "t:min_words"]
        literal = repr([{"instruction_id": ids, "kwargs": [None, {"n": 2}]}])
        as_json = json.dumps([{"instruction_id": ids, "kwargs": [None, {"n": 2}]}])
        nested = repr([json.dumps({"instruction_id": ids, "kwargs": [None, {"n": 2}]})])
        for form in (literal, as_json, nested):
            self.assertEqual(rewards.parse_ground_truth(form), (ids, [None, {"n": 2}]))
        self.assertEqual(rewards.parse_ground_truth([{"instruction_id": ids, "kwargs": [None, None]}])[0], ids)
        for bad in ("not a literal", "[]", repr([{"instruction_id": ["a"], "kwargs": []}]), repr([{"nope": 1}]), 5):
            with self.assertRaises(rewards.RewardsError, msg=str(bad)):
                rewards.parse_ground_truth(bad)

    def test_degenerate_floor(self):
        g1, g2 = gt(["t:no_comma"], [None]), gt(["t:min_words"], [{"n": 5}])
        self.assertEqual(rewards.degenerate_floor([g1, g2], ("Yes.", "ok")), {"Yes.": 0.5, "ok": 0.5})
        with self.assertRaises(ValueError):
            rewards.degenerate_floor([])

    def test_pins_and_required_files_agree(self):
        self.assertEqual(set(rewards.REQUIRED_FILES), set(rewards.IFEVALG_SHA256))
        self.assertEqual(rewards.REQUIRED_FILES, ("instructions.py", "instructions_registry.py", "instructions_util.py"))

    def test_preflight_is_loud_without_nltk_data(self):
        try:
            import nltk  # noqa: F401
        except ImportError:
            with self.assertRaises(rewards.RewardsError):
                rewards.preflight(self.checker)
            return
        with mock.patch.object(nltk.data, "find", side_effect=LookupError("missing")):
            with self.assertRaises(rewards.RewardsError):
                rewards.preflight(self.checker)

    def test_module_is_standalone_stdlib_only(self):
        import ast
        tree = ast.parse((Path(rewards.__file__)).read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods.add((node.module or "").split(".")[0])
        optional = {"langdetect", "nltk"}  # imported lazily inside functions, tolerated when absent
        self.assertNotIn("nightshift", mods)
        self.assertEqual(sorted(m for m in mods if m not in sys.stdlib_module_names and m not in optional), [])

    def test_imports_and_fails_cleanly_on_a_rental_without_nightshift(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "rewards.py").write_text(Path(rewards.__file__).read_text())
            code = ("import sys; sys.path.insert(0, %r); import rewards; "
                    "print(rewards.REQUIRED_FILES[0]); "
                    "import os; os.environ['IFEVALG_PATH'] = %r\n"
                    "try:\n    rewards.reward('x', 'y')\nexcept rewards.RewardsError as e:\n    print('refused')") % (d, str(self.root))
            out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=d)
            self.assertEqual(out.stdout.split(), ["instructions.py", "refused"], out.stderr)


@unittest.skipUnless(__import__("shutil").which("bash"), "bash not available")
class SetupScriptTests(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lab_setup_verifiers.sh"

    def run_script(self, *args):
        import subprocess
        return subprocess.run(["bash", str(self.SCRIPT), *args], capture_output=True, text=True)

    def test_syntax(self):
        import subprocess
        self.assertEqual(subprocess.run(["bash", "-n", str(self.SCRIPT)], capture_output=True).returncode, 0)

    def test_usage_and_unknown_option(self):
        self.assertEqual(self.run_script().returncode, 2)
        self.assertEqual(self.run_script("--nope", "x").returncode, 2)

    def test_verify_only_refuses_files_that_do_not_match_the_pins(self):
        with tempfile.TemporaryDirectory() as d:
            write_stand_in_ifbench(Path(d) / "IFBench")
            write_stand_in_ifevalg(Path(d) / "open-instruct")
            r = self.run_script("--verify-only", d)
            self.assertEqual(r.returncode, 1)
            self.assertIn("pins NOT satisfied", r.stderr)
            self.assertIn("evaluation_lib.py", r.stderr)

    def test_verify_only_on_an_empty_directory_fails_and_executes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_script("--verify-only", d).returncode, 1)


# ---------------------------------------------------------------------------------------------------------------------
# real packages: lead-run acceptance check, opt-in
# ---------------------------------------------------------------------------------------------------------------------
THIRD_PARTY = os.environ.get("LAB_THIRD_PARTY")


@unittest.skipUnless(THIRD_PARTY, "set LAB_THIRD_PARTY to run the verifier parity tests")
class RealVerifierTests(unittest.TestCase):
    """Needs ``$LAB_THIRD_PARTY/IFBench`` and ``$LAB_THIRD_PARTY/open-instruct`` at the pinned revisions, the third-party
    dependencies (absl-py, langdetect, nltk, immutabledict) and the NLTK data. Rows are synthetic, written here."""

    def setUp(self):
        verifiers._reset_for_tests()
        rewards._reset_for_tests()
        self.addCleanup(verifiers._reset_for_tests)
        self.addCleanup(rewards._reset_for_tests)
        self.nltk = os.environ.get("NLTK_DATA")

    def test_ifevalg_real_checkers_on_synthetic_rows(self):
        b = IfEvalG(Path(THIRD_PARTY) / "open-instruct", nltk_data=self.nltk)
        b.preflight()
        rows = [
            {"key": "r1", "prompt": "p", "ground_truth": gt(["punctuation:no_comma", "change_case:english_lowercase"], [None, None]),
             "instruction_ids": ["punctuation:no_comma", "change_case:english_lowercase"]},
            {"key": "r2", "prompt": "p", "ground_truth": gt(["length_constraints:number_words"], [{"num_words": 5, "relation": "at least"}]),
             "instruction_ids": ["length_constraints:number_words"]},
        ]
        out = b.score(rows, ["all lowercase and no commas here", "Too short"])
        self.assertEqual(out[0].strict, (True, True))
        self.assertEqual(out[1].strict, (False,))
        self.assertFalse(any(r.error for r in out))

    def test_ifbench_real_upstream_on_synthetic_rows(self):
        b = IfbenchUpstream(Path(THIRD_PARTY) / "IFBench", nltk_data=self.nltk)
        b.preflight()
        rows = [ib_row("0", "Write a note. Do not use any commas.", ["punctuation:no_comma"], [{}])]
        out = b.score(rows, ["A note without commas."])
        self.assertEqual((out[0].strict, out[0].loose, out[0].error), ((True,), (True,), False))
        out = b.score(rows, ["A note, with a comma."])
        self.assertEqual(out[0].strict, (False,))


if __name__ == "__main__":
    unittest.main()
