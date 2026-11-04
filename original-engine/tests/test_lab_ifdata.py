"""Split builder, salt and manifest (owner 1). Fixtures are synthetic rows written for the tests, never rows of the IFBench test set."""

import hashlib
import json
import os
import random
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from nightshift.lab import ifdata
from nightshift.lab.contracts import (
    TEST_ASSIGNMENT_ID,
    AssignmentManifest,
    AssignmentRole,
    ContractError,
    ItemMode,
    LabLayout,
    MixedRowPolicy,
    ScorerError,
)
from nightshift.lab.ifdata import AssignmentPlan, base_key, build_assignment_manifest, build_splits, ensure_split_salt, load_rows, read_split_salt

FIX = Path(__file__).resolve().parent / "fixtures" / "lab" / "ifbench_mini"
ROWS = FIX / "rows.jsonl"
TESTS = [FIX / "ifbench_test_mini.jsonl", FIX / "ifbench_test_legacy_mini.jsonl"]
EVO_FAMS = ("ifeval25", "counting", "format_new")
HELD_FAMS = ("copy", "firstlast")
LEAK_SENTENCES = ["cartographer unrolled the faded survey", "blue door at the end of the lane", "single green lantern swung above the quay"]
SALT = bytes(range(32))
PLANS = (
    AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, EVO_FAMS, n_selection=8, n_dev=10, n_train=0),
    AssignmentPlan("a-held-1", AssignmentRole.HELDOUT, HELD_FAMS, n_selection=4, n_dev=6, n_train=0),
)


def build(out: Path, *, salt=SALT, rows=ROWS, plans=PLANS, **kw) -> Path:
    return build_splits(rows, TESTS, out, salt=salt, evolution_families=EVO_FAMS + ("keywords_new",), heldout_families=HELD_FAMS, plans=plans, **kw)


def read_split(layout: LabLayout, aid: str, name: str) -> list:
    path = {"train": layout.train_data, "dev": layout.dev_data}.get(name, None)
    p = path(aid) if path else layout.selection_data(aid)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


class LoadRowsTests(unittest.TestCase):
    def test_fixture_loads_and_is_synthetic(self):
        rows = load_rows(ROWS)
        self.assertEqual(len(rows), 124)
        self.assertTrue(all(r["key"].startswith("mini-") for r in rows))

    def test_malformed_files_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            good = {"key": "a", "prompt": "p", "ground_truth": "g", "instruction_ids": ["x:y"]}
            for text in ('{"key": "a"}\n', "not json\n", json.dumps(good) + "\n" + json.dumps(good) + "\n",
                         json.dumps(dict(good, prompt="  ")) + "\n", json.dumps(dict(good, instruction_ids=[])) + "\n",
                         '{"key": "a", "key": "b", "prompt": "p", "ground_truth": "g", "instruction_ids": ["x:y"]}\n',
                         '{"key": "a", "prompt": "p", "ground_truth": NaN, "instruction_ids": ["x:y"]}\n', "[1]\n"):
                p.write_text(text)
                with self.assertRaises(ScorerError, msg=text):
                    load_rows(p)
            p.write_text("\n" + json.dumps(good) + "\n\n")
            self.assertEqual(len(load_rows(p)), 1)
        with self.assertRaises(ScorerError):
            load_rows(Path(d) / "absent.jsonl")


class LineSeparatorTests(unittest.TestCase):
    """U+2028, U+0085 and friends are legal inside JSON strings; only "\\n" ends a JSONL record."""

    def test_unicode_line_separators_inside_strings_do_not_split_rows(self):
        weird = "a\u2028b\u0085c\u2029d\x0be\x1cf and an emoji \U0001f600"
        good = {"key": "k1", "prompt": weird, "ground_truth": "g", "instruction_ids": ["x:y"]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text(json.dumps(good, ensure_ascii=False) + "\n" + json.dumps(dict(good, key="k2"), ensure_ascii=False) + "\n", encoding="utf-8")
            rows = load_rows(p)
            self.assertEqual([r["prompt"] for r in rows], [weird, weird])
            self.assertEqual(len(ifdata._dump(rows).split("\n")), 3)  # what we write is plain ASCII: any reader can split it
            self.assertTrue(ifdata._dump(rows).isascii())

    def test_source_rows_may_repeat_keys_but_written_files_may_not(self):
        row = {"key": "same", "prompt": "p one", "ground_truth": "g", "instruction_ids": ["x:y"]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, prompt="p two")) + "\n")
            with self.assertRaises(ScorerError):
                load_rows(p)
            self.assertEqual(len(load_rows(p, unique_keys=False)), 2)


class FamiliesTests(unittest.TestCase):
    def test_every_if_rlvr_id_has_one_family(self):
        ids = [i for ids in ifdata.FAMILIES.values() for i in ids]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 54)  # 29 IFBench-Train ids + 25 classic IFEval ids
        self.assertEqual(ifdata.family_of("copy:copy"), "copy")
        self.assertIsNone(ifdata.family_of("custom:csv_city"))


class SaltTests(unittest.TestCase):
    def test_salt_is_random_private_and_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            layout = LabLayout(Path(d))
            salt = ensure_split_salt(layout)
            self.assertEqual(len(salt), 32)
            self.assertEqual(stat.S_IMODE(os.stat(layout.split_salt_path).st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(layout.private_dir).st_mode), 0o700)
            self.assertEqual(ensure_split_salt(layout), salt)
            self.assertEqual(read_split_salt(layout), salt)
            other = LabLayout(Path(d) / "other")
            self.assertNotEqual(ensure_split_salt(other), salt)

    def test_bad_salt_files_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            layout = LabLayout(Path(d))
            with self.assertRaises(ScorerError):
                read_split_salt(layout)
            layout.private_dir.mkdir()
            for text in ("zz\n", "abcd\n", ""):
                layout.split_salt_path.write_text(text)
                with self.assertRaises(ScorerError):
                    read_split_salt(layout)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "lab"
        self.report_path = build(self.out)
        self.layout = LabLayout(self.out)
        self.report = json.loads(self.report_path.read_text())

    def split(self, aid, name):
        return read_split(self.layout, aid, name)

    def all_rows(self):
        return {(aid, name): self.split(aid, name) for aid in ("a-evo-1", "a-held-1") for name in ("train", "dev", "selection")}

    def test_files_sizes_and_locations(self):
        self.assertEqual(self.report_path, self.layout.private_dir / "splits_report.json")
        rows = self.all_rows()
        self.assertEqual({k: len(v) for k, v in rows.items() if k[1] != "train"},
                         {("a-evo-1", "dev"): 10, ("a-evo-1", "selection"): 8, ("a-held-1", "dev"): 6, ("a-held-1", "selection"): 4})
        self.assertGreater(len(rows[("a-evo-1", "train")]), 20)
        self.assertGreater(len(rows[("a-held-1", "train")]), 5)
        self.assertEqual(self.layout.dev_data("a-evo-1").parent, self.layout.public_data_dir / "assignments" / "a-evo-1")
        self.assertEqual(stat.S_IMODE(os.stat(self.layout.selection_data("a-evo-1")).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.layout.selection_data("a-evo-1").parent).st_mode), 0o700)

    def test_filters_are_counted_exactly(self):
        s = self.report["stats"]
        self.assertEqual(s["rows_in"], 124)
        self.assertEqual(s["dropped_duplicates"], 2)
        self.assertEqual(s["dropped_unsatisfiable"], 2)
        self.assertEqual(s["dropped_long_prompt"], 1)
        self.assertEqual(s["dropped_decontam"], 3)
        self.assertEqual(s["dropped_mixed_or_unlisted_family"], 6)
        self.assertEqual(s["dropped_unknown_constraint_id"], 0)
        self.assertEqual(s["evolution_pool"] + s["heldout_pool"], 124 - 2 - 2 - 1 - 3 - 6)

    def test_the_test_set_is_not_in_any_output(self):
        test_prompts = [json.loads(l)["prompt"] for f in TESTS for l in f.read_text().splitlines()]
        for rows in self.all_rows().values():
            for r in rows:
                self.assertNotIn(r["prompt"], test_prompts)
                for sentence in LEAK_SENTENCES:
                    self.assertNotIn(sentence, r["prompt"])

    def test_template_boilerplate_does_not_trigger_decontamination(self):
        # every base prompt starts with a template that also opens some test prompts; only the 3 distinctive sentences are leaks
        self.assertGreater(self.report["stats"]["decontam_template_ngrams_ignored"], 0)
        with self.assertRaises(ScorerError):  # treat nothing as template: all rows overlap the test files and the pool is empty
            build(Path(self.tmp.name) / "strict", template_df=10 ** 9)

    def test_partition_is_clean(self):
        held_ids = {i for f in HELD_FAMS for i in ifdata.FAMILIES[f]}
        for (aid, name), rows in self.all_rows().items():
            for r in rows:
                has_held = bool(set(r["instruction_ids"]) & held_ids)
                if aid == "a-evo-1":
                    self.assertFalse(has_held, (aid, name, r["instruction_ids"]))
                else:
                    self.assertTrue(set(r["instruction_ids"]) <= held_ids, (aid, name, r["instruction_ids"]))

    def test_no_row_is_in_two_files_and_keys_are_row_ids(self):
        keys = [r["key"] for rows in self.all_rows().values() for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        prompts = [r["prompt"] for rows in self.all_rows().values() for r in rows]
        self.assertEqual(len(prompts), len(set(prompts)))
        for k in keys:
            self.assertRegex(k, r"^[0-9a-f]{20}$")

    def test_selection_and_heldout_base_prompts_never_reach_train_or_dev(self):
        src = {r["key"]: r for r in load_rows(ROWS)}
        def source(row):
            return src[row["source_key"]]
        sel_bases = {base_key(source(r)) for (aid, name), rows in self.all_rows().items() if name == "selection" for r in rows}
        held_bases = {base_key(source(r)) for (aid, name), rows in self.all_rows().items() if aid == "a-held-1" for r in rows}
        for (aid, name), rows in self.all_rows().items():
            if name in ("train", "dev"):
                bases = {base_key(source(r)) for r in rows}
                self.assertFalse(bases & sel_bases, (aid, name))
                if aid == "a-evo-1":
                    self.assertFalse(bases & held_bases, (aid, name))
        # selection rows have one row per base prompt
        for aid in ("a-evo-1", "a-held-1"):
            bs = [base_key(source(r)) for r in self.split(aid, "selection")]
            self.assertEqual(len(bs), len(set(bs)))

    def test_invariants_hold_for_many_salts(self):
        src = {r["key"]: r for r in load_rows(ROWS)}
        for i in range(6):
            out = Path(self.tmp.name) / f"salt{i}"
            build(out, salt=hashlib.sha256(b"salt%d" % i).digest())
            layout = LabLayout(out)
            sel = {base_key(src[r["source_key"]]) for aid in ("a-evo-1", "a-held-1") for r in read_split(layout, aid, "selection")}
            for aid in ("a-evo-1", "a-held-1"):
                for name in ("train", "dev"):
                    got = {base_key(src[r["source_key"]]) for r in read_split(layout, aid, name)}
                    self.assertFalse(got & sel, (i, aid, name))
            evo_bases = {base_key(src[r["source_key"]]) for name in ("train", "dev") for r in read_split(layout, "a-evo-1", name)}
            held_bases = {base_key(src[r["source_key"]]) for name in ("train", "dev", "selection") for r in read_split(layout, "a-held-1", name)}
            self.assertFalse(evo_bases & held_bases, i)

    def test_row_fields(self):
        sel = self.split("a-evo-1", "selection")[0]
        self.assertEqual(set(sel), {"key", "source_key", "prompt", "ground_truth", "instruction_ids", "family", "base_prompt"})
        self.assertTrue(sel["prompt"].startswith(sel["base_prompt"]))
        self.assertNotIn("Finish your response", sel["base_prompt"])  # the constraint sentences are not part of the base prompt
        dev = self.split("a-evo-1", "dev")[0]
        self.assertEqual(set(dev), set(sel))
        self.assertTrue(set(dev["family"].split("+")) <= set(EVO_FAMS))
        self.assertEqual(sorted(dev["family"].split("+")), dev["family"].split("+"))
        self.assertIn("instruction_id", dev["ground_truth"])

    def test_deterministic_byte_identical_for_the_same_salt_and_input_order_does_not_matter(self):
        files = lambda lay: {p.relative_to(lay.root).as_posix(): p.read_bytes() for p in sorted(lay.root.rglob("*.json*"))}  # noqa: E731
        again = Path(self.tmp.name) / "again"
        build(again)
        self.assertEqual(files(self.layout), files(LabLayout(again)))
        lines = ROWS.read_text().splitlines()
        random.Random(1).shuffle(lines)
        shuffled = Path(self.tmp.name) / "shuffled.jsonl"
        shuffled.write_text("\n".join(lines) + "\n")
        other = Path(self.tmp.name) / "other_order"
        build(other, rows=shuffled)
        self.assertEqual(files(self.layout), files(LabLayout(other)))

    def test_different_salt_gives_a_different_draw(self):
        other = Path(self.tmp.name) / "other_salt"
        build(other, salt=bytes(reversed(range(32))))
        a = [r["key"] for r in self.split("a-evo-1", "selection")]
        b = [r["key"] for r in read_split(LabLayout(other), "a-evo-1", "selection")]
        self.assertNotEqual(a, b)

    def test_public_train_and_dev_files_share_no_nontemplate_gram_with_selection_or_heldout(self):
        template = set(json.loads((self.layout.private_dir / "template_ngrams.json").read_text())["ngrams"])
        self.assertGreater(len(template), 0)
        def grams(rows):
            out = set()
            for r in rows:
                out |= ifdata.ngrams(r["base_prompt"])
            return out - template
        refs = grams(self.split("a-evo-1", "selection") + self.split("a-held-1", "selection") + [r for n in ("train", "dev") for r in self.split("a-held-1", n)])
        for name in ("train", "dev"):
            self.assertFalse(grams(self.split("a-evo-1", name)) & refs, name)
        held_refs = grams(self.split("a-evo-1", "selection") + self.split("a-held-1", "selection"))
        for name in ("train", "dev"):
            self.assertFalse(grams(self.split("a-held-1", name)) & held_refs, name)

    def test_report_has_no_salt_path_or_time(self):
        text = self.report_path.read_text()
        self.assertNotIn(SALT.hex(), text)
        self.assertNotIn(self.tmp.name, text)
        self.assertEqual(self.report["salt_sha256"], hashlib.sha256(SALT).hexdigest())
        self.assertEqual(self.report["item_mode"], "salted_draw")
        for aid, entry in self.report["assignments"].items():
            for name in ("train", "dev", "selection"):
                p = self.layout.root / entry[name]["path"]
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), entry[name]["sha256"])

    def test_too_few_rows_are_refused(self):
        big = (AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, EVO_FAMS, 500, 10, 0), PLANS[1])
        with self.assertRaises(ScorerError) as cm:
            build(Path(self.tmp.name) / "big", plans=big)
        self.assertIn("eligible rows", str(cm.exception))

    def test_plan_validation(self):
        with self.assertRaises(ScorerError):
            AssignmentPlan(TEST_ASSIGNMENT_ID, AssignmentRole.EVOLUTION, EVO_FAMS, 1, 1, 0)
        with self.assertRaises(ScorerError):
            AssignmentPlan("a", AssignmentRole.EVOLUTION, (), 1, 1, 0)
        with self.assertRaises(ScorerError):
            AssignmentPlan("a", AssignmentRole.EVOLUTION, EVO_FAMS, 0, 1, 0)
        bad_family = (AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, ("copy",), 1, 1, 0), PLANS[1])  # a held-out family in an evolution plan
        with self.assertRaises(ScorerError):
            build(Path(self.tmp.name) / "bad1", plans=bad_family)
        with self.assertRaises(ScorerError):
            build(Path(self.tmp.name) / "bad2", plans=(PLANS[0],))  # needs a held-out assignment
        with self.assertRaises(ScorerError):
            build(Path(self.tmp.name) / "bad3", plans=(PLANS[0], PLANS[0]))
        with self.assertRaises(ScorerError):
            build(Path(self.tmp.name) / "bad4", salt=b"short")
        with self.assertRaises(ScorerError):
            build_splits(ROWS, TESTS, Path(self.tmp.name) / "bad5", salt=SALT, evolution_families=("copy",), heldout_families=("copy",), plans=PLANS)

    def test_unknown_constraint_ids_and_unparsable_ground_truth_are_dropped_and_counted(self):
        rows = load_rows(ROWS)[:60]
        extra = [
            {"key": "x1", "prompt": "unknown id row about a thing that is long enough to be unique one", "ground_truth": repr([{"instruction_id": ["custom:csv_city"], "kwargs": [None]}]), "instruction_ids": ["custom:csv_city"]},
            {"key": "x2", "prompt": "unparsable row about another thing that is also long enough to be unique two", "ground_truth": "not a literal", "instruction_ids": ["punctuation:no_comma"]},
        ]
        p = Path(self.tmp.name) / "extra.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows + extra))
        small = (AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, EVO_FAMS, 3, 3, 3), AssignmentPlan("a-held-1", AssignmentRole.HELDOUT, HELD_FAMS, 2, 2, 2))
        rp = build(Path(self.tmp.name) / "extra_out", rows=p, plans=small)
        stats = json.loads(rp.read_text())["stats"]
        self.assertEqual(stats["dropped_unknown_constraint_id"], 1)
        self.assertEqual(stats["dropped_unparsable_ground_truth"], 1)


class ConvertParquetScriptTests(unittest.TestCase):
    """scripts/lab_convert_parquet.py: the conversion is a pure function; pyarrow is imported only inside main()."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("lab_convert_parquet", Path(__file__).resolve().parents[1] / "scripts" / "lab_convert_parquet.py")
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_module_import_does_not_need_pyarrow(self):
        self.assertNotIn("pyarrow", self.mod.__dict__)

    def test_convert_rows(self):
        gt = repr([{"instruction_id": ["punctuation:no_comma", "last_word:last_word_answer"], "kwargs": [None, {"last_word": "brief"}]}])
        recs = [
            {"key": "k1", "messages": [{"role": "user", "content": "Do it. No commas. End with brief."}], "ground_truth": gt, "constraint": "No commas.\tEnd with brief.",
             "constraint_type": "multi", "dataset": "ifeval"},
            {"key": "k2", "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "ground_truth": gt, "constraint": ""},
            {"key": "k3", "messages": [{"role": "user", "content": "x"}], "ground_truth": "garbage", "constraint": ""},
            {"key": 4, "messages": [{"role": "user", "content": "y"}], "ground_truth": gt, "constraint": None},
        ]
        counts = {}
        out = list(self.mod.convert_rows(recs, counts))
        self.assertEqual([r["key"] for r in out], ["k1", "4"])
        self.assertEqual(out[0]["instruction_ids"], ["punctuation:no_comma", "last_word:last_word_answer"])
        self.assertEqual(out[0]["constraint_type"], "multi")
        self.assertEqual(out[1]["constraint"], "")
        self.assertEqual(counts, {"skipped_turns": 1, "skipped_ground_truth": 1})
        with tempfile.TemporaryDirectory() as d:  # what it writes is what load_rows reads
            p = Path(d) / "rows.jsonl"
            p.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in out))
            self.assertEqual(len(load_rows(p)), 2)

    def test_main_refuses_a_parquet_with_the_wrong_hash(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "x.parquet"
            src.write_bytes(b"not really parquet")
            self.assertEqual(self.mod.main([str(src), str(Path(d) / "o.jsonl"), "--expect-sha256", "0" * 64]), 1)
            self.assertFalse((Path(d) / "o.jsonl").exists())
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                self.assertEqual(self.mod.main([str(src), str(Path(d) / "o.jsonl")]), 2)  # names the missing dependency, writes nothing
                self.assertFalse((Path(d) / "o.jsonl").exists())


class ShortPromptTests(unittest.TestCase):
    """Prompts under 13 tokens have no 13-grams: base-prompt keys and exact matches are what protect them."""

    def rows(self):
        out = []
        ids = ["punctuation:no_comma", "change_case:english_lowercase", "detectable_format:title"]
        for g in range(12):
            for j in range(3):
                cons = f"Constraint sentence variant {j} number {g}."
                gt = repr([{"instruction_id": [ids[j]], "kwargs": [None]}])
                out.append({"key": f"s{g}-{j}", "prompt": f"Describe topic number {g}.\n\n{cons}", "ground_truth": gt, "instruction_ids": [ids[j]], "constraint": cons})
        for k in range(10):
            cons = f"Constraint sentence held {k}."
            out.append({"key": f"h{k}", "prompt": f"Repeat idea {k}.\n\n{cons}", "ground_truth": repr([{"instruction_id": ["copy:copy"], "kwargs": [{"prompt_to_repeat": "x"}]}]),
                        "instruction_ids": ["copy:copy"], "constraint": cons})
        out.append({"key": "t1", "prompt": "Say hello to the lighthouse keeper.", "ground_truth": repr([{"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}]),
                    "instruction_ids": ["punctuation:no_comma"]})
        return out

    def build(self, d):
        d = Path(d)
        rows = d / "rows.jsonl"
        rows.write_text("".join(json.dumps(r) + "\n" for r in self.rows()))
        test = d / "test.jsonl"
        test.write_text(json.dumps({"key": "0", "prompt": "Say hello to the lighthouse keeper.", "instruction_id_list": ["x:y"], "kwargs": [{}]}) + "\n")
        plans = (AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, EVO_FAMS, 8, 6, 0), AssignmentPlan("a-held-1", AssignmentRole.HELDOUT, HELD_FAMS, 3, 3, 0))
        rp = build_splits(rows, [test], d / "lab", salt=SALT, evolution_families=EVO_FAMS, heldout_families=HELD_FAMS, plans=plans)
        return LabLayout(d / "lab"), json.loads(rp.read_text())

    def test_selection_has_one_row_per_base_and_its_bases_never_reach_train_or_dev(self):
        with tempfile.TemporaryDirectory() as d:
            layout, report = self.build(d)
            sel = read_split(layout, "a-evo-1", "selection")
            self.assertEqual(len(sel), 8)
            sel_bases = [r["base_prompt"] for r in sel]
            self.assertEqual(len(set(sel_bases)), 8)  # one row per base prompt
            for name in ("train", "dev"):
                rows = read_split(layout, "a-evo-1", name)
                self.assertTrue(rows)
                self.assertFalse({r["base_prompt"] for r in rows} & set(sel_bases), name)
            self.assertEqual(len(read_split(layout, "a-evo-1", "dev")), 6)

    def test_short_prompt_equal_to_a_test_prompt_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            layout, report = self.build(d)
            self.assertEqual(report["stats"]["dropped_decontam"], 1)
            for aid in ("a-evo-1", "a-held-1"):
                for name in ("train", "dev", "selection"):
                    self.assertNotIn("Say hello to the lighthouse keeper.", [r["prompt"] for r in read_split(layout, aid, name)])

    def test_short_test_prompt_with_constraints_appended_is_dropped(self):
        # the base prompt equals a (13-gram-free) test prompt; the full prompt does not, because the constraint sentence follows it
        cons = "Do not use any commas in your response."
        extra = {"key": "t2", "prompt": "Say hello to the lighthouse keeper. " + cons, "constraint": cons, "instruction_ids": ["punctuation:no_comma"],
                 "ground_truth": repr([{"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}])}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = d / "rows.jsonl"
            rows.write_text("".join(json.dumps(r) + "\n" for r in self.rows() + [extra]))
            test = d / "test.jsonl"
            test.write_text(json.dumps({"key": "0", "prompt": "Say hello to the lighthouse keeper.", "instruction_id_list": ["x:y"], "kwargs": [{}]}) + "\n")
            plans = (AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, EVO_FAMS, 8, 6, 0), AssignmentPlan("a-held-1", AssignmentRole.HELDOUT, HELD_FAMS, 3, 3, 0))
            rp = build_splits(rows, [test], d / "lab", salt=SALT, evolution_families=EVO_FAMS, heldout_families=HELD_FAMS, plans=plans)
            layout, report = LabLayout(d / "lab"), json.loads(rp.read_text())
            self.assertEqual(report["stats"]["dropped_decontam"], 2)  # t1 (exact) and t2 (exact base)
            for aid in ("a-evo-1", "a-held-1"):
                for name in ("train", "dev", "selection"):
                    self.assertNotIn("t2", [r["source_key"] for r in read_split(layout, aid, name)])


class ShortPromptEmbeddedTests(unittest.TestCase):
    """A short (5 to 12 token) test or selection prompt has no 13-gram; a longer prompt that wraps it must still be kept out of public data,
    the same rule the contamination scanner applies to the builder's output."""

    TEST_PROMPT = "Describe the blue glass sculpture."
    CONS = "Do not use any commas in your response."
    NC = {"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}

    def row(self, key, base, wrapped=None):
        text = wrapped or base
        return {"key": key, "prompt": f"{text}\n\n{self.CONS}", "constraint": self.CONS, "instruction_ids": ["punctuation:no_comma"], "ground_truth": repr([self.NC])}

    def rows(self):
        out = [self.row(f"b{g}", f"Explain why shape number {g} is fun.") for g in range(14)]
        # same short base wrapped in extra words, so neither whole-text equality nor the base-prompt key sees it
        out += [self.row(f"w{g}", f"Explain why shape number {g} is fun.", f"Task: Explain why shape number {g} is fun. Please answer politely.") for g in range(14)]
        out += [self.row("tw", self.TEST_PROMPT, f"Task: {self.TEST_PROMPT} Please answer politely."), self.row("te", self.TEST_PROMPT)]
        out += [self.row(f"h{k}", f"Repeat idea {k} aloud.") for k in range(8)]
        for r in out[-8:]:  # the held-out rows use the copy family
            r["instruction_ids"] = ["copy:copy"]
            r["ground_truth"] = repr([{"instruction_id": ["copy:copy"], "kwargs": [{"prompt_to_repeat": "x"}]}])
        return out

    def build(self, d):
        d = Path(d)
        rows = d / "rows.jsonl"
        rows.write_text("".join(json.dumps(r) + "\n" for r in self.rows()))
        test = d / "test.jsonl"
        test.write_text(json.dumps({"key": "0", "prompt": self.TEST_PROMPT, "instruction_id_list": ["x:y"], "kwargs": [{}]}) + "\n")
        plans = (AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, EVO_FAMS, 4, 6, 6), AssignmentPlan("a-held-1", AssignmentRole.HELDOUT, HELD_FAMS, 2, 2, 2))
        rp = build_splits(rows, [test], d / "lab", salt=SALT, evolution_families=EVO_FAMS, heldout_families=HELD_FAMS, plans=plans)
        return LabLayout(d / "lab"), json.loads(rp.read_text())

    def test_a_short_test_prompt_wrapped_in_a_longer_prompt_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            layout, report = self.build(d)
            self.assertEqual(report["stats"]["dropped_decontam"], 2)  # the wrapped row and the bare one
            for aid in ("a-evo-1", "a-held-1"):
                for name in ("train", "dev", "selection"):
                    for r in read_split(layout, aid, name):
                        self.assertNotIn("sculpture", r["prompt"])

    def test_a_short_selection_base_wrapped_in_a_longer_prompt_never_reaches_train_or_dev(self):
        with tempfile.TemporaryDirectory() as d:
            layout, _ = self.build(d)
            sel = read_split(layout, "a-evo-1", "selection")
            self.assertEqual(len(sel), 4)
            phrases = ifdata.short_phrases([r["base_prompt"] for r in sel])
            self.assertTrue(phrases)
            for name in ("train", "dev"):
                rows = read_split(layout, "a-evo-1", name)
                self.assertTrue(rows)
                for r in rows:
                    self.assertFalse(ifdata.has_short_phrase(r["prompt"], phrases), (name, r["prompt"]))

    def test_the_builder_agrees_with_the_scanner_on_short_references(self):
        from nightshift.lab import contamination
        self.assertEqual(ifdata.MIN_SHORT_REFERENCE_TOKENS, contamination.MIN_SHORT_REFERENCE_TOKENS)
        phrases = ifdata.short_phrases(["Explain why shape number 1 is fun.", "Too short here", "x " * 13])
        self.assertEqual(phrases, {7: {"explain why shape number 1 is fun"}})
        self.assertTrue(ifdata.has_short_phrase("Task: explain WHY shape number 1 is fun! Thanks", phrases))
        self.assertFalse(ifdata.has_short_phrase("explain why shape number 2 is fun", phrases))


class LongTestPromptInsideConstraintTests(unittest.TestCase):
    """A whole test prompt of 13 tokens or more that sits inside the constraint sentences is stripped by ``base_text``, so only a
    whole-prompt run check keeps it out of the train and dev files the researcher can read."""

    TEST_PROMPT = "Describe the blue glass sculpture standing at the far end of the quiet harbour on a cold winter morning."
    CONS = "Do not use any commas in your response."
    NC = {"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}

    def row(self, key, base, cons):
        return {"key": key, "prompt": f"{base}\n\n{cons}", "constraint": cons, "instruction_ids": ["punctuation:no_comma"], "ground_truth": repr([self.NC])}

    def rows(self):
        out = [self.row(f"b{g}", f"Explain why shape number {g} is fun for children.", self.CONS) for g in range(16)]
        out += [self.row(f"x{g}", f"Explain why cube number {g} is fun for adults.", f"Also answer this: {self.TEST_PROMPT} {self.CONS}") for g in range(4)]
        out += [self.row(f"h{k}", f"Repeat idea {k} aloud.", self.CONS) for k in range(8)]
        for r in out[-8:]:
            r["instruction_ids"] = ["copy:copy"]
            r["ground_truth"] = repr([{"instruction_id": ["copy:copy"], "kwargs": [{"prompt_to_repeat": "x"}]}])
        return out

    def test_a_whole_long_test_prompt_inside_the_constraint_text_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = d / "rows.jsonl"
            rows.write_text("".join(json.dumps(r) + "\n" for r in self.rows()))
            test = d / "test.jsonl"
            test.write_text(json.dumps({"key": "0", "prompt": self.TEST_PROMPT, "instruction_id_list": ["x:y"], "kwargs": [{}]}) + "\n")
            plans = (AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, EVO_FAMS, 4, 6, 6), AssignmentPlan("a-held-1", AssignmentRole.HELDOUT, HELD_FAMS, 2, 2, 2))
            rp = build_splits(rows, [test], d / "lab", salt=SALT, evolution_families=EVO_FAMS, heldout_families=HELD_FAMS, plans=plans)
            layout, report = LabLayout(d / "lab"), json.loads(rp.read_text())
            self.assertEqual(report["stats"]["dropped_decontam"], 4)
            for aid in ("a-evo-1", "a-held-1"):
                for name in ("train", "dev", "selection"):
                    for r in read_split(layout, aid, name):
                        self.assertNotIn("sculpture", r["prompt"])

    def test_long_run_helper_needs_the_whole_prompt(self):
        runs = ifdata.long_runs([self.TEST_PROMPT, "too short here"])
        self.assertTrue(ifdata.has_long_run("Prefix. " + self.TEST_PROMPT.upper() + " Suffix.", runs))
        self.assertFalse(ifdata.has_long_run(self.TEST_PROMPT.rsplit(" ", 1)[0], runs))  # one token short of the whole prompt


class SelectionPromptInsideConstraintTests(unittest.TestCase):
    """A selection prompt of 13 tokens or more that sits inside another row's constraint text is stripped by ``base_text``, so the n-gram
    reservation never sees it; the scanner flags the public file, so the builder must keep that row out of train and dev."""

    CONS = "Do not use any commas in your response."
    NC = {"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}

    def base(self, g):
        return f"Describe the blue glass sculpture number {g} standing at the far end of the quiet harbour on a cold winter morning."

    def row(self, key, base, cons):
        return {"key": key, "prompt": f"{base}\n\n{cons}", "constraint": cons, "instruction_ids": ["punctuation:no_comma"], "ground_truth": repr([self.NC])}

    def rows(self):
        out = [self.row(f"b{g}", self.base(g), self.CONS) for g in range(14)]
        out += [self.row(f"x{g}", f"Explain why cube number {g} is fun for adults who like to build things.", f"Also answer this: {self.base(g)} {self.CONS}") for g in range(14)]
        out += [self.row(f"h{k}", f"Repeat idea {k} aloud.", self.CONS) for k in range(8)]
        for r in out[-8:]:
            r["instruction_ids"] = ["copy:copy"]
            r["ground_truth"] = repr([{"instruction_id": ["copy:copy"], "kwargs": [{"prompt_to_repeat": "x"}]}])
        return out

    def test_a_selection_prompt_inside_another_rows_constraint_text_stays_out_of_train_and_dev(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = d / "rows.jsonl"
            rows.write_text("".join(json.dumps(r) + "\n" for r in self.rows()))
            test = d / "test.jsonl"
            test.write_text(json.dumps({"key": "0", "prompt": "Unrelated test prompt.", "instruction_id_list": ["x:y"], "kwargs": [{}]}) + "\n")
            plans = (AssignmentPlan("a-evo-1", AssignmentRole.EVOLUTION, EVO_FAMS, 4, 4, 1), AssignmentPlan("a-held-1", AssignmentRole.HELDOUT, HELD_FAMS, 2, 2, 2))
            build_splits(rows, [test], d / "lab", salt=SALT, evolution_families=EVO_FAMS, heldout_families=HELD_FAMS, plans=plans)
            layout = LabLayout(d / "lab")
            for aid in ("a-evo-1",):
                sel = read_split(layout, aid, "selection")
                runs = ifdata.long_runs([t for r in sel for t in (r["prompt"], r["base_prompt"])])
                self.assertTrue(runs)
                for name in ("train", "dev"):
                    rs = read_split(layout, aid, name)
                    self.assertTrue(rs)
                    for r in rs:
                        self.assertFalse(ifdata.has_long_run(r["prompt"], runs), (aid, name, r["prompt"]))
            # the lab's own scanner agrees: no public file hits a selection prompt
            from nightshift.lab import contamination
            sc = contamination.ContaminationScanner([t for r in read_split(layout, "a-evo-1", "selection") for t in (r["prompt"], r["base_prompt"])],
                                                    [r["base_prompt"] for r in read_split(layout, "a-evo-1", "selection")],
                                                    template_ngrams=json.loads((layout.private_dir / "template_ngrams.json").read_text())["ngrams"])
            for aid in ("a-evo-1", "a-held-1"):
                for name in ("train", "dev"):
                    for r in read_split(layout, aid, name):
                        self.assertFalse(sc.has_ngram(r["prompt"]), (aid, name))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.layout = LabLayout(Path(self.tmp.name) / "lab")
        self.salt = ensure_split_salt(self.layout)
        build(self.layout.root, salt=self.salt)

    def manifest(self, **kw):
        args = dict(evolution_families=EVO_FAMS + ("keywords_new",), heldout_families=HELD_FAMS, ifrlvr_parquet_sha256="e" * 64,
                    ifbench_commit="1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d", open_instruct_commit="b781e782a8fc530577f235aefe282dd046cb8ad2",
                    clock=lambda: 1_790_000_000.0)
        args.update(kw)
        return build_assignment_manifest(self.layout, PLANS, **args)

    def test_manifest_records_file_hashes_roles_and_salt_hash(self):
        m = self.manifest()
        self.assertIsInstance(m, AssignmentManifest)
        evo = m.get("a-evo-1")
        self.assertEqual(evo.role, AssignmentRole.EVOLUTION)
        for key, path in (("train_sha256", self.layout.train_data), ("dev_sha256", self.layout.dev_data), ("selection_sha256", self.layout.selection_data)):
            self.assertEqual(evo.data_revisions[key], hashlib.sha256(path("a-evo-1").read_bytes()).hexdigest())
        self.assertEqual(m.get("a-held-1").role, AssignmentRole.HELDOUT)
        self.assertEqual(m.source_revisions["split_salt_sha256"], hashlib.sha256(self.salt).hexdigest())
        self.assertEqual(m.item_mode, ItemMode.SALTED_DRAW)
        self.assertEqual(m.mixed_row_policy, MixedRowPolicy.DROPPED)
        self.assertEqual(AssignmentManifest.from_dict(json.loads(json.dumps(m.to_dict()))), m)
        self.assertNotIn(self.salt.hex(), json.dumps(m.to_dict()))

    def test_manifest_requires_the_files(self):
        self.layout.dev_data("a-held-1").unlink()
        with self.assertRaises(ScorerError):
            self.manifest()

    def test_manifest_follows_the_contract_rules(self):
        with self.assertRaises(ContractError):
            self.manifest(heldout_families=("firstlast",))  # a-held-1 uses copy: outside the partition

    def test_scorer_reads_the_built_files_against_the_manifest(self):
        from nightshift.lab.contracts import Arm, BaseModelRef, GenerationParams, Phase, ScoreRequest, Split, Verifier
        from nightshift.lab.scorer import ScoreService
        from tests.support.lab_fake_generator import FakeGenerator, FakeVerifierBackend
        m = self.manifest()
        svc = ScoreService(self.layout, m, {Verifier.IFEVALG: FakeVerifierBackend(Verifier.IFEVALG)}, guard=lambda r: None, clock=lambda: 1_790_000_000.0)
        req = ScoreRequest("sc-1", Phase.SEARCH, "a-evo-1", Split.DEV, None, Arm.STOCK, BaseModelRef("meta-models/Muse-Glimmer-30B", "a" * 40), None,
                           (0,), GenerationParams.recommended(1000), svc.scorer_version, None, None)
        rec = svc.score(req, FakeGenerator(), "m")
        self.assertEqual(rec.n_prompts, 10)
        self.layout.dev_data("a-evo-1").write_text("tampered\n")
        with self.assertRaises(ScorerError):
            svc.score(ScoreRequest("sc-2", Phase.SEARCH, "a-evo-1", Split.DEV, None, Arm.STOCK, BaseModelRef("meta-models/Muse-Glimmer-30B", "a" * 40), None,
                                   (0,), GenerationParams.recommended(1000), svc.scorer_version, None, None), FakeGenerator(), "m")


if __name__ == "__main__":
    unittest.main()
