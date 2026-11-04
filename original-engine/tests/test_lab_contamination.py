"""Contamination scan (owner 1): planted exact and 13-gram hits are found, disjoint base prompts give no hit. Synthetic data only."""

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from nightshift.lab import contamination
from nightshift.lab.contamination import ContaminationScanner, build_scanner, scan_job
from nightshift.lab.contracts import Arm, AssignmentRole, ContaminationReport, FileRef, LabLayout, Phase, ScorerError, write_record
from nightshift.lab.ifdata import AssignmentPlan, build_assignment_manifest, build_splits, ensure_split_salt, ngrams
from tests.test_lab_contracts import spec as make_trial_spec
from tests.test_lab_ifdata import EVO_FAMS, FIX, HELD_FAMS, PLANS, ROWS, TESTS

CLOCK = lambda: 1_790_000_000.0  # noqa: E731


def jl(*rows) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def chat(user, assistant="Sure."):
    return {"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


class LabCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.layout = LabLayout(Path(self.tmp.name) / "lab")
        salt = ensure_split_salt(self.layout)
        build_splits(ROWS, TESTS, self.layout.root, salt=salt, evolution_families=EVO_FAMS + ("keywords_new",), heldout_families=HELD_FAMS, plans=PLANS)
        self.layout.test_data().parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIX / "ifbench_test_mini.jsonl", self.layout.test_data())
        self.manifest = build_assignment_manifest(self.layout, PLANS, evolution_families=EVO_FAMS + ("keywords_new",), heldout_families=HELD_FAMS,
                                                  ifrlvr_parquet_sha256="e" * 64, ifbench_commit="a" * 40, open_instruct_commit="b" * 40, clock=CLOCK)
        self.scanner = build_scanner(self.layout, self.manifest)
        self.sel = [json.loads(l) for l in self.layout.selection_data("a-evo-1").read_text().splitlines()]
        self.held_sel = [json.loads(l) for l in self.layout.selection_data("a-held-1").read_text().splitlines()]
        self.held_train = [json.loads(l) for l in self.layout.train_data("a-held-1").read_text().splitlines()]
        self.evo_train = [json.loads(l) for l in self.layout.train_data("a-evo-1").read_text().splitlines()]
        self.test_rows = [json.loads(l) for l in (FIX / "ifbench_test_mini.jsonl").read_text().splitlines()]

    def put(self, name: str, data: bytes) -> FileRef:
        p = self.layout.trial_train_data_path("t-0001", name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return self.layout.file_ref(p)

    def scan(self, rows, **kw):
        ref = self.put("j-0001", jl(*rows))
        return scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0001", train_data=ref, clock=CLOCK, scanner=self.scanner, **kw)


class ScanJobTests(LabCase):
    def test_clean_public_training_data_passes(self):
        rep = self.scan([chat(r["prompt"], "an answer") for r in self.evo_train])
        self.assertIsInstance(rep, ContaminationReport)
        self.assertTrue(rep.passed)
        self.assertEqual((rep.n_rows, rep.n_exact_hits, rep.n_ngram_hits, rep.ngram_n), (len(self.evo_train), 0, 0, 13))
        self.assertEqual(rep.trial_id, "t-0001")
        self.assertEqual(rep.scanned.path, "trials/t-0001/train_data/j-0001.jsonl")
        self.assertGreaterEqual(len(rep.reference_sha256), 5)

    def test_exact_selection_prompt_is_found(self):
        rep = self.scan([chat(self.evo_train[0]["prompt"]), chat(self.sel[2]["prompt"]), chat(self.evo_train[1]["prompt"])])
        self.assertFalse(rep.passed)
        self.assertEqual(rep.n_exact_hits, 1)
        self.assertEqual(rep.n_rows, 3)

    def test_exact_match_ignores_case_and_whitespace(self):
        rep = self.scan([{"text": "  " + self.sel[0]["prompt"].upper().replace(" ", "   ") + "\n"}])
        self.assertEqual(rep.n_exact_hits, 1)

    def test_base_prompt_embedded_in_a_longer_message_is_an_ngram_hit(self):
        wrapped = "Here is a task. " + self.sel[1]["base_prompt"] + " Please also reply in French and add a poem."
        rep = self.scan([chat(wrapped)])
        self.assertEqual((rep.n_exact_hits, rep.n_ngram_hits), (0, 1))
        self.assertFalse(rep.passed)

    def test_short_test_prompt_wrapped_in_a_longer_message_is_a_hit(self):
        # a test prompt of fewer than 13 tokens has no 13-gram; wrapping it must not hide it
        short = "List three tools a locksmith carries, separated by semicolons only."
        self.assertIn(short, [r["prompt"] for r in self.test_rows])
        self.assertEqual(self.scan([chat(short)]).n_exact_hits, 1)
        rep = self.scan([chat("Task: " + short + " Answer politely.")])
        self.assertEqual((rep.n_exact_hits, rep.n_ngram_hits), (0, 1))
        self.assertFalse(rep.passed)
        self.assertFalse(self.scan([chat("List three tools a carpenter carries, separated by commas only.")]).n_ngram_hits)

    def test_whole_selection_prompt_with_a_short_base_wrapped_in_a_longer_message_is_a_hit(self):
        # the base supplies no n-gram and the whole-string comparison fails once the prompt is wrapped: the whole prompt as a run must still hit
        sc = ContaminationScanner(["Hello. Please answer the following question in exactly three sentences without using any commas at all today."],
                                  ["Hello."], background=[])
        whole = "Hello. Please answer the following question in exactly three sentences without using any commas at all today."
        self.assertTrue(sc.has_ngram("Task: " + whole + " Please respond."))
        self.assertFalse(sc.has_ngram("Task: Hello. Please respond."))
        self.assertFalse(sc.has_ngram(whole.rsplit(" ", 1)[0]))  # not the whole prompt

    def test_held_out_family_data_is_a_hit(self):
        rep = self.scan([chat(self.held_train[0]["prompt"] + " Reply briefly.")])
        self.assertEqual(rep.n_ngram_hits, 1)
        rep = self.scan([chat(self.held_sel[0]["prompt"])])
        self.assertEqual(rep.n_exact_hits, 1)

    def write_spec(self, **kw):
        sp = make_trial_spec(trial_id="t-0001", **kw)
        self.layout.trial_spec_path("t-0001").parent.mkdir(parents=True, exist_ok=True)
        write_record(self.layout.trial_spec_path("t-0001"), sp)

    def test_a_final_trial_may_train_on_its_own_held_out_train_and_dev_but_nothing_else_held_out(self):
        self.write_spec(phase=Phase.FINAL, arm=Arm.V0, assignment_id="a-held-1", final_request_id="fr-1")
        held_dev = [json.loads(l) for l in self.layout.dev_data("a-held-1").read_text().splitlines()]
        rep = self.scan([chat(self.held_train[0]["prompt"]), chat(held_dev[0]["prompt"])])
        self.assertTrue(rep.passed, rep)
        self.assertEqual((rep.n_exact_hits, rep.n_ngram_hits), (0, 0))
        # its own selection and the test prompts stay forbidden
        self.assertFalse(self.scan([chat(self.held_sel[0]["prompt"])]).passed)
        self.assertFalse(self.scan([chat(self.test_rows[0]["prompt"])]).passed)
        # the same data inside a prebuilt scanner of a search trial (evolution assignment) is still a hit
        self.write_spec(assignment_id="a-evo-1")
        self.assertFalse(self.scan([chat(self.held_train[0]["prompt"])]).passed)

    def test_without_a_trial_spec_the_held_out_train_data_is_still_a_hit(self):
        self.assertFalse(self.scan([chat(self.held_train[0]["prompt"])]).passed)

    def test_test_prompt_is_found_exact_and_by_ngram(self):
        distinctive = [r["prompt"] for r in self.test_rows if "harbour" in r["prompt"]][0]
        self.assertEqual(self.scan([chat(distinctive)]).n_exact_hits, 1)
        span = " ".join(distinctive.split()[:14])
        rep = self.scan([chat("Earlier someone wrote: " + span + " and then stopped.")])
        self.assertEqual((rep.n_exact_hits, rep.n_ngram_hits), (0, 1))

    def test_ngram_boundary_is_thirteen_tokens(self):
        gram = sorted(self.scanner._grams)[0].split()
        self.assertEqual(len(gram), 13)
        self.assertTrue(self.scanner.has_ngram("zzzfirst " + " ".join(gram) + " zzzlast"))
        self.assertFalse(self.scanner.has_ngram("zzzfirst " + " ".join(gram[:12]) + " zzzlast"))
        self.assertFalse(self.scanner.has_ngram(" ".join(gram[1:]) + " zzzlast"))

    def test_a_gram_at_the_very_end_or_start_of_a_row_is_found(self):
        gram = sorted(self.scanner._grams)[0]
        self.assertTrue(self.scanner.has_ngram(gram))  # the whole text is exactly the 13-gram
        self.assertTrue(self.scanner.has_ngram("zzz yyy " + gram))
        self.assertTrue(self.scanner.has_ngram(gram + " zzz yyy"))

    def test_shared_template_boilerplate_is_not_a_hit(self):
        template = "Please write a short descriptive note about the following topic and keep it suitable for a general audience:"
        self.assertTrue(ngrams(template) & self.scanner._grams == set())  # filtered out of the reference
        rep = self.scan([chat(template + " glass harmonicas, including a short list of materials.")])
        self.assertTrue(rep.passed)

    def test_rows_of_other_shapes_are_scanned(self):
        sel = self.sel[0]["prompt"].replace("\n", " ")
        data = ("plain text line mentioning nothing\n" + sel + "\n" + json.dumps({"a": [{"b": ["x", sel]}]}) + "\n\n" +
                json.dumps(["not", "an", "object"]) + "\n").encode()
        rep = scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0001", train_data=self.put("j-0001", data), clock=CLOCK, scanner=self.scanner)
        self.assertEqual((rep.n_rows, rep.n_exact_hits), (4, 2))

    def test_unicode_line_separators_do_not_split_a_row_and_hide_a_hit(self):
        # a selection prompt split by U+2028 inside one JSON string must still be one exact hit
        sel = self.sel[0]["prompt"].replace("\n", "\u2028")
        data = (json.dumps(chat(sel), ensure_ascii=False) + "\n").encode()
        rep = scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0001", train_data=self.put("j-0001", data), clock=CLOCK, scanner=self.scanner)
        self.assertEqual(rep.n_rows, 1)
        self.assertEqual(rep.n_exact_hits, 1)

    def test_hit_row_is_counted_once_per_kind(self):
        both = chat(self.sel[0]["prompt"] + " " + self.sel[1]["base_prompt"])
        rep = self.scan([both])
        self.assertEqual((rep.n_rows, rep.n_exact_hits, rep.n_ngram_hits), (1, 0, 1))

    def test_report_contains_counts_only(self):
        rep = self.scan([chat(self.sel[0]["prompt"])])
        text = json.dumps(rep.to_dict())
        self.assertNotIn(self.sel[0]["prompt"][:40], text)

    def test_empty_training_data_passes_with_zero_rows(self):
        rep = scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0002", train_data=self.put("j-0002", b""), clock=CLOCK, scanner=self.scanner)
        self.assertEqual((rep.n_rows, rep.passed), (0, True))

    def test_sha_mismatch_and_symlink_are_refused(self):
        ref = self.put("j-0001", jl(chat("harmless")))
        bad = FileRef(ref.path, "0" * 64, ref.size_bytes)
        with self.assertRaises(ScorerError):
            scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0001", train_data=bad, clock=CLOCK, scanner=self.scanner)
        # a symbolic link planted at the harvest path that points at a canary under private/ is never read
        link = self.layout.trial_train_data_path("t-0001", "j-0003")
        link.symlink_to(self.layout.selection_data("a-evo-1"))
        with self.assertRaises(ScorerError):
            scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0003",
                     train_data=FileRef(link.relative_to(self.layout.root).as_posix(), "0" * 64, 1), clock=CLOCK, scanner=self.scanner)
        with self.assertRaises(ScorerError):
            scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0004",
                     train_data=FileRef("trials/t-0001/train_data/absent.jsonl", "0" * 64, 1), clock=CLOCK, scanner=self.scanner)

    def test_hard_linked_data_is_refused(self):
        p = self.layout.trial_train_data_path("t-0001", "j-0005")
        p.parent.mkdir(parents=True, exist_ok=True)
        os.link(self.layout.selection_data("a-evo-1"), p)
        data = p.read_bytes()
        with self.assertRaises(ScorerError):
            scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0005",
                     train_data=FileRef(p.relative_to(self.layout.root).as_posix(), hashlib.sha256(data).hexdigest(), len(data)), clock=CLOCK, scanner=self.scanner)


def tarball(files: dict, extra=None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            raw = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        if extra:
            extra(tar)
    return buf.getvalue()


class RecipeTarballTests(LabCase):
    def scan_tar(self, data: bytes):
        train = self.put("j-0001", jl(chat("harmless prompt about gardens")))
        recipe = self.put("j-0001-recipe", data)
        return scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0001", train_data=train, recipe=recipe, clock=CLOCK, scanner=self.scanner)

    def test_clean_recipe_passes_and_rows_are_added(self):
        rep = self.scan_tar(tarball({"run.sh": "echo hi\n", "train.py": "print(1)\n"}))
        self.assertTrue(rep.passed)
        self.assertEqual(rep.n_rows, 1 + 2)

    def test_selection_prompt_hidden_in_a_recipe_file_is_found(self):
        rep = self.scan_tar(tarball({"run.sh": "echo hi\n", "data/extra.txt": "line one\n" + self.sel[0]["prompt"] + "\nline three\n"}))
        self.assertFalse(rep.passed)
        self.assertEqual(rep.n_exact_hits, 1)
        rep = self.scan_tar(tarball({"notes.md": "copied from somewhere: " + self.sel[1]["base_prompt"]}))
        self.assertEqual(rep.n_ngram_hits, 1)

    def test_links_and_devices_in_the_tarball_are_refused(self):
        def add_symlink(tar):
            info = tarfile.TarInfo("evil")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        def add_hardlink(tar):
            info = tarfile.TarInfo("evil2")
            info.type = tarfile.LNKTYPE
            info.linkname = "run.sh"
            tar.addfile(info)
        for extra in (add_symlink, add_hardlink):
            with self.assertRaises(ScorerError):
                self.scan_tar(tarball({"run.sh": "x"}, extra))

    def test_not_a_tarball_is_refused(self):
        with self.assertRaises(ScorerError):
            self.scan_tar(b"this is not a gzip file")

    def test_oversize_member_is_refused(self):
        old = contamination.MAX_TAR_MEMBER_BYTES
        contamination.MAX_TAR_MEMBER_BYTES = 10
        self.addCleanup(setattr, contamination, "MAX_TAR_MEMBER_BYTES", old)
        with self.assertRaises(ScorerError):
            self.scan_tar(tarball({"big.txt": "x" * 100}))


class ScannerConstructionTests(LabCase):
    def test_missing_reference_file_is_an_error_not_a_silent_skip(self):
        self.layout.test_data().unlink()
        with self.assertRaises(ScorerError):
            build_scanner(self.layout, self.manifest)
        self.layout.test_data().write_bytes((FIX / "ifbench_test_mini.jsonl").read_bytes())
        self.layout.selection_data("a-held-1").unlink()
        with self.assertRaises(ScorerError):
            build_scanner(self.layout, self.manifest)

    def test_scanner_without_references_is_refused(self):
        with self.assertRaises(ScorerError):
            ContaminationScanner([], [])
        with self.assertRaises(ScorerError):
            ContaminationScanner(["x"], ["x"], n=5)

    def test_scan_job_builds_its_own_scanner_when_none_is_given(self):
        ref = self.put("j-0001", jl(chat(self.sel[0]["prompt"])))
        rep = scan_job(self.layout, self.manifest, trial_id="t-0001", job_id="j-0001", train_data=ref, clock=CLOCK)
        self.assertFalse(rep.passed)

    def test_reference_hashes_cover_every_file_read(self):
        want = {hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [self.layout.selection_data("a-evo-1"), self.layout.selection_data("a-held-1"), self.layout.train_data("a-held-1"),
                 self.layout.dev_data("a-held-1"), self.layout.test_data()]}
        self.assertTrue(want <= set(self.scanner.reference_sha256))


if __name__ == "__main__":
    unittest.main()
