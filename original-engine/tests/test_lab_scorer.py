"""ScoreService (owner 1): access policy row by row, guard injection, files, aggregation, truncation, no re-roll."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from nightshift.lab.contracts import (
    TEST_ASSIGNMENT_ID,
    AccessDenied,
    AdapterFile,
    AdapterRef,
    Arm,
    Assignment,
    AssignmentManifest,
    AssignmentRole,
    BaseModelRef,
    FinalComparisonRefused,
    GenerationParams,
    ItemMode,
    LabLayout,
    MixedRowPolicy,
    Phase,
    ScoreRecord,
    ScoreRequest,
    ScorerError,
    Scorer,
    Split,
    Verifier,
    read_record,
    utc_iso,
)
from nightshift.lab.scorer import ACCESS_POLICY, ROLE_TEST, Access, ScoreService
from tests.support.lab_fake_generator import FakeGenerator, FakeVerifierBackend

BASE = BaseModelRef("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68")
PARAMS = GenerationParams.recommended(2000)
EVO, HELD = "a-evo-1", "a-held-1"
IDS = ["punctuation:no_comma", "change_case:english_lowercase"]
CLOCK = lambda: 1_790_000_000.0  # noqa: E731


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def adapter(tag="1") -> AdapterRef:
    return AdapterRef.from_files("ad1", [AdapterFile("adapter_config.json", tag * 64, 100), AdapterFile("adapter_model.safetensors", "2" * 64, 5000)])


def ifevalg_rows(prefix: str, n: int) -> list:
    return [
        {"key": f"{prefix}{i}", "prompt": f"Synthetic prompt {prefix}{i}: write a short note about topic number {i}.",
         "ground_truth": repr([{"instruction_id": IDS, "kwargs": [None, None]}]), "instruction_ids": IDS, "family": "ifeval25"}
        for i in range(n)
    ]


def ifbench_rows(n: int) -> list:
    return [
        {"key": str(i), "prompt": f"Synthetic test prompt {i}: describe object {i}.", "instruction_id_list": ["count:fake_a", "count:fake_b"],
         "kwargs": [{"N": None}, {"word": "x"}]}
        for i in range(n)
    ]


def write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    path.write_bytes(data)
    return sha_bytes(data)


class Lab:
    """A lab root with one evolution and one held-out assignment, a test file, and a manifest that hashes them."""

    def __init__(self, root: Path, n_dev=4, n_sel=5, n_test=3):
        self.layout = LabLayout(root)
        revs = {}
        self.rows = {}
        for aid in (EVO, HELD):
            tr = write_jsonl(self.layout.train_data(aid), ifevalg_rows(f"{aid}-t", 3))
            self.rows[(aid, "dev")] = ifevalg_rows(f"{aid}-d", n_dev)
            self.rows[(aid, "selection")] = ifevalg_rows(f"{aid}-s", n_sel)
            dv = write_jsonl(self.layout.dev_data(aid), self.rows[(aid, "dev")])
            sl = write_jsonl(self.layout.selection_data(aid), self.rows[(aid, "selection")])
            revs[aid] = {"train_sha256": tr, "dev_sha256": dv, "selection_sha256": sl}
        self.test_rows = ifbench_rows(n_test)
        write_jsonl(self.layout.test_data(), self.test_rows)
        self.manifest = AssignmentManifest(
            assignments=(
                Assignment(EVO, AssignmentRole.EVOLUTION, ("ifeval25",), revs[EVO]),
                Assignment(HELD, AssignmentRole.HELDOUT, ("copy",), revs[HELD]),
            ),
            evolution_families=("ifeval25", "counting"), heldout_families=("copy", "firstlast"),
            mixed_row_policy=MixedRowPolicy.DROPPED, item_mode=ItemMode.SALTED_DRAW,
            source_revisions={"ifrlvr_parquet_sha256": "e" * 64, "ifbench_commit": "b" * 40, "open_instruct_commit": "c" * 40, "split_salt_sha256": "d" * 64},
            created_utc="2026-11-01T12:00:00+00:00",
        )


class ScorerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lab = Lab(Path(self.tmp.name))
        self.layout = self.lab.layout
        self.guard_calls = []
        self.guard_error = None
        self.ev = FakeVerifierBackend(Verifier.IFEVALG)
        self.ib = FakeVerifierBackend(Verifier.IFBENCH_UPSTREAM)
        self.svc = self.make_service()
        self.gen = FakeGenerator()

    def guard(self, request):
        self.guard_calls.append(request)
        if self.guard_error is not None:
            raise self.guard_error

    def make_service(self, **kw):
        verifiers = kw.pop("verifiers", {Verifier.IFEVALG: self.ev, Verifier.IFBENCH_UPSTREAM: self.ib})
        return ScoreService(self.layout, self.lab.manifest, verifiers, guard=self.guard, clock=CLOCK, **kw)

    def request(self, **kw) -> ScoreRequest:
        base = dict(request_id="sc-0001", phase=Phase.SEARCH, assignment_id=EVO, split=Split.DEV, trial_id=None, arm=Arm.STOCK, base=BASE,
                    adapter=None, seeds=(0, 1), params=PARAMS, scorer_version=self.svc.scorer_version, reason=None, final_request_id=None)
        base.update(kw)
        return ScoreRequest(**base)

    def final_request(self, **kw) -> ScoreRequest:
        base = dict(phase=Phase.FINAL, assignment_id=HELD, split=Split.SELECTION, arm=Arm.EVOLVED, adapter=adapter(), final_request_id="f-0001", trial_id="t-0001")
        base.update(kw)
        return self.request(**base)

    def files_in(self, sid="sc-0001"):
        d = self.layout.score_dir(sid)
        return sorted(p.name for p in d.iterdir()) if d.exists() else []


class ScoreFlowTests(ScorerTestBase):
    def test_satisfies_the_scorer_protocol(self):
        self.assertIsInstance(self.svc, Scorer)

    def test_a_concurrent_score_of_the_same_request_is_refused_while_the_first_runs(self):
        req = self.request(split=Split.DEV)
        inner_errors = []

        def respond(model, prompt, params, seed):
            if not inner_errors:  # the first call is generating and has not written score.json yet: a second caller arrives now
                self.svc.precheck(req)  # the early refusal alone does not see it
                other = FakeGenerator()
                with self.assertRaises(ScorerError) as cm:
                    self.svc.score(req, other, "glimmer-base")
                inner_errors.append(cm.exception)
                self.assertEqual(other.calls, [])  # it never generated
            return "OK"

        rec = self.svc.score(req, FakeGenerator(respond), "glimmer-base")
        self.assertEqual(len(inner_errors), 1)
        self.assertIn("already being computed", str(inner_errors[0]))
        self.assertEqual(read_record(self.layout.score_record_path("sc-0001"), ScoreRecord), rec)
        with self.assertRaises(ScorerError):  # and once written, it is still never re-rolled
            self.svc.score(req, self.gen, "glimmer-base")

    def test_a_concurrent_score_of_the_same_logical_score_under_another_id_is_refused(self):
        req = self.request(split=Split.SELECTION, adapter=adapter(), arm=Arm.V0, trial_id="t-0001")
        other_req = dataclasses_replace(req, request_id="sc-0002")
        errors = []

        def respond(model, prompt, params, seed):
            if not errors:
                other = FakeGenerator()
                with self.assertRaises(ScorerError) as cm:
                    self.svc.score(other_req, other, "glimmer-base")
                errors.append(cm.exception)
                self.assertEqual(other.calls, [])
            return "OK"

        self.svc.score(req, FakeGenerator(respond), "glimmer-base")
        self.assertIn("already being computed", str(errors[0]))
        with self.assertRaises(ScorerError):  # once written, the same logical score under a new id is still refused
            self.svc.precheck(other_req)
        dev = self.request(split=Split.DEV, request_id="sc-0003")  # dev scores may be repeated
        self.svc.score(dev, self.gen, "glimmer-base")
        self.svc.score(dataclasses_replace(dev, request_id="sc-0004"), self.gen, "glimmer-base")

    def crash_before_score_json(self, req):
        """The state a process killed between the marker write and score.json leaves: request.json and the logical marker, no score.json."""
        self.svc.score(req, self.gen, "glimmer-base")
        self.layout.score_record_path(req.request_id).unlink()
        marker = self.svc._logical_marker(self.svc.logical_key(req))
        self.assertTrue(marker.is_file())
        return marker

    def test_a_crash_between_the_logical_marker_and_score_json_does_not_block_a_rescore(self):
        req = self.request(split=Split.SELECTION, adapter=adapter(), arm=Arm.V0, trial_id="t-0001")
        marker = self.crash_before_score_json(req)
        new_req = dataclasses_replace(req, request_id="sc-0002")
        self.svc.precheck(new_req)  # no writer holds the lock: the orphan marker does not refuse
        rec = self.svc.score(new_req, self.gen, "glimmer-base")
        self.assertEqual(marker.read_text().strip(), "sc-0002")
        self.assertEqual(read_record(self.layout.score_record_path("sc-0002"), ScoreRecord), rec)
        with self.assertRaises(ScorerError):  # and the recovered score is once again never re-rolled
            self.svc.score(dataclasses_replace(req, request_id="sc-0003"), self.gen, "glimmer-base")

    def test_a_crash_before_score_json_can_be_retried_under_the_same_request_id(self):
        req = self.request(split=Split.SELECTION, adapter=adapter(), arm=Arm.V0, trial_id="t-0001")
        self.crash_before_score_json(req)
        rec = self.svc.score(req, self.gen, "glimmer-base")
        self.assertEqual(read_record(self.layout.score_record_path("sc-0001"), ScoreRecord), rec)

    def test_a_marker_whose_writer_is_still_running_is_not_an_orphan(self):
        req = self.request(split=Split.SELECTION, adapter=adapter(), arm=Arm.V0, trial_id="t-0001")
        marker = self.crash_before_score_json(req)
        fd = self.svc._flock(f"logical-{self.svc.logical_key(req)}.lock", "busy")  # a live writer between its two writes
        self.addCleanup(os.close, fd)
        with self.assertRaises(ScorerError):
            self.svc.precheck(dataclasses_replace(req, request_id="sc-0002"))
        self.assertTrue(marker.is_file())

    def test_arm1_shaped_request_writes_request_and_score(self):
        req = self.request(split=Split.DEV)
        rec = self.svc.score(req, self.gen, "glimmer-base")
        self.assertEqual(sorted(self.files_in()), ["gen_seed0.jsonl", "gen_seed1.jsonl", "per_prompt.jsonl", "request.json", "score.json"])
        self.assertEqual(read_record(self.layout.score_record_path("sc-0001"), ScoreRecord), rec)
        self.assertEqual(read_record(self.layout.score_request_path("sc-0001"), ScoreRequest), req)
        self.assertEqual((rec.score_id, rec.request_id), ("sc-0001", "sc-0001"))
        self.assertEqual((rec.arm, rec.adapter_sha256, rec.verifier, rec.split, rec.phase), (Arm.STOCK, None, Verifier.IFEVALG, Split.DEV, Phase.SEARCH))
        self.assertEqual(rec.n_prompts, 4)
        self.assertEqual([s.seed for s in rec.per_seed], [0, 1])
        self.assertEqual(rec.headline_metric, "prompt_all")
        self.assertEqual(rec.headline_mean(), 1.0)  # every default fake response passes
        self.assertEqual(rec.verifier_pins["fake_verifier"], "1")
        self.assertEqual(rec.created_utc, utc_iso(CLOCK()))
        # the per-prompt file ref is the real file
        self.assertEqual(rec.per_prompt, self.layout.file_ref(self.layout.per_prompt_path("sc-0001")))
        self.assertEqual(self.guard_calls, [])  # dev in the search phase on an evolution assignment needs no guard

    def test_adapter_request_uses_the_adapter_model_name_and_records_its_hash(self):
        req = self.request(arm=Arm.V0, adapter=adapter(), trial_id="t-0001")
        rec = self.svc.score(req, self.gen, "lora-abc")
        self.assertEqual(set(self.gen.models_seen), {"lora-abc"})
        self.assertEqual(rec.adapter_sha256, adapter().sha256)
        self.assertEqual([c[3] for c in self.gen.calls], [0, 1])  # one call per seed, with that seed
        self.assertTrue(all(c[2] == PARAMS for c in self.gen.calls))

    def test_selection_record_metrics_and_per_prompt_flags_hand_computed(self):
        def rule(row, text):
            return [text.startswith("good"), "lower" in text]
        self.ev = FakeVerifierBackend(Verifier.IFEVALG, rule=rule)
        self.svc = self.make_service(verifiers={Verifier.IFEVALG: self.ev})
        answers = {"a-evo-1-s0": "good lower", "a-evo-1-s1": "good", "a-evo-1-s2": "bad lower", "a-evo-1-s3": "bad", "a-evo-1-s4": "good lower"}
        self.gen = FakeGenerator(answers)
        rec = self.svc.score(self.request(split=Split.SELECTION, seeds=(3,)), self.gen, "m")
        m = rec.per_seed[0].metrics
        self.assertAlmostEqual(m["prompt_all"], 2 / 5)
        self.assertAlmostEqual(m["prompt_fraction"], (1 + 0.5 + 0.5 + 0 + 1) / 5)
        rows = [json.loads(l) for l in self.layout.per_prompt_path("sc-0001").read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1], {"prompt_key": "a-evo-1-s1", "seed": 3, "passed": False, "fraction": 0.5, "truncated": False, "error": False,
                                   "constraints": [{"id": IDS[0], "passed": True}, {"id": IDS[1], "passed": False}]})

    def test_test_split_uses_upstream_verifier_and_records_four_metrics(self):
        def rule(row, text):
            return [text == "yes", text in ("yes", "y")]
        self.ib = FakeVerifierBackend(Verifier.IFBENCH_UPSTREAM, rule=rule, loose_rule=lambda row, text: [text in ("yes", "y")] * 2)
        self.svc = self.make_service()
        self.gen = FakeGenerator({"0": "yes", "1": "y", "2": "no"})
        req = self.request(phase=Phase.FINAL, assignment_id=TEST_ASSIGNMENT_ID, split=Split.TEST, final_request_id="f-0001", trial_id="t-0001",
                           arm=Arm.EVOLVED, adapter=adapter(), seeds=(0,))
        rec = self.svc.score(req, self.gen, "m")
        self.assertEqual(rec.verifier, Verifier.IFBENCH_UPSTREAM)
        self.assertEqual(rec.headline_metric, "prompt_strict")
        m = rec.per_seed[0].metrics
        self.assertAlmostEqual(m["prompt_strict"], 1 / 3)
        self.assertAlmostEqual(m["prompt_loose"], 2 / 3)
        self.assertAlmostEqual(m["instruction_strict"], (2 + 1 + 0) / 6)
        self.assertAlmostEqual(m["instruction_loose"], (2 + 2 + 0) / 6)
        rows = [json.loads(l) for l in self.layout.per_prompt_path("sc-0001").read_text().splitlines()]
        self.assertEqual(rows[1], {"prompt_key": "1", "seed": 0, "strict_prompt": False, "loose_prompt": True,
                                   "strict_instructions": [False, True], "loose_instructions": [True, True], "truncated": False, "error": False})
        self.assertEqual(len(self.ib.calls), 1)
        self.assertEqual(self.ib.calls[0][2], "both")

    def test_responses_stay_in_the_private_gen_files_only(self):
        self.gen = FakeGenerator(lambda model, p, params, seed: f"TOPSECRETANSWER-{p.prompt_id}-{seed}")
        rec = self.svc.score(self.request(), self.gen, "m")
        for name in ("per_prompt.jsonl", "score.json", "request.json"):
            self.assertNotIn("TOPSECRETANSWER", (self.layout.score_dir("sc-0001") / name).read_text())
        g = [json.loads(l) for l in self.layout.gen_responses_path("sc-0001", 1).read_text().splitlines()]
        self.assertEqual(g[0], {"prompt_key": "a-evo-1-d0", "text": "TOPSECRETANSWER-a-evo-1-d0-1", "finish_reason": "stop", "completion_tokens": 1, "error": None})
        self.assertEqual(os.stat(self.layout.gen_responses_path("sc-0001", 1)).st_mode & 0o077, 0)
        self.assertEqual(rec.verifier_pins["prompts_sha256"], sha_bytes(self.layout.dev_data(EVO).read_bytes()))

    def test_unicode_line_separators_in_prompts_and_responses_keep_rows_aligned(self):
        weird = "line\u2028separator\u0085inside"
        rows = ifevalg_rows("w", 3)
        rows[1]["prompt"] = "Synthetic prompt with " + weird
        sha = write_jsonl(self.layout.dev_data(EVO), rows)
        self.svc = self.make_service()
        import dataclasses
        a = self.lab.manifest.get(EVO)
        manifest = AssignmentManifest((dataclasses.replace(a, data_revisions=dict(a.data_revisions, dev_sha256=sha)), self.lab.manifest.get(HELD)),
                                      self.lab.manifest.evolution_families, self.lab.manifest.heldout_families, self.lab.manifest.mixed_row_policy,
                                      self.lab.manifest.item_mode, self.lab.manifest.source_revisions, self.lab.manifest.created_utc)
        svc = ScoreService(self.layout, manifest, {Verifier.IFEVALG: self.ev}, guard=self.guard, clock=CLOCK)
        gen = FakeGenerator(lambda model, p, params, seed: "answer " + weird)
        rec = svc.score(self.request(scorer_version=svc.scorer_version, seeds=(0,)), gen, "m")
        self.assertEqual(rec.n_prompts, 3)
        self.assertEqual(self.ev.calls[-1][0], ["w0", "w1", "w2"])
        raw = self.layout.gen_responses_path("sc-0001", 0).read_text()
        self.assertEqual(len(raw.split("\n")), 4)
        self.assertEqual(len(raw.splitlines()), 3)  # plain ASCII on disk: splitlines() agrees with the records
        self.assertEqual(json.loads(raw.splitlines()[1])["text"], "answer " + weird)

    def test_scorer_version_is_stable_and_depends_on_the_pins(self):
        v1 = self.svc.scorer_version
        self.assertRegex(v1, r"^sv1-[0-9a-f]{16}$")
        self.assertEqual(self.make_service().scorer_version, v1)
        other = self.make_service(verifiers={Verifier.IFEVALG: FakeVerifierBackend(Verifier.IFEVALG, pins={"fake_verifier": "2"}), Verifier.IFBENCH_UPSTREAM: self.ib})
        self.assertNotEqual(other.scorer_version, v1)

    def test_request_with_another_scorer_version_is_refused(self):
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(scorer_version="sv1-0000000000000000"), self.gen, "m")
        self.assertEqual(self.gen.calls, [])
        self.assertEqual(self.files_in(), [])


class TruncationAndErrorTests(ScorerTestBase):
    def test_truncated_generation_fails_in_every_arm_and_is_counted(self):
        self.gen = FakeGenerator({"a-evo-1-d1": "a perfectly valid answer"}, truncated=["a-evo-1-d1"])
        rec = self.svc.score(self.request(seeds=(0,)), self.gen, "m")
        agg = rec.per_seed[0]
        self.assertEqual(agg.n_truncated, 1)
        self.assertAlmostEqual(agg.metrics["prompt_all"], 3 / 4)  # without the rule the valid-looking text would pass: 4/4
        self.assertEqual(self.ev.calls[0][1][1], "")  # the verifier never saw the truncated text
        rows = [json.loads(l) for l in self.layout.per_prompt_path("sc-0001").read_text().splitlines()]
        self.assertTrue(rows[1]["truncated"] and not rows[1]["passed"])
        # the raw text is still kept in the private file
        g = [json.loads(l) for l in self.layout.gen_responses_path("sc-0001", 0).read_text().splitlines()]
        self.assertEqual((g[1]["text"], g[1]["finish_reason"]), ("a perfectly valid answer", "length"))

    def test_a_generation_error_aborts_before_anything_is_written(self):
        self.gen = FakeGenerator(errors={"a-evo-1-d2": "timeout"})
        with self.assertRaises(ScorerError) as cm:
            self.svc.score(self.request(), self.gen, "m")
        self.assertIn("timeout", str(cm.exception))
        self.assertEqual(self.files_in(), [])
        self.assertEqual(self.ev.calls, [])

    def test_error_in_a_later_seed_leaves_no_partial_files(self):
        class Flaky(FakeGenerator):
            def generate(inner, model, prompts, params, seed):
                if seed == 1:
                    raise ScorerError("endpoint died")
                return super().generate(model, prompts, params, seed)
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(seeds=(0, 1)), Flaky(), "m")
        self.assertEqual(self.files_in(), [])

    def test_systemic_generator_failure_propagates(self):
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(), FakeGenerator(raise_systemic="dead endpoint"), "m")
        self.assertEqual(self.files_in(), [])

    def test_checker_errors_are_counted_in_the_seed_aggregate(self):
        self.ev = FakeVerifierBackend(Verifier.IFEVALG, error_keys=["a-evo-1-d0", "a-evo-1-d3"])
        self.svc = self.make_service(verifiers={Verifier.IFEVALG: self.ev})
        rec = self.svc.score(self.request(seeds=(0,)), self.gen, "m")
        self.assertEqual(rec.per_seed[0].n_errors, 2)
        self.assertAlmostEqual(rec.per_seed[0].metrics["prompt_all"], 2 / 4)
        rows = [json.loads(l) for l in self.layout.per_prompt_path("sc-0001").read_text().splitlines()]
        self.assertEqual([r["error"] for r in rows], [True, False, False, True])

    def test_generator_returning_the_wrong_prompts_is_refused(self):
        class Wrong(FakeGenerator):
            def generate(inner, model, prompts, params, seed):
                return super().generate(model, list(reversed(prompts)), params, seed)
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(), Wrong(), "m")
        class Short(FakeGenerator):
            def generate(inner, model, prompts, params, seed):
                return super().generate(model, prompts[:-1], params, seed)
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(), Short(), "m")
        self.assertEqual(self.files_in(), [])

    def test_a_verifier_returning_mismatched_results_is_refused(self):
        class Bad(FakeVerifierBackend):
            def score(inner, rows, responses, *, mode="both"):
                return super().score(rows[:-1], responses[:-1], mode=mode)
        self.svc = self.make_service(verifiers={Verifier.IFEVALG: Bad(Verifier.IFEVALG)})
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(), self.gen, "m")
        self.assertEqual(self.files_in(), [])


class DataIntegrityTests(ScorerTestBase):
    def test_a_score_is_never_rewritten(self):
        self.svc.score(self.request(), self.gen, "m")
        before = self.layout.score_record_path("sc-0001").read_bytes()
        n = len(self.gen.calls)
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(), FakeGenerator("different"), "m")
        self.assertEqual(self.layout.score_record_path("sc-0001").read_bytes(), before)
        self.assertEqual(len(self.gen.calls), n)

    def test_modified_data_file_is_refused(self):
        path = self.layout.selection_data(EVO)
        path.write_text(path.read_text() + json.dumps(ifevalg_rows("evil", 1)[0]) + "\n")
        with self.assertRaises(ScorerError) as cm:
            self.svc.score(self.request(split=Split.SELECTION), self.gen, "m")
        self.assertIn("frozen sha256", str(cm.exception))
        self.assertEqual(self.gen.calls, [])

    def test_data_file_replaced_by_symlink_is_refused(self):
        path = self.layout.dev_data(EVO)
        real = path.with_name("real.jsonl")
        path.rename(real)
        path.symlink_to(real)
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(), self.gen, "m")
        self.assertEqual(self.gen.calls, [])

    def test_missing_data_and_malformed_rows_are_scorer_errors(self):
        self.layout.dev_data(EVO).unlink()
        with self.assertRaises(ScorerError):
            self.svc.score(self.request(), self.gen, "m")

    def test_row_without_required_field_or_duplicate_key_is_refused(self):
        for rows in ([{"key": "x", "prompt": "p"}], [ifevalg_rows("d", 1)[0], ifevalg_rows("d", 1)[0]], []):
            d = Lab(Path(tempfile.mkdtemp()))
            self.addCleanup(lambda p=d.layout.root: __import__("shutil").rmtree(p, ignore_errors=True))
            sha = write_jsonl(d.layout.dev_data(EVO), rows)
            a = d.manifest.get(EVO)
            revs = dict(a.data_revisions, dev_sha256=sha)
            manifest = AssignmentManifest((Assignment(EVO, AssignmentRole.EVOLUTION, ("ifeval25",), revs), d.manifest.get(HELD)),
                                          d.manifest.evolution_families, d.manifest.heldout_families, d.manifest.mixed_row_policy,
                                          d.manifest.item_mode, d.manifest.source_revisions, d.manifest.created_utc)
            svc = ScoreService(d.layout, manifest, {Verifier.IFEVALG: self.ev}, guard=self.guard, clock=CLOCK)
            with self.assertRaises(ScorerError):
                svc.score(self.request(scorer_version=svc.scorer_version), self.gen, "m")

    def test_expected_test_hash_is_enforced_when_given(self):
        sha = sha_bytes(self.layout.test_data().read_bytes())
        req = self.request(phase=Phase.FINAL, assignment_id=TEST_ASSIGNMENT_ID, split=Split.TEST, final_request_id="f-0001", trial_id="t-0001", arm=Arm.EVOLVED, adapter=adapter())
        ok = self.make_service(expected_test_sha256=[sha])
        ok.score(req, self.gen, "m")
        bad = self.make_service(expected_test_sha256=["0" * 64])
        # a different adapter: the same logical score would be refused by the no-re-roll marker before the test file is ever read
        req2 = self.request(request_id="sc-0002", phase=Phase.FINAL, assignment_id=TEST_ASSIGNMENT_ID, split=Split.TEST, final_request_id="f-0001", trial_id="t-0001", arm=Arm.EVOLVED, adapter=adapter("5"))
        with self.assertRaisesRegex(ScorerError, "does not match any accepted sha256"):
            bad.score(req2, self.gen, "m")

    def test_missing_verifier_for_the_split_is_a_scorer_error(self):
        svc = self.make_service(verifiers={Verifier.IFEVALG: self.ev})
        req = self.request(phase=Phase.FINAL, assignment_id=TEST_ASSIGNMENT_ID, split=Split.TEST, final_request_id="f-0001", trial_id="t-0001", arm=Arm.EVOLVED, adapter=adapter())
        with self.assertRaises(ScorerError):
            svc.score(dataclasses_replace(req, scorer_version=svc.scorer_version), self.gen, "m")


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


class AccessPolicyTests(ScorerTestBase):
    """Every row of the access-policy table in docs/harness-design.md, plus the rows the table leaves out (deny by default)."""

    def make(self, phase, role, split):
        aid = {"evolution": EVO, "heldout": HELD, ROLE_TEST: TEST_ASSIGNMENT_ID}[role]
        kw = dict(phase=phase, assignment_id=aid, split=split)
        if phase is Phase.FINAL:
            kw.update(final_request_id="f-0001", trial_id="t-0001", arm=Arm.EVOLVED, adapter=adapter())
        return self.request(**kw)

    def test_table_has_the_documented_rows(self):
        S, F = Phase.SEARCH, Phase.FINAL
        want = {
            (S, "evolution", Split.DEV): Access.ALLOW, (S, "evolution", Split.SELECTION): Access.GUARD,
            (S, "heldout", Split.DEV): Access.DENY, (S, "heldout", Split.SELECTION): Access.DENY, (S, "test", Split.TEST): Access.DENY,
            (F, "heldout", Split.DEV): Access.GUARD, (F, "heldout", Split.SELECTION): Access.GUARD,
            (F, "evolution", Split.DEV): Access.DENY, (F, "evolution", Split.SELECTION): Access.DENY, (F, "test", Split.TEST): Access.GUARD,
        }
        self.assertEqual(ACCESS_POLICY, want)

    def test_every_row_behaves_as_the_table_says(self):
        n = 0
        for phase in (Phase.SEARCH, Phase.FINAL):
            for role in ("evolution", "heldout", ROLE_TEST):
                for split in (Split.DEV, Split.SELECTION, Split.TEST):
                    if (split is Split.TEST) != (role == ROLE_TEST):
                        continue  # the contract cannot even build these requests
                    if split is Split.TEST and phase is Phase.SEARCH:
                        continue  # the contract refuses the request: tested below
                    with self.subTest(phase=phase.value, role=role, split=split.value):
                        n += 1
                        self.setUp()
                        decision = ACCESS_POLICY[(phase, role, split)]
                        req = self.make(phase, role, split)
                        if decision is Access.DENY:
                            with self.assertRaises(AccessDenied):
                                self.svc.score(req, self.gen, "m")
                            self.assertEqual((self.guard_calls, self.gen.calls, self.files_in()), ([], [], []))
                        else:
                            rec = self.svc.score(req, self.gen, "m")
                            self.assertEqual(rec.phase, phase)
                            self.assertEqual(len(self.guard_calls), 0 if decision is Access.ALLOW else 1)
                            if decision is Access.GUARD:
                                self.assertIs(self.guard_calls[0], req)
        self.assertGreaterEqual(n, 8)

    def test_the_contract_refuses_a_search_phase_test_request(self):
        with self.assertRaises(Exception):
            self.request(assignment_id=TEST_ASSIGNMENT_ID, split=Split.TEST)

    def test_held_out_dev_is_denied_in_the_search_phase(self):
        req = self.request(assignment_id=HELD, split=Split.DEV)
        with self.assertRaises(AccessDenied):
            self.svc.score(req, self.gen, "m")
        self.assertEqual((self.gen.calls, self.guard_calls), ([], []))

    def test_every_final_phase_score_goes_through_the_guard(self):
        for i, (aid, split) in enumerate([(HELD, Split.DEV), (HELD, Split.SELECTION), (TEST_ASSIGNMENT_ID, Split.TEST)]):
            req = self.final_request(request_id=f"sc-f{i}", assignment_id=aid, split=split)
            self.svc.score(req, self.gen, "m")
        self.assertEqual(len(self.guard_calls), 3)
        self.assertEqual([r.request_id for r in self.guard_calls], ["sc-f0", "sc-f1", "sc-f2"])

    def test_guard_refusal_stops_everything_and_nothing_is_written(self):
        for err in (AccessDenied("search not closed"), FinalComparisonRefused("second request")):
            self.guard_error = err
            for req in (self.final_request(), self.request(split=Split.SELECTION)):
                with self.assertRaises(type(err)):
                    self.svc.score(req, self.gen, "m")
        self.assertEqual((self.gen.calls, self.files_in()), ([], []))

    def test_the_phase_in_a_request_is_not_trusted_by_the_scorer_alone(self):
        # A selection request that claims the search phase while the ledger is in the final phase: the table sends it to the guard,
        # which derives the phase from the ledger and refuses.
        def ledger_phase_guard(request):
            if request.phase is not Phase.FINAL:
                raise AccessDenied("the ledger says the search is closed")
        self.guard = ledger_phase_guard
        svc = ScoreService(self.layout, self.lab.manifest, {Verifier.IFEVALG: self.ev, Verifier.IFBENCH_UPSTREAM: self.ib}, guard=ledger_phase_guard, clock=CLOCK)
        with self.assertRaises(AccessDenied):
            svc.score(self.request(scorer_version=svc.scorer_version, split=Split.SELECTION), self.gen, "m")
        self.assertEqual(self.gen.calls, [])

    def test_a_combination_missing_from_the_table_is_denied(self):
        from unittest import mock
        table = {k: v for k, v in ACCESS_POLICY.items() if k != (Phase.SEARCH, "evolution", Split.DEV)}
        with mock.patch.dict("nightshift.lab.scorer.ACCESS_POLICY", table, clear=True):
            with self.assertRaises(AccessDenied):
                self.svc.score(self.request(), self.gen, "m")
        self.assertEqual(self.gen.calls, [])

    def test_unknown_assignment_is_denied_by_default(self):
        with self.assertRaises(AccessDenied):
            self.svc.score(self.request(assignment_id="a-unknown"), self.gen, "m")
        self.assertEqual(self.gen.calls, [])

    def test_guard_runs_before_any_data_is_read(self):
        self.guard_error = AccessDenied("no")
        self.layout.selection_data(EVO).unlink()  # reading would raise ScorerError
        with self.assertRaises(AccessDenied):
            self.svc.score(self.request(split=Split.SELECTION), self.gen, "m")

    def test_guard_and_clock_must_be_callables(self):
        with self.assertRaises(ScorerError):
            ScoreService(self.layout, self.lab.manifest, {}, guard=None, clock=CLOCK)


if __name__ == "__main__":
    unittest.main()
