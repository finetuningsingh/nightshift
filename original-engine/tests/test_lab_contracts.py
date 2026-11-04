import copy
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

from nightshift.lab import contracts as c
from nightshift.lab.contracts import (
    AdapterFile,
    AdapterRef,
    Arm,
    AssignmentManifest,
    Assignment,
    AssignmentWeight,
    BaseModelRef,
    BudgetCharge,
    BudgetExhaustedEvent,
    BudgetSpec,
    BudgetUsage,
    CandidateRecord,
    ContaminationReport,
    ContractError,
    CostBreakdown,
    Decision,
    DecisionNumbers,
    FileRef,
    FinalAnalysis,
    FinalComparison,
    FinalRequest,
    FinalReveal,
    FrozenFile,
    GateRecord,
    GenerationParams,
    GpuUsage,
    LabConfig,
    LabInit,
    LabKind,
    LabLayout,
    LlmUsage,
    OperatorChange,
    PackCommitted,
    PackEdit,
    PackRef,
    Phase,
    PlannedTrial,
    Proposal,
    Rule,
    ScoreFailure,
    SearchClosed,
    SpreadPlan,
    ScoreCost,
    ScoreRecord,
    ScoreRequest,
    SchemaVersionError,
    SearchReplace,
    SeedAggregate,
    Shape,
    SpreadEstimate,
    SpreadMeasured,
    Split,
    Submission,
    TrainJob,
    TrainResult,
    TrialRecord,
    TrialScore,
    TrialSpec,
)

H64 = "a" * 64
H40 = "b" * 40
NOW = "2026-11-01T12:00:00+00:00"
PACK_REPO = "packs/pack.git"
BASE_COMMIT = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"


def adapter(adapter_id: str = "ad1") -> AdapterRef:
    return AdapterRef.from_files(
        adapter_id,
        [AdapterFile("adapter_config.json", "1" * 64, 1000), AdapterFile("adapter_model.safetensors", "2" * 64, 766_880_464)],
    )


def params() -> GenerationParams:
    return GenerationParams.recommended(8000)


def budget() -> BudgetSpec:
    return BudgetSpec(researcher_tokens=2_000_000, gpu_minutes=120.0, dollars=25.0, wall_seconds=7200.0)


def shape() -> Shape:
    return Shape("a100", 1, 12, 300)


def data_revs(sel: str = "3" * 64) -> dict:
    return {"train_sha256": "1" * 64, "dev_sha256": "2" * 64, "selection_sha256": sel}


def spec(**kw) -> TrialSpec:
    base = dict(
        trial_id="t001",
        phase=Phase.SEARCH,
        arm=None,
        pack_version="v0",
        pack=PackRef(PACK_REPO, H40),
        assignment_id="a-evo-1",
        seed=7,
        budget=budget(),
        shape=shape(),
        base=BaseModelRef("meta-models/Muse-Glimmer-30B", BASE_COMMIT),
        data_revisions=data_revs(),
        scorer_version="s1",
        score_params=params(),
        score_seeds=(0, 1, 2),
        final_request_id=None,
    )
    base.update(kw)
    return TrialSpec(**base)


def final_spec(**kw) -> TrialSpec:
    base = dict(phase=Phase.FINAL, arm=Arm.EVOLVED, assignment_id="a-held-1", final_request_id="f1")
    base.update(kw)
    return spec(**base)


def seed_agg(seed: int, p: float = 0.7) -> SeedAggregate:
    return SeedAggregate(seed, 100, {"prompt_all": p, "prompt_fraction": p + 0.05}, 2, 0)


def ifbench_seed_agg(seed: int, p: float = 0.3) -> SeedAggregate:
    return SeedAggregate(seed, 300, {"prompt_strict": p, "prompt_loose": p + 0.03}, 2, 0)


def score_request(**kw) -> ScoreRequest:
    base = dict(
        request_id="sc1",
        phase=Phase.SEARCH,
        assignment_id="a-evo-1",
        split=Split.SELECTION,
        trial_id="t001",
        arm=None,
        base=spec().base,
        adapter=adapter(),
        seeds=(0, 1),
        params=params(),
        scorer_version="s1",
        reason=None,
        final_request_id=None,
    )
    base.update(kw)
    return ScoreRequest(**base)


def ifbench_request(**kw) -> ScoreRequest:
    base = dict(
        request_id="sc9",
        phase=Phase.FINAL,
        assignment_id=c.TEST_ASSIGNMENT_ID,
        split=Split.TEST,
        trial_id="t-f1",
        arm=Arm.EVOLVED,
        final_request_id="f1",
    )
    base.update(kw)
    return score_request(**base)


def score_record(**kw) -> ScoreRecord:
    base = dict(
        score_id="sc1",
        request_id="sc1",
        phase=Phase.SEARCH,
        assignment_id="a-evo-1",
        split=Split.SELECTION,
        trial_id="t001",
        arm=None,
        final_request_id=None,
        base=spec().base,
        adapter_sha256=adapter().sha256,
        scorer_version="s1",
        verifier=c.Verifier.IFEVALG,
        verifier_pins={"ifevalg_instructions_py": "dfc4629a"},
        params=params(),
        per_seed=(seed_agg(0, 0.7), seed_agg(1, 0.72)),
        n_prompts=100,
        per_prompt=FileRef("scores/sc1/per_prompt.jsonl", H64, 1234),
        headline_metric="prompt_all",
        created_utc=NOW,
    )
    base.update(kw)
    return ScoreRecord(**base)


def ifbench_record(**kw) -> ScoreRecord:
    base = dict(
        score_id="sc9",
        request_id="sc9",
        phase=Phase.FINAL,
        assignment_id=c.TEST_ASSIGNMENT_ID,
        split=Split.TEST,
        trial_id="t-f1",
        arm=Arm.EVOLVED,
        final_request_id="f1",
        verifier=c.Verifier.IFBENCH_UPSTREAM,
        per_seed=(ifbench_seed_agg(0), ifbench_seed_agg(1)),
        n_prompts=300,
        headline_metric="prompt_strict",
    )
    base.update(kw)
    return score_record(**base)


def trial_record(**kw) -> TrialRecord:
    base = dict(
        spec=spec(),
        status=c.TrialStatus.COMPLETED,
        started_utc=NOW,
        finished_utc="2026-11-01T13:00:00+00:00",
        initial_context_sha256=H64,
        budget_used=BudgetUsage(1_500_000, 90.5, 6.5, 5.0, 3600.0),
        submission=Submission("ad1", H64, H40, 0.71),
        adapter=adapter(),
        dev_score_id="sc0",
        selection_score_id="sc1",
        trace=FileRef("trials/t001/trace.jsonl", H64, 10),
        notes=None,
        trial_ledger_anchor=c.LedgerAnchor(12, "c" * 16),
        environment={"coral": "0123dfb939b35228cf2c1fde224cd0561e727408", "opencode": "1.19.3"},
        failure=None,
    )
    base.update(kw)
    return TrialRecord(**base)


def ts(trial: str, assignment: str, score: float, key: str = H64) -> TrialScore:
    return TrialScore(trial, assignment, "sc-" + trial, "prompt_all", "s1", key, score)


def rule(**kw) -> Rule:
    base = dict(
        metric="prompt_all",
        split=Split.SELECTION,
        scorer_version="s1",
        assignments=("a-evo-1",),
        spread=SpreadEstimate(0.02, 4, ("t001", "t002", "t003", "t004", "t005"), 1),
        alpha=0.05,
        min_gain=0.0,
        score_seeds=(0, 1, 2),
        score_params=params(),
        screen_batch_size=3,
        screen_trials_per_candidate=2,
        confirm_trials_per_candidate=3,
        confirm_trials_per_incumbent=3,
        variance_guard_ratio=1.5,
        failed_trial_policy=c.FailedTrialPolicy.INVALIDATES_CANDIDATE,
        assumed_gain=0.05,
        minimum_detectable_gain=0.04,
        degenerate_override=None,
        max_candidates=6,
        stop_after_rejections=None,
        repeated_decisions_note="Screen then confirm; selection results are an adaptive search trace, not evidence of a gain.",
        frozen_utc=NOW,
    )
    base.update(kw)
    return Rule(**base)


def candidate(**kw) -> CandidateRecord:
    base = dict(
        candidate_id="cand-001",
        version_label="cand-001",
        parent=PackRef(PACK_REPO, H40),
        parent_version="v0",
        pack=PackRef(PACK_REPO, "d" * 40),
        proposal=FileRef("candidates/cand-001/proposal.json", H64, 5),
        diff=FileRef("candidates/cand-001/diff.patch", H64, 5),
        prompt=FileRef("candidates/cand-001/prompt.txt", H64, 5),
        response=FileRef("candidates/cand-001/response.txt", H64, 5),
        lever=c.Lever.CONTEXT_NOTES,
        hypothesis="Researchers re-read the full log every turn.",
        reason="Bound the notes summary.",
        lint_passed=True,
        lint_messages=(),
        proposer_model="muse-wonder-1.0",
        proposer_served_model="muse-wonder-1.0",
        fallback_used=False,
        propose_usd=0.31,
        created_utc=NOW,
    )
    base.update(kw)
    return CandidateRecord(**base)


def decision(**kw) -> Decision:
    base = dict(
        candidate_id="cand-001",
        rule_sha256=rule().sha256(),
        stage=c.DecisionStage.CONFIRM,
        round_index=0,
        outcome=c.Outcome.ACCEPTED,
        reason="delta 0.05 exceeds margin 0.03",
        incumbent_version="v0",
        candidate_version="cand-001",
        incumbent_scores=(ts("t001", "a-evo-1", 0.7), ts("t002", "a-evo-1", 0.72)),
        candidate_scores=(ts("t010", "a-evo-1", 0.77),),
        screen_delta=0.05,
        delta=0.06,
        margin=0.03,
        df=8,
        sigma_used=0.02,
        cost=CostBreakdown(0.3, 5.0, 8.0, 1.0),
        decided_utc=NOW,
    )
    base.update(kw)
    return Decision(**base)


def screen_decision(**kw) -> Decision:
    base = dict(
        stage=c.DecisionStage.SCREEN,
        outcome=c.Outcome.REJECTED,
        reason="lost its screening round",
        screen_delta=0.01,
        delta=None,
        margin=None,
        df=None,
        sigma_used=None,
    )
    base.update(kw)
    return decision(**base)


def train_job(**kw) -> TrainJob:
    base = dict(
        job_id="j1",
        trial_id="t001",
        recipe=FileRef("trials/t001/recipe.tar", H64, 100),
        entrypoint=("bash", "run.sh"),
        max_gpu_minutes=60.0,
        shape=shape(),
        dev_request=score_request(split=Split.DEV, adapter=None, request_id="j1-dev"),
    )
    base.update(kw)
    return TrainJob(**base)


def train_result(**kw) -> TrainResult:
    base = dict(
        job_id="j1",
        trial_id="t001",
        status=c.JobStatus.OK,
        adapter_id="ad1",
        adapter=adapter(),
        dev_score_id="sc0",
        gpu=GpuUsage(55.0, 1.2, ("rent1",)),
        timings={"load_s": 300.0, "train_s": 1200.0},
        log=None,
        train_data=FileRef("trials/t001/train_data/j1.jsonl", H64, 4000),
        failure=None,
    )
    base.update(kw)
    return TrainResult(**base)


def manifest(**kw) -> AssignmentManifest:
    base = dict(
        assignments=(
            Assignment("a-evo-1", c.AssignmentRole.EVOLUTION, ("ifeval25",), data_revs()),
            Assignment("a-held-1", c.AssignmentRole.HELDOUT, ("copy",), data_revs("4" * 64)),
        ),
        evolution_families=("ifeval25", "counting"),
        heldout_families=("copy", "firstlast"),
        mixed_row_policy=c.MixedRowPolicy.DROPPED,
        item_mode=c.ItemMode.SALTED_DRAW,
        source_revisions={
            "ifrlvr_parquet_sha256": "5" * 64,
            "ifbench_commit": "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d",
            "open_instruct_commit": "b781e782a8fc530577f235aefe282dd046cb8ad2",
            "split_salt_sha256": "6" * 64,
        },
        created_utc=NOW,
    )
    base.update(kw)
    return AssignmentManifest(**base)


def final_analysis(n_assignments: int = 1, per_cell: int = 5, **kw) -> FinalAnalysis:
    ids = ["a-held-1", "a-held-2"][:n_assignments]
    planned = tuple(
        PlannedTrial(f"f-{arm.value}-{aid}-{k}", arm, aid, 100 + k) for arm in (Arm.V0, Arm.EVOLVED) for aid in ids for k in range(per_cell)
    )
    base = dict(
        heldout_metric="prompt_all",
        test_metric="prompt_strict",
        assignments=tuple(AssignmentWeight(i, 1.0 / n_assignments) for i in ids),
        planned_trials=planned,
        confidence=0.95,
        method=c.ComparisonMethod.WELCH if n_assignments == 1 else c.ComparisonMethod.BLOCKED_T,
        score_seeds=(0, 1, 2),
        score_params=params(),
        failed_trial_policy=c.FailedTrialPolicy.INVALIDATES_CANDIDATE,
        retry_policy="Retry only after an infrastructure failure that revealed no score.",
        budget_sha256=c.record_sha256(budget()),
        frozen_utc=NOW,
    )
    base.update(kw)
    return FinalAnalysis(**base)


def lab_config(**kw) -> LabConfig:
    base = dict(
        lab_id="lab1",
        control_host="ax43",
        researcher_host="ax41",
        researcher_user="researcher",
        base=spec().base,
        shape=shape(),
        pins={"coral_commit": "0123dfb939b35228cf2c1fde224cd0561e727408", "opencode_version": "1.19.3",
              "ifbench_commit": "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d", "open_instruct_commit": "b781e782a8fc530577f235aefe282dd046cb8ad2"},
        secret_env={"meta_api_key": "META_API_KEY", "thunder_api_key": "THUNDER_API_KEY"},
        upstream_base_url="https://api.meta.ai/v1",
        lab_budget_usd=150.0,
        affordable_trials=60,
    )
    base.update(kw)
    return LabConfig(**base)


def final_comparison(**kw) -> FinalComparison:
    base = dict(
        analysis_sha256=H64,
        split=Split.SELECTION,
        metric="prompt_all",
        contrast="evolved_minus_v0",
        method=c.ComparisonMethod.WELCH,
        delta=0.03,
        ci_low=-0.01,
        ci_high=0.07,
        level=0.95,
        df=7.2,
        n_assignments=1,
        trials_v0=5,
        trials_evolved=5,
        mean_v0=0.70,
        mean_evolved=0.73,
        includes_zero=True,
        note="Within one assignment; does not generalise across assignments.",
    )
    base.update(kw)
    return FinalComparison(**base)


# one factory per persisted (versioned) record or ledger payload
PERSISTED = {
    "BudgetSpec": budget,
    "AdapterRef": adapter,
    "AssignmentManifest": manifest,
    "TrialSpec": spec,
    "TrialRecord": trial_record,
    "ScoreRequest": score_request,
    "ScoreRecord": score_record,
    "ScoreCost": lambda: ScoreCost("sc1", GpuUsage(20.0, 0.5, ("rent9",))),
    "ScoreFailure": lambda: ScoreFailure("sc2", Split.SELECTION, "a-evo-1", "t001", H64, "vLLM crashed before the first response", False, NOW),
    "ContaminationReport": lambda: ContaminationReport("t001", "j1", FileRef("trials/t001/train_data/j1.jsonl", H64, 4000), 120, 0, 0, 13, (H64,), True, NOW),
    "Rule": rule,
    "SpreadPlan": lambda: SpreadPlan("prompt_all", "s1", ("a-evo-1",), 5, NOW),
    "SpreadMeasured": lambda: SpreadMeasured("prompt_all", "s1", 5, 0, tuple(ts(f"t00{i}", "a-evo-1", 0.70 + 0.01 * i) for i in range(1, 6)),
                                             SpreadEstimate(0.014, 4, ("t001", "t002", "t003", "t004", "t005"), 1)),
    "Proposal": lambda: Proposal(
        "h", c.Lever.TOOLING, "s", "e", "r",
        (PackEdit("task.yaml", c.EditOp.SEARCH_REPLACE, None, (SearchReplace("a", "b"),)),
         PackEdit("notes/seed.md", c.EditOp.REPLACE_FILE, "text", ())),
    ),
    "CandidateRecord": candidate,
    "Decision": decision,
    "OperatorChange": lambda: OperatorChange("proposer_prompt", "Shortened the lever list.", 2, NOW),
    "LabInit": lambda: LabInit("lab1", NOW, "ax43"),
    "LabConfig": lab_config,
    "FrozenFile": lambda: FrozenFile("rule", FileRef("frozen/rule.json", H64, 10)),
    "GateRecord": lambda: GateRecord(c.Gate.TERMS_10_1_IX, c.GateOutcome.PROCEED_WITH_RISK, "Proceeding; risk stated.", "Ankit", NOW),
    "PackCommitted": lambda: PackCommitted("v0", PackRef(PACK_REPO, H40), None, None),
    "SearchClosed": lambda: SearchClosed("cand-002", PackRef(PACK_REPO, "d" * 40), 6, 29, "max_candidates reached", NOW),
    "FinalAnalysis": final_analysis,
    "FinalRequest": lambda: FinalRequest("f1", Arm.V0, "v0", PackRef(PACK_REPO, H40), H64, (Split.SELECTION, Split.TEST), None, None, NOW),
    "FinalReveal": lambda: FinalReveal(H64, 10, NOW),
    "FinalComparison": final_comparison,
    "BudgetCharge": lambda: BudgetCharge("t001", c.Resource.GPU_MINUTES, 12.5, "gpu:rent1:3", {"rate": 1.31, "ok": True, "who": None}),
    "BudgetExhaustedEvent": lambda: BudgetExhaustedEvent("t001", "gpu_minutes", 120.0, 121.0, "stop runtime"),
    "TrainJob": train_job,
    "TrainResult": train_result,
}


def roundtrip(obj):
    text = c.dumps_record(obj)
    return type(obj).from_dict(c.loads_strict(text))


class RoundTripTests(unittest.TestCase):
    def test_every_persisted_record_round_trips_through_json_text(self):
        for name, make in PERSISTED.items():
            with self.subTest(name):
                obj = make()
                self.assertEqual(roundtrip(obj), obj)
                self.assertEqual(obj.to_dict()["schema_version"], c.SCHEMA_VERSION)

    def test_to_dict_is_strict_json_with_no_python_objects(self):
        for name, make in PERSISTED.items():
            with self.subTest(name):
                json.dumps(make().to_dict(), allow_nan=False)

    def test_embedded_value_objects_round_trip(self):
        for obj in (PackRef(PACK_REPO, H40), FileRef("a/b", H64, 0), BaseModelRef("x/y", H40), params(), BudgetUsage(1, 2.0, 3.0, 4.0, 5.0),
                    CostBreakdown(1, 2, 3, 4), seed_agg(0), SpreadEstimate(0.1, 1, ("a", "b"), 1), ts("t", "a", 0.5),
                    Shape("h100", 1, 12, 300), GpuUsage(1.0, 2.0, ("r",)), Submission("a", H64, H40, None), c.LedgerAnchor(0, "f" * 16),
                    AdapterFile("adapter_config.json", H64, 3), LlmUsage(10, 5, 2), AssignmentWeight("a", 0.5),
                    PlannedTrial("t", Arm.V0, "a", 1)):
            with self.subTest(type(obj).__name__):
                self.assertEqual(type(obj).from_dict(json.loads(json.dumps(obj.to_dict()))), obj)

    def test_dumps_is_deterministic_and_canonical_hash_ignores_key_order(self):
        a = rule()
        self.assertEqual(c.dumps_record(a), c.dumps_record(rule()))
        self.assertEqual(c.canonical_json({"b": 1, "a": 2}), c.canonical_json({"a": 2, "b": 1}))
        self.assertEqual(a.sha256(), rule().sha256())
        self.assertNotEqual(a.sha256(), rule(min_gain=0.01).sha256())

    def test_write_and_read_record_atomic_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "frozen" / "rule.json"
        digest = c.write_record(p, rule())
        self.assertEqual(digest, c.sha256_file(p))
        self.assertEqual(c.read_record(p, Rule), rule())
        self.assertEqual([x.name for x in p.parent.iterdir()], ["rule.json"])

    def test_adapter_hash_is_over_the_canonical_listing(self):
        a = adapter()
        lines = "".join(f"{f.sha256} {f.size_bytes} {f.name}\n" for f in a.files)
        self.assertEqual(a.sha256, c.sha256_text(lines))
        self.assertEqual(a.size_bytes, 1000 + 766_880_464)
        self.assertEqual([f.name for f in a.files], sorted(f.name for f in a.files))

    def test_adapter_ref_is_lab_relative_and_has_an_id(self):
        a = adapter("ad7")
        self.assertEqual(a.path, "artifacts/adapters/ad7")
        self.assertEqual(a.adapter_id, "ad7")
        self.assertEqual(a.resolve("/lab"), Path("/lab/artifacts/adapters/ad7"))
        for bad in ("/artifacts/adapters/a", "artifacts/adapters", "artifacts/adapters/a/b", "adapters/a", "artifacts/other/a", "../artifacts/adapters/a"):
            with self.subTest(bad):
                d = a.to_dict()
                d["path"] = bad
                with self.assertRaises(ContractError):
                    AdapterRef.from_dict(d)

    def test_adapter_ref_from_dir_and_verify(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        (d / "adapter_config.json").write_text('{"r": 32}')
        (d / "adapter_model.safetensors").write_bytes(b"\x00\x01weights")
        (d / "checkpoint-3").mkdir()
        (d / "checkpoint-3" / "noise.bin").write_bytes(b"ignored")
        ref = c.adapter_ref_from_dir(d, adapter_id="ad9")
        self.assertEqual(ref.path, "artifacts/adapters/ad9")
        self.assertEqual(ref.size_bytes, 9 + 9)
        c.verify_adapter_dir(ref, d)
        root = Path(tmp.name) / "root"
        (root / "artifacts" / "adapters").mkdir(parents=True)
        (root / "artifacts" / "adapters" / "ad9").mkdir()
        for name in c.ADAPTER_FILES:
            (root / "artifacts" / "adapters" / "ad9" / name).write_bytes((d / name).read_bytes())
        c.verify_adapter_dir(ref, root=root)  # the lab root may move; the ref stays valid
        with self.assertRaises(ContractError):
            c.verify_adapter_dir(ref)
        (d / "adapter_model.safetensors").write_bytes(b"tampered!")
        with self.assertRaises(ContractError):
            c.verify_adapter_dir(ref, d)
        (d / "adapter_model.safetensors").unlink()
        with self.assertRaises(ContractError):
            c.adapter_ref_from_dir(d, adapter_id="ad9")

    def test_adapter_ref_from_dir_refuses_a_symlinked_weight_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name) / "ad"
        d.mkdir()
        secret = Path(tmp.name) / "secret"
        secret.write_bytes(b"ground truth")
        (d / "adapter_config.json").write_text("{}")
        os.symlink(secret, d / "adapter_model.safetensors")
        with self.assertRaisesRegex(ContractError, "regular file"):
            c.adapter_ref_from_dir(d, adapter_id="ad1")


class EnumAndTimeTests(unittest.TestCase):
    def test_every_enum_formats_as_its_value_on_every_python(self):
        enums = [v for v in vars(c).values() if isinstance(v, type) and issubclass(v, c._StrEnum) and v is not c._StrEnum]
        self.assertGreaterEqual(len(enums), 15)
        for cls in enums:
            for m in cls:
                with self.subTest(m):
                    self.assertEqual(str(m), m.value)
                    self.assertEqual(f"{m}", m.value)
                    self.assertEqual("{}".format(m), m.value)
                    self.assertEqual(format(m, ">30"), m.value.rjust(30))
        self.assertEqual(f"{Arm.V0}-{Phase.SEARCH}", "v0-search")
        self.assertFalse(hasattr(c, "Confidence"))
        self.assertEqual([m.value for m in c.MixedRowPolicy], ["dropped"])

    def test_utc_iso_is_always_plus_zero_seconds_precision_and_accepted_by_the_validators(self):
        self.assertEqual(c.utc_iso(0), "1970-01-01T00:00:00+00:00")
        self.assertEqual(c.utc_iso(1_793_534_400.9), "2026-11-01T12:00:00+00:00")
        LabInit("lab1", c.utc_iso(1_793_534_400.9), "ax43")
        with self.assertRaises(ContractError):
            c.utc_iso(-1)

    def test_owner_tag_and_scorer_version_are_deterministic(self):
        key = "ssh-ed25519 AAAAC3Nza lab@ax43"
        self.assertEqual(c.owner_tag_of_key(key), c.owner_tag_of_key("ssh-ed25519 AAAAC3Nza other-comment"))
        self.assertNotEqual(c.owner_tag_of_key(key), c.owner_tag_of_key("ssh-ed25519 AAAAC3Nzb lab@ax43"))
        self.assertEqual(len(c.owner_tag_of_key(key)), 16)
        with self.assertRaises(ContractError):
            c.owner_tag_of_key("justonefield")
        a = c.scorer_version_of({"ifbench": "1c40f0c", "ifevalg": "b781e78"}, ["prompt_all", "prompt_strict"], 1)
        self.assertEqual(a, c.scorer_version_of({"ifevalg": "b781e78", "ifbench": "1c40f0c"}, ["prompt_strict", "prompt_all"], 1))
        self.assertTrue(a.startswith("sv1-"))
        self.assertNotEqual(a, c.scorer_version_of({"ifbench": "other", "ifevalg": "b781e78"}, ["prompt_all", "prompt_strict"], 1))
        self.assertNotEqual(a, c.scorer_version_of({"ifbench": "1c40f0c", "ifevalg": "b781e78"}, ["prompt_all", "prompt_strict"], 2))


class RejectionTests(unittest.TestCase):
    def test_unknown_and_missing_fields_are_rejected_for_every_record(self):
        for name, make in PERSISTED.items():
            cls = type(make())
            good = make().to_dict()
            with self.subTest(name + " unknown"):
                bad = dict(good, surprise=1)
                with self.assertRaisesRegex(ContractError, "unknown field"):
                    cls.from_dict(bad)
            for key in list(good):
                with self.subTest(f"{name} missing {key}"):
                    bad = {k: v for k, v in good.items() if k != key}
                    with self.assertRaises(ContractError):
                        cls.from_dict(bad)

    def test_unknown_field_inside_a_nested_object_is_rejected(self):
        d = spec().to_dict()
        d["pack"]["extra"] = 1
        with self.assertRaisesRegex(ContractError, "unknown field"):
            TrialSpec.from_dict(d)

    def test_schema_version_must_match(self):
        for bad in (0, 2, "1", True, None):
            with self.subTest(bad):
                d = budget().to_dict()
                d["schema_version"] = bad
                with self.assertRaises(ContractError):
                    BudgetSpec.from_dict(d)
        d = budget().to_dict()
        d["schema_version"] = 2
        with self.assertRaises(SchemaVersionError):
            BudgetSpec.from_dict(d)
        with self.assertRaises(SchemaVersionError):
            BudgetSpec(1, 1.0, 1.0, 1.0, schema_version=9)

    def test_non_object_input_is_rejected(self):
        for bad in (None, [], "x", 3):
            with self.assertRaises(ContractError):
                BudgetSpec.from_dict(bad)

    def test_negative_budgets_are_rejected(self):
        for kw in ({"researcher_tokens": -1}, {"gpu_minutes": -0.5}, {"dollars": -1}, {"wall_seconds": -1}):
            with self.subTest(kw):
                with self.assertRaises(ContractError):
                    dataclasses.replace(budget(), **kw)
                d = budget().to_dict()
                d.update(kw)
                with self.assertRaises(ContractError):
                    BudgetSpec.from_dict(d)
        with self.assertRaises(ContractError):
            BudgetUsage(0, -1.0, 0, 0, 0)
        with self.assertRaises(ContractError):
            CostBreakdown(0, 0, -0.01, 0)

    def test_non_finite_numbers_are_rejected_in_memory_and_in_json_text(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad):
                with self.assertRaises(ContractError):
                    BudgetSpec(1, bad, 1.0, 1.0)
                with self.assertRaises(ContractError):
                    SeedAggregate(0, 10, {"m": bad}, 0, 0)
                with self.assertRaises(ContractError):
                    BudgetCharge("t", c.Resource.GPU_USD, 1.0, "c", {"x": bad})
        for text in ('{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'):
            with self.assertRaisesRegex(ContractError, "non-finite"):
                c.loads_strict(text)
        with self.assertRaisesRegex(ContractError, "duplicate"):
            c.loads_strict('{"a": 1, "a": 2}')
        with self.assertRaises(ValueError):
            c.canonical_json({"x": float("nan")})

    def test_wrong_types_are_rejected(self):
        d = budget().to_dict()
        d["researcher_tokens"] = 1.5
        with self.assertRaises(ContractError):
            BudgetSpec.from_dict(d)
        d = budget().to_dict()
        d["gpu_minutes"] = True
        with self.assertRaises(ContractError):
            BudgetSpec.from_dict(d)
        d = budget().to_dict()
        d["dollars"] = "5"
        with self.assertRaises(ContractError):
            BudgetSpec.from_dict(d)

    def test_identifiers_hashes_paths_and_times_are_validated(self):
        for bad_id in ("", "../x", "a/b", " a", "a" * 65, ".hidden", "a\x00b"):
            with self.subTest(bad_id):
                with self.assertRaises(ContractError):
                    spec(trial_id=bad_id)
        with self.assertRaises(ContractError):
            PackRef(PACK_REPO, "abc")
        with self.assertRaises(ContractError):
            PackRef(PACK_REPO, "B" * 40)
        with self.assertRaises(ContractError):
            PackRef("/lab/packs/pack.git", H40)  # control-host absolute paths are never persisted
        self.assertEqual(PackRef(PACK_REPO, H40).resolve("/lab"), Path("/lab/packs/pack.git"))
        for bad_path in ("/etc/passwd", "../x", "a/../b", "a//b", "a\\b", "", ".git/config", "a/./b"):
            with self.subTest(bad_path):
                with self.assertRaises(ContractError):
                    FileRef(bad_path, H64, 1)
        with self.assertRaises(ContractError):
            LabInit("lab1", "2026-11-01T12:00:00", "ax43")  # no timezone
        with self.assertRaises(ContractError):
            LabInit("lab1", "yesterday", "ax43")

    def test_adapter_ref_integrity(self):
        a = adapter()
        d = a.to_dict()
        d["sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "canonical listing"):
            AdapterRef.from_dict(d)
        d = a.to_dict()
        d["size_bytes"] += 1
        with self.assertRaisesRegex(ContractError, "sum"):
            AdapterRef.from_dict(d)
        with self.assertRaises(ContractError):
            AdapterRef.from_files("x", [AdapterFile("adapter_config.json", H64, 1)])  # weights missing
        with self.assertRaises(ContractError):
            AdapterRef.from_files("x", [AdapterFile("adapter_config.json", H64, 1), AdapterFile("adapter_config.json", H64, 1),
                                        AdapterFile("adapter_model.safetensors", H64, 1)])

    def test_trial_spec_phase_and_arm_consistency(self):
        with self.assertRaises(ContractError):
            final_spec(arm=None)
        with self.assertRaises(ContractError):
            final_spec(arm=Arm.STOCK)  # arm 1 has no trial
        with self.assertRaises(ContractError):
            spec(phase=Phase.SEARCH, arm=Arm.V0)
        self.assertEqual(final_spec().arm.number, 3)
        with self.assertRaises(ContractError):
            spec(score_seeds=())
        with self.assertRaises(ContractError):
            spec(score_seeds=(1, 1))

    def test_trial_spec_final_request_id_and_test_assignment(self):
        with self.assertRaisesRegex(ContractError, "final_request_id"):
            final_spec(final_request_id=None)
        with self.assertRaisesRegex(ContractError, "final_request_id"):
            spec(final_request_id="f1")
        with self.assertRaises(ContractError):
            spec(assignment_id=c.TEST_ASSIGNMENT_ID)
        with self.assertRaisesRegex(ContractError, "keys"):
            spec(data_revisions={"if_rlvr": "2e3a77407b7fce69f95b248d64a884e3ae1c2423"})
        with self.assertRaisesRegex(ContractError, "keys"):
            spec(data_revisions={"train_sha256": H64})

    def test_comparability_key_changes_with_every_comparable_thing_and_nothing_else(self):
        k = c.comparability_key(spec(), "prompt_all")
        self.assertEqual(k, c.comparability_key(spec(trial_id="t002", seed=99), "prompt_all"))
        self.assertNotEqual(k, c.comparability_key(spec(), "prompt_fraction"))
        self.assertNotEqual(k, c.comparability_key(spec(scorer_version="s2"), "prompt_all"))
        self.assertNotEqual(k, c.comparability_key(spec(score_seeds=(0, 1)), "prompt_all"))
        self.assertNotEqual(k, c.comparability_key(spec(budget=BudgetSpec(2_000_000, 121.0, 25.0, 7200.0)), "prompt_all"))
        self.assertNotEqual(k, c.comparability_key(spec(shape=Shape("h100", 1, 12, 300)), "prompt_all"))
        self.assertNotEqual(k, c.comparability_key(spec(data_revisions=data_revs("9" * 64)), "prompt_all"))
        self.assertNotEqual(k, c.comparability_key(spec(score_params=GenerationParams.recommended(4000)), "prompt_all"))
        self.assertNotEqual(k, c.comparability_key(spec(base=BaseModelRef("meta-models/Muse-Glimmer-30B", "c" * 40)), "prompt_all"))

    def test_trial_record_status_requirements(self):
        with self.assertRaises(ContractError):
            trial_record(adapter=None)
        with self.assertRaises(ContractError):
            trial_record(selection_score_id=None)
        with self.assertRaises(ContractError):
            trial_record(status=c.TrialStatus.FAILED, failure=None)
        with self.assertRaisesRegex(ContractError, "initial_context_sha256"):
            trial_record(initial_context_sha256=None)
        ok = trial_record(status=c.TrialStatus.NO_SUBMISSION, submission=None, adapter=None, dev_score_id=None, selection_score_id=None,
                          failure="the team never saved an adapter")
        self.assertEqual(roundtrip(ok), ok)

    def test_a_trial_that_died_before_checkout_is_a_valid_aborted_record(self):
        aborted = trial_record(status=c.TrialStatus.ABORTED, initial_context_sha256=None, submission=None, adapter=None, dev_score_id=None,
                               selection_score_id=None, trial_ledger_anchor=None, failure="coordinator restarted before the workspace was checked out")
        self.assertEqual(roundtrip(aborted), aborted)
        contaminated = trial_record(status=c.TrialStatus.CONTAMINATED, selection_score_id=None, failure="13-gram overlap with a held-out prompt")
        self.assertEqual(roundtrip(contaminated), contaminated)

    def test_trial_record_submission_must_name_the_adapter(self):
        with self.assertRaisesRegex(ContractError, "adapter"):
            trial_record(submission=Submission("other", H64, H40, 0.7))

    def test_score_request_rules(self):
        with self.assertRaisesRegex(ContractError, "final phase"):
            score_request(split=Split.TEST, phase=Phase.SEARCH)
        with self.assertRaises(ContractError):
            score_request(arm=Arm.STOCK)  # adapter present
        with self.assertRaises(ContractError):
            score_request(arm=Arm.V0, adapter=None)
        ok = ifbench_request(arm=Arm.STOCK, adapter=None, trial_id=None)
        self.assertEqual(roundtrip(ok), ok)

    def test_score_request_final_request_id_and_test_assignment_rules(self):
        with self.assertRaisesRegex(ContractError, "final_request_id"):
            ifbench_request(final_request_id=None)  # every final-phase score belongs to a FinalRequest
        with self.assertRaisesRegex(ContractError, "final_request_id"):
            score_request(final_request_id="f1")  # a search-phase score has none
        with self.assertRaisesRegex(ContractError, "test split"):
            ifbench_request(assignment_id="a-held-1")  # the test set belongs to no assignment
        with self.assertRaisesRegex(ContractError, "test split"):
            score_request(assignment_id=c.TEST_ASSIGNMENT_ID)  # no other split may use the test id
        final_dev = score_request(phase=Phase.FINAL, assignment_id="a-held-1", split=Split.DEV, arm=None, adapter=None, final_request_id="f1")
        self.assertEqual(roundtrip(final_dev), final_dev)  # a dev template in the final phase has arm None and builds

    def test_score_record_invariants(self):
        with self.assertRaisesRegex(ContractError, "verified with"):
            score_record(verifier=c.Verifier.IFBENCH_UPSTREAM)  # selection must use IFEvalG
        with self.assertRaisesRegex(ContractError, "verified with"):
            score_record(split=Split.TEST, phase=Phase.FINAL)  # test must use upstream IFBench
        with self.assertRaises(ContractError):
            score_record(per_seed=(seed_agg(0), seed_agg(0)))
        with self.assertRaises(ContractError):
            score_record(n_prompts=99)
        with self.assertRaises(ContractError):
            score_record(headline_metric="missing")
        with self.assertRaises(ContractError):
            SeedAggregate(0, 10, {"m": 1.2}, 0, 0)  # pass rates stay in [0, 1]
        with self.assertRaises(ContractError):
            SeedAggregate(0, 10, {"m": 0.5}, 11, 0)
        self.assertAlmostEqual(score_record().headline_mean(), 0.71)

    def test_score_id_equals_request_id_and_metrics_belong_to_the_split(self):
        with self.assertRaisesRegex(ContractError, "score_id must equal request_id"):
            score_record(score_id="sc1", request_id="r1")
        rec = ifbench_record()
        self.assertEqual(roundtrip(rec), rec)
        self.assertAlmostEqual(rec.headline_mean(), 0.3)
        with self.assertRaisesRegex(ContractError, "not a selection metric|not a selection"):
            score_record(headline_metric="prompt_strict", per_seed=(SeedAggregate(0, 100, {"prompt_strict": 0.3}, 0, 0),))
        with self.assertRaisesRegex(ContractError, "metric"):
            score_record(per_seed=(SeedAggregate(0, 100, {"prompt_all": 0.5, "prompt_loose": 0.5}, 0, 0),))
        with self.assertRaisesRegex(ContractError, "test assignment id|test split"):
            ifbench_record(assignment_id="a-held-1")

    def test_rule_must_use_selection_and_have_a_spread(self):
        with self.assertRaises(ContractError):
            rule(split=Split.DEV)
        with self.assertRaises(ContractError):
            rule(split=Split.TEST)
        with self.assertRaises(ContractError):
            SpreadEstimate(0.1, 3, ("t1", "t2"), 1)  # df must be trials - cells
        with self.assertRaises(ContractError):
            SpreadEstimate(0.1, 0, ("t1", "t2"), 1)
        with self.assertRaises(ContractError):
            SpreadEstimate(0.1, 2, ("t1", "t2", "t3"), 2)  # 3 trials in 2 cells give df 1, not 2
        self.assertEqual(SpreadEstimate(0.1, 2, ("a1", "a2", "b1", "b2"), 2).df, 2)
        with self.assertRaises(ContractError):
            rule(max_candidates=0)
        with self.assertRaises(ContractError):
            rule(repeated_decisions_note="  ")
        with self.assertRaises(ContractError):
            rule(min_gain=-0.01)

    def test_rule_screen_confirm_parameters_and_guards(self):
        with self.assertRaisesRegex(ContractError, "alpha"):
            rule(alpha=0.0)
        with self.assertRaisesRegex(ContractError, "alpha"):
            rule(alpha=0.6)
        with self.assertRaisesRegex(ContractError, "screen_batch_size"):
            rule(screen_batch_size=c.MAX_SCREEN_BATCH + 1)
        with self.assertRaises(ContractError):
            rule(confirm_trials_per_candidate=1)  # a candidate sd needs two trials
        with self.assertRaisesRegex(ContractError, "variance_guard_ratio"):
            rule(variance_guard_ratio=0.9)
        with self.assertRaisesRegex(ContractError, "minimum detectable gain"):
            rule(minimum_detectable_gain=0.06, assumed_gain=0.05)
        with self.assertRaisesRegex(ContractError, "pooled degrees of freedom"):
            rule(spread=SpreadEstimate(0.02, 2, ("t1", "t2", "t3"), 1))
        self.assertEqual(rule().pooled_df(), 8)
        self.assertEqual(rule().planned_search_trials(), 5 + 6 * 2 * 1 + 2 * (3 + 3) * 1)

    def test_rule_with_zero_sigma_and_zero_min_gain_needs_a_recorded_override(self):
        flat = SpreadEstimate(0.0, 4, ("t001", "t002", "t003", "t004", "t005"), 1)
        with self.assertRaisesRegex(ContractError, "override"):
            rule(spread=flat)
        self.assertEqual(roundtrip(rule(spread=flat, min_gain=0.01)).min_gain, 0.01)
        ok = rule(spread=flat, degenerate_override="five identical v0 scores: quantised at 1/100, accepted by the owner")
        self.assertEqual(roundtrip(ok), ok)

    def test_spread_plan_and_measured_account_for_every_planned_trial(self):
        with self.assertRaisesRegex(ContractError, "planned"):
            SpreadMeasured("prompt_all", "s1", 6, 0, tuple(ts(f"t00{i}", "a-evo-1", 0.7) for i in range(1, 6)),
                           SpreadEstimate(0.0, 4, ("t001", "t002", "t003", "t004", "t005"), 1))
        counted = SpreadMeasured("prompt_all", "s1", 6, 1, tuple(ts(f"t00{i}", "a-evo-1", 0.7) for i in range(1, 6)),
                                 SpreadEstimate(0.0, 4, ("t001", "t002", "t003", "t004", "t005"), 1))
        self.assertEqual(counted.n_failed, 1)
        with self.assertRaisesRegex(ContractError, "not listed"):
            SpreadMeasured("prompt_all", "s1", 2, 0, (ts("t001", "a-evo-1", 0.7), ts("t002", "a-evo-1", 0.7)),
                           SpreadEstimate(0.0, 1, ("t001", "t999"), 1))
        with self.assertRaises(ContractError):
            SpreadPlan("prompt_strict", "s1", ("a-evo-1",), 5, NOW)  # not a selection metric
        with self.assertRaises(ContractError):
            SpreadPlan("prompt_all", "s1", ("a-evo-1",), 1, NOW)

    def test_decision_outcome_must_agree_with_the_rule(self):
        with self.assertRaisesRegex(ContractError, "contradicts"):
            decision(outcome=c.Outcome.ACCEPTED, delta=0.01, margin=0.03)
        with self.assertRaisesRegex(ContractError, "contradicts"):
            decision(outcome=c.Outcome.REJECTED, delta=0.06, margin=0.03)
        rejected = decision(outcome=c.Outcome.REJECTED, delta=0.03, margin=0.03, reason="no gain beyond margin")
        self.assertFalse(rejected.accepted)
        invalid = decision(outcome=c.Outcome.INVALID, delta=None, margin=None, df=None, sigma_used=None, screen_delta=None, reason="trials not comparable")
        self.assertEqual(roundtrip(invalid), invalid)
        with self.assertRaises(ContractError):
            decision(outcome=c.Outcome.INVALID)  # numbers present
        self.assertTrue(decision().accepted)

    def test_decision_stages(self):
        lost = screen_decision()
        self.assertEqual(roundtrip(lost), lost)
        with self.assertRaisesRegex(ContractError, "screen cannot accept"):
            screen_decision(outcome=c.Outcome.ACCEPTED)
        with self.assertRaisesRegex(ContractError, "screen_delta only"):
            screen_decision(delta=0.1, margin=0.0, df=8, sigma_used=0.02)
        with self.assertRaisesRegex(ContractError, "screen delta"):
            screen_decision(screen_delta=None)
        with self.assertRaisesRegex(ContractError, "sigma_used"):
            decision(sigma_used=None)
        with self.assertRaisesRegex(ContractError, "candidate_version"):
            decision(candidate_version="cand-002")
        self.assertEqual(DecisionNumbers(c.Outcome.ACCEPTED, 0.06, 0.03, 8, 0.02, "delta exceeds margin").outcome, c.Outcome.ACCEPTED)
        with self.assertRaisesRegex(ContractError, "contradicts"):
            DecisionNumbers(c.Outcome.ACCEPTED, 0.01, 0.03, 8, 0.02, "x")
        with self.assertRaises(ContractError):
            DecisionNumbers(c.Outcome.INVALID, 0.01, None, None, None, "x")

    def test_candidate_and_decision_keep_their_diff_reason_scores_and_cost(self):
        cand = roundtrip(candidate())
        self.assertEqual(cand.diff.path, "candidates/cand-001/diff.patch")
        self.assertEqual(cand.reason, "Bound the notes summary.")
        dec = roundtrip(decision())
        self.assertEqual(len(dec.incumbent_scores), 2)
        self.assertAlmostEqual(dec.cost.total_usd, 14.3)
        with self.assertRaises(ContractError):
            candidate(diff=None)  # lint passed but no diff
        with self.assertRaisesRegex(ContractError, "version_label"):
            candidate(version_label="cand-777")
        failed = candidate(lint_passed=False, pack=None, diff=None, lint_messages=("task.yaml does not parse",))
        self.assertEqual(roundtrip(failed), failed)

    def test_pack_edit_and_proposal_rules(self):
        with self.assertRaises(ContractError):
            PackEdit("a.md", c.EditOp.REPLACE_FILE, None, ())
        with self.assertRaises(ContractError):
            PackEdit("a.md", c.EditOp.REPLACE_FILE, "x", (SearchReplace("a", "b"),))
        with self.assertRaises(ContractError):
            PackEdit("a.md", c.EditOp.SEARCH_REPLACE, "x", ())
        with self.assertRaises(ContractError):
            PackEdit("../a.md", c.EditOp.REPLACE_FILE, "x", ())
        with self.assertRaises(ContractError):
            PackEdit("a.md", c.EditOp.SEARCH_REPLACE, None, (SearchReplace("", "b"),))
        e = PackEdit("a.md", c.EditOp.REPLACE_FILE, "x", ())
        with self.assertRaises(ContractError):
            Proposal("h", c.Lever.OTHER, "s", "e", "r", (e, e))  # one edit per path
        with self.assertRaises(ContractError):
            Proposal("h", c.Lever.OTHER, "s", "e", "r", ())

    def test_final_request_repeat_needs_a_reason(self):
        with self.assertRaisesRegex(ContractError, "reason"):
            FinalRequest("f2", Arm.V0, "v0", PackRef(PACK_REPO, H40), H64, (Split.TEST,), "f1", None, NOW)
        ok = FinalRequest("f2", Arm.V0, "v0", PackRef(PACK_REPO, H40), H64, (Split.TEST,), "f1", "scorer crashed at seed 2", NOW)
        self.assertEqual(roundtrip(ok), ok)

    def test_final_request_for_the_stock_arm_has_no_pack(self):
        stock = FinalRequest("f0", Arm.STOCK, None, None, H64, (Split.TEST,), None, None, NOW)
        self.assertEqual(roundtrip(stock), stock)
        with self.assertRaisesRegex(ContractError, "stock has no pack"):
            FinalRequest("f0", Arm.STOCK, "v0", PackRef(PACK_REPO, H40), H64, (Split.TEST,), None, None, NOW)
        with self.assertRaisesRegex(ContractError, "stock has no pack"):
            FinalRequest("f3", Arm.EVOLVED, None, None, H64, (Split.TEST,), None, None, NOW)
        with self.assertRaisesRegex(ContractError, "dev"):
            FinalRequest("f3", Arm.V0, "v0", PackRef(PACK_REPO, H40), H64, (Split.DEV, Split.TEST), None, None, NOW)

    def test_manifest_partition_rules(self):
        m = manifest()
        self.assertEqual(m.get("a-held-1").role, c.AssignmentRole.HELDOUT)
        self.assertEqual(m.ids_with_role(c.AssignmentRole.EVOLUTION), ("a-evo-1",))
        d = m.to_dict()
        d["heldout_families"] = ["ifeval25"]  # overlaps evolution families
        with self.assertRaises(ContractError):
            AssignmentManifest.from_dict(d)
        d = m.to_dict()
        d["assignments"][0]["families"] = ["copy"]  # an evolution assignment using a held-out family
        with self.assertRaisesRegex(ContractError, "outside"):
            AssignmentManifest.from_dict(d)
        d = m.to_dict()
        d["assignments"] = [d["assignments"][0]]
        with self.assertRaises(ContractError):
            AssignmentManifest.from_dict(d)

    def test_manifest_records_how_items_were_made(self):
        self.assertEqual(manifest().item_mode, c.ItemMode.SALTED_DRAW)
        fresh = manifest(item_mode=c.ItemMode.FRESH_INSTANTIATION)
        self.assertEqual(roundtrip(fresh), fresh)
        with self.assertRaises(ContractError):
            manifest(item_mode="copied")

    def test_manifest_revision_keys_and_the_reserved_test_id(self):
        with self.assertRaises(ContractError):
            Assignment(c.TEST_ASSIGNMENT_ID, c.AssignmentRole.HELDOUT, ("copy",), data_revs())
        with self.assertRaisesRegex(ContractError, "keys"):
            Assignment("a1", c.AssignmentRole.HELDOUT, ("copy",), {"dev": H64})
        with self.assertRaisesRegex(ContractError, "keys"):
            manifest(source_revisions={"ifrlvr_parquet_sha256": "5" * 64})
        bad = dict(manifest().source_revisions, ifbench_commit="short")
        with self.assertRaises(ContractError):
            manifest(source_revisions=bad)

    def test_train_job_dev_request_is_a_dev_template_without_arm_or_adapter(self):
        with self.assertRaises(ContractError):
            dataclasses.replace(train_job(), dev_request=score_request())  # selection split
        with self.assertRaisesRegex(ContractError, "template"):
            train_job(dev_request=score_request(split=Split.DEV, request_id="j1-dev", arm=Arm.V0))
        with self.assertRaisesRegex(ContractError, "template"):
            train_job(dev_request=score_request(split=Split.DEV, request_id="j1-dev"))  # adapter present
        job = train_job()
        filled = job.dev_request.with_adapter("j1-dev2", adapter())
        self.assertEqual(filled.adapter, adapter())
        self.assertIsNone(filled.arm)  # a dev score never carries the final arm
        final_job = train_job(dev_request=score_request(split=Split.DEV, request_id="j2-dev", adapter=None, phase=Phase.FINAL,
                                                        assignment_id="a-held-1", final_request_id="f1"))
        self.assertEqual(roundtrip(final_job), final_job)
        with self.assertRaises(ContractError):
            train_result(status=c.JobStatus.OK, adapter=None)
        with self.assertRaises(ContractError):
            train_result(status=c.JobStatus.TIMEOUT, adapter=None, adapter_id=None, dev_score_id=None, train_data=None, failure=None)

    def test_train_result_timings_adapter_id_and_training_data(self):
        with self.assertRaisesRegex(ContractError, "timings"):
            train_result(timings={"training": 1.0})
        self.assertEqual(set(c.TIMING_KEYS), {"download_s", "load_s", "generation_s", "train_s", "export_s", "eval_s"})
        with self.assertRaisesRegex(ContractError, "train_data"):
            train_result(train_data=None)
        with self.assertRaisesRegex(ContractError, "adapter_id"):
            train_result(adapter_id="other")
        failed = train_result(status=c.JobStatus.FAILED, adapter=None, adapter_id=None, dev_score_id=None, train_data=None, failure="oom")
        self.assertEqual(roundtrip(failed), failed)

    def test_budget_usage_exhausted_axes(self):
        b = BudgetSpec(100, 10.0, 5.0, 60.0)
        self.assertEqual(BudgetUsage(0, 0, 0, 0, 0).exhausted_axes(b), ())
        u = BudgetUsage(100, 9.0, 3.0, 2.0, 61.0)
        self.assertEqual(u.exhausted_axes(b), ("researcher_tokens", "dollars", "wall_seconds"))
        self.assertEqual(BudgetUsage(0, 0, 0, 0, 0).exhausted_axes(BudgetSpec(0, 1.0, 1.0, 1.0)), ("researcher_tokens",))

    def test_axis_of_covers_every_resource_and_maps_both_dollar_sources_to_one_axis(self):
        self.assertEqual(set(c.AXIS_OF), set(c.Resource))
        self.assertEqual(c.AXIS_OF[c.Resource.RESEARCHER_USD], "dollars")
        self.assertEqual(c.AXIS_OF[c.Resource.GPU_USD], "dollars")
        spec_axes = {"researcher_tokens", "gpu_minutes", "dollars", "wall_seconds"}
        self.assertEqual(set(c.AXIS_OF.values()), spec_axes)

    def test_budget_exhausted_event_axes_include_the_lab_wide_cap(self):
        self.assertEqual(BudgetExhaustedEvent("t001", "lab_dollars", 150.0, 150.2, "refuse new trials").axis, "lab_dollars")
        with self.assertRaises(ContractError):
            BudgetExhaustedEvent("t001", "gpu_usd", 1.0, 2.0, "x")

    def test_llm_usage(self):
        u = LlmUsage(100, 20, 40)
        self.assertEqual(u.total_tokens, 120)
        with self.assertRaises(ContractError):
            LlmUsage(10, 1, 11)
        with self.assertRaises(ContractError):
            LlmUsage(-1, 0, 0)

    def test_records_are_frozen(self):
        for name, make in PERSISTED.items():
            with self.subTest(name):
                obj = make()
                field = next(f.name for f in dataclasses.fields(obj))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(obj, field, None)

    def test_records_with_dict_fields_are_unhashable_and_compared_by_value(self):
        with self.assertRaises(TypeError):
            hash(spec())
        with self.assertRaises(TypeError):
            hash(BudgetCharge("t001", c.Resource.GPU_MINUTES, 1.0, "c", {}))
        self.assertEqual(spec(), spec())
        self.assertEqual(hash(Arm.V0), hash(Arm.V0))

    def test_text_limit_protects_against_ledger_truncation(self):
        with self.assertRaises(ContractError):
            GateRecord(c.Gate.TERMS_10_1_IX, c.GateOutcome.CLARIFIED, "x" * (c.MAX_TEXT_BYTES + 1), "me", NOW)
        c.assert_ledger_safe(decision().to_dict())
        with self.assertRaisesRegex(ContractError, "truncated"):
            c.assert_ledger_safe({"a": ["x" * 9000]})
        with self.assertRaisesRegex(ContractError, "non-finite"):
            c.assert_ledger_safe({"a": float("nan")})
        self.assertLess(c.MAX_TEXT_BYTES, c.LEDGER_STRING_CAP_BYTES)


class FinalPhaseTests(unittest.TestCase):
    def test_final_analysis_round_trips_and_derives_its_shape(self):
        fa = final_analysis()
        self.assertEqual(roundtrip(fa), fa)
        self.assertEqual(fa.trials_per_pack_per_assignment(), 5)
        self.assertEqual(len(fa.planned_trials), 10)
        self.assertEqual(fa.planned("f-v0-a-held-1-0").assignment_id, "a-held-1")
        with self.assertRaises(KeyError):
            fa.planned("nope")
        self.assertEqual(fa.sha256(), final_analysis().sha256())
        two = final_analysis(n_assignments=2, per_cell=3)
        self.assertEqual(two.method, c.ComparisonMethod.BLOCKED_T)
        self.assertEqual(roundtrip(two), two)

    def test_final_analysis_rejects_underpowered_or_unbalanced_plans(self):
        with self.assertRaisesRegex(ContractError, "at least 5 trials per pack"):
            final_analysis(per_cell=4)
        with self.assertRaisesRegex(ContractError, "method"):
            final_analysis(method=c.ComparisonMethod.BLOCKED_T)
        with self.assertRaisesRegex(ContractError, "method"):
            final_analysis(n_assignments=2, per_cell=3, method=c.ComparisonMethod.WELCH)
        fa = final_analysis()
        with self.assertRaisesRegex(ContractError, "same number"):
            final_analysis(planned_trials=fa.planned_trials[:-1] + (PlannedTrial("extra", Arm.V0, "a-held-1", 777),))
        with self.assertRaisesRegex(ContractError, "both v0 and evolved"):
            final_analysis(planned_trials=tuple(t for t in fa.planned_trials if t.arm is Arm.V0))
        with self.assertRaisesRegex(ContractError, "sum to 1"):
            final_analysis(assignments=(AssignmentWeight("a-held-1", 0.5),))
        with self.assertRaisesRegex(ContractError, "duplicate trial id"):
            final_analysis(planned_trials=fa.planned_trials + (fa.planned_trials[0],))
        with self.assertRaisesRegex(ContractError, "selection metric"):
            final_analysis(heldout_metric="prompt_strict")
        with self.assertRaisesRegex(ContractError, "test metric"):
            final_analysis(test_metric="prompt_all")
        with self.assertRaises(ContractError):
            PlannedTrial("t", Arm.STOCK, "a", 1)

    def test_check_trial_against_frozen(self):
        m, b, r, fa = manifest(), budget(), rule(), final_analysis()
        c.check_trial_against_frozen(spec(), manifest=m, budget=b, rule=r)
        planned = fa.planned_trials[-1]  # an evolved trial
        ok = final_spec(trial_id=planned.trial_id, arm=planned.arm, seed=planned.seed, data_revisions=data_revs("4" * 64))
        c.check_trial_against_frozen(ok, manifest=m, budget=b, analysis=fa)
        with self.assertRaisesRegex(ContractError, "evolution assignments"):
            c.check_trial_against_frozen(spec(assignment_id="a-held-1", data_revisions=data_revs("4" * 64)), manifest=m, budget=b)
        with self.assertRaisesRegex(ContractError, "heldout assignments"):
            c.check_trial_against_frozen(final_spec(assignment_id="a-evo-1"), manifest=m, budget=b)
        with self.assertRaisesRegex(ContractError, "data_revisions"):
            c.check_trial_against_frozen(spec(data_revisions=data_revs("8" * 64)), manifest=m, budget=b)
        with self.assertRaisesRegex(ContractError, "frozen budget"):
            c.check_trial_against_frozen(spec(budget=BudgetSpec(2_000_000, 121.0, 25.0, 7200.0)), manifest=m, budget=b)
        with self.assertRaisesRegex(ContractError, "frozen rule"):
            c.check_trial_against_frozen(spec(score_seeds=(0, 1)), manifest=m, budget=b, rule=r)
        with self.assertRaisesRegex(ContractError, "not in the frozen final analysis"):
            c.check_trial_against_frozen(final_spec(trial_id="unplanned", data_revisions=data_revs("4" * 64)), manifest=m, budget=b, analysis=fa)
        with self.assertRaisesRegex(ContractError, "planned"):
            c.check_trial_against_frozen(
                final_spec(trial_id=planned.trial_id, arm=Arm.V0, seed=planned.seed, data_revisions=data_revs("4" * 64)), manifest=m, budget=b, analysis=fa)
        with self.assertRaisesRegex(ContractError, "not in the manifest"):
            c.check_trial_against_frozen(spec(assignment_id="nope"), manifest=m, budget=b)

    def test_rule_and_analysis_must_use_the_right_assignment_roles(self):
        m = manifest()
        c.validate_rule_against_manifest(rule(), m)
        c.validate_analysis_against_manifest(final_analysis(), m)
        with self.assertRaisesRegex(ContractError, "held out"):
            c.validate_rule_against_manifest(rule(assignments=("a-held-1",)), m)
        with self.assertRaisesRegex(ContractError, "not in the manifest"):
            c.validate_rule_against_manifest(rule(assignments=("nope",)), m)
        with self.assertRaisesRegex(ContractError, "evolution assignment"):
            c.validate_analysis_against_manifest(final_analysis(assignments=(AssignmentWeight("a-evo-1", 1.0),),
                                                                planned_trials=tuple(PlannedTrial(f"x-{a.value}-{k}", a, "a-evo-1", k) for a in (Arm.V0, Arm.EVOLVED) for k in range(5))), m)

    def test_final_comparison_invariants(self):
        fc = final_comparison()
        self.assertEqual(roundtrip(fc), fc)
        with self.assertRaisesRegex(ContractError, "interval"):
            final_comparison(ci_low=0.05)
        with self.assertRaisesRegex(ContractError, "includes_zero"):
            final_comparison(includes_zero=False)
        with self.assertRaises(ContractError):
            final_comparison(split=Split.DEV)
        positive = final_comparison(ci_low=0.01, includes_zero=False)
        self.assertFalse(positive.includes_zero)

    def test_search_closed_reveal_failure_contamination_and_operator_records(self):
        with self.assertRaises(ContractError):
            FinalReveal(H64, 0, NOW)
        with self.assertRaisesRegex(ContractError, "test split"):
            ScoreFailure("x", Split.TEST, "a-held-1", None, None, "boom", False, NOW)
        ok = ScoreFailure("x", Split.TEST, c.TEST_ASSIGNMENT_ID, None, None, "boom", True, NOW)
        self.assertTrue(ok.result_revealed)
        ref = FileRef("trials/t001/train_data/j1.jsonl", H64, 10)
        with self.assertRaisesRegex(ContractError, "passed"):
            ContaminationReport("t001", "j1", ref, 10, 1, 0, 13, (H64,), True, NOW)
        with self.assertRaisesRegex(ContractError, "passed"):
            ContaminationReport("t001", "j1", ref, 10, 0, 0, 13, (H64,), False, NOW)
        with self.assertRaises(ContractError):
            ContaminationReport("t001", "j1", ref, 10, 0, 0, 5, (H64,), True, NOW)
        hit = ContaminationReport("t001", "j1", ref, 10, 0, 3, 13, (H64,), False, NOW)
        self.assertFalse(hit.passed)
        with self.assertRaises(ContractError):
            OperatorChange("x", "y", -1, NOW)

    def test_lab_config_pins_and_secret_names(self):
        self.assertEqual(roundtrip(lab_config()), lab_config())
        with self.assertRaisesRegex(ContractError, "pins"):
            lab_config(pins={"coral_commit": "x"})
        with self.assertRaisesRegex(ContractError, "secret_env"):
            lab_config(secret_env={"meta_api_key": "META_API_KEY"})
        with self.assertRaisesRegex(ContractError, "environment variable"):
            lab_config(secret_env={"meta_api_key": "sk-live-abc123", "thunder_api_key": "THUNDER_API_KEY"})
        with self.assertRaises(ContractError):
            lab_config(lab_budget_usd=0)

    def test_lint_result(self):
        self.assertTrue(c.LintResult(True, (), ("notes/seed.md",)).passed)
        with self.assertRaises(ContractError):
            c.LintResult(False, (), ())
        with self.assertRaises(ContractError):
            c.LintResult(True, (), ("../x",))


class FileSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "lab"
        (self.root / "private").mkdir(parents=True)
        (self.root / "notes").mkdir()
        self.secret = self.root / "private" / "selection.jsonl"
        self.secret.write_text("GROUND-TRUTH-CANARY\n")

    def test_file_ref_refuses_a_symlink_to_a_private_file(self):
        link = self.root / "notes" / "x"
        os.symlink(self.secret, link)
        with self.assertRaisesRegex(ContractError, "symbolic link"):
            c.file_ref(self.root, link)
        with self.assertRaisesRegex(ContractError, "symbolic link"):
            c.read_regular_bytes(self.root, "notes/x", max_bytes=1000)

    def test_a_symlinked_directory_component_is_refused_too(self):
        os.symlink(self.root / "private", self.root / "notes" / "d")
        with self.assertRaisesRegex(ContractError, "symbolic link"):
            c.read_regular_bytes(self.root, "notes/d/selection.jsonl", max_bytes=1000)

    def test_hard_links_are_refused(self):
        os.link(self.secret, self.root / "notes" / "h")
        with self.assertRaisesRegex(ContractError, "hard-linked"):
            c.file_ref(self.root, self.root / "notes" / "h")

    def test_regular_files_are_read_within_the_size_cap(self):
        p = self.root / "notes" / "n.md"
        p.write_text("hello")
        ref = c.file_ref(self.root, p)
        self.assertEqual((ref.path, ref.size_bytes), ("notes/n.md", 5))
        self.assertEqual(c.read_regular_bytes(self.root, "notes/n.md", max_bytes=5), b"hello")
        with self.assertRaisesRegex(ContractError, "exceeds"):
            c.read_regular_bytes(self.root, "notes/n.md", max_bytes=4)
        with self.assertRaises(ContractError):
            c.read_regular_bytes(self.root, "../outside", max_bytes=10)
        with self.assertRaises(ContractError):
            c.file_ref(self.root, self.root)

    def test_load_frozen_checks_the_bytes_against_the_ledger_ref(self):
        layout = LabLayout(self.root)
        c.write_record(layout.rule_path, rule())
        frozen = FrozenFile("rule", layout.file_ref(layout.rule_path))
        self.assertEqual(c.load_frozen(layout, frozen, Rule), rule())
        self.assertNotEqual(frozen.file.sha256, rule().sha256())  # two different hashes: bytes versus canonical record
        c.write_record(layout.rule_path, rule(min_gain=0.01))  # rewritten after the freeze
        with self.assertRaisesRegex(ContractError, "does not match the hash"):
            c.load_frozen(layout, frozen, Rule)


class KindsLayoutProtocolTests(unittest.TestCase):
    def test_lab_kinds_are_unique_namespaced_and_have_payload_types(self):
        kinds = list(LabKind.MASTER) + list(LabKind.TRIAL)
        self.assertEqual(len(kinds), len(set(kinds)))
        self.assertEqual(LabKind.all(), set(kinds))
        self.assertEqual(set(c.PAYLOAD_TYPES), set(kinds))
        self.assertTrue(all(k.startswith("lab.") for k in kinds))
        self.assertNotIn("experiment.proposed", kinds)  # nightshift.ledger allocates experiment ids from exactly this kind
        self.assertEqual(LabKind.REUSED, ("budget.charged", "budget.exhausted"))
        self.assertFalse(set(LabKind.REUSED) & set(kinds))

    def test_final_phase_kinds_exist_and_freeze_once_kinds_are_master_kinds(self):
        for kind in (LabKind.FINAL_ANALYSIS_FROZEN, LabKind.SEARCH_CLOSED, LabKind.FINAL_REVEALED, LabKind.SCORE_FAILED,
                     LabKind.CONTAMINATION_SCANNED, LabKind.OPERATOR_CHANGE, LabKind.SPREAD_PLANNED):
            self.assertIn(kind, LabKind.MASTER)
        self.assertEqual(c.PAYLOAD_TYPES[LabKind.FINAL_ANALYSIS_FROZEN], FrozenFile)
        self.assertTrue(set(LabKind.FREEZE_ONCE) <= set(LabKind.MASTER))
        for kind in (LabKind.RULE_FROZEN, LabKind.BUDGET_FROZEN, LabKind.ASSIGNMENTS_FROZEN, LabKind.FINAL_ANALYSIS_FROZEN):
            self.assertIn(kind, LabKind.FREEZE_ONCE)
        self.assertNotIn(LabKind.TRIAL_STARTED, LabKind.FREEZE_ONCE)

    def test_payload_from_dict_dispatches_and_refuses_unknown_kinds(self):
        rec = payload = c.payload_from_dict(LabKind.TRIAL_FINISHED, trial_record().to_dict())
        self.assertEqual(rec, trial_record())
        self.assertEqual(payload.spec.trial_id, "t001")
        with self.assertRaises(ContractError):
            c.payload_from_dict("budget.charged", {})
        with self.assertRaises(ContractError):
            c.payload_from_dict(LabKind.RULE_FROZEN, {"nope": 1})

    def test_every_persisted_payload_fits_the_ledger(self):
        for kind, cls in c.PAYLOAD_TYPES.items():
            make = PERSISTED.get(cls.__name__)
            self.assertIsNotNone(make, f"no sample for {cls.__name__}")
            c.assert_ledger_safe(make().to_dict())

    def test_lab_state_fields_are_typed_by_their_key(self):
        st = c.LabState()
        for name in ("trials", "trials_started", "scores", "score_costs", "candidates", "decisions", "packs", "contamination"):
            self.assertEqual(getattr(st, name), {})
        for name in ("gates", "finals", "score_failures", "operator_changes", "malformed"):
            self.assertEqual(getattr(st, name), [])
        hints = c.LabState.__annotations__
        self.assertIn("dict[str, TrialRecord]", hints["trials"])
        self.assertIn("dict[str, ScoreRecord]", hints["scores"])
        self.assertIn("dict[str, Decision]", hints["decisions"])
        self.assertIn("dict[str, PackCommitted]", hints["packs"])
        self.assertIsNone(st.final_analysis)
        self.assertIsNone(st.search_closed)

    def test_layout_paths_stay_inside_the_root(self):
        lay = LabLayout("/lab")
        self.assertEqual(lay.rule_path, Path("/lab/frozen/rule.json"))
        self.assertEqual(lay.trial_ledger_dir("t001"), Path("/lab/trials/t001"))
        self.assertEqual(lay.selection_data("a-evo-1"), Path("/lab/private/assignments/a-evo-1/selection.jsonl"))
        self.assertEqual(lay.dev_data("a-evo-1"), Path("/lab/data/assignments/a-evo-1/dev.jsonl"))
        self.assertNotIn(str(lay.private_dir), str(lay.train_data("a-evo-1")))
        self.assertEqual(lay.master_ledger_dir, Path("/lab"))
        for bad in ("..", "../x", "a/b", ""):
            with self.assertRaises(ContractError):
                lay.trial_dir(bad)
            with self.assertRaises(ContractError):
                lay.score_dir(bad)

    def test_layout_has_every_path_the_design_uses(self):
        lay = LabLayout("/lab")
        expected = {
            lay.lab_config_path: "/lab/lab.json",
            lay.final_analysis_path: "/lab/frozen/final_analysis.json",
            lay.anchors_path: "/lab/anchors/master.anchors",
            lay.third_party_dir: "/lab/third_party",
            lay.split_salt_path: "/lab/private/split_salt",
            lay.gpu_session_dir("s1"): "/lab/gpu/sessions/s1",
            lay.trial_pack_dir("t1"): "/lab/trials/t1/pack",
            lay.trial_run_dir("t1"): "/lab/trials/t1/run",
            lay.trial_trace_dir("t1"): "/lab/trials/t1/trace",
            lay.trial_recipe_dir("t1"): "/lab/trials/t1/recipe",
            lay.trial_train_data_path("t1", "j1"): "/lab/trials/t1/train_data/j1.jsonl",
            lay.trial_spec_path("t1"): "/lab/trials/t1/spec.json",
            lay.trial_record_path("t1"): "/lab/trials/t1/trial.json",
            lay.score_request_path("sc1"): "/lab/scores/sc1/request.json",
            lay.score_record_path("sc1"): "/lab/scores/sc1/score.json",
            lay.per_prompt_path("sc1"): "/lab/scores/sc1/per_prompt.jsonl",
            lay.gen_responses_path("sc1", 3): "/lab/scores/sc1/gen_seed3.jsonl",
            lay.candidate_proposal_path("cand-001"): "/lab/candidates/cand-001/proposal.json",
            lay.candidate_prompt_path("cand-001"): "/lab/candidates/cand-001/prompt.txt",
            lay.candidate_response_path("cand-001"): "/lab/candidates/cand-001/response.txt",
            lay.candidate_diff_path("cand-001"): "/lab/candidates/cand-001/diff.patch",
            lay.candidate_lint_path("cand-001"): "/lab/candidates/cand-001/lint.json",
            lay.adapter_json_path("ad1"): "/lab/artifacts/adapters/ad1/adapter.json",
            lay.test_data(): "/lab/private/test/IFBench_test.jsonl",
        }
        for got, want in expected.items():
            self.assertEqual(got, Path(want))
        self.assertFalse(hasattr(lay, "trial_control_socket"))  # the control API listens on a loopback TCP port
        with self.assertRaises(ContractError):
            lay.gen_responses_path("sc1", -1)
        with self.assertRaises(ContractError):
            lay.gpu_session_dir("../x")

    def test_layout_file_ref(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lay = LabLayout(tmp.name)
        p = lay.per_prompt_path("sc1")
        p.parent.mkdir(parents=True)
        p.write_text("{}\n")
        ref = lay.file_ref(p)
        self.assertEqual(ref.path, "scores/sc1/per_prompt.jsonl")
        self.assertEqual(ref.size_bytes, 3)
        with self.assertRaises(ContractError):
            lay.file_ref("/etc/hosts")

    def test_protocols_accept_conforming_objects_and_reject_others(self):
        class Meter:
            spec = budget()

            def check(self, resource): ...
            def charge(self, resource, amount, call_id, detail=None): ...
            def reserve_llm(self, model, prompt_tokens_estimate, max_tokens, call_id): ...
            def charge_llm(self, model, usage, call_id): ...
            def usage(self): ...

        class OldMeter:  # the pre-reservation shape: no charge_llm, so it cannot be a meter
            spec = budget()

            def check(self, resource): ...
            def charge(self, resource, amount, call_id, detail=None): ...
            def usage(self): ...

        class Gen:
            def generate(self, model, prompts, params, seed): ...

        class Sc:
            scorer_version = "s1"

            def score(self, request, generator, model): ...

        class Ledger:
            def append(self, kind, actor, record): ...
            def anchor(self): ...
            def seal(self): ...
            def close(self): ...

        self.assertIsInstance(Meter(), c.BudgetMeter)
        self.assertNotIsInstance(OldMeter(), c.BudgetMeter)
        self.assertIsInstance(Gen(), c.Generator)
        self.assertIsInstance(Sc(), c.Scorer)
        self.assertIsInstance(Ledger(), c.LedgerWriter)
        self.assertNotIsInstance(object(), c.BudgetMeter)
        self.assertNotIsInstance(Gen(), c.Scorer)
        for proto in (c.GpuProvider, c.ModelServer, c.Scorer, c.Trainer, c.ResearcherRuntime, c.RuntimeHandle, c.ProposalSource, c.TrialExecutor,
                      c.ServeSession, c.LedgerWriter, c.PackStore, c.DecisionRule, c.StateReader, c.BudgetMeter, c.Generator):
            self.assertTrue(getattr(proto, "_is_runtime_protocol", False), proto.__name__)

    def test_world_container_names_every_seam(self):
        names = {f.name for f in dataclasses.fields(c.LabWorld)}
        self.assertEqual(names, {"layout", "config", "clock", "ledger", "state_reader", "new_trial_ledger", "meter_factory", "scorer",
                                 "generator_factory", "trainer", "model_server", "runtime", "packs", "decision_rule", "trial_executor",
                                 "proposal_source"})
        for alias in (c.GeneratorFactory, c.MeterFactory, c.TrialLedgerFactory, c.Clock, c.PortAllocator, c.ScoreGuard):
            self.assertIsNotNone(alias)

    def test_exceptions_share_one_base_and_budget_exhausted_reports_numbers(self):
        for exc in (c.ContractError, c.SchemaVersionError, c.BudgetExhausted, c.AccessDenied, c.FinalComparisonRefused, c.GateNotRecorded,
                    c.LedgerError, c.GpuError, c.ProviderUnavailable, c.CreateRejected, c.CreateAmbiguous, c.OwnershipError, c.ScorerError,
                    c.TrainerError, c.RuntimeFailure, c.ProposalInvalid):
            self.assertTrue(issubclass(exc, c.LabError), exc.__name__)
        self.assertTrue(issubclass(c.ContractError, ValueError))
        self.assertTrue(issubclass(c.CreateAmbiguous, c.GpuError))
        e = c.BudgetExhausted("gpu_minutes", 120.0, 121.5)
        self.assertEqual((e.axis, e.limit, e.used), ("gpu_minutes", 120.0, 121.5))
        self.assertIn("121.5", str(e))

    def test_contracts_import_nothing_from_the_rest_of_nightshift(self):
        text = Path(c.__file__).read_text()
        self.assertNotIn("import nightshift", text)
        self.assertNotIn("from nightshift", text)

    def test_deep_copy_of_a_record_is_equal(self):
        t = trial_record()
        self.assertEqual(copy.deepcopy(t), t)


class IntegratorAdditionsTests(unittest.TestCase):
    """Additive changes applied by the integrator after owners 1 to 6 reported (docs/harness-design.md, Work breakdown, wave C)."""

    def test_shared_keys_are_constants_with_the_values_the_report_reads(self):
        self.assertEqual(c.ENV_ISOLATION, "isolation")
        self.assertEqual(c.DETAIL_USAGE_ESTIMATED, "usage_estimated")
        BudgetCharge(trial_id="t001", resource=c.Resource.RESEARCHER_TOKENS, amount=1.0, call_id="x", detail={c.DETAIL_USAGE_ESTIMATED: True})

    def test_new_layout_paths(self):
        lay = LabLayout("/lab")
        self.assertEqual(lay.split_report_path, Path("/lab/private/splits_report.json"))
        self.assertEqual(lay.template_ngrams_path, Path("/lab/private/template_ngrams.json"))
        self.assertEqual(lay.final_comparison_path, Path("/lab/reports/final_comparison.json"))
        self.assertEqual(lay.trial_job_log_path("t1", "j1"), Path("/lab/trials/t1/job_logs/j1.log"))
        for path in (lay.split_report_path, lay.template_ngrams_path):
            self.assertIn(str(lay.private_dir), str(path))  # both stay on the control host, under private/
        with self.assertRaises(ContractError):
            lay.trial_job_log_path("t1", "../x")

    def test_lab_state_carries_event_seq_as_a_field(self):
        st = c.LabState()
        self.assertEqual(st.event_seq, {})
        st.event_seq[(LabKind.RULE_FROZEN, "")] = 7
        self.assertEqual(c.LabState().event_seq, {})  # no shared default

    def test_protocol_signatures_the_owners_implement(self):
        import inspect

        self.assertEqual(list(inspect.signature(c.BudgetMeter.charge_llm).parameters), ["self", "model", "usage", "call_id", "detail"])
        self.assertEqual(list(inspect.signature(c.StateReader.next_id).parameters), ["self", "state", "prefix", "taken", "reserve"])
        self.assertIn("load_final_analysis", dir(c.StateReader))
        self.assertIn("close", dir(c.RuntimeHandle))

    def test_a_runtime_handle_needs_close(self):
        class Handle:
            def poll(self): ...
            def stop(self, reason): ...
            def collect(self): ...

        class Closing(Handle):
            def close(self): ...

        self.assertNotIsInstance(Handle(), c.RuntimeHandle)
        self.assertIsInstance(Closing(), c.RuntimeHandle)


class ExchangeTypeTests(unittest.TestCase):
    def test_trial_context_hides_its_secrets_from_repr(self):
        ctx = c.TrialContext(spec(), "/srv/lab/trials/t001", "http://127.0.0.1:9000/v1", "tok-SECRET-123", "http://127.0.0.1:9001/control")
        self.assertNotIn("SECRET", repr(ctx))
        self.assertNotIn("9001", repr(ctx))
        self.assertEqual(ctx.llm_token, "tok-SECRET-123")

    def test_runtime_result_carries_submitted_attempts_with_commit_times(self):
        att = c.SubmittedAttempt(Submission("ad1", H64, H40, None), NOW)
        res = c.RuntimeResult(c.ExitReason.FINISHED, 4, (att,), None, None, H64, {"coral": "x"})
        self.assertEqual(res.submissions[0].committed_utc, NOW)
        with self.assertRaises(ContractError):
            c.SubmittedAttempt(Submission("ad1", H64, H40, None), "2026-11-01T12:00:00")  # no timezone
        with self.assertRaises(ContractError):
            c.RuntimeResult(c.ExitReason.FINISHED, 4, (Submission("ad1", H64, H40, None),), None, None, H64, {})

    def test_exchange_types_are_not_persistable(self):
        for obj in (c.Quote(1.0, 0.0, True, "x"), c.CreateReceipt("h", 1.0), c.GenResult("p", "t", "stop", 3)):
            with self.assertRaises(TypeError):
                c.canonical_json(obj)

    def test_gpu_provider_create_takes_no_separate_owner_tag(self):
        import inspect

        self.assertEqual(list(inspect.signature(c.GpuProvider.create).parameters), ["self", "shape", "ssh_public_key"])


if __name__ == "__main__":
    unittest.main()
