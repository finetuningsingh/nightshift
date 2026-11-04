"""Acceptance A10 (agent inputs), A17, A18, A19 (demo end to end)."""

import json

import pytest
from inspect_ai.log import read_eval_log

from nightshift.cli import main
from nightshift.data import assert_payload_is_public, demo_registry
from nightshift.hashing import canonical_json
from nightshift.ledger import verify_ledger
from nightshift.loops import TrialRecord, build_reviser_input, summarize_comparison
from nightshift.pack import create_pack
from nightshift.report import render_report, write_report


def test_a10_reviser_input_contains_no_private_rows_paths_or_per_item_results(tmp_path):
    pack = create_pack(tmp_path / "pack")
    rogue_trial = TrialRecord(
        trial_id="trial-rogue", phase="screening", pack_version="v1", seed=1,
        status="succeeded", fake=True,
        history=[
            {"request": {"type": "evaluate_dev"},
             "result": {"status": "succeeded", "split": "dev", "score": 0.5,
                        "inspect_log_path": "/private/logs/x.eval"}},
            {"request": {"type": "note"},
             "result": {"status": "succeeded",
                        "selection": [{"id": "sel-01", "score": 1.0}],
                        "leaked": "Write a sign for a mushroom stall. Use at most "
                                  "9 words and include the word forest.",
                        "test_path": "test.jsonl"}},
        ],
        dev_feedback=[{"split": "dev", "score": 0.5, "status": "succeeded"}],
        shared_notes="public note about lanterns",
    )
    payload = build_reviser_input(pack, "v1", [rogue_trial], demo_registry())
    text = canonical_json(payload)
    assert_payload_is_public(payload, demo_registry())
    for forbidden in ("sel-01", "test-01", "mushroom", "telescope",
                      "selection.jsonl", "test.jsonl", "/private/logs"):
        assert forbidden not in text
    assert "per_item" not in text
    assert "public note about lanterns" in text  # public material survives


def test_m1_reviser_input_drops_generic_selection_scores_and_fragments(tmp_path):
    pack = create_pack(tmp_path / "pack")
    rogue_trial = TrialRecord(
        trial_id="trial-m1", phase="screening", pack_version="v1", seed=1,
        status="succeeded", fake=True,
        history=[
            {"request": {"type": "evaluate_dev"},
             "result": {"status": "succeeded", "split": "dev", "score": 0.5}},
            {"request": {"type": "note"},
             "result": {"status": "succeeded",
                        "split": "selection", "score": 0.9,
                        "per_seed": [1.0, 0.0]}},
            {"request": {"type": "note"},
             "result": {"status": "succeeded",
                        "leaked_fragment": "mushroom stall. Use at most"}},
        ],
        dev_feedback=[{"split": "dev", "score": 0.5, "status": "succeeded"},
                      {"split": "selection", "score": 0.9, "status": "succeeded"}],
        shared_notes="public note about lanterns",
    )
    payload = build_reviser_input(pack, "v1", [rogue_trial], demo_registry())
    text = canonical_json(payload)
    assert "per_seed" not in text
    assert "mushroom stall" not in text
    assert '"split":"selection"' not in text
    assert '"split":"dev"' in text  # public dev feedback survives


def test_m10_one_scored_pair_is_not_evidence():
    pairs = [{"seed": 1, "fake": False,
              "original": {"status": "succeeded", "score": 0.0, "compute_seconds": 10},
              "final": {"status": "succeeded", "score": 1.0, "compute_seconds": 10}}]
    result = summarize_comparison(
        [dict(p) for p in pairs], margin=0.0, bootstrap_samples=100, seed=1
    )
    assert result["scored_pairs"] == 1 and result["min_scored_pairs"] == 2
    assert result["evidence_of_improvement"] is False
    relaxed = summarize_comparison(
        [dict(p) for p in pairs], margin=0.0, bootstrap_samples=100,
        seed=1, min_scored_pairs=1,
    )
    assert relaxed["evidence_of_improvement"] is True


def test_b1_reviser_writing_outside_scratch_pack_rejects_round(tmp_path):
    from nightshift.agents import AgentRunResult
    from nightshift.config import load_config, write_demo_config
    from nightshift.loops import NightshiftRun, ScreeningResult
    from nightshift.pack import freeze_pack

    create_pack(tmp_path / "pack")
    config_path = write_demo_config(
        tmp_path / "nightshift.toml", output_dir="run",
        budget_file="run/budget.json", pack_dir="pack",
    )
    run = NightshiftRun(load_config(config_path))
    version, pack_dir = freeze_pack(run._source_pack(), run.run_dir / "packs")
    run.pack_paths[version] = pack_dir
    run._screen = lambda v, label: ScreeningResult(pack_version=v)

    class OutsideReviser:
        fake = True

        def revise(self, scratch_pack_dir, public_input=None, round_no=1):
            from pathlib import Path

            scratch = Path(scratch_pack_dir)
            text = (scratch / "instructions.md").read_text(encoding="utf-8")
            (scratch / "instructions.md").write_text(
                text + "\nGenerate two candidates per prompt.\n", encoding="utf-8"
            )
            # Outside the scratch pack, handed to the loop as data.
            (scratch.parent / "outside.txt").write_text("escaped", encoding="utf-8")
            return AgentRunResult(text="stub", fake=True)

    run.reviser = OutsideReviser()
    incumbent, record = run._outer_round(1, version)
    assert incumbent == version
    assert record["accepted"] is False
    assert "outside the scratch pack" in record["reasons"][0]


def test_a17_failed_trials_stay_in_the_comparison_denominator():
    pairs = [
        {"seed": 1, "fake": False,
         "original": {"status": "succeeded", "score": 0.5, "compute_seconds": 100},
         "final": {"status": "succeeded", "score": 0.75, "compute_seconds": 120}},
        {"seed": 2, "fake": False,
         "original": {"status": "failed", "compute_seconds": 90},
         "final": {"status": "succeeded", "score": 1.0, "compute_seconds": 110}},
        {"seed": 3, "fake": False,
         "original": {"status": "succeeded", "score": 0.25, "compute_seconds": 80},
         "final": {"status": "failed", "compute_seconds": 70}},
    ]
    result = summarize_comparison(pairs, margin=0.0, bootstrap_samples=200, seed=4)
    assert result["pair_denominator"] == 3
    assert result["scored_pairs"] == 1
    assert result["original_failures"] == 1 and result["final_failures"] == 1
    assert result["mean_difference"] == pytest.approx(0.25)
    assert "score" not in str(result["pairs"][1]["original"].get("score"))
    assert result["evidence_of_improvement"] is False  # failures block the claim
    assert result["conclusion"] == "no evidence of improvement"
    assert result["original_compute_seconds"] == 270


def test_a18_fake_report_has_banner_and_never_claims_improvement(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "run_name": "fake-run", "status": "succeeded", "fake": True,
        "original_pack_version": "a" * 64, "final_pack_version": "b" * 64,
        "budgets": {"ceiling_usd": 10.0, "committed_and_reserved_usd": 0.5, "fake": True},
        "rounds": [{"round": 1, "candidate_version": "b" * 64,
                    "candidate_frozen": True, "accepted": True,
                    "mean_paired_difference": 0.9, "fake": True,
                    "reasons": ["screening difference (fake)"]}],
        "comparison": {
            "pairs": [{"seed": 1, "fake": True, "difference": 1.0,
                       "original": {"status": "succeeded", "score": 0.0},
                       "final": {"status": "succeeded", "score": 1.0}}],
            "pair_denominator": 1, "scored_pairs": 1,
            "mean_difference": 1.0, "bootstrap_ci": [1.0, 1.0],
            "original_failures": 0, "final_failures": 0,
            "original_compute_seconds": 10, "final_compute_seconds": 10,
            "evidence_of_improvement": True,  # must be overridden for fake runs
            "fake": True, "conclusion": "evidence of improvement",
        },
        "trials": [],
    }
    report_path = write_report(run_dir, summary)
    text = report_path.read_text(encoding="utf-8")
    assert "FAKE RUN" in text
    assert "`fake: true`" in text
    assert "No evidence of improvement" in text
    assert "**Evidence of improvement**" not in text
    assert render_report(run_dir) == text


@pytest.mark.slow
def test_a19_demo_completes_end_to_end_offline(tmp_path):
    root = tmp_path / "demo"
    assert main(["demo", "--root", str(root)]) == 0
    run_dir = root / "run"
    assert (run_dir / "report.md").is_file()
    assert verify_ledger(run_dir / "ledger.jsonl").ok
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert summary["fake"] is True and summary["status"] == "succeeded"
    assert len(summary["rounds"]) >= 2
    for round_record in summary["rounds"]:
        assert round_record["candidate_frozen"] is True, round_record
        frozen = run_dir / "packs" / round_record["candidate_version"]
        assert (frozen / "pack.toml").is_file()
    assert summary["comparison"]["pair_denominator"] == 2
    assert summary["comparison"]["conclusion"] == "no evidence of improvement"
    logs = [
        p for p in (run_dir / "inspect_logs").rglob("*")
        if p.is_file() and p.suffix in (".eval", ".json")
    ]
    assert logs, "demo produced no Inspect logs"
    eval_log = read_eval_log(str(logs[0]))
    assert eval_log.status in ("success", "error")
    assert list((run_dir / "evaluations").glob("*.json"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "FAKE RUN" in report and "No evidence of improvement" in report
    assert "**Evidence of improvement**" not in report
    # CLI status/report/ledger views agree with the artefacts.
    assert main(["ledger", "verify", str(run_dir)]) == 0
    assert main(["status", str(run_dir)]) == 0
    assert main(["report", str(run_dir)]) == 0
