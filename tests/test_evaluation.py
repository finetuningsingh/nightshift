"""Acceptance A13, A14, A16: Inspect evaluation, failures, test gate."""

import pytest
from inspect_ai.log import read_eval_log

from nightshift.checkpoints import write_fake_checkpoint
from nightshift.data import TestAccessGate, demo_registry
from nightshift.errors import PolicyError
from nightshift.evaluation import (
    SCORER_ID,
    SCORER_VERSION,
    TASK_VERSION,
    Evaluator,
    FailurePolicy,
    mean_score,
)
from nightshift.hashing import file_sha256


def make_evaluator(tmp_path, retries=0):
    return Evaluator(
        registry=demo_registry(),
        log_dir=tmp_path / "logs",
        test_gate=TestAccessGate(tmp_path / "gate.json"),
        failure_policy=FailurePolicy(max_retries=retries),
        decoding_params={"temperature": 0.0, "max_tokens": 64},
        records_dir=tmp_path / "evaluations",
    )


def test_a14_fake_inspect_record_links_every_identity_and_log_hash(tmp_path):
    checkpoint = write_fake_checkpoint(tmp_path / "ckpt", marker="a14", quality=1.0)
    evaluator = make_evaluator(tmp_path)
    record = evaluator.evaluate_checkpoint(
        checkpoint, split="dev", trial_id="trial-1", pack_version="pack-v1"
    )
    assert record.status == "succeeded", record.error
    assert record.score == pytest.approx(1.0)
    assert record.fake is True and record.model == "mockllm/model"
    data = record.to_record()
    assert data["dataset_manifest_id"] == "toy-dev"
    assert data["dataset_revision"] == "toy-1"
    assert data["split"] == "dev"
    assert data["task_version"] == TASK_VERSION
    assert (data["scorer_id"], data["scorer_version"]) == (SCORER_ID, SCORER_VERSION)
    assert data["code_revision"] and data["inspect_version"]
    assert data["decoding_params"] == {"temperature": 0.0, "max_tokens": 64}
    assert data["trial_id"] == "trial-1" and data["pack_version"] == "pack-v1"
    log_path = record.inspect_log_path
    assert log_path and file_sha256(log_path) == record.inspect_log_sha256
    # It is a real Inspect log file, readable by Inspect itself.
    eval_log = read_eval_log(log_path)
    assert eval_log.status == "success"
    assert eval_log.samples and len(eval_log.samples) == 4
    # A zero-quality fake checkpoint scores a genuine, succeeded 0.
    bad = write_fake_checkpoint(tmp_path / "ckpt-bad", marker="a14-bad", quality=0.0)
    zero = make_evaluator(tmp_path).evaluate_checkpoint(bad, split="dev", trial_id="trial-2")
    assert zero.status == "succeeded" and zero.score == pytest.approx(0.0)


def test_a13_evaluation_error_is_failed_without_score_and_never_averaged(tmp_path):
    checkpoint = write_fake_checkpoint(tmp_path / "ckpt", marker="a13", quality=1.0)
    evaluator = make_evaluator(tmp_path, retries=1)
    good = evaluator.evaluate_checkpoint(checkpoint, split="dev", trial_id="trial-good")
    assert good.status == "succeeded"
    bad_rows = [{
        "id": "bad-1",
        "prompt": "Say anything at all, in any style you like, for this broken row.",
        "checks": [{"id": "no_such_verifier", "params": {}}],
    }]
    failed = evaluator.evaluate_rows(
        checkpoint, split="dev", rows=bad_rows, trial_id="trial-bad"
    )
    assert failed.status == "failed"
    assert failed.attempts == 2  # prespecified retries, identical settings
    failed_record = failed.to_record()
    assert "score" not in failed_record
    assert failed_record["error"]
    # Averaging a failure as 0 would give 0.5; the policy gives the real mean.
    assert mean_score([failed, good]) == pytest.approx(good.score)


def test_a16_test_split_only_in_final_comparison_and_only_once(tmp_path):
    checkpoint = write_fake_checkpoint(tmp_path / "ckpt", marker="a16", quality=1.0)
    evaluator = make_evaluator(tmp_path)
    with pytest.raises(PolicyError):
        evaluator.evaluate_checkpoint(
            checkpoint, split="test", purpose="outer_screening", comparison_id="cmp-1"
        )
    with pytest.raises(PolicyError):
        evaluator.evaluate_checkpoint(
            checkpoint, split="selection", purpose="researcher_request"
        )
    record = evaluator.evaluate_checkpoint(
        checkpoint, split="test", purpose="final_comparison", comparison_id="cmp-1"
    )
    assert record.status == "succeeded" and record.split == "test"
    assert record.to_record()["dataset_manifest_id"] == "toy-test"
    with pytest.raises(PolicyError, match="exactly once"):
        evaluator.evaluate_checkpoint(
            checkpoint, split="test", purpose="final_comparison", comparison_id="cmp-1"
        )
    # Pre-claimed rows still need the final-comparison purpose.
    with pytest.raises(PolicyError):
        evaluator.evaluate_rows(
            checkpoint, split="test", rows=[], purpose="dev_feedback"
        )
