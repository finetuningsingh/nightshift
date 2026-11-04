"""Acceptance A9, A10 (data side) and the test-split gate behind A16."""

import json
import shutil

import pytest

from nightshift.data import (
    ControllerDataAccess,
    DatasetRegistry,
    PublicDataView,
    TestAccessGate,
    assert_payload_is_public,
    build_manifest_entries,
    demo_data_dir,
    demo_registry,
)
from nightshift.errors import DataError, PolicyError
from nightshift.hashing import canonical_json


def copied_registry(tmp_path):
    dest = tmp_path / "data"
    shutil.copytree(demo_data_dir(), dest)
    return DatasetRegistry(build_manifest_entries(dest), dest), dest


def test_a9_hash_mismatch_is_refused(tmp_path):
    registry, dest = copied_registry(tmp_path)
    dev_file = dest / "dev.jsonl"
    rows = dev_file.read_text(encoding="utf-8").replace("honey", "jammy")
    dev_file.write_text(rows, encoding="utf-8")
    with pytest.raises(DataError, match="sha256 mismatch"):
        registry.verify(registry.manifest("dev"))


def test_a9_row_count_mismatch_is_refused(tmp_path):
    registry, _ = copied_registry(tmp_path)
    entries = [m.to_record() for m in registry.manifests()]
    for entry in entries:
        if entry["split"] == "dev":
            entry["row_count"] = 99
    broken = DatasetRegistry(entries, registry.base_dir)
    with pytest.raises(DataError, match="row_count"):
        broken.verify(broken.manifest("dev"))


def test_bundled_demo_manifests_verify():
    counts = demo_registry().verify_all()
    assert counts == {"train_prompts": 6, "dev": 4, "selection": 4, "test": 4}


def test_a10_public_view_only_serves_train_prompts():
    view = PublicDataView(demo_registry())
    prompts = view.train_prompts()
    assert prompts and all(set(p) == {"id", "prompt", "verifiers"} for p in prompts)
    for split in ("dev", "selection", "test", "nope"):
        with pytest.raises(PolicyError):
            view.public_rows(split)


def test_a10_public_material_contains_no_private_content():
    registry = demo_registry()
    view = PublicDataView(registry)
    material = {
        "train": view.train_prompts(),
        "dev_feedback": view.dev_feedback(0.75, 4, fake=True),
    }
    assert_payload_is_public(material, registry)  # must not raise
    text = canonical_json(material)
    assert "sel-" not in text and "test-" not in text
    assert "mushroom" not in text and "telescope" not in text

    leaked = {"notes": "the hidden prompt was: Write a sign for a mushroom stall. "
                        "Use at most 9 words and include the word forest."}
    with pytest.raises(PolicyError):
        assert_payload_is_public(leaked, registry)
    with pytest.raises(PolicyError):
        assert_payload_is_public({"path": "selection.jsonl"}, registry)


def test_controller_private_access_needs_purpose_and_test_is_single_claim(tmp_path):
    registry = demo_registry()
    gate = TestAccessGate(tmp_path / "gate.json")
    controller = ControllerDataAccess(registry, gate)
    assert controller.rows("selection", "selection_evaluation")
    with pytest.raises(PolicyError):
        controller.rows("selection", "researcher_request")
    with pytest.raises(PolicyError):
        controller.rows("test", "outer_screening", comparison_id="cmp-1")
    assert controller.rows("test", "final_comparison", comparison_id="cmp-1")
    with pytest.raises(PolicyError, match="exactly once"):
        controller.rows("test", "final_comparison", comparison_id="cmp-1")
    # A different comparison id is a different, still single, claim.
    assert controller.rows("test", "final_comparison", comparison_id="cmp-2")
    # The gate persists across instances (a fresh process cannot re-claim).
    assert TestAccessGate(tmp_path / "gate.json").claimed("cmp-1")
    assert json.loads((tmp_path / "gate.json").read_text())["test_claims"]


def test_train_rows_with_targets_are_refused(tmp_path):
    registry, dest = copied_registry(tmp_path)
    train_file = dest / "train_prompts.jsonl"
    first = json.loads(train_file.read_text(encoding="utf-8").splitlines()[0])
    first["target"] = "a smuggled answer"
    lines = train_file.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(first)
    train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Rebuild manifests so only the target rule (not the hash) can fire.
    rebuilt = DatasetRegistry(build_manifest_entries(dest), dest)
    with pytest.raises(DataError, match="target"):
        rebuilt.verify(rebuilt.manifest("train_prompts"))
    assert registry is not rebuilt
