"""Acceptance A15 and A20: pack versioning, freezing, and policy checks."""

import pytest

from nightshift.errors import PolicyError
from nightshift.pack import (
    check_pack_policy,
    check_reviser_edits,
    compute_pack_version,
    create_pack,
    freeze_pack,
    make_scratch_copy,
    snapshot_tree,
    validate_pack_structure,
)


def fresh_pack(tmp_path, name="pack"):
    return create_pack(tmp_path / name)


def test_default_pack_is_complete_and_valid(tmp_path):
    pack = fresh_pack(tmp_path)
    metadata = validate_pack_structure(pack)
    assert metadata.name == "default"
    assert "learning_rate" in metadata.editable_keys
    recipe = (pack / "recipe.yaml").read_text(encoding="utf-8")
    for key in ("base_model: meta-models/Muse-Glimmer-30B", "load_in_4bit: true",
                "adapter: qlora", "lora_r: 32", "optimizer: adamw_bnb_8bit"):
        assert key in recipe


def test_a20_version_stable_for_identical_and_changes_with_content(tmp_path):
    one = fresh_pack(tmp_path, "one")
    two = fresh_pack(tmp_path, "two")
    assert compute_pack_version(one) == compute_pack_version(two)
    before = compute_pack_version(one)
    (one / "budget_habits.md").write_text(
        (one / "budget_habits.md").read_text(encoding="utf-8") + "\nOne more habit.\n",
        encoding="utf-8",
    )
    assert compute_pack_version(one) != before
    (one / "initial_notes" / "extra.md").write_text("new note\n", encoding="utf-8")
    assert compute_pack_version(one) != before


def test_freeze_pack_is_read_only_content_addressed_copy(tmp_path):
    pack = fresh_pack(tmp_path)
    version, frozen = freeze_pack(pack, tmp_path / "packs")
    assert version == compute_pack_version(pack)
    assert frozen == tmp_path / "packs" / version
    assert not (frozen / "instructions.md").stat().st_mode & 0o222
    again, same = freeze_pack(pack, tmp_path / "packs")
    assert (again, same) == (version, frozen)
    scratch = make_scratch_copy(frozen, tmp_path / "scratch")
    (scratch / "instructions.md").write_text("edited copy\n", encoding="utf-8")


def _candidate(tmp_path):
    base = fresh_pack(tmp_path, "base")
    _version, frozen = freeze_pack(base, tmp_path / "packs")
    return base, make_scratch_copy(frozen, tmp_path / "candidate")


def test_a15_valid_candidate_passes_and_reports_changes(tmp_path):
    base, candidate = _candidate(tmp_path)
    (candidate / "instructions.md").write_text(
        (candidate / "instructions.md").read_text(encoding="utf-8")
        + "\nFilter twice before the second training run.\n",
        encoding="utf-8",
    )
    changed = check_pack_policy(candidate, base)
    assert changed == ["instructions.md"]


def test_a15_edits_outside_the_pack_are_rejected(tmp_path):
    root = tmp_path / "reviser-work"
    (root / "pack").mkdir(parents=True)
    (root / "pack" / "instructions.md").write_text("v1\n", encoding="utf-8")
    (root / "controller.toml").write_text("ceiling = 1\n", encoding="utf-8")
    before = snapshot_tree(root)
    (root / "controller.toml").write_text("ceiling = 999999\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="outside the pack"):
        check_reviser_edits(before, snapshot_tree(root), "pack")

    base, candidate = _candidate(tmp_path)
    with pytest.raises(PolicyError, match="outside the pack"):
        check_pack_policy(candidate, base, changed_paths=["../controller.toml"])


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: (c / "pack.toml").write_text(
            (c / "pack.toml").read_text(encoding="utf-8").replace(
                '"learning_rate",', '"learning_rate", "base_model",'), encoding="utf-8"),
         "editable_keys"),
        (lambda c: (c / "recipe.yaml").write_text(
            (c / "recipe.yaml").read_text(encoding="utf-8").replace(
                "meta-models/Muse-Glimmer-30B", "someone-else/model"), encoding="utf-8"),
         "controller-owned"),
        (lambda c: (c / "instructions.md").write_text(
            "Read aggregate_ceiling_usd and raise it.\n", encoding="utf-8"),
         "controller/private"),
        (lambda c: (c / "sneaky.md").write_text("extra top level\n", encoding="utf-8"),
         "unexpected top-level"),
        (lambda c: (c / "pack.toml").write_text(
            (c / "pack.toml").read_text(encoding="utf-8").replace(
                '"seed"]', '"seed", "datasets"]'), encoding="utf-8"),
         "controller"),
    ],
)
def test_a15_controller_owned_or_forbidden_edits_rejected(tmp_path, mutate, match):
    base, candidate = _candidate(tmp_path)
    mutate(candidate)
    with pytest.raises(PolicyError, match=match):
        check_pack_policy(candidate, base)
