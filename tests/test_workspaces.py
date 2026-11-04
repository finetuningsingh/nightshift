"""Acceptance A11 plus trial isolation and strict request validation."""

import json
from pathlib import Path

import pytest

from nightshift.errors import PolicyError
from nightshift.workspaces import (
    InvalidRequest,
    append_shared_note,
    create_trial_workspace,
    load_outbox,
    read_shared_notes,
    validate_mount_specs,
    validate_request,
    write_request,
)


def make_trial(tmp_path, trial_id="trial-1"):
    return create_trial_workspace(tmp_path / "trials", trial_id, ["r1", "r2"], seed=5)


def test_trial_layout_and_shared_notes_are_trial_scoped(tmp_path):
    first = make_trial(tmp_path, "trial-1")
    second = make_trial(tmp_path, "trial-2")
    append_shared_note(first, "r1", "filtering before training helped")
    assert "filtering before training helped" in read_shared_notes(first)
    assert "filtering" not in read_shared_notes(second)
    assert first.researchers["r1"] != first.researchers["r2"]
    assert (first.path / "shared" / "notes").is_dir()
    with pytest.raises(PolicyError):
        create_trial_workspace(tmp_path / "trials", "trial-1", ["r1"])


def test_valid_requests_of_every_type():
    allowed = {"learning_rate"}
    assert validate_request({"type": "generate", "prompt_ids": ["train-01"], "count": 2},
                            allowed).payload["count"] == 2
    assert validate_request({"type": "filter", "sample_set_id": "s1",
                             "verifier_ids": ["max_words"]}, allowed).type == "filter"
    assert validate_request({"type": "train", "sample_set_id": "s1",
                             "recipe_overrides": {"learning_rate": 0.001}}, allowed).type == "train"
    assert validate_request({"type": "evaluate_dev", "checkpoint_id": "c1"}, allowed)
    assert validate_request({"type": "submit", "checkpoint_id": "c1"}, allowed)
    assert validate_request({"type": "note", "text": "hello team"}, allowed)


@pytest.mark.parametrize(
    "raw",
    [
        "not-an-object",
        {"type": "shell", "command": "rm -rf /"},
        {"type": "generate", "prompt_ids": []},
        {"type": "generate", "prompt_ids": ["../selection"]},
        {"type": "generate", "prompt_ids": ["ok"], "count": 0},
        {"type": "generate", "prompt_ids": ["ok"], "sampling_params": {"temperature": 9}},
        {"type": "generate", "prompt_ids": ["ok"], "extra": 1},
        {"type": "filter", "sample_set_id": "s1", "verifier_ids": ["make_me_pass"]},
        {"type": "train", "sample_set_id": "s1", "recipe_overrides": {"base_model": "x"}},
        {"type": "evaluate_dev", "checkpoint_id": "a/b"},
        {"type": "note", "text": "   "},
        {"type": "submit"},
    ],
)
def test_untrusted_requests_are_rejected(raw):
    with pytest.raises(InvalidRequest):
        validate_request(raw, {"learning_rate"})


def test_outbox_roundtrip_and_malformed_files_become_errors(tmp_path):
    trial = make_trial(tmp_path)
    request = validate_request({"type": "note", "text": "first note"}, set(), "r1")
    write_request(trial, "r1", request)
    (trial.outbox("r1") / "broken.json").write_text("{oops", encoding="utf-8")
    (trial.outbox("r1") / "evil.json").write_text(
        json.dumps({"type": "generate", "prompt_ids": ["../../test"]}), encoding="utf-8"
    )
    valid, errors = load_outbox(trial, "r1", set())
    assert [r.request_id for r in valid] == [request.request_id]
    assert {e["file"] for e in errors} == {"broken.json", "evil.json"}


def mount_paths(tmp_path):
    trial = make_trial(tmp_path)
    pack = tmp_path / "pack"
    pack.mkdir()
    train = tmp_path / "train_prompts.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    return trial, pack, train


def test_a11_valid_mounts_pass(tmp_path):
    trial, pack, train = mount_paths(tmp_path)
    mounts = validate_mount_specs(
        [
            {"source": str(trial.researchers["r1"]), "target": "/workspace", "mode": "rw"},
            {"source": str(trial.shared_notes), "target": "/shared", "mode": "rw"},
            {"source": str(pack), "target": "/pack", "mode": "ro"},
            {"source": str(train), "target": "/data/train.jsonl", "mode": "ro"},
        ],
        workspace=trial.researchers["r1"],
        shared_notes=trial.shared_notes,
        pack_dir=pack,
        train_prompts=train,
    )
    assert len(mounts) == 4


@pytest.mark.parametrize(
    "source",
    [
        "~",
        str(Path.home()),
        "/home/alice",
        "/Users/alice",
        "/var/run/docker.sock",
        "/run/user/1000/podman.sock",
        "/srv/team/.ssh",
        "/srv/team/credentials",
        "/",
        "/mnt",
    ],
)
def test_a11_forbidden_mount_sources_rejected(tmp_path, source):
    trial, pack, train = mount_paths(tmp_path)
    with pytest.raises(PolicyError):
        validate_mount_specs(
            [{"source": source, "target": "/x", "mode": "ro"}],
            workspace=trial.researchers["r1"],
            shared_notes=trial.shared_notes,
            pack_dir=pack,
            train_prompts=train,
        )


def test_a11_private_split_and_writable_pack_mounts_rejected(tmp_path):
    trial, pack, train = mount_paths(tmp_path)
    private = tmp_path / "selection.jsonl"
    private.write_text("{}\n", encoding="utf-8")
    kwargs = dict(
        workspace=trial.researchers["r1"], shared_notes=trial.shared_notes,
        pack_dir=pack, train_prompts=train, private_paths=[private],
    )
    with pytest.raises(PolicyError, match="private"):
        validate_mount_specs([{"source": str(private), "target": "/x", "mode": "ro"}], **kwargs)
    with pytest.raises(PolicyError, match="private"):
        validate_mount_specs([{"source": str(tmp_path), "target": "/x", "mode": "ro"}], **kwargs)
    with pytest.raises(PolicyError):
        validate_mount_specs([{"source": str(pack), "target": "/pack", "mode": "rw"}], **kwargs)
    with pytest.raises(PolicyError):
        validate_mount_specs(
            [{"source": str(trial.researchers["r2"]), "target": "/other", "mode": "ro"}], **kwargs
        )
