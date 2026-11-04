"""Acceptance A12: checkpoint validation and the content-addressed store."""

import json
import os
import struct

import pytest

from nightshift.checkpoints import (
    make_safetensors_bytes,
    parse_safetensors_header,
    store_checkpoint,
    validate_checkpoint_dir,
    write_fake_checkpoint,
)
from nightshift.errors import PolicyError


def valid_checkpoint(tmp_path, name="ckpt"):
    path = tmp_path / name
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(make_safetensors_bytes(b"weights"))
    (path / "adapter_config.json").write_text('{"fake": true}', encoding="utf-8")
    return path


def test_a12_valid_safetensors_adapter_accepted_and_stored(tmp_path):
    ckpt = valid_checkpoint(tmp_path)
    info = validate_checkpoint_dir(ckpt)
    assert len(info.checkpoint_hash) == 64
    stored = store_checkpoint(ckpt, tmp_path / "store", fake=True)
    assert stored.path == tmp_path / "store" / info.checkpoint_hash
    record = json.loads((tmp_path / "store" / f"{info.checkpoint_hash}.json").read_text())
    assert record["fake"] is True
    header = parse_safetensors_header(ckpt / "adapter_model.safetensors")
    assert header["adapter.weight"]["shape"] == [7]


def test_a12_fake_checkpoint_helper_is_valid_and_labelled(tmp_path):
    ckpt = write_fake_checkpoint(tmp_path / "fake", marker="trial-1", quality=0.5)
    validate_checkpoint_dir(ckpt)
    config = json.loads((ckpt / "adapter_config.json").read_text())
    assert config["fake"] is True and config["fake_quality"] == 0.5


@pytest.mark.parametrize("suffix", [".bin", ".pt", ".pth", ".pkl", ".ckpt"])
def test_a12_pickle_formats_rejected(tmp_path, suffix):
    ckpt = valid_checkpoint(tmp_path)
    (ckpt / f"weights{suffix}").write_bytes(b"pickle-ish")
    with pytest.raises(PolicyError, match="pickle"):
        validate_checkpoint_dir(ckpt)


def test_a12_symlink_and_path_escape_rejected(tmp_path):
    ckpt = valid_checkpoint(tmp_path)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(make_safetensors_bytes())
    link = ckpt / "escape.safetensors"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(PolicyError, match="symlink"):
        validate_checkpoint_dir(ckpt)


def _raw_safetensors(header_obj, data=b""):
    header = json.dumps(header_obj).encode()
    return struct.pack("<Q", len(header)) + header + data


@pytest.mark.parametrize(
    "blob",
    [
        b"tiny",
        struct.pack("<Q", 999) + b"{}",
        struct.pack("<Q", 5) + b"not json!",
        _raw_safetensors({"__metadata__": {"format": "pt"}}),
        _raw_safetensors({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}),
        _raw_safetensors({"w": {"dtype": "U8", "shape": [2], "data_offsets": [0, 99]}}),
        _raw_safetensors({"w": {"dtype": "NOPE", "shape": [1], "data_offsets": [0, 1]}}, b"x"),
        _raw_safetensors({"w": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]}}, b"x"),
    ],
)
def test_a12_malformed_safetensors_headers_rejected(tmp_path, blob):
    ckpt = tmp_path / "bad"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(blob)
    with pytest.raises(PolicyError, match="malformed safetensors"):
        validate_checkpoint_dir(ckpt)


def test_a12_executable_and_unexpected_files_rejected(tmp_path):
    ckpt = valid_checkpoint(tmp_path)
    script = ckpt / "run_me.json"
    script.write_text("{}", encoding="utf-8")
    os.chmod(script, 0o755)
    with pytest.raises(PolicyError, match="executable"):
        validate_checkpoint_dir(ckpt)
    os.chmod(script, 0o644)
    (ckpt / "notes.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(PolicyError, match="unexpected"):
        validate_checkpoint_dir(ckpt)
