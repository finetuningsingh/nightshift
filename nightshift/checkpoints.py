"""Checkpoint safety: pure-Python safetensors validation (spec 3.7, A12).

No safetensors/torch dependency: headers are parsed by hand and nothing is
deserialised as code. Pickle-based formats, symlinks, path escapes, and
executable files are rejected.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import PolicyError
from .hashing import canonical_json, file_sha256, sha256_hex

PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".pkl", ".ckpt"}
ALLOWED_SUFFIXES = {".safetensors", ".json", ".yaml", ".yml"}
MAX_HEADER_BYTES = 64 * 1024 * 1024
_DTYPE_SIZES = {
    "U8": 1, "I8": 1, "BOOL": 1, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4, "F64": 8, "I64": 8, "U64": 8,
}


def parse_safetensors_header(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    size = file_path.stat().st_size
    if size < 8:
        raise PolicyError(f"malformed safetensors {file_path.name}: shorter than header length")
    with file_path.open("rb") as handle:
        prefix = handle.read(8)
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len <= 0 or header_len > MAX_HEADER_BYTES or 8 + header_len > size:
            raise PolicyError(f"malformed safetensors {file_path.name}: bad header length")
        header_bytes = handle.read(header_len)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyError(f"malformed safetensors {file_path.name}: header is not JSON") from exc
    if not isinstance(header, dict) or not header:
        raise PolicyError(f"malformed safetensors {file_path.name}: header is not an object")
    data_len = size - 8 - header_len
    tensors = 0
    for name, meta in header.items():
        if name == "__metadata__":
            if not isinstance(meta, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
            ):
                raise PolicyError(f"malformed safetensors {file_path.name}: bad __metadata__")
            continue
        tensors += 1
        if not isinstance(meta, dict):
            raise PolicyError(f"malformed safetensors {file_path.name}: tensor {name!r}")
        dtype, shape, offsets = meta.get("dtype"), meta.get("shape"), meta.get("data_offsets")
        if dtype not in _DTYPE_SIZES:
            raise PolicyError(f"malformed safetensors {file_path.name}: bad dtype {dtype!r}")
        if not isinstance(shape, list) or any(
            isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in shape
        ):
            raise PolicyError(f"malformed safetensors {file_path.name}: bad shape for {name!r}")
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or any(isinstance(o, bool) or not isinstance(o, int) for o in offsets)
        ):
            raise PolicyError(f"malformed safetensors {file_path.name}: bad offsets for {name!r}")
        start, end = offsets
        elements = 1
        for dim in shape:
            elements *= dim
        expected = elements * _DTYPE_SIZES[dtype]
        if start < 0 or end < start or end > data_len or (end - start) != expected:
            raise PolicyError(
                f"malformed safetensors {file_path.name}: offsets for {name!r} "
                "do not match dtype/shape or exceed the data segment"
            )
    if tensors == 0:
        raise PolicyError(f"malformed safetensors {file_path.name}: no tensors")
    return header


def make_safetensors_bytes(
    tensors: Mapping[str, tuple[str, list[int], bytes]] | bytes | bytearray | None = None,
    payload: bytes = b"nightshift-fake-adapter",
) -> bytes:
    """Build a small valid safetensors file byte-by-byte (test/fake helper).

    Accepts either a tensor mapping or a bare byte payload for one default
    U8 tensor, plus `payload=` as a keyword.
    """
    if isinstance(tensors, (bytes, bytearray)):
        payload = bytes(tensors)
        tensors = None
    chosen = dict(tensors) if tensors is not None else {
        "adapter.weight": ("U8", [len(payload)], bytes(payload))
    }
    header: dict[str, Any] = {}
    data = bytearray()
    for name in sorted(chosen):
        dtype, shape, raw = chosen[name]
        start = len(data)
        data.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(data)]}
    header_bytes = canonical_json(header).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(data)


@dataclass(frozen=True)
class CheckpointInfo:
    checkpoint_hash: str
    files: tuple[dict[str, Any], ...]
    path: Path

    def to_record(self) -> dict[str, Any]:
        return {"checkpoint_hash": self.checkpoint_hash, "files": list(self.files)}


def validate_checkpoint_dir(path: str | Path) -> CheckpointInfo:
    root = Path(path)
    if root.is_symlink():
        raise PolicyError("checkpoint path is a symlink")
    if not root.is_dir():
        raise PolicyError(f"checkpoint is not a directory: {root}")
    resolved_root = root.resolve()
    files: list[dict[str, Any]] = []
    saw_safetensors = False
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        for dirname in dirnames:
            entry = base / dirname
            if entry.is_symlink():
                raise PolicyError(f"checkpoint contains a symlink: {entry.relative_to(root)}")
        for filename in filenames:
            entry = base / filename
            rel = entry.relative_to(root)
            if entry.is_symlink():
                raise PolicyError(f"checkpoint contains a symlink (path escape risk): {rel}")
            try:
                entry.resolve().relative_to(resolved_root)
            except ValueError as exc:
                raise PolicyError(f"checkpoint path escapes its directory: {rel}") from exc
            suffix = entry.suffix.lower()
            if suffix in PICKLE_SUFFIXES:
                raise PolicyError(f"pickle-based checkpoint format rejected: {rel}")
            if suffix not in ALLOWED_SUFFIXES:
                raise PolicyError(f"unexpected checkpoint file rejected: {rel}")
            if entry.stat().st_size == 0:
                raise PolicyError(f"empty checkpoint file rejected: {rel}")
            if entry.stat().st_mode & 0o111:
                raise PolicyError(f"executable checkpoint file rejected: {rel}")
            if suffix == ".safetensors":
                parse_safetensors_header(entry)
                saw_safetensors = True
            elif suffix == ".json":
                try:
                    json.loads(entry.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PolicyError(f"checkpoint JSON config is invalid: {rel}") from exc
            else:  # YAML config: must at least be text
                try:
                    text = entry.read_bytes().decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise PolicyError(f"checkpoint YAML config is not text: {rel}") from exc
                if "\x00" in text:
                    raise PolicyError(f"checkpoint YAML config contains NUL bytes: {rel}")
            files.append({"path": rel.as_posix(), "sha256": file_sha256(entry)})
    if not saw_safetensors:
        raise PolicyError("checkpoint contains no .safetensors weights")
    files.sort(key=lambda item: item["path"])
    return CheckpointInfo(sha256_hex(canonical_json(files)), tuple(files), root)


def store_checkpoint(
    path: str | Path, store_root: str | Path, fake: bool = False
) -> CheckpointInfo:
    """Validate, then copy by content hash into the controller-owned store."""
    info = validate_checkpoint_dir(path)
    store = Path(store_root)
    dest = store / info.checkpoint_hash
    if not dest.exists():
        store.mkdir(parents=True, exist_ok=True)
        shutil.copytree(info.path, dest)
        stored = validate_checkpoint_dir(dest)
        if stored.checkpoint_hash != info.checkpoint_hash:
            raise PolicyError("stored checkpoint hash does not match the validated source")
    record = dict(info.to_record())
    record["fake"] = bool(fake)
    (store / f"{info.checkpoint_hash}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return CheckpointInfo(info.checkpoint_hash, info.files, dest)


def write_fake_checkpoint(
    dest: str | Path, marker: str = "", base_model: str = "", quality: float | None = None
) -> Path:
    """Deterministic labelled fake adapter (fake training output only)."""
    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    payload = sha256_hex(f"nightshift-fake-adapter:{marker}:{quality}").encode("ascii")
    (target / "adapter_model.safetensors").write_bytes(make_safetensors_bytes(payload=payload))
    config: dict[str, Any] = {
        "fake": True,
        "adapter_type": "qlora-fake",
        "base_model": base_model,
        "marker": sha256_hex(marker)[:16],
    }
    if quality is not None:
        # The fake Inspect model derives deterministic output quality from this.
        config["fake_quality"] = float(quality)
    (target / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    validate_checkpoint_dir(target)
    return target
