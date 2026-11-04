"""Researcher pack: versioning, freezing, policy checks (spec 3.9, 3.10)."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import PolicyError
from .hashing import canonical_json, file_sha256, sha256_hex

REQUIRED_FILES = ("instructions.md", "recipe.yaml", "collaboration.toml", "budget_habits.md", "pack.toml")
REQUIRED_DIRS = ("tools", "initial_notes")
ALLOWED_TOP_LEVEL = set(REQUIRED_FILES) | set(REQUIRED_DIRS)
MAX_PACK_FILE_BYTES = 256 * 1024
MAX_PACK_TOTAL_BYTES = 2 * 1024 * 1024
MAX_PACK_FILES = 100

CONTROLLER_OWNED_RECIPE_KEYS = {
    "base_model", "datasets", "output_dir", "max_steps", "seed",
}

# References a candidate pack may never make (spec 3.10 step 3).
FORBIDDEN_PACK_REFERENCES = (
    "aggregate_ceiling_usd",
    "budget_file",
    "selection.jsonl",
    "test.jsonl",
    "[[data.manifests]]",
    "[budgets]",
    "scorer_id",
    "private selection",
    "private test",
)


@dataclass(frozen=True)
class PackMetadata:
    name: str
    parent_version: str
    editable_keys: tuple[str, ...]
    raw: dict[str, Any]


def default_pack_dir() -> Path:
    return Path(__file__).resolve().parent / "packs" / "default"


def _iter_pack_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        for dirname in list(dirnames):
            if (base / dirname).is_symlink():
                raise PolicyError(f"pack contains a symlink directory: {dirname}")
        for filename in filenames:
            path = base / filename
            if path.is_symlink():
                raise PolicyError(f"pack contains a symlink: {path.relative_to(root)}")
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def snapshot_tree(root: str | Path) -> dict[str, str]:
    """Map relative path -> content hash for every regular file under root."""
    base = Path(root)
    snapshot: dict[str, str] = {}
    if not base.exists():
        return snapshot
    for dirpath, _dirnames, filenames in os.walk(base):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_symlink() or not path.is_file():
                snapshot[path.relative_to(base).as_posix()] = "symlink-or-special"
            else:
                snapshot[path.relative_to(base).as_posix()] = file_sha256(path)
    return dict(sorted(snapshot.items()))


def compute_pack_version(pack_dir: str | Path) -> str:
    """SHA-256 over the sorted (path, content-hash) list (A20)."""
    root = Path(pack_dir)
    entries = [
        {"path": p.relative_to(root).as_posix(), "sha256": file_sha256(p)}
        for p in _iter_pack_files(root)
    ]
    return sha256_hex(canonical_json(entries))


def load_pack_metadata(pack_dir: str | Path) -> PackMetadata:
    pack_toml = Path(pack_dir) / "pack.toml"
    if not pack_toml.is_file():
        raise PolicyError(f"pack is missing pack.toml: {pack_dir}")
    try:
        raw = tomllib.loads(pack_toml.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PolicyError(f"pack.toml is not valid TOML: {exc}") from exc
    section = raw.get("pack", raw)
    if not isinstance(section, dict):
        raise PolicyError("pack.toml must contain a [pack] table")
    editable = section.get("editable_keys")
    if not isinstance(editable, list) or not all(isinstance(k, str) for k in editable):
        raise PolicyError("pack.toml editable_keys must be a list of strings")
    return PackMetadata(
        name=str(section.get("name", "")),
        parent_version=str(section.get("parent_version", "")),
        editable_keys=tuple(editable),
        raw=raw,
    )


def parse_recipe_keys(recipe_text: str) -> dict[str, str]:
    """Top-level `key: value` pairs of the flat recipe template."""
    keys: dict[str, str] = {}
    for line in recipe_text.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if match:
            keys[match.group(1)] = match.group(2).strip()
    return keys


def validate_pack_structure(pack_dir: str | Path) -> PackMetadata:
    root = Path(pack_dir)
    if not root.is_dir():
        raise PolicyError(f"pack directory does not exist: {root}")
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            raise PolicyError(f"pack is missing required file {name}")
    for name in REQUIRED_DIRS:
        if not (root / name).is_dir():
            raise PolicyError(f"pack is missing required directory {name}/")
    for entry in root.iterdir():
        if entry.name not in ALLOWED_TOP_LEVEL:
            raise PolicyError(f"unexpected top-level pack entry: {entry.name}")
    files = _iter_pack_files(root)
    if len(files) > MAX_PACK_FILES:
        raise PolicyError(f"pack has too many files ({len(files)} > {MAX_PACK_FILES})")
    total = 0
    for path in files:
        size = path.stat().st_size
        if size > MAX_PACK_FILE_BYTES:
            raise PolicyError(f"pack file too large: {path.relative_to(root)} ({size} bytes)")
        total += size
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PolicyError(f"pack file is not UTF-8 text: {path.relative_to(root)}") from exc
    if total > MAX_PACK_TOTAL_BYTES:
        raise PolicyError(f"pack too large in total ({total} bytes)")
    try:
        tomllib.loads((root / "collaboration.toml").read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PolicyError(f"collaboration.toml is not valid TOML: {exc}") from exc
    return load_pack_metadata(root)


def _copy_tree_writable(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)
    for dirpath, _dirs, filenames in os.walk(dst):
        os.chmod(dirpath, 0o755)
        for filename in filenames:
            os.chmod(Path(dirpath) / filename, 0o644)


def create_pack(dest: str | Path, source: str | Path | None = None) -> Path:
    """Copy the bundled default pack (or `source`) to `dest`, writable."""
    src = Path(source) if source is not None else default_pack_dir()
    dst = Path(dest)
    if dst.exists() and any(dst.iterdir()):
        raise PolicyError(f"pack destination already exists and is not empty: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.rmdir()
    _copy_tree_writable(src, dst)
    validate_pack_structure(dst)
    return dst


def make_scratch_copy(pack_dir: str | Path, dest: str | Path) -> Path:
    dst = Path(dest)
    if dst.exists():
        raise PolicyError(f"scratch destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree_writable(Path(pack_dir), dst)
    return dst


def freeze_pack(pack_dir: str | Path, packs_root: str | Path) -> tuple[str, Path]:
    """Freeze a pack as a read-only copy under packs/<version>/."""
    src = Path(pack_dir)
    validate_pack_structure(src)
    version = compute_pack_version(src)
    dst = Path(packs_root) / version
    if dst.exists():
        if compute_pack_version(dst) != version:
            raise PolicyError(f"frozen pack {version} exists with different content")
        return version, dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    for dirpath, _dirs, filenames in os.walk(dst):
        for filename in filenames:
            os.chmod(Path(dirpath) / filename, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(dirpath, 0o555)
    if compute_pack_version(dst) != version:
        raise PolicyError("frozen pack content changed during freeze")
    return version, dst


def check_reviser_edits(
    before: dict[str, str], after: dict[str, str], allowed_prefix: str
) -> list[str]:
    """Reject any created/modified/deleted path outside `allowed_prefix` (A15)."""
    prefix = allowed_prefix.strip("/") + "/"
    changed = sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )
    outside = [path for path in changed if not path.startswith(prefix)]
    if outside:
        raise PolicyError(f"reviser edited file(s) outside the pack: {outside}")
    return changed


def check_pack_policy(
    candidate_dir: str | Path,
    base_dir: str | Path,
    changed_paths: list[str] | None = None,
) -> list[str]:
    """Pack policy check (spec 3.10 step 3, A15). Returns changed pack paths."""
    candidate = Path(candidate_dir)
    base = Path(base_dir)
    validate_pack_structure(candidate)
    base_metadata = load_pack_metadata(base)
    cand_metadata = load_pack_metadata(candidate)

    if cand_metadata.name != base_metadata.name:
        raise PolicyError("pack policy: pack.toml name is controller-owned and changed")
    if cand_metadata.editable_keys != base_metadata.editable_keys:
        raise PolicyError("pack policy: pack.toml editable_keys is controller-owned and changed")
    if cand_metadata.parent_version != base_metadata.parent_version:
        raise PolicyError("pack policy: pack.toml parent_version is set by the controller at freeze")
    base_controller = (base_metadata.raw or {}).get("controller")
    cand_controller = (cand_metadata.raw or {}).get("controller")
    if base_controller != cand_controller:
        raise PolicyError("pack policy: pack.toml [controller] fields changed")

    base_recipe = parse_recipe_keys((base / "recipe.yaml").read_text(encoding="utf-8"))
    cand_recipe = parse_recipe_keys((candidate / "recipe.yaml").read_text(encoding="utf-8"))
    for key in CONTROLLER_OWNED_RECIPE_KEYS:
        if key in base_recipe and cand_recipe.get(key) != base_recipe.get(key):
            raise PolicyError(f"pack policy: recipe key {key!r} is controller-owned and changed")
    for key in cand_recipe:
        if key in CONTROLLER_OWNED_RECIPE_KEYS and key not in base_recipe:
            raise PolicyError(f"pack policy: recipe key {key!r} is controller-owned")

    for path in _iter_pack_files(candidate):
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_PACK_REFERENCES:
            if token in text:
                raise PolicyError(
                    f"pack policy: {path.relative_to(candidate)} references "
                    f"controller/private material ({token!r})"
                )

    if changed_paths is not None:
        for changed in changed_paths:
            rel = Path(changed)
            if rel.is_absolute() or ".." in rel.parts:
                raise PolicyError(f"pack policy: edit outside the pack: {changed}")

    before = snapshot_tree(base)
    after = snapshot_tree(candidate)
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
