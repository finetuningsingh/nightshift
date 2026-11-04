"""Frozen data manifests and split access control (spec 3.5).

Researchers and the reviser only ever see `train_prompts` and aggregate dev
feedback. `selection` and `test` rows, paths, and per-item results are
reachable only through the controller-only accessor, and `test` additionally
requires the final-comparison gate (spec 3.11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import DataError, PolicyError
from .hashing import canonical_json, file_sha256

SPLITS = ("train_prompts", "dev", "selection", "test")
PRIVATE_SPLITS = ("selection", "test")

TRAIN_ROW_KEYS = {"id", "prompt", "verifiers"}
TARGET_KEYS = {"target", "answer", "completion", "response", "output"}


@dataclass(frozen=True)
class DataManifest:
    id: str
    source: str
    revision: str
    split: str
    path: str
    row_count: int
    sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataManifest":
        required = ("id", "source", "revision", "split", "path", "row_count", "sha256")
        missing = [k for k in required if k not in raw]
        if missing:
            raise DataError(f"manifest entry missing field(s): {', '.join(missing)}")
        split = str(raw["split"])
        if split not in SPLITS:
            raise DataError(f"unknown split {split!r}; expected one of {SPLITS}")
        row_count = raw["row_count"]
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
            raise DataError("manifest row_count must be a non-negative integer")
        return cls(
            id=str(raw["id"]),
            source=str(raw["source"]),
            revision=str(raw["revision"]),
            split=split,
            path=str(raw["path"]),
            row_count=int(row_count),
            sha256=str(raw["sha256"]).lower(),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "revision": self.revision,
            "split": self.split,
            "path": self.path,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{path}:{line_no} is not valid JSONL") from exc
        if not isinstance(row, dict):
            raise DataError(f"{path}:{line_no} is not a JSON object")
        rows.append(row)
    return rows


def _validate_rows(manifest: DataManifest, rows: list[dict[str, Any]]) -> None:
    if len(rows) != manifest.row_count:
        raise DataError(
            f"manifest {manifest.id}: row_count {manifest.row_count} != "
            f"actual {len(rows)} rows"
        )
    seen: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise DataError(f"manifest {manifest.id}: every row needs a string id")
        if row_id in seen:
            raise DataError(f"manifest {manifest.id}: duplicate row id {row_id!r}")
        seen.add(row_id)
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            raise DataError(f"manifest {manifest.id}: row {row_id!r} needs a prompt")
        if manifest.split == "train_prompts":
            leaked = TARGET_KEYS & set(row)
            if leaked:
                raise DataError(
                    f"train_prompts row {row_id!r} carries target field(s) "
                    f"{sorted(leaked)}; the split is prompts only"
                )
            unknown = set(row) - TRAIN_ROW_KEYS
            if unknown:
                raise DataError(
                    f"train_prompts row {row_id!r} has unexpected field(s) {sorted(unknown)}"
                )


class TestAccessGate:
    """Persistent gate: `test` may be claimed exactly once per comparison id."""

    __test__ = False  # production class; pytest must not collect it

    def __init__(self, state_path: str | Path):
        self.state_path = Path(state_path)

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"test_claims": {}}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataError(f"corrupt test-access state {self.state_path}") from exc

    def claimed(self, comparison_id: str) -> bool:
        return comparison_id in self._load().get("test_claims", {})

    def claim(self, comparison_id: str) -> None:
        if not comparison_id:
            raise PolicyError("test access requires a comparison id")
        state = self._load()
        claims = state.setdefault("test_claims", {})
        if comparison_id in claims:
            raise PolicyError(
                f"test split was already accessed for comparison {comparison_id!r}; "
                "it may be accessed exactly once per comparison id"
            )
        claims[comparison_id] = {"claimed": True}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)


class DatasetRegistry:
    """Verified view of all frozen manifests. Controller-owned."""

    def __init__(
        self,
        manifests: Iterable[DataManifest | dict[str, Any]],
        base_dir: str | Path,
    ):
        self.base_dir = Path(base_dir)
        self._by_split: dict[str, DataManifest] = {}
        self._by_id: dict[str, DataManifest] = {}
        for entry in manifests:
            manifest = entry if isinstance(entry, DataManifest) else DataManifest.from_dict(entry)
            if manifest.split in self._by_split:
                raise DataError(f"duplicate manifest for split {manifest.split!r}")
            if manifest.id in self._by_id:
                raise DataError(f"duplicate manifest id {manifest.id!r}")
            self._by_split[manifest.split] = manifest
            self._by_id[manifest.id] = manifest

    @classmethod
    def from_manifest_file(
        cls, manifest_path: str | Path, base_dir: str | Path | None = None
    ) -> "DatasetRegistry":
        path = Path(manifest_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise DataError(f"unreadable manifest file {path}") from exc
        entries = raw.get("manifests") if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise DataError("manifest file must hold a list of manifest entries")
        return cls(entries, base_dir if base_dir is not None else path.parent)

    @classmethod
    def from_config(cls, config: Any) -> "DatasetRegistry":
        if config.data_manifests:
            return cls(config.data_manifests, config.base_dir)
        return demo_registry()

    def manifest(self, split: str) -> DataManifest:
        if split not in SPLITS:
            raise DataError(f"unknown split {split!r}")
        try:
            return self._by_split[split]
        except KeyError as exc:
            raise DataError(f"no manifest configured for split {split!r}") from exc

    def manifests(self) -> list[DataManifest]:
        return [self._by_split[s] for s in SPLITS if s in self._by_split]

    def resolve_path(self, manifest: DataManifest) -> Path:
        path = Path(manifest.path)
        return path if path.is_absolute() else self.base_dir / path

    def verify(self, manifest: DataManifest) -> list[dict[str, Any]]:
        """Refuse (A9) any file whose hash or row count disagrees."""
        path = self.resolve_path(manifest)
        if not path.is_file():
            raise DataError(f"manifest {manifest.id}: data file missing for split {manifest.split}")
        actual = file_sha256(path)
        if actual != manifest.sha256:
            raise DataError(
                f"manifest {manifest.id}: sha256 mismatch for split {manifest.split} "
                f"(manifest {manifest.sha256}, file {actual}); refusing the file"
            )
        rows = load_jsonl_rows(path)
        _validate_rows(manifest, rows)
        return rows

    def verify_all(self) -> dict[str, int]:
        return {m.split: len(self.verify(m)) for m in self.manifests()}


class PublicDataView:
    """The only data surface handed to researchers and the reviser (A10)."""

    def __init__(self, registry: DatasetRegistry):
        self._registry = registry

    @property
    def registry(self) -> DatasetRegistry:
        return self._registry

    def train_prompts(self) -> list[dict[str, Any]]:
        rows = self._registry.verify(self._registry.manifest("train_prompts"))
        return [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "verifiers": row.get("verifiers", []),
            }
            for row in rows
        ]

    def train_prompt_ids(self) -> list[str]:
        return [row["id"] for row in self.train_prompts()]

    def public_rows(self, split: str) -> list[dict[str, Any]]:
        if split != "train_prompts":
            raise PolicyError(
                f"researcher/reviser access to split {split!r} is denied; "
                "only train_prompts and dev feedback are public"
            )
        return self.train_prompts()

    def rows(self, split: str) -> list[dict[str, Any]]:
        return self.public_rows(split)

    @staticmethod
    def dev_feedback(
        score: float | None,
        n: int,
        failures: int = 0,
        status: str = "succeeded",
        fake: bool = False,
    ) -> dict[str, Any]:
        """Aggregate public dev feedback: never per-item, never a path."""
        feedback: dict[str, Any] = {
            "split": "dev",
            "status": status,
            "n": int(n),
            "failures": int(failures),
            "fake": bool(fake),
        }
        if status == "succeeded" and score is not None:
            feedback["score"] = float(score)
        return feedback


class ControllerDataAccess:
    """Controller-only rows. Private splits need an explicit purpose."""

    def __init__(self, registry: DatasetRegistry, test_gate: TestAccessGate | None = None):
        self.registry = registry
        self.test_gate = test_gate

    def check_purpose(
        self, split: str, purpose: str, comparison_id: str | None = None, claim_test: bool = False
    ) -> None:
        if split == "selection" and purpose not in ("selection_evaluation", "outer_screening"):
            raise PolicyError(
                f"selection access denied for purpose {purpose!r}; "
                "it is reserved for outer-loop acceptance evaluation"
            )
        if split == "test":
            if purpose != "final_comparison":
                raise PolicyError(
                    f"test access denied for purpose {purpose!r}; "
                    "it is reserved for the final matched comparison"
                )
            if claim_test:
                if self.test_gate is None:
                    raise PolicyError("test access requires a TestAccessGate")
                self.test_gate.claim(comparison_id or "")

    def rows(
        self,
        split: str,
        purpose: str,
        comparison_id: str | None = None,
    ) -> list[dict[str, Any]]:
        manifest = self.registry.manifest(split)
        self.check_purpose(split, purpose, comparison_id, claim_test=True)
        return self.registry.verify(manifest)


def private_markers(registry: DatasetRegistry) -> set[str]:
    """Strings that must never appear in researcher/reviser-facing material."""
    markers: set[str] = set()
    for split in PRIVATE_SPLITS:
        try:
            manifest = registry.manifest(split)
        except DataError:
            continue
        markers.add(manifest.path)
        markers.add(Path(manifest.path).name)
        markers.add(str(registry.resolve_path(manifest)))
        for row in registry.verify(manifest):
            markers.add(str(row.get("id", "")))
            markers.add(str(row.get("prompt", "")))
    markers.discard("")
    return markers


def assert_payload_is_public(payload: Any, registry: DatasetRegistry) -> None:
    """Raise if a payload leaks private rows, ids, or paths (A10)."""
    text = canonical_json(payload) if not isinstance(payload, str) else payload
    for marker in private_markers(registry):
        if marker and marker in text:
            raise PolicyError(
                "payload contains private selection/test material and must not "
                "be shown to researchers or the reviser"
            )


DEMO_REVISION = "toy-1"
DEMO_SOURCE = "nightshift-bundled-toy"
DEMO_FILES = {
    "train_prompts": "train_prompts.jsonl",
    "dev": "dev.jsonl",
    "selection": "selection.jsonl",
    "test": "test.jsonl",
}


def demo_data_dir() -> Path:
    return Path(__file__).resolve().parent / "demo_data"


def build_manifest_entries(
    data_dir: str | Path, source: str = DEMO_SOURCE, revision: str = DEMO_REVISION
) -> list[dict[str, Any]]:
    """Materialise frozen manifest entries with freshly computed hashes."""
    root = Path(data_dir)
    entries: list[dict[str, Any]] = []
    for split in SPLITS:
        filename = DEMO_FILES[split]
        path = root / filename
        if not path.is_file():
            raise DataError(f"demo data file missing: {filename}")
        rows = load_jsonl_rows(path)
        entries.append(
            {
                "id": f"toy-{split}",
                "source": source,
                "revision": revision,
                "split": split,
                "path": filename,
                "row_count": len(rows),
                "sha256": file_sha256(path),
            }
        )
    return entries


def write_manifest_file(data_dir: str | Path, out_path: str | Path) -> Path:
    entries = build_manifest_entries(data_dir)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"manifests": entries}, indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def demo_registry() -> DatasetRegistry:
    root = demo_data_dir()
    return DatasetRegistry(build_manifest_entries(root), root)
