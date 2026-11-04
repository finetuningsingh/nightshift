"""Training: recipe rendering, fake training, Axolotl boundary (spec 3.7).

The controller forces base model (id recorded with revision), dataset path,
output dir, step/epoch ceilings, and seed. Only pack-declared editable keys
may vary. Fake training writes a deterministic labelled-fake adapter whose
content depends on samples/seed/quality and path-independent recipe
settings only. Real training is an Axolotl job through the provider/job
interface and is NOT YET VALIDATED LIVE. Axolotl is never imported here.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoints import validate_checkpoint_dir, write_fake_checkpoint
from .config import assert_pinned_base_model
from .errors import NightshiftError, PolicyError, ProviderError
from .hashing import canonical_json, hash_object
from .pack import CONTROLLER_OWNED_RECIPE_KEYS, load_pack_metadata, parse_recipe_keys
from .providers import JobSpec, RentalRequest

DEFAULT_MAX_STEPS_CEILING = 50
DEFAULT_MAX_EPOCHS_CEILING = 1


class TrainingError(NightshiftError):
    """Training was refused or failed."""


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value).strip().strip('"'))
    except (TypeError, ValueError):
        return None


@dataclass
class RenderedRecipe:
    text: str
    effective: dict[str, str]
    forced: dict[str, Any]
    path: Path | None = None

    def content_fingerprint(self) -> dict[str, str]:
        """Path-independent effective settings (for deterministic artefacts)."""
        return {
            key: value
            for key, value in sorted(self.effective.items())
            if key not in ("output_dir", "datasets")
        }


def render_recipe(
    template_text: str,
    overrides: dict[str, Any] | None = None,
    allowed_keys: set[str] | tuple[str, ...] = (),
    base_model: str = "",
    base_revision: str = "",
    dataset_path: str = "",
    output_dir: str = "",
    seed: int = 0,
    max_steps_ceiling: int = DEFAULT_MAX_STEPS_CEILING,
    max_epochs_ceiling: int = DEFAULT_MAX_EPOCHS_CEILING,
) -> RenderedRecipe:
    """Render the pack template with controller-owned keys forced."""
    overrides = dict(overrides or {})
    allowed = set(allowed_keys)
    for key, value in overrides.items():
        if key in CONTROLLER_OWNED_RECIPE_KEYS:
            raise PolicyError(f"recipe key {key!r} is controller-owned and cannot be overridden")
        if key not in allowed:
            raise PolicyError(f"recipe key {key!r} is not in the pack's editable-key list")
        if not isinstance(value, (str, int, float, bool)):
            raise PolicyError(f"recipe override {key!r} must be a scalar")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
            raise PolicyError(f"recipe override {key!r} must be positive")

    template_keys = parse_recipe_keys(template_text)
    epochs = _as_int(overrides.get("num_epochs", template_keys.get("num_epochs", 1))) or 1
    epochs = max(1, min(int(epochs), int(max_epochs_ceiling)))
    forced: dict[str, Any] = {
        "base_model": base_model,
        "output_dir": output_dir,
        "max_steps": int(max_steps_ceiling),
        "seed": int(seed),
        "num_epochs": epochs,
    }
    values: dict[str, Any] = dict(overrides)
    values.update(forced)

    out_lines: list[str] = []
    seen: set[str] = set()
    lines = template_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        is_top_key = bool(line) and not line[0].isspace() and ":" in line and not stripped.startswith("#")
        if is_top_key:
            key = line.split(":", 1)[0].strip()
            if key == "datasets":
                out_lines.append("datasets:")
                out_lines.append(f"  - path: {_yaml_scalar(dataset_path)}")
                out_lines.append("    type: alpaca")
                seen.add("datasets")
                index += 1
                while index < len(lines) and (
                    not lines[index].strip()
                    or lines[index][0].isspace()
                    or lines[index].lstrip().startswith("-")
                ):
                    index += 1
                continue
            if key in values:
                out_lines.append(f"{key}: {_yaml_scalar(values[key])}")
                seen.add(key)
                index += 1
                continue
        out_lines.append(line)
        index += 1
    if "datasets" not in seen:
        out_lines.append("datasets:")
        out_lines.append(f"  - path: {_yaml_scalar(dataset_path)}")
        out_lines.append("    type: alpaca")
    for key in sorted(values):
        if key not in seen and key not in parse_recipe_keys("\n".join(out_lines)):
            out_lines.append(f"{key}: {_yaml_scalar(values[key])}")
    header = (
        f"# Rendered by the Nightshift controller. base_model_revision={base_revision} "
        "(controller-owned; template and overrides cannot change it)."
    )
    text = header + "\n" + "\n".join(out_lines).rstrip() + "\n"
    forced_record = dict(forced)
    forced_record.update({"datasets": dataset_path, "base_revision": base_revision})
    return RenderedRecipe(text=text, effective=parse_recipe_keys(text), forced=forced_record)


def render_pack_recipe(
    pack_dir: str | Path,
    out_path: str | Path,
    overrides: dict[str, Any] | None = None,
    **forced_kwargs: Any,
) -> RenderedRecipe:
    pack = Path(pack_dir)
    metadata = load_pack_metadata(pack)
    rendered = render_recipe(
        (pack / "recipe.yaml").read_text(encoding="utf-8"),
        overrides=overrides,
        allowed_keys=set(metadata.editable_keys),
        **forced_kwargs,
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered.text, encoding="utf-8")
    rendered.path = target
    return rendered


def write_training_dataset(samples: list[dict[str, Any]], path: str | Path) -> Path:
    """Accepted samples as Axolotl alpaca JSONL (prompt -> generation)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                canonical_json(
                    {
                        "instruction": sample.get("prompt", ""),
                        "input": "",
                        "output": sample.get("text") or "",
                    }
                )
                + "\n"
            )
    return target


@dataclass
class TrainingResult:
    checkpoint_path: Path
    checkpoint_hash: str
    recipe_path: Path | None
    fake: bool
    n_samples: int
    quality: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "checkpoint_hash": self.checkpoint_hash,
            "n_samples": self.n_samples,
            "quality": self.quality,
            "fake": bool(self.fake),
        }


def fake_quality_for(samples: list[dict[str, Any]]) -> float:
    """Deterministic fake capability: more accepted samples, a little better."""
    return min(0.95, 0.35 + 0.09 * math.log2(len(samples) + 1)) if samples else 0.0


class FakeTrainer:
    """Deterministic fake QLoRA run. Output is a labelled-fake adapter."""

    fake = True

    def train(
        self,
        samples: list[dict[str, Any]],
        checkpoint_dir: str | Path,
        pack_dir: str | Path | None = None,
        config: Any = None,
        overrides: dict[str, Any] | None = None,
        seed: int = 0,
        quality: float | None = None,
        work_dir: str | Path | None = None,
    ) -> TrainingResult:
        if not samples:
            raise TrainingError("fake training refused: no accepted samples")
        work = Path(work_dir) if work_dir is not None else Path(checkpoint_dir).parent / "training"
        dataset_path = write_training_dataset(samples, work / "dataset.jsonl")
        recipe_path: Path | None = None
        recipe_fingerprint: dict[str, str] = {}
        base_model = getattr(getattr(config, "models", None), "base_model", "")
        if pack_dir is not None:
            rendered = render_pack_recipe(
                pack_dir,
                work / "recipe.rendered.yaml",
                overrides=overrides,
                base_model=base_model,
                base_revision=getattr(getattr(config, "models", None), "base_revision", ""),
                dataset_path=str(dataset_path),
                output_dir=str(checkpoint_dir),
                seed=seed,
            )
            recipe_path = rendered.path
            recipe_fingerprint = rendered.content_fingerprint()
        effective_quality = float(quality) if quality is not None else fake_quality_for(samples)
        marker = hash_object(
            {
                "samples": [s.get("sample_id") for s in samples],
                "recipe": recipe_fingerprint,
                "seed": seed,
                "quality": effective_quality,
            }
        )
        target = write_fake_checkpoint(
            checkpoint_dir, marker=marker, base_model=base_model, quality=effective_quality
        )
        info = validate_checkpoint_dir(target)
        record = {
            "checkpoint_hash": info.checkpoint_hash,
            "n_samples": len(samples),
            "quality": effective_quality,
            "fake": True,
        }
        (work / "training_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        return TrainingResult(target, info.checkpoint_hash, recipe_path, True, len(samples), effective_quality)


class RealTrainer:
    """Axolotl QLoRA on a rented/operator GPU. NOT YET VALIDATED LIVE."""

    fake = False
    validated_live = False

    def __init__(self, resource_manager: Any = None, config: Any = None):
        self.resource_manager = resource_manager
        self.config = config

    @staticmethod
    def build_job(recipe_path: str | Path) -> JobSpec:
        return JobSpec(
            job_id="train-" + hash_object(str(recipe_path))[:12],
            kind="train",
            command=("axolotl", "train", str(recipe_path)),
        )

    def train(
        self,
        samples: list[dict[str, Any]],
        checkpoint_dir: str | Path,
        pack_dir: str | Path,
        request: RentalRequest,
        overrides: dict[str, Any] | None = None,
        seed: int = 0,
        work_dir: str | Path | None = None,
    ) -> TrainingResult:
        if not samples:
            raise TrainingError("real training refused: no accepted samples")
        if any(sample.get("fake") for sample in samples):
            raise TrainingError(
                "real training refused: fake-labelled samples may only flow into fake training"
            )
        if self.config is not None:
            assert_pinned_base_model(self.config)
        if self.resource_manager is None:
            if shutil.which("axolotl") is None:
                raise ProviderError(
                    "real training unavailable: the 'axolotl' executable was not found "
                    "and no provider ResourceManager is attached (not yet validated live)"
                )
            raise ProviderError(
                "real training is not yet validated live and has no "
                "ResourceManager/provider attached; refusing to train"
            )
        work = Path(work_dir) if work_dir is not None else Path(checkpoint_dir).parent / "training"
        dataset_path = write_training_dataset(samples, work / "dataset.jsonl")
        models = getattr(self.config, "models", None)
        rendered = render_pack_recipe(
            pack_dir,
            work / "recipe.rendered.yaml",
            overrides=overrides,
            base_model=getattr(models, "base_model", ""),
            base_revision=getattr(models, "base_revision", ""),
            dataset_path=str(dataset_path),
            output_dir=str(checkpoint_dir),
            seed=seed,
        )
        execution = self.resource_manager.run_and_destroy(
            request, self.build_job(rendered.path or work / "recipe.rendered.yaml")
        )
        if not execution.result.succeeded:
            raise TrainingError(f"Axolotl job failed: {execution.result.reason}")
        produced = Path(checkpoint_dir)
        if not produced.is_dir():
            raise ProviderError(
                "real training is not yet validated live: no checkpoint "
                "artefact was returned to the controller output directory"
            )
        info = validate_checkpoint_dir(produced)
        return TrainingResult(produced, info.checkpoint_hash, rendered.path, False, len(samples))
