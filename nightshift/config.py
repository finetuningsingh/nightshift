"""Configuration loading and validation (spec 3.1).

One TOML file describes a run. Budget numbers are never defaulted: a missing
budget field is a configuration error.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError

REQUIRED_BUDGET_FIELDS = (
    "aggregate_ceiling_usd",
    "budget_file",
    "per_trial_gpu_seconds",
    "max_rental_lifetime_seconds",
    "max_concurrent_rentals",
)

DEFAULT_BASE_MODEL = "meta-models/Muse-Glimmer-30B"
# Placeholder: an operator must replace this with an exact pinned revision
# before any real run. Real-mode code refuses unpinned revisions.
PLACEHOLDER_REVISION = "operator-must-pin-revision"
UNPINNED_REVISIONS = {"", "main", "master", "head", "latest", PLACEHOLDER_REVISION}


def revision_is_pinned(revision: str) -> bool:
    return str(revision).strip().lower() not in UNPINNED_REVISIONS


def assert_pinned_base_model(config: "NightshiftConfig") -> None:
    """Real generation/training must refuse an unpinned base revision."""
    if not config.models.base_model.strip():
        raise ConfigError("models.base_model is empty")
    if not revision_is_pinned(config.models.base_revision):
        raise ConfigError(
            "models.base_revision is unpinned "
            f"({config.models.base_revision!r}); set an exact revision for real runs"
        )


def _section(raw: dict[str, Any], name: str, required: bool = False) -> dict[str, Any]:
    value = raw.get(name)
    if value is None:
        if required:
            raise ConfigError(f"missing required config section [{name}]")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section [{name}] must be a table")
    return value


def _positive_number(section: dict[str, Any], key: str, where: str) -> float:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{where}.{key} must be a positive number, got {value!r}")
    return float(value)


def _positive_int(section: dict[str, Any], key: str, where: str) -> int:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where}.{key} must be a positive integer, got {value!r}")
    return int(value)


def _nonnegative_int(section: dict[str, Any], key: str, where: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{where}.{key} must be a non-negative integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class BudgetsConfig:
    aggregate_ceiling_usd: float
    budget_file: Path
    per_trial_gpu_seconds: float
    max_rental_lifetime_seconds: float
    max_concurrent_rentals: int


@dataclass(frozen=True)
class ModelsConfig:
    researcher: str = "muse-spark-1.3"
    reviser: str = "muse-wonder-full-1.0"
    base_model: str = DEFAULT_BASE_MODEL
    base_revision: str = PLACEHOLDER_REVISION
    fake_agents: bool = False


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "fake"
    fake: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OuterConfig:
    max_rounds: int = 2
    screen_trials: int = 1
    acceptance_margin: float = 0.0
    researchers_per_trial: int = 2


@dataclass(frozen=True)
class EvaluationConfig:
    max_retries: int = 1
    temperature: float = 0.0
    max_tokens: int = 128


@dataclass(frozen=True)
class ComparisonConfig:
    comparison_trials: int = 2
    margin: float = 0.0
    bootstrap_samples: int = 1000
    min_scored_pairs: int = 2


@dataclass(frozen=True)
class NightshiftConfig:
    path: Path | None
    run_name: str
    output_dir: Path
    seed: int
    budgets: BudgetsConfig
    models: ModelsConfig
    provider: ProviderConfig
    outer: OuterConfig
    evaluation: EvaluationConfig
    comparison: ComparisonConfig
    data_manifests: tuple[dict[str, Any], ...] = ()
    verifier_ids: tuple[str, ...] = ()
    pack_dir: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def fake(self) -> bool:
        return bool(self.provider.fake or self.models.fake_agents)

    @property
    def base_dir(self) -> Path:
        return self.path.parent if self.path is not None else Path.cwd()


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path)


def parse_config(raw: dict[str, Any], path: Path | None = None) -> NightshiftConfig:
    """Validate a parsed TOML mapping. Raises ConfigError on any problem."""
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a TOML table")
    base_dir = (path.parent if path is not None else Path.cwd()).resolve()

    run = _section(raw, "run")
    run_name = str(run.get("name", "nightshift-run"))
    output_dir = _resolve(base_dir, str(run.get("output_dir", "run")))
    seed = run.get("seed", 17)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"run.seed must be an integer, got {seed!r}")

    budgets_raw = _section(raw, "budgets", required=True)
    missing = [k for k in REQUIRED_BUDGET_FIELDS if k not in budgets_raw]
    if missing:
        raise ConfigError(
            "missing required budget field(s): "
            + ", ".join(f"budgets.{k}" for k in missing)
            + " (budget numbers have no defaults)"
        )
    if not isinstance(budgets_raw["budget_file"], str) or not budgets_raw["budget_file"]:
        raise ConfigError("budgets.budget_file must be a non-empty path string")
    budgets = BudgetsConfig(
        aggregate_ceiling_usd=_positive_number(budgets_raw, "aggregate_ceiling_usd", "budgets"),
        budget_file=_resolve(base_dir, budgets_raw["budget_file"]),
        per_trial_gpu_seconds=_positive_number(budgets_raw, "per_trial_gpu_seconds", "budgets"),
        max_rental_lifetime_seconds=_positive_number(
            budgets_raw, "max_rental_lifetime_seconds", "budgets"
        ),
        max_concurrent_rentals=_positive_int(budgets_raw, "max_concurrent_rentals", "budgets"),
    )

    models_raw = _section(raw, "models")
    models = ModelsConfig(
        researcher=str(models_raw.get("researcher", "muse-spark-1.3")),
        reviser=str(models_raw.get("reviser", "muse-wonder-full-1.0")),
        base_model=str(models_raw.get("base_model", DEFAULT_BASE_MODEL)),
        base_revision=str(models_raw.get("base_revision", PLACEHOLDER_REVISION)),
        fake_agents=bool(models_raw.get("fake_agents", False)),
    )

    provider_raw = _section(raw, "provider")
    provider_name = str(provider_raw.get("name", "fake"))
    provider = ProviderConfig(
        name=provider_name,
        fake=provider_name == "fake",
        options={k: v for k, v in provider_raw.items() if k != "name"},
    )

    outer_raw = _section(raw, "outer")
    outer = OuterConfig(
        max_rounds=_nonnegative_int(outer_raw, "max_rounds", "outer", 2),
        screen_trials=_nonnegative_int(outer_raw, "screen_trials", "outer", 1),
        acceptance_margin=float(outer_raw.get("acceptance_margin", 0.0)),
        researchers_per_trial=_nonnegative_int(outer_raw, "researchers_per_trial", "outer", 2)
        or 2,
    )
    if outer.screen_trials <= 0:
        raise ConfigError("outer.screen_trials must be positive")

    eval_raw = _section(raw, "evaluation")
    evaluation = EvaluationConfig(
        max_retries=_nonnegative_int(eval_raw, "max_retries", "evaluation", 1),
        temperature=float(eval_raw.get("temperature", 0.0)),
        max_tokens=_nonnegative_int(eval_raw, "max_tokens", "evaluation", 128) or 128,
    )

    comp_raw = _section(raw, "comparison")
    comparison = ComparisonConfig(
        comparison_trials=_nonnegative_int(comp_raw, "comparison_trials", "comparison", 2) or 2,
        margin=float(comp_raw.get("margin", comp_raw.get("acceptance_margin", 0.0))),
        bootstrap_samples=_nonnegative_int(comp_raw, "bootstrap_samples", "comparison", 1000)
        or 1000,
        min_scored_pairs=_nonnegative_int(comp_raw, "min_scored_pairs", "comparison", 2)
        or 2,
    )

    data_raw = _section(raw, "data")
    manifests = data_raw.get("manifests", [])
    if not isinstance(manifests, list) or not all(isinstance(m, dict) for m in manifests):
        raise ConfigError("data.manifests must be a list of tables ([[data.manifests]])")

    verifiers_raw = _section(raw, "verifiers")
    verifier_ids = verifiers_raw.get("training", verifiers_raw.get("ids", []))
    if not isinstance(verifier_ids, list) or not all(isinstance(v, str) for v in verifier_ids):
        raise ConfigError("verifiers.training must be a list of verifier id strings")

    pack_raw = raw.get("pack")
    pack_dir = None
    if isinstance(pack_raw, dict) and pack_raw.get("dir"):
        pack_dir = _resolve(base_dir, str(pack_raw["dir"]))
    elif isinstance(run.get("pack_dir"), str):
        pack_dir = _resolve(base_dir, run["pack_dir"])

    return NightshiftConfig(
        path=path,
        run_name=run_name,
        output_dir=output_dir,
        seed=int(seed),
        budgets=budgets,
        models=models,
        provider=provider,
        outer=outer,
        evaluation=evaluation,
        comparison=comparison,
        data_manifests=tuple(manifests),
        verifier_ids=tuple(verifier_ids),
        pack_dir=pack_dir,
        raw=raw,
    )


def load_config(path: str | Path) -> NightshiftConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    return parse_config(raw, path=config_path.resolve())


DEMO_CONFIG_TEMPLATE = """\
# Nightshift CPU demo config: fake provider, fake agents, bundled toy data.
# Every record from this run is labelled fake.

[run]
name = "nightshift-demo"
output_dir = "{output_dir}"
seed = 17

[models]
researcher = "muse-spark-1.3"
reviser = "muse-wonder-full-1.0"
base_model = "meta-models/Muse-Glimmer-30B"
base_revision = "operator-must-pin-revision"
fake_agents = true

[provider]
name = "fake"

[budgets]
aggregate_ceiling_usd = 10.0
budget_file = "{budget_file}"
per_trial_gpu_seconds = 3600
max_rental_lifetime_seconds = 1800
max_concurrent_rentals = 2

[outer]
max_rounds = 2
screen_trials = 1
acceptance_margin = 0.0
researchers_per_trial = 2

[evaluation]
max_retries = 1
temperature = 0.0
max_tokens = 128

[comparison]
comparison_trials = 2
margin = 0.0
bootstrap_samples = 1000

[verifiers]
training = ["max_words", "required_keyword", "valid_json"]

[pack]
dir = "{pack_dir}"
"""


def write_demo_config(
    path: str | Path,
    output_dir: str | Path = "run",
    budget_file: str = "run/budget.json",
    pack_dir: str = "pack",
) -> Path:
    """Write a complete CPU demo config selecting fakes and toy data."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        DEMO_CONFIG_TEMPLATE.format(
            output_dir=str(output_dir).replace("\\", "/"),
            budget_file=str(budget_file).replace("\\", "/"),
            pack_dir=str(pack_dir).replace("\\", "/"),
        ),
        encoding="utf-8",
    )
    load_config(target)
    return target
