"""Spec 3.1: configuration loading, missing budget fields are errors."""

import pytest

from nightshift.config import load_config, parse_config, write_demo_config
from nightshift.errors import ConfigError


def full_raw():
    return {
        "run": {"name": "r", "output_dir": "run", "seed": 3},
        "budgets": {
            "aggregate_ceiling_usd": 25.0,
            "budget_file": "run/budget.json",
            "per_trial_gpu_seconds": 1000,
            "max_rental_lifetime_seconds": 600,
            "max_concurrent_rentals": 2,
        },
    }


@pytest.mark.parametrize(
    "field",
    [
        "aggregate_ceiling_usd",
        "budget_file",
        "per_trial_gpu_seconds",
        "max_rental_lifetime_seconds",
        "max_concurrent_rentals",
    ],
)
def test_missing_budget_field_is_a_config_error(field):
    raw = full_raw()
    del raw["budgets"][field]
    with pytest.raises(ConfigError, match=field):
        parse_config(raw)


def test_missing_budgets_section_is_a_config_error():
    with pytest.raises(ConfigError, match="budgets"):
        parse_config({"run": {"name": "r"}})


def test_nonpositive_budget_is_a_config_error():
    raw = full_raw()
    raw["budgets"]["aggregate_ceiling_usd"] = 0
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_demo_config_is_complete_fake_and_loadable(tmp_path):
    path = write_demo_config(tmp_path / "demo.toml")
    config = load_config(path)
    assert config.provider.name == "fake"
    assert config.provider.fake is True
    assert config.models.fake_agents is True
    assert config.models.researcher == "muse-spark-1.3"
    assert config.models.reviser == "muse-wonder-full-1.0"
    assert config.budgets.aggregate_ceiling_usd > 0
    assert config.fake is True
