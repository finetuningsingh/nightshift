"""Model prices (USD per 1M tokens, standard mode) from ``docs/META_API_NOTES.md``.

``usd_for`` charges cached prompt tokens at the cached rate and the rest of the prompt at the input rate. A model that
is not in ``PRICES`` is charged at the ``muse-spark-1.3`` rate, so an unknown (e.g. partner-hosted) model never looks free.
"""
from __future__ import annotations

import math

from ..types import Usage

DEFAULT_MODEL = "muse-spark-1.3"

_SPARK = {"cached_input": 0.15, "input": 1.25, "output": 4.25}
_CONTRIBUTOR = {"cached_input": 0.002, "input": 0.10, "output": 0.20}
_WONDER = {"cached_input": 0.20, "input": 4.00, "output": 20.00}

PRICES: dict[str, dict[str, float]] = {
    "muse-spark-1.3": dict(_SPARK),
    "muse-spark-1.2": dict(_SPARK),
    "muse-spark-1.1": dict(_SPARK),
    # /docs/pricing-rate-limits (checked 2026-10-26): the 1.2 and 1.3 contributor variants have the same price.
    "muse-spark-1.3-contributor": dict(_CONTRIBUTOR),
    "muse-spark-1.2-contributor": dict(_CONTRIBUTOR),
    "muse-wonder-1.0": dict(_WONDER),
}


_SNAPSHOT_SEPARATORS = "-@:._"


def price_for(model: str | None) -> dict[str, float]:
    """The price row used for ``model``: an exact match, else the longest known id it extends with a separator (a dated
    snapshot such as ``muse-wonder-1.0-2026-10-20`` is charged like ``muse-wonder-1.0``, never at the cheaper Spark rate),
    else the muse-spark-1.3 row."""
    name = (model or "").strip()
    if name in PRICES:
        return PRICES[name]
    known = [k for k in PRICES if name.startswith(k) and name[len(k):len(k) + 1] in _SNAPSHOT_SEPARATORS and len(name) > len(k) + 1]
    return PRICES[max(known, key=len)] if known else PRICES[DEFAULT_MODEL]


def _tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return max(0, int(value))


def usd_for(model: str, usage: Usage) -> float:
    """Dollar cost of one call. Cached tokens are clamped to the prompt size; negative or non-numeric counts count as 0."""
    price = price_for(model)
    prompt = _tokens(usage.prompt_tokens)
    cached = min(_tokens(usage.cached_tokens), prompt)
    fresh = prompt - cached
    completion = _tokens(usage.completion_tokens)
    return (fresh * price["input"] + cached * price["cached_input"] + completion * price["output"]) / 1_000_000
