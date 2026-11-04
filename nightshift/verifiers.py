"""Pinned, versioned, training-only verifiers (spec 3.5).

Nightshift's own small pure-Python checks. They filter generated training
samples only; evaluation scorers are configured separately. Every filtered
sample records the verifier ids and versions applied to it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .errors import DataError

VERIFIER_SET_VERSION = "nightshift-training-verifiers-1.0.0"


@dataclass(frozen=True)
class VerifierResult:
    verifier_id: str
    version: str
    passed: bool
    detail: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "verifier_id": self.verifier_id,
            "version": self.version,
            "passed": bool(self.passed),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Verifier:
    id: str
    version: str
    description: str
    check: Callable[[str, dict[str, Any]], tuple[bool, str]]

    def __call__(self, text: str, params: dict[str, Any] | None = None) -> VerifierResult:
        passed, detail = self.check(str(text), dict(params or {}))
        return VerifierResult(self.id, self.version, bool(passed), detail)


def _param(params: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in params:
            return params[key]
    return default


def _check_max_words(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    limit = int(_param(params, "max_words", "limit", "value", default=40))
    count = len(text.split())
    return count <= limit, f"{count} words (max {limit})"


def _check_min_words(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    limit = int(_param(params, "min_words", "limit", "value", default=1))
    count = len(text.split())
    return count >= limit, f"{count} words (min {limit})"


def _keywords(params: dict[str, Any]) -> list[str]:
    raw = _param(params, "keywords", "keyword", "value", default=[])
    if isinstance(raw, str):
        return [raw]
    return [str(k) for k in raw]


def _check_required_keyword(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    words = _keywords(params)
    lowered = text.lower()
    missing = [w for w in words if w.lower() not in lowered]
    return not missing, ("all keywords present" if not missing else f"missing {missing}")


def _check_forbidden_keyword(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    words = _keywords(params)
    lowered = text.lower()
    present = [w for w in words if w.lower() in lowered]
    return not present, ("no forbidden keywords" if not present else f"found {present}")


def _check_forbidden_characters(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    raw = _param(params, "characters", "chars", "value", default="")
    chars = list(raw) if isinstance(raw, str) else [str(c) for c in raw]
    present = sorted({c for c in chars if c and c in text})
    return not present, ("no forbidden characters" if not present else f"found {present}")


def _check_valid_json(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return False, "not valid JSON"
    required_keys = _param(params, "required_keys", "keys", default=[])
    if required_keys:
        if not isinstance(parsed, dict):
            return False, "JSON value is not an object"
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            return False, f"missing keys {missing}"
    return True, "valid JSON"


def _check_lowercase(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    passed = bool(letters) and all(c.islower() for c in letters)
    return passed, "all lowercase" if passed else "contains uppercase or no letters"


def _check_uppercase(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    passed = bool(letters) and all(c.isupper() for c in letters)
    return passed, "all uppercase" if passed else "contains lowercase or no letters"


def _check_no_whitespace(text: str, params: dict[str, Any]) -> tuple[bool, str]:
    passed = not re.search(r"\s", text or "")
    return bool(passed and text), "single token" if passed else "contains whitespace"


VERIFIERS: dict[str, Verifier] = {
    v.id: v
    for v in (
        Verifier("max_words", "1.0.0", "At most N words.", _check_max_words),
        Verifier("min_words", "1.0.0", "At least N words.", _check_min_words),
        Verifier("required_keyword", "1.0.0", "Contains every required keyword.", _check_required_keyword),
        Verifier("forbidden_keyword", "1.0.0", "Contains no forbidden keyword.", _check_forbidden_keyword),
        Verifier("forbidden_characters", "1.0.0", "Contains none of the forbidden characters.", _check_forbidden_characters),
        Verifier("valid_json", "1.0.0", "Parses as JSON, optionally with required keys.", _check_valid_json),
        Verifier("lowercase", "1.0.0", "All letters lowercase.", _check_lowercase),
        Verifier("uppercase", "1.0.0", "All letters uppercase.", _check_uppercase),
        Verifier("no_whitespace", "1.0.0", "One whitespace-free token.", _check_no_whitespace),
    )
}


def get_verifier(verifier_id: str) -> Verifier:
    try:
        return VERIFIERS[verifier_id]
    except KeyError as exc:
        raise DataError(
            f"unknown training verifier {verifier_id!r}; known: {sorted(VERIFIERS)}"
        ) from exc


def verifier_catalogue() -> list[dict[str, str]]:
    return [
        {"id": v.id, "version": v.version, "description": v.description}
        for v in sorted(VERIFIERS.values(), key=lambda item: item.id)
    ]


def normalise_check(check: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Accept {"id": ..., "params": {...}} and flat spellings like
    {"id": "max_words", "max_words": 12}."""
    if not isinstance(check, dict):
        raise DataError(f"verifier check must be an object, got {check!r}")
    verifier_id = check.get("id") or check.get("verifier_id") or check.get("verifier")
    if not isinstance(verifier_id, str) or not verifier_id:
        raise DataError(f"verifier check needs an id: {check!r}")
    params = dict(check.get("params") or {})
    for key, value in check.items():
        if key not in ("id", "verifier_id", "verifier", "params"):
            params.setdefault(key, value)
    return verifier_id, params


def check_text(text: str, checks: list[dict[str, Any]]) -> list[VerifierResult]:
    results: list[VerifierResult] = []
    for check in checks or []:
        verifier_id, params = normalise_check(check)
        results.append(get_verifier(verifier_id)(text, params))
    return results


def checks_pass(text: str, checks: list[dict[str, Any]]) -> bool:
    return all(result.passed for result in check_text(text, checks))


def score_checks(text: str, checks: list[dict[str, Any]]) -> float:
    """Fraction of checks passed (1.0 when there are no checks)."""
    results = check_text(text, checks)
    if not results:
        return 1.0
    return sum(1 for r in results if r.passed) / len(results)


def filter_samples(
    samples: list[dict[str, Any]],
    checks_by_prompt_id: dict[str, list[dict[str, Any]]],
    verifier_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition generated samples by the training verifiers.

    Each output sample carries `verifier_results` with ids and versions.
    """
    wanted = set(verifier_ids) if verifier_ids else None
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for sample in samples:
        text = str(sample.get("text") or sample.get("completion") or "")
        checks = list(
            sample.get("verifiers")
            or checks_by_prompt_id.get(str(sample.get("prompt_id", "")), [])
        )
        if wanted is not None:
            checks = [c for c in checks if normalise_check(c)[0] in wanted]
        results = check_text(text, checks)
        record = dict(sample)
        record["verifier_results"] = [r.to_record() for r in results]
        record["verifier_set"] = VERIFIER_SET_VERSION
        (accepted if all(r.passed for r in results) else rejected).append(record)
    return accepted, rejected
