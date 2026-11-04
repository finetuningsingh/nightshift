"""Process-wide secret registry and scrubber.

The API key must never reach the ledger, a log, a report, an exception message or the dashboard. Call ``register``
for every secret at startup; ``Ledger.append`` and the log writers call ``scrub``.
"""
from __future__ import annotations

import re
import threading
from typing import Any

_LOCK = threading.Lock()
_REGISTERED: set[str] = set()
_BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-~+/=]{8,}")
REDACTED = "[REDACTED]"


def register(value: str | None) -> None:
    """Register a secret. Values shorter than 8 characters are ignored (too likely to match ordinary text)."""
    if value and len(value) >= 8:
        with _LOCK:
            _REGISTERED.add(value)


def register_env(*names: str) -> None:
    import os
    for n in names:
        register(os.environ.get(n))


def clear() -> None:  # tests
    with _LOCK:
        _REGISTERED.clear()


def scrub_text(text: str) -> str:
    with _LOCK:
        secrets = sorted(_REGISTERED, key=len, reverse=True)
    for s in secrets:
        if s in text:
            text = text.replace(s, REDACTED)
    return _BEARER.sub(lambda m: f"{m.group(1)} {REDACTED}", text)


def scrub(obj: Any) -> Any:
    """Return a copy of ``obj`` (str / list / tuple / dict, recursively) with secrets replaced."""
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, dict):
        return {(scrub_text(k) if isinstance(k, str) else k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj
