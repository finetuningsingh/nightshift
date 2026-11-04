"""Prompt-safety helpers: untrusted text is wrapped as inert, delimited data before any model sees it.

Guarantees
----------
* ``fence`` is deterministic: the same ``(text, label, max_chars)`` always yields the same string.
* The delimiter embeds ``sha256(text)[:10]`` of the *whole* input. Content cannot contain a matching closing line
  because that line would have to be part of the text it is the hash of, so content such as a fake ``<<<END:...>>>``
  line stays inside the block.
* The label is reduced to ``[A-Za-z0-9_-]``; control characters never survive into the body; truncation is explicit.
* ``sanitize`` never returns more than ``max_chars`` characters.

Fencing marks text as data. It does not make a model obey that; the prompts that embed fenced text still say so.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

_LABEL_BAD = re.compile(r"[^A-Za-z0-9_-]")
MAX_LABEL = 48
_LINE_BREAKS = {" ", " ", "\x85"}


def _clean_label(label: object) -> str:
    cleaned = _LABEL_BAD.sub("_", str(label))[:MAX_LABEL]
    return cleaned or "data"


def _digest10(raw: str) -> str:
    """First 10 hex chars of sha256(raw). A lone surrogate (a model can emit one) must not crash the hash."""
    return hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()[:10]


def _cap(max_chars: object, default: int) -> int:
    """A usable non-negative character cap; ``None`` or junk means ``default``."""
    try:
        return max(int(max_chars), 0)  # type: ignore[call-overload]
    except (TypeError, ValueError, OverflowError):
        return default


def _strip_controls(text: str) -> str:
    """Drop control, format (zero-width, bidi), surrogate and private-use characters; keep ``\\n`` and ``\\t``.

    ``\\r\\n`` and a lone ``\\r`` become ``\\n`` so that a carriage return cannot hide a delimiter line.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            out.append(ch)
        elif ch in _LINE_BREAKS:
            out.append("\n")
        elif unicodedata.category(ch)[0] == "C":
            continue
        else:
            out.append(ch)
    return "".join(out)


def sanitize(text: str, max_chars: int = 300) -> str:
    """Strip control characters (keeping newline and tab), collapse runs of blank lines, cap the length.

    The result is stripped, contains at most one blank line in a row, and is never longer than ``max_chars``
    (an over-long text ends with ``...``). ``None`` is treated as the empty string.
    """
    if text is None:
        return ""
    cleaned = _strip_controls(str(text))
    lines: list[str] = []
    blank = False
    for line in cleaned.split("\n"):
        line = line.rstrip()
        if not line:
            if blank:
                continue
            blank = True
        else:
            blank = False
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    max_chars = _cap(max_chars, 300)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars - 3].rstrip() + "..." if max_chars > 3 else cleaned[:max_chars]
    return cleaned


def fence(text: str, label: str, *, max_chars: int = 4000) -> str:
    """Wrap untrusted ``text`` as a delimited data block.

    Layout::

        <<<DATA:{label}:{sha256(text)[:10]}>>>
        body (control characters removed, at most max_chars characters)
        [truncated N chars]            <- only when the body was cut
        <<<END:{label}:{sha256(text)[:10]}>>>
    """
    raw = "" if text is None else str(text)
    tag = f"{_clean_label(label)}:{_digest10(raw)}"
    body = _strip_controls(raw)
    cut = len(body) - _cap(max_chars, 4000)
    if cut > 0:
        body = body[:len(body) - cut]
    lines = [f"<<<DATA:{tag}>>>", body]
    if cut > 0:
        lines.append(f"[truncated {cut} chars]")
    lines.append(f"<<<END:{tag}>>>")
    return "\n".join(lines)
