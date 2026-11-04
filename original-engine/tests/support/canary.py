"""Sealed-split canary: expand a number the agents must never see into every text form it could leak as, and look for those forms.

The question a canary answers is "did anything derived from the sealed (replication) grade reach a prompt, a lesson, a memory file
or a tool result?". A leak is rarely a verbatim ``repr``: a model that was shown ``1.5625820215618353`` may write ``delta=1.5626``,
"an improvement of 1.56", ``+156.3%`` or ``1.56e+00``. So one number is expanded into

* fixed-point renderings at 2 to 6 decimals, rounded half-up, rounded as Python's ``format`` does, and truncated, plus
  renderings at 3 to 6 SIGNIFICANT digits (the only way a value under 1e-3 shows up at a readable length), with and without the
  leading zero (``0.5`` and ``.5``);
* scientific notation (``1.56e+00``, ``1.56e0``, ``1.56E+00``) and the mantissa prefixes of a tiny value's full ``repr``;
* the full ``repr`` itself and its truncation to 5 or more decimals (a full-precision leak contains it as a prefix);
* percentage (x100, suffix ``%`` or ``%`` after a space), per-mille (x1000, ``‰``) and plain scaled renderings (``156.26``,
  ``1562.6``);
* for named integers (``n_eval``) the labelled forms ``n_eval=210``, ``n_eval: 210``, ``"n_eval": 210``.

THE THRESHOLD (``MIN_SIGNIFICANT_DIGITS``): a needle is kept only if its mantissa has at least 3 significant digits, counted after
stripping leading AND trailing zeros, and is not a bare integer. ``1.5``, ``0.0``, ``2.00`` and ``0.05`` therefore never become
needles (they are in half the numbers any prompt contains), ``1.56`` and ``0.945`` do. A bare integer (``156``) is never a needle
either, because sequence numbers and counts would match; a scaled integer needs its unit (``156%``) and a named integer its label.

MATCHING (``find_leaks``) is boundary-aware so that a short needle cannot fire inside a longer, different number: no digit may
precede a needle, and unless it is a long pure-digit prefix (``SHORT_NEEDLE_DIGITS`` or more digits written, which exists to
be found inside a full-precision ``repr``; trailing zeros count towards that length) no digit may follow it. ``1.56`` is found in ``improvement of 1.56 on`` but not in
``dev score 1.5601``.

NO FALSE ALARMS: ``canary_needles(sealed, dev)`` drops every needle string that a legitimate (dev-split or other allowed) number
of the same run also produces, so a rounding that both splits share is not held against the run. What is left is what only the
sealed numbers can explain.
"""
from __future__ import annotations

import math
import re
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence, Union

from nightshift.types import Kind

MIN_SIGNIFICANT_DIGITS = 3        # a numeric needle needs this many significant digits (leading and trailing zeros do not count)
SHORT_NEEDLE_DIGITS = 6           # fewer digits written than this (leading zeros excluded): a digit may not follow the needle in the text
FIXED_PLACES = (2, 3, 4, 5, 6)    # decimals of the fixed-point renderings of the value itself
SCALED_PLACES = (0, 1, 2, 3, 4)   # decimals of the percentage and per-mille renderings
SIGNIFICANT_ROUNDINGS = (3, 4, 5, 6)
SCI_MANTISSA_PLACES = (2, 3, 4, 5, 6)     # 6 is the default of ``"%e"`` and ``format(x, "e")``
PERCENT_SUFFIXES = ("%", " %")
PER_MILLE_SUFFIXES = ("‰", " ‰", " per mille", " permille")

Value = Union[float, int]
Named = Union[Value, tuple]       # a number, or ("name", number) for a quantity that is also searched under its label


# --------------------------------------------------------------------------------------------------
# one number -> its text forms
# --------------------------------------------------------------------------------------------------
def significant_digits(needle: str) -> int:
    """Significant digits of the leading numeric mantissa of ``needle`` (zeros at either end do not count; 0 if it has none)."""
    found = re.match(r"\.?\d[\d.]*", needle)
    return len(re.sub(r"\D", "", found.group()).strip("0")) if found else 0


def acceptable(needle: str) -> bool:
    """True when ``needle`` clears the threshold: at least ``MIN_SIGNIFICANT_DIGITS`` significant digits, and not a bare integer
    (a number with neither a decimal point, an exponent nor a unit would match counts and sequence numbers)."""
    if significant_digits(needle) < MIN_SIGNIFICANT_DIGITS:
        return False
    return not re.fullmatch(r"\d+", needle)


def _quantized(d: Decimal, places: int, rounding: str) -> Decimal:
    return d.quantize(Decimal(1).scaleb(-places), rounding=rounding)


def _fixed(d: Decimal, places: int, mode: str) -> str:
    return format(_quantized(d, places, mode), "f")


def _sci_forms(x: float, d: Decimal) -> set[str]:
    out: set[str] = set()
    digits = d.as_tuple().digits
    exponent = d.adjusted()
    for k in SCI_MANTISSA_PLACES:
        rounded = f"{x:.{k}e}"                                    # 1.56e+00
        mantissa, _, exp = rounded.partition("e")
        n = int(exp)
        truncated = f"{digits[0]}.{''.join(map(str, digits[1:k + 1])).ljust(k, '0')}"
        for m, e in ((mantissa, n), (truncated, exponent)):
            for upper in ("e", "E"):
                out |= {f"{m}{upper}{e:+03d}", f"{m}{upper}{e}", f"{m}{upper}{e:+d}"}
    return out


def _prefix_forms(d: Decimal) -> set[str]:
    """``1.56258`` and ``1.562582`` for a tiny value: its ``repr`` is ``1.5625820215618353e-05`` and contains them."""
    digits = ''.join(map(str, d.as_tuple().digits))
    return {f"{digits[0]}.{digits[1:k]}" for k in (6, 7) if len(digits) >= k}


def number_forms(value: float, name: str | None = None) -> set[str]:
    """Every text a leak of the float ``value`` could appear as that clears the threshold (see the module docstring).

    ``name`` additionally yields the labelled forms ``name=v``, ``name = v``, ``name: v`` and ``"name": v`` (the value as ``repr``
    and, for a float, rounded to 2 to 4 decimals). The sign is dropped: ``-1.5626`` contains ``1.5626``.
    """
    x = abs(float(value))
    if not math.isfinite(x) or x == 0.0:
        return set()
    out: set[str] = set()
    try:
        with localcontext() as ctx:
            ctx.prec = 120
            d = Decimal(repr(x))
            out |= _forms_with_context(x, d)
    except (InvalidOperation, ValueError):
        pass
    forms = {f for f in out if acceptable(f)}
    if name:
        forms |= labelled_forms(name, value)
    return forms


def _forms_with_context(x: float, d: Decimal) -> set[str]:
    out: set[str] = {repr(x)}
    # the value itself: fixed decimals and significant-digit roundings, rounded and truncated
    sig_places = {max(0, s - 1 - d.adjusted()) for s in SIGNIFICANT_ROUNDINGS}
    for places in set(FIXED_PLACES) | sig_places:
        if places == 0:
            continue
        out |= {_fixed(d, places, ROUND_HALF_UP), _fixed(d, places, ROUND_DOWN), f"{x:.{places}f}"}
    out |= _sci_forms(x, d)
    if x < 1e-3:
        out |= _prefix_forms(d)
    # scaled renderings: percentages (x100) and per-mille (x1000); a whole-number rendering needs its unit
    for scale, suffixes in ((100, PERCENT_SUFFIXES), (1000, PER_MILLE_SUFFIXES), (10000, ())):
        scaled = d * scale
        scaled_float = float(scaled)
        scaled_sig = {max(0, s - 1 - scaled.adjusted()) for s in SIGNIFICANT_ROUNDINGS}
        for places in set(SCALED_PLACES) | scaled_sig:
            texts = {_fixed(scaled, places, ROUND_HALF_UP), _fixed(scaled, places, ROUND_DOWN), f"{scaled_float:.{places}f}"}
            if places == 0:
                out |= {t + s for t in texts for s in suffixes}
            else:
                out |= texts
    # `0.5` is also written `.5`
    out |= {f[1:] for f in out if f.startswith("0.")}
    return out


def labelled_forms(name: str, value: Value) -> set[str]:
    """``name=210``, ``name = 210``, ``name: 210``, ``"name": 210`` for an integer; the same for a float at its ``repr`` and rounded
    to 2 to 4 decimals. Labelled needles carry their own context, so they are exempt from the digit threshold."""
    if isinstance(value, bool):
        return set()
    shown = {str(value)} if isinstance(value, int) else {repr(float(value))} | {f"{abs(float(value)):.{p}f}" for p in (2, 3, 4)}
    out: set[str] = set()
    for v in shown:
        out |= {f"{name}={v}", f"{name} = {v}", f"{name}: {v}", f'"{name}": {v}', f"{name} {v}"}
    return out


# --------------------------------------------------------------------------------------------------
# a run's numbers -> needles
# --------------------------------------------------------------------------------------------------
def _pair(item: Named) -> tuple[str | None, Value]:
    if isinstance(item, tuple):
        return item[0], item[1]
    return None, item


def forms_of_all(items: Iterable[Named]) -> set[str]:
    """The union of ``number_forms`` over numbers or ``(name, number)`` pairs; booleans and non-finite values are skipped."""
    out: set[str] = set()
    for item in items:
        name, value = _pair(item)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, int):
            out |= labelled_forms(name, value) if name else set()
        else:
            out |= number_forms(value, name)
    return out


def canary_needles(sealed: Iterable[Named], allowed: Iterable[Named] = ()) -> list[str]:
    """The needles that only a sealed number explains: the forms of ``sealed`` minus the forms of ``allowed`` (the same run's dev
    numbers and any other legitimate quantity), sorted. Each item is a number or a ``(name, number)`` pair."""
    return sorted(forms_of_all(sealed) - forms_of_all(allowed))


@lru_cache(maxsize=None)
def _is_long_prefix(needle: str) -> bool:
    """A pure-digit needle written with at least ``SHORT_NEEDLE_DIGITS`` digits (leading zeros do not count, trailing zeros DO: the
    six decimals ``0.286500`` of ``0.2865001893`` are a prefix of every longer rendering of it, and only its 4 significant digits
    must not make it a short needle): found as a prefix of a longer number."""
    if re.fullmatch(r"\.?\d[\d.]*", needle) is None:
        return False
    return len(re.sub(r"\D", "", needle).lstrip("0")) >= SHORT_NEEDLE_DIGITS


def occurs_in(needle: str, text: str) -> bool:
    """Boundary-aware containment (see the module docstring): False for ``1.56`` inside ``1.5601`` or ``21.56``, and for
    ``n_eval=21`` inside ``n_eval=210``. No digit may precede a numeric needle (no word character a labelled one), and no digit
    may follow a needle that ends in a digit unless it is a long pure-digit prefix."""
    first, last = needle[:1], needle[-1:]
    trailing = last.isdigit() and not _is_long_prefix(needle)
    at = text.find(needle)
    while at != -1:
        before = text[at - 1] if at else ""
        end = at + len(needle)
        after = text[end] if end < len(text) else ""
        if first.isdigit() or first == ".":
            clean_start = not before.isdigit()
        elif first.isalnum() or first == "_":
            clean_start = not (before.isalnum() or before == "_")
        else:
            clean_start = True
        if clean_start and not (trailing and after.isdigit()):
            return True
        at = text.find(needle, at + 1)
    return False


_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_NUMBER_TOKEN = re.compile(r"\d[\d.]*(?:[eE][+-]?\d+)?")
_UNIT_AFTER = re.compile(r"\s?(?:ms|s|sec|secs|seconds|min|usd|USD)\b")
_KEY_BEFORE = re.compile(r"(\w+)[\\\"']*\s*[:=]?\s*[+-]?$")

# Budget and clock counters are not split data. They are different in every run, and a counter that happens to render like a coarse
# form of a sealed number (spend 0.0763 against s_eff 0.07633...) would be a false alarm, and a flaky one. The number that follows
# one of these keys, in JSON or in prose, is not searched; nor is a number written as an amount of time or money (`1.09s in total`,
# `$0.0763`).
ACCOUNTING_KEYS = frozenset({"usd", "llm_usd", "cost_usd", "cost", "amount", "spent", "remaining", "wall_sec", "elapsed", "elapsed_sec",
                             "duration_sec", "cpu_sec", "idle_sec", "seconds", "latency_ms", "timeout_sec"})


def _mask_accounting(joined: str, ignore_keys: frozenset[str]) -> str:
    """``joined`` with the digits of every accounting number replaced by ``#`` (same length), so that no needle can match them and no
    neighbouring window can carry them along. An accounting number follows one of ``ignore_keys`` (``"usd": 0.0763``) or carries a
    unit of time or money (``1.09s``, ``$0.0763``)."""
    parts: list[str] = []
    last = 0
    for m in _NUMBER_TOKEN.finditer(joined):
        key = _KEY_BEFORE.search(joined, max(0, m.start() - 40), m.start())
        if ((key is not None and key.group(1) in ignore_keys) or _UNIT_AFTER.match(joined, m.end())
                or joined[max(0, m.start() - 1):m.start()] == "$"):
            parts += [joined[last:m.start()], "#" * (m.end() - m.start())]
            last = m.end()
    parts.append(joined[last:])
    return "".join(parts)


def _numeric_windows(joined: str) -> str:
    """The distinct stretches of ``joined`` around every numeric token (two characters before it, twelve after: room for a
    boundary check and a unit such as `` per mille``). A numeric needle can only occur inside one, and the unique windows of a
    long conversation (every request replays the history) are a small fraction of its size."""
    windows = {joined[max(0, m.start() - 2):m.end() + 12] for m in _NUMBER_TOKEN.finditer(joined)}
    return "\x00".join(sorted(windows))


def find_leaks(texts: Iterable[str], needles: Iterable[str], ignore_keys: Iterable[str] = ACCOUNTING_KEYS) -> list[str]:
    """The needles that occur (boundary-aware) in at least one of ``texts``, in the order given. Numbers that follow one of
    ``ignore_keys`` (budget and clock counters, ``ACCOUNTING_KEYS`` by default) are not searched; pass ``()`` to search them."""
    joined = "\x00".join(texts)
    needles = list(needles)
    if "\\u" in joined and not all(n.isascii() for n in needles):
        joined = _UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), joined)      # a JSON body writes ‰ as \u2030
    ignore = frozenset(ignore_keys)
    numeric = _numeric_windows(_mask_accounting(joined, ignore) if ignore else joined)
    found = []
    for n in needles:
        haystack = numeric if n[:1].isdigit() or n[:1] == "." else joined
        if occurs_in(n, haystack):
            found.append(n)
    return found


# --------------------------------------------------------------------------------------------------
# ledger events -> numbers
# --------------------------------------------------------------------------------------------------
_REPLICATION_PURPOSES = ("replication", "replication_baseline")


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _grade_numbers(grade: Mapping[str, Any]) -> list[Named]:
    out: list[Named] = [grade.get("score"), grade.get("std")]
    out += list(grade.get("per_seed") or []) + list((grade.get("metrics") or {}).values())
    for row in grade.get("per_seed_metrics") or []:
        out += list(row.values())
    if _numeric(grade.get("n_eval")):
        out.append(("n_eval", grade["n_eval"]))
    return [v for v in out if _numeric(v) or isinstance(v, tuple)]


def _walk_sealed(node: Any, found: list[Named], key: str = "", force: bool = False) -> None:
    """Every number under a ``sealed_*`` key anywhere in an event's data."""
    if isinstance(node, Mapping):
        for k, v in node.items():
            _walk_sealed(v, found, str(k), force or str(k).startswith("sealed_"))
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk_sealed(v, found, key, force)
    elif force and _numeric(node):
        found.append(node)


def sealed_values(events: Iterable[Any]) -> list[Named]:
    """Every number of a run's ledger that derives from a sealed-split grade: the grade (score, std, per-seed values, metrics,
    ``n_eval``), the sealed-derived statistics of a replication record (``sealed_delta``, ``sealed_margin``, ``s_eff``, ``t``),
    every ``sealed_*`` field of any event, and the promotion's sealed numbers. Named quantities are ``(name, number)`` pairs."""
    out: list[Named] = []
    for e in events:
        d = e.data
        if e.kind == Kind.EXP_GRADED and (d.get("split") == "sealed" or d.get("purpose") in _REPLICATION_PURPOSES):
            out += _grade_numbers(d.get("grade") or {})
        elif e.kind == Kind.EXP_RECORDED and d.get("phase") == "replication":
            out += [d.get(k) for k in ("sealed_delta", "sealed_margin") if _numeric(d.get(k))]
            out += [(k, d[k]) for k in ("s_eff", "t") if _numeric(d.get(k))]
        else:
            _walk_sealed(d, out)
    return [v for v in out if (isinstance(v, tuple) and _numeric(v[1])) or _numeric(v)]


def dev_values(events: Iterable[Any]) -> list[Named]:
    """The numbers that are legitimately visible and so are subtracted from the sealed needles: dev-split grades, the numbers
    of a lane-phase experiment record (``delta``, margins, ``s_eff``, ``t``, ``sigma_prior``), the dev ``score`` of a promotion,
    and the non-sealed statistics (``sigma_prior``, ``noise_margin``) of a replication record."""
    out: list[Named] = []
    for e in events:
        d = e.data
        if e.kind == Kind.EXP_GRADED and d.get("split") == "dev" and d.get("purpose") not in _REPLICATION_PURPOSES:
            out += _grade_numbers(d.get("grade") or {})
        elif e.kind == Kind.EXP_RECORDED:
            keys = ("delta", "noise_margin", "s_eff", "t", "sigma_prior", "train_delta", "train_margin") if d.get("phase") != "replication" \
                else ("sigma_prior", "noise_margin")
            for k in keys:
                if _numeric(d.get(k)):
                    out.append((k, d[k]))
        elif e.kind == Kind.INCUMBENT_PROMOTED and _numeric(d.get("score")):
            out.append(d["score"])
    return [v for v in out if (isinstance(v, tuple) and _numeric(v[1])) or _numeric(v)]


def sealed_floats(events: Iterable[Any]) -> list[float]:
    """The plain float values of ``sealed_values`` (integers and names dropped), for tests that plant a leak by value."""
    return [float(_pair(v)[1]) for v in sealed_values(events) if isinstance(_pair(v)[1], float)]


def run_needles(events: Sequence[Any]) -> list[str]:
    """``canary_needles`` of one run's ledger events: sealed-derived forms minus the forms of the run's dev numbers."""
    events = list(events)
    return canary_needles(sealed_values(events), dev_values(events))
