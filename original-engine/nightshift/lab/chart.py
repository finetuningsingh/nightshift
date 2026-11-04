"""The adaptive search trace as a standard-library SVG (no plotting dependency).

``search_trace_svg(points, title)`` draws one marker per candidate in the order they were judged: accepted (filled diamond), rejected
(hollow circle), invalid (cross), pending (small dot). When the points carry scores it also draws the incumbent as a step line; with
no scores (the bit-only view used during the search) it draws the outcome markers on one row and says so. The chart is a trace of an
ADAPTIVE search: it is not evidence of a gain, and the report titles it "adaptive search trace".

Marker shape carries the outcome as well as colour, and the colours follow the page theme (light or dark) with CSS custom properties
declared on the ``svg`` element itself, so several charts, or a chart inside a page with its own styles, do not interfere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence
from xml.sax.saxutils import escape

__all__ = ["TracePoint", "search_trace_svg", "OUTCOMES"]

OUTCOMES = ("accepted", "rejected", "invalid", "pending")


@dataclass(frozen=True)
class TracePoint:
    """One candidate in judgement order. ``value`` is the candidate's score and ``incumbent`` the incumbent's score at that time; both
    are None in the bit-only view (and ``value`` is None for an invalid candidate, which has no comparable numbers)."""

    index: int
    label: str
    outcome: str
    value: Optional[float] = None
    incumbent: Optional[float] = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"TracePoint.outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        for name in ("value", "incumbent"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
                raise ValueError(f"TracePoint.{name} must be a finite number or None")


_STYLE = """
svg.st-chart{--st-ink:#1f2328;--st-muted:#656d76;--st-grid:#d8dee4;--st-bg:#ffffff;--st-acc:#1a7f37;--st-rej:#9a6700;--st-inv:#cf222e;--st-inc:#0969da;--st-pend:#8c959f}
@media (prefers-color-scheme: dark){svg.st-chart{--st-ink:#e6edf3;--st-muted:#9198a1;--st-grid:#30363d;--st-bg:#0d1117;--st-acc:#3fb950;--st-rej:#d29922;--st-inv:#ff7b72;--st-inc:#58a6ff;--st-pend:#8b949e}}
svg.st-chart text{font:12px system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:var(--st-ink)}
svg.st-chart .st-title{font-size:14px;font-weight:600}
svg.st-chart .st-muted{fill:var(--st-muted)}
svg.st-chart .st-grid{stroke:var(--st-grid);stroke-width:1}
svg.st-chart .st-inc{fill:none;stroke:var(--st-inc);stroke-width:2}
svg.st-chart .st-acc{fill:var(--st-acc);stroke:var(--st-acc);stroke-width:1.5}
svg.st-chart .st-rej{fill:none;stroke:var(--st-rej);stroke-width:2}
svg.st-chart .st-inv{fill:none;stroke:var(--st-inv);stroke-width:2.2}
svg.st-chart .st-pend{fill:var(--st-pend);stroke:none}
"""


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _marker(kind: str, x: float, y: float, tip: str) -> str:
    t = f"<title>{escape(tip)}</title>"
    if kind == "accepted":
        d = 7
        return f'<polygon class="st-acc" points="{x:.1f},{y - d:.1f} {x + d:.1f},{y:.1f} {x:.1f},{y + d:.1f} {x - d:.1f},{y:.1f}">{t}</polygon>'
    if kind == "rejected":
        return f'<circle class="st-rej" cx="{x:.1f}" cy="{y:.1f}" r="6">{t}</circle>'
    if kind == "invalid":
        d = 5.5
        return (
            f'<g class="st-inv">{t}<line x1="{x - d:.1f}" y1="{y - d:.1f}" x2="{x + d:.1f}" y2="{y + d:.1f}"/>'
            f'<line x1="{x - d:.1f}" y1="{y + d:.1f}" x2="{x + d:.1f}" y2="{y - d:.1f}"/></g>'
        )
    return f'<circle class="st-pend" cx="{x:.1f}" cy="{y:.1f}" r="3.5">{t}</circle>'


def search_trace_svg(points: Sequence[TracePoint], title: str, *, y_label: str = "selection score", width: int = 720, height: int = 360) -> str:
    """Render ``points`` (judgement order) as an SVG string. ``title`` is drawn in the chart and used as its accessible name."""
    pts = list(points)
    left, right, top, bottom = 64, 24, 40, 74
    pw, ph = width - left - right, height - top - bottom
    scored = [p for p in pts if p.value is not None or p.incumbent is not None]
    has_values = bool(scored)

    counts = {o: sum(1 for p in pts if p.outcome == o) for o in OUTCOMES}
    desc = (
        f"{len(pts)} candidates: {counts['accepted']} accepted, {counts['rejected']} rejected, {counts['invalid']} invalid"
        + (f", {counts['pending']} pending" if counts["pending"] else "")
        + ". "
        + ("The line is the incumbent's score; markers are candidates." if has_values else "Scores are hidden in this view; markers show outcomes only.")
    )

    out: list[str] = []
    out.append(
        f'<svg class="st-chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px;height:auto" role="img" aria-labelledby="st-t st-d">'
    )
    out.append(f"<style>{_STYLE}</style>")
    out.append(f'<title id="st-t">{escape(title)}</title><desc id="st-d">{escape(desc)}</desc>')
    out.append(f'<rect width="{width}" height="{height}" fill="var(--st-bg)"/>')
    out.append(f'<text class="st-title" x="{left}" y="22">{escape(title)}</text>')

    if not pts:
        out.append(f'<text class="st-muted" x="{left}" y="{top + ph / 2:.0f}">no candidates yet</text></svg>')
        return "".join(out)

    n = len(pts)

    def px(i: int) -> float:
        return left + (pw * (i + 0.5) / n)

    strip_y = top + ph + 26  # the row for markers that have no score (invalid candidates, or everything in the bit-only view)

    if has_values:
        vals = [v for p in pts for v in (p.value, p.incumbent) if v is not None]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.01, hi + 0.01
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad

        def py(v: float) -> float:
            return top + ph * (1.0 - (v - lo) / (hi - lo))

        for k in range(5):
            gv = lo + (hi - lo) * k / 4
            gy = py(gv)
            out.append(f'<line class="st-grid" x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}"/>')
            out.append(f'<text class="st-muted" x="{left - 8}" y="{gy + 4:.1f}" text-anchor="end">{_fmt(gv)}</text>')
        out.append(f'<text class="st-muted" x="14" y="{top + ph / 2:.0f}" transform="rotate(-90 14 {top + ph / 2:.0f})" text-anchor="middle">{escape(y_label)}</text>')

        steps = [(px(i), p.incumbent) for i, p in enumerate(pts) if p.incumbent is not None]
        if steps:
            d = []
            prev_y: Optional[float] = None
            for x, v in steps:
                y = py(v)
                x0, x1 = x - pw / n / 2, x + pw / n / 2
                if prev_y is None:
                    d.append(f"M{x0:.1f},{y:.1f}")
                else:
                    d.append(f"L{x0:.1f},{prev_y:.1f}L{x0:.1f},{y:.1f}")
                d.append(f"L{x1:.1f},{y:.1f}")
                prev_y = y
            out.append(f'<path class="st-inc" d="{"".join(d)}"><title>incumbent score</title></path>')
        out.append(f'<line class="st-grid" x1="{left}" y1="{strip_y - 14:.1f}" x2="{left + pw}" y2="{strip_y - 14:.1f}"/>')
        out.append(f'<text class="st-muted" x="{left - 8}" y="{strip_y + 4:.1f}" text-anchor="end">no score</text>')
    else:
        out.append(f'<text class="st-muted" x="{left}" y="{top + 18}">Bit-only view: scores, deltas and margins are shown in the full report after the search is closed.</text>')
        strip_y = top + ph / 2

    for i, p in enumerate(pts):
        x = px(i)
        y = py(p.value) if (has_values and p.value is not None) else strip_y
        tip = f"{p.label}: {p.outcome}" + (f", score {_fmt(p.value)}" if p.value is not None else "") + (f", incumbent {_fmt(p.incumbent)}" if p.incumbent is not None else "")
        out.append(_marker(p.outcome, x, y, tip))
        lab = p.label if len(p.label) <= 10 else p.label[:9] + "…"
        out.append(f'<text class="st-muted" x="{x:.1f}" y="{top + ph + 48 if has_values else top + ph / 2 + 30:.1f}" text-anchor="middle" font-size="11">{escape(lab)}</text>')

    # legend
    lx, ly = left, height - 14
    for kind, name in (("accepted", "accepted"), ("rejected", "rejected"), ("invalid", "invalid")):
        out.append(_marker(kind, lx + 7, ly - 4, name))
        out.append(f'<text x="{lx + 20}" y="{ly}">{name}</text>')
        lx += 100
    if has_values:
        out.append(f'<line class="st-inc" x1="{lx}" y1="{ly - 4}" x2="{lx + 22}" y2="{ly - 4}"/><text x="{lx + 28}" y="{ly}">incumbent</text>')
    out.append("</svg>")
    return "".join(out)
