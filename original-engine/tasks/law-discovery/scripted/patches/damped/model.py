"""Damped oscillation: y = C + exp(-g t) * (A cos(w t) + B sin(w t)).

For each decay rate g and angular frequency w on a grid, A, B and C follow from linear least squares; the (g, w) with
the smallest training error is refined on successively finer grids around the best cell.
"""
import math

from linalg import lstsq


def _features(t, g, w):
    decay = math.exp(-g * t)
    return [decay * math.cos(w * t), decay * math.sin(w * t), 1.0]


def _fit(ts, ys, g, w):
    """Training sum of squared errors and coefficients (A, B, C) for fixed (g, w); ``None`` if the system is singular."""
    rows = [_features(t, g, w) for t in ts]
    try:
        coef = lstsq(rows, ys, 1e-9)
    except ValueError:
        return None
    sse = sum((sum(c * f for c, f in zip(coef, row)) - y) ** 2 for row, y in zip(rows, ys))
    return sse, coef


def fit_predict(train, query_times):
    ts = [t for t, _ in train]
    ys = [y for _, y in train]
    best = None  # (sse, g, w, coef)
    for gi in range(21):
        for wi in range(36):
            g, w = 0.05 * gi, 0.5 + 0.1 * wi
            res = _fit(ts, ys, g, w)
            if res and (best is None or res[0] < best[0]):
                best = (res[0], g, w, res[1])
    if best is None:
        raise RuntimeError("no stable damped fit on the grid")
    step = 0.025
    for _ in range(3):
        _, g0, w0, _ = best
        for dg in range(-2, 3):
            for dw in range(-2, 3):
                g, w = max(0.0, g0 + dg * step), w0 + dw * 2 * step
                res = _fit(ts, ys, g, w)
                if res and res[0] < best[0]:
                    best = (res[0], g, w, res[1])
        step /= 2
    _, g, w, coef = best
    print("damped fit: g=%.3f w=%.3f" % (g, w))
    return [sum(c * f for c, f in zip(coef, _features(t, g, w))) for t in query_times]
