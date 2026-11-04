"""Backtested polynomial: sweep degree and ridge strength on the latest part of the training window.

The last 25% of the measurements (by time) is held out as a pseudo-future. Every (degree, ridge) pair is fitted on the
earlier 75% and scored on the held-out part; the winner is refitted on all measurements.
"""
from linalg import polyfit

DEGREES = (1, 2, 3, 4)
RIDGES = (0.0, 1e-3, 1e-1, 10.0)
HOLDOUT = 0.25


def _holdout_rmse(early, late, degree, ridge):
    predict = polyfit([t for t, _ in early], [y for _, y in early], degree, ridge)
    return (sum((predict(t) - y) ** 2 for t, y in late) / len(late)) ** 0.5


def fit_predict(train, query_times):
    points = sorted(train)
    cut = int(len(points) * (1 - HOLDOUT))
    early, late = points[:cut], points[cut:]
    best = None
    for degree in DEGREES:
        for ridge in RIDGES:
            try:
                err = _holdout_rmse(early, late, degree, ridge)
            except ValueError:
                continue
            if best is None or err < best[0]:
                best = (err, degree, ridge)
    err, degree, ridge = best
    print("backtest picked degree=%d ridge=%g (holdout rmse %.4f)" % (degree, ridge, err))
    predict = polyfit([t for t, _ in points], [y for _, y in points], degree, ridge)
    return [predict(t) for t in query_times]
