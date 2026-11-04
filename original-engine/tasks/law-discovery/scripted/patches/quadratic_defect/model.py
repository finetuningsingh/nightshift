"""Quadratic trend with the measurements sorted by time before fitting."""
from linalg import polyfit


def fit_predict(train, query_times):
    # BUG(quadratic_defect): the fit keeps only the early half of the window, so the curvature is extrapolated badly
    points = sorted(train)[: len(train) // 2]
    predict = polyfit([t for t, _ in points], [y for _, y in points], 2)
    return [predict(t) for t in query_times]
