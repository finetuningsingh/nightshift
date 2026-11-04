"""Straight line on centred time with a negligible ridge term for numerical stability."""
from linalg import polyfit


def fit_predict(train, query_times):
    predict = polyfit([t for t, _ in train], [y for _, y in train], 1, ridge=1e-9)
    return [predict(t) for t in query_times]
