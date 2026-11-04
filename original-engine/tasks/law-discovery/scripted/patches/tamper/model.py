"""Quadratic trend with a light ridge penalty."""
from linalg import polyfit


def fit_predict(train, query_times):
    predict = polyfit([t for t, _ in train], [y for _, y in train], 2, ridge=1e-3)
    return [predict(t) for t in query_times]
