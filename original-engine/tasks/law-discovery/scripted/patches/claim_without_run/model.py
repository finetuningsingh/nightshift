"""Straight line with a tiny ridge penalty, so the slope is shrunk slightly toward zero."""
from linalg import polyfit


def fit_predict(train, query_times):
    predict = polyfit([t for t, _ in train], [y for _, y in train], 1, ridge=1e-6)
    return [predict(t) for t in query_times]
