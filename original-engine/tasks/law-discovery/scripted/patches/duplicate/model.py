"""Quadratic trend: a degree-2 polynomial in time, fitted by least squares."""
from linalg import polyfit


def fit_predict(train, query_times):
    predict = polyfit([t for t, _ in train], [y for _, y in train], 2)
    return [predict(t) for t in query_times]
