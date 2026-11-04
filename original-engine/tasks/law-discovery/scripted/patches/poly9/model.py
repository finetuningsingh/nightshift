"""Degree-9 polynomial in time with a light ridge penalty (keeps the normal equations well conditioned)."""
from linalg import polyfit


def fit_predict(train, query_times):
    predict = polyfit([t for t, _ in train], [y for _, y in train], 9, ridge=1.0)
    return [predict(t) for t in query_times]
