"""Iterative refinement of a straight line that waits for a convergence test which is never met."""
import time

from linalg import polyfit


def fit_predict(train, query_times):
    predict = polyfit([t for t, _ in train], [y for _, y in train], 1)
    change = 1.0
    while change > 0.0:      # the update is never exactly zero, so this loop never ends
        time.sleep(1.0)
    return [predict(t) for t in query_times]
