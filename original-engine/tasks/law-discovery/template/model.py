"""The editable model. Predict later measurements of a damped oscillator from earlier, noisy ones.

``fit_predict(train, query_times)``
    train        list of ``(t, y)`` measurements taken at times t in [0, 4]
    query_times  list of times to predict: the later times (t in [4, 8]) followed by the training-window times
    returns      a list of floats, one prediction per query time, in the same order

Baseline: an ordinary least-squares straight line, fitted by hand. Standard library only (``linalg.py`` has helpers).
"""


def fit_predict(train, query_times):
    n = len(train)
    mean_t = sum(t for t, _ in train) / n
    mean_y = sum(y for _, y in train) / n
    sxx = sum((t - mean_t) ** 2 for t, _ in train)
    sxy = sum((t - mean_t) * (y - mean_y) for t, y in train)
    slope = sxy / sxx
    return [mean_y + slope * (t - mean_t) for t in query_times]
