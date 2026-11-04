"""The editable model. ``fit(xs, ys)`` returns a function mapping x to a prediction."""


def fit(xs, ys):
    mean = sum(ys) / len(ys)
    return lambda x: mean
