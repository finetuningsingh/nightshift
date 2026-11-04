"""Nearest-neighbour vote: each point takes the majority class of its 7 nearest labelled points."""
import numpy as np

NEIGHBOURS = 7


def fit_predict(X_train, y_train, X_query):
    d2 = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argpartition(d2, NEIGHBOURS, axis=1)[:, :NEIGHBOURS]
    return (y_train[nearest].mean(axis=1) > 0.5).astype(np.int64)
