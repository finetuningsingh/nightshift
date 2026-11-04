"""Polar harmonics with a sweep. Every 5th training point is set aside as a private holdout; each combination of frequency
band, frequency spacing and ridge strength is fitted on the rest and scored on that holdout, and the best-scoring
combination is refitted on all the training points."""
import numpy as np

BANDS = ((6.0, 22.0), (8.0, 20.0), (10.0, 18.0))
SPACINGS = (1.0, 2.0)
RIDGES = (0.01, 0.1, 1.0, 10.0)
HOLDOUT_EVERY = 5
ITERATIONS = 15


def _features(X, low, high, spacing):
    r = np.hypot(X[:, 0], X[:, 1])
    angle = np.arctan2(X[:, 1], X[:, 0])
    phase = angle[:, None] - np.arange(low, high + 1e-9, spacing)[None, :] * r[:, None]
    return np.column_stack([np.ones(len(X)), np.sin(phase), np.cos(phase)])


def _fit(A, y, ridge):
    w = np.zeros(A.shape[1])
    penalty = ridge * np.eye(A.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(ITERATIONS):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30.0, 30.0)))
        hess = (A.T * (p * (1.0 - p))) @ A + penalty + 1e-9 * np.eye(A.shape[1])
        step = np.linalg.solve(hess, A.T @ (p - y) + penalty @ w)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def fit_predict(X_train, y_train, X_query):
    holdout = np.arange(len(X_train)) % HOLDOUT_EVERY == 0
    best = None
    for low, high in BANDS:
        for spacing in SPACINGS:
            A_fit = _features(X_train[~holdout], low, high, spacing)
            A_hold = _features(X_train[holdout], low, high, spacing)
            for ridge in RIDGES:
                w = _fit(A_fit, y_train[~holdout], ridge)
                accuracy = float(np.mean((A_hold @ w > 0.0) == (y_train[holdout] == 1)))
                if best is None or accuracy > best[0]:
                    best = (accuracy, low, high, spacing, ridge)
    accuracy, low, high, spacing, ridge = best
    print("sweep picked band %g-%g, spacing %g, ridge %g (private holdout accuracy %.3f)" % (low, high, spacing, ridge, accuracy))
    w = _fit(_features(X_train, low, high, spacing), y_train, ridge)
    return (_features(X_query, low, high, spacing) @ w > 0.0).astype(np.int64)
