"""Polar harmonics: describe every point by its radius r and angle a, then fit a logistic regression on sin and cos of
(a - w r) for a ladder of spiral tightnesses w. A spiral arm is a straight line in (r, a), so the matching harmonic
separates the two arms."""
import numpy as np

OMEGAS = np.arange(6.0, 22.01, 1.0)
RIDGE = 1.0
ITERATIONS = 30


def polar_features(X):
    r = np.hypot(X[:, 0], X[:, 1])
    angle = np.arctan2(X[:, 1], X[:, 0])
    phase = angle[:, None] - OMEGAS[None, :] * r[:, None]
    return np.column_stack([np.ones(len(X)), np.sin(phase), np.cos(phase)])


def fit_predict(X_train, y_train, X_query):
    A = polar_features(X_train)
    w = np.zeros(A.shape[1])
    penalty = RIDGE * np.eye(A.shape[1])
    penalty[0, 0] = 0.0                                   # the intercept is not shrunk
    for _ in range(ITERATIONS):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30.0, 30.0)))
        grad = A.T @ (p - y_train) + penalty @ w
        hess = (A.T * (p * (1.0 - p))) @ A + penalty + 1e-9 * np.eye(A.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return (polar_features(X_query) @ w > 0.0).astype(np.int64)
