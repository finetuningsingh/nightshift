"""Logistic regression on the raw coordinates with a vanishing ridge penalty (1e-6 instead of 1e-3)."""
import numpy as np

RIDGE = 1e-6
ITERATIONS = 25


def fit_predict(X_train, y_train, X_query):
    A = np.column_stack([np.ones(len(X_train)), X_train])
    w = np.zeros(A.shape[1])
    penalty = RIDGE * np.eye(A.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(ITERATIONS):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30.0, 30.0)))
        grad = A.T @ (p - y_train) + penalty @ w
        hess = (A.T * (p * (1.0 - p))) @ A + penalty + 1e-9 * np.eye(A.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    Q = np.column_stack([np.ones(len(X_query)), X_query])
    return (Q @ w > 0.0).astype(np.int64)
