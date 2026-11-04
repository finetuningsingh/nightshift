"""The editable model. Classify points of the plane into two classes from labelled examples.

``fit_predict(X_train, y_train, X_query)``
    X_train  float array of shape (n, 2): the coordinates (x1, x2) of the labelled training points
    y_train  int array of shape (n,): their classes, 0 or 1
    X_query  float array of shape (m, 2): the points to classify
    returns  an array of m values, each exactly 0 or 1, in the order of X_query

Baseline: logistic regression on the raw coordinates (numpy only), fitted by Newton's method with a small ridge term.
It is deterministic: the same training points always give the same predictions.
"""
import numpy as np

RIDGE = 1e-3
ITERATIONS = 25


def fit_predict(X_train, y_train, X_query):
    A = np.column_stack([np.ones(len(X_train)), X_train])
    w = np.zeros(A.shape[1])
    penalty = RIDGE * np.eye(A.shape[1])
    penalty[0, 0] = 0.0                                   # the intercept is not shrunk
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
