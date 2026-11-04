"""A far larger network, trained until it matches every training point: one hidden layer of 1024 sinusoidal units with a
very high first-layer frequency, and an output layer solved exactly (ridge 1e-6) so that the training error is driven to
zero. Capacity is no longer the limit; the number of training points is."""
import numpy as np

WIDTH = 1024
FREQUENCY = 400.0
RIDGE = 1e-6
INIT_SEED = 11


def fit_predict(X_train, y_train, X_query):
    rng = np.random.default_rng(INIT_SEED)
    W = rng.normal(0.0, FREQUENCY, (2, WIDTH))
    b = rng.uniform(0.0, 2.0 * np.pi, WIDTH)
    scale = np.sqrt(2.0 / WIDTH)
    H = np.cos(X_train @ W + b) * scale
    target = 2.0 * y_train - 1.0
    alpha = np.linalg.solve(H @ H.T + RIDGE * np.eye(len(H)), target)
    readout = H.T @ alpha
    return ((np.cos(X_query @ W + b) * scale) @ readout > 0.0).astype(np.int64)
