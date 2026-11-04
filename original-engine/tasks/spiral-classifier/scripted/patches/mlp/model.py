"""A small neural network written by hand: two tanh layers of 32 units, trained with Adam on the cross-entropy for a fixed
number of full-batch epochs from a fixed random start."""
import numpy as np

HIDDEN = (32, 32)
EPOCHS = 600
LEARNING_RATE = 0.02
INIT_SEED = 7


def _forward(layers, X):
    acts = [X]
    for W, b in layers[:-1]:
        acts.append(np.tanh(acts[-1] @ W + b))
    W, b = layers[-1]
    return acts, (acts[-1] @ W + b).reshape(-1)


def fit_predict(X_train, y_train, X_query):
    rng = np.random.default_rng(INIT_SEED)
    sizes = [2, *HIDDEN, 1]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params += [rng.normal(0.0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)]
    first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    n = len(X_train)
    for epoch in range(1, EPOCHS + 1):
        layers = list(zip(params[0::2], params[1::2]))
        acts, z = _forward(layers, X_train)
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        delta = ((prob - y_train) / n).reshape(-1, 1)
        grads = []
        for i in reversed(range(len(layers))):
            grads = [acts[i].T @ delta, delta.sum(axis=0)] + grads
            if i > 0:
                delta = (delta @ layers[i][0].T) * (1.0 - acts[i] ** 2)
        for k, g in enumerate(grads):
            first[k] = 0.9 * first[k] + 0.1 * g
            second[k] = 0.999 * second[k] + 0.001 * g * g
            m_hat, v_hat = first[k] / (1.0 - 0.9 ** epoch), second[k] / (1.0 - 0.999 ** epoch)
            params[k] = params[k] - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + 1e-8)
    _, z = _forward(list(zip(params[0::2], params[1::2])), X_query)
    return (z > 0.0).astype(np.int64)
