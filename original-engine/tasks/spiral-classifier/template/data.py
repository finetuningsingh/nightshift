"""Read-only data access for spiral-classifier. Do not edit.

``load_train(seed)`` returns a seeded 80% subsample (without replacement) of the fixed pool of labelled points, so each
seed means different training data. ``load_eval_inputs()`` returns the points to classify; their labels are not
available here. Coordinates are float64 arrays of shape (n, 2); labels are int64 arrays of 0 and 1.
"""
import csv
import os
import random

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read(name):
    with open(os.path.join(_HERE, name), newline="") as fh:
        return list(csv.DictReader(fh))


def load_pool():
    """Every labelled point in the pool as ``(ids, X, y)``, in file order."""
    rows = _read("train_pool.csv")
    ids = [r["id"] for r in rows]
    X = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=np.float64).reshape(-1, 2)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    return ids, X, y


def load_train(seed):
    """Seeded 80% subsample of the pool as ``(ids, X, y)``, in pool order."""
    ids, X, y = load_pool()
    rng = random.Random("train:%s" % seed)
    keep = sorted(rng.sample(range(len(ids)), k=int(0.8 * len(ids))))
    return [ids[i] for i in keep], X[keep], y[keep]


def load_eval_inputs():
    """Every point to classify as ``(ids, X)``. No labels are available for these."""
    rows = _read("eval_inputs.csv")
    ids = [r["id"] for r in rows]
    X = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=np.float64).reshape(-1, 2)
    return ids, X
