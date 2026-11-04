"""Read-only data access for the mini-curve fixture. Do not edit.

``load_train(seed)`` returns a seeded 80% subsample (without replacement) of the fixed training pool, so a different
seed means different training data.
"""
import csv
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read(name):
    with open(os.path.join(_HERE, name), newline="") as fh:
        return list(csv.DictReader(fh))


def load_pool():
    """Every training-pool row as ``(id, x, y)``."""
    return [(r["id"], float(r["x"]), float(r["y"])) for r in _read("train_pool.csv")]


def load_train(seed):
    """Seeded 80% subsample of the pool as ``(id, x, y)`` rows, in pool order."""
    rows = load_pool()
    rng = random.Random("train:%s" % seed)
    keep = sorted(rng.sample(range(len(rows)), k=int(0.8 * len(rows))))
    return [rows[i] for i in keep]


def load_eval_inputs():
    """Every row to predict as ``(id, x)``. Labels are not available here."""
    return [(r["id"], float(r["x"])) for r in _read("eval_inputs.csv")]
