"""Read-only data access for law-discovery. Do not edit.

``load_train(seed)`` returns a seeded 80% subsample (without replacement) of the fixed measurement pool, so each
seed means different training data. ``load_eval_inputs()`` returns the later times to predict; their values are
not available here.
"""
import csv
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read(name):
    with open(os.path.join(_HERE, name), newline="") as fh:
        return list(csv.DictReader(fh))


def load_pool():
    """Every measurement in the pool as ``(id, t, y)``, in file order."""
    return [(r["id"], float(r["t"]), float(r["y"])) for r in _read("train_pool.csv")]


def load_train(seed):
    """Seeded 80% subsample of the pool as ``(id, t, y)`` rows, in pool order."""
    rows = load_pool()
    rng = random.Random("train:%s" % seed)
    keep = sorted(rng.sample(range(len(rows)), k=int(0.8 * len(rows))))
    return [rows[i] for i in keep]


def load_eval_inputs():
    """Every time to predict as ``(id, t)``. No measurements are available for these."""
    return [(r["id"], float(r["t"])) for r in _read("eval_inputs.csv")]
