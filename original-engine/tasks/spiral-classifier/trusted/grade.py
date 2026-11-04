"""Trusted grader for spiral-classifier.

    grade.py --workspace STAGED --seed S --split dev|sealed

Reads only STAGED/outputs/predictions.csv and STAGED/outputs/train_predictions.csv (parsed with the csv module, never
executed) and its own labels. It never imports workspace code. Prints one strict-JSON object as the last stdout line and
exits 0 for any graded outcome; ``valid`` is false when the predictions are missing, short, duplicated, malformed, or not
exactly 0 or 1.

Held-out metric: accuracy on a seeded 70% draw of the split's rows. Train metric: accuracy on the training subset the seed
selects from the pool (the grader keeps its own copy of that selection rule).
"""
import argparse
import csv
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 4 * 1024 * 1024
MAX_ROWS = 5000
EVAL_FRACTION = 0.7
TRAIN_FRACTION = 0.8


class Invalid(Exception):
    """The submitted predictions cannot be graded; the message is a short, label-free reason."""


def read_rows(path):
    """All rows of a small CSV file, header included, via the csv module."""
    if os.path.getsize(path) > MAX_BYTES:
        raise Invalid("file too large")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = []
        for row in csv.reader(fh):
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise Invalid("too many rows")
    return rows


def read_labels():
    """id -> (split, class) for every evaluation point."""
    rows = read_rows(os.path.join(HERE, "labels.csv"))
    return {r[0]: (r[1], int(r[2])) for r in rows[1:]}


def read_pool():
    """id -> class for every labelled point in the pool."""
    rows = read_rows(os.path.join(HERE, "train_pool.csv"))
    return {r[0]: int(r[3]) for r in rows[1:]}


def read_predictions(path, expected_ids, what):
    """id -> 0|1, requiring header ``id,pred`` and exactly the expected set of ids, each once, every value 0 or 1."""
    try:
        rows = read_rows(path)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise Invalid("%s unreadable (%s)" % (what, type(exc).__name__))
    if not rows or rows[0] != ["id", "pred"]:
        raise Invalid("%s header must be id,pred" % what)
    preds = {}
    for row in rows[1:]:
        if len(row) != 2:
            raise Invalid("%s has a malformed row" % what)
        try:
            value = float(row[1])
        except ValueError:
            raise Invalid("%s has a non-numeric prediction" % what)
        if not math.isfinite(value) or value not in (0.0, 1.0):
            raise Invalid("%s has a prediction that is not 0 or 1" % what)
        if row[0] in preds:
            raise Invalid("%s repeats an id" % what)
        preds[row[0]] = int(value)
    missing = len(expected_ids) - len(expected_ids & preds.keys())
    if missing:
        raise Invalid("%s is missing %d of %d predictions" % (what, missing, len(expected_ids)))
    if len(preds) != len(expected_ids):
        raise Invalid("%s has predictions for unknown ids" % what)
    return preds


def select_eval_ids(labels, split, seed):
    """The evaluated rows: a seeded 70% draw of the split's fixed row set (sorted ids)."""
    rows = sorted(i for i, (s, _) in labels.items() if s == split)
    return random.Random("%s:%s" % (split, seed)).sample(rows, k=int(EVAL_FRACTION * len(rows)))


def select_train_ids(pool_ids, seed):
    """The seed's training subset: a seeded 80% draw of the pool, in pool order (same rule as the workspace data module)."""
    ids = list(pool_ids)
    keep = sorted(random.Random("train:%s" % seed).sample(range(len(ids)), k=int(TRAIN_FRACTION * len(ids))))
    return [ids[i] for i in keep]


def accuracy(pairs):
    return math.fsum(1.0 for p, y in pairs if p == y) / len(pairs)


def grade(workspace, seed, split):
    """Return the result object for one seed and split."""
    labels, pool = read_labels(), read_pool()
    out = os.path.join(workspace, "outputs")
    try:
        preds = read_predictions(os.path.join(out, "predictions.csv"), set(labels), "predictions.csv")
        train_preds = read_predictions(os.path.join(out, "train_predictions.csv"), set(pool), "train_predictions.csv")
        chosen = select_eval_ids(labels, split, seed)
        heldout = accuracy([(preds[i], labels[i][1]) for i in chosen])
        train = accuracy([(train_preds[i], pool[i]) for i in select_train_ids(pool, seed)])
    except Invalid as exc:
        return {"valid": False, "score": None, "split": split, "n_eval": 0, "metrics": {}, "notes": str(exc)}
    return {"valid": True, "score": heldout, "split": split, "n_eval": len(chosen),
            "metrics": {"heldout_accuracy": heldout, "train_accuracy": train}, "notes": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["dev", "sealed"], required=True)
    args = ap.parse_args()
    print(json.dumps(grade(args.workspace, args.seed, args.split), allow_nan=False))


if __name__ == "__main__":
    sys.exit(main())
