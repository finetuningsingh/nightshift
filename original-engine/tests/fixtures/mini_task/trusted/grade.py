"""Trusted grader for the mini-curve fixture.

    grade.py --workspace STAGED --seed S --split dev|sealed

Reads only STAGED/outputs/*.csv (parsed as CSV, never executed) and its own labels. Prints one JSON object as the
last stdout line.
"""
import argparse
import csv
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 5 * 1024 * 1024


def read_csv(path):
    if os.path.getsize(path) > MAX_BYTES:
        raise ValueError("file too large")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def read_predictions(path):
    preds = {}
    for row in read_csv(path):
        if set(row) != {"id", "pred"}:
            raise ValueError("header must be id,pred")
        value = float(row["pred"])
        if not math.isfinite(value) or row["id"] in preds:
            raise ValueError("non-finite or duplicate prediction")
        preds[row["id"]] = value
    return preds


def rmse(pairs):
    return math.sqrt(sum((p - y) ** 2 for p, y in pairs) / len(pairs))


def emit(valid, split, notes, score=None, n_eval=0, metrics=None):
    print(json.dumps({"valid": valid, "score": score, "split": split, "n_eval": n_eval,
                      "metrics": metrics or {}, "notes": notes}, allow_nan=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["dev", "sealed"], required=True)
    args = ap.parse_args()

    labels = {r["id"]: (r["split"], float(r["y"])) for r in read_csv(os.path.join(HERE, "labels.csv"))}
    pool = {r["id"]: float(r["y"]) for r in read_csv(os.path.join(HERE, "train_pool.csv"))}
    out = os.path.join(args.workspace, "outputs")
    try:
        preds = read_predictions(os.path.join(out, "predictions.csv"))
        train_preds = read_predictions(os.path.join(out, "train_predictions.csv"))
    except (OSError, ValueError, KeyError) as exc:
        return emit(False, args.split, "unreadable predictions: %s" % type(exc).__name__)
    if any(i not in preds for i in labels):
        return emit(False, args.split, "predictions missing for some rows")

    rows = sorted(i for i, (s, _) in labels.items() if s == args.split)
    chosen = random.Random("%s:%s" % (args.split, args.seed)).sample(rows, k=int(0.7 * len(rows)))
    heldout = rmse([(preds[i], labels[i][1]) for i in chosen])

    ids = sorted(pool)
    keep = sorted(random.Random("train:%s" % args.seed).sample(range(len(ids)), k=int(0.8 * len(ids))))
    train_ids = [ids[i] for i in keep]
    if any(i not in train_preds for i in train_ids):
        return emit(False, args.split, "train predictions missing for some rows")
    train = rmse([(train_preds[i], pool[i]) for i in train_ids])
    emit(True, args.split, "", heldout, len(chosen), {"heldout_rmse": heldout, "train_rmse": train})


if __name__ == "__main__":
    sys.exit(main())
