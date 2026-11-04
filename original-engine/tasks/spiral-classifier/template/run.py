"""Train on this run's sample, then classify every evaluation point and every pool point. Read-only; edit model.py."""
import csv
import os
import sys

import numpy as np

import data
import model

HERE = os.path.dirname(os.path.abspath(__file__))


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["id", "pred"])
        writer.writerows(rows)


def main():
    seed = int(os.environ["NIGHTSHIFT_SEED"])
    _, X, y = data.load_train(seed)
    eval_ids, X_eval = data.load_eval_inputs()
    pool_ids, X_pool, _ = data.load_pool()
    print("training on %d of %d labelled points" % (len(X), len(pool_ids)), flush=True)
    queries = np.vstack([X_eval, X_pool])
    preds = np.asarray(model.fit_predict(X, y, queries), dtype=np.float64).reshape(-1)
    if preds.shape[0] != queries.shape[0]:
        sys.exit("fit_predict returned %d predictions for %d points" % (preds.shape[0], queries.shape[0]))
    if not np.all((preds == 0.0) | (preds == 1.0)):
        sys.exit("fit_predict must return the class 0 or 1 for every point")
    out = os.path.join(HERE, "outputs")
    os.makedirs(out, exist_ok=True)
    classes = [str(int(p)) for p in preds]
    _write(os.path.join(out, "predictions.csv"), zip(eval_ids, classes[: len(eval_ids)]))
    _write(os.path.join(out, "train_predictions.csv"), zip(pool_ids, classes[len(eval_ids):]))
    print("wrote %d predictions" % len(classes), flush=True)


if __name__ == "__main__":
    main()
