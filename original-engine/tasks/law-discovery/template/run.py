"""Train on this run's sample, then predict every evaluation time and every pool time. Read-only; edit model.py."""
import csv
import os
import sys

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
    rows = data.load_train(seed)
    evals = data.load_eval_inputs()
    pool = data.load_pool()
    print("training on %d of %d measurements" % (len(rows), len(pool)), flush=True)
    queries = [t for _, t in evals] + [t for _, t, _ in pool]
    preds = [float(p) for p in model.fit_predict([(t, y) for _, t, y in rows], queries)]
    if len(preds) != len(queries):
        sys.exit("fit_predict returned %d predictions for %d query times" % (len(preds), len(queries)))
    out = os.path.join(HERE, "outputs")
    os.makedirs(out, exist_ok=True)
    _write(os.path.join(out, "predictions.csv"), [(i, repr(p)) for (i, _), p in zip(evals, preds)])
    _write(os.path.join(out, "train_predictions.csv"), [(i, repr(p)) for (i, _, _), p in zip(pool, preds[len(evals):])])
    print("wrote %d predictions" % len(preds), flush=True)


if __name__ == "__main__":
    main()
