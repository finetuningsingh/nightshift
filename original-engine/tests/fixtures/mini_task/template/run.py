"""Train on this seed's sample, then predict every evaluation row and every training row."""
import csv
import os
import sys

import data
import model


def _write(path, header, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    seed = int(os.environ["NIGHTSHIFT_SEED"])
    rows = data.load_train(seed)
    predict = model.fit([x for _, x, _ in rows], [y for _, _, y in rows])
    print("trained on %d rows" % len(rows), flush=True)
    os.makedirs("outputs", exist_ok=True)
    _write("outputs/predictions.csv", ["id", "pred"], [(i, repr(float(predict(x)))) for i, x in data.load_eval_inputs()])
    _write("outputs/train_predictions.csv", ["id", "pred"], [(i, repr(float(predict(x)))) for i, x, _ in rows])
    print("wrote predictions", flush=True)


if __name__ == "__main__":
    sys.exit(main())
