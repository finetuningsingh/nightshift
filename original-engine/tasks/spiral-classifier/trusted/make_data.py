"""Regenerate the spiral-classifier data. The committed CSVs are authoritative; this documents and reproduces them.

The world (known only here): two interleaved spiral arms around the origin. A point of class c in {0, 1} sits at

    radius t ~ U(T_MIN, 1),   angle = 2 pi * TURNS * (t - T_MIN) / (1 - T_MIN) + c * pi,   plus Gaussian jitter N(0, JITTER^2) on both coordinates,

and its recorded label is flipped with probability FLIP (label noise, in the training pool and in the evaluation rows alike,
so no model can score much above 1 - FLIP). Everything is deterministic: a fixed string seed, fixed sizes, and a fixed random
50/50 partition of the evaluation points into the ``dev`` and ``sealed`` splits (same distribution). Standard library only.

    python3 trusted/make_data.py [--out DIR]       writes the four CSVs below under DIR (default: the task directory)

Writes: template/train_pool.csv, template/eval_inputs.csv, trusted/train_pool.csv, trusted/labels.csv
"""
import argparse
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)

SEED = "nightshift/spiral-classifier/v1"
TURNS, T_MIN, JITTER, FLIP = 2.0, 0.10, 0.04, 0.06   # exactly two turns over the radii, so no linear rule beats chance by design
POOL_N, EVAL_N = 1200, 800


def sample_point(rng):
    """One ``(x, y, label)`` draw: a fair class, a radius, jitter, then a possible label flip."""
    cls = rng.randrange(2)
    t = rng.uniform(T_MIN, 1.0)
    angle = 2.0 * math.pi * TURNS * (t - T_MIN) / (1.0 - T_MIN) + cls * math.pi
    x = t * math.cos(angle) + rng.gauss(0.0, JITTER)
    y = t * math.sin(angle) + rng.gauss(0.0, JITTER)
    label = cls ^ (1 if rng.random() < FLIP else 0)
    return x, y, label


def generate():
    """Return ``(pool, evals, split)``: rows ``(id, x, y, label)`` and ``id -> 'dev'|'sealed'``."""
    rng = random.Random(SEED)
    pool = [("p%04d" % i, *sample_point(rng)) for i in range(POOL_N)]
    evals = [("e%04d" % i, *sample_point(rng)) for i in range(EVAL_N)]
    dev_idx = set(rng.sample(range(EVAL_N), EVAL_N // 2))
    split = {evals[i][0]: "dev" if i in dev_idx else "sealed" for i in range(EVAL_N)}
    return pool, evals, split


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(row) + "\n")


def write_all(out_dir):
    pool, evals, split = generate()
    pool_rows = [(i, "%.6f" % x, "%.6f" % y, str(c)) for i, x, y, c in pool]
    _write(os.path.join(out_dir, "template", "train_pool.csv"), ["id", "x1", "x2", "label"], pool_rows)
    _write(os.path.join(out_dir, "trusted", "train_pool.csv"), ["id", "x1", "x2", "label"], pool_rows)
    _write(os.path.join(out_dir, "template", "eval_inputs.csv"), ["id", "x1", "x2"],
           [(i, "%.6f" % x, "%.6f" % y) for i, x, y, _ in evals])
    _write(os.path.join(out_dir, "trusted", "labels.csv"), ["id", "split", "label"], [(i, split[i], str(c)) for i, _, _, c in evals])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=TASK_DIR, help="directory that receives template/ and trusted/ (default: the task directory)")
    args = ap.parse_args()
    write_all(args.out)


if __name__ == "__main__":
    main()
