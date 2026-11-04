"""Regenerate the law-discovery data. The committed CSVs are authoritative; this documents and reproduces them.

The law (known only here): a damped oscillation around an offset,

    y(t) = OFFSET + exp(-GAMMA t) * (COS_AMP cos(OMEGA t) + SIN_AMP sin(OMEGA t)) + noise,   noise ~ N(0, SIGMA^2)

Measurements for fitting are taken at t in [0, 4]; the later times to predict are in [4, 8]. Everything is
deterministic: a string seed, fixed sizes, and a random 50/50 partition of the later times into the ``dev`` and
``sealed`` splits (same distribution). The default seed is the one the committed files were made with, and it is PUBLIC:
anyone who reads this repository knows the sealed rows. For a demo whose sealed split nobody has seen, draw a fresh one.

    python3 trusted/make_data.py [--out DIR] [--seed TEXT]
        writes the four CSVs below under DIR (default: the task directory) from the seed TEXT (default: the shared seed)

Writes: template/train_pool.csv, template/eval_inputs.csv, trusted/train_pool.csv, trusted/labels.csv
"""
import argparse
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)

SEED = "nightshift/law-discovery/v1"
OFFSET, COS_AMP, SIN_AMP, GAMMA, OMEGA = 0.30, 1.0, 1.0, 0.46, 1.18
SIGMA = 0.05
POOL_N, EVAL_N = 200, 600
TRAIN_RANGE, EVAL_RANGE = (0.0, 4.0), (4.0, 8.0)


def law(t):
    """The noiseless signal."""
    return OFFSET + math.exp(-GAMMA * t) * (COS_AMP * math.cos(OMEGA * t) + SIN_AMP * math.sin(OMEGA * t))


def generate(seed=SEED):
    """Return ``(pool, evals, split)``: pool rows ``(id, t, y)``, eval rows ``(id, t, y)``, ``id -> 'dev'|'sealed'``.

    ``seed`` is any string; the default reproduces the committed files byte for byte."""
    rng = random.Random(seed)
    pool_t = sorted(round(rng.uniform(*TRAIN_RANGE), 6) for _ in range(POOL_N))
    pool = [("p%03d" % i, t, law(t) + rng.gauss(0.0, SIGMA)) for i, t in enumerate(pool_t)]
    eval_t = [round(rng.uniform(*EVAL_RANGE), 6) for _ in range(EVAL_N)]        # file order is already random in t
    evals = [("e%03d" % i, t, law(t) + rng.gauss(0.0, SIGMA)) for i, t in enumerate(eval_t)]
    dev_idx = set(rng.sample(range(EVAL_N), EVAL_N // 2))
    split = {evals[i][0]: "dev" if i in dev_idx else "sealed" for i in range(EVAL_N)}
    return pool, evals, split


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(row) + "\n")


def write_all(out_dir, seed=SEED):
    pool, evals, split = generate(seed)
    pool_rows = [(i, "%.6f" % t, "%.6f" % y) for i, t, y in pool]
    _write(os.path.join(out_dir, "template", "train_pool.csv"), ["id", "t", "y"], pool_rows)
    _write(os.path.join(out_dir, "trusted", "train_pool.csv"), ["id", "t", "y"], pool_rows)
    _write(os.path.join(out_dir, "template", "eval_inputs.csv"), ["id", "t"], [(i, "%.6f" % t) for i, t, _ in evals])
    _write(os.path.join(out_dir, "trusted", "labels.csv"), ["id", "split", "y"], [(i, split[i], "%.6f" % y) for i, _, y in evals])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=TASK_DIR, help="directory that receives template/ and trusted/ (default: the task directory)")
    ap.add_argument("--seed", default=SEED,
                    help="text that seeds the data (default: %r). The default split is the shared one and is PUBLIC "
                         "(committed to the repository), so its sealed rows are known to anyone who read it. Use a fresh seed "
                         "for a live demo whose sealed split nobody has seen, and keep that seed and the written "
                         "trusted/ files private. Without --out the files are written over the task's committed data "
                         "(use --out DIR to leave them alone, and never commit a private split). The noise calibration in task.toml was measured on the shared split: "
                         "re-run `nightshift tasks` after changing the data." % SEED)
    args = ap.parse_args()
    write_all(args.out, args.seed)


if __name__ == "__main__":
    main()
