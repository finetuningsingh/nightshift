#!/usr/bin/env python3
"""Paired comparison of two scores of the same prompts: python3 paired.py LAB_ROOT STOCK_SCORE_ID ADAPTER_SCORE_ID

Reads each score's per_prompt.jsonl (one sample per prompt), joins on prompt_key, and reports pass-all and mean-fraction for both arms, the
discordant pairs, an exact two-sided McNemar p-value, and a 95% paired bootstrap interval of the pass-all difference (fixed seed)."""
import json, math, random, sys
from pathlib import Path


def load(root, sid):
    rows = [json.loads(l) for l in (Path(root) / "scores" / sid / "per_prompt.jsonl").read_text().splitlines() if l.strip()]
    return {r["prompt_key"]: r for r in rows if r.get("seed") == 0}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main(root, stock_id, adapter_id):
    a, b = load(root, stock_id), load(root, adapter_id)
    keys = sorted(set(a) & set(b))
    only = len(set(a) ^ set(b))
    pa = [int(bool(a[k]["passed"])) for k in keys]
    pb = [int(bool(b[k]["passed"])) for k in keys]
    n = len(keys)
    d = [y - x for x, y in zip(pa, pb)]
    rng = random.Random(0)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000)) if n else [0, 0]
    out = {
        "n_paired": n, "n_unpaired": only,
        "stock": {"pass_all": sum(pa) / n, "mean_fraction": sum(a[k]["fraction"] for k in keys) / n, "truncated": sum(bool(a[k]["truncated"]) for k in keys), "errors": sum(bool(a[k]["error"]) for k in keys)},
        "adapter": {"pass_all": sum(pb) / n, "mean_fraction": sum(b[k]["fraction"] for k in keys) / n, "truncated": sum(bool(b[k]["truncated"]) for k in keys), "errors": sum(bool(b[k]["error"]) for k in keys)},
        "diff_pass_all": sum(d) / n, "ci95_paired_bootstrap": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]],
        "adapter_only_pass": sum(1 for x, y in zip(pa, pb) if y > x), "stock_only_pass": sum(1 for x, y in zip(pa, pb) if x > y),
    }
    out["mcnemar_exact_p"] = mcnemar_exact(out["adapter_only_pass"], out["stock_only_pass"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
