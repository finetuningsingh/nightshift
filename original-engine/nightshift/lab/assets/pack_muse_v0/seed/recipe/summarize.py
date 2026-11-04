#!/usr/bin/env python3
"""Merge generation/filter statistics and stage durations into timings.json."""
import argparse, json, os, sys

def load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-stats", default="")
    p.add_argument("--sft-stats", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--gen-seconds", type=int, default=0)
    p.add_argument("--filter-seconds", type=int, default=0)
    p.add_argument("--train-seconds", type=int, default=0)
    p.add_argument("--total-seconds", type=int, default=0)
    p.add_argument("--smoke", type=int, default=0)
    a = p.parse_args(argv)
    gen, sft = load(a.gen_stats), load(a.sft_stats)
    out = {
        "smoke": bool(a.smoke),
        "stage_seconds": {"generate": a.gen_seconds, "filter": a.filter_seconds,
                          "train": a.train_seconds, "total": a.total_seconds},
        "generation": {k: gen.get(k) for k in ("prompts", "requested", "completed", "errors",
                                               "length", "abandoned", "not_submitted", "timed_out")},
        "yield": {k: sft.get(k) for k in ("samples_in", "dropped", "prompts_total",
                                          "prompts_with_any_pass", "sft_examples", "by_family",
                                          "verifier_error_counts")},
    }
    if gen.get("prompts"):
        out["yield"]["prompt_pass_rate"] = round((sft.get("prompts_with_any_pass") or 0) / gen["prompts"], 3)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[summarize] wrote {a.out}: {json.dumps(out['stage_seconds'])} "
          f"sft_examples={out['yield'].get('sft_examples')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
