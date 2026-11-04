#!/usr/bin/env python3
"""Keep the sampled answers that pass every constraint and write chat-format SFT rows. Standard library only.

    python3 build_sft.py --samples samples.jsonl --out sft.jsonl --per-prompt 1

The checker is /opt/lab/rewards.py (``$REWARDS_PY``) with the pinned IFEvalG files (``$IFEVALG_PATH``): the same code the dev score
uses. Output rows: {"messages": [system, user, assistant]}; the assistant message carries ``reasoning_content`` when the sample had it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

SYSTEM = "You are a helpful AI assistant."


def load_rewards(path):
    spec = importlib.util.spec_from_file_location("lab_rewards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select(samples, passes, per_prompt):
    """Per prompt key, keep up to ``per_prompt`` samples for which ``passes(sample)`` is true (shortest answers first).

    Returns (kept, stats) where stats counts prompts seen, prompts with at least one passing answer, and answers kept.
    """
    by_key = {}
    for s in samples:
        by_key.setdefault(s["key"], []).append(s)
    kept = []
    solved = 0
    for key in sorted(by_key):
        good = [s for s in by_key[key] if passes(s)]
        if good:
            solved += 1
        good.sort(key=lambda s: (len(s["response"]), s["response"]))
        kept.extend(good[:per_prompt])
    return kept, {"prompts": len(by_key), "prompts_solved": solved, "kept": len(kept), "samples": len(samples)}


def to_chat(sample):
    assistant = {"role": "assistant", "content": sample["response"]}
    if sample.get("reasoning"):
        assistant["reasoning_content"] = sample["reasoning"]
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": sample["prompt"]}, assistant]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-prompt", type=int, default=1)
    ap.add_argument("--rewards", default=os.environ.get("REWARDS_PY", "/opt/lab/rewards.py"))
    args = ap.parse_args(argv)
    rewards = load_rewards(args.rewards)
    rewards.preflight()
    with open(args.samples, encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]
    kept, stats = select(samples, lambda s: rewards.reward(s["response"], s["ground_truth"], mode="all") >= 1.0, args.per_prompt)
    errors = rewards.error_counts()
    with open(args.out, "w", encoding="utf-8") as out:
        for s in kept:
            out.write(json.dumps(to_chat(s), ensure_ascii=False) + "\n")
    stats["checker_errors"] = sum(errors.values()) if isinstance(errors, dict) else 0
    print(json.dumps(stats))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
