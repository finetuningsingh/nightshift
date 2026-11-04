#!/usr/bin/env python3
"""Verifier-filter generations into a chat-format SFT file + yield statistics.

Keep rule (all must hold): no request error, finish_reason != "length",
non-empty final answer, verify_constraints(content, ground_truth) is a
non-empty list of all True, and the example fits the training length guard.
Exact duplicates per prompt are dropped, then at most --max-keep-per-prompt
(shortest first) survive per prompt. The verifier sees `content` only, never
the reasoning trace. Zero (or too few) survivors is a hard failure: training
a junk adapter wastes more GPU time than stopping here.

Origin rule: training targets must be recorded, unmodified samples from the
local model (see origin_check.py). Answer/reasoning text is never altered
here (no stripping, re-wrapping or truncation to fit): a sample is either
kept byte-identical or dropped whole. With --train-file/--samples given
(run.sh always does), the approved train file supplies the user prompt and
ground_truth, and a generation survives only if its exact answer matches a
recorded muse-glimmer sample (finish_reason != length) in
glimmer_samples.jsonl; the emitted reasoning_content, if kept, is that
sample's recorded reasoning.
"""
import argparse, collections, hashlib, importlib.util, json, os, random, sys

import origin_check

def load_verifier(path):
    if not path or not os.path.isfile(path):
        raise SystemExit(f"FATAL: verifier module not found at REWARDS_PY={path!r}. "
                         "The harness must set REWARDS_PY to the rewards module file.")
    spec = importlib.util.spec_from_file_location("rewards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("preflight", "verify_constraints", "error_counts"):
        if not hasattr(mod, fn):
            raise SystemExit(f"FATAL: verifier module {path} has no {fn}(); wrong module?")
    return mod

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generations")
    p.add_argument("--out")
    p.add_argument("--stats")
    p.add_argument("--train-file", default="",
                   help="approved train.jsonl: source of the exact user prompt + ground_truth per key")
    p.add_argument("--samples", default="",
                   help="glimmer_samples.jsonl recorded by generate.py (default: next to --generations, if present)")
    p.add_argument("--rewards-py", default=os.environ.get("REWARDS_PY", ""))
    p.add_argument("--max-keep-per-prompt", type=int, default=2)
    p.add_argument("--max-train-examples", type=int, default=800, help="0 = no cap")
    p.add_argument("--min-examples", type=int, default=20)
    p.add_argument("--max-chars", type=int, default=14000,
                   help="prompt+answer character guard approximating sequence_len=4096 tokens (~3.4 chars/token)")
    p.add_argument("--keep-reasoning", action="store_true",
                   help="add reasoning_content to the assistant message (default: train on final answer only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preflight-only", action="store_true", help="load verifier, run preflight(), exit")
    args = p.parse_args(argv)

    verifier = load_verifier(args.rewards_py)
    try:
        verifier.preflight()
    except Exception as e:
        print(f"FATAL: verifier preflight() failed: {type(e).__name__}: {e}. "
              "Checker data (IFEVALG_PATH?) is missing or broken; do not generate against a broken verifier.",
              file=sys.stderr)
        return 2
    if args.preflight_only:
        print("[build_sft] verifier preflight OK")
        return 0
    if not (args.generations and args.out and args.stats):
        print("FATAL: --generations, --out and --stats are required (or use --preflight-only).", file=sys.stderr)
        return 2

    train_by_key = {}
    if args.train_file:
        for t in origin_check.load_jsonl(args.train_file):
            if not t.get("key") or not isinstance(t.get("prompt"), str) or not t.get("ground_truth"):
                raise SystemExit(f"FATAL: train row without key/prompt/ground_truth in {args.train_file}")
            train_by_key[t["key"]] = t
    samples_path = args.samples
    if not samples_path:
        sibling = os.path.join(os.path.dirname(os.path.abspath(args.generations)), "glimmer_samples.jsonl")
        if os.path.isfile(sibling):
            samples_path = sibling
    samples_by_key = None
    if samples_path:
        if not os.path.isfile(samples_path):
            raise SystemExit(f"FATAL: samples file not found: {samples_path}. "
                             "generate.py must record glimmer_samples.jsonl before filtering.")
        samples_by_key = collections.defaultdict(list)
        for s in origin_check.load_jsonl(samples_path):
            samples_by_key[s.get("key")].append(s)

    drop = collections.Counter()
    per_prompt = collections.defaultdict(list)   # key -> [candidate records]
    meta = {}                                    # key -> {prompt, family, instruction_ids}
    fam = collections.defaultdict(lambda: collections.Counter())
    seen_hash = set()
    n = 0
    with open(args.generations) as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            r = json.loads(line)
            key = r["key"]
            meta.setdefault(key, r)
            fam[r.get("family") or "?"]["samples"] += 1
            content = r.get("content")  # exact recorded text; never stripped or otherwise altered
            if r.get("error"):
                drop["request_error"] += 1; continue
            if r.get("finish_reason") == "length":
                drop["truncated_length"] += 1; continue
            if not isinstance(content, str) or not content:
                drop["empty_content"] += 1; continue
            prompt, ground_truth = r["prompt"], r["ground_truth"]
            if train_by_key:
                trow = train_by_key.get(key)
                if trow is None:
                    drop["key_not_in_train"] += 1; continue
                prompt, ground_truth = trow["prompt"], trow["ground_truth"]
            reasoning = r.get("reasoning")
            if samples_by_key is not None:
                cands = [s for s in samples_by_key.get(key, [])
                         if s.get("model") == "muse-glimmer" and s.get("finish_reason") != "length"
                         and s.get("answer") == content]
                if not cands:
                    drop["not_a_recorded_sample"] += 1; continue
                cand = next((s for s in cands if s.get("reasoning") == reasoning), cands[0])
                reasoning = cand.get("reasoning"); r["_origin_reasoning"] = reasoning
            checks = verifier.verify_constraints(content, ground_truth)
            if not checks:
                drop["no_checks_returned"] += 1; continue  # vacuous pass is not a pass
            if not all(checks):
                drop["constraint_failed"] += 1; continue
            fam[r.get("family") or "?"]["passed"] += 1
            extra = len(reasoning or "") if args.keep_reasoning else 0
            if len(prompt) + len(content) + extra > args.max_chars:
                drop["too_long_for_sequence_len"] += 1; continue  # dropped whole, never truncated to fit
            h = (key, hashlib.sha256(content.encode()).hexdigest())
            if h in seen_hash:
                drop["exact_duplicate"] += 1; continue
            seen_hash.add(h)
            per_prompt[key].append(r)

    # Shortest passing answer first within a prompt (cheapest to train, least rambling),
    # then round-robin across prompts so an overall cap keeps prompt breadth.
    for key in per_prompt:
        per_prompt[key].sort(key=lambda r: (len(r["content"]), r["sample_idx"]))
        per_prompt[key] = per_prompt[key][: args.max_keep_per_prompt]
    kept, tier = [], 0
    keys = sorted(per_prompt)
    while True:
        added = [per_prompt[k][tier] for k in keys if tier < len(per_prompt[k])]
        if not added:
            break
        kept += added; tier += 1
    if args.max_train_examples > 0:
        kept = kept[: args.max_train_examples]
    random.Random(args.seed).shuffle(kept)

    with open(args.out, "w") as f:
        for r in kept:
            prompt = train_by_key[r["key"]]["prompt"] if train_by_key else r["prompt"]  # exact approved prompt
            assistant = {"role": "assistant", "content": r["content"]}
            reasoning = r.get("_origin_reasoning", r.get("reasoning"))
            if args.keep_reasoning and reasoning:
                assistant["reasoning_content"] = reasoning
            f.write(json.dumps({"messages": [{"role": "user", "content": prompt}, assistant]}) + "\n")

    fam_out = {}
    prompts_with_pass = {k for k, v in per_prompt.items() if v}
    kept_keys = {r["key"] for r in kept}
    for name, c in sorted(fam.items()):
        fam_out[name] = {"samples": c["samples"], "passed": c["passed"],
                         "pass_rate": round(c["passed"] / c["samples"], 3) if c["samples"] else 0}
    stats = {"samples_in": n, "dropped": dict(drop), "prompts_total": len(meta),
             "prompts_with_any_pass": len(prompts_with_pass), "prompts_kept": len(kept_keys),
             "sft_examples": len(kept), "by_family": fam_out,
             "verifier_error_counts": verifier.error_counts(),
             "keep_reasoning": args.keep_reasoning,
             "train_authority": bool(train_by_key), "samples_authority": samples_path or None,
             "max_keep_per_prompt": args.max_keep_per_prompt}
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[build_sft] {n} samples -> {len(kept)} SFT examples from {len(kept_keys)}/{len(meta)} prompts; "
          f"dropped={dict(drop)}; verifier_error_counts={stats['verifier_error_counts']}", flush=True)
    if stats["verifier_error_counts"]:
        print("[build_sft] WARNING: the verifier swallowed checker exceptions (see stats). "
              "Some pass/fail calls may be wrong; inspect before trusting this yield.", file=sys.stderr)

    if len(kept) == 0:
        print("FATAL: zero yield - no sample passed every constraint. Refusing to train an empty adapter. "
              f"Drop reasons: {dict(drop)}. Inspect generations.jsonl: if most drops are truncated_length, "
              "raise MAX_TOKENS; if constraint_failed, raise SAMPLES_PER_PROMPT or check the verifier.",
              file=sys.stderr)
        return 3
    if len(kept) < args.min_examples:
        print(f"FATAL: only {len(kept)} verified examples (< MIN_EXAMPLES={args.min_examples}). "
              "Training on this little data is unlikely to help and wastes the training stage. "
              "Raise SAMPLES_PER_PROMPT / PROMPT_LIMIT / generation time box, or lower MIN_EXAMPLES deliberately.",
              file=sys.stderr)
        return 3
    return 0

if __name__ == "__main__":
    sys.exit(main())
