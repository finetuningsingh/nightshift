#!/usr/bin/env python3
"""Local training-data origin check (harness rules 2-5), standard library only.

The harness re-checks the file passed to glimmer-axolotl-train inside that
command, before registering data or training; this is the same check run
locally so violations surface before any GPU/training step:

2. Each SFT row is {"messages": [...]}: roles from system/user/assistant
   only, any number of leading system messages, then exactly one user and
   exactly one assistant message. Other roles or extra turns are refused.
3. The user content is byte-for-byte the `prompt` of an approved train row.
4. The assistant content exactly equals the `answer` of a recorded sample
   for that row's key, with "model": "muse-glimmer" and a finish_reason
   that is not "length".
5. A reasoning_content/reasoning field on the assistant message, if present,
   exactly equals that sample's `reasoning`. Omitting it is allowed.

All comparisons are exact equality -- the recorded answer/reasoning are used
unmodified, so even trimming one trailing space is a violation. (Rule 6,
the verifier re-check, and rule 7, the origin report, are the harness's;
build_sft.py already applies the verifier when selecting samples.)
"""
import argparse, json, sys


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"FATAL: {path}:{i}: not valid JSON: {e}")
    return rows


def check_sft_origin(sft_rows, train_rows, sample_rows):
    """Return a list of violation strings; empty means every row passes
    rules 2-5. Inputs are parsed JSONL rows (SFT file, approved train
    file, glimmer_samples.jsonl)."""
    keys_by_prompt = {}
    for r in train_rows:
        keys_by_prompt.setdefault(r.get("prompt"), []).append(r.get("key"))
    samples_by_key = {}
    for s in sample_rows:
        samples_by_key.setdefault(s.get("key"), []).append(s)

    errors = []
    for i, row in enumerate(sft_rows, 1):
        msgs = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(msgs, list) or len(msgs) < 2:
            errors.append(f"row {i}: not a {{\"messages\": [...]}} row with user+assistant turns")
            continue
        roles = [m.get("role") if isinstance(m, dict) else None for m in msgs]
        lead, tail = roles[:-2], roles[-2:]
        if tail != ["user", "assistant"] or any(r != "system" for r in lead):
            errors.append(f"row {i}: roles {roles} are not (system)* user assistant "
                          "(other roles / extra turns are refused)")
            continue
        user, assistant = msgs[-2], msgs[-1]
        keys = keys_by_prompt.get(user.get("content"))
        if not keys:
            errors.append(f"row {i}: user content is not byte-for-byte an approved train prompt")
            continue
        candidates = [s for k in keys for s in samples_by_key.get(k, [])
                      if s.get("model") == "muse-glimmer"
                      and s.get("finish_reason") != "length"
                      and s.get("answer") == assistant.get("content")]
        if not candidates:
            errors.append(f"row {i}: assistant content is not the exact answer of a recorded "
                          f"muse-glimmer sample (finish_reason != length) for key(s) {keys}")
            continue
        given = [assistant[f] for f in ("reasoning_content", "reasoning") if f in assistant]
        if given and not any(all(s.get("reasoning") == g for g in given) for s in candidates):
            errors.append(f"row {i}: assistant reasoning field does not exactly equal "
                          "the recorded sample's reasoning")
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sft", required=True, help="SFT JSONL that will be passed to glimmer-axolotl-train")
    p.add_argument("--train", required=True, help="approved train.jsonl (the harness uses /work/data/train.jsonl)")
    p.add_argument("--samples", required=True, help="glimmer_samples.jsonl recorded by generate.py")
    args = p.parse_args(argv)
    sft_rows = load_jsonl(args.sft)
    errors = check_sft_origin(sft_rows, load_jsonl(args.train), load_jsonl(args.samples))
    if errors:
        for e in errors:
            print(f"ORIGIN CHECK FAILED: {e}", file=sys.stderr)
        print(f"FATAL: {len(errors)} training-data origin violation(s) in {args.sft}. "
              "The harness would refuse this file inside glimmer-axolotl-train; "
              "fix the data, do not edit recorded samples by hand.", file=sys.stderr)
        return 3
    print(f"[origin_check] OK: {len(sft_rows)} rows are exact recorded muse-glimmer samples "
          f"for approved prompts ({args.sft})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
