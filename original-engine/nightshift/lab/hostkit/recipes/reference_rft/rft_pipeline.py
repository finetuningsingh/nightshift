"""The reference method (rejection-sampling fine-tuning) as two small steps that need only the standard library.

    python3 rft_pipeline.py generate --prompts TRAIN.jsonl --out samples.jsonl [--base-url URL] [--n-prompts N] [--samples K] ...
    python3 rft_pipeline.py filter   --samples samples.jsonl --prompts TRAIN.jsonl --rewards REWARDS.py --out sft.jsonl --stats stats.json
    python3 rft_pipeline.py timing   --key generation_s --seconds 123.4

``generate`` samples K answers per prompt from the BASE model served by vLLM (``glimmer-vllm-up``), with the same decoding settings and the same
explicit system message the scorer uses. ``filter`` keeps the answers that pass EVERY constraint under the pinned IFEvalG checkers (through
``rewards.py``), drops truncated and errored generations, keeps at most M distinct answers per prompt, and writes chat-format SFT rows.
Samples come from Glimmer itself, so the method needs no hosted-API-generated training data. It trains on public train-split prompts only.

UNVERIFIED on a GPU: vLLM's extra body fields (top_k, chat_template_kwargs), the response field that carries the reasoning, and that Axolotl
accepts ``reasoning_content``. The measured yield (``stats.json``), memory and cost are what the lead records before v0 is frozen.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

SYSTEM_PROMPT = "You are a helpful AI assistant."  # same as GenerationParams.recommended: an explicit system message, no injected date
DEFAULT_URL = "http://127.0.0.1:8000"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise SystemExit(f"{path}:{n}: not JSON ({exc})")
    return rows


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def post(url, body, timeout, retries=3):
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            time.sleep(min(10, 2 ** attempt))
    raise RuntimeError(f"request failed after {retries} tries: {last}")


def one_prompt(base_url, model, row, args):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": args.system}, {"role": "user", "content": row["prompt"]}],
        "n": args.samples, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens,
        "seed": args.seed, "chat_template_kwargs": {"reasoning_strength": args.reasoning_strength},
    }  # fmt: skip
    out = []
    try:
        resp = post(base_url.rstrip("/") + "/v1/chat/completions", body, args.timeout)
        for i, ch in enumerate(resp["choices"]):
            msg = ch.get("message") or {}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            out.append({
                "key": row["key"], "sample": i, "text": msg.get("content") or "", "reasoning": reasoning,
                "finish_reason": ch.get("finish_reason") or "", "completion_tokens": int((resp.get("usage") or {}).get("completion_tokens", 0)),
                "error": None,
            })  # fmt: skip
    except Exception as exc:  # noqa: BLE001 - a per-prompt failure is recorded, not fatal
        out.append({"key": row["key"], "sample": 0, "text": "", "reasoning": "", "finish_reason": "", "completion_tokens": 0, "error": str(exc)[:300]})
    return out


def cmd_generate(a):
    rows = [r for r in read_jsonl(a.prompts) if isinstance(r.get("prompt"), str) and r.get("key") is not None]
    if not rows:
        raise SystemExit("no usable prompts (each row needs 'key' and 'prompt')")
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n_prompts]
    t0 = time.time()
    samples = []
    # Each prompt's samples are appended to the output as soon as that prompt finishes, so a run stopped by a time limit keeps what it has.
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool, open(a.out, "w", encoding="utf-8") as fh:
        futures = [pool.submit(one_prompt, a.base_url, a.model, r, a) for r in rows]
        for fut in concurrent.futures.as_completed(futures):
            part = fut.result()
            samples.extend(part)
            for sample in part:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fh.flush()
    add_timing("generation_s", time.time() - t0)
    errors = sum(1 for s in samples if s["error"])
    print(f"generated {len(samples)} samples for {len(rows)} prompts in {time.time() - t0:.0f} s ({errors} errors)")
    if errors == len(samples):
        raise SystemExit("every generation failed")


def load_checker(path):
    spec = importlib.util.spec_from_file_location("lab_rewards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def passes_all(mod, text, ground_truth):
    """True iff every constraint of the row passes (the headline 'prompt_all' criterion)."""
    if hasattr(mod, "verify_constraints"):
        flags = mod.verify_constraints(text, ground_truth)
        return bool(flags) and all(flags)
    return float(mod.reward(text, ground_truth, "all")) >= 1.0


def cmd_filter(a):
    mod = load_checker(a.rewards)
    if hasattr(mod, "preflight"):
        mod.preflight()
    prompts = {r["key"]: r for r in read_jsonl(a.prompts)}
    samples = read_jsonl(a.samples)
    kept, per_prompt, stats = [], {}, {"n_samples": len(samples), "n_pass": 0, "n_truncated": 0, "n_errors": 0, "n_empty": 0}
    for s in samples:
        row = prompts.get(s["key"])
        if s.get("error"):
            stats["n_errors"] += 1
            continue
        if row is None:
            continue
        if s["finish_reason"] == "length":
            stats["n_truncated"] += 1
            continue
        if not s["text"].strip():
            stats["n_empty"] += 1
            continue
        if not passes_all(mod, s["text"], row["ground_truth"]):
            continue
        stats["n_pass"] += 1
        seen = per_prompt.setdefault(s["key"], set())
        if s["text"] in seen or len(seen) >= a.keep_per_prompt:
            continue
        seen.add(s["text"])
        assistant = {"role": "assistant", "content": s["text"]}
        if a.include_reasoning and s.get("reasoning"):
            assistant["reasoning_content"] = s["reasoning"]
        kept.append({"messages": [{"role": "system", "content": a.system}, {"role": "user", "content": row["prompt"]}, assistant]})
    stats.update(
        n_kept=len(kept), n_prompts_sampled=len({s["key"] for s in samples}), n_prompts_with_pass=len(per_prompt),
        yield_fraction=round(stats["n_pass"] / max(1, stats["n_samples"]), 6),
    )  # fmt: skip
    if hasattr(mod, "error_counts"):
        stats["checker_errors"] = mod.error_counts()
    write_jsonl(a.out, kept)
    with open(a.stats, "w") as fh:
        json.dump(stats, fh, indent=1, sort_keys=True)
    print(json.dumps(stats, sort_keys=True))
    if not kept:
        raise SystemExit("no sample passed the checkers: nothing to train on")


def add_timing(key, seconds):
    out = os.environ.get("GLIMMER_OUT", ".")
    path = os.path.join(out, "timings.json")
    try:
        doc = json.load(open(path))
    except (OSError, ValueError):
        doc = {}
    doc[key] = round(doc.get(key, 0.0) + float(seconds), 3)
    os.makedirs(out, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--base-url", default=os.environ.get("RFT_BASE_URL", DEFAULT_URL))
    g.add_argument("--model", default="muse-glimmer")
    g.add_argument("--n-prompts", type=int, default=int(os.environ.get("RFT_N_PROMPTS", "500")))
    g.add_argument("--samples", type=int, default=int(os.environ.get("RFT_SAMPLES", "4")))
    g.add_argument("--max-tokens", type=int, default=int(os.environ.get("RFT_MAX_TOKENS", "6000")))
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=64)
    g.add_argument("--reasoning-strength", default=os.environ.get("RFT_REASONING", "high"), choices=["low", "medium", "high", "xhigh"])
    g.add_argument("--system", default=SYSTEM_PROMPT)
    g.add_argument("--seed", type=int, default=int(os.environ.get("RFT_SEED", "42")))
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--timeout", type=float, default=1800.0)
    f = sub.add_parser("filter")
    f.add_argument("--samples", required=True)
    f.add_argument("--prompts", required=True)
    f.add_argument("--rewards", default=os.environ.get("REWARDS_PY", "/opt/lab/rewards.py"))
    f.add_argument("--out", required=True)
    f.add_argument("--stats", required=True)
    f.add_argument("--keep-per-prompt", type=int, default=int(os.environ.get("RFT_KEEP_PER_PROMPT", "2")))
    f.add_argument("--system", default=SYSTEM_PROMPT)
    f.add_argument("--include-reasoning", type=int, default=int(os.environ.get("RFT_INCLUDE_REASONING", "1")))
    t = sub.add_parser("timing")
    t.add_argument("--key", required=True)
    t.add_argument("--seconds", type=float, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        cmd_generate(a)
    elif a.cmd == "filter":
        cmd_filter(a)
    else:
        add_timing(a.key, a.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
