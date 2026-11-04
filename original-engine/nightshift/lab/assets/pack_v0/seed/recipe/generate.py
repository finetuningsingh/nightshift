#!/usr/bin/env python3
"""Sample answers from the served base model for a deterministic subset of the training prompts. Standard library only.

    python3 generate.py --wait-only --base-url http://127.0.0.1:8000/v1 --wait-seconds 1500
    python3 generate.py --data /work/data/train.jsonl --base-url ... --out samples.jsonl --n-prompts 400 --samples 4

Output rows: {"key", "prompt", "ground_truth", "response", "reasoning", "finish_reason"}; only complete answers
(finish_reason == "stop") with non-empty text are written.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

SYSTEM = "You are a helpful AI assistant."  # an explicit system message keeps the rendered prompt identical from day to day


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick(rows, n):
    """The first ``n`` rows in a fixed pseudo-random order (by hash of the row key), so every run picks the same prompts."""
    ordered = sorted(rows, key=lambda r: hashlib.sha256(str(r["key"]).encode("utf-8")).hexdigest())
    return ordered[:n]


def wait_ready(base_url, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(10)
    return False


def request_body(model, prompt, max_tokens, seed):
    return {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": max_tokens,
        "seed": seed,
    }


def complete(base_url, body, timeout=900):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sample_one(base_url, model, row, index, max_tokens):
    seed = int(hashlib.sha256(f"{row['key']}:{index}".encode("utf-8")).hexdigest()[:8], 16)
    try:
        data = complete(base_url, request_body(model, row["prompt"], max_tokens, seed))
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        finish = choice.get("finish_reason")
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as exc:
        print(f"sample failed for {row['key']}#{index}: {type(exc).__name__}", file=sys.stderr)
        return None
    if finish != "stop" or not text.strip():
        return None
    return {
        "key": row["key"],
        "prompt": row["prompt"],
        "ground_truth": row["ground_truth"],
        "response": text,
        "reasoning": reasoning,
        "finish_reason": finish,
    }


def served_model(base_url):
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))["data"][0]["id"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--wait-only", action="store_true")
    ap.add_argument("--wait-seconds", type=int, default=1200)
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--n-prompts", type=int, default=400)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args(argv)
    if args.wait_only:
        if not wait_ready(args.base_url, args.wait_seconds):
            print("the model server did not become ready in time", file=sys.stderr)
            return 1
        return 0
    if not args.data or not args.out:
        ap.error("--data and --out are required")
    rows = pick(load_rows(args.data), args.n_prompts)
    model = served_model(args.base_url)
    jobs = [(row, i) for row in rows for i in range(args.samples)]
    kept = 0
    with open(args.out, "w", encoding="utf-8") as out, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(sample_one, args.base_url, model, row, i, args.max_tokens) for row, i in jobs]
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            if rec is not None:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    print(json.dumps({"prompts": len(rows), "requested": len(jobs), "complete_answers": kept}))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
