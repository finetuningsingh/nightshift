#!/usr/bin/env python3
"""Sample answers from the base model (vLLM, OpenAI chat-completions API).

Standard library only. One HTTP request per sample, bounded concurrency,
a wall-clock time box, and each finished sample is appended to the output
JSONL immediately (single writer = main thread), so a killed job keeps
whatever completed.

Task order is sample-major: sample 0 for every prompt, then sample 1, ...
If the time box cuts generation short, the prompts that were attempted are
a breadth-first, shuffled subset rather than the first families in the file.

Origin rule: alongside the richer --out file, every successful server
response is also recorded in --samples-out (glimmer_samples.jsonl) as
{"key", "model", "answer", "reasoning", "finish_reason"}, with the answer
and reasoning exactly as returned -- never stripped, normalised or
rewritten. That file is the authority the harness checks training targets
against; errored requests produced no server text and are never recorded.
Both files are written as samples complete, so a time-boxed run keeps what
finished.
"""
import argparse, json, os, random, sys, time, urllib.error, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

def log(msg):
    print(f"[generate {time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_prompts(path, limit, seed):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"FATAL: {path}:{i}: not valid JSON: {e}")
            for field in ("key", "prompt", "ground_truth"):
                if not r.get(field):
                    raise SystemExit(f"FATAL: {path}:{i}: missing/empty '{field}' (key={r.get('key')!r})")
            rows.append(r)
    if not rows:
        raise SystemExit(f"FATAL: no prompts in {path}")
    random.Random(seed).shuffle(rows)  # a limit must not mean "first family only"
    return rows[:limit] if limit > 0 else rows

def http_json(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def extract(choice):
    # Exact server values, unmodified: no stripping or "" coercion, so what
    # lands in glimmer_samples.jsonl is byte-identical to the response.
    # Absent reasoning (neither field, or null) is recorded as None.
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    return msg.get("content"), reasoning, choice.get("finish_reason")

def one_sample(args, row, sample_idx):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "seed": args.seed + 100000 * sample_idx + zlib.crc32(row["key"].encode()) % 100000,
    }
    if args.top_k > 0:
        payload["top_k"] = args.top_k
    if args.reasoning_strength:
        payload["chat_template_kwargs"] = {"reasoning_strength": args.reasoning_strength}
    t0, last_err = time.time(), None
    for attempt in range(3):  # 1 try + 2 retries on transient (5xx / connection) errors only
        try:
            r = http_json(args.base_url.rstrip("/") + "/v1/chat/completions", payload, args.request_timeout)
            content, reasoning, finish = extract(r["choices"][0])
            return {**base(row, sample_idx), "model": args.model,
                    "content": content, "reasoning": reasoning,
                    "finish_reason": finish, "usage": r.get("usage") or {},
                    "error": "", "latency_s": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code < 500:
                break  # a 4xx (bad request / too long) will not fix itself
        except Exception as e:  # URLError, timeout, malformed JSON, missing choices
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    return {**base(row, sample_idx), "model": args.model, "content": "", "reasoning": None,
            "finish_reason": "", "usage": {}, "error": last_err or "unknown",
            "latency_s": round(time.time() - t0, 2)}

def base(row, sample_idx):
    return {"key": row["key"], "prompt": row["prompt"], "ground_truth": row["ground_truth"],
            "instruction_ids": row.get("instruction_ids") or [], "family": row.get("family") or "",
            "sample_idx": sample_idx}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples-out", default="",
                   help="glimmer_samples.jsonl (origin authority); default: next to --out")
    p.add_argument("--stats", required=True)
    p.add_argument("--model", default="muse-glimmer")
    p.add_argument("--prompt-limit", type=int, default=250)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0, help="0 = do not send top_k")
    p.add_argument("--reasoning-strength", default="high", help="empty string = omit")
    p.add_argument("--timebox-seconds", type=int, default=2700, help="0 = no time box")
    p.add_argument("--grace-seconds", type=int, default=180, help="after the time box, wait this long for in-flight requests")
    p.add_argument("--request-timeout", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="skip (key, sample_idx) already in --out")
    p.add_argument("--exit-hard-on-abandon", action="store_true",
                   help="os._exit after flushing if requests were abandoned (run.sh passes this)")
    args = p.parse_args(argv)

    try:  # fail before burning any time if the server is not there
        http_json(args.base_url.rstrip("/") + "/v1/models", timeout=15)
    except Exception as e:
        print(f"FATAL: vLLM server unreachable at {args.base_url} ({type(e).__name__}: {e}). "
              f"Check $GLIMMER_OUT/vllm.log and that glimmer-vllm-up printed this base URL.", file=sys.stderr)
        return 2

    rows = load_prompts(args.train_file, args.prompt_limit, args.seed)
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line); done.add((r["key"], r["sample_idx"]))
                except Exception:
                    pass
        log(f"resume: {len(done)} samples already in {args.out}")
    tasks = [(r, s) for s in range(args.samples) for r in rows if (r["key"], s) not in done]
    if not tasks and done:
        stats = {"prompts": len(rows), "requested": 0, "completed": 0, "errors": 0, "length": 0,
                 "abandoned": 0, "not_submitted": 0, "timed_out": False, "elapsed_s": 0.0,
                 "resumed_all_present": len(done)}
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)
        log(f"resume: all {len(done)} samples already present in {args.out}; nothing to generate")
        return 0
    log(f"{len(rows)} prompts x {args.samples} samples = {len(tasks)} requests, "
        f"concurrency={args.concurrency}, timebox={'off' if not args.timebox_seconds else f'{args.timebox_seconds}s'}")

    start = time.time()
    deadline = start + args.timebox_seconds if args.timebox_seconds > 0 else float("inf")
    counts = {"completed": 0, "errors": 0, "length": 0, "abandoned": 0, "not_submitted": 0,
              "samples_recorded": 0}
    timed_out = False
    mode = "a" if args.resume else "w"
    samples_path = args.samples_out or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                    "glimmer_samples.jsonl")
    ex = ThreadPoolExecutor(max_workers=args.concurrency)  # not a context manager: its __exit__ waits for abandoned requests
    with open(args.out, mode) as out, open(samples_path, mode) as samples_out:
        it = iter(tasks)
        inflight = {}
        def submit_more():
            while len(inflight) < args.concurrency and time.time() < deadline:
                try:
                    r, s = next(it)
                except StopIteration:
                    return
                inflight[ex.submit(one_sample, args, r, s)] = (r["key"], s)
        submit_more()
        while inflight:
            now = time.time()
            if now >= deadline:
                timed_out = True
            wait_s = 1.0 if not timed_out else max(0.0, min(1.0, deadline + args.grace_seconds - now))
            finished, _ = wait(inflight, timeout=wait_s, return_when=FIRST_COMPLETED)
            for fut in finished:
                inflight.pop(fut)
                rec = fut.result()
                out.write(json.dumps(rec) + "\n"); out.flush()  # survive a hard kill of the machine
                if not rec["error"]:  # only real server responses enter the origin record
                    samples_out.write(json.dumps({
                        "key": rec["key"], "model": rec["model"], "answer": rec["content"],
                        "reasoning": rec["reasoning"], "finish_reason": rec["finish_reason"]}) + "\n")
                    samples_out.flush()
                    counts["samples_recorded"] += 1
                counts["completed"] += 1
                counts["errors"] += bool(rec["error"])
                counts["length"] += rec["finish_reason"] == "length"
                if counts["completed"] % 50 == 0:
                    log(f"{counts['completed']}/{len(tasks)} done, {counts['errors']} errors, "
                        f"{counts['length']} truncated, {time.time()-start:.0f}s elapsed")
            if timed_out and time.time() > deadline + args.grace_seconds:
                counts["abandoned"] = len(inflight)
                for fut in inflight:
                    fut.cancel()
                inflight.clear()
                ex.shutdown(wait=False, cancel_futures=True)
                break
            submit_more()
        counts["not_submitted"] = len(tasks) - counts["completed"] - counts["abandoned"]
        stats = {"prompts": len(rows), "requested": len(tasks), **counts,
                 "timed_out": timed_out, "elapsed_s": round(time.time() - start, 1),
                 "settings": {"samples": args.samples, "max_tokens": args.max_tokens,
                              "temperature": args.temperature, "top_p": args.top_p,
                              "reasoning_strength": args.reasoning_strength}}
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)
        log(f"finished: {stats}")
        out.flush(); os.fsync(out.fileno())
        samples_out.flush(); os.fsync(samples_out.fileno())
    ex.shutdown(wait=False, cancel_futures=True)

    if counts["completed"] == 0:
        print("FATAL: generation produced zero completed samples. Nothing to train on; "
              "see $GLIMMER_OUT/vllm.log.", file=sys.stderr)
        return 3
    if counts["errors"] == counts["completed"]:
        print(f"FATAL: all {counts['completed']} generation requests errored "
              f"(first errors are in {args.out}). Refusing to continue.", file=sys.stderr)
        return 3
    if counts["abandoned"] and args.exit_hard_on_abandon:
        # In-flight urllib calls cannot be cancelled; results are flushed, so exit hard
        # rather than let worker threads hold the process (and the GPU) for a request timeout.
        log(f"{counts['abandoned']} requests abandoned at the time box; exiting hard after flush")
        os._exit(0)
    return 0

if __name__ == "__main__":
    sys.exit(main())
