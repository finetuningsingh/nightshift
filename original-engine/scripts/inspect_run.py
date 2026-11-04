#!/usr/bin/env python3
"""Run lab evaluations through Inspect AI (see nightshift/lab/inspect_eval.py for the rules).

    inspect_run.py replay LAB_ROOT SCORE_ID        re-score the stored generations of a ledger score with the Inspect scorer and compare
                                                   every prompt with the ledger's per_prompt.jsonl (no model, no GPU, no cost)
    inspect_run.py run LAB_ROOT --assignment E1 --base-url URL --model NAME [--adapter-id ID] [--rental-id RID] [--limit N]
                                                   evaluate a served model on the frozen DEV split (OpenAI-compatible endpoint, e.g. vLLM)

Logs go to LAB_ROOT/inspect_logs/ (inspect_ai's .eval files); a run prints the log path and its sha256. Needs the pinned verifiers
(LAB_ROOT/third_party) and NLTK data (NLTK_DATA)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nightshift.lab import inspect_eval as ie  # noqa: E402
from nightshift.lab.contracts import AdapterRef, GenerationParams, LabLayout, read_record  # noqa: E402
from nightshift.lab.verifiers import IfEvalG  # noqa: E402


def make_backend(layout: LabLayout) -> IfEvalG:
    return IfEvalG(layout.third_party_dir / "open-instruct", nltk_data=os.environ.get("NLTK_DATA"))


def scorer_version(backend: IfEvalG) -> str:
    from nightshift.lab.contracts import scorer_version_of
    from nightshift.lab.scorer import ALL_METRICS, SCORER_MODULE_VERSION

    pins = {f"ifevalg.{k}": str(v) for k, v in backend.pins().items()}
    return scorer_version_of(pins, ALL_METRICS, SCORER_MODULE_VERSION)


def cmd_replay(args) -> int:
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import solver

    layout = LabLayout(Path(args.lab_root))
    sdir = layout.root / "scores" / args.score_id
    req = json.loads((sdir / "request.json").read_text())
    assignment = req["assignment_id"]
    gens = {}
    for line in (sdir / "gen_seed0.jsonl").read_text().splitlines():
        g = json.loads(line)
        gens[g["prompt_key"]] = g
    ledger = {}
    for line in (sdir / "per_prompt.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["seed"] == 0:
            ledger[r["prompt_key"]] = r

    @solver
    def replay():
        async def solve(state, generate):
            g = gens[state.sample_id]
            out = ModelOutput.from_content("replay", g["text"])
            out.choices[0].stop_reason = "max_tokens" if g["finish_reason"] == "length" else "stop"
            state.output = out
            return state
        return solve

    backend = make_backend(layout)
    params = GenerationParams.from_dict(req["params"])
    rows, digest = ie.load_dev_rows(layout, assignment)
    meta = ie.provenance(layout, assignment, params, dataset_sha256=digest, scorer_version=scorer_version(backend), pins=backend.pins(),
                         base_revision=req["base"]["revision"], adapter_sha256=(req.get("adapter") or {}).get("sha256"), rental_id=None, repo=REPO)
    task = ie.build_task(layout, assignment, backend, meta, params, solver=replay(), limit_note=f"replay of ledger score {args.score_id}")
    logdir = layout.root / "inspect_logs"
    logdir.mkdir(exist_ok=True)
    log = inspect_eval(task, model="mockllm/model", log_dir=str(logdir), display="none", metadata={"replay_of": args.score_id})[0]
    bad = 0
    for s in log.samples:
        v = next(iter(s.scores.values())).value
        l = ledger[s.id]
        if (bool(v["prompt_all"]) != bool(l["passed"])) or abs(v["fraction"] - l["fraction"]) > 1e-9 or bool(v["truncated"]) != bool(l["truncated"]):
            bad += 1
            print("MISMATCH", s.id, v, {k: l[k] for k in ("passed", "fraction", "truncated")})
    print(json.dumps({"status": log.status, "n": len(log.samples), "mismatches": bad, "log": log.location,
                      "log_sha256": hashlib.sha256(Path(log.location.replace("file://", "")).read_bytes()).hexdigest()}))
    return 1 if bad or log.status != "success" else 0


def cmd_run(args) -> int:
    from inspect_ai import eval as inspect_eval

    layout = LabLayout(Path(args.lab_root))
    backend = make_backend(layout)
    params = GenerationParams.recommended(args.max_tokens)
    rows, digest = ie.load_dev_rows(layout, args.assignment)
    adapter_sha = read_record(layout.adapter_json_path(args.adapter_id), AdapterRef).sha256 if args.adapter_id else None
    from nightshift.lab.contracts import LabConfig, loads_strict

    cfg = LabConfig.from_dict(loads_strict(layout.lab_config_path.read_bytes().decode("utf-8")))
    meta = ie.provenance(layout, args.assignment, params, dataset_sha256=digest, scorer_version=scorer_version(backend), pins=backend.pins(),
                         base_revision=cfg.base.revision, adapter_sha256=adapter_sha, rental_id=args.rental_id, repo=REPO)
    if meta["git"]["dirty"] and not args.allow_dirty:
        print("refusing: the working tree has uncommitted changes to tracked files; commit first (or --allow-dirty, recorded in the log)", file=sys.stderr)
        return 2
    task = ie.build_task(layout, args.assignment, backend, meta, params)
    logdir = layout.root / "inspect_logs"
    logdir.mkdir(exist_ok=True)
    os.environ.setdefault("VLLM_API_KEY", "none")
    log = inspect_eval(task, model=f"openai-api/vllm/{args.model}", model_base_url=args.base_url.rstrip("/") + "/v1", limit=args.limit, log_dir=str(logdir),
                       max_connections=args.workers, display="none", metadata={"arm": "adapter" if adapter_sha else "stock"})[0]
    path = Path(log.location.replace("file://", ""))
    results = {}
    if log.results is not None:
        for sc in log.results.scores:
            results[sc.name] = {k: m.value for k, m in sc.metrics.items()}
    print(json.dumps({"status": log.status, "error": (log.error.message[:300] if log.error else None), "log": str(path),
                      "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "results": results}, default=str))
    return 0 if log.status == "success" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay"); r.add_argument("lab_root"); r.add_argument("score_id")
    u = sub.add_parser("run"); u.add_argument("lab_root"); u.add_argument("--assignment", required=True); u.add_argument("--base-url", required=True)
    u.add_argument("--model", required=True); u.add_argument("--adapter-id"); u.add_argument("--rental-id"); u.add_argument("--limit", type=int)
    u.add_argument("--max-tokens", type=int, default=6000); u.add_argument("--workers", type=int, default=128); u.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    return cmd_replay(args) if args.cmd == "replay" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
