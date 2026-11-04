#!/usr/bin/env python3
"""Run the cheap live probes against the Meta Model API (or any Chat Completions endpoint) and write a JSON report.

    MODEL_API_KEY=... python3 scripts/live_smoke.py [--model muse-spark-1.3] [--base-url URL] [--out demo/live_probe.json]

The report holds pass/fail per probe, models, latency and whether cached-token counts are reported. It never contains the
API key or any prompt or reply text. Exit status: 0 all required probes passed, 1 a required probe failed, 2 setup error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nightshift.models.base import ModelConfigError  # noqa: E402
from nightshift.models.live_smoke import WONDER_MODEL, build_client_factory, run_smoke, summarize  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cheap live probes of a Chat Completions endpoint.")
    parser.add_argument("--model", help="default model (else NIGHTSHIFT_MODEL, else muse-spark-1.3)")
    parser.add_argument("--base-url", help="endpoint base URL (else NIGHTSHIFT_BASE_URL, else https://api.meta.ai/v1)")
    parser.add_argument("--out", default="demo/live_probe.json", help="where to write the JSON report")
    parser.add_argument("--wonder-model", default=WONDER_MODEL, help="model used for the informational Wonder probe")
    parser.add_argument("--no-wonder", action="store_true", help="skip the Wonder probe")
    return parser


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None, stdout: TextIO | None = None,
         stderr: TextIO | None = None) -> int:
    out, err = stdout or sys.stdout, stderr or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        factory = build_client_factory(model=args.model, base_url=args.base_url, env=env)
    except ModelConfigError as exc:
        print(f"error: {exc}", file=err)
        return 2
    report = run_smoke(factory, args.out, probe_wonder=not args.no_wonder, wonder_model=args.wonder_model)
    for line in summarize(report):
        print(line, file=out)
    print(f"report written to {args.out}", file=out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
