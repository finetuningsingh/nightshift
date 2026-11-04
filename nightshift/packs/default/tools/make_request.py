#!/usr/bin/env python3
"""Write one researcher request JSON into ./outbox (pack helper tool)."""

import argparse
import json
import sys
import uuid
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_json", help="request object as a JSON string")
    parser.add_argument("--outbox", default="outbox")
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request_json)
    except json.JSONDecodeError as exc:
        print(f"invalid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(request, dict) or "type" not in request:
        print("request must be a JSON object with a type field", file=sys.stderr)
        return 2
    request.setdefault("request_id", f"{request['type']}-{uuid.uuid4().hex[:8]}")
    outbox = Path(args.outbox)
    outbox.mkdir(parents=True, exist_ok=True)
    target = outbox / f"{request['request_id']}.json"
    target.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
