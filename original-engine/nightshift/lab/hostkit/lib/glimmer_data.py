"""Training-data registration for the glimmer helper commands. ``python3 -I``; standard library only.

    python3 -I glimmer_data.py register DATA.jsonl OUT_DIR

Validates that every line is a JSON object with a non-empty ``messages`` list of ``{"role", "content"}`` objects (no top-level
``images``; Axolotl 0.19.0 keeps only the first), then APPENDS the file to ``OUT_DIR/train_data.jsonl``. That file is the processed
training dataset the control host scans for contamination before any selection score: a job that never calls a helper has none and fails.
"""
import json
import os
import sys

ROLES = {"system", "user", "assistant", "tool", "developer"}
MAX_LINES = 2_000_000


def validate_line(n, line):
    try:
        row = json.loads(line)
    except ValueError as exc:
        raise SystemExit(f"line {n}: not JSON ({exc})")
    if not isinstance(row, dict) or "images" in row:
        raise SystemExit(f"line {n}: expected an object without a top-level 'images'")
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise SystemExit(f"line {n}: 'messages' must be a non-empty list")
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content", ""), (str, list, type(None))):
            raise SystemExit(f"line {n}: each message needs a known 'role' and a 'content'")
    if not any(m.get("role") == "assistant" for m in msgs):
        raise SystemExit(f"line {n}: no assistant message to train on")
    return row


def register(data, out_dir):
    if os.path.islink(data) or not os.path.isfile(data):
        raise SystemExit(f"{data}: not a regular file")
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    with open(data, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            validate_line(n, line)
            lines.append(line if line.endswith("\n") else line + "\n")
            if n > MAX_LINES:
                raise SystemExit("too many lines")
    if not lines:
        raise SystemExit(f"{data}: no training rows")
    target = os.path.join(out_dir, "train_data.jsonl")
    with open(target, "a", encoding="utf-8") as out:
        out.writelines(lines)
    return len(lines)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "register":
        raise SystemExit("usage: glimmer_data.py register DATA.jsonl OUT_DIR")
    print(f"registered {register(sys.argv[2], sys.argv[3])} training rows")
