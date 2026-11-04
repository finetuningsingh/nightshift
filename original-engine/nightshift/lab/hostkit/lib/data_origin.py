"""Origin check for training data: every assistant target must be a verifier-passing sample of the local base model. ``python3 -I``; stdlib only.

    python3 -I data_origin.py check DATA.jsonl

Run by ``bin/glimmer-axolotl-train`` BEFORE the data is registered or any training starts. Rule (owner, 2026-11-03): answers written by Claude,
Muse models, other hosted agents, fixtures or by hand never become Glimmer's training targets; targets are samples that the local Glimmer model
produced for the approved public training prompts and that the pinned verifier accepts.

Inputs (all inside the job): ``$LAB_WORK/data/train.jsonl`` (the approved prompts, with ``key``, ``prompt``, ``ground_truth``);
``$GLIMMER_OUT/glimmer_samples.jsonl`` written by the recipe's generation step, one object per line:
``{"key": <train key>, "model": "muse-glimmer", "answer": str, "reasoning": str or null, "finish_reason": str}``; the data file itself.

A training row must be ``[system]* user assistant`` (no other roles, one user message, one assistant message) where
  * the user message equals the prompt of some approved train row (exactly);
  * the assistant ``content`` equals the ``answer`` of a recorded sample for that same key, whose ``model`` is ``muse-glimmer`` and whose
    ``finish_reason`` is not ``length``;
  * a ``reasoning_content`` / ``reasoning`` field on the assistant message, if present, equals that sample's ``reasoning``;
  * the verifier (``$REWARDS_PY``, the pinned checkers) accepts the answer on every constraint of that row.
Any other row stops the job before training. The check cannot tell whether a recipe wrote a false samples file: it records the hashes of the
train file, the samples file and the dataset so that the chain from approved prompt to dataset is traceable, and it catches hand-written,
fixture, or other-model targets and any use of non-approved prompts. A summary line goes to the job log and ``origin_report.json`` is appended
to in ``$GLIMMER_OUT``.
"""
import hashlib
import importlib.util
import json
import os
import sys

TRAIN_MODEL = "muse-glimmer"
SHOW = 5


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append((n, json.loads(line)))
                except ValueError as exc:
                    raise SystemExit(f"{path} line {n}: not JSON ({exc})")
    return rows


def _load_verifier(path):
    spec = importlib.util.spec_from_file_location("lab_rewards", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the verifier module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.preflight()
    return mod


def check(data, work, out, rewards_py):
    train_path = os.path.join(work, "data", "train.jsonl")
    samples_path = os.path.join(out, "glimmer_samples.jsonl")
    for p, what in ((train_path, "the approved training prompts"), (samples_path, "the recorded Glimmer samples (glimmer_samples.jsonl)"), (data, "the data file")):
        if os.path.islink(p) or not os.path.isfile(p):
            raise SystemExit(f"origin of the training data cannot be established: {what} is missing at {p}")
    train = {}
    for n, row in _jsonl(train_path):
        if isinstance(row, dict) and isinstance(row.get("key"), str):
            train[row["key"]] = row
    by_prompt = {}
    for key, row in train.items():
        by_prompt.setdefault(row.get("prompt"), []).append(key)
    samples = {}
    for n, s in _jsonl(samples_path):
        if isinstance(s, dict) and isinstance(s.get("key"), str) and isinstance(s.get("answer"), str):
            samples.setdefault(s["key"], []).append(s)
    verifier = _load_verifier(rewards_py)
    bad, rows = [], 0
    for n, row in _jsonl(data):
        rows += 1
        why = None
        msgs = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(msgs, list):
            why = "no messages list"
        else:
            users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            asst = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
            other = [m for m in msgs if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")]
            if other or len(users) != 1 or len(asst) != 1:
                why = "a row must be system* user assistant with one user and one assistant message"
            else:
                keys = by_prompt.get(users[0].get("content"), [])
                if not keys:
                    why = "the user message is not an approved training prompt"
                else:
                    reasoning = asst[0].get("reasoning_content", asst[0].get("reasoning"))
                    ok = False
                    for key in keys:
                        for s in samples.get(key, []):
                            if s.get("model") != TRAIN_MODEL or s.get("finish_reason") == "length" or s["answer"] != asst[0].get("content"):
                                continue
                            if reasoning is not None and reasoning != s.get("reasoning"):
                                continue
                            if all(verifier.verify_constraints(s["answer"], train[key].get("ground_truth"))):
                                ok = True
                                break
                        if ok:
                            break
                    if not ok:
                        why = "the assistant target is not a verifier-passing, untruncated sample of the local Glimmer model recorded for this prompt"
        if why:
            bad.append((n, why))
    if rows == 0:
        raise SystemExit("origin of the training data cannot be established: the data file has no rows")
    if bad:
        lines = "; ".join(f"row {n}: {w}" for n, w in bad[:SHOW])
        raise SystemExit(f"training stopped: {len(bad)} of {rows} rows do not come from the local Glimmer model on approved prompts ({lines})")
    report = {"rows": rows, "data_sha256": _sha(data), "samples_sha256": _sha(samples_path), "train_file_sha256": _sha(train_path),
              "rewards_py_sha256": _sha(rewards_py), "samples_recorded": sum(len(v) for v in samples.values())}
    path = os.path.join(out, "origin_report.json")
    prev = []
    if os.path.isfile(path):
        try:
            prev = json.load(open(path, encoding="utf-8"))
        except ValueError:
            prev = []
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(prev + [report], fh)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "check":
        raise SystemExit("usage: data_origin.py check DATA.jsonl")
    work = os.environ.get("LAB_WORK", "/work")
    r = check(sys.argv[2], work, os.environ.get("GLIMMER_OUT", os.path.join(work, "out")),
              os.environ.get("REWARDS_PY", os.path.join(os.environ.get("LAB_OPT", "/opt/lab"), "rewards.py")))
    print("origin check passed: " + json.dumps(r, sort_keys=True))
