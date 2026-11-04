#!/usr/bin/env python3
"""Replay the recorded Glimmer Lab study from its public summary (no GPU, no network, no keys, standard library only).

    python scripts/replay_recorded_study.py [evidence/recorded-study/lab2p-summary.json] [--private-ledger DIR]

The summary was built from the private hash-chained ledgers of lab root lab2p by whitelisting fields; free-text decision records are
omitted. This script checks that the summary's event chain is complete and linked (every prev_hash equals the previous event's hash), then
prints every trial's arm, pack, seed, status and failure; every score's aggregate pass-all and mean constraint fraction; the candidate
proposal; the contamination scans; the scoring failures; and the GPU rental total.

With --private-ledger DIR (the private lab root), it also verifies the private ledger's own hash chain and that every event hash in the
summary equals the private ledger's hash for the same seq. Per-prompt results, prompts, answers and data splits are never needed.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("summary", nargs="?", default=str(HERE / "evidence/recorded-study/lab2p-summary.json"))
ap.add_argument("--private-ledger")
a = ap.parse_args()
doc = json.loads(Path(a.summary).read_text())
ev = doc["main_ledger_events"]
linked = all(ev[i]["prev_hash"] == ev[i - 1]["hash"] for i in range(1, len(ev))) and [e["seq"] for e in ev] == list(range(1, len(ev) + 1))
print(f"summary: {len(ev)} main-ledger events, sequence complete and hash-linked: {'OK' if linked else 'NO'}")
if a.private_ledger:
    sys.path.insert(0, str(HERE))
    from nightshift.lab.ledger import LabLedger  # noqa: E402
    ok, bad = LabLedger.read_only(a.private_ledger).verify()
    priv = {json.loads(l)["seq"]: json.loads(l)["hash"] for l in (Path(a.private_ledger) / "ledger.jsonl").read_text().splitlines() if l.strip()}
    same = all(priv.get(e["seq"]) == e["hash"] for e in ev) and len(priv) == len(ev)
    print(f"private ledger chain: {'OK' if ok else f'BROKEN at seq {bad}'}; summary hashes match it: {'OK' if same else 'NO'}")

print("\nTrials:")
started = {e["data"]["trial_id"]: e["data"] for e in ev if e["kind"] == "lab.trial_started"}
for e in ev:
    if e["kind"] == "lab.trial_finished":
        d = e["data"]; s = started.get(d["trial_id"], {})
        rt = d.get("researcher_runtime", {})
        print(f"  {d['trial_id']:7} {s.get('phase', '?'):7} arm={s.get('arm') or '-':8} {s.get('assignment_id', '?'):3} pack={s.get('pack_version', '?'):9} "
              f"seed={s.get('seed')} status={d['status']:17} researchers={rt.get('model', '-')}  {(d.get('failure') or '')[:80]}")
print("\nScores (aggregates):")
for e in ev:
    if e["kind"] == "lab.score_recorded":
        d = e["data"]
        for ps in d["per_seed"]:
            m = ps["metrics"]
            print(f"  {d['score_id']:12} {d['assignment_id']} {d['split']:9} trial={d.get('trial_id') or 'stock':7} pass-all {m['prompt_all']:.1%}  "
                  f"constraint fraction {m['prompt_fraction']:.1%}  n={ps['n']} truncated={ps['n_truncated']}  via {d['verifier_pins'].get('inspect_task', '-')}")
print("\nOther recorded events:")
for e in ev:
    if e["kind"] in ("lab.candidate_proposed", "lab.rule_frozen", "lab.search_closed", "lab.operator_change", "lab.final_analysis_frozen",
                     "lab.contamination_scanned", "lab.score_failed", "lab.gate_recorded"):
        brief = {k: v for k, v in e["data"].items() if k in ("candidate_id", "lever", "lint_passed", "proposer_model", "target", "trial_id", "passed", "n_ngram_hits", "failure")}
        print(f"  seq {e['seq']:3} {e['kind']:28} {json.dumps(brief)[:180]}")
r = doc["rentals"]
usd = sum(x["minutes"] / 60 * x["usd_per_hour"] for x in r)
print(f"\nGPU rentals in this root: {len(r)}, all deleted: {all(x['state'] == 'deleted' for x in r)}, estimate {usd:.2f} USD")
