#!/usr/bin/env python3
"""Rental accounting for the two-loop experiment, read from the rental ledger (no network, no credentials).
    spend_report.py LAB_ROOT [PRIOR_SPEND_USD=2.54]
Prints one line per distinct provider rental (session, handle, trial, state, minutes, usd) and the totals against the owner's caps
(owner decision 8: 40.00 total cumulative ceiling including the earlier 2.54, i.e. 37.46 additional across all lab roots). A lab root being replaced never
resets these totals: pass every earlier root's rentals too (see ALSO)."""
import json, sys, time
root = sys.argv[1]
prior = float(sys.argv[2]) if len(sys.argv) > 2 else 2.54
also = sys.argv[3:]  # more LAB_ROOTs whose rentals count toward the same allowance
rows, total = [], 0.0
for r_ in [root] + also:
    d = json.load(open(f"{r_}/gpu/rentals.json"))
    for r in d["rentals"]:
        end = r.get("deleted_epoch") or time.time()
        minutes = max(0.0, (end - r["create_attempt_epoch"]) / 60.0)
        usd = minutes / 60.0 * r["usd_per_hour"] + float(r.get("fixed_usd") or 0)
        total += usd
        rows.append((r["session"], r["handle"], r["trial_id"], r["state"], round(minutes, 1), round(usd, 3)))
for row in rows:
    print(*row, sep="\t")
print(f"distinct rentals: {len(rows)}; additional spend so far: ${total:.2f} of $37.46; cumulative with prior ${prior:.2f}: ${total + prior:.2f} of $40.00")
