#!/usr/bin/env python3
"""Operator lease service for the separate API evidence lane: GPU procurement as a service, so the lane's new code never calls the provider,
never sees the operator's code and never touches another lane's ledger.

    lease_service.py LEASE_DIR LEDGER [--max-live 2] [--deadline-epoch E]

LEASE_DIR layout (the lane writes requests/ and releases/; this service writes everything else):
  requests/<request_id>.json   {"request_id", "trial_id", "purpose": "train"|"score", "max_lifetime_minutes" (<= 105), "note"?}
  leases/<request_id>.json     {"status": "granted"|"refused", "reason", "lease_id", "handle", "ssh": {host, port, user, key_path,
                                 known_hosts}, "cap_utc", "usd_per_hour", "created_utc", "ledger"}
  releases/<lease_id>.json     {"lease_id", "reason"?}       (early release; the cap deletes it anyway)
  released/<lease_id>.json     {"lease_id", "handle", "deleted_utc", "confirmed_absent": true, "billed_minutes", "usd_estimate"}

Refusals (status "refused"): an owner hold file PUBLIC_RUN_HOLD.json next to LEASE_DIR (checked before every create), the ledger's hard stop would be exceeded (the rental manager refuses before any create), more than --max-live
rentals of this ledger would be live, a lifetime above 105 minutes, a cap past the deadline, or a malformed request. Every lease is created
with the rental manager's detached watchdog (deletes at the cap) and is mirrored to the off-host reaper replica by the operator's replicator.
The provider token is read from this machine's Thunder login into this process's environment only; it is never written anywhere.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nightshift.lab.contracts import Shape  # noqa: E402
from nightshift.lab.gpu_rentals import RentalManager  # noqa: E402
from nightshift.lab.gpu_thunder import ThunderProvider  # noqa: E402

SHAPE = Shape("a100", 1, 8, 200, "base")
MAX_LIFETIME_MIN = 105
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def utc(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lease_dir")
    ap.add_argument("ledger")
    ap.add_argument("--max-live", type=int, default=2)
    ap.add_argument("--deadline-epoch", type=float, default=1793844000.0)  # 2026-11-04 21:00 ET
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    os.environ["LAB_THUNDER_TOKEN"] = json.load(open(os.path.expanduser("~/.thunder/cli_config.json")))["token"]
    d = Path(a.lease_dir)
    for sub in ("requests", "leases", "releases", "released"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    mgr = RentalManager(Path(a.ledger), ThunderProvider("LAB_THUNDER_TOKEN"), clock=time.time, machine="ankit-mac")
    while True:
        for req_path in sorted((d / "requests").glob("*.json")):
            rid = req_path.stem
            out = d / "leases" / f"{rid}.json"
            if out.exists():
                continue
            try:
                req = json.loads(req_path.read_text())
                tid, purpose, life = str(req["trial_id"]), str(req["purpose"]), float(req["max_lifetime_minutes"])
                if not (ID.match(rid) and ID.match(tid)) or purpose not in ("train", "score"):
                    raise ValueError("request_id/trial_id must be plain identifiers and purpose train or score")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                write(out, {"status": "refused", "reason": f"malformed request: {exc}"[:300]})
                continue
            hold = d.parent / "PUBLIC_RUN_HOLD.json"
            if hold.exists():  # owner hold: checked before every create; releases and the cap's deletion still work
                write(out, {"status": "refused", "reason": f"owner hold in force ({hold}); no lease is created until the owner lifts it"})
                continue
            ledger = json.load(open(a.ledger))
            live = [r for r in ledger["rentals"] if r["state"] != "deleted"]
            if not 0 < life <= MAX_LIFETIME_MIN:
                write(out, {"status": "refused", "reason": f"max_lifetime_minutes must be in (0, {MAX_LIFETIME_MIN}]"})
                continue
            if len(live) >= a.max_live:
                continue  # concurrency: leave the request pending; it is granted when a lease is released (the file says so meanwhile)
            if time.time() + life * 60 > a.deadline_epoch:
                write(out, {"status": "refused", "reason": "the lease cap would end after the 21:00 ET deadline"})
                continue
            session = f"api-{rid}"[:60]
            try:
                rental = mgr.create(session, SHAPE, life / 60.0, tid, f"{purpose}:{rid}")
                mgr.wait_ready(rental, 900.0)
                t = mgr.endpoint(rental)
                rec = next(r for r in json.load(open(a.ledger))["rentals"] if r["session"] == session)
                write(out, {"status": "granted", "reason": "", "lease_id": session, "handle": rec["handle"],
                            "ssh": {"host": t.host, "port": t.port, "user": t.user, "key_path": t.key_path, "known_hosts": t.known_hosts},
                            "cap_utc": utc(rec["cap_epoch"]), "usd_per_hour": rec["usd_per_hour"], "created_utc": utc(rec["create_attempt_epoch"]),
                            "ledger": a.ledger, "charge_basis": "minutes from the create attempt to confirmed deletion, at usd_per_hour"})
            except Exception as exc:  # noqa: BLE001 - every failure is reported to the requester; the rental manager deletes what it created
                write(out, {"status": "refused", "reason": f"{type(exc).__name__}: {exc}"[:400]})
        for rel in sorted((d / "releases").glob("*.json")):
            lid = rel.stem
            done = d / "released" / f"{lid}.json"
            if done.exists():
                continue
            rec = next((r for r in json.load(open(a.ledger))["rentals"] if r["session"] == lid), None)
            if rec is None:
                write(done, {"lease_id": lid, "confirmed_absent": False, "reason": "no such lease in this ledger"})
                continue
            try:
                if rec["state"] != "deleted":
                    mgr.delete(lid, "released by the API lane")
                rec = next(r for r in json.load(open(a.ledger))["rentals"] if r["session"] == lid)
                end = rec.get("deleted_epoch") or time.time()
                minutes = (end - rec["create_attempt_epoch"]) / 60.0
                write(done, {"lease_id": lid, "handle": rec["handle"], "deleted_utc": utc(end), "confirmed_absent": rec["state"] == "deleted",
                             "billed_minutes": round(minutes, 1), "usd_estimate": round(minutes / 60.0 * rec["usd_per_hour"], 3)})
            except Exception as exc:  # noqa: BLE001
                write(done, {"lease_id": lid, "confirmed_absent": False, "reason": f"{type(exc).__name__}: {exc}"[:300]})
        if a.once or time.time() > a.deadline_epoch:
            return 0
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
