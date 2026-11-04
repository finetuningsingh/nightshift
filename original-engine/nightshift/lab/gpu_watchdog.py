"""Detached watchdog and reaper for rented GPUs (the money backstop when the coordinator is dead).

    python -m nightshift.lab.gpu_watchdog watchdog <session> --ledger <rentals.json> --at-epoch E --provider thunder
    python -m nightshift.lab.gpu_watchdog reap --ledger <rentals.json> --provider thunder          # run every 5 minutes from cron
    python -m nightshift.lab.gpu_watchdog canary --provider thunder                                 # daily: can we still list the account?

``watchdog`` is spawned (detached, new session) by ``RentalManager.create`` BEFORE the provider is called. It sleeps to the rental's
armed ``cap_epoch``, then deletes in a loop with 60 s sleeps until absence is confirmed. It refuses an epoch later than the ledger's
cap (only ``rearm`` moves a cap), exits quietly when the rental is already deleted or when a rearm moved the cap past its epoch and a
replacement watchdog is alive (``rearm`` publishes the later cap before it arms the replacement; if the coordinator dies in between, this
watchdog still deletes at its own deadline), never deletes an instance that does not carry the rental's ownership tag (a rental whose handle is unknown is searched by tag; zero or
several matches write an ``ALARM`` file and the loop keeps retrying), and holds an advisory lock on ``watchdog.<pid>.lock`` so
liveness is "the lock is held" (no /proc scan; works on macOS).

``reap`` is idempotent: it deletes every ledger rental that is live or unsettled and more than ten minutes past its cap. It acts on
ledger rows only. Run it from a SECOND always-on host with that host's own provider token against a replicated copy of the ledger
(``--no-record`` makes it write nothing to the ledger). This is required before the first paid rental (docs/harness-design.md).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from nightshift.lab.contracts import Clock, GpuError, GpuProvider, OwnershipError, ProviderUnavailable, utc_iso
from nightshift.lab.gpu_rentals import LIVE_STATES, REAP_GRACE_SECONDS, RentalManager, alive_watchdogs

RETRY_SECONDS = 60.0
POLL_SECONDS = 5.0


class ReadOnlyRentalManager(RentalManager):
    """A manager that never writes the ledger (the replicated copy on a reaper host). Deletion goes through the same code path."""

    @contextlib.contextmanager
    def _locked(self) -> Iterator[dict]:  # type: ignore[override]
        yield self._load()


def _log(message: str) -> None:
    print(f"{utc_iso(time.time())} {message}", flush=True)


def watchdog(
    session: str,
    ledger: Path,
    provider: GpuProvider,
    *,
    at_epoch: Optional[int] = None,
    clock: Clock = time.time,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = POLL_SECONDS,
    retry_seconds: float = RETRY_SECONDS,
    log: Callable[[str], None] = _log,
) -> int:
    """Run one watchdog to completion. Returns 0 when the rental is gone (or no longer ours to delete), 2 on a refusal at start."""
    mgr = RentalManager(ledger, provider, clock=clock, sleep=sleep, poll_seconds=poll_seconds)
    try:
        row = mgr._row(session)
    except GpuError as exc:
        log(f"refusing: {exc}")
        return 2
    armed_cap = row["cap_epoch"]
    deadline = int(armed_cap if at_epoch is None else at_epoch)
    if deadline > row["cap_epoch"]:
        log(f"refusing --at-epoch {deadline}: it is past the ledger cap {row['cap_epoch']}; only rearm moves the cap")
        return 2
    sd = mgr.sessions_dir / session
    sd.mkdir(parents=True, exist_ok=True)
    lock = open(sd / f"watchdog.{os.getpid()}.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another watchdog with this pid holds the lock")
        return 2
    try:
        (sd / "watchdog.json").write_text(
            json.dumps({"armed_utc": utc_iso(clock()), "delete_at_epoch": deadline, "pid": os.getpid(), "session": session}, indent=2) + "\n"
        )
        log(f"armed: session {session} deleted at epoch {deadline}")
        while clock() < deadline:
            row = mgr._row(session)
            if row["state"] in ("deleted", "not_created"):
                log(f"{session} is {row['state']}; exiting")
                return 0
            sleep(max(0.0, min(poll_seconds, deadline - clock())))
        while True:
            row = mgr._row(session)
            if row["state"] in ("deleted", "not_created"):
                log(f"{session} is {row['state']}; exiting")
                return 0
            if row["cap_epoch"] != armed_cap and row["cap_epoch"] > deadline:
                # rearm publishes the later cap BEFORE its replacement watchdog exists: yield only to a watchdog that is really alive
                # (a coordinator that died in between leaves none, and then this one deletes at its own deadline)
                if [pid for pid in alive_watchdogs(sd) if pid != os.getpid()]:
                    log(f"the cap moved to {row['cap_epoch']}: a later watchdog owns the deadline; exiting")
                    return 0
                log(f"the cap moved to {row['cap_epoch']} but no other watchdog is alive: deleting at the armed deadline")
            try:
                mgr.delete(session, "watchdog: spend-cap deadline reached")
                log(f"{session} deleted")
                return 0
            except OwnershipError as exc:
                log(f"ALARM: {exc}; not deleting an instance that does not carry our tag")
                (sd / "ALARM").write_text(f"{utc_iso(clock())} {exc}\n")
            except (GpuError, ProviderUnavailable) as exc:
                log(f"delete attempt failed: {exc}; retrying in {retry_seconds:g} s")
                if row.get("handle") is None:
                    (sd / "ALARM").write_text(f"{utc_iso(clock())} handle unknown and not found by tag: {exc}\n")
            sleep(retry_seconds)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def reap(
    ledger: Path,
    provider: GpuProvider,
    *,
    clock: Clock = time.time,
    sleep: Callable[[float], None] = time.sleep,
    grace_seconds: float = REAP_GRACE_SECONDS,
    record: bool = True,
    poll_seconds: float = POLL_SECONDS,
) -> list[dict]:  # fmt: skip
    """Delete every ledger rental that is live or unsettled and more than ``grace_seconds`` past its cap. Returns one dict per rental it
    tried: ``{"session", "action": "deleted" | "failed", "error"}``. Idempotent."""
    cls = RentalManager if record else ReadOnlyRentalManager
    mgr = cls(ledger, provider, clock=clock, sleep=sleep, poll_seconds=poll_seconds)
    now = clock()
    out = []
    for row in mgr.read()["rentals"]:
        if row["state"] not in LIVE_STATES or now <= row["cap_epoch"] + grace_seconds:
            continue
        try:
            mgr.delete(row["session"], f"reaper: {int(now - row['cap_epoch'])} s past the cap")
            out.append({"session": row["session"], "action": "deleted", "error": None})
        except (GpuError, ProviderUnavailable) as exc:
            out.append({"session": row["session"], "action": "failed", "error": str(exc)[:300]})
    return out


def build_provider(args: argparse.Namespace) -> GpuProvider:
    if args.provider_factory:
        mod, _, fn = args.provider_factory.partition(":")
        return getattr(importlib.import_module(mod), fn)()
    if args.provider == "thunder":
        from nightshift.lab.gpu_thunder import DEFAULT_BASE_URL, ThunderProvider

        return ThunderProvider(args.token_env, args.base_url or DEFAULT_BASE_URL)
    raise GpuError(f"unknown provider {args.provider!r}")


def main(argv: Optional[Sequence[str]] = None, *, provider: Optional[GpuProvider] = None, clock: Clock = time.time, sleep: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(prog="python -m nightshift.lab.gpu_watchdog", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("watchdog", "reap", "canary"):
        p = sub.add_parser(name)
        p.add_argument("--provider", default="thunder")
        p.add_argument("--token-env", default="THUNDER_API_KEY", help="NAME of the environment variable that holds the provider token")
        p.add_argument("--base-url", default=None)
        p.add_argument("--provider-factory", default=None, help="module:callable returning a GpuProvider (tests)")
        if name != "canary":
            p.add_argument("--ledger", required=True, type=Path)
        if name == "watchdog":
            p.add_argument("session")
            p.add_argument("--at-epoch", type=int, default=None)
        if name == "reap":
            p.add_argument("--grace-seconds", type=float, default=REAP_GRACE_SECONDS)
            p.add_argument("--no-record", action="store_true", help="never write the ledger (a replicated copy on a reaper host)")
    args = ap.parse_args(argv)
    try:
        prov = provider if provider is not None else build_provider(args)
    except (GpuError, ProviderUnavailable) as exc:
        print(f"cannot build the provider: {exc}", file=sys.stderr)
        return 3
    if args.cmd == "watchdog":
        return watchdog(args.session, args.ledger, prov, at_epoch=args.at_epoch, clock=clock, sleep=sleep)
    if args.cmd == "canary":
        try:
            n = len(prov.list_instances())
        except (GpuError, ProviderUnavailable) as exc:
            print(f"CANARY FAILED: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"ok": True, "instances_in_account": n}))
        return 0
    results = reap(args.ledger, prov, clock=clock, sleep=sleep, grace_seconds=args.grace_seconds, record=not args.no_record)
    for r in results:
        print(json.dumps(r))
    return 1 if any(r["action"] == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
