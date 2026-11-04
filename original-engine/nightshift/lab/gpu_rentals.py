"""GPU rental manager: the money rule, the rental state machine, ownership, watchdog arming and the usage reports.

Adapted from the evolved kge-train ``lifecycle.py`` (K004; docs notes: gpu-lifecycle.md). Provider specifics live in a
``GpuProvider`` (``gpu_thunder.ThunderProvider``, or a fake in tests); the rules below are NOT provider code.

State machine of one rental (a row of the rental ledger): ``creating -> live -> deleted``, plus ``not_created`` (the provider
definitely refused) and ``uuid_unknown`` (a create may have succeeded and no handle is known). ``creating`` and ``uuid_unknown`` are
"unsettled": no further rental can be created while any is. A rental is ``deleted`` only after ``provider.exists(handle)`` returned
exactly False (``None`` means "could not tell"). The row is written BEFORE the provider is called, and a detached watchdog is armed
BEFORE the provider is called (case C of the notes), so a coordinator that dies between create and the rest still leaves a deadline.

Money rule (every create and every rearm): billed dollars of all rentals (minutes rounded up) + what every OTHER live rental can still
bill up to its armed cap + the new cap's dollars + the reserve must not exceed ``hard_stop_usd``. The hard stop lives in the ledger
file and has no default in code: ``init(hard_stop_usd, machine, why)`` sets it, and a create without one is refused.

Ownership: every row carries ``trial_id``, ``item`` and an owner tag (``owner_tag_of_key`` of the per-session public key). ``delete``
only acts on a row of this ledger and passes the tag to the provider, which re-resolves the provider's current row and refuses a
foreign one (``OwnershipError``). The Thunder account is shared with other projects and machines.

All ledger reads and writes are under an exclusive ``flock`` on ``<ledger>.lock`` (several processes write: the coordinator, the
watchdogs, the reaper).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from nightshift.lab.contracts import (
    Clock,
    CreateAmbiguous,
    CreateRejected,
    GpuError,
    GpuInstance,
    GpuProvider,
    GpuUsage,
    OwnershipError,
    ProviderUnavailable,
    Shape,
    owner_tag_of_key,
    utc_iso,
)
from nightshift.lab.gpu_remote import SshTarget, write_ssh_config

LEDGER_SCHEMA = "lab-gpu-rentals/v1"
UNSETTLED = ("creating", "uuid_unknown")
LIVE_STATES = ("creating", "live", "uuid_unknown")
MAX_CAP_HOURS = 12.0
MAX_REARM_HOURS_AHEAD = 24.0
WATCHDOG_ARM_SECONDS = 15.0
REAP_GRACE_SECONDS = 600.0  # a rental is reaped when it is this far past its cap
_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class RentalRefused(GpuError):
    """The money rule, an unsettled rental, a missing hard stop or an unavailable shape says no. Nothing was created."""

    def __init__(self, message: str, check: Optional[dict] = None, code: str = "refused") -> None:
        super().__init__(message)
        self.check = check or {}
        self.code = code  # "unsettled": another create is in flight (or stuck); "refused": the money rule or an argument


@dataclass(frozen=True)
class Rental:
    """A snapshot of one ledger row. ``session`` is the rental id (it names ``gpu/sessions/<session>`` and is what ``GpuUsage`` lists)."""

    session: str
    trial_id: str
    item: str
    shape: Shape
    state: str
    handle: Optional[str]
    owner_tag: str
    usd_per_hour: float
    fixed_usd: float
    create_attempt_epoch: float
    cap_epoch: float
    session_dir: Path

    @property
    def rental_id(self) -> str:
        return self.session


def _check_session(session: str) -> str:
    if not isinstance(session, str) or not _SESSION.match(session):
        raise GpuError(f"session {session!r} is not a safe identifier")
    return session


def billed_minutes(row: dict, now: float) -> int:
    """Minutes from the create attempt to the confirmed deletion (or now), rounded UP. Billing starts at the attempt (conservative)."""
    end = row["deleted_epoch"] if row.get("deleted_epoch") is not None else now
    return max(0, math.ceil((end - row["create_attempt_epoch"]) / 60.0 - 1e-9))


def cap_minutes(row: dict) -> int:
    return max(0, math.ceil((row["cap_epoch"] - row["create_attempt_epoch"]) / 60.0 - 1e-9))


def row_usd(row: dict, minutes: float) -> float:
    return minutes * row["usd_per_hour"] / 60.0 + row.get("fixed_usd", 0.0)


def spend(doc: dict, now: float) -> tuple[float, list[dict]]:
    """Billed dollars over the ledger and per rental; a live rental also carries ``usd_to_cap`` (what it bills by its cap, or by now if
    it is past its cap)."""
    rows = []
    for r in doc["rentals"]:
        minutes, live = billed_minutes(r, now), r["state"] in LIVE_STATES
        to_cap = max(minutes, cap_minutes(r)) if live else minutes
        usd = round(row_usd(r, minutes), 6) if (minutes or live) else 0.0
        rows.append({
            "session": r["session"], "trial_id": r["trial_id"], "item": r["item"], "state": r["state"], "handle": r.get("handle"),
            "billed_minutes": minutes, "usd_per_hour": r["usd_per_hour"], "usd": usd, "live": live,
            "usd_to_cap": round(row_usd(r, to_cap), 6) if live else usd, "cap_epoch": r["cap_epoch"],
        })  # fmt: skip
    return round(sum(x["usd"] for x in rows), 6), rows


def money_check(doc: dict, new_usd: float, now: float, *, exclude_session: Optional[str] = None, extra_reserve: float = 0.0) -> dict:
    """The hard-stop rule. ``exclude_session`` is a rental being rearmed: it counts only up to now and ``new_usd`` is its window from now."""
    spent, rows = spend(doc, now)
    others = round(sum(r["usd_to_cap"] - r["usd"] for r in rows if r["live"] and r["session"] != exclude_session), 6)
    reserve = round(float(doc.get("reserve_usd", 0.0)) + extra_reserve, 6)
    total = round(spent + others + new_usd + reserve, 6)
    stop = doc.get("hard_stop_usd")
    return {
        "spent_usd": spent, "committed_others_usd": others, "new_cap_usd": round(new_usd, 6), "reserve_usd": reserve,
        "total_usd": total, "hard_stop_usd": stop, "ok": stop is not None and total <= stop + 1e-9, "rentals": rows,
    }  # fmt: skip


# ----------------------------------------------------------------------------------------------------
# watchdog processes
# ----------------------------------------------------------------------------------------------------
def alive_watchdogs(session_dir: Path) -> list[int]:
    """Pids of live watchdogs of a session. A watchdog holds an advisory lock on ``watchdog.<pid>.lock`` for its whole life, so liveness
    is "the lock is held" (no /proc scan, works on macOS)."""
    pids = []
    for lock in sorted(Path(session_dir).glob("watchdog.*.lock")):
        m = re.fullmatch(r"watchdog\.(\d+)\.lock", lock.name)
        if not m:
            continue
        try:
            fd = os.open(lock, os.O_RDWR)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                pids.append(int(m.group(1)))
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    return pids


def spawn_watchdog_process(provider_args: Sequence[str]) -> Callable[[Path, str, Path, int], None]:
    """The default ``launcher``: a detached ``python -m nightshift.lab.gpu_watchdog watchdog`` (new session, stdin /dev/null, output to
    ``watchdog.log``), which survives the death of the coordinator and of the terminal."""

    def launch(session_dir: Path, session: str, ledger: Path, at_epoch: int) -> None:
        env = dict(os.environ)
        pkg_parent = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        argv = [
            sys.executable, "-m", "nightshift.lab.gpu_watchdog", "watchdog", session, "--ledger", str(ledger),
            "--at-epoch", str(at_epoch), *provider_args,
        ]  # fmt: skip
        log = open(session_dir / "watchdog.log", "ab")
        try:
            subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=env, close_fds=True)
        finally:
            log.close()

    return launch


def ssh_keygen_factory(session_dir: Path) -> str:
    """Generate a per-session ed25519 keypair with ssh-keygen (private key 0600) and return the public key line."""
    session_dir.mkdir(parents=True, exist_ok=True)
    key = session_dir / "id_ed25519"
    try:
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"lab-{session_dir.name}", "-f", str(key)],
            check=True, capture_output=True, timeout=60,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuError(f"ssh-keygen failed: {exc}") from None
    key.chmod(0o600)
    return (session_dir / "id_ed25519.pub").read_text().strip()


class RentalManager:
    """See the module docstring. ``launcher(session_dir, session, ledger_path, at_epoch)`` starts a watchdog (default: a detached
    process; ``watchdog_provider_args`` are appended to its command line, e.g. ``["--provider", "thunder"]``)."""

    def __init__(
        self,
        rental_ledger_path: Path,
        provider: GpuProvider,
        *,
        clock: Clock,
        poll_seconds: float = 5.0,
        launcher: Optional[Callable[[Path, str, Path, int], None]] = None,
        watchdog_provider_args: Sequence[str] = (),
        key_factory: Callable[[Path], str] = ssh_keygen_factory,
        sleep: Callable[[float], None] = time.sleep,
        machine: Optional[str] = None,
        max_cap_hours: float = MAX_CAP_HOURS,
        stopper: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.ledger = Path(rental_ledger_path)
        self.provider = provider
        self.clock = clock
        self.poll_seconds = float(poll_seconds)
        self.launcher = launcher or spawn_watchdog_process(watchdog_provider_args)
        self.key_factory = key_factory
        self.sleep = sleep
        self.machine = machine
        self.max_cap_hours = float(max_cap_hours)
        self.stopper = stopper or (lambda pid: os.kill(pid, signal.SIGTERM))
        self.sessions_dir = self.ledger.parent / "sessions"

    # ledger file ---------------------------------------------------------------------------------
    @contextlib.contextmanager
    def _locked(self) -> Iterator[dict]:
        """Exclusive flock; yields the document, saves it on exit (atomic replace)."""
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.ledger) + ".lock", "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                doc = self._load()
                yield doc
                self._save(doc)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _load(self) -> dict:
        if not self.ledger.is_file():
            return {"schema": LEDGER_SCHEMA, "hard_stop_usd": None, "machine": None, "why": None, "reserve_usd": 0.0, "history": [], "rentals": []}
        doc = json.loads(self.ledger.read_text())
        if doc.get("schema") != LEDGER_SCHEMA:
            raise GpuError(f"{self.ledger}: schema {doc.get('schema')!r} is not {LEDGER_SCHEMA}")
        return doc

    def _save(self, doc: dict) -> None:
        tmp = self.ledger.with_name(self.ledger.name + ".tmp")
        with open(tmp, "w") as fh:
            fh.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.ledger)

    def read(self) -> dict:
        """A consistent copy of the ledger document (for reports and tests)."""
        with self._locked() as doc:
            return json.loads(json.dumps(doc))

    @staticmethod
    def _find(doc: dict, session: str) -> Optional[dict]:
        for r in doc["rentals"]:
            if r["session"] == session:
                return r
        return None

    def _update(self, session: str, **fields: Any) -> dict:
        with self._locked() as doc:
            row = self._find(doc, session)
            if row is None:
                raise GpuError(f"{session} is not in the rental ledger {self.ledger}; refusing to touch it")
            row.update(fields)
            return dict(row)

    def _row(self, session: str) -> dict:
        with self._locked() as doc:
            row = self._find(doc, session)
            if row is None:
                raise GpuError(f"{session} is not in the rental ledger {self.ledger}; refusing to touch its instance")
            return dict(row)

    def _to_rental(self, row: dict) -> Rental:
        return Rental(
            row["session"], row["trial_id"], row["item"], Shape.from_dict(row["shape"]), row["state"], row.get("handle"),
            row["owner_tag"], row["usd_per_hour"], row.get("fixed_usd", 0.0), row["create_attempt_epoch"], row["cap_epoch"],
            self.sessions_dir / row["session"],
        )  # fmt: skip

    def rental(self, session: str) -> Rental:
        return self._to_rental(self._row(_check_session(session)))

    # operator commands ---------------------------------------------------------------------------
    def init(self, hard_stop_usd: float, machine: str, why: str) -> None:
        """Set the dollar hard stop (no default exists in code). Refuses to overwrite one: use ``set_hard_stop``."""
        if isinstance(hard_stop_usd, bool) or not isinstance(hard_stop_usd, (int, float)) or not math.isfinite(hard_stop_usd) or hard_stop_usd <= 0:
            raise GpuError("init: hard_stop_usd must be a positive number")
        if not why or not why.strip() or not machine or not machine.strip():
            raise GpuError("init: machine and why are required")
        with self._locked() as doc:
            if doc.get("hard_stop_usd") is not None:
                raise GpuError(f"init: {self.ledger} already has a hard stop of {doc['hard_stop_usd']}; use set_hard_stop")
            doc.update(hard_stop_usd=float(hard_stop_usd), machine=machine, why=why)
            doc["history"].append({"at": utc_iso(self.clock()), "hard_stop_usd": float(hard_stop_usd), "machine": machine, "why": why})

    def set_hard_stop(self, hard_stop_usd: float, why: str) -> None:
        if not why or not why.strip() or isinstance(hard_stop_usd, bool) or not isinstance(hard_stop_usd, (int, float)) or hard_stop_usd <= 0:
            raise GpuError("set_hard_stop: a positive amount and a reason are required")
        with self._locked() as doc:
            doc["hard_stop_usd"] = float(hard_stop_usd)
            doc["history"].append({"at": utc_iso(self.clock()), "hard_stop_usd": float(hard_stop_usd), "why": why})

    def quote(self, shape: Shape):
        return self.provider.quote(shape)

    # create --------------------------------------------------------------------------------------
    def create(self, session: str, shape: Shape, cap_hours: float, trial_id: str, item: str) -> Rental:
        """Rent one machine. Raises RentalRefused (nothing created), CreateRejected (nothing exists), CreateAmbiguous (a rental may exist;
        the row is ``uuid_unknown`` and blocks every later create until ``recover``), ProviderUnavailable (nothing created)."""
        _check_session(session)
        if isinstance(cap_hours, bool) or not isinstance(cap_hours, (int, float)) or not 0 < cap_hours <= self.max_cap_hours:
            raise RentalRefused(f"cap_hours must be in (0, {self.max_cap_hours:g}], got {cap_hours!r}")
        if not isinstance(trial_id, str) or not trial_id or not isinstance(item, str) or not item:
            raise RentalRefused("trial_id and item are required")
        session_dir = self.sessions_dir / session
        if session_dir.exists():
            raise RentalRefused(f"session {session} already exists")
        quote = self.provider.quote(shape)  # may raise ProviderUnavailable
        if quote.available is False:
            raise RentalRefused(f"{shape.gpu} x{shape.num_gpus} is not available right now ({quote.source})")
        now = self.clock()
        cap_min = math.ceil(cap_hours * 60.0 - 1e-9)
        new_usd = cap_min * quote.usd_per_hour / 60.0 + quote.fixed_usd
        # A whole second, like a rearm target, rounded DOWN: rounding up could add a billable minute (cap_minutes) the admission check above did
        # not price. The watchdog is armed at an integer epoch and refuses one later than the ledger cap.
        epoch_cap = float(math.floor(now + cap_min * 60.0))
        # Key first (a failure here has created nothing), then the row, under the lock.
        pub = self.key_factory(session_dir)
        tag = owner_tag_of_key(pub)
        try:
            with self._locked() as doc:
                if doc.get("hard_stop_usd") is None:
                    raise RentalRefused("no hard stop is set in the rental ledger: run init(hard_stop_usd, machine, why) first")
                if self.machine is not None and doc.get("machine") != self.machine:
                    raise RentalRefused(f"this ledger belongs to machine {doc.get('machine')!r}, not {self.machine!r}")
                if self._find(doc, session) is not None:
                    raise RentalRefused(f"session {session} is already in the rental ledger")
                busy = [r["session"] for r in doc["rentals"] if r["state"] in UNSETTLED]
                if busy:
                    raise RentalRefused(
                        f"rental(s) {busy} are unsettled (creating or uuid_unknown); settle them with recover before creating another", code="unsettled"
                    )  # fmt: skip
                check = money_check(doc, new_usd, now)
                if not check["ok"]:
                    raise RentalRefused(
                        f"the hard-stop rule refuses: spent {check['spent_usd']:.2f} + committed {check['committed_others_usd']:.2f} + new "
                        f"{check['new_cap_usd']:.2f} + reserve {check['reserve_usd']:.2f} = {check['total_usd']:.2f} > {check['hard_stop_usd']:.2f}",
                        check,
                    )  # fmt: skip
                doc["rentals"].append({
                    "session": session, "trial_id": trial_id, "item": item, "provider": getattr(self.provider, "name", "?"),
                    "shape": shape.to_dict(), "state": "creating", "handle": None, "owner_tag": tag, "usd_per_hour": quote.usd_per_hour,
                    "fixed_usd": quote.fixed_usd, "create_attempt_epoch": now, "cap_epoch": epoch_cap, "before_handles": [],
                    "deleted_epoch": None, "delete_reason": None, "rearms": [], "anomalies": [], "created_utc": utc_iso(now),
                })  # fmt: skip
        except BaseException:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise
        # Watchdog BEFORE the provider is called (case C): a coordinator that dies after the create still leaves a deadline.
        try:
            self._arm_watchdog(session, session_dir, int(epoch_cap))
        except BaseException:
            self._update(session, state="not_created", deleted_epoch=now, delete_reason="watchdog could not be armed; nothing was created")
            raise
        # The listing before the create is what find_created compares against.
        try:
            before = sorted(i.handle for i in self.provider.list_instances())
        except ProviderUnavailable:
            self._update(session, state="not_created", deleted_epoch=now, delete_reason="provider listing failed before the create; nothing was created")
            raise
        self._update(session, before_handles=before)
        try:
            receipt = self.provider.create(shape, pub)
        except CreateRejected:
            self._update(session, state="not_created", deleted_epoch=now, delete_reason="the provider refused the create")
            raise
        except CreateAmbiguous as exc:
            self._update(session, state="uuid_unknown")
            handle = self._find_handle(session)
            if handle is None:
                raise CreateAmbiguous(f"{exc}; the rental may exist and its handle is unknown. Do NOT create again: run recover({session!r})") from None
            return self._adopt(session, handle)
        except Exception as exc:  # any other failure after the request was sent is ambiguous, not a refusal
            self._update(session, state="uuid_unknown")
            handle = self._find_handle(session)
            if handle is None:
                raise CreateAmbiguous(f"create failed with {type(exc).__name__}; the rental may exist. Do NOT create again: run recover({session!r})") from None
            return self._adopt(session, handle)
        handle = receipt.handle
        if handle is None:
            self._update(session, state="uuid_unknown")
            handle = self._find_handle(session)
            if handle is None:
                raise CreateAmbiguous(f"the provider returned no handle. Do NOT create again: run recover({session!r})")
        return self._adopt(session, handle)

    def _arm_watchdog(self, session: str, session_dir: Path, at_epoch: int) -> None:
        before = set(alive_watchdogs(session_dir))
        self.launcher(session_dir, session, self.ledger, at_epoch)
        tries = max(1, int(WATCHDOG_ARM_SECONDS / 0.25))
        for _ in range(tries):
            now_alive = set(alive_watchdogs(session_dir)) - before
            info = session_dir / "watchdog.json"
            if now_alive and info.is_file():
                try:
                    doc = json.loads(info.read_text())
                except ValueError:
                    doc = {}
                if doc.get("pid") in now_alive and doc.get("delete_at_epoch") == at_epoch:
                    return
            self.sleep(0.25)
        raise GpuError(f"the watchdog of {session} did not arm within {WATCHDOG_ARM_SECONDS:g} s; nothing was created")

    def _find_handle(self, session: str, tries: int = 6) -> Optional[str]:
        row = self._row(session)
        for i in range(tries):
            try:
                handle = self.provider.find_created(set(row.get("before_handles", [])), Shape.from_dict(row["shape"]), row["create_attempt_epoch"], row["owner_tag"])
            except ProviderUnavailable:
                handle = None
            if handle is not None:
                return handle
            if i < tries - 1:
                self.sleep(self.poll_seconds)
        return None

    def _adopt(self, session: str, handle: str) -> Rental:
        row = self._update(session, state="live", handle=handle)
        sd = self.sessions_dir / session
        (sd / "handle.txt").write_text(handle + "\n")
        return self._to_rental(row)

    def recover(self, session: str, *, confirm_none: bool = False) -> Rental:
        """Settle an unsettled rental: adopt a unique new instance carrying our tag, or (``confirm_none``, only after reading the provider
        console) record that none exists, billed up to now."""
        row = self._row(_check_session(session))
        if row["state"] not in UNSETTLED:
            return self._to_rental(row)
        handle = self._find_handle(session)
        if handle is not None:
            return self._adopt(session, handle)
        if confirm_none:
            row = self._update(session, state="not_created", deleted_epoch=self.clock(), delete_reason="operator confirmed that no instance exists")
            return self._to_rental(row)
        raise CreateAmbiguous(f"{session}: no unique instance carries this rental's tag; read the provider console, then recover(confirm_none=True)")

    # readiness and endpoint ----------------------------------------------------------------------
    def _list(self, tries: int = 3) -> list[GpuInstance]:
        last: Optional[Exception] = None
        for _ in range(tries):
            try:
                return self.provider.list_instances()
            except ProviderUnavailable as exc:
                last = exc
                self.sleep(self.poll_seconds)
        raise last  # type: ignore[misc]

    def wait_ready(self, rental: Rental, timeout_s: float = 900.0) -> GpuInstance:
        """Poll until the instance is RUNNING with a host and a port. On timeout it does NOT delete (the watchdog owns that) and says so."""
        row = self._row(rental.session)
        if row["handle"] is None:
            raise GpuError(f"{rental.session} has no handle yet")
        start = self.clock()
        while True:
            state = self._row(rental.session)["state"]
            if state in ("deleted", "not_created"):
                raise GpuError(f"{rental.session} was {state} while waiting for it to become ready")
            try:
                for inst in self.provider.list_instances():
                    if inst.handle == row["handle"] and inst.status.upper() == "RUNNING" and inst.host and inst.port:
                        return inst
            except ProviderUnavailable:
                pass
            if self.clock() - start > timeout_s:
                raise GpuError(f"{rental.session} was not ready after {timeout_s:g} s; it is STILL BILLING until deleted (the watchdog deletes it at its cap)")
            self.sleep(self.poll_seconds)

    def endpoint(self, rental: Rental) -> SshTarget:
        """The ssh target of a live rental (addresses are dynamic: always read from the provider's current row)."""
        row = self._row(rental.session)
        if row["state"] != "live" or row["handle"] is None:
            raise GpuError(f"{rental.session} is {row['state']}, not live")
        inst = next((i for i in self._list() if i.handle == row["handle"]), None)
        if inst is None or not inst.host or not inst.port:
            raise GpuError(f"{rental.session}: the provider shows no address for {row['handle']}")
        sd = self.sessions_dir / rental.session
        target = SshTarget(inst.host, inst.port, inst.user, str(sd / "id_ed25519"), str(sd / "known_hosts"))
        with contextlib.suppress(OSError):
            write_ssh_config(sd / "ssh_config", f"lab-{rental.session}", target)
        return target

    # delete --------------------------------------------------------------------------------------
    def delete(self, rental: Rental | str, reason: str, *, confirm_tries: int = 36) -> None:
        """Ensure the instance is gone and record it. Idempotent. Raises OwnershipError for a foreign row (nothing is sent),
        ProviderUnavailable when the provider cannot be asked, GpuError when absence cannot be confirmed (the rental may still bill)."""
        session = rental if isinstance(rental, str) else rental.session
        row = self._row(_check_session(session))
        if row["state"] in ("deleted", "not_created"):
            return
        handle = row.get("handle")
        if handle is None:
            handle = self._find_handle(session, tries=2)
            if handle is None:
                raise GpuError(f"{session}: no handle is known, so nothing can be deleted; run recover. The instance may still bill.")
            self._update(session, handle=handle)
        before = {i.handle for i in self._list(tries=1)}  # a failure here sends nothing
        if handle in before:
            self.provider.delete(handle, row["owner_tag"])
        for attempt in range(confirm_tries):
            state = self.provider.exists(handle)
            if state is False:
                break
            if attempt < confirm_tries - 1:
                self.sleep(max(0.0, self.poll_seconds / 2))
        else:
            raise GpuError(f"{session}: deletion of {handle} is not confirmed; the instance may still bill")
        anomalies = []
        try:
            vanished = before - {i.handle for i in self.provider.list_instances()} - {handle}
            anomalies = sorted(vanished)
        except ProviderUnavailable:
            pass
        self._record_deleted(session, reason, anomalies)

    def _record_deleted(self, session: str, reason: str, anomalies: Sequence[str] = ()) -> None:
        now = self.clock()
        with self._locked() as doc:
            row = self._find(doc, session)
            if row is None:
                raise GpuError(f"{session} is not in the rental ledger")
            if row.get("deleted_epoch") is None or row["state"] not in ("deleted", "not_created"):
                row.update(state="deleted", deleted_epoch=now, delete_reason=reason)
            if anomalies:
                row["anomalies"].append({"at": utc_iso(now), "other_instances_vanished_during_delete": list(anomalies)})
        sd = self.sessions_dir / session
        with contextlib.suppress(OSError):
            (sd / "deleted-utc.txt").write_text(utc_iso(now) + "\n")
        with contextlib.suppress(OSError):  # the per-rental private key is deleted at teardown
            (sd / "id_ed25519").unlink()

    # rearm ---------------------------------------------------------------------------------------
    def rearm(self, rental: Rental | str, *, at_epoch: Optional[float] = None, hours_from_now: Optional[float] = None) -> Rental:
        """Move the armed cap. A later cap is checked against the money rule (the rental counts up to now, the new window from now) and
        against ``MAX_REARM_HOURS_AHEAD``; an earlier cap is always allowed. The new watchdog is armed BEFORE the old one is stopped, and
        the old cap is restored if the new watchdog cannot be armed."""
        session = rental if isinstance(rental, str) else rental.session
        if (at_epoch is None) == (hours_from_now is None):
            raise GpuError("rearm: give exactly one of at_epoch and hours_from_now")
        now = self.clock()
        target = float(at_epoch) if at_epoch is not None else now + float(hours_from_now) * 3600.0  # type: ignore[arg-type]
        target = float(math.ceil(target))
        if target <= now:
            raise RentalRefused("rearm: the new cap must be in the future")
        if target > now + MAX_REARM_HOURS_AHEAD * 3600.0:
            raise RentalRefused(f"rearm: at most {MAX_REARM_HOURS_AHEAD:g} hours ahead of now")
        sd = self.sessions_dir / session
        with self._locked() as doc:
            row = self._find(doc, session)
            if row is None:
                raise GpuError(f"{session} is not in the rental ledger")
            if row["state"] not in ("live", "creating"):
                raise RentalRefused(f"rearm: {session} is {row['state']}")
            old_cap = row["cap_epoch"]
            if target > old_cap:
                window = math.ceil((target - now) / 60.0 - 1e-9)
                check = money_check(doc, window * row["usd_per_hour"] / 60.0, now, exclude_session=session)
                if not check["ok"]:
                    raise RentalRefused(
                        f"the hard-stop rule refuses the rearm: total {check['total_usd']:.2f} > {check['hard_stop_usd']:.2f}", check
                    )  # fmt: skip
            # A later cap is published BEFORE its watchdog exists (the watchdog refuses a deadline past the ledger cap) and restored if arming
            # fails: restoring lowers the commitment, so it needs no money check. An earlier cap keeps the old, larger commitment in the
            # ledger until its watchdog is armed, so a concurrent create cannot spend the released money while the old cap is still in force.
            if target > old_cap:
                row["cap_epoch"] = target
                row["rearms"].append({"at": utc_iso(now), "from_cap_epoch": old_cap, "to_cap_epoch": target})
        old_pids = alive_watchdogs(sd)
        try:
            self._arm_watchdog(session, sd, int(target))
        except BaseException:
            if target > old_cap:
                with self._locked() as doc:
                    row = self._find(doc, session)
                    if row is not None and row["cap_epoch"] == target:
                        row["cap_epoch"] = old_cap
            raise
        if target <= old_cap:
            with self._locked() as doc:
                row = self._find(doc, session)
                if row is None or row["state"] not in ("live", "creating"):
                    raise RentalRefused(f"rearm: {session} is no longer live")
                row["cap_epoch"] = target
                row["rearms"].append({"at": utc_iso(now), "from_cap_epoch": old_cap, "to_cap_epoch": target})
        for pid in old_pids:
            with contextlib.suppress(OSError):
                self.stopper(pid)
        return self.rental(session)

    # reports -------------------------------------------------------------------------------------
    def usage_for(self, trial_id: str) -> GpuUsage:
        """Billed GPU use of every rental of a trial (minutes rounded up per rental, at each rental's quoted rate)."""
        return self._usage(lambda r: r["trial_id"] == trial_id)

    def usage_for_session(self, session: str) -> GpuUsage:
        return self._usage(lambda r: r["session"] == session)

    def _usage(self, pick: Callable[[dict], bool]) -> GpuUsage:
        now = self.clock()
        with self._locked() as doc:
            rows = [r for r in doc["rentals"] if pick(r)]
        used = [r for r in rows if r["state"] != "not_created" or billed_minutes(r, now) > 0]
        minutes = sum(billed_minutes(r, now) for r in used)
        usd = sum(row_usd(r, billed_minutes(r, now)) for r in used)
        return GpuUsage(float(minutes), round(usd, 6), tuple(r["session"] for r in used))

    def cost(self) -> dict:
        """Spent, committed to caps, remaining under the hard stop, per rental, and whether each live rental has a live watchdog."""
        now = self.clock()
        with self._locked() as doc:
            doc = json.loads(json.dumps(doc))
        check = money_check(doc, 0.0, now)
        stop = doc.get("hard_stop_usd")
        rows = check["rentals"]
        for r in rows:
            r["watchdog_alive"] = bool(alive_watchdogs(self.sessions_dir / r["session"])) if r["live"] else None
        return {
            "hard_stop_usd": stop, "spent_usd": check["spent_usd"], "committed_usd": check["committed_others_usd"],
            "remaining_usd": None if stop is None else round(stop - check["spent_usd"] - check["committed_others_usd"] - check["reserve_usd"], 6),
            "rentals": rows,
        }  # fmt: skip
