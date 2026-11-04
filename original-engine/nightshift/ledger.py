"""Append-only, hash-chained event ledger. The ledger is the only authoritative state.

* One JSON object per line in ``<run_dir>/ledger.jsonl``.
* Each event stores ``prev_hash`` and ``hash``. ``verify_chain`` detects edits, deletions in the middle and reordering.
  It does NOT detect truncation or a full rewrite by someone who can write the file: for that, record ``head_anchor()``
  somewhere else (the CLI prints it) and later run ``nightshift verify --expect-head``.
* One writer process per run (advisory ``flock`` on ``ledger.jsonl.lock``). Any number of read-only openers can follow
  the file with ``refresh``.
* Only a torn FINAL line (crash mid-write) is dropped on open, after a backup copy. A bad line anywhere else raises
  ``LedgerCorrupt``; it is never auto-repaired.
* Every payload is scrubbed of registered secrets and has oversized strings truncated before it is written.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from . import secrets
from .types import (GENESIS_HASH, Clock, Event, LedgerCorrupt, LedgerView, compute_event_hash, format_exp_id)

try:  # pragma: no cover - fcntl is unavailable on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

LEDGER_NAME = "ledger.jsonl"
MAX_STRING_BYTES = 8192
_REQUIRED = ("seq", "ts", "run_id", "kind", "actor", "data", "prev_hash", "hash")


class LedgerLocked(RuntimeError):
    """Another process is already writing this run."""


def cap_strings(obj: Any, limit: int = MAX_STRING_BYTES) -> Any:
    """Truncate every string longer than ``limit`` bytes (UTF-8) to its head plus a marker."""
    if isinstance(obj, str):
        # lone surrogates (e.g. from a model reply decoded from JSON "\ud800") cannot be UTF-8 encoded
        raw = obj.encode("utf-8", errors="replace")
        if len(raw) <= limit:
            return raw.decode("utf-8")
        head = raw[:limit].decode("utf-8", errors="ignore")
        return f"{head}...[truncated {len(raw) - limit} bytes]"
    if isinstance(obj, dict):
        return {k: cap_strings(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [cap_strings(v, limit) for v in obj]
    return obj


def _parse_line(raw: bytes) -> Event:
    obj = json.loads(raw)
    if not isinstance(obj, dict) or any(k not in obj for k in _REQUIRED):
        raise ValueError("not an event")
    return Event.from_dict(obj)


class Ledger:
    def __init__(self, path: Path | str, run_id: str | None = None, *, clock: Clock = time.time,
                 writer: bool = True, fsync: bool = True):
        self.path = Path(path)
        self._clock = clock
        self._writer = writer
        self._fsync = fsync
        self._lock = threading.RLock()
        self._events: list[Event] = []
        self._offset = 0
        self._last_hash = GENESIS_HASH
        self._exp_allocated = 0
        self._view_cache: tuple[int, LedgerView] | None = None
        self._fd: int | None = None
        self._lock_fd: int | None = None
        self.recovered_torn_tail = False
        self.run_id = ""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if writer:
            self._acquire_writer_lock()
        try:
            self._load()
            if run_id is None:
                run_id = self._events[0].run_id if self._events else "r-" + uuid.uuid4().hex[:8]
            elif self._events and self._events[0].run_id != run_id:
                raise ValueError(f"ledger {self.path} belongs to run {self._events[0].run_id}, not {run_id}")
            self.run_id = run_id
            if writer:
                self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        except BaseException:
            self.close()
            raise

    # ---- construction helpers ------------------------------------------------------------------
    @classmethod
    def create(cls, run_dir: Path | str, *, clock: Clock = time.time, fsync: bool = True) -> "Ledger":
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / LEDGER_NAME).exists() and (run_dir / LEDGER_NAME).stat().st_size > 0:
            raise FileExistsError(f"{run_dir} already contains a ledger; use Ledger.open to resume")
        return cls(run_dir / LEDGER_NAME, clock=clock, fsync=fsync)

    @classmethod
    def open(cls, run_dir: Path | str, *, clock: Clock = time.time, fsync: bool = True) -> "Ledger":
        return cls(Path(run_dir) / LEDGER_NAME, clock=clock, fsync=fsync)

    @classmethod
    def read_only(cls, run_dir: Path | str) -> "Ledger":
        return cls(Path(run_dir) / LEDGER_NAME, writer=False)

    # ---- internals -----------------------------------------------------------------------------
    def _acquire_writer_lock(self) -> None:
        if fcntl is None:  # pragma: no cover
            return
        lock_path = self.path.with_name(self.path.name + ".lock")
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise LedgerLocked(f"{self.path} is being written by another process") from exc
        self._lock_fd = fd

    def _load(self) -> None:
        self._events, self._offset, self._last_hash = [], 0, GENESIS_HASH
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        lines = data.splitlines(keepends=True)
        good_bytes = 0
        for i, raw in enumerate(lines):
            is_last = i == len(lines) - 1
            try:
                if not raw.endswith(b"\n"):
                    raise ValueError("no newline")
                ev = _parse_line(raw)
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                if is_last:
                    break  # torn tail
                raise LedgerCorrupt(f"{self.path}: line {i + 1} is not a valid event ({exc}); refusing to continue",
                                    i + 1) from exc
            self._events.append(ev)
            self._last_hash = ev.hash
            good_bytes += len(raw)
        if good_bytes < len(data):
            self.recovered_torn_tail = True
            if self._writer:
                backup = self.path.with_name(f"{self.path.name}.torn-{int(time.time())}")
                shutil.copyfile(self.path, backup)
                with open(self.path, "r+b") as fh:
                    fh.truncate(good_bytes)
        self._offset = good_bytes
        ids = [int(str(e.data["exp_id"])[1:]) for e in self._events
               if e.kind == "experiment.proposed" and str(e.data.get("exp_id", "")).startswith("e")
               and str(e.data["exp_id"])[1:].isdigit()]
        self._exp_allocated = (max(ids) + 1) if ids else 0

    # ---- public API ----------------------------------------------------------------------------
    def append(self, kind: str, actor: str, data: dict) -> Event:
        if not self._writer or self._fd is None:
            raise PermissionError("ledger opened read-only")
        with self._lock:
            seq = (self._events[-1].seq + 1) if self._events else 1
            ts = float(self._clock())
            # normalise to JSON types (copy), scrub secrets, cap oversized strings
            payload = cap_strings(secrets.scrub(json.loads(json.dumps(data, default=str))))
            h = compute_event_hash(self._last_hash, seq, ts, self.run_id, kind, actor, payload)
            ev = Event(seq=seq, ts=ts, run_id=self.run_id, kind=kind, actor=actor, data=payload,
                       prev_hash=self._last_hash, hash=h)
            line = (json.dumps(ev.to_dict(), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            base = (len(self._events), self._offset, self._last_hash)
            try:
                view = memoryview(line)
                while view:  # os.write may be short
                    n = os.write(self._fd, view)
                    view = view[n:]
                if self._fsync:
                    os.fsync(self._fd)
                self._events.append(ev)
                self._offset += len(line)
                self._last_hash = h
            except BaseException:
                # A Ctrl-C (KeyboardInterrupt) can land anywhere in here, including after the line reached the file but
                # before memory knew about it; the next append would then reuse this seq and prev_hash and break the chain.
                self._recover_after_interrupted_append(*base)
                raise
            return ev

    def _recover_after_interrupted_append(self, base_len: int, base_offset: int, base_hash: str) -> None:
        """Make memory agree with the file after an append was interrupted (the file is authoritative).

        Restores the in-memory tail to its state before the append, then re-reads the file past ``base_offset``: a complete
        line of ours that is hash-consistent with the chain is ADOPTED (the event is durable, so it must stay in the
        chain), and anything else (a torn partial line, a line that does not chain) is truncated away so the next append
        continues the chain. Never raises ``OSError``; a second interrupt during recovery is the only way it can fail."""
        del self._events[base_len:]
        self._offset, self._last_hash = base_offset, base_hash
        try:
            with open(self.path, "rb") as fh:
                fh.seek(base_offset)
                tail = fh.read()
            good = 0
            for raw in tail.splitlines(keepends=True):
                if not raw.endswith(b"\n"):
                    break
                try:
                    ev = _parse_line(raw)
                except (ValueError, TypeError, AttributeError, KeyError):
                    break
                seq = (self._events[-1].seq + 1) if self._events else 1
                if (ev.seq != seq or ev.prev_hash != self._last_hash or ev.run_id != self.run_id
                        or ev.hash != compute_event_hash(self._last_hash, ev.seq, ev.ts, ev.run_id, ev.kind, ev.actor, ev.data)):
                    break
                self._events.append(ev)
                self._offset += len(raw)
                self._last_hash = ev.hash
                good += len(raw)
            if good < len(tail) and self._fd is not None:
                os.ftruncate(self._fd, base_offset + good)      # the fd is O_APPEND: the next write lands at the new end
        except OSError:
            pass

    def events(self, since_seq: int = 0) -> list[Event]:
        with self._lock:
            if since_seq <= 0:
                return list(self._events)
            return [e for e in self._events if e.seq > since_seq]

    def refresh(self) -> int:
        """Pick up events another process appended (read-only openers). Returns the number of new events."""
        with self._lock:
            if self._writer or not self.path.exists():
                return 0
            added = 0
            with open(self.path, "rb") as fh:
                fh.seek(self._offset)
                for raw in fh:
                    if not raw.endswith(b"\n"):
                        break
                    try:
                        ev = _parse_line(raw)
                    except (ValueError, TypeError, AttributeError, KeyError):
                        break
                    self._events.append(ev)
                    self._last_hash = ev.hash
                    self._offset += len(raw)
                    added += 1
            if self._events:
                self.run_id = self._events[0].run_id
            return added

    def view(self) -> LedgerView:
        from .view import build_view  # local import keeps module import order simple
        with self._lock:
            last = self._events[-1].seq if self._events else 0
            if self._view_cache and self._view_cache[0] == last:
                return self._view_cache[1]
            v = build_view(self._events)
            self._view_cache = (last, v)
            return v

    def next_exp_id(self) -> str:
        with self._lock:
            n = self._exp_allocated
            self._exp_allocated += 1
            return format_exp_id(n)

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        return verify_events(self.events())

    def head_anchor(self) -> tuple[int, str]:
        """(last seq, first 16 hex chars of the head hash). Record it outside this file to detect truncation."""
        with self._lock:
            if not self._events:
                return 0, GENESIS_HASH[:16]
            return self._events[-1].seq, self._events[-1].hash[:16]

    def seal(self) -> None:
        """Called when a run finishes: close the writer and make the file read-only (chmod 0444)."""
        self.close()
        try:
            os.chmod(self.path, 0o444)
        except OSError:  # pragma: no cover
            pass

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            if self._lock_fd is not None:
                try:
                    if fcntl is not None:
                        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def verify_events(events: list[Event]) -> tuple[bool, Optional[int]]:
    """Check sequence numbers, hash links, hashes and run id constancy. Returns (ok, first_bad_seq)."""
    prev = GENESIS_HASH
    expected_seq = 1
    run_id = events[0].run_id if events else None
    for ev in events:
        if ev.seq != expected_seq or ev.prev_hash != prev or ev.run_id != run_id:
            return False, ev.seq
        if ev.hash != compute_event_hash(ev.prev_hash, ev.seq, ev.ts, ev.run_id, ev.kind, ev.actor, ev.data):
            return False, ev.seq
        prev = ev.hash
        expected_seq += 1
    return True, None
