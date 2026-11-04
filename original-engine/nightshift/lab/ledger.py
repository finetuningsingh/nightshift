"""Write side of the lab ledgers: a thin, thread-safe, contract-checking wrapper around ``nightshift.ledger.Ledger``.

The reused ``Ledger`` accepts any string as a kind and any dict as data. ``LabLedger`` is the only object that appends to a lab ledger,
and it refuses before any write:

* a kind that is not in ``LabKind.all()``;
* a payload that is not exactly the contract class ``PAYLOAD_TYPES[kind]``;
* a payload with a non-finite number or a string the ledger would truncate (``assert_ledger_safe``);
* an actor that is not ``coordinator``, ``scorer``, ``outer`` or ``trial:<id>``;
* a second event of a ``LabKind.FREEZE_ONCE`` kind (also across a reopen: the kinds already in the file count).

One coordinator process owns each ledger file (``Ledger`` takes an advisory ``flock``); inside it every thread shares one ``LabLedger``
(``Ledger.append`` is serialised by an RLock and this class adds its own lock around the freeze-once check and the append, so the check and
the write are one step). Register credentials with ``register_secrets`` before the first append: the reused scrubber only removes
values it knows.

Import this module by its absolute name (``nightshift.lab.ledger``); it shares a basename with ``nightshift/ledger.py``.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from nightshift import secrets as _secrets
from nightshift.ledger import Ledger, LedgerLocked
from nightshift.lab.contracts import (
    PAYLOAD_TYPES,
    Clock,
    ContractError,
    LabKind,
    LedgerAnchor,
    LedgerError,
    assert_ledger_safe,
)
from nightshift.types import Event

__all__ = ["LabLedger", "LedgerPayloadError", "register_secrets", "ACTOR_PATTERN"]

# coordinator (CLI, trial runner, outer loop wiring), scorer (a score the scorer service appends), outer (the proposal loop), trial:<id>.
ACTOR_PATTERN = re.compile(r"^(coordinator|scorer|outer|trial:[A-Za-z0-9][A-Za-z0-9_.-]{0,63})$")


class LedgerPayloadError(LedgerError, ContractError):
    """The payload is not a lab contract record or is unsafe to write. Both a ``LedgerError`` and a ``ContractError``."""


def register_secrets(*env_names: str) -> None:
    """Register the values of these environment variables with the scrubber (``nightshift.secrets.register_env``). Call it for the Meta
    key, the Thunder token and every trial token BEFORE the first append."""
    _secrets.register_env(*env_names)


class LabLedger:
    """Implements ``LedgerWriter`` (``append``, ``anchor``, ``seal``, ``close``) plus ``events``, ``verify`` and ``refresh``.

    Build it with ``create`` (a new ledger; refuses to overwrite one), ``open`` (resume a ledger written by this coordinator) or
    ``read_only`` (no lock, no appends; ``refresh`` follows a file another process writes).
    """

    def __init__(self, ledger: Ledger, *, writer: bool) -> None:
        self._ledger = ledger
        self._writer = writer
        self._lock = threading.RLock()
        self._closed = False
        # FREEZE_ONCE kinds already present in the file (an earlier process may have written them).
        self._frozen: set[str] = {e.kind for e in ledger.events() if e.kind in LabKind.FREEZE_ONCE}

    # ---- construction ---------------------------------------------------------------------------
    @classmethod
    def create(cls, dir: Path | str, *, clock: Clock = time.time, fsync: bool = True) -> "LabLedger":
        try:
            return cls(Ledger.create(dir, clock=clock, fsync=fsync), writer=True)
        except FileExistsError as exc:
            raise LedgerError(f"{dir} already contains a ledger; use LabLedger.open to resume") from exc
        except LedgerLocked as exc:
            raise LedgerError(str(exc)) from exc

    @classmethod
    def open(cls, dir: Path | str, *, clock: Clock = time.time, fsync: bool = True) -> "LabLedger":
        try:
            return cls(Ledger.open(dir, clock=clock, fsync=fsync), writer=True)
        except LedgerLocked as exc:
            raise LedgerError(str(exc)) from exc

    @classmethod
    def read_only(cls, dir: Path | str) -> "LabLedger":
        return cls(Ledger.read_only(dir), writer=False)

    # ---- LedgerWriter ---------------------------------------------------------------------------
    def append(self, kind: str, actor: str, record: Any) -> Event:
        """Append one contract record. Raises ``LedgerError`` (a ``LedgerPayloadError`` for payload problems) before any write when the
        kind, payload class, payload content, actor or the freeze-once rule is violated."""
        if not self._writer:
            raise LedgerError("this ledger was opened read-only")
        if not isinstance(kind, str) or kind not in LabKind.all():
            raise LedgerError(f"{kind!r} is not a lab event kind")
        cls = PAYLOAD_TYPES[kind]
        if not isinstance(record, cls):
            raise LedgerPayloadError(f"{kind}: payload must be a {cls.__name__}, not {type(record).__name__}")
        if not isinstance(actor, str) or not ACTOR_PATTERN.match(actor):
            raise LedgerError(f"{actor!r} is not a lab actor (coordinator, scorer, outer or trial:<id>)")
        try:
            data = record.to_dict()
            assert_ledger_safe(data)
        except ContractError as exc:
            raise LedgerPayloadError(f"{kind}: {exc}") from exc
        with self._lock:
            if self._closed:
                raise LedgerError("this ledger is closed")
            if kind in LabKind.FREEZE_ONCE and kind in self._frozen:
                raise LedgerError(f"{kind} may be appended once; it is already in this ledger")
            try:
                event = self._ledger.append(kind, actor, data)
            except PermissionError as exc:
                raise LedgerError(str(exc)) from exc
            if kind in LabKind.FREEZE_ONCE:
                self._frozen.add(kind)
            return event

    def anchor(self) -> LedgerAnchor:
        seq, hash16 = self._ledger.head_anchor()
        return LedgerAnchor(seq, hash16)

    def seal(self) -> None:
        """Close the writer and make the file read-only (``Ledger.seal``). Idempotent."""
        with self._lock:
            self._closed = True
            self._ledger.seal()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._ledger.close()

    # ---- reading --------------------------------------------------------------------------------
    def events(self, since_seq: int = 0) -> list[Event]:
        return self._ledger.events(since_seq)

    def verify(self) -> tuple[bool, Optional[int]]:
        """``(ok, first_bad_seq)`` for the hash chain (sequence numbers, links, hashes, one run id)."""
        return self._ledger.verify_chain()

    def refresh(self) -> int:
        """Read-only openers: pick up events another process appended. Returns the number of new events."""
        return self._ledger.refresh()

    @property
    def path(self) -> Path:
        return self._ledger.path

    @property
    def run_id(self) -> str:
        return self._ledger.run_id

    @property
    def writable(self) -> bool:
        return self._writer and not self._closed

    def __enter__(self) -> "LabLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
