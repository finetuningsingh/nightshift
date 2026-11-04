"""Append-only hash-chained JSONL event ledger (spec 3.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .errors import LedgerError
from .hashing import canonical_json, sha256_hex

GENESIS_HASH = "0" * 64


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def event_hash(event_without_hash: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(event_without_hash))


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    count: int
    broken_seq: int | None = None
    message: str = ""

    def __bool__(self) -> bool:
        return self.ok


class EventLedger:
    """Append-only JSONL ledger. Each event chains to the previous hash."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._prev_hash = self._read_tail()

    def _read_tail(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, GENESIS_HASH
        last: dict[str, Any] | None = None
        count = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            count += 1
            try:
                last = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"corrupt ledger line {count} in {self.path}") from exc
        if last is None:
            return 0, GENESIS_HASH
        return int(last.get("seq", count)), str(last.get("hash", GENESIS_HASH))

    @property
    def next_seq(self) -> int:
        return self._seq + 1

    @property
    def last_hash(self) -> str:
        return self._prev_hash

    def append(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "seq": self._seq + 1,
            "timestamp": timestamp or utc_now_iso(),
            "type": event_type,
            "actor": actor,
            "payload": dict(payload or {}),
            "prev_hash": self._prev_hash,
        }
        # Hash is over the canonical JSON of the event without `hash`.
        try:
            event["hash"] = event_hash(event)
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"event payload is not JSON-serialisable: {exc}") from exc
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event) + "\n")
        self._seq = event["seq"]
        self._prev_hash = event["hash"]
        return event

    def events(self) -> Iterator[dict[str, Any]]:
        yield from read_events(self.path)

    def verify(self) -> VerifyResult:
        return verify_ledger(self.path)


def read_events(path: str | Path) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        raise LedgerError(f"ledger not found: {ledger_path}")
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LedgerError(f"corrupt ledger line {line_no} in {ledger_path}") from exc
    return events


def verify_ledger(path: str | Path) -> VerifyResult:
    """Recompute the chain; report the first broken sequence."""
    ledger_path = Path(path)
    if not ledger_path.exists():
        return VerifyResult(False, 0, 1, f"ledger not found: {ledger_path}")
    prev_hash = GENESIS_HASH
    expected_seq = 1
    count = 0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        count += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return VerifyResult(False, count, expected_seq, "unparseable event line")
        if not isinstance(event, dict):
            return VerifyResult(False, count, expected_seq, "event is not an object")
        seq = event.get("seq")
        if seq != expected_seq:
            return VerifyResult(
                False, count, expected_seq, f"expected seq {expected_seq}, found {seq!r}"
            )
        if event.get("prev_hash") != prev_hash:
            return VerifyResult(False, count, expected_seq, "prev_hash does not chain")
        claimed = event.get("hash")
        without_hash = {k: v for k, v in event.items() if k != "hash"}
        if not isinstance(claimed, str) or event_hash(without_hash) != claimed:
            return VerifyResult(False, count, expected_seq, "event hash mismatch")
        prev_hash = claimed
        expected_seq += 1
    return VerifyResult(True, count, None, "ok")
