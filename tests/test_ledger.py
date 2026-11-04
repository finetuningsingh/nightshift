"""Acceptance A8: ledger verification detects tampering (spec 3.2)."""

import json

from nightshift.ledger import EventLedger, read_events, verify_ledger


def make_ledger(tmp_path):
    ledger = EventLedger(tmp_path / "ledger.jsonl")
    ledger.append("run_started", "controller", {"run": "r"})
    ledger.append("spend_reserved", "controller", {"usd": 1.5})
    ledger.append("rental_destroyed", "provider", {"rental_id": "x", "fake": True})
    return ledger


def rewrite(path, events):
    path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events))


def test_valid_ledger_verifies_and_chains(tmp_path):
    ledger = make_ledger(tmp_path)
    result = verify_ledger(ledger.path)
    assert result.ok and result.count == 3
    events = read_events(ledger.path)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["prev_hash"] == "0" * 64
    assert events[1]["prev_hash"] == events[0]["hash"]
    assert all(e["timestamp"].endswith("Z") for e in events)


def test_modified_event_is_detected_at_its_sequence(tmp_path):
    ledger = make_ledger(tmp_path)
    events = read_events(ledger.path)
    events[1]["payload"]["usd"] = 999.0
    rewrite(ledger.path, events)
    result = verify_ledger(ledger.path)
    assert not result.ok and result.broken_seq == 2


def test_deleted_event_is_detected(tmp_path):
    ledger = make_ledger(tmp_path)
    events = read_events(ledger.path)
    rewrite(ledger.path, [events[0], events[2]])
    result = verify_ledger(ledger.path)
    assert not result.ok and result.broken_seq == 2


def test_reordered_event_is_detected(tmp_path):
    ledger = make_ledger(tmp_path)
    events = read_events(ledger.path)
    rewrite(ledger.path, [events[0], events[2], events[1]])
    result = verify_ledger(ledger.path)
    assert not result.ok and result.broken_seq == 2


def test_cli_verify_exit_codes(tmp_path):
    from nightshift.cli import main

    make_ledger(tmp_path)
    assert main(["ledger", "verify", str(tmp_path)]) == 0
