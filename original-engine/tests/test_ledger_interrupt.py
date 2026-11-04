"""A Ctrl-C (KeyboardInterrupt) inside ``Ledger.append`` must never break the hash chain.

Before the fix, an interrupt that landed after the line had reached the file but before the in-memory update left the event on
disk and unknown to memory, so the NEXT append reused the same ``seq`` and ``prev_hash``: ``nightshift verify`` then reported a
broken chain for a run whose only "tampering" was the operator pressing Ctrl-C. The file is authoritative, so after an
interrupted append the in-memory tail is restored, a complete hash-consistent line is adopted, and a torn tail is truncated.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nightshift import ledger as ledger_module
from nightshift.ledger import Ledger, verify_events



class _InterruptingList(list):
    """A list whose ``append`` raises KeyboardInterrupt ONCE, after it has appended (the narrowest possible window)."""

    def __init__(self, items, armed: bool = True):
        super().__init__(items)
        self.armed = armed

    def append(self, item):
        super().append(item)
        if self.armed:
            self.armed = False
            raise KeyboardInterrupt


class InterruptedAppendTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run"
        self.ledger = Ledger.create(self.run_dir)
        self.addCleanup(self.ledger.close)
        for i in range(3):
            self.ledger.append("note", "test", {"i": i})

    def assert_chain_sound_in_memory_and_on_disk(self, expected_events: int):
        self.assertEqual(len(self.ledger.events()), expected_events)
        self.assertEqual([e.seq for e in self.ledger.events()], list(range(1, expected_events + 1)))
        self.assertEqual(self.ledger.verify_chain(), (True, None))
        reread = Ledger.read_only(self.run_dir)
        self.assertEqual(len(reread.events()), expected_events)
        self.assertEqual(verify_events(reread.events()), (True, None))
        self.assertEqual(reread.head_anchor(), self.ledger.head_anchor())

    def test_an_interrupt_during_fsync_keeps_the_durable_event_and_the_next_append_chains(self):
        """The line is fully on disk; memory must adopt it, not forget it and reuse its seq."""
        with mock.patch.object(ledger_module.os, "fsync", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.ledger.append("note", "test", {"i": 3})
        self.assert_chain_sound_in_memory_and_on_disk(4)
        self.ledger.append("note", "test", {"i": 4})
        self.assert_chain_sound_in_memory_and_on_disk(5)

    def test_an_interrupt_in_the_middle_of_the_write_leaves_no_torn_tail(self):
        real_write = os.write

        def half_then_interrupt(fd, data):
            real_write(fd, bytes(data)[: len(data) // 2])
            raise KeyboardInterrupt

        with mock.patch.object(ledger_module.os, "write", side_effect=half_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.ledger.append("note", "test", {"i": 3})
        self.assert_chain_sound_in_memory_and_on_disk(3)          # the half-written event is gone, not adopted
        self.ledger.append("note", "test", {"i": 3})
        self.assert_chain_sound_in_memory_and_on_disk(4)
        self.ledger.close()
        reopened = Ledger.open(self.run_dir)
        self.addCleanup(reopened.close)
        self.assertFalse(reopened.recovered_torn_tail)             # nothing was left for the loader to repair
        self.assertEqual(len(reopened.events()), 4)

    def test_an_interrupt_between_the_file_and_the_in_memory_update_does_not_fork_the_chain(self):
        """The original bug: event on disk, memory not told, next append reuses seq and prev_hash."""
        self.ledger._events = _InterruptingList(self.ledger._events)
        with self.assertRaises(KeyboardInterrupt):
            self.ledger.append("note", "test", {"i": 3})
        self.assert_chain_sound_in_memory_and_on_disk(4)
        self.ledger.append("note", "test", {"i": 4})
        self.assert_chain_sound_in_memory_and_on_disk(5)

    def test_interrupt_before_anything_was_written_changes_nothing(self):
        with mock.patch.object(ledger_module.os, "write", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.ledger.append("note", "test", {"i": 3})
        self.assert_chain_sound_in_memory_and_on_disk(3)
        self.ledger.append("note", "test", {"i": 3})
        self.assert_chain_sound_in_memory_and_on_disk(4)

    def test_repeated_interrupts_at_every_point_never_break_the_chain(self):
        real_write, real_fsync = os.write, os.fsync
        kinds = ("before_write", "half_write", "fsync", "memory")
        for round_no in range(8):
            kind = kinds[round_no % len(kinds)]
            if kind == "before_write":
                patcher = mock.patch.object(ledger_module.os, "write", side_effect=KeyboardInterrupt)
            elif kind == "half_write":
                def half(fd, data, _w=real_write):
                    _w(fd, bytes(data)[: max(1, len(data) // 3)])
                    raise KeyboardInterrupt
                patcher = mock.patch.object(ledger_module.os, "write", side_effect=half)
            elif kind == "fsync":
                patcher = mock.patch.object(ledger_module.os, "fsync", side_effect=KeyboardInterrupt)
            else:
                self.ledger._events = _InterruptingList(self.ledger._events)
                patcher = mock.patch.object(ledger_module.os, "fsync", side_effect=real_fsync)
            with patcher:
                with self.assertRaises(KeyboardInterrupt):
                    self.ledger.append("note", "test", {"round": round_no, "kind": kind})
            self.ledger._events = list(self.ledger._events)
            self.assertEqual(self.ledger.verify_chain(), (True, None), kind)
            self.ledger.append("note", "test", {"after": round_no})
            reread = Ledger.read_only(self.run_dir)
            self.assertEqual(verify_events(reread.events()), (True, None), kind)
            self.assertEqual(len(reread.events()), len(self.ledger.events()), kind)

    def test_an_ordinary_oserror_is_recovered_the_same_way(self):
        with mock.patch.object(ledger_module.os, "fsync", side_effect=OSError("disk went away")):
            with self.assertRaises(OSError):
                self.ledger.append("note", "test", {"i": 3})
        self.assertEqual(self.ledger.verify_chain(), (True, None))
        self.ledger.append("note", "test", {"i": 4})
        self.assertEqual(verify_events(Ledger.read_only(self.run_dir).events()), (True, None))


if __name__ == "__main__":
    unittest.main()
