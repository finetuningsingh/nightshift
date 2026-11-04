"""Tests for nightshift.workflows.engine over the real sandbox and grader (tests/fixtures/mini_task) with fake agents.

Run: python3 -m unittest tests.test_engine -v
"""
import dataclasses
import inspect
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from nightshift import grading, sandbox, secrets
from nightshift.task import trusted_hash
from nightshift.types import (BudgetExceeded, BudgetLimits, ExperimentPlan, ExpStatus, HaltRequested, ImplementResult, Kind, LaneSpec,
                              ModelError, Outcome,
                              ReflectContext, ReflectionResult, ReviewContext, ReviewVerdict, SimulatedCrash, ProtectedPathViolation, short_hash,
                              DesignContext, ImplementContext)
from nightshift.workflows import engine as engine_module
from nightshift.workflows import gates
from nightshift.workflows.engine import BaselineFailed
from tests.support.fake_suite import (ALWAYS_INFRA, CRASHING, DEFECT_MARKER, MINI_TASK_DIR, PATCHES, SEALED_SENTINEL, STAGES_FULL,
                                      STAGES_MIN, EngineHarness, FakeAgents, Patch, binned_model, copy_mini_task)
from tests.support.synthetic import RunBuilder

KEYS = {
    Kind.EXP_PROPOSED: {"exp_id", "lane", "hypothesis", "plan", "parent_exp_id", "base", "hyp_fingerprint", "replicate_of",
                        "scripted_id", "planted_fault"},
    Kind.EXP_BLOCKED: {"exp_id", "lane", "hypothesis", "hyp_fingerprint", "reason", "detail", "dup_of"},
    Kind.EXP_WORKSPACE: {"exp_id", "path", "base_hash"},
    Kind.EXP_IMPLEMENTED: {"exp_id", "worker", "diff_hash", "files_changed", "diff_summary", "idem_key", "attempt", "steps",
                           "tree_hash", "plan_summary"},
    Kind.EXP_PROTECTED: {"exp_id", "paths", "kind"},
    Kind.EXP_RUN_STARTED: {"exp_id", "run_idx", "purpose", "seed", "cmd", "pid", "attempt", "isolation"},
    Kind.EXP_HEARTBEAT: {"exp_id", "run_idx", "output_bytes", "cpu_sec"},
    Kind.EXP_RUN_FINISHED: {"exp_id", "run_idx", "purpose", "seed", "exit_code", "duration_sec", "classification", "stalled",
                            "stdout_tail", "stderr_tail", "note"},
    Kind.EXP_GRADED: {"exp_id", "purpose", "split", "grade", "grader_hash", "seeds", "conditions_hash", "outputs_hash"},
    Kind.EXP_REVIEWED: {"exp_id", "review_round", "reviewer", "reviewer_model", "implementer_model", "independent",
                        "independent_basis", "verdict", "findings"},
    Kind.EXP_REPAIRED: {"exp_id", "repair_round", "diff_hash", "addressed"},
    Kind.EXP_RECORDED: {"exp_id", "phase", "outcome", "delta", "vs_exp_id", "noise_margin", "train_delta", "train_margin",
                        "overfit_suspect", "replicated", "sigma_prior", "n", "s_eff", "t", "sealed_delta", "sealed_margin", "note"},
    Kind.INCUMBENT_PROMOTED: {"exp_id", "score", "prev_exp_id", "replicated", "replication_exp_ids", "tree_hash", "sealed_score",
                              "sealed_delta", "sealed_margin"},
    Kind.WATCHDOG_STALLED: {"exp_id", "run_idx", "idle_sec"},
}


_SNAPSHOT = {}


def setUpModule():
    """One real baseline for the whole module: tests that start from a recorded baseline copy it instead of re-running it."""
    _SNAPSHOT["tmp"] = tempfile.TemporaryDirectory()
    _SNAPSHOT["dir"] = EngineHarness.baseline_snapshot(Path(_SNAPSHOT["tmp"].name) / "snap")


def tearDownModule():
    _SNAPSHOT["tmp"].cleanup()


def crash_when(engine, predicate):
    """Make the engine raise SimulatedCrash instead of writing the first event for which ``predicate(kind, data)`` holds."""
    real, fired = engine._append, []

    def wrapper(kind, actor, data):
        if not fired and predicate(kind, data):
            fired.append(kind)
            raise SimulatedCrash("test crash before " + kind)
        return real(kind, actor, data)

    engine._append = wrapper
    return fired


class EngineCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self._harnesses = []

    def make(self, sub="h", **kw) -> EngineHarness:
        h = EngineHarness(self.root / sub, **kw)
        self._harnesses.append(h)
        self.addCleanup(h.close)
        return h

    def with_baseline(self, **kw) -> EngineHarness:
        """A harness whose baseline is recorded. The default task starts from the shared snapshot; any other task (or a crash
        hook) runs its own real baseline."""
        if kw.get("task") is None and kw.get("crash_after_runs") is None and kw.get("write_run_started", True):
            return self.make(snapshot=_SNAPSHOT["dir"], **kw)
        h = self.make(**kw)
        h.baseline()
        return h

    @staticmethod
    def lane_events(h, exp_id):
        return [e for e in h.events(exp_id=exp_id) if e.kind.startswith("experiment.")]

    @staticmethod
    def first(h, kind, exp_id=None, **match):
        for e in h.events(kind, exp_id=exp_id):
            if all(e.data.get(k) == v for k, v in match.items()):
                return e
        raise AssertionError(f"no {kind} event for {exp_id} {match}")

    def charged(self, h, kind):
        return [e for e in h.events(Kind.BUDGET_CHARGED) if e.data.get("kind") == kind and e.data.get("amount", 0) > 0]


# =====================================================================================================
# baseline
# =====================================================================================================
class BaselineTests(EngineCase):
    def test_baseline_end_to_end(self):
        h = self.make()
        with mock.patch.object(grading, "evaluate", wraps=grading.evaluate) as spy:
            res = h.baseline()
        self.assertEqual((res.exp_id, res.lane, res.outcome, res.status), ("e000", "baseline", "baseline", ExpStatus.RECORDED))
        kinds = [e.kind for e in self.lane_events(h, "e000")]
        self.assertEqual(kinds, [Kind.EXP_PROPOSED, Kind.EXP_WORKSPACE] + [Kind.EXP_RUN_STARTED, Kind.EXP_RUN_FINISHED] * 3
                         + [Kind.EXP_GRADED, Kind.EXP_RECORDED])
        proposed = self.first(h, Kind.EXP_PROPOSED).data
        self.assertEqual((proposed["lane"], proposed["scripted_id"], proposed["base"], proposed["planted_fault"]),
                         ("baseline", None, "template", None))
        ws = self.first(h, Kind.EXP_WORKSPACE).data
        self.assertEqual(ws["path"], "exp/e000/workspace")           # relative posix, never absolute
        graded = self.first(h, Kind.EXP_GRADED).data
        self.assertEqual((graded["purpose"], graded["split"], graded["seeds"]), ("baseline", "dev", [0, 1, 2]))
        self.assertTrue(graded["grade"]["valid"])
        self.assertEqual(graded["grader_hash"], trusted_hash(h.task))
        self.assertEqual(graded["grader_hash"], h.view().run["grader_hash"])
        self.assertAlmostEqual(res.score, graded["grade"]["score"])
        rec = self.first(h, Kind.EXP_RECORDED).data
        self.assertEqual((rec["phase"], rec["outcome"], rec["delta"], rec["replicated"]), ("lane", "baseline", None, False))
        view = h.view()
        self.assertEqual((view.baseline_id, view.incumbent_id), ("e000", "e000"))
        reserve = [e for e in h.events(Kind.BUDGET_CHARGED, exp_id="e000")]
        self.assertEqual([(e.data["amount"], e.data["purpose"]) for e in reserve], [(0, "baseline")])
        # evaluate was told which grader hash the run started with, and which tree to expect
        call = spy.call_args
        self.assertEqual(call.kwargs["expected_grader_hash"], trusted_hash(h.task))
        self.assertEqual(call.kwargs["purpose"], "baseline")
        self.assertEqual(call.kwargs["expected_tree_hash"], sandbox.tree_hash(h.run_dir / "exp/e000/workspace", h.task))

    def test_run_started_records_the_isolation_that_the_sandbox_reports(self):
        h = self.make(overrides={})                                       # isolation "auto"
        h.baseline()
        isolations = {e.data["isolation"] for e in h.events(Kind.EXP_RUN_STARTED)}
        self.assertEqual(isolations, {sandbox.available_isolation()})

    def test_run_started_labels_a_forced_mode_with_what_the_sandbox_applied(self):
        """Regression: the configured override was echoed back, so ``userns`` was recorded on a machine whose runs got ``none``."""
        for forced, proved, expected in (("userns", "none", "none"), ("userns", "unshare-net", "unshare-net"), ("none", "userns", "none")):
            with self.subTest(forced=forced, proved=proved):
                h = self.make(f"label-{forced}-{proved}", overrides={"isolation": forced})
                with mock.patch.object(sandbox, "_probed", return_value={"level": proved, "reason": "test"}), \
                        mock.patch.object(sandbox, "_net_prefix_for", return_value=["unshare", "-n"]), \
                        mock.patch.dict(os.environ, {"NIGHTSHIFT_ISOLATION": ""}):
                    self.assertEqual(h.engine._isolation_label(), expected)

    def test_run_started_label_follows_the_environment_override_too(self):
        h = self.make("label-env", overrides={})
        with mock.patch.dict(os.environ, {"NIGHTSHIFT_ISOLATION": "none"}):
            self.assertEqual(h.engine._isolation_label(), "none")

    def test_a_real_baseline_with_a_forced_stronger_mode_is_labelled_with_the_applied_mode(self):
        h = self.make("label-real", overrides={"isolation": "userns"})
        h.baseline()
        labels = {e.data["isolation"] for e in h.events(Kind.EXP_RUN_STARTED)}
        self.assertEqual(labels, {sandbox.effective_isolation("userns")})

    def test_baseline_tree_is_frozen_read_only(self):
        h = self.with_baseline()
        frozen = h.run_dir / "exp" / "e000" / "frozen"
        self.assertEqual(sandbox.tree_hash(frozen, h.task), sandbox.tree_hash(h.run_dir / "exp/e000/workspace", h.task))
        mode = (frozen / "model.py").stat().st_mode
        self.assertEqual(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), 0)

    def test_baseline_is_idempotent(self):
        h = self.with_baseline()
        before = len(h.ledger.events())
        again = h.baseline()
        self.assertEqual((again.exp_id, again.outcome), ("e000", "baseline"))
        self.assertEqual(len(h.ledger.events()), before)

    def test_baseline_crash_is_recorded_failed_and_raises(self):
        task = copy_mini_task(self.root / "bad", template_model=CRASHING)
        h = self.make(task=task)
        with self.assertRaises(BaselineFailed) as ctx:
            h.baseline()
        self.assertEqual(ctx.exception.outcome, Outcome.FAILED)
        exp = h.view().exp("e000")
        self.assertEqual((exp.status, exp.outcome), (ExpStatus.RECORDED, Outcome.FAILED))
        self.assertIsNone(h.view().baseline_id)

    def test_baseline_infra_failure_is_retried_once_then_infra_error(self):
        task = copy_mini_task(self.root / "infra", template_model=ALWAYS_INFRA)
        h = self.make(task=task)
        with self.assertRaises(BaselineFailed) as ctx:
            h.baseline()
        self.assertEqual(ctx.exception.outcome, Outcome.INFRA)
        attempts = [e.data["attempt"] for e in h.events(Kind.EXP_RUN_STARTED)]
        self.assertEqual(attempts, [1, 1, 1, 2, 2, 2])
        self.assertEqual(h.view().exp("e000").outcome, Outcome.INFRA)
        self.assertEqual(h.events(Kind.EXP_GRADED), [])               # an infrastructure failure grades nothing

    def test_baseline_resume_replaces_a_half_copied_frozen_tree(self):
        h = self.make()
        crash_when(h.engine, lambda k, d: k == Kind.EXP_RUN_STARTED)
        with self.assertRaises(SimulatedCrash):
            h.baseline()
        frozen = h.run_dir / "exp/e000/frozen"
        (frozen / "data.py").unlink()                                       # a kill in the middle of the copy
        h.restart()
        res = h.engine.resume_pending()[0]
        self.assertEqual(res.outcome, "baseline")
        self.assertTrue((frozen / "data.py").is_file())
        self.assertEqual(sandbox.tree_hash(frozen, h.task), sandbox.tree_hash(h.run_dir / "exp/e000/workspace", h.task))

    def test_baseline_task_hash_mismatch_is_a_trusted_violation(self):
        h = self.make(write_run_started=False)
        h.ledger.append(Kind.RUN_STARTED, "engine", {"task": "x", "backend": "fake", "limits": h.limits.to_dict(),
                                                     "task_hash": "0" * 12, "grader_hash": trusted_hash(h.task)})
        with self.assertRaises(BaselineFailed) as ctx:
            h.baseline()
        self.assertEqual(ctx.exception.outcome, Outcome.INVALID)
        self.assertEqual(self.first(h, Kind.EXP_PROTECTED).data["kind"], "trusted")
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED), [])


# =====================================================================================================
# explore lanes
# =====================================================================================================
class LaneTests(EngineCase):
    def test_improving_lane_events_and_payloads(self):
        h = self.with_baseline()
        res = h.lane("improve")
        view = h.view()
        base = view.baseline.primary_grade
        self.assertEqual((res.exp_id, res.lane, res.status, res.outcome, res.review_verdict),
                         ("e001", "A", ExpStatus.RECORDED, "improved", "accept"))
        self.assertGreater(res.delta, 0.05)
        self.assertAlmostEqual(res.score, base.score - res.delta, places=9)
        self.assertFalse(res.replicated)
        kinds = [e.kind for e in self.lane_events(h, "e001")]
        self.assertEqual(kinds, [Kind.EXP_PROPOSED, Kind.EXP_WORKSPACE, Kind.EXP_IMPLEMENTED] + [Kind.EXP_RUN_STARTED, Kind.EXP_RUN_FINISHED] * 3
                         + [Kind.EXP_GRADED, Kind.EXP_REVIEWED, Kind.EXP_RECORDED])
        proposed = self.first(h, Kind.EXP_PROPOSED, "e001")
        self.assertEqual(proposed.actor, "designer:A")
        self.assertEqual((proposed.data["parent_exp_id"], proposed.data["base"], proposed.data["scripted_id"],
                          proposed.data["planted_fault"], proposed.data["replicate_of"]), ("e000", "e000", "improve", None, None))
        self.assertEqual(proposed.data["hyp_fingerprint"], h.memory.fingerprint(proposed.data["hypothesis"]))
        implemented = self.first(h, Kind.EXP_IMPLEMENTED, "e001")
        ws_event = self.first(h, Kind.EXP_WORKSPACE, "e001")
        self.assertEqual(implemented.actor, "implementer:A")
        self.assertEqual(implemented.data["idem_key"], short_hash(f"{ws_event.data['base_hash']}|{implemented.data['diff_hash']}", 16))
        self.assertEqual(implemented.data["tree_hash"], sandbox.tree_hash(h.run_dir / "exp/e001/workspace", h.task))
        self.assertEqual((implemented.data["worker"], implemented.data["files_changed"], implemented.data["attempt"]),
                         ("fake-implementer", ["model.py"], 1))
        runs = h.events(Kind.EXP_RUN_STARTED, exp_id="e001")
        self.assertEqual([(r.data["purpose"], r.data["seed"], r.data["run_idx"], r.data["attempt"]) for r in runs],
                         [("primary", 0, 0, 1), ("primary", 1, 1, 1), ("primary", 2, 2, 1)])
        self.assertEqual(runs[0].data["cmd"], h.task.run_cmd)
        reviewed = self.first(h, Kind.EXP_REVIEWED, "e001")
        self.assertEqual(reviewed.actor, "reviewer:A")
        self.assertTrue(reviewed.data["independent"])
        self.assertNotEqual(reviewed.data["reviewer"], implemented.data["worker"])
        rec = self.first(h, Kind.EXP_RECORDED, "e001").data
        t = grading.T95[2]
        self.assertEqual((rec["phase"], rec["outcome"], rec["vs_exp_id"], rec["replicated"], rec["n"]), ("lane", "improved", "e000", False, 3))
        self.assertAlmostEqual(rec["sigma_prior"], base.std)                  # the baseline's per-seed std
        self.assertAlmostEqual(rec["t"], t)
        self.assertGreaterEqual(rec["s_eff"], rec["sigma_prior"])
        self.assertAlmostEqual(rec["noise_margin"], max(h.task.noise_margin, t * rec["s_eff"] / math.sqrt(3)))
        self.assertAlmostEqual(rec["delta"], res.delta)
        self.assertIsNone(rec["sealed_delta"])
        # not promoted: it waits for replication
        self.assertEqual(view.incumbent_id, "e000")
        self.assertTrue(view.exp("e001").awaiting_replication)
        exp_dir = h.run_dir / "exp" / "e001"
        self.assertIn("model.py", (exp_dir / "diff.patch").read_text())
        self.assertIn("round 0", (exp_dir / "worker.log").read_text())
        self.assertIn("e001", (h.run_dir / "memory.md").read_text())

    def test_every_payload_has_exactly_the_contract_keys(self):
        h = self.with_baseline()
        h.lane("defective")
        h.lane("tamper")
        h.lane("defective", hypothesis="a second, differently worded try")      # blocked as an idempotency duplicate
        h.engine.replicate("e001")
        seen = set()
        for event in h.ledger.events():
            if event.kind in KEYS:
                seen.add(event.kind)
                self.assertEqual(set(event.data), KEYS[event.kind], event.kind)
        for kind in (Kind.EXP_PROPOSED, Kind.EXP_BLOCKED, Kind.EXP_WORKSPACE, Kind.EXP_IMPLEMENTED, Kind.EXP_PROTECTED,
                     Kind.EXP_RUN_STARTED, Kind.EXP_RUN_FINISHED, Kind.EXP_GRADED, Kind.EXP_REVIEWED, Kind.EXP_REPAIRED,
                     Kind.EXP_RECORDED, Kind.INCUMBENT_PROMOTED):
            self.assertIn(kind, seen)

    def test_watchdog_and_heartbeat_payloads(self):
        h = self.with_baseline(overrides={"isolation": "none", "stall_sec": 0.3, "heartbeat_sec": 0.05, "run_timeout_sec": 10})
        res = h.lane("hang")
        self.assertEqual(res.outcome, Outcome.FAILED)
        stalled = h.events(Kind.WATCHDOG_STALLED)
        self.assertGreaterEqual(len(stalled), 1)
        for e in stalled:
            self.assertEqual((e.actor, set(e.data)), ("watchdog", KEYS[Kind.WATCHDOG_STALLED]))
        beats = h.events(Kind.EXP_HEARTBEAT)
        self.assertTrue(beats)
        self.assertEqual(set(beats[0].data), KEYS[Kind.EXP_HEARTBEAT])
        finished = h.events(Kind.EXP_RUN_FINISHED, exp_id="e001")
        self.assertTrue(all(e.data["stalled"] and e.data["classification"] == "timeout" for e in finished))
        self.assertTrue(all(len(e.data["stdout_tail"].encode()) <= 4096 for e in finished))

    def test_worse_lane_records_worse_and_a_dev_only_lesson(self):
        h = self.with_baseline()
        res = h.lane("worse")
        self.assertEqual(res.outcome, Outcome.WORSE)
        self.assertLess(res.delta, 0)
        lesson = h.view().lessons[-1]
        self.assertEqual((lesson["classification"], lesson["source_exps"], lesson["kind"]), ("model", ["e001"], "observation"))
        self.assertIn("worse", lesson["text"])
        self.assertIn(f"{res.delta:+.4g}", lesson["text"])
        self.assertEqual(self.first(h, Kind.LESSON).actor, "engine")

    def test_improved_lane_writes_no_lesson(self):
        h = self.with_baseline()
        h.lane("improve")
        self.assertEqual(h.view().lessons, [])

    def test_noop_is_no_gain(self):
        h = self.with_baseline()
        res = h.lane("noop")
        self.assertEqual(res.outcome, Outcome.NO_GAIN)
        self.assertAlmostEqual(res.delta, 0.0, places=9)
        self.assertEqual(len(h.view().lessons), 1)

    def test_overfit_is_flagged_with_train_numbers(self):
        h = self.with_baseline()
        res = h.lane("overfit")
        rec = self.first(h, Kind.EXP_RECORDED, "e001").data
        self.assertEqual(res.outcome, Outcome.NO_GAIN)
        self.assertTrue(rec["overfit_suspect"])
        self.assertGreater(rec["train_delta"], rec["train_margin"])
        self.assertIn("overfitting", h.view().lessons[-1]["text"])
        reviews = h.agents.contexts_of(ReviewContext)
        self.assertTrue(reviews[-1].grade_summary["overfit_suspect"])

    def test_design_stage_is_optional(self):
        h = self.with_baseline()
        res = h.lane("improve", stages=STAGES_MIN)
        self.assertEqual(h.agents.design_calls, 0)
        self.assertEqual(res.outcome, "improved")
        proposed = self.first(h, Kind.EXP_PROPOSED, "e001").data
        self.assertEqual(proposed["plan"], proposed["hypothesis"])        # no designer: the plan is the hypothesis
        h.lane("improve_mid", stages=STAGES_FULL)
        self.assertEqual(h.agents.design_calls, 1)
        ctx = h.agents.contexts_of(DesignContext)[0]
        self.assertIn("model.py", ctx.editable_files)
        self.assertIn("e000", ctx.incumbent_summary)
        self.assertNotIn("planted", str(ctx).lower())

    def test_planted_fault_is_only_written_for_the_scripted_backend(self):
        with mock.patch("nightshift.agents.scripted.planted_fault_for", return_value="tamper") as fault:
            h = self.with_baseline(sub="scripted", backend="scripted")
            h.lane("improve")
            self.assertEqual(self.first(h, Kind.EXP_PROPOSED, "e001").data["planted_fault"], "tamper")
            self.assertEqual(fault.call_args.args, (h.task, "improve", False))
            fake = self.with_baseline(sub="fake")
            fake.lane("improve")
            self.assertIsNone(self.first(fake, Kind.EXP_PROPOSED, "e001").data["planted_fault"])
        with mock.patch("nightshift.agents.scripted.planted_fault_for", side_effect=RuntimeError("bank broken")):
            h2 = self.with_baseline(sub="broken", backend="scripted")
            self.assertEqual(h2.lane("improve").outcome, "improved")     # display metadata never breaks a lane

    def test_overrides_apply_to_the_engines_own_copy_of_the_task(self):
        h = self.with_baseline(overrides={"isolation": "none", "run_timeout_sec": 7, "grade_timeout_sec": 8, "stall_sec": 1.5})
        self.assertEqual((h.engine.task.run_timeout_sec, h.engine.task.grade_timeout_sec, h.engine.task.stall_sec), (7.0, 8.0, 1.5))
        self.assertEqual((h.task.run_timeout_sec, h.task.stall_sec), (20.0, 10.0))      # the caller's task is not mutated
        self.assertEqual(h.lane("improve").outcome, "improved")

    def test_a_streak_ended_by_two_lanes_at_once_reflects_once(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=3, stall_after=2)
        h = self.with_baseline(limits=limits)
        h.agents.patches["noop3"] = Patch({"model.py": PATCHES["noop"].files["model.py"] + "\n# v3\n"})
        h.lanes(["noop", "noop2", "noop3"])
        self.assertEqual(h.agents.reflect_calls, 1)

    def test_the_public_signatures_are_the_contracts(self):
        engine_cls = engine_module.WorkflowEngine
        init = inspect.signature(engine_cls.__init__).parameters
        self.assertEqual([n for n in init if n != "self"], ["task", "run_dir", "ledger", "gate", "memory", "suite", "auditor", "limits",
                                                            "clock", "overrides", "crash_after_runs"])
        self.assertTrue(all(p.kind is inspect.Parameter.KEYWORD_ONLY for n, p in init.items() if n != "self"))
        self.assertIs(init["clock"].default, __import__("time").time)
        self.assertIsNone(init["overrides"].default)
        self.assertIsNone(init["crash_after_runs"].default)
        for name, params in (("run_baseline", []), ("run_lane", ["stages", "lane"]), ("run_lanes", ["stages", "lanes"]),
                             ("replicate", ["exp_id"]), ("reflect", ["trigger"]), ("reconcile", []), ("resume_pending", []),
                             ("regenerate_memory", [])):
            self.assertEqual([n for n in inspect.signature(getattr(engine_cls, name)).parameters if n != "self"], params, name)
        self.assertTrue(issubclass(BaselineFailed, engine_module.NightshiftError))

    def test_a_planted_fault_label_changes_nothing_the_engine_decides(self):
        outcomes = []
        for sub, label in (("with_label", "tamper"), ("without_label", None)):
            with mock.patch("nightshift.agents.scripted.planted_fault_for", return_value=label):
                h = self.with_baseline(sub=sub, backend="scripted")
                res = h.lane("improve")
                rec = self.first(h, Kind.EXP_RECORDED, "e001").data
                outcomes.append((res.outcome, res.status, rec["delta"], rec["noise_margin"], rec["outcome"]))
                self.assertEqual(self.first(h, Kind.EXP_PROPOSED, "e001").data["planted_fault"], label)
        self.assertEqual(outcomes[0], outcomes[1])

    def test_engine_never_reads_planted_fault(self):
        source = inspect.getsource(engine_module)
        self.assertNotRegex(source, r"\.planted_fault\b")

    def test_tamper_with_a_protected_file_ends_invalid_before_any_run(self):
        h = self.with_baseline()
        res = h.lane("tamper")
        self.assertEqual((res.outcome, res.status), (Outcome.INVALID, ExpStatus.RECORDED))
        violation = self.first(h, Kind.EXP_PROTECTED, "e001").data
        self.assertEqual(violation["kind"], "workspace")
        self.assertIn("protected:run.py", violation["paths"])
        kinds = [e.kind for e in self.lane_events(h, "e001")]
        self.assertNotIn(Kind.EXP_IMPLEMENTED, kinds)
        self.assertNotIn(Kind.EXP_RUN_STARTED, kinds)
        self.assertEqual(kinds[-1], Kind.EXP_RECORDED)
        self.assertEqual(h.view().exp("e001").protected_violations[0]["kind"], "workspace")
        self.assertIn("invalid", h.view().lessons[-1]["text"])
        self.assertNotIn("run.py", h.view().lessons[-1]["text"])          # file names are not repeated into memory

    def test_probe_for_the_trusted_directory_is_a_trusted_probe(self):
        h = self.with_baseline()
        res = h.lane("probe")
        self.assertEqual(res.outcome, Outcome.INVALID)
        violation = self.first(h, Kind.EXP_PROTECTED, "e001").data
        self.assertEqual(violation["kind"], "trusted_probe")
        self.assertTrue(violation["paths"][0].startswith("probe:"))

    def test_implementer_that_edits_the_trusted_directory_is_a_trusted_violation(self):
        task = copy_mini_task(self.root / "t")
        h = self.with_baseline(task=task)
        h.agents.on_implement["tamper"] = lambda ctx: (task.trusted_dir / "grade.py").write_text(
            (task.trusted_dir / "grade.py").read_text() + "\n# edited\n")
        res = h.lane("tamper")
        self.assertEqual(res.outcome, Outcome.INVALID)
        self.assertIn("trusted", [e.data["kind"] for e in h.events(Kind.EXP_PROTECTED, exp_id="e001")])
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED, exp_id="e001"), [])
        # the run's recorded grader hash no longer matches: nothing more is evaluated, the next lane ends the same way
        res2 = h.lane("improve")
        self.assertEqual(res2.outcome, Outcome.INVALID)
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED, exp_id="e002"), [])
        self.assertEqual(self.first(h, Kind.EXP_PROTECTED, "e002").data["kind"], "trusted")

    def test_template_edit_after_run_start_is_a_trusted_violation(self):
        task = copy_mini_task(self.root / "t2")
        h = self.with_baseline(task=task)
        with open(task.template_dir / "data.py", "a") as fh:
            fh.write("\n# someone edited the template for future workspaces\n")
        res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.INVALID)
        self.assertEqual(self.first(h, Kind.EXP_PROTECTED, "e001").data["kind"], "trusted")
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED, exp_id="e001"), [])

    def test_a_run_that_edits_the_template_is_the_experiment_that_ends_invalid(self):
        task = copy_mini_task(self.root / "t3")
        h = self.with_baseline(task=task)
        poisoned = str(task.template_dir / "data.py")
        evil = binned_model(5, extra=f"import atexit\natexit.register(lambda: open({poisoned!r}, 'a').write('\\n# poisoned\\n'))\n")
        h.agents.patches["evil"] = Patch({"model.py": evil})
        res = h.lane("evil")
        self.assertEqual((res.exp_id, res.outcome), ("e001", Outcome.INVALID))        # not "improved", and not blamed on e002
        self.assertEqual(self.first(h, Kind.EXP_PROTECTED, "e001").data["kind"], "trusted")
        self.assertEqual(h.events(Kind.EXP_GRADED, exp_id="e001"), [])
        self.assertFalse(h.view().exp("e001").awaiting_replication)

    def test_identical_change_is_blocked_with_the_experiment_id_and_no_implemented_event(self):
        h = self.with_baseline()
        h.lane("improve")
        res = h.lane("improve_again")
        self.assertEqual((res.exp_id, res.status, res.blocked_reason), ("e002", ExpStatus.BLOCKED, "duplicate"))
        blocked = self.first(h, Kind.EXP_BLOCKED, "e002").data
        self.assertEqual((blocked["reason"], blocked["dup_of"]), ("duplicate", "e001"))
        kinds = [e.kind for e in self.lane_events(h, "e002")]
        self.assertEqual(kinds, [Kind.EXP_PROPOSED, Kind.EXP_WORKSPACE, Kind.EXP_BLOCKED])
        self.assertEqual(h.view().exp("e002").status, ExpStatus.BLOCKED)

    def test_parallel_identical_changes_run_once(self):
        h = self.with_baseline()
        results = h.lanes(["improve", "improve_again"])
        self.assertEqual(sorted(r.status for r in results), [ExpStatus.BLOCKED, ExpStatus.RECORDED])
        self.assertEqual(len(h.events(Kind.EXP_IMPLEMENTED)), 1)
        self.assertEqual(len(h.events(Kind.EXP_BLOCKED)), 1)

    def test_recorded_duplicate_hypothesis_is_blocked_before_a_proposal(self):
        h = self.with_baseline()
        first = h.lane("improve", hypothesis="Use five bins to model the curve.")
        before_charges = len(self.charged(h, "experiment"))
        res = h.lane("noop", hypothesis="use FIVE bins, to model the curve!!")
        self.assertEqual((res.exp_id, res.blocked_reason, res.status), ("", "duplicate", ExpStatus.BLOCKED))
        blocked = [e for e in h.events(Kind.EXP_BLOCKED)][-1].data
        self.assertEqual((blocked["exp_id"], blocked["reason"], blocked["dup_of"]), (None, "duplicate", first.exp_id))
        self.assertEqual(len(h.events(Kind.EXP_PROPOSED)), 2)             # baseline + first only
        self.assertEqual(len(self.charged(h, "experiment")), before_charges)

    def test_identical_hypotheses_in_one_batch_run_once(self):
        h = self.with_baseline()
        lanes = [h.agents.lane("improve", hypothesis="Same idea, twice."), h.agents.lane("improve_mid", hypothesis="same idea twice")]
        results = h.engine.run_lanes(STAGES_FULL, lanes)
        self.assertEqual(sorted(r.blocked_reason or "ok" for r in results), ["duplicate", "ok"])
        self.assertEqual(len(h.events(Kind.EXP_PROPOSED)), 2)

    def test_crashing_code_is_failed_with_an_invalid_grade_and_no_review(self):
        h = self.with_baseline()
        res = h.lane("crash")
        self.assertEqual((res.outcome, res.score, res.delta), (Outcome.FAILED, None, None))
        self.assertEqual({e.data["classification"] for e in h.events(Kind.EXP_RUN_FINISHED, exp_id="e001")}, {"crash"})
        graded = self.first(h, Kind.EXP_GRADED, "e001").data
        self.assertFalse(graded["grade"]["valid"])
        self.assertEqual(h.events(Kind.EXP_REVIEWED, exp_id="e001"), [])
        self.assertEqual(h.view().lessons[-1]["classification"], "model")
        self.assertEqual(self.first(h, Kind.EXP_RECORDED, "e001").data["delta"], None)

    def test_infrastructure_failure_is_retried_once_as_a_whole(self):
        h = self.with_baseline()
        res = h.lane("infra_flaky")
        runs = h.events(Kind.EXP_RUN_STARTED, exp_id="e001")
        self.assertEqual([r.data["attempt"] for r in runs], [1, 1, 1, 2, 2, 2])
        self.assertEqual([r.data["run_idx"] for r in runs], list(range(6)))      # indices are never reused
        classes = [e.data["classification"] for e in h.events(Kind.EXP_RUN_FINISHED, exp_id="e001")]
        self.assertEqual(classes, ["infra"] * 3 + ["ok"] * 3)
        self.assertEqual(len(h.events(Kind.EXP_GRADED, exp_id="e001")), 1)
        self.assertEqual(res.outcome, "improved")

    def test_persistent_infrastructure_failure_is_infra_error_without_a_hypothesis_lesson(self):
        h = self.with_baseline()
        spec = h.agents.lane("infra_always", hypothesis="Try a dependency heavy approach.")
        res = h.engine.run_lane(STAGES_FULL, spec)
        self.assertEqual(res.outcome, Outcome.INFRA)
        self.assertEqual(h.events(Kind.EXP_GRADED, exp_id="e001"), [])
        self.assertEqual(h.events(Kind.EXP_REVIEWED, exp_id="e001"), [])
        lessons = h.view().lessons
        self.assertTrue(lessons)
        self.assertEqual({x["classification"] for x in lessons}, {"infra"})
        self.assertIsNone(h.memory.seen(spec.hypothesis))                 # the idea may be tried again
        retry = h.engine.run_lane(STAGES_FULL, LaneSpec(lane="B", hypothesis=spec.hypothesis, scripted_id="improve"))
        self.assertEqual((retry.blocked_reason, retry.outcome), (None, "improved"))

    def test_lane_label_baseline_is_reserved(self):
        h = self.with_baseline()
        res = h.lane("improve", label="Baseline")
        self.assertNotEqual(res.lane.lower(), "baseline")
        self.assertEqual(h.view().baseline_id, "e000")

    def test_stage_list_is_only_tested_for_membership(self):
        h = self.with_baseline()
        res = h.lane("defective", stages=STAGES_MIN)                       # no 'repair'
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(h.events(Kind.EXP_REPAIRED), [])
        self.assertIn("no repair stage", self.first(h, Kind.EXP_RECORDED, "e001").data["note"])

    def test_named_base_experiment_is_frozen_and_used(self):
        h = self.with_baseline()
        h.lane("improve")                                                  # e001: improved, not promoted
        res = h.engine.run_lane(STAGES_FULL, h.agents.lane("improve_mid", base="e001"))
        proposed = self.first(h, Kind.EXP_PROPOSED, res.exp_id).data
        self.assertEqual((proposed["base"], proposed["parent_exp_id"]), ("e001", "e001"))
        self.assertTrue((h.run_dir / "exp/e001/frozen/model.py").is_file())
        meta = json.loads((h.run_dir / "exp" / res.exp_id / "base.json").read_text())
        self.assertEqual(meta["base_ref"], "e001")
        self.assertIn("bins = 5", (h.run_dir / "exp/e001/frozen/model.py").read_text())
        self.assertIn("-    bins = 5", (h.run_dir / "exp" / res.exp_id / "diff.patch").read_text())

    def test_unknown_base_is_refused_before_a_proposal(self):
        h = self.with_baseline()
        res = h.lane("improve", base="e099")
        self.assertEqual((res.exp_id, res.status, res.blocked_reason), ("", ExpStatus.BLOCKED, "invalid_composition"))
        self.assertEqual(len(h.events(Kind.EXP_PROPOSED)), 1)

    def test_a_symlink_in_the_workspace_is_a_workspace_violation(self):
        h = self.with_baseline()
        h.agents.on_implement["noop"] = lambda ctx: os.symlink("/etc/hostname", Path(ctx.workspace) / "link.txt")
        res = h.lane("noop")
        self.assertEqual(res.outcome, Outcome.INVALID)
        violation = self.first(h, Kind.EXP_PROTECTED, "e001").data
        self.assertEqual(violation["kind"], "workspace")
        self.assertTrue(any(p.startswith("special:") for p in violation["paths"]))
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED, exp_id="e001"), [])

    def test_predictions_the_grader_rejects_are_failed_not_invalid(self):
        h = self.with_baseline()
        res = h.lane("nan_output")
        self.assertEqual(res.outcome, Outcome.FAILED)
        self.assertEqual({e.data["classification"] for e in h.events(Kind.EXP_RUN_FINISHED, exp_id="e001")}, {"ok"})
        self.assertIn("grader", self.first(h, Kind.EXP_RECORDED, "e001").data["note"])
        self.assertEqual(h.events(Kind.EXP_REVIEWED, exp_id="e001"), [])

    def test_lane_requests_are_sanitised_before_they_become_ids_or_text(self):
        h = self.with_baseline()
        res = h.engine.run_lane(STAGES_FULL, LaneSpec(lane="../x y", hypothesis="idea\x00 with\x1b[31m control", scripted_id="../noop"))
        self.assertEqual(res.lane, "___x_y")
        proposed = self.first(h, Kind.EXP_PROPOSED, res.exp_id)
        self.assertEqual(proposed.actor, "designer:___x_y")
        self.assertIsNone(proposed.data["scripted_id"])               # not a usable id: dropped, never used as a key
        self.assertNotRegex(proposed.data["hypothesis"], r"[\x00-\x08\x0b-\x1f]")

    def test_empty_hypothesis_is_refused_before_a_proposal(self):
        h = self.with_baseline()
        res = h.engine.run_lane(STAGES_FULL, LaneSpec(lane="A", hypothesis="  \n ", scripted_id="noop"))
        self.assertEqual((res.exp_id, res.status), ("", ExpStatus.BLOCKED))
        self.assertEqual(len(h.events(Kind.EXP_PROPOSED)), 1)

    def test_every_ledger_write_goes_through_append(self):
        source = inspect.getsource(engine_module)
        self.assertEqual(len(re.findall(r"self\.ledger\.append\(", source)), 1)

    def test_replicate_of_lane_is_delegated(self):
        h = self.with_baseline()
        h.lane("improve")
        res = h.engine.run_lane(STAGES_FULL, LaneSpec(lane="R", hypothesis="", replicate_of="e001"))
        self.assertTrue(res.replicated)


# =====================================================================================================
# review and repair
# =====================================================================================================
class ReviewTests(EngineCase):
    def test_review_context_is_dev_only_and_has_no_self_report(self):
        h = self.with_baseline()
        res = h.lane("improve")
        ctx = h.agents.contexts_of(ReviewContext)[0]
        base = h.view().baseline.primary_grade
        self.assertEqual(set(ctx.grade_summary), {"valid", "score", "std", "per_seed", "metrics", "incumbent_score", "delta",
                                                  "margin", "train_delta", "overfit_suspect", "failure_class"})
        self.assertTrue(ctx.grade_summary["valid"])
        self.assertAlmostEqual(ctx.grade_summary["incumbent_score"], base.score)
        self.assertAlmostEqual(ctx.grade_summary["delta"], res.delta)
        self.assertIsNone(ctx.grade_summary["failure_class"])
        self.assertEqual(len(ctx.grade_summary["per_seed"]), 3)
        self.assertEqual(ctx.implementer_identity, "fake-implementer")
        self.assertEqual(ctx.repair_round, 0)
        self.assertTrue(ctx.diff_text.startswith("<<<DATA:diff_e001:"))   # untrusted diff arrives fenced
        self.assertIn("3 run(s)", ctx.run_summary)
        self.assertNotIn("everything improved a lot", str(ctx))           # the implementer's self-report never reaches the reviewer
        self.assertFalse(hasattr(ctx, "self_report"))

    def test_review_repair_rerun_regrade_rereview(self):
        h = self.with_baseline()
        res = h.lane("defective")
        self.assertEqual(res.outcome, "improved")
        ev = self.lane_events(h, "e001")
        repaired = self.first(h, Kind.EXP_REPAIRED, "e001")
        self.assertEqual(repaired.data["repair_round"], 1)
        self.assertEqual(repaired.data["addressed"], ["defect"])
        after = [e for e in ev if e.seq > repaired.seq]
        self.assertEqual([e.kind for e in after], [Kind.EXP_RUN_STARTED, Kind.EXP_RUN_FINISHED] * 3
                         + [Kind.EXP_GRADED, Kind.EXP_REVIEWED, Kind.EXP_RECORDED])
        reviews = h.events(Kind.EXP_REVIEWED, exp_id="e001")
        self.assertEqual([(r.data["review_round"], r.data["verdict"]) for r in reviews], [(0, "repair"), (1, "accept")])
        self.assertEqual(reviews[0].data["findings"][0]["id"], "defect")
        self.assertEqual({r.data["run_idx"] for r in h.events(Kind.EXP_RUN_STARTED, exp_id="e001")}, set(range(6)))
        self.assertEqual({e.data["purpose"] for e in h.events(Kind.EXP_GRADED, exp_id="e001")}, {"primary"})
        impl = [c for c in h.agents.contexts_of(ImplementContext) if c.repair_findings]
        self.assertEqual(len(impl), 1)
        self.assertEqual(impl[0].repair_findings[0]["id"], "defect")
        self.assertIn(DEFECT_MARKER, impl[0].prior_diff)
        self.assertTrue(impl[0].prior_diff.startswith("<<<DATA:diff_e001:"))
        self.assertIn(DEFECT_MARKER, h.agents.contexts_of(ReviewContext)[0].diff_text)
        self.assertNotIn(DEFECT_MARKER, h.agents.contexts_of(ReviewContext)[1].diff_text)
        self.assertEqual(h.agents.contexts_of(ReviewContext)[1].repair_round, 1)
        self.assertEqual(len(h.agents.contexts_of(ReviewContext)[1].prior_findings), 1)

    def test_repairs_are_bounded_by_max_repairs(self):
        h = self.with_baseline()
        h.agents.vary_finding_ids = True
        res = h.lane("defective_stubborn")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(len(h.events(Kind.EXP_REPAIRED, exp_id="e001")), h.limits.max_repairs)
        self.assertEqual(len(h.events(Kind.EXP_REVIEWED, exp_id="e001")), h.limits.max_repairs + 1)
        self.assertIn("repair limit", self.first(h, Kind.EXP_RECORDED, "e001").data["note"])
        self.assertIn("unresolved", h.view().lessons[-1]["text"])

    def test_same_findings_twice_is_unresolved(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=2, max_repairs=5, stall_after=9)
        h = self.with_baseline(limits=limits)
        res = h.lane("defective_stubborn")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(len(h.events(Kind.EXP_REPAIRED, exp_id="e001")), 1)
        self.assertEqual(len(h.events(Kind.EXP_REVIEWED, exp_id="e001")), 2)
        self.assertIn("same review findings", self.first(h, Kind.EXP_RECORDED, "e001").data["note"])

    def test_a_repair_that_changes_nothing_is_unresolved_without_a_repaired_event(self):
        h = self.with_baseline()
        text = binned_model(5, extra=f"# {DEFECT_MARKER}stuck\n")
        h.agents.patches["stuck"] = Patch({"model.py": text}, repair_files={"model.py": text})
        res = h.lane("stuck")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(h.events(Kind.EXP_REPAIRED), [])                # a repair event would demand a re-run it will not get
        self.assertEqual(len(h.events(Kind.EXP_RUN_STARTED, exp_id="e001")), 3)
        self.assertIn("changed nothing", self.first(h, Kind.EXP_RECORDED, "e001").data["note"])

    def test_unavailable_reviewer_is_unresolved_without_a_repair_loop(self):
        h = self.with_baseline()
        h.agents.review_queue.append(ReviewVerdict(verdict="reject", reviewer="fake-reviewer", findings=[
            {"id": "reviewer_unavailable", "severity": "info", "text": "no review", "evidence": ""}]))
        res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(h.events(Kind.EXP_REPAIRED), [])
        reviewed = self.first(h, Kind.EXP_REVIEWED, "e001").data
        self.assertTrue(reviewed["reviewer"])
        self.assertNotEqual(reviewed["reviewer"], self.first(h, Kind.EXP_IMPLEMENTED, "e001").data["worker"])

    def test_reject_without_block_findings_is_unresolved(self):
        h = self.with_baseline()
        h.agents.review_queue.append(ReviewVerdict(verdict="reject", reviewer="fake-reviewer", findings=[
            {"id": "meh", "severity": "warn", "text": "not convincing", "evidence": ""}]))
        res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(h.events(Kind.EXP_REPAIRED), [])

    def test_accept_with_a_block_finding_is_treated_as_repair(self):
        h = self.with_baseline()
        h.agents.review_queue.append(ReviewVerdict(verdict="accept", reviewer="fake-reviewer", findings=[
            {"id": "contradiction", "severity": "block", "text": "bad", "evidence": ""}]))
        res = h.lane("improve")
        self.assertEqual(len(h.events(Kind.EXP_REPAIRED, exp_id="e001")), 1)
        self.assertEqual(res.outcome, "improved")

    def test_reviewer_model_error_is_an_unavailable_review(self):
        h = self.with_baseline()

        def boom(ctx):
            raise ModelError("down", status=503, retryable=True)
        h.agents.review_script = boom
        res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(self.first(h, Kind.EXP_REVIEWED, "e001").data["findings"][0]["id"], "reviewer_unavailable")

    def test_independence_is_recorded_honestly(self):
        h = self.with_baseline()
        h.agents.review_queue.append(ReviewVerdict(verdict="accept", reviewer="fake-implementer", findings=[]))
        h.lane("improve")
        same = self.first(h, Kind.EXP_REVIEWED, "e001").data
        self.assertEqual((same["reviewer"], same["independent"]), ("fake-implementer", False))
        agents = FakeAgents()
        agents.reviewer_identity = ""
        h2 = self.with_baseline(sub="anon", agents=agents)
        h2.lane("improve")
        anon = self.first(h2, Kind.EXP_REVIEWED, "e001").data
        self.assertEqual(anon["reviewer"], "reviewer:A")              # never empty (A.4), never claimed independent
        self.assertFalse(anon["independent"])

    def test_independent_basis_follows_the_models(self):
        h = self.with_baseline()
        h.lane("improve")
        self.assertEqual(self.first(h, Kind.EXP_REVIEWED, "e001").data["independent_basis"], "context")
        h.suite.models["reviewer"] = "another-model"
        h.lane("improve_mid")
        reviewed = self.first(h, Kind.EXP_REVIEWED, "e002").data
        self.assertEqual((reviewed["independent_basis"], reviewed["reviewer_model"], reviewed["implementer_model"]),
                         ("model", "another-model", "fake-model"))

    def test_review_hard_runs_after_review_when_composed(self):
        h = self.with_baseline(reviewer_hard=True)
        res = h.lane("improve", stages=["design", "implement", "run", "grade", "review", "review_hard", "repair", "record"])
        reviews = h.events(Kind.EXP_REVIEWED, exp_id="e001")
        self.assertEqual([r.data["reviewer"] for r in reviews], ["fake-reviewer", "fake-reviewer-hard"])
        self.assertEqual(reviews[1].data["reviewer_model"], "fake-hard-model")
        self.assertEqual(res.outcome, "improved")

    def test_review_hard_rejection_goes_through_repair(self):
        h = self.with_baseline(reviewer_hard=True)
        h.agents.hard_queue.append(ReviewVerdict(verdict="repair", reviewer="fake-reviewer-hard", findings=[
            {"id": "deep-issue", "severity": "block", "text": "subtle", "evidence": ""}]))
        res = h.lane("improve", stages=["implement", "run", "grade", "review", "review_hard", "repair", "record"])
        self.assertEqual(len(h.events(Kind.EXP_REPAIRED, exp_id="e001")), 1)
        names = [r.data["reviewer"] for r in h.events(Kind.EXP_REVIEWED, exp_id="e001")]
        self.assertEqual(names, ["fake-reviewer", "fake-reviewer-hard", "fake-reviewer", "fake-reviewer-hard"])
        self.assertEqual(res.outcome, "improved")

    def test_review_hard_without_a_hard_reviewer_is_skipped_with_a_note(self):
        h = self.with_baseline()
        res = h.lane("improve", stages=["implement", "run", "grade", "review", "review_hard", "record"])
        self.assertEqual(len(h.events(Kind.EXP_REVIEWED, exp_id="e001")), 1)
        self.assertIn("review_hard skipped", self.first(h, Kind.EXP_RECORDED, "e001").data["note"])
        self.assertEqual(res.outcome, "improved")

    def test_model_written_findings_are_sanitised_and_bounded(self):
        h = self.with_baseline()
        junk = [{"id": "bad id!\n<script>", "severity": "CRITICAL", "text": "\x00\x1b[31mred\x07 text", "evidence": "e\x00v"}]
        junk += [{"id": "bad id!\n<script>", "severity": "info", "text": "dup id"}]
        junk += [{"id": f"f{i}", "severity": "info", "text": "x"} for i in range(30)] + ["not a dict", None]
        h.agents.review_queue.append(ReviewVerdict(verdict="accept", reviewer="fake-reviewer", findings=junk))
        h.lane("improve")
        findings = self.first(h, Kind.EXP_REVIEWED, "e001").data["findings"]
        self.assertLessEqual(len(findings), 12)
        self.assertRegex(findings[0]["id"], r"^[A-Za-z0-9_.-]+$")
        self.assertEqual(findings[0]["severity"], "warn")
        self.assertNotRegex(findings[0]["text"], r"[\x00-\x08\x0b-\x1f]")
        self.assertEqual(len({f["id"] for f in findings}), len(findings))

    def test_same_hard_findings_twice_is_unresolved_even_though_the_regular_review_sits_between(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=2, max_repairs=5, stall_after=9)
        h = self.with_baseline(limits=limits, reviewer_hard=True)
        deep = ReviewVerdict(verdict="repair", reviewer="fake-reviewer-hard", findings=[
            {"id": "deep-issue", "severity": "block", "text": "subtle", "evidence": ""}])
        h.agents.hard_queue.extend([deep, deep])
        res = h.lane("improve", stages=["implement", "run", "grade", "review", "review_hard", "repair", "record"])
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)                 # review(accept) always separates the two hard rounds
        self.assertEqual(len(h.events(Kind.EXP_REPAIRED, exp_id="e001")), 1)
        self.assertIn("same review findings", self.first(h, Kind.EXP_RECORDED, "e001").data["note"])

    def test_an_unusable_verdict_is_an_unavailable_review(self):
        h = self.with_baseline()
        h.agents.review_queue.append(ReviewVerdict(verdict="approve!!!", reviewer="fake-reviewer", findings=[]))
        res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(self.first(h, Kind.EXP_REVIEWED, "e001").data["verdict"], "reject")


# =====================================================================================================
# claims, secrets, budget, halt, errors
# =====================================================================================================
class RulesTests(EngineCase):
    def test_a_claim_is_recorded_but_is_never_evidence(self):
        h = self.with_baseline()
        res = h.lane("claims_win")
        self.assertEqual(res.outcome, Outcome.NO_GAIN)
        claim = h.view().claims["e001.c1"]
        self.assertEqual((claim["claim_type"], claim["exp_id"], claim["actor"]), ("improved", "e001", "implementer:A"))
        self.assertEqual(h.view().exp("e001").dev_outcome, Outcome.NO_GAIN)

    def test_claim_ids_count_per_experiment_and_types_are_validated(self):
        h = self.with_baseline()

        def two_claims(ctx):
            self.assertEqual(ctx.emit_claim("improved", "first"), "e001.c1")
            self.assertEqual(ctx.emit_claim("totally-made-up", "second\x00"), "e001.c2")
        h.agents.on_implement["noop"] = two_claims
        h.lane("noop")
        self.assertEqual(h.view().claims["e001.c2"]["claim_type"], "other")
        self.assertNotIn("\x00", h.view().claims["e001.c2"]["text"])

    def test_registered_secret_never_reaches_the_ledger_or_the_logs(self):
        key = "sk-live-0123456789abcdef"
        secrets.register(key)
        self.addCleanup(secrets.clear)
        h = self.with_baseline()
        leak = _TEMPLATE_WITH(f'print("{key}")\n# {key}\n')
        h.agents.patches["leaky"] = Patch({"model.py": leak})
        h.lane("leaky")
        self.assertNotIn(key, (h.run_dir / "ledger.jsonl").read_text())
        for name in ("diff.patch", "worker.log"):
            self.assertNotIn(key, (h.run_dir / "exp/e001" / name).read_text())
        self.assertIn("[REDACTED]", (h.run_dir / "exp/e001/diff.patch").read_text())

    def test_budget_exhausted_before_a_proposal_blocks_with_a_null_experiment(self):
        h = self.with_baseline(limits=BudgetLimits(max_experiments=1, max_wall_sec=1e5, max_parallel=2))
        h.lane("improve")
        res = h.lane("noop")
        self.assertEqual((res.exp_id, res.status, res.blocked_reason), ("", ExpStatus.BLOCKED, "budget"))
        blocked = h.events(Kind.EXP_BLOCKED)[-1].data
        self.assertEqual((blocked["exp_id"], blocked["reason"]), (None, "budget"))
        self.assertEqual(len(h.events(Kind.EXP_PROPOSED)), 2)
        self.assertTrue(h.events(Kind.BUDGET_EXHAUSTED))

    def test_budget_exceeded_mid_lane_stops_it_as_unresolved(self):
        h = self.with_baseline()

        def out_of_money(ctx):
            raise BudgetExceeded("llm budget exhausted", kind="llm", limit=1.0, used=1.0)
        h.agents.on_implement["improve"] = out_of_money
        res = h.lane("improve")
        self.assertEqual((res.outcome, res.status), (Outcome.UNRESOLVED, ExpStatus.RECORDED))
        self.assertEqual(self.first(h, Kind.EXP_RECORDED, "e001").data["note"], "stopped by the budget")
        self.assertEqual(h.view().lessons, [])                          # nothing was learned about the hypothesis
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED, exp_id="e001"), [])

    def test_designer_budget_error_is_handled_the_same_way(self):
        h = self.with_baseline()

        def out_of_money(ctx):
            raise BudgetExceeded("llm budget exhausted", kind="llm")
        h.agents.designer_hook = out_of_money
        res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertEqual(h.events(Kind.EXP_IMPLEMENTED), [])

    def test_halt_file_stops_a_lane_before_it_starts(self):
        h = self.with_baseline()
        before = len(h.ledger.events())
        h.halt()
        with self.assertRaises(HaltRequested):
            h.lane("improve")
        self.assertEqual(len(h.ledger.events()), before)
        with self.assertRaises(HaltRequested):
            h.engine.replicate("e000")
        with self.assertRaises(HaltRequested):
            h.engine.run_baseline()

    def test_halt_between_stages_propagates_and_the_lane_is_closed_as_unresolved(self):
        """A halted run is finished and sealed, so a lane cut off by HALT cannot be resumed: it is recorded ``unresolved``
        (``stopped by HALT``) before the halt propagates, instead of being left mid-stage in a sealed ledger."""
        h = self.with_baseline()
        h.agents.on_implement["improve"] = lambda ctx: h.halt()
        with self.assertRaises(HaltRequested):
            h.lane("improve")
        kinds = [e.kind for e in self.lane_events(h, "e001")]
        self.assertEqual(kinds[-2:], [Kind.EXP_IMPLEMENTED, Kind.EXP_RECORDED])
        recorded = self.lane_events(h, "e001")[-1].data
        self.assertEqual((recorded["phase"], recorded["outcome"], recorded["note"]), ("lane", "unresolved", "stopped by HALT"))
        self.assertEqual(h.view().exp("e001").status, ExpStatus.RECORDED)
        self.assertEqual(h.view().in_flight(), [])
        (h.run_dir / "HALT").unlink()
        h.restart()
        self.assertEqual(h.engine.resume_pending(), [])                  # nothing is left to continue

    def test_run_lanes_reraises_halt(self):
        h = self.with_baseline()
        h.halt()
        with self.assertRaises(HaltRequested):
            h.lanes(["improve", "noop"])

    def test_simulated_crash_in_a_role_propagates(self):
        h = self.with_baseline()

        def crash(ctx):
            raise SimulatedCrash("kill -9")
        h.agents.on_implement["improve"] = crash
        with self.assertRaises(SimulatedCrash):
            h.lane("improve")
        self.assertEqual(h.view().exp("e001").status, ExpStatus.WORKSPACE)     # nothing was swallowed or recorded

    def test_model_error_from_the_implementer_is_an_environment_failure(self):
        h = self.with_baseline()

        def down(ctx):
            raise ModelError("overloaded", status=503, retryable=True)
        h.agents.on_implement["improve"] = down
        res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.INFRA)
        self.assertEqual({x["classification"] for x in h.view().lessons}, {"infra"})

    def test_a_hostile_implementer_result_does_not_cost_the_lane_its_work(self):
        h = self.with_baseline()
        real = h.suite.implementer

        class Hostile:
            identity = "fake-implementer"

            def implement(self, ctx):
                real.implement(ctx)
                return ImplementResult(summary=None, steps="many", worker=None, raw_log=["x"], self_report=b"bytes")
        h.suite.implementer = Hostile()
        res = h.lane("improve")
        self.assertEqual(res.outcome, "improved")
        implemented = self.first(h, Kind.EXP_IMPLEMENTED, "e001").data
        self.assertEqual((implemented["steps"], implemented["worker"]), (0, "fake-implementer"))
        h.suite.implementer = type("Junk", (), {"identity": "fake-implementer", "implement": lambda self, ctx: (real.implement(ctx), 42)[1]})()
        self.assertEqual(h.lane("improve_mid").outcome, "improved")        # not an ImplementResult at all

    def test_a_lane_that_dies_mid_run_closes_its_open_runs(self):
        h = self.with_baseline()

        def evaluate(task, ws, seeds, **kw):
            kw["hooks"].on_run_start(0, int(seeds[0]), 4242)               # a run started ...
            raise OSError("the disk went away")                            # ... and the evaluation never came back
        with mock.patch.object(grading, "evaluate", side_effect=evaluate):
            res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        runs = h.view().exp("e001").runs
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].classification, "interrupted")            # no run_started without a run_finished
        self.assertEqual(h.engine.reconcile(), [])

    def test_unexpected_error_becomes_unresolved_and_leaks_nothing(self):
        h = self.with_baseline()

        def bug(ctx):
            raise RuntimeError("secret detail from a model")
        h.agents.on_implement["improve"] = bug
        res = h.lane("improve")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        note = self.first(h, Kind.EXP_RECORDED, "e001").data["note"]
        self.assertEqual(note, "engine_error:RuntimeError")
        self.assertNotIn("secret detail", (h.run_dir / "ledger.jsonl").read_text())
        self.assertEqual(h.view().lessons, [])

    def test_workspace_modified_between_review_and_replication_is_a_violation(self):
        h = self.with_baseline()
        h.lane("improve")
        with open(h.run_dir / "exp/e001/workspace/model.py", "a") as fh:
            fh.write("\n# sneaky edit after the lane ended\n")
        res = h.engine.replicate("e001")
        self.assertFalse(res.replicated)
        self.assertEqual(self.first(h, Kind.EXP_PROTECTED, "e001").data["kind"], "workspace")
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED, exp_id="e001")[-1].data["purpose"], "primary")   # nothing sealed ran


def _TEMPLATE_WITH(extra: str) -> str:
    return extra + (MINI_TASK_DIR / "template" / "model.py").read_text()


# =====================================================================================================
# parallelism, the crash hook, reconcile and resume
# =====================================================================================================
class ParallelAndCrashTests(EngineCase):
    def test_run_lanes_is_bounded_and_ordered(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=2)
        h = self.with_baseline(limits=limits)
        h.agents.implement_delay = 0.2
        lanes = [h.agents.lane("improve"), h.agents.lane("improve_mid"), h.agents.lane("noop")]
        results = h.engine.run_lanes(STAGES_FULL, lanes)
        self.assertEqual([r.lane for r in results], [lane.lane for lane in lanes])
        self.assertEqual([r.outcome for r in results], ["improved", "improved", "no_gain"])
        self.assertEqual(len({r.exp_id for r in results}), 3)
        self.assertEqual(h.agents.peak_concurrency, 2)
        paths = {self.first(h, Kind.EXP_WORKSPACE, r.exp_id).data["path"] for r in results}
        self.assertEqual(len(paths), 3)                                    # one workspace each

    def test_parallel_lanes_keep_the_ledger_chain_valid(self):
        h = self.with_baseline()
        h.lanes(["improve", "worse", "noop"])
        self.assertEqual(h.ledger.verify_chain(), (True, None))
        self.assertFalse(h.engine.crashed)

    def test_run_lanes_of_nothing(self):
        self.assertEqual(self.with_baseline().engine.run_lanes(STAGES_FULL, []), [])

    def test_crash_hook_counts_run_started_and_blocks_every_later_append(self):
        h = self.make(crash_after_runs=2)
        with self.assertRaises(SimulatedCrash):
            h.baseline()
        self.assertEqual(len(h.events(Kind.EXP_RUN_STARTED)), 2)
        self.assertTrue(h.engine.crashed)
        self.assertTrue(getattr(h.ledger, "crashed", False))
        count = len(h.ledger.events())
        for _ in range(3):
            with self.assertRaises(SimulatedCrash):
                h.engine._append(Kind.SUP_NUDGE, "engine", {"text": "x"})
        with self.assertRaises(SimulatedCrash):
            h.engine.run_lane(STAGES_FULL, h.agents.lane("improve"))
        self.assertEqual(len(h.ledger.events()), count)

    def test_reconcile_closes_interrupted_runs_and_is_idempotent(self):
        h = self.make(crash_after_runs=2)
        with self.assertRaises(SimulatedCrash):
            h.baseline()
        h.restart()
        self.assertEqual(h.engine.reconcile(), ["e000"])
        interrupted = [e for e in h.events(Kind.EXP_RUN_FINISHED) if e.data["classification"] == "interrupted"]
        self.assertEqual([(e.data["run_idx"], e.data["exit_code"]) for e in interrupted], [(1, None)])
        self.assertEqual(set(interrupted[0].data), KEYS[Kind.EXP_RUN_FINISHED])
        count = len(h.ledger.events())
        self.assertEqual(h.engine.reconcile(), [])
        self.assertEqual(len(h.ledger.events()), count)

    def test_baseline_resumes_after_a_crash_with_the_next_attempt(self):
        h = self.make(crash_after_runs=2)
        with self.assertRaises(SimulatedCrash):
            h.baseline()
        h.restart()
        results = h.engine.resume_pending()
        self.assertEqual([(r.exp_id, r.outcome) for r in results], [("e000", "baseline")])
        runs = h.events(Kind.EXP_RUN_STARTED, exp_id="e000")
        self.assertEqual([r.data["attempt"] for r in runs], [1, 1, 2, 2, 2])
        self.assertEqual([r.data["run_idx"] for r in runs], [0, 1, 2, 3, 4])
        self.assertEqual(len(h.events(Kind.EXP_PROPOSED)), 1)
        self.assertEqual(len(h.events(Kind.EXP_WORKSPACE)), 1)
        self.assertEqual(len(h.events(Kind.BUDGET_CHARGED, exp_id="e000")), 1)
        self.assertEqual(h.view().baseline_id, "e000")

    def test_parallel_lanes_crash_then_resume_without_reproposing_or_recharging(self):
        h = self.with_baseline()
        h.restart(crash_after_runs=4)
        h.agents.implement_delay = 0.05
        with self.assertRaises(SimulatedCrash):
            h.lanes(["improve", "improve_mid", "noop"])
        self.assertEqual(len([e for e in h.events(Kind.EXP_RUN_STARTED) if e.data["exp_id"] != "e000"]), 4)
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith("nightshift-lane")])
        proposed_before = len(h.events(Kind.EXP_PROPOSED))
        charged_before = len(self.charged(h, "experiment"))
        h.restart()
        results = h.engine.resume_pending()
        self.assertEqual(len(h.events(Kind.EXP_PROPOSED)), proposed_before)
        self.assertEqual(len(self.charged(h, "experiment")), charged_before)
        self.assertEqual(sorted(r.exp_id for r in results), ["e001", "e002", "e003"])
        view = h.view()
        for exp_id in ("e001", "e002", "e003"):
            self.assertEqual(view.exp(exp_id).status, ExpStatus.RECORDED)
            self.assertEqual(len([e for e in h.events(Kind.EXP_RECORDED, exp_id=exp_id)]), 1)
        self.assertEqual(sorted(view.exp(e).outcome for e in ("e001", "e002", "e003")), ["improved", "improved", "no_gain"])
        self.assertEqual(h.engine.reconcile(), [])

    def test_run_lanes_reraises_the_crash_after_joining_the_pool(self):
        h = self.with_baseline()
        h.agents.implement_delay = 0.05

        def crash(ctx):
            raise SimulatedCrash("kill -9")
        h.agents.on_implement["worse"] = crash
        with self.assertRaises(SimulatedCrash):
            h.lanes(["improve", "worse", "noop"])
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith("nightshift-lane")])

    def _resume_after_crash_before(self, predicate, patch, expected, sub):
        """Crash instead of writing the first matching event of e001, restart, resume; returns the harness."""
        h = self.with_baseline(sub=sub)
        crash_when(h.engine, lambda k, d: d.get("exp_id") == "e001" and predicate(k, d))
        with self.assertRaises(SimulatedCrash):
            h.lane(patch)
        h.restart()
        results = h.engine.resume_pending()
        self.assertEqual([(r.exp_id, r.outcome) for r in results], [("e001", expected)])
        self.assertEqual(h.view().exp("e001").status, ExpStatus.RECORDED)
        self.assertEqual(len(h.events(Kind.EXP_PROPOSED)), 2)                      # never re-proposed
        self.assertEqual(len(self.charged(h, "experiment")), 1)                    # never re-charged
        self.assertEqual(len(h.events(Kind.EXP_RECORDED, exp_id="e001")), 1)
        self.assertLessEqual(len(h.events(Kind.EXP_IMPLEMENTED, exp_id="e001")), 1)
        self.assertLessEqual(len(h.events(Kind.EXP_WORKSPACE, exp_id="e001")), 1)
        self.assertEqual(h.engine.resume_pending(), [])                            # and resuming again is a no-op
        return h

    def test_resume_proposed_only(self):
        self._resume_after_crash_before(lambda k, d: k == Kind.EXP_WORKSPACE, "improve", "improved", "r1")

    def test_resume_workspace_ready_not_implemented(self):
        h = self._resume_after_crash_before(lambda k, d: k == Kind.EXP_IMPLEMENTED, "improve", "improved", "r2")
        self.assertEqual(len(h.events(Kind.EXP_IMPLEMENTED, exp_id="e001")), 1)

    def test_resume_implemented_never_run(self):
        h = self._resume_after_crash_before(lambda k, d: k == Kind.EXP_RUN_STARTED, "improve", "improved", "r3")
        self.assertEqual([e.data["attempt"] for e in h.events(Kind.EXP_RUN_STARTED, exp_id="e001")], [1, 1, 1])

    def test_resume_interrupted_run_is_rerun_with_the_next_attempt(self):
        h = self._resume_after_crash_before(lambda k, d: k == Kind.EXP_RUN_STARTED and d["run_idx"] == 1, "improve", "improved", "r4")
        self.assertEqual([e.data["attempt"] for e in h.events(Kind.EXP_RUN_STARTED, exp_id="e001")], [1, 2, 2, 2])

    def test_resume_graded_not_reviewed_reviews_without_rerunning(self):
        h = self._resume_after_crash_before(lambda k, d: k == Kind.EXP_REVIEWED, "improve", "improved", "r5")
        self.assertEqual(len(h.events(Kind.EXP_GRADED, exp_id="e001")), 1)
        self.assertEqual(len(h.events(Kind.EXP_RUN_STARTED, exp_id="e001")), 3)

    def test_resume_reviewed_not_recorded_records_without_rereviewing(self):
        h = self._resume_after_crash_before(lambda k, d: k == Kind.EXP_RECORDED, "improve", "improved", "r6")
        self.assertEqual(len(h.events(Kind.EXP_REVIEWED, exp_id="e001")), 1)

    def test_resume_repaired_not_rerun(self):
        h = self._resume_after_crash_before(lambda k, d: k == Kind.EXP_RUN_STARTED and d["run_idx"] == 3, "defective", "improved", "r7")
        self.assertEqual(len(h.events(Kind.EXP_REPAIRED, exp_id="e001")), 1)
        self.assertEqual(len(h.events(Kind.EXP_REVIEWED, exp_id="e001")), 2)

    def test_resume_reviewed_with_a_repair_request_repairs(self):
        h = self._resume_after_crash_before(lambda k, d: k == Kind.EXP_REPAIRED, "defective", "improved", "r8")
        self.assertEqual(len(h.events(Kind.EXP_REPAIRED, exp_id="e001")), 1)
        self.assertEqual(len(h.events(Kind.EXP_REVIEWED, exp_id="e001")), 2)

    def test_resume_violation_written_but_not_recorded(self):
        h = self._resume_after_crash_before(lambda k, d: k == Kind.EXP_RECORDED, "tamper", "invalid", "r9")
        self.assertEqual(len(h.events(Kind.EXP_PROTECTED, exp_id="e001")), 1)

    def test_resume_refuses_a_workspace_that_changed_while_the_process_was_down(self):
        h = self.with_baseline()
        crash_when(h.engine, lambda k, d: k == Kind.EXP_RUN_STARTED)
        with self.assertRaises(SimulatedCrash):
            h.lane("improve")
        with open(h.run_dir / "exp/e001/workspace/model.py", "a") as fh:
            fh.write("\n# edited while the engine was down\n")
        h.restart()
        results = h.engine.resume_pending()
        self.assertEqual(results[0].outcome, Outcome.INVALID)
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED, exp_id="e001"), [])

    def test_replication_resumes_after_a_crash_without_recharging(self):
        h = self.with_baseline()
        h.lane("improve")
        h.restart(crash_after_runs=3)                 # dies inside the baseline's sealed grading
        with self.assertRaises(SimulatedCrash):
            h.engine.replicate("e001")
        self.assertEqual(len(self.charged(h, "replication")), 1)
        h.restart()
        results = h.engine.resume_pending()
        self.assertEqual([(r.exp_id, r.replicated) for r in results], [("e001", True)])
        self.assertEqual(len(self.charged(h, "replication")), 1)
        self.assertEqual(h.view().incumbent_id, "e001")
        self.assertEqual(len(h.events(Kind.EXP_RECORDED, exp_id="e001")), 2)         # the lane record and one replication record
        self.assertEqual(h.engine.resume_pending(), [])


    def test_a_crash_between_the_replication_record_and_the_promotion_is_finished_on_resume(self):
        h = self.with_baseline()
        h.lane("improve")
        crash_when(h.engine, lambda k, d: k == Kind.INCUMBENT_PROMOTED)
        with self.assertRaises(SimulatedCrash):
            h.engine.replicate("e001")
        view = h.view()
        self.assertEqual((view.incumbent_id, view.exp("e001").replicated), ("e000", True))       # replicated, never promoted
        runs, charges = len(h.events(Kind.EXP_RUN_STARTED)), len(self.charged(h, "replication"))
        h.restart()
        out = h.engine.resume_pending()
        self.assertEqual([(r.exp_id, r.replicated) for r in out], [("e001", True)])
        self.assertEqual(h.view().incumbent_id, "e001")
        self.assertEqual(len(h.events(Kind.INCUMBENT_PROMOTED)), 1)
        self.assertEqual((len(h.events(Kind.EXP_RUN_STARTED)), len(self.charged(h, "replication"))), (runs, charges))
        self.assertEqual(h.engine.resume_pending(), [])                                         # and it is not promoted twice

    def test_resume_never_promotes_a_tree_that_is_not_the_one_that_was_graded(self):
        h = self.with_baseline()
        h.lane("improve")
        crash_when(h.engine, lambda k, d: k == Kind.INCUMBENT_PROMOTED)
        with self.assertRaises(SimulatedCrash):
            h.engine.replicate("e001")
        with open(h.run_dir / "exp/e001/workspace/model.py", "a") as fh:
            fh.write("\n# edited while the engine was down\n")
        h.restart()
        self.assertEqual(h.engine.resume_pending(), [])
        self.assertEqual((h.view().incumbent_id, len(h.events(Kind.INCUMBENT_PROMOTED))), ("e000", 0))

    def test_resume_does_not_promote_what_a_gate_still_refuses(self):
        h = self.with_baseline()
        lane = h.lane("improve")
        RunBuilder(h.ledger).finding("I01", "e001", exp_id="e001")
        self.assertTrue(h.engine.replicate(lane.exp_id).replicated)                             # replicated, not promoted
        h.restart()
        self.assertEqual(h.engine.resume_pending(), [])
        self.assertEqual(h.view().incumbent_id, "e000")


# =====================================================================================================
# replication and promotion
# =====================================================================================================
class ReplicationTests(EngineCase):
    def test_replicate_grades_sealed_compares_and_promotes(self):
        h = self.with_baseline()
        lane = h.lane("improve")
        res = h.engine.replicate(lane.exp_id)
        self.assertEqual((res.exp_id, res.status, res.outcome, res.replicated, res.score, res.delta),
                         ("e001", ExpStatus.RECORDED, "improved", True, None, None))
        charges = [e.data for e in h.events(Kind.BUDGET_CHARGED, exp_id="e001") if e.data["kind"] == "replication"]
        self.assertEqual([(c["amount"], c["purpose"]) for c in charges], [(1, "replication")])
        base_runs = [e.data for e in h.events(Kind.EXP_RUN_STARTED, exp_id="e000") if e.data["purpose"] == "replication_baseline"]
        self.assertEqual([(r["seed"], r["run_idx"]) for r in base_runs], list(zip(h.task.replication_seeds, range(3, 8))))
        base_graded = self.first(h, Kind.EXP_GRADED, "e000", purpose="replication_baseline").data
        cand_graded = self.first(h, Kind.EXP_GRADED, "e001", purpose="replication").data
        for g in (base_graded, cand_graded):
            self.assertEqual((g["split"], g["grade"]["split"], g["seeds"]), ("sealed", "sealed", h.task.replication_seeds))
        self.assertEqual(base_graded["conditions_hash"], cand_graded["conditions_hash"])
        self.assertEqual(base_graded["grader_hash"], cand_graded["grader_hash"])
        rec = [e for e in h.events(Kind.EXP_RECORDED, exp_id="e001")][-1].data
        self.assertEqual((rec["phase"], rec["outcome"], rec["replicated"], rec["vs_exp_id"]), ("replication", "improved", True, "e000"))
        self.assertGreater(rec["sealed_delta"], rec["sealed_margin"])
        self.assertEqual(rec["t"], grading.T975[4])
        promoted = self.first(h, Kind.INCUMBENT_PROMOTED, "e001").data
        self.assertEqual((promoted["prev_exp_id"], promoted["replicated"], promoted["replication_exp_ids"]), ("e000", True, ["e001"]))
        self.assertEqual(promoted["sealed_delta"], rec["sealed_delta"])
        frozen = h.run_dir / "exp/e001/frozen"
        self.assertEqual(promoted["tree_hash"], sandbox.tree_hash(frozen, h.task))
        view = h.view()
        exp = view.exp("e001")
        self.assertEqual((view.incumbent_id, exp.replicated, exp.dev_outcome), ("e001", True, "improved"))
        self.assertAlmostEqual(exp.delta, lane.delta)                      # the dev delta is untouched by replication
        self.assertFalse(exp.awaiting_replication)

    def test_second_candidate_is_compared_with_the_new_incumbent_and_the_baseline_is_graded_once(self):
        h = self.with_baseline()
        h.engine.replicate(h.lane("improve").exp_id)
        runs_before = len(h.events(Kind.EXP_RUN_STARTED, exp_id="e000"))
        second = h.lane("improve_best")
        self.assertEqual(second.outcome, "improved")                     # better than the new incumbent on dev
        self.assertEqual(self.first(h, Kind.EXP_PROPOSED, second.exp_id).data["base"], "e001")
        rec = self.first(h, Kind.EXP_RECORDED, second.exp_id).data
        self.assertEqual(rec["vs_exp_id"], "e001")
        res = h.engine.replicate(second.exp_id)
        self.assertTrue(res.replicated)
        self.assertEqual(len(h.events(Kind.EXP_RUN_STARTED, exp_id="e000")), runs_before)   # no second sealed baseline grading
        replication = [e for e in h.events(Kind.EXP_RECORDED, exp_id=second.exp_id)][-1].data
        self.assertEqual(replication["vs_exp_id"], "e001")
        promoted = [e for e in h.events(Kind.INCUMBENT_PROMOTED)][-1].data
        self.assertEqual((promoted["exp_id"], promoted["prev_exp_id"]), (second.exp_id, "e001"))
        self.assertEqual(h.view().incumbent_id, second.exp_id)

    def test_failed_replication_records_it_writes_a_number_free_lesson_and_does_not_promote(self):
        task = copy_mini_task(self.root / "flat", sealed="flat")
        h = self.with_baseline(task=task)
        lane = h.lane("improve")
        self.assertEqual(lane.outcome, "improved")                        # dev disagrees with the sealed split
        res = h.engine.replicate(lane.exp_id)
        self.assertEqual((res.replicated, res.outcome, res.score, res.delta), (False, "no_gain", None, None))
        rec = [e for e in h.events(Kind.EXP_RECORDED, exp_id="e001")][-1].data
        self.assertEqual((rec["phase"], rec["outcome"], rec["replicated"]), ("replication", "no_gain", False))
        self.assertLessEqual(rec["sealed_delta"], rec["sealed_margin"])
        self.assertEqual(h.events(Kind.INCUMBENT_PROMOTED), [])
        view = h.view()
        self.assertEqual(view.incumbent_id, "e000")
        self.assertAlmostEqual(view.exp("e001").delta, lane.delta)         # still the dev delta (contract section 13)
        self.assertFalse(view.exp("e001").awaiting_replication)
        lesson = view.lessons[-1]
        self.assertEqual((lesson["classification"], lesson["source_exps"]), ("model", ["e001"]))
        self.assertIn("did not replicate on the independent re-check; possible overfitting to the evaluation it was tuned on",
                      lesson["text"])
        self.assertNotIn("sealed", lesson["text"].lower())
        self.assertNotRegex(re.sub(r"\be\d{3}\b", "", lesson["text"]), r"\d")     # WITHOUT NUMBERS
        self.assertIn("no_gain (failed replication)", h.memory.tried_table())

    def test_refusals_are_blocked_events_with_a_null_experiment(self):
        h = self.with_baseline()
        h.lane("noop")
        for target, why in (("e000", "baseline"), ("e001", "not improved"), ("e099", "unknown"), ("../../etc/passwd", "path"),
                            (None, "none")):
            with self.subTest(target=target):
                before = len(self.charged(h, "replication"))
                res = h.engine.replicate(target)
                self.assertEqual((res.status, res.blocked_reason, res.replicated), (ExpStatus.BLOCKED, "not_replicable", None))
                blocked = h.events(Kind.EXP_BLOCKED)[-1].data
                self.assertEqual((blocked["exp_id"], blocked["reason"], blocked["lane"]), (None, "not_replicable", "replicate"))
                self.assertNotIn("/", blocked["detail"])
                self.assertEqual(len(self.charged(h, "replication")), before)

    def test_replicating_twice_is_refused(self):
        h = self.with_baseline()
        h.engine.replicate(h.lane("improve").exp_id)
        res = h.engine.replicate("e001")
        self.assertEqual((res.blocked_reason, res.replicated), ("not_replicable", None))

    def test_replication_budget_is_a_budget_block(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=3, max_replications=1)
        h = self.with_baseline(limits=limits)
        lanes = h.lanes(["improve", "improve_best"])
        self.assertTrue(h.engine.replicate(lanes[0].exp_id).replicated)
        res = h.engine.replicate(lanes[1].exp_id)
        self.assertEqual((res.status, res.blocked_reason, res.replicated), (ExpStatus.BLOCKED, "budget", None))
        self.assertEqual(h.events(Kind.EXP_BLOCKED)[-1].data["reason"], "budget")
        self.assertEqual(len(self.charged(h, "replication")), 1)

    def test_superseded_when_the_incumbent_changes_during_replication(self):
        h = self.with_baseline()
        a, b = h.lanes(["improve", "improve_best"])
        real = grading.evaluate
        state = {"done": False}

        def evaluate(task, ws, seeds, **kw):
            if kw["purpose"] == "replication" and ws.exp_id == a.exp_id and not state["done"]:
                state["done"] = True
                inner = h.engine.replicate(b.exp_id)         # the better candidate is promoted while A is still being replicated
                self.assertTrue(inner.replicated)
            return real(task, ws, seeds, **kw)

        with mock.patch.object(grading, "evaluate", side_effect=evaluate):
            res = h.engine.replicate(a.exp_id)
        self.assertFalse(res.replicated)
        rec = [e for e in h.events(Kind.EXP_RECORDED, exp_id=a.exp_id)][-1].data
        self.assertEqual((rec["note"], rec["replicated"], rec["vs_exp_id"]), ("superseded", False, b.exp_id))
        self.assertEqual(h.view().incumbent_id, b.exp_id)
        self.assertEqual([e.data["exp_id"] for e in h.events(Kind.INCUMBENT_PROMOTED)], [b.exp_id])

    def test_concurrent_replications_are_serialised(self):
        h = self.with_baseline()
        a, b = h.lanes(["improve", "improve_best"])
        results, errors = {}, []

        def go(exp_id):
            try:
                results[exp_id] = h.engine.replicate(exp_id)
            except BaseException as exc:      # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=go, args=(x.exp_id,)) for x in (a, b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(h.view().incumbent_id, b.exp_id)                  # the better one wins whatever the order
        promoted = [e.data for e in h.events(Kind.INCUMBENT_PROMOTED)]
        self.assertIn(len(promoted), (1, 2))
        for prev, cur in zip([None] + promoted, promoted):
            if prev is not None:
                self.assertEqual(cur["prev_exp_id"], prev["exp_id"])
        for x in (a, b):
            self.assertEqual(len([e for e in h.events(Kind.EXP_RECORDED, exp_id=x.exp_id) if e.data["phase"] == "replication"]), 1)
        self.assertEqual(len([e for e in h.events(Kind.EXP_GRADED, exp_id="e000") if e.data["purpose"] == "replication_baseline"]), 1)

    def test_an_open_block_finding_stops_promotion_but_not_the_record(self):
        h = self.with_baseline()
        lane = h.lane("improve")
        RunBuilder(h.ledger).finding("I01", "e001", exp_id="e001")
        res = h.engine.replicate(lane.exp_id)
        self.assertTrue(res.replicated)
        self.assertTrue(res.notes.startswith("replicated, not promoted"))
        self.assertEqual(h.events(Kind.INCUMBENT_PROMOTED), [])
        self.assertEqual(h.view().incumbent_id, "e000")

    def test_an_infrastructure_failure_in_replication_leaves_the_candidate_awaiting(self):
        h = self.with_baseline()
        lane = h.lane("improve")
        real = grading.evaluate

        def flaky(task, ws, seeds, **kw):
            if kw["purpose"] == "replication":
                grade = grading.GradeResult(valid=False, score=None, grader_hash=trusted_hash(task), split="sealed", seeds=list(seeds),
                                            error="simulated")
                return grading.EvalResult(grade=grade, runs=[], failure_class="infra", conditions_hash="x" * 12, outputs_hashes=[])
            return real(task, ws, seeds, **kw)

        with mock.patch.object(grading, "evaluate", side_effect=flaky):
            res = h.engine.replicate(lane.exp_id)
        self.assertEqual((res.outcome, res.replicated), (Outcome.INFRA, False))
        rec = [e for e in h.events(Kind.EXP_RECORDED, exp_id="e001")][-1].data
        self.assertEqual((rec["phase"], rec["outcome"]), ("replication", Outcome.INFRA))
        self.assertEqual([e for e in h.events(Kind.EXP_GRADED, exp_id="e001") if e.data["purpose"] == "replication"], [])
        self.assertTrue(h.view().exp("e001").awaiting_replication)
        self.assertEqual(h.view().lessons[-1]["classification"], "infra")
        self.assertEqual(h.view().incumbent_id, "e000")
        again = h.engine.replicate("e001")                                   # a retry works and is not charged twice
        self.assertTrue(again.replicated)
        self.assertEqual(len(self.charged(h, "replication")), 1)

    def test_a_protected_violation_during_replication_ends_the_wait(self):
        h = self.with_baseline()
        h.lane("improve")
        with open(h.run_dir / "exp/e001/workspace/model.py", "a") as fh:
            fh.write("\n# sneaky edit after the lane ended\n")
        self.assertTrue(h.view().exp("e001").awaiting_replication)
        res = h.engine.replicate("e001")
        self.assertEqual((res.replicated, res.outcome), (False, Outcome.INVALID))
        exp = h.view().exp("e001")
        self.assertFalse(exp.awaiting_replication)                          # it can no longer be replicated over and over
        sealed = [e.data for e in h.events(Kind.EXP_GRADED, exp_id="e001") if e.data["purpose"] == "replication"]
        self.assertEqual([(g["split"], g["grade"]["valid"], g["grade"]["score"]) for g in sealed], [("sealed", False, None)])
        again = h.engine.replicate("e001")
        self.assertEqual((again.blocked_reason, again.replicated), ("not_replicable", None))
        self.assertEqual(len(h.events(Kind.EXP_PROTECTED, exp_id="e001")), 1)
        self.assertEqual(len([e for e in h.events(Kind.EXP_RECORDED, exp_id="e001") if e.data["phase"] == "replication"]), 1)
        self.assertIn("no_gain (failed replication)", h.memory.tried_table())
        self.assertEqual(h.view().incumbent_id, "e000")

    def test_a_runtime_violation_on_the_sealed_seeds_ends_the_wait(self):
        h = self.with_baseline()
        h.lane("improve")
        real = grading.evaluate

        def evaluate(task, ws, seeds, **kw):
            if kw["purpose"] == "replication":
                raise ProtectedPathViolation(["model.py"], "runtime")
            return real(task, ws, seeds, **kw)
        with mock.patch.object(grading, "evaluate", side_effect=evaluate):
            res = h.engine.replicate("e001")
        self.assertEqual((res.replicated, res.outcome), (False, Outcome.INVALID))
        self.assertEqual(self.first(h, Kind.EXP_PROTECTED, "e001").data["kind"], "runtime")
        self.assertFalse(h.view().exp("e001").awaiting_replication)
        self.assertEqual(h.engine.replicate("e001").blocked_reason, "not_replicable")

    def test_a_promotion_that_fails_after_the_record_does_not_write_a_second_record(self):
        h = self.with_baseline()
        h.lane("improve")
        with mock.patch.object(sandbox, "freeze", side_effect=OSError("disk full")):
            res = h.engine.replicate("e001")
        self.assertTrue(res.replicated)
        self.assertIn("not promoted", res.notes)
        records = [e.data for e in h.events(Kind.EXP_RECORDED, exp_id="e001") if e.data["phase"] == "replication"]
        self.assertEqual([(r["outcome"], r["replicated"]) for r in records], [("improved", True)])
        self.assertEqual(h.view().incumbent_id, "e000")
        # the next resume finishes the promotion: nothing is re-run and nothing is charged again
        runs, charges = len(h.events(Kind.EXP_RUN_STARTED)), len(self.charged(h, "replication"))
        h.restart()
        out = h.engine.resume_pending()
        self.assertEqual([(r.exp_id, r.replicated, r.notes) for r in out], [("e001", True, "replicated and promoted")])
        self.assertEqual(h.view().incumbent_id, "e001")
        self.assertEqual((len(h.events(Kind.EXP_RUN_STARTED)), len(self.charged(h, "replication"))), (runs, charges))

    def test_replicating_on_a_task_without_a_sealed_split_is_refused(self):
        h = self.with_baseline()
        h.lane("improve")
        h.engine.task.sealed = False
        res = h.engine.replicate("e001")
        self.assertEqual((res.blocked_reason, res.replicated), ("not_replicable", None))


# =====================================================================================================
# the sealed canary (contract section 13)
# =====================================================================================================
def _forms(x):
    return {repr(x), f"{x:.8g}", f"{x:.6g}", f"{x:.4g}", f"{x:+.4g}", f"{x:+.6g}", f"{x:.3f}"}


def _floats(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and math.isfinite(obj):
            yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _floats(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _floats(v)


class SealedCanaryTests(EngineCase):
    def sealed_and_dev_forms(self, h):
        sealed, dev = set(), set()
        for e in h.ledger.events():
            d = e.data
            if e.kind == Kind.EXP_GRADED:
                target = sealed if d["split"] == "sealed" else dev
                for x in _floats(d["grade"]):
                    target |= _forms(x)
            elif e.kind == Kind.EXP_RECORDED:
                if d["phase"] == "replication":
                    for key in ("sealed_delta", "sealed_margin", "s_eff", "t"):
                        for x in _floats(d.get(key)):
                            sealed |= _forms(x)
                else:
                    for x in _floats({k: v for k, v in d.items() if k not in ("sealed_delta", "sealed_margin")}):
                        dev |= _forms(x)
            elif e.kind == Kind.INCUMBENT_PROMOTED:
                for key in ("sealed_score", "sealed_delta", "sealed_margin"):
                    for x in _floats(d.get(key)):
                        sealed |= _forms(x)
        return {f for f in sealed if len(f) >= 4} - dev, dev

    def haystack(self, h, results, extra=""):
        lessons = "\n".join(x["text"] for x in h.view().lessons)
        memory = (h.run_dir / "memory.md").read_text()
        return "\n".join([h.agents.context_text(), memory, lessons, json.dumps([r.to_dict() for r in results]), h.memory.digest(),
                          h.memory.tried_table(), extra])

    def check_canary(self, h, results, extra=""):
        sealed, dev = self.sealed_and_dev_forms(h)
        self.assertTrue(sealed, "the scenario produced no sealed numbers")
        hay = self.haystack(h, results, extra)
        for form in sorted(sealed):
            # a number, not a prefix of a longer one: "0.153" must not match "0.15312"
            found = re.search(r"(?<![0-9.])" + re.escape(form) + r"(?![0-9])", hay)
            self.assertIsNone(found, f"sealed number {form} reached an agent-facing string")
        self.assertNotIn("314159", hay)
        self.assertNotIn(str(SEALED_SENTINEL), hay)
        # the haystack is not trivially empty: dev numbers are there
        base = h.view().baseline.primary_grade.score
        self.assertIn(f"{base:.6g}", hay)

    def test_successful_replication_leaks_no_sealed_number(self):
        task = copy_mini_task(self.root / "offset", sealed="offset")
        h = self.with_baseline(task=task, reviewer_hard=True)
        results = h.lanes(["improve", "worse"])
        results.append(h.engine.replicate(results[0].exp_id))                     # promoted: sealed numbers now exist
        results += h.lanes(["improve_best"])                                       # a lane that starts from the promoted incumbent
        results.append(h.engine.replicate(results[-1].exp_id))
        reflected = h.engine.reflect("manual")
        self.assertTrue(h.view().promotions)
        self.check_canary(h, results, extra=json.dumps(reflected))
        self.assertTrue(h.agents.contexts_of(ReflectContext))
        sealed_grades = [e for e in h.events(Kind.EXP_GRADED) if e.data["split"] == "sealed"]
        self.assertTrue(all(g.data["grade"]["score"] > 0.3 for g in sealed_grades))   # the offset really applied

    def test_failed_replication_leaks_no_sealed_number(self):
        task = copy_mini_task(self.root / "flat2", sealed="flat")
        h = self.with_baseline(task=task)
        results = h.lanes(["improve", "improve_best"])
        results.append(h.engine.replicate(results[0].exp_id))
        results.append(h.engine.replicate(results[1].exp_id))
        self.assertEqual([r.replicated for r in results[2:]], [False, False])
        results += h.lanes(["noop"])
        reflected = h.engine.reflect("manual")
        self.check_canary(h, results, extra=json.dumps(reflected))

    def test_the_canary_catches_a_leak(self):
        task = copy_mini_task(self.root / "offset2", sealed="offset")
        h = self.with_baseline(task=task)
        res = h.lane("improve")
        rep = h.engine.replicate(res.exp_id)
        self.check_canary(h, [res, rep])                                           # clean
        leaked = repr(h.view().exp("e001").replication_grade.score)
        with self.assertRaises(AssertionError):
            self.check_canary(h, [res, rep], extra="the sealed score was " + leaked)
        with self.assertRaises(AssertionError):
            self.check_canary(h, [res, rep], extra=f"sealed delta {h.view().exp('e001').sealed_delta:.6g}")

    def test_sealed_grades_are_only_in_the_ledger(self):
        h = self.with_baseline()
        res = h.lane("improve")
        rep = h.engine.replicate(res.exp_id)
        sealed = [e.data["grade"]["score"] for e in h.events(Kind.EXP_GRADED) if e.data["split"] == "sealed"]
        self.assertEqual(len(sealed), 2)
        text = json.dumps([res.to_dict(), rep.to_dict()]) + h.agents.context_text() + (h.run_dir / "memory.md").read_text()
        for score in sealed:
            self.assertNotIn(repr(score), text)
            self.assertNotIn(f"{score:.6g}", text)


# =====================================================================================================
# reflection, memory
# =====================================================================================================
class ReflectionTests(EngineCase):
    def test_reflect_writes_lessons_and_pivots(self):
        h = self.with_baseline()
        h.lane("noop")
        h.lane("worse")
        out = h.engine.reflect("manual")
        self.assertEqual(set(out), {"lessons", "pivots"})
        self.assertEqual(out["pivots"], ["try a different model family"])
        self.assertEqual(len(out["lessons"]), 1)
        lesson = h.view().lessons[-1]
        self.assertEqual((lesson["actor"], lesson["classification"], lesson["kind"]), ("reflector", "model", "strategy"))
        self.assertEqual(lesson["source_exps"], ["e001", "e002"])
        reflection = h.view().reflections[-1]
        self.assertEqual((reflection["trigger"], reflection["consolidated"], reflection["pivots"]),
                         ("manual", [lesson["lesson_id"]], out["pivots"]))
        ctx = h.agents.contexts_of(ReflectContext)[-1]
        self.assertEqual(ctx.trigger, "manual")
        self.assertEqual(ctx.experiments_table, h.memory.tried_table())
        self.assertIn("e001", ctx.incumbent_summary + ctx.experiments_table)
        self.assertIn("Fake reflection 1", (h.run_dir / "memory.md").read_text())

    def test_reflection_output_is_untrusted_and_classified_by_the_engine(self):
        h = self.with_baseline()
        h.lane("infra_always")
        h.lane("noop")
        h.agents.reflection = ReflectionResult(lessons=[
            {"text": "env was broken", "source_exps": ["e001"], "classification": "model", "kind": "observation"},
            {"text": "no valid source\x00", "source_exps": ["e999", "../etc", 7], "kind": "nonsense"},
            {"text": "mixed sources", "source_exps": ["e001", "e002"]},
            {"text": "   ", "source_exps": []}, "garbage", {"source_exps": ["e001"]}],
            pivots=["p" * 500, 5, "second pivot"])
        out = h.engine.reflect("bogus-trigger")
        lessons = h.view().lessons
        self.assertEqual([(x["classification"], x["source_exps"]) for x in lessons[-3:]],
                         [("infra", ["e001"]), ("model", []), ("model", ["e001", "e002"])])
        self.assertEqual(out["lessons"], ["env was broken", "no valid source", "mixed sources"])
        self.assertEqual(h.view().reflections[-1]["trigger"], "manual")
        self.assertEqual([len(p) for p in out["pivots"]], [200, len("second pivot")])

    def test_unavailable_reflector_returns_empty_and_writes_nothing(self):
        h = self.with_baseline()

        def down(ctx):
            raise ModelError("down", status=503)
        h.agents.reflect_hook = down
        before = len(h.ledger.events())
        self.assertEqual(h.engine.reflect("manual"), {"lessons": [], "pivots": []})
        self.assertEqual(len(h.ledger.events()), before)

    def test_reflector_budget_error_propagates_from_an_explicit_reflect(self):
        h = self.with_baseline()

        def broke(ctx):
            raise BudgetExceeded("no money", kind="llm")
        h.agents.reflect_hook = broke
        with self.assertRaises(BudgetExceeded):
            h.engine.reflect("manual")

    def test_auto_reflection_fires_once_per_streak(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=2, stall_after=2)
        h = self.with_baseline(limits=limits)
        for i in (3, 4, 5, 6):
            h.agents.patches[f"noop{i}"] = Patch({"model.py": PATCHES["noop"].files["model.py"] + f"\n# variant {i}\n"})
        h.lane("noop")
        self.assertEqual(h.agents.reflect_calls, 0)
        h.lane("noop2")
        self.assertEqual(h.agents.reflect_calls, 1)                      # two in a row
        h.lane("worse")                                                  # the streak continues: no second reflection
        self.assertEqual(h.agents.reflect_calls, 1)
        h.lane("improve")                                                # an improvement ends the streak
        h.lane("noop3")
        self.assertEqual(h.agents.reflect_calls, 1)
        h.lane("noop4")
        self.assertEqual(h.agents.reflect_calls, 2)
        self.assertEqual([r["trigger"] for r in h.view().reflections], ["stall", "stall"])

    def test_auto_reflection_after_a_restart_does_not_repeat_within_a_streak(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=2, stall_after=2)
        h = self.with_baseline(limits=limits)
        h.lane("noop")
        h.lane("noop2")
        self.assertEqual(h.agents.reflect_calls, 1)
        h.restart()
        h.lane("worse")
        self.assertEqual(h.agents.reflect_calls, 1)

    def test_auto_reflection_with_no_budget_does_not_undo_the_experiment(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=2, stall_after=1)
        h = self.with_baseline(limits=limits)

        def broke(ctx):
            raise BudgetExceeded("no money", kind="llm")
        h.agents.reflect_hook = broke
        res = h.lane("noop")
        self.assertEqual((res.outcome, res.status), ("no_gain", ExpStatus.RECORDED))
        self.assertEqual(h.view().reflections, [])

    def test_memory_md_is_regenerated_atomically_after_each_record(self):
        h = self.with_baseline()
        path = h.run_dir / "memory.md"
        self.assertTrue(path.is_file())
        self.assertNotIn("e001", path.read_text())
        h.lane("improve")
        self.assertIn("e001", path.read_text())
        h.lane("worse")
        self.assertIn("e002", path.read_text())
        self.assertEqual([p.name for p in h.run_dir.iterdir() if p.name.endswith(".tmp") or p.name.startswith(".memory")], [])
        h.engine.regenerate_memory()
        self.assertIn("Dev-split numbers only", path.read_text())

    def test_wall_time_is_charged_after_each_experiment(self):
        h = self.with_baseline()
        h.lane("improve")
        self.assertGreaterEqual(len([e for e in h.events(Kind.BUDGET_CHARGED) if e.data["kind"] == "wall"]), 2)


# =====================================================================================================
# the reviewer event names the model that answered
# =====================================================================================================
class PlantedLaneTests(EngineCase):
    """Red-team lanes (``nightshift run --plant``): in a meta run whose suite carries ``suite.planted``, a lane with a
    ``scripted_id`` is designed and implemented by the planted (scripted attacker) suite; the review and everything else stay
    with the run's own roles."""

    def planted_harness(self, sub="meta", **kw):
        h = self.with_baseline(sub=sub, backend="meta", **kw)
        attackers = FakeAgents(h.task)
        attackers.implementer_identity = "scripted-implementer"
        h.suite.planted = attackers.suite("scripted", models={"designer": "scripted", "implementer": "scripted"})
        h.attackers = attackers
        return h

    def test_a_scripted_lane_is_designed_and_implemented_by_the_attackers_and_reviewed_by_the_real_reviewer(self):
        h = self.planted_harness()
        with mock.patch("nightshift.agents.scripted.planted_fault_for", return_value="overfit") as fault:
            res = h.lane("improve")
        self.assertEqual((res.outcome, res.review_verdict), ("improved", "accept"))
        self.assertEqual(fault.call_args.args, (h.task, "improve", False))
        self.assertEqual(self.first(h, Kind.EXP_PROPOSED, "e001").data["planted_fault"], "overfit")
        self.assertEqual(self.first(h, Kind.EXP_PROPOSED, "e001").data["scripted_id"], "improve")
        self.assertEqual(h.attackers.design_calls, 1)
        self.assertEqual(len(h.attackers.contexts_of(ImplementContext)), 1)
        self.assertEqual(h.agents.design_calls, 0)
        self.assertEqual(h.agents.contexts_of(ImplementContext), [])           # the real implementer never touched the lane
        self.assertEqual(h.agents.review_calls, ["fake-reviewer"])             # the real reviewer did
        self.assertEqual(h.attackers.review_calls, [])
        self.assertEqual(self.first(h, Kind.EXP_IMPLEMENTED, "e001").data["worker"], "scripted-implementer")

    def test_the_review_record_names_the_attacker_as_implementer_and_the_real_model_as_reviewer(self):
        h = self.planted_harness()
        h.lane("improve")
        reviewed = self.first(h, Kind.EXP_REVIEWED, "e001").data
        self.assertEqual(reviewed["implementer_model"], "scripted")           # not the run's implementer model
        self.assertEqual(reviewed["reviewer_model"], "fake-model")
        self.assertTrue(reviewed["independent"])
        self.assertEqual(reviewed["independent_basis"], "model")

    def test_a_lane_without_a_scripted_id_uses_the_real_roles_even_when_attackers_are_attached(self):
        h = self.planted_harness()
        res = h.engine.run_lane(STAGES_FULL, LaneSpec(lane="A", hypothesis="a real idea from the supervisor", scripted_id=None))
        self.assertEqual(res.status, ExpStatus.RECORDED)
        self.assertEqual(h.agents.design_calls, 1)
        self.assertEqual(len(h.agents.contexts_of(ImplementContext)), 1)
        self.assertEqual(h.attackers.design_calls, 0)
        self.assertEqual(h.attackers.contexts_of(ImplementContext), [])
        proposed = self.first(h, Kind.EXP_PROPOSED, "e001").data
        self.assertEqual((proposed["scripted_id"], proposed["planted_fault"]), (None, None))
        self.assertEqual(self.first(h, Kind.EXP_REVIEWED, "e001").data["implementer_model"], "fake-model")

    def test_a_scripted_id_without_attackers_is_never_labelled_a_planted_fault_in_a_meta_run(self):
        h = self.with_baseline(sub="plain_meta", backend="meta")
        with mock.patch("nightshift.agents.scripted.planted_fault_for", return_value="tamper") as fault:
            h.lane("improve")
        fault.assert_not_called()
        self.assertIsNone(self.first(h, Kind.EXP_PROPOSED, "e001").data["planted_fault"])

    def test_the_planted_fault_of_a_lane_changes_nothing_the_engine_decides(self):
        outcomes = []
        for sub, label in (("labelled", "tamper"), ("unlabelled", None)):
            h = self.planted_harness(sub=sub)
            with mock.patch("nightshift.agents.scripted.planted_fault_for", return_value=label):
                res = h.lane("improve")
            rec = self.first(h, Kind.EXP_RECORDED, "e001").data
            outcomes.append((res.outcome, res.status, rec["delta"], rec["noise_margin"], rec["outcome"]))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_a_broken_bank_never_breaks_a_planted_lane(self):
        h = self.planted_harness()
        with mock.patch("nightshift.agents.scripted.planted_fault_for", side_effect=RuntimeError("bank broken")):
            res = h.lane("improve")
        self.assertEqual(res.outcome, "improved")
        self.assertIsNone(self.first(h, Kind.EXP_PROPOSED, "e001").data["planted_fault"])

    def test_the_attackers_tampering_is_caught_by_the_engine_not_by_a_role(self):
        h = self.planted_harness()
        res = h.lane("tamper")
        self.assertEqual((res.status, res.outcome), (ExpStatus.RECORDED, Outcome.INVALID))
        self.assertTrue(h.events(Kind.EXP_PROTECTED, exp_id="e001"))
        self.assertEqual(h.agents.review_calls, [])                              # a protected-path violation is never reviewed

    def test_a_resumed_planted_lane_goes_back_to_the_attackers(self):
        h = self.planted_harness()
        crash_when(h.engine, lambda kind, data: kind == Kind.EXP_IMPLEMENTED)
        with self.assertRaises(SimulatedCrash):
            h.lane("improve")
        h.restart()
        h.suite.planted = h.attackers.suite("scripted")
        with mock.patch("nightshift.agents.scripted.planted_fault_for", return_value="overfit"):
            h.engine.reconcile()
            h.engine.resume_pending()
        self.assertEqual(len(h.attackers.contexts_of(ImplementContext)), 2)     # once before the crash, once after
        self.assertEqual(h.agents.contexts_of(ImplementContext), [])
        self.assertEqual(self.first(h, Kind.EXP_RECORDED, "e001").data["outcome"], "improved")

    def test_a_scripted_attackers_patch_can_never_be_replicated_or_promoted(self):
        """Verifier regression: a planted lane whose patch improves on dev (the bank's ``duplicate`` is a twin of an honest idea) could be
        replicated and promoted by the supervisor, so a SCRIPTED patch would have decided the verdict of a run reported as the model
        roles' work."""
        h = self.planted_harness()
        lane = h.lane("improve")
        self.assertEqual(lane.outcome, "improved")
        res = h.engine.replicate(lane.exp_id)
        self.assertEqual((res.status, res.blocked_reason), (ExpStatus.BLOCKED, "not_replicable"))
        self.assertIn("scripted red-team attacker", res.notes)
        self.assertEqual(h.events(Kind.INCUMBENT_PROMOTED), [])
        self.assertEqual([e for e in h.events(Kind.EXP_GRADED) if e.data.get("purpose") == "replication"], [])
        # a scripted-backend run has no attackers suite and replicates exactly as before
        plain = self.with_baseline(sub="plain_scripted", backend="scripted")
        again = plain.lane("improve")
        self.assertNotEqual(plain.engine.replicate(again.exp_id).blocked_reason, "not_replicable")

    def test_a_real_lane_cannot_be_built_on_an_attackers_patch_but_an_attacker_can_be_built_on_the_incumbent(self):
        h = self.planted_harness()
        attacker = h.lane("improve")
        res = h.engine.run_lane(STAGES_FULL, LaneSpec(lane="A", hypothesis="a real idea from the supervisor", base=attacker.exp_id))
        self.assertEqual((res.status, res.blocked_reason), (ExpStatus.BLOCKED, "invalid_composition"))
        self.assertIn("cannot be the base", res.notes)
        ok = h.engine.run_lane(STAGES_FULL, LaneSpec(lane="B", hypothesis="another real idea", base="incumbent"))
        self.assertEqual(ok.status, ExpStatus.RECORDED)

    def test_the_attackers_get_no_sealed_number(self):
        h = self.planted_harness(sub="canary", task=copy_mini_task(self.root / "canary_task", sealed="offset"))
        lane = h.lane("improve")
        h.engine.replicate(lane.exp_id)
        self.assertNotIn(str(SEALED_SENTINEL), h.attackers.context_text())
        self.assertNotIn(str(SEALED_SENTINEL), h.agents.context_text())


class AnsweringReviewer:
    """A reviewer that exposes ``answering_model`` like ``LLMReviewer`` (the model of its latest call in this thread)."""

    def __init__(self, identity="answering-reviewer", answering=None, *, raises=None, broken_property=False):
        self.identity, self._answering, self.raises, self.broken_property = identity, answering, raises, broken_property
        self.calls = 0

    @property
    def answering_model(self):
        if self.broken_property:
            raise RuntimeError("the property is broken")
        return self._answering

    def review(self, ctx):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return ReviewVerdict(verdict="accept", reviewer=self.identity, findings=[])


class ReviewerHonestyTests(EngineCase):
    def reviewed(self, h, exp_id="e001", rnd=0):
        return self.first(h, Kind.EXP_REVIEWED, exp_id, review_round=rnd).data

    def test_a_fallback_model_that_answered_is_named_not_the_configured_one(self):
        h = self.with_baseline()
        h.suite.models["reviewer"] = "wonder-configured"
        h.suite.reviewer = AnsweringReviewer(answering="spark-answered")
        h.lane("improve")
        data = self.reviewed(h)
        self.assertEqual((data["reviewer_model"], data["implementer_model"]), ("spark-answered", "fake-model"))
        self.assertEqual(data["independent_basis"], "model")             # it really is another model than the implementer's
        self.assertEqual(data["reviewer"], "answering-reviewer")

    def test_independence_by_model_is_dropped_when_the_model_that_answered_is_the_implementers(self):
        h = self.with_baseline()
        h.suite.models["reviewer"] = "wonder-configured"                  # configured: different, so the old code said "model"
        h.suite.reviewer = AnsweringReviewer(answering="fake-model")      # but the implementer's own model answered
        h.lane("improve")
        data = self.reviewed(h)
        self.assertEqual((data["reviewer_model"], data["independent_basis"]), ("fake-model", "context"))

    def test_a_version_suffixed_answer_of_the_implementers_model_is_not_an_independent_model(self):
        # the host may answer with a dated id of the very model that implemented the change ("fake-model-0615" for
        # "fake-model"); LLMReviewer already counts that as the same model, so the ledger must not call it independent
        for sub, answered, basis in (("a", "fake-model-0615", "context"), ("b", "fake-model", "context"),
                                     ("c", "fake-model2", "model"), ("d", "fake", "context"), ("e", "fake-modelling", "model")):
            with self.subTest(answered=answered):
                h = self.with_baseline(sub=sub)
                h.suite.models["reviewer"] = "wonder-configured"
                h.suite.reviewer = AnsweringReviewer(answering=answered)
                h.lane("improve")
                data = self.reviewed(h)
                self.assertEqual((data["reviewer_model"], data["independent_basis"]), (answered, basis))

    def test_without_an_answering_model_the_configured_one_is_recorded_as_before(self):
        h = self.with_baseline()
        h.suite.models["reviewer"] = "another-model"
        h.suite.reviewer = AnsweringReviewer(answering=None)
        h.lane("improve")
        data = self.reviewed(h)
        self.assertEqual((data["reviewer_model"], data["independent_basis"]), ("another-model", "model"))

    def test_a_stale_answering_model_is_not_credited_with_a_review_that_failed(self):
        h = self.with_baseline()
        h.suite.reviewer = AnsweringReviewer(answering="stale-from-an-earlier-call", raises=ModelError("down", status=503))
        res = h.lane("improve")
        data = self.reviewed(h)
        self.assertEqual(data["findings"][0]["id"], "reviewer_unavailable")
        self.assertEqual(data["reviewer_model"], "fake-model")           # the configured name: nobody answered
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)

    def test_a_broken_or_non_string_attribute_never_costs_the_lane_its_review(self):
        for sub, reviewer in (("a", AnsweringReviewer(broken_property=True)), ("b", AnsweringReviewer(answering=7)),
                              ("c", AnsweringReviewer(answering="   "))):
            with self.subTest(sub=sub):
                h = self.with_baseline(sub=sub)
                h.suite.reviewer = reviewer
                res = h.lane("improve")
                self.assertEqual(res.outcome, "improved")
                self.assertEqual(self.reviewed(h)["reviewer_model"], "fake-model")

    def test_a_simulated_crash_raised_while_reading_the_attribute_is_not_swallowed(self):
        class Crashing(AnsweringReviewer):
            @property
            def answering_model(self):
                raise SimulatedCrash("kill -9")
        h = self.with_baseline()
        h.suite.reviewer = Crashing()
        with self.assertRaises(SimulatedCrash):
            h.lane("improve")

    def test_the_hard_review_reads_the_hard_reviewers_own_answering_model(self):
        h = self.with_baseline(reviewer_hard=True)
        h.suite.reviewer = AnsweringReviewer("soft", answering="spark-soft")
        h.suite.reviewer_hard = AnsweringReviewer("hard", answering="spark-instead-of-wonder")
        res = h.lane("improve", stages=["implement", "run", "grade", "review", "review_hard", "repair", "record"])
        self.assertEqual(res.outcome, "improved")
        self.assertEqual((self.reviewed(h, rnd=0)["reviewer_model"], self.reviewed(h, rnd=1)["reviewer_model"]),
                         ("spark-soft", "spark-instead-of-wonder"))
        self.assertEqual(self.reviewed(h, rnd=1)["reviewer"], "hard")


# =====================================================================================================
# HALT in the middle of a lane
# =====================================================================================================
class HaltInFlightTests(EngineCase):
    def recorded_lane(self, h, exp_id):
        return [e.data for e in h.events(Kind.EXP_RECORDED, exp_id=exp_id) if e.data["phase"] == "lane"]

    def test_a_lane_halted_after_grading_and_review_is_recorded_unresolved_with_its_dev_numbers_kept(self):
        h = self.with_baseline()

        def accept_and_halt(ctx):
            h.halt()
            return ReviewVerdict(verdict="accept", reviewer="fake-reviewer", findings=[])
        h.agents.review_script = accept_and_halt
        with self.assertRaises(HaltRequested):
            h.lane("improve")
        (rec,) = self.recorded_lane(h, "e001")
        self.assertEqual((rec["outcome"], rec["note"]), ("unresolved", "stopped by HALT"))
        self.assertIsNotNone(rec["delta"])                                # the dev comparison is still on record
        view = h.view()
        self.assertEqual((view.exp("e001").status, view.exp("e001").dev_outcome), (ExpStatus.RECORDED, "unresolved"))
        self.assertEqual(view.in_flight(), [])
        self.assertFalse(view.exp("e001").awaiting_replication)
        self.assertEqual(h.ledger.verify_chain(), (True, None))

    def test_open_runs_are_closed_as_interrupted_before_the_record(self):
        h = self.with_baseline()

        def orphan_run_then_halt(ctx):
            h.ledger.append(Kind.EXP_RUN_STARTED, "engine", {"exp_id": ctx.exp_id, "run_idx": 0, "purpose": "primary", "seed": 0,
                                                             "cmd": ["python"], "pid": 1, "attempt": 1, "isolation": "none"})
            h.halt()
        h.agents.on_implement["improve"] = orphan_run_then_halt
        with self.assertRaises(HaltRequested):
            h.lane("improve")
        finished = h.events(Kind.EXP_RUN_FINISHED, exp_id="e001")
        self.assertEqual([(e.data["run_idx"], e.data["classification"]) for e in finished], [(0, "interrupted")])
        kinds = [e.kind for e in self.lane_events(h, "e001")]
        self.assertLess(kinds.index(Kind.EXP_RUN_FINISHED), kinds.index(Kind.EXP_RECORDED))
        self.assertEqual(h.engine.reconcile(), [])                          # nothing left for a restart to clean up

    def test_a_halt_writes_no_lesson_and_spends_no_reflection(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=3, max_replications=6, max_repairs=2, stall_after=1)
        h = self.with_baseline(limits=limits)
        h.lane("noop")
        self.assertEqual(h.agents.reflect_calls, 1)                         # a normal non-improvement does reflect at stall_after=1
        lessons_before = len(h.view().lessons)
        h.agents.on_implement["improve"] = lambda ctx: h.halt()
        with self.assertRaises(HaltRequested):
            h.lane("improve")
        self.assertEqual(h.agents.reflect_calls, 1)
        self.assertEqual(len(h.view().lessons), lessons_before)

    def test_every_in_flight_lane_of_a_parallel_batch_is_closed(self):
        h = self.with_baseline()
        h.agents.implement_delay = 0.05
        h.agents.on_implement["improve"] = lambda ctx: h.halt()
        with self.assertRaises(HaltRequested):
            h.lanes(["improve", "improve_mid", "noop"])
        view = h.view()
        self.assertEqual(view.in_flight(), [])
        proposed = [e for e in view.experiments.values() if e.lane != "baseline"]
        self.assertTrue(proposed)
        self.assertTrue(all(e.status == ExpStatus.RECORDED for e in proposed))
        halted = [e for e in proposed if e.dev_outcome == "unresolved"]
        self.assertTrue(halted)
        for e in halted:
            (rec,) = self.recorded_lane(h, e.exp_id)
            self.assertEqual(rec["note"], "stopped by HALT")
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith("nightshift-lane")])
        self.assertEqual(h.ledger.verify_chain(), (True, None))

    def test_a_halt_before_the_proposal_leaves_nothing_to_close(self):
        h = self.with_baseline()
        h.halt()
        before = len(h.ledger.events())
        with self.assertRaises(HaltRequested):
            h.lane("improve")
        self.assertEqual(len(h.ledger.events()), before)

    def test_a_lane_that_is_already_recorded_is_not_recorded_twice(self):
        h = self.with_baseline()
        h.lane("improve")
        ctx = engine_module._Lane(exp_id="e001", spec=LaneSpec(lane="A", hypothesis="x"), stages=frozenset(),
                                  plan=ExperimentPlan(hypothesis="x", approach=""), base_id=None, base_dir=None)
        before = len(h.ledger.events())
        h.engine._stop_lane(ctx, "stopped by HALT")
        self.assertEqual(len(h.ledger.events()), before)

    def test_a_failure_while_closing_never_replaces_the_halt(self):
        h = self.with_baseline()
        h.agents.on_implement["improve"] = lambda ctx: h.halt()

        def broken(ctx, note, outcome=Outcome.UNRESOLVED):
            raise RuntimeError("bookkeeping failed")
        h.engine._abort = broken
        with self.assertRaises(HaltRequested):
            h.lane("improve")

    def test_a_halted_baseline_is_recorded_unresolved(self):
        h = self.make()
        real = h.engine._freeze_baseline

        def freeze_then_halt(exp_id, ws):
            out = real(exp_id, ws)
            h.halt()
            return out
        h.engine._freeze_baseline = freeze_then_halt
        with self.assertRaises(HaltRequested):
            h.baseline()
        (rec,) = self.recorded_lane(h, "e000")
        self.assertEqual((rec["outcome"], rec["note"]), ("unresolved", "stopped by HALT"))
        view = h.view()
        self.assertIsNone(view.baseline_id)
        self.assertEqual(view.in_flight(), [])

    def test_a_replication_cut_off_by_halt_propagates_it_and_leaves_no_open_run(self):
        h = self.with_baseline()
        lane = h.lane("improve")
        real = h.engine._ensure_baseline_sealed

        def sealed_baseline_then_halt(base_id):
            out = real(base_id)
            h.halt()
            return out
        h.engine._ensure_baseline_sealed = sealed_baseline_then_halt
        with self.assertRaises(HaltRequested):
            h.engine.replicate(lane.exp_id)
        self.assertEqual(h.engine.reconcile(), [])
        self.assertEqual(len(self.charged(h, "replication")), 1)            # the query was charged: it is not refunded


# =====================================================================================================
# the designed plan is on record
# =====================================================================================================
class PlanSummaryTests(EngineCase):
    def test_implemented_carries_the_designers_plan_not_the_supervisors_hint(self):
        h = self.with_baseline()
        h.lane("improve", approach="the supervisor's hint")
        proposed = self.first(h, Kind.EXP_PROPOSED, "e001").data
        implemented = self.first(h, Kind.EXP_IMPLEMENTED, "e001").data
        self.assertEqual(proposed["plan"], "the supervisor's hint")           # unchanged: the hint as proposed
        self.assertEqual(implemented["plan_summary"], "fake plan for improve Expected effect: fake")

    def test_without_a_design_stage_the_summary_is_the_hint(self):
        h = self.with_baseline()
        h.lane("improve", stages=STAGES_MIN, approach="just do it")
        self.assertEqual(self.first(h, Kind.EXP_IMPLEMENTED, "e001").data["plan_summary"], "just do it")
        h.lane("improve_mid", stages=STAGES_MIN)
        self.assertEqual(self.first(h, Kind.EXP_IMPLEMENTED, "e002").data["plan_summary"], "")

    def test_the_summary_is_sanitised_and_capped_at_600_characters(self):
        class Rambler:
            identity = "rambler"

            def design(self, ctx):
                return ExperimentPlan(hypothesis=ctx.lane.hypothesis, approach="a\x00b\x07 " + "x" * 3000,
                                      expected_effect="effect " + "y" * 3000)
        h = self.with_baseline()
        h.suite.designer = Rambler()
        h.lane("improve")
        summary = self.first(h, Kind.EXP_IMPLEMENTED, "e001").data["plan_summary"]
        self.assertLessEqual(len(summary), 600)
        self.assertTrue(summary.startswith("ab xxx"))
        self.assertNotIn("\x00", summary)
        self.assertNotIn("\x07", summary)
        self.assertTrue(summary.endswith("..."))

    def test_the_plan_never_carries_a_sealed_number(self):
        task = copy_mini_task(self.root / "offset", sealed="offset")
        h = self.with_baseline(task=task)
        lane = h.lane("improve")
        h.engine.replicate(lane.exp_id)
        for e in h.events(Kind.EXP_IMPLEMENTED):
            self.assertNotIn("314159", e.data.get("plan_summary", ""))

    def test_a_resumed_repair_starts_from_the_designed_plan_not_the_hint(self):
        h = self.with_baseline()
        crash_when(h.engine, lambda k, d: d.get("exp_id") == "e001" and k == Kind.EXP_REPAIRED)
        with self.assertRaises(SimulatedCrash):
            h.lane("defective", approach="the supervisor's hint")
        planned = self.first(h, Kind.EXP_IMPLEMENTED, "e001").data["plan_summary"]
        self.assertIn("fake plan for defective", planned)
        h.restart()
        seen = len(h.agents.contexts_of(ImplementContext))
        results = h.engine.resume_pending()
        self.assertEqual([(r.exp_id, r.outcome) for r in results], [("e001", "improved")])
        resumed = h.agents.contexts_of(ImplementContext)[seen]
        self.assertTrue(resumed.repair_findings)
        self.assertEqual(resumed.plan.approach, planned)
        self.assertNotIn("supervisor's hint", resumed.plan.approach)

    def test_a_ledger_from_before_the_key_existed_falls_back_to_the_hint(self):
        h = self.with_baseline()
        crash_when(h.engine, lambda k, d: d.get("exp_id") == "e001" and k == Kind.EXP_REPAIRED)
        with self.assertRaises(SimulatedCrash):
            h.lane("defective", approach="the supervisor's hint")
        h.restart()
        h.engine._recorded_plan = lambda exp_id: ""                       # as if the implemented event had no plan_summary
        seen = len(h.agents.contexts_of(ImplementContext))
        h.engine.resume_pending()
        self.assertEqual(h.agents.contexts_of(ImplementContext)[seen].plan.approach, "the supervisor's hint")

    def test_recorded_plan_of_an_experiment_that_was_never_implemented_is_empty(self):
        h = self.with_baseline()
        self.assertEqual(h.engine._recorded_plan("e000"), "")
        self.assertEqual(h.engine._recorded_plan("e999"), "")


# =====================================================================================================
# an implementer that changes nothing
# =====================================================================================================
class EmptyImplementationTests(EngineCase):
    NOTE = "the implementer made no change; the hypothesis is untested"

    def setUp(self):
        super().setUp()
        self.h = self.with_baseline()
        self.h.agents.patches["empty"] = Patch({}, note="writes nothing")

    def test_it_is_not_evaluated_and_is_recorded_unresolved(self):
        h = self.h
        res = h.lane("empty")
        self.assertEqual((res.outcome, res.status), (Outcome.UNRESOLVED, ExpStatus.RECORDED))
        self.assertIn(self.NOTE, res.notes)
        self.assertEqual(h.events(Kind.EXP_RUN_STARTED, exp_id="e001"), [])
        self.assertEqual(h.events(Kind.EXP_GRADED, exp_id="e001"), [])
        self.assertEqual(h.events(Kind.EXP_REVIEWED, exp_id="e001"), [])
        self.assertEqual(h.agents.contexts_of(ReviewContext), [])
        (rec,) = [e.data for e in h.events(Kind.EXP_RECORDED, exp_id="e001")]
        self.assertEqual((rec["outcome"], rec["note"], rec["delta"], rec["vs_exp_id"]), ("unresolved", self.NOTE, None, None))
        self.assertEqual(h.view().in_flight(), [])

    def test_no_implemented_event_is_written_so_empty_changes_never_look_like_duplicates(self):
        h = self.h
        first = h.lane("empty")
        second = h.lane("empty")
        self.assertEqual(h.events(Kind.EXP_IMPLEMENTED), [])
        self.assertEqual(h.events(Kind.EXP_BLOCKED), [])
        self.assertEqual((first.outcome, second.outcome), (Outcome.UNRESOLVED, Outcome.UNRESOLVED))
        self.assertNotEqual(first.exp_id, second.exp_id)

    def test_the_lesson_is_infra_classified_and_does_not_call_the_hypothesis_a_failure(self):
        h = self.h
        h.lane("empty")
        (lesson,) = h.view().lessons
        self.assertEqual((lesson["classification"], lesson["source_exps"]), ("infra", ["e001"]))
        self.assertIn("could not be turned into a change", lesson["text"])
        self.assertIn("not tested", lesson["text"])
        self.assertNotIn("fix the code", lesson["text"])
        self.assertIn("Environment notes", h.memory.digest())            # shown to agents as "not evidence against any idea"

    def test_it_does_not_count_toward_the_stall_streak_or_trigger_a_reflection(self):
        limits = BudgetLimits(max_experiments=16, max_wall_sec=1e5, max_parallel=3, max_replications=6, max_repairs=2, stall_after=2)
        h = self.with_baseline(sub="stall", limits=limits)
        h.agents.patches["empty"] = Patch({})
        for _ in range(3):
            h.lane("empty")
        self.assertEqual(gates.stall_streak(h.view()), 0)
        self.assertEqual(h.agents.reflect_calls, 0)
        h.lane("noop")
        h.lane("noop2")
        self.assertEqual(gates.stall_streak(h.view()), 2)
        self.assertEqual(h.agents.reflect_calls, 1)                        # two real non-improvements do trigger it, once

    def test_a_reflection_lesson_about_only_untested_experiments_is_infra(self):
        h = self.h
        h.lane("empty")
        h.agents.reflection = ReflectionResult(lessons=[{"text": "Ideas must be concrete enough to implement.", "kind": "strategy",
                                                         "source_exps": ["e001"], "classification": "unknown"}], pivots=[])
        h.engine.reflect("manual")
        self.assertEqual(h.view().lessons[-1]["classification"], "infra")

    def test_the_artifacts_are_still_written_and_the_budget_slot_is_spent(self):
        h = self.h
        h.lane("empty")
        self.assertTrue((h.run_dir / "exp/e001/diff.patch").is_file())
        self.assertTrue((h.run_dir / "exp/e001/worker.log").is_file())
        self.assertEqual(len(self.charged(h, "experiment")), 1)            # the baseline is free, e001 is the only charge

    def test_the_same_hypothesis_can_be_tried_again_because_it_was_never_tested(self):
        h = self.h
        first = h.engine.run_lane(STAGES_FULL, h.agents.lane("empty", hypothesis="Make the model smarter"))
        self.assertEqual(first.outcome, Outcome.UNRESOLVED)
        h.agents.patches["empty"] = PATCHES["improve"]                 # this time the implementer manages to change something
        second = h.engine.run_lane(STAGES_FULL, h.agents.lane("empty", hypothesis="make the model SMARTER!"))
        self.assertEqual((second.outcome, second.status), ("improved", ExpStatus.RECORDED))
        self.assertEqual(h.events(Kind.EXP_BLOCKED), [])

    def test_a_repair_that_changes_nothing_keeps_its_own_note(self):
        h = self.h
        h.agents.patches["defective_stubborn"] = Patch(
            {"model.py": PATCHES["defective_stubborn"].files["model.py"]},
            repair_files={"model.py": PATCHES["defective_stubborn"].files["model.py"]})
        res = h.lane("defective_stubborn")
        self.assertEqual(res.outcome, Outcome.UNRESOLVED)
        self.assertNotIn(self.NOTE, res.notes)


# =====================================================================================================
# exactly crash_after_runs runs start
# =====================================================================================================
class CrashHookRaceTests(EngineCase):
    def slow_run_started(self, h, delay=0.05):
        """Make every ``run_started`` append slow: before the fix, other lanes slipped in between the append and the count."""
        real = h.ledger.append

        def slow(kind, actor, data):
            if kind == Kind.EXP_RUN_STARTED:
                time.sleep(delay)
            return real(kind, actor, data)
        h.ledger.append = slow

    def started(self, h):
        return [e for e in h.events(Kind.EXP_RUN_STARTED) if e.data["exp_id"] != "e000"]

    def test_exactly_that_many_runs_start_with_parallel_lanes(self):
        for n in (1, 2, 3):
            with self.subTest(crash_after_runs=n):
                h = self.with_baseline(sub=f"n{n}")
                h.restart(crash_after_runs=n)
                self.slow_run_started(h)
                with self.assertRaises(SimulatedCrash):
                    h.lanes(["improve", "improve_mid", "noop"])
                self.assertEqual(len(self.started(h)), n)
                self.assertTrue(h.engine.crashed)
                self.assertFalse([t for t in threading.enumerate() if t.name.startswith("nightshift-lane")])

    def test_one_crash_leaves_exactly_one_interrupted_run_to_reconcile(self):
        h = self.with_baseline()
        h.restart(crash_after_runs=1)
        self.slow_run_started(h)
        with self.assertRaises(SimulatedCrash):
            h.lanes(["improve", "improve_mid", "noop"])
        h.restart()
        touched = h.engine.reconcile()
        interrupted = [e for e in h.events(Kind.EXP_RUN_FINISHED) if e.data["classification"] == "interrupted"]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(touched, [interrupted[0].data["exp_id"]])

    def test_a_losing_lane_writes_nothing_after_the_crash(self):
        h = self.with_baseline()
        h.restart(crash_after_runs=1)
        self.slow_run_started(h)
        with self.assertRaises(SimulatedCrash):
            h.lanes(["improve", "improve_mid", "noop"])
        tripped = self.started(h)[0].seq
        self.assertEqual([e.kind for e in h.ledger.events() if e.seq > tripped and e.kind == Kind.EXP_RUN_STARTED], [])

    def test_the_run_counter_and_the_append_share_one_lock(self):
        source = inspect.getsource(engine_module.WorkflowEngine._append)
        self.assertEqual(len(re.findall(r"self\.ledger\.append\(", source)), 1)
        self.assertIn("self._crash_lock", source)


# =====================================================================================================
# a candidate a better incumbent already beats is not replicated
# =====================================================================================================
class SupersededReplicationTests(EngineCase):
    def two_winners(self, h):
        """weak (5 bins) and strong (20 bins) both improve on the baseline in the same batch; strong is replicated first."""
        weak = h.lane("improve")
        strong = h.lane("improve_best", base="template")           # both start from the baseline, so both are "improved"
        self.assertEqual((weak.outcome, strong.outcome), ("improved", "improved"))
        return weak, strong

    def test_the_dominated_candidate_is_refused_and_no_sealed_query_is_spent(self):
        h = self.with_baseline()
        weak, strong = self.two_winners(h)
        self.assertTrue(h.engine.replicate(strong.exp_id).replicated)
        charged = len(self.charged(h, "replication"))
        runs = len(h.events(Kind.EXP_RUN_STARTED, exp_id=weak.exp_id))
        res = h.engine.replicate(weak.exp_id)
        self.assertEqual((res.status, res.blocked_reason, res.replicated), (ExpStatus.BLOCKED, "not_replicable", None))
        self.assertIn("superseded on dev: a better result has been promoted since", res.notes)
        blocked = h.events(Kind.EXP_BLOCKED)[-1].data
        self.assertEqual((blocked["exp_id"], blocked["reason"], blocked["lane"]), (None, "not_replicable", "replicate"))
        self.assertEqual(len(self.charged(h, "replication")), charged)
        self.assertEqual(len(h.events(Kind.EXP_RUN_STARTED, exp_id=weak.exp_id)), runs)
        self.assertIsNone(h.view().exp(weak.exp_id).replication_grade)
        self.assertEqual(h.view().incumbent_id, strong.exp_id)

    def test_the_candidate_that_still_wins_can_be_replicated(self):
        h = self.with_baseline()
        weak, strong = self.two_winners(h)
        self.assertTrue(h.engine.replicate(weak.exp_id).replicated)        # weak first: strong still beats it afterwards
        self.assertTrue(h.engine.replicate(strong.exp_id).replicated)
        self.assertEqual(h.view().incumbent_id, strong.exp_id)

    def test_the_refusal_names_neither_the_hidden_split_nor_a_number(self):
        h = self.with_baseline()
        weak, strong = self.two_winners(h)
        h.engine.replicate(strong.exp_id)
        res = h.engine.replicate(weak.exp_id)
        self.assertNotIn("sealed", res.notes.lower())
        self.assertIsNone(re.search(r"\d", re.sub(r"\be\d{3}\b", "", res.notes)))


# =====================================================================================================
# what the agents read after a replication never names the hidden split
# =====================================================================================================
class ReplicationWordingTests(EngineCase):
    def test_a_failed_replication_leaves_no_mention_of_the_hidden_split_in_anything_an_agent_reads(self):
        task = copy_mini_task(self.root / "flat2", sealed="flat")
        h = self.with_baseline(task=task)
        lane = h.lane("improve")
        res = h.engine.replicate(lane.exp_id)
        agent_facing = "\n".join([(h.run_dir / "memory.md").read_text(), h.memory.digest(), h.memory.tried_table(),
                                  json.dumps(res.to_dict()), "\n".join(x["text"] for x in h.view().lessons)])
        self.assertNotIn("sealed", agent_facing.lower())
        self.assertEqual(res.notes, "did not replicate")

    def test_an_infrastructure_failure_during_replication_is_worded_the_same_way(self):
        h = self.with_baseline()
        lane = h.lane("improve")
        h.engine._finish_replication(lane.exp_id, Outcome.INFRA, False, "infrastructure failure during replication", lesson="infra")
        text = h.view().lessons[-1]["text"]
        self.assertNotIn("sealed", text.lower())
        self.assertIn("independent re-check", text)
        self.assertEqual(h.view().lessons[-1]["classification"], "infra")

    def test_a_replicated_but_unpromoted_note_does_not_name_the_split(self):
        h = self.with_baseline()
        lane = h.lane("improve")
        with mock.patch.object(gates, "promotion_gate", return_value=["not replicated: it did not replicate on the independent re-check"]):
            res = h.engine.replicate(lane.exp_id)
        self.assertTrue(res.replicated)
        self.assertNotIn("sealed", res.notes.lower())
        self.assertTrue(res.notes.startswith("replicated, not promoted:"))

    def test_a_task_without_a_sealed_split_is_refused_without_naming_it(self):
        h = self.with_baseline()
        lane = h.lane("improve")
        h.engine.task = dataclasses.replace(h.engine.task, sealed=False)
        res = h.engine.replicate(lane.exp_id)
        self.assertEqual(res.blocked_reason, "not_replicable")
        self.assertNotIn("sealed", res.notes.lower())


if __name__ == "__main__":
    unittest.main()
