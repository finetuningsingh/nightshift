"""Golden end-to-end test of the offline demo: real subprocesses, the real law-discovery task, the scripted backend.

The scripted backend is a FAULT-INJECTION HARNESS, not a language model. This test pins what the whole system does with
its planted faults: the exact outcome of every scripted proposal, the incumbent chain, the single finish refusal, the
audit findings (all closed, no false positive), crash and resume, the hash chain, the report, and the run-level sealed
canary (no sealed number reaches memory.md or any supervisor turn or nudge). ``ScriptedAgentCanary`` runs the golden demo once
more IN PROCESS with every role recorded, so that the default suite (not only the opt-in ``NIGHTSHIFT_INTEGRATION=1`` test of
``tests/test_orchestrator.py``) scans everything the scripted agents are handed for the needles of ``tests/support/canary.py``.

Runtime: about 90 to 110 seconds. ``setUpClass`` makes six CLI runs once (the full run about 17 s, the control run about
5 s, two killed runs under 1 s each and their resumes about 20 s each) and every test reads their ledgers. All runs go to a
temporary directory; nothing is written under the repository's ``runs/``.

Experiment ids of lanes that run in parallel are assigned in the order their threads reach the engine, so the full run is
compared by scripted id, never by exp id. The first crash run uses ``--parallel 1``: that makes the id order deterministic
(quadratic is e001, damped e011). The second crash run uses the default parallelism: the engine counts started runs and
writes ``run_started`` under one lock, so exactly ``crash_after_runs`` runs start however the lanes race and exactly one
untrusted run is in flight when the process is killed in both runs.

Run: ``python3 -m unittest tests.test_demo_e2e -v``
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable

from nightshift.ledger import Ledger
from nightshift.orchestrator import Orchestrator
from nightshift.task import load_task
from nightshift.types import Event, Kind, LedgerView
from nightshift.view import build_view
from tests.support.canary import find_leaks, number_forms, run_needles, sealed_floats

ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = ROOT / "tasks" / "law-discovery"
MANIFEST = json.loads((TASK_DIR / "scripted" / "manifest.json").read_text(encoding="utf-8"))
CLI_TIMEOUT_SEC = 300
CRASH_AFTER_RUNS = 4          # the baseline takes 3 untrusted runs, so the 4th dies inside the first candidate

# scripted_id -> (status, dev outcome, final outcome, replicated, repairs). "Final outcome" is the lane's outcome after any
# replication: poly_tuned improved on dev and then failed the sealed split.
EXPECTED_OUTCOMES = {
    "quadratic": ("recorded", "improved", "improved", True, 0),
    "poly9": ("recorded", "worse", "worse", False, 0),
    "tamper": ("recorded", "invalid", "invalid", False, 0),
    "claim_without_run": ("recorded", "no_gain", "no_gain", False, 0),
    "duplicate": ("blocked", None, None, False, 0),
    "infra_flake": ("recorded", "no_gain", "no_gain", False, 0),
    "quadratic_defect": ("recorded", "improved", "improved", False, 1),
    "injection": ("recorded", "no_gain", "no_gain", False, 0),
    "hang": ("recorded", "failed", "failed", False, 0),
    "poly_tuned": ("recorded", "improved", "no_gain", False, 0),
    "damped": ("recorded", "improved", "improved", True, 0),
}
# Findings by scripted id (exp ids are assigned in thread-completion order within a batch, so they are not stable).
EXPECTED_FINDINGS = {"I01@poly9.c1", "I01@tamper.c1", "I02@poly9", "I03@tamper", "I01@claim_without_run.c1",
                     "I09@quadratic", "I14@quadratic.c2", "I02@poly_tuned"}
# The words that assert a result. The scripted ``say`` lines may not contain them; only the generated sentence may.
OUTCOME_WORDS = re.compile(r"\b(replicated|beats?|improves?|wins?|succeeded|failed)\b", re.IGNORECASE)


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-m", "nightshift", *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=CLI_TIMEOUT_SEC, check=False)


def events_of(run_dir: Path) -> list[Event]:
    return list(Ledger.read_only(run_dir).events())


def lane_ids(view: LedgerView) -> dict[str, str]:
    """scripted_id -> exp_id for every non-replication lane experiment."""
    return {e.scripted_id: e.exp_id for e in view.experiments.values() if e.scripted_id and not e.replicate_of}


def outcomes(view: LedgerView) -> dict[str, tuple]:
    out = {}
    for sid, exp_id in lane_ids(view).items():
        e = view.experiments[exp_id]
        out[sid] = (e.status, e.dev_outcome, e.outcome, bool(e.replicated), len(e.repairs))
    return out


def chain(view: LedgerView) -> list[str]:
    """The incumbent chain as scripted ids: the baseline, then each promotion in order."""
    names = {exp_id: sid for sid, exp_id in lane_ids(view).items()}
    names[view.baseline_id] = "baseline"
    steps = [view.baseline_id]
    for promotion in view.promotions:
        assert promotion["prev_exp_id"] == steps[-1], "every promotion replaces the incumbent before it"
        steps.append(promotion["exp_id"])
    return [names[s] for s in steps]


def normalised_findings(view: LedgerView) -> set[str]:
    names = {exp_id: sid for sid, exp_id in lane_ids(view).items()}
    return {re.sub(r"e\d{3}", lambda m: names.get(m.group(0), m.group(0)), fid) for fid in view.findings}


def charges(events: Iterable[Event]) -> collections.Counter:
    """(kind, purpose, exp_id-or-None) of every experiment or replication charge. Wall-clock charges are timing, not work."""
    return collections.Counter((e.data.get("kind"), e.data.get("purpose"), e.data.get("exp_id")) for e in events
                               if e.kind == Kind.BUDGET_CHARGED and e.data.get("kind") in ("experiment", "replication"))


def sealed_numbers(events: Iterable[Event]) -> list[float]:
    """Every float with at least 6 significant digits that was derived from a replication grade: the grades themselves
    (score, std, per-seed values, metrics) and every ``sealed_*`` field of any event."""
    found: list[float] = []

    def walk(node: Any, force: bool = False) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, force or str(key).startswith("sealed_"))
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value, force)
        elif force and isinstance(node, float):
            found.append(node)

    for ev in events:
        if ev.kind == Kind.EXP_GRADED and ev.data.get("purpose") in ("replication", "replication_baseline"):
            walk(ev.data.get("grade"), True)
        else:
            walk(ev.data)
    keep = {x for x in found if x == x and abs(x) not in (0.0, float("inf")) and len(_digits(x)) >= 6}
    return sorted(keep)


def _digits(x: float) -> str:
    """The significant decimal digits of ``repr(x)``, without sign, point, leading zeros or exponent."""
    mantissa = repr(abs(float(x))).lower().split("e")[0]
    return re.sub(r"\D", "", mantissa).lstrip("0")


def digit_strings(x: float) -> set[str]:
    """First 6 significant digits of ``x``, both truncated (as printed in full) and rounded (as printed short)."""
    truncated = _digits(x)[:6]
    rounded = re.sub(r"\D", "", f"{abs(x):.5e}".split("e")[0])
    return {truncated, rounded}


def leaked(numbers: Iterable[float], text: str) -> list[float]:
    """The numbers one of whose 6-digit strings occurs in ``text`` (decimal points inside numbers are ignored)."""
    flat = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    return [x for x in numbers if any(d in flat for d in digit_strings(x))]


def agent_facing_text(run_dir: Path, events: Iterable[Event]) -> str:
    """memory.md plus the text of every supervisor turn and nudge (and the lessons, which are fed to later prompts)."""
    parts = [(run_dir / "memory.md").read_text(encoding="utf-8")]
    parts += [json.dumps(e.data, sort_keys=True) for e in events
              if e.kind in (Kind.SUP_TURN, Kind.SUP_NUDGE, Kind.LESSON, Kind.REFLECTION)]
    return "\n".join(parts)


# --------------------------------------------------------------------------------------------------
class GoldenRuns(unittest.TestCase):
    """Four CLI runs, made once: the full scripted run, the clean control, and a killed run that is then resumed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="nightshift-e2e-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.main_dir, cls.control_dir, cls.crash_dir = cls.tmp / "main", cls.tmp / "control", cls.tmp / "crash"
        cls.pcrash_dir = cls.tmp / "crash-parallel"
        base = ["run", "--task", str(TASK_DIR), "--backend", "scripted"]
        cls.main_proc = cli(*base, "--run-dir", str(cls.main_dir))
        cls.control_proc = cli(*base, "--control", "--run-dir", str(cls.control_dir))
        cls.crash_proc = cli(*base, "--run-dir", str(cls.crash_dir), "--parallel", "1", "--crash-after-runs", str(CRASH_AFTER_RUNS))
        cls.resume_proc = cli("resume", str(cls.crash_dir), "--task", str(TASK_DIR))
        cls.pcrash_proc = cli(*base, "--run-dir", str(cls.pcrash_dir), "--crash-after-runs", str(CRASH_AFTER_RUNS))
        cls.pcrash_events_at_crash = events_of(cls.pcrash_dir)
        cls.presume_proc = cli("resume", str(cls.pcrash_dir), "--task", str(TASK_DIR))
        cls.main_events, cls.control_events = events_of(cls.main_dir), events_of(cls.control_dir)
        cls.crash_events = events_of(cls.crash_dir)
        cls.pcrash_events = events_of(cls.pcrash_dir)
        cls.main, cls.control = build_view(cls.main_events), build_view(cls.control_events)
        cls.resumed = build_view(cls.crash_events)
        cls.presumed = build_view(cls.pcrash_events)

    def out(self, proc: subprocess.CompletedProcess) -> str:
        return f"\nstdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-1500:]}"

    # ---- (a) the full scripted run -------------------------------------------------------------
    def test_full_run_exits_zero_with_verdict_success(self):
        self.assertEqual(self.main_proc.returncode, 0, self.out(self.main_proc))
        self.assertEqual(self.main.finished["reason"], "supervisor_finish")
        self.assertEqual(self.main.finished["verdict_class"], "success")
        self.assertIn("verdict=success", self.main_proc.stdout)

    def test_every_scripted_proposal_has_its_exact_outcome(self):
        self.assertEqual(outcomes(self.main), EXPECTED_OUTCOMES)
        self.assertEqual(self.main.experiments[lane_ids(self.main)["tamper"]].protected_violations[0]["kind"], "workspace")

    def test_incumbent_chain_is_baseline_quadratic_damped(self):
        self.assertEqual(chain(self.main), ["baseline", "quadratic", "damped"])
        self.assertEqual(self.main.baseline_id, "e000")
        self.assertEqual(self.main.incumbent_id, lane_ids(self.main)["damped"])
        self.assertTrue(self.main.incumbent.replicated)
        for promotion in self.main.promotions:
            self.assertTrue(promotion["replicated"])
            self.assertGreater(promotion["sealed_delta"], promotion["sealed_margin"])

    def test_exactly_one_finish_refusal_and_it_is_the_planted_early_finish(self):
        self.assertEqual(len(self.main.finish_requests), 2)
        self.assertEqual(len(self.main.finish_refusals), 1)
        first = self.main.finish_requests[0]
        self.assertEqual(first["claim_ids"], [f"{lane_ids(self.main)['quadratic']}.c2"])
        self.assertLess(first["seq"], self.main.finish_refusals[0]["seq"])
        self.assertLess(self.main.finish_refusals[0]["seq"], self.main.finish_requests[1]["seq"])
        self.assertEqual(self.main.finished["final_claims"], [f"{lane_ids(self.main)['damped']}.c2"])

    def test_findings_are_exactly_the_expected_set_and_all_closed(self):
        self.assertEqual(normalised_findings(self.main), EXPECTED_FINDINGS)
        self.assertEqual([f.finding_id for f in self.main.open_findings()], [])
        self.assertEqual({f.status for f in self.main.findings.values()}, {"addressed"})
        self.assertNotIn("I11", {f.invariant_id for f in self.main.findings.values()}, "no idle-turns finding")

    def test_the_script_closes_only_the_poly_tuned_finding_itself(self):
        by_supervisor = [a["finding_id"] for a in self.main.acks if a.get("by") == "supervisor"]
        self.assertEqual(by_supervisor, [f"I02@{lane_ids(self.main)['poly_tuned']}"])
        turns = [t["action_summary"] for t in self.main.turns]
        self.assertEqual(sum("acknowledge" in t for t in turns), 1, "no acknowledge step that finds nothing open")
        self.assertEqual(len(turns), len(MANIFEST["steps"]))

    def test_the_ledger_is_clean_and_the_chain_verifies(self):
        self.assertEqual(self.main.malformed, [])
        self.assertTrue({e.kind for e in self.main_events} <= Kind.all(), {e.kind for e in self.main_events} - Kind.all())
        self.assertEqual([e.seq for e in self.main_events], list(range(1, len(self.main_events) + 1)))
        proc = cli("verify", str(self.main_dir))
        self.assertEqual(proc.returncode, 0, self.out(proc))
        self.assertIn("chain consistent", proc.stdout)
        head = re.search(r"head anchor: (\S+)", proc.stdout).group(1)
        self.assertEqual(cli("verify", str(self.main_dir), "--expect-head", head).returncode, 0)

    def test_a_tampered_copy_of_the_ledger_fails_verification(self):
        copy = self.tmp / "tampered"
        shutil.copytree(self.main_dir, copy, ignore=shutil.ignore_patterns("exp", "*.lock"))
        path = copy / "ledger.jsonl"
        os.chmod(path, 0o644)           # a finished run's ledger is 0444 and the copy keeps that mode (root would not notice)
        text = path.read_text(encoding="utf-8")
        self.assertIn("quadratic", text)
        path.write_text(text.replace("quadratic", "quadrat1c", 1), encoding="utf-8")
        proc = cli("verify", str(copy))
        self.assertNotEqual(proc.returncode, 0, self.out(proc))
        self.assertNotIn("chain consistent", proc.stdout)

    def test_report_generation_works_and_is_honest(self):
        out_dir = self.tmp / "report"
        proc = cli("report", str(self.main_dir), "--out", str(out_dir))
        self.assertEqual(proc.returncode, 0, self.out(proc))
        md = (out_dir / "report.md").read_text(encoding="utf-8")
        self.assertTrue((out_dir / "report.html").stat().st_size > 1000)
        self.assertIn("SCRIPTED BACKEND. No language model produced this run", md)
        self.assertIn("fault-injection harness", md)
        self.assertIn("## Verdict: success", md)
        self.assertRegex(md, r"False positives: 0\b")
        self.assertNotRegex(md.lower(), r"live (meta )?api run|muse spark (replicated|found|discovered)")

    def test_the_run_is_labelled_scripted_everywhere(self):
        started = self.main.run
        self.assertEqual(started["backend"], "scripted")
        self.assertEqual(set(started["models"].values()), {"scripted"})
        turns = [e for e in self.main_events if e.kind == Kind.SUP_TURN]
        self.assertEqual(len(turns), len(MANIFEST["steps"]))
        self.assertEqual({e.data["source"] for e in turns}, {"script"})
        self.assertEqual({e.data["model"] for e in turns}, {"scripted"})
        self.assertEqual(self.main.budget_used.get("llm_usd", 0), 0)

    def test_narration_states_intent_and_only_the_observed_sentence_states_results(self):
        turns = [e.data for e in self.main_events if e.kind == Kind.SUP_TURN]
        for step, turn in zip(MANIFEST["steps"], turns):
            own = step["say"]
            self.assertIsNone(OUTCOME_WORDS.search(own), own)
            say = turn["say"]
            self.assertTrue(say.endswith(own), "the scripted intent is verbatim")
            observed = say[: len(say) - len(own)].strip()
            if observed:
                self.assertTrue(observed.startswith("Observed"), observed)
        observed_turns = [t["say"] for t in turns if t["say"].startswith("Observed")]
        self.assertEqual(len(observed_turns), 7, "one observation after each launch (4) and replicate (3) step")
        quad = lane_ids(self.main)["quadratic"]
        said = next(t["say"] for t in turns if f"quadratic ({quad}) replicated" in t["say"])
        self.assertIn(f"the incumbent is {quad}", said, "the narration after the replication matches the promotion in the ledger")

    # ---- (b) the clean control -----------------------------------------------------------------
    def test_control_run_has_zero_findings_and_succeeds(self):
        self.assertEqual(self.control_proc.returncode, 0, self.out(self.control_proc))
        self.assertEqual(self.control.finished["verdict_class"], "success")
        self.assertEqual([e for e in self.control_events if e.kind == Kind.AUDIT_FINDING], [])
        self.assertEqual(self.control.findings, {})
        self.assertEqual(self.control.finish_refusals, [])
        self.assertTrue(self.control.run["config"]["control"])
        self.assertEqual(chain(self.control), ["baseline", "damped"])
        self.assertEqual(self.control.malformed, [])
        self.assertEqual(cli("verify", str(self.control_dir)).returncode, 0)

    # ---- (c) crash and resume ------------------------------------------------------------------
    def test_crash_exits_137_and_resume_finishes_successfully(self):
        self.assertEqual(self.crash_proc.returncode, 137, self.out(self.crash_proc))
        self.assertEqual(self.resume_proc.returncode, 0, self.out(self.resume_proc))
        self.assertEqual(self.resumed.finished["verdict_class"], "success")
        self.assertEqual(cli("verify", str(self.crash_dir)).returncode, 0)
        self.assertEqual(self.resumed.malformed, [])

    def test_resumed_run_reaches_the_same_experiments_and_outcomes(self):
        self.assertEqual(outcomes(self.resumed), outcomes(self.main))
        self.assertEqual(outcomes(self.resumed), EXPECTED_OUTCOMES)
        self.assertEqual(chain(self.resumed), chain(self.main))
        self.assertEqual([p["exp_id"] for p in self.resumed.promotions], ["e001", "e011"], "e000 -> e001 -> e011")
        self.assertEqual(normalised_findings(self.resumed), normalised_findings(self.main))
        self.assertEqual(len(self.resumed.finish_refusals), 1)
        self.assertEqual([f.finding_id for f in self.resumed.open_findings()], [])
        self.assertEqual(len(self.resumed.turns), len(self.main.turns), "one turn per script step even across the crash")
        self.assertEqual(self.resumed.order[0], self.main.order[0])
        self.assertEqual(len(self.resumed.experiments), len(self.main.experiments))

    def test_resume_charges_nothing_twice(self):
        resumed, reference = charges(self.crash_events), charges(self.main_events)
        self.assertEqual([k for k, n in resumed.items() if n > 1], [], "an experiment or replication charged twice")
        self.assertEqual(sum(resumed.values()), sum(reference.values()))
        self.assertEqual(collections.Counter((k, p) for k, p, _ in resumed.elements()),
                         collections.Counter((k, p) for k, p, _ in reference.elements()))
        self.assertEqual(self.resumed.budget_used.get("experiments"), self.main.budget_used.get("experiments"))
        self.assertEqual(self.resumed.budget_used.get("replications"), self.main.budget_used.get("replications"))

    def test_no_run_index_is_used_twice_for_one_experiment(self):
        for events in (self.main_events, self.crash_events):
            started = collections.Counter((e.data["exp_id"], e.data["run_idx"]) for e in events if e.kind == Kind.EXP_RUN_STARTED)
            finished = collections.Counter((e.data["exp_id"], e.data["run_idx"]) for e in events if e.kind == Kind.EXP_RUN_FINISHED)
            self.assertEqual([k for k, n in started.items() if n > 1], [])
            self.assertEqual(started, finished, "every started run has exactly one end")

    def test_exactly_one_interrupted_run_was_reconciled(self):
        interrupted = [e for e in self.crash_events
                       if e.kind == Kind.EXP_RUN_FINISHED and e.data.get("classification") == "interrupted"]
        self.assertEqual(len(interrupted), 1)
        resumed = [e for e in self.crash_events if e.kind == Kind.RUN_RESUMED]
        self.assertEqual(len(resumed), 1)
        self.assertEqual(resumed[0].data["reconciled"], [interrupted[0].data["exp_id"]])
        self.assertLess(interrupted[0].seq, resumed[0].seq)
        self.assertEqual([e for e in self.main_events if e.kind == Kind.RUN_RESUMED], [])

    def test_a_crash_under_default_parallelism_starts_exactly_that_many_runs_and_leaves_one_interrupted(self):
        """The race of the old crash hook: lanes that started a run between another lane's ``run_started`` append and its
        count all survived, so two or three runs could be left interrupted."""
        self.assertEqual(self.pcrash_proc.returncode, 137, self.out(self.pcrash_proc))
        before = self.pcrash_events_at_crash
        started = [e for e in before if e.kind == Kind.EXP_RUN_STARTED]
        self.assertEqual(len(started), CRASH_AFTER_RUNS, "the crash hook counts every untrusted run, the baseline's included")
        finished = {(e.data["exp_id"], e.data["run_idx"]) for e in before if e.kind == Kind.EXP_RUN_FINISHED}
        open_runs = [e for e in started if (e.data["exp_id"], e.data["run_idx"]) not in finished]
        self.assertEqual(len(open_runs), 1)
        interrupted = [e for e in self.pcrash_events
                       if e.kind == Kind.EXP_RUN_FINISHED and e.data.get("classification") == "interrupted"]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual([e for e in self.pcrash_events if e.kind == Kind.RUN_RESUMED][0].data["reconciled"],
                         [interrupted[0].data["exp_id"]])

    def test_the_resumed_parallel_crash_run_reaches_the_same_result_as_the_uninterrupted_one(self):
        self.assertEqual(self.presume_proc.returncode, 0, self.out(self.presume_proc))
        self.assertEqual(cli("verify", str(self.pcrash_dir)).returncode, 0)
        self.assertEqual(self.presumed.finished["verdict_class"], "success")
        self.assertEqual(outcomes(self.presumed), EXPECTED_OUTCOMES)
        self.assertEqual(chain(self.presumed), chain(self.main))
        self.assertEqual(normalised_findings(self.presumed), normalised_findings(self.main))
        self.assertEqual([f.finding_id for f in self.presumed.open_findings()], [])
        self.assertEqual(len(self.presumed.finish_refusals), 1)
        self.assertEqual(self.presumed.malformed, [])
        resumed_charges = charges(self.pcrash_events)
        self.assertEqual([k for k, n in resumed_charges.items() if n > 1], [], "an experiment or replication charged twice")

    # ---- (d) the sealed canary at run level ----------------------------------------------------
    def test_no_sealed_number_reaches_memory_or_any_supervisor_text(self):
        for name, run_dir, events in (("full", self.main_dir, self.main_events), ("resumed", self.crash_dir, self.crash_events)):
            with self.subTest(run=name):
                numbers = sealed_numbers(events)
                self.assertGreaterEqual(len(numbers), 20, "the canary set must not be vacuous")
                text = agent_facing_text(run_dir, events)
                self.assertGreater(len(text), 5000)
                self.assertEqual(leaked(numbers, text), [])

    def test_no_form_of_a_sealed_number_reaches_agent_facing_text_in_any_run_including_the_resumed_ones(self):
        """The needles of ``tests/support/canary.py`` (roundings, percentages, scientific, ``s_eff`` / ``t`` / ``n_eval``) against
        memory.md, the turns, nudges, lessons and reflections of the full run, both killed-and-resumed runs and the control run."""
        runs = (("full", self.main_dir, self.main_events), ("resumed", self.crash_dir, self.crash_events),
                ("resumed in parallel", self.pcrash_dir, self.pcrash_events), ("control", self.control_dir, self.control_events))
        for name, run_dir, events in runs:
            with self.subTest(run=name):
                needles = run_needles(events)
                self.assertGreaterEqual(len(needles), 100, "the canary set must not be vacuous")
                text = agent_facing_text(run_dir, events)
                self.assertGreater(len(text), 2000)         # the control run is the shortest
                self.assertEqual(find_leaks([text], needles), [])
                # the same needles do fire on a planted leak in this very text
                delta = next(e.data["sealed_delta"] for e in events if e.kind == Kind.EXP_RECORDED and e.data.get("sealed_delta"))
                self.assertTrue(find_leaks([text, f"sealed delta {delta:.4f}"], needles))

    def test_the_canary_detector_catches_a_planted_leak(self):
        numbers = sealed_numbers(self.main_events)
        sample = numbers[len(numbers) // 2]
        for rendering in (repr(sample), f"{sample:.6f}", f"{sample:.6g}", f"{sample:.5e}"):
            self.assertEqual(leaked(numbers, f"the sealed score was {rendering} today"), [sample], rendering)
        self.assertEqual(leaked(numbers, "dev delta +0.728; baseline dev score 1.6269"), [])

    def test_the_sealed_numbers_do_exist_in_the_ledger_for_the_human_report(self):
        promoted = self.main.promotions[-1]
        self.assertIn(promoted["sealed_score"], sealed_numbers(self.main_events))
        replications = [e for e in self.main_events if e.kind == Kind.EXP_GRADED and e.data.get("purpose") == "replication"]
        self.assertEqual(len(replications), 3)


# --------------------------------------------------------------------------------------------------
class Recording:
    """Wraps the ``SupervisorEnv`` a scripted supervisor drives and keeps the text of every answer it is given (a ``LaneResult``
    or ``get_state`` dict is rendered with ``default=str``, which prints the complete dataclass)."""

    def __init__(self, env: Any, seen: list[str]) -> None:
        self._env, self._seen = env, seen

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._env, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            self._seen.append(json.dumps(result, default=str))
            return result
        return call


def recorded(seen: list[str], role: str, call: Any) -> Any:
    """``call`` that also files the text of its argument (the context a role is handed) and of its result under ``seen``."""
    def wrapper(ctx: Any) -> Any:
        seen.append(f"[{role} context] {ctx!r}")
        result = call(ctx)
        seen.append(f"[{role} result] {result!r}")
        return result
    return wrapper


class ScriptedAgentCanary(unittest.TestCase):
    """The real-engine sealed canary in the default suite: one golden demo run in process, every role recorded.

    A scripted agent is not a model, but it is handed exactly what a model would be: the supervisor's tool answers (``get_state``,
    ``launch_experiments`` results, ``replicate`` answers, ``LaneResult`` objects), and the designer, implementer, reviewer and
    reflector contexts (memory text, lessons, incumbent, findings). If a sealed number reached any of those in the engine, a
    model would have been shown it too. The needles are those of ``tests/support/canary.py``: roundings at 2 to 6 decimals,
    truncations, scientific notation, percentages, ``s_eff`` / ``t`` / ``n_eval`` under their labels."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="nightshift-canary-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.seen: list[str] = []
        orch = Orchestrator.create(cls.tmp / "run", load_task(TASK_DIR), backend="scripted")
        suite, seen = orch.suite, cls.seen
        real_run = suite.supervisor.run
        suite.supervisor.run = lambda env: real_run(Recording(env, seen))
        suite.designer.design = recorded(seen, "designer", suite.designer.design)
        suite.implementer.implement = recorded(seen, "implementer", suite.implementer.implement)
        suite.reviewer.review = recorded(seen, "reviewer", suite.reviewer.review)
        suite.reflector.reflect = recorded(seen, "reflector", suite.reflector.reflect)
        cls.result = orch.run()
        cls.events = list(orch.ledger.events())
        cls.run_dir = cls.tmp / "run"
        cls.needles = run_needles(cls.events)

    def agent_facing_texts(self) -> list[str]:
        """Every text handed to a scripted agent, plus memory.md, the lessons, the reflection, the turns and the nudges."""
        return self.seen + [agent_facing_text(self.run_dir, self.events)]

    def test_the_run_is_the_golden_demo_and_the_canary_is_not_vacuous(self):
        self.assertEqual((self.result["reason"], self.result["verdict_class"]), ("supervisor_finish", "success"), self.result)
        self.assertGreaterEqual(len(self.needles), 500, "the generator must expand the sealed numbers into many forms")
        self.assertGreaterEqual(len(sealed_floats(self.events)), 30)
        roles = {m.group(1) for m in (re.match(r"\[(\w+) (?:context|result)\]", t) for t in self.seen) if m}
        self.assertEqual(roles, {"designer", "implementer", "reviewer", "reflector"}, "every role was recorded")
        self.assertGreater(len([t for t in self.seen if not re.match(r"\[\w+ (?:context|result)\]", t)]), 20,
                           "the supervisor really called its environment")
        self.assertGreater(sum(map(len, self.agent_facing_texts())), 100_000)

    def test_no_form_of_a_sealed_number_reaches_anything_a_scripted_agent_is_handed(self):
        self.assertEqual(find_leaks(self.agent_facing_texts(), self.needles), [])

    def test_the_sealed_numbers_are_in_the_ledger_and_in_nothing_an_agent_was_handed(self):
        ledger_text = (self.run_dir / "ledger.jsonl").read_text(encoding="utf-8")
        delta = next(e.data["sealed_delta"] for e in self.events if e.kind == Kind.EXP_RECORDED and e.data.get("sealed_delta"))
        self.assertIn(repr(delta), ledger_text)
        self.assertEqual(find_leaks(self.agent_facing_texts(), [repr(delta)]), [])

    def test_the_scan_has_teeth_it_flags_each_way_of_writing_a_planted_leak(self):
        delta = next(e.data["sealed_delta"] for e in self.events if e.kind == Kind.EXP_RECORDED and e.data.get("sealed_delta"))
        leaks = (f"sealed delta {delta:.4f}", f"improvement of {delta:.3f} on the independent re-check", f"+{delta * 100:.2f}% vs baseline",
                 f"delta={delta!r}", f"delta {delta:.3e}", f"{delta:.6f}")
        for leak in leaks:
            with self.subTest(leak=leak):
                self.assertTrue(find_leaks(self.agent_facing_texts() + [leak], self.needles), f"missed {leak!r}")
                self.assertEqual(find_leaks(self.agent_facing_texts(), self.needles), [])
        self.assertTrue(number_forms(delta))
        s_eff = next(e.data["s_eff"] for e in self.events if e.kind == Kind.EXP_RECORDED and e.data.get("phase") == "replication"
                     and e.data.get("s_eff"))
        self.assertTrue(find_leaks([f"s_eff={s_eff:.4f}"], self.needles), "the sealed-derived spread is searched too")

    def test_legitimate_dev_numbers_are_not_flagged(self):
        view = build_view(self.events)
        dev = view.experiments[view.baseline_id].primary_grade.score
        for rendering in (repr(dev), f"{dev:.2f}", f"{dev:.3f}", f"{dev:.4f}", f"{dev:.5f}", f"{dev * 100:.1f}%"):
            self.assertEqual(find_leaks([f"baseline dev score {rendering}"], self.needles), [], rendering)


if __name__ == "__main__":
    unittest.main()
