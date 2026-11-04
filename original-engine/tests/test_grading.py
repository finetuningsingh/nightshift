import dataclasses
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from nightshift import grading, sandbox, secrets
from nightshift.grading import (EvalHooks, GraderError, T95, T975, decide_outcome, evaluate, run_trusted_grader)
from nightshift.task import conditions_hash, load_task, trusted_hash
from nightshift.types import GradeResult, OutcomeDecision, ProtectedPathViolation, Purpose, SimulatedCrash, TaskSpec

FIXTURE = Path(__file__).parent / "fixtures" / "mini_task"
LINUX = sys.platform.startswith("linux")

BINNED = textwrap.dedent("""
    def fit(xs, ys):
        bins = 10
        sums, counts = [0.0] * bins, [0] * bins
        for x, y in zip(xs, ys):
            b = min(int(x * bins), bins - 1)
            sums[b] += y
            counts[b] += 1
        overall = sum(ys) / len(ys)
        means = [sums[i] / counts[i] if counts[i] else overall for i in range(bins)]
        return lambda x: means[min(max(int(x * bins), 0), bins - 1)]
""")
MEMORISER = textwrap.dedent("""
    def fit(xs, ys):
        pairs = list(zip(xs, ys))
        return lambda x: min(pairs, key=lambda p: abs(p[0] - x))[1]
""")


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            raw = fh.read()
        return raw[raw.rindex(b")") + 2:].split()[0] != b"Z"
    except FileNotFoundError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class EvalCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(secrets.clear)
        shutil.copytree(FIXTURE, self.dir / "task", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.task = load_task(self.dir / "task")
        self.run_dir = self.dir / "run"

    def workspace(self, model_src: str | None = None, exp_id: str | None = None, task: TaskSpec | None = None):
        task = task or self.task
        self._n = getattr(self, "_n", 0) + 1
        ws = sandbox.create_workspace(task, self.run_dir, exp_id or f"e{self._n:03d}")
        if model_src is not None:
            (ws.path / "model.py").write_text(textwrap.dedent(model_src))
        return ws, sandbox.tree_hash(ws.path, task)

    def evaluate(self, model_src: str | None = None, *, seeds=None, purpose=Purpose.PRIMARY, task: TaskSpec | None = None,
                 hooks=None, isolation="none", **kw) -> grading.EvalResult:
        task = task or self.task
        ws, tree = self.workspace(model_src, task=task)
        return evaluate(task, ws, seeds if seeds is not None else task.seeds, purpose=purpose, run_dir=self.run_dir,
                        expected_tree_hash=tree, hooks=hooks, isolation=isolation, **kw)


class EvaluateHappyPathTests(EvalCase):
    def test_baseline_on_the_dev_split(self):
        res = self.evaluate(purpose=Purpose.BASELINE)
        g = res.grade
        self.assertTrue(g.valid)
        self.assertIsNone(res.failure_class)
        self.assertEqual((g.split, g.seeds, g.n_eval), ("dev", [0, 1, 2], 70))
        self.assertEqual(len(g.per_seed), 3)
        self.assertEqual(len(set(g.per_seed)), 3, "seeds must change the score (training sample and evaluated rows)")
        self.assertAlmostEqual(g.score, statistics.mean(g.per_seed))
        self.assertAlmostEqual(g.std, statistics.stdev(g.per_seed))
        self.assertGreater(g.std, 0)
        self.assertEqual(set(g.metrics), {"heldout_rmse", "train_rmse"})
        self.assertAlmostEqual(g.metrics["heldout_rmse"], g.score)
        self.assertEqual(len(g.per_seed_metrics), 3)
        self.assertEqual([m["heldout_rmse"] for m in g.per_seed_metrics], g.per_seed)
        self.assertEqual(g.grader_hash, trusted_hash(self.task))
        self.assertEqual(res.conditions_hash, conditions_hash(self.task, [0, 1, 2], "dev"))
        self.assertEqual([r.classification for r in res.runs], ["ok"] * 3)
        self.assertEqual(len(res.outputs_hashes), 3)
        self.assertEqual(len(set(res.outputs_hashes)), 3)
        self.assertTrue(all(len(h) == 12 for h in res.outputs_hashes))
        self.assertIsNone(g.error)
        json.dumps(g.to_dict())

    def test_run_artifacts_land_where_the_layout_says(self):
        self.evaluate()
        exp = self.run_dir / "exp" / "e001"
        for idx in range(3):
            self.assertTrue((exp / "runs" / str(idx) / "ws" / "run.py").exists())
            self.assertTrue((exp / "runs" / str(idx) / "staged" / "outputs" / "predictions.csv").exists())
            self.assertIn("trained on 160 rows", (exp / "logs" / f"run_{idx}.stdout").read_text())
        self.assertEqual(sorted(p.name for p in (exp / "runs" / "0" / "staged").iterdir()), ["outputs"])

    def test_the_live_workspace_is_never_run_in(self):
        ws, tree = self.workspace()
        evaluate(self.task, ws, [0], purpose=Purpose.PRIMARY, run_dir=self.run_dir, expected_tree_hash=tree, isolation="none")
        self.assertFalse((ws.path / "outputs").exists())
        self.assertEqual(sandbox.tree_hash(ws.path, self.task), tree)

    def test_sealed_purposes_use_the_sealed_split(self):
        dev = self.evaluate(purpose=Purpose.BASELINE)
        sealed = self.evaluate(seeds=self.task.replication_seeds, purpose=Purpose.REPLICATION_BASELINE)
        self.assertTrue(sealed.grade.valid)
        self.assertEqual((sealed.grade.split, sealed.grade.seeds, sealed.grade.n_eval),
                         ("sealed", [100, 101, 102, 103, 104], 70))
        self.assertEqual(len(sealed.grade.per_seed), 5)
        self.assertEqual(sealed.conditions_hash, conditions_hash(self.task, self.task.replication_seeds, "sealed"))
        self.assertNotEqual(sealed.conditions_hash, dev.conditions_hash)
        self.assertEqual(self.evaluate(seeds=[100], purpose=Purpose.REPLICATION).grade.split, "sealed")

    def test_sealed_purpose_on_a_task_without_a_sealed_split_is_refused(self):
        task = dataclasses.replace(self.task, sealed=False)
        ws, tree = self.workspace(task=task)
        with self.assertRaises(ValueError):
            evaluate(task, ws, [100], purpose=Purpose.REPLICATION, run_dir=self.run_dir, expected_tree_hash=tree)

    def test_needs_seeds(self):
        ws, tree = self.workspace()
        with self.assertRaises(ValueError):
            evaluate(self.task, ws, [], purpose=Purpose.PRIMARY, run_dir=self.run_dir, expected_tree_hash=tree)

    def test_same_workspace_same_numbers(self):
        a = self.evaluate(seeds=[0, 1]).grade
        ws, tree = self.workspace(exp_id="e002")
        b = evaluate(self.task, ws, [0, 1], purpose=Purpose.PRIMARY, run_dir=self.run_dir, expected_tree_hash=tree,
                     isolation="none").grade
        self.assertEqual(a.per_seed, b.per_seed)

    def test_a_better_model_is_judged_improved_and_numbers_come_from_really_running_it(self):
        base = self.evaluate(purpose=Purpose.BASELINE).grade
        ws, tree = self.workspace(BINNED, "e002")
        cand = evaluate(self.task, ws, self.task.seeds, purpose=Purpose.PRIMARY, run_dir=self.run_dir,
                        expected_tree_hash=tree, isolation="none").grade
        self.assertLess(cand.score, base.score * 0.8)
        d = decide_outcome(self.task, cand, base, sigma_prior=base.std)
        self.assertEqual(d.outcome, "improved")
        self.assertGreater(d.delta, d.margin)

    def test_a_memoriser_overfits_the_train_metric(self):
        base = self.evaluate(purpose=Purpose.BASELINE).grade
        ws, tree = self.workspace(MEMORISER, "e002")
        cand = evaluate(self.task, ws, self.task.seeds, purpose=Purpose.PRIMARY, run_dir=self.run_dir,
                        expected_tree_hash=tree, isolation="none").grade
        self.assertAlmostEqual(cand.metrics["train_rmse"], 0.0, places=6)
        d = decide_outcome(self.task, cand, base, sigma_prior=base.std)
        self.assertGreater(d.train_delta, d.train_margin)

    def test_trusted_directory_stays_clean(self):
        before = trusted_hash(self.task)
        self.evaluate()
        self.assertEqual(trusted_hash(self.task), before)
        self.assertEqual(list((self.dir / "task" / "trusted").rglob("__pycache__")), [])

    def test_secrets_never_reach_the_run(self):
        src = """
            import json, os
            os.makedirs("outputs", exist_ok=True)
            open("outputs/env.json", "w").write(json.dumps(dict(os.environ)))
            def fit(xs, ys):
                return lambda x: 0.0
        """
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "sk-eval-secret-1234", "META_API_KEY": "meta-eval-secret-5678",
                                          "HOME": "/parent/home"}):
            self.evaluate(src, seeds=[0])
        env = (self.run_dir / "exp" / "e001" / "runs" / "0" / "staged" / "outputs" / "env.json").read_text()
        for secret in ("sk-eval-secret", "meta-eval-secret", "/parent/home"):
            self.assertNotIn(secret, env)
        self.assertNotIn("MODEL_API_KEY", env)


class ParallelTests(EvalCase):
    def test_lanes_evaluated_in_threads_do_not_disturb_each_other(self):
        from concurrent.futures import ThreadPoolExecutor
        slow = "import time\ndef fit(xs, ys):\n    time.sleep(0.2)\n    m = sum(ys) / len(ys)\n    return lambda x: m\n"
        jobs = [self.workspace(slow) for _ in range(4)]

        def go(job):
            ws, tree = job
            return evaluate(self.task, ws, [0, 1], purpose=Purpose.PRIMARY, run_dir=self.run_dir, expected_tree_hash=tree,
                            isolation="auto")

        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(go, jobs))
        self.assertTrue(all(r.grade.valid for r in results), [r.grade.error for r in results])
        self.assertEqual(len({tuple(r.grade.per_seed) for r in results}), 1, "identical workspaces give identical numbers")


RESTORE_SNIPPET = """
import os
LABELS = {labels!r}
st = os.stat(LABELS)
orig = open(LABELS, "rb").read()
with open(LABELS, "r+b") as fh:
    fh.write(b"X" * 8)
with open(LABELS, "r+b") as fh:
    fh.write(orig[:8])
os.utime(LABELS, ns=(st.st_atime_ns, st.st_mtime_ns))
"""


class TrustedStatSnapshotTests(EvalCase):
    """A change to trusted/ that is undone before the untrusted process exits leaves the content hash untouched; only the
    stat snapshot (ctime cannot be put back) sees it. These run with isolation "none", the weakest mode."""

    @property
    def labels(self) -> Path:
        return self.task.trusted_dir / "labels.csv"

    def restoring_model(self, extra: str = "") -> str:
        body = RESTORE_SNIPPET.format(labels=str(self.labels)) + extra
        return body + "def fit(xs, ys):\n    m = sum(ys) / len(ys)\n    return lambda x: m\n"

    def test_write_then_restore_is_a_trusted_violation_although_the_content_hash_is_unchanged(self):
        before, stat_before = trusted_hash(self.task), sandbox.trusted_stat(self.task.trusted_dir)
        log = []
        hooks = EvalHooks(on_run_end=lambda i, s, r: log.append(r))
        with self.assertRaises(ProtectedPathViolation) as cm:
            self.evaluate(self.restoring_model(), seeds=[0], hooks=hooks)
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("trusted", ["trusted/labels.csv"]))
        self.assertEqual(trusted_hash(self.task), before, "this is the case the content hash cannot see")
        self.assertEqual(len(log), 1, "the finished run is still reported")
        self.assertEqual(log[0].trusted_changed, ["trusted/labels.csv"])
        self.assertIn("trusted_tree_changed", log[0].note)
        self.assertNotEqual(sandbox.trusted_stat(self.task.trusted_dir), stat_before)

    def test_the_same_restore_against_the_old_checks_alone_passes(self):
        """Documents what the finding was: with only the manifest comparison nothing is noticed."""
        before = grading.trusted_manifest(self.task)
        ws, tree = self.workspace(self.restoring_model())
        res = sandbox.run_untrusted(self.task, ws.path, 0, isolation="none")
        self.assertEqual(res.classification, "ok")
        self.assertEqual(grading.trusted_manifest(self.task), before)
        self.assertEqual(res.trusted_changed, ["trusted/labels.csv"])

    def test_an_honest_evaluation_leaves_trusted_exactly_as_it_found_it(self):
        grader = self.task.trusted_dir / "grade.py"
        grader.write_text(grader.read_text().replace("import argparse\n", "import argparse\nimport helper_mod\n", 1))
        (self.task.trusted_dir / "helper_mod.py").write_text("VALUE = 1\n")
        before = sandbox.trusted_stat(self.task.trusted_dir)
        res = self.evaluate(seeds=[0, 1, 2], purpose=Purpose.PRIMARY)
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual(sandbox.trusted_stat_diff(before, sandbox.trusted_stat(self.task.trusted_dir)), [])
        self.assertFalse((self.task.trusted_dir / "__pycache__").exists(), "the grader must not leave bytecode behind")
        self.assertEqual(sorted(p.name for p in self.task.trusted_dir.iterdir()),
                         sorted(before["."][8]))

    def test_the_grader_environment_forbids_bytecode(self):
        self.assertEqual(sandbox.trusted_env(self.task, "/h")["PYTHONDONTWRITEBYTECODE"], "1")

    def test_reading_the_labels_is_not_a_violation(self):
        res = self.evaluate("import os\nopen(%r, 'rb').read()\nos.listdir(%r)\n"
                            "def fit(xs, ys):\n    m = sum(ys) / len(ys)\n    return lambda x: m\n"
                            % (str(self.labels), str(self.task.trusted_dir)), seeds=[0])
        self.assertTrue(res.grade.valid, res.grade.error)

    def test_a_planted_bytecode_file_next_to_the_grader_is_a_violation_though_the_manifest_ignores_it(self):
        planted = self.task.trusted_dir / "random.pyc"
        manifest = grading.trusted_manifest(self.task)
        with self.assertRaises(ProtectedPathViolation) as cm:
            self.evaluate(f"open({str(planted)!r}, 'wb').write(b'not python')\n"
                          "def fit(xs, ys):\n    return lambda x: 0.0\n", seeds=[0])
        self.assertEqual(cm.exception.kind, "trusted")
        self.assertIn("trusted/random.pyc", cm.exception.paths)
        self.assertEqual(grading.trusted_manifest(self.task), manifest)

    def test_a_grader_that_rewrites_a_label_file_with_identical_bytes_is_caught_after_the_call(self):
        grader = self.task.trusted_dir / "grade.py"
        grader.write_text(grader.read_text().replace(
            "def main():", "def main():\n    p = os.path.join(HERE, 'labels.csv')\n    d = open(p, 'rb').read()\n"
            "    open(p, 'wb').write(d)", 1))
        manifest = grading.trusted_manifest(self.task)
        with self.assertRaises(ProtectedPathViolation) as cm:
            self.evaluate(seeds=[0])
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("trusted", ["trusted/labels.csv"]))
        self.assertEqual(grading.trusted_manifest(self.task), manifest)

    def test_a_grader_that_writes_its_own_bytecode_is_still_tolerated(self):
        # unchanged expectation of test_bytecode_written_into_trusted_is_ignored, here through the stat snapshot as well
        grader = self.task.trusted_dir / "grade.py"
        grader.write_text(grader.read_text().replace(
            "def main():", "def main():\n    os.makedirs(os.path.join(HERE, '__pycache__'), exist_ok=True)\n"
            "    open(os.path.join(HERE, '__pycache__', 'x.pyc'), 'wb').write(b'x')", 1))
        self.assertTrue(self.evaluate(seeds=[0]).grade.valid)


class StrayBytecodeEvaluateTests(EvalCase):
    """A source-less ``*.pyc`` in ``trusted/`` (importable by the grader, ignored by the content hash) stops the evaluation."""

    def plant(self, name: str = "random.pyc") -> Path:
        path = self.task.trusted_dir / name
        path.write_bytes(b"not python")
        return path

    def test_an_existing_one_stops_the_evaluation_before_any_run_starts(self):
        self.plant()
        starts = []
        with self.assertRaises(ProtectedPathViolation) as cm:
            self.evaluate(seeds=[0], hooks=EvalHooks(on_run_start=lambda i, s, pid: starts.append(i)))
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("trusted", ["trusted/random.pyc"]))
        self.assertEqual(starts, [], "nothing runs on top of a poisoned grader")

    def test_it_is_a_persistent_plant_that_poisons_the_next_evaluation_of_an_honest_lane_too(self):
        planted = self.task.trusted_dir / "random.pyc"
        with self.assertRaises(ProtectedPathViolation):
            self.evaluate(f"open({str(planted)!r}, 'wb').write(b'x')\ndef fit(xs, ys):\n    return lambda x: 0.0\n", seeds=[0])
        self.assertTrue(planted.exists())
        with self.assertRaises(ProtectedPathViolation) as cm:
            self.evaluate(seeds=[0])
        self.assertEqual(cm.exception.paths, ["trusted/random.pyc"])

    def test_bytecode_inside_pycache_is_not_a_problem(self):
        (self.task.trusted_dir / "__pycache__").mkdir()
        (self.task.trusted_dir / "__pycache__" / "grade.cpython-313.pyc").write_bytes(b"x")
        self.assertTrue(self.evaluate(seeds=[0]).grade.valid)

    def test_one_that_appears_between_the_swept_run_and_the_grader_is_caught(self):
        planted = self.task.trusted_dir / "random.pyc"

        def late_plant(*a, **k):
            raise AssertionError("the grader must not start")

        original = sandbox.run_untrusted

        def run_then_plant(*a, **k):
            res = original(*a, **k)
            planted.write_bytes(b"x")                        # a survivor of the run, after the post-run check
            return res

        with mock.patch.object(sandbox, "run_untrusted", run_then_plant), \
                mock.patch.object(grading, "run_trusted_grader", late_plant):
            with self.assertRaises(ProtectedPathViolation) as cm:
                self.evaluate(seeds=[0])
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("trusted", ["trusted/random.pyc"]))

    def test_one_that_appears_while_the_grader_runs_is_caught_although_bytecode_is_ignored_there(self):
        planted = self.task.trusted_dir / "random.pyc"
        real = grading.run_trusted_grader

        def plant_during(*a, **k):
            out = real(*a, **k)
            planted.write_bytes(b"x")
            return out

        with mock.patch.object(grading, "run_trusted_grader", plant_during):
            with self.assertRaises(ProtectedPathViolation) as cm:
                self.evaluate(seeds=[0])
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("trusted", ["trusted/random.pyc"]))


class SymlinkedTrustedDirectoryTests(EvalCase):
    """The task loader and the content hash accept a ``trusted`` that is a link to the real directory; the stat watch must too."""

    def link_trusted(self) -> Path:
        real = self.dir / "task" / "trusted_real"
        os.rename(self.dir / "task" / "trusted", real)
        os.symlink(real, self.dir / "task" / "trusted")
        self.task = load_task(self.dir / "task")
        return real

    def test_write_then_restore_through_a_linked_trusted_directory_is_detected(self):
        real = self.link_trusted()
        manifest = grading.trusted_manifest(self.task)
        self.assertIn("labels.csv", manifest, "the content hash follows the link")
        src = RESTORE_SNIPPET.format(labels=str(real / "labels.csv")) + "def fit(xs, ys):\n    m = sum(ys) / len(ys)\n    return lambda x: m\n"
        with self.assertRaises(ProtectedPathViolation) as cm:
            self.evaluate(src, seeds=[0])
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("trusted", ["trusted/labels.csv"]))
        self.assertEqual(grading.trusted_manifest(self.task), manifest, "the content hash alone cannot see this")

    def test_an_honest_evaluation_through_a_linked_trusted_directory_is_clean(self):
        self.link_trusted()
        self.assertTrue(self.evaluate(seeds=[0, 1]).grade.valid)


class ConcurrentLaneTests(EvalCase):
    """Lanes share one trusted/ directory. A difference seen by a run that overlapped another is ambiguous: ``infra``, never an
    accusation of the honest run; the retry (attempt 2) runs alone, so the hostile run is then caught and the honest one is not."""

    STARTED, DONE = "honest.started", "hostile.done"

    def sources(self) -> tuple[str, str]:
        marks = self.dir / "marks"
        marks.mkdir(exist_ok=True)
        started, done = str(marks / self.STARTED), str(marks / self.DONE)
        honest = ("import os, time\n"
                  "def fit(xs, ys):\n"
                  "    if os.environ['NIGHTSHIFT_ATTEMPT'] == '1':\n"
                  f"        open({started!r}, 'w').write('x')\n"
                  f"        end = time.time() + 30\n        while not os.path.exists({done!r}) and time.time() < end:\n"
                  "            time.sleep(0.01)\n"
                  "    m = sum(ys) / len(ys)\n    return lambda x: m\n")
        hostile = ("import os, time\n"
                   "if os.environ['NIGHTSHIFT_ATTEMPT'] == '1':\n"
                   f"    end = time.time() + 30\n    while not os.path.exists({started!r}) and time.time() < end:\n"
                   "        time.sleep(0.01)\n"
                   + RESTORE_SNIPPET.format(labels=str(self.task.trusted_dir / "labels.csv"))
                   + "if os.environ['NIGHTSHIFT_ATTEMPT'] == '1':\n"
                   f"    open({done!r}, 'w').write('x')\n"
                   "def fit(xs, ys):\n    m = sum(ys) / len(ys)\n    return lambda x: m\n")
        return honest, hostile

    def run_both(self, honest_ws, hostile_ws, attempt: int, idx: int):
        from concurrent.futures import ThreadPoolExecutor

        def go(job):
            ws, tree = job
            try:
                return evaluate(self.task, ws, [0], purpose=Purpose.PRIMARY, run_dir=self.run_dir, expected_tree_hash=tree,
                                isolation="none", attempt=attempt, run_idx_start=idx)
            except ProtectedPathViolation as exc:
                return exc

        with ThreadPoolExecutor(2) as pool:
            return list(pool.map(go, [honest_ws, hostile_ws]))

    def test_ambiguous_on_the_first_attempt_decisive_on_the_solo_retry(self):
        honest_src, hostile_src = self.sources()
        honest, hostile = self.workspace(honest_src), self.workspace(hostile_src)
        h1, x1 = self.run_both(honest, hostile, attempt=1, idx=0)
        for name, res in (("honest", h1), ("hostile", x1)):
            self.assertIsInstance(res, grading.EvalResult, (name, res))
            self.assertEqual(res.failure_class, "infra", name)
            self.assertFalse(res.grade.valid)
            run = res.runs[0]
            self.assertEqual((run.classification, run.trusted_contested, run.trusted_changed),
                             ("infra", True, ["trusted/labels.csv"]), name)
            self.assertIn("may come from a concurrent lane", run.note)
            self.assertIn("trusted_tree_changed_during_run", res.grade.error)
        # the retry: honest and hostile run (at the same time, but alone one after the other)
        h2, x2 = self.run_both(honest, hostile, attempt=2, idx=1)
        self.assertIsInstance(h2, grading.EvalResult, "an honest lane is never accused")
        self.assertTrue(h2.grade.valid, h2.grade.error)
        self.assertIsNone(h2.failure_class)
        self.assertFalse(h2.runs[0].trusted_changed)
        self.assertIsInstance(x2, ProtectedPathViolation)
        self.assertEqual((x2.kind, x2.paths), ("trusted", ["trusted/labels.csv"]))

    def test_two_honest_concurrent_runs_never_trip_the_snapshot(self):
        slow = "import time\ndef fit(xs, ys):\n    time.sleep(0.3)\n    m = sum(ys) / len(ys)\n    return lambda x: m\n"
        a, b = self.workspace(slow), self.workspace(slow)
        ra, rb = self.run_both(a, b, attempt=1, idx=0)
        self.assertTrue(ra.grade.valid and rb.grade.valid, (getattr(ra, "grade", ra), getattr(rb, "grade", rb)))
        self.assertTrue(not ra.runs[0].trusted_changed and not rb.runs[0].trusted_changed)


class AggregationOverflowTests(EvalCase):
    def test_finite_scores_whose_mean_overflows_make_an_invalid_grade_not_a_crash(self):
        raw = {"valid": True, "score": 1.7e308, "split": "dev", "n_eval": 1, "metrics": {}, "notes": ""}
        with mock.patch.object(grading, "run_trusted_grader", return_value=dict(raw)):
            res = self.evaluate(seeds=[0, 1])
        self.assertFalse(res.grade.valid)
        self.assertEqual(res.failure_class, "invalid")
        self.assertIn("overflow", res.grade.error)


class HookTests(EvalCase):
    def test_hooks_fire_in_order_with_offset_run_indices(self):
        events = []
        hooks = EvalHooks(
            on_run_start=lambda i, s, pid: events.append(("start", i, s, isinstance(pid, int))),
            on_heartbeat=lambda i, b, c: events.append(("beat", i)),
            on_run_end=lambda i, s, r: events.append(("end", i, s, r.classification)),
            on_stalled=lambda i, idle: events.append(("stalled", i)))
        slow = """
            import time
            def fit(xs, ys):
                time.sleep(0.25)
                return lambda x: 0.0
        """
        self.evaluate(slow, seeds=[4, 5, 6], hooks=hooks, run_idx_start=7, heartbeat_sec=0.05)
        kinds = [(e[0], e[1]) for e in events if e[0] != "beat"]
        self.assertEqual(kinds, [("start", 7), ("end", 7), ("start", 8), ("end", 8), ("start", 9), ("end", 9)])
        self.assertEqual([e[2] for e in events if e[0] == "start"], [4, 5, 6])
        self.assertTrue(all(e[3] for e in events if e[0] == "start"))
        self.assertTrue(all(e[3] == "ok" for e in events if e[0] == "end"))
        # heartbeats happen between the matching start and end
        positions = {(e[0], e[1]): n for n, e in enumerate(events) if e[0] != "beat"}
        for n, e in enumerate(events):
            if e[0] == "beat":
                self.assertTrue(positions[("start", e[1])] < n < positions[("end", e[1])])
        self.assertTrue(any(e[0] == "beat" for e in events))

    def test_stalled_hook_fires_once_and_the_run_is_a_timeout(self):
        task = dataclasses.replace(self.task, stall_sec=0.5)
        events = []
        hooks = EvalHooks(on_stalled=lambda i, idle: events.append(("stalled", i, idle)),
                          on_run_end=lambda i, s, r: events.append(("end", i, r.stalled, r.classification)))
        hang = """
            import time
            def fit(xs, ys):
                time.sleep(30)
        """
        res = self.evaluate(hang, seeds=[0], task=task, hooks=hooks)
        self.assertEqual(res.failure_class, "timeout")
        self.assertEqual([e[0] for e in events], ["stalled", "end"])
        self.assertEqual(events[0][1], 0)
        self.assertEqual(events[1][2:], (True, "timeout"))

    def test_an_exception_in_a_hook_propagates_and_kills_the_run(self):
        pids = []

        def start(i, s, pid):
            pids.append(pid)
            raise SimulatedCrash("boom")

        hang = """
            import time
            def fit(xs, ys):
                time.sleep(30)
        """
        with self.assertRaises(SimulatedCrash):
            self.evaluate(hang, hooks=EvalHooks(on_run_start=start))
        self.assertFalse(alive(pids[0]))


class FailureClassTests(EvalCase):
    CRASH_BODY = "raise RuntimeError('model bug')"
    HANG_BODY = "import time; time.sleep(30)"
    INFRA_BODY = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    NAN_BODY = "return lambda x: float('nan')"
    OK_BODY = "return lambda x: 0.0"

    @staticmethod
    def model(body: str) -> str:
        return f"def fit(xs, ys):\n    {body}\n"

    def by_seed(self, a: str, b: str) -> str:
        """seed 0 behaves like body ``a`` and every other seed like body ``b``."""
        return ("import os\n_s = int(os.environ['NIGHTSHIFT_SEED'])\n"
                f"def _a(xs, ys):\n    {a}\n"
                f"def _b(xs, ys):\n    {b}\n"
                "def fit(xs, ys):\n    return (_a if _s == 0 else _b)(xs, ys)\n")

    def test_crash(self):
        res = self.evaluate(self.model(self.CRASH_BODY))
        self.assertEqual(res.failure_class, "crash")
        self.assertFalse(res.grade.valid)
        self.assertIsNone(res.grade.score)
        self.assertEqual(res.grade.per_seed, [])
        self.assertEqual(res.grade.seeds, [0, 1, 2])
        self.assertEqual(len(res.runs), 3, "every seed is attempted")
        self.assertIn("seed 0: run crash", res.grade.error)
        self.assertEqual(res.outputs_hashes, ["", "", ""])
        self.assertIn("model bug", res.runs[0].stderr)

    def test_timeout(self):
        res = self.evaluate(self.model(self.HANG_BODY), seeds=[0], timeout_sec=0.5)
        self.assertEqual((res.failure_class, res.grade.valid), ("timeout", False))
        self.assertTrue(res.runs[0].timed_out)

    def test_infra_from_a_missing_required_module(self):
        task = dataclasses.replace(self.task, requires=["no_such_module_xyz"])
        res = self.evaluate("import no_such_module_xyz\ndef fit(xs, ys):\n    return lambda x: 0\n", task=task, seeds=[0])
        self.assertEqual(res.failure_class, "infra")

    def test_invalid_when_the_grader_rejects_the_output(self):
        res = self.evaluate(self.model(self.NAN_BODY), seeds=[0, 1])
        self.assertEqual((res.failure_class, res.grade.valid), ("invalid", False))
        self.assertEqual([r.classification for r in res.runs], ["ok", "ok"])
        self.assertIn("unreadable predictions", res.grade.error)       # dev notes are shown

    def test_sealed_failures_never_echo_grader_notes(self):
        res = self.evaluate(self.model(self.NAN_BODY), seeds=[100], purpose=Purpose.REPLICATION)
        self.assertEqual(res.failure_class, "invalid")
        self.assertNotIn("unreadable", res.grade.error)
        self.assertEqual(res.grade.notes, "")

    def test_precedence_timeout_over_crash(self):
        res = self.evaluate(self.by_seed(self.CRASH_BODY, self.HANG_BODY), seeds=[0, 1], timeout_sec=1.0)
        self.assertEqual(sorted(r.classification for r in res.runs), ["crash", "timeout"])
        self.assertEqual(res.failure_class, "timeout")

    def test_precedence_infra_over_timeout(self):
        res = self.evaluate(self.by_seed(self.INFRA_BODY, self.HANG_BODY), seeds=[0, 1], timeout_sec=1.0)
        self.assertEqual(sorted(r.classification for r in res.runs), ["infra", "timeout"])
        self.assertEqual(res.failure_class, "infra")

    def test_precedence_crash_over_invalid(self):
        res = self.evaluate(self.by_seed(self.CRASH_BODY, self.NAN_BODY), seeds=[0, 1])
        self.assertEqual(res.failure_class, "crash")

    def test_one_bad_seed_invalidates_the_whole_grade(self):
        res = self.evaluate(self.by_seed(self.CRASH_BODY, self.OK_BODY), seeds=[0, 1, 2])
        self.assertFalse(res.grade.valid)
        self.assertEqual(res.grade.per_seed, [])
        self.assertEqual([r.classification for r in res.runs], ["crash", "ok", "ok"])

    def test_output_flood_is_a_capped_crash(self):
        flood = ("import sys\ndef fit(xs, ys):\n    chunk = 'x' * 65536\n    for _ in range(400):\n"
                 "        sys.stdout.write(chunk)\n        sys.stdout.flush()\n")
        res = self.evaluate(flood, seeds=[0])
        self.assertEqual(res.failure_class, "crash")
        self.assertIn("output_limit", res.runs[0].note)
        self.assertEqual((self.run_dir / "exp" / "e001" / "logs" / "run_0.stdout").stat().st_size, 8 * 1024 * 1024)


class ProtectedViolationTests(EvalCase):
    def expect_violation(self, model_src: str, kind: str, **kw):
        hooks_log = []
        hooks = EvalHooks(on_run_start=lambda i, s, p: hooks_log.append("start"),
                          on_run_end=lambda i, s, r: hooks_log.append("end"))
        with self.assertRaises(ProtectedPathViolation) as cm:
            self.evaluate(model_src, seeds=[0], hooks=hooks, **kw)
        self.assertEqual(cm.exception.kind, kind)
        return cm.exception, hooks_log

    def test_tree_hash_mismatch_is_a_runtime_violation_before_anything_runs(self):
        ws, tree = self.workspace()
        (ws.path / "model.py").write_text("# edited after the hash was recorded\n" + (ws.path / "model.py").read_text())
        log = []
        with self.assertRaises(ProtectedPathViolation) as cm:
            evaluate(self.task, ws, [0], purpose=Purpose.PRIMARY, run_dir=self.run_dir, expected_tree_hash=tree,
                     isolation="none", hooks=EvalHooks(on_run_start=lambda *a: log.append(a)))
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("runtime", ["tree_hash_mismatch"]))
        self.assertEqual(log, [])

    def test_special_file_in_the_workspace_is_refused_before_running(self):
        ws, tree = self.workspace()
        os.symlink("/etc/hostname", ws.path / "link")
        with self.assertRaises(ProtectedPathViolation) as cm:
            evaluate(self.task, ws, [0], purpose=Purpose.PRIMARY, run_dir=self.run_dir, expected_tree_hash=tree, isolation="none")
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("runtime", ["special:link"]))

    def test_run_that_edits_an_editable_file_is_a_runtime_violation(self):
        exc, log = self.expect_violation(
            "def fit(xs, ys):\n    open('model.py', 'a').write('# sneaky\\n')\n    return lambda x: 0.0\n", "runtime")
        self.assertEqual(exc.paths, ["model.py"])
        self.assertEqual(log, ["start", "end"], "the finished run is still reported")

    def test_run_that_writes_a_new_non_scratch_file(self):
        exc, _ = self.expect_violation(
            "def fit(xs, ys):\n    open('extra.txt', 'w').write('x')\n    return lambda x: 0.0\n", "runtime")
        self.assertEqual(exc.paths, ["extra.txt"])

    def test_run_that_edits_or_deletes_readonly_code(self):
        exc, _ = self.expect_violation(
            "import os\ndef fit(xs, ys):\n    open('data.py', 'a').write('#\\n')\n    os.unlink('eval_inputs.csv')\n"
            "    return lambda x: 0.0\n", "runtime")
        self.assertEqual(sorted(exc.paths), ["data.py", "eval_inputs.csv"])

    def test_a_scratch_file_is_fine(self):
        res = self.evaluate("import os\ndef fit(xs, ys):\n    os.makedirs('outputs', exist_ok=True)\n"
                            "    open('outputs/notes.txt', 'w').write('x')\n    return lambda x: 0.0\n", seeds=[0])
        self.assertTrue(res.grade.valid)
        staged = self.run_dir / "exp" / "e001" / "runs" / "0" / "staged" / "outputs"
        self.assertEqual((staged / "notes.txt").read_text(), "x")

    def test_outputs_replaced_by_a_symlink_to_a_directory(self):
        elsewhere = self.dir / "elsewhere"
        elsewhere.mkdir()
        exc, _ = self.expect_violation(
            f"import os\ndef fit(xs, ys):\n    os.symlink({str(elsewhere)!r}, 'outputs')\n    return lambda x: 0.0\n", "runtime")
        self.assertEqual(exc.paths, ["special:outputs"])
        self.assertFalse((self.run_dir / "exp" / "e001" / "runs" / "0" / "staged").exists(), "nothing may be staged")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "no mkfifo")
    def test_fifo_in_outputs(self):
        exc, _ = self.expect_violation(
            "import os\ndef fit(xs, ys):\n    os.makedirs('outputs', exist_ok=True)\n    os.mkfifo('outputs/pipe')\n"
            "    return lambda x: 0.0\n", "runtime")
        self.assertEqual(exc.paths, ["special:outputs/pipe"])

    def test_hardlink_in_the_workspace(self):
        exc, _ = self.expect_violation(
            "import os\ndef fit(xs, ys):\n    os.makedirs('outputs', exist_ok=True)\n    os.link('data.py', 'outputs/d.py')\n"
            "    return lambda x: 0.0\n", "runtime")
        self.assertEqual(sorted(exc.paths), ["special:data.py", "special:outputs/d.py"])

    def test_newline_in_a_filename(self):
        exc, _ = self.expect_violation(
            "import os\ndef fit(xs, ys):\n    os.makedirs('outputs', exist_ok=True)\n    open('outputs/a\\nb', 'w').write('x')\n"
            "    return lambda x: 0.0\n", "runtime")
        self.assertEqual(exc.paths, ["special:outputs/a\\nb"])

    def test_grader_tampering_is_a_trusted_violation_and_the_grader_never_runs(self):
        grader = self.dir / "task" / "trusted" / "grade.py"
        marker = self.dir / "grader_ran.txt"
        src = f"""
            def fit(xs, ys):
                open({str(grader)!r}, 'a').write("\\nopen({str(marker)!r}, 'w').write('ran')\\n")
                return lambda x: 0.0
        """
        exc, log = self.expect_violation(src, "trusted")
        self.assertEqual(exc.paths, ["trusted/grade.py"])
        self.assertFalse(marker.exists(), "a tampered grader must not be executed")
        self.assertEqual(log, ["start", "end"])

    def test_tampering_followed_by_a_crash_or_timeout_is_still_a_trusted_violation(self):
        grader = self.dir / "task" / "trusted" / "grade.py"
        for name, tail in (("crash", "    raise SystemExit(3)\n"), ("timeout", "    import time\n    time.sleep(60)\n")):
            with self.subTest(name):
                before = grader.read_text()
                src = f"def fit(xs, ys):\n    open({str(grader)!r}, 'a').write('\\n# evil\\n')\n{tail}"
                task = dataclasses.replace(self.task, run_timeout_sec=1.5)
                with self.assertRaises(ProtectedPathViolation) as cm:
                    self.evaluate(src, seeds=[0], task=task)
                self.assertEqual((cm.exception.kind, cm.exception.paths), ("trusted", ["trusted/grade.py"]))
                grader.write_text(before)

    def test_expected_grader_hash_refuses_a_trusted_tree_that_changed_between_evaluations(self):
        ws, tree = self.workspace()
        good = trusted_hash(self.task)
        kw = dict(purpose=Purpose.PRIMARY, run_dir=self.run_dir, expected_tree_hash=tree, isolation="none")
        self.assertTrue(evaluate(self.task, ws, [0], expected_grader_hash=good, **kw).grade.valid)
        grader = self.dir / "task" / "trusted" / "grade.py"
        grader.write_text(grader.read_text() + "\n# changed while no evaluation was running\n")
        with self.assertRaises(ProtectedPathViolation) as cm:
            evaluate(self.task, ws, [0], expected_grader_hash=good, **kw)
        self.assertEqual(cm.exception.kind, "trusted")
        # without the argument the new tree is the baseline (documented default)
        self.assertTrue(evaluate(self.task, ws, [0], **kw).grade.valid)

    def test_new_or_deleted_trusted_files_are_tampering_too(self):
        trusted = self.dir / "task" / "trusted"
        exc, _ = self.expect_violation(
            f"import os\ndef fit(xs, ys):\n    open({str(trusted / 'extra.json')!r}, 'w').write('{{}}')\n"
            f"    os.unlink({str(trusted / 'train_pool.csv')!r})\n    return lambda x: 0.0\n", "trusted")
        self.assertEqual(sorted(exc.paths), ["trusted/extra.json", "trusted/train_pool.csv"])

    def test_a_grader_that_modifies_trusted_files_is_caught_after_the_call(self):
        grader = self.dir / "task" / "trusted" / "grade.py"
        real = grader.read_text()
        grader.write_text(real.replace("def main():", "def main():\n    open(os.path.join(HERE, 'cache.bin'), 'w').write('x')"))
        with self.assertRaises(ProtectedPathViolation) as cm:
            self.evaluate(seeds=[0])
        self.assertEqual((cm.exception.kind, cm.exception.paths), ("trusted", ["trusted/cache.bin"]))

    def test_bytecode_written_into_trusted_is_ignored(self):
        grader = self.dir / "task" / "trusted" / "grade.py"
        real = grader.read_text()
        grader.write_text(real.replace("def main():", "def main():\n    os.makedirs(os.path.join(HERE, '__pycache__'), exist_ok=True)\n"
                                       "    open(os.path.join(HERE, '__pycache__', 'x.cpython-313.pyc'), 'wb').write(b'x')"))
        self.assertTrue(self.evaluate(seeds=[0]).grade.valid)

    def test_a_detached_process_cannot_alter_the_staged_outputs(self):
        src = """
            import os, subprocess, sys
            _real_fit = None
            def fit(xs, ys):
                code = ("import time\\ntime.sleep(0.7)\\n"
                        "open('outputs/predictions.csv', 'w').write('id,pred\\\\nr000,999\\\\n')\\n"
                        "open('outputs/late.txt', 'w').write('late')\\n")
                d = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
                open('daemon.pid', 'w').write(str(d.pid))
                mean = sum(ys) / len(ys)
                return lambda x: mean
        """
        # the daemon pid file is a non-scratch file, so put it under outputs instead
        src = src.replace("open('daemon.pid', 'w')", "os.makedirs('outputs', exist_ok=True); open('outputs/daemon.pid', 'w')")
        res = self.evaluate(src, seeds=[0])
        self.assertTrue(res.grade.valid, res.grade.error)
        run_dir = self.run_dir / "exp" / "e001" / "runs" / "0"
        pid = int((run_dir / "staged" / "outputs" / "daemon.pid").read_text())
        self.assertFalse(alive(pid))
        time.sleep(1.0)
        for root in (run_dir / "staged" / "outputs", run_dir / "ws" / "outputs"):
            self.assertNotIn("999", (root / "predictions.csv").read_text())
            self.assertFalse((root / "late.txt").exists())

    @unittest.skipUnless(LINUX, "namespaces are Linux only")
    def test_isolation_plumbing_hides_trusted_ledger_and_siblings_from_the_run(self):
        if sandbox.available_isolation() != "userns":
            self.skipTest("user namespaces unavailable here")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "ledger.jsonl").write_text("ledger contents\n")
        (self.run_dir / "exp" / "e009").mkdir(parents=True)
        (self.run_dir / "exp" / "e009" / "workspace").write_text("sibling")
        trusted = self.task.trusted_dir
        src = f"""
            import json, os
            def fit(xs, ys):
                os.makedirs('outputs', exist_ok=True)
                seen = dict(trusted=sorted(os.listdir({str(trusted)!r})), ledger=os.path.getsize({str(self.run_dir / 'ledger.jsonl')!r}),
                            sibling=sorted(os.listdir({str(self.run_dir / 'exp' / 'e009')!r})))
                open('outputs/seen.json', 'w').write(json.dumps(seen))
                mean = sum(ys) / len(ys)
                return lambda x: mean
        """
        res = self.evaluate(src, seeds=[0], isolation="auto")
        self.assertTrue(res.grade.valid, res.grade.error)
        self.assertEqual(res.runs[0].isolation, "userns")
        seen = json.loads((self.run_dir / "exp" / "e001" / "runs" / "0" / "ws" / "outputs" / "seen.json").read_text())
        self.assertEqual(seen, {"trusted": [], "ledger": 0, "sibling": []})


# --------------------------------------------------------------------------------------------------
# the trusted grader contract
# --------------------------------------------------------------------------------------------------
class TrustedGraderTests(unittest.TestCase):
    ECHO = textwrap.dedent("""
        import os, sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.stdout.write(open(os.path.join(here, "payload.txt"), encoding="utf-8", errors="surrogateescape").read())
        sys.stdout.flush()
        code_file = os.path.join(here, "exitcode.txt")
        sys.exit(int(open(code_file).read()) if os.path.exists(code_file) else 0)
    """)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name).resolve()     # the grader reports its resolved cwd (macOS TMPDIR is a symlink)
        self.addCleanup(self.tmp.cleanup)
        shutil.copytree(FIXTURE, self.dir / "task", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (self.dir / "task" / "trusted" / "grade.py").write_text(self.ECHO)
        self.task = load_task(self.dir / "task")
        self.staged = self.dir / "staged"
        self.staged.mkdir()

    def payload(self, text: str, exit_code: int = 0) -> None:
        (self.dir / "task" / "trusted" / "payload.txt").write_text(text, encoding="utf-8", errors="surrogateescape")
        (self.dir / "task" / "trusted" / "exitcode.txt").write_text(str(exit_code))

    def grade(self, text: str, *, split: str = "dev", exit_code: int = 0, task: TaskSpec | None = None) -> dict:
        self.payload(text, exit_code)
        return run_trusted_grader(task or self.task, self.staged, 0, split)

    @staticmethod
    def line(**over) -> str:
        obj = {"valid": True, "score": 0.25, "split": "dev", "n_eval": 7, "metrics": {"heldout_rmse": 0.25, "train_rmse": 0.1},
               "notes": "fine"}
        obj.update(over)
        return json.dumps({k: v for k, v in obj.items() if v is not ...}) + "\n"

    def rejected(self, text: str, **kw) -> GraderError:
        with self.assertRaises(ValueError) as cm:
            self.grade(text, **kw)
        self.assertIsInstance(cm.exception, GraderError)
        return cm.exception

    def test_accepts_the_contract_example(self):
        got = self.grade(self.line())
        self.assertEqual(got, {"valid": True, "score": 0.25, "split": "dev", "n_eval": 7,
                               "metrics": {"heldout_rmse": 0.25, "train_rmse": 0.1}, "notes": "fine"})

    def test_integers_are_normalised_to_floats(self):
        got = self.grade(self.line(score=1, metrics={"heldout_rmse": 1}))
        self.assertIsInstance(got["score"], float)
        self.assertIsInstance(got["metrics"]["heldout_rmse"], float)

    def test_only_the_last_line_counts_and_earlier_noise_is_ignored(self):
        self.assertTrue(self.grade("warning: something\n{not json}\n\n" + self.line() + "\n\n")["valid"])
        self.rejected(self.line() + "trailing garbage\n")

    def test_echoes_the_requested_split(self):
        self.assertEqual(self.grade(self.line(split="sealed"), split="sealed")["split"], "sealed")
        self.assertIn("split", str(self.rejected(self.line(split="dev"), split="sealed")))
        self.assertIn("split", str(self.rejected(self.line(split=...))))
        self.rejected(self.line(split="sealed"), split="dev")

    def test_unicode_line_separators_inside_the_json_do_not_split_the_last_line(self):
        for sep in ("\u2028", "\u2029", "\x85", "\x1c", "\x0b", "\x0c"):
            with self.subTest(sep=repr(sep)):
                text = json.dumps({"valid": True, "score": 0.5, "split": "dev", "n_eval": 1, "metrics": {},
                                   "notes": f"a{sep}b"}, ensure_ascii=False) + "\n"
                self.assertEqual(self.grade(text)["notes"], f"a{sep}b")

    def test_exit_code_must_be_zero(self):
        exc = self.rejected(self.line(), exit_code=1)
        self.assertIn("code 1", str(exc))
        self.assertEqual(exc.kind, "invalid")

    def test_nothing_or_non_json(self):
        for text in ("", "\n\n", "hello\n", "[1, 2]\n", '"str"\n', "{'valid': true}\n", "{\"valid\": true,"):
            with self.subTest(text=text):
                self.rejected(text)

    def test_nan_and_infinity_are_rejected_wherever_they_appear(self):
        base = '{"valid": true, "score": %s, "split": "dev", "n_eval": 1, "metrics": {"m": %s}, "notes": ""}\n'
        for score, metric in (("NaN", "1.0"), ("Infinity", "1.0"), ("-Infinity", "1.0"), ("1.0", "NaN"), ("1.0", "Infinity"),
                              ("1e999", "1.0"), ("1.0", "-1e999")):
            with self.subTest(score=score, metric=metric):
                self.rejected(base % (score, metric))
        self.rejected('{"valid": false, "score": NaN, "split": "dev", "n_eval": 0, "metrics": {}, "notes": ""}\n')

    def test_valid_must_be_a_real_boolean(self):
        for bad in ('"true"', "1", "0", "null", "[]"):
            with self.subTest(valid=bad):
                self.rejected('{"valid": %s, "score": 0.1, "split": "dev", "n_eval": 1, "metrics": {}, "notes": ""}\n' % bad)
        self.rejected(self.line(valid=...))

    def test_score_must_be_a_finite_number_when_valid(self):
        for bad in ('"0.5"', "true", "null", "[0.5]", "{}"):
            with self.subTest(score=bad):
                self.rejected('{"valid": true, "score": %s, "split": "dev", "n_eval": 1, "metrics": {}, "notes": ""}\n' % bad)
        self.rejected(self.line(score=...))

    def test_valid_false_is_returned_not_raised(self):
        got = self.grade(self.line(valid=False, score=None, metrics={}, notes="bad shape"))
        self.assertEqual((got["valid"], got["score"], got["notes"]), (False, None, "bad shape"))
        self.assertEqual(self.grade(self.line(valid=False, score=3.0))["score"], 3.0)
        self.rejected(self.line(valid=False, score="x"))

    def test_metrics_must_be_an_object_of_finite_numbers(self):
        for bad in ("[]", '"x"', '{"a": "1"}', '{"a": true}', '{"a": null}', '{"a": [1]}'):
            with self.subTest(metrics=bad):
                self.rejected('{"valid": true, "score": 0.1, "split": "dev", "n_eval": 1, "metrics": %s, "notes": ""}\n' % bad)

    def test_n_eval_and_notes_types(self):
        for field, bad in (("n_eval", "-1"), ("n_eval", "1.5"), ("n_eval", "true"), ("n_eval", '"7"'), ("notes", "3"),
                           ("notes", "null"), ("notes", "[]")):
            with self.subTest(field=field, bad=bad):
                good = {"valid": True, "score": 0.1, "split": "dev", "n_eval": 1, "metrics": {}, "notes": ""}
                text = json.dumps({**good, field: None}).replace("null", bad, 1) if bad != "null" else json.dumps({**good, field: None})
                self.rejected(text + "\n")

    def test_missing_metrics_and_notes_default_sensibly(self):
        got = self.grade('{"valid": true, "score": 0.5, "split": "dev"}\n')
        self.assertEqual((got["metrics"], got["notes"], got["n_eval"]), ({}, "", 0))

    def test_stdout_is_capped_at_64_kib(self):
        exc = self.rejected("x" * (64 * 1024 + 10) + "\n" + self.line())
        self.assertIn("64 KiB", str(exc))
        self.assertTrue(self.grade("y" * 60000 + "\n" + self.line())["valid"])

    def test_non_utf8_output_is_rejected(self):
        self.payload("\udcff\udcfe\n" + self.line())
        with self.assertRaisesRegex(GraderError, "UTF-8"):
            run_trusted_grader(self.task, self.staged, 0, "dev")

    def test_unknown_split_argument(self):
        with self.assertRaises(ValueError):
            run_trusted_grader(self.task, self.staged, 0, "test")

    def test_timeout_kills_the_grader(self):
        (self.dir / "task" / "trusted" / "grade.py").write_text(
            "import os, time\nopen(os.path.join(os.path.dirname(__file__), 'pid.txt'), 'w').write(str(os.getpid()))\ntime.sleep(30)\n")
        task = dataclasses.replace(self.task, grade_timeout_sec=0.5)
        t0 = time.monotonic()
        with self.assertRaises(GraderError) as cm:
            run_trusted_grader(task, self.staged, 0, "dev")
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual(cm.exception.kind, "timeout")
        self.assertFalse(alive(int((self.dir / "task" / "trusted" / "pid.txt").read_text())))

    def test_grader_children_are_killed_with_the_group(self):
        (self.dir / "task" / "trusted" / "grade.py").write_text(
            "import os, subprocess, sys\nd = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "open(os.path.join(os.path.dirname(__file__), 'child.txt'), 'w').write(str(d.pid))\n"
            "print('{\"valid\": true, \"score\": 1.0, \"split\": \"dev\", \"n_eval\": 1, \"metrics\": {}, \"notes\": \"\"}')\n")
        run_trusted_grader(self.task, self.staged, 0, "dev")
        self.assertFalse(alive(int((self.dir / "task" / "trusted" / "child.txt").read_text())))

    def test_failed_start_is_infra(self):
        with mock.patch.object(sandbox.subprocess, "Popen", side_effect=OSError("no fork")):
            with self.assertRaises(GraderError) as cm:
                run_trusted_grader(self.task, self.staged, 0, "dev")
        self.assertEqual(cm.exception.kind, "infra")

    def test_missing_required_module_in_the_grader_is_infra_and_stderr_stays_out_of_the_message(self):
        (self.dir / "task" / "trusted" / "grade.py").write_text("import no_such_module_xyz\n")
        task = dataclasses.replace(self.task, requires=["no_such_module_xyz"])
        with self.assertRaises(GraderError) as cm:
            run_trusted_grader(task, self.staged, 0, "dev")
        self.assertEqual(cm.exception.kind, "infra")
        self.assertIn("no_such_module_xyz", cm.exception.stderr_tail)
        self.assertNotIn("no_such_module_xyz", str(cm.exception))

    def test_crashing_grader_is_invalid_not_infra(self):
        (self.dir / "task" / "trusted" / "grade.py").write_text("raise RuntimeError('grader bug SECRET-LABEL')\n")
        with self.assertRaises(GraderError) as cm:
            run_trusted_grader(self.task, self.staged, 0, "dev")
        self.assertEqual(cm.exception.kind, "invalid")
        self.assertNotIn("SECRET-LABEL", str(cm.exception))

    def test_environment_cwd_group_and_rlimits(self):
        (self.dir / "task" / "trusted" / "grade.py").write_text(textwrap.dedent("""
            import json, os, resource, sys
            info = dict(env=dict(os.environ), cwd=os.getcwd(), argv=sys.argv[1:], pgid=os.getpgrp(), pid=os.getpid(),
                        core=resource.getrlimit(resource.RLIMIT_CORE))
            print(json.dumps({"valid": True, "score": 1.0, "split": sys.argv[sys.argv.index("--split") + 1], "n_eval": 1,
                              "metrics": {}, "notes": json.dumps(info)}))
        """))
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "sk-grader-secret-1234", "META_API_KEY": "meta-grader-secret",
                                          "PYTHONPATH": "/parent/pp"}):
            got = run_trusted_grader(self.task, self.staged, 3, "sealed")
        info = json.loads(got["notes"])
        self.assertEqual(info["cwd"], str(self.dir / "task"))
        self.assertEqual(info["argv"], ["--workspace", str(self.staged), "--seed", "3", "--split", "sealed"])
        self.assertEqual(info["pgid"], info["pid"], "own process group")
        self.assertNotEqual(info["pgid"], os.getpgrp())
        self.assertEqual(info["core"], [0, 0])
        env = info["env"]
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["HOME"], str(self.staged))
        self.assertTrue({"PATH", "PYTHONHASHSEED"} <= set(env))
        self.assertLessEqual(set(env) - {"PYTHONDONTWRITEBYTECODE", "HOME", "PATH", "PYTHONHASHSEED", "OPENBLAS_NUM_THREADS",
                                         "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LC_CTYPE"}, set())
        self.assertNotIn("sk-grader-secret", json.dumps(env))
        self.assertNotIn("/parent/pp", json.dumps(env))


# --------------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------------
def make_task(**over) -> TaskSpec:
    base = dict(name="t", description="", brief="b", task_dir=Path("/x"), template_dir=Path("/x/template"),
                trusted_dir=Path("/x/trusted"), objective="minimize", primary_metric="m", train_metric="tr", editable=["a.py"],
                scratch=["outputs/**"], run_cmd=["python3", "run.py"], run_timeout_sec=5, grade_cmd=[], grade_timeout_sec=5,
                seeds=[0, 1, 2], replication_seeds=[100, 101, 102], noise_margin=0.02, requires=[])
    base.update(over)
    return TaskSpec(**base)


def grade(per_seed, train=None, *, seeds=None, split="dev", gh="g" * 12, n_eval=70, valid=True, std=None, score=...):
    seeds = list(seeds if seeds is not None else range(len(per_seed)))
    finite = all(math.isfinite(v) for v in per_seed)
    mean = statistics.mean(per_seed) if per_seed and finite else None
    if std is None:
        std = statistics.stdev(per_seed) if len(per_seed) > 1 and finite else 0.0
    return GradeResult(
        valid=valid, score=mean if score is ... else score, std=std,
        per_seed=list(per_seed), metrics={}, per_seed_metrics=[{"m": s, **({"tr": t} if train is not None else {})}
                                                              for s, t in zip(per_seed, train or per_seed)],
        grader_hash=gh, split=split, n_eval=n_eval, seeds=seeds)


# --------------------------------------------------------------------------------------------------
# an independent computation of Student-t quantiles: regularised incomplete beta (continued fraction) + bisection
# --------------------------------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = d if abs(d) > tiny else tiny
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_quantile(p: float, df: int) -> float:
    """The ``p`` quantile of Student's t with ``df`` degrees of freedom (bisection on the exact cdf)."""
    def cdf(t: float) -> float:
        tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
        return 1.0 - tail if t > 0 else tail
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2.0


class CriticalValueTableTests(unittest.TestCase):
    """T95 (one-sided 95%, quantile 0.95) and T975 (two-sided 95%, quantile 0.975), checked against an independent computation."""

    def test_the_helper_reproduces_textbook_values(self):
        for p, df, want in ((0.95, 1, 6.3138), (0.95, 10, 1.8125), (0.975, 1, 12.7062), (0.975, 30, 2.0423), (0.975, 120, 1.9799)):
            self.assertAlmostEqual(_t_quantile(p, df), want, places=4)

    def test_every_tabulated_value_matches_the_exact_quantile(self):
        for name, table, p in (("T95", T95, 0.95), ("T975", T975, 0.975)):
            for df, value in table.items():
                with self.subTest(table=name, df=df):
                    # df 1..10 are the historical three-decimal values (rounded down in two places); the rest are 4-decimal
                    self.assertAlmostEqual(value, _t_quantile(p, df), delta=6e-4 if df <= 10 else 5.1e-5)

    def test_values_are_non_increasing_in_the_degrees_of_freedom_through_the_boundary_and_beyond(self):
        for table in (T95, T975):
            values = [table[df] for df in range(1, 400)]
            self.assertTrue(all(a >= b for a, b in zip(values, values[1:])), "a larger sample never needs a wider margin")
            self.assertGreater(table[10], table[11], "the old table stepped UP at the boundary (1.812 -> 1.645)")
            self.assertGreaterEqual(table[120], table.far)
            self.assertEqual(table[10**6], table.far)
            self.assertEqual(table[0], table[1])

    def test_untabulated_degrees_of_freedom_up_to_120_are_never_below_the_exact_quantile(self):
        # beyond df 120 the normal quantile is used, as documented (its error there is under 1%)
        for table, p in ((T95, 0.95), (T975, 0.975)):
            for df in range(11, 121):
                with self.subTest(p=p, df=df):
                    self.assertGreaterEqual(table[df] + 5.1e-5, _t_quantile(p, df))


class DecideOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.task = make_task()

    def decide(self, cand, inc, **kw):
        return decide_outcome(kw.pop("task", self.task), cand, inc, **kw)

    def test_tables_are_the_contract_values(self):
        # df 1..10 are the values the contract has always shipped (no existing result may change); 11.. were extended
        first = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}
        first975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        self.assertEqual({k: v for k, v in T95.items() if k <= 10}, first)
        self.assertEqual({k: v for k, v in T975.items() if k <= 10}, first975)
        self.assertEqual(sorted(T95), list(range(1, 31)) + [40, 60, 120])
        self.assertEqual(sorted(T975), sorted(T95))
        self.assertEqual((T95[11], T975[11]), (1.7959, 2.2010), "exact t at df 11, not the normal quantile")
        self.assertEqual((T95[200], T975[200]), (1.645, 1.96), "beyond the table: the normal quantile")

    def test_minimize_improved_with_the_noise_margin_floor(self):
        d = self.decide(grade([0.8, 0.8, 0.8]), grade([1.0, 1.0, 1.0]))
        self.assertIsInstance(d, OutcomeDecision)
        self.assertEqual(d.outcome, "improved")
        self.assertAlmostEqual(d.delta, 0.2)
        self.assertEqual((d.n, d.t, d.s_eff), (3, 2.920, 0.0))
        self.assertEqual(d.margin, 0.02)
        self.assertFalse(d.overfit_suspect)

    def test_worse_and_no_gain(self):
        self.assertEqual(self.decide(grade([1.2] * 3), grade([1.0] * 3)).outcome, "worse")
        self.assertEqual(self.decide(grade([1.0] * 3), grade([1.0] * 3)).outcome, "no_gain")
        self.assertEqual(self.decide(grade([0.99] * 3), grade([1.0] * 3)).outcome, "no_gain")
        self.assertEqual(self.decide(grade([1.01] * 3), grade([1.0] * 3)).outcome, "no_gain")

    def test_threshold_is_strict_and_uses_eps(self):
        self.assertEqual(self.decide(grade([0.98] * 3), grade([1.0] * 3)).outcome, "no_gain")     # delta == margin up to rounding
        self.assertEqual(self.decide(grade([0.979] * 3), grade([1.0] * 3)).outcome, "improved")
        self.assertEqual(self.decide(grade([0.98] * 3), grade([1.0] * 3), eps=-1e-3).outcome, "improved")
        self.assertEqual(self.decide(grade([1.02] * 3), grade([1.0] * 3)).outcome, "no_gain")
        self.assertEqual(self.decide(grade([1.021] * 3), grade([1.0] * 3)).outcome, "worse")

    def test_maximize_flips_the_sign(self):
        t = make_task(objective="maximize")
        self.assertEqual(self.decide(grade([1.2] * 3), grade([1.0] * 3), task=t).outcome, "improved")
        self.assertEqual(self.decide(grade([0.8] * 3), grade([1.0] * 3), task=t).outcome, "worse")
        self.assertAlmostEqual(self.decide(grade([1.2] * 3), grade([1.0] * 3), task=t).delta, 0.2)
        self.assertEqual(self.decide(grade([0.8] * 3), grade([1.0] * 3)).outcome, "improved")

    def test_margin_grows_with_the_variance_of_the_paired_differences(self):
        inc = grade([1.0, 1.0, 1.0])
        d = self.decide(grade([0.9, 0.7, 0.8]), inc)             # d = .1 .3 .2
        self.assertAlmostEqual(d.delta, 0.2)
        self.assertAlmostEqual(d.s_eff, 0.1)
        self.assertAlmostEqual(d.margin, 2.920 * 0.1 / math.sqrt(3))
        self.assertEqual(d.outcome, "improved")
        d = self.decide(grade([1.0, 0.6, 0.8]), inc)             # d = 0 .4 .2
        self.assertAlmostEqual(d.margin, 2.920 * 0.2 / math.sqrt(3))
        self.assertEqual(d.outcome, "no_gain")

    def test_pairing_is_by_seed_index_not_by_sorted_value(self):
        d = self.decide(grade([3.0, 2.0, 1.0]), grade([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(d.delta, 0.0)
        self.assertAlmostEqual(d.s_eff, 2.0)         # paired differences are -2, 0, 2
        self.assertEqual(d.outcome, "no_gain")

    def test_a_lucky_low_variance_candidate_cannot_shrink_the_margin_below_sigma_prior(self):
        cand, inc = grade([0.95, 0.95, 0.9499]), grade([1.0, 1.0, 1.0])
        loose = self.decide(cand, inc)
        self.assertEqual(loose.outcome, "improved")
        strict = self.decide(cand, inc, sigma_prior=0.1)
        self.assertEqual(strict.outcome, "no_gain")
        self.assertAlmostEqual(strict.s_eff, 0.1)
        self.assertAlmostEqual(strict.margin, 2.920 * 0.1 / math.sqrt(3))
        # the prior is a floor, not an additive term
        noisy = self.decide(grade([0.9, 0.7, 0.8]), grade([1.0] * 3), sigma_prior=0.05)
        self.assertAlmostEqual(noisy.s_eff, 0.1)

    def test_sigma_floor_does_the_same_job(self):
        t = make_task(sigma_floor=0.1)
        d = self.decide(grade([0.95, 0.95, 0.9499]), grade([1.0] * 3), task=t)
        self.assertEqual((d.outcome, round(d.s_eff, 6)), ("no_gain", 0.1))

    def test_noise_margin_is_a_floor_even_when_statistics_are_tight(self):
        t = make_task(noise_margin=0.5)
        d = self.decide(grade([0.8, 0.8, 0.8]), grade([1.0] * 3), task=t)
        self.assertEqual((d.margin, d.outcome), (0.5, "no_gain"))

    def test_single_seed(self):
        d = self.decide(grade([0.5]), grade([1.0]))
        self.assertEqual((d.outcome, d.n, d.t, d.s_eff, d.margin), ("improved", 1, 6.314, 0.0, 0.02))
        d = self.decide(grade([0.5]), grade([1.0]), sigma_prior=0.1)
        self.assertEqual((d.outcome, d.n), ("no_gain", 1))
        self.assertAlmostEqual(d.margin, 6.314 * 0.1)
        self.assertEqual(self.decide(grade([0.5]), grade([1.0]), k_sigma=T975, sigma_prior=0.01).t, 12.706)

    def test_identical_grades_have_zero_variance(self):
        d = self.decide(grade([0.4, 0.5, 0.6]), grade([0.4, 0.5, 0.6]))
        self.assertEqual((d.delta, d.s_eff, d.outcome), (0.0, 0.0, "no_gain"))

    def test_critical_value_follows_the_sample_size(self):
        for n, expected in ((2, 6.314), (3, 2.920), (4, 2.353), (5, 2.132), (11, 1.812), (12, 1.7959), (30, 1.6991),
                            (31, 1.6973), (41, 1.6839), (121, 1.6577), (122, 1.645)):
            with self.subTest(n=n):
                d = self.decide(grade([0.5] * n), grade([1.0] * n))
                self.assertEqual((d.n, d.t), (n, expected))

    def test_sealed_k_sigma_is_stricter(self):
        offsets = (-0.1, -0.05, 0.0, 0.05, 0.1)
        diffs = [0.07 + 0.85 * o for o in offsets]
        inc, cand = grade([1.0] * 5, seeds=range(5)), grade([1.0 - x for x in diffs], seeds=range(5))
        dev = self.decide(cand, inc)
        self.assertEqual((dev.outcome, dev.t), ("improved", 2.132))
        for k in (T975, 2.776, T975[4]):
            sealed = self.decide(cand, inc, k_sigma=k)
            self.assertEqual((sealed.outcome, sealed.t), ("no_gain", 2.776))
            self.assertEqual(sealed.delta, dev.delta)
            self.assertGreater(sealed.margin, dev.margin)

    def test_k_sigma_table_covers_every_sample_size(self):
        self.assertEqual(self.decide(grade([0.5] * 3), grade([1.0] * 3), k_sigma=T975).t, 4.303)
        self.assertEqual(self.decide(grade([0.5] * 12), grade([1.0] * 12), k_sigma=T975).t, 2.2010)
        self.assertEqual(self.decide(grade([0.5] * 150), grade([1.0] * 150), k_sigma=T975).t, 1.96)
        self.assertEqual(self.decide(grade([0.5] * 3), grade([1.0] * 3), k_sigma={2: 9.0}).t, 9.0)
        with self.assertRaises(ValueError):
            self.decide(grade([0.5] * 3), grade([1.0] * 3), k_sigma={5: 9.0})

    def test_bad_k_sigma(self):
        for bad in (0, -1.0, float("nan"), float("inf")):
            with self.subTest(k=bad):
                with self.assertRaises(ValueError):
                    self.decide(grade([0.5] * 3), grade([1.0] * 3), k_sigma=bad)
        for bad in ("2.0", True, [1.0]):
            with self.subTest(k=bad):
                with self.assertRaises(TypeError):
                    self.decide(grade([0.5] * 3), grade([1.0] * 3), k_sigma=bad)

    def assertNotComparable(self, d):
        self.assertEqual((d.outcome, d.reason, d.delta, d.overfit_suspect, d.train_delta), ("invalid", "not_comparable", None, False, None))
        self.assertTrue(d.detail)

    def test_not_comparable_cases(self):
        ok = grade([0.8] * 3)
        inc = grade([1.0] * 3)
        cases = {
            "candidate invalid": (grade([0.8] * 3, valid=False), inc),
            "incumbent invalid": (ok, grade([1.0] * 3, valid=False)),
            "grader hash": (grade([0.8] * 3, gh="x" * 12), inc),
            "split": (grade([0.8] * 3, split="sealed"), inc),
            "seeds": (grade([0.8] * 3, seeds=[0, 1, 9]), inc),
            "seed order": (grade([0.8] * 3, seeds=[2, 1, 0]), inc),
            "n": (grade([0.8] * 2, seeds=[0, 1]), inc),
            "n_eval": (grade([0.8] * 3, n_eval=71), inc),
            "empty": (grade([], seeds=[], score=0.0), grade([], seeds=[], score=0.0)),
            "nan per_seed": (grade([0.8, float("nan"), 0.8]), inc),
            "inf per_seed": (grade([0.8, float("inf"), 0.8]), inc),
            "nan incumbent": (ok, grade([1.0, float("nan"), 1.0])),
            "nan score": (grade([0.8] * 3, score=float("nan")), inc),
            "none score": (grade([0.8] * 3, score=None), inc),
            "nan std": (grade([0.8] * 3, std=float("nan")), inc),
            "nan train": (grade([0.8] * 3, train=[0.5, float("nan"), 0.5]), grade([1.0] * 3, train=[1.0] * 3)),
        }
        for name, (cand, incumbent) in cases.items():
            with self.subTest(name):
                self.assertNotComparable(self.decide(cand, incumbent))
        self.assertNotComparable(self.decide(ok, inc, sigma_prior=float("nan")))

    def test_arithmetic_overflow_in_the_paired_differences_is_not_comparable(self):
        big = 1e308
        self.assertNotComparable(self.decide(grade([big, -big, big]), grade([-big, big, -big])))
        self.assertNotComparable(self.decide(grade([-big] * 3), grade([big] * 3)))
        # the train metric overflowing is reported the same way, never raised
        self.assertNotComparable(self.decide(grade([0.8] * 3, train=[big] * 3), grade([1.0] * 3, train=[-big] * 3)))

    def test_infinite_paired_differences_are_not_comparable_on_every_python(self):
        # statistics.stdev raises AttributeError on inf/nan on 3.11 and 3.12 but ValueError on 3.13: the verdict must not depend on it
        big = 1e308
        for cand, inc in (([big, -big], [-big, big]), ([big, 0.0, 0.0], [-big, 0.0, 0.0]), ([-big, -big], [big, big])):
            with self.subTest(cand=cand):
                self.assertNotComparable(self.decide(grade(cand, seeds=range(len(cand))), grade(inc, seeds=range(len(inc)))))
        task = make_task(objective="maximize")
        self.assertNotComparable(self.decide(grade([big, big, big]), grade([-big] * 3), task=task))
        # a finite spread that is merely huge is still a verdict, not a crash
        d = self.decide(grade([big / 2, -big / 2]), grade([-big / 2, big / 2]))
        self.assertIn(d.outcome, ("improved", "worse", "no_gain", "invalid"))

    def test_n_eval_zero_means_unknown_and_is_comparable(self):
        self.assertEqual(self.decide(grade([0.8] * 3, n_eval=0), grade([1.0] * 3)).outcome, "improved")

    def test_overfit_suspect_when_train_improves_but_heldout_does_not(self):
        inc = grade([1.0, 1.0, 1.0], train=[0.9, 0.9, 0.9])
        cand = grade([1.0, 1.0, 1.0], train=[0.5, 0.5, 0.5])
        d = self.decide(cand, inc)
        self.assertEqual(d.outcome, "no_gain")
        self.assertAlmostEqual(d.train_delta, 0.4)
        self.assertEqual(d.train_margin, 0.02)
        self.assertTrue(d.overfit_suspect)
        worse = self.decide(grade([1.3] * 3, train=[0.5] * 3), inc)
        self.assertEqual((worse.outcome, worse.overfit_suspect), ("worse", True))

    def test_no_overfit_flag_when_heldout_improves_or_train_does_not(self):
        inc = grade([1.0] * 3, train=[0.9] * 3)
        both = self.decide(grade([0.7] * 3, train=[0.5] * 3), inc)
        self.assertEqual((both.outcome, both.overfit_suspect), ("improved", False))
        self.assertGreater(both.train_delta, 0)
        flat = self.decide(grade([1.0] * 3, train=[0.9] * 3), inc)
        self.assertEqual((flat.overfit_suspect, flat.train_delta), (False, 0.0))
        small = self.decide(grade([1.0] * 3, train=[0.89] * 3), inc)
        self.assertFalse(small.overfit_suspect)

    def test_train_margin_uses_the_train_variance(self):
        inc = grade([1.0] * 3, train=[1.0, 1.0, 1.0])
        d = self.decide(grade([1.0] * 3, train=[0.9, 0.7, 0.8]), inc)
        self.assertAlmostEqual(d.train_delta, 0.2)
        self.assertAlmostEqual(d.train_margin, 2.920 * 0.1 / math.sqrt(3))

    def test_train_follows_the_objective_direction(self):
        t = make_task(objective="maximize")
        inc = grade([1.0] * 3, train=[0.5] * 3)
        d = self.decide(grade([1.0] * 3, train=[0.9] * 3), inc, task=t)
        self.assertTrue(d.overfit_suspect)
        self.assertAlmostEqual(d.train_delta, 0.4)

    def test_no_train_information(self):
        t = make_task(train_metric=None)
        d = self.decide(grade([0.8] * 3, train=[0.1] * 3), grade([1.0] * 3, train=[0.9] * 3), task=t)
        self.assertEqual((d.train_delta, d.train_margin, d.overfit_suspect), (None, None, False))
        d = self.decide(grade([1.0] * 3), grade([1.0] * 3))          # per_seed_metrics lack the train metric
        self.assertEqual((d.train_delta, d.overfit_suspect), (None, False))

    def test_decision_is_serialisable_and_carries_the_detail_field(self):
        d = self.decide(grade([0.8] * 3), grade([1.0] * 3))
        as_dict = dataclasses.asdict(d)
        json.dumps(as_dict)
        self.assertEqual(as_dict["outcome"], "improved")
        self.assertEqual(as_dict["detail"], "")


if __name__ == "__main__":
    unittest.main()
