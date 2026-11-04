"""Tests for scripts/null_benchmark.py and the document it generates (docs/NULL_BENCHMARK.md).

The full benchmark (20,000 campaigns) takes minutes and is not run here. These tests check that the script models what ships
(its defaults are read from ``task.toml``), that it calls the shipped ``decide_outcome``, that it is deterministic, and that the
committed document states the script's parameters, has the script's structure and contains numbers the script reproduces
for the stated seed and campaign count (a few cheap rows are regenerated in full).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import random
import re
import statistics
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from nightshift import grading
from nightshift.types import BudgetLimits

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "null_benchmark.py"
DOC = ROOT / "docs" / "NULL_BENCHMARK.md"
TASK_TOML = ROOT / "tasks" / "law-discovery" / "task.toml"


def load_script():
    """Import the script by path without leaving bytecode next to it."""
    keep = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("null_benchmark_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module            # @dataclass looks its module up here
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = keep


nb = load_script()
GRADE = tomllib.loads(TASK_TOML.read_text(encoding="utf-8"))["grade"]


def parse(*argv: str):
    return nb.build_parser().parse_args(list(argv))


def small_report(campaigns: int = 30, *extra: str) -> str:
    return nb.report(parse("--campaigns", str(campaigns), *extra))


class DefaultsAreTheShippedConstants(unittest.TestCase):
    def test_defaults_equal_the_task_toml_and_the_budget_defaults(self):
        args = parse()
        self.assertEqual(args.noise_margin, GRADE["noise_margin"])
        self.assertEqual(args.sigma_floor, GRADE["sigma_floor"])
        self.assertEqual(args.dev_seeds, GRADE["seeds"])
        self.assertEqual(args.sealed_seeds, GRADE["replication_seeds"])
        self.assertEqual(args.max_replications, BudgetLimits().max_replications)
        self.assertEqual((len(args.dev_seeds), len(args.sealed_seeds), args.max_replications), (3, 5, 6))

    def test_the_documented_literals_match_the_task(self):
        self.assertEqual((nb.DEFAULT_SEED, nb.DEFAULT_CAMPAIGNS, nb.DEFAULT_K), (7, 20000, 16))
        # the floor is calibrated to the baseline's per-seed std, which is also the model's true baseline std
        self.assertEqual(nb.SIGMA_BASELINE, GRADE["sigma_floor"])
        self.assertEqual(nb.PREVIOUS_SIGMA_FLOOR, 0.02)
        self.assertLess(nb.SHIPPED_PRIOR, GRADE["sigma_floor"], "the realized prior is below the floor: the floor binds")

    def test_the_realized_prior_is_the_baselines_std_on_the_dev_seeds(self):
        """SHIPPED_PRIOR is the real unmodified baseline's std on dev seeds 0, 1, 2: measured through the real sandbox."""
        from nightshift import sandbox
        from nightshift.task import load_task
        task = load_task(ROOT / "tasks" / "law-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            ws = sandbox.create_workspace(task, Path(tmp), "b")
            res = grading.evaluate(task, ws, task.seeds, purpose="baseline", run_dir=Path(tmp), expected_tree_hash=ws.tree_hash,
                                   isolation="none")
        self.assertTrue(res.grade.valid)
        self.assertAlmostEqual(res.grade.std, nb.SHIPPED_PRIOR, places=3)

    def test_the_decision_code_is_the_shipped_one(self):
        self.assertIs(nb.decide_outcome, grading.decide_outcome)
        self.assertIs(nb.T95, grading.T95)
        self.assertIs(nb.T975, grading.T975)

    def test_the_script_never_hard_codes_the_floor_or_margin(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"noise_margin\s*=\s*0\.0\d")
        self.assertIn("load_task(TASK_DIR)", text)


class SimulationTests(unittest.TestCase):
    def rules(self, **kw):
        a = parse()
        return nb.Rules(a.noise_margin, kw.get("sigma_floor", a.sigma_floor), a.dev_seeds, a.sealed_seeds,
                        shortcut=kw.get("shortcut", True))

    def test_decide_outcome_is_called_for_the_dev_and_the_sealed_gate(self):
        calls: list[tuple[str, object, float]] = []
        real = grading.decide_outcome

        def spy(task, candidate, incumbent, **kw):
            calls.append((candidate.split, kw.get("k_sigma"), kw.get("sigma_prior")))
            return real(task, candidate, incumbent, **kw)

        with mock.patch.object(nb, "decide_outcome", spy):
            cell = nb.simulate(40, 0.15, 0.036, self.rules(), seed=3, ks=(1, 4))
        self.assertGreater(cell.ns_run[4], 0, "a real effect of 0.15 is promoted")
        dev = [c for c in calls if c[0] == "dev"]
        sealed = [c for c in calls if c[0] == "sealed"]
        self.assertTrue(dev and sealed)
        self.assertTrue(all(c[1] is None for c in dev), "dev uses the default 95% one-sided critical values")
        self.assertTrue(all(c[1] is grading.T975 for c in sealed), "sealed uses the two-sided 95% table")
        self.assertTrue(all(c[2] is not None and c[2] > 0 for c in calls), "sigma_prior is passed on every call")

    def test_deterministic_for_a_seed_and_different_for_another(self):
        a = nb.simulate(60, 0.0, 0.07, self.rules(), seed=5, ks=(1, 16))
        b = nb.simulate(60, 0.0, 0.07, self.rules(), seed=5, ks=(1, 16))
        c = nb.simulate(60, 0.0, 0.07, self.rules(), seed=6, ks=(1, 16))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_the_report_is_deterministic_for_a_seed(self):
        self.assertEqual(small_report(12), small_report(12))
        self.assertNotEqual(small_report(12), small_report(12, "--seed", "8"))

    def test_the_shortcut_gives_the_same_tallies_as_calling_decide_outcome_for_every_candidate(self):
        for theta, sd in ((0.0, 0.07), (0.08, 0.036), (0.12, 0.05)):
            fast = nb.simulate(80, theta, sd, self.rules(shortcut=True), seed=11, ks=(1, 4, 16))
            slow = nb.simulate(80, theta, sd, self.rules(shortcut=False), seed=11, ks=(1, 4, 16))
            self.assertEqual(fast, slow, (theta, sd))

    def test_nothing_at_or_below_the_noise_margin_can_be_improved(self):
        """The shortcut's premise, checked against the shipped code: delta > margin >= noise_margin."""
        rules = self.rules(shortcut=False)
        rng = random.Random(1)
        for _ in range(400):
            sd = rng.choice((0.005, 0.02, 0.05, 0.1))
            dev = [rng.gauss(rng.uniform(-0.05, 0.2), sd) for _ in range(3)]
            sealed = [rng.gauss(rng.uniform(-0.05, 0.2), sd) for _ in range(5)]
            prior = rng.uniform(0.0, 0.08)
            if rules.dev_gate(dev, prior):
                self.assertGreater(statistics.fmean(dev), rules.task.noise_margin)
            if rules.sealed_gate(sealed, prior):
                self.assertGreater(statistics.fmean(sealed), rules.task.noise_margin)

    def test_the_sealed_cap_limits_how_many_dev_winners_can_be_promoted(self):
        capped = nb.simulate(60, 0.2, 0.036, self.rules(), seed=2, ks=(48,), max_replications=1)
        free = nb.simulate(60, 0.2, 0.036, self.rules(), seed=2, ks=(48,), max_replications=6)
        self.assertGreaterEqual(free.ns_run[48], capped.ns_run[48])
        zero = nb.simulate(60, 0.2, 0.036, self.rules(), seed=2, ks=(48,), max_replications=0)
        self.assertEqual(zero.ns_run[48], 0, "no sealed query, no promotion")
        self.assertEqual(zero.dev_run[48], capped.dev_run[48], "the cap does not change the dev gate")

    def test_a_fixed_prior_is_used_instead_of_a_drawn_one(self):
        seen: list[float] = []
        real = grading.decide_outcome

        def spy(task, candidate, incumbent, **kw):
            seen.append(kw["sigma_prior"])
            return real(task, candidate, incumbent, **kw)

        with mock.patch.object(nb, "decide_outcome", spy):
            nb.simulate(20, 0.2, 0.036, self.rules(), seed=1, ks=(4,), fixed_prior=0.0123)
            self.assertEqual(set(seen), {0.0123})
            seen.clear()
            nb.simulate(20, 0.2, 0.036, self.rules(), seed=1, ks=(4,))
            self.assertGreater(len(set(seen)), 1, "without fixed_prior every campaign draws its own sigma_prior")

    def test_the_calibrated_floor_lowers_the_false_pass_rate_of_noisy_candidates_and_raises_the_bar(self):
        old = nb.simulate(400, 0.0, 0.05, self.rules(sigma_floor=0.02), seed=7, ks=(16,), fixed_prior=nb.SHIPPED_PRIOR)
        new = nb.simulate(400, 0.0, 0.05, self.rules(), seed=7, ks=(16,), fixed_prior=nb.SHIPPED_PRIOR)
        self.assertGreater(old.cand_dev, 1.5 * new.cand_dev, "the old floor let about twice as many nulls through the dev gate")
        self.assertGreaterEqual(old.dev_run[16], new.dev_run[16])
        # a modest real effect is promoted less often under the calibrated floor: that is the price
        old_p = nb.simulate(400, 0.05, 0.036, self.rules(sigma_floor=0.02), seed=7, ks=(1,), fixed_prior=nb.SHIPPED_PRIOR)
        new_p = nb.simulate(400, 0.05, 0.036, self.rules(), seed=7, ks=(1,), fixed_prior=nb.SHIPPED_PRIOR)
        self.assertGreater(old_p.ns_run[1], new_p.ns_run[1])

    def test_the_prior_is_drawn_per_campaign_from_three_baseline_scores(self):
        """Different campaigns see different priors: the sample std of 3 normal scores, not a constant."""
        seen: list[float] = []
        real = grading.decide_outcome

        def spy(task, candidate, incumbent, **kw):
            seen.append(kw["sigma_prior"])
            return real(task, candidate, incumbent, **kw)

        with mock.patch.object(nb, "decide_outcome", spy):
            nb.simulate(60, 0.2, 0.036, self.rules(), seed=4, ks=(1,))
        self.assertGreater(len(set(seen)), 20)
        self.assertGreater(max(seen), 2 * min(seen), "a std of 3 scores varies by a large factor")


class CommandLineTests(unittest.TestCase):
    """Hostile command lines are usage errors, never a traceback from deep inside a table."""

    def refused(self, *argv: str) -> str:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            nb.build_parser().parse_args(list(argv))
        self.assertEqual(caught.exception.code, 2)
        return err.getvalue()

    def test_a_count_below_one_is_refused(self):
        for flag in ("--campaigns", "--k"):
            for bad in ("0", "-3", "x", "1.5"):
                with self.subTest(flag=flag, bad=bad):
                    self.assertIn(flag, self.refused(flag, bad))

    def test_a_noise_level_floor_or_margin_must_be_a_finite_number_of_at_least_zero(self):
        for flag in ("--sigma-d", "--sigma-baseline", "--sigma-floor", "--noise-margin", "--compare-floor", "--shipped-prior"):
            for bad in ("-0.1", "nan", "inf", "abc"):
                with self.subTest(flag=flag, bad=bad):
                    self.assertIn(flag, self.refused(flag, bad))

    def test_zero_sealed_queries_is_allowed_and_negative_is_not(self):
        self.assertEqual(parse("--max-replications", "0").max_replications, 0)
        self.assertIn("--max-replications", self.refused("--max-replications", "-1"))

    def test_zero_sealed_queries_promotes_nothing_in_a_full_report(self):
        text = small_report(8, "--max-replications", "0")
        self.assertIn("with at most 0 sealed queries per run", text)


class WilsonIntervalTests(unittest.TestCase):
    def test_known_values_and_edges(self):
        low, high = nb.wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)
        low, high = nb.wilson_interval(0, 100)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.0370, places=3)
        self.assertEqual(nb.wilson_interval(0, 0), (0.0, 1.0))
        low, high = nb.wilson_interval(2, 2)
        self.assertEqual(high, 1.0)
        self.assertGreater(low, 0.3)

    def test_the_interval_contains_the_rate_and_narrows_with_more_trials(self):
        for k, n in ((1, 10), (3, 50), (24, 416), (0, 7)):
            low, high = nb.wilson_interval(k, n)
            self.assertLessEqual(low, k / n)
            self.assertGreaterEqual(high, k / n)
        narrow, wide = nb.wilson_interval(24, 416), nb.wilson_interval(3, 52)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])


class ComparisonVerdictTests(unittest.TestCase):
    def test_the_three_verdicts_and_the_empty_case(self):
        self.assertIn("too few candidates", nb.comparison_verdict(0, 0, 0.05, 48))
        # 24 of 416 = 5.8%, Wilson interval about 3.9% to 8.4%
        self.assertIn("agree within sampling error", nb.comparison_verdict(24, 416, 0.0472, 48))
        under = nb.comparison_verdict(24, 416, 0.01, 48)
        self.assertIn("understates the real rate", under)
        self.assertIn("optimistic", under)
        self.assertIn("overstates the real rate", nb.comparison_verdict(24, 416, 0.20, 48))

    def test_zero_promotions_in_a_few_runs_is_never_called_a_disagreement(self):
        self.assertIn("agree within sampling error", nb.comparison_verdict(0, 6, 0.02, 16))


class LawReplicaTests(unittest.TestCase):
    """Section 6: nulls drawn through the task's own data module, baseline model and trusted grader, in process."""

    BAG_MODEL = """\
import random

SALT = %r
FRACTION = %r


def fit_predict(train, query_times):
    sub = random.Random(f"{SALT}|{train!r}").sample(train, k=max(2, int(FRACTION * len(train))))
    n = len(sub)
    mean_t = sum(t for t, _ in sub) / n
    mean_y = sum(y for _, y in sub) / n
    sxx = sum((t - mean_t) ** 2 for t, _ in sub)
    sxy = sum((t - mean_t) * (y - mean_y) for t, y in sub)
    slope = sxy / sxx
    return [mean_y + slope * (t - mean_t) for t in query_times]
"""

    @classmethod
    def setUpClass(cls) -> None:
        args = parse()
        cls.args = args
        cls.replica = nb.LawReplica(args.dev_seeds, args.sealed_seeds)

    def test_the_replica_leaves_no_bytecode_in_the_task_directory(self):
        self.assertEqual(list((ROOT / "tasks" / "law-discovery").rglob("__pycache__")), [])

    def test_the_replicas_prior_is_the_shipped_realized_prior(self):
        self.assertAlmostEqual(self.replica.prior, nb.SHIPPED_PRIOR, places=3)
        self.assertEqual(len(self.replica.baseline), 8)

    def test_a_bagged_null_scores_the_same_in_process_as_through_the_real_sandbox(self):
        """The replica is only worth something if it reproduces the sandboxed pipeline: same per-seed RMSE, dev and sealed."""
        from nightshift import sandbox
        from nightshift.task import load_task
        task = load_task(ROOT / "tasks" / "law-discovery")
        salt, fraction = "equivalence:1", 0.5
        with tempfile.TemporaryDirectory() as tmp:
            ws = sandbox.create_workspace(task, Path(tmp), "bag")
            (ws.path / "model.py").write_text(self.BAG_MODEL % (salt, fraction), encoding="utf-8")
            tree = sandbox.tree_hash(ws.path, task)
            dev = grading.evaluate(task, ws, task.seeds, purpose="primary", run_dir=Path(tmp), expected_tree_hash=tree, isolation="none")
            sealed = grading.evaluate(task, ws, task.replication_seeds, purpose="replication", run_dir=Path(tmp),
                                      expected_tree_hash=tree, isolation="none", run_idx_start=len(task.seeds))
        self.assertTrue(dev.grade.valid and sealed.grade.valid, (dev.grade.error, sealed.grade.error))
        mine = self.replica.bag(fraction, salt)
        for got, want in zip(dev.grade.per_seed + sealed.grade.per_seed, mine):
            self.assertAlmostEqual(got, want, places=12)
        # and it is not the baseline: the subset changes the score
        self.assertNotAlmostEqual(mine[0], self.replica.baseline[0], places=6)

    def test_candidates_are_deterministic_and_differ_by_salt_and_by_seed(self):
        a, b = self.replica.bag(0.5, "x"), self.replica.bag(0.5, "x")
        self.assertEqual(a, b)
        self.assertNotEqual(a, self.replica.bag(0.5, "y"))
        self.assertEqual(len(set(a)), len(a), "each seed trains on different data, so each seed's score differs")

    def test_the_decisions_go_through_the_shipped_decide_outcome(self):
        rules = nb.Rules(self.args.noise_margin, self.args.sigma_floor, self.args.dev_seeds, self.args.sealed_seeds)
        draws = nb.real_null_draw(self.replica, 0.3, 40, 7)
        self.assertEqual(len(draws), 40)
        calls = []
        real = grading.decide_outcome

        def spy(task, candidate, incumbent, **kw):
            calls.append((candidate.split, kw["sigma_prior"]))
            return real(task, candidate, incumbent, **kw)

        with mock.patch.object(nb, "decide_outcome", spy):
            decisions = nb.real_null_decisions(draws, rules, self.replica.prior)
        self.assertEqual(len(decisions), 40)
        self.assertTrue(any(c[0] == "dev" for c in calls))
        self.assertEqual({c[1] for c in calls}, {self.replica.prior}, "the real baseline's std is the prior")
        self.assertTrue(all(not sealed or dev for dev, sealed in decisions), "the sealed gate is only asked of a dev winner")

    def test_the_sealed_cap_and_the_run_grouping(self):
        winners = [(True, False)] * 6 + [(True, True)]           # the 7th dev winner would pass the sealed gate
        quiet = [(False, False)] * 9
        run = winners + quiet                                  # one run of 16 candidates
        capped = nb.real_null_cell(run, (16,), 6)
        self.assertEqual(capped.runs[16], (0, 1, 1), "a dev winner beyond the cap is never replicated")
        self.assertEqual(nb.real_null_cell(run, (16,), 7).runs[16], (1, 1, 1))
        self.assertEqual(nb.real_null_cell(run, (16,), 0).runs[16], (0, 1, 1), "no sealed query, no promotion")
        self.assertEqual((capped.n, capped.dev, capped.both), (16, 7, 1), "per-candidate tallies ignore the cap")
        self.assertEqual(nb.real_null_cell(run + run[:5], (16,), 6).runs[16][2], 1, "a partial run is dropped")
        self.assertEqual(nb.real_null_cell(run, (48,), 6).runs[48], (0, 0, 0))

    def test_a_run_count_of_zero_says_why_instead_of_dividing_by_zero(self):
        cell = nb.real_null_cell([(False, False)] * 5, (16,), 6)
        self.assertEqual(nb._runs_cell(cell, 16), "n/a (fewer than 16 candidates)")
        self.assertEqual(nb._runs_cell(nb.real_null_cell([(False, False)] * 32, (16,), 6), 16), "0 of 2")

    def test_the_report_has_the_section_and_its_rows_for_every_family(self):
        text = small_report(20)
        body = text.split("## 6. Nulls drawn from the real grader")[1].split("<!-- measured-on-the-real-pipeline:begin -->")[0]
        for frac in nb.REAL_NULL_FRACTIONS:
            for label in ("previous floor 0.02, real", "shipped floor 0.036, real", "shipped floor 0.036, model"):
                self.assertIn(f"| bag {round(100 * frac)}% |", body)
                self.assertIn(label, body)
        self.assertIn("n/a (fewer than 48 candidates)", body, "20 candidates make no 48-experiment run")
        self.assertIn("not the audit's candidates", body)
        self.assertIn("A floor can only help when the candidate's own sample std", body)


class ReportTests(unittest.TestCase):
    def test_the_script_runs_and_prints_every_section(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(nb.main(["--campaigns", "15"]), 0)
        text = out.getvalue()
        for heading in ("# Null-hypothesis benchmark", "## Setup", "## 1. False wins per run", "## 2. False promotion by candidate noise",
                        "## 3. The winner's curse", "## 4. The price: sensitivity", "## 5. What calibrating `sigma_floor` changed",
                        "## 6. Nulls drawn from the real grader", "## Measured on the real pipeline", "## Reading it"):
            self.assertIn(heading, text)

    def test_write_creates_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "out.md"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(nb.main(["--campaigns", "10", "--write", str(target)]), 0)
            self.assertTrue(target.read_text(encoding="utf-8").startswith("# Null-hypothesis benchmark"))
            self.assertIn("wrote", out.getvalue())

    def test_the_report_models_the_noisy_estimate_of_the_prior_and_a_row_family_above_it(self):
        text = small_report(10)
        self.assertIn("The prior is not a constant", text)
        self.assertIn("sample std of those scores is that campaign's `sigma_prior`", text)
        for sd in ("0.018", "0.036", "0.050", "0.070"):
            self.assertRegex(text, rf"\| {sd} \|")
        self.assertIn("per run, 16 experiments", text)
        self.assertIn("per run, 48 experiments", text)
        self.assertIn("one-sided 5% t-test", text)
        self.assertIn("What the sealed stage then buys", text)
        self.assertIn("at most 6 sealed queries per run", text)

    def test_the_measured_section_says_who_measured_it_and_that_it_is_not_reproducible_here(self):
        self.assertIn("throwaway harness that is **not part of this repository**", nb.MEASURED_SECTION)
        self.assertIn("independent statistics audit", nb.MEASURED_SECTION)
        self.assertIn("have **not** been re-measured", nb.MEASURED_SECTION)
        for token in ("190", "4.2%", "1.1%", "11,000", "0.0088 / 0.0304 / 0.0639", "wls1 0.5%, bag70 1.6%, bag50 3.7%, bag30 4.8%",
                      "7 of 3,000", "+0.73", "+1.56", "0.029 RMSE improvement promoted 0%", "0.124 97%"):
            self.assertIn(token, nb.MEASURED_SECTION)


# --------------------------------------------------------------------------------------------------
# the committed document
# --------------------------------------------------------------------------------------------------
def outside_measured(text: str) -> str:
    a, b = text.index(nb.MEASURED_BEGIN), text.index(nb.MEASURED_END) + len(nb.MEASURED_END)
    return (text[:a] + text[b:]).strip()


NUMERIC_CELL = re.compile(r"^\**(?:[+-]?[\d.,]+%?(?: of [\d,]+)?(?: \([\d,]+ of [\d,]+\))?|n/a(?: \(fewer than \d+ candidates\))?)\**$")


def shape(text: str) -> list[str]:
    """The structure of a generated document: prose and labels kept verbatim, every number replaced by '#'."""
    out = []
    for line in text.splitlines():
        if line.startswith("|") and not set(line) <= set("|- "):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            line = "| " + " | ".join("#" if NUMERIC_CELL.match(c) else c for c in cells) + " |"
        elif line.startswith("* sigma_d "):
            line = "* sigma_d #: #"
        elif line.startswith("In the noisiest family"):
            line = "In the noisiest family #"          # which comparison verdict applies depends on the campaign count
        out.append(re.sub(r"\d[\d,]*\.?\d*", "#", line))
    return out


class CommittedDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = DOC.read_text(encoding="utf-8")

    def test_the_measured_section_is_exactly_the_scripts(self):
        a, b = self.doc.index(nb.MEASURED_BEGIN), self.doc.index(nb.MEASURED_END) + len(nb.MEASURED_END)
        self.assertEqual(self.doc[a:b] + "\n", nb.MEASURED_SECTION)

    def test_the_setup_states_the_parameters_the_script_defaults_to(self):
        args = parse()
        setup = self.doc.split("## Setup")[1].split("## 1.")[0]
        self.assertIn(f"* {args.campaigns:,} campaigns per cell, up to {3 * args.k} candidates per campaign", setup)
        self.assertIn(f"random seed {nb.DEFAULT_SEED}.", setup)
        self.assertIn(f"noise_margin {args.noise_margin}, sigma_floor {args.sigma_floor}, {len(args.dev_seeds)} dev seeds, "
                      f"{len(args.sealed_seeds)} sealed seeds. Budget: max_replications {args.max_replications}", setup)
        self.assertIn(f"(true baseline per-seed std {args.sigma_baseline})", setup)
        self.assertIn(f"--campaigns {nb.DEFAULT_CAMPAIGNS} --seed {nb.DEFAULT_SEED} --write docs/NULL_BENCHMARK.md",
                      self.doc.split("## Setup")[0])

    def test_the_documented_margins_follow_from_the_constants(self):
        args = parse()
        dev = max(args.noise_margin, grading.T95[len(args.dev_seeds) - 1] * args.sigma_floor / len(args.dev_seeds) ** 0.5)
        sealed = max(args.noise_margin, grading.T975[len(args.sealed_seeds) - 1] * args.sigma_floor / len(args.sealed_seeds) ** 0.5)
        self.assertIn(f"= {dev:.4f} when s_eff is the floor", self.doc)
        self.assertIn(f"= {sealed:.4f}. When", self.doc)

    def test_structure_labels_and_prose_equal_what_the_script_generates(self):
        """Regenerate at a tiny campaign count: every label, heading and sentence must be the committed document's."""
        generated = nb.report(parse("--campaigns", "25"))
        want, got = shape(outside_measured(self.doc)), shape(outside_measured(generated))
        self.assertEqual(len(want), len(got))
        for i, (a, b) in enumerate(zip(want, got)):
            self.assertEqual(a, b, f"line {i + 1} of the document differs from what the script generates")

    def test_power_rows_are_reproduced_for_the_stated_seed_and_campaign_count(self):
        """Regenerate the theta = 0 and theta = 0.075 rows of section 4 in full (20,000 campaigns each, single candidates)."""
        args = parse()
        rules = nb.Rules(args.noise_margin, args.sigma_floor, args.dev_seeds, args.sealed_seeds)
        table = self.doc.split("## 4. The price: sensitivity")[1].split("## 5.")[0]
        rows = {}
        for line in table.splitlines():
            m = re.match(r"\| (\d\.\d{3}) \| (.+) \|$", line)
            if m:
                rows[float(m.group(1))] = [c.strip() for c in m.group(2).split("|")]
        for theta in (0.0, 0.075):
            expected = []
            for sd in nb.NOISE_ROWS:
                cell = nb.simulate(args.campaigns, theta, sd, rules, seed=nb.DEFAULT_SEED, ks=(1,),
                                   sigma_baseline=args.sigma_baseline, max_replications=args.max_replications)
                expected.append(nb.pct_n(cell.ns_run[1], args.campaigns) if theta == 0.0 else nb.pct(cell.rate(cell.ns_run, 1)))
            self.assertEqual(rows[theta], expected, f"theta {theta}")

    def test_section_6_is_in_the_document_with_one_of_the_three_verdicts(self):
        body = self.doc.split("## 6. Nulls drawn from the real grader")[1].split(nb.MEASURED_BEGIN)[0]
        self.assertIn("20,000 candidates per family", body)
        self.assertRegex(body, r"(understates|overstates) the real rate of the 48-experiment runs|agree within sampling error for the 48-experiment runs")
        for frac in nb.REAL_NULL_FRACTIONS:
            self.assertEqual(body.count(f"| bag {round(100 * frac)}% |"), 3, "previous, shipped and model row")

    def test_the_document_says_what_it_is_not(self):
        self.assertIn("it says nothing about any language model", self.doc)
        self.assertIn("independent audit measured on the real pipeline", self.doc)
        # the old headline claim must be gone: nothing here may state that the shipped rule never lets a null through
        self.assertNotIn("0.00%", self.doc.split("## 1.")[1].split("## 2.")[0])


if __name__ == "__main__":
    unittest.main()
