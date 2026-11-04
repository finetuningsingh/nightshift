#!/usr/bin/env python3
"""How often does an auto-researcher fool itself, and what does Nightshift's decision rule cost in sensitivity?

A Monte Carlo over simulated research campaigns that calls the SHIPPED decision code (``nightshift.grading.decide_outcome``
and its critical-value tables). It is a statistical check of the decision rule under a stated noise model. It says nothing
about any language model, and the model is a simplification of the real pipeline: the section "Measured on the real
pipeline" of the generated document is a separate, static record of an independent audit.

Model (all stated, all adjustable on the command line; the defaults are read from ``tasks/law-discovery/task.toml``):
  * A campaign (one run of the system) first measures the baseline on ``n_dev`` dev seeds. Its sample std is the campaign's
    ``sigma_prior``: the engine uses the baseline's OWN per-seed std on the dev seeds as the statistical prior of every
    comparison, a noisy estimate (it is drawn afresh per campaign, from ``--sigma-baseline``, the true per-seed std of the
    baseline). ``decide_outcome`` then applies the task's ``sigma_floor`` on top: ``s_eff = max(s, sigma_prior, sigma_floor)``.
  * The campaign tests ``k`` candidates against an incumbent. A candidate has a true improvement ``theta`` (0 for the null).
    Its paired per-seed differences (incumbent minus candidate, positive is better) are i.i.d. Normal(theta, sigma_d^2),
    drawn independently on the dev split (``n_dev`` seeds) and on the sealed split (``n_sealed`` seeds): the sealed split is
    a disjoint set of evaluation rows and new seeds, so its noise is independent of the dev noise.
  * A candidate that passes the dev gate is replicated on the sealed split, at most ``--max-replications`` times per run
    (the shipped budget cap; later dev passes are never queried). A run "falsely promotes" when a null passes both gates
    inside the cap.
  * Section 6 replaces the normal draws by null candidates that go through the law-discovery task's own data module, baseline model
    and the trusted grader's row selection and RMSE, in process: the model's check against something closer to the real pipeline.
  * Decision rules compared:
      raw         promote the best candidate whenever its mean dev improvement is above zero (what a sloppy loop does)
      t-test      classic one-sided 95 percent t-test on the dev seeds, no prior on the noise
      dev gate    Nightshift's dev margin alone: max(noise_margin, t * max(s, sigma_prior, sigma_floor) / sqrt(n)), paired
      nightshift  dev gate AND the sealed gate (two-sided 95 percent critical values, same margin formula)
  * The larger a candidate's paired noise s is compared with the floor and the prior, the closer its dev gate gets to a
    one-sided 5 percent t-test on ``n_dev`` seeds (the margin is then ``t * s / sqrt(n)`` with the candidate's own s, plus the
    ``noise_margin`` as a minimum effect). With a floor far below the real noise that is the whole gate; the generated
    document says what the sealed stage then buys.

    python3 scripts/null_benchmark.py                        # prints a markdown report (20,000 campaigns: about 20 minutes)
    python3 scripts/null_benchmark.py --campaigns 20000 --write docs/NULL_BENCHMARK.md
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nightshift.grading import T95, T975, decide_outcome  # noqa: E402
from nightshift.task import load_task  # noqa: E402
from nightshift.types import BudgetLimits, GradeResult  # noqa: E402

TASK_DIR = ROOT / "tasks" / "law-discovery"
DEFAULT_SEED = 7
DEFAULT_CAMPAIGNS = 20000
DEFAULT_K = 16
LONG_K = 48                       # the longer run the document also reports
CHECKPOINTS = (1, 4, DEFAULT_K, LONG_K)
SIGMA_BASELINE = 0.036            # true per-seed std of the unmodified baseline (0.0363 dev / 0.0380 sealed over 60 seeds)
NOISE_ROWS = (0.018, 0.036, 0.05, 0.07)      # paired noise sd of a candidate: small, equal to the prior, and larger than it
PREVIOUS_SIGMA_FLOOR = 0.02       # what the task shipped before the floor was calibrated (kept for the comparison table)
SHIPPED_PRIOR = 0.0100            # law-discovery's REALIZED prior: the unmodified baseline's std on dev seeds 0, 1, 2 (0.010012)
POWER_THETAS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.125, 0.15, 0.20)
BASELINE_SCORE = 1.6              # the unmodified law-discovery baseline's heldout RMSE (mean over seeds, 1.60 measured)
MEASURED_BEGIN = "<!-- measured-on-the-real-pipeline:begin -->"
MEASURED_END = "<!-- measured-on-the-real-pipeline:end -->"


# --------------------------------------------------------------------------------------------------
# decision rules (all go through the shipped decide_outcome)
# --------------------------------------------------------------------------------------------------
def _grade(scores: Sequence[float], seeds: Sequence[int], split: str) -> GradeResult:
    mean = statistics.fmean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return GradeResult(valid=True, score=mean, std=std, per_seed=list(scores), grader_hash="g", split=split, n_eval=100,
                       seeds=list(seeds))


class Rules:
    """The dev and sealed gates for one task configuration. The incumbent scores 1.0 on every seed (objective: minimize),
    a candidate is better by ``diffs[i]`` on seed i; only the candidate's grade is rebuilt per call."""

    def __init__(self, noise_margin: float, sigma_floor: float, dev_seeds: Sequence[int], sealed_seeds: Sequence[int],
                 *, shortcut: bool = True):
        self.task = SimpleNamespace(minimize=True, noise_margin=noise_margin, sigma_floor=sigma_floor, train_metric=None)
        self.dev_seeds, self.sealed_seeds = list(dev_seeds), list(sealed_seeds)
        self.shortcut = shortcut
        self._inc = {"dev": _grade([1.0] * len(dev_seeds), dev_seeds, "dev"),
                     "sealed": _grade([1.0] * len(sealed_seeds), sealed_seeds, "sealed")}

    def _improved(self, diffs: Sequence[float], seeds: Sequence[int], split: str, k_sigma, sigma_prior: float) -> bool:
        # Exact shortcut: decide_outcome returns "improved" only when delta > margin and margin >= noise_margin, so a mean
        # improvement at or below noise_margin can never pass and the call is skipped. ``shortcut=False`` (--exact) calls it
        # for every candidate; tests/test_null_benchmark.py checks that both give identical tallies.
        if self.shortcut and statistics.fmean(diffs) <= self.task.noise_margin:
            return False
        cand = _grade([1.0 - d for d in diffs], seeds, split)
        return decide_outcome(self.task, cand, self._inc[split], k_sigma=k_sigma, sigma_prior=sigma_prior).outcome == "improved"

    def dev_gate(self, diffs: Sequence[float], sigma_prior: float) -> bool:
        return self._improved(diffs, self.dev_seeds, "dev", None, sigma_prior)

    def sealed_gate(self, diffs: Sequence[float], sigma_prior: float) -> bool:
        return self._improved(diffs, self.sealed_seeds, "sealed", T975, sigma_prior)

    @staticmethod
    def t_test(diffs: Sequence[float]) -> bool:
        n = len(diffs)
        return statistics.fmean(diffs) > T95[n - 1] * statistics.stdev(diffs) / math.sqrt(n)


# --------------------------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------------------------
@dataclass
class Cell:
    """Tallies for one (theta, sigma_d) over all campaigns. ``*_run[k]`` counts campaigns whose first ``k`` candidates
    produced the event; ``cand_*`` count single candidates (every candidate of every campaign)."""

    campaigns: int
    ks: tuple[int, ...]
    raw_run: dict[int, int] = field(default_factory=dict)
    t_run: dict[int, int] = field(default_factory=dict)
    dev_run: dict[int, int] = field(default_factory=dict)
    ns_run: dict[int, int] = field(default_factory=dict)
    cand_n: int = 0
    cand_t: int = 0
    cand_dev: int = 0
    cand_sealed_given_dev: int = 0
    cand_both: int = 0
    winners: list[tuple[int, float, float]] = field(default_factory=list)   # (index, dev mean, sealed mean) of the first dev winner

    def rate(self, tally: dict[int, int], k: int) -> float:
        return tally[k] / self.campaigns


def simulate(campaigns: int, theta: float, sigma_d: float, rules: Rules, *, seed: int, ks: Sequence[int] = CHECKPOINTS,
             sigma_baseline: float = SIGMA_BASELINE, max_replications: int = 6, fixed_prior: Optional[float] = None) -> Cell:
    """Run ``campaigns`` campaigns of ``max(ks)`` candidates and tally the events at every checkpoint in ``ks`` (the first
    ``k`` candidates of a campaign are a campaign of ``k`` candidates). Deterministic for a given argument list.

    ``fixed_prior`` replaces the per-campaign random ``sigma_prior`` by a constant: the shipped task has fixed seeds and a fixed
    template, so its real prior is one number (``SHIPPED_PRIOR``), not a random variable."""
    ks = tuple(sorted(set(ks)))
    cell = Cell(campaigns=campaigns, ks=ks)
    for tally in (cell.raw_run, cell.t_run, cell.dev_run, cell.ns_run):
        tally.update({k: 0 for k in ks})
    rng = random.Random(f"{seed}:{theta}:{sigma_d}:{sigma_baseline}:{rules.task.sigma_floor}:{rules.task.noise_margin}:{fixed_prior}")
    n_dev, n_sealed = len(rules.dev_seeds), len(rules.sealed_seeds)
    for _ in range(campaigns):
        prior = fixed_prior if fixed_prior is not None else \
            statistics.stdev([rng.gauss(0.0, sigma_baseline) for _ in range(n_dev)])   # this campaign's sigma_prior
        best_mean, any_t, any_dev, promoted, queries = -math.inf, False, False, False, 0
        first_winner: Optional[tuple[int, float, float]] = None
        for i in range(1, max(ks) + 1):
            dev = [rng.gauss(theta, sigma_d) for _ in range(n_dev)]
            mean = statistics.fmean(dev)
            best_mean = max(best_mean, mean)
            cell.cand_n += 1
            if rules.t_test(dev):
                any_t = True
                cell.cand_t += 1
            if rules.dev_gate(dev, prior):
                any_dev = True
                cell.cand_dev += 1
                sealed = [rng.gauss(theta, sigma_d) for _ in range(n_sealed)]
                passed = rules.sealed_gate(sealed, prior)
                cell.cand_sealed_given_dev += passed
                cell.cand_both += passed
                if first_winner is None:
                    first_winner = (i, mean, statistics.fmean(sealed))
                if queries < max_replications:           # the budget gate refuses a query beyond the cap
                    queries += 1
                    promoted = promoted or passed
            if i in cell.raw_run:
                cell.raw_run[i] += best_mean > 0
                cell.t_run[i] += any_t
                cell.dev_run[i] += any_dev
                cell.ns_run[i] += promoted
        if first_winner is not None:
            cell.winners.append(first_winner)
    return cell


# --------------------------------------------------------------------------------------------------
# nulls drawn from the real law-discovery grader (in process)
# --------------------------------------------------------------------------------------------------
REAL_NULL_FRACTIONS = (0.3, 0.5, 0.7)      # a null fits the baseline's straight line on this share of the seed's training sample


def _load_by_path(path: Path, name: str):
    """Import ``path`` as module ``name`` without writing bytecode next to it (the task files must stay clean)."""
    keep = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = keep


class LawReplica:
    """The law-discovery pipeline run in process: the task's own ``data.load_train``, its baseline ``model.fit_predict`` and the
    trusted grader's own row selection and RMSE, with no sandbox (so it is cheap enough for tens of thousands of candidates).

    A null candidate is the baseline's straight line fitted on a random ``fraction`` of the seed's training sample. The subset is
    drawn from ``random.Random(f"{salt}|{train!r}")``: seeded by the salt (one per candidate) and by the training data, so it
    differs from seed to seed, exactly as a candidate that bags its training rows inside ``fit_predict`` would behave.
    ``tests/test_null_benchmark.py`` checks the scores against a real sandboxed run of such a candidate."""

    def __init__(self, dev_seeds: Sequence[int], sealed_seeds: Sequence[int], task_dir: Path = TASK_DIR):
        self.data = _load_by_path(task_dir / "template" / "data.py", "null_benchmark_ld_data")
        self.model = _load_by_path(task_dir / "template" / "model.py", "null_benchmark_ld_model")
        self.grade = _load_by_path(task_dir / "trusted" / "grade.py", "null_benchmark_ld_grade")
        labels = self.grade.read_labels()
        times = dict(self.data.load_eval_inputs())
        self.plan = []                       # (split, seed, train rows, query times, labels) per evaluated seed, dev first
        for split, seeds in (("dev", dev_seeds), ("sealed", sealed_seeds)):
            for seed in seeds:
                ids = self.grade.select_eval_ids(labels, split, seed)
                train = [(t, y) for _, t, y in self.data.load_train(seed)]
                self.plan.append((split, seed, train, [times[i] for i in ids], [labels[i][1] for i in ids]))
        self.n_dev = len(list(dev_seeds))
        self.baseline = [self._rmse(self.model.fit_predict(train, q), y) for _, _, train, q, y in self.plan]
        self.prior = statistics.stdev(self.baseline[:self.n_dev])        # the engine's sigma_prior for this task

    def _rmse(self, preds: Sequence[float], truth: Sequence[float]) -> float:
        return self.grade.rmse(list(zip(preds, truth)))

    def bag(self, fraction: float, salt: str) -> list[float]:
        """The null candidate's score on every planned seed (dev seeds first, then sealed seeds)."""
        out = []
        for _, _, train, q, y in self.plan:
            sub = random.Random(f"{salt}|{train!r}").sample(train, k=max(2, int(fraction * len(train))))
            out.append(self._rmse([float(p) for p in self.model.fit_predict(sub, q)], y))
        return out

    def diffs(self, scores: Sequence[float]) -> tuple[list[float], list[float]]:
        """Paired improvements (baseline minus candidate, positive is better) on the dev seeds and on the sealed seeds."""
        d = [b - c for b, c in zip(self.baseline, scores)]
        return d[:self.n_dev], d[self.n_dev:]


@dataclass
class RealNullCell:
    """One null family under one floor: per-candidate and per-run tallies over the candidates in ``decisions``."""

    n: int
    dev: int
    both: int
    runs: dict[int, tuple[int, int, int]]     # k -> (runs with a promoted null, runs with a dev winner, number of runs)


def real_null_draw(replica: LawReplica, fraction: float, n: int, seed: int, salt: str = "") -> list[tuple[list[float], list[float]]]:
    """``n`` null candidates of one family: each is its paired improvements on the dev seeds and on the sealed seeds."""
    return [replica.diffs(replica.bag(fraction, f"{salt}{seed}:{fraction}:{c}")) for c in range(n)]


def real_null_summary(draws: Sequence[tuple[list[float], list[float]]]) -> tuple[float, float, float]:
    """``(sd, mean improvement, skewness)`` of a family's paired improvements pooled over every candidate and every seed. The
    sd is the model's sigma_d for the family (a mean of 3-seed sample stds would understate it by about 11%); a negative mean
    means the null is slightly worse than the baseline."""
    xs = [d for dev, sealed in draws for d in dev + sealed]
    mean, sd = statistics.fmean(xs), statistics.stdev(xs)
    return sd, mean, statistics.fmean(((x - mean) / sd) ** 3 for x in xs)


def real_null_decisions(draws: Sequence[tuple[list[float], list[float]]], rules: Rules, prior: float) -> list[tuple[bool, bool]]:
    """Decide every candidate with ``rules`` through ``decide_outcome``: ``(dev gate passed, sealed gate passed)``; the sealed
    gate is only evaluated for a candidate that passed dev."""
    out = []
    for dev, sealed in draws:
        passed = rules.dev_gate(dev, prior)
        out.append((passed, passed and rules.sealed_gate(sealed, prior)))
    return out


def real_null_cell(decisions: Sequence[tuple[bool, bool]], ks: Sequence[int], cap: int) -> RealNullCell:
    """Tallies of ``decisions`` (in candidate order): consecutive groups of ``k`` candidates are runs, each allowed ``cap``
    sealed queries (the first ``cap`` dev winners; a later one is never replicated)."""
    runs = {}
    for k in ks:
        promoted = devwin = 0
        n_runs = len(decisions) // k
        for r in range(n_runs):
            queries, hit, win = 0, False, False
            for dev, sealed in decisions[r * k:(r + 1) * k]:
                if dev:
                    win = True
                    if queries < cap:
                        queries += 1
                        hit = hit or sealed
            promoted, devwin = promoted + hit, devwin + win
        runs[k] = (promoted, devwin, n_runs)
    return RealNullCell(n=len(decisions), dev=sum(d for d, _ in decisions), both=sum(b for _, b in decisions), runs=runs)


# --------------------------------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{100 * x:.2f}%" if x < 0.1 else f"{100 * x:.1f}%"


def pct_n(count: int, total: int) -> str:
    """A rate; an event that never happened says how many trials it was out of, since 0.00% alone claims too much."""
    return f"0 of {total:,}" if count == 0 else pct(count / total)


def theta_at_power(power: dict, sd: float, level: float) -> Optional[float]:
    """The improvement at which Nightshift promotes with probability ``level`` for noise ``sd``, linearly interpolated over
    the simulated grid (None when the grid never reaches ``level``)."""
    pts = [(theta, c.rate(c.ns_run, 1)) for (theta, s_d), c in sorted(power.items()) if s_d == sd]
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if p0 < level <= p1:
            return t0 + (t1 - t0) * (level - p0) / (p1 - p0)
    return None


def defaults() -> argparse.Namespace:
    """The shipped constants the script models, read from the task file and the budget defaults."""
    task = load_task(TASK_DIR)
    return argparse.Namespace(noise_margin=task.noise_margin, sigma_floor=task.sigma_floor, dev_seeds=list(task.seeds),
                              sealed_seeds=list(task.replication_seeds), max_replications=BudgetLimits().max_replications)


MEASURED_SECTION = f"""{MEASURED_BEGIN}
## Measured on the real pipeline (independent audit, not produced by this script)

Everything above is a model. This section is a static record of what an independent statistics audit measured on the real
pipeline (the real sandbox, the real trusted grader, the engine's own decision code) for the law-discovery task. It was
produced with a throwaway harness that is **not part of this repository**, so the script cannot reproduce these numbers and the
tests do not check them. The audit ran **before** `sigma_floor` was calibrated: its configuration was `noise_margin` 0.035 and
`sigma_floor` 0.02, with the engine's `sigma_prior` (the baseline's own per-seed std on the 3 dev seeds) equal to 0.0100 on the
real task: below the floor, so the prior was inert, and the floor was itself far below the true baseline std (about 0.036 to
0.040). For a candidate with small paired noise the dev margin therefore collapsed to `noise_margin` 0.035; for a noisy one it
was the candidate's own `t * s / sqrt(3)`.

**The nulls.** Candidate families that do not truly improve on the baseline: bagged fits (bag30, bag50, bag70) and a weighted
fit (wls1); the names are the audit's labels. Their training subset or weights are seeded by the training data, so they differ
from seed to seed, and their effect on the population loss is within about +-0.005 of zero. That makes them valid nulls (no
real improvement) whose paired per-seed noise is nevertheless large: standard deviation 0.03 to 0.064, comparable to or larger
than the baseline's own 0.036.

**What was measured (old configuration, `sigma_floor` 0.02).**

| measurement | value |
|---|---|
| nulls run through the real sandbox | 190 |
| dev gate passed | 4.2% |
| of those, passed the sealed gate | 1.1% |
| nulls promoted | 0 |
| in-process replica of the pipeline, nulls | 11,000, with 0 mismatches against the sandbox's decisions |
| bag50, per candidate (of 3,000): dev gate / sealed gate / both | 5.2% / 2.4% / 7 of 3,000 |
| end-to-end false promotion, per candidate | about 0.1% to 0.3% |
| end-to-end false promotion, per 16-experiment run | wls1 0.5%, bag70 1.6%, bag50 3.7%, bag30 4.8% |
| end-to-end false promotion, per 48-experiment run | 5% to 12% |
| baseline `sigma_prior` (3 dev seeds), 5% / 50% / 95% quantiles over resampled seed sets | 0.0088 / 0.0304 / 0.0639 |

**Power (old configuration).** Promoted 100% of the time: the quadratic model (+0.73 RMSE improvement) and the damped
oscillator (+1.56). For graded effects: 0.029 RMSE improvement promoted 0%, 0.039 30%, 0.049 46%, 0.061 64%, 0.074 75%, 0.124 97%.

**How it compares with the model.** In section 5 the model's rows for the previous configuration give the same order of
magnitude for the dev gate (a few percent per candidate) but a lower "both gates" rate than the audit measured for bag50
(7 of 3,000). The real nulls are not independent normal draws, so for that configuration trust the audit's figures, not the
model's.

**What this means.** For the audit's noisy nulls the old dev gate was essentially a one-sided 5% t-test on 3 seeds, because the
prior (0.0100) and the floor (0.02) were both below their paired noise (0.03 to 0.064). Section 5 above shows in the model what the
calibrated floor changes. The shipped configuration now sets `sigma_floor` to the baseline's multi-seed std (0.036), which makes
the prior real; the audit's numbers describe the configuration it measured and are **not** a claim about the shipped one. The
power figures were measured with the old margins and are lower under the new ones (the dev margin for a candidate with small
paired noise rises from 0.035 to about 0.061, the sealed margin from 0.035 to about 0.045); the power table above is the model's
estimate of that shift. The real-pipeline false-promotion figures under the calibrated floor have **not** been re-measured by the
audit; section 6 above is this script's own in-process estimate with the real grader, not a re-measurement of the sandbox.
{MEASURED_END}
"""


def _rules(args: argparse.Namespace, sigma_floor: Optional[float] = None) -> Rules:
    return Rules(args.noise_margin, args.sigma_floor if sigma_floor is None else sigma_floor, args.dev_seeds, args.sealed_seeds,
                 shortcut=not args.exact)


def _sim(args: argparse.Namespace, theta: float, sigma_d: float, ks: Sequence[int] = CHECKPOINTS,
         sigma_floor: Optional[float] = None, fixed_prior: Optional[float] = None) -> Cell:
    return simulate(args.campaigns, theta, sigma_d, _rules(args, sigma_floor), seed=args.seed, ks=ks,
                    sigma_baseline=args.sigma_baseline, max_replications=args.max_replications, fixed_prior=fixed_prior)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a rate (well behaved at 0 successes and for few trials); (0, 1) for no trials."""
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def comparison_verdict(promoted: int, n_runs: int, model_rate: float, k: int) -> str:
    """One sentence comparing the model's rate with the real-grader nulls' rate (``promoted`` of ``n_runs`` runs of ``k``
    experiments): the model is called wrong only when its rate is outside the 95% Wilson interval of the real one."""
    if not n_runs:
        return "There are too few candidates for a comparison"
    low, high = wilson_interval(promoted, n_runs)
    if model_rate < low:
        return (f"The normal-noise model understates the real rate of the {k}-experiment runs (outside the 95% interval "
                f"{pct(low)} to {pct(high)}): treat the normal-noise rows of sections 1 and 2 as optimistic for such candidates")
    if model_rate > high:
        return (f"The normal-noise model overstates the real rate of the {k}-experiment runs (outside the 95% interval "
                f"{pct(low)} to {pct(high)})")
    return (f"The model and the real-grader nulls agree within sampling error for the {k}-experiment runs (95% interval "
            f"of the real rate {pct(low)} to {pct(high)})")


def _runs_cell(c: RealNullCell, k: int) -> str:
    """Runs of ``k`` candidates that ended with a promoted null: 'x of N' with the rate once x is not zero, or why there is no
    figure (fewer candidates than one run needs)."""
    promoted, _, runs = c.runs[k]
    if not runs:
        return f"n/a (fewer than {k} candidates)"
    return f"{pct(promoted / runs)} ({promoted} of {runs:,})" if promoted else f"0 of {runs:,}"


def report(args: argparse.Namespace) -> str:
    n_dev, n_sealed = len(args.dev_seeds), len(args.sealed_seeds)
    n = args.campaigns
    k_ref, k_long = args.k, 3 * args.k
    ks = (1, 4, k_ref, k_long)
    cap = args.max_replications
    dev_margin = max(args.noise_margin, T95[n_dev - 1] * args.sigma_floor / math.sqrt(n_dev))
    sealed_margin = max(args.noise_margin, T975[n_sealed - 1] * args.sigma_floor / math.sqrt(n_sealed))
    lines = [
        "# Null-hypothesis benchmark: how often does an auto-researcher fool itself?",
        "",
        "A Monte Carlo over simulated research campaigns using the shipped decision code "
        "(`nightshift.grading.decide_outcome`, `T95`, `T975`). It checks the **decision rule** under a stated noise model; it says "
        "nothing about any language model. It models the configuration that ships (`tasks/law-discovery/task.toml`), including the "
        "noisy estimate of the baseline's std that the engine uses as its prior. Section 6 re-checks the model with null candidates "
        "drawn through the task's real data module and grader, and the section after it is a separate record of what an "
        "independent audit measured on the real pipeline. Regenerate with "
        f"`python3 scripts/null_benchmark.py --campaigns {n} --seed {args.seed} --write docs/NULL_BENCHMARK.md` "
        "(about 20 minutes on one core).",
        "",
        "## Setup",
        "",
        f"* {n:,} campaigns per cell, up to {k_long} candidates per campaign (every table reads the first k of them), "
        f"random seed {args.seed}.",
        f"* Task constants (law-discovery, read from `task.toml`): noise_margin {args.noise_margin}, sigma_floor {args.sigma_floor}, "
        f"{n_dev} dev seeds, {n_sealed} sealed seeds. Budget: max_replications {cap} (sealed queries per run).",
        f"* The prior is not a constant. Each campaign first measures the baseline on {n_dev} dev seeds; the sample std of those "
        f"scores is that campaign's `sigma_prior` (true baseline per-seed std {args.sigma_baseline}), and `decide_outcome` takes "
        f"`s_eff = max(paired s, sigma_prior, sigma_floor)`. The sample std of {n_dev} scores is a very noisy estimate, which is "
        "why the floor matters.",
        f"* Paired per-seed differences are i.i.d. Normal(theta, sigma_d^2), independent between the dev split ({n_dev} seeds) and "
        f"the sealed split ({n_sealed} seeds). Row families: small noise (sigma_d 0.018, 0.036) and noise **larger than the prior** "
        "(0.05, 0.07).",
        f"* Margins at the shipped floor: dev gate `max({args.noise_margin}, {T95[n_dev - 1]} * s_eff / sqrt({n_dev}))` = "
        f"{dev_margin:.4f} when s_eff is the floor; sealed gate `max({args.noise_margin}, {T975[n_sealed - 1]} * s_eff / "
        f"sqrt({n_sealed}))` = {sealed_margin:.4f}. When the sample std s of a candidate's paired differences exceeds the "
        f"floor and the prior, s_eff = s and the dev gate is a one-sided 5% t-test on {n_dev} seeds (with the noise_margin as an "
        "additional minimum effect).",
        "* **raw**: promote the best candidate whenever its mean dev improvement is above zero. **t-test**: one-sided 95% t-test "
        "on the dev seeds, no prior on the noise. **dev gate**: Nightshift's dev margin alone. **Nightshift**: dev gate and then "
        f"the sealed gate (two-sided 95% critical values), with at most {cap} sealed queries per run.",
        "",
    ]

    # ---- 1. headline sigma_d --------------------------------------------------------------------
    sd0 = args.sigma_d
    cell = _sim(args, 0.0, sd0, ks)
    lines += [f"## 1. False wins per run: no candidate has any real effect (theta = 0, sigma_d {sd0})", "",
              "Share of runs (campaigns) that end with a claimed improvement although nothing improved. A run tests the stated "
              "number of candidates.", "",
              "| candidates per run | raw | t-test | dev gate | **Nightshift** (cap " + str(cap) + " sealed queries) |",
              "|---|---|---|---|---|"]
    for k in ks:
        lines.append(f"| {k} | {pct(cell.rate(cell.raw_run, k))} | {pct(cell.rate(cell.t_run, k))} | "
                     f"{pct(cell.rate(cell.dev_run, k))} | **{pct_n(cell.ns_run[k], n)}** |")

    # ---- 2. noise rows ---------------------------------------------------------------------------
    lines += ["", "## 2. False promotion by candidate noise (theta = 0)", "",
              "The same simulation for each row family of paired noise. Per candidate: the share that pass the dev gate, the share "
              "of those that then also pass the sealed gate, and the share that pass both. Per run: the probability that at least "
              f"one null is promoted within {k_ref} and {k_long} experiments (the sealed cap of {cap} applied).", "",
              f"| paired noise sigma_d | per candidate: dev gate | sealed, given dev | both | t-test per candidate | per run, {k_ref} "
              f"experiments: dev gate | **Nightshift** | per run, {k_long} experiments: dev gate | **Nightshift** |",
              "|---|---|---|---|---|---|---|---|---|"]
    cells: dict[float, Cell] = {}
    for sd in NOISE_ROWS:
        c = cells[sd] = _sim(args, 0.0, sd, (k_ref, k_long)) if sd != sd0 else cell
        both = c.cand_both
        given = c.cand_sealed_given_dev / c.cand_dev if c.cand_dev else 0.0
        lines.append(f"| {sd:.3f} | {pct(c.cand_dev / c.cand_n)} | {pct(given)} | {pct_n(both, c.cand_n)} | "
                     f"{pct(c.cand_t / c.cand_n)} | {pct(c.rate(c.dev_run, k_ref))} | **{pct_n(c.ns_run[k_ref], n)}** | "
                     f"{pct(c.rate(c.dev_run, k_long))} | **{pct_n(c.ns_run[k_long], n)}** |")
    lines += ["",
              f"The larger a candidate's paired noise is compared with the floor ({args.sigma_floor}) and the prior, the closer its "
              f"dev gate gets to a plain one-sided 5% t-test on {n_dev} seeds: the margin is then `t * s / sqrt({n_dev})` with the "
              "candidate's own s instead of the floor, and the `noise_margin` only adds a minimum effect size. The t-test column "
              "is that limit (5% per candidate). The floor keeps moderately noisy candidates well below it (the dev-gate "
              "column), the gap closes as the noise grows, and a run that tries many experiments gives the dev gate many chances "
              "(the per-run dev-gate columns). What the sealed stage then buys: "
              f"a fresh draw of independent noise on {n_sealed} new seeds against a two-sided 95% threshold, so a null that "
              "got through dev passes it only a few percent of the time or less (column 'sealed, given dev'). The product of the two "
              "gates, and the cap on sealed queries, is the protection, not the dev gate: the per-run figures above are what a "
              f"long run costs. The cap of {cap} sealed queries bounds how many nulls the sealed stage can ever wave through.", ""]

    # ---- 3. winner's curse ---------------------------------------------------------------------
    lines += ["## 3. The winner's curse: what the dev score of a winner is worth", "",
              f"Among runs where some null candidate passed the dev gate within {k_ref} candidates (theta = 0): the mean dev "
              "improvement the researcher would have reported for the first such candidate, versus the same candidate's mean "
              "improvement on fresh sealed data.", "",
              "| paired noise sigma_d | runs with a dev-gate winner | mean reported dev improvement | mean sealed improvement "
              "(true value 0) |", "|---|---|---|---|"]
    for sd in NOISE_ROWS:
        w = [x for x in cells[sd].winners if x[0] <= k_ref]
        if w:
            lines.append(f"| {sd:.3f} | {len(w):,} of {n:,} | {statistics.fmean(x[1] for x in w):+.4f} | "
                         f"{statistics.fmean(x[2] for x in w):+.4f} |")
        else:
            lines.append(f"| {sd:.3f} | 0 of {n:,} | n/a | n/a |")

    # ---- 4. the price: sensitivity ------------------------------------------------------------
    lines += ["", "## 4. The price: sensitivity", "",
              "Probability that a single candidate with a real improvement `theta` (in the units of the metric, RMSE for "
              "law-discovery) is promoted by Nightshift (dev gate and sealed gate). Trust costs sensitivity, and the table says "
              "how much. The baseline's RMSE is about 1.6, so 0.05 is about 3% of it.", "",
              "| true improvement theta | " + " | ".join(f"sigma_d {sd:.3f}" for sd in NOISE_ROWS) + " |",
              "|---|" + "---|" * len(NOISE_ROWS)]
    power: dict[tuple[float, float], Cell] = {}
    for theta in (0.0,) + POWER_THETAS:
        row = []
        for sd in NOISE_ROWS:
            c = power[(theta, sd)] = _sim(args, theta, sd, (1,))
            row.append(pct_n(c.ns_run[1], n) if theta == 0.0 else pct(c.rate(c.ns_run, 1)))
        lines.append(f"| {theta:.3f} | " + " | ".join(row) + " |")
    lines += ["", f"The smallest improvement that can pass at all is the larger of the two margins: {dev_margin:.4f} on dev and "
              f"{sealed_margin:.4f} on the sealed split (candidates with small paired noise). The improvement at which half of such "
              "candidates are promoted (50% power, interpolated in the table), the minimum detectable effect:", ""]
    for sd in NOISE_ROWS:
        t50 = theta_at_power(power, sd, 0.5)
        lines.append(f"* sigma_d {sd:.3f}: " + (f"about {t50:.3f}, {100 * t50 / BASELINE_SCORE:.1f}% of the baseline RMSE ({BASELINE_SCORE})"
                                               if t50 is not None else "not reached in the table"))
    lines += ["", f"A smaller real effect cannot be told from noise with {n_dev} dev seeds and is mostly not promoted: that is the "
              "intended trade, and a ceiling as well: once the incumbent is within the margin of the best achievable score, "
              "nothing further can be promoted.", ""]

    # ---- 5. what the calibrated floor changed -----------------------------------------------------
    old, prior = args.compare_floor, args.shipped_prior
    configs = [(f"previous: floor {old}, prior {prior:.4f} (the audited configuration)", old, prior),
               (f"shipped: floor {args.sigma_floor}, prior {prior:.4f} (the task's realized prior)", None, prior),
               (f"floor {args.sigma_floor}, prior drawn per campaign (sections 1 to 4)", None, None)]
    lines += ["## 5. What calibrating `sigma_floor` changed", "",
              f"The shipped task has fixed seeds and a fixed template, so its REAL prior is one number: the unmodified baseline's "
              f"std on dev seeds 0, 1, 2, which is {prior:.4f}. Before the floor was calibrated (floor {old}) that prior and the "
              "floor were both below the real paired noise of many candidates, so their margin was the candidate's own "
              f"`t * s / sqrt(n)` (at least the `noise_margin`): a bare t-test. With the floor at {args.sigma_floor} the margin uses "
              "`max(s, floor)`. The sections above draw the prior per campaign instead (a task whose baseline seeds are not "
              "fixed); the three configurations compared here:", "",
              f"| paired noise sigma_d | configuration | per candidate: dev gate | per candidate: both gates | per run, {k_ref} "
              f"experiments | per run, {k_long} experiments |", "|---|---|---|---|---|---|"]
    for sd in (0.036, 0.05, 0.07):
        for label, floor, fixed in configs:
            c = cells[sd] if (floor is None and fixed is None) else _sim(args, 0.0, sd, (k_ref, k_long), sigma_floor=floor,
                                                                          fixed_prior=fixed)
            lines.append(f"| {sd:.3f} | {label} | {pct(c.cand_dev / c.cand_n)} | {pct_n(c.cand_both, c.cand_n)} | "
                         f"{pct_n(c.ns_run[k_ref], n)} | {pct_n(c.ns_run[k_long], n)} |")
    lines += ["", f"| true improvement theta (sigma_d 0.036) | promoted: previous (floor {old}) | promoted: shipped (floor "
              f"{args.sigma_floor}, realized prior) | promoted: floor {args.sigma_floor}, drawn prior |", "|---|---|---|---|"]
    for theta in (0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.125):
        c_old = _sim(args, theta, 0.036, (1,), sigma_floor=old, fixed_prior=prior)
        c_ship = _sim(args, theta, 0.036, (1,), fixed_prior=prior)
        c_new = power[(theta, 0.036)]
        lines.append(f"| {theta:.3f} | {pct(c_old.rate(c_old.ns_run, 1))} | {pct(c_ship.rate(c_ship.ns_run, 1))} | "
                     f"{pct(c_new.rate(c_new.ns_run, 1))} |")
    lines += ["", "The floor trades sensitivity for protection: the minimum effect that can be promoted rises, and the false "
              "promotion rate of noisy candidates falls. The real-pipeline numbers in the next section were measured under the "
              "previous configuration.", ""]

    # ---- 6. nulls drawn from the real grader ----------------------------------------------------
    replica = LawReplica(args.dev_seeds, args.sealed_seeds)
    n_real, shipped_rules, previous_rules = n, _rules(args), _rules(args, args.compare_floor)
    lines += ["## 6. Nulls drawn from the real grader (law-discovery, in process)", "",
              "Sections 1 to 5 draw independent normal noise. This section checks the model against candidates that go through the "
              "task's own data module, its baseline model and the trusted grader's own row selection and RMSE, in process (no "
              "sandbox). A null here is the baseline's straight line fitted on a random share of the seed's training sample, "
              "with a fresh random subset for every candidate and every seed. It does not improve the population loss (the mean "
              "improvement column shows it is slightly worse), but its per-seed score moves with the subset, so its paired "
              "differences are noisy, and they are not symmetric (skewness column): with only 3 dev and 5 sealed seeds a skewed "
              "sample makes one tail of a t-test heavier than the normal model assumes. The dev and the sealed draws of one "
              f"candidate are independent, as in the model. {n_real:,} candidates per family, cut into runs of {k_ref} and {k_long} candidates with the cap of {cap} sealed "
              f"queries; the engine's prior is the real baseline's std on the dev seeds, {replica.prior:.4f}. "
              "The families are this script's own re-creation of the audit's kind of null, not the audit's candidates. "
              "'model' rows are section 2's model at the same paired noise sd (independent normal draws, same prior).", "",
              f"| null family | paired noise sd | mean improvement | skewness | configuration | per candidate: dev gate | both gates | "
              f"per run, {k_ref} experiments | per run, {k_long} experiments |", "|---|---|---|---|---|---|---|---|---|"]
    real_rows = {}
    for frac in REAL_NULL_FRACTIONS:
        draws = real_null_draw(replica, frac, n_real, args.seed)
        s_mean, gain, skew = real_null_summary(draws)
        s_model = round(s_mean, 3)
        configs = [(f"previous floor {args.compare_floor}, real", real_null_cell(real_null_decisions(draws, previous_rules, replica.prior), (k_ref, k_long), cap)),
                   (f"shipped floor {args.sigma_floor}, real", real_null_cell(real_null_decisions(draws, shipped_rules, replica.prior), (k_ref, k_long), cap))]
        real_rows[frac] = (s_mean, configs[1][1])
        for label, c in configs:
            lines.append(f"| bag {round(100 * frac)}% | {s_mean:.3f} | {gain:+.4f} | {skew:+.2f} | {label} | {pct(c.dev / c.n)} | {pct_n(c.both, c.n)} | "
                         f"{_runs_cell(c, k_ref)} | {_runs_cell(c, k_long)} |")
        m = _sim(args, 0.0, s_model, (k_ref, k_long), fixed_prior=replica.prior)
        real_rows[frac] += (m,)
        lines.append(f"| bag {round(100 * frac)}% | {s_mean:.3f} | n/a | n/a | shipped floor {args.sigma_floor}, model | {pct(m.cand_dev / m.cand_n)} | "
                     f"{pct_n(m.cand_both, m.cand_n)} | {pct_n(m.ns_run[k_ref], n)} | {pct_n(m.ns_run[k_long], n)} |")
    noisiest = max(REAL_NULL_FRACTIONS, key=lambda f: real_rows[f][0])
    s_noisy, c_noisy, m_noisy = real_rows[noisiest]
    promoted, _, n_runs = c_noisy.runs[k_long]
    verdict = comparison_verdict(promoted, n_runs, m_noisy.rate(m_noisy.ns_run, k_long), k_long)
    lines += ["", f"In the noisiest family (bag {round(100 * noisiest)}%, paired noise sd {s_noisy:.3f}, above the floor {args.sigma_floor}) "
              f"the runs that end with a promoted null at the shipped floor are {_runs_cell(c_noisy, k_ref)} of the {k_ref}-experiment runs "
              f"and {_runs_cell(c_noisy, k_long)} of the {k_long}-experiment runs; the model at the same noise gives "
              f"{pct_n(m_noisy.ns_run[k_ref], n)} and {pct_n(m_noisy.ns_run[k_long], n)}. {verdict}. A floor can only help when the "
              "candidate's own sample std of its paired differences (from only 3 seeds) is below it, and the noisier the family the "
              "less often that happens: compare the previous and the shipped rows of each family. For the noisiest family the dev "
              f"gate is nearly the plain {n_dev}-seed t-test, and what protects a run is the sealed stage and its cap. This is a model of "
              "bagged straight-line fits, not of an arbitrary researcher.", ""]

    lines += [MEASURED_SECTION,
              "## Reading it", "",
              "* A loop that promotes on the raw dev score is wrong almost every time it runs a few candidates.",
              "* A classic t-test on three seeds still lets a null through at the textbook rate, and that rate compounds over "
              "candidates (the multiple-comparison problem every long auto-research session has).",
              "* The dev gate alone is not a strong test for noisy candidates: the noisier a candidate's paired differences are "
              "compared with the floor, the closer the gate is to a 5% t-test on three seeds, and over many experiments the "
              "chance that some null reaches the sealed stage grows quickly (the per-run dev-gate columns). The sealed stage, with a hard cap on queries, is what keeps the per-run rate "
              "in the low percent for the model's noisiest rows.",
              "* Section 6 checks the normal-noise rows against nulls drawn through the real grader. Read it for what a very noisy family "
              "costs at the shipped floor: the floor cannot help a candidate whose own paired noise is above it.",
              "* The per-run false-promotion rate grows with the number of experiments (the tables give 16 and 48) and is not "
              "zero for noisy candidates. A longer run should be read with that in mind; the report states the expected number "
              "and an adaptive worst case for the sealed stage.",
              "* Real runs differ from the model: seeds that share a training sample have correlated noise, a language model does "
              "not draw candidates independently, and a supervisor that sees dev scores can tune against them. Treat this as a check "
              "of the decision rule, not a forecast.", ""]
    return "\n".join(lines)


def _int_at_least(minimum: int):
    """argparse type: an integer >= ``minimum`` (campaigns and candidates must be >= 1: 0 would divide by zero in every table)."""
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value
    return parse


_positive_int = _int_at_least(1)


def _non_negative_float(text: str) -> float:
    """argparse type: a finite float >= 0 (a noise level, a floor or a margin)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"must be a finite number >= 0, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    d = defaults()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaigns", type=_positive_int, default=DEFAULT_CAMPAIGNS)
    ap.add_argument("--k", type=_positive_int, default=DEFAULT_K, help="experiments per run in the main tables (the long run is 3x)")
    ap.add_argument("--sigma-d", type=_non_negative_float, default=0.036, help="paired per-seed noise sd of the headline table")
    ap.add_argument("--sigma-baseline", type=_non_negative_float, default=SIGMA_BASELINE,
                    help="true per-seed std of the unmodified baseline (the campaign's sigma_prior is the sample std of "
                         "its dev scores)")
    ap.add_argument("--sigma-floor", type=_non_negative_float, default=d.sigma_floor, help="task.toml sigma_floor (default: read from it)")
    ap.add_argument("--noise-margin", type=_non_negative_float, default=d.noise_margin, help="task.toml noise_margin (default: read from it)")
    ap.add_argument("--compare-floor", type=_non_negative_float, default=PREVIOUS_SIGMA_FLOOR,
                    help="the previous shipped sigma_floor, for the comparison table")
    ap.add_argument("--shipped-prior", type=_non_negative_float, default=SHIPPED_PRIOR,
                    help="law-discovery's realized sigma_prior (baseline std on dev seeds 0, 1, 2), for section 5")
    ap.add_argument("--max-replications", type=_int_at_least(0), default=d.max_replications,
                    help="sealed queries per run (BudgetLimits)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--exact", action="store_true",
                    help="call decide_outcome for every candidate (slower; the default skips candidates whose mean improvement "
                         "cannot exceed noise_margin, which gives identical results)")
    ap.add_argument("--write", type=Path, default=None)
    ap.set_defaults(dev_seeds=d.dev_seeds, sealed_seeds=d.sealed_seeds)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    text = report(args)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.write}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
