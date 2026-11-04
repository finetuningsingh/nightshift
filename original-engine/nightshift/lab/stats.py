"""Statistics for the Glimmer three-arm harness: spread, the screen-then-confirm keep rule, the final comparison, diagnostics.

Everything here is pure and deterministic (no clock, no files, no randomness except the seeded bootstrap) and depends on
``nightshift.lab.contracts`` only. The module itself is a ``DecisionRule`` (``trial_spread``, ``screen``, ``decide``), so the CLI passes
``stats`` straight in as ``LabWorld.decision_rule``.

The method (docs/harness-design.md, "Statistics"):

* The unit of independent evidence is the TRIAL. A trial's number is the mean over generation seeds of the headline metric on the
  selection split (``TrialScore.score``). Resampling prompts is a diagnostic only (``paired_bootstrap_prompts``, ``mcnemar``) and never
  decides a keep.
* ``trial_spread``: cells are assignments with at least two trials; ``df = sum(n_a - 1)``; ``sigma^2`` is the pooled within-cell variance.
* ``screen``: ``delta_k = (1/A) * sum_a (m_k,a - m_i,a)`` with no threshold. ``select_screen_winner`` picks the one candidate that
  advances (largest delta > 0, ties to the smaller id).
* ``decide``: the confirmation of that one candidate with FRESH trials on both sides.
  ``margin = t(alpha, df) * sigma_used * se_factor + min_gain`` and accept iff ``delta > margin`` where
  ``se_factor = (1/A) * sqrt(sum_a (1/n_c,a + 1/n_i,a))`` and ``sigma_used^2`` pools the frozen v0 variance (``rule.spread``) with the
  candidate's and the incumbent's own within-cell variances, ``df = df0 + sum_a (n_c,a - 1) + sum_a (n_i,a - 1)``. A candidate whose
  pooled sample sd exceeds ``variance_guard_ratio`` times the frozen sigma is INVALID.
* ``compare_final``: the confirmatory contrast (blocked t over held-out assignments, or Welch over trials for one assignment).

Nothing is imported from ``nightshift.grading``: ``t_quantile`` is implemented here (regularised incomplete beta, bisection) and the
tests compare it with the reused ``T95`` / ``T975`` tables.
"""

from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from nightshift.lab.contracts import (
    Arm,
    ComparisonMethod,
    ContractError,
    DecisionNumbers,
    FinalAnalysis,
    FinalComparison,
    Outcome,
    Rule,
    ScoreRecord,
    SpreadEstimate,
    Split,
    TrialScore,
)

__all__ = [
    "t_quantile",
    "trial_spread",
    "screen",
    "select_screen_winner",
    "decide",
    "minimum_detectable_gain",
    "minimum_detectable_gain_from_se",
    "compare_final",
    "compare_to_stock",
    "StockComparison",
    "seed_spread",
    "assignment_spread",
    "prompt_outcomes",
    "paired_bootstrap_prompts",
    "mcnemar",
    "false_accept_bound",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
]

# Reporting resolution of the paired bootstrap (not a decision threshold; the report prints it).
DEFAULT_BOOTSTRAP_RESAMPLES = 2000

_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------------------------------
# distribution helpers
# --------------------------------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_upper_tail(t: float, df: float) -> float:
    """P(T > t) for t >= 0."""
    if t <= 0.0:
        return 0.5
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _finite(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ContractError(f"{what}: expected a finite number, got {v!r}")
    return float(v)


@functools.lru_cache(maxsize=4096)
def _t_quantile_cached(alpha: float, df: float) -> float:
    if alpha == 0.5:
        return 0.0
    if alpha > 0.5:
        return -_t_quantile_cached(1.0 - alpha, df)
    lo, hi = 0.0, 1.0
    while _t_upper_tail(hi, df) > alpha:
        lo, hi = hi, hi * 2.0
        if hi > 1e15:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_upper_tail(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def t_quantile(alpha: float, df: float) -> float:
    """One-sided Student-t critical value: the ``t`` with ``P(T > t) = alpha`` (so ``t_quantile(0.05, 8)`` is about 1.860 and
    ``t_quantile(0.025, 8)`` about 2.306). Works for any ``alpha`` in (0, 1) and any real ``df > 0``; ``alpha > 0.5`` gives the
    negative of the corresponding upper value."""
    a = _finite(alpha, "alpha")
    d = _finite(df, "df")
    if not 0.0 < a < 1.0:
        raise ContractError(f"t_quantile: alpha must be in (0, 1), got {alpha!r}")
    if d <= 0.0:
        raise ContractError(f"t_quantile: df must be > 0, got {df!r}")
    return _t_quantile_cached(a, d)


@functools.lru_cache(maxsize=1024)
def _power_noncentrality(tc: float, df: float, power: float) -> float:
    """The noncentrality ``ncp`` (true gain divided by the standard error of the estimate) at which a test that accepts iff
    ``delta_hat > tc * s * se`` has the given power, where ``s`` is a variance estimate with ``df`` degrees of freedom around the true
    sigma. Exact: ``power(ncp) = E_s[Phi(ncp - tc * s)]`` with ``s = sqrt(chi2_df / df)``, integrated numerically."""
    if df > 1e6:  # s is 1 to within 4e-4
        return tc + _norm_ppf(power)

    smax = 1.0 + 9.0 / math.sqrt(df)
    n = 2000
    h = smax / n
    log_c = -(df / 2.0) * math.log(2.0) - math.lgamma(df / 2.0)
    nodes: list[tuple[float, float]] = []
    total = 0.0
    for i in range(n):
        s = (i + 0.5) * h
        v = df * s * s
        # density of s = sqrt(V/df), V ~ chi2(df): f_V(v) * dv/ds with dv/ds = 2 df s
        logf = (df / 2.0 - 1.0) * math.log(v) - v / 2.0 + log_c + math.log(2.0 * df * s)
        w = math.exp(logf) * h
        nodes.append((s, w))
        total += w
    nodes = [(s, w / total) for s, w in nodes]  # absorb the (tiny) quadrature error

    def pw(ncp: float) -> float:
        return math.fsum(w * _phi(ncp - tc * s) for s, w in nodes)

    lo, hi = 0.0, tc + 10.0
    while pw(hi) < power:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pw(mid) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _norm_ppf(p: float) -> float:
    """Standard normal quantile by bisection (only used for the large-df shortcut)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------------------
# small numeric helpers (order-independent: sums use fsum over id-sorted data)
# --------------------------------------------------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _ss(xs: Sequence[float]) -> float:
    """Sum of squared deviations from the mean."""
    if min(xs) == max(xs):  # identical values have exactly zero spread (the mean of equal floats can be off by one ulp)
        return 0.0
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs)


def _sd(xs: Sequence[float]) -> float:
    return math.sqrt(_ss(xs) / (len(xs) - 1))


def _by_assignment(scores: Iterable[TrialScore]) -> dict:
    """assignment_id -> list of TrialScore sorted by trial id (so every sum is independent of input order)."""
    out: dict = {}
    for s in scores:
        out.setdefault(s.assignment_id, []).append(s)
    for v in out.values():
        v.sort(key=lambda t: t.trial_id)
    return out


# --------------------------------------------------------------------------------------------------
# trial spread
# --------------------------------------------------------------------------------------------------
def trial_spread(scores: Sequence[TrialScore]) -> SpreadEstimate:
    """Pooled within-assignment trial-to-trial standard deviation of the v0 trials.

    Cells are assignments with at least two trials; a cell with one trial adds no degrees of freedom and is left out of ``trial_ids``.
    ``df = sum(n_a - 1)``. Raises ContractError("too few trials to estimate spread") when no assignment has two trials.
    """
    scores = list(scores)
    ids = [s.trial_id for s in scores]
    if len(set(ids)) != len(ids):
        raise ContractError("trial_spread: duplicate trial ids")
    cells = {a: v for a, v in _by_assignment(scores).items() if len(v) >= 2}
    if not cells:
        raise ContractError("too few trials to estimate spread")
    ss = math.fsum(_ss([t.score for t in v]) for v in cells.values())
    df = sum(len(v) - 1 for v in cells.values())
    trial_ids = tuple(t.trial_id for a in sorted(cells) for t in cells[a])
    return SpreadEstimate(math.sqrt(ss / df), df, trial_ids, len(cells))


# --------------------------------------------------------------------------------------------------
# comparing an incumbent with a candidate
# --------------------------------------------------------------------------------------------------
def _prepare(
    rule: Rule,
    incumbent: Sequence[TrialScore],
    candidate: Sequence[TrialScore],
    *,
    n_candidate: int,
    n_incumbent: Optional[int],
):
    """Group both sides by assignment, or return a reason string when they cannot be compared."""
    inc, cand = list(incumbent), list(candidate)
    if not inc:
        return "no incumbent trials"
    if not cand:
        return "no candidate trials"
    ids = [t.trial_id for t in inc] + [t.trial_id for t in cand]
    if len(set(ids)) != len(ids):
        return "a trial appears twice (within a side or on both sides)"
    allowed = set(rule.assignments)
    for side, name in ((inc, "incumbent"), (cand, "candidate")):
        for t in side:
            if t.assignment_id not in allowed:
                return f"{name} trial {t.trial_id} is on assignment {t.assignment_id}, which is not a rule assignment"
            if t.metric != rule.metric:
                return f"{name} trial {t.trial_id} has metric {t.metric}, the rule uses {rule.metric}"
            if t.scorer_version != rule.scorer_version:
                return f"{name} trial {t.trial_id} was scored with scorer version {t.scorer_version}, the rule uses {rule.scorer_version}"
    inc_by, cand_by = _by_assignment(inc), _by_assignment(cand)
    for a in rule.assignments:
        if a not in inc_by:
            return f"assignment {a} has no incumbent trials"
        if a not in cand_by:
            return f"assignment {a} has no candidate trials"
        keys = {t.comparability_sha256 for t in inc_by[a]} | {t.comparability_sha256 for t in cand_by[a]}
        if len(keys) != 1:
            return f"assignment {a}: trials carry different comparability keys (scorer, decoding, seeds, selection file, budget, shape or base differ)"
        if len(cand_by[a]) != n_candidate:
            return f"assignment {a}: {len(cand_by[a])} candidate trials, the rule needs {n_candidate}"
        if n_incumbent is not None and len(inc_by[a]) != n_incumbent:
            return f"assignment {a}: {len(inc_by[a])} incumbent trials, the rule needs {n_incumbent}"
    return inc_by, cand_by


def _delta(rule: Rule, inc_by: Mapping, cand_by: Mapping) -> float:
    a_n = len(rule.assignments)
    diffs = [_mean([t.score for t in cand_by[a]]) - _mean([t.score for t in inc_by[a]]) for a in rule.assignments]
    return math.fsum(diffs) / a_n


def screen(rule: Rule, incumbent: Sequence[TrialScore], candidates: Mapping[str, Sequence[TrialScore]]) -> dict:
    """Delta of each candidate against the incumbent's existing trials: ``(1/A) * sum_a (m_k,a - m_i,a)``.

    The value is None when the candidate cannot be compared: a rule assignment has no trials on either side, the candidate does not
    have ``rule.screen_trials_per_candidate`` trials in every assignment, or the comparability key, metric or scorer version differ.
    There is no threshold here; ``select_screen_winner`` picks who advances.
    """
    out: dict = {}
    for cid in sorted(candidates):
        prepared = _prepare(rule, incumbent, candidates[cid], n_candidate=rule.screen_trials_per_candidate, n_incumbent=None)
        out[cid] = None if isinstance(prepared, str) else _delta(rule, *prepared)
    return out


def select_screen_winner(deltas: Mapping[str, Optional[float]]) -> Optional[str]:
    """The candidate that advances to confirmation: the largest delta that is greater than 0, ties to the smaller id. None if no
    candidate has a positive delta (the round then has no confirmation)."""
    best_id: Optional[str] = None
    best = 0.0
    for cid in sorted(deltas):
        d = deltas[cid]
        if d is not None and d > best:
            best_id, best = cid, d
    return best_id


def decide(rule: Rule, incumbent: Sequence[TrialScore], candidate: Sequence[TrialScore]) -> DecisionNumbers:
    """The confirmation of one candidate with FRESH trials on both sides (see the module docstring for the formulas).

    Never raises for data problems: an assignment without trials on either side, a comparability mismatch, trial counts that differ
    from the rule's, or a candidate whose pooled sample sd exceeds ``variance_guard_ratio`` times the frozen sigma give INVALID with
    no numbers.
    """
    n_c, n_i = rule.confirm_trials_per_candidate, rule.confirm_trials_per_incumbent
    prepared = _prepare(rule, incumbent, candidate, n_candidate=n_c, n_incumbent=n_i)
    if isinstance(prepared, str):
        return DecisionNumbers(Outcome.INVALID, None, None, None, None, prepared)
    inc_by, cand_by = prepared
    a_n = len(rule.assignments)

    ss_c = math.fsum(_ss([t.score for t in cand_by[a]]) for a in rule.assignments)
    ss_i = math.fsum(_ss([t.score for t in inc_by[a]]) for a in rule.assignments)
    df_c = sum(len(cand_by[a]) - 1 for a in rule.assignments)
    df_i = sum(len(inc_by[a]) - 1 for a in rule.assignments)
    sd_c = math.sqrt(ss_c / df_c)
    sigma0 = rule.spread.sigma
    if sd_c > rule.variance_guard_ratio * sigma0:
        return DecisionNumbers(
            Outcome.INVALID,
            None,
            None,
            None,
            None,
            f"variance guard: candidate trial sd {sd_c:.4f} exceeds {rule.variance_guard_ratio:g} x the frozen sigma {sigma0:.4f}",
        )

    df = rule.spread.df + df_c + df_i
    sigma_used = math.sqrt((rule.spread.df * sigma0 * sigma0 + ss_c + ss_i) / df)
    delta = _delta(rule, inc_by, cand_by)
    se_factor = math.sqrt(math.fsum(1.0 / len(cand_by[a]) + 1.0 / len(inc_by[a]) for a in rule.assignments)) / a_n
    tq = t_quantile(rule.alpha, df)
    margin = tq * sigma_used * se_factor + rule.min_gain
    accepted = delta > margin
    reason = (
        f"delta {delta:.4f} {'exceeds' if accepted else 'does not exceed'} margin {margin:.4f} "
        f"(t {tq:.3f} at alpha {rule.alpha:g} and df {df}, sigma_used {sigma_used:.4f}, se_factor {se_factor:.4f}, min_gain {rule.min_gain:g})"
    )
    return DecisionNumbers(Outcome.ACCEPTED if accepted else Outcome.REJECTED, delta, margin, df, sigma_used, reason)


def false_accept_bound(alpha: float, rounds: int) -> float:
    """The nominal chance of at least one false accept over ``rounds`` independent confirmations under a true null with candidate trials as
    noisy as the incumbent's: ``1 - (1 - alpha)^rounds``. It is NOT a guaranteed ceiling: with candidates noisier than the incumbent (the
    variance guard only rejects those above its ratio) the simulated rate exceeds it slightly (tests/test_lab_stats.py). The report prints
    it next to the rule, worded as nominal (the rule is a heuristic filter, not evidence)."""
    a = _finite(alpha, "alpha")
    if not 0.0 < a < 1.0 or isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ContractError("false_accept_bound: alpha in (0, 1) and rounds >= 1 required")
    return 1.0 - (1.0 - a) ** rounds


# --------------------------------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------------------------------
def minimum_detectable_gain(
    sigma: float,
    df: int,
    n_candidate: int,
    n_incumbent: int,
    n_assignments: int,
    alpha: float,
    power: float = 0.8,
    *,
    min_gain: float = 0.0,
) -> float:
    """The true gain the confirmation detects with probability ``power`` (default 0.8), assuming the trial sd is ``sigma`` and the
    rule's variance estimate has ``df`` degrees of freedom (``Rule.pooled_df()``), with ``n_candidate`` and ``n_incumbent`` trials per
    assignment over ``n_assignments`` assignments. It is exact for the t test of ``decide`` (the margin's own sigma estimate is
    treated as random), so it is a little larger than the textbook ``(z_alpha + z_power) * sigma * se``. ``min_gain`` shifts the margin.
    """
    s = _finite(sigma, "sigma")
    d = _finite(df, "df")
    p = _finite(power, "power")
    mg = _finite(min_gain, "min_gain")
    if s < 0 or mg < 0:
        raise ContractError("minimum_detectable_gain: sigma and min_gain must be >= 0")
    for name, n in (("n_candidate", n_candidate), ("n_incumbent", n_incumbent), ("n_assignments", n_assignments)):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ContractError(f"minimum_detectable_gain: {name} must be an integer >= 1")
    if not 0.0 < p < 1.0:
        raise ContractError("minimum_detectable_gain: power must be in (0, 1)")
    se_factor = math.sqrt((1.0 / n_candidate + 1.0 / n_incumbent) / n_assignments)  # equal n in every assignment
    tc = t_quantile(alpha, d)
    return mg + _power_noncentrality(tc, d, p) * s * se_factor


def minimum_detectable_gain_from_se(se: float, df: float, alpha: float, power: float = 0.8) -> float:
    """The true effect that a one-sided test at level ``alpha`` (use ``(1 - confidence) / 2`` for a two-sided interval) detects with
    probability ``power`` when the estimate's standard error is ``se`` with ``df`` degrees of freedom. Used next to every final
    interval."""
    s = _finite(se, "se")
    if s < 0:
        raise ContractError("minimum_detectable_gain_from_se: se must be >= 0")
    p = _finite(power, "power")
    if not 0.0 < p < 1.0:
        raise ContractError("minimum_detectable_gain_from_se: power must be in (0, 1)")
    return _power_noncentrality(t_quantile(alpha, df), _finite(df, "df"), p) * s


# --------------------------------------------------------------------------------------------------
# final comparison
# --------------------------------------------------------------------------------------------------
def _welch(x_ev: Sequence[float], x_v0: Sequence[float]) -> tuple:
    """(delta, se, df) of mean(x_ev) - mean(x_v0); df is the Welch-Satterthwaite value, floored (at least 1)."""
    n_e, n_0 = len(x_ev), len(x_v0)
    if n_e < 2 or n_0 < 2:
        raise ContractError("Welch comparison needs at least two trials per arm")
    v_e, v_0 = _ss(x_ev) / (n_e - 1) / n_e, _ss(x_v0) / (n_0 - 1) / n_0
    delta = _mean(x_ev) - _mean(x_v0)
    se = math.sqrt(v_e + v_0)
    if se == 0.0:
        return delta, 0.0, float(n_e + n_0 - 2)
    df = (v_e + v_0) ** 2 / (v_e**2 / (n_e - 1) + v_0**2 / (n_0 - 1))
    return delta, se, float(max(1, math.floor(df)))


def _collect_planned(analysis: FinalAnalysis, arm: Arm, scores: Sequence[TrialScore], metric: str, split: Split) -> dict:
    planned = {t.trial_id: t for t in analysis.planned_trials if t.arm is arm}
    got = {}
    for s in scores:
        if s.trial_id in got:
            raise ContractError(f"compare_final: trial {s.trial_id} is scored twice")
        got[s.trial_id] = s
    unplanned = sorted(set(got) - set(planned))
    if unplanned:
        raise ContractError(f"compare_final: {arm.value} scores for trials that are not in the {arm.value} plan: {unplanned}")
    missing = sorted(set(planned) - set(got))
    if missing:
        raise ContractError(f"compare_final: planned {arm.value} trials have no score: {missing}")
    for tid, s in got.items():
        if s.metric != metric:
            raise ContractError(f"compare_final: trial {tid} has metric {s.metric}, the analysis uses {metric}")
        if split is Split.SELECTION and s.assignment_id != planned[tid].assignment_id:
            raise ContractError(f"compare_final: trial {tid} is scored on assignment {s.assignment_id}, planned on {planned[tid].assignment_id}")
    return {tid: (planned[tid], got[tid]) for tid in sorted(got)}


def compare_final(analysis: FinalAnalysis, v0: Sequence[TrialScore], evolved: Sequence[TrialScore], split: Split) -> FinalComparison:
    """The confirmatory contrast ``evolved - v0`` on the held-out metric (``Split.SELECTION``, the held-out assignments) or the test
    metric (``Split.TEST``) of a frozen ``FinalAnalysis``.

    With two or more held-out assignments (``ComparisonMethod.BLOCKED_T``) trials are averaged within an assignment, the per-assignment
    differences are combined with the analysis weights, and the interval is a two-sided t interval over assignments (``df = H - 1``;
    the standard error is the reliability-weighted one, which equals ``s / sqrt(H)`` for equal weights). With one assignment
    (``WELCH``) it is a Welch interval over trials. Both arms must supply exactly their planned trials (the analysis fixes them before
    any score exists), and every trial of an assignment must carry the same comparability key, otherwise ContractError: a confirmatory
    comparison cannot be marked INVALID and retried. For the test split, trials are grouped by the held-out assignment they were
    planned on (the test set itself has no assignment).
    """
    if split is Split.SELECTION:
        metric = analysis.heldout_metric
    elif split is Split.TEST:
        metric = analysis.test_metric
    else:
        raise ContractError("compare_final: dev is not a final split")
    pv = _collect_planned(analysis, Arm.V0, v0, metric, split)
    pe = _collect_planned(analysis, Arm.EVOLVED, evolved, metric, split)

    weights = {w.assignment_id: w.weight for w in analysis.assignments}
    groups: dict = {a: {"v0": [], "ev": [], "keys": set()} for a in weights}
    for tid, (plan, ts) in pv.items():
        groups[plan.assignment_id]["v0"].append(ts.score)
        groups[plan.assignment_id]["keys"].add(ts.comparability_sha256)
    for tid, (plan, ts) in pe.items():
        groups[plan.assignment_id]["ev"].append(ts.score)
        groups[plan.assignment_id]["keys"].add(ts.comparability_sha256)
    for a, g in groups.items():
        if len(g["keys"]) != 1:
            raise ContractError(f"compare_final: assignment {a}: trials carry different comparability keys; the arms were not budget-matched")

    level = analysis.confidence
    alpha2 = (1.0 - level) / 2.0
    n_v0, n_ev = len(pv), len(pe)
    if analysis.method is ComparisonMethod.BLOCKED_T:
        ids = sorted(weights)
        h = len(ids)
        m0 = {a: _mean(groups[a]["v0"]) for a in ids}
        me = {a: _mean(groups[a]["ev"]) for a in ids}
        d = {a: me[a] - m0[a] for a in ids}
        w = {a: weights[a] for a in ids}
        delta = math.fsum(w[a] * d[a] for a in ids)
        sw2 = math.fsum(w[a] ** 2 for a in ids)
        var_w = math.fsum(w[a] * (d[a] - delta) ** 2 for a in ids) / (1.0 - sw2)
        se = math.sqrt(var_w * sw2)
        df = float(h - 1)
        mean_v0 = math.fsum(w[a] * m0[a] for a in ids)
        mean_ev = math.fsum(w[a] * me[a] for a in ids)
        note = (
            f"Blocked over {h} held-out assignments: trials are averaged within an assignment and the interval is a t interval over the "
            f"per-assignment differences with {h - 1} degree(s) of freedom. A small number of assignments gives a wide interval."
        )
        n_assign = h
    else:
        (only,) = list(groups)
        delta, se, df = _welch(groups[only]["ev"], groups[only]["v0"])
        mean_v0, mean_ev = _mean(groups[only]["v0"]), _mean(groups[only]["ev"])
        note = (
            "Within one assignment; does not generalise across assignments. Welch interval over trials "
            f"(Welch-Satterthwaite degrees of freedom {df:g}, floored)."
        )
        n_assign = 1
    half = t_quantile(alpha2, df) * se
    ci_low, ci_high = delta - half, delta + half
    return FinalComparison(
        analysis_sha256=analysis.sha256(),
        split=split,
        metric=metric,
        contrast="evolved_minus_v0",
        method=analysis.method,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        level=level,
        df=df,
        n_assignments=n_assign,
        trials_v0=n_v0,
        trials_evolved=n_ev,
        mean_v0=mean_v0,
        mean_evolved=mean_ev,
        includes_zero=ci_low <= 0.0 <= ci_high,
        note=note,
    )


@dataclass(frozen=True)
class StockComparison:
    """Arm 1 (stock Glimmer, one fixed score) against the trial scores of arm 2 or arm 3. Not persisted by the contracts; the report
    prints it. The interval is a one-sample t interval over trials around the stock mean, which is treated as a constant whose own
    generation-seed spread is reported in ``stock_seed_sd``."""

    metric: str
    stock_mean: float
    stock_seed_sd: Optional[float]
    arm_mean: float
    arm_trial_sd: float
    n_trials: int
    delta: float
    ci_low: float
    ci_high: float
    level: float
    df: int
    includes_zero: bool
    note: str


def compare_to_stock(
    arm_scores: Sequence[TrialScore], stock: ScoreRecord, *, level: float = 0.95, weights: Optional[Mapping[str, float]] = None, metric: Optional[str] = None
) -> StockComparison:
    """Trial scores of arm 2 (or 3) minus the mean over generation seeds of the stock record (arm 1).

    ``weights`` (assignment id -> weight, summing to 1; the frozen ``FinalAnalysis`` weights) makes the arm mean the weighted mean of the
    per-assignment trial means, the same estimator ``compare_final`` uses, so a stock comparison and the arm table agree. With two or
    more weighted assignments the interval is then the blocked one over assignments (``df = H - 1``, reliability-weighted standard error);
    with one assignment, or no ``weights``, it is the one-sample t interval over trials.

    ``metric`` (default: the stock record's headline metric) is the metric both sides are compared on; the frozen analysis passes its
    ``test_metric`` so the stock side is read from the same metric as the trial scores, whatever the record's headline is.
    """
    scores = list(arm_scores)
    lv = _finite(level, "level")
    if not 0.5 < lv < 1.0:
        raise ContractError("compare_to_stock: level must be in (0.5, 1)")
    if len(scores) < 2:
        raise ContractError("compare_to_stock: needs at least two trial scores")
    metric = metric or stock.headline_metric
    if any(metric not in s.metrics for s in stock.per_seed):
        raise ContractError(f"compare_to_stock: the stock record has no {metric} metric")
    for t in scores:
        if t.metric != metric:
            raise ContractError(f"compare_to_stock: trial {t.trial_id} has metric {t.metric}, the stock comparison is on {metric}")
    stock_vals = stock.metric_by_seed(metric)
    stock_mean = math.fsum(stock_vals) / len(stock_vals)
    xs = [t.score for t in sorted(scores, key=lambda t: t.trial_id)]
    n = len(xs)
    sd = _sd(xs)
    note = "The stock score is one fixed measurement (generation-seed spread only); the interval reflects trial-to-trial spread of the arm."
    cells: dict = {}
    for t in scores:
        cells.setdefault(t.assignment_id, []).append(t.score)
    if weights is not None:
        missing = sorted(a for a in cells if a not in weights)
        if missing:
            raise ContractError(f"compare_to_stock: no weight for assignment(s) {missing}")
        cells = {a: v for a, v in cells.items() if weights[a] > 0}
    if weights is not None and len(cells) >= 2:
        total = math.fsum(weights[a] for a in cells)
        w = {a: weights[a] / total for a in cells}
        d = {a: _mean(v) - stock_mean for a, v in cells.items()}
        delta = math.fsum(w[a] * d[a] for a in cells)
        arm_mean = stock_mean + delta
        sw2 = math.fsum(x ** 2 for x in w.values())
        var_w = math.fsum(w[a] * (d[a] - delta) ** 2 for a in cells) / (1.0 - sw2)
        df = len(cells) - 1
        half = t_quantile((1.0 - lv) / 2.0, df) * math.sqrt(var_w * sw2)
        note = (
            f"The stock score is one fixed measurement (generation-seed spread only). The arm mean is weighted by the frozen assignment weights over "
            f"{len(cells)} assignments; the interval is a t interval over the per-assignment differences with {df} degree(s) of freedom."
        )
    else:
        arm_mean = _mean(xs)
        delta = arm_mean - stock_mean
        df = n - 1
        half = t_quantile((1.0 - lv) / 2.0, df) * sd / math.sqrt(n)
    lo, hi = delta - half, delta + half
    return StockComparison(
        metric=metric,
        stock_mean=stock_mean,
        stock_seed_sd=_sd(list(stock_vals)) if len(stock_vals) > 1 else None,
        arm_mean=arm_mean,
        arm_trial_sd=sd,
        n_trials=n,
        delta=delta,
        ci_low=lo,
        ci_high=hi,
        level=lv,
        df=df,
        includes_zero=lo <= 0.0 <= hi,
        note=note,
    )


# --------------------------------------------------------------------------------------------------
# diagnostics (describe two checkpoints on identical prompts; they never decide a keep)
# --------------------------------------------------------------------------------------------------
def seed_spread(record: ScoreRecord, metric: Optional[str] = None) -> dict:
    """Generation-seed spread inside one ScoreRecord: mean, sample sd (None for a single seed), min and max over seeds of ``metric``
    (default: the headline). This is scorer noise, the smallest of the three spreads."""
    vals = list(record.metric_by_seed(metric))
    return {
        "metric": metric or record.headline_metric,
        "n_seeds": len(vals),
        "mean": _mean(vals),
        "sd": _sd(vals) if len(vals) > 1 else None,
        "min": min(vals),
        "max": max(vals),
    }


def assignment_spread(scores: Sequence[TrialScore]) -> Optional[dict]:
    """Spread of the per-assignment means (sample sd over assignments); None with fewer than two assignments."""
    by = _by_assignment(scores)
    if len(by) < 2:
        return None
    means = {a: _mean([t.score for t in v]) for a, v in by.items()}
    vals = [means[a] for a in sorted(means)]
    return {"n_assignments": len(vals), "means": {a: means[a] for a in sorted(means)}, "sd": _sd(vals)}


_FIELD_FOR_METRIC = {
    "prompt_all": "passed",
    "prompt_strict": "strict_prompt",
    "prompt_loose": "loose_prompt",
}


def prompt_outcomes(rows: Iterable[Mapping[str, Any]], *, field: str = "passed", per_seed: bool = False) -> dict:
    """Collapse ``per_prompt.jsonl`` rows (dev/selection: ``passed``; test: ``strict_prompt`` / ``loose_prompt``) to ``prompt_key ->
    pass rate over seeds`` or, with ``per_seed``, to ``(prompt_key, seed) -> bool``. Rows whose ``error`` flag is set count as failures."""
    if per_seed:
        out: dict = {}
        for r in rows:
            key = (r["prompt_key"], r["seed"])
            if key in out:
                raise ContractError(f"prompt_outcomes: duplicate row for {key}")
            out[key] = bool(r[field]) and not r.get("error", False)
        return out
    acc: dict = {}
    for r in rows:
        acc.setdefault(r["prompt_key"], []).append(1.0 if (bool(r[field]) and not r.get("error", False)) else 0.0)
    return {k: _mean(v) for k, v in acc.items()}


def _paired(a: Mapping, b: Mapping) -> list:
    if set(a) != set(b):
        raise ContractError("the two checkpoints were not scored on identical prompts")
    if not a:
        raise ContractError("no prompts to compare")
    return sorted(a)


def paired_bootstrap_prompts(
    per_prompt_a: Mapping[Any, float],
    per_prompt_b: Mapping[Any, float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int,
) -> dict:
    """Paired bootstrap over PROMPTS of ``mean(b) - mean(a)`` for two checkpoints scored on identical prompts (percentile 95% interval).
    It resamples prompts only, so it measures prompt-sampling noise and not trial-to-trial variation: a diagnostic, never a keep rule.
    ``resamples`` is a reporting resolution, not a decision threshold; the seed makes the result reproducible."""
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ContractError("paired_bootstrap_prompts: resamples must be an integer >= 1")
    keys = _paired(per_prompt_a, per_prompt_b)
    diffs = [float(per_prompt_b[k]) - float(per_prompt_a[k]) for k in keys]
    n = len(diffs)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        stats.append(math.fsum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    stats.sort()
    lo = stats[int(0.025 * (resamples - 1))]
    hi = stats[int(math.ceil(0.975 * (resamples - 1)))]
    return {
        "n_prompts": n,
        "mean_a": _mean([float(per_prompt_a[k]) for k in keys]),
        "mean_b": _mean([float(per_prompt_b[k]) for k in keys]),
        "delta": math.fsum(diffs) / n,
        "ci_low": lo,
        "ci_high": hi,
        "resamples": resamples,
        "seed": seed,
        "label": "checkpoint-level, prompts resampled; trial-to-trial variation is not included; never decides a keep",
    }


def mcnemar(per_prompt_a: Mapping[Any, Any], per_prompt_b: Mapping[Any, Any]) -> dict:
    """Exact McNemar test (two-sided binomial on the discordant pairs) for two checkpoints on identical prompts with boolean outcomes.
    ``b_only`` counts prompts only b passes, ``a_only`` prompts only a passes. A diagnostic, never a keep rule."""
    keys = _paired(per_prompt_a, per_prompt_b)
    for m in (per_prompt_a, per_prompt_b):
        for k in keys:
            if not isinstance(m[k], bool) and m[k] not in (0, 1):
                raise ContractError("mcnemar: outcomes must be booleans (collapse seeds first)")
    b_only = sum(1 for k in keys if bool(per_prompt_b[k]) and not bool(per_prompt_a[k]))
    a_only = sum(1 for k in keys if bool(per_prompt_a[k]) and not bool(per_prompt_b[k]))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        # exact integers: float powers and sums overflow from n >= 1024 discordant prompts; int / int true division stays correct
        tail = sum(math.comb(n, i) for i in range(0, min(a_only, b_only) + 1)) / (1 << n)
        p = min(1.0, 2.0 * tail)
    return {
        "n_prompts": len(keys),
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": n,
        "p_value": p,
        "label": "checkpoint-level, exact two-sided test on discordant prompts; never decides a keep",
    }
