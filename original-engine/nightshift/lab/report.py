"""The lab report: tables, the search-trace chart and the labels every claim must carry (docs/harness-design.md, section 4).

``build_report(state, layout, *, full=False)`` turns a ``LabState`` (made by ``fold``) into a module-local ``Report``; ``render_markdown``
and ``render_html`` print it. Everything is deterministic: no clock, no randomness except the seeded bootstrap in ``stats``.

What it shows, and when:

* Default view (``full=False``): the outer loop's view of the search, bits and costs only (outcome of each candidate, its cost). No
  selection score, delta, margin or sigma, no decision reason text, no per-prompt rows. This is also what ``status`` shows.
* ``full=True``: refused (``AccessDenied``) before ``lab.search_closed``. After it, adds the decision numbers, the three spreads, the
  search trace with scores, and the per-trial scores of the search.
* Per-trial held-out and test scores, and every aggregate made from them, appear only after ``lab.final_revealed``.

The report reads the ledger only through the ``LabState`` it is given. Files it needs besides that (frozen records, per-prompt rows,
trial ledgers) are read with the link-refusing helpers of ``contracts``; a file whose sha256 does not match its reference is listed as
a warning and its numbers are not used.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from nightshift.lab import stats
from nightshift.lab.chart import TracePoint, search_trace_svg
from nightshift.lab.contracts import (
    AccessDenied,
    Arm,
    AssignmentManifest,
    BudgetCharge,
    ContractError,
    DecisionStage,
    FailedTrialPolicy,
    FinalAnalysis,
    FinalComparison,
    ItemMode,
    LabKind,
    LabLayout,
    LabState,
    Outcome,
    Phase,
    Rule,
    ScoreRecord,
    Split,
    TrainResult,
    TrialScore,
    TrialStatus,
    Verifier,
    comparability_key,
    loads_strict,
    load_frozen,
    read_regular_bytes,
)

__all__ = [
    "Report",
    "Section",
    "Block",
    "Table",
    "TrialFacts",
    "build_report",
    "render_markdown",
    "render_html",
    "load_trial_facts",
    "TRACE_TITLE",
    "IFBENCH_TEST_PROMPTS",
]

TRACE_TITLE = "adaptive search trace"
IFBENCH_TEST_PROMPTS = 300  # rows of IFBench_test.jsonl; a test score over fewer prompts is labelled partial
_PER_PROMPT_MAX_BYTES = 64 << 20
_LEDGER_MAX_BYTES = 256 << 20
_MAX_ESTIMATE_WARNINGS = 200

_SCORED = (TrialStatus.COMPLETED, TrialStatus.BUDGET_EXHAUSTED)
_FIELD_OF_METRIC = {"prompt_all": "passed", "prompt_strict": "strict_prompt", "prompt_loose": "loose_prompt"}

# Labels (exact strings; tests and readers search for them).
LABEL_COMPONENTS_ONLY = "components tested, system not yet demonstrated"
LABEL_ARM3_NOT_TESTED = "arm 3 versus arm 2 was not tested"
LABEL_SPREAD_NOT_ESTIMATED = "spread not estimated"
LABEL_DEV_INFLUENCED = "dev scores are researcher-influenced; decisions use control-host numbers only"
LABEL_ADAPTIVE = "the bits and the operator's choices steer the search"
LABEL_LEDGER = "the ledger proves order and non-editing, not truth"


# --------------------------------------------------------------------------------------------------
# report structure
# --------------------------------------------------------------------------------------------------
@dataclass
class Table:
    caption: str
    headers: list
    rows: list


@dataclass
class Block:
    kind: str  # "para" | "table" | "bullets" | "svg" | "trace"
    text: str = ""
    table: Optional[Table] = None
    items: list = field(default_factory=list)


@dataclass
class Section:
    key: str
    title: str
    blocks: list = field(default_factory=list)

    def para(self, text: str) -> None:
        self.blocks.append(Block("para", text=text))

    def bullets(self, items: Sequence[str]) -> None:
        self.blocks.append(Block("bullets", items=list(items)))

    def table(self, caption: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        self.blocks.append(Block("table", table=Table(caption, list(headers), [[str(c) for c in r] for r in rows])))


@dataclass
class Report:
    title: str
    full: bool
    search_closed: bool
    revealed: bool
    head: Optional[str]
    sections: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def section(self, key: str) -> Optional[Section]:
        for s in self.sections:
            if s.key == key:
                return s
        return None


# --------------------------------------------------------------------------------------------------
# facts read from files (never from the ledger directly)
# --------------------------------------------------------------------------------------------------
@dataclass
class TrialFacts:
    """What a trial ledger holds that the master ledger does not: budget charges and the trainer's job results."""

    charges: list = field(default_factory=list)
    jobs: list = field(default_factory=list)
    problems: list = field(default_factory=list)


def load_trial_facts(layout: LabLayout, trial_ids: Sequence[str]) -> dict:
    """Read ``trials/<id>/ledger.jsonl`` of each trial (read-only, no lock, link-refusing). A missing ledger gives no entry; a line that
    does not parse is listed in ``problems``. Only ``lab.budget_charged`` and ``lab.gpu_job_finished`` events are used."""
    out: dict = {}
    for tid in trial_ids:
        rel = layout.trial_dir(tid).relative_to(layout.root).as_posix() + "/ledger.jsonl"
        try:
            raw = read_regular_bytes(layout.root, rel, max_bytes=_LEDGER_MAX_BYTES)
        except (OSError, ContractError):
            continue
        facts = TrialFacts()
        for n, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                kind, data = ev["kind"], ev["data"]
                if kind == LabKind.BUDGET_CHARGED:
                    facts.charges.append(BudgetCharge.from_dict(data))
                elif kind == LabKind.GPU_JOB_FINISHED:
                    facts.jobs.append(TrainResult.from_dict(data))
            except (ValueError, KeyError, TypeError, ContractError) as e:
                facts.problems.append(f"line {n}: {type(e).__name__}")
        out[tid] = facts
    return out


def _is_estimated(charge: BudgetCharge) -> bool:
    for k, v in charge.detail.items():
        if k == "usage_estimated" and v not in (False, None, 0, "", "false", "False"):
            return True
        if v == "usage_estimated":
            return True
    return False


def _read_json_frozen(layout: LabLayout, frozen, cls, warnings: list, what: str):
    if frozen is None:
        return None
    try:
        return load_frozen(layout, frozen, cls)
    except (OSError, ContractError, ValueError) as e:
        warnings.append(f"frozen {what} could not be verified against the ledger ({type(e).__name__}: {e}); its numbers are not used")
        return None


def _read_per_prompt(layout: LabLayout, rec: ScoreRecord, warnings: list) -> Optional[list]:
    ref = rec.per_prompt
    try:
        data = read_regular_bytes(layout.root, ref.path, max_bytes=_PER_PROMPT_MAX_BYTES)
    except (OSError, ContractError) as e:
        warnings.append(f"per-prompt file of score {rec.score_id} is unreadable ({type(e).__name__}); its numbers are not used")
        return None
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        warnings.append(f"per-prompt file of score {rec.score_id} does not match its recorded sha256; its numbers are not used")
        return None
    try:
        return [loads_strict(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    except (ValueError, ContractError):
        warnings.append(f"per-prompt file of score {rec.score_id} is not valid JSON lines; its numbers are not used")
        return None


# --------------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------------
def _f3(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _f4(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _num(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.4f}"


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def _short(text: str, n: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _mean_sd(xs: Sequence[float]) -> str:
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.fmean(xs):.3f} ± {statistics.stdev(xs):.3f}"


# --------------------------------------------------------------------------------------------------
# final-phase numbers (only computed after lab.final_revealed)
# --------------------------------------------------------------------------------------------------
@dataclass
class _Final:
    analysis: FinalAnalysis
    complete: bool = True
    missing: list = field(default_factory=list)
    v0_heldout: list = field(default_factory=list)
    ev_heldout: list = field(default_factory=list)
    v0_test: list = field(default_factory=list)
    ev_test: list = field(default_factory=list)
    heldout: Optional[FinalComparison] = None
    test: Optional[FinalComparison] = None
    stock: Optional[ScoreRecord] = None
    stock_vs_v0: Any = None
    stock_vs_evolved: Any = None
    records: dict = field(default_factory=dict)  # (trial_id or "stock", split) -> ScoreRecord


def _final_numbers(state: LabState, analysis: FinalAnalysis, warnings: list) -> _Final:
    fin = _Final(analysis)
    by_key: dict = {}
    for rec in state.scores.values():
        if rec.phase is Phase.FINAL and rec.split in (Split.SELECTION, Split.TEST):
            by_key.setdefault((rec.trial_id, rec.split), rec)
            if rec.arm is Arm.STOCK and rec.split is Split.TEST and fin.stock is None:
                fin.stock = rec
    fin.records = {k: v for k, v in by_key.items()}
    if fin.stock is not None:
        fin.records[("stock", Split.TEST)] = fin.stock

    def trial_scores(arm: Arm, split: Split, metric: str) -> list:
        out = []
        for plan in analysis.planned_trials:
            if plan.arm is not arm:
                continue
            rec = by_key.get((plan.trial_id, split))
            spec = state.trials_started.get(plan.trial_id) or (state.trials[plan.trial_id].spec if plan.trial_id in state.trials else None)
            done = state.trials.get(plan.trial_id)
            if (
                rec is None
                and spec is not None
                and done is not None
                and (done.status not in _SCORED or done.adapter is None)
                and analysis.failed_trial_policy is FailedTrialPolicy.SCORED_ZERO
            ):
                # a finished trial that produced no scored adapter contributes 0.0 under the frozen policy (as in the search phase)
                try:
                    out.append(TrialScore(plan.trial_id, plan.assignment_id, None, metric, spec.scorer_version, comparability_key(spec, metric), 0.0))
                except (ContractError, KeyError):
                    fin.missing.append(f"{plan.trial_id} ({split.value}: spec unusable)")
                continue
            if rec is None or spec is None or metric not in rec.per_seed[0].metrics:
                fin.missing.append(f"{plan.trial_id} ({split.value})")
                continue
            vals = rec.metric_by_seed(metric)
            try:
                key = comparability_key(spec, metric)
            except (ContractError, KeyError):
                fin.missing.append(f"{plan.trial_id} ({split.value}: spec unusable)")
                continue
            out.append(TrialScore(plan.trial_id, plan.assignment_id, rec.score_id, metric, rec.scorer_version, key, math.fsum(vals) / len(vals)))
        return out

    fin.v0_heldout = trial_scores(Arm.V0, Split.SELECTION, analysis.heldout_metric)
    fin.ev_heldout = trial_scores(Arm.EVOLVED, Split.SELECTION, analysis.heldout_metric)
    explore = bool(getattr(analysis, "exploratory", False))  # exploratory: held-out selection only, no test scoring was planned
    fin.v0_test = [] if explore else trial_scores(Arm.V0, Split.TEST, analysis.test_metric)
    fin.ev_test = [] if explore else trial_scores(Arm.EVOLVED, Split.TEST, analysis.test_metric)
    if explore:
        warnings.append("EXPLORATORY final analysis (owner-approved small design below the confirmatory floor): held-out selection only, no test scores; not a confirmation")
    fin.complete = not fin.missing
    if fin.missing:
        warnings.append(f"final scores are incomplete: no usable score for {len(fin.missing)} planned trial/split pair(s) ({', '.join(fin.missing[:6])}{'...' if len(fin.missing) > 6 else ''}); the comparison is not computed")
        return fin
    for attr, split, v0, ev in (("heldout", Split.SELECTION, fin.v0_heldout, fin.ev_heldout), ("test", Split.TEST, fin.v0_test, fin.ev_test)):
        if explore and split is Split.TEST:
            continue
        try:
            setattr(fin, attr, stats.compare_final(analysis, v0, ev, split))
        except ContractError as e:
            warnings.append(f"final comparison on the {split.value} split could not be computed: {e}")
    if fin.stock is not None:
        for attr, arm_scores in (("stock_vs_v0", fin.v0_test), ("stock_vs_evolved", fin.ev_test)):
            try:
                setattr(fin, attr, stats.compare_to_stock(arm_scores, fin.stock, level=analysis.confidence, weights={w.assignment_id: w.weight for w in analysis.assignments}, metric=analysis.test_metric))
            except ContractError as e:
                warnings.append(f"comparison with stock could not be computed: {e}")
    return fin


def final_numbers(state: LabState, analysis: FinalAnalysis, warnings: list) -> _Final:
    """The final-phase numbers the report prints, computed from ``state`` alone. ``cli reveal`` calls this to decide whether the whole plan is
    scored (``.complete``) and to write ``reports/final_comparison.json`` from the same numbers the report shows."""
    return _final_numbers(state, analysis, warnings)


def _se_of(fc: FinalComparison) -> float:
    alpha2 = (1.0 - fc.level) / 2.0
    t = stats.t_quantile(alpha2, fc.df)
    return (fc.ci_high - fc.delta) / t if t > 0 else 0.0


# --------------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------------
def build_report(state: LabState, layout: LabLayout, *, full: bool = False, trial_facts: Optional[Mapping[str, TrialFacts]] = None) -> Report:
    """Make the report. ``full=True`` raises ``AccessDenied`` before ``lab.search_closed``. ``trial_facts`` (trial id -> ``TrialFacts``) is
    read from the trial ledgers when not given."""
    if full and state.search_closed is None:
        raise AccessDenied("the full report is refused before lab.search_closed: during the search only bits and costs are shown")

    warnings: list = []
    revealed = state.reveal is not None
    head = f"{state.head.seq}:{state.head.hash16}" if state.head is not None else None
    report = Report("Glimmer three-arm experiment: lab report", full, state.search_closed is not None, revealed, head)

    rule: Optional[Rule] = _read_json_frozen(layout, state.rule, Rule, warnings, "rule")
    manifest: Optional[AssignmentManifest] = _read_json_frozen(layout, state.manifest, AssignmentManifest, warnings, "assignment manifest")
    analysis: Optional[FinalAnalysis] = _read_json_frozen(layout, state.final_analysis, FinalAnalysis, warnings, "final analysis")

    if state.malformed:
        shown = "; ".join(f"seq {s}: {r}" for s, r in state.malformed[:5])
        warnings.append(f"the ledger has {len(state.malformed)} malformed event(s) that were skipped ({shown}{'; ...' if len(state.malformed) > 5 else ''})")

    if trial_facts is None:
        trial_facts = load_trial_facts(layout, sorted(state.trials_started))
    for tid, facts in trial_facts.items():
        for p in facts.problems:
            warnings.append(f"trial ledger of {tid}: {p} could not be read")

    fin: Optional[_Final] = None
    if revealed and analysis is not None:
        fin = _final_numbers(state, analysis, warnings)
    elif revealed and analysis is None:
        warnings.append("lab.final_revealed exists but the final analysis could not be loaded; no final numbers are shown")

    joined = [t for t in state.trials.values() if t.status in _SCORED and t.selection_score_id in state.scores]

    labels = _labels(state, rule, manifest, trial_facts, fin, joined, warnings)
    report.labels = labels

    report.sections.append(_sec_read_first(report, labels))
    report.sections.append(_sec_status(state, rule, manifest, analysis, revealed))
    report.sections.append(_sec_search(state, rule, full))
    if full:
        report.sections.append(_sec_spreads(state, rule))
    report.sections.append(_sec_final(state, fin, analysis, revealed, full))
    if full and revealed and fin is not None and fin.complete:
        sec = _sec_per_prompt(state, layout, fin, warnings)
        if sec is not None:
            report.sections.append(sec)
    report.sections.append(_sec_scoring(state))
    report.sections.append(_sec_costs(state, trial_facts))
    report.sections.append(_sec_trials(state))
    report.sections.append(_sec_caveats(state, rule, manifest, trial_facts))
    if full and state.search_closed is not None:
        sec = _sec_search_trials(state)
        if sec is not None:
            report.sections.insert(report.sections.index(report.section("trials")) + 1, sec)
    report.warnings = warnings
    report.sections.append(_sec_warnings(warnings))
    return report


# --------------------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------------------
def _labels(state, rule, manifest, trial_facts, fin, joined, warnings) -> list:
    out: list = []
    if not joined:
        out.append(f"{LABEL_COMPONENTS_ONLY}: no trial has run the whole chain (researchers, trainer, scorer) to a scored adapter.")
    if fin is None or fin.heldout is None:
        out.append(f"{LABEL_ARM3_NOT_TESTED}: no held-out comparison exists" + ("" if state.reveal is not None else " (final scores stay hidden until lab.final_revealed)") + ".")
    if rule is None:
        out.append(f"{LABEL_SPREAD_NOT_ESTIMATED}: the rule is not frozen, so no keep decision has an evidence base (or the frozen rule could not be verified; see Warnings)." + (" A v0 spread has been measured but is not yet part of a frozen rule." if state.spread is not None else ""))
    out.append(LABEL_DEV_INFLUENCED + ".")
    out.append(f"{LABEL_ADAPTIVE[0].upper()}{LABEL_ADAPTIVE[1:]}: {len(state.operator_changes)} operator change(s) are recorded after the first candidate; the accept/reject bits are an adaptive channel.")
    out.append("The search rule is a heuristic filter, not evidence of a gain; the claim rests on the held-out comparison.")
    out.append(f"{LABEL_LEDGER[0].upper()}{LABEL_LEDGER[1:]}; its head anchor is printed above and should be compared with the anchors file in git.")

    # verifier and prompt count: custom or partial scoring
    custom = sorted({(r.split.value, r.verifier.value) for r in state.scores.values() if r.verifier is Verifier.IFEVALG})
    if custom:
        out.append("Custom scoring: dev and selection scores use the IFEvalG verifier (open-instruct's constraint checks), not upstream IFBench, and are not comparable with a published IFBench number.")
    partial = [r for r in state.scores.values() if r.split is Split.TEST and r.n_prompts < IFBENCH_TEST_PROMPTS]
    if partial:
        out.append(f"Partial scoring: {len(partial)} test score(s) cover fewer than {IFBENCH_TEST_PROMPTS} prompts ({', '.join(sorted({str(r.n_prompts) for r in partial}))}).")
    # truncation and error counts are outcome-derived (a score of exactly zero shows as every generation truncated), so a final-phase record
    # contributes to them only after lab.final_revealed and a search-phase selection record only after lab.search_closed
    countable = [r for r in state.scores.values() if _counts_visible(state, r.split, r.phase)]
    errored = [r for r in countable if any(s.n_errors for s in r.per_seed)]
    if errored:
        out.append(f"Partial scoring: {sum(s.n_errors for r in errored for s in r.per_seed)} prompt(s) errored in {len(errored)} score record(s); errors count as failures.")
    trunc = sum(s.n_truncated for r in countable for s in r.per_seed)
    if trunc:
        out.append(f"{trunc} truncated generation(s) (finish_reason length) are counted as failures in every arm.")

    n_est = 0
    for tid in sorted(trial_facts):
        for ch in trial_facts[tid].charges:
            if _is_estimated(ch):
                n_est += 1
                if n_est <= _MAX_ESTIMATE_WARNINGS:
                    warnings.append(f"estimated usage (usage_estimated): trial {tid}, call {ch.call_id}, {ch.resource.value} {ch.amount:g} was estimated by the proxy, not reported by the provider")
    if n_est > _MAX_ESTIMATE_WARNINGS:
        warnings.append(f"estimated usage (usage_estimated): {n_est - _MAX_ESTIMATE_WARNINGS} more estimated charge(s) not listed individually")
    if n_est:
        out.append(f"{n_est} budget charge(s) were estimated (usage_estimated), not provider-reported; each is listed under Warnings.")
    return out


# --------------------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------------------
def _sec_read_first(report: Report, labels: Sequence[str]) -> Section:
    sec = Section("labels", "Read this first")
    sec.para("Every statement below carries the labels that qualify it. " + ("This is the full report." if report.full else "This is the default view: during the search it shows bits and costs only."))
    sec.bullets(labels)
    return sec


def _gate_line(state) -> str:
    """Closed only by a CLARIFIED record. A PROCEED_WITH_RISK record is a waiver: the gate is still open and the line says so."""
    if not state.gates:
        return "no (open)"
    outcomes = {g.outcome.value for g in state.gates}
    if "clarified" in outcomes:
        return "closed (clarified)"
    return "OPEN: proceeding at the owner's stated risk (not closed)"


def _sec_status(state, rule, manifest, analysis, revealed) -> Section:
    sec = Section("status", "Status")
    by_status: dict = {}
    for t in state.trials.values():
        by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
    pending = len(state.trials_started) - len(state.trials)
    rows = [
        ["ledger head", f"{state.head.seq}:{state.head.hash16}" if state.head else "not available"],
        ["assignments frozen", "yes" if state.manifest else "no"],
        ["budget frozen", "yes" if state.budget else "no"],
        ["v0 spread measured", "yes" if state.spread else "no"],
        ["rule frozen", "yes" if state.rule else "no"],
        ["terms gate", _gate_line(state)],
        ["candidates proposed / decided", f"{len(state.candidates)} / {len(state.decisions)}"],
        ["search closed", "yes" if state.search_closed else "no"],
        ["final analysis frozen", "yes" if state.final_analysis else "no"],
        ["final requests", str(len(state.finals))],
        ["final revealed", "yes" if revealed else "no"],
        ["trials finished by status", ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())) or "none"],
        ["trials started but not finished", str(pending)],
    ]
    sec.table("Where the lab stands", ["item", "value"], rows)
    return sec


def _candidate_outcome(state: LabState, cid: str) -> str:
    dec = state.decisions.get(cid)
    if dec is not None:
        return dec.outcome.value
    cand = state.candidates[cid]
    return "pending" if cand.lint_passed else "invalid"


def _sec_search(state: LabState, rule: Optional[Rule], full: bool) -> Section:
    sec = Section("search", "The search (outer loop)")
    if rule is not None:
        rounds = -(-rule.max_candidates // rule.screen_batch_size)
        bound = stats.false_accept_bound(rule.alpha, rounds)
        sec.para(
            f"Rule: screen up to {rule.screen_batch_size} candidates per round ({rule.screen_trials_per_candidate} trial(s) per assignment each), confirm only the best with "
            f"{rule.confirm_trials_per_candidate} fresh candidate and {rule.confirm_trials_per_incumbent} fresh incumbent trials per assignment, one-sided alpha {rule.alpha:g}, "
            f"failed-trial policy {rule.failed_trial_policy.value}. With up to {rule.max_candidates} candidates in {rounds} round(s), the chance of at least one false accept "
            f"under a true null is nominally {bound:.3f} (1 - (1 - alpha)^rounds) when candidate trials are as noisy as the incumbent's. This is not a ceiling: simulations with candidates up to twice as noisy "
            "(which the variance guard does not reject) gave somewhat higher rates. The rule is a heuristic filter, not evidence of a gain."
        )
    else:
        sec.para("The rule is not frozen: no candidate can be decided.")

    cands = list(state.candidates)
    rows = []
    for cid in cands:
        cand = state.candidates[cid]
        dec = state.decisions.get(cid)
        outcome = _candidate_outcome(state, cid)
        row = [cid, cand.lever.value, _short(cand.hypothesis), "passed" if cand.lint_passed else "failed", outcome, _usd(dec.cost.total_usd if dec else cand.propose_usd)]
        if full:
            if dec is not None and dec.outcome is not Outcome.INVALID:
                row += [dec.stage.value, _num(dec.screen_delta), _num(dec.delta), _num(dec.margin), str(dec.df) if dec.df is not None else "-", _num(dec.sigma_used)]
            else:
                row += [dec.stage.value if dec else "-"] + ["-"] * 5
            row.append(_short(dec.reason, 200) if dec else "")
        rows.append(row)
    headers = ["candidate", "lever", "hypothesis", "lint", "outcome", "cost (candidate-attributed)"]
    if full:
        headers += ["stage", "screen delta", "delta", "margin", "df", "sigma used", "reason"]
    if rows:
        sec.table("Candidates in the order they were proposed" + (" (full view)" if full else " (bits and costs only)"), headers, rows)
    else:
        sec.para("No candidate has been proposed yet.")

    points = []
    for i, cid in enumerate(cands, 1):
        dec = state.decisions.get(cid)
        value = incumbent = None
        if full and dec is not None and dec.outcome is not Outcome.INVALID:
            if dec.candidate_scores:
                value = _mean([t.score for t in dec.candidate_scores])
            if dec.incumbent_scores:
                incumbent = _mean([t.score for t in dec.incumbent_scores])
        points.append(TracePoint(i, cid, _candidate_outcome(state, cid), value, incumbent))
    sec.blocks.append(Block("trace", text=TRACE_TITLE, items=points))
    sec.para(
        "The trace is of an adaptive search: each judgement was made after seeing earlier ones, so it shows what the search did, not that the evolved pack is better."
        + ("" if full else " Scores, deltas and margins are hidden in this view.")
    )
    return sec


def _sec_search_trials(state: LabState) -> Optional[Section]:
    rows = []
    for tid, t in state.trials.items():
        if t.spec.phase is not Phase.SEARCH or t.selection_score_id not in state.scores:
            continue
        rec = state.scores[t.selection_score_id]
        rows.append([tid, t.spec.pack_version, t.spec.assignment_id, t.status.value, _f3(rec.headline_mean()), rec.headline_metric, f"{rec.verifier.value}, {rec.n_prompts} prompts, {len(rec.per_seed)} seeds"])
    if not rows:
        return None
    sec = Section("search_trials", "Search trials (full view)")
    sec.table("Selection score of each scored search trial", ["trial", "pack", "assignment", "status", "score", "metric", "verifier and prompts"], rows)
    return sec


def _sec_spreads(state: LabState, rule: Optional[Rule]) -> Section:
    sec = Section("spreads", "The three spreads")
    rows = []
    sp = state.spread
    # 1. generation seed
    seed_sds = []
    if sp is not None:
        for t in sp.trials:
            rec = state.scores.get(t.score_id) if t.score_id else None
            if rec is not None and len(rec.per_seed) > 1:
                s = stats.seed_spread(rec, sp.metric)
                if s["sd"] is not None:
                    seed_sds.append(s["sd"])
    rows.append(["generation seed (scorer noise)", f"mean sd over seeds within a record {_f4(_mean(seed_sds))} (max {_f4(max(seed_sds))}, {len(seed_sds)} records)" if seed_sds else "n/a"])
    # 2. trial
    if sp is not None:
        e = sp.estimate
        rows.append(["trial (v0, within assignment)", f"sigma {_f4(e.sigma)} with df {e.df} from {len(e.trial_ids)} trial(s) in {e.n_assignments} assignment(s); planned {sp.planned_trials}, failed {sp.n_failed}"])
    else:
        rows.append(["trial (v0, within assignment)", LABEL_SPREAD_NOT_ESTIMATED])
    # 3. assignment
    asp = stats.assignment_spread(sp.trials) if sp is not None else None
    rows.append(["assignment (sd of per-assignment means)", f"sd {_f4(asp['sd'])} over {asp['n_assignments']} assignments" if asp else "not estimable: fewer than two assignments"])
    sec.table("Spread at three levels, kept separate", ["level", "value"], rows)
    if rule is not None:
        sec.para(
            f"Frozen sigma {_f4(rule.spread.sigma)}; the confirmation pools {rule.pooled_df()} degrees of freedom at least. Minimum detectable gain of the rule (80% power): "
            f"{_f4(rule.minimum_detectable_gain)} against the assumed gain {_f4(rule.assumed_gain)}."
        )
    return sec


def _arm_cell(trials: Sequence[TrialScore], mean: Optional[float] = None) -> str:
    """``mean ± sd`` of the trial scores; ``mean`` (the frozen comparison's assignment-weighted mean) replaces the plain trial mean."""
    xs = [t.score for t in trials]
    if mean is None or not xs:
        return f"{_mean_sd(xs)} (n = {len(trials)} trials)"
    sd = f" ± {statistics.stdev(xs):.3f}" if len(xs) > 1 else ""
    return f"{mean:.3f}{sd} (n = {len(trials)} trials)"


def _sec_final(state, fin: Optional[_Final], analysis, revealed: bool, full: bool) -> Section:
    sec = Section("final", "Three-arm results and the held-out comparison")
    if not revealed:
        done = 0
        if analysis is not None:
            have = {r.trial_id for r in state.scores.values() if r.phase is Phase.FINAL and r.trial_id}
            done = len({t.trial_id for t in analysis.planned_trials} & have)
        sec.para(
            "Hidden until lab.final_revealed: per-trial held-out and test scores, and every aggregate made from them, so that trial counts cannot be adapted to results."
            + (f" Progress: {done} of {len(analysis.planned_trials)} planned final trials have a score." if analysis is not None else " The final analysis is not frozen yet.")
        )
        return sec
    if fin is None or not fin.complete:
        sec.para("The final numbers are not available (see Warnings).")
        return sec
    an = fin.analysis
    rows = []
    stock = fin.stock
    if stock is not None:
        vals = stock.metric_by_seed(an.test_metric)
        rows.append(["1  stock Glimmer (no adapter)", "-", f"{_f3(_mean(vals))} (seed sd {_f4(stats.seed_spread(stock, an.test_metric)['sd'])}, {len(vals)} seeds)", f"{stock.verifier.value}, {stock.n_prompts} prompts"])
    else:
        rows.append(["1  stock Glimmer (no adapter)", "-", "not scored", "-"])
    for name, ho, te, mh, mt in (
        ("2  v0 pack", fin.v0_heldout, fin.v0_test, fin.heldout.mean_v0 if fin.heldout else None, fin.test.mean_v0 if fin.test else None),
        ("3  evolved pack", fin.ev_heldout, fin.ev_test, fin.heldout.mean_evolved if fin.heldout else None, fin.test.mean_evolved if fin.test else None),
    ):
        ex = next((state.scores.get(t.score_id) for t in te if t.score_id is not None), None)
        eh = next((state.scores.get(t.score_id) for t in ho if t.score_id is not None), None)
        rows.append([name, _arm_cell(ho, mh), _arm_cell(te, mt), f"held-out {eh.verifier.value if eh else '-'}, {eh.n_prompts if eh else '-'} prompts; test {ex.verifier.value if ex else '-'}, {ex.n_prompts if ex else '-'} prompts"])
    sec.table(
        f"Three arms (held-out metric {an.heldout_metric} on {len(an.assignments)} held-out assignment(s); test metric {an.test_metric}; mean weighted by the frozen assignment weights ± sd over trials)",
        ["arm", f"held-out {an.heldout_metric}", f"test {an.test_metric}", "verifier and prompt count"],
        rows,
    )

    cmp_rows = []
    notes = []
    for label, fc in (("held-out (primary)", fin.heldout), ("IFBench test (secondary)", fin.test)):
        if fc is None:
            cmp_rows.append([label, "not computed", "", "", "", ""])
            continue
        alpha2 = (1.0 - fc.level) / 2.0
        se = _se_of(fc)
        mdg = stats.minimum_detectable_gain_from_se(se, fc.df, alpha2) if se > 0 else None
        cmp_rows.append([
            f"{label}: arm 3 minus arm 2 ({fc.metric})",
            f"{fc.delta:+.4f}",
            f"[{fc.ci_low:+.4f}, {fc.ci_high:+.4f}] ({fc.level:.0%}, {fc.method.value}, df {fc.df:g})",
            f"{fc.n_assignments} assignment(s), {fc.trials_v0} v0 and {fc.trials_evolved} evolved trials",
            f"{mdg:.4f}" if mdg is not None else "n/a (zero standard error)",
            "includes zero" if fc.includes_zero else "excludes zero",
        ])
        notes.append(fc.note)
        if fc.includes_zero:
            notes.append(f"The {label} interval includes zero: this experiment does not show that arm 3 differs from arm 2 on {fc.metric}.")
        else:
            notes.append(f"The {label} interval excludes zero; it is one frozen analysis of {fc.trials_evolved} evolved and {fc.trials_v0} v0 trials and says nothing about other budgets or tasks.")
    sec.table("Confirmatory comparison (frozen before any held-out or test score existed)", ["comparison", "difference", "interval", "design", "minimum detectable gain (80% power)", "zero"], cmp_rows)
    sec.bullets(notes)

    st_rows = []
    for name, sc in (("arm 2 (v0)", fin.stock_vs_v0), ("arm 3 (evolved)", fin.stock_vs_evolved)):
        if sc is not None:
            st_rows.append([name, f"{sc.arm_mean:.4f}", f"{sc.stock_mean:.4f}", f"{sc.delta:+.4f}", f"[{sc.ci_low:+.4f}, {sc.ci_high:+.4f}] ({sc.level:.0%}, df {sc.df})", "includes zero" if sc.includes_zero else "excludes zero"])
    if st_rows:
        sec.table("Arm 1 (stock) versus the trial scores of arms 2 and 3 on the IFBench test metric", ["arm", "mean of trials", "stock", "difference", "interval", "zero"], st_rows)
        sec.para("The stock score is one fixed measurement; the interval reflects only the arm's trial-to-trial spread.")

    if full:
        trows = []
        for arm, ho, te in (("v0", fin.v0_heldout, fin.v0_test), ("evolved", fin.ev_heldout, fin.ev_test)):
            te_by = {t.trial_id: t for t in te}
            for t in ho:
                trows.append([t.trial_id, arm, t.assignment_id, _f3(t.score), _f3(te_by[t.trial_id].score) if t.trial_id in te_by else "n/a"])
        sec.table("Per-trial final scores (full view, after reveal)", ["trial", "arm", "assignment", f"held-out {an.heldout_metric}", f"test {an.test_metric}"], trows)
    return sec


def _arm_prompt_rates(layout, state, fin: _Final, arm_key: str, metric: str, warnings: list) -> Optional[dict]:
    field_name = _FIELD_OF_METRIC.get(metric)
    if field_name is None:
        return None
    trials: list = []
    if arm_key == "stock":
        recs = [fin.stock] if fin.stock is not None else []
    else:
        trials = fin.v0_test if arm_key == "v0" else fin.ev_test
        recs = [fin.records[(t.trial_id, Split.TEST)] for t in trials if t.score_id is not None and (t.trial_id, Split.TEST) in fin.records]
    # a trial the frozen failure policy scored as 0.0 has no record: it counts as failing every prompt, as in the arm table
    n_zero = sum(1 for t in trials if t.score_id is None)
    acc: dict = {}
    for rec in recs:
        rows = _read_per_prompt(layout, rec, warnings)
        if rows is None:
            return None
        per_trial: dict = {}
        for r in rows:
            per_trial.setdefault(r["prompt_key"], []).append(1.0 if (bool(r.get(field_name)) and not r.get("error", False)) else 0.0)
        for k, v in per_trial.items():  # one value per trial (mean over its seeds), so a zero trial weighs the same as a scored one
            acc.setdefault(k, []).append(_mean(v))
    if not acc:
        return None
    for v in acc.values():
        v.extend([0.0] * n_zero)
    return {k: _mean(v) for k, v in acc.items()}


def _sec_per_prompt(state, layout, fin: _Final, warnings: list) -> Optional[Section]:
    metric = fin.analysis.test_metric
    rates = {a: _arm_prompt_rates(layout, state, fin, a, metric, warnings) for a in ("stock", "v0", "evolved")}
    rows = []
    for a, b, name in (("stock", "v0", "arm 1 to arm 2"), ("v0", "evolved", "arm 2 to arm 3"), ("stock", "evolved", "arm 1 to arm 3")):
        ra, rb = rates[a], rates[b]
        if ra is None or rb is None:
            continue
        try:
            boot = stats.paired_bootstrap_prompts(ra, rb, seed=20261104)
            mc = stats.mcnemar({k: v >= 0.5 for k, v in ra.items()}, {k: v >= 0.5 for k, v in rb.items()})
        except ContractError as e:
            warnings.append(f"per-prompt comparison {name} could not be computed: {e}")
            continue
        rows.append([name, str(boot["n_prompts"]), f"{boot['delta']:+.4f}", f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] ({boot['resamples']} resamples)", f"{mc['a_only']} / {mc['b_only']}", f"{mc['p_value']:.4f}"])
    if not rows:
        return None
    sec = Section("per_prompt", "Paired per-prompt comparison between arms (checkpoint-level)")
    sec.para(
        f"Each arm is reduced to a per-prompt pass rate on the IFBench test metric {metric} (mean over its trials and seeds). The bootstrap resamples prompts and McNemar counts prompts "
        "that pass in the majority of an arm's generations. These describe two checkpoint-level results on identical prompts; they ignore trial-to-trial variation and never decide a keep."
    )
    sec.table("Prompt-level diagnostics", ["pair", "prompts", "difference in pass rate", "bootstrap interval", "only first / only second passes", "McNemar p"], rows)
    return sec


def _counts_visible(state: LabState, split, phase) -> bool:
    """Whether the truncation and error counts of a score record may be printed. They are outcome-derived: a selection score of exactly zero
    shows as every generation truncated. Dev records are what the researchers see; search-phase selection records are hidden until
    ``lab.search_closed`` and final-phase records until ``lab.final_revealed``."""
    if phase in (Phase.FINAL, Phase.FINAL.value):
        return state.reveal is not None
    if split in (Split.SELECTION, Split.SELECTION.value):
        return state.search_closed is not None
    return True


def _hidden_until(phase) -> str:
    return "hidden until reveal" if phase == Phase.FINAL.value else "hidden until search closed"


def _sec_scoring(state: LabState) -> Section:
    sec = Section("scoring", "What the scores are made of")
    groups: dict = {}
    for r in state.scores.values():
        k = (r.split.value, r.phase.value, r.verifier.value, r.n_prompts, len(r.per_seed), r.scorer_version)
        g = groups.setdefault(k, [0, 0, 0])
        g[0] += 1
        g[1] += sum(s.n_truncated for s in r.per_seed)
        g[2] += sum(s.n_errors for s in r.per_seed)
    # outcome-derived: selection and final-phase truncation and error counts would disclose the hidden scores
    rows = [
        [*k[:5], k[5], v[0], v[1], v[2]] if _counts_visible(state, k[0], k[1]) else [*k[:5], k[5], v[0], *([_hidden_until(k[1])] * 2)]
        for k, v in sorted(groups.items())
    ]
    if rows:
        sec.table("Score records by split, phase, verifier and prompt count (counts only; no score values)", ["split", "phase", "verifier", "prompts per seed", "seeds", "scorer version", "records", "truncated", "errors"], rows)
    else:
        sec.para("No score has been recorded yet.")
    return sec


def _gpu_ledger_usd(facts: Optional[TrialFacts]):
    if facts is None:
        return None
    return math.fsum(j.gpu.usd for j in facts.jobs), math.fsum(j.gpu.minutes for j in facts.jobs)


def _sec_costs(state: LabState, trial_facts: Mapping[str, TrialFacts]) -> Section:
    sec = Section("costs", "Cost")
    phase_of_score = {sid: r.phase for sid, r in state.scores.items()}
    rows = []
    notes = []

    def add_phase(phase: Phase, label: str) -> None:
        trials = [t for t in state.trials.values() if t.spec.phase is phase]
        res_usd = math.fsum(t.budget_used.researcher_usd for t in trials)
        meter_gpu_usd = math.fsum(t.budget_used.gpu_usd for t in trials)
        meter_gpu_min = math.fsum(t.budget_used.gpu_minutes for t in trials)
        tokens = sum(t.budget_used.researcher_tokens for t in trials)
        led = [_gpu_ledger_usd(trial_facts.get(t.spec.trial_id)) for t in trials]
        if trials and all(x is not None for x in led):
            led_usd, led_min = math.fsum(x[0] for x in led), math.fsum(x[1] for x in led)
            led_cell, led_min_cell = _usd(led_usd), f"{led_min:.0f}"
            if abs(led_usd - meter_gpu_usd) > 0.005 or abs(led_min - meter_gpu_min) > 0.5:
                notes.append(f"{label}: the meter's GPU ticks ({_usd(meter_gpu_usd)}, {meter_gpu_min:.0f} min, quoted rate, charged from the create attempt) differ from the rental ledger ({led_cell}, {led_min_cell} min, rounded-up minutes); both are shown.")
        else:
            led_cell, led_min_cell = ("n/a (no trial ledger read)" if trials else "-"), "-"
        scoring = math.fsum(c.gpu.usd for sid, c in state.score_costs.items() if phase_of_score.get(sid) is phase)
        rows.append([f"{label}: researcher inference", _usd(res_usd), "-", f"{tokens:,} tokens, {len(trials)} trial(s)"])
        rows.append([f"{label}: GPU (training jobs and dev scoring)", _usd(meter_gpu_usd), led_cell, f"meter {meter_gpu_min:.0f} min; ledger {led_min_cell} min"])
        rows.append([f"{label}: scoring (selection and final scorer rentals)", "-", _usd(scoring), f"{sum(1 for sid in state.score_costs if phase_of_score.get(sid) is phase)} scoring session(s), rental ledger"])

    add_phase(Phase.SEARCH, "search (candidate trials)")
    propose = math.fsum(c.propose_usd for c in state.candidates.values())
    rows.append(["search: outer proposals (Wonder)", "-", _usd(propose), f"{len(state.candidates)} proposal(s), metered client"])
    add_phase(Phase.FINAL, "final (budget-matched trials)")
    sec.table("Cost centres, kept apart (meter = enforcement ticks; rental ledger = money)", ["cost centre", "meter", "rental ledger / metered", "detail"], rows)
    sec.para(
        "Budget enforcement uses the meter's one-minute ticks; money reporting uses the rental ledger's rounded-up minutes. The two are printed whenever they differ. "
        "LLM overshoot is bounded by concurrent researchers times the error of the proxy's prompt-token estimate (the worst case is reserved before each call); GPU overshoot by one tick (one minute). "
        "The measured estimate error is not available from the ledgers read here."
    )
    if notes:
        sec.bullets(notes)
    return sec


def _isolation_of(t) -> str:
    env = t.environment
    return env.get("isolation") or env.get("isolation_level") or "not recorded"


def _sec_trials(state: LabState) -> Section:
    sec = Section("trials", "Trials and isolation")
    rows = []
    for tid, spec in state.trials_started.items():
        rec = state.trials.get(tid)
        rows.append([
            tid,
            spec.phase.value + (f" / {spec.arm.value}" if spec.arm else ""),
            spec.pack_version,
            spec.assignment_id,
            rec.status.value if rec else "started, not finished",
            _isolation_of(rec) if rec else "not recorded",
            f"{rec.budget_used.researcher_tokens:,} tok, {rec.budget_used.gpu_minutes:.0f} GPU min, {_usd(rec.budget_used.dollars)}" if rec else "-",
        ])
    if rows:
        sec.table("Every trial with its isolation level (no scores)", ["trial", "phase / arm", "pack", "assignment", "status", "isolation level", "budget used (meter)"], rows)
        levels = sorted({_isolation_of(t) for t in state.trials.values()})
        sec.para("Isolation levels recorded: " + (", ".join(levels) if levels else "none") + ". Trials run serially in one shared unix user are labelled as such; they could read each other's files.")
    else:
        sec.para("No trial has started.")
    return sec


def _sec_caveats(state, rule, manifest, trial_facts) -> Section:
    sec = Section("caveats", "Residual risks")
    items = []
    if manifest is not None:
        if manifest.item_mode is ItemMode.FRESH_INSTANTIATION:
            items.append("Selection and held-out items are fresh constraint instantiations: their verbatim strings are not in the public dataset, which lowers (does not remove) the chance that training data contains them.")
        else:
            items.append("Selection and held-out items were drawn from the public dataset with a secret salt (salted draw): a researcher who retrieves that dataset could train on them. The contamination scan is the defence, not a guarantee.")
    else:
        items.append("The assignment manifest is not available, so how selection and held-out items were made is unknown.")
    scans = list(state.contamination.values())
    hits = [s for s in scans if not s.passed]
    contaminated = [t for t in state.trials.values() if t.status is TrialStatus.CONTAMINATED]
    items.append(f"Contamination scans: {len(scans)} recorded, {len(hits)} with hits; {len(contaminated)} trial(s) marked contaminated (counted as failed trials).")
    items.append("Residual contamination risk: a determined researcher who routes data around the glimmer helpers and never trips the scan is not detected; recipes run as root on rentals.")
    items.append(f"{LABEL_DEV_INFLUENCED[0].upper()}{LABEL_DEV_INFLUENCED[1:]}; the recipe that produced a dev score ran with full control of its rental.")
    items.append(f"Adaptive channels: {LABEL_ADAPTIVE}. Operator changes after the first candidate: {len(state.operator_changes)}" + ("." if not state.operator_changes else " (" + "; ".join(f"{o.target}" for o in state.operator_changes[:6]) + ")."))
    items.append("Money: the meter enforces, the rental ledger reports; the first trial's token and dollar totals should be reconciled against the provider's billing before the token axis is trusted.")
    sec.bullets(items)
    return sec


def _sec_warnings(warnings: Sequence[str]) -> Section:
    sec = Section("warnings", "Warnings")
    if warnings:
        sec.bullets(warnings)
    else:
        sec.para("None.")
    return sec


# --------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------
def _md_cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _trace_text(points: Sequence[TracePoint]) -> str:
    return " -> ".join(f"{p.label} {p.outcome}" for p in points) if points else "no candidates yet"


def render_markdown(report: Report) -> str:
    out = [f"# {report.title}", ""]
    out.append(f"Ledger head: {report.head or 'not available'}. View: {'full' if report.full else 'default (bits and costs during the search)'}.")
    out.append("")
    for sec in report.sections:
        out.append(f"## {sec.title}")
        out.append("")
        for b in sec.blocks:
            if b.kind == "para":
                out += [b.text, ""]
            elif b.kind == "bullets":
                out += [f"- {_md_cell(i)}" for i in b.items] + [""]
            elif b.kind == "table" and b.table is not None:
                t = b.table
                out += [f"**{t.caption}**", "", "| " + " | ".join(_md_cell(h) for h in t.headers) + " |", "|" + "---|" * len(t.headers)]
                out += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in t.rows] + [""]
            elif b.kind == "trace":
                out += [f"### {b.text}", "", _trace_text(b.items), "", "(the SVG chart is in the HTML report)", ""]
    return "\n".join(out).rstrip() + "\n"


_CSS = """
:root{--bg:#ffffff;--ink:#1f2328;--muted:#59636e;--line:#d1d9e0;--head:#f6f8fa;--link:#0969da}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d1117;--ink:#e6edf3;--muted:#9198a1;--line:#30363d;--head:#161b22;--link:#58a6ff}}
:root[data-theme="dark"]{--bg:#0d1117;--ink:#e6edf3;--muted:#9198a1;--line:#30363d;--head:#161b22;--link:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.15rem;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.meta{color:var(--muted);margin:0 0 12px}
.wrap{overflow-x:auto;margin:8px 0 16px}
table{border-collapse:collapse;width:100%;font-size:.9rem}
caption{text-align:left;font-weight:600;padding:4px 0}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
th{background:var(--head)}
ul{padding-left:20px}li{margin:3px 0}
"""


def render_html(report: Report) -> str:
    e = html.escape
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">']
    out.append(f"<title>{e(report.title)}</title><style>{_CSS}</style></head><body><main>")
    out.append(f"<h1>{e(report.title)}</h1>")
    out.append(f'<p class="meta">Ledger head: {e(report.head or "not available")}. View: {"full" if report.full else "default (bits and costs during the search)"}.</p>')
    for sec in report.sections:
        out.append(f'<section id="{e(sec.key)}"><h2>{e(sec.title)}</h2>')
        for b in sec.blocks:
            if b.kind == "para":
                out.append(f"<p>{e(b.text)}</p>")
            elif b.kind == "bullets":
                out.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b.items) + "</ul>")
            elif b.kind == "table" and b.table is not None:
                t = b.table
                out.append(f'<div class="wrap"><table><caption>{e(t.caption)}</caption><thead><tr>' + "".join(f"<th>{e(h)}</th>" for h in t.headers) + "</tr></thead><tbody>")
                out += ["<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in t.rows]
                out.append("</tbody></table></div>")
            elif b.kind == "trace":
                out.append(f'<div class="wrap">{search_trace_svg(b.items, b.text)}</div>')
        out.append("</section>")
    out.append("</main></body></html>")
    return "".join(out)
