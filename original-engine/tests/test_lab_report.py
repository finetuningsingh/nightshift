"""Tests for nightshift.lab.report and nightshift.lab.chart (owner 3).

The states come from ``tests/support/lab_state_builder.py`` (a real experiment's records with real files behind them, decisions computed
by the real ``stats``). The hidden-until-reveal and bits-only checks search the rendered text for the exact numbers that must not appear,
and each has a positive control (the same numbers do appear in the view that is allowed to show them), so none is vacuous.

Run:  PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_report
"""

import dataclasses
import json
import os
import re
import tempfile
import unittest
import xml.dom.minidom
from pathlib import Path

from nightshift.lab import chart, report, stats
from nightshift.lab import contracts as c
from nightshift.lab.chart import TracePoint, search_trace_svg
from nightshift.lab.contracts import AccessDenied, Arm, BudgetCharge, ItemMode, Resource, Split, TrialStatus
from tests.support.lab_state_builder import LabStateBuilder


def render(rep):
    return report.render_markdown(rep) + "\n" + report.render_html(rep)


def numbers(values, *, places=(3, 4)):
    out = set()
    for v in values:
        for p in places:
            out.add(f"{v:.{p}f}")
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self._n = 0

    def make(self, **kw):
        self._n += 1
        root = self.dir / f"lab{self._n}"
        kw.setdefault("ledgers", True)
        b = LabStateBuilder.standard(root, **kw)
        return b, b.build(), b.layout

    def assertAbsent(self, needles, text, what=""):
        found = sorted(n for n in needles if n in text)
        self.assertEqual(found, [], f"{what}: these must not be printed: {found}")

    def assertPresent(self, needles, text, what=""):
        missing = sorted(n for n in needles if n not in text)
        self.assertEqual(missing, [], f"{what}: these must be printed: {missing}")


# --------------------------------------------------------------------------------------------------
class RenderTests(Base):
    def test_a_table_and_an_svg_chart_render_from_a_synthetic_state(self):
        b, st, lay = self.make()
        rep = report.build_report(st, lay, full=True)
        md, html_ = report.render_markdown(rep), report.render_html(rep)
        # the three-arm table, the candidate table, the cost table and the chart title
        for needle in ("## Three-arm results and the held-out comparison", "stock Glimmer", "2 v0 pack", "3 evolved pack", "| candidate |", "cand-002 | context_notes", "## Cost", "### adaptive search trace"):
            self.assertIn(needle, md)
        self.assertIn("<svg", html_)
        self.assertIn("adaptive search trace", html_)
        svg = re.search(r"<svg.*?</svg>", html_, re.S).group(0)
        xml.dom.minidom.parseString(svg)  # well-formed
        self.assertIn("<table>", html_)
        self.assertTrue(html_.startswith("<!doctype html>"))
        self.assertIn('name="viewport"', html_)

    def test_a_risk_waiver_is_reported_as_an_open_terms_gate(self):
        b, st, lay = self.make()  # the builder records PROCEED_WITH_RISK
        md = report.render_markdown(report.build_report(st, lay, full=True))
        self.assertIn("OPEN: proceeding at the owner's stated risk (not closed)", md)
        self.assertNotIn("closed (clarified)", md)

    def test_the_chart_marks_accepted_rejected_and_invalid_candidates(self):
        b, st, lay = self.make()
        html_ = report.render_html(report.build_report(st, lay, full=True))
        svg = re.search(r"<svg.*?</svg>", html_, re.S).group(0)
        self.assertEqual(svg.count('class="st-acc"'), 2)  # cand-002 and the legend entry
        self.assertEqual(svg.count('class="st-rej"'), 4)  # cand-001, cand-004, cand-005 and the legend entry
        self.assertEqual(svg.count("<polygon"), 2)  # accepted marker and its legend entry
        self.assertIn("cand-003: invalid", svg)
        self.assertIn("cand-002: accepted, score", svg)  # full view: the score is in the tooltip

    def test_render_is_deterministic(self):
        b, st, lay = self.make()
        self.assertEqual(render(report.build_report(st, lay, full=True)), render(report.build_report(st, lay, full=True)))

    def test_text_that_came_from_a_model_is_escaped(self):
        b, st, lay = self.make()
        evil = "<script>alert(1)</script> | pipe & \"quote\"\nnewline"
        cand = st.candidates["cand-001"]
        st.candidates["cand-001"] = dataclasses.replace(cand, hypothesis=evil)
        rep = report.build_report(st, lay)
        md, html_ = report.render_markdown(rep), report.render_html(rep)
        self.assertNotIn("<script>", html_)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html_)
        self.assertIn("\\| pipe", md)
        row = [ln for ln in md.splitlines() if ln.startswith("| cand-001")][0]  # the newline did not split the row
        cells = re.split(r"(?<!\\)\|", row)
        self.assertEqual(len(cells) - 2, len(report.build_report(st, lay).section("search").blocks[1].table.headers))

    def test_two_held_out_assignments_use_the_blocked_t_comparison(self):
        b, st, lay = self.make(n_heldout=2)
        md = report.render_markdown(report.build_report(st, lay))
        self.assertIn("blocked_t", md)
        self.assertIn("Blocked over 2 held-out assignments", md)
        self.assertIn("2 assignment(s), 6 v0 and 6 evolved trials", md)

    def test_one_held_out_assignment_says_it_does_not_generalise(self):
        b, st, lay = self.make()
        md = report.render_markdown(report.build_report(st, lay))
        self.assertIn("Within one assignment; does not generalise across assignments", md)
        self.assertIn("welch", md)


class FalseAcceptWordingTests(Base):
    def test_the_false_accept_figure_is_called_nominal_not_a_ceiling(self):
        b, st, lay = self.make()
        text = report.render_markdown(report.build_report(st, lay))
        self.assertIn("nominally", text)
        self.assertIn("not a ceiling", text)
        self.assertNotRegex(text, r"false accept\s+under a true null is at most")


class EmptyStateTests(Base):
    def test_an_empty_state_renders_without_files(self):
        lay = c.LabLayout(self.dir / "nothing")
        rep = report.build_report(c.LabState(), lay)
        text = render(rep)
        self.assertIn("components tested, system not yet demonstrated", text)
        self.assertIn("arm 3 versus arm 2 was not tested", text)
        self.assertIn("spread not estimated", text)
        self.assertIn("No candidate has been proposed yet.", text)
        self.assertIn("No trial has started.", text)
        self.assertIn("No score has been recorded yet.", text)
        svg = re.search(r"<svg.*?</svg>", text, re.S).group(0)
        self.assertIn("no candidates yet", svg)
        with self.assertRaises(AccessDenied):
            report.build_report(c.LabState(), lay, full=True)


# --------------------------------------------------------------------------------------------------
class BitsOnlyTests(Base):
    """During the search the default view shows bits and costs only (design: Report; open decision 7)."""

    def search_numbers(self, b, st):
        nums = set()
        for rec in st.scores.values():
            if rec.phase is c.Phase.SEARCH:
                nums |= numbers([rec.headline_mean()])
        for d in st.decisions.values():
            nums |= numbers([x for x in (d.screen_delta, d.delta, d.margin, d.sigma_used) if x is not None], places=(4,))
        nums |= numbers([st.spread.estimate.sigma, b.rule.minimum_detectable_gain, b.rule.assumed_gain], places=(4,))
        return nums

    def test_the_default_view_shows_no_selection_number_and_the_full_view_does(self):
        b, st, lay = self.make(close=False, final=False, reveal=False)
        nums = self.search_numbers(b, st)
        self.assertGreater(len(nums), 20)
        default = render(report.build_report(st, lay))
        self.assertAbsent(nums, default, "default view")
        self.assertAbsent({"exceeds margin", "does not exceed margin", "lost its screening round", "sigma_used"}, default, "decision reasons are not printed in the default view")
        self.assertIn("outcome", default)
        self.assertPresent({"accepted", "rejected", "invalid"}, default)
        self.assertPresent({"$33.10"}, default, "costs are shown")

    def test_the_default_view_stays_bits_only_after_the_search_is_closed(self):
        b, st, lay = self.make(final=False, reveal=False)  # search closed
        default = render(report.build_report(st, lay))
        self.assertAbsent(self.search_numbers(b, st), default)
        full = render(report.build_report(st, lay, full=True))
        # positive control: the allowed view prints them
        self.assertPresent({f"{st.decisions['cand-002'].delta:.4f}", f"{st.decisions['cand-002'].margin:.4f}", f"{st.spread.estimate.sigma:.4f}", "0.665"}, full)

    def test_the_default_chart_has_no_scores(self):
        b, st, lay = self.make(close=False, final=False, reveal=False)
        html_ = report.render_html(report.build_report(st, lay))
        svg = re.search(r"<svg.*?</svg>", html_, re.S).group(0)
        self.assertIn("Bit-only view", svg)
        self.assertNotIn("incumbent", svg)
        self.assertNotRegex(svg, r"score 0\.\d+")

    def test_full_is_refused_before_search_closed_and_allowed_after(self):
        b, st, lay = self.make(close=False, final=False, reveal=False)
        with self.assertRaises(AccessDenied):
            report.build_report(st, lay, full=True)
        report.build_report(st, lay)  # the default view is always available
        b2, st2, lay2 = self.make(final=False, reveal=False)
        report.build_report(st2, lay2, full=True)

    def test_no_per_prompt_row_or_ground_truth_is_ever_printed(self):
        b, st, lay = self.make()
        text = render(report.build_report(st, lay, full=True))
        self.assertAbsent({"p000", "p001", "ground_truth", "strict_instructions", '"passed"'}, text)


# --------------------------------------------------------------------------------------------------
class HiddenUntilRevealTests(Base):
    def final_numbers(self, st):
        nums = set()
        for rec in st.scores.values():
            if rec.phase is c.Phase.FINAL:
                nums |= numbers([rec.headline_mean()] + [s.metrics[rec.headline_metric] for s in rec.per_seed])
        return nums

    def test_no_per_trial_or_aggregate_final_score_before_reveal(self):
        b, st, lay = self.make(reveal=False)
        nums = self.final_numbers(st)
        self.assertGreater(len(nums), 20)
        for full in (False, True):
            text = render(report.build_report(st, lay, full=full))
            self.assertAbsent(nums, text, f"before reveal, full={full}")
            self.assertAbsent({"0.373", "0.3733"}, text, "the stock test score is hidden too")
            self.assertIn("Hidden until lab.final_revealed", text)
            self.assertIn("10 of 10 planned final trials have a score", text)

    def test_after_reveal_the_aggregates_show_and_the_per_trial_rows_only_in_the_full_view(self):
        b, st, lay = self.make(reveal=True)
        default = render(report.build_report(st, lay))
        full = render(report.build_report(st, lay, full=True))
        # positive control: aggregates (arm means) are printed in both views
        self.assertPresent({"0.780 ± 0.008", "0.835 ± 0.008", "0.373"}, default)
        # per-trial rows (e.g. the single trial score 0.775 / 0.845) only in the full view
        self.assertAbsent({"0.775", "0.845", "0.4033", "0.403 "}, default, "default view after reveal")
        self.assertPresent({"0.775", "0.845", "t-f-v0-1-0"}, full, "full view after reveal")

    def test_arm_3_versus_arm_2_is_labelled_untested_until_it_exists(self):
        b, st, lay = self.make(reveal=False)
        self.assertIn("arm 3 versus arm 2 was not tested", render(report.build_report(st, lay)))
        b2, st2, lay2 = self.make(reveal=True)
        self.assertNotIn("arm 3 versus arm 2 was not tested", render(report.build_report(st2, lay2)))

    def test_incomplete_final_scores_give_no_comparison_and_a_warning(self):
        b, st, lay = self.make(reveal=True)
        victim = next(sid for sid, r in st.scores.items() if r.phase is c.Phase.FINAL and r.trial_id == "t-f-evolved-1-2" and r.split is Split.SELECTION)
        del st.scores[victim]
        rep = report.build_report(st, lay)
        text = render(rep)
        self.assertTrue(any("final scores are incomplete" in w for w in rep.warnings))
        self.assertIn("arm 3 versus arm 2 was not tested", text)
        self.assertNotIn("Confirmatory comparison", text)


# --------------------------------------------------------------------------------------------------
class FinalPolicyAndWeightsTests(Base):
    def test_a_failed_final_trial_counts_as_zero_under_the_scored_zero_policy(self):
        b, st, lay = self.make()
        victim = "t-f-evolved-1-2"
        for sid in [sid for sid, r in st.scores.items() if r.trial_id == victim]:
            del st.scores[sid]
        st.trials[victim] = dataclasses.replace(st.trials[victim], status=TrialStatus.FAILED, adapter=None, submission=None, selection_score_id=None, failure="the trainer crashed")
        # the frozen policy is INVALIDATES_CANDIDATE: the plan stays incomplete (positive control)
        warnings = []
        self.assertFalse(report.final_numbers(st, b.analysis, warnings).complete)
        zero = dataclasses.replace(b.analysis, failed_trial_policy=c.FailedTrialPolicy.SCORED_ZERO)
        warnings = []
        fin = report.final_numbers(st, zero, warnings)
        self.assertTrue(fin.complete, (fin.missing, warnings))
        zeros = [t for t in fin.ev_heldout if t.trial_id == victim]
        self.assertEqual([(t.score, t.score_id) for t in zeros], [(0.0, None)])
        self.assertEqual([t.score for t in fin.ev_test if t.trial_id == victim], [0.0])
        self.assertIsNotNone(fin.heldout)
        self.assertEqual(len(fin.ev_heldout), 5)

    def test_per_prompt_rates_include_the_trials_scored_zero_by_the_failure_policy(self):
        b, st, lay = self.make()
        victim = "t-f-evolved-1-2"
        for sid in [sid for sid, r in st.scores.items() if r.trial_id == victim]:
            del st.scores[sid]
        st.trials[victim] = dataclasses.replace(st.trials[victim], status=TrialStatus.FAILED, adapter=None, submission=None, selection_score_id=None, failure="the trainer crashed")
        zero = dataclasses.replace(b.analysis, failed_trial_policy=c.FailedTrialPolicy.SCORED_ZERO)
        fin = report.final_numbers(st, zero, [])
        self.assertTrue(fin.complete)
        rates = report._arm_prompt_rates(lay, st, fin, "evolved", "prompt_strict", [])
        # every test record scores the same prompts, so the mean per-prompt pass rate is the plain mean of the trial scores, the zero trial included
        trial_mean = sum(t.score for t in fin.ev_test) / len(fin.ev_test)
        self.assertAlmostEqual(sum(rates.values()) / len(rates), trial_mean, places=9)
        scored = [t.score for t in fin.ev_test if t.score_id is not None]
        self.assertGreater(sum(scored) / len(scored), trial_mean + 0.01, "positive control: leaving the zero trial out would raise the mean")

    def test_a_trial_that_has_not_finished_is_still_missing_under_scored_zero(self):
        b, st, lay = self.make()
        victim = "t-f-evolved-1-2"
        for sid in [sid for sid, r in st.scores.items() if r.trial_id == victim]:
            del st.scores[sid]
        del st.trials[victim]
        zero = dataclasses.replace(b.analysis, failed_trial_policy=c.FailedTrialPolicy.SCORED_ZERO)
        fin = report.final_numbers(st, zero, [])
        self.assertFalse(fin.complete)

    def test_the_three_arm_table_uses_the_frozen_assignment_weights(self):
        b, st, lay = self.make(final=False, reveal=False, n_heldout=2)
        weights = (c.AssignmentWeight("a-held-1", 0.9), c.AssignmentWeight("a-held-2", 0.1))
        b.freeze_final_analysis(per_cell=3, assignments=weights)
        b.request_final()
        held = {"v0": {"a-held-1": [0.9] * 3, "a-held-2": [0.1] * 3}, "ev": {"a-held-1": [0.8] * 3, "a-held-2": [0.6] * 3}}
        test = {"a-held-1": [0.4] * 3, "a-held-2": [0.4] * 3}
        b.add_final_trials(held["v0"], held["ev"], test, test)
        b.add_stock_score(0.3733)
        b.reveal()
        st = b.build()
        rep = report.build_report(st, b.layout)
        tables = {bl.table.caption.split(" (")[0]: bl.table for bl in rep.section("final").blocks if bl.kind == "table"}
        arms = tables["Three arms"]
        v0_cell = [r for r in arms.rows if r[0].startswith("2")][0][1]
        ev_cell = [r for r in arms.rows if r[0].startswith("3")][0][1]
        self.assertTrue(v0_cell.startswith("0.820"), v0_cell)  # 0.9 * 0.9 + 0.1 * 0.1, not the plain mean 0.500
        self.assertTrue(ev_cell.startswith("0.780"), ev_cell)  # 0.9 * 0.8 + 0.1 * 0.6, not the plain mean 0.700
        cmp_row = [bl.table for bl in rep.section("final").blocks if bl.kind == "table" and bl.table.caption.startswith("Confirmatory")][0].rows[0]
        self.assertEqual(cmp_row[1], "-0.0400")  # the table and the frozen comparison now agree: 0.78 - 0.82

    def test_the_stock_comparison_uses_the_frozen_assignment_weights(self):
        b, st, lay = self.make(final=False, reveal=False, n_heldout=2)
        weights = (c.AssignmentWeight("a-held-1", 0.9), c.AssignmentWeight("a-held-2", 0.1))
        b.freeze_final_analysis(per_cell=3, assignments=weights)
        b.request_final()
        held = {"a-held-1": [0.5, 0.5, 0.5], "a-held-2": [0.5, 0.5, 0.5]}
        test_v0 = {"a-held-1": [0.9, 0.9, 0.9], "a-held-2": [0.1, 0.1, 0.1]}
        test_ev = {"a-held-1": [0.5, 0.5, 0.5], "a-held-2": [0.5, 0.5, 0.5]}
        b.add_final_trials(held, held, test_v0, test_ev)
        b.add_stock_score(0.6)
        b.reveal()
        st = b.build()
        fin = report.final_numbers(st, b.analysis, [])
        sc = fin.stock_vs_v0
        self.assertAlmostEqual(sc.arm_mean, 0.82, places=9)  # 0.9 * 0.9 + 0.1 * 0.1; the plain trial mean is 0.5
        self.assertAlmostEqual(sc.delta, 0.22, places=9)  # not 0.5 - 0.6 = -0.1
        self.assertEqual(sc.df, 1)  # blocked over the two assignments
        self.assertAlmostEqual(fin.stock_vs_evolved.arm_mean, 0.5, places=9)
        # the same figures reach the rendered stock table
        rep = report.build_report(st, b.layout)
        tbl = [bl.table for bl in rep.section("final").blocks if bl.kind == "table" and bl.table.caption.startswith("Arm 1 (stock)")][0]
        row = [r for r in tbl.rows if r[0].startswith("arm 2")][0]
        self.assertEqual((row[1], row[3]), ("0.8200", "+0.2200"))

    def test_the_stock_row_and_comparison_use_the_frozen_test_metric_not_the_records_headline(self):
        b, st, lay = self.make(final=False, reveal=False)
        b.freeze_final_analysis(per_cell=5, test_metric="prompt_loose")
        b.request_final()
        held = {"a-held-1": [0.8] * 5}
        b.add_final_trials(held, held, {"a-held-1": [0.60, 0.61, 0.62, 0.63, 0.64]}, {"a-held-1": [0.70, 0.71, 0.72, 0.73, 0.74]})
        stock = b.add_stock_score(0.30)  # the stock record's headline is prompt_strict 0.30; its prompt_loose is 0.31
        b.reveal()
        st = b.build()
        self.assertEqual(stock.headline_metric, "prompt_strict")
        warnings = []
        fin = report.final_numbers(st, b.analysis, warnings)
        self.assertEqual(warnings, [])
        self.assertEqual(fin.stock_vs_v0.metric, "prompt_loose")
        self.assertAlmostEqual(fin.stock_vs_v0.stock_mean, 0.31, places=9)
        rep = report.build_report(st, b.layout)
        arms = [bl.table for bl in rep.section("final").blocks if bl.kind == "table" and bl.table.caption.startswith("Three arms")][0]
        self.assertTrue([r for r in arms.rows if r[0].startswith("1")][0][2].startswith("0.310"), arms.rows)


# --------------------------------------------------------------------------------------------------
class BlindedCountsTests(Base):
    def truncate_final(self, st):
        for sid, r in list(st.scores.items()):
            if r.phase is c.Phase.FINAL and r.split is Split.TEST and r.trial_id == "t-f-v0-1-0":
                st.scores[sid] = dataclasses.replace(r, per_seed=tuple(dataclasses.replace(s, n_truncated=r.n_prompts) for s in r.per_seed))
                return r.n_prompts * len(r.per_seed)
        raise AssertionError("no such record")

    def scoring_rows(self, st, lay):
        return [bl.table.rows for bl in report.build_report(st, lay).section("scoring").blocks if bl.kind == "table"][0]

    def test_truncation_and_error_counts_of_final_records_are_hidden_until_reveal(self):
        b, st, lay = self.make(reveal=False)
        total = self.truncate_final(st)
        rep = report.build_report(st, lay)
        rows = self.scoring_rows(st, lay)
        final_rows = [r for r in rows if r[1] == "final"]
        self.assertTrue(final_rows)
        for r in final_rows:
            self.assertEqual(r[7:], ["hidden until reveal", "hidden until reveal"], r)
        self.assertNotIn(str(total), render(rep))
        self.assertFalse(any("truncated generation" in x for x in rep.labels), rep.labels)

    def test_counts_are_printed_after_reveal_and_for_search_records_before(self):
        b, st, lay = self.make(reveal=True)
        total = self.truncate_final(st)
        rep = report.build_report(st, lay)
        final_rows = [r for r in self.scoring_rows(st, lay) if r[1] == "final" and r[0] == "test"]
        self.assertIn(str(total), [r[7] for r in final_rows])
        self.assertTrue(any(f"{total} truncated generation" in x for x in rep.labels), rep.labels)
        # search records stay visible before the reveal
        b2, st2, lay2 = self.make(reveal=False)
        sid = next(sid for sid, r in st2.scores.items() if r.phase is c.Phase.SEARCH)
        r = st2.scores[sid]
        st2.scores[sid] = dataclasses.replace(r, per_seed=tuple(dataclasses.replace(s, n_truncated=1) for s in r.per_seed))
        rep2 = report.build_report(st2, lay2)
        self.assertTrue(any("truncated generation" in x for x in rep2.labels), rep2.labels)
        self.assertTrue(any(r_[1] == "search" and r_[7].isdigit() and int(r_[7]) > 0 for r_ in self.scoring_rows(st2, lay2)))


    def test_search_selection_counts_are_hidden_until_the_search_is_closed(self):
        # a selection score of exactly zero shows as every generation truncated, so the counts would disclose it during the search
        b, st, lay = self.make(close=False, final=False, reveal=False)
        sid = next(sid for sid, r in st.scores.items() if r.split is Split.SELECTION and r.phase is c.Phase.SEARCH)
        r = st.scores[sid]
        st.scores[sid] = dataclasses.replace(r, per_seed=tuple(dataclasses.replace(s, n_truncated=r.n_prompts, n_errors=7) for s in r.per_seed))
        total = r.n_prompts * len(r.per_seed)
        rep = report.build_report(st, lay)
        sel_rows = [x for x in self.scoring_rows(st, lay) if x[0] == "selection" and x[1] == "search"]
        self.assertTrue(sel_rows)
        for x in sel_rows:
            self.assertEqual(x[7:], ["hidden until search closed"] * 2, x)
        self.assertFalse(any("truncated generation" in x or "errored" in x for x in rep.labels), rep.labels)
        self.assertAbsent({f"{total} truncated"}, render(rep))
        # positive control: once the search is closed the same record's counts are printed
        b.close_search()
        rep2 = report.build_report(st, lay)
        self.assertTrue(any(f"{total} truncated generation" in x for x in rep2.labels), rep2.labels)


# --------------------------------------------------------------------------------------------------
class LabelTests(Base):
    def labels(self, st, lay, **kw):
        return "\n".join(report.build_report(st, lay, **kw).labels)

    def test_components_only_label_when_no_trial_joined_the_chain(self):
        b = LabStateBuilder(self.dir / "empty")
        st = b.build()
        text = render(report.build_report(st, b.layout))
        self.assertIn("components tested, system not yet demonstrated", text)
        b2, st2, lay2 = self.make()
        self.assertNotIn("components tested, system not yet demonstrated", render(report.build_report(st2, lay2)))

    def test_a_trial_that_failed_before_scoring_is_not_a_joined_trial(self):
        b = LabStateBuilder(self.dir / "failed").freeze_manifest().freeze_budget()
        b.add_pack("v0")
        b.add_trial(score=None, status=TrialStatus.NO_SUBMISSION)
        self.assertIn("components tested, system not yet demonstrated", self.labels(b.build(), b.layout))

    def test_spread_not_estimated_when_the_rule_is_missing(self):
        b = LabStateBuilder(self.dir / "norule").freeze_manifest().freeze_budget()
        self.assertIn("spread not estimated", self.labels(b.build(), b.layout))
        b.add_pack("v0")
        b.measure_spread([0.60, 0.62, 0.61, 0.63, 0.64])
        self.assertIn("spread not estimated", self.labels(b.build(), b.layout))  # measured but no frozen rule
        b2, st2, lay2 = self.make()
        self.assertNotIn("spread not estimated", self.labels(st2, lay2))

    def test_a_tampered_rule_file_is_a_warning_and_the_rule_is_not_used(self):
        b, st, lay = self.make()
        lay.rule_path.write_text(lay.rule_path.read_text().replace("0.05", "0.5"), encoding="utf-8")
        rep = report.build_report(st, lay)
        self.assertTrue(any("frozen rule could not be verified" in w for w in rep.warnings))
        self.assertIn("spread not estimated", "\n".join(rep.labels))
        self.assertIn("The rule is not frozen: no candidate can be decided.", render(rep))

    def test_a_tampered_final_analysis_shows_no_final_numbers(self):
        b, st, lay = self.make()
        lay.final_analysis_path.write_text(lay.final_analysis_path.read_text().replace("0.95", "0.80"), encoding="utf-8")
        rep = report.build_report(st, lay)
        self.assertTrue(any("frozen final analysis could not be verified" in w for w in rep.warnings))
        self.assertAbsent({"0.780", "0.835"}, render(rep))

    def test_dev_scores_label_always_prints(self):
        b = LabStateBuilder(self.dir / "empty2")
        self.assertIn("dev scores are researcher-influenced; decisions use control-host numbers only", self.labels(b.build(), b.layout))

    def test_adaptive_search_label_counts_operator_changes(self):
        b, st, lay = self.make()
        self.assertIn("the bits and the operator's choices steer the search", self.labels(st, lay).replace("The bits", "the bits"))
        self.assertIn("1 operator change(s)", self.labels(st, lay))
        b.add_operator_change("rule_input", "x")
        self.assertIn("2 operator change(s)", self.labels(st, lay))

    def test_ledger_limitation_is_stated(self):
        b, st, lay = self.make()
        self.assertIn("proves order and non-editing, not truth", render(report.build_report(st, lay)))

    def test_custom_and_partial_scoring_labels(self):
        b, st, lay = self.make()
        self.assertIn("Custom scoring", self.labels(st, lay))
        self.assertNotIn("Partial scoring", self.labels(st, lay))
        # a test score over fewer than 300 prompts, and a score with errored prompts
        rid = next(i for i, r in st.scores.items() if r.split is Split.TEST)
        rec = st.scores[rid]
        st.scores[rid] = dataclasses.replace(rec, n_prompts=293, per_seed=tuple(dataclasses.replace(s, n=293) for s in rec.per_seed))
        sel = next(i for i, r in st.scores.items() if r.split is Split.SELECTION)
        st.scores[sel] = dataclasses.replace(st.scores[sel], per_seed=tuple(dataclasses.replace(s, n_errors=2, n_truncated=1) for s in st.scores[sel].per_seed))
        text = self.labels(st, lay)
        self.assertIn("Partial scoring: 1 test score(s) cover fewer than 300 prompts (293)", text)
        self.assertIn("Partial scoring: 6 prompt(s) errored", text)
        self.assertIn("3 truncated generation(s)", text)

    def test_the_verifier_and_prompt_count_are_printed_next_to_scores(self):
        b, st, lay = self.make()
        md = report.render_markdown(report.build_report(st, lay))
        self.assertIn("ifbench_upstream, 300 prompts", md)
        self.assertIn("held-out ifevalg, 200 prompts; test ifbench_upstream, 300 prompts", md)
        scoring = report.build_report(st, lay).section("scoring").blocks[0].table
        self.assertIn(["selection", "search", "ifevalg", "200", "3", "sv1-0123456789abcdef", "25", "0", "0"], scoring.rows)

    def test_usage_estimated_charges_each_get_a_warning(self):
        b, st, lay = self.make()
        charges = [
            BudgetCharge("t-0001", Resource.RESEARCHER_USD, 0.5, "call-77:usd", {"usage_estimated": True}),
            BudgetCharge("t-0001", Resource.RESEARCHER_TOKENS, 900.0, "call-77:tokens", {"detail": "usage_estimated"}),
            BudgetCharge("t-0001", Resource.RESEARCHER_USD, 0.1, "call-78:usd", {"usage_estimated": False}),  # provider-reported
            BudgetCharge("t-0002", Resource.GPU_MINUTES, 1.0, "gpu:r:1", {"rate": 1.3}),
        ]
        b.write_trial_ledger("t-0001", charges=charges)
        rep = report.build_report(st, lay)
        est = [w for w in rep.warnings if w.startswith("estimated usage")]
        self.assertEqual(len(est), 2)
        self.assertTrue(any("call-77:usd" in w for w in est) and any("call-77:tokens" in w for w in est))
        self.assertFalse(any("call-78" in w or "gpu:r:1" in w for w in rep.warnings))
        self.assertIn("2 budget charge(s) were estimated", "\n".join(rep.labels))
        self.assertIn("call-77:usd", render(rep))
        # no estimated charge, no warning
        b2, st2, lay2 = self.make()
        self.assertFalse([w for w in report.build_report(st2, lay2).warnings if "estimated" in w])

    def test_a_symlinked_trial_ledger_is_never_read(self):
        b, st, lay = self.make()
        canary = self.dir / "outside.jsonl"
        canary.write_text(json.dumps({"kind": "lab.budget_charged", "data": BudgetCharge("t-0001", Resource.RESEARCHER_USD, 1.0, "CANARY-CALL", {"usage_estimated": True}).to_dict()}) + "\n")
        ledger = lay.trial_dir("t-0001") / "ledger.jsonl"
        ledger.unlink()
        os.symlink(canary, ledger)
        text = render(report.build_report(st, lay))
        self.assertNotIn("CANARY-CALL", text)

    def test_isolation_level_of_each_trial_and_unrecorded(self):
        b, st, lay = self.make()
        st.trials["t-0002"] = dataclasses.replace(st.trials["t-0002"], environment={"isolation": "serial shared user (no per-trial users)"})
        st.trials["t-0003"] = dataclasses.replace(st.trials["t-0003"], environment={})
        t = report.build_report(st, lay).section("trials").blocks[0].table
        rows = {r[0]: r for r in t.rows}
        self.assertEqual(rows["t-0001"][5], "per-trial unix user")
        self.assertEqual(rows["t-0002"][5], "serial shared user (no per-trial users)")
        self.assertEqual(rows["t-0003"][5], "not recorded")
        self.assertEqual(len(t.rows), len(st.trials_started))

    def test_contamination_risk_text_follows_the_item_mode(self):
        b, st, lay = self.make()
        self.assertIn("salted draw", render(report.build_report(st, lay)))
        b2, st2, lay2 = self.make(item_mode=ItemMode.FRESH_INSTANTIATION)
        text = render(report.build_report(st2, lay2))
        self.assertIn("fresh constraint instantiations", text)
        self.assertNotIn("salted draw", text)
        self.assertIn("Residual contamination risk", text)
        self.assertIn("Contamination scans: 1 recorded, 0 with hits", text)

    def test_contamination_hits_are_counted(self):
        b, st, lay = self.make()
        b.add_contamination("t-0002", hits=3)
        st.trials["t-0002"] = dataclasses.replace(st.trials["t-0002"], status=TrialStatus.CONTAMINATED, failure="training data matched held-out prompts")
        self.assertIn("2 recorded, 1 with hits; 1 trial(s) marked contaminated", render(report.build_report(st, lay)))

    def test_malformed_ledger_events_are_listed(self):
        b, st, lay = self.make()
        st.malformed.append((17, "lab.trial_finished for 't-9' without lab.trial_started"))
        rep = report.build_report(st, lay)
        self.assertTrue(any("1 malformed event" in w and "seq 17" in w for w in rep.warnings))


# --------------------------------------------------------------------------------------------------
class FinalIntervalTests(Base):
    def test_minimum_detectable_gain_sits_next_to_every_final_interval(self):
        b, st, lay = self.make()
        rep = report.build_report(st, lay)
        tbl = [bl.table for bl in rep.section("final").blocks if bl.kind == "table" and bl.table.caption.startswith("Confirmatory")][0]
        self.assertEqual(len(tbl.rows), 2)
        an = b.analysis
        v0 = [b.trial_scores[t.trial_id] for t in an.planned_trials if t.arm is Arm.V0]
        ev = [b.trial_scores[t.trial_id] for t in an.planned_trials if t.arm is Arm.EVOLVED]
        fc = stats.compare_final(an, v0, ev, Split.SELECTION)
        se = (fc.ci_high - fc.delta) / stats.t_quantile((1 - fc.level) / 2, fc.df)
        want = stats.minimum_detectable_gain_from_se(se, fc.df, (1 - fc.level) / 2)
        self.assertEqual(tbl.rows[0][4], f"{want:.4f}")
        self.assertEqual(tbl.rows[0][1], f"{fc.delta:+.4f}")
        self.assertIn(f"[{fc.ci_low:+.4f}, {fc.ci_high:+.4f}]", tbl.rows[0][2])
        self.assertTrue(tbl.rows[1][4].replace(".", "").isdigit())  # the test comparison has one too
        self.assertEqual(tbl.headers[4], "minimum detectable gain (80% power)")

    def test_a_plain_statement_when_the_interval_includes_zero(self):
        vals = {"heldout_v0": [0.775, 0.785, 0.780, 0.790, 0.770], "heldout_ev": [0.790, 0.770, 0.785, 0.795, 0.775], "test_v0": [0.4033, 0.41, 0.4, 0.4133, 0.4067], "test_ev": [0.41, 0.4, 0.4133, 0.4033, 0.4067]}
        b, st, lay = self.make(final_values=vals)
        text = render(report.build_report(st, lay))
        self.assertIn("includes zero: this experiment does not show that arm 3 differs from arm 2 on prompt_all", text)
        self.assertIn("includes zero: this experiment does not show that arm 3 differs from arm 2 on prompt_strict", text)
        b2, st2, lay2 = self.make()
        self.assertNotIn("does not show that arm 3 differs", render(report.build_report(st2, lay2)))

    def test_the_number_of_assignments_and_trials_per_arm_are_printed(self):
        b, st, lay = self.make()
        self.assertIn("1 assignment(s), 5 v0 and 5 evolved trials", render(report.build_report(st, lay)))


# --------------------------------------------------------------------------------------------------
class CostTests(Base):
    def cost_notes(self, st, lay):
        sec = report.build_report(st, lay).section("costs")
        return [i for bl in sec.blocks if bl.kind == "bullets" for i in bl.items]

    def test_the_four_cost_centres_are_separate_rows(self):
        b, st, lay = self.make()
        tbl = report.build_report(st, lay).section("costs").blocks[0].table
        names = [r[0] for r in tbl.rows]
        self.assertTrue(any("researcher inference" in n and "search" in n for n in names))
        self.assertTrue(any("GPU" in n and "search" in n for n in names))
        self.assertTrue(any("outer proposals" in n for n in names))
        self.assertTrue(any("scoring" in n and "search" in n for n in names))
        self.assertTrue(any("final" in n for n in names))
        gpu = [r for r in tbl.rows if "GPU" in r[0] and "search" in r[0]][0]
        self.assertEqual((gpu[1], gpu[2]), ("$20.00", "$22.50"))  # 25 trials x $0.80 on the meter; 25 x $0.90 in the rental ledger
        self.assertEqual([r for r in tbl.rows if "outer proposals" in r[0]][0][2], "$1.50")

    def test_both_gpu_numbers_are_called_out_when_they_differ(self):
        b, st, lay = self.make()
        notes = self.cost_notes(st, lay)
        self.assertTrue(any("differ from the rental ledger" in n for n in notes), notes)
        self.assertIn("one-minute ticks", render(report.build_report(st, lay)))

    def test_no_difference_note_when_meter_and_ledger_agree(self):
        b, st, lay = self.make(ledgers=False)
        for tid in st.trials:
            b.write_trial_ledger(tid, gpu_usd=0.80, gpu_minutes=40.0)
        self.assertEqual(self.cost_notes(st, lay), [])

    def test_missing_trial_ledgers_are_stated_not_guessed(self):
        b, st, lay = self.make(ledgers=False)
        tbl = report.build_report(st, lay).section("costs").blocks[0].table
        self.assertIn("n/a (no trial ledger read)", [r for r in tbl.rows if "GPU" in r[0] and "search" in r[0]][0][2])


# --------------------------------------------------------------------------------------------------
class FileIntegrityTests(Base):
    def test_a_per_prompt_file_that_does_not_match_its_hash_is_a_warning_and_unused(self):
        b, st, lay = self.make()
        rid = next(i for i, r in st.scores.items() if r.trial_id == "t-f-v0-1-0" and r.split is Split.TEST)
        path = lay.root / st.scores[rid].per_prompt.path
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        rep = report.build_report(st, lay, full=True)
        self.assertTrue(any(f"score {rid} does not match its recorded sha256" in w for w in rep.warnings), rep.warnings)
        rows = rep.section("per_prompt").blocks[1].table.rows
        self.assertEqual([r[0] for r in rows], ["arm 1 to arm 3"])  # every pair that needed arm 2's rows is gone

    def test_a_missing_per_prompt_file_is_a_warning(self):
        b, st, lay = self.make()
        rec = next(r for r in st.scores.values() if r.arm is Arm.STOCK)
        (lay.root / rec.per_prompt.path).unlink()
        rep = report.build_report(st, lay, full=True)
        self.assertTrue(any("is unreadable" in w for w in rep.warnings))
        self.assertEqual([r[0] for r in rep.section("per_prompt").blocks[1].table.rows], ["arm 2 to arm 3"])

    def test_a_symlinked_per_prompt_file_is_refused(self):
        b, st, lay = self.make()
        rec = next(r for r in st.scores.values() if r.arm is Arm.STOCK)
        p = lay.root / rec.per_prompt.path
        other = self.dir / "other.jsonl"
        other.write_text(p.read_text())
        p.unlink()
        os.symlink(other, p)
        rep = report.build_report(st, lay, full=True)
        self.assertTrue(any("is unreadable" in w for w in rep.warnings))

    def test_per_prompt_diagnostics_are_labelled_checkpoint_level(self):
        b, st, lay = self.make()
        text = render(report.build_report(st, lay, full=True))
        self.assertIn("checkpoint-level", text)
        self.assertIn("never decide a keep", text)
        self.assertIn("2000 resamples", text)
        self.assertNotIn("Paired per-prompt", render(report.build_report(st, lay)))  # not in the default view


# --------------------------------------------------------------------------------------------------
class SpreadTests(Base):
    def test_three_spreads_are_kept_separate_in_the_full_view(self):
        b, st, lay = self.make()
        rows = report.build_report(st, lay, full=True).section("spreads").blocks[0].table.rows
        self.assertEqual([r[0] for r in rows], ["generation seed (scorer noise)", "trial (v0, within assignment)", "assignment (sd of per-assignment means)"])
        self.assertIn(f"sigma {st.spread.estimate.sigma:.4f} with df 4", rows[1][1])
        self.assertIn("mean sd over seeds", rows[0][1])
        self.assertIn("not estimable", rows[2][1])

    def test_assignment_spread_with_two_assignments(self):
        b = LabStateBuilder(self.dir / "two", n_evolution=2).freeze_manifest().freeze_budget()
        b.add_pack("v0")
        ids = [b.add_trial(assignment=a, score=s) for a, s in (("a-evo-1", 0.60), ("a-evo-1", 0.62), ("a-evo-2", 0.70), ("a-evo-2", 0.72))]
        st = b.state
        est = stats.trial_spread([b.trial_scores[i] for i in ids])
        st.spread = c.SpreadMeasured("prompt_all", "sv1-0123456789abcdef", 4, 0, tuple(b.trial_scores[i] for i in ids), est)
        b.state.search_closed = c.SearchClosed("v0", b.packs["v0"], 0, 4, "test", b.utc())
        rows = report.build_report(b.build(), b.layout, full=True).section("spreads").blocks[0].table.rows
        self.assertIn("sd 0.0707 over 2 assignments", rows[2][1])  # sd of {0.61, 0.71}
        self.assertIn("df 2", rows[1][1])


# --------------------------------------------------------------------------------------------------
class ChartTests(unittest.TestCase):
    def pts(self):
        return [
            TracePoint(1, "cand-001", "rejected", 0.62, 0.60),
            TracePoint(2, "cand-002", "accepted", 0.66, 0.60),
            TracePoint(3, "cand-003", "invalid"),
            TracePoint(4, "cand-004", "rejected", 0.65, 0.66),
            TracePoint(5, "cand-005", "pending"),
        ]

    def test_well_formed_titled_and_accessible(self):
        svg = search_trace_svg(self.pts(), "adaptive search trace")
        doc = xml.dom.minidom.parseString(svg)
        self.assertEqual(doc.documentElement.getAttribute("role"), "img")
        self.assertIn("adaptive search trace", svg)
        self.assertIn("1 accepted, 2 rejected, 1 invalid, 1 pending", svg)
        self.assertEqual(svg.count("<polygon"), 2)  # accepted marker + legend
        self.assertEqual(svg.count("<g class=\"st-inv\">"), 2)  # invalid cross + legend
        self.assertIn("prefers-color-scheme: dark", svg)
        self.assertIn('viewBox="0 0 720 360"', svg)

    def test_scores_draw_the_incumbent_line_and_place_markers_by_value(self):
        svg = search_trace_svg(self.pts(), "t")
        self.assertIn('class="st-inc" d="M', svg)
        # cand-002 (0.66) is drawn higher on the page (smaller y) than cand-001 (0.62)
        y1 = float(re.search(r'cy="([\d.]+)" r="6"><title>cand-001', svg).group(1))
        pt = re.search(r'points="([\d.]+),([\d.]+) [^"]*"><title>cand-002', svg)
        y2 = float(pt.group(2)) + 7  # the diamond's top vertex is 7 above its centre
        self.assertLess(y2, y1)

    def test_no_scores_means_a_bit_only_chart(self):
        svg = search_trace_svg([TracePoint(p.index, p.label, p.outcome) for p in self.pts()], "t")
        xml.dom.minidom.parseString(svg)
        self.assertIn("Bit-only view", svg)
        self.assertNotIn('class="st-inc" d=', svg)
        self.assertIn("Scores are hidden in this view", svg)

    def test_empty_trace(self):
        svg = search_trace_svg([], "adaptive search trace")
        xml.dom.minidom.parseString(svg)
        self.assertIn("no candidates yet", svg)

    def test_text_is_escaped(self):
        svg = search_trace_svg([TracePoint(1, "<b>&x", "accepted", 0.5, 0.4)], "a <script> & b")
        xml.dom.minidom.parseString(svg)
        self.assertNotIn("<script>", svg)
        self.assertNotIn("<b>", svg)

    def test_one_value_does_not_divide_by_zero(self):
        xml.dom.minidom.parseString(search_trace_svg([TracePoint(1, "c", "accepted", 0.5, 0.5)], "t"))

    def test_bad_points_are_refused(self):
        with self.assertRaises(ValueError):
            TracePoint(1, "c", "maybe")
        with self.assertRaises(ValueError):
            TracePoint(1, "c", "accepted", float("nan"))

    def test_outcomes_are_distinguished_by_shape_not_only_colour(self):
        svg = search_trace_svg(self.pts(), "t")
        self.assertIn("<polygon", svg)  # diamond
        self.assertIn("<circle", svg)  # circle
        self.assertIn("<line", svg)  # cross


if __name__ == "__main__":
    unittest.main()
