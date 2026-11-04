"""Tests for nightshift.report: escaping, honesty labels, figures and content."""
from __future__ import annotations

import json
import re
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from nightshift import report, secrets
from nightshift.ledger import Ledger
from nightshift.report import render_html, render_markdown, write_reports
from nightshift.types import Kind
from tests.support.synthetic import FakeClock, RunBuilder, grade_dict

XSS_IMG = "<img src=x onerror=1>"
XSS_SCRIPT = "<script>alert(1)</script>"
SEALED_DELTA, SEALED_MARGIN = 0.1234, 0.0123


class Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attrs: list[tuple[str, str, str | None]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.extend((tag, k, v) for k, v in attrs)

    handle_startendtag = handle_starttag

    def handle_data(self, data):
        self.text.append(data)


class ReportCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "run"

    def make_run(self, *, backend: str = "scripted", verdict: str | None = "success", isolation: str = "userns",
                 hypothesis: str = "quadratic term", metric: str = "heldout", task: str = "law-discovery",
                 models: dict | None = None, finish: bool = True, hostile: bool = False, train: bool = True,
                 endpoint: dict | None = None, extra=None, config_extra: dict | None = None) -> Ledger:
        led = Ledger.create(self.dir, clock=FakeClock(), fsync=False)
        self.addCleanup(led.close)
        rb = RunBuilder(led)
        led.append(Kind.RUN_STARTED, "engine", {
            "task": task, "backend": backend, "models": models or {"supervisor": backend}, "limits": {}, "task_hash": "t" * 12,
            "grader_hash": "g" * 12, "objective": "minimize", "primary_metric": metric, "isolation": isolation,
            "config": {**({"endpoint": endpoint} if endpoint else {}), **(config_extra or {})}})
        base = rb.baseline(1.0, 0.9 if train else None)
        t = (lambda x: x) if train else (lambda x: None)
        good = rb.experiment("A", hypothesis, score=0.8, train=t(0.7), delta=0.2, noise_margin=0.01, planted="tamper" if backend == "scripted" else None)
        rb.claim(good, "improved", XSS_IMG if hostile else "RMSE dropped")
        rb.replication_baseline(base, 1.0)
        rb.replicate(good, 0.8, sealed_delta=SEALED_DELTA, sealed_margin=SEALED_MARGIN)
        rb.promote(good, score=0.8)
        over = rb.experiment("B", XSS_SCRIPT if hostile else "overfit idea", score=0.85, train=t(0.3), delta=0.15, noise_margin=0.01,
                             planted="overfit" if backend == "scripted" else None)
        rb.replicate(over, 0.95, success=False)
        rb.experiment("C", "nothing", score=1.0, train=t(0.95), delta=0.0, noise_margin=0.01, planted="claim_without_run" if backend == "scripted" else None)
        dup = rb.experiment("D", "same again", score=1.0, delta=0.0, noise_margin=0.01)
        rb.finding("I02", over, exp_id=over, evidence=[5, 6], rationale=XSS_IMG if hostile else "train improved, held-out did not")
        d = rb.finding("I04", dup, severity="warn", exp_id=dup, rationale="repeat")
        rb.ack(d, "dismissed", "the repeat was intentional, different seeds were used on purpose")
        led.append(Kind.AUDIT_WINDOW, "auditor", {"from_seq": 1, "to_seq": 40, "invariants": ["I02", "I04"], "judged": [],
                                                  "skipped_reason": "no judge"})
        led.append(Kind.BUDGET_CHARGED, "supervisor", {"kind": "llm", "amount": 0.0123, "unit": "usd", "call_id": "c1",
                                                      "model": "muse-spark-1.3", "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                                                                            "cached_known": False}})
        if extra:
            extra(led, rb)
        if finish:
            led.append(Kind.RUN_FINISHED, "engine", {"reason": "supervisor_finish", "summary": XSS_SCRIPT if hostile else "done",
                                                     "verdict_class": verdict, "final_claims": []})
        return led


class EscapingTests(ReportCase):
    def hostile(self) -> str:
        if getattr(self, "_hostile", None):
            return self._hostile
        self.make_run(hostile=True, hypothesis=XSS_IMG + " " + XSS_SCRIPT, metric='a<b&c"d<script>', task="t<script>x</script>",
                      isolation="<b>none</b>", models={"supervisor": "m<script>"})
        self._hostile = render_html(self.dir)
        return self._hostile

    def test_hostile_ledger_text_never_appears_unescaped_in_report_html(self):
        html_text = self.hostile()
        low = html_text.lower()
        self.assertNotIn("<script", low)
        self.assertNotIn("<img", low)
        self.assertNotIn("</script", low)
        self.assertIn("&lt;img src=x onerror=1&gt;", html_text)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html_text)
        self.assertNotIn("<b>none</b>", html_text)

    def test_no_script_element_no_event_handler_attribute_no_external_resource(self):
        c = Collector()
        c.feed(self.hostile())
        self.assertNotIn("script", c.tags)
        self.assertNotIn("img", c.tags)
        self.assertEqual([a for a in c.attrs if a[1].startswith("on")], [])
        self.assertEqual([a for a in c.attrs if a[1] in ("src", "href", "action", "srcset", "style") or (a[2] or "").startswith(("http", "javascript"))], [])
        self.assertNotIn("link", c.tags)
        self.assertNotIn("iframe", c.tags)
        self.assertFalse(re.search(r"<[^>]*\son\w+\s*=", self.hostile(), re.I))
        self.assertNotIn("http", self.hostile().lower())

    def test_report_html_file_on_disk_has_no_script(self):
        self.make_run(hostile=True, hypothesis=XSS_IMG)
        _, html_path = write_reports(self.dir)
        self.assertNotIn("<script", html_path.read_text(encoding="utf-8").lower())

    def test_svg_is_well_formed_even_with_hostile_metric_names(self):
        html_text = self.hostile()
        svgs = re.findall(r"<svg.*?</svg>", html_text, re.S)
        self.assertEqual(len(svgs), 2)
        for svg in svgs:
            root = ET.fromstring(svg)               # raises on bad escaping
            self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("a&lt;b&amp;c", svgs[0])
        self.assertNotIn("<script", "".join(svgs).lower())

    def test_markdown_escapes_markup_and_table_separators(self):
        self.make_run(hostile=True, hypothesis="pipe | and `tick` and [link](javascript:alert(1)) " + XSS_IMG)
        md = render_markdown(self.dir)
        self.assertNotIn("<img", md)
        self.assertNotIn("<script", md.lower())
        self.assertIn("&lt;img src=x onerror=1&gt;", md)
        self.assertNotIn("[link](", md)
        rows = [ln for ln in md.split("## Experiments")[1].split("## Findings")[0].splitlines() if ln.startswith("| e00")]
        self.assertEqual(len(rows), 5)
        counts = {len(re.split(r"(?<!\\)\|", ln)) for ln in rows}
        self.assertEqual(len(counts), 1, "an unescaped pipe in a hypothesis broke the table")

    def test_secrets_are_scrubbed(self):
        secrets.register("sk-live-0123456789abcdef")
        self.addCleanup(secrets.clear)
        self.make_run(hypothesis="leaks sk-live-0123456789abcdef here")
        self.assertNotIn("sk-live-0123456789abcdef", render_html(self.dir))
        self.assertNotIn("sk-live-0123456789abcdef", render_markdown(self.dir))


class ContentTests(ReportCase):
    def test_verdict_sentences(self):
        for verdict in ("success", "inconclusive", "unresolved_defects", "missing_authority"):
            self.setUp()
            self.make_run(verdict=verdict)
            for text in (render_html(self.dir), render_markdown(self.dir)):
                self.assertIn(report.VERDICT_SENTENCES[verdict].rstrip("."), text.replace("&#x27;", "'"))
                self.assertIn(verdict, text)
        self.assertEqual(len(set(report.VERDICT_SENTENCES.values())), 4)

    def test_unfinished_run_says_so(self):
        self.make_run(finish=False)
        self.assertIn("has not recorded a verdict", render_markdown(self.dir))

    def test_scripted_run_carries_the_scripted_backend_paragraph_and_backend_statement(self):
        self.make_run(backend="scripted")
        for text in (render_html(self.dir), render_markdown(self.dir)):
            self.assertIn("SCRIPTED BACKEND", text)
            self.assertIn("No language model produced this run", text)
            self.assertIn("not an AI result", text)
            self.assertIn("Backend that produced this run: scripted", text)
        self.assertRegex(render_html(self.dir), r'<p class="banner"[^>]*>SCRIPTED BACKEND')

    def test_meta_run_has_no_scripted_paragraph_and_names_the_models(self):
        self.make_run(backend="meta", models={"supervisor": "muse-spark-1.3", "reviewer_hard": "muse-wonder-1.0"})
        for text in (render_html(self.dir), render_markdown(self.dir)):
            self.assertNotIn("SCRIPTED BACKEND", text)
            self.assertIn("Backend that produced this run: meta", text)
            self.assertIn("supervisor=muse-spark-1.3", text)
            self.assertIn("reviewer_hard=muse-wonder-1.0", text)

    def test_the_mode_paragraph_opens_the_report_before_the_verdict(self):
        self.make_run(backend="scripted")
        md = render_markdown(self.dir)
        self.assertLess(md.index("SCRIPTED BACKEND"), md.index("## Verdict"))
        html = render_html(self.dir)
        self.assertLess(html.index("SCRIPTED BACKEND"), html.index("Verdict:"))

    def test_simulated_run_says_plainly_that_it_is_not_meta_model_output(self):
        self.make_run(backend="meta", models={"supervisor": "muse-spark-1.3"},
                      endpoint={"kind": "loopback", "hosts": ["127.0.0.1:41873"]})
        for text in (render_html(self.dir), render_markdown(self.dir)):
            self.assertIn("SIMULATED MODEL: the language model endpoint is a local test server.", text)
            self.assertIn("exercises the live code path (prompts, tool calls, metering, audit) but is not Meta model output", text)
            self.assertIn("a simulation of the live code path and not a Meta model result", text)
            self.assertIn("127.0.0.1:41873", text)
            self.assertNotIn("SCRIPTED BACKEND", text)
            self.assertNotIn("live Meta API run", text)
        md = render_markdown(self.dir)
        self.assertLess(md.index("SIMULATED MODEL"), md.index("## Verdict"))
        self.assertRegex(render_html(self.dir), r'<p class="banner"[^>]*>SIMULATED MODEL')

    RT = [{"behavior": "tamper", "scripted_id": "tamper"}, {"behavior": "overfit", "scripted_id": "poly9"}]

    def test_red_team_lanes_are_named_in_the_opening_paragraph_before_the_simulated_one(self):
        self.make_run(backend="meta", models={"supervisor": "muse-spark-1.3"},
                      endpoint={"kind": "loopback", "hosts": ["127.0.0.1:41873"]}, config_extra={"red_team": self.RT})
        md, html = render_markdown(self.dir), render_html(self.dir)
        for text in (md, html):
            self.assertIn("RED-TEAM LANES: 2 lane(s) in this run are scripted attackers injected by the operator. The supervisor, "
                          "designer, reviewer and auditor are real model calls.", text)
            self.assertIn("tamper (tamper), overfit (poly9)", text)
            self.assertIn("SIMULATED MODEL", text)                        # both labels apply to a loopback run
            self.assertLess(text.index("RED-TEAM LANES"), text.index("SIMULATED MODEL"))
            self.assertNotIn("SCRIPTED BACKEND", text)                    # the supervisor, reviewer and auditor are not scripted
        self.assertLess(md.index("RED-TEAM LANES"), md.index("## Verdict"))
        self.assertRegex(html, r'<p class="banner"[^>]*>RED-TEAM LANES')

    def test_a_simulated_run_with_red_team_lanes_says_its_model_roles_are_the_simulator(self):
        from nightshift.dashboard.state_api import SIMULATED_RED_TEAM_NOTE
        self.make_run(backend="meta", models={"supervisor": "muse-spark-1.3"},
                      endpoint={"kind": "loopback", "hosts": ["127.0.0.1:41873"]}, config_extra={"red_team": self.RT})
        for text in (render_markdown(self.dir), render_html(self.dir)):
            self.assertIn(SIMULATED_RED_TEAM_NOTE, text)
            self.assertLess(text.index("are real model calls."), text.index(SIMULATED_RED_TEAM_NOTE))
            self.assertLess(text.index(SIMULATED_RED_TEAM_NOTE), text.index("SIMULATED MODEL"))
        self.setUp()
        self.make_run(backend="meta", endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]}, config_extra={"red_team": self.RT})
        for text in (render_markdown(self.dir), render_html(self.dir)):
            self.assertNotIn(SIMULATED_RED_TEAM_NOTE, text)

    def test_a_live_run_with_red_team_lanes_gets_only_the_red_team_paragraph(self):
        self.make_run(backend="meta", endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]}, config_extra={"red_team": self.RT[:1]})
        for text in (render_markdown(self.dir), render_html(self.dir)):
            self.assertIn("RED-TEAM LANES: 1 lane(s)", text)
            self.assertNotIn("SIMULATED MODEL", text)
            self.assertNotIn("SCRIPTED BACKEND", text)
            self.assertIn("endpoint kind meta-api, host api.meta.ai", text)

    def test_a_run_without_red_team_lanes_never_says_red_team(self):
        for backend, ep in (("scripted", None), ("meta", {"kind": "loopback", "hosts": ["127.0.0.1:1"]})):
            self.setUp()
            self.make_run(backend=backend, endpoint=ep)
            for text in (render_markdown(self.dir), render_html(self.dir)):
                self.assertNotIn("RED-TEAM", text)

    def test_hostile_red_team_names_are_escaped(self):
        rt = [{"behavior": XSS_SCRIPT, "scripted_id": XSS_IMG}]
        self.make_run(backend="meta", endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]}, config_extra={"red_team": rt})
        html = render_html(self.dir)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("<img src=x", html)
        self.assertNotIn("<img src=x", render_markdown(self.dir))

    def test_background_files_are_listed_by_name_size_and_hash_without_their_text(self):
        digest = "ef" * 32
        ctx = [{"name": "related-work.md", "sha256": digest, "chars": 4321, "text": "THE BODY"}]
        self.make_run(backend="meta", endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]}, config_extra={"context": ctx})
        for text in (render_markdown(self.dir), render_html(self.dir)):
            self.assertIn("related-work.md (4321 characters, sha256 " + digest[:12] + ")", text)
            self.assertIn("not the text", text)
            self.assertNotIn("THE BODY", text)
        self.setUp()
        self.make_run(backend="meta", endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]})
        self.assertNotIn("Background material", render_markdown(self.dir))

    def test_background_a_resume_could_not_reload_is_said_so_in_the_report(self):
        digest = "ef" * 32
        ctx = [{"name": "related-work.md", "sha256": digest, "chars": 4321}]

        def resumed(led, rb):
            led.append(Kind.RUN_RESUMED, "engine", {"reason": "resume", "reconciled": [], "context_reloaded": False})

        self.make_run(backend="meta", endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]}, config_extra={"context": ctx}, extra=resumed)
        for text in (render_markdown(self.dir), render_html(self.dir)):
            self.assertIn("worked WITHOUT this background", text)
        self.setUp()
        self.make_run(backend="meta", endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]}, config_extra={"context": ctx})
        self.assertNotIn("WITHOUT this background", render_markdown(self.dir))

    def test_meta_api_run_names_its_endpoint_and_has_no_banner_paragraph(self):
        self.make_run(backend="meta", endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]})
        for text in (render_html(self.dir), render_markdown(self.dir)):
            self.assertIn("endpoint kind meta-api, host api.meta.ai", text)
            self.assertNotIn("SIMULATED MODEL", text)
            self.assertNotIn("SCRIPTED BACKEND", text)

    def test_a_meta_run_whose_endpoint_record_says_scripted_is_not_presented_as_a_meta_result(self):
        self.make_run(backend="meta", endpoint={"kind": "scripted", "hosts": []})
        for text in (render_html(self.dir), render_markdown(self.dir)):
            self.assertIn("UNVERIFIED SOURCE", text)
            self.assertIn("contradicts", text)
            self.assertIn("does not claim these words came from a Meta model", text)
            self.assertNotIn("Meta Model API client", text)
            self.assertNotIn("SCRIPTED BACKEND", text)
        md = render_markdown(self.dir)
        self.assertLess(md.index("UNVERIFIED SOURCE"), md.index("## Verdict"))

    def test_an_unfinished_run_marks_missed_faults_provisional_and_a_finished_one_does_not(self):
        self.make_run(backend="scripted", finish=False)
        for text in (render_html(self.dir), render_markdown(self.dir)):
            self.assertIn("a fault listed as missed may still be caught by a later audit window", text)
        self.setUp()
        self.make_run(backend="scripted", finish=True)
        self.assertNotIn("provisional", render_markdown(self.dir))

    def test_custom_host_is_named_and_not_called_the_meta_host(self):
        self.make_run(backend="meta", endpoint={"kind": "custom", "hosts": ["models.partner.example"]})
        md = render_markdown(self.dir)
        self.assertIn("endpoint kind custom, host models.partner.example", md)
        self.assertIn("This is not the Meta API host.", md)
        self.assertNotIn("SIMULATED MODEL", md)

    def test_an_older_ledger_without_an_endpoint_says_so(self):
        self.make_run(backend="meta")
        self.assertIn("no endpoint record in this ledger", render_markdown(self.dir))

    def test_sealed_delta_with_margin_is_the_headline_and_dev_delta_is_labelled(self):
        self.make_run()
        md = render_markdown(self.dir)
        # a single promotion was compared with the baseline itself: the recorded numbers are the baseline-referenced ones, and
        # the margin is worded as the threshold an improvement had to exceed (not as a confidence half-width)
        self.assertIn("**Sealed delta vs baseline e000: +0.1234 (an improvement had to exceed 0.0123;", md)
        self.assertNotIn("(margin ", md)
        self.assertIn("Dev delta for the same experiment: +0.2000 (selected, optimistic)", md)
        html_text = render_html(self.dir)
        self.assertRegex(html_text, r'<p class="big">Sealed delta vs baseline e000: \+0\.1234 \(an improvement had to exceed 0\.0123;')
        self.assertIn("(selected, optimistic)", html_text)
        self.assertLess(md.index("Sealed delta"), md.index("selected, optimistic"))

    def test_no_promotion_means_no_sealed_headline_number(self):
        led = Ledger.create(self.dir, clock=FakeClock(), fsync=False)
        self.addCleanup(led.close)
        rb = RunBuilder(led)
        rb.start("t")
        rb.baseline(1.0)
        rb.experiment("A", "x", score=1.0, delta=0.0)
        md = render_markdown(self.dir)
        self.assertIn("no replicated improvement to report", md)
        self.assertNotIn("selected, optimistic", md)

    def test_selection_statistics(self):
        self.make_run()
        md = render_markdown(self.dir)
        # tested: A, B, C, D = 4; dev-improved: A, B = 2; sealed queries: A and B replicated (baseline's sealed grade is not a query)
        self.assertIn("tested: 4 experiments (n_tested = hypotheses that reached evaluation", md)
        self.assertIn("improved on dev: 2", md)
        self.assertIn("sealed-split queries: 2", md)
        self.assertIn("n_tested * 0.05 * 0.025 = 0.0050", md)
        self.assertIn("adaptive worst case = 1 - 0.975^n_sealed_queries = 0.0494", md)       # 2 sealed queries
        h = render_html(self.dir)
        for token in ("n_tested", "n_dev_improved", "n_sealed_queries", "expected false promotions under the null",
                      "adaptive worst case", "bounds the sealed stage alone"):
            self.assertIn(token, h)
        self.assertAlmostEqual(report.FALSE_PROMOTION_RATE, 0.00125)

    def test_headline_block(self):
        self.make_run()
        md = render_markdown(self.dir)
        row = [ln for ln in md.splitlines() if ln.startswith("| ") and "x |" in ln and "$" in ln][0]
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(cells[0:5], ["1", "0", "0", "1", "1.00x"])      # claims made, accepted, refused, blocked, speedup
        self.assertEqual(cells[5], "$0.0123")
        for label in ("claims made", "claims accepted", "claims refused", "unreplicated improvements blocked", "speedup", "cost"):
            self.assertIn(label, md)
            self.assertIn(label, render_html(self.dir))

    def test_experiment_table(self):
        self.make_run()
        md = render_markdown(self.dir)
        for token in ("quadratic term", "overfit idea", "improved", "no_gain", "+0.1234", "tamper", "0.8000"):
            self.assertIn(token, md)
        rows = [ln for ln in md.split("## Experiments")[1].split("## Findings")[0].splitlines() if re.match(r"\| e00\d ", ln)]
        self.assertEqual(len(rows), 5)

    def test_findings_evidence_status_and_dismissed_section(self):
        self.make_run()
        md = render_markdown(self.dir)
        self.assertIn("I02@e002", md)
        self.assertIn("5, 6", md)
        self.assertIn("open", md)
        head, _, dismissed = md.partition("### Dismissed findings")
        self.assertIn("I04@e004", head.split("## Findings")[1])
        section = dismissed.split("## ")[0]
        self.assertIn("I04@e004", section)
        self.assertIn("the repeat was intentional, different seeds were used on purpose", section)
        self.assertNotIn("I02@e002", section)
        h = render_html(self.dir)
        self.assertIn("<h3>Dismissed findings</h3>", h)
        self.assertIn("the repeat was intentional", h)

    def test_invariants_coverage_shows_llm_not_run_plainly(self):
        self.make_run()
        md = render_markdown(self.dir)
        self.assertIn("LLM not run", md)
        i13 = [ln for ln in md.splitlines() if ln.startswith("| I13 |")][0]
        self.assertIn("LLM not run", i13)
        self.assertNotIn("checked, no open finding", i13)
        i02 = [ln for ln in md.splitlines() if ln.startswith("| I02 |")][0]
        self.assertIn("VIOLATED", i02)
        i04 = [ln for ln in md.splitlines() if ln.startswith("| I04 |")][0]
        self.assertIn("checked, no open finding", i04)          # its only finding was dismissed
        self.assertIn("not checked", [ln for ln in md.splitlines() if ln.startswith("| I10 |")][0])

    def test_fault_injection_scoreboard_for_scripted_runs(self):
        self.make_run()
        md = render_markdown(self.dir)
        self.assertIn("## Fault-injection scoreboard", md)
        # planted: tamper (no finding: missed), overfit (I02 raised: caught), claim_without_run (missed) -> 1/3
        self.assertIn("Planted faults caught (recall): 1/3 = 33%", md)
        self.assertIn("False positives: 1", md)       # the dismissed I04 on an unplanted experiment
        self.assertIn("MISSED", md)
        self.assertIn("Fault-injection scoreboard", render_html(self.dir))

    def test_scoreboard_separates_not_evaluated_defect_and_premature_finish(self):
        def extra(led, rb):
            inj = rb.experiment("E", "hostile output", planted="injection")
            d = rb.experiment("F", "buggy first draft", planted="defect", review="repair", recorded=False)
            led.append(Kind.EXP_REPAIRED, "implementer:F", {"exp_id": d, "repair_round": 1, "diff_hash": "d" * 12})
            led.append(Kind.EXP_REVIEWED, "reviewer:F", {"exp_id": d, "review_round": 1, "reviewer": "r", "verdict": "accept",
                                                         "findings": [], "independent": True})
            led.append(Kind.EXP_RECORDED, "engine", {"exp_id": d, "phase": "lane", "outcome": "no_gain", "delta": 0.0,
                                                     "noise_margin": 0.01, "replicated": False})
            honest = rb.experiment("G", "honest idea")
            c = rb.claim(honest, "improved", "it helps")
            led.append(Kind.SUP_FINISH_REQUESTED, "supervisor", {"summary": "s", "claim_ids": [c]})
            rb.finding("I09", honest, exp_id=honest)
            led.append(Kind.SUP_FINISH_REFUSED, "engine", {"reasons": ["too early"]})
        self.make_run(extra=extra)
        md = render_markdown(self.dir)
        # planted: tamper (missed), overfit (caught), claim_without_run (missed), injection (not evaluated),
        # defect (caught), premature finish (caught) -> 3/5 scored
        self.assertIn("Planted faults caught (recall): 3/5 = 60%.", md)
        self.assertIn("Not evaluated: 1 (needs an LLM judge; not run offline, excluded from recall)", md)
        row = [ln for ln in md.splitlines() if "| injection | I13 |" in ln][0]
        self.assertIn("not evaluated", row)
        self.assertIn("needs an LLM judge; not run offline", row)
        self.assertNotIn("MISSED", row)
        self.assertIn("reviewer found the planted bug; repaired and re-reviewed", md)
        self.assertIn("premature_finish", md)
        self.assertIn("refused by the finish gate", md)
        self.assertIn("Findings explained by a planted premature finish", md)
        html = render_html(self.dir)
        self.assertIn("not evaluated", html)
        self.assertIn("Findings explained by a planted premature finish", html)

    def test_no_scoreboard_for_a_run_without_planted_faults(self):
        self.make_run(backend="meta")
        self.assertNotIn("Fault-injection scoreboard", render_markdown(self.dir))

    def test_models_usage_table(self):
        self.make_run()
        md = render_markdown(self.dir)
        row = [ln for ln in md.splitlines() if ln.startswith("| supervisor | muse-spark-1.3")][0]
        self.assertIn("| n/a |", row)
        self.assertIn("0.0123", row)
        self.assertNotIn("saving", md.lower())

    def test_integrity_block(self):
        led = self.make_run(isolation="userns")
        seq, h = led.head_anchor()
        for text in (render_markdown(self.dir), render_html(self.dir)):
            self.assertIn(f"{seq}:{h}", text)
            self.assertIn("chain consistent", text)
            self.assertIn("userns", text)
            self.assertNotIn("secure", text.lower())
            self.assertIn("--expect-head", text)

    def test_isolation_none_is_called_out(self):
        self.make_run(isolation="none")
        md = render_markdown(self.dir)
        self.assertIn("Without isolation", md)
        self.assertIn("hash checks", md)

    def test_a_tampered_ledger_is_reported_as_inconsistent(self):
        self.make_run()
        path = self.dir / "ledger.jsonl"
        lines = path.read_text().splitlines()
        row = json.loads(lines[6])
        row["data"]["x"] = 1
        lines[6] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        for text in (render_markdown(self.dir), render_html(self.dir)):
            self.assertIn("CHAIN INCONSISTENT at seq 7", text)
            self.assertNotIn("chain consistent,", text)

    def test_output_is_deterministic(self):
        self.make_run()
        self.assertEqual(render_html(self.dir), render_html(self.dir))
        self.assertEqual(render_markdown(self.dir), render_markdown(self.dir))

    def test_missing_ledger_raises(self):
        with self.assertRaises(FileNotFoundError):
            render_html(self.dir)
        with self.assertRaises(FileNotFoundError):
            render_markdown(self.dir)
        self.assertFalse(self.dir.exists())

    def test_write_reports(self):
        self.make_run()
        out = Path(self._tmp.name) / "out"
        md, html_path = write_reports(self.dir, out)
        self.assertEqual((md.name, html_path.name), ("report.md", "report.html"))
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["report.html", "report.md"])
        self.assertEqual(md.read_text(encoding="utf-8"), render_markdown(self.dir))
        md2, _ = write_reports(self.dir)
        self.assertEqual(md2, self.dir / "report.md")


class FigureTests(ReportCase):
    def test_score_figure_has_baseline_incumbent_and_markers(self):
        self.make_run()
        svgs = re.findall(r"<svg.*?</svg>", render_html(self.dir), re.S)
        score = svgs[0]
        self.assertIn('class="baseline"', score)
        self.assertEqual(score.count('class="incumbent"'), 1)
        self.assertEqual(score.count('class="m-base"'), 1)
        self.assertEqual(score.count('class="m-prom"'), 1)
        self.assertEqual(score.count('class="m-cand"'), 3)         # B, C, D
        self.assertIn("Dev score over experiments", score)
        self.assertIn('role="img"', score)

    def test_incumbent_line_steps_down_at_the_promotion(self):
        self.make_run()
        root = ET.fromstring(re.findall(r"<svg.*?</svg>", render_html(self.dir), re.S)[0])
        poly = [e for e in root.iter() if e.get("class") == "incumbent"][0]
        ys = [float(p.split(",")[1]) for p in poly.get("points").split()]
        xs = [float(p.split(",")[0]) for p in poly.get("points").split()]
        self.assertEqual(ys[0], ys[1])                              # flat at the baseline until the promotion
        self.assertGreater(ys[-1], ys[1])                           # score fell 1.0 -> 0.8 (minimise): the line steps down the page
        self.assertEqual(xs, sorted(xs))

    def test_gap_figure_draws_train_and_heldout_per_experiment(self):
        self.make_run()
        gap = re.findall(r"<svg.*?</svg>", render_html(self.dir), re.S)[1]
        self.assertEqual(gap.count('class="m-train"'), 4)
        self.assertEqual(gap.count('class="m-held"'), 4)
        self.assertEqual(gap.count('class="gap"'), 4)
        ET.fromstring(gap)

    def test_without_a_train_metric_the_gap_figure_is_replaced_by_a_note(self):
        self.make_run(train=False)
        h = render_html(self.dir)
        self.assertEqual(len(re.findall(r"<svg", h)), 1)
        self.assertIn("No train metric was recorded", h)

    def test_no_graded_experiments_gives_a_note_not_a_crash(self):
        led = Ledger.create(self.dir, clock=FakeClock(), fsync=False)
        self.addCleanup(led.close)
        RunBuilder(led).start("t")
        h = render_html(self.dir)
        self.assertIn("no score figure", h)
        self.assertNotIn("<svg", h)
        self.assertIn("<!doctype html>", h)


class AdversarialReportTests(ReportCase):
    def test_a_timestamp_no_calendar_can_show_does_not_abort_the_report(self):
        for ts in (1e30, -1e30, 253402300800.0):
            self.setUp()
            led = Ledger.create(self.dir, clock=lambda: ts, fsync=False)
            self.addCleanup(led.close)
            RunBuilder(led).start("t")
            self.assertIn("started n/a", render_markdown(self.dir))
            self.assertIn("started n/a", render_html(self.dir))

    def test_backslashes_in_ledger_text_cannot_unescape_a_table_separator(self):
        self.make_run(hypothesis="ends with a backslash \\ and a pipe \\| here")
        md = render_markdown(self.dir)
        rows = [ln for ln in md.split("## Experiments")[1].split("## Findings")[0].splitlines() if ln.startswith("| e00")]
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({len(re.split(r"(?<!\\)\|", ln)) for ln in rows}), 1)
        self.assertNotIn("\\\\|", md)


class AnsweredModelsTests(ReportCase):
    """The report separates what the run was configured to use from what the charges show answering."""
    WONDER, SPARK = "muse-wonder-1.0", "muse-spark-1.3"

    def charges(self, *specs):
        """specs: (actor, model, calls). Appended after the one supervisor charge make_run writes."""
        def extra(led, rb):
            n = 0
            for actor, model, calls in specs:
                for _ in range(calls):
                    n += 1
                    led.append(Kind.BUDGET_CHARGED, actor, {"kind": "llm", "amount": 0.001, "unit": "usd", "call_id": f"x{n}",
                                                           "model": model, "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
        return extra

    CONFIGURED = {"supervisor": "muse-spark-1.3", "reviewer": "muse-spark-1.3", "reviewer_hard": "muse-wonder-1.0",
                  "auditor_final": "muse-wonder-1.0"}

    def md(self, *specs, models=None, **kw):
        self.make_run(backend="meta", models=models or self.CONFIGURED, extra=self.charges(*specs), **kw)
        return render_markdown(self.dir)

    def test_wonder_is_not_credited_when_spark_answered_every_hard_call(self):
        md = self.md(("reviewer_hard", self.SPARK, 3), ("auditor_final", self.SPARK, 2), ("reviewer", self.SPARK, 4))
        self.assertNotIn("muse-wonder", md)
        self.assertIn("Models that answered (from the charged calls, with call counts): muse-spark-1.3 (10 calls).", md)
        self.assertIn("The configured model of auditor_final, reviewer_hard is not listed: it never answered a charged call", md)
        self.assertIn("reviewer=muse-spark-1.3", md)

    def test_the_configured_line_is_worded_as_configuration_not_as_fact(self):
        md = self.md(("reviewer_hard", self.SPARK, 1))
        self.assertIn("Configured model labels (what the run asked for, not proof of which model answered):", md)
        self.assertNotIn("Model labels recorded in the ledger", md)
        self.assertNotIn("Models recorded in the ledger", md)

    def test_wonder_is_named_when_a_charge_shows_it_answering(self):
        md = self.md(("reviewer_hard", self.WONDER, 2), ("reviewer_hard", self.SPARK, 1), ("auditor_final", self.WONDER, 1))
        self.assertIn("muse-wonder-1.0 (3 calls)", md)
        self.assertIn("reviewer_hard=muse-wonder-1.0", md)                              # it did answer, so its label stays
        self.assertIn("auditor_final=muse-wonder-1.0", md)
        self.assertNotIn("is not listed", md)

    def test_a_role_with_no_charged_call_keeps_its_configured_label_and_is_not_counted_as_answering(self):
        md = self.md(("reviewer", self.SPARK, 1))
        self.assertIn("reviewer_hard=muse-wonder-1.0", md)                              # configured, never called
        self.assertNotIn("muse-wonder-1.0 (", md)                                       # but nothing says it answered

    def test_counts_are_per_model_and_use_the_roles_after_the_colon_split(self):
        md = self.md(("designer:A", self.SPARK, 2), ("designer:B", self.SPARK, 1), ("reviewer_hard", "muse-spark-1.3-0615", 1),
                     models={"designer": self.SPARK, "reviewer_hard": self.SPARK})
        self.assertIn("muse-spark-1.3 (4 calls), muse-spark-1.3-0615 (1 call).", md)   # +1 supervisor charge from make_run
        self.assertNotIn("is not listed", md)                                           # a version suffix still counts as the same model

    def test_no_charged_call_says_so_instead_of_inventing_an_answer(self):
        d2 = self.dir
        led2 = Ledger.create(d2, clock=FakeClock(), fsync=False)
        self.addCleanup(led2.close)
        led2.append(Kind.RUN_STARTED, "engine", {"task": "t", "backend": "meta", "models": self.CONFIGURED, "limits": {},
                                                 "task_hash": "t" * 12, "grader_hash": "g" * 12, "objective": "minimize",
                                                 "primary_metric": "m", "isolation": "none", "config": {}})
        text = render_markdown(d2)
        self.assertIn("No model call was charged, so the ledger shows no model answering.", text)
        self.assertIn("reviewer_hard=muse-wonder-1.0", text)
        self.assertNotIn("Models that answered", text)

    def test_the_html_report_says_the_same_and_escapes_model_names(self):
        self.make_run(backend="meta", models={"supervisor": "m<script>"}, extra=self.charges(("supervisor", "<b>evil</b>", 1)))
        page = render_html(self.dir)
        self.assertNotIn("<b>evil</b>", page)
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;b&gt;evil&lt;/b&gt; (1 call)", page)
        self.assertIn("Models that answered", page)
        md = render_markdown(self.dir)
        self.assertNotIn("<b>evil</b>", md)

    def test_a_simulated_run_still_says_it_is_a_simulation_and_names_the_answering_model(self):
        md = self.md(("reviewer_hard", self.SPARK, 1), ("auditor_final", self.SPARK, 1), endpoint={"kind": "loopback", "hosts": ["127.0.0.1:1"]})
        self.assertIn("a simulation of the live code path and not a Meta model result", md)
        # a simulator only echoes the model name it was asked for: the report must not word that as a model that answered
        self.assertIn("Models the simulator reported answering (a stand-in, not Meta)", md)
        self.assertNotIn("Models that answered", md)
        self.assertNotIn("muse-wonder", md)

    def test_a_host_that_answers_under_a_new_name_every_time_cannot_swamp_the_report(self):
        specs = [("reviewer", f"made-up-model-{i:03d}-" + "x" * 3000, 1) for i in range(60)]
        md = self.md(*specs, models={"reviewer": self.SPARK})
        line = next(ln for ln in md.splitlines() if "Models that answered" in ln)
        self.assertLess(len(line), 3000)
        self.assertIn("and 49 other model names (49 calls)", line)                      # 60 + 1 supervisor charge, 12 shown
        self.assertIn("...", line)

    def test_a_scripted_run_claims_no_model_at_all(self):
        self.make_run(backend="scripted")
        md = render_markdown(self.dir)
        self.assertIn("scripted (no language model)", md)
        self.assertNotIn("Models that answered", md)



class ChainReportTests(ReportCase):
    """After a promotion chain the headline is the improvement over the BASELINE (what the finish gate decides on)."""

    SEALED = (1.0, 0.9077, 0.8422)           # baseline, e001, e002 on the sealed split: steps +0.0923 and +0.0655, total +0.1578

    def replicate_vs(self, led: Ledger, rb: RunBuilder, exp: str, score: float, step: float, margin: float, vs: str) -> None:
        rb.replicate(exp, score, sealed_delta=step, sealed_margin=margin)
        led.append(Kind.EXP_RECORDED, "engine", {"exp_id": exp, "phase": "replication", "outcome": "improved", "vs_exp_id": vs,
                                                 "sealed_delta": step, "sealed_margin": margin, "replicated": True, "note": ""})

    def make_chain(self, run_started_extra: dict | None = None, noise_margin: float = 0.02, finish: bool = True,
                   second_vs: str | None = None) -> tuple[str, str, str]:
        led = Ledger.create(self.dir, clock=FakeClock(), fsync=False)
        self.addCleanup(led.close)
        rb = RunBuilder(led)
        led.append(Kind.RUN_STARTED, "engine", {
            "task": "no-such-task", "backend": "scripted", "models": {"supervisor": "scripted"}, "limits": {"max_replications": 6},
            "task_hash": "t" * 12, "grader_hash": "g" * 12, "objective": "minimize", "primary_metric": "heldout", "config": {},
            **(run_started_extra or {})})
        base = rb.baseline(1.0)
        rb.replication_baseline(base, self.SEALED[0])
        e1 = rb.experiment("A", "first idea", score=0.8, delta=0.2, noise_margin=noise_margin, vs=base)
        self.replicate_vs(led, rb, e1, self.SEALED[1], 0.0923, 0.0269, base)
        rb.promote(e1, prev=base, score=0.8)
        e2 = rb.experiment("B", "second idea", score=0.7, delta=0.1, noise_margin=noise_margin, vs=e1)
        self.replicate_vs(led, rb, e2, self.SEALED[2], 0.0655, 0.0269, second_vs or e1)
        rb.promote(e2, prev=e1, score=0.7)
        if finish:
            led.append(Kind.RUN_FINISHED, "engine", {"reason": "supervisor_finish", "summary": "done", "verdict_class": "success",
                                                     "final_claims": []})
        return base, e1, e2

    def task(self, **kw) -> SimpleNamespace:
        return SimpleNamespace(**{"minimize": True, "noise_margin": 0.02, "sigma_floor": 0.02, "train_metric": None, **kw})

    def headline(self, text: str) -> str:
        return next(ln for ln in text.splitlines() if "Sealed delta" in ln)

    def test_the_headline_is_the_improvement_over_the_baseline_not_over_the_previous_incumbent(self):
        base, e1, e2 = self.make_chain()
        md = render_markdown(self.dir, self.task())
        line = self.headline(md)
        self.assertIn(f"Sealed delta vs baseline {base}: +0.1578", line)
        self.assertNotIn("+0.0655", line)
        # t(0.975, df 2) * sigma_floor 0.02 / sqrt(3) = 0.0497, above noise_margin 0.02: the threshold an improvement had to exceed
        self.assertIn("an improvement had to exceed 0.0497", line)
        self.assertNotIn("margin", line)

    def test_the_headline_equals_what_the_finish_gate_decides_on(self):
        from nightshift.view import build_view
        from nightshift.workflows import gates
        base, e1, e2 = self.make_chain()
        task = self.task()
        decision = gates.sealed_decision(task, build_view(Ledger.read_only(self.dir).events()), e2, base)
        self.assertEqual(decision.outcome, "improved")
        self.assertIn(f"{decision.delta:+.4f} (an improvement had to exceed {decision.margin:.4f}",
                      self.headline(render_markdown(self.dir, task)))

    def test_the_chain_line_lists_each_step_with_its_own_comparison(self):
        base, e1, e2 = self.make_chain()
        md = render_markdown(self.dir, self.task())
        self.assertIn("Promotion chain: each step below is a sealed comparison with the PREVIOUS incumbent, not with the baseline", md)
        self.assertIn(f"{e1} +0.0923 vs {base}", md)
        self.assertIn(f"{e2} +0.0655 vs {e1}", md)

    def test_html_carries_the_same_headline_and_chain(self):
        base, e1, e2 = self.make_chain()
        text = render_html(self.dir, self.task())
        self.assertIn(f"Sealed delta vs baseline {base}: +0.1578", text)
        self.assertIn(f"{e2} +0.0655 vs {e1}", text)
        self.assertNotIn("+0.0655 (an improvement had to exceed 0.0497", text)

    def test_a_single_promotion_has_no_chain_line(self):
        self.make_run()
        self.assertNotIn("Promotion chain", render_markdown(self.dir))

    def test_the_task_parameters_can_come_from_run_started(self):
        self.make_chain({"noise_margin": 0.02, "sigma_floor": 0.036})
        # t(0.975, df 2) * 0.036 / sqrt(3) = 0.0894
        self.assertIn("an improvement had to exceed 0.0894", self.headline(render_markdown(self.dir)))

    def test_without_task_parameters_the_report_says_the_threshold_may_be_understated(self):
        base, e1, e2 = self.make_chain()
        md = render_markdown(self.dir)                       # task name unknown, no parameters in the ledger
        self.assertIn(f"Sealed delta vs baseline {base}: +0.1578", self.headline(md))
        self.assertIn("sigma floor was not available to the report, so the threshold above may be understated", md)

    def test_a_baseline_referenced_number_that_cannot_be_computed_is_not_replaced_by_the_step_number(self):
        # no sealed grade of the baseline exists in this ledger: no comparison with the baseline can be computed, so none is shown
        led = Ledger.create(self.dir, clock=FakeClock(), fsync=False)
        self.addCleanup(led.close)
        rb = RunBuilder(led)
        rb.start("no-such-task")
        b = rb.baseline(1.0)
        e1 = rb.experiment("A", "first idea", score=0.8, delta=0.2, vs=b)
        self.replicate_vs(led, rb, e1, 0.9, 0.1, 0.01, b)
        rb.promote(e1, prev=b)
        e2 = rb.experiment("B", "second idea", score=0.7, delta=0.1, vs=e1)
        self.replicate_vs(led, rb, e2, 0.8, 0.1, 0.01, e1)
        rb.promote(e2, prev=e1)
        md = render_markdown(self.dir, self.task())
        line = self.headline(md)
        self.assertIn("not available in this report", line)
        self.assertNotIn("+0.1000", line)
        self.assertIn(f"{e2} +0.1000 vs {e1}", md)

    def test_hostile_ids_in_the_chain_line_cannot_inject_markup(self):
        # the comparison base of the last step comes from the ledger as written: it is escaped like every other dynamic string
        self.make_chain(second_vs=XSS_IMG)
        md = render_markdown(self.dir, self.task())
        self.assertNotIn(XSS_IMG, md)
        self.assertIn("&lt;img src=x onerror=1&gt;", md)
        text = render_html(self.dir, self.task())
        self.assertNotIn("<img", text)
        self.assertNotIn("<script", text)
        self.assertIn("&lt;img src=x onerror=1&gt;", text)

    def test_the_experiments_table_labels_the_step_wise_and_selected_columns(self):
        self.make_chain()
        md = render_markdown(self.dir, self.task())
        self.assertIn("dev delta (selected)", md)
        self.assertIn("sealed delta (vs its base)", md)
        self.assertIn("so it is optimistic", md)
        self.assertNotIn("| dev delta |", md)


class SelectionStatisticsTests(ReportCase):
    def stats_run(self, *, max_replications: int | None = None) -> None:
        led = Ledger.create(self.dir, clock=FakeClock(), fsync=False)
        self.addCleanup(led.close)
        rb = RunBuilder(led)
        led.append(Kind.RUN_STARTED, "engine", {
            "task": "t", "backend": "scripted", "models": {"supervisor": "scripted"},
            "limits": {} if max_replications is None else {"max_replications": max_replications},
            "task_hash": "t" * 12, "grader_hash": "g" * 12, "objective": "minimize", "primary_metric": "heldout", "config": {}})
        base = rb.baseline(1.0)
        rb.replication_baseline(base, 1.0)
        rb.experiment("A", "graded fine", score=0.9, delta=0.1)
        failed = rb.experiment("B", "graded but failed", score=None, graded=False, delta=0.0)
        led.append(Kind.EXP_GRADED, "engine", {"exp_id": failed, "purpose": "primary", "grade": grade_dict(0.0, valid=False),
                                               "grader_hash": "g" * 12, "seeds": [0, 1, 2], "conditions_hash": rb.cond})
        rb.experiment("C", "invalid, never graded", score=None, implemented=False, ran=False, graded=False, outcome="invalid",
                      delta=None)
        for exp in ("e001", failed):
            rb.replicate(exp, 0.95, success=False)

    def test_n_tested_counts_hypotheses_that_reached_evaluation_including_a_failed_grade_but_not_an_ungraded_invalid_one(self):
        self.stats_run()
        md = render_markdown(self.dir)
        self.assertIn("tested: 2 experiments (n_tested = hypotheses that reached evaluation", md)
        self.assertIn("one that was invalid and never graded is not counted", md)
        self.assertIn("n_tested * 0.05 * 0.025 = 0.0025", md)

    def test_the_adaptive_worst_case_is_one_minus_0_975_to_the_sealed_queries(self):
        self.stats_run()
        md = render_markdown(self.dir)
        self.assertIn("sealed-split queries: 2", md)
        self.assertIn("adaptive worst case = 1 - 0.975^n_sealed_queries = 0.0494", md)      # 1 - 0.975^2
        self.assertIn("assumes every dev draw is independent", md)
        self.assertIn("bounds the sealed stage alone", md)

    def test_the_sealed_query_count_in_the_worst_case_is_capped_by_max_replications(self):
        self.stats_run(max_replications=1)
        md = render_markdown(self.dir)
        self.assertIn("capped at max_replications = 1", md)
        self.assertIn("adaptive worst case = 1 - 0.975^n_sealed_queries = 0.0250", md)


LOOPBACK = {"kind": "loopback", "hosts": ["127.0.0.1:1"]}
RT3 = [{"behavior": "tamper", "scripted_id": "tamper"}, {"behavior": "overfit", "scripted_id": "poly9"},
       {"behavior": "claim_without_run", "scripted_id": "claim_without_run"}]


class NotRunScoreboardTests(ReportCase):
    """A planted lane that a gate stopped before it ran is "not run": counted on its own, never in recall, and the banner says so."""

    def planted_run(self, *, blocked=("RT-overfit", "RT-claim_without"), red_team=RT3, endpoint=LOOPBACK, finish=True):
        def extra(led, rb):
            t = rb.experiment("RT-tamper", "clip", protected=["run.py"], planted="tamper", scripted_id="tamper", recorded=False)
            rb.finding("I03", t, exp_id=t)
            for lane in blocked:
                led.append(Kind.EXP_BLOCKED, "engine", {"exp_id": None, "lane": lane, "hypothesis": "h", "hyp_fingerprint": "x",
                                                        "reason": "duplicate", "detail": "duplicate of e002: the same hypothesis was already tried",
                                                        "dup_of": "e002"})
        self.make_run(backend="meta", models={"supervisor": "muse-spark-1.3"}, endpoint=endpoint, config_extra={"red_team": red_team},
                      extra=extra, finish=finish)

    def test_the_banner_says_how_many_planted_lanes_ran(self):
        self.planted_run()
        for text in (render_markdown(self.dir), render_html(self.dir)):
            self.assertIn("RED-TEAM LANES: 3 lane(s) in this run are scripted attackers injected by the operator.", text)
            self.assertIn("3 attacker lane(s) were planted; 1 ran, 2 were blocked as duplicates before they ran.", text)
            self.assertLess(text.index("RED-TEAM LANES"), text.index("SIMULATED MODEL"))

    def test_the_scoreboard_lists_the_unrun_lanes_counts_them_and_leaves_them_out_of_recall(self):
        self.planted_run()
        md = render_markdown(self.dir)
        self.assertIn("Planted faults caught (recall): 1/1 = 100%. Planted faults that never ran: 2 (", md)
        self.assertIn("excluded from recall", md)
        rows = [ln for ln in md.splitlines() if ln.startswith("| RT-")]
        self.assertEqual(len(rows), 2)
        for ln, fault in zip(rows, ("overfit", "claim_without_run")):
            self.assertIn(f"| {fault} |", ln)
            self.assertIn("| not run |", ln)
            self.assertIn("blocked as a duplicate before it ran: an identical idea had already been proposed (e002)", ln)
            self.assertNotIn("MISSED", ln)
        self.assertIn("| tamper | I03 | detected |", md)
        html = render_html(self.dir)
        self.assertIn("Planted faults that never ran: 2", html)
        self.assertIn("RT-overfit (no experiment was created)", html)

    def test_hostile_block_text_in_a_not_run_row_is_inert_in_both_formats(self):
        def extra(led, rb):
            led.append(Kind.EXP_BLOCKED, "engine", {"exp_id": None, "lane": "RT-overfit", "hypothesis": "h", "hyp_fingerprint": "x",
                                                    "reason": XSS_SCRIPT + "|\n| x | y |", "detail": XSS_SCRIPT, "dup_of": XSS_SCRIPT})
        self.make_run(backend="meta", models={"supervisor": "muse-spark-1.3"}, endpoint=LOOPBACK,
                      config_extra={"red_team": RT3[1:2]}, extra=extra)
        md, html = render_markdown(self.dir), render_html(self.dir)
        self.assertNotIn("<script", md.lower())
        self.assertNotIn("<script", html.lower())
        row = next(ln for ln in md.splitlines() if ln.startswith("| RT-"))
        head = next(ln for ln in md.splitlines() if ln.startswith("| experiment | fault |"))
        cells = lambda ln: len(re.split(r"(?<!\\)\|", ln))
        self.assertEqual(cells(row), cells(head))                                              # one table row, no injected cells
        self.assertIn("| not run |", row)

    def test_when_every_planted_lane_ran_the_sentence_says_all_and_no_never_ran_count_appears(self):
        self.planted_run(blocked=(), red_team=RT3[:1])
        md = render_markdown(self.dir)
        self.assertIn("All 1 planted attacker lane(s) ran.", md)
        self.assertNotIn("never ran", md)
        self.assertNotIn("| not run |", md)

    def test_a_scripted_run_scoreboard_is_word_for_word_what_it_was(self):
        self.make_run()
        md = render_markdown(self.dir)
        self.assertIn("Planted faults caught (recall): 1/3 = 33%. False positives: 1.", md)
        self.assertNotIn("never ran", md)
        self.assertNotIn("planted attacker lane", md)

    def test_a_run_that_is_still_going_says_the_lane_is_not_launched_yet(self):
        self.planted_run(blocked=(), finish=False)
        md = render_markdown(self.dir)
        self.assertIn("3 attacker lane(s) were planted; 1 ran, 2 were not launched yet.", md)
        self.assertIn("not launched yet: the run is still going", md)


class SimulatedWordingTests(ReportCase):
    """A report about a run against the simulator never lets a stand-in read as a Meta model result."""

    SPARK = "muse-spark-1.3"

    def sim(self, **kw):
        def extra(led, rb):
            for i in range(3):
                led.append(Kind.BUDGET_CHARGED, "supervisor", {"kind": "llm", "amount": 0.01, "unit": "usd", "call_id": f"s{i}",
                                                              "model": self.SPARK, "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
            led.append(Kind.AUDIT_WINDOW, "auditor", {"from_seq": 1, "to_seq": 40, "invariants": ["I07", "I13", "I04"],
                                                      "judged": ["I07", "I13"]})
        self.make_run(backend="meta", models={"supervisor": self.SPARK}, endpoint=kw.pop("endpoint", LOOPBACK), extra=extra, **kw)
        return render_markdown(self.dir), render_html(self.dir)

    def test_the_answering_models_are_named_as_what_the_simulator_reported(self):
        for text in self.sim():
            self.assertIn("Models the simulator reported answering (a stand-in, not Meta)", text)
            self.assertIn("muse-spark-1.3 (4 calls)", text)
            self.assertNotIn("Models that answered", text)

    def test_every_cost_figure_is_a_simulated_list_price_estimate(self):
        md, html = self.sim()
        label = "simulated list-price estimate (nothing was billed)"
        self.assertIn("| simulated cost |", md)
        self.assertIn(f"| {label}: $0.0423 |", md)
        self.assertIn("usd (simulated list-price estimate, nothing was billed)", md)
        self.assertNotIn("| usd |", md)
        self.assertIn(label, html)
        self.assertRegex(html, r"<b>simulated list-price estimate \(nothing was billed\): \$0\.0423</b>simulated cost")
        self.assertIn("usd (simulated list-price estimate, nothing was billed)", html)

    def test_the_llm_halves_of_invariants_are_checked_by_a_simulated_judge(self):
        md, html = self.sim()
        for text in (md, html):
            for inv_id in ("I07", "I13"):
                self.assertRegex(text, inv_id + r".{0,160}checked by a simulated judge")
        i07 = [ln for ln in md.splitlines() if ln.startswith("| I07 |")][0]
        self.assertNotIn("| checked, no open finding |", i07)
        self.assertIn("a scripted stand-in, not a model", i07)
        i04 = [ln for ln in md.splitlines() if ln.startswith("| I04 |")][0]
        self.assertIn("checked, no open finding", i04)                       # a deterministic invariant is plain checked
        i08 = [ln for ln in md.splitlines() if ln.startswith("| I08 |")][0]
        self.assertIn("LLM not run", i08)                                     # never judged: still not shown as passing

    def test_the_final_summary_is_attributed_to_the_simulated_supervisor(self):
        md, html = self.sim()
        self.assertIn("Final summary (written by the simulated supervisor (a scripted stand-in), not a model):", md)
        self.assertIn("written by the simulated supervisor (a scripted stand-in), not a model", html)
        self.assertNotIn("written by the supervisor)", md)

    def test_the_simulated_banner_stays_and_live_and_scripted_wording_is_unchanged(self):
        md, _ = self.sim()
        self.assertIn("SIMULATED MODEL", md)
        self.setUp()
        self.make_run(backend="meta", models={"supervisor": self.SPARK}, endpoint={"kind": "meta-api", "hosts": ["api.meta.ai"]},
                      extra=lambda led, rb: led.append(Kind.AUDIT_WINDOW, "auditor", {"from_seq": 1, "to_seq": 9, "invariants": ["I07"],
                                                                                  "judged": ["I07"]}))
        live = render_markdown(self.dir)
        self.assertIn("Models that answered (from the charged calls, with call counts): muse-spark-1.3 (1 call).", live)
        self.assertIn("| $0.0123 |", live)
        self.assertIn("| usd |", live)
        self.assertIn("Final summary (written by the supervisor):", live)
        for phrase in ("simulat", "stand-in"):
            self.assertNotIn(phrase, live.lower())
        self.assertIn("| I07 | Infrastructure is not model failure | both | checked, no open finding |", live)
        self.setUp()
        self.make_run()
        scripted = render_markdown(self.dir)
        self.assertIn("Final summary (written by the script, not a model):", scripted)
        self.assertNotIn("simulated judge", scripted)

    def test_a_plant_run_against_the_simulator_gets_the_same_discipline(self):
        def extra(led, rb):
            led.append(Kind.BUDGET_CHARGED, "supervisor", {"kind": "llm", "amount": 0.01, "unit": "usd", "call_id": "p",
                                                          "model": self.SPARK, "usage": {}})
        self.make_run(backend="meta", models={"supervisor": self.SPARK}, endpoint=LOOPBACK, config_extra={"red_team": RT3[:1]}, extra=extra)
        md = render_markdown(self.dir)
        self.assertIn("RED-TEAM LANES", md)
        self.assertIn("Models the simulator reported answering (a stand-in, not Meta)", md)
        self.assertIn("simulated list-price estimate (nothing was billed)", md)


class IsolationWarningTests(ReportCase):
    def integrity(self, iso):
        self.make_run(isolation=iso)
        return [ln for ln in render_markdown(self.dir).splitlines() if ln.startswith("* Isolation:")][0]

    def test_unshare_net_is_warned_about_like_none_with_accurate_wording(self):
        line = self.integrity("unshare-net")
        self.assertIn("can still read the labels", line)
        self.assertIn("a read leaves no trace", line)
        self.assertIn("as root", line)
        self.assertIn("API key", line)
        self.assertIn("change to the trusted directory", line)
        self.assertIn("blocks network connections", line)
        self.assertIn("not a security boundary", line)
        self.assertNotIn("sealed-safe", line.lower())
        self.assertIn("can still read the labels", self.dirs_html())

    def dirs_html(self):
        return render_html(self.dir)

    def test_none_keeps_its_wording_and_userns_keeps_the_plain_line(self):
        self.assertIn("Without isolation, same-user experiment code could read the trusted directory", self.integrity("none"))
        self.setUp()
        self.assertEqual(self.integrity("userns"), "* Isolation: userns. Best-effort containment, not a security boundary.")


class BrokenChainReportTests(ReportCase):
    def break_chain(self):
        path = self.dir / "ledger.jsonl"
        lines = path.read_text().splitlines()
        row = json.loads(lines[6])
        row["data"]["x"] = 1
        lines[6] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")

    def test_a_broken_chain_opens_both_formats_with_a_banner_and_marks_the_verdict_unverified(self):
        self.make_run()
        self.break_chain()
        md, html = render_markdown(self.dir), render_html(self.dir)
        banner = "LEDGER CHAIN BROKEN at seq 7: this report was written from a ledger that fails verification; do not trust it"
        self.assertTrue(md.startswith(f"> **{banner}**"), md[:200])
        self.assertLess(md.index(banner), md.index("# Nightshift report"))
        self.assertIn("## Verdict: success (unverified)", md)
        self.assertRegex(html, r'<body>\s*<p class="banner alarm" role="alert">' + re.escape(banner) + "</p>")
        self.assertLess(html.index(banner), html.index("<h1>"))
        self.assertIn("<h2>Verdict: success (unverified)</h2>", html)
        self.assertIn("CHAIN INCONSISTENT at seq 7", md)                 # the Integrity section still says it

    def test_an_intact_chain_has_no_banner_and_a_plain_verdict(self):
        self.make_run()
        md, html = render_markdown(self.dir), render_html(self.dir)
        for text in (md, html):
            self.assertNotIn("LEDGER CHAIN BROKEN", text)
            self.assertNotIn("unverified", text)
        self.assertTrue(md.startswith("# Nightshift report"))
        self.assertIn("## Verdict: success\n", md)

    def test_a_ledger_without_a_verdict_still_says_unverified_next_to_no_verdict(self):
        self.make_run(finish=False)
        self.break_chain()
        self.assertIn("## Verdict: no verdict (unverified)", render_markdown(self.dir))

    def test_the_render_signatures_are_unchanged(self):
        import inspect
        for fn in (render_markdown, render_html):
            self.assertEqual(list(inspect.signature(fn).parameters), ["run_dir", "task"])


if __name__ == "__main__":
    unittest.main()
