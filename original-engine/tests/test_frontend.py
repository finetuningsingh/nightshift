"""Dashboard frontend tests.

* ``StaticAssetTests`` need nothing but Python: they read the three static files and the two sample states.
* ``BrowserTests`` run ``tests/frontend/check.js`` (node + the preinstalled Playwright) and are skipped, never failed,
  when node or playwright is unavailable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_SAMPLES = ("state_scripted_sample.json", "state_live_sample.json", "state_simulated_sample.json", "state_scripted_real.json",
               "state_simulated_real.json", "state_redteam_sim.json", "state_redteam_partial.json")
DASH = ROOT / "nightshift" / "dashboard"
FIXTURES = ROOT / "tests" / "fixtures"
CHECK_JS = ROOT / "tests" / "frontend" / "check.js"
SKIP_EXIT = 77

STATE_KEYS = {"run", "banner", "budget", "baseline", "incumbent", "experiments", "series", "claims", "finish", "lessons",
              "pivots", "findings", "invariants", "models_usage", "turns", "timeline", "parallel", "fault_injection",
              "headline", "ledger", "replay"}
EXPERIMENT_KEYS = {"exp_id", "lane", "hypothesis", "status", "stage", "worker", "reviewer", "outcome", "dev_outcome", "score",
                   "train", "delta", "sealed_delta", "train_delta", "review", "replicated", "awaiting_replication",
                   "diff_summary", "parent_exp_id", "blocked_reason", "protected", "claim_ids", "finding_ids",
                   "planted_fault", "started_ts", "ended_ts", "cost_usd", "tokens", "llm_calls", "cost_by_role"}
HEADLINE_COST_KEYS = {"cost_usd", "cost_known", "llm_calls", "cost_attributed_usd", "cost_unattributed_usd", "promoted",
                      "cost_per_improvement_usd"}


class _Tags(HTMLParser):
    """Collects every start tag with its attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.script_bodies: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script and data.strip():
            self.script_bodies.append(data)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class StaticAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = _read(DASH / "index.html")
        cls.js = _read(DASH / "app.js")
        cls.css = _read(DASH / "app.css")
        parser = _Tags()
        parser.feed(cls.html)
        cls.parsed = parser

    def test_files_exist_and_are_utf8(self) -> None:
        for name in ("index.html", "app.js", "app.css"):
            self.assertTrue((DASH / name).is_file(), name)
            (DASH / name).read_bytes().decode("utf-8")

    def test_html_has_no_inline_script_or_style(self) -> None:
        for tag, attrs in self.parsed.tags:
            self.assertNotEqual(tag, "style", "inline <style> is forbidden by the CSP")
            if tag == "script":
                self.assertTrue(attrs.get("src"), "every <script> must load a file")
        self.assertEqual(self.parsed.script_bodies, [], "no inline script body")
        self.assertFalse([1 for _, attrs in self.parsed.tags if "style" in attrs], "no style attribute")
        self.assertNotIn("unsafe-inline", self.html + self.js + self.css)

    def test_html_has_no_inline_event_handlers_or_javascript_urls(self) -> None:
        for tag, attrs in self.parsed.tags:
            for name, value in attrs.items():
                self.assertFalse(name.lower().startswith("on"), f"<{tag} {name}> is an inline handler")
                self.assertFalse((value or "").strip().lower().startswith("javascript:"), f"<{tag} {name}> javascript: url")

    def test_html_requests_nothing_external(self) -> None:
        for tag, attrs in self.parsed.tags:
            for name in ("src", "href", "action", "poster", "srcset", "data"):
                value = (attrs.get(name) or "").strip()
                if not value:
                    continue
                self.assertTrue(value.startswith(("/", "#", "data:")), f"<{tag} {name}={value[:40]!r}> is not same-origin")
        self.assertEqual({a["src"] for t, a in self.parsed.tags if t == "script"}, {"/app.js"})
        self.assertEqual({a["href"] for t, a in self.parsed.tags if t == "link" and a.get("rel") == "stylesheet"}, {"/app.css"})
        self.assertNotRegex(self.html, r"(?i)<(iframe|object|embed|form|base)\b")

    def test_css_has_no_external_references(self) -> None:
        self.assertNotRegex(self.css, r"(?i)@import")
        for url in re.findall(r"url\(([^)]*)\)", self.css, flags=re.I):
            self.assertTrue(url.strip("'\" ").startswith("data:"), f"css url({url}) is not a data URI")
        self.assertNotRegex(self.css, r"(?i)https?://")
        self.assertNotRegex(self.css, r"(?i)@font-face")

    def test_css_defines_both_themes_with_the_required_guards(self) -> None:
        self.assertIn("prefers-color-scheme: dark", self.css)
        self.assertIn(':root:not([data-theme="light"])', self.css)
        self.assertIn(':root[data-theme="dark"]', self.css)
        self.assertIn("prefers-reduced-motion", self.css)
        self.assertIn(":focus-visible", self.css)
        self.assertRegex(self.css, r"font-variant-numeric:\s*tabular-nums")
        for token in ("--ok", "--bad", "--warn", "--neutral", "--scripted", "--live"):
            self.assertIn(token + ":", self.css)

    def test_js_never_parses_markup_or_evaluates_strings(self) -> None:
        forbidden = {
            "innerHTML": r"\binnerHTML\b",
            "outerHTML": r"\bouterHTML\b",
            "insertAdjacentHTML": r"\binsertAdjacentHTML\b",
            "document.write": r"\bdocument\s*\.\s*write(ln)?\s*\(",
            "eval": r"(?<![\w.])eval\s*\(",
            "Function constructor": r"(?<![\w.])(new\s+)?Function\s*\(",
            "string timer": r"\bset(Timeout|Interval)\s*\(\s*['\"`]",
            "DOMParser": r"\bDOMParser\b",
            "createContextualFragment": r"\bcreateContextualFragment\b",
            "import()": r"\bimport\s*\(",
            "srcdoc": r"\bsrcdoc\b",
            "setAttribute style": r"setAttribute\(\s*['\"]style['\"]",
            "cssText": r"\bcssText\b",
            "XMLHttpRequest": r"\bXMLHttpRequest\b",
            "WebSocket": r"\bWebSocket\b",
            "sendBeacon": r"\bsendBeacon\b",
            "postMessage": r"\bpostMessage\b",
            "document.cookie": r"\bdocument\s*\.\s*cookie\b",
        }
        for label, pattern in forbidden.items():
            self.assertIsNone(re.search(pattern, self.js), f"app.js must not use {label}")

    def test_js_only_talks_to_the_state_endpoint(self) -> None:
        fetches = re.findall(r"\bfetch\s*\(([^,)]*)", self.js)
        self.assertEqual([f.strip() for f in fetches], ["'/api/state'"])
        urls = set(re.findall(r"https?://[^\s'\"`)]+", self.js))
        self.assertEqual(urls, {"http://www.w3.org/2000/svg"}, "the only URL in app.js is the SVG namespace")

    def test_js_dynamic_attributes_go_through_an_allow_list(self) -> None:
        self.assertIn("SAFE_ATTR", self.js)
        # the generic element factory must refuse handler and style attributes
        pattern = re.search(r"const SAFE_ATTR = /(.*)/;", self.js)
        self.assertIsNotNone(pattern)
        allow = re.compile(pattern.group(1))
        for ok in ("class", "aria-label", "data-act", "tabindex", "role", "title"):
            self.assertTrue(allow.match(ok), ok)
        for bad in ("onclick", "onerror", "style", "href", "src", "srcdoc", "formaction", "xlink:href"):
            self.assertFalse(allow.match(bad), bad)

    def test_css_dimming_never_drops_text_below_aa(self) -> None:
        """Opacity multiplies into text contrast: a skipped step name at 0.8 measured 3.98:1, a struck-out filter chip at 0.6 3.3:1."""
        self.assertNotRegex(self.css, r"\.step\.is-skipped\s*\{[^}]*opacity")
        for sel, value in re.findall(r"(\.chip-toggle:not\(\.is-on\))\s*\{[^}]*?opacity:\s*([0-9.]+)", self.css):
            self.assertGreaterEqual(float(value), 0.75, sel)

    def test_js_lookups_by_state_value_use_own_properties(self) -> None:
        """A state value such as "constructor" must not resolve through Object.prototype."""
        for table in ("INV_STATE", "INV_KIND", "CLAIM_STATUS", "STATUS_ACTIVE", "STAGE_INDEX"):
            self.assertNotRegex(self.js, table + r"\[\s*str\(", f"{table} must be read through own()")

    def test_js_banner_texts_match_the_state_api(self) -> None:
        from nightshift.dashboard.state_api import BANNER, SIMULATED_BANNER
        def js_const(name: str) -> str:
            m = re.search(r"const " + name + r" = ((?:'[^']*'(?:\s*\+\s*)?)+);", self.js)
            self.assertIsNotNone(m, name)
            return "".join(re.findall(r"'([^']*)'", m.group(1)))
        self.assertEqual(js_const("SCRIPTED_BANNER"), BANNER)
        self.assertEqual(js_const("SIMULATED_BANNER"), SIMULATED_BANNER)

    def test_js_simulated_labels_match_the_state_api_and_the_report(self) -> None:
        from nightshift import report
        from nightshift.dashboard import state_api
        pairs = (("SIM_COST_LABEL", state_api.SIMULATED_COST_LABEL), ("SIM_MODELS_LABEL", state_api.SIMULATED_MODELS_LABEL),
                 ("SIM_JUDGE_LABEL", state_api.SIMULATED_JUDGE_LABEL), ("SIM_SUPERVISOR_LABEL", state_api.SIMULATED_SUPERVISOR_LABEL),
                 ("SIM_RED_TEAM_NOTE", state_api.SIMULATED_RED_TEAM_NOTE))
        for name, expected in pairs:
            m = re.search(r"const " + name + r" = ((?:'[^']*'(?:\s*\+\s*)?)+);", self.js)
            self.assertIsNotNone(m, name)
            self.assertEqual("".join(re.findall(r"'([^']*)'", m.group(1))), expected, name)
        # the report words the same four things with the same strings
        self.assertIs(report.SIMULATED_COST_LABEL, state_api.SIMULATED_COST_LABEL)
        self.assertIs(report.SIMULATED_SUPERVISOR_LABEL, state_api.SIMULATED_SUPERVISOR_LABEL)

    def test_the_page_reads_the_new_scoreboard_keys_through_the_total_helpers(self) -> None:
        ids = {a.get("id") for _, a in self.parsed.tags}
        self.assertIn("usage-usd-h", ids)
        self.assertIn("not_run: arr(fi.not_run)", self.js)
        self.assertNotRegex(self.js, r"fault_injection\.red_team\.")
        self.assertIn("ok_simulated", self.js)
        # the unshare-net isolation mode warns like none and never calls itself sealed-safe
        self.assertIn("iso === 'unshare-net'", self.js)
        self.assertNotRegex(self.js, r"(?i)sealed-safe")

    def test_js_red_team_notice_matches_the_state_api_and_has_no_dismiss_control(self) -> None:
        from nightshift.dashboard.state_api import RED_TEAM_NOTICE
        m = re.search(r"const RED_TEAM_NOTICE = ((?:'[^']*'(?:\s*\+\s*)?)+);", self.js)
        self.assertIsNotNone(m)
        self.assertEqual("".join(re.findall(r"'([^']*)'", m.group(1))), RED_TEAM_NOTICE)
        # index.html has no slot for it: the page builds it, never with a control that could close it
        start = self.js.index("function renderRedTeam")
        body = self.js[start:self.js.index("function renderReplay", start)]
        self.assertNotRegex(body, r"h\(\s*'(button|a|input)'")
        self.assertNotIn("addEventListener", body)
        steps = self.js.index("const steps = [")
        self.assertIn("renderRedTeam", self.js[steps:self.js.index("];", steps)])

    def test_simulated_has_its_own_tone_in_every_theme_and_the_page_has_a_models_line(self) -> None:
        self.assertEqual(len(re.findall(r"--sim:", self.css)), 3, "light, dark media query and data-theme dark")
        self.assertIn(".tone-simulated", self.css)
        self.assertIn(".banner.is-simulated", self.css)
        ids = {a.get("id") for _, a in self.parsed.tags}
        self.assertTrue({"models-line", "faults-note", "faults-lists", "faults-score", "banner-text"} <= ids)
        self.assertIn("'SIMULATED'", self.js)
        self.assertIn("not_evaluated", self.js)

    def test_the_banner_has_no_dismiss_control(self) -> None:
        start = self.html.index('id="banner"')
        block = self.html[start:self.html.index("</div>", start)]
        self.assertNotRegex(block, r"<(button|a|input)\b")

    def test_the_page_has_the_cost_and_header_hooks_and_guards_every_new_key(self) -> None:
        ids = {a.get("id") for _, a in self.parsed.tags}
        self.assertTrue({"run-implementer", "run-implementer-wrap"} <= ids)
        self.assertIn("lc-cost", self.js)
        self.assertIn("cost_per_improvement_usd", self.js)
        self.assertIn("par.reason", self.js)
        # every new state key is read through the total helpers, never indexed raw
        for raw in (r"S\.headline\.cost_usd\.", r"e\.cost_by_role\.", r"run\.models_answered\.", r"par\.reason\.",
                    r"\be\.cost_usd\.toFixed"):
            self.assertNotRegex(self.js, raw)

    def test_the_readme_screenshots_exist(self) -> None:
        for name in ("dashboard-scripted-light.png", "dashboard-scripted-dark.png", "dashboard-mobile.png"):
            data = (ROOT / "docs" / "img" / name).read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n", name)
            self.assertGreater(len(data), 20000, name)

    def test_js_parses(self) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        out = subprocess.run([node, "--check", str(DASH / "app.js")], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)


def _simulated_banner() -> str:
    from nightshift.dashboard.state_api import SIMULATED_BANNER
    return SIMULATED_BANNER


class SampleStateTests(unittest.TestCase):
    """The hand-built states must follow docs/ARCHITECTURE.md section 10 so the page is tested against the real shape."""

    def load(self, name: str) -> dict:
        return json.loads(_read(FIXTURES / name))

    def test_every_sample_has_every_top_level_key(self) -> None:
        for name in ALL_SAMPLES:
            state = self.load(name)
            self.assertEqual(STATE_KEYS - set(state), set(), name)
            for exp in state["experiments"]:
                self.assertEqual(EXPERIMENT_KEYS - set(exp), set(), f"{name} {exp.get('exp_id')}")

    def test_scripted_sample_is_rich_and_honest(self) -> None:
        s = self.load("state_scripted_sample.json")
        self.assertEqual((s["run"]["backend"], s["run"]["mode"]), ("scripted", "scripted"))
        self.assertTrue(s["banner"] and s["banner"].startswith("SCRIPTED BACKEND"))
        self.assertGreaterEqual(len(s["experiments"]), 12)
        self.assertEqual(len(s["invariants"]), 14)
        self.assertTrue(s["run"]["finished"])
        self.assertTrue(s["ledger"]["chain_ok"])
        self.assertTrue(all(t["source"] == "script" for t in s["turns"]))
        self.assertEqual({i["state"] for i in s["invariants"]} - {"ok", "violated", "llm_not_run"}, set())
        fi = s["fault_injection"]
        self.assertTrue(fi["planted"] and fi["detected"] and fi["not_evaluated"])
        # every fault kind the scripted bank can plant is on the board
        self.assertGreaterEqual({p["fault"] for p in fi["planted"]},
                                {"tamper", "hang", "overfit", "claim_without_run", "injection", "duplicate", "infra_flake",
                                 "defect", "premature_finish"})
        self.assertEqual({p["fault"] for p in fi["not_evaluated"]}, {"injection"})
        self.assertEqual(fi["not_evaluated"][0]["reason"], "needs an LLM judge; not run offline")
        self.assertIn("premature_finish", {p["fault"] for p in fi["detected"]})
        self.assertEqual(len(fi["planted"]), len(fi["detected"]) + len(fi["missed"]) + len(fi["not_evaluated"]))
        self.assertEqual(fi["recall"]["scored"], len([p for p in fi["planted"] if not p["negative"] and p["result"] != "not_evaluated"]))
        self.assertEqual(s["finish"]["accepted_claims"], [c["claim_id"] for c in s["claims"] if c["accepted"]])
        outcomes = {e["outcome"] for e in s["experiments"]}
        self.assertTrue({"baseline", "improved", "no_gain", "worse", "failed", "invalid", "infra_error"} <= outcomes)
        self.assertTrue({c["status"] for c in s["claims"]} >= {"supported", "contradicted", "refused"})
        self.assertTrue(any(e["blocked_reason"] for e in s["experiments"]))
        self.assertTrue(any(e["protected"] for e in s["experiments"]))

    def test_live_sample_is_a_run_in_progress(self) -> None:
        s = self.load("state_live_sample.json")
        self.assertEqual((s["run"]["backend"], s["run"]["mode"], s["run"]["origin"]), ("meta", "live", "live"))
        self.assertEqual(s["run"]["endpoint"], {"kind": "meta-api", "hosts": ["api.meta.ai"]})
        self.assertIsNone(s["banner"])
        self.assertFalse(s["run"]["finished"])
        self.assertTrue(any(e["status"] not in ("recorded", "blocked", "violation") for e in s["experiments"]))
        self.assertTrue(any(u["cache"] == "known" for u in s["models_usage"]))
        self.assertTrue(all(t["source"] == "supervisor" for t in s["turns"]))
        self.assertEqual(s["fault_injection"]["planted"], [])
        self.assertTrue(any(i["state"] == "insufficient" for i in s["invariants"]))

    def test_simulated_sample_is_a_local_test_server_run(self) -> None:
        s = self.load("state_simulated_sample.json")
        self.assertEqual((s["run"]["backend"], s["run"]["mode"], s["run"]["origin"]), ("meta", "simulated", "simulated"))
        self.assertEqual(s["run"]["endpoint"]["kind"], "loopback")
        self.assertTrue(all(h.startswith("127.0.0.1") for h in s["run"]["endpoint"]["hosts"]))
        self.assertEqual(s["banner"], _simulated_banner())
        self.assertNotEqual(s["run"]["run_id"], self.load("state_live_sample.json")["run"]["run_id"])

    def test_real_scripted_sample_comes_from_a_cli_run(self) -> None:
        s = self.load("state_scripted_real.json")
        self.assertEqual((s["run"]["backend"], s["run"]["mode"]), ("scripted", "scripted"))
        self.assertGreaterEqual((len(s["experiments"]), len(s["findings"]), len(s["claims"]), len(s["turns"])), (12, 8, 13, 10))
        self.assertTrue(s["run"]["summary"])
        self.assertEqual(s["run"]["verdict_class"], "success")
        states = {i["id"]: i["state"] for i in s["invariants"]}
        for llm_id in ("I07", "I08", "I13", "I14"):
            self.assertEqual(states[llm_id], "llm_not_run")
        self.assertEqual(sum(1 for v in states.values() if v == "ok"), 10)
        fi = s["fault_injection"]
        results = {p["fault"]: p["result"] for p in fi["planted"]}
        self.assertEqual((results["defect"], results["injection"], results["premature_finish"]), ("detected", "not_evaluated", "detected"))
        self.assertEqual(sorted(x["invariant_id"] for x in fi["explained"]), ["I09", "I14"])
        self.assertEqual(fi["missed"], [])
        self.assertTrue(s["ledger"]["chain_ok"])

    def test_simulated_real_sample_is_a_recorded_loopback_run_with_attributed_cost(self) -> None:
        s = self.load("state_simulated_real.json")
        self.assertEqual((s["run"]["backend"], s["run"]["mode"], s["run"]["origin"]), ("meta", "simulated", "simulated"))
        self.assertEqual(s["run"]["endpoint"]["kind"], "loopback")
        self.assertEqual(s["banner"], _simulated_banner())
        self.assertTrue(s["ledger"]["chain_ok"] and s["run"]["finished"])
        self.assertEqual(s["run"]["implementer"], "patch")
        self.assertTrue(any(e["cost_usd"] for e in s["experiments"]))
        self.assertTrue(s["headline"]["promoted"] >= 1 and s["headline"]["cost_per_improvement_usd"])

    def test_red_team_sample_is_a_recorded_simulated_meta_run_with_three_attackers(self) -> None:
        s = self.load("state_redteam_sim.json")
        self.assertEqual((s["run"]["backend"], s["run"]["mode"], s["run"]["origin"]), ("meta", "simulated", "simulated"))
        self.assertEqual(s["banner"], _simulated_banner())                  # the simulated banner stays; the notice is separate
        self.assertEqual([r["behavior"] for r in s["run"]["red_team"]], ["tamper", "overfit", "claim_without_run"])
        self.assertEqual(s["run"]["verdict_class"], "success")
        self.assertTrue(s["ledger"]["chain_ok"] and s["run"]["finished"])
        planted = {e["lane"]: e["planted_fault"] for e in s["experiments"] if e["planted_fault"]}
        self.assertEqual(planted, {"RT-tamper": "tamper", "RT-overfit": "overfit", "RT-claim_without": "claim_without_run"})
        fi = s["fault_injection"]
        self.assertEqual((fi["recall"]["caught"], fi["recall"]["scored"]), (3, 3))
        self.assertEqual(fi["missed"], [])
        self.assertTrue(all(t["source"] == "supervisor" for t in s["turns"]))   # no script narration in a meta run
        for e in s["experiments"]:
            if e["planted_fault"]:
                self.assertEqual(e["worker"] in ("scripted-implementer", ""), True, e["exp_id"])

    def test_red_team_partial_sample_has_two_planted_lanes_that_never_ran(self) -> None:
        s = self.load("state_redteam_partial.json")
        self.assertEqual((s["run"]["backend"], s["run"]["origin"]), ("meta", "simulated"))
        self.assertEqual(STATE_KEYS - set(s), set())
        self.assertTrue(s["ledger"]["chain_ok"] and s["run"]["finished"])
        fi = s["fault_injection"]
        self.assertEqual({p["fault"]: p["result"] for p in fi["planted"]},
                         {"tamper": "detected", "overfit": "not_run", "claim_without_run": "not_run"})
        self.assertEqual((fi["recall"]["caught"], fi["recall"]["scored"], fi["recall"]["not_run"]), (1, 1, 2))
        self.assertEqual((fi["red_team"]["planted"], fi["red_team"]["ran"], fi["red_team"]["blocked_duplicate"]), (3, 1, 2))
        self.assertEqual(len(fi["planted"]), len(fi["detected"]) + len(fi["missed"]) + len(fi["not_evaluated"]) + len(fi["not_run"]))
        self.assertTrue(all(x["exp_id"] is None and x["lane"].startswith("RT-") for x in fi["not_run"]))
        self.assertEqual({i["state"] for i in s["invariants"] if i["id"] in ("I07", "I08", "I13", "I14")}, {"ok_simulated"})

    def test_cost_fields_are_consistent_in_every_sample(self) -> None:
        """Per-experiment costs add up to the attributed total, which never exceeds the run total; scripted runs have none."""
        for name in ALL_SAMPLES:
            s = self.load(name)
            h = s["headline"]
            self.assertEqual(HEADLINE_COST_KEYS - set(h), set(), name)
            self.assertEqual({"implementer", "models_answered"} - set(s["run"]), set(), name)
            self.assertEqual({"max_parallel", "reason"} - set(s["parallel"]), set(), name)
            costs = [e["cost_usd"] for e in s["experiments"] if e["cost_usd"] is not None]
            if s["run"]["origin"] == "scripted":
                self.assertEqual((costs, h["cost_known"], h["cost_usd"], h["cost_per_improvement_usd"]), ([], False, 0.0, None), name)
                self.assertEqual(s["run"]["models_answered"], {}, name)
                self.assertEqual(s["run"]["implementer"], "scripted", name)
                continue
            self.assertTrue(h["cost_known"], name)
            self.assertAlmostEqual(sum(costs), h["cost_attributed_usd"], places=4, msg=name)
            self.assertLessEqual(h["cost_attributed_usd"], h["cost_usd"] + 1e-9, name)
            self.assertAlmostEqual(h["cost_attributed_usd"] + h["cost_unattributed_usd"], h["cost_usd"], places=4, msg=name)
            self.assertAlmostEqual(h["cost_usd"], sum(u["usd"] for u in s["models_usage"]), places=3, msg=name)
            self.assertAlmostEqual(h["cost_per_improvement_usd"], h["cost_usd"] / h["promoted"], places=5, msg=name)
            for e in s["experiments"]:
                if e["cost_usd"] is None:
                    self.assertEqual((e["tokens"], e["llm_calls"], e["cost_by_role"]), (None, None, {}), name)
                else:
                    self.assertAlmostEqual(sum(e["cost_by_role"].values()), e["cost_usd"], places=4, msg=name)

    def test_samples_reference_only_existing_seqs_and_experiments(self) -> None:
        for name in ALL_SAMPLES:
            s = self.load(name)
            seqs = {t["seq"] for t in s["timeline"]}
            ids = {e["exp_id"] for e in s["experiments"]}
            for f in s["findings"]:
                self.assertTrue(set(f["evidence_seqs"]) <= seqs, f"{name} {f['finding_id']}")
                self.assertTrue(f["exp_id"] is None or f["exp_id"] in ids)
            for c in s["claims"]:
                self.assertIn(c["exp_id"], ids | {None})
            self.assertEqual(len(seqs), len(s["timeline"]), "timeline seqs are unique")

    def test_samples_contain_no_sealed_score_leak_in_free_text(self) -> None:
        # sealed numbers may appear only in the clearly named sealed_delta field
        for name in ALL_SAMPLES:
            s = self.load(name)
            blob = json.dumps([s["timeline"], s["turns"], s["lessons"], s["claims"], s["series"]])
            self.assertNotIn("0.314159265", blob)


class BrowserTests(unittest.TestCase):
    def test_dashboard_in_chromium(self) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        probe = subprocess.run([node, "-e", "require('playwright')"], capture_output=True, text=True, timeout=60, cwd=str(ROOT))
        if probe.returncode != 0:
            self.skipTest("playwright is not available to node")
        with tempfile.TemporaryDirectory() as shots:
            env = dict(os.environ, NIGHTSHIFT_SHOTS_DIR=shots)
            out = subprocess.run([node, str(CHECK_JS)], capture_output=True, text=True, timeout=240, cwd=str(ROOT), env=env)
            if out.returncode == SKIP_EXIT:
                self.skipTest("chromium is not available: " + out.stdout.strip()[-200:])
            self.assertEqual(out.returncode, 0, f"check.js failed\n{out.stdout[-6000:]}\n{out.stderr[-2000:]}")
            produced = {p.name for p in Path(shots).iterdir()}
            for name in ("desktop-light.png", "desktop-dark.png", "mobile.png", "invariants.png", "simulated-header.png",
                         "fault-scoreboard.png", "real-desktop-light.png", "mobile-real.png", "cost-lanes.png", "cost-headline.png",
                         "sim-real-desktop-light.png", "mobile-costs.png", "replay-scripted.png", "redteam-header.png",
                         "redteam-mobile.png", "redteam-partial.png", "simulated-labels.png", "simulated-labels-mobile.png"):
                self.assertIn(name, produced)
                self.assertGreater((Path(shots) / name).stat().st_size, 5000, name)


if __name__ == "__main__":
    unittest.main()
