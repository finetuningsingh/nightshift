"""Tests for nightshift.models.live_smoke: every probe, run against the fake Meta server (loopback only, no external network).

The fake follows the documented API (``tests/support/fake_meta_server.py``); these tests prove what each probe sends, what it
concludes, that a reply cut off by the token cap is INCONCLUSIVE rather than a failure, and that the report never holds the key
or any message content.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nightshift.models import live_smoke
from nightshift.models.base import strictify_schema
from nightshift.models.meta import MetaChatClient
from nightshift.types import ModelError
from tests.support.fake_meta_server import FakeMetaServer, Fault, Reply

SPARK, WONDER = "muse-spark-1.3", "muse-wonder-1.0"


class Sleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class SmokeCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeMetaServer().start()
        self.addCleanup(self.fake.stop)

    def factory(self, **kw):
        return live_smoke.build_client_factory(base_url=self.fake.base_url, env={"MODEL_API_KEY": self.fake.api_key}, **kw)

    def run_smoke(self, **kw) -> dict:
        return live_smoke.run_smoke(self.factory(), **kw)

    @staticmethod
    def by_name(report: dict) -> dict:
        return {p["name"]: p for p in report["probes"]}

    def bodies_where(self, predicate) -> list[dict]:
        return [b for b in self.fake.bodies() if predicate(b)]


class ReportShapeTests(SmokeCase):
    def test_every_probe_is_recorded_with_a_status_and_the_baseline_passes(self):
        self.fake.ratelimit_headers = True
        report = self.run_smoke()
        probes = self.by_name(report)
        self.assertEqual(report["version"], 2)
        self.assertEqual((report["ok"], report["required_failed"], report["required_inconclusive"], report["conclusive"]),
                         (True, [], [], True))
        for name in ("effort_max", "effort_none_rejected", "strict_PLAN", "strict_REVIEW", "strict_REFLECT", "strict_VERDICT",
                     "supervisor_tools", "temperature_with_reasoning", "completion_cap_ceiling", "assistant_tool_calls_content",
                     "tools_with_response_format", "system_and_developer_roles", "ratelimit_headers", "prompt_cache_large"):
            with self.subTest(name):
                self.assertEqual((probes[name]["status"], probes[name]["required"]), ("pass", False))
        # the fake never issues two tool calls on its own, so the parallel probe says so instead of failing
        self.assertEqual(probes["parallel_tool_calls"]["status"], "inconclusive")
        self.assertTrue(probes["parallel_tool_calls"]["detail"]["message"])
        self.assertTrue(all(p["status"] in ("pass", "fail", "inconclusive") for p in report["probes"]))

    def test_required_probes_leave_room_for_reasoning(self):
        self.run_smoke()
        required_chat = self.bodies_where(lambda b: b.get("max_completion_tokens") == 2000 and b.get("reasoning_effort") == "minimal")
        self.assertGreaterEqual(len(required_chat), 6)
        self.assertEqual(live_smoke.DEFAULT_MAX_COMPLETION_TOKENS, 2000)
        report = self.run_smoke(max_completion_tokens=3000)
        self.assertEqual(report["max_completion_tokens"], 3000)
        self.assertTrue(self.bodies_where(lambda b: b.get("max_completion_tokens") == 3000))

    def test_the_report_holds_no_key_and_no_message_content_from_the_new_probes(self):
        report = self.run_smoke()
        text = json.dumps(report)
        self.assertNotIn(self.fake.api_key, text)
        for fragment in ("twice in the same turn", "exactly one word", "large caching probe", "item3799", "Say ready",
                         "Add 2 and 3, then answer", "Return a minimal", "Reply with: ok"):
            self.assertNotIn(fragment, text)

    def test_the_report_is_written_as_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo" / "live_probe.json"
            report = self.run_smoke(out_path=out)
            self.assertEqual(json.loads(out.read_text()), json.loads(json.dumps(report)))

    def test_the_agent_schemas_failing_to_import_is_one_failed_probe_not_an_abort(self):
        with mock.patch.object(live_smoke, "_agent_schemas", side_effect=ImportError("boom")):
            report = self.run_smoke()
        probes = self.by_name(report)
        self.assertEqual(probes["strict_agent_schemas"]["status"], "fail")
        self.assertEqual(probes["strict_agent_schemas"]["error"]["kind"], "ImportError")
        self.assertEqual((probes["supervisor_tools"]["status"], report["ok"]), ("pass", True))


class InconclusiveTests(SmokeCase):
    """Muse Spark always reasons and reasoning counts against max_completion_tokens: a cut-off reply proves nothing."""

    def test_a_cap_eaten_by_reasoning_is_inconclusive_never_a_failure(self):
        self.fake.simulate_reasoning = 2500                       # more than the default cap of 2000
        report = self.run_smoke()
        probes = self.by_name(report)
        for name in ("tool_roundtrip", "strict_schema", "stream"):
            with self.subTest(name):
                self.assertEqual(probes[name]["status"], "inconclusive")
                self.assertFalse(probes[name]["ok"])
                self.assertEqual(probes[name]["detail"]["finish_reason"], "length")
                self.assertIn("length", probes[name]["detail"]["message"])
                self.assertIn("max_completion_tokens", probes[name]["detail"]["message"])
        self.assertEqual(report["required_inconclusive"], ["tool_roundtrip", "strict_schema", "stream"])
        self.assertEqual((report["required_failed"], report["ok"], report["conclusive"]), ([], True, False))
        # the effort probes test what was SENT, which the cut-off does not change
        self.assertTrue(all(probes[f"effort_{e}"]["status"] == "pass" for e in ("minimal", "low", "medium", "high", "xhigh")))

    def test_the_same_server_with_room_to_think_is_conclusive(self):
        self.fake.simulate_reasoning = 2500
        report = self.run_smoke(max_completion_tokens=6000)
        self.assertEqual((report["required_inconclusive"], report["conclusive"]), ([], True))
        self.assertEqual(self.by_name(report)["tool_roundtrip"]["status"], "pass")

    def test_tool_result_turn_cut_off_is_inconclusive(self):
        # first turn answers with the tool call; the tool-result turn runs out of tokens
        calls = {"n": 0}

        def handler(req):
            if any(m["role"] == "tool" for m in req.body["messages"]) and req.body.get("tools") and calls["n"] == 0:
                calls["n"] += 1
                return Reply(content="", finish_reason="length")
            return None
        self.fake.set_handler(handler)
        probe = self.by_name(self.run_smoke())["tool_roundtrip"]
        self.assertEqual(probe["status"], "inconclusive")
        self.assertEqual(probe["detail"]["second_finish_reason"], "length")

    def test_a_real_failure_is_still_a_failure(self):
        self.fake.enqueue(Reply.text("I will not call the tool"))              # finish_reason stop, no tool call
        report = self.run_smoke()
        self.assertEqual((report["required_failed"], self.by_name(report)["tool_roundtrip"]["status"]), (["tool_roundtrip"], "fail"))
        self.assertFalse(report["ok"])

    def test_summarize_shows_the_message_of_an_inconclusive_probe(self):
        self.fake.simulate_reasoning = 2500
        lines = live_smoke.summarize(self.run_smoke())
        joined = "\n".join(lines)
        self.assertIn("????  tool_roundtrip", joined)
        self.assertIn("finish_reason was 'length'", joined)
        self.assertTrue(lines[-1].startswith("overall: OK [inconclusive: tool_roundtrip, strict_schema, stream]"))


class StreamProbeTests(SmokeCase):
    def test_the_stream_probe_sends_include_usage_and_requires_real_usage(self):
        report = self.run_smoke()
        stream_bodies = self.bodies_where(lambda b: b.get("stream"))
        self.assertEqual([b["stream_options"] for b in stream_bodies], [{"include_usage": True}])
        detail = self.by_name(report)["stream"]["detail"]
        self.assertEqual((detail["usage_estimated"], detail["include_usage_stripped"]), (False, False))
        self.assertEqual(self.by_name(report)["stream"]["status"], "pass")

    def test_an_estimated_stream_usage_fails_the_probe_and_says_why(self):
        self.fake.stream_usage = False                              # an endpoint that never sends the usage chunk
        report = self.run_smoke()
        probe = self.by_name(report)["stream"]
        self.assertEqual((probe["status"], "stream" in report["required_failed"]), ("fail", True))
        self.assertTrue(probe["detail"]["usage_estimated"])
        self.assertIn("include_usage", probe["detail"]["problem"])
        self.assertIn("under-charge", probe["detail"]["problem"])

    def test_a_host_that_rejects_stream_options_fails_the_stream_probe(self):
        with FakeMetaServer(unsupported_params=("stream_options",)) as fake:
            report = live_smoke.run_smoke(live_smoke.build_client_factory(
                base_url=fake.base_url, env={"MODEL_API_KEY": fake.api_key}))
        probe = self.by_name(report)["stream"]
        self.assertEqual(probe["status"], "fail")
        self.assertTrue(probe["detail"]["include_usage_stripped"])


class SchemaProbeTests(SmokeCase):
    def test_each_agent_schema_is_sent_strictified_as_the_live_path_sends_it(self):
        from nightshift.agents import prompts
        from nightshift.audit import judge
        self.run_smoke()
        sent = {b["response_format"]["json_schema"]["name"]: b["response_format"]["json_schema"] for b in self.fake.bodies()
                if b.get("response_format", {}).get("json_schema", {}).get("name", "").startswith("smoke_") and "tools" not in b}
        for label, schema in (("plan", prompts.PLAN_SCHEMA), ("review", prompts.REVIEW_SCHEMA), ("reflect", prompts.REFLECT_SCHEMA),
                              ("verdict", judge.VERDICT_SCHEMA)):
            with self.subTest(label):
                self.assertEqual(sent[f"smoke_{label}"]["schema"], strictify_schema(schema))
                self.assertTrue(sent[f"smoke_{label}"]["strict"])

    def test_the_supervisor_tools_are_sent_exactly_as_the_orchestrator_defines_them(self):
        self.run_smoke()
        bodies = self.bodies_where(lambda b: len(b.get("tools") or []) == 6)
        self.assertEqual(len(bodies), 1)
        names = [t["function"]["name"] for t in bodies[0]["tools"]]
        self.assertEqual(names, ["get_state", "launch_experiments", "replicate", "reflect", "acknowledge_finding", "finish"])
        launch = next(t for t in bodies[0]["tools"] if t["function"]["name"] == "launch_experiments")["function"]["parameters"]
        lane = launch["properties"]["lanes"]["items"]["properties"]["lane"]
        self.assertEqual((lane["minLength"], lane["maxLength"]), (1, 16))
        self.assertIn("pattern", lane)

    def test_a_rejected_tool_keyword_is_learned_per_tool(self):
        self.fake.rejected_schema_keywords = {"minLength"}
        report = self.run_smoke()
        probe = self.by_name(report)["supervisor_tools"]
        self.assertEqual(probe["status"], "fail")
        detail = probe["detail"]
        self.assertEqual(detail["status"], 400)
        self.assertEqual({k: v for k, v in detail["status_per_tool"].items() if v == 400},
                         {"launch_experiments": 400, "acknowledge_finding": 400, "finish": 400})
        for tool in ("launch_experiments", "acknowledge_finding", "finish"):
            self.assertEqual(detail["keywords_per_rejected_tool"][tool]["rejected_keywords"], ["minLength"], tool)
            self.assertTrue(detail["keywords_per_rejected_tool"][tool]["accepted_without_all_constraints"])
        self.assertIn("cannot run", detail["message"])
        self.assertIn("minLength", json.dumps(detail["constraint_keywords_used"]))
        self.assertTrue(report["ok"])                                        # informational: the required probes do not use them

    def test_a_rejected_strict_schema_keyword_is_learned_and_the_message_says_the_live_path_breaks(self):
        schema = {"type": "object", "properties": {
            "code": {"type": "string", "pattern": "^[a-z]+$", "maxLength": 9},
            "items": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
            "required": ["code", "items"], "additionalProperties": False}
        self.fake.rejected_schema_keywords = {"pattern"}
        with mock.patch.object(live_smoke, "_agent_schemas", return_value={"CUSTOM": schema}):
            report = self.run_smoke()
        probe = self.by_name(report)["strict_CUSTOM"]
        self.assertEqual(probe["status"], "fail")
        self.assertEqual(probe["detail"]["status"], 400)
        self.assertEqual(probe["detail"]["rejected_keywords"], ["pattern"])
        self.assertEqual(probe["detail"]["constraint_keywords_used"], ["maxItems", "maxLength", "pattern"])
        self.assertTrue(probe["detail"]["accepted_without_all_constraints"])
        self.assertIn("pattern", probe["detail"]["api_message"])
        self.assertIn("every call that uses it will fail", probe["detail"]["message"])

    def test_a_schema_the_server_accepts_reports_its_keywords_but_no_rejection(self):
        schema = {"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}, "required": ["x"],
                  "additionalProperties": False}
        with mock.patch.object(live_smoke, "_agent_schemas", return_value={"CUSTOM": schema}):
            probe = self.by_name(self.run_smoke())["strict_CUSTOM"]
        self.assertEqual((probe["status"], probe["detail"]["constraint_keywords_used"]), ("pass", ["maxLength"]))
        self.assertNotIn("rejected_keywords", probe["detail"])


class EffortAndParameterProbeTests(SmokeCase):
    def test_max_effort_is_probed_on_spark_13_and_none_must_be_a_400(self):
        report = self.run_smoke(probe_wonder=False)
        probes = self.by_name(report)
        max_bodies = self.bodies_where(lambda b: b.get("reasoning_effort") == "max")
        self.assertEqual([b["model"] for b in max_bodies], [SPARK])
        self.assertEqual(probes["effort_max"]["detail"]["sent_effort"], "max")
        none_requests = [r for r in self.fake.requests if (r.body or {}).get("reasoning_effort") == "none"]
        self.assertEqual([r.status for r in none_requests], [400])
        self.assertEqual((probes["effort_none_rejected"]["status"], probes["effort_none_rejected"]["detail"]["status"]), ("pass", 400))

    def test_max_effort_uses_spark_13_even_when_the_default_model_is_another(self):
        asked: list = []
        inner = self.factory()

        def factory(model=None):
            asked.append(model)
            return inner("muse-spark-1.2" if model is None else model)
        report = live_smoke.run_smoke(factory, probe_wonder=False)
        self.assertEqual(asked, [None, SPARK])
        self.assertEqual(self.by_name(report)["effort_max"]["status"], "pass")

    def test_temperature_is_probed_with_reasoning_on(self):
        probe = self.by_name(self.run_smoke())["temperature_with_reasoning"]
        self.assertEqual((probe["status"], probe["detail"]["temperature_stripped"]), ("pass", False))
        bodies = self.bodies_where(lambda b: b.get("temperature") == 0.2)
        self.assertEqual([(b["reasoning_effort"], len(b["messages"])) for b in bodies], [("low", 1)])

    def test_a_host_that_refuses_temperature_is_reported_with_its_consequence(self):
        with FakeMetaServer(unsupported_params=("temperature",)) as fake:
            report = live_smoke.run_smoke(live_smoke.build_client_factory(
                base_url=fake.base_url, env={"MODEL_API_KEY": fake.api_key}), probe_wonder=False)
        probe = self.by_name(report)["temperature_with_reasoning"]
        self.assertEqual(probe["status"], "fail")
        self.assertTrue(probe["detail"]["temperature_stripped"])
        self.assertIn("provider default", probe["detail"]["message"])

    def test_the_completion_cap_ceiling_is_found_between_the_probe_values(self):
        self.fake.max_completion_tokens_ceiling = 20000
        probe = self.by_name(self.run_smoke())["completion_cap_ceiling"]
        self.assertEqual(probe["detail"]["statuses"], {"16000": 200, "32000": 400, "64000": None})
        self.assertEqual((probe["detail"]["highest_accepted"], probe["detail"]["lowest_rejected"], probe["status"]), (16000, 32000, "pass"))
        self.assertIn("20000", probe["detail"]["api_message"])
        sizes = [b["max_completion_tokens"] for b in self.fake.bodies() if b.get("max_completion_tokens", 0) >= 16000]
        self.assertEqual(sizes, [16000, 32000])                       # 64000 is not asked once 32000 was refused

    def test_everything_accepted_means_no_ceiling_below_64000(self):
        probe = self.by_name(self.run_smoke())["completion_cap_ceiling"]
        self.assertEqual((probe["detail"]["statuses"], probe["detail"]["lowest_rejected"], probe["detail"]["highest_accepted"]),
                         ({"16000": 200, "32000": 200, "64000": 200}, None, 64000))

    def test_a_ceiling_below_the_retry_cap_ceiling_is_flagged(self):
        self.fake.max_completion_tokens_ceiling = 10000
        probe = self.by_name(self.run_smoke())["completion_cap_ceiling"]
        self.assertEqual((probe["status"], probe["detail"]["highest_accepted"]), ("fail", None))
        self.assertIn("RETRY_CAP_CEILING", probe["detail"]["message"])
        self.assertIn("completion_cap_ceiling", probe["detail"]["message"])


class MessageShapeProbeTests(SmokeCase):
    def test_assistant_tool_call_content_null_and_empty_string_are_both_tried(self):
        probe = self.by_name(self.run_smoke())["assistant_tool_calls_content"]
        self.assertEqual(probe["status"], "pass")
        self.assertEqual((probe["detail"]["null"]["accepted"], probe["detail"]["empty_string"]["accepted"]), (True, True))
        replays = [b["messages"][1] for b in self.fake.bodies() if len(b["messages"]) == 3 and b["messages"][1]["role"] == "assistant"
                   and b["messages"][1].get("tool_calls") and b["messages"][1]["tool_calls"][0]["id"] == "call_smoke1"]
        self.assertEqual([m["content"] for m in replays], [None, ""])

    def test_a_server_that_refuses_null_content_breaks_the_tool_loop_and_the_probe_says_so(self):
        def handler(req):
            for m in req.body["messages"]:
                if m["role"] == "assistant" and m.get("tool_calls") and m["content"] is None and m["tool_calls"][0]["id"] == "call_smoke1":
                    return Fault.bad_request("content must be a string", "messages")
            return None
        self.fake.set_handler(handler)
        probe = self.by_name(self.run_smoke())["assistant_tool_calls_content"]
        self.assertEqual(probe["status"], "fail")
        self.assertEqual((probe["detail"]["null"]["accepted"], probe["detail"]["null"]["status"], probe["detail"]["empty_string"]["accepted"]),
                         (False, 400, True))
        self.assertIn("every tool loop turn after the first would fail", probe["detail"]["message"])

    def test_two_parallel_tool_calls_are_answered_in_one_turn(self):
        def handler(req):
            has_result = any(m["role"] == "tool" for m in req.body["messages"])
            if req.body.get("tools") and not has_result and "twice" in req.body["messages"][0]["content"]:
                return Reply.tools(("add_numbers", {"a": 2, "b": 3}, "call_p1"), ("add_numbers", {"a": 10, "b": 20}, "call_p2"))
            return None
        self.fake.set_handler(handler)
        probe = self.by_name(self.run_smoke())["parallel_tool_calls"]
        self.assertEqual((probe["status"], probe["detail"]["tool_calls"]), ("pass", 2))
        answer = [b for b in self.fake.bodies() if [m.get("tool_call_id") for m in b["messages"] if m["role"] == "tool"] == ["call_p1", "call_p2"]]
        self.assertEqual(len(answer), 1)
        self.assertEqual([m["role"] for m in answer[0]["messages"]], ["user", "assistant", "tool", "tool"])

    def test_one_tool_call_is_inconclusive_not_a_failure(self):
        probe = self.by_name(self.run_smoke())["parallel_tool_calls"]
        self.assertEqual((probe["status"], probe["detail"]["tool_calls"]), ("inconclusive", 1))

    def test_tools_and_response_format_travel_together(self):
        probe = self.by_name(self.run_smoke())["tools_with_response_format"]
        self.assertEqual(probe["status"], "pass")
        both = self.bodies_where(lambda b: b.get("tools") and b.get("response_format") and len(b.get("tools")) == 1)
        self.assertEqual(len(both), 1)
        self.assertTrue(both[0]["response_format"]["json_schema"]["strict"])

    def test_a_server_that_refuses_the_combination_is_a_failed_informational_probe(self):
        self.fake.set_handler(lambda req: Fault.bad_request("tools cannot be combined with response_format", "tools")
                              if req.body.get("tools") and req.body.get("response_format") else None)
        report = self.run_smoke()
        probe = self.by_name(report)["tools_with_response_format"]
        self.assertEqual((probe["status"], probe["error"]["status"], report["ok"]), ("fail", 400, True))

    def test_system_and_developer_roles_are_each_tried_as_a_one_turn_call(self):
        probe = self.by_name(self.run_smoke())["system_and_developer_roles"]
        self.assertEqual(probe["status"], "pass")
        roles = [b["messages"][0]["role"] for b in self.fake.bodies() if len(b["messages"]) == 2
                 and b["messages"][0]["content"] == "You answer with exactly one word."]
        self.assertEqual(roles, ["system", "developer"])

    def test_a_server_without_the_developer_role_is_reported_per_role(self):
        self.fake.set_handler(lambda req: Fault.bad_request("invalid role developer", "messages")
                              if req.body["messages"][0]["role"] == "developer" else None)
        probe = self.by_name(self.run_smoke())["system_and_developer_roles"]
        self.assertEqual(probe["status"], "fail")
        self.assertEqual((probe["detail"]["system"]["accepted"], probe["detail"]["developer"]["accepted"],
                          probe["detail"]["developer"]["status"]), (True, False, 400))
        self.assertNotIn("message", probe["detail"])                        # system works: the agents are not broken


class RateLimitAndCacheProbeTests(SmokeCase):
    def test_ratelimit_headers_are_recorded_without_content(self):
        self.fake.ratelimit_headers = True
        report = self.run_smoke()
        probe = self.by_name(report)["ratelimit_headers"]
        self.assertEqual(probe["status"], "pass")
        self.assertEqual(set(probe["detail"]["headers"]), {
            "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests"})
        self.assertEqual(set(report["rate_limit"]["headers"]), set(probe["detail"]["headers"]))
        self.assertEqual(report["rate_limit"]["rate_limited_responses"], 0)

    def test_missing_ratelimit_headers_make_the_informational_probe_fail_quietly(self):
        report = self.run_smoke()
        probe = self.by_name(report)["ratelimit_headers"]
        self.assertEqual((probe["status"], probe["detail"]["headers"], report["ok"]), ("fail", {}, True))

    def test_a_429_with_retry_after_is_recorded(self):
        sleeper = Sleeper()
        client = MetaChatClient(base_url=self.fake.base_url, api_key=self.fake.api_key, max_retries=0, sleep=sleeper)
        self.fake.enqueue(Fault.rate_limit(retry_after=7))
        report = live_smoke.run_smoke(lambda: client)
        probe = self.by_name(report)["tool_roundtrip"]
        self.assertEqual((probe["status"], probe["error"]["status"]), ("fail", 429))
        self.assertEqual((report["rate_limit"]["rate_limited_responses"], report["rate_limit"]["retry_after_seconds"]), (1, [7.0]))

    def test_the_large_prefix_probe_sends_at_least_8k_tokens_twice_with_one_key(self):
        probe = self.by_name(self.run_smoke())["prompt_cache_large"]
        self.assertTrue(probe["detail"]["prefix_at_least_8k_tokens"])
        self.assertGreaterEqual(probe["detail"]["first_prompt_tokens"], 8000)
        self.assertTrue(probe["detail"]["cache_hit"])
        large = self.bodies_where(lambda b: b.get("prompt_cache_key") == "nightshift-live-smoke-large")
        self.assertEqual(len(large), 2)
        self.assertEqual(large[0]["messages"], large[1]["messages"])
        self.assertGreater(len(large[0]["messages"][0]["content"]), 32000)

    def test_the_small_prefix_probe_is_unchanged(self):
        probe = self.by_name(self.run_smoke())["prompt_cache"]
        self.assertEqual(probe["status"], "pass")
        self.assertFalse(probe["detail"]["prefix_at_least_8k_tokens"])
        self.assertGreater(probe["detail"]["second_cached_tokens"], 0)


class ClientWithoutRawRequestTests(SmokeCase):
    def test_raw_probes_fail_cleanly_for_a_client_that_cannot_send_raw_requests(self):
        from nightshift.models.scripted import ScriptedClient, text_response
        report = live_smoke.run_smoke(lambda: ScriptedClient([text_response("only one reply")]), probe_wonder=False)
        probes = self.by_name(report)
        self.assertEqual(probes["effort_none_rejected"]["status"], "fail")
        self.assertEqual(probes["completion_cap_ceiling"]["error"]["kind"], "RuntimeError")
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()


class VerifierRegressionSmokeTests(SmokeCase):
    """Defects found by the adversarial verification of the API-conformance workstream."""

    def test_a_tool_call_cut_off_mid_arguments_is_inconclusive_not_a_failure(self):
        # finish_reason 'length' with a half-written arguments string is a token-cap symptom, not a broken tool path
        self.fake.enqueue(Reply.tools(("add_numbers", '{"a": 2, "b"'), finish_reason="length"))
        probe = self.by_name(self.run_smoke())["tool_roundtrip"]
        self.assertEqual(probe["status"], "inconclusive")
        self.assertIn("max_completion_tokens", probe["detail"]["message"])

    def test_bad_arguments_without_a_cut_off_are_still_a_failure(self):
        self.fake.enqueue(Reply.tools(("add_numbers", '{"a": 2, "b"'), finish_reason="tool_calls"))
        probe = self.by_name(self.run_smoke())["tool_roundtrip"]
        self.assertEqual(probe["status"], "fail")

    def test_a_429_absorbed_by_the_clients_own_retries_is_still_recorded(self):
        self.fake.enqueue(Fault.rate_limit(retry_after=3))
        client = MetaChatClient(base_url=self.fake.base_url, api_key=self.fake.api_key, sleep=Sleeper())
        report = live_smoke.run_smoke(lambda: client)
        self.assertTrue(self.by_name(report)["tool_roundtrip"]["ok"])
        self.assertEqual((report["rate_limit"]["rate_limited_responses"], report["rate_limit"]["retry_after_seconds"]), (1, [3.0]))
