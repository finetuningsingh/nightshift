"""Tests for nightshift.models: base helpers, MetaChatClient (over real loopback sockets against the fake server),
pricing, ScriptedClient, routing config and the live smoke probes. No external network."""
from __future__ import annotations

import copy
import hashlib
import http.client
import importlib.util
import io
import json
import os
import socket
import tempfile
import threading
import traceback
import unittest
from pathlib import Path
from unittest import mock

from nightshift import secrets
from nightshift.ledger import Ledger
from nightshift.models import base, config, live_smoke, pricing
from nightshift.models.base import ModelConfigError
from nightshift.models.meta import NEVER_SENT, MetaChatClient, check_base_url
from nightshift.models.scripted import ScriptedClient, ScriptExhausted, text_response, tool_response
from nightshift.types import (BudgetLimits, ChatMessage, ChatResponse, ModelClient, ModelError, ToolCall, ToolSpec, Usage)
from tests.support.fake_meta_server import (FakeMetaServer, Fault, Reply, sample_from_schema, strict_schema_error)
from tests.support.synthetic import FakeClock, RunBuilder

try:
    from nightshift.budget import BudgetGate, MeteredClient
    BUDGET_IMPORT_ERROR = ""
except Exception as _exc:  # noqa: BLE001 - budget.py is built by another engineer
    BUDGET_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

ROOT = Path(__file__).resolve().parent.parent
SPARK, WONDER = "muse-spark-1.3", "muse-wonder-1.0"
ADD = ToolSpec("add", "Add two integers.", {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}})


def user(text: str = "hello") -> list[ChatMessage]:
    return [ChatMessage("user", text)]


class Sleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_client(fake: FakeMetaServer, **kw) -> MetaChatClient:
    """Client for the fake server with a recording sleep and a constant 0.25 s jitter."""
    sleeper = Sleeper()
    kw.setdefault("sleep", sleeper)
    kw.setdefault("jitter", lambda: 0.25)
    client = MetaChatClient(base_url=fake.base_url, api_key=fake.api_key, **kw)
    client.sleeps = sleeper  # type: ignore[attr-defined]
    return client


def raw_post(fake: FakeMetaServer, body: dict, key: str | None | bool = None, path: str = "/v1/chat/completions"):
    """POST straight to the fake (no client) and return (status, headers, body bytes)."""
    conn = http.client.HTTPConnection("127.0.0.1", fake.port, timeout=5)
    headers = {"Content-Type": "application/json"}
    if key is not False:
        headers["Authorization"] = "Bearer " + (key or fake.api_key)
    conn.request("POST", path, json.dumps(body), headers)
    resp = conn.getresponse()
    data = resp.read()
    out = (resp.status, dict(resp.getheaders()), data)
    conn.close()
    return out


class ServerCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeMetaServer().start()
        self.addCleanup(self.fake.stop)


# ==================================================================================================
# strict schema
# ==================================================================================================
class StrictifySchemaTests(unittest.TestCase):
    def test_every_object_closed_and_fully_required(self):
        schema = {"type": "object", "required": ["a"], "properties": {
            "a": {"type": "string"},
            "b": {"type": "object", "properties": {"c": {"type": "integer"}}},
            "d": {"type": "array", "items": {"type": "object", "properties": {"e": {"type": "boolean"}}, "required": []}}}}
        out = base.strictify_schema(schema)
        self.assertEqual(out["required"], ["a", "b", "d"])
        self.assertIs(out["additionalProperties"], False)
        self.assertEqual(out["properties"]["b"]["required"], ["c"])
        self.assertIs(out["properties"]["b"]["additionalProperties"], False)
        self.assertEqual(out["properties"]["d"]["items"]["required"], ["e"])
        self.assertIs(out["properties"]["d"]["items"]["additionalProperties"], False)
        self.assertIsNone(strict_schema_error(out))   # the fake's independent checker agrees

    def test_input_is_not_mutated_and_result_is_idempotent(self):
        schema = {"type": "object", "properties": {"x": {"type": "object", "properties": {"y": {"type": "string"}}}}}
        before = copy.deepcopy(schema)
        once = base.strictify_schema(schema)
        self.assertEqual(schema, before)
        self.assertEqual(base.strictify_schema(once), once)

    def test_local_refs_are_inlined(self):
        schema = {"type": "object", "properties": {"p": {"$ref": "#/$defs/Point", "description": "where"}},
                  "$defs": {"Point": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}}}
        out = base.strictify_schema(schema)
        self.assertNotIn("$defs", out)
        self.assertNotIn("$ref", json.dumps(out))
        self.assertEqual(out["properties"]["p"]["required"], ["x", "y"])
        self.assertEqual(out["properties"]["p"]["description"], "where")

    def test_recursion_is_rejected(self):
        schema = {"type": "object", "properties": {"node": {"$ref": "#/$defs/Node"}},
                  "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}}}
        with self.assertRaisesRegex(ValueError, "recursive"):
            base.strictify_schema(schema)

    def test_unresolved_and_foreign_refs_are_rejected(self):
        for ref in ("#/$defs/Missing", "http://example.com/schema.json", "#/properties/x"):
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, "reference"):
                base.strictify_schema({"type": "object", "properties": {"x": {"$ref": ref}}})

    def test_all_of_of_objects_is_flattened(self):
        out = base.strictify_schema({"type": "object", "properties": {"c": {"type": "boolean"}}, "allOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"properties": {"b": {"type": "integer"}}}]})
        self.assertNotIn("allOf", out)
        self.assertEqual(sorted(out["properties"]), ["a", "b", "c"])
        self.assertEqual(sorted(out["required"]), ["a", "b", "c"])

    def test_all_of_of_non_objects_and_conflicts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "allOf"):
            base.strictify_schema({"type": "object", "properties": {"x": {"allOf": [{"type": "string"}]}}})
        with self.assertRaisesRegex(ValueError, "twice"):
            base.strictify_schema({"type": "object", "allOf": [
                {"properties": {"a": {"type": "string"}}}, {"properties": {"a": {"type": "integer"}}}]})

    def test_unsupported_constructs_name_the_path(self):
        for keyword in ("oneOf", "not", "patternProperties", "if"):
            schema = {"type": "object", "properties": {"deep": {"type": "object", "properties": {
                "x": {"type": "string"}}, keyword: [{"type": "string"}]}}}
            with self.subTest(keyword=keyword), self.assertRaisesRegex(ValueError, rf"\$\.deep: '{keyword}'"):
                base.strictify_schema(schema)

    def test_root_must_be_a_plain_object(self):
        for bad in ({"type": "array", "items": {"type": "string"}}, {"type": "string"}, {"anyOf": [{"type": "object", "properties": {}}]}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "root"):
                base.strictify_schema(bad)
        with self.assertRaises(ValueError):
            base.strictify_schema("not a schema")  # type: ignore[arg-type]

    def test_free_form_objects_arrays_without_items_and_untyped_nodes_are_rejected(self):
        cases = {
            "free-form true": {"type": "object", "properties": {}, "additionalProperties": True},
            "free-form map": {"type": "object", "properties": {"m": {"type": "object", "additionalProperties": {"type": "string"}}}},
            "array without items": {"type": "object", "properties": {"a": {"type": "array"}}},
            "untyped": {"type": "object", "properties": {"x": {}}},
            "unknown type": {"type": "object", "properties": {"x": {"type": "date"}}},
            "required undeclared": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["y"]},
        }
        for name, schema in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                base.strictify_schema(schema)

    def test_explicit_additional_properties_false_and_nullable_forms_are_accepted(self):
        out = base.strictify_schema({"type": "object", "additionalProperties": False, "properties": {
            "a": {"type": ["string", "null"]}, "b": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "c": {"enum": ["x", "y"]}}})
        self.assertEqual(out["required"], ["a", "b", "c"])

    def test_nesting_limit_is_ten_levels(self):
        def nest(wraps: int) -> dict:
            node: dict = {"type": "string"}
            for _ in range(wraps):
                node = {"type": "object", "properties": {"k": node}}
            return node
        base.strictify_schema(nest(9))                    # the leaf sits at depth 10
        with self.assertRaisesRegex(ValueError, "deeper than 10"):
            base.strictify_schema(nest(10))

    def test_schema_name_is_cleaned(self):
        self.assertEqual(base.clean_schema_name("my schema!"), "my_schema_")
        self.assertEqual(base.clean_schema_name(""), "response")
        self.assertEqual(len(base.clean_schema_name("x" * 100)), 64)


# ==================================================================================================
# wire helpers
# ==================================================================================================
class WireHelperTests(unittest.TestCase):
    def test_assistant_tool_calls_and_tool_results_serialise_as_documented(self):
        call = ToolCall("call_1", "add", {"a": 1}, raw_arguments='{"a": 1}')
        wire = base.messages_to_wire([ChatMessage("user", "q"), ChatMessage("assistant", None, [call]),
                                      ChatMessage("tool", "2", tool_call_id="call_1")])
        self.assertEqual(wire, [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "2"}])

    def test_replayed_arguments_are_always_valid_json(self):
        cases = [(ToolCall("c", "f", {"x": 1}), '{"x": 1}'),
                 (ToolCall("c", "f", {}, raw_arguments="{broken", parse_error="invalid JSON"), "{}"),
                 (ToolCall("c", "f", {}), "{}")]
        for call, expected in cases:
            with self.subTest(expected=expected):
                wire = base.message_to_wire(ChatMessage("assistant", None, [call]))
                self.assertEqual(wire["tool_calls"][0]["function"]["arguments"], expected)

    def test_tool_result_sequence_is_enforced(self):
        two = ChatMessage("assistant", None, [ToolCall("a", "f", {}), ToolCall("b", "f", {})])
        bad = {
            "unanswered": [ChatMessage("user", "q"), ChatMessage("assistant", None, [ToolCall("a", "f", {})])],
            "wrong order": [two, ChatMessage("tool", "x", tool_call_id="b"), ChatMessage("tool", "y", tool_call_id="a")],
            "one missing": [two, ChatMessage("tool", "x", tool_call_id="a"), ChatMessage("user", "next")],
            "stray tool": [ChatMessage("user", "q"), ChatMessage("tool", "x", tool_call_id="a")],
            "duplicate ids": [ChatMessage("assistant", None, [ToolCall("a", "f", {}), ToolCall("a", "f", {})])],
            "empty": [],
        }
        for name, msgs in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                base.messages_to_wire(msgs)
        ok = base.messages_to_wire([two, ChatMessage("tool", "x", tool_call_id="a"), ChatMessage("tool", "y", tool_call_id="b")])
        self.assertEqual([m["role"] for m in ok], ["assistant", "tool", "tool"])

    def test_tool_message_needs_an_id_and_roles_are_checked(self):
        with self.assertRaises(ValueError):
            base.message_to_wire(ChatMessage("tool", "x"))
        with self.assertRaises(ValueError):
            base.message_to_wire(ChatMessage("wizard", "x"))
        self.assertEqual(base.message_to_wire(ChatMessage("developer", "rules"))["role"], "developer")

    def test_tools_use_the_nested_form_and_names_are_validated(self):
        wire = base.tools_to_wire([ADD])
        self.assertEqual(wire, [{"type": "function", "function": {"name": "add", "description": "Add two integers.",
                                                                  "parameters": ADD.parameters}}])
        base.tools_to_wire([ToolSpec("ns.tool-name_1", "d", {})])   # one dot is fine
        for name in ("two.dots.here", "has space", "", "slash/name"):
            with self.subTest(name), self.assertRaises(ValueError):
                base.tools_to_wire([ToolSpec(name, "d", {})])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            base.tools_to_wire([ADD, ADD])

    def test_parse_arguments(self):
        self.assertEqual(base.parse_arguments('{"a": 1}'), ({"a": 1}, None))
        self.assertEqual(base.parse_arguments(""), ({}, None))
        self.assertEqual(base.parse_arguments({"a": 2}), ({"a": 2}, None))
        for raw in ("{bad", "[1, 2]", "42", 5):
            with self.subTest(raw=raw):
                value, error = base.parse_arguments(raw)
                self.assertEqual(value, {})
                self.assertTrue(error)

    def test_tool_calls_from_wire_keep_raw_text_and_report_parse_errors(self):
        calls = base.tool_calls_from_wire([
            {"id": "x1", "type": "function", "function": {"name": "f", "arguments": '{"k": [1, 2]}'}},
            {"id": "x2", "type": "function", "function": {"name": "g", "arguments": "{oops"}},
            {"type": "function", "function": {"name": "h", "arguments": {"already": "dict"}}}, "junk"])
        self.assertEqual([c.name for c in calls], ["f", "g", "h"])
        self.assertEqual(calls[0].arguments, {"k": [1, 2]})
        self.assertEqual((calls[1].arguments, calls[1].raw_arguments), ({}, "{oops"))
        self.assertIn("invalid JSON", calls[1].parse_error)
        self.assertEqual((calls[2].id, calls[2].arguments), ("call_2", {"already": "dict"}))

    def test_usage_cached_tokens_are_read_defensively(self):
        nested = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "prompt_tokens_details": {"cached_tokens": 4}}
        flat = {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 6}
        for wire, usage, known in ((nested, Usage(10, 5, 4, 15), True), (flat, Usage(10, 5, 6, 15), True),
                                   ({"prompt_tokens": 10, "completion_tokens": 5}, Usage(10, 5, 0, 15), False),
                                   ({"prompt_tokens": 10, "cached_tokens": 0}, Usage(10, 0, 0, 10), True),
                                   ({"prompt_tokens": 10, "cached_tokens": True}, Usage(10, 0, 0, 10), False),
                                   ({"prompt_tokens": -3, "completion_tokens": "x"}, Usage(0, 0, 0, 0), False),
                                   (None, Usage(0, 0, 0, 0), False)):
            with self.subTest(wire=wire):
                self.assertEqual(base.usage_from_wire(wire), (usage, known))

    def test_error_envelope(self):
        info = base.parse_error_envelope(json.dumps({"error": {"message": "m", "type": "t", "param": None, "code": 429}}))
        self.assertEqual(info, {"message": "m", "type": "t", "param": None, "code": "429"})
        html = base.parse_error_envelope("<html>  oops \n</html>")
        self.assertEqual((html["message"], html["type"]), ("<html> oops </html>", None))
        self.assertEqual(base.parse_error_envelope(b"")["message"], None)
        self.assertEqual(base.parse_error_envelope('{"error": "plain"}')["message"], "plain")
        self.assertLessEqual(len(base.parse_error_envelope("x" * 5000)["message"]), 600)

    def test_reasoning_effort_normalisation(self):
        for given, expected in (("none", "minimal"), ("minimal", "minimal"), ("LOW", "low"), (" high ", "high"), ("xhigh", "xhigh"), (None, None)):
            self.assertEqual(base.normalize_reasoning_effort(given, SPARK), expected)
        self.assertEqual(base.normalize_reasoning_effort("max", WONDER), "max")
        for given in ("turbo", "", 3):
            with self.subTest(given=given), self.assertRaises(ValueError):
                base.normalize_reasoning_effort(given, SPARK)

    def test_max_effort_follows_the_docs_per_model(self):
        # /docs/reasoning and /docs/models: max is supported by muse-wonder-1.0 and muse-spark-1.3, not by -contributor
        for model in (WONDER, SPARK, "muse-spark-1.3-2026-10-20"):
            with self.subTest(model=model):
                self.assertTrue(base.supports_max_effort(model))
                self.assertEqual(base.normalize_reasoning_effort("max", model), "max")
        for model in ("muse-spark-1.3-contributor", "muse-spark-1.2", "muse-spark-1.2-contributor", "some-partner-model"):
            with self.subTest(model=model):
                self.assertFalse(base.supports_max_effort(model))
                with self.assertRaises(ValueError):
                    base.normalize_reasoning_effort("max", model)
        self.assertEqual(base.normalize_reasoning_effort("none", SPARK), "minimal")   # none is mapped, never sent


class SseTests(unittest.TestCase):
    STREAM = (b': comment\n\ndata: {"a": 1}\n\nevent: x\ndata: {"b":\ndata:  2}\n\n'
              b'data: {"c": 3}\r\n\r\ndata: [DONE]\n\ndata: {"after": "done"}\n\n')
    EVENTS = [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_events_are_parsed_and_done_stops_the_stream(self):
        self.assertEqual(list(base.parse_sse([self.STREAM])), self.EVENTS)

    def test_result_is_independent_of_chunk_boundaries(self):
        whole = list(base.parse_sse([self.STREAM]))
        for size in (1, 2, 3, 7, 50):
            chunks = [self.STREAM[i:i + size] for i in range(0, len(self.STREAM), size)]
            self.assertEqual(list(base.parse_sse(chunks)), whole, f"chunk size {size}")

    def test_crlf_split_across_chunks_is_one_line_break(self):
        self.assertEqual(list(base.parse_sse([b'data: {"a": 1}\r', b"\n\r", b"\n"])), [{"a": 1}])

    def test_last_event_without_blank_line_and_string_chunks(self):
        self.assertEqual(list(base.parse_sse(['data: {"a": 1}'])), [{"a": 1}])

    def test_malformed_payloads_and_oversize_streams_raise(self):
        with self.assertRaisesRegex(ValueError, "malformed"):
            list(base.parse_sse([b"data: {not json}\n\n"]))
        with self.assertRaisesRegex(ValueError, "not a JSON object"):
            list(base.parse_sse([b"data: [1, 2]\n\n"]))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            list(base.parse_sse([b"data: {}\n\n" * 10], max_bytes=20))


class StreamAssemblerTests(unittest.TestCase):
    @staticmethod
    def chunk(delta, finish=None, **extra):
        return {"model": "m1", "id": "s1", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

    def test_content_and_interleaved_tool_call_fragments_are_grouped_by_index(self):
        asm = base.StreamAssembler()
        for c in (
            self.chunk({"role": "assistant", "content": "Hel"}),
            self.chunk({"content": "lo"}),
            self.chunk({"tool_calls": [{"index": 0, "id": "ca", "type": "function", "function": {"name": "f", "arguments": ""}}]}),
            self.chunk({"tool_calls": [{"index": 1, "id": "cb", "type": "function", "function": {"name": "g", "arguments": ""}}]}),
            self.chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"y"'}}]}),
            self.chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"x":'}}]}),
            self.chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' 1}'}}]}),
            self.chunk({"tool_calls": [{"index": 1, "function": {"arguments": ': 2}'}}]}),
            self.chunk({}, "tool_calls"),
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        ):
            asm.feed(c)
        self.assertTrue(asm.finished)
        out = asm.to_response()
        msg = out["choices"][0]["message"]
        self.assertEqual(msg["content"], "Hello")
        self.assertEqual([(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in msg["tool_calls"]],
                         [("ca", "f", '{"x": 1}'), ("cb", "g", '{"y": 2}')])
        self.assertEqual((out["model"], out["usage"]["prompt_tokens"]), ("m1", 7))

    def test_unfinished_stream_and_error_chunks(self):
        asm = base.StreamAssembler()
        asm.feed(self.chunk({"content": "partial"}))
        self.assertFalse(asm.finished)
        with self.assertRaises(ModelError) as ctx:
            asm.feed({"error": {"message": "boom", "type": "server_error", "code": "x"}})
        self.assertEqual((ctx.exception.error_type, ctx.exception.code, ctx.exception.retryable), ("server_error", "x", False))

    def test_fragments_without_index_continue_the_previous_call(self):
        asm = base.StreamAssembler()
        asm.feed(self.chunk({"tool_calls": [{"id": "c0", "function": {"name": "f", "arguments": '{"a":'}}]}))
        asm.feed(self.chunk({"tool_calls": [{"function": {"arguments": ' 1}'}}]}, "tool_calls"))
        call = asm.to_response()["choices"][0]["message"]["tool_calls"]
        self.assertEqual([(c["id"], c["function"]["arguments"]) for c in call], [("c0", '{"a": 1}')])


# ==================================================================================================
# pricing and scripted client
# ==================================================================================================
class PricingTests(unittest.TestCase):
    def test_price_table_from_the_docs(self):
        expected = {
            "muse-spark-1.3": (0.15, 1.25, 4.25), "muse-spark-1.2": (0.15, 1.25, 4.25), "muse-spark-1.1": (0.15, 1.25, 4.25),
            "muse-wonder-1.0": (0.20, 4.00, 20.00), "muse-spark-1.3-contributor": (0.002, 0.10, 0.20)}
        for model, (cached, inp, out) in expected.items():
            with self.subTest(model):
                self.assertEqual(pricing.PRICES[model], {"cached_input": cached, "input": inp, "output": out})
        self.assertIn("muse-spark-1.2-contributor", pricing.PRICES)
        for row in pricing.PRICES.values():
            self.assertEqual(set(row), {"input", "cached_input", "output"})

    def test_cached_tokens_use_the_cached_rate_and_the_rest_the_input_rate(self):
        usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, cached_tokens=500_000)
        self.assertAlmostEqual(pricing.usd_for(SPARK, usage), 0.5 * 1.25 + 0.5 * 0.15 + 4.25)
        self.assertAlmostEqual(pricing.usd_for(WONDER, Usage(1000, 1000, 0)), (1000 * 4.0 + 1000 * 20.0) / 1e6)
        self.assertAlmostEqual(pricing.usd_for("muse-spark-1.3-contributor", Usage(1_000_000, 0, 1_000_000)), 0.002)

    def test_unknown_model_is_charged_at_the_spark_rate(self):
        usage = Usage(1234, 567, 100)
        self.assertEqual(pricing.usd_for("some-partner-model", usage), pricing.usd_for(SPARK, usage))
        self.assertEqual(pricing.usd_for("", usage), pricing.usd_for(SPARK, usage))
        self.assertNotEqual(pricing.usd_for(WONDER, usage), pricing.usd_for(SPARK, usage))

    def test_odd_counts_never_make_a_negative_or_free_cost(self):
        self.assertAlmostEqual(pricing.usd_for(SPARK, Usage(100, 0, 500)), 100 * 0.15 / 1e6)   # cached clamped to prompt
        self.assertEqual(pricing.usd_for(SPARK, Usage(-5, -5, -5)), 0.0)
        self.assertEqual(pricing.usd_for(SPARK, Usage(0, 0, 0)), 0.0)

    @unittest.skipIf(BUDGET_IMPORT_ERROR, f"nightshift.budget cannot be imported: {BUDGET_IMPORT_ERROR}")
    def test_budget_module_charges_the_same_amount(self):
        from nightshift import budget
        usage = Usage(5000, 700, 1200)
        for model in (SPARK, WONDER, "unknown-model"):
            self.assertAlmostEqual(budget.usd_for(model, usage), pricing.usd_for(model, usage))


class ScriptedClientTests(unittest.TestCase):
    def test_replies_are_served_in_order_and_calls_are_recorded(self):
        client = ScriptedClient([text_response("one", prompt_tokens=3, completion_tokens=2), "two"])
        self.assertIsInstance(client, ModelClient)
        first = client.chat(user("a"), reasoning_effort="high", cache_key="k", max_completion_tokens=9)
        second = client.chat(user("b"), tools=[ADD], response_schema={"type": "object"}, schema_name="s")
        self.assertEqual((first.text, first.usage), ("one", Usage(3, 2, 0, 5)))
        self.assertEqual(second.text, "two")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["messages"][0].content, "a")
        self.assertEqual((client.calls[0]["reasoning_effort"], client.calls[0]["cache_key"], client.calls[0]["max_completion_tokens"]),
                         ("high", "k", 9))
        self.assertEqual((client.calls[1]["tools"], client.calls[1]["schema_name"]), ([ADD], "s"))

    def test_exhausted_script_raises_a_clear_non_model_error(self):
        client = ScriptedClient([text_response("only")])
        client.chat(user())
        with self.assertRaises(ScriptExhausted) as ctx:
            client.chat(user())
        self.assertNotIsInstance(ctx.exception, ModelError)
        self.assertIn("exhausted", str(ctx.exception))
        self.assertEqual(len(client.calls), 2)     # the unanswered call is still recorded

    def test_callable_items_and_a_single_callable_script(self):
        client = ScriptedClient([lambda messages, **kw: text_response(f"echo:{messages[-1].content}:{kw['reasoning_effort']}")])
        self.assertEqual(client.chat(user("hi"), reasoning_effort="low").text, "echo:hi:low")
        always = ScriptedClient(lambda messages, **kw: text_response(str(len(messages))))
        self.assertEqual([always.chat(user()).text, always.chat(user() * 3).text], ["1", "3"])

    def test_exception_items_are_raised_for_failure_path_tests(self):
        client = ScriptedClient([ModelError("down", status=503, retryable=True), text_response("back")])
        with self.assertRaises(ModelError):
            client.chat(user())
        self.assertEqual(client.chat(user()).text, "back")

    def test_wrong_item_type_is_reported(self):
        with self.assertRaisesRegex(TypeError, "must be a ChatResponse"):
            ScriptedClient([42]).chat(user())

    def test_tool_response_builds_unique_ids_and_parsed_arguments(self):
        resp = tool_response(("add", {"a": 1, "b": 2}), ("add", {"a": 3}), text="thinking")
        calls = resp.message.tool_calls
        self.assertEqual(resp.finish_reason, "tool_calls")
        self.assertEqual(resp.message.content, "thinking")
        self.assertEqual([c.arguments for c in calls], [{"a": 1, "b": 2}, {"a": 3}])
        self.assertEqual(len({c.id for c in calls}), 2)
        self.assertEqual(json.loads(calls[0].raw_arguments), {"a": 1, "b": 2})
        again = tool_response(("add", {}))
        self.assertNotIn(again.message.tool_calls[0].id, {c.id for c in calls})
        self.assertEqual(tool_response(("f", {}, "fixed-id")).message.tool_calls[0].id, "fixed-id")

    def test_tool_response_with_invalid_raw_arguments_reports_a_parse_error(self):
        call = tool_response(("f", "{not json")).message.tool_calls[0]
        self.assertEqual((call.arguments, call.raw_arguments), ({}, "{not json"))
        self.assertTrue(call.parse_error)

    def test_text_response_rejects_unknown_usage_keys(self):
        with self.assertRaises(TypeError):
            text_response("x", prompt_token=3)

    def test_scripted_responses_survive_the_wire_round_trip(self):
        resp = tool_response(("add", {"a": 1}))
        wire = base.messages_to_wire([resp.message, ChatMessage("tool", "1", tool_call_id=resp.message.tool_calls[0].id)])
        self.assertEqual(wire[0]["tool_calls"][0]["function"]["name"], "add")

    def test_texts_returns_every_string_sent(self):
        client = ScriptedClient([text_response("a")])
        client.chat([ChatMessage("system", "SYS-CANARY"), ChatMessage("user", "USER-CANARY")])
        self.assertEqual(client.texts(), ["SYS-CANARY", "USER-CANARY"])

    def test_concurrent_calls_get_distinct_replies(self):
        client = ScriptedClient([text_response(str(i)) for i in range(20)])
        seen: list[str] = []
        threads = [threading.Thread(target=lambda: seen.append(client.chat(user()).text)) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(seen, key=int), [str(i) for i in range(20)])


# ==================================================================================================
# the fake server itself (other engineers' tests rely on it)
# ==================================================================================================
class FakeServerTests(ServerCase):
    BODY = {"model": SPARK, "messages": [{"role": "user", "content": "hi"}]}

    def post(self, **extra):
        status, headers, data = raw_post(self.fake, {**self.BODY, **extra})
        return status, json.loads(data), headers

    def test_valid_request_is_answered_and_recorded(self):
        status, body, headers = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        self.assertEqual(headers["x-request-id"], "req_fake1")
        self.assertEqual([(r.path, r.status) for r in self.fake.requests], [("/v1/chat/completions", 200)])

    def test_forbidden_parameters_get_400_naming_the_parameter(self):
        for name, value in (("stop", ["x"]), ("logit_bias", {"1": 1}), ("prediction", {}), ("verbosity", "low"), ("n", 2),
                            ("logprobs", True), ("web_search_options", {}), ("modalities", ["text"]), ("audio", {}),
                            ("text", {"format": {}}), ("max_output_tokens", 5), ("tool_choice", "required"),
                            ("tool_choice", "none"), ("tool_choice", {"type": "function", "function": {"name": "f"}}),
                            ("reasoning_effort", "none"), ("surprise", 1)):
            with self.subTest(name=name, value=value):
                status, body, _ = self.post(**{name: value})
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["param"], name)
                self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_allowed_variants(self):
        self.assertEqual(self.post(tool_choice="auto")[0], 200)
        self.assertEqual(self.post(n=1)[0], 200)
        self.assertEqual(self.post(logprobs=False)[0], 200)
        self.assertEqual(self.post(max_tokens=5)[0], 200)             # documented deprecated alias
        self.assertEqual(self.post(model=WONDER, reasoning_effort="max")[0], 200)

    def test_message_and_tool_rules(self):
        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        asst = {"role": "assistant", "content": None, "tool_calls": [call]}
        bad_histories = [[{"role": "user", "content": "q"}, asst],
                         [asst, {"role": "user", "content": "x"}],
                         [{"role": "tool", "tool_call_id": "c1", "content": "x"}],
                         [asst, {"role": "tool", "tool_call_id": "other", "content": "x"}]]
        for messages in bad_histories:
            with self.subTest(messages=messages):
                self.assertEqual(self.post(messages=messages)[0], 400)
        good = [{"role": "user", "content": "q"}, asst, {"role": "tool", "tool_call_id": "c1", "content": "x"}]
        self.assertEqual(self.post(messages=good)[0], 200)
        for tools in ([{"type": "function", "name": "f"}], [{"type": "function", "function": {"name": "a.b.c"}}], []):
            with self.subTest(tools=tools):
                self.assertEqual(self.post(tools=tools)[0], 400)

    def test_strict_schema_is_enforced(self):
        def rf(schema):
            return {"type": "json_schema", "json_schema": {"name": "s", "schema": schema, "strict": True}}
        loose = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.assertEqual(self.post(response_format=rf(loose))[0], 400)
        self.assertEqual(self.post(response_format=rf({"type": "object", "properties": {"a": {"type": "string"}},
                                                       "required": ["a"], "additionalProperties": False}))[0], 200)
        self.assertEqual(self.post(response_format=rf({"oneOf": []}))[0], 400)

    def test_unknown_model_is_404_and_unsupported_params_are_400(self):
        self.assertEqual(self.post(model="nope")[0], 404)
        self.fake.unsupported_params.add("prompt_cache_key")
        status, body, _ = self.post(prompt_cache_key="k")
        self.assertEqual((status, body["error"]["param"]), (400, "prompt_cache_key"))

    def test_wrong_key_gets_401_that_echoes_the_authorization_header(self):
        status, _, data = raw_post(self.fake, self.BODY, key="wrong-key-0123456789")
        self.assertEqual(status, 401)
        self.assertIn(b"Bearer wrong-key-0123456789", data)
        status, _, data = raw_post(self.fake, self.BODY, key=False)
        self.assertEqual(status, 401)

    def test_faults_come_before_validation_and_replies_wait_for_a_valid_request(self):
        self.fake.enqueue(Fault.rate_limit(retry_after=3), Reply.text("scripted"))
        status, headers, _ = raw_post(self.fake, {**self.BODY, "stop": ["x"]})
        self.assertEqual((status, headers["Retry-After"]), (429, "3"))      # the invalid request got the fault first
        self.assertEqual(self.post(stop=["x"])[0], 400)                       # validation failure leaves the reply queued
        self.assertEqual(self.fake.queued, 1)
        self.assertEqual(self.post()[1]["choices"][0]["message"]["content"], "scripted")
        self.assertEqual(self.fake.queued, 0)

    def test_fault_constructors(self):
        for fault, status in ((Fault.server_error(503), 503), (Fault.server_error(500), 500), (Fault.gateway_timeout(), 504),
                              (Fault.bad_request("bad", "temperature"), 400), (Fault.redirect("/elsewhere"), 302),
                              (Fault.garbage("not json", 502), 502)):
            with self.subTest(status=status):
                self.fake.enqueue(fault)
                got, headers, data = raw_post(self.fake, self.BODY)
                self.assertEqual(got, status)
        self.fake.enqueue(Fault.redirect("/elsewhere"))
        self.assertEqual(raw_post(self.fake, self.BODY)[1]["Location"], "/elsewhere")

    def test_streaming_replies_are_server_sent_events(self):
        self.fake.enqueue(Reply.tools(("f", {"x": "y" * 12}), ("g", {"k": 1})))
        status, headers, data = raw_post(self.fake, {**self.BODY, "stream": True})
        self.assertEqual((status, headers["Content-Type"]), (200, "text/event-stream"))
        self.assertTrue(data.rstrip().endswith(b"data: [DONE]"))
        self.assertGreater(data.count(b"data: "), 6)

    def test_callable_and_dict_items_and_default_handler(self):
        seen = []
        self.fake.enqueue(lambda req: seen.append(req.body["model"]) or Reply.text("from callable"),
                          {"choices": [{"message": {"role": "assistant", "content": "from dict"}, "finish_reason": "stop"}]})
        self.assertEqual(self.post()[1]["choices"][0]["message"]["content"], "from callable")
        self.assertEqual(self.post()[1]["choices"][0]["message"]["content"], "from dict")
        self.fake.set_handler(lambda req: Reply.text("handler"))
        self.assertEqual(self.post()[1]["choices"][0]["message"]["content"], "handler")
        self.assertEqual(seen, [SPARK])

    def test_default_policy_calls_a_tool_then_answers_in_text(self):
        status, body, _ = self.post(tools=[{"type": "function", "function": {"name": "add", "parameters": ADD.parameters}}])
        call = body["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual((call["function"]["name"], json.loads(call["function"]["arguments"])), ("add", {"a": 1, "b": 1}))
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")

    def test_cached_tokens_reported_on_a_repeat_with_the_same_cache_key(self):
        _, first, _ = self.post(prompt_cache_key="k")
        _, second, _ = self.post(prompt_cache_key="k")
        _, other, _ = self.post(prompt_cache_key="different")
        self.assertEqual(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(second["usage"]["prompt_tokens_details"]["cached_tokens"], second["usage"]["prompt_tokens"] // 2)
        self.assertEqual(other["usage"]["prompt_tokens_details"]["cached_tokens"], 0)

    def test_sample_from_schema_satisfies_the_schema_shape(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "array", "items": {"type": "string"}},
                                                   "c": {"enum": ["x"]}, "d": {"type": ["null", "boolean"]}}}
        self.assertEqual(sample_from_schema(schema), {"a": 1, "b": ["sample"], "c": "x", "d": True})

    def test_reset_clears_queue_and_history(self):
        self.fake.enqueue(Reply.text("x"))
        self.post()
        self.fake.reset()
        self.assertEqual((self.fake.requests, self.fake.queued), ([], 0))


# ==================================================================================================
# MetaChatClient: request shape
# ==================================================================================================
class ClientRequestTests(ServerCase):
    def test_plain_request_shape_and_headers(self):
        client = make_client(self.fake)
        resp = client.chat([ChatMessage("system", "be brief"), ChatMessage("user", "héllo ✓")])
        req = self.fake.last_request
        self.assertEqual((req.method, req.path), ("POST", "/v1/chat/completions"))
        self.assertEqual(req.authorization, f"Bearer {self.fake.api_key}")
        self.assertEqual(req.headers["content-type"], "application/json")
        self.assertEqual(req.body, {"model": SPARK, "messages": [{"role": "system", "content": "be brief"},
                                                                  {"role": "user", "content": "héllo ✓"}]})
        self.assertEqual((resp.text, resp.finish_reason, resp.model, resp.request_id), ("ok", "stop", SPARK, "req_fake1"))
        self.assertIsInstance(client, ModelClient)

    def test_forbidden_parameters_are_never_sent(self):
        client = make_client(self.fake, safety_identifier="u", default_reasoning="none", cache_retention="24h")
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        client.chat(user(), tools=[ADD], reasoning_effort="high", max_completion_tokens=77, temperature=0.3, cache_key="k")
        client.chat(user(), response_schema=schema, stream=True)
        client.chat(user())
        self.assertEqual(len(self.fake.requests), 3)
        for req in self.fake.requests:
            self.assertEqual(req.status, 200, req.note)
            self.assertEqual(set(req.body) & set(NEVER_SENT), set(), req.body.keys())

    def test_reasoning_effort_mapping(self):
        client = make_client(self.fake)
        for given, sent in (("none", "minimal"), ("minimal", "minimal"), ("low", "low"), ("medium", "medium"),
                            ("high", "high"), ("xhigh", "xhigh"), ("XHigh", "xhigh")):
            with self.subTest(given=given):
                client.chat(user(), reasoning_effort=given)
                self.assertEqual(self.fake.last_request.body["reasoning_effort"], sent)
        client.chat(user())
        self.assertNotIn("reasoning_effort", self.fake.last_request.body)

    def test_default_reasoning_applies_unless_overridden(self):
        client = make_client(self.fake, default_reasoning="medium")
        client.chat(user())
        self.assertEqual(self.fake.last_request.body["reasoning_effort"], "medium")
        client.chat(user(), reasoning_effort="low")
        self.assertEqual(self.fake.last_request.body["reasoning_effort"], "low")

    def test_max_effort_for_wonder_and_spark_13_but_not_contributor_and_bad_values_send_nothing(self):
        client = make_client(self.fake)
        for bad in ("turbo", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                client.chat(user(), reasoning_effort=bad)
        contributor = make_client(self.fake, model="muse-spark-1.3-contributor")
        with self.assertRaises(ValueError):
            contributor.chat(user(), reasoning_effort="max")
        self.assertEqual(self.fake.requests, [])
        for model in (WONDER, SPARK):
            make_client(self.fake, model=model).chat(user(), reasoning_effort="max")
            self.assertEqual(self.fake.last_request.body["reasoning_effort"], "max")
        make_client(self.fake, default_reasoning="max").chat(user())      # muse-spark-1.3 is the default model
        self.assertEqual(self.fake.last_request.body["reasoning_effort"], "max")
        with self.assertRaises(ValueError):
            make_client(self.fake, model="muse-spark-1.2", default_reasoning="max")

    def test_strict_json_schema_response_format(self):
        client = make_client(self.fake)
        schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}, "y": {"type": "string"}}}
        resp = client.chat(user(), response_schema=schema, schema_name="my schema!")
        self.assertEqual(self.fake.last_request.body["response_format"], {"type": "json_schema", "json_schema": {
            "name": "my_schema_", "strict": True, "schema": {
                "type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "string"}},
                "required": ["x", "y"], "additionalProperties": False}}})
        self.assertEqual(json.loads(resp.text), {"x": 1, "y": "sample"})
        self.assertTrue(resp.raw["_nightshift"]["schema_requested"])

    def test_an_unsupported_schema_fails_before_anything_is_sent(self):
        client = make_client(self.fake)
        with self.assertRaisesRegex(ValueError, "oneOf"):
            client.chat(user(), response_schema={"type": "object", "properties": {"x": {"oneOf": [{"type": "string"}]}}})
        self.assertEqual(self.fake.requests, [])

    def test_token_cap_cache_key_and_hashed_safety_identifier(self):
        client = make_client(self.fake, safety_identifier="user-42", cache_retention="in_memory")
        client.chat(user(), max_completion_tokens=50, cache_key="run-7", temperature=0.5)
        body = self.fake.last_request.body
        self.assertEqual((body["max_completion_tokens"], body["prompt_cache_key"], body["temperature"]), (50, "run-7", 0.5))
        self.assertEqual(body["prompt_cache_retention"], "in_memory")
        self.assertEqual(body["safety_identifier"], hashlib.sha256(b"user-42").hexdigest()[:32])
        self.assertEqual(len(body["safety_identifier"]), 32)
        self.assertNotIn(b"user-42", self.fake.last_request.raw)

    def test_malformed_requests_raise_value_error_without_sending(self):
        client = make_client(self.fake)
        bad_calls = [dict(messages=[]), dict(max_completion_tokens=0), dict(max_completion_tokens=True),
                     dict(temperature=float("nan")), dict(tools=[ToolSpec("bad name", "d", {})]),
                     dict(messages=[ChatMessage("assistant", None, [ToolCall("a", "f", {})])])]
        for kw in bad_calls:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                client.chat(kw.pop("messages", user()), **kw)
        self.assertEqual(self.fake.requests, [])

    def test_tool_call_round_trip_replays_the_documented_message_shapes(self):
        client = make_client(self.fake)
        first = client.chat(user("add 2 and 3"), tools=[ADD])
        call = first.message.tool_calls[0]
        self.assertEqual((call.name, call.arguments, first.finish_reason), ("add", {"a": 1, "b": 1}, "tool_calls"))
        self.assertIsNone(first.message.content)
        history = user("add 2 and 3") + [first.message, ChatMessage("tool", "5", tool_call_id=call.id)]
        second = client.chat(history, tools=[ADD])
        self.assertEqual(second.text, "ok")
        sent = self.fake.last_request.body["messages"]
        self.assertEqual(sent[1], {"role": "assistant", "content": None, "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": "add", "arguments": call.raw_arguments}}]})
        self.assertEqual(sent[2], {"role": "tool", "tool_call_id": call.id, "content": "5"})
        self.assertEqual(self.fake.last_request.body["tools"][0]["function"]["name"], "add")

    def test_parallel_tool_calls_are_all_returned_in_order(self):
        self.fake.enqueue(Reply.tools(("add", {"a": 1}), ("add", {"a": 2}), ("other", {})))
        resp = make_client(self.fake).chat(user(), tools=[ADD])
        self.assertEqual([(c.name, c.arguments) for c in resp.message.tool_calls], [("add", {"a": 1}), ("add", {"a": 2}), ("other", {})])
        self.assertEqual(len({c.id for c in resp.message.tool_calls}), 3)

    def test_invalid_tool_arguments_become_a_parse_error_and_can_be_replayed(self):
        self.fake.enqueue(Reply.tools(("add", "{not json"), ("add", "[1]")))
        client = make_client(self.fake)
        resp = client.chat(user(), tools=[ADD])
        bad, not_object = resp.message.tool_calls
        self.assertEqual((bad.arguments, bad.raw_arguments), ({}, "{not json"))
        self.assertIn("invalid JSON", bad.parse_error)
        self.assertEqual(not_object.arguments, {})
        self.assertTrue(not_object.parse_error)
        history = user() + [resp.message] + [ChatMessage("tool", "error: bad arguments", tool_call_id=c.id) for c in resp.message.tool_calls]
        client.chat(history, tools=[ADD])
        replayed = self.fake.last_request.body["messages"][1]["tool_calls"]
        self.assertEqual([c["function"]["arguments"] for c in replayed], ["{}", "{}"])     # the API never sees broken JSON

    def test_usage_cached_tokens_and_cached_known(self):
        client = make_client(self.fake)
        self.fake.enqueue(Reply.text("a", prompt_tokens=100, completion_tokens=20, cached_tokens=60))
        resp = client.chat(user())
        self.assertEqual(resp.usage, Usage(100, 20, 60, 120))
        self.assertTrue(resp.raw["_nightshift"]["cached_known"])
        self.fake.cache_report = "flat"
        self.fake.enqueue(Reply.text("b", prompt_tokens=10, completion_tokens=1, cached_tokens=7))
        flat = client.chat(user())
        self.assertEqual((flat.usage.cached_tokens, flat.raw["_nightshift"]["cached_known"]), (7, True))
        self.fake.cache_report = "none"
        self.fake.enqueue(Reply.text("c", prompt_tokens=10, completion_tokens=1, cached_tokens=7))
        unknown = client.chat(user())
        self.assertEqual((unknown.usage.cached_tokens, unknown.raw["_nightshift"]["cached_known"]), (0, False))

    def test_response_model_field_is_reported_and_falls_back_to_the_requested_model(self):
        client = make_client(self.fake)
        self.fake.enqueue(Reply.text("a", model=SPARK), Reply.text("b", model=None))
        self.assertEqual(client.chat(user()).model, SPARK)
        self.assertEqual(client.chat(user()).model, SPARK)
        other = make_client(self.fake, model="muse-spark-1.2")
        self.fake.enqueue(Reply.text("c", model="muse-spark-1.2"), Reply.text("d", model=None))
        self.assertEqual(other.chat(user()).model, "muse-spark-1.2")
        self.assertEqual(other.chat(user()).model, "muse-spark-1.2")

    def test_nightshift_usage_detail(self):
        ticks = iter([10.0, 10.25])
        client = make_client(self.fake, default_reasoning="high", timer=lambda: next(ticks))
        resp = client.chat(user())
        meta = resp.raw["_nightshift"]
        self.assertEqual({k: meta[k] for k in ("cached_known", "latency_ms", "retries", "stream_fallback", "requested_model",
                                                "reasoning_effort", "schema_requested")},
                         {"cached_known": True, "latency_ms": 250.0, "retries": 0, "stream_fallback": False,
                          "requested_model": SPARK, "reasoning_effort": "high", "schema_requested": False})
        self.assertIs(resp.raw["nightshift"], meta)        # budget.MeteredClient reads this spelling

    def test_concurrent_calls_share_one_client_safely(self):
        self.fake.unsupported_params.add("prompt_cache_key")
        client = make_client(self.fake)
        results: list[str] = []
        errors: list[BaseException] = []

        barrier = threading.Barrier(8)

        def work(i: int) -> None:
            try:
                barrier.wait(timeout=5)
                results.append(client.chat(user(f"m{i}"), cache_key="k").text)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual((errors, results), ([], ["ok"] * 8))
        self.assertEqual(client.disabled_params, {"prompt_cache_key"})


# ==================================================================================================
# retries
# ==================================================================================================
class ClientRetryTests(ServerCase):
    def test_retry_after_is_honoured(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.rate_limit(retry_after=2))
        resp = client.chat(user())
        self.assertEqual(client.sleeps.calls, [2.0])
        self.assertEqual(len(self.fake.requests), 2)
        self.assertEqual(resp.raw["_nightshift"]["retries"], 1)

    def test_backoff_doubles_from_half_a_second_plus_jitter(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.server_error(503), Fault.server_error(500), Fault.server_error(503))
        client.chat(user())
        self.assertEqual(client.sleeps.calls, [0.5 + 0.25, 1.0 + 0.25, 2.0 + 0.25])
        self.assertEqual(len(self.fake.requests), 4)

    def test_jitter_is_clamped_and_defaults_to_zero_to_one_second(self):
        client = make_client(self.fake, jitter=lambda: 7.0)
        self.fake.enqueue(Fault.server_error(503))
        client.chat(user())
        self.assertEqual(client.sleeps.calls, [1.5])
        sleeper = Sleeper()
        real = MetaChatClient(base_url=self.fake.base_url, api_key=self.fake.api_key, sleep=sleeper)   # default jitter source
        self.fake.enqueue(*[Fault.server_error(503)] * 3)
        real.chat(user())
        self.assertEqual(len(sleeper.calls), 3)
        for k, delay in enumerate(sleeper.calls):
            self.assertTrue(0.5 * 2 ** k <= delay <= 0.5 * 2 ** k + 1.0, delay)

    def test_gives_up_after_max_retries_with_a_retryable_error(self):
        client = make_client(self.fake)       # max_retries defaults to 4
        self.fake.enqueue(*[Fault.rate_limit()] * 5)
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        err = ctx.exception
        self.assertEqual((err.status, err.retryable, err.error_type), (429, True, "rate_limit_error"))
        self.assertEqual(len(self.fake.requests), 5)
        self.assertEqual(client.sleeps.calls, [0.75, 1.25, 2.25, 4.25])

    def test_max_retries_is_configurable(self):
        client = make_client(self.fake, max_retries=1)
        self.fake.enqueue(Fault.server_error(503), Fault.server_error(503), Fault.server_error(503))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.status, len(self.fake.requests)), (503, 2))
        zero = make_client(self.fake, max_retries=0)
        self.fake.reset()
        self.fake.enqueue(Fault.server_error(500))
        with self.assertRaises(ModelError):
            zero.chat(user())
        self.assertEqual((len(self.fake.requests), zero.sleeps.calls), (1, []))

    def test_an_absurd_retry_after_is_surfaced_instead_of_slept(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.rate_limit(retry_after=900))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.retry_after, ctx.exception.retryable, client.sleeps.calls), (900.0, True, []))
        self.assertEqual(len(self.fake.requests), 1)

    def test_unparseable_retry_after_falls_back_to_backoff(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.error(429, "slow down", type="rate_limit_error", headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
                          Fault.error(429, "slow down", type="rate_limit_error", headers={"Retry-After": "-5"}))
        client.chat(user())
        self.assertEqual(client.sleeps.calls, [0.75, 1.25])

    def test_other_statuses_are_not_retried_and_carry_the_envelope(self):
        for status in (400, 401, 402, 403, 404, 413, 422):
            with self.subTest(status=status):
                self.fake.reset()
                client = make_client(self.fake)
                self.fake.enqueue(Fault.error(status, f"nope {status}", type="thing_error", code="c1", param="p1"))
                with self.assertRaises(ModelError) as ctx:
                    client.chat(user())
                err = ctx.exception
                self.assertEqual((err.status, err.error_type, err.code, err.retryable), (status, "thing_error", "c1", False))
                self.assertIn(f"nope {status}", str(err))
                self.assertEqual((len(self.fake.requests), client.sleeps.calls), (1, []))

    def test_retry_then_non_retryable_stops(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.rate_limit(), Fault.error(402, "payment required", type="billing_error"))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.status, ctx.exception.retryable, len(self.fake.requests)), (402, False, 2))


# ==================================================================================================
# 504 and streaming
# ==================================================================================================
class ClientStreamTests(ServerCase):
    def test_504_is_retried_once_as_a_stream_and_reassembled(self):
        client = make_client(self.fake)
        self.fake.non_stream_status = 504
        self.fake.enqueue(Reply.tools(("add", {"a": 1, "b": "x" * 30}), ("add", {"a": 2}), text="thinking aloud",
                                      prompt_tokens=11, completion_tokens=9, cached_tokens=5))
        resp = client.chat(user(), tools=[ADD])
        first, second = self.fake.requests
        self.assertEqual((first.status, first.body.get("stream")), (504, None))
        self.assertEqual((second.status, second.body["stream"]), (200, True))
        self.assertEqual(resp.text, "thinking aloud")
        self.assertEqual([c.arguments for c in resp.message.tool_calls], [{"a": 1, "b": "x" * 30}, {"a": 2}])
        self.assertEqual(resp.finish_reason, "tool_calls")
        self.assertEqual(resp.usage, Usage(11, 9, 5, 20))
        meta = resp.raw["_nightshift"]
        self.assertEqual((meta["stream_fallback"], meta["streamed"], meta["attempts"]), (True, True, 2))

    def test_a_504_on_the_stream_attempt_is_not_retried_again(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.gateway_timeout(), Fault.gateway_timeout(), Reply.text("never"))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.status, ctx.exception.retryable), (504, False))
        self.assertEqual(len(self.fake.requests), 2)

    def test_the_streamed_retry_still_retries_rate_limits(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.gateway_timeout(), Fault.rate_limit(retry_after=1), Reply.text("finally"))
        resp = client.chat(user())
        self.assertEqual((resp.text, client.sleeps.calls, resp.raw["_nightshift"]["retries"]), ("finally", [1.0], 1))
        self.assertEqual([r.status for r in self.fake.requests], [504, 429, 200])

    def test_forced_stream(self):
        client = make_client(self.fake)
        self.fake.enqueue(Reply.text("streamed text"))
        resp = client.chat(user(), stream=True)
        self.assertTrue(self.fake.last_request.body["stream"])
        self.assertEqual(resp.text, "streamed text")
        self.assertEqual((resp.raw["_nightshift"]["streamed"], resp.raw["_nightshift"]["stream_fallback"]), (True, False))

    def test_streamed_reply_without_usage_gets_an_estimate_and_cached_unknown(self):
        self.fake.stream_usage = False
        client = make_client(self.fake)
        self.fake.enqueue(Reply.text("x" * 400))
        resp = client.chat(user("y" * 800), stream=True)
        self.assertGreaterEqual(resp.usage.completion_tokens, 90)
        self.assertGreaterEqual(resp.usage.prompt_tokens, 190)
        meta = resp.raw["_nightshift"]
        self.assertEqual((meta["usage_estimated"], meta["cached_known"], resp.usage.cached_tokens), (True, False, 0))

    def test_stream_model_and_request_id(self):
        client = make_client(self.fake)
        self.fake.enqueue(Reply.text("a", model="served-model"))
        resp = client.chat(user(), stream=True)
        self.assertEqual((resp.model, resp.request_id), ("served-model", "req_fake1"))


class StubTransport:
    """Injected transport returning canned (status, headers, body) tuples and recording what it was given."""

    def __init__(self, *results):
        self.results, self.calls = list(results), []

    def __call__(self, method, url, headers, body, timeout, stream):
        self.calls.append(dict(method=method, url=url, headers=dict(headers), body=body, timeout=timeout, stream=stream))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def ok_json(content="hi", **extra):
    body = {"id": "x", "model": SPARK, "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}, **extra}
    return 200, {"X-Request-Id": "r1"}, json.dumps(body).encode()


def sse(*events):
    return [(b"data: " + json.dumps(e).encode() + b"\n\n") if isinstance(e, dict) else e for e in events]


class InjectedTransportTests(unittest.TestCase):
    def client(self, *results, **kw):
        transport = StubTransport(*results)
        client = MetaChatClient(api_key="k-injected-0123456789", transport=transport, sleep=lambda s: None, **kw)
        return client, transport

    def test_transport_receives_method_url_headers_body_timeout(self):
        client, transport = self.client(ok_json(), timeout=33.0)
        resp = client.chat(user())
        call = transport.calls[0]
        self.assertEqual((call["method"], call["url"], call["timeout"], call["stream"]),
                         ("POST", "https://api.meta.ai/v1/chat/completions", 33.0, False))
        self.assertEqual(call["headers"]["Authorization"], "Bearer k-injected-0123456789")
        self.assertEqual(json.loads(call["body"])["model"], SPARK)
        self.assertEqual(resp.request_id, "r1")

    def test_per_call_timeout_overrides_the_default(self):
        client, transport = self.client(ok_json())
        client.chat(user(), timeout=4.5)
        self.assertEqual(transport.calls[0]["timeout"], 4.5)

    def test_headers_may_use_any_case_and_list_bodies_are_read(self):
        client, _ = self.client((429, {"RETRY-AFTER": "0"}, b'{"error": {"message": "x", "type": "rate_limit_error"}}'),
                                (200, {}, json.dumps(json.loads(ok_json()[2])).encode()))
        self.assertEqual(client.chat(user()).raw["_nightshift"]["retries"], 1)

    def test_stream_chunks_may_split_anywhere(self):
        events = sse({"id": "s", "model": SPARK, "choices": [{"delta": {"content": "ab"}, "finish_reason": None}]},
                     {"choices": [{"delta": {"content": "cd"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
                     b"data: [DONE]\n\n")
        blob = b"".join(events)
        client, _ = self.client((200, {}, iter([blob[i:i + 5] for i in range(0, len(blob), 5)])))
        resp = client.chat(user(), stream=True)
        self.assertEqual((resp.text, resp.usage.prompt_tokens), ("abcd", 4))

    def test_truncated_stream_is_an_error_not_a_partial_answer(self):
        client, _ = self.client((200, {}, iter(sse({"choices": [{"delta": {"content": "par"}, "finish_reason": None}]}))))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user(), stream=True)
        self.assertEqual(ctx.exception.error_type, "truncated_stream")

    def test_error_event_inside_a_stream(self):
        client, _ = self.client((200, {}, iter(sse({"error": {"message": "overloaded", "type": "server_error"}}))))
        with self.assertRaisesRegex(ModelError, "overloaded"):
            client.chat(user(), stream=True)

    def test_malformed_stream_and_interrupted_stream(self):
        client, _ = self.client((200, {}, iter([b"data: {nope}\n\n"])))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user(), stream=True)
        self.assertEqual(ctx.exception.error_type, "malformed_response")

        def broken():
            yield sse({"choices": [{"delta": {"content": "a"}, "finish_reason": None}]})[0]
            raise ConnectionResetError("reset")
        client2, _ = self.client((200, {}, broken()))
        with self.assertRaises(ModelError) as ctx2:
            client2.chat(user(), stream=True)
        self.assertEqual((ctx2.exception.error_type, ctx2.exception.retryable), ("network", True))

    def test_transport_failures_become_retryable_model_errors_without_internal_retry(self):
        for exc, kind in ((ConnectionRefusedError("refused"), "network"), (TimeoutError("slow"), "timeout"),
                          (OSError("dns"), "network")):
            with self.subTest(kind=kind):
                client, transport = self.client(exc, ok_json())
                with self.assertRaises(ModelError) as ctx:
                    client.chat(user())
                self.assertEqual((ctx.exception.error_type, ctx.exception.retryable, ctx.exception.status), (kind, True, None))
                self.assertEqual(len(transport.calls), 1)

    def test_malformed_success_bodies(self):
        for body in (b"<html>", b"[]", b'{"choices": []}', b'{"choices": "x"}', b""):
            with self.subTest(body=body):
                client, _ = self.client((200, {}, body))
                with self.assertRaises(ModelError) as ctx:
                    client.chat(user())
                self.assertFalse(ctx.exception.retryable)
        client, _ = self.client((200, {}, b'{"error": {"message": "soft failure", "type": "server_error", "code": "e1"}}'))
        with self.assertRaisesRegex(ModelError, "soft failure") as ctx:
            client.chat(user())
        self.assertEqual(ctx.exception.code, "e1")

    def test_message_content_as_part_list_and_missing_fields_are_tolerated(self):
        body = {"choices": [{"message": {"content": [{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}]}}]}
        client, _ = self.client((200, {}, json.dumps(body).encode()))
        resp = client.chat(user())
        self.assertEqual((resp.text, resp.finish_reason, resp.model), ("hello", "stop", SPARK))
        # no usage object in the reply: the client estimates, flags it, and cached tokens stay unknown
        self.assertGreater(resp.usage.prompt_tokens, 0)
        self.assertGreater(resp.usage.completion_tokens, 0)
        self.assertEqual(resp.usage.cached_tokens, 0)
        self.assertTrue(resp.raw["_nightshift"]["usage_estimated"])
        self.assertFalse(resp.raw["_nightshift"]["cached_known"])

    def test_non_json_error_body_gives_a_capped_snippet(self):
        client, _ = self.client((403, {}, b"<html>" + b"bad gateway " * 500 + b"</html>"))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.status, ctx.exception.error_type, ctx.exception.retryable), (403, None, False))
        self.assertLess(len(str(ctx.exception)), 800)



# ==================================================================================================
# optional-parameter stripping and fallback model
# ==================================================================================================
class ClientParamStrippingTests(ServerCase):
    CASES = {
        "prompt_cache_key": dict(chat=dict(cache_key="k")),
        "prompt_cache_retention": dict(init=dict(cache_retention="24h"), chat={}),
        "safety_identifier": dict(init=dict(safety_identifier="user-1"), chat={}),
        "reasoning_effort": dict(chat=dict(reasoning_effort="high")),
        "temperature": dict(chat=dict(temperature=0.2)),
    }

    def test_a_named_optional_parameter_is_dropped_retried_once_and_stays_disabled(self):
        for name, case in self.CASES.items():
            with self.subTest(name):
                self.fake.reset()
                self.fake.unsupported_params = {name}
                client = make_client(self.fake, **case.get("init", {}))
                resp = client.chat(user(), **case["chat"])
                first, second = self.fake.requests
                self.assertEqual((first.status, name in first.body), (400, True))
                self.assertEqual((second.status, name in second.body), (200, False))
                self.assertEqual(client.disabled_params, {name})
                self.assertIn(name, resp.raw["_nightshift"]["disabled_params"])
                client.chat(user(), **case["chat"])
                self.assertEqual(len(self.fake.requests), 3)                    # no further probing
                self.assertNotIn(name, self.fake.last_request.body)
                self.assertEqual(client.sleeps.calls, [])                       # not a backoff retry

    def test_each_parameter_is_stripped_in_turn(self):
        self.fake.unsupported_params = {"prompt_cache_key", "temperature"}
        client = make_client(self.fake)
        client.chat(user(), cache_key="k", temperature=0.1, reasoning_effort="low")
        self.assertEqual(len(self.fake.requests), 3)
        self.assertEqual(client.disabled_params, {"prompt_cache_key", "temperature"})
        self.assertEqual(self.fake.last_request.body["reasoning_effort"], "low")

    def test_the_error_param_field_alone_is_enough(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.bad_request("Invalid value", param="temperature"))
        client.chat(user(), temperature=0.4)
        self.assertEqual(client.disabled_params, {"temperature"})
        self.assertNotIn("temperature", self.fake.last_request.body)

    def test_reasoning_effort_stripped_is_reported_as_not_sent(self):
        self.fake.unsupported_params = {"reasoning_effort"}
        resp = make_client(self.fake).chat(user(), reasoning_effort="high")
        self.assertIsNone(resp.raw["_nightshift"]["reasoning_effort"])

    def test_a_400_about_a_parameter_that_was_not_sent_is_not_retried(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.bad_request("temperature is not supported"))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.status, len(self.fake.requests), client.disabled_params), (400, 1, set()))

    def test_only_a_400_triggers_stripping(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.error(404, "route for temperature not found", type="not_found_error"))
        with self.assertRaises(ModelError):
            client.chat(user(), temperature=0.4)
        self.assertEqual((len(self.fake.requests), client.disabled_params), (1, set()))

    def test_a_parameter_is_only_retried_once(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.bad_request("prompt_cache_key rejected"), Fault.bad_request("prompt_cache_key rejected again"))
        with self.assertRaises(ModelError):
            client.chat(user(), cache_key="k")
        self.assertEqual(len(self.fake.requests), 2)

    def test_tools_and_response_format_are_never_stripped(self):
        for name, kw in (("tools", dict(tools=[ADD])),
                         ("response_format", dict(response_schema={"type": "object", "properties": {"a": {"type": "string"}}}))):
            with self.subTest(name):
                self.fake.reset()
                self.fake.unsupported_params = {name}
                client = make_client(self.fake)
                with self.assertRaises(ModelError) as ctx:
                    client.chat(user(), **kw)
                self.assertEqual((ctx.exception.status, len(self.fake.requests), client.disabled_params), (400, 1, set()))
                self.assertIn(name, ctx.exception.param)


class ClientFallbackTests(ServerCase):
    def test_404_on_the_primary_retries_on_the_fallback_and_reports_who_answered(self):
        self.fake.models = {SPARK}
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK, default_reasoning="high", fallback_reasoning="xhigh")
        resp = client.chat(user())
        first, second = self.fake.requests
        self.assertEqual((first.status, first.body["model"], first.body["reasoning_effort"]), (404, WONDER, "high"))
        self.assertEqual((second.status, second.body["model"], second.body["reasoning_effort"]), (200, SPARK, "xhigh"))
        self.assertEqual(resp.model, SPARK)
        meta = resp.raw["_nightshift"]
        self.assertEqual((meta["requested_model"], meta["fallback_used"], meta["reasoning_effort"]), (WONDER, True, "xhigh"))

    def test_400_on_the_primary_also_falls_back(self):
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
        self.fake.enqueue(Fault.bad_request("this model does not accept chat completions"))
        resp = client.chat(user())
        self.assertEqual((resp.model, len(self.fake.requests)), (SPARK, 2))

    def test_max_effort_is_kept_on_a_spark_13_fallback(self):
        self.fake.models = {SPARK}
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
        client.chat(user(), reasoning_effort="max")
        self.assertEqual([r.body.get("reasoning_effort") for r in self.fake.requests], ["max", "max"])

    def test_max_effort_becomes_xhigh_only_when_the_fallback_cannot_take_max(self):
        for fallback in ("muse-spark-1.3-contributor", "muse-spark-1.2"):
            with self.subTest(fallback=fallback):
                self.fake.reset()
                self.fake.models = {fallback}
                client = make_client(self.fake, model=WONDER, fallback_model=fallback)
                resp = client.chat(user(), reasoning_effort="max")
                self.assertEqual([r.body.get("reasoning_effort") for r in self.fake.requests], ["max", "xhigh"])
                self.assertEqual(resp.raw["_nightshift"]["reasoning_effort"], "xhigh")

    def test_reported_model_falls_back_to_the_fallback_when_the_reply_has_no_model_field(self):
        self.fake.models = {SPARK}
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
        self.fake.enqueue(Fault.error(404, "no such model", type="not_found_error"), Reply.text("x", model=None))
        self.assertEqual(client.chat(user()).model, SPARK)

    def test_both_models_failing_raises_the_last_error(self):
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
        self.fake.enqueue(Fault.error(404, "primary gone", type="not_found_error"), Fault.error(400, "fallback bad", type="invalid_request_error"))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.status, len(self.fake.requests)), (400, 2))
        self.assertIn("fallback bad", str(ctx.exception))

    def test_other_failures_do_not_trigger_the_fallback(self):
        for status in (401, 402, 403, 413, 429):
            with self.subTest(status=status):
                self.fake.reset()
                client = make_client(self.fake, model=WONDER, fallback_model=SPARK, max_retries=0)
                self.fake.enqueue(Fault.error(status, "stop", type="x_error"))
                with self.assertRaises(ModelError):
                    client.chat(user())
                self.assertEqual(len(self.fake.requests), 1)

    def test_without_a_fallback_the_404_is_raised(self):
        self.fake.models = {SPARK}
        with self.assertRaises(ModelError) as ctx:
            make_client(self.fake, model=WONDER).chat(user())
        self.assertEqual((ctx.exception.status, ctx.exception.code, len(self.fake.requests)), (404, "model_not_found", 1))

    def test_invalid_effort_for_the_fallback_is_caught_at_construction(self):
        with self.assertRaises(ValueError):
            MetaChatClient(WONDER, api_key="k-0123456789", fallback_model="muse-spark-1.3-contributor", fallback_reasoning="max")
        MetaChatClient(WONDER, api_key="k-0123456789", fallback_model=SPARK, fallback_reasoning="max")   # documented for 1.3


class StickyFallbackTests(ServerCase):
    """A primary model that does not exist is found out once; the client then stops sending doomed requests to it."""

    def spark_only(self, **kw):
        self.fake.models = {SPARK}
        return make_client(self.fake, model=WONDER, fallback_model=SPARK, default_reasoning="high", fallback_reasoning="xhigh", **kw)

    def test_after_a_model_not_found_every_later_call_goes_straight_to_the_fallback(self):
        client = self.spark_only()
        first = client.chat(user())
        self.assertEqual([(r.body["model"], r.status) for r in self.fake.requests], [(WONDER, 404), (SPARK, 200)])
        self.assertIsNotNone(client.primary_unavailable)
        for _ in range(3):
            resp = client.chat(user())
        self.assertEqual([r.body["model"] for r in self.fake.requests[2:]], [SPARK] * 3)       # no further 404s
        self.assertEqual(len(self.fake.requests), 5)
        self.assertEqual((resp.model, resp.raw["_nightshift"]["fallback_used"], resp.raw["_nightshift"]["requested_model"]),
                         (SPARK, True, WONDER))
        self.assertEqual(self.fake.requests[-1].body["reasoning_effort"], "xhigh")              # the fallback's own effort
        self.assertEqual(resp.raw["_nightshift"]["reasoning_effort"], "xhigh")
        self.assertIn("fallback_reason", first.raw["_nightshift"])

    def test_the_reason_is_logged_once_on_the_call_that_found_out(self):
        client = self.spark_only()
        first, second, third = client.chat(user()), client.chat(user()), client.chat(user())
        reason = first.raw["_nightshift"]["fallback_reason"]
        self.assertIn(WONDER, reason)
        self.assertIn("model_not_found", reason)
        self.assertIn(SPARK, reason)
        self.assertEqual(reason, client.primary_unavailable)
        for later in (second, third):
            self.assertNotIn("fallback_reason", later.raw["_nightshift"])
            self.assertNotIn("fallback_reason", later.raw["nightshift"])

    def test_a_400_on_the_primary_falls_back_for_that_call_only(self):
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
        self.fake.enqueue(Fault.bad_request("this model does not accept chat completions"))
        self.assertEqual(client.chat(user()).model, SPARK)
        self.assertIsNone(client.primary_unavailable)
        self.assertEqual(client.chat(user()).model, WONDER)                                     # the primary is tried again
        self.assertEqual([r.body["model"] for r in self.fake.requests], [WONDER, SPARK, WONDER])

    def test_a_404_that_is_not_about_the_model_is_not_remembered(self):
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
        self.fake.enqueue(Fault.error(404, "Unknown route POST /v1/chat/completions", type="not_found_error"))
        self.assertEqual(client.chat(user()).model, SPARK)
        self.assertIsNone(client.primary_unavailable)
        self.assertNotIn("fallback_reason", client.chat(user()).raw["_nightshift"])
        self.assertEqual([r.body["model"] for r in self.fake.requests], [WONDER, SPARK, WONDER])

    def test_a_missing_model_is_recognised_by_code_param_or_message(self):
        for fault in (Fault.error(404, "gone", type="not_found_error", code="model_not_found"),
                      Fault.error(404, "gone", type="not_found_error", param="model"),
                      Fault.error(404, "The model 'x' does not exist", type="not_found_error")):
            with self.subTest(fault=str(fault)):
                self.fake.reset()
                client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
                self.fake.enqueue(fault)
                client.chat(user())
                self.assertIsNotNone(client.primary_unavailable)

    def test_without_a_fallback_nothing_is_remembered_and_the_primary_keeps_being_asked(self):
        self.fake.models = {SPARK}
        client = make_client(self.fake, model=WONDER)
        for _ in range(2):
            with self.assertRaises(ModelError):
                client.chat(user())
        self.assertEqual([r.body["model"] for r in self.fake.requests], [WONDER, WONDER])
        self.assertIsNone(client.primary_unavailable)

    def test_a_fallback_that_fails_too_still_raises_and_stays_the_only_target(self):
        client = self.spark_only()
        client.chat(user())
        self.fake.enqueue(Fault.error(400, "fallback bad", type="invalid_request_error"))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual([r.body["model"] for r in self.fake.requests], [WONDER, SPARK, SPARK])

    def test_concurrent_first_calls_all_succeed_and_the_reason_is_set_once(self):
        client = self.spark_only()
        barrier = threading.Barrier(4)
        out: list = []

        def work():
            barrier.wait(timeout=5)
            out.append(client.chat(user()))
        ts = [threading.Thread(target=work) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        self.assertEqual(len(out), 4)
        self.assertEqual({r.model for r in out}, {SPARK})
        self.assertEqual(sum(1 for r in out if "fallback_reason" in r.raw["_nightshift"]), 1)
        self.assertIsNotNone(client.primary_unavailable)

    def test_the_metered_client_charges_the_model_that_answered_and_the_effort_that_was_sent(self):
        if BUDGET_IMPORT_ERROR:
            self.skipTest(BUDGET_IMPORT_ERROR)
        with tempfile.TemporaryDirectory() as tmp:
            led = Ledger.create(Path(tmp), clock=FakeClock(), fsync=False)
            self.addCleanup(led.close)
            RunBuilder(led).start()
            gate = BudgetGate(BudgetLimits(), led)
            metered = MeteredClient(self.spark_only(), gate, "reviewer_hard")
            metered.chat(user(), reasoning_effort="high", charge_context={"exp_id": "e002", "purpose": "review"})
            (ev,) = [e for e in led.events() if e.kind == "budget.charged" and e.data["kind"] == "llm"]
            self.assertEqual((ev.data["model"], ev.data["usage"]["requested_model"], ev.data["usage"]["reasoning_effort"]),
                             (SPARK, WONDER, "xhigh"))
            self.assertEqual((ev.data["exp_id"], ev.data["purpose"]), ("e002", "review"))


class RequestTimeoutTests(unittest.TestCase):
    """Reasoning calls are slow: the default fits them, a large completion cap gets more, and an explicit value wins."""

    def timeout_of(self, *, ctor=None, **chat_kw) -> float:
        kw = {} if ctor is None else {"timeout": ctor}
        transport = StubTransport(ok_json())
        client = MetaChatClient(api_key="k-injected-0123456789", transport=transport, sleep=lambda s: None, **kw)
        client.chat(user(), **chat_kw)
        return transport.calls[0]["timeout"]

    def test_the_documented_constants(self):
        from nightshift.models import meta
        self.assertEqual((meta.DEFAULT_TIMEOUT, meta.REASONING_TIMEOUT, meta.REASONING_TOKEN_THRESHOLD), (300.0, 600.0, 8000))
        self.assertEqual(MetaChatClient(api_key="k-0123456789").timeout, 300.0)

    def test_the_default_is_300_seconds(self):
        self.assertEqual(self.timeout_of(), 300.0)
        self.assertEqual(self.timeout_of(max_completion_tokens=6000), 300.0)

    def test_the_threshold_itself_is_still_the_default_and_one_more_token_is_not(self):
        self.assertEqual(self.timeout_of(max_completion_tokens=8000), 300.0)
        self.assertEqual(self.timeout_of(max_completion_tokens=8001), 600.0)
        self.assertEqual(self.timeout_of(max_completion_tokens=16000), 600.0)           # the retry cap of an 8000-token role

    def test_a_per_call_timeout_wins_over_everything(self):
        self.assertEqual(self.timeout_of(max_completion_tokens=16000, timeout=12.5), 12.5)
        self.assertEqual(self.timeout_of(ctor=33.0, max_completion_tokens=16000, timeout=4.0), 4.0)

    def test_a_constructor_timeout_is_explicit_and_is_not_scaled(self):
        self.assertEqual(self.timeout_of(ctor=33.0), 33.0)
        self.assertEqual(self.timeout_of(ctor=33.0, max_completion_tokens=16000), 33.0)

    def test_the_stream_retry_after_a_504_keeps_the_same_timeout(self):
        stream = iter(sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, b"data: [DONE]\n\n"))
        transport = StubTransport((504, {}, b"{}"), (200, {}, stream))
        client = MetaChatClient(api_key="k-injected-0123456789", transport=transport, sleep=lambda s: None)
        client.chat(user(), max_completion_tokens=12000)
        self.assertEqual([c["timeout"] for c in transport.calls], [600.0, 600.0])

    def test_bad_constructor_timeouts_are_still_refused(self):
        for bad in (0, -1, float("nan"), float("inf"), "5", True):
            with self.subTest(timeout=bad), self.assertRaises(ValueError):
                MetaChatClient(api_key="k-0123456789", timeout=bad)


# ==================================================================================================
# errors, redirects, timeouts over real sockets
# ==================================================================================================
class ClientNetworkTests(ServerCase):
    def test_redirects_are_never_followed(self):
        for status in (301, 302, 307, 308):
            with self.subTest(status=status):
                self.fake.reset()
                client = make_client(self.fake)
                self.fake.enqueue(Fault.redirect("/v1/chat/completions", status), Reply.text("must not be reached"))
                with self.assertRaises(ModelError) as ctx:
                    client.chat(user())
                self.assertEqual((ctx.exception.status, ctx.exception.retryable), (status, False))
                self.assertEqual(len(self.fake.requests), 1)            # the Location was not requested
                self.assertIn("redirect", str(ctx.exception))

    def test_a_redirect_to_another_host_does_not_leak_the_key(self):
        other = FakeMetaServer().start()
        self.addCleanup(other.stop)
        self.fake.enqueue(Fault.redirect(other.base_url + "/chat/completions"))
        with self.assertRaises(ModelError):
            make_client(self.fake).chat(user())
        self.assertEqual(other.requests, [])

    def test_connection_refused_is_a_retryable_error_naming_no_secret(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        client = MetaChatClient(base_url=f"http://127.0.0.1:{port}/v1", api_key="k-0123456789", sleep=lambda s: None)
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.error_type, ctx.exception.retryable, ctx.exception.status), ("network", True, None))
        self.assertNotIn("k-0123456789", str(ctx.exception))

    def test_read_timeout(self):
        self.fake.enqueue(Fault.slow(1.0))
        client = make_client(self.fake, timeout=0.3)
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.error_type, ctx.exception.retryable), ("timeout", True))
        self.fake.enqueue(Fault.slow(0.6))
        with self.assertRaises(ModelError):
            make_client(self.fake, timeout=30).chat(user(), timeout=0.2)     # per-call timeout wins

    def test_server_hanging_up_is_a_retryable_network_error(self):
        self.fake.enqueue(Fault.disconnect_now())
        with self.assertRaises(ModelError) as ctx:
            make_client(self.fake).chat(user())
        self.assertEqual((ctx.exception.error_type, ctx.exception.retryable), ("network", True))

    def test_html_error_and_garbage_success_bodies(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.garbage("<html>forbidden</html>", 403), Fault.garbage("not json", 200))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.status, ctx.exception.error_type), (403, None))
        with self.assertRaises(ModelError) as ctx2:
            client.chat(user())
        self.assertEqual((ctx2.exception.error_type, ctx2.exception.retryable), ("malformed_response", False))

    def test_loopback_requests_ignore_proxy_environment_variables(self):
        env = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
               "https_proxy": "http://127.0.0.1:9"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(make_client(self.fake).chat(user()).text, "ok")


# ==================================================================================================
# security
# ==================================================================================================
class ClientSecurityTests(ServerCase):
    def test_base_url_must_be_https_unless_loopback(self):
        for url in ("http://example.com/v1", "http://api.meta.ai/v1", "http://127.0.0.1.evil.com/v1", "http://localhost.evil.com/v1",
                    "ftp://example.com/v1", "https:///v1", "example.com/v1", "", "http://10.0.0.5/v1"):
            with self.subTest(url=url), self.assertRaises(ModelConfigError):
                check_base_url(url)
        for url, expected in (("https://api.meta.ai/v1", "https://api.meta.ai/v1"), ("https://api.meta.ai/v1/", "https://api.meta.ai/v1"),
                              ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1"), ("http://localhost/v1", "http://localhost/v1"),
                              ("http://[::1]:9/v1", "http://[::1]:9/v1"), ("HTTPS://API.EXAMPLE.COM/v1?x=1#f", "https://api.example.com/v1")):
            with self.subTest(url=url):
                self.assertEqual(check_base_url(url), expected)

    def test_userinfo_is_stripped_and_never_echoed(self):
        self.assertEqual(check_base_url("https://alice:hunter2xx@api.example.com/v1"), "https://api.example.com/v1")
        with self.assertRaises(ModelConfigError) as ctx:
            check_base_url("http://alice:hunter2xx@example.com/v1")
        self.assertNotIn("hunter2xx", str(ctx.exception))
        self.assertNotIn("alice", str(ctx.exception))
        client = MetaChatClient(base_url=f"http://alice:hunter2xx@127.0.0.1:{self.fake.port}/v1", api_key=self.fake.api_key)
        self.assertNotIn("hunter2xx", repr(client) + client.base_url)
        client.chat(user())
        req = self.fake.last_request
        self.assertEqual(req.authorization, f"Bearer {self.fake.api_key}")            # not Basic alice:...
        self.assertNotIn(b"hunter2xx", req.raw)

    def test_the_key_is_registered_with_the_secret_registry(self):
        key = "sk-registry-test-0123456789"
        MetaChatClient(api_key=key)
        self.assertEqual(secrets.scrub_text(f"error talking with {key}!"), f"error talking with {secrets.REDACTED}!")

    def test_a_401_that_echoes_the_authorization_header_does_not_leak_the_key(self):
        wrong = "sk-wrong-key-abcdefghijklmnop"
        client = MetaChatClient(base_url=self.fake.base_url, api_key=wrong, sleep=lambda s: None)
        # premise: the fake really does put the header value in the response body
        _, _, data = raw_post(self.fake, {"model": SPARK, "messages": [{"role": "user", "content": "x"}]}, key=wrong)
        self.assertIn(wrong.encode(), data)
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        err = ctx.exception
        self.assertEqual((err.status, err.retryable, err.error_type, err.code), (401, False, "authentication_error", "invalid_api_key"))
        everything = " ".join([str(err), repr(err), str(err.args), str(err.error_type), str(err.code), str(getattr(err, "param", "")),
                               "".join(traceback.format_exception(err)), repr(client), str(client.__dict__.get("base_url"))])
        self.assertNotIn(wrong, everything)
        self.assertIn(secrets.REDACTED, str(err))

    def test_a_key_too_short_for_the_registry_is_still_scrubbed_from_errors(self):
        short = "abc12"          # below secrets.register's 8-character floor
        client = MetaChatClient(base_url=self.fake.base_url, api_key=short, sleep=lambda s: None)
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertNotIn(short, str(ctx.exception))

    def test_the_key_never_reaches_the_ledger(self):
        wrong = "sk-ledger-key-abcdefghijklmnop"
        client = MetaChatClient(base_url=self.fake.base_url, api_key=wrong, sleep=lambda s: None)
        with tempfile.TemporaryDirectory() as tmp:
            led = Ledger.create(tmp, clock=FakeClock(), fsync=False)
            rb = RunBuilder(led)
            rb.start()
            try:
                client.chat(user())
            except ModelError as exc:
                led.append("budget.charged", "engine", {"kind": "llm", "note": str(exc), "error": repr(exc)})
            led.close()
            text = (Path(tmp) / "ledger.jsonl").read_text()
        self.assertIn("401", text)
        self.assertNotIn(wrong, text)

    def test_the_key_travels_only_in_the_authorization_header(self):
        client = make_client(self.fake, safety_identifier="someone")
        client.chat(user("a prompt"), tools=[ADD], cache_key="k")
        req = self.fake.last_request
        key = self.fake.api_key.encode()
        self.assertNotIn(key, req.raw)
        self.assertNotIn(key, req.path.encode())
        self.assertEqual([k for k, v in req.headers.items() if self.fake.api_key in v], ["authorization"])

    def test_raw_request_scrubs_what_the_server_echoes(self):
        wrong = "sk-raw-key-abcdefghijklmnopqr"
        client = MetaChatClient(base_url=self.fake.base_url, api_key=wrong)
        status, payload = client.raw_request({"model": SPARK, "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 401)
        self.assertNotIn(wrong, json.dumps(payload))

    def test_missing_key_is_a_clear_error_that_prints_no_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelConfigError) as ctx:
                MetaChatClient()
            self.assertIn("MODEL_API_KEY", str(ctx.exception))
            self.assertIn("META_API_KEY", str(ctx.exception))
            with self.assertRaisesRegex(ModelConfigError, "MY_KEY"):
                MetaChatClient(api_key_env="MY_KEY")

    def test_key_environment_lookup_order(self):
        with mock.patch.dict(os.environ, {"META_API_KEY": "meta-key-0123456789"}, clear=True):
            self.assertEqual(MetaChatClient(base_url=self.fake.base_url)._api_key, "meta-key-0123456789")
        with mock.patch.dict(os.environ, {"META_API_KEY": "meta-key-0123456789", "MODEL_API_KEY": "model-key-0123456789"}, clear=True):
            self.assertEqual(MetaChatClient(base_url=self.fake.base_url)._api_key, "model-key-0123456789")
        with mock.patch.dict(os.environ, {"META_API_KEY": "meta-key-0123456789", "CUSTOM": "custom-key-0123456789"}, clear=True):
            self.assertEqual(MetaChatClient(base_url=self.fake.base_url, api_key_env="CUSTOM")._api_key, "custom-key-0123456789")
            with self.assertRaises(ModelConfigError):          # an explicit env name does not silently fall back
                MetaChatClient(base_url=self.fake.base_url, api_key_env="UNSET_NAME")

    def test_keys_that_could_inject_headers_are_refused_without_echo(self):
        for bad in ("good-prefix\r\nX-Evil: 1", "has space inside", "tab\tinside", "", "ab"):
            with self.subTest(bad=bad), self.assertRaises(ModelConfigError) as ctx:
                MetaChatClient(api_key=bad)
            self.assertNotIn("X-Evil", str(ctx.exception))

    def test_constructor_argument_validation(self):
        for kw in (dict(model=""), dict(max_retries=-1), dict(timeout=0), dict(timeout=float("nan")), dict(default_reasoning="turbo")):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                MetaChatClient(api_key="k-0123456789", **kw)

    def test_repr_has_no_secret(self):
        client = MetaChatClient(base_url=self.fake.base_url, api_key=self.fake.api_key, fallback_model=SPARK)
        self.assertNotIn(self.fake.api_key, repr(client) + str(client))
        self.assertIn(SPARK, repr(client))


class FromEnvTests(unittest.TestCase):
    def test_defaults_and_key_precedence(self):
        c = MetaChatClient.from_env(env={"META_API_KEY": "meta-key-0123456789"})
        self.assertEqual((c.model, c.base_url, c._api_key), (SPARK, "https://api.meta.ai/v1", "meta-key-0123456789"))
        c = MetaChatClient.from_env(env={"META_API_KEY": "meta-key-0123456789", "MODEL_API_KEY": "model-key-0123456789"})
        self.assertEqual(c._api_key, "model-key-0123456789")

    def test_nightshift_variables_override_arguments(self):
        env = {"MODEL_API_KEY": "model-key-0123456789", "NIGHTSHIFT_BASE_URL": "http://127.0.0.1:9/v1", "NIGHTSHIFT_MODEL": "muse-spark-1.2"}
        c = MetaChatClient.from_env("muse-wonder-1.0", base_url="https://elsewhere.example/v1", env=env)
        self.assertEqual((c.model, c.base_url), ("muse-spark-1.2", "http://127.0.0.1:9/v1"))
        c = MetaChatClient.from_env("muse-wonder-1.0", base_url="https://elsewhere.example/v1", env={"MODEL_API_KEY": "model-key-0123456789"})
        self.assertEqual((c.model, c.base_url), ("muse-wonder-1.0", "https://elsewhere.example/v1"))

    def test_extra_keywords_are_passed_through(self):
        c = MetaChatClient.from_env(env={"MODEL_API_KEY": "model-key-0123456789"}, fallback_model=WONDER, max_retries=1)
        self.assertEqual((c.fallback_model, c.max_retries), (WONDER, 1))

    def test_no_key_and_unsafe_url_are_clear_errors(self):
        with self.assertRaisesRegex(ModelConfigError, "MODEL_API_KEY"):
            MetaChatClient.from_env(env={})
        with self.assertRaises(ModelConfigError):
            MetaChatClient.from_env(env={"MODEL_API_KEY": "model-key-0123456789", "NIGHTSHIFT_BASE_URL": "http://evil.example/v1"})

    def test_reads_the_process_environment_by_default(self):
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "proc-key-0123456789"}, clear=True):
            self.assertEqual(MetaChatClient.from_env()._api_key, "proc-key-0123456789")


# ==================================================================================================
# routing config
# ==================================================================================================
class RoutingConfigTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "nightshift.toml"
        p.write_text(text)
        return p

    def test_constants_match_the_contract(self):
        self.assertEqual(config.ROLES, ("supervisor", "designer", "implementer", "reviewer", "reviewer_hard", "reflector", "judge", "auditor_final"))
        self.assertEqual(config.DEFAULT_EFFORT, {"supervisor": "medium", "designer": "low", "implementer": "low", "reviewer": "high",
                                                 "reviewer_hard": "high", "reflector": "medium", "judge": "high", "auditor_final": "high"})

    def test_defaults(self):
        routing = config.load_routing(None)
        self.assertEqual(set(routing), set(config.ROLES))
        for role, ep in routing.items():
            with self.subTest(role):
                self.assertEqual((ep.base_url, ep.api_key_env, ep.reasoning_effort), ("https://api.meta.ai/v1", "MODEL_API_KEY", config.DEFAULT_EFFORT[role]))
                if role in ("reviewer_hard", "auditor_final"):
                    self.assertEqual((ep.model, ep.fallback_model, ep.fallback_reasoning), (WONDER, SPARK, "xhigh"))
                else:
                    self.assertEqual((ep.model, ep.fallback_model), (SPARK, None))

    def test_no_default_is_a_contributor_model(self):
        for source in (config.load_routing(None), config.load_routing(self.write("[models.default]\nreasoning_effort = 'low'\n"))):
            for role, ep in source.items():
                with self.subTest(role):
                    self.assertNotIn("contributor", ep.model)
                    self.assertNotIn("contributor", ep.fallback_model or "")

    def test_a_contributor_model_only_by_explicit_choice(self):
        routing = config.load_routing(self.write("[roles.judge]\nmodel = 'muse-spark-1.3-contributor'\n"))
        self.assertEqual(routing["judge"].model, "muse-spark-1.3-contributor")
        self.assertTrue(all("contributor" not in ep.model for r, ep in routing.items() if r != "judge"))

    def test_toml_defaults_and_role_overrides(self):
        routing = config.load_routing(self.write("""
[models.default]
reasoning_effort = "low"
api_key_env = "MY_KEY"

[roles.reviewer]
model = "muse-spark-1.2"
base_url = "http://127.0.0.1:8123/v1"
reasoning_effort = "xhigh"
fallback_model = "muse-spark-1.3"

[roles.reviewer_hard]
reasoning_effort = "max"
"""))
        rev = routing["reviewer"]
        self.assertEqual((rev.model, rev.base_url, rev.reasoning_effort, rev.fallback_model, rev.api_key_env),
                         ("muse-spark-1.2", "http://127.0.0.1:8123/v1", "xhigh", SPARK, "MY_KEY"))
        self.assertEqual((routing["supervisor"].reasoning_effort, routing["supervisor"].api_key_env), ("low", "MY_KEY"))
        self.assertEqual((routing["reviewer_hard"].model, routing["reviewer_hard"].reasoning_effort), (WONDER, "max"))

    def test_changing_the_default_model_or_host_moves_the_hard_roles_off_wonder(self):
        for toml in ("[models.default]\nmodel = 'partner-llama'\n", "[models.default]\nbase_url = 'https://partner.example/v1'\n"):
            with self.subTest(toml=toml):
                routing = config.load_routing(self.write(toml))
                for role in ("reviewer_hard", "auditor_final"):
                    self.assertNotEqual(routing[role].model, WONDER)
                    self.assertIsNone(routing[role].fallback_model)
        routing = config.load_routing(self.write("[models.default]\nmodel = 'partner-llama'\n[roles.auditor_final]\nmodel = 'muse-wonder-1.0'\n"))
        self.assertEqual(routing["auditor_final"].model, WONDER)

    def test_an_empty_fallback_model_removes_the_fallback(self):
        routing = config.load_routing(self.write("[roles.reviewer_hard]\nfallback_model = ''\n"))
        self.assertIsNone(routing["reviewer_hard"].fallback_model)
        self.assertIsNone(routing["reviewer_hard"].fallback_reasoning)

    def test_bad_configuration_is_rejected_with_a_precise_message(self):
        cases = {
            "unknown role": ("[roles.oracle]\nmodel = 'x'\n", "oracle"),
            "unknown key": ("[roles.judge]\nmodle = 'x'\n", "modle"),
            "unknown default key": ("[models.default]\ntemperature = 'x'\n", "temperature"),
            "non-string": ("[roles.judge]\nmodel = 5\n", "must be a string"),
            "http host": ("[roles.judge]\nbase_url = 'http://example.com/v1'\n", "https"),
            "bad effort": ("[roles.judge]\nreasoning_effort = 'turbo'\n", "turbo"),
            "max on contributor": ("[roles.judge]\nmodel = 'muse-spark-1.3-contributor'\nreasoning_effort = 'max'\n", "contributor"),
            "max on spark 1.2": ("[roles.judge]\nmodel = 'muse-spark-1.2'\nreasoning_effort = 'max'\n", "muse-spark-1.3"),
            "ceiling not an integer": ("[roles.judge]\ncompletion_cap_ceiling = '9000'\n", "positive integer"),
            "ceiling zero": ("[roles.judge]\ncompletion_cap_ceiling = 0\n", "positive integer"),
            "ceiling bool": ("[roles.judge]\ncompletion_cap_ceiling = true\n", "positive integer"),
            "empty model": ("[roles.judge]\nmodel = ''\n", "model"),
            "bad models table": ("[models.other]\nmodel = 'x'\n", "models.default"),
            "bad toml": ("[roles.judge\n", "cannot read"),
        }
        for name, (toml, needle) in cases.items():
            with self.subTest(name), self.assertRaises(ModelConfigError) as ctx:
                config.load_routing(self.write(toml))
            self.assertIn(needle, str(ctx.exception))

    def test_missing_file_is_an_error_but_none_means_defaults(self):
        with self.assertRaisesRegex(ModelConfigError, "not found"):
            config.load_routing(Path("/nonexistent/nightshift.toml"))
        self.assertEqual(config.load_routing(None)["judge"].model, SPARK)

    def test_client_for_builds_a_configured_client(self):
        routing = config.load_routing(None)
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "env-key-0123456789"}, clear=True):
            reviewer = config.client_for("reviewer", routing)
            hard = config.client_for("reviewer_hard", routing)
            positional = config.client_for("judge", routing, None, None)
        self.assertIsInstance(reviewer, MetaChatClient)
        self.assertEqual((reviewer.model, reviewer.default_reasoning, reviewer.fallback_model), (SPARK, "high", None))
        self.assertEqual((hard.model, hard.default_reasoning, hard.fallback_model, hard.fallback_reasoning), (WONDER, "high", SPARK, "xhigh"))
        self.assertEqual(positional.model, SPARK)
        with self.assertRaises(ModelConfigError):
            config.client_for("oracle", routing)

    def test_client_for_leaves_the_timeout_to_the_client_and_accepts_an_override(self):
        routing = config.load_routing(None)
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "env-key-0123456789"}, clear=True):
            default = config.client_for("reviewer", routing)
            custom = config.client_for("reviewer", routing, timeout=45.0)
        self.assertEqual(default.timeout, 300.0)
        self.assertFalse(default._timeout_explicit)                      # so a large completion cap still gets the longer timeout
        self.assertEqual((custom.timeout, custom._timeout_explicit), (45.0, True))

    @unittest.skipIf(BUDGET_IMPORT_ERROR, f"nightshift.budget cannot be imported: {BUDGET_IMPORT_ERROR}")
    def test_a_gated_client_advertises_charge_context_and_a_plain_one_does_not(self):
        routing = config.load_routing(None)
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "env-key-0123456789"}, clear=True):
            self.assertTrue(config.client_for("judge", routing, gate=mock.Mock()).supports_charge_context)
            self.assertFalse(getattr(config.client_for("judge", routing), "supports_charge_context", False))

    def test_client_for_without_a_key_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ModelConfigError, "API key"):
            config.client_for("supervisor", config.load_routing(None))

    def test_client_for_end_to_end_with_fallback_and_effort(self):
        with FakeMetaServer(models={SPARK}) as fake:
            routing = config.load_routing(None)
            for ep in routing.values():
                ep.base_url = fake.base_url
            with mock.patch.dict(os.environ, {"MODEL_API_KEY": fake.api_key}, clear=True):
                resp = config.client_for("reviewer_hard", routing, sleep=lambda s: None).chat(user())
            first, second = fake.requests
            self.assertEqual((first.body["model"], first.body["reasoning_effort"], first.status), (WONDER, "high", 404))
            self.assertEqual((second.body["model"], second.body["reasoning_effort"], second.status), (SPARK, "xhigh", 200))
            self.assertEqual(resp.model, SPARK)

    @unittest.skipIf(BUDGET_IMPORT_ERROR, f"nightshift.budget cannot be imported: {BUDGET_IMPORT_ERROR}")
    def test_client_for_with_a_gate_charges_the_budget(self):
        from nightshift.types import BudgetLimits, Kind
        with FakeMetaServer() as fake, tempfile.TemporaryDirectory() as tmp:
            led = Ledger.create(tmp, clock=FakeClock(), fsync=False)
            self.addCleanup(led.close)
            RunBuilder(led).start()
            gate = BudgetGate(BudgetLimits(max_llm_usd=1.0), led, clock=FakeClock(step=0.0))
            routing = config.load_routing(None)
            for ep in routing.values():
                ep.base_url = fake.base_url
            with mock.patch.dict(os.environ, {"MODEL_API_KEY": fake.api_key}, clear=True):
                client = config.client_for("reviewer", routing, gate=gate, actor="reviewer:A", sleep=lambda s: None)
                self.assertIsInstance(client, MeteredClient)
                fake.enqueue(Fault.server_error(503), Reply.text("fine", prompt_tokens=1000, completion_tokens=200, cached_tokens=400))
                client.chat(user(), reasoning_effort="high")
            charge = [e for e in led.events() if e.kind == Kind.BUDGET_CHARGED and e.data["kind"] == "llm"][0]
            self.assertEqual((charge.actor, charge.data["model"]), ("reviewer:A", SPARK))
            self.assertAlmostEqual(charge.data["amount"], (600 * 1.25 + 400 * 0.15 + 200 * 4.25) / 1e6)
            usage = charge.data["usage"]
            self.assertEqual((usage["cached_tokens"], usage["cached_known"], usage["retries"], usage["reasoning_effort"]), (400, True, 1, "high"))

    def test_client_for_with_a_gate_uses_the_role_as_default_actor(self):
        if BUDGET_IMPORT_ERROR:
            self.skipTest(f"nightshift.budget cannot be imported: {BUDGET_IMPORT_ERROR}")
        gate = mock.Mock()
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "env-key-0123456789"}, clear=True):
            client = config.client_for("judge", config.load_routing(None), gate=gate)
        self.assertEqual((client._actor, client.model), ("judge", SPARK))


# ==================================================================================================
# live smoke
# ==================================================================================================
class LiveSmokeTests(ServerCase):
    def factory(self, **kw):
        return live_smoke.build_client_factory(base_url=self.fake.base_url, env={"MODEL_API_KEY": self.fake.api_key}, **kw)

    def test_all_probes_pass_against_a_documented_endpoint(self):
        report = live_smoke.run_smoke(self.factory(), clock=lambda: 1234.5)
        names = [p["name"] for p in report["probes"]]
        self.assertEqual(names, ["tool_roundtrip", "strict_schema", "effort_minimal", "effort_low", "effort_medium", "effort_high",
                                 "effort_xhigh", "prompt_cache", "stream", "rejected_param", "effort_max", "effort_none_rejected",
                                 "strict_PLAN", "strict_REVIEW", "strict_REFLECT", "strict_VERDICT", "supervisor_tools",
                                 "temperature_with_reasoning", "completion_cap_ceiling", "assistant_tool_calls_content",
                                 "parallel_tool_calls", "tools_with_response_format", "system_and_developer_roles",
                                 "ratelimit_headers", "prompt_cache_large", "wonder_chat"])
        self.assertTrue(report["ok"], [p for p in report["probes"] if p["required"] and not p["ok"]])
        self.assertEqual(report["required_inconclusive"], [])
        self.assertEqual((report["required_failed"], report["generated_at"], report["cached_known"], report["cache_hit"]), ([], 1234.5, True, True))
        self.assertEqual(report["models"], {"requested": SPARK, "answered": [SPARK, WONDER]})
        self.assertEqual(report["wonder_chat_completions"], "supported")
        by_name = {p["name"]: p for p in report["probes"]}
        self.assertEqual(by_name["tool_roundtrip"]["detail"]["tool_name"], "add_numbers")
        self.assertTrue(by_name["stream"]["detail"]["streamed"])
        self.assertEqual((by_name["rejected_param"]["detail"]["status"], by_name["rejected_param"]["detail"]["has_message"]), (400, True))
        self.assertGreater(by_name["prompt_cache"]["detail"]["second_cached_tokens"], 0)
        # chat calls and raw probe requests alike; the Responses probes have their own tally in report["responses"]["usage"]
        self.assertEqual(report["usage"]["calls"], sum(1 for r in self.fake.requests if r.endpoint == "chat"))
        self.assertEqual(report["responses"]["usage"]["calls"], sum(1 for r in self.fake.requests if r.endpoint == "responses"))
        self.assertGreater(report["usage"]["estimated_usd"], 0)
        self.assertTrue(all(p["latency_ms"] >= 0 for p in report["probes"]))

    def test_each_probe_exercises_what_it_claims(self):
        live_smoke.run_smoke(self.factory())
        bodies = self.fake.bodies()
        first_two = [b for b in bodies if b.get("tools")][:2]                         # tool call + tool-result turn
        self.assertEqual([any(m["role"] == "tool" for m in b["messages"]) for b in first_two], [False, True])
        self.assertTrue(any(b.get("response_format", {}).get("json_schema", {}).get("strict") is True for b in bodies))
        self.assertEqual([b["reasoning_effort"] for b in bodies if "reasoning_effort" in b and len(b["messages"]) == 1
                          and b.get("max_completion_tokens") == 2000 and not b.get("tools") and "response_format" not in b
                          and "stream" not in b][:5], ["minimal", "low", "medium", "high", "xhigh"])
        cached = [b for b in bodies if b.get("prompt_cache_key") == "nightshift-live-smoke"]
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached[0]["messages"], cached[1]["messages"])
        self.assertEqual(sum(1 for b in bodies if b.get("stream")), 1)
        self.assertEqual(self.fake.requests[[b.get("stream") for b in bodies].index(True)].body["stream_options"],
                         {"include_usage": True})
        rejected = [r.body for r in self.fake.requests if r.status == 400]
        self.assertIn("stop", rejected[0])                                                  # the deliberately rejected parameter
        self.assertEqual(len(rejected), 2)                                                  # ... and reasoning_effort "none"
        for req in self.fake.requests:
            if req.status == 200:
                self.assertEqual(set(req.body) & set(NEVER_SENT), set())

    def test_report_holds_no_key_and_no_message_content(self):
        report = live_smoke.run_smoke(self.factory())
        text = json.dumps(report)
        self.assertNotIn(self.fake.api_key, text)
        strings: list[str] = []
        for body in self.fake.bodies():
            for m in body["messages"]:
                strings.append(m["content"] or "")
                strings.extend(c["function"]["arguments"] for c in m.get("tool_calls", []))
        probe_texts = [s for s in strings if len(s) >= 6]
        self.assertGreaterEqual(len(probe_texts), 6)
        for s in probe_texts:
            self.assertNotIn(s, text)
        for word in ("single word", "add_numbers tool", "Reference list", "item0001"):
            self.assertNotIn(word, text)

    def test_report_is_written_atomically_as_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "dir" / "live_probe.json"
            report = live_smoke.run_smoke(self.factory(), out)
            self.assertEqual(json.loads(out.read_text()), json.loads(json.dumps(report)))
            self.assertEqual([p.name for p in out.parent.iterdir()], ["live_probe.json"])

    def test_cached_tokens_not_reported_means_cached_known_false(self):
        self.fake.cache_report = "none"
        report = live_smoke.run_smoke(self.factory())
        self.assertEqual((report["ok"], report["cached_known"], report["cache_hit"]), (True, False, False))

    def test_a_failing_probe_is_recorded_and_the_rest_still_run(self):
        self.fake.enqueue(Reply.text("I will not call the tool"))
        report = live_smoke.run_smoke(self.factory())
        self.assertFalse(report["ok"])
        self.assertEqual(report["required_failed"], ["tool_roundtrip"])
        probe = report["probes"][0]
        self.assertEqual(probe["detail"]["tool_calls"], 0)
        self.assertTrue(all(p["ok"] for p in report["probes"][1:] if p["required"]))

    def test_server_errors_are_recorded_with_status_type_and_code_only(self):
        self.fake.enqueue(Fault.error(402, "SECRET-PROMPT-ECHO should not matter", type="billing_error", code="credits"))
        report = live_smoke.run_smoke(self.factory())
        probe = report["probes"][0]
        self.assertEqual((probe["ok"], probe["error"]["status"], probe["error"]["error_type"], probe["error"]["code"]),
                         (False, 402, "billing_error", "credits"))
        self.assertEqual(probe["error"]["retryable"], False)

    def test_secrets_in_error_fields_are_scrubbed_from_the_report(self):
        key = "sk-smoke-scrub-0123456789"
        secrets.register(key)
        err = ModelError(f"denied for {key}", status=401, error_type=f"type-{key}", code=f"code-{key}")
        report = live_smoke.run_smoke(lambda: ScriptedClient([err] * 20), probe_wonder=False)
        self.assertNotIn(key, json.dumps(report))
        self.assertEqual(report["probes"][0]["error"]["status"], 401)

    def test_wonder_not_served_is_informational(self):
        self.fake.models = {SPARK}
        report = live_smoke.run_smoke(self.factory())
        wonder = report["probes"][-1]
        self.assertEqual((wonder["name"], wonder["ok"], wonder["required"], wonder["error"]["status"]), ("wonder_chat", False, False, 404))
        self.assertEqual((report["ok"], report["wonder_chat_completions"]), (True, "unsupported"))

    def test_wonder_probe_can_be_skipped(self):
        report = live_smoke.run_smoke(self.factory(), probe_wonder=False)
        self.assertEqual((report["wonder_chat_completions"], "wonder_chat" in [p["name"] for p in report["probes"]]), ("skipped", False))

    def test_factory_is_asked_for_the_default_model_then_wonder(self):
        asked: list[str | None] = []
        inner = self.factory()

        def factory(model=None):
            asked.append(model)
            return inner(model)
        live_smoke.run_smoke(factory)
        self.assertEqual(asked, [None, WONDER])

    def test_zero_argument_factories_work_and_skip_wonder(self):
        client = make_client(self.fake)
        report = live_smoke.run_smoke(lambda: client)
        self.assertTrue(report["ok"])
        self.assertEqual(report["wonder_chat_completions"], "skipped")

    def test_a_client_that_cannot_do_the_probes_yields_a_failed_report_not_an_exception(self):
        report = live_smoke.run_smoke(lambda: ScriptedClient([text_response("only one reply")]))
        self.assertFalse(report["ok"])
        kinds = {p["name"]: (p["error"] or {}).get("kind") for p in report["probes"]}
        self.assertEqual(kinds["strict_schema"], "ScriptExhausted")
        rejected = next(p for p in report["probes"] if p["name"] == "rejected_param")
        self.assertFalse(rejected["ok"])

    def test_building_the_factory_fails_early_without_a_key_or_with_an_unsafe_url(self):
        with self.assertRaises(ModelConfigError):
            live_smoke.build_client_factory(env={})
        with self.assertRaises(ModelConfigError):
            live_smoke.build_client_factory(base_url="http://example.com/v1", env={"MODEL_API_KEY": "key-0123456789"})

    def test_factory_precedence_is_argument_then_environment_then_default(self):
        env = {"MODEL_API_KEY": "key-0123456789", "NIGHTSHIFT_MODEL": "muse-spark-1.2", "NIGHTSHIFT_BASE_URL": "http://127.0.0.1:5/v1"}
        self.assertEqual(live_smoke.build_client_factory(env=env)().model, "muse-spark-1.2")
        self.assertEqual(live_smoke.build_client_factory(model=SPARK, env=env)().model, SPARK)
        self.assertEqual(live_smoke.build_client_factory(env=env)().base_url, "http://127.0.0.1:5/v1")
        self.assertEqual(live_smoke.build_client_factory(env={"MODEL_API_KEY": "key-0123456789"})().base_url, "https://api.meta.ai/v1")
        self.assertEqual(live_smoke.build_client_factory(env=env)(WONDER).model, WONDER)


class SmokeCliTests(ServerCase):
    @staticmethod
    def load_script():
        spec = importlib.util.spec_from_file_location("live_smoke_script", ROOT / "scripts" / "live_smoke.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_cli_writes_the_report_and_exits_zero(self):
        script = self.load_script()
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            code = script.main(["--base-url", self.fake.base_url, "--model", "muse-spark-1.2", "--out", str(path)],
                               env={"MODEL_API_KEY": self.fake.api_key}, stdout=out, stderr=err)
            report = json.loads(path.read_text())
        self.assertEqual(code, 0, out.getvalue())
        self.assertTrue(report["ok"])
        self.assertEqual(report["models"]["requested"], "muse-spark-1.2")
        self.assertEqual(self.fake.requests[0].body["model"], "muse-spark-1.2")
        self.assertIn("overall: OK", out.getvalue())
        self.assertIn("PASS  tool_roundtrip", out.getvalue())
        self.assertNotIn(self.fake.api_key, out.getvalue() + err.getvalue() + json.dumps(report))

    def test_cli_exit_codes_for_failure_and_setup_errors(self):
        script = self.load_script()
        self.fake.enqueue(Reply.text("no tool call"))
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            code = script.main(["--base-url", self.fake.base_url, "--out", str(Path(tmp) / "p.json"), "--no-wonder"],
                               env={"MODEL_API_KEY": self.fake.api_key}, stdout=out, stderr=err)
            self.assertTrue((Path(tmp) / "p.json").exists())
        self.assertEqual(code, 1)
        self.assertIn("FAIL  tool_roundtrip", out.getvalue())
        self.assertNotIn("wonder_chat", out.getvalue())
        code = script.main(["--base-url", self.fake.base_url], env={}, stdout=io.StringIO(), stderr=err)
        self.assertEqual(code, 2)
        self.assertIn("API key", err.getvalue())

    def test_cli_defaults(self):
        args = self.load_script().build_parser().parse_args([])
        self.assertEqual((args.out, args.model, args.base_url, args.no_wonder), ("demo/live_probe.json", None, None, False))


# ==================================================================================================
# regressions found by adversarial review
# ==================================================================================================
class ReviewRegressionTests(ServerCase):
    def stub_client(self, *results, **kw):
        transport = StubTransport(*results)
        return MetaChatClient(api_key="k-review-0123456789", transport=transport, sleep=lambda s: None, **kw), transport

    # ---- secrets ------------------------------------------------------------------------------------
    def test_the_key_is_removed_before_an_error_message_is_capped(self):
        # A cap that fell in the middle of the key used to leave its first characters in the message.
        key = self.fake.api_key
        pad = "x" * 590
        for fault in (Fault(400, pad + key + " tail"), Fault(400, {"error": {"message": pad + key, "type": "t"}}),
                      Fault(400, {"error": {"message": "y" * 595 + "Bearer " + key, "type": "t"}})):
            self.fake.enqueue(fault)
            with self.subTest(body=str(fault.body)[-20:]), self.assertRaises(ModelError) as ctx:
                make_client(self.fake).chat(user())
            text = str(ctx.exception)
            self.assertNotIn(key[:6], text.replace("Bearer", ""))
            self.assertNotIn(key[:6], text)

    def test_a_200_error_body_is_scrubbed_before_it_is_capped(self):
        key = "k-review-0123456789"
        body = json.dumps({"error": {"message": "z" * 595 + key, "type": "t"}}).encode()
        client, _ = self.stub_client((200, {}, body))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertNotIn(key[:6], str(ctx.exception))

    def test_model_errors_raised_by_a_transport_or_a_stream_are_scrubbed(self):
        key = "k-review-0123456789"
        client, _ = self.stub_client(ModelError(f"proxy said {key}", error_type=f"t-{key}", code=f"c-{key}"))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        err = ctx.exception
        self.assertNotIn(key, f"{err} {err.error_type} {err.code} {err.args}")
        client, _ = self.stub_client((200, {}, sse({"error": {"message": f"echo {key}", "type": "oops"}})))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user(), stream=True)
        self.assertNotIn(key, str(ctx.exception))
        self.assertIn("oops", str(ctx.exception))

    def test_nightshift_errors_from_a_transport_are_not_turned_into_network_errors(self):
        from nightshift.types import BudgetExceeded, SimulatedCrash
        for exc in (BudgetExceeded("out of money", kind="llm"), SimulatedCrash("boom")):
            client, _ = self.stub_client(exc)
            with self.subTest(exc=type(exc).__name__), self.assertRaises(type(exc)):
                client.chat(user())

    # ---- malformed replies never escape as AttributeError/TypeError ---------------------------------------
    def test_replies_without_usage_and_with_odd_shapes_still_give_a_response_or_a_model_error(self):
        bodies = [{"choices": [{"message": "just a string"}]},
                  {"choices": [{"message": {"content": None, "tool_calls": [{"id": "a", "function": "oops"}]}}]},
                  {"choices": [{"message": {"content": None, "tool_calls": "abc"}}]},
                  {"choices": ["x"]},
                  {"choices": [{"message": {"content": ["not", "text"], "tool_calls": [None, 3]}}]}]
        for body in bodies:
            client, _ = self.stub_client((200, {}, json.dumps(body).encode()))
            with self.subTest(body=body):
                try:
                    resp = client.chat(user())
                except ModelError:
                    continue
                self.assertTrue(resp.raw["_nightshift"]["usage_estimated"])
                self.assertGreaterEqual(resp.usage.prompt_tokens, 1)

    # ---- tool-call ids ---------------------------------------------------------------------------------------
    def test_duplicate_or_missing_tool_call_ids_from_the_server_are_made_unique_so_the_history_can_be_replayed(self):
        self.assertEqual([c.id for c in base.tool_calls_from_wire([
            {"id": "a", "function": {"name": "x", "arguments": "{}"}}, {"id": "a", "function": {"name": "y", "arguments": "{}"}},
            {"function": {"name": "z", "arguments": "{}"}}, {"id": "a_1", "function": {"name": "w", "arguments": "{}"}}])],
            ["a", "a_1", "call_2", "a_1_1"])
        self.fake.enqueue(Reply.tools(("add", {"a": 1}, "dup"), ("add", {"a": 2}, "dup")))
        client = make_client(self.fake)
        first = client.chat(user(), tools=[ADD])
        ids = [c.id for c in first.message.tool_calls]
        self.assertEqual(len(set(ids)), 2)
        history = user() + [first.message] + [ChatMessage("tool", "r", tool_call_id=i) for i in ids]
        self.assertEqual(client.chat(history, tools=[ADD]).text, "ok")      # the fake enforces id order and uniqueness of results
        sent = self.fake.last_request.body["messages"]
        self.assertEqual([m["tool_call_id"] for m in sent if m["role"] == "tool"], ids)

    # ---- arguments ---------------------------------------------------------------------------------------------
    def test_per_call_timeout_must_be_a_positive_finite_number(self):
        for bad in (0, -1, float("nan"), float("inf"), True, "5"):
            with self.subTest(timeout=bad), self.assertRaises(ValueError):
                make_client(self.fake).chat(user(), timeout=bad)
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(make_client(self.fake).chat(user(), timeout=5).text, "ok")

    def test_from_env_with_a_custom_key_variable_never_falls_back_to_the_meta_key(self):
        env = {"MODEL_API_KEY": "meta-key-0123456789", "PARTNER_KEY": "partner-key-0123456789"}
        c = MetaChatClient.from_env(env=env, api_key_env="PARTNER_KEY", base_url="https://partner.example/v1")
        self.assertEqual(c._api_key, "partner-key-0123456789")
        with self.assertRaisesRegex(ModelConfigError, "PARTNER_KEY"):
            MetaChatClient.from_env(env={"MODEL_API_KEY": "meta-key-0123456789"}, api_key_env="PARTNER_KEY",
                                    base_url="https://partner.example/v1")
        c = MetaChatClient.from_env(env=env, api_key_env="MODEL_API_KEY")
        self.assertEqual(c._api_key, "meta-key-0123456789")

    # ---- 504 -> stream ---------------------------------------------------------------------------------------------
    def test_after_a_504_every_later_attempt_of_the_call_is_streamed(self):
        # 504 on the plain request, then the stream attempt is rejected for a named optional parameter. The retry
        # must stay on the stream instead of repeating the request the gateway already timed out.
        self.fake.enqueue(Fault.gateway_timeout(), Fault.bad_request("Unsupported value for reasoning_effort", param="reasoning_effort"))
        client = make_client(self.fake)
        resp = client.chat(user(), reasoning_effort="high")
        self.assertEqual([r.stream for r in self.fake.requests], [False, True, True])
        self.assertTrue(resp.raw["_nightshift"]["stream_fallback"])
        self.assertEqual(client.disabled_params, {"reasoning_effort"})

    def test_a_fallback_model_after_a_504_is_also_streamed(self):
        self.fake.models = {WONDER}
        self.fake.enqueue(Fault.gateway_timeout())
        client = make_client(self.fake, fallback_model=SPARK)
        self.fake.models = {SPARK}
        resp = client.chat(user())
        self.assertEqual(resp.model, SPARK)
        self.assertEqual([r.stream for r in self.fake.requests], [False, True])


class FakeServerRobustnessTests(unittest.TestCase):
    def test_many_simultaneous_clients_are_all_served(self):
        # The default listen backlog of 5 reset connections when parallel lanes called at once.
        with FakeMetaServer() as fake:
            self.assertGreaterEqual(fake._server.request_queue_size, 64)
            barrier, errors = threading.Barrier(48), []

            def worker():
                client = MetaChatClient(base_url=fake.base_url, api_key=fake.api_key, sleep=lambda s: None)
                barrier.wait(timeout=10)
                try:
                    client.chat(user())
                except Exception as exc:  # noqa: BLE001
                    errors.append(repr(exc))

            threads = [threading.Thread(target=worker) for _ in range(48)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
            self.assertEqual(errors, [])
            self.assertEqual(len(fake.requests), 48)


class VerifierRegressionTests(unittest.TestCase):
    """Defects found when the sticky fallback and the recorded effort were attacked."""

    MISSING = (404, {}, json.dumps({"error": {"message": "The model does not exist", "type": "not_found_error",
                                              "code": "model_not_found", "param": "model"}}).encode())

    def test_the_fallback_reason_is_not_lost_when_the_call_that_found_out_could_not_be_answered(self):
        transport = StubTransport(self.MISSING, (503, {}, b"{}"), ok_json(model=SPARK), ok_json(model=SPARK))
        client = MetaChatClient(WONDER, api_key="k-injected-0123456789", fallback_model=SPARK, max_retries=0,
                                transport=transport, sleep=lambda s: None)
        with self.assertRaises(ModelError):                       # primary 404, then the fallback is down too
            client.chat(user())
        self.assertIsNotNone(client.primary_unavailable)
        second, third = client.chat(user()), client.chat(user())
        self.assertIn("model_not_found", second.raw["_nightshift"]["fallback_reason"])      # reported on the first answer
        self.assertNotIn("fallback_reason", third.raw["_nightshift"])                       # and only once
        self.assertEqual([json.loads(c["body"])["model"] for c in transport.calls], [WONDER, SPARK, SPARK, SPARK])

    def test_the_recorded_effort_is_the_one_on_the_wire_even_if_another_call_disables_it_meanwhile(self):
        sent_a, sent_b = threading.Event(), threading.Event()
        calls: list[dict] = []
        lock = threading.Lock()
        holder: dict = {}

        def transport(method, url, headers, body, timeout, stream):
            parsed = json.loads(body)
            with lock:
                idx = len(calls)
                calls.append(parsed)
            if idx == 0:                                          # call A: rejected, and it names reasoning_effort
                sent_a.set()
                sent_b.wait(5)
                return 400, {}, json.dumps({"error": {"message": "unknown parameter reasoning_effort",
                                                      "type": "invalid_request_error", "param": "reasoning_effort"}}).encode()
            if idx == 1:                                          # call B: already sent WITH the effort; answered after A disabled it
                sent_b.set()
                for _ in range(500):
                    if "reasoning_effort" in holder["client"].disabled_params:
                        break
                    threading.Event().wait(0.01)
                return ok_json()
            return ok_json()                                      # A's retry without the effort

        client = holder["client"] = MetaChatClient(api_key="k-injected-0123456789", transport=transport, sleep=lambda s: None)
        out: dict = {}
        ta = threading.Thread(target=lambda: out.setdefault("a", client.chat(user(), reasoning_effort="high")))
        ta.start()
        sent_a.wait(5)
        tb = threading.Thread(target=lambda: out.setdefault("b", client.chat(user(), reasoning_effort="high")))
        tb.start()
        ta.join(15)
        tb.join(15)
        self.assertEqual(calls[1].get("reasoning_effort"), "high")
        self.assertEqual(out["b"].raw["_nightshift"]["reasoning_effort"], "high")           # it WAS sent
        self.assertIsNone(out["a"].raw["_nightshift"]["reasoning_effort"])                  # A's final request had none


class PricingSnapshotTests(unittest.TestCase):
    def test_a_dated_snapshot_of_a_known_model_is_charged_at_that_models_rate(self):
        u = Usage(prompt_tokens=1000, completion_tokens=1000)
        for model, base_model in (("muse-wonder-1.0-2026-10-20", "muse-wonder-1.0"), ("muse-wonder-1.0@preview", "muse-wonder-1.0"),
                                  ("muse-spark-1.3-contributor-2026", "muse-spark-1.3-contributor"),
                                  ("muse-spark-1.2-latest", "muse-spark-1.2")):
            with self.subTest(model=model):
                self.assertEqual(pricing.usd_for(model, u), pricing.usd_for(base_model, u))
        self.assertGreater(pricing.usd_for("muse-wonder-1.0-x", u), pricing.usd_for("muse-spark-1.3", u))

    def test_lookalike_and_unknown_ids_still_use_the_spark_rate(self):
        u = Usage(prompt_tokens=1000, completion_tokens=1000)
        for model in ("muse-wonder-1.00", "muse-wonder-1.0-", "muse-wonder-2.0", "other-model", "", None):
            with self.subTest(model=model):
                self.assertEqual(pricing.usd_for(model, u), pricing.usd_for("muse-spark-1.3", u))


# ==================================================================================================
# API conformance (audit findings checked against https://dev.meta.ai/docs)
# ==================================================================================================
def sse_events(data: bytes) -> list[dict]:
    """The JSON payloads of a raw SSE body (comments and [DONE] dropped)."""
    out = []
    for line in data.decode("utf-8").splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            out.append(json.loads(line[6:]))
    return out


class StreamUsageTests(ServerCase):
    """/docs/cookbook/streaming-responses: a stream omits usage unless stream_options.include_usage is set."""

    def test_a_forced_stream_asks_for_usage_and_gets_the_real_counts(self):
        client = make_client(self.fake)
        self.fake.enqueue(Reply.text("hi", prompt_tokens=10, completion_tokens=350, reasoning_tokens=300))
        resp = client.chat(user(), stream=True)
        self.assertEqual(self.fake.last_request.body["stream_options"], {"include_usage": True})
        meta = resp.raw["_nightshift"]
        self.assertFalse(meta["usage_estimated"])
        self.assertEqual((resp.usage.prompt_tokens, resp.usage.completion_tokens), (10, 350))   # hidden reasoning is counted
        self.assertEqual(meta["reasoning_tokens"], 300)

    def test_the_504_then_stream_fallback_asks_for_usage_too(self):
        client = make_client(self.fake)
        self.fake.non_stream_status = 504
        self.fake.enqueue(Reply.text("hi", prompt_tokens=10, completion_tokens=350, reasoning_tokens=300))
        resp = client.chat(user())
        first, second = self.fake.requests
        self.assertNotIn("stream_options", first.body)
        self.assertEqual((second.body["stream"], second.body["stream_options"]), (True, {"include_usage": True}))
        self.assertEqual((resp.raw["_nightshift"]["usage_estimated"], resp.usage.completion_tokens), (False, 350))

    def test_a_request_that_is_not_streamed_does_not_send_stream_options(self):
        make_client(self.fake).chat(user())
        self.assertNotIn("stream_options", self.fake.last_request.body)

    def test_a_host_that_rejects_stream_options_gets_it_stripped_and_an_estimate_flagged(self):
        with FakeMetaServer(unsupported_params=("stream_options",)) as fake:
            client = make_client(fake)
            resp = client.chat(user(), stream=True)
            self.assertEqual([("stream_options" in r.body, r.status) for r in fake.requests], [(True, 400), (False, 200)])
            self.assertIn("stream_options", client.disabled_params)
            self.assertTrue(resp.raw["_nightshift"]["usage_estimated"])
            client.chat(user(), stream=True)
            self.assertNotIn("stream_options", fake.last_request.body)          # remembered, like any disabled parameter

    def test_the_fake_streams_usage_only_when_asked_and_in_a_final_chunk_with_no_choices(self):
        body = {"model": SPARK, "messages": [{"role": "user", "content": "hi"}], "stream": True}
        plain = sse_events(raw_post(self.fake, body)[2])
        self.assertFalse(any("usage" in c for c in plain))
        asked = sse_events(raw_post(self.fake, {**body, "stream_options": {"include_usage": True}})[2])
        self.assertEqual([c["choices"] for c in asked if "usage" in c], [[]])
        self.assertIn("usage", asked[-1])
        self.assertEqual(raw_post(self.fake, {**body, "stream_options": {"include_usage": "yes"}})[0], 400)

    def test_without_include_usage_the_old_estimate_is_what_the_budget_would_have_seen(self):
        # The regression in one picture: a server that follows the docs sends no usage unless asked, so a client that never
        # asked saw only visible text (here 1 token) while 300 hidden reasoning tokens were spent.
        body = {"model": SPARK, "messages": [{"role": "user", "content": "hi"}], "stream": True}
        self.fake.enqueue(Reply.text("ok", reasoning_tokens=300))
        chunks = sse_events(raw_post(self.fake, body)[2])
        self.assertFalse(any("usage" in c for c in chunks))


class CompletionCapTests(ServerCase):
    """The docs: a request above the model's configured maximum is an HTTP 400; no maximum is stated."""

    def test_retry_cap_doubles_but_never_beyond_the_ceiling_or_below_the_original(self):
        from nightshift.agents import prompts
        self.assertEqual([prompts.retry_cap(c) for c in (1000, 6000, 8000, 9000, 16000, 20000)],
                         [2000, 12000, 16000, 16000, 16000, 20000])
        self.assertEqual((prompts.retry_cap(None), prompts.retry_cap(0)), (None, None))
        self.assertEqual((prompts.retry_cap(6000, 8000), prompts.retry_cap(6000, 32000), prompts.retry_cap(20000, 8000)),
                         (8000, 12000, 20000))
        self.assertEqual((prompts.retry_cap(6000, 0), prompts.retry_cap(9000, True), prompts.retry_cap(9000, "x")),
                         (12000, 16000, 16000))
        self.assertEqual(prompts.RETRY_CAP_CEILING, 16000)

    def test_a_400_naming_the_cap_is_retried_once_at_the_limit_the_message_states(self):
        self.fake.max_completion_tokens_ceiling = 8000
        client = make_client(self.fake)
        resp = client.chat(user(), max_completion_tokens=12000)
        self.assertEqual([(r.body["max_completion_tokens"], r.status) for r in self.fake.requests], [(12000, 400), (8000, 200)])
        self.assertEqual(resp.raw["_nightshift"]["completion_cap_reduced"], {"from": 12000, "to": 8000})
        client.chat(user(), max_completion_tokens=12000)                        # remembered: no second doomed request
        self.assertEqual([r.body["max_completion_tokens"] for r in self.fake.requests][2:], [8000])

    def test_without_a_stated_limit_the_retry_uses_the_largest_value_seen_accepted(self):
        self.fake.max_completion_tokens_ceiling = 8000
        self.fake.ceiling_message_states_limit = False
        client = make_client(self.fake)
        client.chat(user(), max_completion_tokens=5000)
        resp = client.chat(user(), max_completion_tokens=16000)
        self.assertEqual([(r.body["max_completion_tokens"], r.status) for r in self.fake.requests],
                         [(5000, 200), (16000, 400), (5000, 200)])
        self.assertEqual(resp.raw["_nightshift"]["completion_cap_reduced"], {"from": 16000, "to": 5000})

    def test_with_nothing_known_the_retry_halves_and_a_second_rejection_is_raised(self):
        self.fake.max_completion_tokens_ceiling = 1000
        self.fake.ceiling_message_states_limit = False
        client = make_client(self.fake)
        with self.assertRaises(ModelError) as ctx:
            client.chat(user(), max_completion_tokens=16000)
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual([r.body["max_completion_tokens"] for r in self.fake.requests], [16000, 8000])    # exactly ONE retry

    def test_a_lower_cap_that_does_not_help_is_not_remembered(self):
        # a host that dislikes the parameter itself (not its size) must not leave the client with a bogus, lower cap
        client = make_client(self.fake)
        self.fake.enqueue(Fault.bad_request("max_completion_tokens is not supported here", "max_completion_tokens"),
                          Fault.bad_request("max_completion_tokens is not supported here", "max_completion_tokens"))
        with self.assertRaises(ModelError):
            client.chat(user(), max_completion_tokens=12000)
        client.chat(user(), max_completion_tokens=12000)
        self.assertEqual([r.body["max_completion_tokens"] for r in self.fake.requests], [12000, 6000, 12000])

    def test_a_400_that_is_not_about_the_cap_is_not_retried(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.bad_request("messages are malformed", "messages"))
        with self.assertRaises(ModelError):
            client.chat(user(), max_completion_tokens=12000)
        self.assertEqual(len(self.fake.requests), 1)

    def test_a_configured_ceiling_clamps_every_request(self):
        client = make_client(self.fake, completion_cap_ceiling=5000)
        resp = client.chat(user(), max_completion_tokens=12000)
        client.chat(user(), max_completion_tokens=3000)
        client.chat(user())
        self.assertEqual([r.body.get("max_completion_tokens") for r in self.fake.requests], [5000, 3000, None])
        self.assertEqual(resp.raw["_nightshift"]["max_completion_tokens_sent"], 5000)
        for bad in (0, -1, True, 2.5, "9"):
            with self.subTest(bad=bad), self.assertRaises(ModelConfigError):
                MetaChatClient(api_key="k-0123456789", completion_cap_ceiling=bad)

    def test_the_learned_limit_belongs_to_the_model_that_refused(self):
        self.fake.max_completion_tokens_ceiling = {WONDER: 8000}
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
        client.chat(user(), max_completion_tokens=12000)
        self.assertEqual([(r.body["model"], r.body["max_completion_tokens"]) for r in self.fake.requests],
                         [(WONDER, 12000), (WONDER, 8000)])
        self.fake.reset()
        self.fake.models = {SPARK}
        client = make_client(self.fake, model=WONDER, fallback_model=SPARK)
        client.chat(user(), max_completion_tokens=12000)
        self.assertEqual(self.fake.requests[-1].body["max_completion_tokens"], 12000)   # the fallback was never limited

    def test_the_cap_retry_works_on_a_stream_and_keeps_stream_options(self):
        self.fake.max_completion_tokens_ceiling = 8000
        client = make_client(self.fake)
        resp = client.chat(user(), max_completion_tokens=12000, stream=True)
        self.assertEqual([r.status for r in self.fake.requests], [400, 200])
        self.assertEqual(self.fake.last_request.body["stream_options"], {"include_usage": True})
        self.assertTrue(resp.raw["_nightshift"]["streamed"])

    def test_the_routing_config_carries_the_ceiling_to_the_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "n.toml"
            path.write_text("[models.default]\ncompletion_cap_ceiling = 9000\n[roles.judge]\ncompletion_cap_ceiling = 4000\n")
            routing = config.load_routing(path)
        self.assertEqual((routing["judge"].completion_cap_ceiling, routing["reviewer"].completion_cap_ceiling), (4000, 9000))
        self.assertIsNone(config.load_routing(None)["judge"].completion_cap_ceiling)
        client = config.client_for("judge", routing, api_key="k-0123456789")
        self.assertEqual(client.completion_cap_ceiling, 4000)


class RetryClassificationTests(ServerCase):
    def test_502_is_retried_like_429_500_and_503(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.server_error(502), Fault.server_error(502), Reply.text("fine"))
        resp = client.chat(user())
        self.assertEqual((resp.text, [r.status for r in self.fake.requests], resp.raw["_nightshift"]["retries"]), ("fine", [502, 502, 200], 2))
        self.assertEqual(client.sleeps.calls, [0.75, 1.25])

    def test_502_that_never_clears_raises_a_retryable_error(self):
        client = make_client(self.fake, max_retries=1)
        self.fake.enqueue(Fault.server_error(502), Fault.server_error(502))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user())
        self.assertEqual((ctx.exception.status, ctx.exception.retryable), (502, True))
        self.assertEqual(len(self.fake.requests), 2)

    def test_504_keeps_its_own_stream_path_and_is_not_in_the_retry_set(self):
        from nightshift.models import meta
        self.assertEqual(meta.RETRYABLE_STATUSES, (429, 500, 502, 503))
        client = make_client(self.fake)
        self.fake.non_stream_status = 504
        client.chat(user())
        self.assertEqual([r.status for r in self.fake.requests], [504, 200])
        self.assertEqual(client.sleeps.calls, [])

    def test_the_fake_504_body_uses_the_documented_code(self):
        self.fake.non_stream_status = 504
        status, _, data = raw_post(self.fake, {"model": SPARK, "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual((status, json.loads(data)["error"]["code"]), (504, "gateway_timeout"))
        self.fake.reset()
        self.fake.enqueue(Fault.gateway_timeout())
        _, _, data = raw_post(self.fake, {"model": SPARK, "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(json.loads(data)["error"]["code"], "gateway_timeout")


class InStreamErrorTests(ServerCase):
    """/docs/error-handling: a mid-stream server_shutting_down is handled by re-sending the whole request."""

    def test_a_shutting_down_error_event_re_sends_the_request(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.stream_error("server_shutting_down"), Reply.text("second try"))
        resp = client.chat(user(), stream=True)
        self.assertEqual((resp.text, len(self.fake.requests), resp.raw["_nightshift"]["retries"]), ("second try", 2, 1))
        self.assertEqual(len(client.sleeps.calls), 1)

    def test_service_overloaded_and_backend_unavailable_are_transient_too(self):
        for code in ("service_overloaded", "backend_unavailable"):
            with self.subTest(code=code):
                self.fake.reset()
                client = make_client(self.fake)
                self.fake.enqueue(Fault.stream_error(code), Reply.text("ok again"))
                self.assertEqual(client.chat(user(), stream=True).text, "ok again")

    def test_other_in_stream_errors_are_raised_without_a_retry(self):
        client = make_client(self.fake)
        self.fake.enqueue(Fault.stream_error("content_policy_violation", "blocked"), Reply.text("never"))
        with self.assertRaises(ModelError) as ctx:
            client.chat(user(), stream=True)
        self.assertEqual((ctx.exception.code, ctx.exception.retryable, len(self.fake.requests)), ("content_policy_violation", False, 1))

    def test_a_stream_that_keeps_shutting_down_stops_at_max_retries(self):
        client = make_client(self.fake, max_retries=2)
        self.fake.enqueue(*[Fault.stream_error("server_shutting_down")] * 5)
        with self.assertRaises(ModelError) as ctx:
            client.chat(user(), stream=True)
        self.assertEqual((ctx.exception.retryable, len(self.fake.requests)), (True, 3))

    def test_every_documented_and_defensive_error_shape_is_understood(self):
        shapes = {
            "openai chunk": {"error": {"message": "going down", "type": "server_error", "code": "server_shutting_down"}},
            "responses error event": {"type": "error", "code": "server_shutting_down", "message": "going down"},
            "response.failed": {"type": "response.failed", "response": {"status": "failed", "error": {
                "code": "service_overloaded", "message": "busy"}}},
            "code carried in type": {"error": {"message": "busy", "type": "service_overloaded"}},
        }
        for name, chunk in shapes.items():
            with self.subTest(name):
                with self.assertRaises(ModelError) as ctx:
                    base.StreamAssembler().feed(chunk)
                self.assertTrue(ctx.exception.retryable)
        for name, chunk in {"unknown code": {"error": {"message": "x", "code": "other"}},
                            "bare error event": {"type": "error", "message": "x"},
                            "failed without detail": {"type": "response.failed"}}.items():
            with self.subTest(name):
                with self.assertRaises(ModelError) as ctx:
                    base.StreamAssembler().feed(chunk)
                self.assertFalse(ctx.exception.retryable)

    def test_a_usage_only_chunk_after_the_finish_chunk_does_not_end_or_break_the_reply(self):
        asm = base.StreamAssembler()
        asm.feed({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
        asm.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        asm.feed({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 9}})
        out = asm.to_response()
        self.assertEqual((out["choices"][0]["message"]["content"], out["usage"]["completion_tokens"]), ("hi", 9))


class FakeFidelityTests(ServerCase):
    BODY = {"model": SPARK, "messages": [{"role": "user", "content": "hi"}]}

    def post(self, **extra):
        status, headers, data = raw_post(self.fake, {**self.BODY, **extra})
        return status, json.loads(data), headers

    def test_max_effort_follows_the_docs_for_every_model(self):
        for model, expected in ((WONDER, 200), (SPARK, 200), ("muse-spark-1.3-contributor", 400), ("muse-spark-1.2", 400)):
            with self.subTest(model=model):
                status, body, _ = self.post(model=model, reasoning_effort="max")
                self.assertEqual(status, expected)
                if expected == 400:
                    self.assertEqual(body["error"]["param"], "reasoning_effort")
        self.assertEqual(self.post(reasoning_effort="none")[0], 400)

    def test_responses_carry_refusal_null_and_empty_reasoning_content(self):
        _, body, _ = self.post()
        message = body["choices"][0]["message"]
        self.assertIsNone(message["refusal"])
        self.assertEqual(message["reasoning_content"], "")
        self.assertEqual(body["usage"]["completion_tokens_details"], {"reasoning_tokens": 0})
        self.assertEqual(body["usage"]["prompt_tokens_details"], {"cached_tokens": 0})

    def test_reasoning_token_reporting_is_configurable_and_part_of_completion_tokens(self):
        self.fake.enqueue(Reply.text("abcd", reasoning_tokens=40))
        usage = self.post()[1]["usage"]
        self.assertEqual((usage["completion_tokens"], usage["completion_tokens_details"]["reasoning_tokens"]), (41, 40))
        self.fake.report_reasoning_tokens = False
        self.assertNotIn("completion_tokens_details", self.post()[1]["usage"])

    def test_simulated_reasoning_can_eat_the_whole_completion_cap(self):
        self.fake.simulate_reasoning = 1500
        status, body, _ = self.post(max_completion_tokens=1000)
        choice = body["choices"][0]
        self.assertEqual((status, choice["finish_reason"], choice["message"]["content"]), (200, "length", ""))
        self.assertEqual(body["usage"]["completion_tokens"], 1000)
        self.assertEqual(body["usage"]["completion_tokens_details"]["reasoning_tokens"], 1000)
        enough = self.post(max_completion_tokens=2000)[1]
        self.assertEqual((enough["choices"][0]["finish_reason"], enough["choices"][0]["message"]["content"]), ("stop", "ok"))
        self.assertEqual(enough["usage"]["completion_tokens_details"]["reasoning_tokens"], 1500)
        self.assertGreater(enough["usage"]["completion_tokens"], 1500)

    def test_simulated_reasoning_per_effort_and_a_cut_off_answer(self):
        self.fake.simulate_reasoning = {"minimal": 10, "high": 900, "*": 100}
        self.fake.enqueue(Reply.text("x" * 400), Reply.text("x" * 400))
        short = self.post(max_completion_tokens=105, reasoning_effort="minimal")[1]["choices"][0]
        self.assertEqual((short["finish_reason"], len(short["message"]["content"])), ("length", 380))
        eaten = self.post(max_completion_tokens=500, reasoning_effort="high")[1]["choices"][0]
        self.assertEqual((eaten["finish_reason"], eaten["message"]["content"]), ("length", ""))

    def test_the_client_sees_a_reasoning_exhausted_reply_as_length(self):
        self.fake.simulate_reasoning = 1500
        resp = make_client(self.fake).chat(user(), max_completion_tokens=1000)
        self.assertEqual((resp.finish_reason, resp.text, resp.usage.completion_tokens), ("length", "", 1000))
        self.assertEqual(resp.raw["_nightshift"]["reasoning_tokens"], 1000)

    def test_ratelimit_headers_are_optional_and_reach_the_response_meta(self):
        self.assertFalse([h for h in self.post()[2] if h.lower().startswith("x-ratelimit")])
        self.fake.ratelimit_headers = True
        headers = {k.lower(): v for k, v in self.post()[2].items()}
        for name in ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests"):
            self.assertIn(name, headers)
        resp = make_client(self.fake).chat(user())
        self.assertEqual(set(resp.raw["_nightshift"]["ratelimit"]), {
            "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests"})

    def test_content_may_be_a_string_or_an_array_of_parts(self):
        parts = [{"type": "text", "text": "hello"}]
        for role in ("system", "developer", "user"):
            with self.subTest(role=role):
                self.assertEqual(self.post(messages=[{"role": role, "content": parts}, {"role": "user", "content": parts}])[0], 200)
        for bad in (5, [], [{"type": "text"}], [{"type": "image_url", "image_url": {}}], None):
            with self.subTest(bad=bad):
                self.assertEqual(self.post(messages=[{"role": "user", "content": bad}])[0], 400)

    def test_assistant_content_may_be_null_a_string_or_parts(self):
        call = {"id": "c1", "type": "function", "function": {"name": "add", "arguments": "{}"}}
        for content in (None, "", "thinking", [{"type": "text", "text": "t"}], [{"type": "refusal", "refusal": "no"}]):
            with self.subTest(content=content):
                messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": content, "tool_calls": [call]},
                            {"role": "tool", "tool_call_id": "c1", "content": "5"}]
                self.assertEqual(self.post(messages=messages)[0], 200)
        self.assertEqual(self.post(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": 5}])[0], 400)

    def test_max_completion_tokens_above_the_ceiling_gets_the_documented_400_envelope(self):
        self.fake.max_completion_tokens_ceiling = 8000
        status, body, _ = self.post(max_completion_tokens=8001)
        self.assertEqual(status, 400)
        self.assertEqual((body["error"]["type"], body["error"]["param"]), ("invalid_request_error", "max_completion_tokens"))
        self.assertIn("8000", body["error"]["message"])
        self.assertEqual(self.post(max_completion_tokens=8000)[0], 200)

    def test_rejected_schema_keywords_simulate_a_stricter_subset(self):
        self.fake.rejected_schema_keywords = {"minLength"}
        schema = {"type": "object", "properties": {"minLength": {"type": "string"}, "a": {"type": "string", "minLength": 1}},
                  "required": ["minLength", "a"], "additionalProperties": False}
        rf = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": schema}}
        status, body, _ = self.post(response_format=rf)
        self.assertEqual((status, "'minLength'" in body["error"]["message"]), (400, True))
        schema["properties"]["a"].pop("minLength")                 # a property NAMED minLength is not the keyword
        self.assertEqual(self.post(response_format=rf)[0], 200)
        tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {
            "q": {"type": "string", "minLength": 2}}}}}
        self.assertEqual(self.post(tools=[tool])[0], 400)


class PackageLayoutTests(unittest.TestCase):
    def test_models_init_stays_empty(self):
        self.assertEqual((ROOT / "nightshift" / "models" / "__init__.py").read_text().strip(), "")

    def test_clients_satisfy_the_model_client_protocol(self):
        self.assertIsInstance(ScriptedClient([]), ModelClient)
        self.assertIsInstance(MetaChatClient(api_key="k-0123456789"), ModelClient)
        self.assertIsInstance(text_response("x"), ChatResponse)


if __name__ == "__main__":
    unittest.main()


class VerifierRegressionClientTests(ServerCase):
    """Defects found by the adversarial verification of the API-conformance workstream."""

    def test_a_context_window_400_is_not_taken_for_the_models_output_maximum(self):
        # Remembering a cap lowered because ONE prompt was huge would clamp every later call of the client for that model.
        self.fake.enqueue(Fault.error(400, "Input plus max_completion_tokens exceeds the context window of 131072 tokens",
                                      param="max_completion_tokens", code="context_length_exceeded"))
        client = make_client(self.fake)
        with self.assertRaises(ModelError) as ctx:
            client.chat(user(), max_completion_tokens=8000)
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(self.fake.requests), 1)                        # not retried at a lower cap
        client.chat(user(), max_completion_tokens=8000)
        self.assertEqual(self.fake.requests[-1].body["max_completion_tokens"], 8000)

    def test_the_documented_configured_maximum_400_is_still_handled_when_the_context_is_mentioned(self):
        self.fake.max_completion_tokens_ceiling = 8000
        client = make_client(self.fake)
        client.chat(user(), max_completion_tokens=12000)
        self.assertEqual([r.status for r in self.fake.requests], [400, 200])

    def test_a_host_naming_only_include_usage_still_gets_stream_options_stripped(self):
        self.fake.enqueue(Fault.bad_request("Unknown field 'include_usage'"))
        client = make_client(self.fake)
        resp = client.chat(user(), stream=True)
        self.assertIn("stream_options", client.disabled_params)
        self.assertNotIn("stream_options", self.fake.requests[-1].body)
        self.assertTrue(resp.raw["_nightshift"]["usage_estimated"])

    def test_429s_absorbed_by_the_retry_loop_are_reported_in_the_response_meta(self):
        self.fake.enqueue(Fault.rate_limit(retry_after=2), Fault.rate_limit())
        meta = make_client(self.fake).chat(user()).raw["_nightshift"]
        self.assertEqual((meta["rate_limited"], meta["retry_after_seen"]), (2, [2.0]))
