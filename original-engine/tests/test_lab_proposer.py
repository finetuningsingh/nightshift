"""Tests for nightshift.lab.proposer (owner 4): parsing and retry logic with scripted clients, and the REAL MetaChatClient +
MeteredClient + BudgetGate against the repo's FakeMetaServer (there is no Meta key here).

Run: PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_proposer
"""

import json
import tempfile
import unittest
from pathlib import Path

from nightshift.agents import llm as agents_llm
from nightshift.budget import BudgetGate, MeteredClient
from nightshift.ledger import Ledger
from nightshift.lab.contracts import BudgetExhausted, EditOp, Lever, PackRef, Proposal, ProposalContext, ProposalInvalid
from nightshift.lab.proposer import (
    PROPOSAL_SCHEMA,
    ProposerUnavailable,
    UnparseableProposal,
    WonderProposalSource,
    parse_json_object,
    proposal_from_dict,
)
from nightshift.models.meta import MetaChatClient
from nightshift.models.pricing import usd_for
from nightshift.types import BudgetLimits, ChatMessage, ChatResponse, ModelError, Usage
from tests.support.fake_meta_server import FakeMetaServer, Fault, Reply
from tests.support.synthetic import FakeClock

CTX = ProposalContext("cand-001", PackRef("packs/pack.git", "b" * 40), "v0", "SYSTEM RULES", "USER FACTS")


def good_reply(**over) -> dict:
    d = {
        "hypothesis": "Researchers re-read the whole log every turn.",
        "lever": "context_notes",
        "change_summary": "Bound the notes summary.",
        "expected_effect": "Less context per turn.",
        "risk": "Loses detail.",
        "edits": [
            {"path": "notes/seed.md", "op": "search_replace", "content": None, "blocks": [{"search": "## Log", "replace": "## Log (newest first)"}]},
            {"path": "notes/new.md", "op": "replace_file", "content": "fresh\n", "blocks": None},
        ],
    }
    d.update(over)
    return d


class ParseTests(unittest.TestCase):
    def test_parse_json_object_equals_the_agents_copy_on_a_fixture_set(self):
        cases = [
            None, "", "   ", "not json", "{}", '{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go:\n{"a": {"b": [1, 2]}}\nHope that helps.',
            "[1, 2, 3]", '"just a string"', '{"a": 1', '{"a": 1} trailing {"b": 2}', 'text {"a": "x}y"} more', "}{", '{"a": NaN}', "\n\n{\"k\": \"v\"}\n",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(parse_json_object(text), agents_llm.parse_json_object(text))

    def test_proposal_from_dict_accepts_both_edit_kinds(self):
        p = proposal_from_dict(good_reply())
        self.assertIsInstance(p, Proposal)
        self.assertEqual(p.lever, Lever.CONTEXT_NOTES)
        self.assertEqual([e.op for e in p.edits], [EditOp.SEARCH_REPLACE, EditOp.REPLACE_FILE])
        self.assertEqual(p.edits[0].blocks[0].replace, "## Log (newest first)")

    def test_proposal_from_dict_rejects_each_kind_of_bad_reply(self):
        edit = good_reply()["edits"][0]
        bad = {
            "unknown lever": good_reply(lever="world_domination"),
            "empty text field": good_reply(risk="  "),
            "over-long text field": good_reply(hypothesis="x" * 5000),
            "no edits": good_reply(edits=[]),
            "edits not a list": good_reply(edits="do it"),
            "too many edits": good_reply(edits=[dict(edit, path=f"notes/{i}.md") for i in range(13)]),
            "duplicate paths": good_reply(edits=[edit, dict(edit)]),
            "traversal path": good_reply(edits=[dict(edit, path="../../etc/passwd")]),
            "dot-git path": good_reply(edits=[dict(edit, path=".git/config")]),
            "replace_file with blocks": good_reply(edits=[{"path": "a.md", "op": "replace_file", "content": "x", "blocks": [{"search": "a", "replace": "b"}]}]),
            "replace_file without content": good_reply(edits=[{"path": "a.md", "op": "replace_file", "content": None, "blocks": None}]),
            "search_replace with content": good_reply(edits=[dict(edit, content="x")]),
            "search_replace with no blocks": good_reply(edits=[dict(edit, blocks=[])]),
            "unknown op": good_reply(edits=[dict(edit, op="delete")]),
            "extra key": dict(good_reply(), confidence=0.9),
            "missing key": {k: v for k, v in good_reply().items() if k != "risk"},
            "edit with extra key": good_reply(edits=[dict(edit, mode="0777")]),
            "block missing replace": good_reply(edits=[dict(edit, blocks=[{"search": "a"}])]),
            "empty search": good_reply(edits=[dict(edit, blocks=[{"search": "", "replace": "x"}])]),
        }
        for why, obj in bad.items():
            with self.subTest(why):
                with self.assertRaises(ProposalInvalid):
                    proposal_from_dict(obj)

    def test_the_schema_is_strict_mode_shaped(self):
        from nightshift.models.base import strictify_schema

        strict = strictify_schema(PROPOSAL_SCHEMA)
        self.assertEqual(strict["required"], list(strict["properties"]))
        self.assertEqual(strict["properties"]["lever"]["enum"], [lv.value for lv in Lever])
        self.assertFalse(strict["additionalProperties"])
        item = strict["properties"]["edits"]["items"]
        self.assertEqual(set(item["required"]), {"path", "op", "content", "blocks"})


class ScriptedClient:
    """A ModelClient that replays canned replies; counts calls."""

    supports_charge_context = False

    def __init__(self, replies, model="muse-wonder-1.0"):
        self.model = model
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((list(messages), kw))
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def resp(text, *, finish="stop", prompt=1000, completion=500, model="muse-wonder-1.0", raw=None):
    return ChatResponse(ChatMessage(role="assistant", content=text), finish, Usage(prompt, completion, 0, prompt + completion), model, raw=raw or {})


class LogicTests(unittest.TestCase):
    def test_a_good_reply_becomes_a_proposal_result_with_cost(self):
        c = ScriptedClient([resp(json.dumps(good_reply()))])
        res = WonderProposalSource(c).propose(CTX)
        self.assertEqual(res.proposal.change_summary, "Bound the notes summary.")
        self.assertEqual((res.model, res.served_model, res.fallback_used), ("muse-wonder-1.0", "muse-wonder-1.0", False))
        self.assertAlmostEqual(res.usd, usd_for("muse-wonder-1.0", Usage(1000, 500, 0, 1500)))
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (1000, 500))
        self.assertEqual(len(c.calls), 1)
        msgs, kw = c.calls[0]
        self.assertEqual([m.role for m in msgs], ["system", "user"])
        self.assertEqual((msgs[0].content, msgs[1].content), ("SYSTEM RULES", "USER FACTS"))
        self.assertEqual(kw["max_completion_tokens"], 16000)
        self.assertEqual(kw["schema_name"], "pack_proposal")
        self.assertIs(kw["response_schema"] is PROPOSAL_SCHEMA or kw["response_schema"] == PROPOSAL_SCHEMA, True)

    def test_an_unusable_reply_is_retried_once_and_both_calls_are_paid_for(self):
        c = ScriptedClient([resp("I cannot do that.", prompt=2000, completion=100), resp("still prose", prompt=2000, completion=100)])
        with self.assertRaises(UnparseableProposal) as cm:
            WonderProposalSource(c).propose(CTX)
        self.assertEqual(len(c.calls), 2, "exactly one retry")
        e = cm.exception
        self.assertEqual(e.raw_response, "still prose")
        self.assertAlmostEqual(e.usd, 2 * usd_for("muse-wonder-1.0", Usage(2000, 100, 0, 2100)))
        self.assertEqual((e.prompt_tokens, e.completion_tokens), (4000, 200))
        self.assertIsInstance(e, ProposalInvalid)

    def test_a_failed_retry_after_a_paid_unreadable_reply_still_carries_the_spend(self):
        c = ScriptedClient([resp("I cannot do that.", prompt=2000, completion=100), ModelError("boom", status=503)])
        with self.assertRaises(UnparseableProposal) as cm:
            WonderProposalSource(c).propose(CTX)
        e = cm.exception
        self.assertEqual(len(c.calls), 2)
        self.assertAlmostEqual(e.usd, usd_for("muse-wonder-1.0", Usage(2000, 100, 0, 2100)))
        self.assertGreater(e.usd, 0)
        self.assertEqual(e.raw_response, "I cannot do that.")
        self.assertIn("retry failed", str(e))

    def test_budget_exhaustion_during_the_retry_still_carries_the_paid_reply(self):
        from nightshift.types import BudgetExceeded

        c = ScriptedClient([resp("I cannot do that.", prompt=2000, completion=100), BudgetExceeded("llm budget exhausted", kind="llm", limit=1.0, used=1.0)])
        with self.assertRaises(UnparseableProposal) as cm:
            WonderProposalSource(c).propose(CTX)
        e = cm.exception
        self.assertAlmostEqual(e.usd, usd_for("muse-wonder-1.0", Usage(2000, 100, 0, 2100)))
        self.assertEqual(e.raw_response, "I cannot do that.")
        self.assertIn("budget", str(e))

    def test_a_retry_that_works_returns_the_proposal_and_the_total_cost(self):
        c = ScriptedClient([resp("prose"), resp(json.dumps(good_reply()))])
        res = WonderProposalSource(c).propose(CTX)
        self.assertEqual(len(c.calls), 2)
        self.assertEqual(res.prompt_tokens, 2000)
        self.assertAlmostEqual(res.usd, 2 * usd_for("muse-wonder-1.0", Usage(1000, 500, 0, 1500)))

    def test_a_json_object_that_is_not_a_proposal_is_not_asked_again(self):
        c = ScriptedClient([resp(json.dumps(good_reply(lever="nonsense")))])
        with self.assertRaises(UnparseableProposal) as cm:
            WonderProposalSource(c).propose(CTX)
        self.assertEqual(len(c.calls), 1)
        self.assertIn("lever", str(cm.exception))
        self.assertGreater(cm.exception.usd, 0)

    def test_a_cut_off_reply_is_reported_as_such(self):
        c = ScriptedClient([resp('{"hypothesis": "x', finish="length"), resp('{"hypothesis": "y', finish="length")])
        with self.assertRaisesRegex(UnparseableProposal, "cut off"):
            WonderProposalSource(c).propose(CTX)

    def test_the_provider_answering_with_another_model_is_recorded(self):
        raw = {"_nightshift": {"fallback_used": True}}
        c = ScriptedClient([resp(json.dumps(good_reply()), model="muse-spark-1.3", raw=raw)])
        res = WonderProposalSource(c).propose(CTX)
        self.assertEqual((res.model, res.served_model, res.fallback_used), ("muse-wonder-1.0", "muse-spark-1.3", True))

    def test_provider_failure_and_budget_exhaustion_are_distinguished(self):
        with self.assertRaises(ProposerUnavailable):
            WonderProposalSource(ScriptedClient([ModelError("boom", status=503)])).propose(CTX)
        from nightshift.types import BudgetExceeded

        with self.assertRaises(BudgetExhausted) as cm:
            WonderProposalSource(ScriptedClient([BudgetExceeded("llm budget exhausted", kind="llm", limit=1.0, used=1.0)])).propose(CTX)
        self.assertEqual(cm.exception.axis, "llm")


class FakeServerTests(unittest.TestCase):
    """The real client stack: MetaChatClient -> MeteredClient (BudgetGate on a real hash-chained ledger) -> WonderProposalSource."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeMetaServer().start()
        self.addCleanup(self.fake.stop)
        self.led = Ledger.create(Path(self.tmp.name), clock=FakeClock(), fsync=False)
        self.addCleanup(self.led.close)

    def source(self, *, max_usd=5.0, fallback=None, model="muse-wonder-1.0"):
        client = MetaChatClient(model, base_url=self.fake.base_url, api_key=self.fake.api_key, fallback_model=fallback, max_retries=1, sleep=lambda s: None)
        gate = BudgetGate(BudgetLimits(max_llm_usd=max_usd), self.led, clock=lambda: 1000.0)
        self.gate = gate
        return WonderProposalSource(MeteredClient(client, gate, "wonder"))

    def test_a_structured_reply_is_parsed_and_the_call_is_charged_to_the_outer_ledger(self):
        self.fake.enqueue(Reply.json(good_reply(), prompt_tokens=3000, completion_tokens=1200))
        res = self.source().propose(CTX)
        self.assertEqual(res.proposal.lever, Lever.CONTEXT_NOTES)
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (3000, 1200))
        body = self.fake.last_request.body
        self.assertEqual(body["model"], "muse-wonder-1.0")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["schema"]["required"][0], "hypothesis")
        self.assertEqual(body["max_completion_tokens"], 16000)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertIn("SYSTEM RULES", json.dumps(body["messages"][0]))
        self.assertNotIn("tool_choice", body)
        charged = [e for e in self.led.events() if e.kind == "budget.charged"]
        self.assertEqual(len(charged), 1)
        self.assertEqual(charged[0].actor, "wonder")
        self.assertAlmostEqual(charged[0].data["amount"], res.usd)
        self.assertEqual(charged[0].data["model"], "muse-wonder-1.0")
        self.assertEqual(charged[0].data["exp_id"], "cand-001")

    def test_the_default_reply_of_the_fake_satisfies_the_strict_schema_or_fails_cleanly(self):
        """With nothing queued the fake builds a sample from the schema; the sample is a schema-shaped object, and the source either
        parses it or reports it as unparseable (never crashes)."""
        try:
            res = self.source().propose(CTX)
        except UnparseableProposal as exc:
            self.assertGreater(exc.usd, 0)
        else:
            self.assertIsInstance(res.proposal, Proposal)

    def test_a_garbled_reply_is_retried_once_through_the_real_client(self):
        self.fake.enqueue(Reply.text("Here is my plan, in prose."), Reply.json(good_reply(), prompt_tokens=100, completion_tokens=50))
        res = self.source().propose(CTX)
        self.assertEqual(len(self.fake.requests), 2)
        charged = [e for e in self.led.events() if e.kind == "budget.charged"]
        self.assertEqual(len(charged), 2)
        self.assertEqual(res.prompt_tokens, sum(e.data["usage"]["prompt_tokens"] for e in charged))
        self.assertAlmostEqual(res.usd, sum(e.data["amount"] for e in charged))

    def test_the_outer_budget_is_a_hard_stop(self):
        self.fake.enqueue(Reply.json(good_reply(), prompt_tokens=100_000, completion_tokens=50_000))
        src = self.source(max_usd=0.5)
        res = src.propose(CTX)  # $0.40 + $1.00 = $1.40 > $0.5: the call itself was allowed (checked before), the next is not
        self.assertGreater(res.usd, 0.5)
        n = len(self.fake.requests)
        with self.assertRaises(BudgetExhausted):
            src.propose(CTX)
        self.assertEqual(len(self.fake.requests), n, "no request is sent once the budget is gone")

    def test_a_server_error_is_a_provider_failure_not_an_invalid_candidate(self):
        self.fake.enqueue(Fault.server_error(503), Fault.server_error(503))
        with self.assertRaises(ProposerUnavailable):
            self.source().propose(CTX)

    def test_a_silent_fallback_to_spark_is_visible_in_the_result(self):
        fake = FakeMetaServer(models=("muse-spark-1.3",)).start()
        self.addCleanup(fake.stop)
        fake.enqueue(Reply.json(good_reply(), prompt_tokens=100, completion_tokens=50))
        client = MetaChatClient("muse-wonder-1.0", base_url=fake.base_url, api_key=fake.api_key, fallback_model="muse-spark-1.3", max_retries=0, sleep=lambda s: None)
        gate = BudgetGate(BudgetLimits(max_llm_usd=5.0), self.led, clock=lambda: 1000.0)
        res = WonderProposalSource(MeteredClient(client, gate, "wonder")).propose(CTX)
        self.assertTrue(res.fallback_used)
        self.assertEqual(res.model, "muse-wonder-1.0")
        self.assertEqual(res.served_model, "muse-spark-1.3")

    def test_no_fallback_configured_means_a_missing_model_is_a_failure(self):
        fake = FakeMetaServer(models=("muse-spark-1.3",)).start()
        self.addCleanup(fake.stop)
        client = MetaChatClient("muse-wonder-1.0", base_url=fake.base_url, api_key=fake.api_key, fallback_model=None, max_retries=0, sleep=lambda s: None)
        gate = BudgetGate(BudgetLimits(max_llm_usd=5.0), self.led, clock=lambda: 1000.0)
        with self.assertRaises(ProposerUnavailable):
            WonderProposalSource(MeteredClient(client, gate, "wonder")).propose(CTX)


if __name__ == "__main__":
    unittest.main()
