"""Opt-in Responses API client for the Meta Model API. Standard library only.

UNTESTED AGAINST THE REAL SERVICE. No Meta API key was available when this was written: every behaviour below was checked only
against ``tests/support/fake_meta_server.py``, a fake that follows the documentation. The default protocol of Nightshift stays
Chat Completions (``MetaChatClient``); this client is used only for a role whose routing says ``api = "responses"``.

Why it exists (``docs/META_API_NOTES.md``, "Chat Completions versus Responses"): on Chat Completions the model re-reasons from
scratch on every tool turn for external API keys (``reasoning_content`` is redacted), which costs output tokens on every turn and
can make it lose the thread in a long tool loop. On Responses, requesting ``include: ["reasoning.encrypted_content"]`` with
``store: false`` returns encrypted reasoning items that the caller sends back as input on the next turn. The supervisor loop (up to
200 turns) and the patch worker (12 turns) are the roles that benefit; the single-turn structured roles gain nothing.

The same ``ModelClient`` protocol as ``MetaChatClient``, which this class extends: constructor arguments, retry policy (429/500/
502/503, ``Retry-After``, 504 retried once as a stream, transient in-stream errors re-sent), ``disabled_params`` learning, model
fallback, ``completion_cap_ceiling``, secret scrubbing and the ``response.raw["_nightshift"]`` facts that ``budget.MeteredClient``
reads are the same, so charging by call id, cost and ledger fields are unchanged. ``response.usage`` is the Chat-shaped
``Usage`` (``input_tokens`` -> ``prompt_tokens``, ``output_tokens`` -> ``completion_tokens``, reasoning tokens are part of the
output tokens and priced as output).

What goes on the wire
---------------------
``POST {base_url}/responses`` with ``input`` (typed items), ``instructions`` (the leading system/developer messages), flat function
``tools`` (``strict: false``), ``text.format`` (strict ``json_schema``) for structured output, ``reasoning.effort``,
``max_output_tokens`` (at least 16), ``store: false`` and, for a ``muse-*`` model, ``include: ["reasoning.encrypted_content"]``.
Never sent: ``previous_response_id`` (state is held by the caller), ``tool_choice``, ``stop``, ``n``, ``logprobs``, ``logit_bias``,
``truncation``, ``stream_options`` (documented as having no effect: usage arrives in ``response.completed``).

Reasoning replay
----------------
The reply to ``chat`` is Chat-shaped (``message.content``, ``message.tool_calls`` with ids, a Chat ``finish_reason``) so every
role consumes it unchanged. The assistant message additionally carries a private attribute ``_responses_items`` (an
``OpaqueItems``: the encrypted ``reasoning`` items, the ``phase`` of the commentary message and the ``function_call`` ids of that
turn, in the order the model produced them). When a later ``chat`` call receives that message in its history, the client rebuilds
the Responses input from the Chat fields (the single source of truth for text, tool names and arguments) and weaves the reasoning
items back in at their original positions. Rules:

* Replay only to the model that produced the items (a Wonder-to-Spark fallback attempt gets none), only when ``include`` is in
  use, and only when the carried ``function_call`` ids equal the message's ``tool_calls`` ids.
* A ``reasoning`` item that would not be followed by an assistant message or a ``function_call`` (the documented ordering rule) is
  dropped.
* A 400 that names a replayed reasoning item ("not found or has expired") is retried once without any replay; three such
  rejections turn replay off for the client and the response meta says so.
* A role that builds its own assistant message from a response instead of appending the returned one loses the attribute. The
  supervisor copies it (``base.carry_items``); the patch worker does not, so the client also remembers the carrier of every
  tool-calling turn under its tool-call ids (a bounded LRU) and uses it for a message that has NO attribute at all. The memo
  entry is scoped to the conversation (a hash of its system messages and first user message) and to the position of the turn in
  it (how many assistant messages precede it), so parallel lanes sharing a client, or a model that repeats a tool-call id and
  its arguments, never receive another turn's reasoning. History compaction marks the turns it drops with an empty carrier,
  which wins over the memo.
* ``MetaChatClient`` builds its wire messages from the ``ChatMessage`` fields only, so histories are interchangeable between the
  two clients; the attribute is ignored there.

Untrusted data. Everything in an output item is model/server data. ``OpaqueItems`` is inert (not iterable, not JSON serialisable,
``repr`` shows a count, a byte length and a hash only); ``response.raw`` holds a structural summary in which every encrypted blob is
replaced by its byte length and a hash; nothing here puts an opaque item in a prompt, a log or an exception message.

Assumptions taken from the documentation and NOT verified live (the probes in ``live_smoke`` exist to settle them)
------------------------------------------------------------------------------------------------------------------
1. ``store: false`` together with ``include: ["reasoning.encrypted_content"]`` is accepted, and a returned reasoning item replayed as
   ``{"type": "reasoning", "id", "summary": [], "encrypted_content"}`` is accepted on the next turn (documented under
   "Reasoning items" at ``/docs/protocols/responses``).
2. A replayed ``function_call`` may carry the ``id`` the server returned next to ``call_id``, ``name`` and ``arguments``.
3. Commentary text before a ``function_call`` must be replayed as an assistant message with ``phase: "commentary"`` (documented);
   a message that the model returned with a ``phase`` is replayed with it.
4. Function tools take the flat shape (documented). ``strict: false`` is sent explicitly because the default is not stated.
5. Encrypted reasoning is only valid for the model that issued it (NOT documented; the conservative reading, so a fallback model
   never receives it).
6. ``status: "incomplete"`` means the output limit was reached when ``incomplete_details.reason`` is ``max_output_tokens`` or absent
   (the shape of ``incomplete_details`` is not documented); ``content_filter`` maps to ``content_filter``.
7. Usage always arrives in the ``response.completed`` / ``response.incomplete`` event of a stream.
8. ``text.format`` strict ``json_schema`` accepts the same subset as Chat Completions ``response_format``.
9. ``max_output_tokens`` above the model's configured maximum is a 400 (documented), the maximum itself is not.
10. ``temperature`` is accepted while reasoning is on (unknown; a 400 naming it strips it, like on Chat Completions).
11. A ``response.failed`` event with error code ``server_error`` is transient and the request can be re-sent.
12. Partner hosts (Fireworks, Baseten, Modal) are not known to implement the Responses API: a role needs ``allow_partner_responses``
    in the routing config to use a non-Meta ``base_url``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..types import ChatMessage, ChatResponse, ModelError, NightshiftError, ToolCall, ToolSpec
from .base import (REPLAY_ATTR, STREAM_RETRYABLE_CODES, UNSET, OpaqueItems, StreamAssembler, _wire_arguments, carried_items,
                   clean_schema_name, estimate_tokens, normalize_reasoning_effort, parse_sse, reasoning_tokens_from_responses,
                   strictify_schema, tool_calls_from_wire, tools_to_wire, usage_from_responses, validate_tool_sequence)
from .meta import DEFAULT_MODEL, OPTIONAL_PARAMS, RETRYABLE_STATUSES, MetaChatClient, _Call

# One line per numbered assumption in the module docstring; the live smoke report and docs/META_API_NOTES.md quote them.
ASSUMPTIONS: tuple[str, ...] = (
    "store=false together with include reasoning.encrypted_content is accepted, and a returned reasoning item replayed as "
    "{type, id, summary: [], encrypted_content} is accepted on the next turn",
    "a replayed function_call may carry the id the server returned next to call_id, name and arguments",
    "commentary text before a function_call is replayed as an assistant message with phase 'commentary'; a phase the model "
    "returned is replayed with it",
    "function tools use the flat shape; strict: false is sent explicitly because the default is not stated",
    "encrypted reasoning is only valid for the model that issued it (not documented; a fallback model never receives it)",
    "status 'incomplete' means the output limit was reached when incomplete_details.reason is max_output_tokens or absent "
    "(the shape of incomplete_details is not documented)",
    "usage always arrives in the response.completed / response.incomplete event of a stream",
    "text.format strict json_schema accepts the same subset as Chat Completions response_format",
    "max_output_tokens above the model's configured maximum is a 400 (documented); the maximum itself is not",
    "temperature is accepted while reasoning is on (unknown; a 400 naming it strips it)",
    "a response.failed event with error code server_error is transient and the request can be re-sent",
    "partner hosts (Fireworks, Baseten, Modal) are not known to implement the Responses API",
)

RESPONSES_PATH = "/responses"
INCLUDE_REASONING = "reasoning.encrypted_content"
MIN_OUTPUT_TOKENS = 16               # /docs/protocols/responses: max_output_tokens "minimum 16"
MAX_CARRIED_BYTES = 2 * 1024 * 1024  # one turn's opaque items above this are not carried (the turn is simply not replayed)
MEMO_PER_KEY = 8                     # conversations remembered per identical set of tool calls
MEMO_ENTRIES = 256                   # tool-calling turns remembered for callers that rebuild assistant messages
MAX_CALL_ID = 64                     # /docs/protocols/responses: call_id is 1-64 characters
REPLAY_REJECTIONS_BEFORE_OFF = 3
# Terminal in-stream / response.failed codes after which the whole request is re-sent. "server_error" is the example code
# /docs/protocols/responses gives for response.failed (assumption 11).
TRANSIENT_CODES = STREAM_RETRYABLE_CODES | {"server_error"}
_REPLAY_REJECTED = re.compile(r"reasoning|encrypted|\brs_[A-Za-z0-9]|not found or has expired", re.I)
_CONTEXT_OVERFLOW = re.compile(r"context[ _-]?(?:window|length)|maximum context|too many tokens|prompt is too long", re.I)
# wire key -> the logical parameter name kept in ``disabled_params`` (the same names the Chat client and the smoke probes use).
# ``store`` is deliberately absent: a host that rejects ``store: false`` must fail, not silently keep every prompt server-side.
_WIRE_PARAMS = {"reasoning": "reasoning_effort", "temperature": "temperature", "prompt_cache_key": "prompt_cache_key",
                "prompt_cache_retention": "prompt_cache_retention", "safety_identifier": "safety_identifier",
                "include": "include"}
_PARAM_WORDS = {"reasoning_effort": ("reasoning.effort", "reasoning_effort", "effort")}


# --------------------------------------------------------------------------------------------------
# pure conversion: Chat-shaped history -> Responses input
# --------------------------------------------------------------------------------------------------
def reasoning_capable(model: str) -> bool:
    """Muse models always reason (``docs/META_API_NOTES.md``), so ``include`` is requested for every ``muse-*`` model. A model
    the docs do not list is not assumed to emit encrypted reasoning."""
    return str(model).strip().lower().startswith("muse-")


def wire_call_id(call_id: str) -> str:
    """The id sent as ``call_id``: unchanged when it is 1-64 characters, else a stable hash of it (the matching
    ``function_call_output`` gets the same mapping, so the pair stays matched)."""
    if call_id and len(call_id) <= MAX_CALL_ID:
        return call_id
    return "call_" + hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:32]


def flat_tools(tools: Sequence[ToolSpec]) -> list[dict]:
    """Chat's nested tool specs (validated by ``tools_to_wire``) in the flat Responses shape, ``strict: false``."""
    out = []
    for item in tools_to_wire(tools):
        fn = item["function"]
        out.append({"type": "function", "name": fn["name"], "description": fn["description"], "parameters": fn["parameters"],
                    "strict": False})
    return out


def text_format(schema: dict, name: str) -> dict:
    """``text`` parameter for strict structured output (``text.format``, not ``response_format``)."""
    return {"format": {"type": "json_schema", "name": clean_schema_name(name), "schema": strictify_schema(schema), "strict": True}}


@dataclass
class Converted:
    """The Responses ``input`` for one request plus facts about it."""

    instructions: str | None
    items: list[dict]
    replayed: int = 0                  # reasoning items sent back
    skipped_turns: int = 0             # carried turns that were NOT replayed (other model, ids changed, include off)


def _reasoning_input(item: Mapping[str, Any]) -> dict:
    out: dict[str, Any] = {"type": "reasoning", "summary": list(item.get("summary") or []),
                           "encrypted_content": item["encrypted_content"]}
    if isinstance(item.get("id"), str) and item["id"]:
        out["id"] = item["id"]
    return out


def _message_input(role: str, text: str, phase: str | None = None) -> dict:
    kind = "output_text" if role == "assistant" else "input_text"
    out: dict[str, Any] = {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}
    if phase and role == "assistant":
        out["phase"] = phase
    return out


def _call_input(call: ToolCall, item_id: str | None) -> dict:
    out: dict[str, Any] = {"type": "function_call", "call_id": wire_call_id(call.id), "name": call.name,
                           "arguments": _wire_arguments(call)}
    if item_id:
        out["id"] = item_id
    return out


def _usable_state(state: OpaqueItems | None, calls: Sequence[ToolCall], model: str) -> bool:
    if state is None or not state or state.model != model:
        return False
    carried = [i.get("call_id") for i in state.items if i.get("type") == "function_call"]
    return carried == [c.id for c in calls]


def assistant_items(msg: ChatMessage, state: OpaqueItems | None, model: str) -> tuple[list[dict], int, bool]:
    """Input items for one assistant message: ``(items, reasoning_items_replayed, carried_state_ignored)``.

    Without usable state: an assistant message (``phase: "commentary"`` when tool calls follow, which the API requires) and one
    ``function_call`` per call. With usable state the carried items give the ORDER: ``reasoning`` items are replayed as they were,
    the message and function calls are rebuilt from ``msg``. Text the role added to a turn that had none is placed before the first
    call."""
    text = msg.content if (msg.content or "").strip() else ""
    calls = list(msg.tool_calls or [])
    usable = _usable_state(state, calls, model)
    ignored = state is not None and bool(state) and not usable
    phase_default = "commentary" if calls else None
    items: list[dict] = []
    replayed = 0
    if usable and state is not None:
        by_id = {c.id: c for c in calls}
        message_done = False
        for it in state.items:
            kind = it.get("type")
            if kind == "reasoning" and isinstance(it.get("encrypted_content"), str):
                items.append(_reasoning_input(it))
                replayed += 1
            elif kind == "message" and not message_done:
                message_done = True
                if text:
                    items.append(_message_input("assistant", text, "commentary" if calls else (it.get("phase") or None)))
            elif kind == "function_call" and it.get("call_id") in by_id:
                if text and not message_done:        # text the model did not produce in this slot: before the first call
                    items.append(_message_input("assistant", text, "commentary"))
                    message_done = True
                items.append(_call_input(by_id.pop(it["call_id"]), it.get("id")))
        if text and not message_done:
            items.append(_message_input("assistant", text, phase_default))
        return items, replayed, False
    if text:
        items.append(_message_input("assistant", text, phase_default))
    items.extend(_call_input(c, None) for c in calls)
    return items, 0, ignored


def _drop_orphan_reasoning(items: list[dict]) -> list[dict]:
    """Remove every ``reasoning`` item that is not followed (after any further reasoning items) by an assistant message or a
    ``function_call``: the API answers that ordering with HTTP 400 (``param: "input"``)."""
    out: list[dict] = []
    for i, item in enumerate(items):
        if item.get("type") == "reasoning":
            j = i + 1
            while j < len(items) and items[j].get("type") == "reasoning":
                j += 1
            nxt = items[j] if j < len(items) else None
            ok = nxt is not None and (nxt.get("type") == "function_call"
                                      or (nxt.get("type") == "message" and nxt.get("role") == "assistant"))
            if not ok:
                continue
        out.append(item)
    return out


def messages_to_input(messages: Sequence[ChatMessage], model: str, *,
                      lookup: Callable[[ChatMessage], OpaqueItems | None] | None = None, replay: bool = True) -> Converted:
    """Chat-shaped messages to Responses ``instructions`` + ``input``.

    Leading ``system``/``developer`` messages become ``instructions``; a later one stays an input message. ``tool`` messages become
    ``function_call_output`` items (order preserved: without ``previous_response_id`` the outputs must follow their calls in
    order). ``lookup`` supplies the carrier for an assistant message (default: its ``_responses_items`` attribute). With
    ``replay=False`` no reasoning item is sent. Raises ``ValueError`` for an unknown role or a history ``validate_tool_sequence``
    rejects; nothing is sent in that case."""
    if not messages:
        raise ValueError("at least one message is required")
    validate_tool_sequence(messages)
    find = lookup or _attribute_lookup
    lead: list[str] = []
    items: list[dict] = []
    replayed = skipped = 0
    for i, msg in enumerate(messages):
        role = msg.role
        if role in ("system", "developer"):
            if not items:
                lead.append(msg.content or "")
            else:
                items.append(_message_input(role, msg.content or ""))
        elif role == "user":
            items.append(_message_input("user", msg.content or ""))
        elif role == "assistant":
            state = find(msg) if replay else None
            part, n, ignored = assistant_items(msg, state, model)
            items.extend(part)
            replayed += n
            skipped += 1 if ignored else 0
        elif role == "tool":
            if not msg.tool_call_id:
                raise ValueError(f"message {i}: a tool message needs tool_call_id")
            items.append({"type": "function_call_output", "call_id": wire_call_id(msg.tool_call_id), "output": msg.content or ""})
        else:
            raise ValueError(f"unknown message role {role!r}")
    items = _drop_orphan_reasoning(items)
    kept = sum(1 for it in items if it.get("type") == "reasoning")
    return Converted(instructions="\n\n".join(lead) if lead else None, items=items, replayed=kept, skipped_turns=skipped)


def _attribute_lookup(msg: ChatMessage) -> OpaqueItems | None:
    value = carried_items(msg)
    return None if value is UNSET else value  # type: ignore[return-value]


# --------------------------------------------------------------------------------------------------
# pure conversion: Responses output -> Chat-shaped result
# --------------------------------------------------------------------------------------------------
@dataclass
class ParsedOutput:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    carried: list[dict] = field(default_factory=list)        # structural items to replay (see OpaqueItems)
    reasoning_items: int = 0
    refusal: bool = False
    ignored_types: list[str] = field(default_factory=list)


def _output_text(content: Any) -> tuple[str, bool]:
    """``(visible text, saw_refusal)`` from a message item's ``content``: ``output_text`` blocks count, ``refusal`` blocks are
    flagged and never become text."""
    if isinstance(content, str):
        return content, False
    text, refused = [], False
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, Mapping):
            continue
        kind = block.get("type")
        if kind == "refusal":
            refused = True
        elif kind in ("output_text", "text") and isinstance(block.get("text"), str):
            text.append(block["text"])
    return "".join(text), refused


def parse_output(output: Any) -> ParsedOutput:
    """Output items to text, tool calls and the structural carrier. Unknown item types are ignored (and listed in
    ``ignored_types``), never an error. A ``reasoning`` item without ``encrypted_content`` (the host ignored ``include``) is not
    carried."""
    parsed = ParsedOutput()
    texts: list[str] = []
    wire_calls: list[dict] = []
    call_items: list[Mapping[str, Any]] = []
    order: list[tuple[str, Mapping[str, Any]]] = []
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "reasoning":
            parsed.reasoning_items += 1
            if isinstance(item.get("encrypted_content"), str) and item["encrypted_content"]:
                order.append(("reasoning", item))
        elif kind == "message":
            text, refused = _output_text(item.get("content"))
            parsed.refusal = parsed.refusal or refused
            if text:
                texts.append(text)
            order.append(("message", item))
        elif kind == "function_call":
            fn_id = item.get("call_id") or item.get("id") or f"call_{len(wire_calls)}"
            wire_calls.append({"id": str(fn_id), "function": {"name": item.get("name"), "arguments": item.get("arguments")}})
            call_items.append(item)
            order.append(("function_call", item))
        else:
            parsed.ignored_types.append(str(kind)[:40])
    parsed.text = "\n\n".join(texts)
    parsed.calls = tool_calls_from_wire(wire_calls)
    next_call = iter(parsed.calls)
    for kind, item in order:
        if kind == "reasoning":
            summary = item.get("summary")        # server data: anything but a list of objects is treated as no summary
            carried: dict[str, Any] = {"type": "reasoning", "encrypted_content": item["encrypted_content"],
                                       "summary": [s for s in (summary if isinstance(summary, list) else []) if isinstance(s, Mapping)][:16]}
            if isinstance(item.get("id"), str):
                carried["id"] = item["id"][:128]
        elif kind == "message":
            carried = {"type": "message"}
            if item.get("phase") in ("commentary", "final_answer"):
                carried["phase"] = item["phase"]
        else:
            call = next(next_call)
            carried = {"type": "function_call", "call_id": call.id}
            if isinstance(item.get("id"), str):
                carried["id"] = item["id"][:128]
        parsed.carried.append(carried)
    return parsed


def finish_reason_for(data: Mapping[str, Any], has_calls: bool) -> str:
    """Chat ``finish_reason`` for a Responses object (assumption 6): ``incomplete`` -> ``length`` unless the reason says
    ``content_filter``; ``completed`` -> ``tool_calls`` when there are calls, else ``stop``."""
    if data.get("status") == "incomplete":
        details = data.get("incomplete_details")
        reason = str(details.get("reason") if isinstance(details, Mapping) else "").lower()
        return "content_filter" if "content_filter" in reason else "length"
    return "tool_calls" if has_calls else "stop"


def describe_output(output: Any) -> list[dict]:
    """A structural, content-free summary of the output items for ``response.raw``: types, ids and call names, and for a reasoning
    item only the byte length and a hash of its encrypted blob."""
    out: list[dict] = []
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, Mapping):
            continue
        entry: dict[str, Any] = {"type": str(item.get("type"))[:40]}
        if isinstance(item.get("id"), str):
            entry["id"] = item["id"][:128]
        blob = item.get("encrypted_content")
        if isinstance(blob, str):
            entry["encrypted_content_bytes"] = len(blob.encode("utf-8"))
            entry["encrypted_content_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
        if item.get("type") == "function_call":
            entry["name"] = str(item.get("name"))[:80]
        if item.get("phase"):
            entry["phase"] = str(item["phase"])[:20]
        out.append(entry)
    return out


# --------------------------------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------------------------------
class ResponsesStreamAssembler:
    """Fold Responses stream events into one response object (the shape a non-streaming call returns).

    The terminal ``response.completed`` / ``response.incomplete`` event carries the whole response (output items and usage); if its
    ``output`` is empty the items are rebuilt from ``response.output_item.done`` events plus the text and argument deltas. An
    ``error`` or ``response.failed`` event raises ``ModelError`` (shared with the Chat assembler). Unknown events are ignored.
    """

    def __init__(self) -> None:
        self._items: dict[int, dict] = {}
        self._text: dict[tuple[int, int], list[str]] = {}
        self._args: dict[int, list[str]] = {}
        self._args_done: dict[int, str] = {}
        self.response: dict | None = None
        self.terminal: str | None = None
        self.id: str | None = None
        self.model: str | None = None
        self.chunks = 0

    @property
    def finished(self) -> bool:
        return self.terminal is not None

    def feed(self, event: Mapping[str, Any]) -> None:
        self.chunks += 1
        StreamAssembler._raise_on_error(event)
        kind = event.get("type")
        if not isinstance(kind, str):
            return
        resp = event.get("response") if isinstance(event.get("response"), Mapping) else None
        if resp is not None:
            self.id = resp.get("id") or self.id
            self.model = resp.get("model") or self.model
        index = event.get("output_index")
        index = index if isinstance(index, int) and not isinstance(index, bool) else None
        if kind in ("response.output_item.added", "response.output_item.done") and index is not None:
            if isinstance(event.get("item"), Mapping):
                self._items[index] = dict(event["item"])
        elif kind == "response.output_text.delta" and index is not None and isinstance(event.get("delta"), str):
            cidx = event.get("content_index") if isinstance(event.get("content_index"), int) else 0
            self._text.setdefault((index, cidx), []).append(event["delta"])
        elif kind == "response.function_call_arguments.delta" and index is not None and isinstance(event.get("delta"), str):
            self._args.setdefault(index, []).append(event["delta"])
        elif kind == "response.function_call_arguments.done" and index is not None and isinstance(event.get("arguments"), str):
            self._args_done[index] = event["arguments"]
        elif kind in ("response.completed", "response.incomplete") and resp is not None:
            self.terminal, self.response = kind, dict(resp)

    def to_response(self) -> dict:
        resp = dict(self.response or {})
        if not isinstance(resp.get("output"), list) or not resp["output"]:
            resp["output"] = self._rebuilt()
        if not resp.get("status"):
            resp["status"] = "incomplete" if self.terminal == "response.incomplete" else "completed"
        resp.setdefault("id", self.id)
        resp.setdefault("model", self.model)
        return resp

    def _rebuilt(self) -> list[dict]:
        out: list[dict] = []
        for index in sorted(self._items):
            item = dict(self._items[index])
            if item.get("type") == "function_call" and not item.get("arguments"):
                item["arguments"] = self._args_done.get(index) or "".join(self._args.get(index, []))
            elif item.get("type") == "message" and not item.get("content"):
                parts = [("".join(chunks)) for (idx, _), chunks in sorted(self._text.items()) if idx == index]
                item["content"] = [{"type": "output_text", "text": t} for t in parts]
            out.append(item)
        return out


# --------------------------------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------------------------------
class _RCall(_Call):
    """``_Call`` plus the chat history (converted per model, lazily) and per-call replay bookkeeping."""

    def __init__(self, history: list[ChatMessage], tools: list[dict] | None, text: dict | None, max_tokens: int | None,
                 temperature: float | None, cache_key: str | None, timeout: float, stream: bool):
        super().__init__([], tools, None, max_tokens, temperature, cache_key, timeout, stream)
        self.history, self.text = history, text
        self.no_replay: set[str] = set()                  # models whose replay a 400 of THIS call rejected
        self.converted: dict[tuple[str, bool], Converted] = {}
        self.info.update(replayed=0, skipped_turns=0, replay_dropped=False)


class MetaResponsesClient(MetaChatClient):
    """Responses API client; implements ``nightshift.types.ModelClient``. See the module docstring. Thread-safe like its parent.

    Extra constructor argument: ``replay_reasoning`` (default True) turns the replay of encrypted reasoning off entirely (items are
    still requested and carried, which costs nothing, but never sent back)."""

    api = "responses"

    def __init__(self, model: str = DEFAULT_MODEL, *, replay_reasoning: bool = True, **kw: Any):
        super().__init__(model, **kw)
        self._url = self.base_url + RESPONSES_PATH
        self.replay_reasoning = bool(replay_reasoning)
        self.replay_disabled_reason: str | None = None
        self._replay_rejections = 0
        self._memo: "OrderedDict[tuple[tuple[str, str, str], ...], list[tuple[str, int, OpaqueItems]]]" = OrderedDict()

    def __repr__(self) -> str:
        return (f"MetaResponsesClient(model={self.model!r}, base_url={self.base_url!r}, "
                f"fallback_model={self.fallback_model!r})")

    # ---- public API -------------------------------------------------------------------------------
    def chat(self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolSpec] | None = None,
             response_schema: dict | None = None, schema_name: str = "response", reasoning_effort: str | None = None,
             max_completion_tokens: int | None = None, temperature: float | None = None, cache_key: str | None = None,
             timeout: float | None = None, stream: bool = False) -> ChatResponse:
        """One Responses call with a Chat-shaped result. ``max_completion_tokens`` is sent as ``max_output_tokens`` (raised to the
        documented minimum of 16). Raises ``ValueError`` for a malformed request (nothing is sent) and ``ModelError`` for
        anything the server or network did. The returned assistant message carries ``_responses_items`` (see module docstring)."""
        history = list(messages)
        if not history:
            raise ValueError("at least one message is required")
        validate_tool_sequence(history)
        wire_tools = flat_tools(tools) if tools else None
        fmt = text_format(response_schema, schema_name) if response_schema is not None else None
        if max_completion_tokens is not None and (not isinstance(max_completion_tokens, int)
                                                  or isinstance(max_completion_tokens, bool) or max_completion_tokens < 1):
            raise ValueError("max_completion_tokens must be a positive integer")
        if temperature is not None and not (isinstance(temperature, (int, float)) and math.isfinite(temperature)):
            raise ValueError("temperature must be a finite number")
        if timeout is not None and not (isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
                                        and math.isfinite(timeout) and timeout > 0):
            raise ValueError("timeout must be a positive number of seconds")
        requested = reasoning_effort if reasoning_effort is not None else self.default_reasoning
        primary_effort = normalize_reasoning_effort(requested, self.model)
        call = _RCall(history, wire_tools, fmt, max_completion_tokens, temperature, cache_key,
                      self._timeout_for(max_completion_tokens, timeout), stream)
        self._converted(self.model, call)         # an unconvertible message (unknown role, tool message without id) fails now
        started = self._timer()
        attempts: list[tuple[str, str | None, bool]] = [(self.model, primary_effort, False)]
        if self.fallback_model:
            attempts.append((self.fallback_model, self._fallback_effort(requested), True))
            if self.primary_unavailable is not None:
                attempts = attempts[1:]
        for i, (model, effort, is_fallback) in enumerate(attempts):
            try:
                data, info = self._complete(model, effort, call)
            except ModelError as exc:
                if i + 1 < len(attempts) and exc.status in (400, 404):
                    if exc.status == 404 and self._model_missing(exc):
                        self._remember_missing_primary(call, model)
                    continue
                raise
            return self._to_chat_response(data, info, model, effort, is_fallback, started, response_schema is not None, call)
        raise AssertionError("unreachable")  # pragma: no cover

    # ---- reasoning memo -----------------------------------------------------------------------------------
    def _carrier_for(self, msg: ChatMessage, scope: str | None = None, ordinal: int = 0) -> OpaqueItems | None:
        """The carrier to replay for an assistant message: its own attribute (an empty one means "dropped on purpose"), else
        the memo entry for its tool-call ids in the same conversation (``scope``) and at the same place in it (``ordinal``:
        how many assistant messages precede it), else nothing."""
        value = carried_items(msg)
        if value is not UNSET:
            return value  # type: ignore[return-value]
        if not msg.tool_calls:
            return None
        key = self._memo_key(msg.tool_calls)
        with self._lock:
            entries = self._memo.get(key)
            if not entries:
                return None
            self._memo.move_to_end(key)
            if scope is None:                      # no conversation given (a direct lookup): the latest turn with these calls
                return entries[-1][2]
            for entry_scope, entry_ordinal, carrier in reversed(entries):
                if entry_scope == scope and entry_ordinal == ordinal:
                    return carrier
            return None

    @staticmethod
    def _scope_of(history: Sequence[ChatMessage]) -> str:
        """Identifies the conversation: a hash of its leading system/developer messages and its first user message (what history
        compaction always keeps, and what does not change as the conversation grows). Two conversations that share a client,
        such as parallel lanes, never receive each other's reasoning even if a model reuses a tool-call id."""
        head: list[tuple[str, str]] = []
        for m in history:
            if m.role not in ("system", "developer", "user") or len(head) >= 8:
                break
            head.append((m.role, m.content or ""))
            if m.role == "user":
                break
        return hashlib.sha256(json.dumps(head, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _memo_key(calls: Sequence[ToolCall]) -> tuple[tuple[str, str, str], ...]:
        """Tool-call ids, names and a hash of the raw arguments: a model that reuses an id in a later turn cannot be given the
        earlier turn's reasoning unless it also repeats the call."""
        return tuple((c.id, c.name, hashlib.sha256(c.raw_arguments.encode("utf-8")).hexdigest()[:16]) for c in calls)

    @staticmethod
    def _ordinals(history: Sequence[ChatMessage]) -> dict[int, int]:
        """``id(message) -> number of assistant messages before it`` for the assistant messages of ``history``."""
        out: dict[int, int] = {}
        seen = 0
        for m in history:
            if m.role == "assistant":
                out[id(m)] = seen
                seen += 1
        return out

    def _remember(self, scope: str, ordinal: int, calls: Sequence[ToolCall], carrier: OpaqueItems) -> None:
        """Remember ``carrier`` for the tool-calling turn that will be appended after ``ordinal`` assistant messages. A model
        that repeats an id and arguments in a later turn of the same conversation therefore does not receive an earlier (or
        later) turn's reasoning."""
        if not calls or not carrier:
            return
        with self._lock:
            key = self._memo_key(calls)
            entries = [e for e in self._memo.get(key, []) if (e[0], e[1]) != (scope, ordinal)]
            entries.append((scope, ordinal, carrier))
            self._memo[key] = entries[-MEMO_PER_KEY:]
            self._memo.move_to_end(key)
            while len(self._memo) > MEMO_ENTRIES:
                self._memo.popitem(last=False)

    # ---- request construction -------------------------------------------------------------------------------
    def _replay_on(self, model: str, call: _RCall) -> bool:
        return (self.replay_reasoning and "include" not in self.disabled_params and model not in call.no_replay
                and reasoning_capable(model))

    def _converted(self, model: str, call: _RCall) -> Converted:
        replay = self._replay_on(model, call)
        key = (model, replay)
        found = call.converted.get(key)
        if found is None:
            scope = self._scope_of(call.history)
            ordinals = self._ordinals(call.history)
            found = call.converted[key] = messages_to_input(
                call.history, model, replay=replay, lookup=lambda m: self._carrier_for(m, scope, ordinals.get(id(m), -1)))
        return found

    def _body(self, model: str, effort: str | None, call: "_Call", stream: bool) -> dict:
        assert isinstance(call, _RCall)
        off = self.disabled_params
        conv = self._converted(model, call)
        body: dict[str, Any] = {"model": model, "input": conv.items}
        if conv.instructions:
            body["instructions"] = conv.instructions
        body["store"] = False
        if reasoning_capable(model) and "include" not in off:
            body["include"] = [INCLUDE_REASONING]
        if call.tools:
            body["tools"] = call.tools
        if call.text:
            body["text"] = call.text
        if effort and "reasoning_effort" not in off:
            body["reasoning"] = {"effort": effort}
        cap = self._cap_for(model, call.max_tokens, call.cap_override.get(model))
        if cap is not None:
            body["max_output_tokens"] = max(MIN_OUTPUT_TOKENS, cap)
        if call.temperature is not None and "temperature" not in off:
            body["temperature"] = call.temperature
        if call.cache_key and "prompt_cache_key" not in off:
            body["prompt_cache_key"] = call.cache_key
        if self.cache_retention and "prompt_cache_retention" not in off:
            body["prompt_cache_retention"] = self.cache_retention
        if self._safety and "safety_identifier" not in off:
            body["safety_identifier"] = self._safety
        if stream:
            self._make_streamed(body)
        call.info["replayed"], call.info["skipped_turns"] = conv.replayed, conv.skipped_turns
        return body

    def _make_streamed(self, body: dict) -> None:
        """A streamed request. ``stream_options`` is not sent: on Responses it "has no effect" and usage always arrives in the
        terminal event."""
        body["stream"] = True

    def _note_accepted(self, body: Mapping[str, Any]) -> None:
        cap, model = body.get("max_output_tokens"), body.get("model")
        if isinstance(cap, int) and isinstance(model, str):
            with self._lock:
                self._accepted_cap[model] = max(cap, self._accepted_cap.get(model, 0))

    @staticmethod
    def _names_completion_cap(exc: ModelError, body: Mapping[str, Any]) -> bool:
        if not isinstance(body.get("max_output_tokens"), int):
            return False
        text = f"{exc} {getattr(exc, 'param', None) or ''}".lower()
        if (exc.code or "").lower() == "context_length_exceeded" or (
                _CONTEXT_OVERFLOW.search(text) and "configured maximum" not in text):
            return False
        return "max_output_tokens" in text or "max output" in text or "maximum output" in text

    @staticmethod
    def _named_optional(exc: ModelError, body: Mapping[str, Any]) -> list[str]:
        """Logical names (the ones kept in ``disabled_params``) of the optional parameters a 400 names."""
        text = f"{exc} {getattr(exc, 'param', None) or ''}".lower()
        found = []
        for wire, logical in _WIRE_PARAMS.items():
            if wire not in body:
                continue
            # whole words only ("include" must not match "included"), and "reasoning" alone also appears in replayed-reasoning
            # errors, so the effort parameter needs the word "effort"
            words = _PARAM_WORDS.get(logical) or (wire, logical)
            if any(re.search(rf"(?<![a-z0-9_]){re.escape(w)}(?![a-z0-9_])", text) for w in words):
                found.append(logical)
        return found

    def _replayed_reasoning_rejected(self, exc: ModelError, body: Mapping[str, Any]) -> bool:
        items = body.get("input")
        has = isinstance(items, list) and any(isinstance(i, Mapping) and i.get("type") == "reasoning" for i in items)
        return has and bool(_REPLAY_REJECTED.search(f"{exc} {getattr(exc, 'param', None) or ''}"))

    # ---- the request loop -------------------------------------------------------------------------------------
    def _complete(self, model: str, effort: str | None, call: "_Call") -> tuple[dict, dict]:
        """One model: retry without any optional parameter the server names in a 400, once without replayed reasoning when the
        400 names a reasoning item, and once at a lower ``max_output_tokens`` when the 400 names the cap."""
        assert isinstance(call, _RCall)
        last: ModelError | None = None
        cap_retried = replay_retried = False
        for _ in range(len(OPTIONAL_PARAMS) + 6):
            body = self._body(model, effort, call, call.stream)
            try:
                result = self._post(body, call)
                if model in call.cap_override:
                    with self._lock:
                        self._learned_cap[model] = call.cap_override[model]
                return result
            except ModelError as exc:
                if exc.status != 400:
                    raise
                named = self._named_optional(exc, body)
                if named:
                    with self._lock:
                        self.disabled_params.update(named)
                elif not replay_retried and self._replayed_reasoning_rejected(exc, body):
                    replay_retried = True
                    call.no_replay.add(model)
                    call.info["replay_dropped"] = True
                    self._note_replay_rejected()
                elif not cap_retried and self._names_completion_cap(exc, body):
                    lower = self._lower_cap(model, body["max_output_tokens"], exc)
                    if lower is None:
                        raise
                    cap_retried = True
                    call.cap_override[model] = lower
                    call.info["cap_reduced"] = {"from": body["max_output_tokens"], "to": lower}
                else:
                    raise
                last = exc
        assert last is not None
        raise last

    def _note_replay_rejected(self) -> None:
        with self._lock:
            self._replay_rejections += 1
            if self._replay_rejections >= REPLAY_REJECTIONS_BEFORE_OFF and self.replay_reasoning:
                self.replay_reasoning = False
                self.replay_disabled_reason = (f"the server rejected replayed reasoning items {self._replay_rejections} times; "
                                               f"this client stopped replaying them")

    def _post(self, body: dict, call: "_Call") -> tuple[dict, dict]:
        info = call.info
        stream = bool(body.get("stream"))
        retries = 0
        while True:
            payload = json.dumps(body, allow_nan=False).encode("utf-8")
            info["attempts"] += 1
            reasoning = body.get("reasoning")
            info["sent_effort"] = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
            info["sent_cap"] = body.get("max_output_tokens")
            status, headers, raw = self._send(payload, stream, call.timeout)
            if 200 <= status < 300:
                info["streamed"] = stream
                info["request_id"] = headers.get("x-request-id") or headers.get("request-id")
                info["ratelimit"] = {k: v[:40] for k, v in headers.items() if k.startswith("x-ratelimit-")}
                try:
                    data = self._read_stream(raw) if stream else self._read_json(raw)
                except ModelError as exc:
                    if self._transient_stream_error(exc) and retries < self.max_retries:
                        self._sleep(self._delay(retries, None))
                        retries += 1
                        info["retries"] += 1
                        continue
                    raise
                self._ensure_usage(data, payload)
                self._note_accepted(body)
                return data, info
            err = self._http_error(status, headers, raw)
            if status == 429:
                info["rate_limited"] += 1
                if err.retry_after is not None:
                    info["retry_after_seen"].append(err.retry_after)
            if status == 504 and not stream:
                stream = True
                body = dict(body)
                self._make_streamed(body)
                call.stream = True
                info["stream_fallback"] = True
                continue
            if status in RETRYABLE_STATUSES:
                if retries >= self.max_retries or (err.retry_after or 0) > self._max_retry_after:
                    raise err
                self._sleep(self._delay(retries, err.retry_after))
                retries += 1
                info["retries"] += 1
                continue
            raise err

    @staticmethod
    def _transient_stream_error(exc: ModelError) -> bool:
        return (exc.code or exc.error_type or "").lower() in TRANSIENT_CODES

    # ---- response parsing -------------------------------------------------------------------------------------
    def _read_json(self, raw: Any) -> dict:
        text = self._body_text(raw)
        try:
            data = json.loads(text)
        except ValueError:
            raise ModelError("the API returned a response that is not JSON", error_type="malformed_response") from None
        if not isinstance(data, dict):
            raise ModelError("the API response is not a JSON object", error_type="malformed_response")
        self._check_status(data, text)
        return data

    def _check_status(self, data: Mapping[str, Any], text: str = "") -> None:
        status = data.get("status")
        error = data.get("error")
        if error and (status in (None, "failed") or not isinstance(data.get("output"), list)):
            env = self._envelope(json.dumps({"error": error}) if isinstance(error, Mapping) else text)
            code = (env["code"] or "").lower()
            raise ModelError(self._clean(f"the API reported an error in a 200 response: {env['message']}"),
                             error_type=env["type"], code=env["code"], retryable=code in TRANSIENT_CODES)
        if status == "failed":
            raise ModelError("the response failed", error_type="response.failed")
        if status not in ("completed", "incomplete") or not isinstance(data.get("output"), list):
            raise ModelError(self._clean(f"the API response is not a finished response (status {str(status)[:30]!r})"),
                             error_type="malformed_response")

    def _read_stream(self, raw: Any) -> dict:
        asm = ResponsesStreamAssembler()
        try:
            chunks = raw if not isinstance(raw, (bytes, bytearray)) else [bytes(raw)]
            for event in parse_sse(chunks):
                asm.feed(event)
        except ModelError as exc:
            raise self._scrub_error(exc) from None
        except NightshiftError:
            raise
        except ValueError as exc:
            raise ModelError(self._clean(f"malformed stream from the API: {exc}"), error_type="malformed_response") from None
        except Exception as exc:
            raise ModelError(f"stream interrupted: {type(exc).__name__}", error_type="network", retryable=True) from None
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()
        if not asm.finished:
            raise ModelError("the stream ended before the reply was complete", error_type="truncated_stream", retryable=True)
        data = asm.to_response()
        self._check_status(data)
        return data

    @staticmethod
    def _ensure_usage(data: dict, request_payload: bytes) -> None:
        """A reply without token counts is charged an estimate (visible text only) and flagged ``_usage_estimated``."""
        usage = data.get("usage")
        if isinstance(usage, Mapping) and (usage.get("input_tokens") is not None or usage.get("output_tokens") is not None):
            return
        parsed = parse_output(data.get("output"))
        produced = parsed.text + "".join(c.name + c.raw_arguments for c in parsed.calls)
        data["usage"] = {"input_tokens": estimate_tokens(request_payload.decode("utf-8", errors="replace")),
                         "output_tokens": estimate_tokens(produced)}
        data["_usage_estimated"] = True

    def _to_chat_response(self, data: dict, info: dict, model: str, effort: str | None, is_fallback: bool, started: float,
                          schema_requested: bool, call: "_RCall") -> ChatResponse:
        parsed = parse_output(data.get("output"))
        usage, cached_known = usage_from_responses(data.get("usage"))
        carrier = OpaqueItems(parsed.carried if any(c.get("type") == "reasoning" for c in parsed.carried) else [], model=model)
        if carrier.nbytes > MAX_CARRIED_BYTES:
            carrier = OpaqueItems((), model=model)
        if carrier:
            self._remember(self._scope_of(call.history), sum(1 for m in call.history if m.role == "assistant"), parsed.calls, carrier)
        with self._lock:
            disabled = set(self.disabled_params)
            reason = self.primary_unavailable if (self._reason_pending and is_fallback) else None
            if reason:
                self._reason_pending = False
            replay_off = self.replay_disabled_reason
        details = data.get("incomplete_details")
        meta = {
            "cached_known": cached_known, "latency_ms": round((self._timer() - started) * 1000.0, 1),
            "retries": info["retries"], "stream_fallback": info["stream_fallback"], "requested_model": self.model,
            "reasoning_effort": info.get("sent_effort"), "schema_requested": schema_requested,
            "streamed": info["streamed"], "attempts": info["attempts"], "fallback_used": is_fallback,
            "disabled_params": sorted(disabled), "usage_estimated": bool(data.get("_usage_estimated")),
            "reasoning_tokens": reasoning_tokens_from_responses(data.get("usage")), "ratelimit": dict(info.get("ratelimit") or {}),
            "max_completion_tokens_sent": info.get("sent_cap"), "completion_cap_reduced": info.get("cap_reduced"),
            "rate_limited": info["rate_limited"], "retry_after_seen": list(info["retry_after_seen"]),
            # Responses extras
            "api": "responses", "replayed_reasoning_items": info.get("replayed", 0), "replay_skipped_turns": info.get("skipped_turns", 0),
            "replay_dropped": bool(info.get("replay_dropped")), "reasoning_items_returned": parsed.reasoning_items,
            "reasoning_items_carried": sum(1 for c in carrier.items if c.get("type") == "reasoning"),
            "opaque_bytes": carrier.nbytes, "opaque_sha256": carrier.digest or None, "refusal": parsed.refusal,
            "ignored_output_types": sorted(set(parsed.ignored_types)), "response_status": str(data.get("status")),
            "incomplete_reason": str(details.get("reason"))[:40] if isinstance(details, Mapping) and details.get("reason") else None,
        }
        if reason:
            meta["fallback_reason"] = reason
        if replay_off:
            meta["replay_disabled"] = replay_off
        raw = {"id": data.get("id") if isinstance(data.get("id"), str) else None, "object": "response",
               "model": data.get("model") if isinstance(data.get("model"), str) else None, "status": data.get("status"),
               "output": describe_output(data.get("output")), "usage": data.get("usage") if isinstance(data.get("usage"), Mapping) else None,
               "incomplete_details": details if isinstance(details, Mapping) else None, "_nightshift": meta, "nightshift": meta}
        answered = data.get("model") if isinstance(data.get("model"), str) and data.get("model") else model
        request_id = info.get("request_id") or (data.get("id") if isinstance(data.get("id"), str) else None)
        message = ChatMessage(role="assistant", content=parsed.text if (parsed.text or not parsed.calls) else None,
                              tool_calls=parsed.calls)
        if carrier:
            setattr(message, REPLAY_ATTR, carrier)
        return ChatResponse(message=message, finish_reason=finish_reason_for(data, bool(parsed.calls)), usage=usage,
                            model=answered, request_id=request_id, raw=raw)
