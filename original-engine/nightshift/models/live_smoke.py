"""Cheap live probes of a Chat Completions endpoint. ``nightshift doctor --live`` and ``scripts/live_smoke.py`` run this.

``run_smoke(client_factory, out_path)`` makes a few dozen tiny calls and returns (and optionally writes) a JSON report.

Required probes (``ok`` in the report is false when one FAILS):

* ``tool_roundtrip``: a tool call, then the tool-result turn.
* ``strict_schema``: ``response_format`` json_schema strict.
* ``effort_<level>``: one call per reasoning effort.
* ``stream``: one forced-stream call; the usage must be REAL (``stream_options.include_usage``), not estimated.
* ``rejected_param``: a deliberately rejected parameter, to confirm the documented error envelope.

Informational probes (each independent, each cheap; they teach the client what the live API really does):

* ``prompt_cache`` and ``prompt_cache_large`` (prefix of at least 8k tokens; the minimum cacheable prefix is unstated).
* ``strict_<NAME>`` for every strict schema the agents use (PLAN, REVIEW, REFLECT, VERDICT) and ``supervisor_tools`` for the
  six supervisor tool schemas; on a rejection the probe strips JSON-Schema keywords one at a time to learn which one the
  strict subset refuses (``rejected_keywords``).
* ``effort_max`` (muse-spark-1.3 must accept it) and ``effort_none_rejected`` (must be a 400).
* ``temperature_with_reasoning``, ``completion_cap_ceiling`` (16000/32000/64000), ``assistant_tool_calls_content`` (null versus
  ``""``), ``parallel_tool_calls``, ``tools_with_response_format``, ``ratelimit_headers``, ``system_and_developer_roles``.
* ``wonder_chat``: does ``muse-wonder-1.0`` answer on Chat Completions?

Responses probes (``report["responses"]``, their own key, never part of ``ok``): when a Responses client factory is available
(``build_client_factory`` attaches one as ``factory.responses_factory``; ``run_smoke(responses_factory=...)`` overrides it) the
same run also settles whether ``MetaResponsesClient`` works on this endpoint: ``responses_plain``, ``responses_include_encrypted``
(``include: ["reasoning.encrypted_content"]`` with ``store: false`` is accepted and the reply carries an encrypted reasoning item),
``responses_tool_roundtrip_replay`` (a tool call, then the tool-result turn with that reasoning item REPLAYED), ``responses_structured``
(``text.format`` strict) and ``responses_stream``. The section ends in a one-line recommendation: ``responses OK: set api =
"responses" for the supervisor in nightshift.toml``, or ``chat only``. The Responses client was written from the documentation and
has never run against the real service before these probes (see ``nightshift/models/responses.py`` for the assumptions).

A probe is ``pass``, ``fail`` or ``inconclusive``. Muse Spark always reasons and reasoning tokens count against
``max_completion_tokens``, so a reply with ``finish_reason == "length"`` that lacks what the probe needs says nothing about the
feature: it is reported ``inconclusive`` (never ``fail``), with a message, and listed in ``required_inconclusive``.

The report holds status, models, latencies, token counts and ``cached_known``. It never holds the API key or any message
content (the prompts, replies and tool arguments are not recorded; error text is limited to the server's own message,
scrubbed and capped).
"""
from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .. import secrets
from ..types import ChatMessage, ModelClient, ModelError, ToolCall, ToolSpec, Usage
from .base import EFFORTS, strictify_schema, tools_to_wire
from .meta import DEFAULT_BASE_URL, DEFAULT_MODEL, MetaChatClient, resolve_api_key
from .pricing import usd_for

REPORT_VERSION = 2
WONDER_MODEL = "muse-wonder-1.0"
# Reasoning tokens count against the cap and Muse Spark always reasons, so 200 can be spent entirely on hidden reasoning.
DEFAULT_MAX_COMPLETION_TOKENS = 2000
CAP_PROBE_VALUES = (16000, 32000, 64000)
# JSON-Schema keywords whose support in strict mode is unknown; the schema probes learn which are refused.
CONSTRAINT_KEYWORDS = ("minLength", "maxLength", "pattern", "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum",
                       "exclusiveMaximum", "multipleOf", "format", "uniqueItems", "minProperties", "maxProperties")
_NOT_KEYWORD_VALUES = ("enum", "const", "default", "examples", "required")
_ADD_TOOL = ToolSpec("add_numbers", "Add two integers and return the sum.",
                     {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                      "required": ["a", "b"]})
_SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}, "unit": {"type": "string"}}}
_SHARED_PREFIX = "Reference list for a caching probe. " + " ".join(f"item{i:04d}" for i in range(900))
# At least 8k tokens even by the crude 4-characters-per-token rule (about 15k with a real tokenizer).
_LARGE_PREFIX = "Reference list for a large caching probe. " + " ".join(f"item{i:04d}" for i in range(3800))
_READY = "Reply with the single word: ready."


class _Run:
    """Accumulates per-call facts while the probes run."""

    def __init__(self, timer: Callable[[], float]):
        self.timer = timer
        self.calls = 0
        self.prompt = self.completion = self.cached = 0
        self.usd = 0.0
        self.models: set[str] = set()
        self.cached_known = False
        self.ratelimit: dict[str, str] = {}
        self.rate_limited = 0
        self.retry_after: list[float] = []

    def chat(self, client: Any, messages: list[ChatMessage], **kw: Any) -> tuple[Any, float]:
        t0 = self.timer()
        resp = client.chat(messages, **kw)
        latency = round((self.timer() - t0) * 1000.0, 1)
        self.calls += 1
        self.prompt += resp.usage.prompt_tokens
        self.completion += resp.usage.completion_tokens
        self.cached += resp.usage.cached_tokens
        self.usd += usd_for(resp.model, resp.usage)
        self.models.add(resp.model)
        meta = resp.raw.get("_nightshift") if isinstance(resp.raw, dict) else None
        if isinstance(meta, Mapping) and meta.get("cached_known"):
            self.cached_known = True
        if isinstance(meta, Mapping) and isinstance(meta.get("ratelimit"), Mapping):
            self.ratelimit.update({str(k): str(v) for k, v in meta["ratelimit"].items()})
        if isinstance(meta, Mapping):     # 429s the client's retry loop absorbed
            seen = meta.get("rate_limited")
            self.rate_limited += seen if isinstance(seen, int) and not isinstance(seen, bool) else 0
            self.retry_after.extend(float(v) for v in (meta.get("retry_after_seen") or ()) if isinstance(v, (int, float)))
        return resp, latency

    def raw(self, client: Any, body: Mapping[str, Any]) -> tuple[int, dict]:
        """A request sent exactly as given (no retries, no parameter policy): a rejection stays visible as a status."""
        raw_request = getattr(client, "raw_request", None)
        if raw_request is None:
            raise RuntimeError("this client cannot send a raw request")
        status, payload = raw_request(dict(body))
        self.calls += 1
        if status == 429:
            self.rate_limited += 1
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else 0
            completion = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else 0
            self.prompt += prompt
            self.completion += completion
            model = payload.get("model") if isinstance(payload.get("model"), str) else getattr(client, "model", DEFAULT_MODEL)
            self.usd += usd_for(model, Usage(prompt_tokens=prompt, completion_tokens=completion))
        return status, payload


def _meta(resp: Any) -> Mapping[str, Any]:
    meta = resp.raw.get("_nightshift") if isinstance(resp.raw, dict) else None
    return meta if isinstance(meta, Mapping) else {}


def _error_info(exc: BaseException) -> dict:
    if isinstance(exc, ModelError):
        return {"kind": "ModelError", "status": exc.status, "error_type": exc.error_type, "code": exc.code,
                "retryable": exc.retryable, "message": secrets.scrub_text(str(exc))[:300]}
    return {"kind": type(exc).__name__, "status": None, "error_type": None, "code": None, "retryable": None, "message": ""}


def _cut_off(resp: Any) -> bool:
    return str(getattr(resp, "finish_reason", "") or "").strip().lower() == "length"


def _inconclusive(what: str, cap: int) -> str:
    return (f"finish_reason was 'length': hidden reasoning tokens used up max_completion_tokens={cap} before {what}. This says "
            f"nothing about the feature; re-run with a larger max_completion_tokens (run_smoke(max_completion_tokens=...)).")


def _api_message(payload: Mapping[str, Any]) -> str | None:
    err = payload.get("error") if isinstance(payload, Mapping) else None
    msg = err.get("message") if isinstance(err, Mapping) else None
    return secrets.scrub_text(msg)[:300] if isinstance(msg, str) and msg else None


def _strip_keywords(node: Any, keywords: Sequence[str]) -> Any:
    """``node`` without the given JSON-Schema keywords at any depth (property NAMES and literal values are never touched)."""
    if isinstance(node, list):
        return [_strip_keywords(item, keywords) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in keywords:
            continue
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {name: _strip_keywords(sub, keywords) for name, sub in value.items()}
        elif key in _NOT_KEYWORD_VALUES:
            out[key] = value
        else:
            out[key] = _strip_keywords(value, keywords)
    return out


def _keywords_used(node: Any) -> list[str]:
    found: set[str] = set()

    def walk(n: Any) -> None:
        if isinstance(n, list):
            for item in n:
                walk(item)
        elif isinstance(n, dict):
            for key, value in n.items():
                if key in CONSTRAINT_KEYWORDS:
                    found.add(key)
                elif key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                    for sub in value.values():
                        walk(sub)
                elif key not in _NOT_KEYWORD_VALUES:
                    walk(value)

    walk(node)
    return sorted(found)


def _agent_schemas() -> dict[str, dict]:
    """The strict response schemas the agents really send (imported lazily; the smoke test must not need them to import)."""
    from ..agents import prompts
    from ..audit import judge
    return {"PLAN": prompts.PLAN_SCHEMA, "REVIEW": prompts.REVIEW_SCHEMA, "REFLECT": prompts.REFLECT_SCHEMA,
            "VERDICT": judge.VERDICT_SCHEMA}


def _supervisor_tools() -> list[ToolSpec]:
    """The six supervisor tool schemas exactly as the orchestrator builds them (they carry minLength, pattern, maxItems ...)."""
    from .. import orchestrator
    schemas = orchestrator._schemas(["default"], ["stage"])
    return [ToolSpec(name, desc, schema) for name, (desc, schema) in schemas.items()]


def run_smoke(client_factory: Callable[..., ModelClient], out_path: str | Path | None = None, *,
              probe_wonder: bool = True, wonder_model: str = WONDER_MODEL,
              max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
              clock: Callable[[], float] = time.time, timer: Callable[[], float] = time.monotonic,
              responses_factory: Callable[[str | None], ModelClient] | None = None) -> dict:
    """Run every probe and return the report. A failing probe is recorded, never raised.

    ``client_factory(model)`` returns a client for ``model``; it is called with ``None`` for the default model. A factory
    that takes no arguments is also accepted (the Wonder probe and the muse-spark-1.3 effort probe then use the default
    client). ``ok`` in the report is true when no required probe failed; ``required_inconclusive`` lists required probes
    that could not tell (the reply was cut off by the token cap), and ``conclusive`` is true when there are none.

    ``responses_factory(model)`` returns a ``MetaResponsesClient`` for the Responses probes (default: the
    ``responses_factory`` attribute of ``client_factory``, which ``build_client_factory`` sets). Without one the report's
    ``responses`` section says ``skipped``. Those probes never affect ``ok``.
    """
    try:
        accepts_model = len([p for p in inspect.signature(client_factory).parameters.values()
                             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)]) > 0
    except (TypeError, ValueError):
        accepts_model = False
    make = (lambda model=None: client_factory(model)) if accepts_model else (lambda model=None: client_factory())
    run = _Run(timer)
    probes: list[dict] = []
    client = make(None)
    cap = max_completion_tokens

    def probe(name: str, fn: Callable[[], tuple[bool | None, dict]], *, required: bool = True,
              into: list[dict] | None = None, tally: "_Run | None" = None) -> dict:
        """``fn`` returns ``(True, detail)`` pass, ``(False, detail)`` fail or ``(None, detail)`` inconclusive."""
        into = probes if into is None else into
        tally = run if tally is None else tally
        t0 = timer()
        entry: dict[str, Any] = {"name": name, "required": required, "ok": False, "status": "fail", "latency_ms": 0.0,
                                 "detail": {}, "error": None}
        try:
            result, entry["detail"] = fn()
            entry["ok"] = result is True
            entry["status"] = "pass" if result is True else ("inconclusive" if result is None else "fail")
        except Exception as exc:  # noqa: BLE001 - a probe must never abort the run
            entry["error"] = _error_info(exc)
            if isinstance(exc, ModelError) and exc.status == 429:
                tally.rate_limited += 1
                if exc.retry_after is not None:
                    tally.retry_after.append(exc.retry_after)
        entry["latency_ms"] = round((timer() - t0) * 1000.0, 1)
        into.append(entry)
        return entry

    def tool_roundtrip() -> tuple[bool | None, dict]:
        history = [ChatMessage("user", "Use the add_numbers tool to add 2 and 3.")]
        first, _ = run.chat(client, history, tools=[_ADD_TOOL], max_completion_tokens=cap, reasoning_effort="minimal")
        calls = first.message.tool_calls
        detail: dict[str, Any] = {"tool_calls": len(calls), "finish_reason": first.finish_reason}
        if not calls:
            if _cut_off(first):
                detail["message"] = _inconclusive("the tool call was written", cap)
                return None, detail
            detail["problem"] = "model answered in text instead of calling the tool (tool_choice cannot force it)"
            return False, detail
        call = calls[0]
        detail.update(tool_name=call.name, arguments_valid=call.parse_error is None)
        if call.parse_error is not None and _cut_off(first):
            detail["message"] = _inconclusive("the tool-call arguments were complete", cap)     # a cut-off fragment, not bad JSON
            return None, detail
        history += [first.message, ChatMessage("tool", "5", tool_call_id=call.id)]
        history.extend(ChatMessage("tool", "5", tool_call_id=c.id) for c in calls[1:])
        second, _ = run.chat(client, history, tools=[_ADD_TOOL], max_completion_tokens=cap, reasoning_effort="minimal")
        answered = bool(second.text.strip()) or bool(second.message.tool_calls)
        detail.update(second_finish_reason=second.finish_reason, second_has_text=bool(second.text.strip()),
                      second_has_tool_calls=bool(second.message.tool_calls))
        if not answered and _cut_off(second):
            detail["message"] = _inconclusive("the tool-result turn was answered", cap)
            return None, detail
        return call.name == _ADD_TOOL.name and call.parse_error is None and answered, detail

    def strict_schema() -> tuple[bool | None, dict]:
        resp, _ = run.chat(client, [ChatMessage("user", "Answer with the sum of 2 and 3 and the unit 'count'.")],
                           response_schema=_SCHEMA, schema_name="smoke_answer", max_completion_tokens=cap,
                           reasoning_effort="minimal")
        try:
            parsed = json.loads(resp.text)
        except ValueError:
            parsed = None
        keys_ok = isinstance(parsed, dict) and set(parsed) == set(_SCHEMA["properties"])
        types_ok = keys_ok and isinstance(parsed["answer"], int) and isinstance(parsed["unit"], str)
        detail = {"json_parsed": parsed is not None, "keys_match": bool(keys_ok), "types_match": bool(types_ok),
                  "finish_reason": resp.finish_reason}
        if not types_ok and parsed is None and _cut_off(resp):
            detail["message"] = _inconclusive("the JSON answer was complete", cap)
            return None, detail
        return bool(types_ok), detail

    def effort_probe(effort: str) -> Callable[[], tuple[bool | None, dict]]:
        def go() -> tuple[bool | None, dict]:
            resp, _ = run.chat(client, [ChatMessage("user", _READY)], reasoning_effort=effort, max_completion_tokens=cap)
            sent = _meta(resp).get("reasoning_effort")
            stripped = "reasoning_effort" in getattr(client, "disabled_params", ())
            return (sent == effort and not stripped), {
                "effort": effort, "sent_effort": sent, "parameter_stripped": stripped, "finish_reason": resp.finish_reason,
                "completion_tokens": resp.usage.completion_tokens, "reasoning_tokens": _meta(resp).get("reasoning_tokens")}
        return go

    def prompt_cache_probe(prefix: str, key: str) -> Callable[[], tuple[bool | None, dict]]:
        def go() -> tuple[bool | None, dict]:
            messages = [ChatMessage("system", prefix), ChatMessage("user", _READY)]
            kw = dict(cache_key=key, max_completion_tokens=min(cap, 64), reasoning_effort="minimal")
            first, _ = run.chat(client, messages, **kw)
            second, _ = run.chat(client, messages, **kw)
            m1, m2 = _meta(first), _meta(second)
            return True, {"cached_known": bool(m1.get("cached_known") or m2.get("cached_known")),
                          "first_cached_tokens": first.usage.cached_tokens, "second_cached_tokens": second.usage.cached_tokens,
                          "first_prompt_tokens": first.usage.prompt_tokens, "second_prompt_tokens": second.usage.prompt_tokens,
                          "prefix_at_least_8k_tokens": first.usage.prompt_tokens >= 8000,
                          "cache_hit": second.usage.cached_tokens > 0,
                          "prompt_cache_key_stripped": "prompt_cache_key" in getattr(client, "disabled_params", ())}
        return go

    def stream() -> tuple[bool | None, dict]:
        resp, _ = run.chat(client, [ChatMessage("user", _READY)], stream=True, max_completion_tokens=cap,
                           reasoning_effort="minimal")
        meta = _meta(resp)
        has_text = bool(resp.text.strip())
        estimated = bool(meta.get("usage_estimated"))
        detail: dict[str, Any] = {
            "streamed": bool(meta.get("streamed")), "has_text": has_text, "usage_estimated": estimated,
            "include_usage_stripped": "stream_options" in getattr(client, "disabled_params", ()),
            "reasoning_tokens": meta.get("reasoning_tokens"), "finish_reason": resp.finish_reason}
        if not has_text and _cut_off(resp):
            detail["message"] = _inconclusive("any text was streamed", cap)
            return None, detail
        if estimated:
            detail["problem"] = ("the stream carried no usage although stream_options.include_usage was requested: every streamed "
                                 "call is charged an estimate of the visible text only, so hidden reasoning tokens would "
                                 "under-charge the budget gate")
        return bool(meta.get("streamed")) and has_text and not estimated, detail

    def rejected_param() -> tuple[bool, dict]:
        if getattr(client, "raw_request", None) is None:
            return False, {"problem": "this client cannot send a raw request"}
        status, payload = run.raw(client, {"model": getattr(client, "model", DEFAULT_MODEL), "stop": ["\n"],
                                           "messages": [{"role": "user", "content": "ping"}]})
        err = payload.get("error") if isinstance(payload, dict) else None
        err = err if isinstance(err, dict) else {}
        has_message, has_type = isinstance(err.get("message"), str) and bool(err.get("message")), isinstance(err.get("type"), str)
        return status == 400 and has_message and has_type, {
            "status": status, "error_type": err.get("type") if has_type else None, "has_message": has_message,
            "has_type": has_type, "has_param": "param" in err, "has_code": "code" in err}

    # ---- informational probes ------------------------------------------------------------------------------
    def raw_body(**extra: Any) -> dict:
        body = {"model": getattr(client, "model", DEFAULT_MODEL), "messages": [{"role": "user", "content": "Reply with: ok"}],
                "reasoning_effort": "minimal", "max_completion_tokens": cap}
        body.update(extra)
        return body

    def learn_keywords(build: Callable[[Any], dict], schema: Any) -> dict[str, Any]:
        """After a rejected schema: which constraint keywords, removed one at a time, make the request pass?"""
        used = _keywords_used(schema)
        culprits = []
        for kw in used:
            status, _ = run.raw(client, build(_strip_keywords(schema, [kw])))
            if status == 200:
                culprits.append(kw)
        all_status = run.raw(client, build(_strip_keywords(schema, used)))[0] if used else None
        return {"constraint_keywords_used": used, "rejected_keywords": culprits,
                "accepted_without_all_constraints": all_status == 200 if used else None}

    def agent_schema_probe(name: str, schema: dict) -> Callable[[], tuple[bool | None, dict]]:
        def go() -> tuple[bool | None, dict]:
            strict = strictify_schema(schema)

            def build(sch: Any) -> dict:
                return raw_body(response_format={"type": "json_schema", "json_schema": {
                    "name": f"smoke_{name.lower()}", "schema": sch, "strict": True}})

            status, payload = run.raw(client, build(strict))
            detail: dict[str, Any] = {"status": status, "constraint_keywords_used": _keywords_used(strict)}
            if status == 200:
                choice = (payload.get("choices") or [{}])[0] if isinstance(payload, dict) else {}
                detail["finish_reason"] = choice.get("finish_reason") if isinstance(choice, dict) else None
                return True, detail
            detail["api_message"] = _api_message(payload)
            if status == 400:
                detail.update(learn_keywords(build, strict))
            detail["message"] = (f"the {name} strict schema was not accepted (HTTP {status}); response_format is never stripped, "
                                 f"so every call that uses it will fail on this endpoint")
            return False, detail
        return go

    def supervisor_tools() -> tuple[bool | None, dict]:
        tools = _supervisor_tools()
        wire = tools_to_wire(tools)
        status, payload = run.raw(client, raw_body(tools=wire))
        detail: dict[str, Any] = {"status": status, "tools": [t.name for t in tools],
                                  "constraint_keywords_used": _keywords_used([t.parameters for t in tools])}
        if status == 200:
            return True, detail
        detail["api_message"] = _api_message(payload)
        per_tool = {tool.name: run.raw(client, raw_body(tools=[wired]))[0] for tool, wired in zip(tools, wire)}
        detail["status_per_tool"] = per_tool
        learned = {}
        for tool in tools:
            if per_tool[tool.name] == 400:
                learned[tool.name] = learn_keywords(
                    lambda sch, name=tool.name: raw_body(tools=[{"type": "function", "function": {
                        "name": name, "description": "", "parameters": sch}}]), tool.parameters)
        detail["keywords_per_rejected_tool"] = learned
        detail["message"] = ("the supervisor tool schemas were not accepted; the supervisor cannot run on this endpoint "
                             "(tools are never stripped)")
        return False, detail

    def effort_max() -> tuple[bool | None, dict]:
        spark = client if (getattr(client, "model", None) == "muse-spark-1.3" or not accepts_model) else make("muse-spark-1.3")
        resp, _ = run.chat(spark, [ChatMessage("user", _READY)], reasoning_effort="max", max_completion_tokens=cap)
        sent = _meta(resp).get("reasoning_effort")
        return sent == "max", {"sent_effort": sent, "model": resp.model, "finish_reason": resp.finish_reason,
                               "message": "documented as supported by muse-spark-1.3 and muse-wonder-1.0, not by -contributor"}

    def effort_none_rejected() -> tuple[bool | None, dict]:
        status, payload = run.raw(client, raw_body(reasoning_effort="none"))
        return status == 400, {"status": status, "api_message": _api_message(payload),
                               "message": "documented: 'none' is a 400 (the client sends 'minimal' instead)"}

    def temperature_with_reasoning() -> tuple[bool | None, dict]:
        resp, _ = run.chat(client, [ChatMessage("user", _READY)], reasoning_effort="low", temperature=0.2,
                           max_completion_tokens=cap)
        stripped = "temperature" in getattr(client, "disabled_params", ())
        detail: dict[str, Any] = {"temperature_stripped": stripped, "finish_reason": resp.finish_reason}
        if stripped:
            detail["message"] = ("temperature is rejected while reasoning is on; the client drops it and the provider default "
                                 "applies")
        return not stripped, detail

    def completion_cap_ceiling() -> tuple[bool | None, dict]:
        statuses: dict[str, int | None] = {}
        message = None
        for value in CAP_PROBE_VALUES:
            if message is not None:
                statuses[str(value)] = None             # an earlier, smaller value was already refused: do not ask again
                continue
            status, payload = run.raw(client, raw_body(max_completion_tokens=value))
            statuses[str(value)] = status
            if status != 200:
                message = _api_message(payload)
        accepted = [v for v in CAP_PROBE_VALUES if statuses[str(v)] == 200]
        rejected = [v for v in CAP_PROBE_VALUES if statuses[str(v)] not in (200, None)]
        detail: dict[str, Any] = {"statuses": statuses, "highest_accepted": max(accepted) if accepted else None,
                                  "lowest_rejected": min(rejected) if rejected else None, "api_message": message}
        if not accepted or min(CAP_PROBE_VALUES) in rejected:
            detail["message"] = ("16000 completion tokens were refused: the retry cap ceiling (prompts.RETRY_CAP_CEILING) is too "
                                 "high for this endpoint; set a lower completion_cap_ceiling in the routing config")
        return bool(accepted) and min(CAP_PROBE_VALUES) not in rejected, detail

    def assistant_tool_calls_content() -> tuple[bool | None, dict]:
        out: dict[str, Any] = {}
        for label, content in (("null", None), ("empty_string", "")):
            call = ToolCall(id="call_smoke1", name="add_numbers", arguments={"a": 2, "b": 3}, raw_arguments='{"a": 2, "b": 3}')
            history = [ChatMessage("user", "Use the add_numbers tool to add 2 and 3."),
                       ChatMessage("assistant", content, tool_calls=[call]),
                       ChatMessage("tool", "5", tool_call_id="call_smoke1")]
            try:
                resp, _ = run.chat(client, history, tools=[_ADD_TOOL], max_completion_tokens=cap, reasoning_effort="minimal")
                out[label] = {"accepted": True, "finish_reason": resp.finish_reason}
            except ModelError as exc:
                out[label] = {"accepted": False, "status": exc.status, "api_message": secrets.scrub_text(str(exc))[:200]}
        detail: dict[str, Any] = dict(out)
        if not out["null"]["accepted"]:
            detail["message"] = ("an assistant tool-call message with content null is refused, and that is what the client "
                                 "replays after every tool call: every tool loop turn after the first would fail")
        return bool(out["null"]["accepted"]), detail

    def parallel_tool_calls() -> tuple[bool | None, dict]:
        history = [ChatMessage("user", "Use the add_numbers tool twice in the same turn: once for 2 and 3, once for 10 and 20.")]
        first, _ = run.chat(client, history, tools=[_ADD_TOOL], max_completion_tokens=cap, reasoning_effort="minimal")
        calls = first.message.tool_calls
        detail: dict[str, Any] = {"tool_calls": len(calls), "finish_reason": first.finish_reason}
        if len(calls) < 2:
            detail["message"] = (_inconclusive("two tool calls were written", cap) if _cut_off(first) else
                                 "the model issued fewer than two tool calls, so answering parallel calls was not exercised")
            return None, detail
        history += [first.message] + [ChatMessage("tool", str(7 * (i + 1)), tool_call_id=c.id) for i, c in enumerate(calls)]
        second, _ = run.chat(client, history, tools=[_ADD_TOOL], max_completion_tokens=cap, reasoning_effort="minimal")
        detail.update(second_finish_reason=second.finish_reason, second_has_text=bool(second.text.strip()))
        if not second.text.strip() and not second.message.tool_calls and _cut_off(second):
            detail["message"] = _inconclusive("the tool results were answered", cap)
            return None, detail
        return True, detail

    def tools_with_response_format() -> tuple[bool | None, dict]:
        resp, _ = run.chat(client, [ChatMessage("user", "Add 2 and 3, then answer with the sum and the unit 'count'.")],
                           tools=[_ADD_TOOL], response_schema=_SCHEMA, schema_name="smoke_answer", max_completion_tokens=cap,
                           reasoning_effort="minimal")
        return True, {"accepted": True, "finish_reason": resp.finish_reason, "tool_calls": len(resp.message.tool_calls),
                      "has_text": bool(resp.text.strip())}

    def ratelimit_headers() -> tuple[bool | None, dict]:
        resp, _ = run.chat(client, [ChatMessage("user", _READY)], max_completion_tokens=cap, reasoning_effort="minimal")
        headers = dict(_meta(resp).get("ratelimit") or {})
        return bool(headers), {"headers": headers, "rate_limited_responses": run.rate_limited,
                               "retry_after_seconds": list(run.retry_after),
                               "message": "documented: x-ratelimit-limit/remaining-tokens/requests on successful responses"}

    def system_and_developer_roles() -> tuple[bool | None, dict]:
        out: dict[str, Any] = {}
        for role in ("system", "developer"):
            try:
                resp, _ = run.chat(client, [ChatMessage(role, "You answer with exactly one word."), ChatMessage("user", "Say ready.")],
                                   max_completion_tokens=cap, reasoning_effort="minimal")
                out[role] = {"accepted": True, "finish_reason": resp.finish_reason}
            except ModelError as exc:
                out[role] = {"accepted": False, "status": exc.status, "api_message": secrets.scrub_text(str(exc))[:200]}
        detail: dict[str, Any] = dict(out)
        if not out["system"]["accepted"]:
            detail["message"] = "role 'system' is refused: every agent prompt starts with a system message and would fail"
        return all(v["accepted"] for v in out.values()), detail

    probe("tool_roundtrip", tool_roundtrip)
    probe("strict_schema", strict_schema)
    for effort in EFFORTS:
        probe(f"effort_{effort}", effort_probe(effort))
    probe("prompt_cache", prompt_cache_probe(_SHARED_PREFIX, "nightshift-live-smoke"), required=False)
    probe("stream", stream)
    probe("rejected_param", rejected_param)
    probe("effort_max", effort_max, required=False)
    probe("effort_none_rejected", effort_none_rejected, required=False)
    try:
        agent_schemas = _agent_schemas()
    except Exception as exc:  # noqa: BLE001 - recorded as one failed probe instead of aborting
        agent_schemas = {}
        probes.append({"name": "strict_agent_schemas", "required": False, "ok": False, "status": "fail", "latency_ms": 0.0,
                       "detail": {"message": "the agent schemas could not be imported"}, "error": _error_info(exc)})
    for schema_name, schema in agent_schemas.items():
        probe(f"strict_{schema_name}", agent_schema_probe(schema_name, schema), required=False)
    probe("supervisor_tools", supervisor_tools, required=False)
    probe("temperature_with_reasoning", temperature_with_reasoning, required=False)
    probe("completion_cap_ceiling", completion_cap_ceiling, required=False)
    probe("assistant_tool_calls_content", assistant_tool_calls_content, required=False)
    probe("parallel_tool_calls", parallel_tool_calls, required=False)
    probe("tools_with_response_format", tools_with_response_format, required=False)
    probe("system_and_developer_roles", system_and_developer_roles, required=False)
    probe("ratelimit_headers", ratelimit_headers, required=False)
    probe("prompt_cache_large", prompt_cache_probe(_LARGE_PREFIX, "nightshift-live-smoke-large"), required=False)

    wonder_state = "skipped"
    if probe_wonder and accepts_model:
        def wonder() -> tuple[bool | None, dict]:
            wonder_client = make(wonder_model)
            resp, _ = run.chat(wonder_client, [ChatMessage("user", _READY)], reasoning_effort="high", max_completion_tokens=cap)
            return True, {"model": resp.model, "finish_reason": resp.finish_reason}
        entry = probe("wonder_chat", wonder, required=False)
        if entry["ok"]:
            wonder_state = "supported"
        elif entry["error"] and entry["error"].get("status") in (400, 404):
            wonder_state = "unsupported"
        else:
            wonder_state = "unknown"

    responses_section = _responses_section(
        responses_factory or getattr(client_factory, "responses_factory", None), probe, cap, timer, client)

    failed = [p["name"] for p in probes if p["required"] and p["status"] == "fail"]
    inconclusive = [p["name"] for p in probes if p["required"] and p["status"] == "inconclusive"]
    cache = next((p for p in probes if p["name"] == "prompt_cache"), None)
    report: dict[str, Any] = {
        "version": REPORT_VERSION, "generated_at": clock(), "ok": not failed, "required_failed": failed,
        "required_inconclusive": inconclusive, "conclusive": not inconclusive,
        "max_completion_tokens": cap,
        "endpoint": getattr(client, "base_url", None),
        "models": {"requested": getattr(client, "model", None), "answered": sorted(run.models)},
        "cached_known": run.cached_known, "cache_hit": bool(cache and cache["detail"].get("cache_hit")),
        "wonder_chat_completions": wonder_state,
        "rate_limit": {"headers": dict(run.ratelimit), "rate_limited_responses": run.rate_limited,
                       "retry_after_seconds": list(run.retry_after)},
        "usage": {"calls": run.calls, "prompt_tokens": run.prompt, "completion_tokens": run.completion,
                  "cached_tokens": run.cached, "estimated_usd": round(run.usd, 6)},
        "latency_ms": {"total": round(sum(p["latency_ms"] for p in probes), 1),
                       "max": max((p["latency_ms"] for p in probes), default=0.0)},
        "probes": probes,
        "responses": responses_section,
    }
    report = secrets.scrub(report)
    if out_path is not None:
        _write_json(Path(out_path), report)
    return report


RESPONSES_UNTESTED = ("The Responses client (nightshift/models/responses.py) was written from the documentation and had never run "
                      "against the real service before this probe.")


def _responses_section(make: Callable[[str | None], ModelClient] | None, probe: Callable[..., dict], cap: int,
                       timer: Callable[[], float], chat_client: Any) -> dict:
    """The ``responses`` section of the report: five independent probes of ``MetaResponsesClient`` and a recommendation.
    Nothing here can fail the run: a probe that raises is recorded, and the section is never part of ``ok``."""
    from .responses import ASSUMPTIONS

    base: dict[str, Any] = {"tested": False, "status": "skipped", "untested_note": RESPONSES_UNTESTED,
                            "assumptions": list(ASSUMPTIONS), "probes": [],
                            "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "estimated_usd": 0.0},
                            "endpoint": None, "models": {"requested": None, "answered": []}, "failed": [], "inconclusive": [],
                            "recommendation": 'chat only: the Responses probes did not run; keep api = "chat"'}
    if make is None:
        base["reason"] = "no Responses client factory was supplied"
        return base
    try:
        rc = make(None)
    except Exception as exc:  # noqa: BLE001 - recorded, never raised
        base.update(status="fail", reason="the Responses client could not be built", error=_error_info(exc),
                    recommendation='chat only: the Responses client could not be built; keep api = "chat"')
        return base
    run = _Run(timer)
    entries: list[dict] = []

    def go(name: str, fn: Callable[[], tuple[bool | None, dict]]) -> None:
        probe(name, fn, required=False, into=entries, tally=run)

    def meta_of(resp: Any) -> Mapping[str, Any]:
        return _meta(resp)

    def plain() -> tuple[bool | None, dict]:
        resp, _ = run.chat(rc, [ChatMessage("user", _READY)], reasoning_effort="minimal", max_completion_tokens=cap)
        meta = meta_of(resp)
        detail: dict[str, Any] = {"finish_reason": resp.finish_reason, "has_text": bool(resp.text.strip()), "model": resp.model,
                                  "usage_estimated": bool(meta.get("usage_estimated")), "reasoning_tokens": meta.get("reasoning_tokens"),
                                  "response_status": meta.get("response_status"), "ignored_output_types": meta.get("ignored_output_types")}
        if not detail["has_text"] and _cut_off(resp):
            detail["message"] = _inconclusive("any text was written", cap)
            return None, detail
        return detail["has_text"], detail

    def include_encrypted() -> tuple[bool | None, dict]:
        resp, _ = run.chat(rc, [ChatMessage("user", _READY)], reasoning_effort="minimal", max_completion_tokens=cap)
        meta = meta_of(resp)
        stripped = sorted({"include"} & set(getattr(rc, "disabled_params", ())))
        carried = int(meta.get("reasoning_items_carried") or 0)
        detail: dict[str, Any] = {"parameters_stripped": stripped, "reasoning_items_returned": meta.get("reasoning_items_returned"),
                                  "reasoning_items_carried": carried, "opaque_bytes": meta.get("opaque_bytes"),
                                  "finish_reason": resp.finish_reason}
        if stripped:
            detail["message"] = ("the host rejected include: reasoning cannot be carried across turns here, so the Responses client "
                                 "has no advantage over Chat Completions")
            return False, detail
        if not carried:
            detail["message"] = ("include was accepted but the reply holds no encrypted reasoning item: nothing can be replayed"
                                 if not _cut_off(resp) else _inconclusive("a reasoning item was complete", cap))
            return (None if _cut_off(resp) else False), detail
        return True, detail

    def tool_roundtrip_replay() -> tuple[bool | None, dict]:
        history = [ChatMessage("user", "Use the add_numbers tool to add 2 and 3.")]
        first, _ = run.chat(rc, history, tools=[_ADD_TOOL], max_completion_tokens=cap, reasoning_effort="minimal")
        calls = first.message.tool_calls
        m1 = meta_of(first)
        detail: dict[str, Any] = {"tool_calls": len(calls), "finish_reason": first.finish_reason,
                                  "reasoning_items_carried": m1.get("reasoning_items_carried")}
        if not calls:
            detail["message"] = (_inconclusive("the tool call was written", cap) if _cut_off(first) else
                                 "the model answered in text instead of calling the tool, so no tool turn was exercised")
            return None, detail
        if calls[0].parse_error is not None and _cut_off(first):
            detail["message"] = _inconclusive("the tool-call arguments were complete", cap)
            return None, detail
        history += [first.message] + [ChatMessage("tool", "5", tool_call_id=c.id) for c in calls]
        second, _ = run.chat(rc, history, tools=[_ADD_TOOL], max_completion_tokens=cap, reasoning_effort="minimal")
        m2 = meta_of(second)
        replayed = int(m2.get("replayed_reasoning_items") or 0)
        answered = bool(second.text.strip()) or bool(second.message.tool_calls)
        detail.update(replayed_reasoning_items=replayed, replay_dropped=bool(m2.get("replay_dropped")),
                      replay_disabled=m2.get("replay_disabled"), second_finish_reason=second.finish_reason,
                      second_has_text=bool(second.text.strip()), second_has_tool_calls=bool(second.message.tool_calls),
                      store_false_sent=True)
        if not m1.get("reasoning_items_carried"):
            detail["message"] = "the first turn returned no encrypted reasoning item, so there was nothing to replay"
            return False, detail
        if m2.get("replay_dropped"):
            detail["message"] = ("the server rejected the replayed reasoning item; the client retried without it. Reasoning is NOT "
                                 "carried across turns on this endpoint")
            return False, detail
        if not answered and _cut_off(second):
            detail["message"] = _inconclusive("the tool-result turn was answered", cap)
            return None, detail
        return replayed >= 1 and answered, detail

    def structured() -> tuple[bool | None, dict]:
        resp, _ = run.chat(rc, [ChatMessage("user", "Answer with the sum of 2 and 3 and the unit 'count'.")],
                           response_schema=_SCHEMA, schema_name="smoke_answer", max_completion_tokens=cap, reasoning_effort="minimal")
        try:
            parsed = json.loads(resp.text)
        except ValueError:
            parsed = None
        ok = isinstance(parsed, dict) and set(parsed) == set(_SCHEMA["properties"]) and isinstance(parsed["answer"], int) \
            and isinstance(parsed["unit"], str)
        detail = {"json_parsed": parsed is not None, "types_match": bool(ok), "finish_reason": resp.finish_reason}
        if not ok and parsed is None and _cut_off(resp):
            detail["message"] = _inconclusive("the JSON answer was complete", cap)
            return None, detail
        return bool(ok), detail

    def streamed() -> tuple[bool | None, dict]:
        resp, _ = run.chat(rc, [ChatMessage("user", _READY)], stream=True, max_completion_tokens=cap, reasoning_effort="minimal")
        meta = meta_of(resp)
        has_text, estimated = bool(resp.text.strip()), bool(meta.get("usage_estimated"))
        detail: dict[str, Any] = {"streamed": bool(meta.get("streamed")), "has_text": has_text, "usage_estimated": estimated,
                                  "reasoning_tokens": meta.get("reasoning_tokens"), "finish_reason": resp.finish_reason}
        if not has_text and _cut_off(resp):
            detail["message"] = _inconclusive("any text was streamed", cap)
            return None, detail
        if estimated:
            detail["problem"] = "the stream's terminal event carried no usage: streamed calls would be charged an estimate"
        return bool(meta.get("streamed")) and has_text and not estimated, detail

    go("responses_plain", plain)
    go("responses_include_encrypted", include_encrypted)
    go("responses_tool_roundtrip_replay", tool_roundtrip_replay)
    go("responses_structured", structured)
    go("responses_stream", streamed)

    failed = [p["name"] for p in entries if p["status"] == "fail"]
    inconclusive = [p["name"] for p in entries if p["status"] == "inconclusive"]
    if failed:
        status = "fail"
        recommendation = f'chat only: keep api = "chat" (Responses probes failed: {", ".join(failed)})'
    elif inconclusive:
        status = "inconclusive"
        recommendation = (f'chat only: the Responses probes were inconclusive ({", ".join(inconclusive)}); re-run with a larger '
                          f'max_completion_tokens before setting api = "responses"')
    else:
        status = "pass"
        recommendation = 'responses OK: set api = "responses" for the supervisor in nightshift.toml'
    base.update(tested=True, status=status, probes=entries, failed=failed, inconclusive=inconclusive, recommendation=recommendation,
                endpoint=getattr(rc, "base_url", None) or getattr(chat_client, "base_url", None),
                models={"requested": getattr(rc, "model", None), "answered": sorted(run.models)},
                usage={"calls": run.calls, "prompt_tokens": run.prompt, "completion_tokens": run.completion,
                       "cached_tokens": run.cached, "estimated_usd": round(run.usd, 6)})
    base.pop("reason", None)
    return base


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_client_factory(*, model: str | None = None, base_url: str | None = None,
                         env: Mapping[str, str] | None = None) -> Callable[[str | None], MetaChatClient]:
    """A factory for the probes. Precedence for the default model and base URL: argument, then ``NIGHTSHIFT_MODEL`` /
    ``NIGHTSHIFT_BASE_URL``, then the built-in default. Raises ``ModelConfigError`` now if there is no key or the URL is
    unsafe. Retries are limited so a dead endpoint fails quickly."""
    env = os.environ if env is None else env
    key = resolve_api_key(env)
    url = base_url or (env.get("NIGHTSHIFT_BASE_URL") or "").strip() or DEFAULT_BASE_URL
    default_model = model or (env.get("NIGHTSHIFT_MODEL") or "").strip() or DEFAULT_MODEL

    def factory(wanted: str | None = None) -> MetaChatClient:
        return MetaChatClient(wanted or default_model, base_url=url, api_key=key, timeout=60.0, max_retries=2)

    def responses_factory(wanted: str | None = None) -> Any:
        from .responses import MetaResponsesClient     # lazy: a chat-only smoke run never imports it
        return MetaResponsesClient(wanted or default_model, base_url=url, api_key=key, timeout=60.0, max_retries=2)

    factory()  # validate eagerly (URL, key shape)
    factory.responses_factory = responses_factory  # type: ignore[attr-defined]  # run_smoke finds it here
    return factory


def summarize(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for the CLI (no content, no key). A failed or inconclusive probe is followed by its message."""
    labels = {"pass": "PASS", "inconclusive": "????"}
    lines: list[str] = []
    for p in report["probes"]:
        status = p.get("status") or ("pass" if p["ok"] else "fail")
        label = labels.get(status) or ("FAIL" if p["required"] else "info")
        lines.append(f"{label}  {p['name']:<30} {p['latency_ms']:>8.1f} ms")
        note = (p.get("detail") or {}).get("message") or (p.get("detail") or {}).get("problem")
        if note and status != "pass":
            lines.append(f"      {note}")
    lines.append(f"models answered: {', '.join(report['models']['answered']) or '-'}; cached tokens reported: "
                 f"{'yes' if report['cached_known'] else 'no'}; wonder on chat completions: {report['wonder_chat_completions']}")
    resp = report.get("responses")
    if isinstance(resp, Mapping) and resp.get("status") not in (None, "skipped"):     # also when the client could not be built
        for p in resp.get("probes", ()):
            status = p.get("status") or "fail"
            lines.append(f"{labels.get(status) or 'info'}  {p['name']:<30} {p['latency_ms']:>8.1f} ms")
            note = (p.get("detail") or {}).get("message") or (p.get("detail") or {}).get("problem")
            if note and status != "pass":
                lines.append(f"      {note}")
        lines.append(str(resp.get("recommendation")))
        lines.append(f"note: {resp.get('untested_note')}")
    overall = "OK" if report["ok"] else "FAILED (" + ", ".join(report["required_failed"]) + ")"
    if report.get("required_inconclusive"):
        overall += " [inconclusive: " + ", ".join(report["required_inconclusive"]) + "]"
    lines.append("overall: " + overall)
    return lines
