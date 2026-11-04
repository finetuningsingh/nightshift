"""Shared helpers for chat-completion clients. Pure functions, no I/O, standard library only.

Guarantees
----------
* ``strictify_schema`` returns a new schema in the strict structured-output subset (``docs/META_API_NOTES.md``) or
  raises ``ValueError`` naming the offending path. It never mutates its input and is idempotent.
* ``messages_to_wire`` produces the OpenAI-shaped message list and refuses (``ValueError``) a history in which an
  assistant message with ``tool_calls`` is not followed by exactly one ``tool`` message per call, in order, because the
  API answers that with HTTP 400.
* ``parse_sse`` / ``StreamAssembler`` reassemble streamed content and tool-call argument fragments by ``index``.
* ``OpaqueItems`` is the private carrier of the Responses client's encrypted reasoning items on an assistant
  ``ChatMessage``. It is never iterable, never JSON-serialisable and prints only a byte length and a hash, so it cannot reach a
  prompt, a dashboard, a report or the ledger by accident. The Chat Completions helpers ignore it.
* Nothing here ever sees the API key.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..types import ChatMessage, ModelError, NightshiftError, ToolCall, ToolSpec, Usage

EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
MAX_SCHEMA_DEPTH = 10
MAX_STREAM_BYTES = 32 * 1024 * 1024
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_.-]+$")
_SCHEMA_NAME_BAD = re.compile(r"[^a-zA-Z0-9_-]")
_JSON_TYPES = {"string", "number", "integer", "boolean", "null", "object", "array"}
# Error codes that /docs/error-handling documents as transient 503s. Delivered inside a stream they mean "re-send the whole request".
STREAM_RETRYABLE_CODES = frozenset({"server_shutting_down", "service_overloaded", "backend_unavailable"})
_UNSUPPORTED_KEYWORDS = ("oneOf", "not", "if", "then", "else", "patternProperties", "prefixItems",
                         "dependentSchemas", "unevaluatedProperties", "propertyNames")


class ModelConfigError(NightshiftError, ValueError):
    """A client was configured wrongly (no API key, unsafe base URL). Raised at construction, never during a call."""


def is_wonder(model: str) -> bool:
    return str(model).startswith("muse-wonder-")


def supports_max_effort(model: str) -> bool:
    """Whether ``reasoning_effort: "max"`` is documented for ``model`` (``/docs/reasoning``, ``/docs/models``).

    Supported by ``muse-wonder-1.0`` and ``muse-spark-1.3``; explicitly not by ``muse-spark-1.3-contributor``. Any other
    id (older Spark releases, partner-hosted models) is treated as unsupported because the docs do not promise it.
    """
    name = str(model).strip()
    if is_wonder(name):
        return True
    return name.startswith("muse-spark-1.3") and "contributor" not in name


# --------------------------------------------------------------------------------------------------
# reasoning effort
# --------------------------------------------------------------------------------------------------
def normalize_reasoning_effort(effort: str | None, model: str) -> str | None:
    """Map a requested effort to the wire value. ``None`` stays ``None`` (parameter omitted).

    ``none`` becomes ``minimal`` (the API answers ``none`` with 400); ``max`` is valid for ``muse-wonder-*`` and
    ``muse-spark-1.3`` but not for ``muse-spark-1.3-contributor`` or models the docs do not list (``supports_max_effort``).
    """
    if effort is None:
        return None
    if not isinstance(effort, str):
        raise ValueError(f"reasoning_effort must be a string, got {type(effort).__name__}")
    value = effort.strip().lower()
    if value == "none":
        return "minimal"
    if value == "max":
        if supports_max_effort(model):
            return "max"
        raise ValueError(f"reasoning_effort 'max' is only valid for muse-wonder-* and muse-spark-1.3, not {model!r}")
    if value in EFFORTS:
        return value
    raise ValueError(f"reasoning_effort must be one of none, {', '.join(EFFORTS)}, max; got {effort!r}")


# --------------------------------------------------------------------------------------------------
# strict JSON schema
# --------------------------------------------------------------------------------------------------
def strictify_schema(schema: Mapping[str, Any]) -> dict:
    """Normalise ``schema`` to the strict structured-output subset.

    Every object gets ``additionalProperties: false`` and ``required`` = all its property keys (an optional property
    therefore becomes required; declare it nullable if ``null`` should be allowed). Local ``$ref`` to ``$defs`` /
    ``definitions`` is inlined and ``allOf`` of object schemas is merged. Raises ``ValueError`` for ``oneOf`` and the other
    unsupported keywords, recursion, free-form objects (``additionalProperties`` other than false), arrays without
    ``items``, untyped nodes, a non-object root, and nesting deeper than 10.
    """
    if not isinstance(schema, Mapping):
        raise ValueError("schema must be a mapping")
    root = copy.deepcopy(dict(schema))
    defs: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        found = root.pop(key, None)
        if isinstance(found, Mapping):
            defs.update(found)
    out = _strict_node(root, defs, 1, (), "$")
    if out.get("type") != "object":
        raise ValueError("$: the root of a strict schema must be a plain object (type 'object')")
    return out


def _resolve_ref(node: dict, defs: Mapping[str, Any], stack: tuple[str, ...], path: str) -> tuple[dict, tuple[str, ...]]:
    ref = node.pop("$ref")
    match = re.fullmatch(r"#/(?:\$defs|definitions)/([^/]+)", ref) if isinstance(ref, str) else None
    if not match:
        raise ValueError(f"{path}: only local '#/$defs/<name>' references are supported, got {ref!r}")
    name = match.group(1)
    if name in stack:
        raise ValueError(f"{path}: recursive schema reference to {name!r} is not supported")
    if name not in defs or not isinstance(defs[name], Mapping):
        raise ValueError(f"{path}: unresolved reference {ref!r}")
    target = copy.deepcopy(dict(defs[name]))
    target.update(node)  # siblings such as 'description' win
    return target, stack + (name,)


def _merge_all_of(node: dict, defs: Mapping[str, Any], stack: tuple[str, ...], path: str) -> dict:
    branches = node.pop("allOf")
    if not isinstance(branches, list) or not branches:
        raise ValueError(f"{path}: allOf must be a non-empty list")
    merged: dict[str, Any] = {k: v for k, v in node.items()}
    props: dict[str, Any] = dict(merged.get("properties") or {})
    required: list[str] = list(merged.get("required") or [])
    for i, branch in enumerate(branches):
        if not isinstance(branch, Mapping):
            raise ValueError(f"{path}.allOf[{i}]: must be an object schema")
        part = dict(branch)
        sub_stack = stack
        while "$ref" in part:
            part, sub_stack = _resolve_ref(part, defs, sub_stack, f"{path}.allOf[{i}]")
        if "allOf" in part:
            part = _merge_all_of(part, defs, sub_stack, f"{path}.allOf[{i}]")
        if part.get("type", "object") != "object" or "anyOf" in part:
            raise ValueError(f"{path}.allOf[{i}]: only allOf of plain object schemas can be flattened")
        for k, v in (part.get("properties") or {}).items():
            if k in props and props[k] != v:
                raise ValueError(f"{path}.allOf[{i}]: property {k!r} is defined twice with different schemas")
            props[k] = v
        required.extend(part.get("required") or [])
        for k, v in part.items():
            if k not in ("properties", "required", "type"):
                merged.setdefault(k, v)
    merged["type"] = "object"
    merged["properties"] = props
    merged["required"] = list(dict.fromkeys(required))
    return merged


def _strict_node(node: Any, defs: Mapping[str, Any], depth: int, stack: tuple[str, ...], path: str) -> dict:
    if not isinstance(node, Mapping):
        raise ValueError(f"{path}: a schema node must be an object, got {type(node).__name__}")
    if depth > MAX_SCHEMA_DEPTH:
        raise ValueError(f"{path}: nesting deeper than {MAX_SCHEMA_DEPTH} levels is not supported")
    node = dict(node)
    while "$ref" in node:
        node, stack = _resolve_ref(node, defs, stack, path)
    for keyword in _UNSUPPORTED_KEYWORDS:
        if keyword in node:
            raise ValueError(f"{path}: '{keyword}' is not supported by strict structured output")
    if "allOf" in node:
        node = _merge_all_of(node, defs, stack, path)

    if "anyOf" in node:
        branches = node["anyOf"]
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"{path}: anyOf must be a non-empty list")
        node["anyOf"] = [_strict_node(b, defs, depth + 1, stack, f"{path}.anyOf[{i}]") for i, b in enumerate(branches)]
        return node

    declared = node.get("type")
    types = declared if isinstance(declared, list) else ([declared] if declared is not None else [])
    for t in types:
        if t not in _JSON_TYPES:
            raise ValueError(f"{path}: unknown type {t!r}")
    if declared is None and "properties" in node:
        types = ["object"]
        node["type"] = "object"
    if not types and "enum" not in node and "const" not in node:
        raise ValueError(f"{path}: every schema node needs a 'type' (or 'enum'/'const'/'anyOf') in strict mode")

    if "object" in types:
        props = node.get("properties", {})
        if not isinstance(props, Mapping):
            raise ValueError(f"{path}.properties must be an object")
        extra = node.get("additionalProperties", False)
        if extra is not False:
            raise ValueError(f"{path}: free-form objects are not supported (additionalProperties must be false)")
        listed = node.get("required", [])
        if not isinstance(listed, list) or any(k not in props for k in listed):
            raise ValueError(f"{path}: 'required' names a property that is not declared")
        node["properties"] = {k: _strict_node(v, defs, depth + 1, stack, f"{path}.{k}") for k, v in props.items()}
        node["required"] = list(node["properties"])
        node["additionalProperties"] = False
    if "array" in types:
        items = node.get("items")
        if not isinstance(items, Mapping):
            raise ValueError(f"{path}: an array needs a single 'items' schema")
        node["items"] = _strict_node(items, defs, depth + 1, stack, f"{path}[]")
    return node


def clean_schema_name(name: str | None) -> str:
    """A legal ``json_schema.name`` (letters, digits, ``_``, ``-``; at most 64 characters)."""
    cleaned = _SCHEMA_NAME_BAD.sub("_", name or "")[:64]
    return cleaned or "response"


# --------------------------------------------------------------------------------------------------
# messages and tools
# --------------------------------------------------------------------------------------------------
def parse_arguments(raw: Any) -> tuple[dict, str | None]:
    """Tool-call arguments to ``(dict, parse_error)``. Empty text is ``{}`` with no error; anything else that is not a
    JSON object yields ``({}, reason)``."""
    if isinstance(raw, Mapping):
        return dict(raw), None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if not isinstance(raw, str):
        return {}, f"arguments have type {type(raw).__name__}, expected a JSON string"
    try:
        value = json.loads(raw)
    except ValueError as exc:
        return {}, f"invalid JSON: {exc}"
    if not isinstance(value, dict):
        return {}, f"arguments must be a JSON object, got {type(value).__name__}"
    return value, None


def tool_calls_from_wire(wire: Any) -> list[ToolCall]:
    """OpenAI ``tool_calls`` to ``ToolCall``s. Invalid argument JSON becomes ``arguments={}`` plus ``parse_error``."""
    calls: list[ToolCall] = []
    if not isinstance(wire, list):
        return calls
    seen: set[str] = set()
    for i, item in enumerate(wire):
        if not isinstance(item, Mapping):
            continue
        fn = item.get("function") if isinstance(item.get("function"), Mapping) else {}
        raw = fn.get("arguments")
        args, error = parse_arguments(raw)
        raw_text = raw if isinstance(raw, str) else (json.dumps(raw) if isinstance(raw, Mapping) else "")
        # Every tool result is matched to its call by id, so ids must be unique even if the server repeats one.
        call_id, n = str(item.get("id") or f"call_{i}"), 1
        while call_id in seen:
            call_id = f"{item.get('id') or f'call_{i}'}_{n}"
            n += 1
        seen.add(call_id)
        calls.append(ToolCall(id=call_id, name=str(fn.get("name") or ""), arguments=args,
                              raw_arguments=raw_text, parse_error=error))
    return calls


def _wire_arguments(call: ToolCall) -> str:
    """Argument text for a replayed assistant message: the model's own text if it is a valid object (keeps the prompt
    prefix byte-stable for caching), else a re-serialisation, else ``{}``. Never invalid JSON."""
    if call.raw_arguments and parse_arguments(call.raw_arguments) == (call.arguments, None):
        return call.raw_arguments
    if call.arguments:
        return json.dumps(call.arguments, ensure_ascii=False)
    if call.raw_arguments and parse_arguments(call.raw_arguments)[1] is None:
        return call.raw_arguments
    return "{}"


def message_to_wire(msg: ChatMessage) -> dict:
    role = msg.role
    if role in ("system", "developer", "user"):
        return {"role": role, "content": msg.content or ""}
    if role == "assistant":
        if msg.tool_calls:
            return {"role": "assistant", "content": msg.content,
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": _wire_arguments(c)}}
                                   for c in msg.tool_calls]}
        return {"role": "assistant", "content": msg.content or ""}
    if role == "tool":
        if not msg.tool_call_id:
            raise ValueError("a tool message needs tool_call_id")
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content or ""}
    raise ValueError(f"unknown message role {role!r}")


def validate_tool_sequence(messages: Sequence[ChatMessage]) -> None:
    """Raise ``ValueError`` unless every assistant ``tool_calls`` is answered by one tool message per call, in order."""
    pending: list[str] = []
    for i, m in enumerate(messages):
        if pending:
            if m.role != "tool" or m.tool_call_id != pending[0]:
                raise ValueError(f"message {i}: expected the tool result for call {pending[0]!r} next "
                                 f"(every tool call needs exactly one tool message, in order)")
            pending.pop(0)
            continue
        if m.role == "tool":
            raise ValueError(f"message {i}: tool message {m.tool_call_id!r} does not follow an assistant tool call")
        if m.role == "assistant" and m.tool_calls:
            ids = [c.id for c in m.tool_calls]
            if not all(ids) or len(set(ids)) != len(ids):
                raise ValueError(f"message {i}: tool calls need unique non-empty ids")
            pending = ids
    if pending:
        raise ValueError(f"tool call(s) {pending} were never answered by a tool message")


def messages_to_wire(messages: Sequence[ChatMessage]) -> list[dict]:
    if not messages:
        raise ValueError("at least one message is required")
    validate_tool_sequence(messages)
    return [message_to_wire(m) for m in messages]


def tools_to_wire(tools: Sequence[ToolSpec]) -> list[dict]:
    """Nested ``{"type":"function","function":{...}}`` form. Names must match ``^[a-zA-Z0-9_.-]+$`` with at most one dot."""
    wire, seen = [], set()
    for tool in tools:
        name = tool.name
        if not isinstance(name, str) or not _TOOL_NAME.match(name) or name.count(".") > 1:
            raise ValueError(f"invalid tool name {name!r}: use letters, digits, '_', '-' and at most one '.'")
        if name in seen:
            raise ValueError(f"duplicate tool name {name!r}")
        seen.add(name)
        params = tool.parameters if tool.parameters else {"type": "object", "properties": {}}
        wire.append({"type": "function", "function": {"name": name, "description": tool.description or "",
                                                      "parameters": params}})
    return wire


def text_from_content(content: Any) -> str:
    """Message ``content`` as text: a string, or the ``text`` parts of a content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, Mapping) and isinstance(p.get("text"), str))
    return ""


# --------------------------------------------------------------------------------------------------
# usage and errors
# --------------------------------------------------------------------------------------------------
def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value)) if math.isfinite(value) else 0


def usage_from_wire(wire: Any) -> tuple[Usage, bool]:
    """``(Usage, cached_known)``. Cached tokens come from ``prompt_tokens_details.cached_tokens`` or ``cached_tokens``;
    when neither field exists they are 0 and ``cached_known`` is False."""
    u = wire if isinstance(wire, Mapping) else {}
    prompt, completion = _count(u.get("prompt_tokens")), _count(u.get("completion_tokens"))
    details = u.get("prompt_tokens_details")
    cached_raw = None
    if isinstance(details, Mapping) and "cached_tokens" in details:
        cached_raw = details["cached_tokens"]
    elif "cached_tokens" in u:
        cached_raw = u["cached_tokens"]
    known = isinstance(cached_raw, (int, float)) and not isinstance(cached_raw, bool)
    total = _count(u.get("total_tokens")) or prompt + completion
    return Usage(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=_count(cached_raw) if known else 0,
                 total_tokens=total), known


def reasoning_tokens_from_wire(wire: Any) -> int | None:
    """``usage.completion_tokens_details.reasoning_tokens`` or None when the server does not report it. Reasoning tokens
    are a subset of ``completion_tokens`` (``/docs/token-counting``), so they are already charged; this is for display."""
    details = wire.get("completion_tokens_details") if isinstance(wire, Mapping) else None
    value = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    return _count(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def estimate_tokens(text: str) -> int:
    """Rough token count (4 characters per token) used only when a reply reports no usage at all. It counts visible text
    only, so it under-counts hidden reasoning tokens; a streamed request therefore asks for real usage (include_usage)."""
    return max(1, len(text) // 4) if text else 0


def parse_error_envelope(body: bytes | str | None, limit: int = 600) -> dict:
    """Parse ``{"error": {"message","type","param","code"}}``. Always returns the four keys; ``message`` falls back to a
    capped snippet of the body when it is not an envelope."""
    out: dict[str, Any] = {"message": None, "type": None, "param": None, "code": None}
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    text = (body or "").strip()
    err: Any = None
    try:
        parsed = json.loads(text) if text else None
        err = parsed.get("error") if isinstance(parsed, Mapping) else None
        if isinstance(parsed, Mapping) and err is None and "message" in parsed:
            err = parsed
    except ValueError:
        pass
    if isinstance(err, Mapping):
        for key in out:
            val = err.get(key)
            out[key] = None if val is None else str(val)
    elif isinstance(err, str):
        out["message"] = err
    if out["message"] is None and text:
        out["message"] = re.sub(r"\s+", " ", text)
    for key in ("message", "type", "param", "code"):
        if out[key] is not None:
            out[key] = out[key][:limit]
    return out


# --------------------------------------------------------------------------------------------------
# Responses API: reasoning carried on assistant messages, usage
# --------------------------------------------------------------------------------------------------
REPLAY_ATTR = "_responses_items"


class OpaqueItems:
    """The structural output items of ONE Responses turn that must be replayed with it: encrypted ``reasoning`` items, the
    ``phase`` of the commentary message and the ids of the ``function_call`` items. Attached to an assistant ``ChatMessage`` as
    the private attribute ``_responses_items`` by ``MetaResponsesClient``; the message text, tool names and arguments stay in
    the ``ChatMessage`` (the single source of truth for what the model said).

    Model output is untrusted data, so the carrier is deliberately inert: it is not iterable, ``json.dumps`` raises
    ``TypeError`` on it, and ``repr``/``str`` show only the item count, byte length and a short hash. Read the items through
    ``.items`` (code that replays them) and nothing else. ``len(carrier) == 0`` marks "reasoning dropped on purpose" (history
    compaction), which is different from an attribute that is absent.
    """

    __slots__ = ("model", "items", "nbytes", "digest")

    def __init__(self, items: Sequence[Mapping[str, Any]] = (), *, model: str = "") -> None:
        frozen = tuple(copy.deepcopy(dict(i)) for i in items if isinstance(i, Mapping))
        blob = json.dumps(frozen, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8") if frozen else b""
        object.__setattr__(self, "model", str(model))
        object.__setattr__(self, "items", frozen)
        object.__setattr__(self, "nbytes", len(blob))
        object.__setattr__(self, "digest", hashlib.sha256(blob).hexdigest()[:12] if blob else "")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("OpaqueItems is immutable")

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("OpaqueItems is opaque: read .items only where the items are replayed")

    def __repr__(self) -> str:
        return f"<opaque response items: {len(self.items)} items, {self.nbytes} bytes, sha256:{self.digest or '-'}>"

    __str__ = __repr__

    def __reduce__(self) -> tuple:
        return (_rebuild_opaque, (self.items, self.model))

    def __deepcopy__(self, memo: dict) -> "OpaqueItems":
        return OpaqueItems(self.items, model=self.model)

    def __copy__(self) -> "OpaqueItems":
        return self


def _rebuild_opaque(items: Sequence[Mapping[str, Any]], model: str) -> OpaqueItems:
    return OpaqueItems(items, model=model)


_UNSET = object()


def carried_items(msg: Any) -> OpaqueItems | None | object:
    """The ``OpaqueItems`` attached to ``msg``; ``None`` when something else is stored there; ``UNSET`` when the attribute
    is absent (a message that never passed through a Responses client, or one rebuilt by a role)."""
    value = getattr(msg, REPLAY_ATTR, _UNSET)
    if value is _UNSET:
        return _UNSET
    return value if isinstance(value, OpaqueItems) else None


UNSET = _UNSET


def carries_reasoning(msg: Any) -> bool:
    """True when ``msg`` has a non-empty ``OpaqueItems`` attached."""
    value = carried_items(msg)
    return isinstance(value, OpaqueItems) and bool(value)


def carry_items(source: Any, target: Any) -> Any:
    """Copy the carrier from ``source`` to ``target`` (both ``ChatMessage``s) when ``source`` has one, and return ``target``.
    Roles use it when they rebuild an assistant message for their history (they do not append the returned object itself)."""
    value = carried_items(source)
    if value is not UNSET and value is not None:
        setattr(target, REPLAY_ATTR, value)
    return target


def drop_items(msg: Any) -> Any:
    """A copy of ``msg`` that carries NO reasoning (explicitly empty, so no client replays anything for it). Used by history
    compaction for the turns it no longer keeps verbatim. The input is not modified."""
    out = copy.copy(msg)
    setattr(out, REPLAY_ATTR, OpaqueItems((), model=""))
    return out


def usage_from_responses(wire: Any) -> tuple[Usage, bool]:
    """``(Usage, cached_known)`` from a Responses ``usage`` object: ``input_tokens`` / ``output_tokens`` /
    ``input_tokens_details.cached_tokens``. Same shape and meaning as ``usage_from_wire``: reasoning tokens are part of
    ``output_tokens`` (``/docs/api-reference/responses``: commentary counts toward ``output_tokens`` but not
    ``reasoning_tokens``), so ``completion_tokens`` is already the billed output."""
    u = wire if isinstance(wire, Mapping) else {}
    prompt, completion = _count(u.get("input_tokens")), _count(u.get("output_tokens"))
    details = u.get("input_tokens_details")
    cached_raw = details.get("cached_tokens") if isinstance(details, Mapping) else None
    known = isinstance(cached_raw, (int, float)) and not isinstance(cached_raw, bool)
    total = _count(u.get("total_tokens")) or prompt + completion
    return Usage(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=_count(cached_raw) if known else 0,
                 total_tokens=total), known


def reasoning_tokens_from_responses(wire: Any) -> int | None:
    """``usage.output_tokens_details.reasoning_tokens`` or None when the server does not report it (display only)."""
    details = wire.get("output_tokens_details") if isinstance(wire, Mapping) else None
    value = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    return _count(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# --------------------------------------------------------------------------------------------------
# server-sent events
# --------------------------------------------------------------------------------------------------
def parse_sse(chunks: Iterable[bytes | str], *, max_bytes: int = MAX_STREAM_BYTES) -> Iterator[dict]:
    """Yield each ``data:`` payload of an SSE stream as a JSON object, stopping at ``[DONE]``.

    ``chunks`` may be split anywhere (including inside a line or a ``\\r\\n`` pair). Comments and non-data fields are
    ignored; multiple ``data:`` lines of one event are joined with newlines. Raises ``ValueError`` on a payload that is
    not a JSON object or when the stream exceeds ``max_bytes``.
    """
    buf = b""
    data: list[str] = []
    total = 0

    def dispatch() -> Iterator[dict | None]:
        if not data:
            return
        payload = "\n".join(data)
        data.clear()
        if payload.strip() == "[DONE]":
            yield None
            return
        try:
            obj = json.loads(payload)
        except ValueError as exc:
            raise ValueError(f"malformed SSE event: {exc}") from None
        if not isinstance(obj, dict):
            raise ValueError("malformed SSE event: payload is not a JSON object")
        yield obj

    def handle(line: str) -> Iterator[dict | None]:
        if line == "":
            yield from dispatch()
        elif line.startswith(":"):
            return
        else:
            field, _, value = line.partition(":")
            if field == "data":
                data.append(value[1:] if value.startswith(" ") else value)

    for chunk in chunks:
        raw = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
        total += len(raw)
        if total > max_bytes:
            raise ValueError(f"SSE stream exceeds {max_bytes} bytes")
        buf += raw
        while True:
            m = re.search(rb"\r\n|\n|\r", buf)
            if m is None or (m.group() == b"\r" and m.end() == len(buf)):
                break  # no complete line yet (a trailing \r may be half of \r\n)
            line, buf = buf[:m.start()].decode("utf-8", errors="replace"), buf[m.end():]
            for event in handle(line):
                if event is None:
                    return
                yield event
    if buf:
        for event in handle(buf.decode("utf-8", errors="replace").rstrip("\r")):
            if event is None:
                return
            yield event
    for event in dispatch():
        if event is None:
            return
        yield event


class StreamAssembler:
    """Fold streamed chunks into one OpenAI-shaped response dict (choice 0 only).

    Content deltas are concatenated; tool-call fragments are grouped by ``index`` (name and id taken from the first
    fragment that carries them, argument text concatenated in arrival order). A terminal error event raises
    ``ModelError`` (see ``_raise_on_error``); a usage-only chunk (empty ``choices``, sent last when the request set
    ``stream_options.include_usage``) is recorded without ending the reply.
    """

    def __init__(self) -> None:
        self._content: list[str] = []
        self._calls: dict[int, dict[str, Any]] = {}
        self._last_index: int | None = None
        self.finish_reason: str | None = None
        self.usage: Mapping | None = None
        self.model: str | None = None
        self.id: str | None = None
        self.chunks = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def feed(self, chunk: Mapping[str, Any]) -> None:
        self.chunks += 1
        self._raise_on_error(chunk)
        self.model = chunk.get("model") or self.model
        self.id = chunk.get("id") or self.id
        if isinstance(chunk.get("usage"), Mapping):
            self.usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            return
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice.get("delta"), Mapping) else {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            self._content.append(content)
        for frag in delta.get("tool_calls") or []:
            if isinstance(frag, Mapping):
                self._add_fragment(frag)
        if choice.get("finish_reason"):
            self.finish_reason = str(choice["finish_reason"])

    @staticmethod
    def _raise_on_error(chunk: Mapping[str, Any]) -> None:
        """Raise ``ModelError`` if ``chunk`` is a terminal error event.

        /docs/error-handling documents mid-stream errors only for the Responses stream (``type: "error"`` with ``code`` and
        ``message`` at the top level, or ``response.failed`` carrying ``response.error``); the Chat Completions chunk shape
        for an error is not documented. All three shapes are therefore accepted, plus the OpenAI-style ``{"error": {...}}``
        chunk. An error whose code is in ``STREAM_RETRYABLE_CODES`` (``server_shutting_down``, ``service_overloaded``,
        ``backend_unavailable``) is marked retryable: the caller re-sends the whole request, as the docs prescribe.
        """
        kind = chunk.get("type")
        err: Any = chunk.get("error")
        if not err and kind == "error":
            err = {k: chunk.get(k) for k in ("message", "param", "code") if chunk.get(k) is not None}
            err["type"] = "error"
        if not err and kind == "response.failed":
            resp = chunk.get("response")
            err = resp.get("error") if isinstance(resp, Mapping) else None
            err = err or {"message": "the response failed", "type": "response.failed"}
        if not err:
            return
        info = parse_error_envelope(json.dumps({"error": err if isinstance(err, Mapping) else {"message": str(err)}}))
        label = f" ({info['type']})" if info["type"] else ""
        code = (info["code"] or "").lower()
        retryable = code in STREAM_RETRYABLE_CODES or (info["type"] or "").lower() in STREAM_RETRYABLE_CODES
        raise ModelError(f"error inside the stream{label}: {info['message'] or 'unknown'}",
                         error_type=info["type"], code=info["code"], retryable=retryable)

    def _add_fragment(self, frag: Mapping[str, Any]) -> None:
        index = frag.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            if frag.get("id") or self._last_index is None:
                index = (max(self._calls) + 1) if self._calls else 0
            else:
                index = self._last_index
        self._last_index = index
        call = self._calls.setdefault(index, {"id": "", "name": "", "args": []})
        fn = frag.get("function") if isinstance(frag.get("function"), Mapping) else {}
        if frag.get("id") and not call["id"]:
            call["id"] = str(frag["id"])
        if fn.get("name") and not call["name"]:
            call["name"] = str(fn["name"])
        if isinstance(fn.get("arguments"), str):
            call["args"].append(fn["arguments"])

    def to_response(self) -> dict:
        calls = [{"id": c["id"] or f"call_{i}", "type": "function",
                  "function": {"name": c["name"], "arguments": "".join(c["args"])}}
                 for i, c in sorted(self._calls.items())]
        text = "".join(self._content)
        message: dict[str, Any] = {"role": "assistant", "content": text if (text or not calls) else None}
        if calls:
            message["tool_calls"] = calls
        out: dict[str, Any] = {"id": self.id, "model": self.model, "object": "chat.completion",
                               "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}]}
        if self.usage is not None:
            out["usage"] = dict(self.usage)
        return out
