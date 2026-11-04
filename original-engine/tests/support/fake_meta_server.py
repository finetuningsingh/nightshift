"""An in-process fake of the Meta Model API (Chat Completions) for tests. Real loopback sockets, standard library only.

    with FakeMetaServer() as fake:                      # ThreadingHTTPServer on 127.0.0.1, port chosen by the OS
        client = MetaChatClient(base_url=fake.base_url, api_key=fake.api_key)
        fake.enqueue(Fault.rate_limit(retry_after=1))   # first request: 429 with Retry-After
        fake.enqueue(Reply.text("hello", prompt_tokens=10, completion_tokens=2, cached_tokens=4))
        ...
        fake.requests            # every request received, rejected ones included (RecordedRequest)
        fake.last_request.body   # the parsed JSON body

It behaves as ``docs/META_API_NOTES.md`` describes the real API:

* ``POST /v1/chat/completions`` with ``Authorization: Bearer <api_key>``. A wrong or missing key gets **401 whose body echoes
  the Authorization header value** (so tests can prove the client never repeats it).
* Rejected with 400: ``stop``, ``logit_bias``, ``prediction``, ``verbosity``, ``n > 1``, ``logprobs: true``,
  ``web_search_options``, ``modalities``, ``audio``, ``text``, ``max_output_tokens``; ``tool_choice`` other than ``"auto"``;
  ``reasoning_effort`` outside minimal..xhigh (``"none"`` too; ``"max"`` is accepted for ``muse-wonder-*`` and
  ``muse-spark-1.3`` and rejected for ``-contributor`` and older Spark, per ``/docs/reasoning``); unknown top-level
  parameters; any name in ``unsupported_params`` (simulates a partner host without e.g. ``prompt_cache_key``);
  ``max_completion_tokens`` above ``max_completion_tokens_ceiling`` (the documented "more than the model's configured maximum"
  400; the real maximum is not documented, so there is none unless a test sets one).
* Messages: an assistant message with ``tool_calls`` must be followed by exactly one ``tool`` message per call, in order.
  Tool names must match ``^[a-zA-Z0-9_.-]+$`` with at most one dot.
* ``response_format`` json_schema ``strict`` must be in the strict subset (plain object root, ``additionalProperties:false``
  and ``required`` = all keys on every object, no ``allOf``/``oneOf``/``$ref``, depth <= 10), else 400.
* Unknown model -> 404 ``model_not_found`` (restrict ``models`` to test fallback). ``stream: true`` answers with SSE.
* A stream carries ``usage`` ONLY when the request set ``stream_options.include_usage`` (then as a last chunk with empty
  ``choices``), exactly as the docs describe; ``stream_usage=False`` drops it even then (an interrupted stream).
* Message ``content`` is accepted as a string or as an array of ``{"type": "text", "text": ...}`` parts for system / developer /
  user messages; an assistant message may have ``content`` null, a string or parts.
* Responses always carry ``message.refusal: null`` and ``message.reasoning_content: ""`` (redacted for external callers).
* Usage is reported with a cached-token breakdown according to ``cache_report``: ``"details"``
  (``usage.prompt_tokens_details.cached_tokens``), ``"flat"`` (``usage.cached_tokens``) or ``"none"``, and with
  ``usage.completion_tokens_details.reasoning_tokens`` unless ``report_reasoning_tokens=False``. Reasoning tokens are part of
  ``completion_tokens``. A repeated request with the same ``prompt_cache_key`` and messages reports half the prompt as cached.
* ``simulate_reasoning=N`` (or ``{effort: N, "*": default}``) makes the "model" reason for N hidden tokens before it answers,
  all of which count against ``max_completion_tokens``: a cap at or below N yields ``finish_reason: "length"`` with empty
  content, a cap inside the answer yields a truncated answer, like the real API.
* ``rejected_schema_keywords={"minLength", ...}`` makes strict ``response_format`` schemas and tool parameters that use one of
  those JSON-Schema keywords fail with a 400 naming the keyword (the real subset is unknown; the live smoke probes learn it).
* ``ratelimit_headers=True`` adds ``x-ratelimit-limit/remaining-tokens/requests`` to successful responses.

``POST /v1/responses`` (the Responses API, ``docs/META_API_NOTES.md`` and ``nightshift/models/responses.py``) is served by the same
queue, handler and fault machinery. What it enforces, from the documentation of ``/docs/protocols/responses``:

* ``input`` is a string or typed items (``message`` with ``input_text`` / ``output_text`` blocks, ``function_call``,
  ``function_call_output``, ``reasoning``); ``instructions``; flat function ``tools`` (the nested Chat form is a 400);
  ``text.format`` (a ``response_format`` is a 400); ``reasoning.effort`` (``none`` is a 400; ``max`` per model); ``include`` (only
  ``reasoning.encrypted_content``; other values, and ``include`` with ``previous_response_id``, are a 400); ``store``;
  ``max_output_tokens`` (at least 16; above ``max_completion_tokens_ceiling`` is a 400); ``stream_options`` is accepted and has no
  effect (usage always arrives in the terminal event). ``stop``, ``verbosity``, ``logit_bias``, ``prediction``, ``n > 1``,
  ``logprobs``, ``web_search_options``, ``modalities``, ``audio`` and unknown parameters (``messages``, ``max_completion_tokens`` ...) are a 400.
* Each ``function_call_output`` must answer a ``function_call`` of the same request (``call_id`` 1-64 characters), in order; a
  ``function_call`` left unanswered is a 400 (the docs do not say; OpenAI does, and so does this fake).
* A ``reasoning`` item must be followed by an assistant message or a ``function_call`` before any user/system/developer message,
  else 400 with ``param: "input"``. An assistant message directly followed by a ``function_call`` must have ``phase: "commentary"``.
* Reasoning is ENCRYPTED and VERIFIED. A response to a request with ``include: ["reasoning.encrypted_content"]`` carries a
  ``reasoning`` item whose ``encrypted_content`` this server issued; replaying an item it did not issue (or one without a blob
  whose id it never stored) is a 400 "Referenced reasoning item ... was not found or has expired". ``reasoning_model_bound=True``
  also rejects a blob replayed to a different model (an assumption: the docs do not say). ``fake.issued_reasoning`` lists what was
  issued and ``fake.replays`` records, per accepted request, the reasoning items that were replayed, so a test can PROVE replay.
* Output: ``reasoning`` (when ``emit_reasoning``), ``message`` (``phase: "commentary"`` when tool calls follow; a ``Reply.refusal``
  becomes a ``refusal`` block) and ``function_call`` items; ``Reply.extra_output`` appends further items (unknown types).
  ``usage`` is ``input_tokens`` / ``output_tokens`` / ``input_tokens_details.cached_tokens`` / ``output_tokens_details.reasoning_tokens``.
  ``finish_reason: "length"`` becomes ``status: "incomplete"`` with ``incomplete_details.reason: "max_output_tokens"``.
* Streams are the documented event sequence (``response.created`` ... ``response.output_item.*``, ``response.output_text.delta``,
  ``response.function_call_arguments.delta``, ``response.completed``/``response.incomplete``, ``[DONE]``); ``Fault.stream_error``
  becomes an ``error`` event. ``stream_usage=False`` drops ``usage`` from the terminal event.
* ``serve_responses=False`` makes ``/v1/responses`` answer 404 (a host that does not implement the Responses API).
* A handler set with ``set_handler`` receives a CHAT-SHAPED view of a Responses request (``messages``, nested ``tools``,
  ``response_format``, ``reasoning_effort``, ``max_completion_tokens``) so a chat brain such as ``SimBrain`` serves both endpoints;
  queued callables and ``enqueue``d items see the request as received (``RecordedRequest.endpoint`` tells them apart).

Default replies (empty queue): tools offered and no tool result yet -> a call to the first tool with sample arguments;
``response_format`` -> JSON that satisfies the schema; otherwise the text ``"ok"``.

Scripting: ``enqueue(*items)`` consumes one item per request, in order. Items:

* ``Fault`` (a raw HTTP failure: ``Fault.rate_limit``, ``server_error``, ``gateway_timeout``, ``unauthorized``,
  ``bad_request``, ``error``, ``redirect``, ``disconnect``, ``garbage``). A Fault at the head of the queue is returned
  *before* request validation, like a gateway would.
* ``Reply`` (a successful completion: ``Reply.text``, ``Reply.tools``, ``Reply.json``, ``Reply.from_openai``), rendered as JSON
  or SSE to match the request. Validation runs first; a request that fails validation gets its 400 and the Reply stays queued.
* a dict (an OpenAI-shaped response body) or a callable ``(RecordedRequest) -> Fault | Reply | dict``.

``set_handler(fn)`` replaces the default reply policy. ``non_stream_status = 504`` makes every non-streaming request fail
with that status while streamed ones succeed (the "504 then stream" behaviour). ``reset()`` clears queue and history.
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
import threading
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

DEFAULT_API_KEY = "sk-test-1234567890abcdefghijkl"
KNOWN_MODELS = ("muse-spark-1.3", "muse-spark-1.2", "muse-spark-1.1", "muse-spark-1.3-contributor",
                "muse-spark-1.2-contributor", "muse-wonder-1.0")
FORBIDDEN_PARAMS = ("stop", "logit_bias", "prediction", "verbosity", "web_search_options", "modalities", "audio", "text",
                    "max_output_tokens")
KNOWN_PARAMS = {"model", "messages", "tools", "parallel_tool_calls", "tool_choice", "response_format", "reasoning_effort",
                "max_completion_tokens", "max_tokens", "temperature", "top_p", "prompt_cache_key",
                "prompt_cache_retention", "safety_identifier", "stream", "stream_options", "user", "n", "logprobs",
                "top_logprobs", "store", "seed", "metadata", "service_tier", "frequency_penalty", "presence_penalty",
                } | set(FORBIDDEN_PARAMS)
EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
RESPONSES_FORBIDDEN = ("stop", "logit_bias", "prediction", "verbosity", "web_search_options", "modalities", "audio")
RESPONSES_KNOWN = {"model", "input", "instructions", "previous_response_id", "tools", "tool_choice", "text", "reasoning", "include",
                   "store", "stream", "stream_options", "background", "max_output_tokens", "max_tool_calls", "truncation",
                   "temperature", "top_p", "top_logprobs", "frequency_penalty", "presence_penalty", "prompt_cache_key",
                   "prompt_cache_retention", "safety_identifier", "user", "parallel_tool_calls", "service_tier", "metadata",
                   "context_management", "prompt", "client_metadata", "n", "logprobs"} | set(RESPONSES_FORBIDDEN)
REASONING_INCLUDE = "reasoning.encrypted_content"
_USE_REQUEST_MODEL = object()


def supports_max_effort(model: str) -> bool:
    """Independent restatement of ``/docs/reasoning``: max is for muse-wonder-1.0 and muse-spark-1.3, not -contributor."""
    return model.startswith("muse-wonder-") or (model.startswith("muse-spark-1.3") and "contributor" not in model)


# --------------------------------------------------------------------------------------------------
# scripting objects
# --------------------------------------------------------------------------------------------------
@dataclass
class RecordedRequest:
    seq: int
    method: str
    path: str
    headers: dict[str, str]                  # lower-case names
    raw: bytes
    body: dict | None = None                 # parsed JSON object, None if it was not one
    stream: bool = False
    status: int | None = None                # status the server answered with
    note: str = ""                           # why it was rejected / which scripted item answered
    endpoint: str = "chat"                   # "chat" (/v1/chat/completions) or "responses" (/v1/responses)

    @property
    def authorization(self) -> str:
        return self.headers.get("authorization", "")


@dataclass
class Fault:
    """A raw HTTP failure."""

    status: int
    body: Any = None                         # dict/list -> JSON, str/bytes sent as is
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    disconnect: bool = False                 # close the socket without answering
    echo_authorization: bool = False         # 401: put the request's Authorization header value in the body

    @staticmethod
    def error(status: int, message: str, *, type: str = "invalid_request_error", param: str | None = None,
              code: str | None = None, headers: dict[str, str] | None = None) -> "Fault":
        return Fault(status, {"error": {"message": message, "type": type, "param": param, "code": code}},
                     dict(headers or {}))

    @staticmethod
    def rate_limit(retry_after: float | None = None) -> "Fault":
        headers = {"Retry-After": f"{retry_after:g}"} if retry_after is not None else {}
        return Fault.error(429, "Rate limit exceeded", type="rate_limit_error", headers=headers)

    @staticmethod
    def server_error(status: int = 503) -> "Fault":
        message = {503: "The server is temporarily unavailable", 502: "Bad gateway"}.get(status, "Internal server error")
        return Fault.error(status, message, type="server_error")

    @staticmethod
    def gateway_timeout() -> "Fault":
        return Fault.error(504, "Gateway timeout", type="timeout_error", code="gateway_timeout")

    @staticmethod
    def stream_error(code: str = "server_shutting_down", message: str = "The server is shutting down") -> "Fault":
        """A terminal error EVENT inside a 200 stream (the HTTP status is already sent). The Chat Completions shape of such an
        event is not documented; this uses the OpenAI-style ``{"error": {...}}`` chunk. Streamed requests only: a
        non-streaming request gets the same error as an HTTP 503."""
        return Fault(-1, {"error": {"message": message, "type": "server_error", "param": None, "code": code}})

    @staticmethod
    def unauthorized() -> "Fault":
        return Fault(401, None, echo_authorization=True)

    @staticmethod
    def bad_request(message: str, param: str | None = None) -> "Fault":
        return Fault.error(400, message, param=param)

    @staticmethod
    def redirect(location: str, status: int = 302) -> "Fault":
        return Fault(status, "", {"Location": location})

    @staticmethod
    def disconnect_now() -> "Fault":
        return Fault(0, disconnect=True)

    @staticmethod
    def garbage(text: str = "<html>not json</html>", status: int = 200) -> "Fault":
        return Fault(status, text)

    @staticmethod
    def slow(seconds: float) -> "Fault":
        """Hold the answer for ``seconds`` (pair with a short client timeout); the request then continues as normal."""
        return Fault(0, delay=seconds)


@dataclass
class Reply:
    """A successful completion. Rendered as a JSON body or as SSE chunks depending on the request's ``stream``."""

    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)      # {"name", "arguments": str, "id"?}
    finish_reason: str | None = None
    model: Any = _USE_REQUEST_MODEL                            # None omits the response's "model" field
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None                        # hidden reasoning; part of completion_tokens
    usage: dict | None = None                                  # complete override of the usage object
    delay: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)
    refusal: str | None = None                                 # Responses only: the message holds a refusal block
    extra_output: list[dict] = field(default_factory=list)     # Responses only: further output items (e.g. unknown types)

    @staticmethod
    def text(text: str, **kw: Any) -> "Reply":
        return Reply(content=text, **kw)

    @staticmethod
    def json(obj: Any, **kw: Any) -> "Reply":
        return Reply(content=json.dumps(obj), **kw)

    @staticmethod
    def tools(*calls: tuple, text: str | None = None, **kw: Any) -> "Reply":
        """``calls`` are ``(name, args)`` or ``(name, args, id)``; ``args`` may be a dict or a (possibly invalid) JSON string."""
        built = []
        for i, call in enumerate(calls):
            name, args = call[0], call[1]
            built.append({"name": name, "arguments": args if isinstance(args, str) else json.dumps(args),
                          "id": call[2] if len(call) > 2 else f"call_fake{i}"})
        return Reply(content=text, tool_calls=built, **kw)

    @staticmethod
    def from_openai(body: Mapping[str, Any]) -> "Reply":
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = [{"name": (c.get("function") or {}).get("name", ""), "arguments": (c.get("function") or {}).get("arguments", ""),
                  "id": c.get("id")} for c in msg.get("tool_calls") or []]
        usage = body.get("usage")
        return Reply(content=msg.get("content"), tool_calls=calls, finish_reason=choice.get("finish_reason"),
                     model=body.get("model", _USE_REQUEST_MODEL), usage=dict(usage) if isinstance(usage, Mapping) else None)


# --------------------------------------------------------------------------------------------------
# schema helpers (independent of nightshift.models.base so they can catch its bugs)
# --------------------------------------------------------------------------------------------------
def strict_schema_error(schema: Any, depth: int = 1, path: str = "schema") -> str | None:
    """None if ``schema`` is in the strict subset, else the reason it is not."""
    if not isinstance(schema, dict):
        return f"{path} must be an object"
    if depth > 10:
        return f"{path}: nesting is deeper than 10"
    for bad in ("allOf", "oneOf", "$ref", "$defs", "definitions"):
        if bad in schema:
            return f"{path}: '{bad}' is not supported in strict mode"
    if "anyOf" in schema:
        for i, branch in enumerate(schema["anyOf"]):
            problem = strict_schema_error(branch, depth + 1, f"{path}.anyOf[{i}]")
            if problem:
                return problem
        return None
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "object" in kinds:
        props = schema.get("properties")
        if not isinstance(props, dict):
            return f"{path}: an object needs 'properties'"
        if schema.get("additionalProperties") is not False:
            return f"{path}: additionalProperties must be false"
        if sorted(schema.get("required") or []) != sorted(props):
            return f"{path}: 'required' must list every property"
        for key, sub in props.items():
            problem = strict_schema_error(sub, depth + 1, f"{path}.{key}")
            if problem:
                return problem
    if "array" in kinds:
        problem = strict_schema_error(schema.get("items"), depth + 1, f"{path}[]")
        if problem:
            return problem
    return None


_NOT_KEYWORDS = ("enum", "const", "default", "examples", "required")


def find_schema_keyword(node: Any, keywords: set[str]) -> str | None:
    """The first of ``keywords`` used as a JSON-Schema keyword anywhere in ``node`` (property NAMES do not count)."""
    if isinstance(node, list):
        return next((hit for item in node if (hit := find_schema_keyword(item, keywords))), None)
    if not isinstance(node, dict):
        return None
    for key, value in node.items():
        if key in keywords:
            return key
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            hit = next((h for sub in value.values() if (h := find_schema_keyword(sub, keywords))), None)
        elif key in _NOT_KEYWORDS:
            hit = None
        else:
            hit = find_schema_keyword(value, keywords)
        if hit:
            return hit
    return None


def sample_from_schema(schema: Any) -> Any:
    """A small value that satisfies ``schema``."""
    if not isinstance(schema, dict):
        return None
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema and schema["anyOf"]:
        return sample_from_schema(schema["anyOf"][0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object" or (kind is None and "properties" in schema):
        return {k: sample_from_schema(v) for k, v in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [sample_from_schema(schema.get("items"))]
    return {"string": "sample", "integer": 1, "number": 0.5, "boolean": True, "null": None}.get(kind)


def _valid_content(content: Any) -> bool:
    """A string, or a non-empty array of ``{"type": "text", "text": str}`` parts."""
    if isinstance(content, str):
        return True
    return (isinstance(content, list) and bool(content)
            and all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str) for p in content))


def _estimate(text: str) -> int:
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------------------------------
class FakeMetaServer:
    def __init__(self, api_key: str = DEFAULT_API_KEY, *, models: tuple[str, ...] | set[str] = KNOWN_MODELS,
                 unsupported_params: tuple[str, ...] | set[str] = (), cache_report: str = "details",
                 stream_usage: bool = True, enforce_auth: bool = True, report_reasoning_tokens: bool = True,
                 simulate_reasoning: int | Mapping[str, int] | None = None,
                 max_completion_tokens_ceiling: int | Mapping[str, int] | None = None, ceiling_message_states_limit: bool = True,
                 ratelimit_headers: bool = False, rejected_schema_keywords: tuple[str, ...] | set[str] = (),
                 emit_reasoning: bool = True, reasoning_model_bound: bool = False, reasoning_blob_bytes: int = 96,
                 serve_responses: bool = True):
        if cache_report not in ("details", "flat", "none"):
            raise ValueError("cache_report must be 'details', 'flat' or 'none'")
        self.api_key = api_key
        self.models = set(models)
        self.unsupported_params = set(unsupported_params)
        self.cache_report = cache_report
        self.stream_usage = stream_usage
        self.enforce_auth = enforce_auth
        self.report_reasoning_tokens = report_reasoning_tokens
        self.simulate_reasoning = simulate_reasoning
        self.max_completion_tokens_ceiling = max_completion_tokens_ceiling
        self.ceiling_message_states_limit = ceiling_message_states_limit
        self.ratelimit_headers = ratelimit_headers
        self.rejected_schema_keywords = set(rejected_schema_keywords)      # e.g. {"minLength"}: simulates a stricter schema subset
        self.serve_responses = serve_responses                     # False: /v1/responses answers 404 like a host without the API
        self.emit_reasoning = emit_reasoning                       # Responses: every reply starts with a reasoning item
        self.reasoning_model_bound = reasoning_model_bound         # Responses: a blob only decrypts for the model that issued it
        self.reasoning_blob_bytes = max(16, reasoning_blob_bytes)
        self.issued_reasoning: list[dict] = []                     # {"seq", "id", "token", "model"} for every reasoning item issued
        self.replays: list[dict] = []                              # {"seq", "ids", "count"} per accepted request that replayed reasoning
        self._issued_tokens: dict[str, str] = {}
        self._stored_reasoning: set[str] = set()
        self.ratelimit_limit_requests, self.ratelimit_limit_tokens = 3000, 4_000_000
        self._tokens_used = 0
        self.non_stream_status: int | None = None
        self._lock = threading.Lock()
        self._queue: list[Any] = []
        self._requests: list[RecordedRequest] = []
        self._cache_seen: set[tuple] = set()
        self._handler: Callable[[RecordedRequest], Any] | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    # ---- lifecycle ------------------------------------------------------------------------------------
    def start(self) -> "FakeMetaServer":
        fake = self

        class Handler(_Handler):
            owner = fake

        server = _Server(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        self._server, self.port = server, server.server_address[1]
        self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> "FakeMetaServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    # ---- scripting API ----------------------------------------------------------------------------------
    def enqueue(self, *items: Any) -> None:
        with self._lock:
            self._queue.extend(items)

    def set_handler(self, fn: Callable[[RecordedRequest], Any] | None) -> None:
        self._handler = fn

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self._requests.clear()
            self._cache_seen.clear()
            self._tokens_used = 0
            self.issued_reasoning.clear()
            self.replays.clear()
            self._issued_tokens.clear()
            self._stored_reasoning.clear()
        self.non_stream_status = None

    @property
    def requests(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self._requests)

    @property
    def last_request(self) -> RecordedRequest | None:
        with self._lock:
            return self._requests[-1] if self._requests else None

    @property
    def queued(self) -> int:
        with self._lock:
            return len(self._queue)

    def bodies(self, endpoint: str | None = "chat") -> list[dict]:
        """Parsed bodies of the requests that had one. By default only the Chat Completions requests (the tests that predate the
        Responses endpoint read them as chat bodies); ``endpoint="responses"`` gives the Responses bodies, ``None`` both."""
        return [r.body for r in self.requests if r.body is not None and (endpoint is None or r.endpoint == endpoint)]

    # ---- request validation (the documented API rules) ------------------------------------------------------
    def validate(self, body: dict) -> Fault | None:
        def bad(message: str, param: str | None = None, code: str | None = None) -> Fault:
            return Fault.error(400, message, param=param, code=code)

        for name in FORBIDDEN_PARAMS:
            if name in body:
                return bad(f"Unsupported parameter: '{name}'", name)
        if "n" in body and body["n"] != 1:
            return bad("Unsupported value for 'n': only n=1 is supported", "n")
        if body.get("logprobs"):
            return bad("Unsupported parameter: 'logprobs'", "logprobs")
        if "tool_choice" in body and body["tool_choice"] != "auto":
            return bad("Unsupported value for 'tool_choice': only 'auto' is supported", "tool_choice")
        for name in sorted(self.unsupported_params):
            if name in body:
                return bad(f"Unrecognized request argument supplied: {name}", name)
        for name in body:
            if name not in KNOWN_PARAMS:
                return bad(f"Unknown parameter: '{name}'", name)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return bad("'model' is required", "model")
        if model not in self.models:
            return Fault.error(404, f"The model '{model}' does not exist", type="not_found_error", param="model",
                               code="model_not_found")
        effort = body.get("reasoning_effort")
        if effort is not None:
            allowed = EFFORTS + (("max",) if supports_max_effort(model) else ())
            if effort not in allowed:
                return bad(f"Invalid value for reasoning_effort: {effort!r}. Supported values: {', '.join(allowed)}",
                           "reasoning_effort")
        if "temperature" in body and not (isinstance(body["temperature"], (int, float)) and 0 <= body["temperature"] <= 2):
            return bad("temperature must be a number between 0 and 2", "temperature")
        if "safety_identifier" in body and not (isinstance(body["safety_identifier"], str) and 0 < len(body["safety_identifier"]) <= 64):
            return bad("safety_identifier must be a string of at most 64 characters", "safety_identifier")
        if "prompt_cache_retention" in body and body["prompt_cache_retention"] not in ("in_memory", "24h"):
            return bad("prompt_cache_retention must be 'in_memory' or '24h'", "prompt_cache_retention")
        if "prompt_cache_key" in body and not isinstance(body["prompt_cache_key"], str):
            return bad("prompt_cache_key must be a string", "prompt_cache_key")
        mct = body.get("max_completion_tokens")
        if mct is not None and not (isinstance(mct, int) and not isinstance(mct, bool) and mct > 0):
            return bad("max_completion_tokens must be a positive integer", "max_completion_tokens")
        ceiling = self.max_completion_tokens_ceiling
        ceiling = ceiling.get(model) if isinstance(ceiling, Mapping) else ceiling
        if mct is not None and ceiling is not None and mct > ceiling:
            detail = f" of {ceiling}" if self.ceiling_message_states_limit else ""
            return bad(f"Requested max_completion_tokens is more than the model's configured maximum{detail}",
                       "max_completion_tokens")
        stream_options = body.get("stream_options")
        if stream_options is not None and not (isinstance(stream_options, dict)
                                               and isinstance(stream_options.get("include_usage", False), bool)):
            return bad("stream_options must be an object whose include_usage is a boolean", "stream_options")
        problem = self._check_messages(body.get("messages"))
        if problem:
            return bad(problem, "messages")
        problem = self._check_tools(body.get("tools"))
        if problem:
            return bad(problem, "tools")
        rf = body.get("response_format")
        if rf is not None:
            if not isinstance(rf, dict) or rf.get("type") not in ("json_schema", "json_object", "text"):
                return bad("invalid response_format", "response_format")
            if rf["type"] == "json_schema":
                js = rf.get("json_schema")
                if not isinstance(js, dict) or not isinstance(js.get("name"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", js["name"]):
                    return bad("response_format.json_schema.name is invalid", "response_format")
                if js.get("strict") is True:
                    problem = strict_schema_error(js.get("schema"))
                    if problem:
                        return bad(f"Invalid schema for response_format: {problem}", "response_format")
                    hit = find_schema_keyword(js.get("schema"), self.rejected_schema_keywords)
                    if hit:
                        return bad(f"Invalid schema for response_format: '{hit}' is not supported in strict mode", "response_format")
        return None

    @staticmethod
    def _check_messages(messages: Any) -> str | None:
        if not isinstance(messages, list) or not messages:
            return "'messages' must be a non-empty list"
        pending: list[str] = []
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or m.get("role") not in ("system", "developer", "user", "assistant", "tool"):
                return f"messages[{i}] has an invalid role"
            role = m["role"]
            if pending:
                if role != "tool" or m.get("tool_call_id") != pending[0]:
                    return (f"messages[{i}]: an assistant message with 'tool_calls' must be followed by tool messages "
                            f"responding to each 'tool_call_id' in order (expected {pending[0]!r})")
                pending.pop(0)
                continue
            if role == "tool":
                return f"messages[{i}]: a tool message must follow an assistant message with tool_calls"
            if role == "assistant" and m.get("tool_calls"):
                for j, call in enumerate(m["tool_calls"]):
                    fn = call.get("function") if isinstance(call, dict) else None
                    if (not isinstance(fn, dict) or call.get("type") != "function" or not call.get("id")
                            or not isinstance(fn.get("name"), str) or not isinstance(fn.get("arguments"), str)):
                        return f"messages[{i}].tool_calls[{j}] is malformed"
                    pending.append(call["id"])
            elif role == "assistant":
                content = m.get("content")
                if content is not None and not isinstance(content, (str, list)):
                    return f"messages[{i}].content must be a string, an array of parts or null"
                if isinstance(content, list) and not all(isinstance(p, dict) and p.get("type") in ("text", "refusal")
                                                         for p in content):
                    return f"messages[{i}].content has an invalid part"
            elif not _valid_content(m.get("content")):
                return f"messages[{i}].content must be a string or an array of text parts"
        if pending:
            return "the conversation ends with tool calls that have no tool message"
        return None

    def _check_tools(self, tools: Any) -> str | None:
        if tools is None:
            return None
        if not isinstance(tools, list) or not tools:
            return "'tools' must be a non-empty list"
        for i, tool in enumerate(tools):
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(fn, dict):
                return f"tools[{i}] must use the nested form {{'type':'function','function':{{...}}}}"
            name = fn.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or name.count(".") > 1:
                return f"tools[{i}].function.name {name!r} is invalid"
            if not isinstance(fn.get("parameters", {}), dict):
                return f"tools[{i}].function.parameters must be an object"
            hit = find_schema_keyword(fn.get("parameters"), self.rejected_schema_keywords)
            if hit:
                return f"tools[{i}].function.parameters: '{hit}' is not supported"
        return None

    # ---- reply construction -------------------------------------------------------------------------------------
    def _default_reply(self, req: RecordedRequest) -> Reply:
        body = req.body or {}
        tools = body.get("tools")
        has_result = any(m.get("role") == "tool" for m in body.get("messages", []))
        if tools and not has_result:
            fn = tools[0]["function"]
            return Reply.tools((fn["name"], sample_from_schema(fn.get("parameters") or {})))
        rf = body.get("response_format") or {}
        if rf.get("type") == "json_schema":
            return Reply.json(sample_from_schema(rf["json_schema"].get("schema")))
        return Reply.text("ok")

    def _usage_numbers(self, reply: Reply, body: dict) -> tuple[int, int, int, int]:
        """``(prompt, completion, cached, reasoning)`` token counts of a reply; counted once towards the rate-limit headers."""
        prompt = reply.prompt_tokens if reply.prompt_tokens is not None else _estimate(json.dumps(body.get("messages", [])))
        out_text = (reply.content if isinstance(reply.content, str) else "") + "".join(c["name"] + c["arguments"] for c in reply.tool_calls)
        reasoning = reply.reasoning_tokens or 0
        completion = reply.completion_tokens if reply.completion_tokens is not None else (
            (_estimate(out_text) if out_text else 0) + reasoning)
        cached = reply.cached_tokens
        if cached is None:
            key = body.get("prompt_cache_key")
            fingerprint = (key, json.dumps(body.get("messages", []), sort_keys=True))
            with self._lock:
                cached = prompt // 2 if key and fingerprint in self._cache_seen else 0
                if key:
                    self._cache_seen.add(fingerprint)
        with self._lock:
            self._tokens_used += prompt + completion
        return prompt, completion, cached, reasoning

    def _usage(self, reply: Reply, body: dict) -> dict:
        if reply.usage is not None:
            return reply.usage
        prompt, completion, cached, reasoning = self._usage_numbers(reply, body)
        usage: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
        if self.cache_report == "details":
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        elif self.cache_report == "flat":
            usage["cached_tokens"] = cached
        if self.report_reasoning_tokens:
            usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
        return usage

    def _usage_responses(self, reply: Reply, view: dict) -> dict:
        """The Responses ``usage`` object. ``Reply.usage`` (a Chat-shaped override) is converted; ``cache_report="none"`` omits the
        cached-token detail, ``report_reasoning_tokens=False`` the reasoning detail."""
        if reply.usage is not None:
            u = reply.usage
            prompt, completion = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            details = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
            cached, reasoning = int(details.get("cached_tokens") or u.get("cached_tokens") or 0), reply.reasoning_tokens or 0
        else:
            prompt, completion, cached, reasoning = self._usage_numbers(reply, view)
        usage: dict[str, Any] = {"input_tokens": prompt, "output_tokens": completion, "total_tokens": prompt + completion}
        if self.cache_report != "none":
            usage["input_tokens_details"] = {"cached_tokens": cached}
        if self.report_reasoning_tokens:
            usage["output_tokens_details"] = {"reasoning_tokens": reasoning}
        return usage

    def _reasoned(self, reply: Reply, body: dict) -> Reply:
        """Apply ``simulate_reasoning``: hidden reasoning tokens come out of ``max_completion_tokens`` before any visible text."""
        budget = self.simulate_reasoning
        if isinstance(budget, Mapping):
            budget = budget.get(body.get("reasoning_effort") or "*", budget.get("*"))
        if budget is None or reply.usage is not None or reply.finish_reason == "length":
            return reply
        cap = body.get("max_completion_tokens")
        if cap is not None and budget >= cap:                       # reasoning ate the whole allowance
            return replace(reply, content="", tool_calls=[], finish_reason="length", reasoning_tokens=cap, completion_tokens=cap)
        out_text = (reply.content or "") + "".join(c["name"] + c["arguments"] for c in reply.tool_calls)
        visible = reply.completion_tokens if reply.completion_tokens is not None else (_estimate(out_text) if out_text else 0)
        if cap is not None and budget + visible > cap:              # the answer itself is cut off
            room = max(0, cap - budget) * 4
            return replace(reply, content=(reply.content or "")[:room], tool_calls=[], finish_reason="length",
                           reasoning_tokens=budget, completion_tokens=cap)
        return replace(reply, reasoning_tokens=budget, completion_tokens=budget + visible)

    def render_json(self, reply: Reply, req: RecordedRequest) -> dict:
        body = req.body or {}
        message: dict[str, Any] = {"role": "assistant", "content": reply.content, "refusal": None, "reasoning_content": ""}
        if reply.tool_calls:
            message["tool_calls"] = [{"id": c.get("id") or f"call_fake{i}", "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for i, c in enumerate(reply.tool_calls)]
        finish = reply.finish_reason or ("tool_calls" if reply.tool_calls else "stop")
        out: dict[str, Any] = {"id": f"chatcmpl-fake{req.seq}", "object": "chat.completion",
                               "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                               "usage": self._usage(reply, body)}
        if reply.model is not None:
            out["model"] = body.get("model") if reply.model is _USE_REQUEST_MODEL else reply.model
        return out

    def render_sse(self, reply: Reply, req: RecordedRequest) -> list[bytes]:
        body = req.body or {}
        model = body.get("model") if reply.model is _USE_REQUEST_MODEL else reply.model
        base: dict[str, Any] = {"id": f"chatcmpl-fake{req.seq}", "object": "chat.completion.chunk"}
        if model is not None:
            base["model"] = model

        def chunk(choice: dict | None, **extra: Any) -> bytes:
            data = dict(base, choices=[dict(choice, index=0)] if choice is not None else [], **extra)
            return b"data: " + json.dumps(data).encode("utf-8") + b"\n\n"

        events = [b": keep-alive comment\n\n",
                  chunk({"delta": {"role": "assistant", "content": "", "reasoning_content": ""}, "finish_reason": None})]
        text = reply.content or ""
        for i in range(0, len(text), 3):
            events.append(chunk({"delta": {"content": text[i:i + 3]}, "finish_reason": None}))
        calls = reply.tool_calls
        for idx, c in enumerate(calls):  # first fragment of each call carries id and name
            events.append(chunk({"delta": {"tool_calls": [{"index": idx, "id": c.get("id") or f"call_fake{idx}", "type": "function",
                                                           "function": {"name": c["name"], "arguments": ""}}]},
                                 "finish_reason": None}))
        pieces = [[c["arguments"][i:i + 5] for i in range(0, len(c["arguments"]), 5)] for c in calls]
        for round_no in range(max((len(p) for p in pieces), default=0)):  # interleave the calls' fragments by index
            for idx, p in enumerate(pieces):
                if round_no < len(p):
                    events.append(chunk({"delta": {"tool_calls": [{"index": idx, "function": {"arguments": p[round_no]}}]},
                                         "finish_reason": None}))
        events.append(chunk({"delta": {}, "finish_reason": reply.finish_reason or ("tool_calls" if calls else "stop")}))
        options = body.get("stream_options")
        if self.stream_usage and isinstance(options, dict) and options.get("include_usage") is True:
            events.append(chunk(None, usage=self._usage(reply, body)))
        events.append(b"data: [DONE]\n\n")
        return events

    def ratelimit_header_values(self, req: RecordedRequest) -> dict[str, str]:
        """The documented ``x-ratelimit-*`` headers of a successful response (empty unless ``ratelimit_headers``)."""
        if not self.ratelimit_headers:
            return {}
        with self._lock:
            used_requests, used_tokens = len(self._requests), self._tokens_used
        return {"x-ratelimit-limit-requests": str(self.ratelimit_limit_requests),
                "x-ratelimit-remaining-requests": str(max(0, self.ratelimit_limit_requests - used_requests)),
                "x-ratelimit-limit-tokens": str(self.ratelimit_limit_tokens),
                "x-ratelimit-remaining-tokens": str(max(0, self.ratelimit_limit_tokens - used_tokens))}

    # ---- the Responses API -------------------------------------------------------------------------------------
    def validate_responses(self, body: dict) -> Fault | None:
        """The documented rules of ``POST /v1/responses`` (see the module docstring). None when the request is acceptable."""
        def bad(message: str, param: str | None = None, code: str | None = None) -> Fault:
            return Fault.error(400, message, param=param, code=code)

        for name in RESPONSES_FORBIDDEN:
            if name in body:
                return bad(f"Unsupported parameter: '{name}'", name)
        if "n" in body and body["n"] != 1:
            return bad("Unsupported value for 'n': only n=1 is supported", "n")
        if body.get("logprobs"):
            return bad("Unsupported parameter: 'logprobs'", "logprobs")
        for name in sorted(self.unsupported_params):
            if name in body:
                return bad(f"Unrecognized request argument supplied: {name}", name)
        for name in body:
            if name not in RESPONSES_KNOWN:
                return bad(f"Unknown parameter: '{name}'", name)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return bad("'model' is required", "model")
        if model not in self.models:
            return Fault.error(404, f"The model '{model}' does not exist", type="not_found_error", param="model",
                               code="model_not_found")
        data = body.get("input")
        if not ((isinstance(data, str) and data) or (isinstance(data, list) and data)):
            return bad("'input' must be a non-empty string or a non-empty array of items", "input")
        if "instructions" in body and not (body["instructions"] is None or isinstance(body["instructions"], str)):
            return bad("'instructions' must be a string", "instructions")
        reasoning = body.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                return bad("'reasoning' must be an object", "reasoning")
            effort = reasoning.get("effort")
            if effort is not None:
                allowed = EFFORTS + (("max",) if supports_max_effort(model) else ())
                if effort not in allowed:
                    return bad(f"Invalid value for reasoning.effort: {effort!r}. Supported values: {', '.join(allowed)}",
                               "reasoning.effort")
        include = body.get("include")
        if include is not None:
            if not isinstance(include, list) or not all(isinstance(i, str) for i in include):
                return bad("'include' must be an array of strings", "include")
            for value in include:
                if value != REASONING_INCLUDE:
                    return bad(f"Invalid value for include: {value!r}", "include")
            if body.get("previous_response_id"):
                return bad("'include' cannot be combined with 'previous_response_id'", "include")
        if body.get("previous_response_id") is not None:
            return bad("This server keeps no responses: previous_response_id is not supported by the fake", "previous_response_id")
        if "store" in body and not isinstance(body["store"], bool):
            return bad("'store' must be a boolean", "store")
        if body.get("truncation", "disabled") != "disabled":
            return bad("Only truncation 'disabled' is supported", "truncation")
        if body.get("background") and body.get("stream"):
            return bad("'background' cannot be combined with 'stream'", "background")
        mot = body.get("max_output_tokens")
        if mot is not None:
            if not (isinstance(mot, int) and not isinstance(mot, bool) and mot >= 16):
                return bad("max_output_tokens must be an integer of at least 16", "max_output_tokens")
            ceiling = self.max_completion_tokens_ceiling
            ceiling = ceiling.get(model) if isinstance(ceiling, Mapping) else ceiling
            if ceiling is not None and mot > ceiling:
                detail = f" of {ceiling}" if self.ceiling_message_states_limit else ""
                return bad(f"Requested max_output_tokens is more than the model's configured maximum{detail}", "max_output_tokens")
        if "temperature" in body and not (isinstance(body["temperature"], (int, float)) and 0 <= body["temperature"] <= 2):
            return bad("temperature must be a number between 0 and 2", "temperature")
        if "safety_identifier" in body and not (isinstance(body["safety_identifier"], str) and 0 < len(body["safety_identifier"]) <= 64):
            return bad("safety_identifier must be a string of at most 64 characters", "safety_identifier")
        if "prompt_cache_retention" in body and body["prompt_cache_retention"] not in ("in_memory", "24h"):
            return bad("prompt_cache_retention must be 'in_memory' or '24h'", "prompt_cache_retention")
        if "prompt_cache_key" in body and not isinstance(body["prompt_cache_key"], str):
            return bad("prompt_cache_key must be a string", "prompt_cache_key")
        so = body.get("stream_options")
        if so is not None and not isinstance(so, dict):
            return bad("stream_options must be an object", "stream_options")
        problem = self._check_response_tools(body.get("tools"))
        if problem:
            return bad(problem, "tools")
        text = body.get("text")
        if text is not None:
            problem = self._check_text_format(text)
            if problem:
                return bad(problem, "text.format")
        return self._check_input(data, model)

    def _check_response_tools(self, tools: Any) -> str | None:
        if tools is None:
            return None
        if not isinstance(tools, list) or not tools:
            return "'tools' must be a non-empty list"
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict) or tool.get("type") != "function":
                return f"tools[{i}].type must be 'function'"
            if "function" in tool:
                return f"tools[{i}] uses the nested Chat Completions form; the Responses form is flat {{'type','name','parameters'}}"
            name = tool.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or name.count(".") > 1:
                return f"tools[{i}].name {name!r} is invalid"
            if not isinstance(tool.get("parameters", {}), dict):
                return f"tools[{i}].parameters must be an object"
            if tool.get("strict") not in (None, True, False):
                return f"tools[{i}].strict must be a boolean"
            if tool.get("strict") is True:
                problem = strict_schema_error(tool.get("parameters"))
                if problem:
                    return f"tools[{i}].parameters: {problem}"
            hit = find_schema_keyword(tool.get("parameters"), self.rejected_schema_keywords)
            if hit:
                return f"tools[{i}].parameters: '{hit}' is not supported"
        return None

    def _check_text_format(self, text: Any) -> str | None:
        if not isinstance(text, dict):
            return "'text' must be an object"
        fmt = text.get("format")
        if fmt is None:
            return None
        if not isinstance(fmt, dict) or fmt.get("type") not in ("json_schema", "json_object", "text"):
            return "invalid text.format"
        if fmt["type"] == "json_schema":
            if not isinstance(fmt.get("name"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", fmt["name"]):
                return "text.format.name is invalid"
            if not isinstance(fmt.get("schema"), dict):
                return "text.format.schema must be an object"
            if fmt.get("strict") is True:
                problem = strict_schema_error(fmt["schema"])
                if problem:
                    return f"Invalid schema for text.format: {problem}"
                hit = find_schema_keyword(fmt["schema"], self.rejected_schema_keywords)
                if hit:
                    return f"Invalid schema for text.format: '{hit}' is not supported in strict mode"
        return None

    def _reasoning_gone(self, item: dict, model: str) -> Fault | None:
        """None when a replayed reasoning item is one this server issued (and, if bound, for this model)."""
        ident = item.get("id") if isinstance(item.get("id"), str) else "(no id)"
        blob = item.get("encrypted_content")
        gone = Fault.error(400, f"Referenced reasoning item '{ident}' was not found or has expired", param="input")
        if isinstance(blob, str):
            issued_for = self._issued_tokens.get(blob)
            if issued_for is None:
                return gone
            if self.reasoning_model_bound and issued_for != model:
                return Fault.error(400, f"The encrypted content of reasoning item '{ident}' cannot be used with model '{model}'",
                                   param="input")
            return None
        return None if (isinstance(item.get("id"), str) and item["id"] in self._stored_reasoning) else gone

    def _check_input(self, data: Any, model: str) -> Fault | None:
        def bad(message: str, param: str = "input") -> Fault:
            return Fault.error(400, message, param=param)

        if isinstance(data, str):
            return None
        pending: list[str] = []            # call ids of the function_call items still waiting for their output
        reasoning_open = False             # a reasoning item that has not yet been followed by an assistant message / function_call
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return bad(f"input[{i}] must be an object")
            kind = item.get("type") or ("message" if "role" in item else None)
            if pending and kind != "function_call_output" and kind != "function_call":
                return bad(f"input[{i}]: function_call {pending[0]!r} has no function_call_output", "call_id")
            if kind == "message":
                role, content = item.get("role"), item.get("content")
                if role not in ("user", "assistant", "system", "developer"):
                    return bad(f"input[{i}].role is invalid")
                if reasoning_open and role != "assistant":
                    return bad(f"input[{i}]: a reasoning item must be followed by an assistant message or a function_call "
                               f"before any {role} message")
                reasoning_open = False
                if isinstance(content, list):
                    want = "output_text" if role == "assistant" else "input_text"
                    for j, block in enumerate(content):
                        ok = isinstance(block, dict) and (block.get("type") == want or (role == "assistant" and block.get("type") == "refusal"))
                        if not ok or (block.get("type") != "refusal" and not isinstance(block.get("text"), str)):
                            return bad(f"input[{i}].content[{j}] must be an {want} block")
                elif not isinstance(content, str):
                    return bad(f"input[{i}].content must be a string or an array of blocks")
                phase = item.get("phase")
                if phase is not None and (role != "assistant" or phase not in ("commentary", "final_answer")):
                    return bad(f"input[{i}].phase is invalid")
                if role == "assistant" and phase != "commentary":
                    nxt = next((x for x in data[i + 1:] if not (isinstance(x, dict) and x.get("type") == "reasoning")), None)
                    if isinstance(nxt, dict) and nxt.get("type") == "function_call":
                        return bad(f"input[{i}]: assistant text before a function_call must have phase 'commentary'")
            elif kind == "function_call":
                cid = item.get("call_id")
                if not (isinstance(cid, str) and 1 <= len(cid) <= 64):
                    return bad(f"input[{i}].call_id must be 1-64 characters", "call_id")
                if not isinstance(item.get("name"), str) or not isinstance(item.get("arguments"), str):
                    return bad(f"input[{i}] needs a name and arguments string")
                pending.append(cid)
                reasoning_open = False
            elif kind == "function_call_output":
                cid = item.get("call_id")
                if reasoning_open:
                    return bad(f"input[{i}]: a reasoning item must be followed by an assistant message or a function_call")
                if not pending or cid != pending[0]:
                    return bad(f"input[{i}]: no matching function_call for call_id {cid!r} (outputs follow their calls in order)",
                               "call_id")
                if not isinstance(item.get("output"), (str, list)):
                    return bad(f"input[{i}].output must be a string or an array")
                pending.pop(0)
            elif kind == "reasoning":
                if not isinstance(item.get("summary"), list):
                    return bad(f"input[{i}].summary is required (send [] when empty)")
                gone = self._reasoning_gone(item, model)
                if gone is not None:
                    return gone
                reasoning_open = True
            else:
                return bad(f"input[{i}]: unsupported item type {kind!r}")
        if pending:
            return bad(f"function_call {pending[0]!r} has no function_call_output", "call_id")
        return None

    @staticmethod
    def _replayed_ids(data: Any) -> list[str]:
        return [str(i.get("id") or i.get("encrypted_content", "")[:12]) for i in data
                if isinstance(i, dict) and i.get("type") == "reasoning"] if isinstance(data, list) else []

    @staticmethod
    def chat_view(body: dict) -> dict:
        """A chat-shaped copy of a (valid) Responses request, for handlers written against Chat Completions."""
        def text_of(content: Any) -> str:
            if isinstance(content, str):
                return content
            return "".join(b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))

        view: dict[str, Any] = {"model": body.get("model")}
        messages: list[dict] = []
        if body.get("instructions"):
            messages.append({"role": "system", "content": body["instructions"]})
        data = body.get("input")
        items = [{"type": "message", "role": "user", "content": data}] if isinstance(data, str) else list(data or [])
        cur: dict | None = None
        for item in items:
            kind = item.get("type") or "message"
            if kind == "message":
                role = item["role"]
                if role == "assistant":
                    cur = {"role": "assistant", "content": text_of(item.get("content"))}
                    messages.append(cur)
                else:
                    cur = None
                    messages.append({"role": role, "content": text_of(item.get("content"))})
            elif kind == "function_call":
                if cur is None:
                    cur = {"role": "assistant", "content": None}
                    messages.append(cur)
                cur.setdefault("tool_calls", []).append({"id": item["call_id"], "type": "function",
                                                         "function": {"name": item["name"], "arguments": item["arguments"]}})
            elif kind == "function_call_output":
                cur = None
                out = item.get("output")
                messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": out if isinstance(out, str) else text_of(out)})
        view["messages"] = messages
        if body.get("tools"):
            view["tools"] = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
                             for t in body["tools"]]
        fmt = (body.get("text") or {}).get("format") if isinstance(body.get("text"), dict) else None
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            view["response_format"] = {"type": "json_schema", "json_schema": {k: fmt[k] for k in ("name", "schema", "strict") if k in fmt}}
        if isinstance(body.get("reasoning"), dict) and body["reasoning"].get("effort"):
            view["reasoning_effort"] = body["reasoning"]["effort"]
        for src, dst in (("max_output_tokens", "max_completion_tokens"), ("temperature", "temperature"),
                         ("prompt_cache_key", "prompt_cache_key"), ("stream", "stream")):
            if src in body:
                view[dst] = body[src]
        return view

    def _issue_reasoning(self, req: RecordedRequest, body: dict, index: int) -> dict:
        """A reasoning output item. It carries an encrypted blob only when the request asked for ``reasoning.encrypted_content``."""
        seq, model = req.seq, body.get("model")
        item: dict[str, Any] = {"type": "reasoning", "id": f"rs_fake{seq}_{index}", "summary": [], "status": "completed"}
        if REASONING_INCLUDE in (body.get("include") or []):
            seed = f"{seq}:{index}:{model}"
            blob = "enc_" + (hashlib.sha256(seed.encode()).hexdigest() * (self.reasoning_blob_bytes // 64 + 1))[:self.reasoning_blob_bytes]
            item["encrypted_content"] = blob
            with self._lock:
                self._issued_tokens[blob] = str(model)
                self.issued_reasoning.append({"seq": seq, "id": item["id"], "token": blob, "model": model})
        if body.get("store", True):
            with self._lock:
                self._stored_reasoning.add(item["id"])
        return item

    def render_response(self, reply: Reply, req: RecordedRequest, view: dict, *, final: bool = True) -> dict:
        """The Responses object for ``reply``. ``final=False`` is the in-progress skeleton of a stream."""
        body = req.body or {}
        model = body.get("model") if reply.model is _USE_REQUEST_MODEL else reply.model
        output: list[dict] = []
        if final:
            if self.emit_reasoning:
                output.append(self._issue_reasoning(req, body, 0))
            text = reply.content if isinstance(reply.content, str) else (
                "".join(p.get("text", "") for p in reply.content if isinstance(p, dict)) if isinstance(reply.content, list) else None)
            has_calls = bool(reply.tool_calls)
            if reply.refusal is not None:
                output.append({"type": "message", "id": f"msg_fake{req.seq}", "role": "assistant", "status": "completed",
                               "content": [{"type": "refusal", "refusal": reply.refusal}]})
            elif text is not None and (text or reply.finish_reason != "length") and (text or not has_calls):
                msg: dict[str, Any] = {"type": "message", "id": f"msg_fake{req.seq}", "role": "assistant", "status": "completed",
                                       "content": [{"type": "output_text", "text": text, "annotations": []}]}
                if has_calls:
                    msg["phase"] = "commentary"
                output.append(msg)
            for i, c in enumerate(reply.tool_calls):
                output.append({"type": "function_call", "id": f"fc_fake{req.seq}_{i}", "call_id": c.get("id") or f"call_fake{i}",
                               "name": c["name"], "arguments": c["arguments"], "status": "completed"})
            output.extend(dict(x) for x in reply.extra_output)
        finish = reply.finish_reason
        incomplete = final and finish in ("length", "content_filter")
        out: dict[str, Any] = {
            "id": f"resp_fake{req.seq}", "object": "response", "created_at": int(time.time()),
            "completed_at": int(time.time()) if final and not incomplete else None,
            "status": ("incomplete" if incomplete else "completed") if final else "in_progress",
            "model": model, "store": bool(body.get("store", True)), "output": output, "error": None,
            "incomplete_details": ({"reason": "max_output_tokens" if finish == "length" else "content_filter"} if incomplete else None),
            "instructions": body.get("instructions"), "tools": body.get("tools") or [], "tool_choice": "auto",
            "parallel_tool_calls": True, "metadata": body.get("metadata"), "temperature": body.get("temperature", 1.0),
            "top_p": 1.0, "max_output_tokens": body.get("max_output_tokens")}
        if model is None:
            out.pop("model")
        if final and (self.stream_usage or not req.stream):
            out["usage"] = self._usage_responses(reply, view)
        return out

    def render_response_sse(self, reply: Reply, req: RecordedRequest, view: dict) -> list[bytes]:
        seq = [0]

        def event(kind: str, **fields: Any) -> bytes:
            data = {"type": kind, **fields, "sequence_number": seq[0]}
            seq[0] += 1
            return f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode("utf-8")

        final = self.render_response(reply, req, view)
        skeleton = self.render_response(reply, req, view, final=False)
        events = [event("response.created", response=skeleton), event("response.in_progress", response=skeleton)]
        for index, item in enumerate(final["output"]):
            started = dict(item, status="in_progress")
            if item["type"] == "function_call":
                started["arguments"] = ""
            elif item["type"] == "message":
                started["content"] = []
            events.append(event("response.output_item.added", output_index=index, item=started))
            if item["type"] == "message":
                block = item["content"][0] if item["content"] else {"type": "output_text", "text": "", "annotations": []}
                events.append(event("response.content_part.added", item_id=item["id"], output_index=index, content_index=0,
                                    part=dict(block, text="") if block["type"] == "output_text" else block))
                if block["type"] == "output_text":
                    text = block["text"]
                    for i in range(0, len(text), 3):
                        events.append(event("response.output_text.delta", item_id=item["id"], output_index=index, content_index=0,
                                            delta=text[i:i + 3], logprobs=[]))
                    events.append(event("response.output_text.done", item_id=item["id"], output_index=index, content_index=0,
                                        text=text, logprobs=[]))
                events.append(event("response.content_part.done", item_id=item["id"], output_index=index, content_index=0, part=block))
            elif item["type"] == "function_call":
                args = item["arguments"]
                for i in range(0, len(args), 5):
                    events.append(event("response.function_call_arguments.delta", item_id=item["id"], output_index=index,
                                        delta=args[i:i + 5]))
                events.append(event("response.function_call_arguments.done", item_id=item["id"], output_index=index,
                                    name=item["name"], arguments=args))
            events.append(event("response.output_item.done", output_index=index, item=item))
        events.append(event("response.incomplete" if final["status"] == "incomplete" else "response.completed", response=final))
        events.append(b"data: [DONE]\n\n")
        return events

    # ---- dispatch (called by the handler thread) ------------------------------------------------------------------
    def dispatch(self, req: RecordedRequest) -> tuple[str, Any]:
        """Return ``("fault", Fault)`` or ``("reply", Reply)`` for a recorded request, updating ``req.note``."""
        if self.enforce_auth and req.authorization != f"Bearer {self.api_key}":
            req.note = "bad authorization"
            return "fault", Fault.unauthorized()
        with self._lock:
            head = self._queue[0] if self._queue else None
            if isinstance(head, Fault):
                self._queue.pop(0)
            else:
                head = None
        if head is not None and head.status == 0 and not head.disconnect:
            time.sleep(head.delay)  # Fault.slow: answer late, then carry on normally
        elif head is not None:
            req.note = "scripted fault"
            return "fault", head
        if req.body is None:
            req.note = "body is not a JSON object"
            return "fault", Fault.bad_request("The request body must be a JSON object")
        if self.non_stream_status and not req.stream:
            req.note = "non-stream status"
            return "fault", Fault.error(self.non_stream_status, "Gateway timeout", type="timeout_error",
                                        code="gateway_timeout" if self.non_stream_status == 504 else None)
        responses = req.endpoint == "responses"
        problem = self.validate_responses(req.body) if responses else self.validate(req.body)
        if problem is not None:
            req.note = "rejected: " + problem.body["error"]["message"]
            return "fault", problem
        view_req = req
        if responses:
            ids = self._replayed_ids(req.body.get("input"))
            if ids:
                with self._lock:
                    self.replays.append({"seq": req.seq, "ids": ids, "count": len(ids)})
            view_req = replace(req, body=self.chat_view(req.body))
        with self._lock:
            item = self._queue.pop(0) if self._queue else None
        from_handler = False
        if item is None and self._handler is not None:
            item, from_handler = self._handler, True
        if callable(item):
            item = item(view_req if from_handler else req)
        if item is None:
            req.note = "default reply"
            item = self._default_reply(view_req)
        else:
            req.note = "scripted reply"
        if isinstance(item, dict):
            item = Reply.from_openai(item)
        if isinstance(item, Fault):
            return "fault", item
        return "reply", self._reasoned(item, view_req.body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256      # the default backlog of 5 resets connections when many lanes call at once


class _Handler(BaseHTTPRequestHandler):
    owner: FakeMetaServer
    server_version = "FakeMeta/1.0"
    disable_nagle_algorithm = True    # headers and body are separate writes; Nagle plus delayed ACK would add ~40 ms per request

    def log_message(self, *args: Any) -> None:  # silence
        pass

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def _handle(self, method: str) -> None:
        owner = self.owner
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(length, 16 * 1024 * 1024)) if length else b""
        body: dict | None
        try:
            parsed = json.loads(raw) if raw else None
            body = parsed if isinstance(parsed, dict) else None
        except ValueError:
            body = None
        with owner._lock:
            seq = len(owner._requests) + 1
            path = self.path.split("?")[0]
            req = RecordedRequest(seq=seq, method=method, path=path,
                                  headers={k.lower(): v for k, v in self.headers.items()}, raw=raw, body=body,
                                  stream=bool(body and body.get("stream")),
                                  endpoint="responses" if path == "/v1/responses" else "chat")
            owner._requests.append(req)
        if method != "POST" or req.path not in ("/v1/chat/completions", "/v1/responses") or (
                req.endpoint == "responses" and not owner.serve_responses):
            req.status = 404
            return self._send_fault(req, Fault.error(404, f"Unknown route {method} {req.path}", type="not_found_error"))
        kind, item = owner.dispatch(req)
        if item.delay:
            time.sleep(item.delay)
        if kind == "fault":
            if item.disconnect:
                req.status = 0
                self.close_connection = True
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return
            return self._send_fault(req, item)
        return self._send_reply(req, item)

    # ---- writers ---------------------------------------------------------------------------------------------------
    def _send_fault(self, req: RecordedRequest, fault: Fault) -> None:
        if fault.status == -1:                 # Fault.stream_error
            if req.stream and req.endpoint == "responses":
                req.status = 200
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                err = (fault.body or {}).get("error") or {}
                created = {"type": "response.created", "sequence_number": 0,
                           "response": self.owner.render_response(Reply(), req, {}, final=False)}
                failure = {"type": "error", "code": err.get("code"), "message": err.get("message"), "param": None,
                           "sequence_number": 1}
                for ev in (created, failure):
                    self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode("utf-8"))
                self.wfile.flush()
                return
            if req.stream:
                req.status = 200
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                first = {"id": f"chatcmpl-fake{req.seq}", "object": "chat.completion.chunk", "model": (req.body or {}).get("model"),
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(first).encode("utf-8") + b"\n\n")
                self.wfile.write(b"data: " + json.dumps(fault.body).encode("utf-8") + b"\n\n")
                self.wfile.flush()
                return
            fault = Fault(503, fault.body, fault.headers)
        req.status = fault.status
        payload = fault.body
        if fault.echo_authorization:
            payload = {"error": {"message": f"Invalid authentication credentials: {req.authorization or '(none)'}",
                                 "type": "authentication_error", "param": None, "code": "invalid_api_key"}}
        if isinstance(payload, (dict, list)):
            data, ctype = json.dumps(payload).encode("utf-8"), "application/json"
        else:
            data = payload.encode("utf-8") if isinstance(payload, str) else (payload or b"")
            ctype = "application/json" if fault.status == 200 else "text/plain"
        self._write(fault.status, data, ctype, fault.headers)

    def _send_reply(self, req: RecordedRequest, reply: Reply) -> None:
        owner = self.owner
        req.status = 200
        if req.endpoint == "responses":
            view = owner.chat_view(req.body or {})
            rendered = owner.render_response_sse(reply, req, view) if req.stream else owner.render_response(reply, req, view)
        else:
            rendered = owner.render_sse(reply, req) if req.stream else owner.render_json(reply, req)
        headers = {"x-request-id": f"req_fake{req.seq}", **owner.ratelimit_header_values(req), **reply.headers}
        if req.stream:
            events = rendered
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            for event in events:
                self.wfile.write(event)
                self.wfile.flush()
            return
        self._write(200, json.dumps(rendered).encode("utf-8"), "application/json", headers)

    def _write(self, status: int, data: bytes, ctype: str, headers: Mapping[str, str]) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):  # the client gave up (e.g. a timeout test)
            pass
