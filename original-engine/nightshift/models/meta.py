"""Chat Completions client for the Meta Model API (and any OpenAI-compatible host). Standard library only.

Behaviour follows ``docs/META_API_NOTES.md`` and ``docs/ARCHITECTURE.md`` section 6.5:

* ``POST {base_url}/chat/completions``. ``tool_choice``, ``stop``, ``n``, ``logprobs``, ``logit_bias`` and ``max_tokens``
  are never sent. Reasoning effort is validated; ``none`` is sent as ``minimal``; ``max`` only to models that document it
  (``muse-wonder-*`` and ``muse-spark-1.3``, not ``-contributor``).
* Every streamed request (``stream=True`` and the 504-then-stream fallback) carries ``stream_options.include_usage``: the
  docs say a stream omits ``usage`` otherwise, and the estimate that replaces it counts visible text only, so hidden reasoning
  tokens would never reach the budget gate. A host that rejects ``stream_options`` gets it stripped (and then an estimate,
  flagged ``usage_estimated``).
* Retries only HTTP 429/500/502/503 (``Retry-After`` honoured, 0.5 s doubling plus 0-1 s jitter, ``max_retries`` retries) and
  then raises ``ModelError(retryable=True)``. A 504 on a non-streaming call is retried once as a stream. A terminal in-stream
  error with code ``server_shutting_down`` / ``service_overloaded`` / ``backend_unavailable`` re-sends the whole request under
  the same retry budget. 400/401/402/403/404/413 and every other status raise ``ModelError(retryable=False)`` carrying
  ``status``/``error_type``/``code``.
* A 400 that names an optional parameter retries once without it and keeps it in ``disabled_params``. ``tools`` and
  ``response_format`` are never stripped. A 400/404 on the primary model retries on ``fallback_model``.
* ``max_completion_tokens``: the docs say a request above the model's configured maximum is a 400 and state no maximum. A
  ``completion_cap_ceiling`` given to the constructor clamps every request; a 400 that names ``max_completion_tokens`` is
  retried ONCE at a lower value (the limit the message states, else the largest value this client has seen accepted for
  the model, else half) and that value is remembered per model, like ``disabled_params``.
  A context-window 400 is not a statement about the output maximum and is never handled this way.
* A 404 ``model_not_found`` from the primary model, when a fallback is configured, makes the fallback STICKY for the
  client: later calls go straight to the fallback instead of sending a doomed request to the primary every time. The
  first response after that (normally the call that discovered it) reports why in
  ``ChatResponse.raw["_nightshift"]["fallback_reason"]``, once; ``primary_unavailable``
  keeps the reason. A 400 on the primary still falls back for that one call only.
* Timeouts have to fit reasoning calls (Muse Spark always reasons and a non-streamed reply can take minutes): the default
  is ``DEFAULT_TIMEOUT`` (300 s), and a call whose ``max_completion_tokens`` is above ``REASONING_TOKEN_THRESHOLD`` (8000)
  gets ``REASONING_TIMEOUT`` (600 s). A ``timeout`` given to the constructor or to ``chat`` is explicit and always wins.
* Security: ``base_url`` must be https (loopback hosts may use http), userinfo is dropped, redirects are never followed,
  the key travels only in the ``Authorization`` header and is registered with ``nightshift.secrets``; no exception message,
  ``repr`` or ledger text produced here contains it.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .. import secrets
from ..types import ChatMessage, ChatResponse, ModelError, NightshiftError, ToolSpec
from .base import (STREAM_RETRYABLE_CODES, ModelConfigError, StreamAssembler, clean_schema_name, estimate_tokens,
                   messages_to_wire, normalize_reasoning_effort, parse_error_envelope, parse_sse, reasoning_tokens_from_wire,
                   strictify_schema, supports_max_effort, text_from_content, tool_calls_from_wire, tools_to_wire,
                   usage_from_wire)

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
KEY_ENV_NAMES = ("MODEL_API_KEY", "META_API_KEY")
OPTIONAL_PARAMS = ("prompt_cache_key", "prompt_cache_retention", "safety_identifier", "reasoning_effort", "temperature",
                   "stream_options")
NEVER_SENT = ("tool_choice", "stop", "n", "logprobs", "logit_bias", "max_tokens")
# 502 is listed by the cookbook retry recipe ("retry 429, 5xx"); /docs/error-handling names only 429/500/503. 504 is not here:
# it has its own path (the same request would time out again, so a non-streaming call is retried once as a stream).
RETRYABLE_STATUSES = (429, 500, 502, 503)
_LIMIT_IN_MESSAGE = re.compile(r"(?:maximum|limit|at most|no more than|allowed|up to|max output tokens|<=|\u2264)[^0-9]{0,30}(\d[\d,_]*)", re.I)
_MIN_PLAUSIBLE_CAP = 16
# A host that does not know stream_options may name only the inner field ("unknown field include_usage").
_PARAM_ALIASES = {"stream_options": ("include_usage",)}
# A 400 about the context window is not a statement about the model's maximum output: lowering the cap may rescue that one
# call, but the lower value must never be remembered as the model's limit (a later call with a short prompt would be clamped).
_CONTEXT_OVERFLOW = re.compile(r"context[ _-]?(?:window|length)|maximum context|too many tokens|prompt is too long", re.I)
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
DEFAULT_TIMEOUT = 300.0             # seconds for one request when nothing else is said
REASONING_TIMEOUT = 600.0           # seconds when max_completion_tokens is above REASONING_TOKEN_THRESHOLD
REASONING_TOKEN_THRESHOLD = 8000
_MODEL_MISSING = re.compile(r"\b(?:does not exist|do not exist|not found|no such|unknown|unavailable|not available)\b", re.I)
_KEY_OK = re.compile(r"^[\x21-\x7e]{4,512}$")

Transport = Callable[[str, str, Mapping[str, str], bytes, float, bool], tuple]


# --------------------------------------------------------------------------------------------------
# configuration helpers
# --------------------------------------------------------------------------------------------------
def check_base_url(base_url: str) -> str:
    """Validate and normalise a base URL: https, or http for loopback hosts only; userinfo, query and fragment dropped.

    Raises ``ModelConfigError`` (a ``ValueError``). The message never repeats userinfo.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ModelConfigError("base_url must be a non-empty string")
    try:
        parts = urlsplit(base_url.strip())
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        raise ModelConfigError("base_url is not a valid URL") from None
    if parts.scheme not in ("http", "https") or not host:
        raise ModelConfigError("base_url must be an http(s) URL with a host")
    if parts.scheme == "http" and host not in LOOPBACK_HOSTS:
        raise ModelConfigError(f"base_url must use https (plain http is allowed only for {', '.join(LOOPBACK_HOSTS)}); "
                               f"host is {host!r}")
    netloc = f"[{host}]" if ":" in host else host
    if port:
        netloc += f":{port}"
    return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))


def is_loopback(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_api_key(env: Mapping[str, str] | None = None, names: Sequence[str] = KEY_ENV_NAMES) -> str:
    """The first non-empty key among ``names`` in ``env`` (default ``os.environ``); ``ModelConfigError`` if none."""
    env = os.environ if env is None else env
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    raise ModelConfigError(f"no API key found: set {' or '.join(names)} in the environment")


def _check_key(key: str) -> str:
    key = key.strip()
    if not _KEY_OK.match(key):
        # Never echo the value: it may be a secret, or contain a newline meant to inject headers.
        raise ModelConfigError("the API key must be 4-512 printable ASCII characters without spaces")
    return key


# --------------------------------------------------------------------------------------------------
# default transport
# --------------------------------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None  # makes urllib surface the 3xx as an HTTPError instead of following it


def _stream_lines(resp: Any) -> Iterator[bytes]:
    try:
        total = 0
        for line in resp:
            total += len(line)
            if total > MAX_RESPONSE_BYTES:
                raise ModelError("response stream is too large", error_type="malformed_response")
            yield line
    finally:
        resp.close()


def default_transport(method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float,
                      stream: bool) -> tuple[int, dict[str, str], Any]:
    """urllib transport. Never follows redirects; loopback requests bypass any proxy. On a 2xx with ``stream`` the body
    is an iterator of byte lines; in every other case it is the (size-capped) response bytes."""
    handlers: list[Any] = [_NoRedirect()]
    if is_loopback(url):
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        resp = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as err:  # any non-2xx, including an unfollowed redirect
        try:
            payload = err.read(MAX_RESPONSE_BYTES)
        finally:
            err.close()
        return err.code, {k.lower(): v for k, v in err.headers.items()}, payload
    status, hdrs = resp.status, {k.lower(): v for k, v in resp.headers.items()}
    if stream and 200 <= status < 300:
        return status, hdrs, _stream_lines(resp)
    try:
        return status, hdrs, resp.read(MAX_RESPONSE_BYTES)
    finally:
        resp.close()


# --------------------------------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------------------------------
class MetaChatClient:
    """Chat Completions client; implements ``nightshift.types.ModelClient``. Thread-safe: concurrent ``chat`` calls only
    share ``disabled_params``, which is updated under a lock."""

    def __init__(self, model: str = DEFAULT_MODEL, *, base_url: str = DEFAULT_BASE_URL, api_key: str | None = None,
                 api_key_env: str = "MODEL_API_KEY", timeout: float | None = None, max_retries: int = 4,
                 sleep: Callable[[float], None] = time.sleep, transport: Transport | None = None,
                 safety_identifier: str | None = None, default_reasoning: str | None = None,
                 fallback_model: str | None = None,
                 # keyword-only extras beyond the contract
                 fallback_reasoning: str | None = None, cache_retention: str | None = None,
                 jitter: Callable[[], float] = random.random, timer: Callable[[], float] = time.monotonic,
                 backoff_base: float = 0.5, max_retry_after: float = 120.0, completion_cap_ceiling: int | None = None):
        if not isinstance(model, str) or not model.strip():
            raise ModelConfigError("model must be a non-empty string")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ModelConfigError("max_retries must be a non-negative integer")
        if timeout is not None and not (isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
                                        and math.isfinite(timeout) and timeout > 0):
            raise ModelConfigError("timeout must be a positive number of seconds")
        self.model = model.strip()
        self.base_url = check_base_url(base_url)
        if api_key is None:
            names = [api_key_env] + ([n for n in KEY_ENV_NAMES if n != api_key_env] if api_key_env == KEY_ENV_NAMES[0] else [])
            api_key = resolve_api_key(names=names)
        self._api_key = _check_key(api_key)
        secrets.register(self._api_key)
        self._timeout_explicit = timeout is not None
        self.timeout = float(DEFAULT_TIMEOUT if timeout is None else timeout)
        self.max_retries = max_retries
        self.fallback_model = fallback_model.strip() if fallback_model and fallback_model.strip() else None
        self.default_reasoning = default_reasoning
        normalize_reasoning_effort(default_reasoning, self.model)  # fail at construction, not mid-run
        self.fallback_reasoning = fallback_reasoning
        if fallback_reasoning is not None and self.fallback_model:
            normalize_reasoning_effort(fallback_reasoning, self.fallback_model)
        self.cache_retention = cache_retention
        if completion_cap_ceiling is not None and (not isinstance(completion_cap_ceiling, int)
                                                   or isinstance(completion_cap_ceiling, bool) or completion_cap_ceiling < 1):
            raise ModelConfigError("completion_cap_ceiling must be a positive integer")
        self.completion_cap_ceiling = completion_cap_ceiling   # configured: no request asks for more than this
        self._learned_cap: dict[str, int] = {}                 # model -> value a 400 made this client lower its cap to
        self._accepted_cap: dict[str, int] = {}                # model -> largest max_completion_tokens answered with a 2xx
        self._safety = hashlib.sha256(safety_identifier.encode("utf-8")).hexdigest()[:32] if safety_identifier else None
        self.disabled_params: set[str] = set()
        self.primary_unavailable: str | None = None   # why calls skip the primary model (set by a 404 model_not_found)
        self._reason_pending = False                  # primary_unavailable has not been reported on a response yet
        self._lock = threading.Lock()
        self._sleep, self._jitter, self._timer = sleep, jitter, timer
        self._transport: Transport = transport or default_transport
        self._backoff_base = float(backoff_base)
        self._max_retry_after = float(max_retry_after)
        self._url = self.base_url + "/chat/completions"

    def __repr__(self) -> str:
        return (f"MetaChatClient(model={self.model!r}, base_url={self.base_url!r}, "
                f"fallback_model={self.fallback_model!r})")

    @classmethod
    def from_env(cls, model: str | None = None, *, env: Mapping[str, str] | None = None, **kw: Any) -> "MetaChatClient":
        """Build from the environment. Key: ``MODEL_API_KEY`` then ``META_API_KEY`` (``ModelConfigError`` if neither).
        ``NIGHTSHIFT_BASE_URL`` and ``NIGHTSHIFT_MODEL`` override the ``base_url`` / ``model`` arguments."""
        env = os.environ if env is None else env
        if kw.get("api_key") is None:
            custom = kw.get("api_key_env")
            # An explicitly named variable is the only source: falling back to the Meta key would send it to another host.
            kw["api_key"] = resolve_api_key(env, (custom,) if custom and custom != KEY_ENV_NAMES[0] else KEY_ENV_NAMES)
        explicit_url = kw.pop("base_url", None)
        base_url = (env.get("NIGHTSHIFT_BASE_URL") or "").strip() or explicit_url or DEFAULT_BASE_URL
        chosen = (env.get("NIGHTSHIFT_MODEL") or "").strip() or model or DEFAULT_MODEL
        return cls(chosen, base_url=base_url, **kw)

    # ---- public API -------------------------------------------------------------------------------
    def chat(self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolSpec] | None = None,
             response_schema: dict | None = None, schema_name: str = "response", reasoning_effort: str | None = None,
             max_completion_tokens: int | None = None, temperature: float | None = None, cache_key: str | None = None,
             timeout: float | None = None, stream: bool = False) -> ChatResponse:
        """One chat completion. ``stream=True`` forces a streamed request (the reply is reassembled). Raises
        ``ValueError`` for a malformed request (nothing is sent) and ``ModelError`` for anything the server or network did."""
        wire_messages = messages_to_wire(messages)
        wire_tools = tools_to_wire(tools) if tools else None
        response_format = None
        if response_schema is not None:
            response_format = {"type": "json_schema", "json_schema": {
                "name": clean_schema_name(schema_name), "schema": strictify_schema(response_schema), "strict": True}}
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
        call = _Call(wire_messages, wire_tools, response_format, max_completion_tokens, temperature, cache_key,
                     self._timeout_for(max_completion_tokens, timeout), stream)
        started = self._timer()
        attempts: list[tuple[str, str | None, bool]] = [(self.model, primary_effort, False)]
        if self.fallback_model:
            attempts.append((self.fallback_model, self._fallback_effort(requested), True))
            if self.primary_unavailable is not None:
                attempts = attempts[1:]          # the primary is known to be missing: do not send a doomed request
        for i, (model, effort, is_fallback) in enumerate(attempts):
            try:
                data, info = self._complete(model, effort, call)
            except ModelError as exc:
                if i + 1 < len(attempts) and exc.status in (400, 404):
                    if exc.status == 404 and self._model_missing(exc):
                        self._remember_missing_primary(call, model)
                    continue
                raise
            return self._to_response(data, info, model, effort, is_fallback, started, response_schema is not None, call)
        raise AssertionError("unreachable")  # pragma: no cover

    def _timeout_for(self, max_completion_tokens: int | None, timeout: float | None) -> float:
        """Seconds one request may take: an explicit ``timeout`` (this call's, else the constructor's) wins; otherwise
        ``REASONING_TIMEOUT`` for a large completion cap and ``DEFAULT_TIMEOUT`` for the rest."""
        if timeout is not None:
            return float(timeout)
        if self._timeout_explicit:
            return self.timeout
        if max_completion_tokens is not None and max_completion_tokens > REASONING_TOKEN_THRESHOLD:
            return REASONING_TIMEOUT
        return self.timeout

    @staticmethod
    def _model_missing(exc: ModelError) -> bool:
        """The 404 says the MODEL does not exist (code ``model_not_found``, ``param`` model, or a message about the model),
        as opposed to a missing route or any other 404."""
        if (exc.code or "").lower() == "model_not_found" or (getattr(exc, "param", None) or "").lower() == "model":
            return True
        text = str(exc)
        return "model" in text.lower() and bool(_MODEL_MISSING.search(text))

    def _remember_missing_primary(self, call: "_Call", model: str) -> None:
        """Make the fallback sticky. The reason is reported once, on the first response that follows (normally the call that
        found out; if that call's fallback attempt failed too, the next call that gets an answer)."""
        with self._lock:
            if self.primary_unavailable is not None:
                return
            self.primary_unavailable = (f"the primary model {model} answered 404 model_not_found; every call of this client "
                                        f"uses {self.fallback_model} instead")
            self._reason_pending = True

    def raw_request(self, body: Mapping[str, Any], *, timeout: float | None = None) -> tuple[int, dict]:
        """Diagnostics only: POST ``body`` exactly as given (no retries, no parameter policy) and return
        ``(status, parsed_json_or_{})`` with secrets scrubbed. Used by the live smoke test to provoke a documented 400."""
        payload = json.dumps(body, allow_nan=False).encode("utf-8")
        status, _headers, raw = self._send(payload, False, self.timeout if timeout is None else float(timeout))
        text = self._body_text(raw)
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = {}
        return status, self._clean_obj(parsed if isinstance(parsed, dict) else {})

    # ---- request construction -----------------------------------------------------------------------
    def _fallback_effort(self, requested: str | None) -> str | None:
        assert self.fallback_model
        effort = self.fallback_reasoning if self.fallback_reasoning is not None else requested
        if effort is not None and str(effort).strip().lower() == "max" and not supports_max_effort(self.fallback_model):
            effort = "xhigh"     # only when the fallback model cannot take max (e.g. a -contributor or a partner-hosted model)
        return normalize_reasoning_effort(effort, self.fallback_model)

    def _body(self, model: str, effort: str | None, call: "_Call", stream: bool) -> dict:
        off = self.disabled_params
        body: dict[str, Any] = {"model": model, "messages": call.messages}
        if call.tools:
            body["tools"] = call.tools
        if call.response_format:
            body["response_format"] = call.response_format
        if effort and "reasoning_effort" not in off:
            body["reasoning_effort"] = effort
        cap = self._cap_for(model, call.max_tokens, call.cap_override.get(model))
        if cap is not None:
            body["max_completion_tokens"] = cap
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
        return body

    def _make_streamed(self, body: dict) -> None:
        """Turn ``body`` into a streamed request that asks for real usage (a stream omits ``usage`` unless
        ``stream_options.include_usage`` is set; without it the budget would only see a visible-text estimate)."""
        body["stream"] = True
        if "stream_options" not in self.disabled_params:
            body["stream_options"] = {"include_usage": True}

    def _cap_for(self, model: str, wanted: int | None, trial: int | None = None) -> int | None:
        """``wanted`` clamped to the configured ceiling, to any lower value learned from a 400 for ``model`` and to ``trial``
        (the value being tried after a rejection; it is only remembered once a request with it succeeds)."""
        if wanted is None:
            return None
        with self._lock:
            limits = [v for v in (self.completion_cap_ceiling, self._learned_cap.get(model), trial) if v is not None]
        return min([wanted, *limits])

    def _note_accepted(self, body: Mapping[str, Any]) -> None:
        cap, model = body.get("max_completion_tokens"), body.get("model")
        if isinstance(cap, int) and isinstance(model, str):
            with self._lock:
                self._accepted_cap[model] = max(cap, self._accepted_cap.get(model, 0))

    def _lower_cap(self, model: str, requested: int, exc: ModelError) -> int | None:
        """The value to retry a rejected ``max_completion_tokens`` at: the limit the message states, else the largest value
        this client has had accepted for ``model``, else half. None when nothing lower than ``requested`` exists."""
        stated = []
        for match in _LIMIT_IN_MESSAGE.finditer(str(exc)):
            try:
                value = int(re.sub(r"[,_]", "", match.group(1)))
            except ValueError:
                continue
            if _MIN_PLAUSIBLE_CAP <= value < requested:
                stated.append(value)
        if stated:
            return max(stated)
        with self._lock:
            accepted = self._accepted_cap.get(model)
        if accepted is not None and accepted < requested:
            return accepted
        return requested // 2 if requested // 2 >= 1 else None

    # ---- the request loop -----------------------------------------------------------------------------
    def _complete(self, model: str, effort: str | None, call: "_Call") -> tuple[dict, dict]:
        """One model: retry without any optional parameter the server names in a 400.

        A parameter named in a 400 is retried even if a concurrent call already disabled it (this request was built
        before that). The loop is bounded by the number of optional parameters, so it always ends.
        """
        last: ModelError | None = None
        cap_retried = False
        for _ in range(len(OPTIONAL_PARAMS) + 2):
            body = self._body(model, effort, call, call.stream)
            try:
                result = self._post(body, call)
                if model in call.cap_override:      # the lower value worked, so the server's limit really is about that
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
                elif not cap_retried and self._names_completion_cap(exc, body):
                    lower = self._lower_cap(model, body["max_completion_tokens"], exc)
                    if lower is None:
                        raise
                    cap_retried = True    # one retry only: a second rejection is the caller's problem
                    call.cap_override[model] = lower
                    call.info["cap_reduced"] = {"from": body["max_completion_tokens"], "to": lower}
                else:
                    raise
                last = exc
        assert last is not None
        raise last

    @staticmethod
    def _names_completion_cap(exc: ModelError, body: Mapping[str, Any]) -> bool:
        """The 400 is about the completion-token cap (param or message names ``max_completion_tokens`` / the max output)."""
        if not isinstance(body.get("max_completion_tokens"), int):
            return False
        text = f"{exc} {getattr(exc, 'param', None) or ''}".lower()
        if (exc.code or "").lower() == "context_length_exceeded" or (
                _CONTEXT_OVERFLOW.search(text) and "configured maximum" not in text):
            return False        # input plus requested output overflowed the window; not the model's output maximum
        return "max_completion_tokens" in text or "max output" in text or "maximum output" in text

    @staticmethod
    def _named_optional(exc: ModelError, body: Mapping[str, Any]) -> list[str]:
        text = f"{exc} {getattr(exc, 'param', None) or ''}".lower()
        return [name for name in OPTIONAL_PARAMS
                if name in body and (name in text or any(alias in text for alias in _PARAM_ALIASES.get(name, ())))]

    def _post(self, body: dict, call: "_Call") -> tuple[dict, dict]:
        info = call.info
        stream = bool(body.get("stream"))
        retries = 0
        while True:
            payload = json.dumps(body, allow_nan=False).encode("utf-8")
            info["attempts"] += 1
            info["sent_effort"] = body.get("reasoning_effort")     # what goes on the wire now, whatever is disabled later
            info["sent_cap"] = body.get("max_completion_tokens")
            status, headers, raw = self._send(payload, stream, call.timeout)
            if 200 <= status < 300:
                info["streamed"] = stream
                info["request_id"] = headers.get("x-request-id") or headers.get("request-id")
                info["ratelimit"] = {k: v[:40] for k, v in headers.items() if k.startswith("x-ratelimit-")}
                try:
                    data = self._read_stream(raw) if stream else self._read_json(raw)
                except ModelError as exc:
                    if stream and self._transient_stream_error(exc) and retries < self.max_retries:
                        self._sleep(self._delay(retries, None))   # re-send the whole request, as /docs/error-handling says
                        retries += 1
                        info["retries"] += 1
                        continue
                    raise
                self._ensure_usage(data, payload)
                self._note_accepted(body)
                return data, info
            err = self._http_error(status, headers, raw)
            if status == 429:       # absorbed by the retry loop, so it would otherwise be invisible (the live smoke reports it)
                info["rate_limited"] += 1
                if err.retry_after is not None:
                    info["retry_after_seen"].append(err.retry_after)
            if status == 504 and not stream:
                stream = True  # a streamed request is exempt from the gateway timeout; try that exactly once
                body = dict(body)
                self._make_streamed(body)
                call.stream = True  # later attempts of this call (parameter strip, fallback model) go straight to a stream
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
        """A terminal in-stream error the docs call transient (server_shutting_down, service_overloaded, ...)."""
        return bool(exc.retryable) and (exc.code or exc.error_type or "").lower() in STREAM_RETRYABLE_CODES

    def _delay(self, retries: int, retry_after: float | None) -> float:
        jitter = min(1.0, max(0.0, float(self._jitter())))
        backoff = self._backoff_base * (2 ** retries) + jitter
        return max(backoff, retry_after) if retry_after is not None else backoff

    def _send(self, payload: bytes, stream: bool, timeout: float) -> tuple[int, Mapping[str, str], Any]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                   "Accept": "text/event-stream" if stream else "application/json", "User-Agent": "nightshift/0.1"}
        try:
            status, resp_headers, body = self._transport("POST", self._url, headers, payload, timeout, stream)
        except ModelError as exc:
            raise self._scrub_error(exc) from None
        except NightshiftError:
            raise  # BudgetExceeded, HaltRequested, SimulatedCrash ... are never network failures
        except Exception as exc:  # trust boundary: sockets, TLS, DNS and injected transports all fail in their own ways
            timed_out = isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout))
            raise ModelError(f"network error contacting {urlsplit(self._url).netloc}: "
                             f"{'timeout' if timed_out else type(exc).__name__}",
                             error_type="timeout" if timed_out else "network", retryable=True) from None
        if not isinstance(status, int):
            raise ModelError("transport returned an invalid status", error_type="malformed_response")
        return status, {str(k).lower(): str(v) for k, v in dict(resp_headers or {}).items()}, body

    # ---- response parsing -------------------------------------------------------------------------------
    def _body_text(self, raw: Any) -> str:
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw).decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return raw
        chunks = []
        try:
            for part in raw or ():
                chunks.append(part if isinstance(part, (bytes, bytearray)) else str(part).encode("utf-8"))
        except Exception:  # an error body that cannot be read is simply empty
            pass
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()
        return b"".join(chunks).decode("utf-8", errors="replace")

    def _read_json(self, raw: Any) -> dict:
        text = self._body_text(raw)
        try:
            data = json.loads(text)
        except ValueError:
            raise ModelError("the API returned a response that is not JSON", error_type="malformed_response") from None
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            info = self._envelope(json.dumps({"error": data["error"]}) if isinstance(data["error"], Mapping) else text)
            raise ModelError(self._clean(f"the API reported an error in a 200 response: {info['message']}"),
                             error_type=info["type"], code=info["code"])
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise ModelError("the API response has no choices", error_type="malformed_response")
        return data

    def _read_stream(self, raw: Any) -> dict:
        asm = StreamAssembler()
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
        return asm.to_response()

    @staticmethod
    def _ensure_usage(data: dict, request_payload: bytes) -> None:
        """If the reply carries no token counts (a stream without include_usage, or one cut off before its last chunk),
        estimate them from visible text so the budget is never charged nothing. The estimate misses hidden reasoning tokens,
        which is why streamed requests ask for real usage; an estimated reply is flagged ``usage_estimated``."""
        usage = data.get("usage")
        if isinstance(usage, Mapping) and (usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None):
            return
        first = data["choices"][0]
        message = first.get("message") if isinstance(first, Mapping) else None
        message = message if isinstance(message, Mapping) else {}
        calls = message.get("tool_calls")
        fns = [c.get("function") for c in calls if isinstance(c, Mapping)] if isinstance(calls, list) else []
        produced = text_from_content(message.get("content")) + "".join(
            str(f.get("name", "")) + str(f.get("arguments", "")) for f in fns if isinstance(f, Mapping))
        data["usage"] = {"prompt_tokens": estimate_tokens(request_payload.decode("utf-8", errors="replace")),
                         "completion_tokens": estimate_tokens(produced)}
        data["_usage_estimated"] = True

    def _envelope(self, text: str, limit: int = 600) -> dict:
        """``parse_error_envelope`` with the key removed *before* the fields are capped, so a cap can never leave half a key."""
        env = parse_error_envelope(text, limit=1 << 22)
        return {k: (self._clean(v)[:limit] if v is not None else None) for k, v in env.items()}

    def _scrub_error(self, exc: ModelError) -> ModelError:
        exc.args = (self._clean(str(exc)),)
        exc.error_type = self._clean(exc.error_type) if exc.error_type else exc.error_type
        exc.code = self._clean(exc.code) if exc.code else exc.code
        return exc

    def _http_error(self, status: int, headers: Mapping[str, str], raw: Any) -> ModelError:
        env = self._envelope(self._body_text(raw))
        if 300 <= status < 400:
            err = ModelError(f"HTTP {status}: the API answered with a redirect, which is never followed",
                             status=status, error_type="redirect")
            err.param = None  # type: ignore[attr-defined]
            return err
        retry_after = _parse_retry_after(headers.get("retry-after"))
        label = f" {env['type']}" if env["type"] else ""
        message = self._clean(f"Meta API HTTP {status}{label}: {env['message'] or 'no error message'}")
        err = ModelError(message, status=status, error_type=self._clean(env["type"]) if env["type"] else None,
                         code=self._clean(env["code"]) if env["code"] else None,
                         retryable=status in RETRYABLE_STATUSES, retry_after=retry_after)
        err.param = self._clean(env["param"]) if env["param"] else None  # type: ignore[attr-defined]
        return err

    def _to_response(self, data: dict, info: dict, model: str, effort: str | None, is_fallback: bool, started: float,
                     schema_requested: bool, call: "_Call") -> ChatResponse:
        choice = data["choices"][0] if isinstance(data["choices"][0], Mapping) else {}
        wire_msg = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
        calls = tool_calls_from_wire(wire_msg.get("tool_calls"))
        text = text_from_content(wire_msg.get("content"))
        usage, cached_known = usage_from_wire(data.get("usage"))
        with self._lock:
            disabled = set(self.disabled_params)
            reason = self.primary_unavailable if (self._reason_pending and is_fallback) else None
            if reason:
                self._reason_pending = False
        sent_effort = info["sent_effort"] if "sent_effort" in info else self._body(model, effort, call, False).get("reasoning_effort")
        meta = {
            "cached_known": cached_known, "latency_ms": round((self._timer() - started) * 1000.0, 1),
            "retries": info["retries"], "stream_fallback": info["stream_fallback"], "requested_model": self.model,
            "reasoning_effort": sent_effort, "schema_requested": schema_requested,
            # extras
            "streamed": info["streamed"], "attempts": info["attempts"], "fallback_used": is_fallback,
            "disabled_params": sorted(disabled), "usage_estimated": bool(data.get("_usage_estimated")),
            "reasoning_tokens": reasoning_tokens_from_wire(data.get("usage")), "ratelimit": dict(info.get("ratelimit") or {}),
            "max_completion_tokens_sent": info.get("sent_cap"), "completion_cap_reduced": info.get("cap_reduced"),
            "rate_limited": info["rate_limited"], "retry_after_seen": list(info["retry_after_seen"]),
        }
        if reason:
            meta["fallback_reason"] = reason     # logged once, on the first response after the primary was found missing
        raw = {k: v for k, v in data.items() if k != "_usage_estimated"}
        raw["_nightshift"] = meta
        raw["nightshift"] = meta  # budget.MeteredClient reads this spelling
        answered = data.get("model") if isinstance(data.get("model"), str) and data.get("model") else model
        request_id = info.get("request_id") or (data.get("id") if isinstance(data.get("id"), str) else None)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=text if (text or not calls) else None, tool_calls=calls),
            finish_reason=str(choice.get("finish_reason") or "stop"), usage=usage, model=answered,
            request_id=request_id, raw=raw)

    # ---- secrets ------------------------------------------------------------------------------------------
    def _clean(self, text: str | None) -> str:
        """Remove the API key (and anything the secret registry knows) from text that may reach an exception or a log."""
        if not text:
            return ""
        return secrets.scrub_text(text.replace(self._api_key, secrets.REDACTED))

    def _clean_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self._clean(obj)
        if isinstance(obj, dict):
            return {self._clean(k) if isinstance(k, str) else k: self._clean_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean_obj(v) for v in obj]
        return obj


class _Call:
    """The immutable request parts of one ``chat`` call plus its mutable statistics."""

    def __init__(self, messages: list[dict], tools: list[dict] | None, response_format: dict | None,
                 max_tokens: int | None, temperature: float | None, cache_key: str | None, timeout: float, stream: bool):
        self.messages, self.tools, self.response_format = messages, tools, response_format
        self.max_tokens, self.temperature, self.cache_key = max_tokens, temperature, cache_key
        self.timeout, self.stream = timeout, stream
        self.cap_override: dict[str, int] = {}      # model -> max_completion_tokens being tried after a 400 named the cap
        self.info: dict[str, Any] = {"retries": 0, "attempts": 0, "stream_fallback": False, "streamed": False,
                                     "request_id": None, "ratelimit": {}, "cap_reduced": None, "sent_cap": None,
                                     "rate_limited": 0, "retry_after_seen": []}


def _parse_retry_after(value: str | None) -> float | None:
    """``Retry-After`` in seconds; an HTTP date or garbage is ignored."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None
