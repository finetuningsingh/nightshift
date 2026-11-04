"""An offline ``ModelClient`` for tests and the scripted demo.

``ScriptedClient`` answers from a queue. Each queue item is a ``ChatResponse``, a callable ``(messages, **kwargs)``
returning a ``ChatResponse`` (or plain text), or an ``Exception`` instance, which is raised (use a ``ModelError`` to
test failure paths). Every call is recorded in ``.calls`` before it is answered. An empty queue raises
``ScriptExhausted`` (never a ``ModelError``, so a test that asks for one reply too many cannot be mistaken for a model
outage and silently degrade a role).
"""
from __future__ import annotations

import itertools
import json
import threading
from typing import Any, Callable, Sequence

from ..types import ChatMessage, ChatResponse, NightshiftError, ToolCall, ToolSpec, Usage
from .base import parse_arguments

_IDS = itertools.count(1)
_IDS_LOCK = threading.Lock()


class ScriptExhausted(NightshiftError, RuntimeError):
    """The scripted client was called more times than it has replies."""


def text_response(text: str, *, model: str = "scripted", finish_reason: str = "stop", **usage: int) -> ChatResponse:
    """A plain assistant reply. ``usage`` takes ``prompt_tokens``, ``completion_tokens``, ``cached_tokens``,
    ``total_tokens`` (default: prompt + completion); an unknown key raises ``TypeError``."""
    u = Usage(**usage)
    if "total_tokens" not in usage:
        u.total_tokens = u.prompt_tokens + u.completion_tokens
    return ChatResponse(message=ChatMessage(role="assistant", content=text), finish_reason=finish_reason, usage=u,
                        model=model)


def _next_id() -> str:
    with _IDS_LOCK:
        return f"call_{next(_IDS)}"


def tool_response(*calls: Any, text: str | None = None, model: str = "scripted", **usage: int) -> ChatResponse:
    """An assistant reply that calls tools. Each call is ``(name, args)``, ``(name, args, id)`` or a ``ToolCall``;
    ``args`` is a dict, or a string kept verbatim (a string that is not a JSON object yields ``parse_error``).
    Ids are unique per process (``call_<n>``) unless given."""
    built: list[ToolCall] = []
    for call in calls:
        if isinstance(call, ToolCall):
            built.append(call)
            continue
        if not isinstance(call, (tuple, list)) or len(call) not in (2, 3):
            raise TypeError("a tool call is (name, args) or (name, args, id) or a ToolCall")
        name, args = call[0], call[1]
        call_id = call[2] if len(call) == 3 else _next_id()
        raw = args if isinstance(args, str) else json.dumps(args, sort_keys=True)
        parsed, error = parse_arguments(raw)
        built.append(ToolCall(id=call_id, name=name, arguments=parsed, raw_arguments=raw, parse_error=error))
    u = Usage(**usage)
    if "total_tokens" not in usage:
        u.total_tokens = u.prompt_tokens + u.completion_tokens
    return ChatResponse(message=ChatMessage(role="assistant", content=text, tool_calls=built),
                        finish_reason="tool_calls" if built else "stop", usage=u, model=model)


class ScriptedClient:
    """Queue-driven ``ModelClient``. Thread-safe. ``calls`` holds one dict per request: ``messages`` (a list copy)
    plus every keyword argument given to ``chat`` (``tools``, ``response_schema``, ``schema_name``, ``reasoning_effort``,
    ``max_completion_tokens``, ``temperature``, ``cache_key``, ``timeout``, and any extra)."""

    def __init__(self, script: Sequence[Any] | Callable[..., Any] = (), model: str = "scripted"):
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._handler: Callable[..., Any] | None = script if callable(script) else None
        self._queue: list[Any] = [] if self._handler else list(script)

    def push(self, *items: Any) -> None:
        """Append more replies."""
        with self._lock:
            self._queue.extend(items)

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._queue)

    def chat(self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolSpec] | None = None,
             response_schema: dict | None = None, schema_name: str = "response", reasoning_effort: str | None = None,
             max_completion_tokens: int | None = None, temperature: float | None = None, cache_key: str | None = None,
             timeout: float | None = None, **extra: Any) -> ChatResponse:
        kwargs: dict[str, Any] = dict(tools=tools, response_schema=response_schema, schema_name=schema_name,
                                      reasoning_effort=reasoning_effort, max_completion_tokens=max_completion_tokens,
                                      temperature=temperature, cache_key=cache_key, timeout=timeout, **extra)
        with self._lock:
            self.calls.append({"messages": list(messages), **kwargs})
            index = len(self.calls)
            item = self._handler if self._handler else (self._queue.pop(0) if self._queue else None)
        if item is None:
            raise ScriptExhausted(f"ScriptedClient({self.model!r}) has no reply for call #{index}: "
                                  f"the script is exhausted after {index - 1} replies")
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(messages, **kwargs)
        if isinstance(item, str):
            item = text_response(item, model=self.model)
        if not isinstance(item, ChatResponse):
            raise TypeError(f"scripted reply #{index} must be a ChatResponse, str, Exception or callable returning one, "
                            f"got {type(item).__name__}")
        return item

    def texts(self) -> list[str]:
        """Every message content string and every tool-call argument text ever sent to this client."""
        out: list[str] = []
        for call in list(self.calls):
            for m in call["messages"]:
                if m.content:
                    out.append(m.content)
                out.extend(c.raw_arguments or json.dumps(c.arguments) for c in m.tool_calls)
        return out
