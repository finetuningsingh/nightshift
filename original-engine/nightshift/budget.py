"""Budget enforcement. Limits are checked by code before an action, never by a prompt.

* ``BudgetGate`` reads usage from the ledger view (``view.budget_used``, ``view.supervisor_turns``) and writes
  ``budget.charged`` / ``budget.exhausted``. One re-entrant lock makes check-then-charge atomic, so with one
  experiment slot left exactly one concurrent ``reserve_experiment`` succeeds.
* Charges are idempotent: an LLM call id, an ``(exp_id, purpose)`` experiment slot and a replication slot are each
  charged at most once, including across a resume (the ledger is consulted, not only process memory).
* ``MeteredClient`` wraps a ``ModelClient``: ``check("llm")`` before the call, charge from the real usage after it. The
  charge names the model that ANSWERED and the reasoning effort the client really put on the wire (not the one that was
  requested: a fallback model is called with its own effort). An optional ``charge_context`` makes LLM spend attributable
  to an experiment (``exp_id``, ``purpose``) and to a lane (actor suffix such as ``designer:A``); it is never forwarded to
  the inner client.
* ``remaining_from_view`` is pure and is what the dashboard shows too.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
import weakref
from typing import Any, Mapping, Optional, Sequence

from .types import (BudgetExceeded, BudgetLimits, ChatMessage, ChatResponse, Clock, Kind, LedgerProtocol,
                    LedgerView, ModelClient, ToolSpec, Usage)

try:  # models.pricing is written by another module; tests must not depend on it
    from .models.pricing import usd_for as _usd_for
except ImportError:  # pragma: no cover - exercised only while pricing.py does not exist
    _usd_for = None

# muse-spark-1.3 per 1M tokens: input, cached input, output. Used only when models.pricing is missing.
_FALLBACK_PRICE = (1.25, 0.15, 4.25)

ACTIONS = ("experiment", "replication", "llm", "supervisor_turn", "wall")
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cached_known", "reasoning_effort",
              "latency_ms", "retries", "schema_ok", "stream_fallback", "requested_model")
_CONTEXT_BAD = re.compile(r"[^A-Za-z0-9_.-]")
MAX_CONTEXT_TOKEN = 40


_LEDGER_LOCKS: "weakref.WeakKeyDictionary[Any, threading.RLock]" = weakref.WeakKeyDictionary()
_LEDGER_LOCKS_GUARD = threading.Lock()


def _lock_for(ledger: Any) -> threading.RLock:
    """One RLock per ledger object, so two gates on the same ledger still make check-then-charge atomic together."""
    try:
        with _LEDGER_LOCKS_GUARD:
            lock = _LEDGER_LOCKS.get(ledger)
            if lock is None:
                lock = _LEDGER_LOCKS[ledger] = threading.RLock()
            return lock
    except TypeError:   # not hashable / not weak-referenceable (a test double): a private lock
        return threading.RLock()


def usd_for(model: str, usage: Usage) -> float:
    """Dollar cost of one call. Delegates to ``models.pricing`` when present, else the muse-spark-1.3 price."""
    if _usd_for is not None:
        return float(_usd_for(model, usage))
    cached = max(0, min(int(usage.cached_tokens or 0), int(usage.prompt_tokens or 0)))
    fresh = max(0, int(usage.prompt_tokens or 0) - cached)
    inp, cin, out = _FALLBACK_PRICE
    return (fresh * inp + cached * cin + max(0, int(usage.completion_tokens or 0)) * out) / 1_000_000


def _num(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def remaining_from_view(limits: BudgetLimits, view: LedgerView, now: float, *,
                        wall_mark: Optional[float] = None) -> dict:
    """What is left of every budget, as a pure function of the ledger view.

    Wall time: ``wall_mark`` (the instant up to which wall time has been charged) lets ``BudgetGate`` be exact:
    used = charged wall seconds + time since the mark. Without it (dashboard) used is the larger of the charged
    wall seconds and the elapsed time since the first event (frozen at ``run.finished``), which is an estimate.

    Returns ``{"limits", "used", "remaining", "exhausted"}``; remaining values are clamped at zero.
    """
    used_map = view.budget_used
    charged_wall = _num(used_map.get("wall_sec"))
    if wall_mark is not None:
        wall = charged_wall + max(0.0, now - wall_mark)
    else:
        end = _num((view.finished or {}).get("ts")) or now
        start = view.first_ts or end
        wall = max(charged_wall, max(0.0, end - start))
    used = {
        "experiments": _num(used_map.get("experiments")),
        "replications": _num(used_map.get("replications")),
        "llm_usd": _num(used_map.get("usd")),
        "llm_calls": _num(used_map.get("llm_calls")),
        "supervisor_turns": float(view.supervisor_turns),
        "wall_sec": wall,
    }
    caps = {
        "experiments": limits.max_experiments,
        "replications": limits.max_replications,
        "llm_usd": limits.max_llm_usd,
        "supervisor_turns": limits.max_supervisor_turns,
        "wall_sec": limits.max_wall_sec,
    }
    remaining = {k: max(0.0, caps[k] - used[k]) for k in caps}
    for k in ("experiments", "replications", "supervisor_turns"):
        used[k] = int(used[k]) if float(used[k]).is_integer() else used[k]
        remaining[k] = int(remaining[k]) if float(remaining[k]).is_integer() else remaining[k]
    used["llm_calls"] = int(used["llm_calls"])
    return {"limits": limits.to_dict(), "used": used, "remaining": remaining,
            "exhausted": sorted(k for k in caps if used[k] >= caps[k])}


class BudgetGate:
    """Hard limits. Every public method is thread-safe; ``reserve_*`` do check and charge as one atomic step."""

    def __init__(self, limits: BudgetLimits, ledger: LedgerProtocol, *, clock: Clock = time.time):
        self.limits = limits
        self.ledger = ledger
        self._clock = clock
        self._lock = _lock_for(ledger)
        self._wall_mark = float(clock())
        self._exhausted: set[str] = set()
        self._slots: set[tuple[str, str, str]] = set()   # (kind, exp_id, purpose) already charged
        self._calls: set[str] = set()
        self._scanned_seq = 0
        with self._lock:
            self._sync()

    # ---- ledger sync ---------------------------------------------------------------------------
    def _sync(self) -> None:
        """Learn charges and exhaustion events written by an earlier process or another gate instance."""
        for ev in self.ledger.events(self._scanned_seq):
            self._scanned_seq = max(self._scanned_seq, ev.seq)
            d = ev.data or {}
            if ev.kind == Kind.BUDGET_EXHAUSTED and d.get("kind"):
                self._exhausted.add(str(d["kind"]))
            elif ev.kind == Kind.BUDGET_CHARGED and d.get("kind") in ("experiment", "replication") and d.get("exp_id"):
                self._slots.add((d["kind"], str(d["exp_id"]), str(d.get("purpose") or "")))

    def _usage_of(self, action: str, view: LedgerView, wall_used: float) -> tuple[float, float]:
        """(used, limit) for one action."""
        used = view.budget_used
        if action == "experiment":
            return _num(used.get("experiments")), float(self.limits.max_experiments)
        if action == "replication":
            return _num(used.get("replications")), float(self.limits.max_replications)
        if action == "llm":
            return _num(used.get("usd")), float(self.limits.max_llm_usd)
        if action == "supervisor_turn":
            return float(view.supervisor_turns), float(self.limits.max_supervisor_turns)
        return wall_used, float(self.limits.max_wall_sec)

    # ---- checks --------------------------------------------------------------------------------
    def check(self, action: str) -> None:
        """Raise ``BudgetExceeded(kind, limit, used)`` if ``action`` may not start. Writes ``budget.exhausted`` once per kind."""
        if action not in ACTIONS:
            raise ValueError(f"unknown budget action {action!r}; expected one of {ACTIONS}")
        with self._lock:
            self._sync()
            view = self.ledger.view()
            wall = _num(view.budget_used.get("wall_sec")) + max(0.0, self._clock() - self._wall_mark)
            used, limit = self._usage_of(action, view, wall)
            if used < limit:
                return
            if action not in self._exhausted:
                self.ledger.append(Kind.BUDGET_EXHAUSTED, "engine", {"kind": action, "limit": limit, "used": used})
                self._exhausted.add(action)
        raise BudgetExceeded(f"{action} budget exhausted ({used:g} of {limit:g})", kind=action, limit=limit, used=used)

    def _reserve(self, kind: str, exp_id: str, purpose: str, amount: float, *, check: Optional[str]) -> bool:
        if not isinstance(exp_id, str) or not exp_id:
            raise ValueError("exp_id must be a non-empty string")
        with self._lock:
            self._sync()
            key = (kind, exp_id, purpose)
            if key in self._slots:
                return False
            if check:
                self.check(check)
            self.ledger.append(Kind.BUDGET_CHARGED, "engine", {
                "kind": kind, "amount": amount, "unit": "count", "call_id": None, "model": None,
                "exp_id": exp_id, "purpose": purpose, "usage": {}})
            self._slots.add(key)
            return True

    def reserve_experiment(self, exp_id: str, purpose: str = "primary") -> None:
        """Check the experiment limit and charge one slot atomically.

        Idempotent per ``(exp_id, purpose)``: a second call neither charges nor checks. A baseline charges 0 and
        does not consume a slot (``max_experiments`` excludes it); it still refuses when wall time is exhausted.
        """
        purpose = str(purpose or "primary")   # None must key the same slot in memory and in the ledger
        baseline = purpose == "baseline"
        self._reserve("experiment", exp_id, purpose, 0 if baseline else 1, check="wall" if baseline else "experiment")

    def reserve_replication(self, exp_id: str) -> None:
        """Check ``limits.max_replications`` and charge one sealed-split query atomically; idempotent per ``exp_id``."""
        self._reserve("replication", exp_id, "replication", 1, check="replication")

    # ---- charges -------------------------------------------------------------------------------
    def charge_llm(self, model: str, usage: Usage, call_id: str, actor: str, **detail: Any) -> float:
        """Record one model call and return its cost in USD (0.0 if ``call_id`` was already charged).

        ``detail`` may carry any of the usage keys in ``USAGE_KEYS`` plus ``exp_id`` and ``purpose``; they are
        written into the ``budget.charged`` payload. ``cached_known`` defaults to whether any cached token was reported.
        """
        if not call_id:
            raise ValueError("call_id is required so the charge can be idempotent")
        usage = usage if usage is not None else Usage()
        with self._lock:
            if call_id in self._calls or call_id in self.ledger.view().charged_call_ids:
                self._calls.add(call_id)
                return 0.0
            usd = usd_for(model, usage)
            usd = usd if math.isfinite(usd) and usd > 0 else 0.0
            cached = int(usage.cached_tokens or 0)
            payload_usage = {
                "prompt_tokens": int(usage.prompt_tokens or 0),
                "completion_tokens": int(usage.completion_tokens or 0),
                "cached_tokens": cached,
                "cached_known": bool(detail.get("cached_known", cached > 0)),
                "reasoning_effort": detail.get("reasoning_effort"),
                "latency_ms": detail.get("latency_ms"),
                "retries": int(detail.get("retries") or 0),
                "schema_ok": detail.get("schema_ok"),
                "stream_fallback": bool(detail.get("stream_fallback", False)),
                "requested_model": detail.get("requested_model") or model,
            }
            self.ledger.append(Kind.BUDGET_CHARGED, actor, {
                "kind": "llm", "amount": usd, "unit": "usd", "call_id": call_id, "model": model,
                "exp_id": detail.get("exp_id"), "purpose": detail.get("purpose"), "usage": payload_usage})
            self._calls.add(call_id)
            return usd

    def charge_wall(self) -> float:
        """Charge the seconds since the previous wall charge in this process (the first call measures from construction)."""
        with self._lock:
            now = float(self._clock())
            delta = max(0.0, now - self._wall_mark)
            if delta <= 0.0:
                return 0.0
            self.ledger.append(Kind.BUDGET_CHARGED, "engine", {
                "kind": "wall", "amount": delta, "unit": "sec", "call_id": None, "model": None,
                "exp_id": None, "purpose": None, "usage": {}})
            self._wall_mark = now
            return delta

    # ---- reporting -----------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """``{"limits", "used", "remaining", "exhausted"}`` for now (see ``remaining_from_view``)."""
        with self._lock:
            return remaining_from_view(self.limits, self.ledger.view(), float(self._clock()), wall_mark=self._wall_mark)

    def remaining(self) -> dict:
        """What is left per budget: experiments, replications, llm_usd, supervisor_turns, wall_sec (never negative)."""
        return self.snapshot()["remaining"]


def _context_token(value: Any, limit: int = MAX_CONTEXT_TOKEN) -> str:
    """A caller-supplied label reduced to ``[A-Za-z0-9_.-]`` and capped (``""`` for anything that is not a string)."""
    return _CONTEXT_BAD.sub("_", value)[:limit].strip("_") if isinstance(value, str) else ""


class MeteredClient:
    """A ``ModelClient`` that is charged to a ``BudgetGate``. Implements the ``ModelClient`` protocol.

    ``check("llm")`` runs before every call; the charge uses the response's own usage and the model that answered.
    A ``ModelError`` (or any other exception) from the inner client passes through and nothing is charged.
    Optional per-call facts the inner client may expose in ``response.raw["_nightshift"]`` (or the identical
    ``raw["nightshift"]``) are forwarded into the charge: ``retries``, ``stream_fallback``, ``cached_known`` and
    ``reasoning_effort``. The recorded effort is the one the client SENT (``None`` when it sent none, for instance after a
    host rejected the parameter); only an inner client that does not report it gets the effort that was requested.

    ``chat`` accepts one keyword beyond the ``ModelClient`` protocol, ``charge_context``: a mapping with any of ``exp_id``
    and ``purpose`` (written into ``budget.charged``, so LLM spend can be attributed per experiment) and ``actor`` (a lane
    suffix such as ``"A"`` or ``"designer:A"``; the charge's actor becomes ``"<this client's actor>:A"``, so one shared
    client per role still attributes each call to its lane). Labels are reduced to ``[A-Za-z0-9_.-]``. The context is never
    forwarded to the inner client. Callers pass it only to a client that sets ``supports_charge_context`` (this class does;
    scripted clients and other fakes do not).
    """

    supports_charge_context = True

    def __init__(self, inner: ModelClient, gate: BudgetGate, actor: str, *, timer=time.monotonic):
        self._inner = inner
        self._gate = gate
        self._actor = actor
        self._timer = timer

    @property
    def model(self) -> str:
        return self._inner.model

    def chat(self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolSpec] | None = None,
             response_schema: dict | None = None, schema_name: str = "response",
             reasoning_effort: str | None = None, max_completion_tokens: int | None = None,
             temperature: float | None = None, cache_key: str | None = None,
             timeout: float | None = None, charge_context: Mapping[str, Any] | None = None) -> ChatResponse:
        actor, attribution = self._attribution(charge_context)
        self._gate.check("llm")
        call_id = f"{self._actor}:{uuid.uuid4().hex}"
        started = self._timer()
        resp = self._inner.chat(messages, tools=tools, response_schema=response_schema, schema_name=schema_name,
                                reasoning_effort=reasoning_effort, max_completion_tokens=max_completion_tokens,
                                temperature=temperature, cache_key=cache_key, timeout=timeout)
        latency_ms = round((self._timer() - started) * 1000.0, 1)
        meta = self._client_meta(resp)
        sent_effort = meta["reasoning_effort"] if "reasoning_effort" in meta else reasoning_effort
        detail: dict[str, Any] = {
            "reasoning_effort": sent_effort, "latency_ms": latency_ms, "requested_model": self._inner.model,
            "schema_ok": self._schema_ok(resp, response_schema), **attribution}
        for key in ("retries", "stream_fallback", "cached_known"):
            if key in meta:
                detail[key] = meta[key]
        self._gate.charge_llm(resp.model or self._inner.model, resp.usage, call_id, actor, **detail)
        return resp

    def _attribution(self, context: Mapping[str, Any] | None) -> tuple[str, dict[str, str]]:
        """``(actor, {exp_id, purpose})`` for one call; both parts are empty/unchanged when there is no usable context."""
        if not isinstance(context, Mapping):
            return self._actor, {}
        extra = {key: token for key in ("exp_id", "purpose") if (token := _context_token(context.get(key)))}
        suffix = _context_token(context.get("actor").rsplit(":", 1)[-1] if isinstance(context.get("actor"), str) else None)
        actor = f"{self._actor}:{suffix}" if suffix and suffix != self._actor else self._actor
        return actor, extra

    @staticmethod
    def _client_meta(resp: ChatResponse) -> Mapping[str, Any]:
        """What the inner client says about the call (``raw["_nightshift"]``, else ``raw["nightshift"]``), or ``{}``."""
        raw = resp.raw if isinstance(resp.raw, dict) else {}
        for name in ("_nightshift", "nightshift"):
            meta = raw.get(name)
            if isinstance(meta, Mapping):
                return meta
        return {}

    @staticmethod
    def _schema_ok(resp: ChatResponse, response_schema: dict | None) -> Optional[bool]:
        """None when no schema was requested; else whether the reply parsed as a JSON object."""
        if response_schema is None:
            return None
        try:
            return isinstance(json.loads(resp.text), dict)
        except (TypeError, ValueError):
            return False
