"""The roles that talk to Muse Spark: designer, reviewer, reflector, supervisor, and the factory that wires them.

Guarantees
----------
* Every role catches ``ModelError`` and nothing else. ``BudgetExceeded``, ``HaltRequested``, ``SimulatedCrash`` and every
  other ``NightshiftError`` propagate out of ``design``/``review``/``reflect``/``run``.
* A role that cannot get a usable answer degrades conservatively and never crashes: the reviewer rejects with one
  ``reviewer_unavailable`` info finding (and a non-empty identity that differs from the implementer's), the designer plans
  with the lane's own hypothesis, the reflector returns no lessons. Unparseable replies are treated the same way.
* Everything parsed from a model reply is validated and capped before it leaves the role (ids, severities, kinds,
  file names), because model output is untrusted data.
* Muse Spark always reasons and reasoning tokens count against ``max_completion_tokens``, so the structured roles have
  generous caps (the named constants below) and retry once, with the cap doubled and the effort lowered a notch, a reply that
  was cut off or is not JSON (``prompts.chat_json``) before they degrade. The supervisor is uncapped.
* ``LLMSupervisor.run`` never sends ``tool_choice``, executes and answers every tool call in order (even when one fails),
  merges consecutive ``launch_experiments`` calls into one ``env.launch`` batch while answering each call with its own
  share, writes ``env.note_turn`` for every model call, and keeps the history compact without ever breaking the
  tool-call/tool-message pairing the API requires. Nudges are static templates; model text never enters one.
* The supervisor loop survives a model that stalls: three identical turns with no new information get a nudge, six end the
  run with a static summary that makes no claims; a model that alternates between the same two calls (period-2 cycle:
  two cycles nudge, four close) or keeps asking to finish with the same refusal (the 2nd refusal nudges with the closing
  options, the 4th closes) is handled the same way; a reply cut off at the output limit is neither a summary nor a plain
  reply (it is nudged, and repeated cuts end the run the same way); an empty reply is never replayed to the API.
* A ``ModelError`` with status 401, 402 or 403 (bad key, no credit, no access) ends ``LLMSupervisor.run`` at once, after the
  turn is recorded, instead of being retried: retrying cannot help and the orchestrator can report the cause. Other model
  errors are retried up to ``max_model_errors`` times.
* Operator background material (``background=``) is untrusted data: the supervisor's first user message (which history
  compaction keeps verbatim) and the designer's lane message carry it fenced, capped (60,000 and 8,000 characters) and visibly
  marked when cut; the static system prompts and the cache keys do not depend on it, and with none every prompt is unchanged.
  The cost trade-off is in ``prompts.background_section``.
* With a Responses client (routing ``api = "responses"``) the supervisor keeps the opaque reasoning that client attaches to an
  assistant message (``base.carry_items``) when it rebuilds the message for its history, and ``compact_history`` drops it with the
  turns it stubs and keeps it on the turns it keeps verbatim. The Chat client ignores it, and it is never rendered in a prompt, the
  ledger or the dashboard (``base.OpaqueItems`` prints a byte length and a hash only). A reply cut off at the output limit is not
  replayed with its unfinished reasoning.
* ``LLMReviewer`` names the model that ANSWERED: ``answering_model`` is the model of the latest call in the calling thread
  (``None`` when that call produced no answer) and the identity of each review is built from it, so a fallback model is never
  recorded under the configured model's name.
"""
from __future__ import annotations

import importlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..fence import sanitize
from ..models.base import carries_reasoning, carry_items, drop_items
from ..types import (AgentSuite, ChatMessage, DesignContext, ExperimentPlan, InvalidComposition, LaneSpec, ModelClient,
                     ModelError, NightshiftError, ReflectContext, ReflectionResult, ReviewContext, ReviewVerdict, TaskSpec,
                     ToolCall, canonical_json, sha256_hex)
from . import prompts
from .patch_worker import PatchWorker

STATE_TOOL = "get_state"
LAUNCH_TOOL = "launch_experiments"
FINISH_TOOL = "finish"
MAX_LANES_PER_CALL = 8
# Completion-token caps. They include reasoning tokens, so they are far above the size of the JSON they bound.
DESIGNER_MAX_TOKENS = 6000
REVIEWER_MAX_TOKENS = 8000
REFLECTOR_MAX_TOKENS = 6000
SUPERVISOR_MAX_TOKENS = None        # uncapped: its turns are tool calls the budget gate already limits
REPEAT_NUDGE_AT = 3                 # identical consecutive turns (same calls, no new information) before the nudge
REPEAT_FINISH_AT = 6                # ... before the run is closed with a static summary
MAX_CUT_REPLIES = 3                 # consecutive replies cut off at the output limit before the run is closed
REFUSAL_NUDGE_AT = 2                # identical refused finishes in a row before the nudge with the closing options
REFUSAL_FINISH_AT = 4               # ... before the run is closed with a static summary
CYCLE_NUDGE_AT = 2                  # repetitions of a period-2 pattern of tool calls (ABAB) before the nudge
CYCLE_FINISH_AT = 4                 # ... before the run is closed
NON_RETRYABLE_STATUSES = frozenset({401, 402, 403})   # bad key, no credit, no access: the supervisor stops at once
VOLATILE_KEYS = frozenset({"turns_left", "budget", "latency_ms"})   # change every turn without being information
MAX_TOOL_RESULT_CHARS = 20_000
KEEP_EXCHANGES = 8
STUB_PREFIX = "[older result omitted"
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_SLUG_BAD = re.compile(r"[^A-Za-z0-9_.-]")
_LABEL_BAD = re.compile(r"[^A-Za-z0-9_-]")
# The three patterns below are always used with fullmatch: "$" would also accept a trailing newline.
_EXP_ID = re.compile(r"e\d{1,6}")
_TEMPLATE = re.compile(r"[A-Za-z0-9_-]{1,40}")
_STAGE = re.compile(r"[a-z_]{1,24}")
MAX_LABEL = 16                      # the orchestrator's lane label limit


# --------------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------------
def parse_json_object(text: str | None) -> dict | None:
    """The JSON object in a model reply (tolerates a code fence or surrounding prose), or ``None``."""
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"):text.rfind("}") + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _text(value: Any, limit: int) -> str:
    return sanitize(value, limit) if isinstance(value, str) else ""


def _slug(value: Any, limit: int = 40) -> str:
    return _SLUG_BAD.sub("_", str(value if value is not None else "")).strip("_")[:limit]


def _max_effort(current: str | None, floor: str) -> str:
    """The higher of two reasoning efforts (an unknown or missing ``current`` yields ``floor``)."""
    if current not in _EFFORT_ORDER:
        return floor
    return current if _EFFORT_ORDER.index(current) >= _EFFORT_ORDER.index(floor) else floor


def _same_model(a: str, b: str) -> bool:
    """Two model ids name the same model (an answer may carry a version suffix such as ``muse-spark-1.3-0615``)."""
    return a == b or a.startswith(b + "-") or b.startswith(a + "-")


def _model_of(client: ModelClient) -> str:
    return str(getattr(client, "model", "") or "model")


def _lane_label(value: Any) -> str:
    """A lane label as the orchestrator accepts it: ``[A-Za-z0-9_-]`` and at most 16 characters (``""`` if nothing is left)."""
    return _LABEL_BAD.sub("_", value).strip("_")[:MAX_LABEL] if isinstance(value, str) else ""


def _ident_or_none(value: Any) -> str | None:
    """A model-independent id (an experiment id) reduced to safe characters, or ``None`` when nothing is left."""
    return prompts.ident(value, 40) or None


def _lane_actor(role: str, lane: Any) -> str | None:
    """``"<role>:<lane>"`` for a charge context, or ``None`` when the lane has no usable label."""
    label = _lane_label(getattr(lane, "lane", None))
    return f"{role}:{label}" if label else None


def _round(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return round(float(value), 8)


# --------------------------------------------------------------------------------------------------
# designer
# --------------------------------------------------------------------------------------------------
class LLMDesigner:
    """Turns a lane's hypothesis into a plan. The plan's hypothesis is always the lane's own: the designer refines the
    approach, it cannot rewrite what was proposed (and fingerprinted)."""

    def __init__(self, client: ModelClient, *, reasoning_effort: str | None = "low",
                 max_completion_tokens: int | None = DESIGNER_MAX_TOKENS, background: str = ""):
        self.client = client
        self.background = background or ""
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.identity = f"designer[{_model_of(client)}]"

    def design(self, ctx: DesignContext) -> ExperimentPlan:
        fallback = ExperimentPlan(hypothesis=ctx.lane.hypothesis, approach=ctx.lane.approach or "")
        try:
            _, obj = prompts.chat_json(self.client, prompts.designer_messages(ctx, self.background), schema=prompts.PLAN_SCHEMA,
                                       schema_name="experiment_plan", parse=parse_json_object,
                                       reasoning_effort=self.reasoning_effort, max_completion_tokens=self.max_completion_tokens,
                                       cache_key=prompts.cache_key("designer", ctx.task.name),
                                       charge_context={"exp_id": _ident_or_none(ctx.exp_id), "purpose": "design",
                                                       "actor": _lane_actor("designer", ctx.lane)})
        except ModelError:
            return fallback
        if obj is None:
            return fallback
        approach = _text(obj.get("approach"), 1500)
        if not approach:
            return fallback
        touch: list[str] = []
        allowed = _editable_names(ctx)
        for name in obj.get("files_to_touch") if isinstance(obj.get("files_to_touch"), list) else []:
            if isinstance(name, str) and name in allowed and name not in touch:
                touch.append(name)
        return ExperimentPlan(hypothesis=ctx.lane.hypothesis, approach=approach,
                              expected_effect=_text(obj.get("expected_effect"), 400), risks=_text(obj.get("risks"), 400),
                              files_to_touch=touch[:10])


def _editable_names(ctx: DesignContext) -> set[str]:
    """Files the designer may name: the editable files it was shown, plus literal (glob-free) editable entries."""
    names = set(ctx.editable_files)
    names.update(g for g in ctx.task.editable if not any(c in g for c in "*?["))
    return names


# --------------------------------------------------------------------------------------------------
# reviewer
# --------------------------------------------------------------------------------------------------
class LLMReviewer:
    """Independent review of one experiment. ``independent_basis`` is ``"model"`` when this reviewer's model differs from
    the implementer's, else ``"context"`` (separate context and prompt only).

    A client with a fallback model (``reviewer_hard``: Muse Wonder, falling back to Muse Spark) can answer with the very model
    that implemented the change. The basis is therefore never upgraded on the strength of a configuration alone: if the model
    that answered the last review in this thread is the implementer's model, the basis is ``"context"``. (It is kept per
    thread: lanes review concurrently and the engine reads the basis in the same thread right after ``review``.)

    ``answering_model`` is the same per-thread fact exposed on its own: the model that answered the most recent ``review`` in
    the calling thread, or ``None`` when that call got no answer (the engine reads it right after ``review``). The identity
    on each ``ReviewVerdict`` is built from it (``name[answering model]``); ``self.identity`` is the configured one and is used
    only for a call that got no answer at all."""

    def __init__(self, client: ModelClient, *, name: str = "reviewer", implementer_model: str | None = None,
                 reasoning_effort: str | None = "high", max_completion_tokens: int | None = REVIEWER_MAX_TOKENS):
        self.client = client
        self.name = _slug(name, 30) or "reviewer"
        self.implementer_model = implementer_model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.identity = f"{self.name}[{_model_of(client)}]"
        self._last = threading.local()

    @property
    def answering_model(self) -> str | None:
        """The model that answered the latest ``review`` call made in THIS thread (``None`` before any call and after a call
        that got no answer)."""
        answered = getattr(self._last, "model", None)
        return answered if isinstance(answered, str) and answered else None

    @property
    def independent_basis(self) -> str:
        other = self.implementer_model
        if not other or other == _model_of(self.client):
            return "context"
        answered = self.answering_model
        return "context" if answered and _same_model(answered, other) else "model"

    def _identity_for(self, answered: str | None, implementer_identity: str) -> str:
        """The reviewer identity of one call: built from the model that answered, else the configured identity; never equal
        to the implementer's."""
        label = prompts.ident(answered, 60) if answered else ""
        identity = f"{self.name}[{label}]" if label else self.identity
        return identity if identity != implementer_identity else identity + ":review"

    def review(self, ctx: ReviewContext) -> ReviewVerdict:
        self._last.model = None
        try:
            resp, obj = prompts.chat_json(self.client, prompts.reviewer_messages(ctx), schema=prompts.REVIEW_SCHEMA,
                                          schema_name="review", parse=parse_json_object, reasoning_effort=self.reasoning_effort,
                                          max_completion_tokens=self.max_completion_tokens,
                                          cache_key=prompts.cache_key(self.name, ctx.task.name),
                                          charge_context={"exp_id": _ident_or_none(ctx.exp_id), "purpose": "review"})
        except ModelError as exc:
            return self._unavailable(self._identity_for(None, ctx.implementer_identity),
                                     f"the model could not be reached (status {exc.status})")
        answered = getattr(resp, "model", None)
        self._last.model = answered if isinstance(answered, str) and answered else None
        identity = self._identity_for(self._last.model, ctx.implementer_identity)
        if obj is None:
            return self._unavailable(identity, "the reply was cut off at the token limit and was not a JSON object"
                                     if prompts.was_cut_off(resp) else "the reply was not a JSON object")
        verdict = str(obj.get("verdict", "")).strip().lower()
        if verdict not in prompts.REVIEW_VERDICTS:
            return self._unavailable(identity, "the reply had no valid verdict")
        findings = self._findings(obj.get("findings"))
        if verdict == "accept" and any(f["severity"] == "block" for f in findings):
            verdict = "repair"          # an accept that lists a blocking defect is a contradiction: the safer reading wins
        return ReviewVerdict(verdict=verdict, findings=findings, summary=_text(obj.get("summary"), 400), reviewer=identity)

    @staticmethod
    def _findings(raw: Any) -> list[dict]:
        out: list[dict] = []
        used: set[str] = set()
        for item in raw[:40] if isinstance(raw, list) else []:
            if len(out) >= 10:
                break
            if not isinstance(item, dict):
                continue
            text = _text(item.get("text"), 400)
            if not text:
                continue
            fid = _slug(item.get("id")) or f"finding_{len(out) + 1}"
            while fid in used:
                fid = f"{fid}_{len(used) + 1}"
            used.add(fid)
            severity = str(item.get("severity", "")).strip().lower()
            out.append({"id": fid, "severity": severity if severity in prompts.FINDING_SEVERITIES else "warn",
                        "text": text, "evidence": _text(item.get("evidence"), 300)})
        return out

    @staticmethod
    def _unavailable(identity: str, why: str) -> ReviewVerdict:
        finding = {"id": "reviewer_unavailable", "severity": "info", "evidence": "",
                   "text": f"No review was performed: {why}."}
        return ReviewVerdict(verdict="reject", findings=[finding], summary="reviewer unavailable", reviewer=identity)


# --------------------------------------------------------------------------------------------------
# reflector
# --------------------------------------------------------------------------------------------------
class LLMReflector:
    """Consolidates lessons and proposes pivots from the tried table (development numbers only)."""

    def __init__(self, client: ModelClient, *, reasoning_effort: str | None = "medium",
                 max_completion_tokens: int | None = REFLECTOR_MAX_TOKENS):
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.identity = f"reflector[{_model_of(client)}]"

    def reflect(self, ctx: ReflectContext) -> ReflectionResult:
        try:
            _, obj = prompts.chat_json(self.client, prompts.reflector_messages(ctx), schema=prompts.REFLECT_SCHEMA,
                                       schema_name="reflection", parse=parse_json_object, reasoning_effort=self.reasoning_effort,
                                       max_completion_tokens=self.max_completion_tokens,
                                       cache_key=prompts.cache_key("reflector", ctx.task.name),
                                       charge_context={"purpose": "reflect"})
        except ModelError:
            return ReflectionResult(lessons=[], pivots=[], summary="reflector unavailable")
        if obj is None:
            return ReflectionResult(lessons=[], pivots=[], summary="reflector reply was not usable")
        lessons: list[dict] = []
        for item in obj.get("lessons")[:30] if isinstance(obj.get("lessons"), list) else []:
            if len(lessons) >= 6:
                break
            if not isinstance(item, dict):
                continue
            text = _text(item.get("text"), 400)
            if not text:
                continue
            kind = str(item.get("kind", "")).strip().lower()
            cls = str(item.get("classification", "")).strip().lower()
            sources = [s for s in (item.get("source_exps") if isinstance(item.get("source_exps"), list) else [])[:10]
                       if isinstance(s, str) and _EXP_ID.fullmatch(s)]
            lessons.append({"text": text, "kind": kind if kind in prompts.LESSON_KINDS else "observation",
                            "source_exps": list(dict.fromkeys(sources)),
                            "classification": cls if cls in prompts.LESSON_CLASSES else "unknown"})
        pivots = [p for p in (_text(x, 200) for x in (obj.get("pivots") if isinstance(obj.get("pivots"), list) else [])[:5]) if p]
        return ReflectionResult(lessons=lessons, pivots=pivots, summary=_text(obj.get("summary"), 400))


# --------------------------------------------------------------------------------------------------
# supervisor: history compaction, launch merging, tool loop
# --------------------------------------------------------------------------------------------------
def compact_history(messages: Sequence[ChatMessage], keep: int = KEEP_EXCHANGES) -> list[ChatMessage]:
    """Keep the system message, the first user message and the last ``keep`` assistant/tool exchanges verbatim; replace the
    content of older tool results with a one-line stub. Tool-call ids stay paired, so ``validate_tool_sequence`` accepts the
    result whenever it accepted the input. Pure: the input list and its messages are not modified.

    Reasoning carried by the Responses client (the private ``_responses_items`` attribute of an assistant message) follows the
    turns: it stays on the exchanges kept verbatim and is dropped, on a copy explicitly marked as carrying none, with the older
    exchanges whose tool results are stubbed (their encrypted reasoning would otherwise grow the prompt forever and belongs to a
    train of thought the stubs no longer support). The Chat client ignores the attribute either way."""
    msgs = list(messages)
    head, rest = msgs[:2], msgs[2:]
    exchanges: list[list[ChatMessage]] = []
    for m in rest:
        if m.role == "assistant" or not exchanges:
            exchanges.append([m])
        else:
            exchanges[-1].append(m)
    if len(exchanges) <= keep:
        return msgs
    out = list(head)
    for ex in exchanges[:-keep] if keep > 0 else exchanges:
        names = {c.id: c.name for m in ex if m.role == "assistant" for c in m.tool_calls}
        for m in ex:
            if m.role == "assistant":
                out.append(drop_items(m) if carries_reasoning(m) else m)     # only a message that carries reasoning is copied
                continue
            if m.role != "tool" or (m.content or "").startswith(STUB_PREFIX):
                out.append(m)
                continue
            status = "error" if (m.content or "").startswith('{"error"') else "ok"
            stub = f"{STUB_PREFIX} ({_slug(names.get(m.tool_call_id or '', 'tool'), 40)}, {status}); call {STATE_TOOL} for current facts]"
            out.append(ChatMessage(role="tool", content=stub, tool_call_id=m.tool_call_id, name=m.name))
    for ex in exchanges[-keep:] if keep > 0 else []:
        out.extend(ex)
    return out


def strip_sealed(obj: Any) -> Any:
    """Defence in depth for tool results: drop any key whose name contains ``sealed`` (recursively)."""
    if isinstance(obj, dict):
        return {k: strip_sealed(v) for k, v in obj.items() if not (isinstance(k, str) and "sealed" in k.lower())}
    if isinstance(obj, (list, tuple)):
        return [strip_sealed(v) for v in obj]
    return obj


def lane_result_view(result: Any) -> dict:
    """A launch result as the model may see it: development numbers only, notes fenced as data."""
    d = result.to_dict() if hasattr(result, "to_dict") else dict(result) if isinstance(result, Mapping) else {}
    out: dict[str, Any] = {k: (prompts.ident(d[k], 40) if d.get(k) is not None else None)
                           for k in ("exp_id", "lane", "status", "outcome", "review_verdict", "blocked_reason")}
    out["delta"], out["score"] = _round(d.get("delta")), _round(d.get("score"))
    out["replicated"] = bool(d.get("replicated"))
    out["notes"] = prompts.data(d["notes"], "notes", 400) if d.get("notes") else ""
    return strip_sealed(out)


@dataclass
class _LaunchRequest:
    template: str
    lanes: list[LaneSpec]
    stages: list[str] | None

    @property
    def key(self) -> tuple:
        return (self.template, tuple(self.stages) if self.stages is not None else None)


DEFAULT_HYPOTHESIS_CHARS = 600


def hypothesis_cap() -> int:
    """The hypothesis length the orchestrator keeps (``orchestrator.HYPOTHESIS_CHARS``), so a lane is never silently cut after
    it was accepted here. Imported lazily (the orchestrator imports this module); ``DEFAULT_HYPOTHESIS_CHARS`` when the
    orchestrator or the constant is unavailable."""
    try:
        from ..orchestrator import HYPOTHESIS_CHARS
    except (ImportError, AttributeError):
        return DEFAULT_HYPOTHESIS_CHARS
    return HYPOTHESIS_CHARS if isinstance(HYPOTHESIS_CHARS, int) and not isinstance(HYPOTHESIS_CHARS, bool) \
        and HYPOTHESIS_CHARS > 0 else DEFAULT_HYPOTHESIS_CHARS


def parse_launch_args(args: Mapping[str, Any]) -> _LaunchRequest | str:
    """Validate ``launch_experiments`` arguments. Returns the request, or an error message (never echoing untrusted text).

    A hypothesis is capped at ``hypothesis_cap()`` characters, the same limit the orchestrator applies."""
    template = args.get("template", "explore")
    if not isinstance(template, str) or not _TEMPLATE.fullmatch(template):
        return "template must be a short name such as 'explore'"
    raw_lanes = args.get("lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        return "lanes must be a non-empty list of {lane, hypothesis, approach?, base?}"
    if len(raw_lanes) > MAX_LANES_PER_CALL:
        return f"at most {MAX_LANES_PER_CALL} lanes per call; a round of 2 to 4 lanes is usually right"
    lanes: list[LaneSpec] = []
    cap = hypothesis_cap()
    for i, raw in enumerate(raw_lanes):
        if not isinstance(raw, dict):
            return f"lanes[{i}] must be an object"
        hypothesis = _text(raw.get("hypothesis"), cap)
        if not hypothesis:
            return f"lanes[{i}].hypothesis must be a non-empty string"
        label = _lane_label(raw.get("lane"))
        approach = raw.get("approach")
        if approach is not None and not isinstance(approach, str):
            return f"lanes[{i}].approach must be a string"
        base = raw.get("base")
        if base in (None, ""):
            base = "incumbent"
        elif not (isinstance(base, str) and (base in ("incumbent", "template") or _EXP_ID.fullmatch(base))):
            return f"lanes[{i}].base must be 'incumbent', 'template' or an experiment id such as e003"
        lanes.append(LaneSpec(lane=label or f"L{i + 1}", hypothesis=hypothesis, approach=_text(approach, 1200), base=base))
    stages = args.get("stages")
    if stages is not None:
        if not (isinstance(stages, list) and len(stages) <= 12 and all(isinstance(s, str) and _STAGE.fullmatch(s) for s in stages)):
            return "stages must be a list of stage names"
        stages = list(stages)
    return _LaunchRequest(template=template, lanes=lanes, stages=stages)


def _unique_labels(requests: Sequence[_LaunchRequest]) -> dict[str, str]:
    """Rename lanes in place so labels are unique across the batch (suffix ``-2``, ``-3`` ...). Returns ``{new: old}``."""
    used: set[str] = set()
    renamed: dict[str, str] = {}
    for req in requests:
        for lane in req.lanes:
            label, n, base = lane.lane, 1, lane.lane
            while label in used:
                n += 1
                suffix = f"-{n}"
                label = f"{base[:MAX_LABEL - len(suffix)]}{suffix}"
            if label != lane.lane:
                renamed[label] = lane.lane
                lane.lane = label
            used.add(label)
    return renamed


def _call_summary(call: ToolCall) -> dict:
    if call.parse_error:
        return {"name": prompts.ident(call.name, 40), "args_summary": "parse_error"}
    args = sanitize(json.dumps(call.arguments, sort_keys=True, ensure_ascii=False, default=str), 200)
    return {"name": prompts.ident(call.name, 40), "args_summary": args}


def _turn_summary(calls: Sequence[ToolCall]) -> str:
    if not calls:
        return "plain-text reply"
    counts: dict[str, int] = {}
    for c in calls:
        counts[prompts.ident(c.name, 40) or "?"] = counts.get(prompts.ident(c.name, 40) or "?", 0) + 1
    return ", ".join(f"{n} x{k}" if k > 1 else n for n, k in counts.items())


def _unique_ids(calls: Sequence[ToolCall]) -> list[ToolCall]:
    """Tool-call ids must be unique and non-empty or the next request is a 400; repair them if a model repeats one."""
    seen: set[str] = set()
    out: list[ToolCall] = []
    for i, c in enumerate(calls):
        cid, n = c.id or f"call_{i}", 1
        while cid in seen:
            cid, n = f"{c.id or f'call_{i}'}_{n}", n + 1
        seen.add(cid)
        out.append(c if cid == c.id else ToolCall(id=cid, name=c.name, arguments=c.arguments, raw_arguments=c.raw_arguments,
                                                  parse_error=c.parse_error))
    return out


def _stable(value: Any) -> Any:
    """``value`` without the keys that change every turn (budget counters, turn counts) and so are not information."""
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _turn_digest(calls: Sequence[ToolCall], replies: Sequence[ChatMessage]) -> str:
    """A fingerprint of one turn: the calls made and what came back, minus the volatile keys. Two equal digests in a row
    mean the model asked the same thing and learned nothing new."""
    parts = []
    for call, reply in zip(calls, replies):
        try:
            body = _stable(json.loads(reply.content or "null"))
        except ValueError:
            body = reply.content
        parts.append([call.name, call.raw_arguments if call.parse_error else call.arguments, body])
    return sha256_hex(canonical_json(parts))


def _calls_signature(calls: Sequence[ToolCall]) -> str:
    """A fingerprint of the tool calls of one turn (names and arguments, not results)."""
    return sha256_hex(canonical_json([[c.name, c.raw_arguments if c.parse_error else c.arguments] for c in calls]))


@dataclass
class _Loop:
    plain: int = 0
    refused: bool = False
    stalled: bool = False
    cut: bool = False               # the last reply was cut off at the output limit: ask for less reasoning next time
    cut_replies: int = 0            # consecutive cut-off replies that called no tool
    same: int = 0                   # consecutive turns with the same calls and the same (non-volatile) results
    last_digest: str = ""
    model_errors: int = 0
    done: bool = False
    refusals: int = 0               # consecutive refused finishes with the same reasons (any other tool call keeps the count)
    refusal_key: str = ""           # fingerprint of those reasons
    refusal_new: bool = False       # a refusal was counted since the loop last looked
    sigs: list[str] = field(default_factory=list)   # signatures of the last two tool turns
    cycle: int = 0                  # consecutive tool turns equal to the turn two back but not to the previous one (ABAB...)
    acted: set[str] = field(default_factory=set)    # (action, pattern) pairs already used: each is nudged and closed once


class LLMSupervisor:
    """Drives the run through ``SupervisorEnv`` tools with Muse Spark. See the module docstring for the guarantees."""

    def __init__(self, client: ModelClient, task: TaskSpec, *, reasoning_effort: str | None = "medium", max_iterations: int = 200,
                 keep_exchanges: int = KEEP_EXCHANGES, max_model_errors: int = 3, max_plain_replies: int = 2,
                 max_completion_tokens: int | None = SUPERVISOR_MAX_TOKENS, timer: Callable[[], float] = time.monotonic,
                 background: str = ""):
        self.client = client
        self.background = background or ""
        self.task = task
        self.reasoning_effort = reasoning_effort
        self.max_iterations = max_iterations
        self.keep_exchanges = keep_exchanges
        self.max_model_errors = max_model_errors
        self.max_plain_replies = max_plain_replies
        self.max_completion_tokens = max_completion_tokens
        self._timer = timer
        self.cache_key = prompts.cache_key("supervisor", task.name)
        self.identity = f"supervisor[{_model_of(client)}]"

    # ---- main loop -------------------------------------------------------------------------------
    def run(self, env: Any) -> None:
        specs = list(env.tool_specs())
        known = {s.name for s in specs}
        messages = prompts.supervisor_messages(self.task, specs, self.background)
        st = _Loop()
        for _ in range(self.max_iterations):
            if env.should_stop():
                return
            self._poll_state(env, st)
            messages = compact_history(messages, self.keep_exchanges)
            effort = _max_effort(self.reasoning_effort, "high") if (st.refused or st.stalled) else self.reasoning_effort
            if st.cut:
                effort = prompts.lower_effort(effort)
            started = self._timer()
            try:
                resp = self.client.chat(messages, tools=specs or None, reasoning_effort=effort,
                                        max_completion_tokens=self.max_completion_tokens, cache_key=self.cache_key)
            except ModelError as exc:
                st.model_errors += 1
                env.note_turn(_model_of(self.client), f"model unavailable (status {exc.status})", tool_calls=[],
                              reasoning_effort=effort, latency_ms=self._ms(started), text=None)
                if exc.status in NON_RETRYABLE_STATUSES:
                    raise                # a rejected key, exhausted credit or a forbidden model: retrying cannot help
                if st.model_errors >= self.max_model_errors:
                    return
                continue
            st.model_errors = 0
            msg = resp.message
            calls = _unique_ids(msg.tool_calls or [])
            text = msg.content or ""
            st.cut = prompts.was_cut_off(resp)
            env.note_turn(str(resp.model or _model_of(self.client)), _turn_summary(calls), tool_calls=[_call_summary(c) for c in calls],
                          reasoning_effort=effort, latency_ms=self._ms(started), text=sanitize(text, 2000) or None)
            if calls:
                st.plain = st.cut_replies = 0
                messages.append(carry_items(msg, ChatMessage(role="assistant", content=msg.content, tool_calls=calls)))
                replies = self._execute(env, calls, known, st)
                messages.extend(replies)
                turn: tuple[Sequence[ToolCall], Sequence[ChatMessage]] | None = (calls, replies)
            elif st.cut:
                self._cut_reply(env, messages, text, st)
                turn = None
            else:
                st.cut_replies = 0
                if text.strip():                  # an empty assistant message is not replayed: the API may refuse it
                    messages.append(carry_items(msg, ChatMessage(role="assistant", content=text)))
                self._plain_reply(env, messages, text, st)
                turn = None
            if not st.done:
                self._watch_loops(env, messages, turn, st)
            if st.done:
                return

    def _ms(self, started: float) -> float:
        return round((self._timer() - started) * 1000.0, 1)

    def _poll_state(self, env: Any, st: _Loop) -> None:
        fn = getattr(env, "state", None)
        if not callable(fn):
            return
        try:
            state = fn()
        except NightshiftError:
            raise
        except Exception:               # a failing state read must not stop the supervisor; it only informs the effort
            return
        if isinstance(state, Mapping):
            st.stalled = bool(state.get("stalled"))

    # ---- plain-text replies ----------------------------------------------------------------------
    def _plain_reply(self, env: Any, messages: list[ChatMessage], text: str, st: _Loop) -> None:
        st.plain += 1
        if st.plain < self.max_plain_replies:
            env.nudge(prompts.NUDGE_NO_TOOL)
            messages.append(ChatMessage(role="user", content=prompts.NUDGE_NO_TOOL))
            return
        st.plain = 0
        self._finish(env, messages, st, sanitize(text, 1500) or "The supervisor ended its turn without a summary.")

    def _finish(self, env: Any, messages: list[ChatMessage], st: _Loop, summary: str) -> None:
        """Call ``env.finish`` on the loop's own initiative. A refusal is relayed to the model and the loop goes on."""
        try:
            result = env.finish(summary=summary)
        except NightshiftError:
            raise                        # BudgetExceeded, HaltRequested, SimulatedCrash and the rest stop the run
        except Exception:
            result = {"accepted": False, "reasons": ["finish failed unexpectedly"]}
        if isinstance(result, Mapping) and result.get("accepted") is True:
            st.done = True
            return
        st.refused = True
        reasons = result.get("reasons") if isinstance(result, Mapping) and isinstance(result.get("reasons"), list) else []
        self._note_refusal(st, reasons)
        messages.append(ChatMessage(role="user", content=prompts.finish_refused_message(reasons)))

    # ---- replies that are cut off, and turns that repeat -----------------------------------------
    def _cut_reply(self, env: Any, messages: list[ChatMessage], text: str, st: _Loop) -> None:
        """A reply with no tool call that hit the output limit (reasoning can use all of it). It is not a summary and not a plain
        reply: nudge for a shorter turn, and close the run after ``MAX_CUT_REPLIES`` in a row."""
        st.plain = 0
        st.cut_replies += 1
        if text.strip():
            messages.append(ChatMessage(role="assistant", content=text))
        if st.cut_replies >= MAX_CUT_REPLIES:
            st.cut_replies = 0
            self._finish(env, messages, st, prompts.FORCED_FINISH_CUT)
            return
        env.nudge(prompts.NUDGE_CUT)
        messages.append(ChatMessage(role="user", content=prompts.NUDGE_CUT))

    @staticmethod
    def _note_refusal(st: _Loop, reasons: Any) -> None:
        """Count a refused finish: consecutive refusals with the same reasons build a streak, different reasons restart it."""
        items = [sanitize(r, 300) for r in list(reasons)[:8]] if isinstance(reasons, (list, tuple)) else []
        key = sha256_hex(canonical_json(items))
        st.refusals = st.refusals + 1 if st.refusals and key == st.refusal_key else 1
        st.refusal_key, st.refusal_new = key, True

    def _watch_loops(self, env: Any, messages: list[ChatMessage], turn: tuple[Sequence[ToolCall], Sequence[ChatMessage]] | None,
                     st: _Loop) -> None:
        """Look for a model that is going nowhere and act at most once per iteration (closing beats nudging).

        Three patterns, each with a static nudge and a static, claim-free closing summary:
        * the same tool turn with the same non-volatile results ``REPEAT_NUDGE_AT`` / ``REPEAT_FINISH_AT`` times in a row;
        * a period-2 cycle of tool calls (get_state, finish, get_state, finish ...): ``CYCLE_NUDGE_AT`` / ``CYCLE_FINISH_AT``
          repetitions of the pair;
        * the same refused finish (equal reasons) ``REFUSAL_NUDGE_AT`` / ``REFUSAL_FINISH_AT`` times in a row, wherever the
          requests came from. The nudge lists the ways out: fix the cause, acknowledge, or finish claiming nothing.
        The cycle and refusal patterns are acted on ONCE each (per pair of calls, per set of reasons): a closing finish that the
        gate refused would be refused again for the same reasons, and every finish request costs a final audit sweep, so after
        it the loop only goes on until ``env.should_stop()``. The counters restart after a closing finish."""
        nudge: list[str] = []
        close: list[str] = []
        if turn is not None:
            calls, replies = turn
            digest = _turn_digest(calls, replies)
            st.same = st.same + 1 if digest == st.last_digest else 1
            st.last_digest = digest
            if st.same == REPEAT_NUDGE_AT:
                nudge.append(prompts.NUDGE_REPEAT)
            elif st.same >= REPEAT_FINISH_AT:
                close.append(prompts.FORCED_FINISH_STUCK)
            sig = _calls_signature(calls)
            two_back, previous = (st.sigs[-2] if len(st.sigs) >= 2 else None), (st.sigs[-1] if st.sigs else None)
            st.cycle = st.cycle + 1 if (sig == two_back and sig != previous) else 0
            st.sigs = (st.sigs + [sig])[-2:]
            pair = "|".join(sorted((sig, previous or "")))
            if st.cycle >= 2 * CYCLE_FINISH_AT - 2 and self._once(st, "close:cycle", pair):
                close.append(prompts.FORCED_FINISH_CYCLE)
            elif st.cycle == 2 * CYCLE_NUDGE_AT - 2 and self._once(st, "nudge:cycle", pair):
                nudge.append(prompts.NUDGE_CYCLE)
        if st.refusal_new:
            st.refusal_new = False
            if st.refusals >= REFUSAL_FINISH_AT and self._once(st, "close:refused", st.refusal_key):
                close.insert(0, prompts.FORCED_FINISH_REFUSED)
            elif st.refusals == REFUSAL_NUDGE_AT and self._once(st, "nudge:refused", st.refusal_key):
                nudge.insert(0, prompts.NUDGE_REFUSED)
        if close:
            st.same = st.cycle = st.refusals = 0
            st.refusal_key, st.sigs = "", []
            self._finish(env, messages, st, close[0])
        elif nudge:
            env.nudge(nudge[0])
            messages.append(ChatMessage(role="user", content=nudge[0]))

    @staticmethod
    def _once(st: _Loop, action: str, key: str) -> bool:
        """True the first time ``(action, key)`` is seen in this run, False afterwards."""
        token = f"{action}:{key}"
        if token in st.acted:
            return False
        st.acted.add(token)
        return True

    # ---- tool execution --------------------------------------------------------------------------
    def _execute(self, env: Any, calls: Sequence[ToolCall], known: set[str], st: _Loop) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        i = 0
        while i < len(calls):
            if calls[i].name == LAUNCH_TOOL and LAUNCH_TOOL in known:
                j = i
                while j < len(calls) and calls[j].name == LAUNCH_TOOL:
                    j += 1
                out.extend(self._launch_group(env, calls[i:j], st))
                i = j
            else:
                out.append(self._reply(calls[i], self._run_tool(env, calls[i], known, st)))
                i += 1
        return out

    @staticmethod
    def _reply(call: ToolCall, result: Any) -> ChatMessage:
        body = json.dumps(result, default=str)
        if len(body) > MAX_TOOL_RESULT_CHARS:
            body = body[:MAX_TOOL_RESULT_CHARS] + '..."[truncated]'
        return ChatMessage(role="tool", content=body, tool_call_id=call.id, name=call.name or None)

    def _run_tool(self, env: Any, call: ToolCall, known: set[str], st: _Loop) -> dict:
        if call.parse_error:
            return {"error": f"parse_error: {sanitize(call.parse_error, 160)}", "hint": self._parse_hint(st)}
        if call.name not in known:
            return {"error": "unknown_tool", "valid_tools": sorted(known)}
        try:
            result = env.call_tool(call.name, dict(call.arguments))
        except InvalidComposition as exc:
            result = {"error": "invalid_composition", "details": [sanitize(e, 200) for e in exc.errors[:5]]}
        except NightshiftError:
            raise                        # BudgetExceeded, HaltRequested, SimulatedCrash and the rest stop the run
        except Exception as exc:         # a tool bug must not leave the call unanswered
            result = {"error": f"tool_failed: {type(exc).__name__}"}
        if not isinstance(result, dict):
            result = {"result": result}
        result = strip_sealed(result)
        if call.name == STATE_TOOL:
            st.stalled = bool(result.get("stalled"))
        elif call.name == FINISH_TOOL:
            if result.get("accepted") is True:
                st.done = True
            elif result.get("accepted") is False:
                st.refused = True
                self._note_refusal(st, result.get("reasons"))
        elif call.name == "replicate" and "error" not in result:
            st.refusals = 0              # real work was done since the last refusal: a new refusal starts a new streak
        return result

    @staticmethod
    def _parse_hint(st: _Loop) -> str:
        if st.cut:
            return "your reply was cut off at the output limit; send shorter arguments as a single JSON object"
        return "send the arguments as a single JSON object that matches the tool's schema"

    def _launch_group(self, env: Any, calls: Sequence[ToolCall], st: _Loop) -> list[ChatMessage]:
        """Consecutive launch calls: validate each, merge those with equal (template, stages) into one ``env.launch``."""
        results: dict[int, dict] = {}
        valid: list[tuple[int, _LaunchRequest]] = []
        for idx, call in enumerate(calls):
            if call.parse_error:
                results[idx] = {"error": f"parse_error: {sanitize(call.parse_error, 160)}", "hint": self._parse_hint(st)}
                continue
            parsed = parse_launch_args(call.arguments)
            if isinstance(parsed, str):
                results[idx] = {"error": "invalid_arguments", "detail": parsed}
            else:
                valid.append((idx, parsed))
        batches: list[list[tuple[int, _LaunchRequest]]] = []
        for item in valid:
            # Calls with equal (template, stages) share one batch, up to the 8 lanes the environment accepts per launch (lanes
            # beyond that would be blocked and lost, so a larger group becomes a second batch).
            if (batches and batches[-1][0][1].key == item[1].key
                    and sum(len(r.lanes) for _, r in batches[-1]) + len(item[1].lanes) <= MAX_LANES_PER_CALL):
                batches[-1].append(item)
            else:
                batches.append([item])
        for batch in batches:
            results.update(self._launch_batch(env, batch))
        if batches:
            st.refusals = 0              # experiments were launched since the last refusal: a new refusal starts a new streak
        return [self._reply(call, results[idx]) for idx, call in enumerate(calls)]

    def _launch_batch(self, env: Any, batch: Sequence[tuple[int, _LaunchRequest]]) -> dict[int, dict]:
        reqs = [r for _, r in batch]
        renamed = _unique_labels(reqs)
        lanes = [lane for r in reqs for lane in r.lanes]
        try:
            raw = env.launch(reqs[0].template, lanes, reqs[0].stages)
        except InvalidComposition as exc:
            err = {"error": "invalid_composition", "details": [sanitize(e, 200) for e in exc.errors[:5]]}
            return {idx: dict(err) for idx, _ in batch}
        except NightshiftError:
            raise
        except Exception as exc:
            err = {"error": f"launch_failed: {type(exc).__name__}"}
            return {idx: dict(err) for idx, _ in batch}
        views = [lane_result_view(r) for r in (raw or [])]
        if len(views) != len(lanes):                       # align by label when the env returned a different number
            by_label = {v["lane"]: v for v in views}
            views = [by_label.get(lane.lane, {"lane": lane.lane, "status": "unknown", "notes": "no result was returned"})
                     for lane in lanes]
        out: dict[int, dict] = {}
        pos = 0
        for idx, req in batch:
            share = views[pos:pos + len(req.lanes)]
            pos += len(req.lanes)
            body: dict[str, Any] = {"results": share}
            mine = {new: old for new, old in renamed.items() if new in {lane.lane for lane in req.lanes}}
            if mine:
                body["lane_renames"] = mine
            out[idx] = body
        return out


# --------------------------------------------------------------------------------------------------
# suite factory
# --------------------------------------------------------------------------------------------------
def _load_muse_code_worker(task: TaskSpec, run_dir: Path, routing: Mapping[str, Any], gate: Any, ledger: Any) -> Any:
    try:
        module = importlib.import_module("nightshift.agents.muse_code")
    except ImportError as exc:
        raise ValueError("implementer 'muse_code' needs nightshift.agents.muse_code, which is not installed or failed to import; "
                         "use implementer 'patch' instead") from exc
    factory = getattr(module, "build_implementer", None) or getattr(module, "MuseCodeWorker", None)
    if not callable(factory):
        raise ValueError("nightshift.agents.muse_code exposes neither build_implementer(...) nor MuseCodeWorker")
    return factory(task, run_dir, routing=routing, gate=gate, ledger=ledger)


def build_llm_suite(task: TaskSpec, run_dir: Path, routing: Mapping[str, Any], gate: Any, ledger: Any, *,
                    implementer: str = "patch", background: str = "") -> AgentSuite:
    """Wire every Muse Spark role. Each role gets its own metered client (charges are attributed to the role name).

    ``run_dir`` and ``ledger`` are accepted for factory symmetry (and handed to a Muse Code worker); the LLM roles read
    everything they need from their contexts. ``implementer="muse_code"`` imports ``nightshift.agents.muse_code`` lazily
    and raises ``ValueError`` when it is not installed.

    ``background`` is operator-supplied background material (related-work notes, paper excerpts). The supervisor gets it in
    its first user message (capped at 60,000 characters, about 15,000 tokens, which every supervisor turn then carries) and
    the designer in each lane's message (capped at 8,000 characters); both fence it as untrusted data and say so when it was
    cut. The other roles never see it. The default (``""``) leaves every prompt unchanged.
    """
    from ..audit.judge import LLMJudge
    from ..models.config import client_for

    if implementer not in ("patch", "muse_code"):
        raise ValueError(f"unknown implementer {implementer!r}; choose 'patch' or 'muse_code'")
    worker: Any = _load_muse_code_worker(task, Path(run_dir), routing, gate, ledger) if implementer == "muse_code" else None

    def client(role: str) -> ModelClient:
        return client_for(role, routing, gate=gate, actor=role)

    def effort(role: str) -> str | None:
        return routing[role].reasoning_effort

    if worker is None:
        worker = PatchWorker(client("implementer"), reasoning_effort=effort("implementer"))
    implementer_model = getattr(worker, "model", None) or routing["implementer"].model
    return AgentSuite(
        backend="meta",
        supervisor=LLMSupervisor(client("supervisor"), task, reasoning_effort=effort("supervisor"), background=background),
        designer=LLMDesigner(client("designer"), reasoning_effort=effort("designer"), background=background),
        implementer=worker,
        reviewer=LLMReviewer(client("reviewer"), name="reviewer", implementer_model=implementer_model,
                             reasoning_effort=effort("reviewer")),
        reflector=LLMReflector(client("reflector"), reasoning_effort=effort("reflector")),
        judge=LLMJudge(client("judge"), reasoning_effort=effort("judge")),
        reviewer_hard=LLMReviewer(client("reviewer_hard"), name="reviewer-hard", implementer_model=implementer_model,
                                  reasoning_effort=effort("reviewer_hard")),
        final_judge=LLMJudge(client("auditor_final"), reasoning_effort=effort("auditor_final")),
        models={role: ep.model for role, ep in routing.items()},
    )
