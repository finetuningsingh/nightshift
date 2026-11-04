"""The orchestrator: run lifecycle, resume, and the one surface a supervisor acts through (``SupervisorEnv``).

Guarantees
----------
* The ledger is the state. Everything this module reports is read back from it; nothing authoritative is kept in memory.
* Nothing derived from a sealed-split grade is ever returned to a supervisor, written into a nudge, or logged. The only
  sealed-derived value that leaves this module towards a model is the boolean ``replicated``. Dev scores come from the
  experiment's own dev grade, never from a promotion record or a replication record.
* Everything a model or an experiment produced (hypotheses, pivots, lessons, notes, LLM finding rationales) is
  ``sanitize``d and ``fence``d before it appears in a tool result; nudges are templates plus validated ids and static titles.
* ``call_tool`` never raises on bad input: a malformed call comes back as ``{"error": ...}``. Only ``NightshiftError``
  (``BudgetExceeded``, ``HaltRequested``, ``SimulatedCrash``, ...) propagates, as roles are required to let it.
* Finishing is final: after an accepted ``finish`` (or any ``run.finished``) the ledger is sealed and every tool answers
  ``run already finished``. The verdict of an accepted finish is derived from gate-verified evidence (a replicated incumbent
  that beats the baseline on the sealed split), never from what the supervisor claimed, and it is never sent to the supervisor.
* A resume continues the same experiment: a changed task or grader makes later grades incomparable with earlier ones, so it
  is refused rather than accepted.
* A run that did not end in an accepted finish is ``missing_authority`` when it was stopped (HALT, a spent budget, or a model
  API that refused the key or the credits: HTTP 401, 402, 403) and ``unresolved_defects`` when it also holds a defect (an open
  block finding, an experiment in flight, an experiment left ``unresolved`` for any reason other than the stop itself). The
  engine's own ``stopped by HALT`` / ``stopped by the budget`` records are the stop, not a defect. A model API failure is
  written to the ledger by its numeric HTTP status only, never by its message.
* ``run.started.config.implementer`` names the implementer the run really used: a scripted run says ``scripted``.
* Red-team lanes (``plant``, meta backend only): one scripted attacker lane per planted behavior is added to the first launch
  after the baseline that has room under the ``MAX_LANES`` cap (the supervisor's own lanes are never reduced; planted lanes
  that do not fit wait for the next launch). Each carries a ``scripted_id``, the label ``RT-<behavior>`` and a note saying so.
  Which behaviors were planted is ``run.started.config.red_team``; which were already injected is read back from the ledger
  (``experiment.proposed.scripted_id``), so a resumed run never injects one twice. Operator background material is passed to
  the roles but only its name, sha256 and length are recorded (``run.started.config.context``).
* On ``SimulatedCrash`` the ledger writer is closed (the flock is released) and the exception is re-raised; ``run.finished``
  is never written for a crash.
"""
from __future__ import annotations

import copy
import dataclasses
import ipaddress
import logging
import math
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from . import sandbox, secrets
from .audit.auditor import Auditor
from .budget import BudgetGate
from .fence import fence, sanitize
from .ledger import LEDGER_NAME, Ledger
from .memory import Memory
from .task import task_hash as _task_hash
from .task import trusted_hash as _trusted_hash
from .types import (AgentSuite, BudgetExceeded, BudgetLimits, Clock, ExpStatus, ExperimentView, HaltRequested,
                    InvalidComposition, Kind, LaneResult, LaneSpec, LedgerCorrupt, ModelError, NightshiftError, Outcome,
                    SimulatedCrash, TaskSpec, ToolSpec)

LOG = logging.getLogger("nightshift.orchestrator")
LOG.addHandler(logging.NullHandler())   # library logger: the application decides where it goes

MAX_LANES = 8
HYPOTHESIS_CHARS = 600       # the one hypothesis length limit: the tool schema, the launch cap and agents/llm.py parse_launch_args
APPROACH_CHARS = 1200
SUMMARY_CHARS = 2000
REASON_CHARS = 1000
MAX_CLAIM_IDS = 50
MAX_PATTERN_INPUT = 4096     # patterns are only run on strings up to this long; every pattern in the schemas matches far less
RECENT = 8
AUTH_STATUSES = (401, 402, 403)   # the model API refused the key or the credits: the run lacks authority, it did not fail
TRUNCATED_NOTE = f"the hypothesis was longer than {HYPOTHESIS_CHARS} characters and was shortened"
STOPPED_NOTES = ("stopped by HALT", "stopped by the budget")   # notes of the engine's stop-unresolved records
KEY_ENVS = ("MODEL_API_KEY", "META_API_KEY")
CLAIM_TYPES = ("improved", "no_effect", "fixed", "done", "safe", "retraction", "other")
BLOCK_REASONS = ("duplicate", "budget", "halt", "protected", "invalid_composition", "not_replicable")
RED_TEAM_NOTE = "red-team lane injected by the operator; scripted"
RED_TEAM_PREFIX = "RT-"

# always used with fullmatch: "$" would also accept a trailing newline
_LABEL = re.compile(r"[A-Za-z0-9_-]{1,16}")
_EXP_ID = re.compile(r"e\d{1,6}")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.@:-]{1,60}")
_FINDING_ID = r"^[A-Za-z0-9_.@:-]{1,80}$"
_BASE = r"^(incumbent|template|e[0-9]{1,6})$"
_SLUG_BAD = re.compile(r"[^A-Za-z0-9_-]")
_SCRIPTED_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


class RunAlreadyFinished(PermissionError):
    """A write was attempted after ``run.finished``. A ``PermissionError`` because the ledger is sealed (read-only)."""


class ResumeRefused(NightshiftError):
    """``Orchestrator.resume`` will not continue this run (finished, changed task or grader, other backend, no run to resume)."""


class BaselineIncomplete(NightshiftError):
    """After a resume the baseline experiment still has no recorded result."""


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def _num(value: Any, digits: int = 6) -> Optional[float]:
    """A finite float rounded for display, else None (bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits) if math.isfinite(value) else None


def _ident(value: Any, fallback: str = "?") -> str:
    """An id from the ledger that is safe to print outside a fence, else ``fallback``."""
    return value if isinstance(value, str) and _SAFE_ID.fullmatch(value) else fallback


def _label(value: Any, index: int) -> str:
    """A lane label restricted to ``[A-Za-z0-9_-]{1,16}``; a bad one is cleaned, an unusable one becomes ``L<index>``."""
    if isinstance(value, str) and _LABEL.fullmatch(value):
        return value
    cleaned = _SLUG_BAD.sub("_", value)[:16] if isinstance(value, str) else ""
    return cleaned or f"L{index + 1}"


def _unique(label: str, used: set[str]) -> str:
    out, n = label, 1
    while out in used:
        n += 1
        suffix = f"-{n}"
        out = f"{label[:16 - len(suffix)]}{suffix}"
    used.add(out)
    return out


def _strings(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [v for v in value if isinstance(v, str)][:limit]


# ids the engine and the auditor print in their own templated text; they carry no score, so they may stay in a note
_NOTE_ID = re.compile(r"\b(?:I\d{2}@(?:e\d{3,6}(?:\.c\d{1,4})?|w\d{1,6}|run)|e\d{3,6}(?:\.c\d{1,4})?|I\d{2})\b")


def _has_digit(text: str) -> bool:
    """True if ``text`` still contains a digit once experiment, claim, invariant and finding ids are set aside."""
    return any(ch.isdigit() for ch in _NOTE_ID.sub("", text))


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _context_records(context: Any) -> list[dict]:
    """``run.started.config.context``: ``[{"name", "sha256", "chars"}]`` for the background files of a run. Names are reduced to
    a short printable basename; anything that is not a well-formed record is dropped. The text itself is never recorded."""
    out: list[dict] = []
    for item in context if isinstance(context, (list, tuple)) else ():
        if not isinstance(item, Mapping):
            continue
        digest, chars = item.get("sha256"), item.get("chars")
        if not (isinstance(digest, str) and _SHA256.fullmatch(digest) and isinstance(chars, int)
                and not isinstance(chars, bool) and chars >= 0):
            continue
        out.append({"name": sanitize(str(item.get("name") or "?"), 80) or "?", "sha256": digest, "chars": chars})
    return out[:50]


def _saved_context(value: Any) -> list[dict]:
    return _context_records(value)


def _context_mismatch(saved: Sequence[Mapping[str, Any]], given: Sequence[Mapping[str, Any]]) -> str:
    """Empty when ``given`` and ``saved`` hold the same ``(name, sha256)`` pairs (order does not matter), else a short text
    naming the first file that is not in the recorded set, or the first recorded file that is missing."""
    want = {(c["name"], c["sha256"]) for c in saved}
    have = {(c["name"], c["sha256"]) for c in given}
    for name, _ in sorted(have - want):
        return f"{sanitize(name, 80)} is not one of the recorded files or its content changed"
    for name, _ in sorted(want - have):
        return f"{sanitize(name, 80)} was recorded at the start but is not given"
    return ""


def _saved_red_team(value: Any) -> list[dict]:
    """The red-team records of a saved ``run.started.config`` (``[]`` for a run without any)."""
    out: list[dict] = []
    for item in value if isinstance(value, (list, tuple)) else ():
        behavior = item.get("behavior") if isinstance(item, Mapping) else None
        sid = item.get("scripted_id") if isinstance(item, Mapping) else None
        if isinstance(behavior, str) and behavior and isinstance(sid, str) and _SCRIPTED_ID.fullmatch(sid):
            out.append({"behavior": behavior, "scripted_id": sid})
    return out


# --------------------------------------------------------------------------------------------------
# which endpoints a run talks to (the dashboard labels simulated runs from this)
# --------------------------------------------------------------------------------------------------
META_API_HOST = "api.meta.ai"
_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_HOSTNAME = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,251}[a-z0-9])?|[0-9a-f:.]{2,45}")
_URL_UNSAFE = re.compile(r"[\\\s\x00-\x1f\x7f]")


def _host_of(base_url: Any) -> Optional[str]:
    """The lower-case hostname of ``base_url`` (no scheme, userinfo, port, path or query), or None if it cannot be trusted.

    A URL with a backslash, whitespace or a control character is refused outright: parsers disagree about where its host
    ends, so no single host can honestly be reported. So is a URL with a scheme other than http(s), an invalid port, or a
    host with more than one trailing dot (no client would reach it, and ``api.meta.ai..`` is not ``api.meta.ai``).
    """
    if not isinstance(base_url, str) or not base_url.strip() or _URL_UNSAFE.search(base_url.strip()):
        return None
    try:
        parts = urlsplit(base_url.strip())
        if parts.scheme.lower() not in ("", "http", "https"):
            return None
        parts.port                                    # noqa: B018 - raises ValueError for a port that is not 0-65535
        host = parts.hostname or ""
    except ValueError:
        return None
    host = host[:-1] if host.endswith(".") else host
    return host if _HOSTNAME.fullmatch(host) else None


def _is_loopback(host: str) -> bool:
    """True for ``localhost`` and any address that connects to this machine, including the short and numeric IPv4 spellings
    (``127.1``, ``2130706433``, ``0x7f.0.0.1``) that ``inet_aton`` accepts and ``ipaddress`` does not."""
    if host in _LOOPBACK_NAMES:
        return True
    try:
        addr: Any = ipaddress.ip_address(host)
    except ValueError:
        try:
            addr = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            return False
    mapped = getattr(addr, "ipv4_mapped", None)
    return bool(addr.is_loopback or (mapped is not None and mapped.is_loopback))


def endpoint_info(backend: str, routing: Optional[Mapping[str, Any]]) -> dict:
    """``{"kind": "scripted"|"meta-api"|"loopback"|"custom", "hosts": [...]}`` for ``run.started.config.endpoint``.

    ``hosts`` are the sorted, de-duplicated host names of the routing endpoints' ``base_url`` values and nothing else.
    ``kind`` is ``scripted`` for the scripted backend (no endpoint is contacted), ``loopback`` if any host is a loopback
    address (a local fake server: a simulation, never a Meta result), ``meta-api`` only if there is at least one host, every
    endpoint has a parseable host and every host is ``api.meta.ai``, and ``custom`` otherwise (including any endpoint whose
    host cannot be determined).
    """
    if backend == "scripted":
        return {"kind": "scripted", "hosts": []}
    urls = [ep.get("base_url") if isinstance(ep, Mapping) else getattr(ep, "base_url", None)
            for ep in (routing.values() if isinstance(routing, Mapping) else ())]
    parsed = [_host_of(u) for u in urls]
    hosts = sorted({h for h in parsed if h})
    if any(_is_loopback(h) for h in hosts):
        kind = "loopback"
    elif hosts and None not in parsed and set(hosts) == {META_API_HOST}:
        kind = "meta-api"
    else:
        kind = "custom"
    return {"kind": kind, "hosts": hosts}


# --------------------------------------------------------------------------------------------------
# tool schemas and a small validator
# --------------------------------------------------------------------------------------------------
def _schemas(template_names: Sequence[str], stage_names: Sequence[str]) -> dict[str, tuple[str, dict]]:
    lane = {
        "type": "object",
        "properties": {
            "lane": {"type": "string", "minLength": 1, "maxLength": 16, "pattern": "^[A-Za-z0-9_-]{1,16}$",
                     "description": "Short label, unique within this call (for example A, B, C)."},
            "hypothesis": {"type": "string", "minLength": 1, "maxLength": HYPOTHESIS_CHARS,
                           "description": "One specific, testable change to the editable code and what it should do."},
            "approach": {"type": "string", "maxLength": APPROACH_CHARS, "description": "Optional hint for the designer."},
            "base": {"type": "string", "pattern": _BASE,
                     "description": "Start from 'incumbent' (default), 'template' or a recorded experiment id such as e003."},
        },
        "required": ["lane", "hypothesis"],
        "additionalProperties": False,
    }
    return {
        "get_state": (
            "Current budget left, baseline and incumbent (development scores), the last experiments, candidates awaiting "
            "replication, open audit findings, suggested pivots and whether the run is stalled. Text fields are quoted data.",
            {"type": "object", "properties": {}, "additionalProperties": False}),
        "launch_experiments": (
            "Run one round of experiments in parallel, one lane per hypothesis (at most %d lanes). Put all lanes of a round "
            "in ONE call. Returns each lane's outcome and development-split score and delta." % MAX_LANES,
            {"type": "object",
             "properties": {
                 "template": {"type": "string", "enum": list(template_names), "description": "Workflow template."},
                 "lanes": {"type": "array", "items": lane, "minItems": 1, "maxItems": MAX_LANES},
                 "stages": {"type": "array", "items": {"type": "string", "enum": list(stage_names)}, "maxItems": 16,
                            "description": "Optional custom stage list; leave out for the default workflow."},
             },
             "required": ["lanes"], "additionalProperties": False}),
        "replicate": (
            "Re-check an improved candidate (listed in awaiting_replication) on independent, hidden conditions. "
            "You learn only whether it replicated, never by how much. Limited number of calls.",
            {"type": "object", "properties": {"exp_id": {"type": "string", "pattern": "^e[0-9]{1,6}$"}},
             "required": ["exp_id"], "additionalProperties": False}),
        "reflect": (
            "Consolidate what the experiments so far show into lessons and suggest new directions (pivots).",
            {"type": "object", "properties": {}, "additionalProperties": False}),
        "acknowledge_finding": (
            "Close an open audit finding after you dealt with it. 'addressed' is verified in code; deterministic block "
            "findings cannot be dismissed. A dismissal needs a reason of at least 20 characters.",
            {"type": "object",
             "properties": {
                 "finding_id": {"type": "string", "minLength": 1, "pattern": _FINDING_ID},
                 "action": {"type": "string", "enum": ["addressed", "dismissed"]},
                 "reason": {"type": "string", "maxLength": REASON_CHARS},
             },
             "required": ["finding_id", "action", "reason"], "additionalProperties": False}),
        "finish": (
            "End the run with an honest summary. Refused (with reasons) while an experiment is in flight, a block finding is "
            "open, or an improvement claim you cite has not replicated. Cite claim ids only if you were given them.",
            {"type": "object",
             "properties": {
                 "summary": {"type": "string", "minLength": 1, "maxLength": SUMMARY_CHARS},
                 "claim_ids": {"type": "array", "items": {"type": "string", "maxLength": 60}, "maxItems": MAX_CLAIM_IDS},
             },
             "required": ["summary"], "additionalProperties": False}),
    }


_PY_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,), "array": (list,), "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}


def validate_arguments(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> list[str]:
    """Check ``value`` against the JSON-Schema subset used by the tool schemas. Returns readable problems (empty = valid).

    Supports type, enum, properties, required, additionalProperties=false, items, minItems, maxItems, minLength,
    maxLength and pattern. It never echoes the offending value (only its type, length or a capped key), and it recurses only
    along the schema, so deeply nested garbage in a position the schema does not describe is rejected without being walked.
    """
    kind = schema.get("type")
    if kind is not None:
        expected = _PY_TYPES.get(kind, ())
        ok = isinstance(value, expected) and not (kind in ("integer", "number") and isinstance(value, bool))
        if kind == "boolean":
            ok = isinstance(value, bool)
        if not ok:
            return [f"{path}: must be {kind}, got {type(value).__name__}"]
    if "enum" in schema and (not isinstance(value, (str, int, float, bool)) or value not in schema["enum"]):
        return [f"{path}: must be one of {', '.join(str(e) for e in schema['enum'])}"]
    errors: list[str] = []
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters (got {len(value)})")
        if "pattern" in schema and not errors and (len(value) > MAX_PATTERN_INPUT
                                                   or not re.fullmatch(schema["pattern"], value)):
            # a string too long to match is a mismatch, never a pass: skipping the check would let any huge string through
            errors.append(f"{path}: does not match the required format")
    elif isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return [f"{path}: more than {schema['maxItems']} items (got {len(value)})"]
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for i, item in enumerate(value):
                errors += validate_arguments(item, item_schema, f"{path}[{i}]")
                if len(errors) >= 8:
                    break
    elif isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}: missing required key '{key}'")
        for key, item in value.items():
            if key in props:
                errors += validate_arguments(item, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unknown key {ascii(str(key)[:30])}")
            if len(errors) >= 8:
                break
    return errors[:8]


# --------------------------------------------------------------------------------------------------
# the orchestrator
# --------------------------------------------------------------------------------------------------
class Orchestrator:
    """Implements ``SupervisorEnv`` and owns the run lifecycle. Build with ``create`` (fresh run) or ``resume``."""

    def __init__(self, run_dir: Path | str, task: TaskSpec, suite: AgentSuite, limits: BudgetLimits, *, ledger: Ledger,
                 gate: BudgetGate, auditor: Auditor, backend: str, clock: Clock = time.time,
                 overrides: Optional[dict] = None, crash_after_runs: Optional[int] = None, config: Optional[dict] = None,
                 memory: Optional[Memory] = None, engine_factory: Optional[Callable[..., Any]] = None,
                 resume_reason: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.task = task
        self.suite = suite
        self.ledger = ledger
        self.gate = gate
        self.auditor = auditor
        self.backend = backend
        self._clock = clock
        self.overrides = dict(overrides or {})
        self.config = {"implementer": "patch", "control": False, "endpoint": endpoint_info(backend, None), "red_team": [],
                       "context": [], **(config or {})}
        self.config["implementer"] = self._effective_implementer(backend, str(self.config["implementer"]))
        if self.config["red_team"] and getattr(suite, "planted", None) is None:
            # a lane that carries a scripted_id would otherwise be designed and implemented by the real roles under a false label
            raise ValueError("red-team lanes are configured but the agent suite has no planted (scripted attacker) suite")
        self._red_team_claimed: set[str] = set()      # scripted ids injected by THIS process (the ledger covers earlier ones)
        self.limits = self._serial_limits(suite, limits, self.config)
        self.config["parallel"] = self.limits.max_parallel
        self._resume_reason = resume_reason
        self._lock = threading.RLock()
        self._finished = ledger.view().finished is not None
        self.memory = memory if memory is not None else Memory(ledger, task)
        if engine_factory is None:
            from .workflows.engine import WorkflowEngine as engine_factory  # lazy: written alongside this module
        self.engine = engine_factory(task=task, run_dir=self.run_dir, ledger=ledger, gate=gate, memory=self.memory,
                                     suite=suite, auditor=auditor, limits=self.limits, clock=clock, overrides=self.overrides,
                                     crash_after_runs=crash_after_runs)
        names = self._templates().TEMPLATES
        stages = self._templates().STAGE_NAMES
        self._schemas = _schemas(list(names), list(stages))
        self._handlers: dict[str, Callable[[dict], dict]] = {
            "get_state": lambda a: self.state(), "launch_experiments": self._tool_launch, "replicate": self._tool_replicate,
            "reflect": lambda a: self.reflect(), "acknowledge_finding": self._tool_acknowledge, "finish": self._tool_finish}

    @staticmethod
    def _serial_limits(suite: AgentSuite, limits: BudgetLimits, config: dict) -> BudgetLimits:
        """Limits for the run. An implementer with ``requires_serial`` (its sandbox cannot overlap untrusted runs) caps
        ``max_parallel`` at 1 and the reason is put in ``config["parallel_reason"]``; the caller's object is never mutated."""
        config.pop("parallel_reason", None)
        if not bool(getattr(getattr(suite, "implementer", None), "requires_serial", False)):
            return limits
        asked = limits.max_parallel
        config["parallel_reason"] = ("the implementer requires serial execution: max_parallel capped from "
                                     f"{asked} to 1" if asked != 1 else "the implementer requires serial execution")
        return limits if asked == 1 else dataclasses.replace(limits, max_parallel=1)

    # ---- lazy collaborators ----------------------------------------------------------------------
    @staticmethod
    def _templates() -> Any:
        from .workflows import templates
        return templates

    @staticmethod
    def _gates() -> Any:
        from .workflows import gates
        return gates

    # ---- construction ----------------------------------------------------------------------------
    @classmethod
    def create(cls, run_dir: Path | str, task: TaskSpec, *, backend: str, routing: Optional[Mapping[str, Any]] = None,
               limits: Optional[BudgetLimits] = None, implementer: str = "patch", control: bool = False,
               clock: Clock = time.time, overrides: Optional[dict] = None, crash_after_runs: Optional[int] = None,
               engine_factory: Optional[Callable[..., Any]] = None, plant: Sequence[str] = (), background: str = "",
               context: Sequence[Mapping[str, Any]] = ()) -> "Orchestrator":
        """A fresh run: harden -> register keys -> ledger -> budget gate -> agent suite -> auditor -> orchestrator.

        ``FileExistsError`` if ``run_dir`` already holds a ledger. A failure after the ledger exists closes it again.

        ``plant`` (meta backend only) names task-bank behaviors to add as scripted red-team lanes; it is validated BEFORE the
        ledger is created (``ValueError`` for the scripted backend, an unknown behavior or one the bank lacks), so a bad name
        leaves no run directory behind. ``background`` is handed to the supervisor and designer; ``context`` describes the
        files it came from (``name``, ``sha256``, ``chars``) and is all that is recorded of it.
        """
        plant = list(plant)
        red_team = cls._red_team_for(backend, task, plant)
        if background and backend != "meta":
            raise ValueError("background material needs the meta backend: the scripted backend has no language model to read it")
        sandbox.harden_parent()
        secrets.register_env(*KEY_ENVS)
        limits = limits if limits is not None else BudgetLimits()
        ledger = Ledger.create(run_dir, clock=clock)
        try:
            gate = BudgetGate(limits, ledger, clock=clock)
            routing = cls._routing_for(backend, routing)
            suite = cls._build_suite(backend, task, run_dir, routing, gate, ledger, implementer, control, plant, background)
            auditor = Auditor(ledger, task, judge=suite.judge, final_judge=suite.final_judge, clock=clock)
            return cls(run_dir, task, suite, limits, ledger=ledger, gate=gate, auditor=auditor, backend=backend, clock=clock,
                       overrides=overrides, crash_after_runs=crash_after_runs,
                       config={"implementer": implementer, "control": bool(control), "endpoint": endpoint_info(backend, routing),
                               "red_team": red_team, "context": _context_records(context)},
                       engine_factory=engine_factory)
        except BaseException:
            ledger.close()
            raise

    @classmethod
    def resume(cls, run_dir: Path | str, task: TaskSpec, *, backend: str, routing: Optional[Mapping[str, Any]] = None,
               accept_broken_chain: bool = False, implementer: Optional[str] = None,
               control: Optional[bool] = None, clock: Clock = time.time, overrides: Optional[dict] = None,
               crash_after_runs: Optional[int] = None, reason: str = "resume",
               engine_factory: Optional[Callable[..., Any]] = None, background: str = "",
               context: Sequence[Mapping[str, Any]] = ()) -> "Orchestrator":
        """Reopen an unfinished run. Limits come from ``run.started``; the work (reconcile, continue) happens in ``run()``.

        Refuses (``ResumeRefused`` / ``LedgerCorrupt`` / ``LedgerLocked``) a finished run, a ledger with no ``run.started``,
        another backend or task, a different kind of model endpoint than the run started with, a broken hash chain (unless
        ``accept_broken_chain``) and, without exception, a task whose
        ``task_hash`` or ``grader_hash`` differs from ``run.started``: grades from before and after such a change are not
        comparable, so the run cannot honestly continue.

        ``background`` / ``context`` hand the operator's background material to the roles again (the ledger holds only the
        files' name, sha256 and length, never the text). They are accepted only when ``context`` is the same set of
        ``(name, sha256)`` pairs that ``run.started.config.context`` recorded, in any order; otherwise ``ResumeRefused``, so a
        resumed run never works from different background than it was started with. ``run.resumed.context_reloaded`` is
        true only then. Without them the roles continue without the background and ``context_reloaded`` is false.
        """
        sandbox.harden_parent()
        secrets.register_env(*KEY_ENVS)
        if not (Path(run_dir) / LEDGER_NAME).is_file():       # opening a ledger creates it: a mistyped path must not leave one behind
            raise ResumeRefused(f"{sanitize(str(run_dir), 200)} holds no ledger; there is nothing to resume")
        try:
            ledger = Ledger.open(run_dir, clock=clock)
        except PermissionError:
            # a finished run's ledger is chmod 0444: say so instead of failing to open it for writing
            peek = Ledger.read_only(run_dir)
            if peek.view().finished is not None:
                raise ResumeRefused(f"run {peek.run_id} already finished; a finished run cannot be resumed") from None
            raise
        try:
            ok, bad = ledger.verify_chain()
            if not ok and not accept_broken_chain:
                raise LedgerCorrupt(f"the hash chain is broken at event {bad}; refusing to resume "
                                    "(accept_broken_chain overrides this at your own risk)", int(bad or 0))
            view = ledger.view()
            started = view.run
            if not started:
                raise ResumeRefused(f"{run_dir} has no run.started event; there is nothing to resume")
            if view.finished is not None:
                raise ResumeRefused(f"run {ledger.run_id} already finished ({sanitize(str(view.finished.get('reason')), 40)}); "
                                    "a finished run cannot be resumed")
            if started.get("backend") not in (None, backend):
                raise ResumeRefused(f"the run was started with backend {sanitize(str(started.get('backend')), 20)}, "
                                    f"not {sanitize(str(backend), 20)}")
            if started.get("task") not in (None, task.name):
                raise ResumeRefused(f"the run was started for task {sanitize(str(started.get('task')), 60)}, not {sanitize(task.name, 60)}")
            if any(started.get(name) and started.get(name) != now
                   for name, now in (("task_hash", _task_hash(task)), ("grader_hash", _trusted_hash(task)))):
                raise ResumeRefused("the task or grader changed since this run started; start a new run")
            limits = BudgetLimits.from_dict(started.get("limits") if isinstance(started.get("limits"), dict) else {})
            saved = started.get("config") if isinstance(started.get("config"), dict) else {}
            impl = implementer if implementer else str(saved.get("implementer") or "patch")
            ctrl = bool(control) if control is not None else bool(saved.get("control"))
            red_team = _saved_red_team(saved.get("red_team"))
            plant = [r["behavior"] for r in red_team]
            try:
                mapped = [sid for _, sid in cls._red_team_pairs(backend, task, plant)]
            except ValueError as exc:
                raise ResumeRefused(f"the planted behaviors of this run cannot be served any more: {sanitize(str(exc), 200)}") from exc
            if red_team and [r["scripted_id"] for r in red_team] != mapped:
                raise ResumeRefused("the task's scripted bank no longer maps the planted behaviors to the proposals this run "
                                    "injected; start a new run")
            saved_context = _saved_context(saved.get("context"))
            if background or context:
                if backend != "meta":
                    raise ResumeRefused("background material needs the meta backend: the scripted backend has no language model "
                                        "to read it")
                missing = _context_mismatch(saved_context, _context_records(context))
                if missing or not saved_context:
                    raise ResumeRefused("the background files do not match the ones this run started with"
                                        + (f" ({missing})" if missing else " (it started without any)"))
            else:
                background = ""
            routing = cls._routing_for(backend, routing)
            endpoint = endpoint_info(backend, routing)
            was = saved.get("endpoint")
            was_kind = was.get("kind") if isinstance(was, dict) else None
            if isinstance(was_kind, str) and was_kind != endpoint["kind"]:
                # run.started.config.endpoint is what labels the whole run (simulation, live, custom host): one run, one kind
                raise ResumeRefused(f"the run started against a {sanitize(was_kind, 20)} endpoint and would now continue against a "
                                    f"{endpoint['kind']} one; start a new run")
            gate = BudgetGate(limits, ledger, clock=clock)
            suite = cls._build_suite(backend, task, run_dir, routing, gate, ledger, impl, ctrl, plant, background)
            auditor = Auditor(ledger, task, judge=suite.judge, final_judge=suite.final_judge, clock=clock)
            return cls(run_dir, task, suite, limits, ledger=ledger, gate=gate, auditor=auditor, backend=backend, clock=clock,
                       overrides=overrides, crash_after_runs=crash_after_runs,
                       config={"implementer": impl, "control": ctrl, "endpoint": endpoint, "red_team": red_team,
                               "context": saved_context, "context_reloaded": bool(background)},
                       engine_factory=engine_factory, resume_reason=sanitize(reason, 80) or "resume")
        except BaseException:
            ledger.close()
            raise

    @staticmethod
    def _effective_implementer(backend: str, implementer: str) -> str:
        """The implementer the run really used, as ``run.started.config.implementer`` records it. The scripted suite has its own
        implementer and ignores the requested one, so a scripted run says ``scripted``, never ``muse_code``."""
        return "scripted" if backend == "scripted" else implementer

    @staticmethod
    def _routing_for(backend: str, routing: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        """The routing the suite will use: the caller's, else the documented defaults for ``meta``, else None."""
        if routing is None and backend == "meta":
            from .models.config import load_routing
            return load_routing(None)
        return routing

    @staticmethod
    def _build_suite(backend: str, task: TaskSpec, run_dir: Path | str, routing: Optional[Mapping[str, Any]], gate: BudgetGate,
                     ledger: Ledger, implementer: str, control: bool, plant: Sequence[str] = (), background: str = "") -> AgentSuite:
        from .agents.factory import build_suite
        extra: dict[str, Any] = {}
        if plant:
            extra["plant"] = list(plant)
        if background:
            extra["background"] = background
        return build_suite(backend, task, run_dir, routing, gate, ledger, implementer=implementer, control=control, **extra)

    @staticmethod
    def _red_team_pairs(backend: str, task: TaskSpec, plant: Sequence[str]) -> list[tuple[str, str]]:
        """``[(behavior, scripted_id)]`` for ``plant`` (``ValueError`` for the scripted backend or a name the bank cannot serve)."""
        if not plant:
            return []
        if backend != "meta":
            raise ValueError("--plant needs the meta backend: the scripted backend already plants its own faults")
        from .agents.scripted import planted_proposals
        return planted_proposals(task, plant)

    @classmethod
    def _red_team_for(cls, backend: str, task: TaskSpec, plant: Sequence[str]) -> list[dict]:
        """The ``run.started.config.red_team`` records: ``[{"behavior", "scripted_id"}]``."""
        return [{"behavior": b, "scripted_id": sid} for b, sid in cls._red_team_pairs(backend, task, plant)]

    # ---- the run ---------------------------------------------------------------------------------
    def run(self) -> dict:
        """Run (or continue) to a finish and return ``{run_id, reason, verdict_class, incumbent, counts, head_anchor}``.

        ``head_anchor`` is the ledger head after ``run.finished`` (what ``nightshift verify --expect-head`` takes); the
        ``run.finished`` event itself carries the anchor of the head just before it, as an event cannot contain its own hash.
        """
        try:
            return self._run()
        except SimulatedCrash:
            self._release()
            raise

    def _run(self) -> dict:
        if self._finished:
            raise ResumeRefused("this run has already finished")
        try:
            self._begin()
            if not self._finished:
                self.suite.supervisor.run(self)
        except (SimulatedCrash, ResumeRefused):
            raise
        except KeyboardInterrupt:
            self._conclude("halt", "the run was interrupted by the operator")
        except HaltRequested:
            self._conclude("halt", "the run was halted")
        except BudgetExceeded as exc:
            self._conclude("budget_exhausted", f"the {_ident(exc.kind, 'budget')} budget was exhausted")
        except ModelError as exc:
            self._conclude_model_error(exc)
        except NightshiftError as exc:
            if any(c.__name__ in ("BaselineFailed", "BaselineIncomplete") for c in type(exc).__mro__):
                self._conclude("error", "the baseline could not be graded, so no comparison is possible", defects=True)
            else:
                LOG.error("run failed: %s", type(exc).__name__)
                self._conclude("error", f"the run stopped on an error ({type(exc).__name__})")
        except Exception as exc:  # noqa: BLE001 - an unexpected supervisor/engine failure ends the run, it does not hide it
            LOG.error("supervisor failed: %s", type(exc).__name__)
            self._conclude("error", f"the supervisor failed unexpectedly ({type(exc).__name__})")
        else:
            if not self._finished:
                self._conclude_after_return()
        return self._result()

    def _begin(self) -> None:
        if self._resume_reason is not None:
            self._begin_resume()
            return
        if self.ledger.view().run:
            raise ResumeRefused("this ledger already holds a run; use Orchestrator.resume to continue it")
        self.ledger.append(Kind.RUN_STARTED, "engine", self._run_started_payload())
        self.engine.run_baseline()
        self.gate.charge_wall()

    def _begin_resume(self) -> None:
        known = {f.finding_id for f in self.ledger.view().open_findings()}
        reconciled = [_ident(e) for e in self.engine.reconcile()]
        payload: dict[str, Any] = {"reason": self._resume_reason, "reconciled": reconciled,
                                   "task_hash": _task_hash(self.task), "grader_hash": _trusted_hash(self.task),
                                   "endpoint": self.config["endpoint"], "parallel": self.config["parallel"]}
        if "parallel_reason" in self.config:
            payload["parallel_reason"] = self.config["parallel_reason"]
        if self.config["context"]:
            # only name, hash and length were recorded: the text is not in the ledger, so it is true only when the operator
            # gave the same files again (``Orchestrator.resume(background=..., context=...)``, checked by sha256)
            payload["context_reloaded"] = bool(self.config.get("context_reloaded"))
        self.ledger.append(Kind.RUN_RESUMED, "engine", payload)
        self.engine.resume_pending()
        view = self.ledger.view()
        if not any(e.lane == "baseline" for e in view.experiments.values()):
            self.engine.run_baseline()
            view = self.ledger.view()
        if view.baseline is None or view.baseline.status != ExpStatus.RECORDED:
            raise BaselineIncomplete("the baseline has no recorded result")
        self.gate.charge_wall()
        self._audit_and_nudge(known)

    def _run_started_payload(self) -> dict:
        t = self.task
        forced = self.overrides.get("isolation")
        # the mode the runs will really get: a forced mode is only a request, lowered to what the probe proved
        isolation = sandbox.effective_isolation(forced if isinstance(forced, str) and forced else "auto")
        optional = ("parallel_reason", "red_team", "context")      # recorded only when there is something to say
        return {"task": t.name, "backend": self.backend, "models": dict(self.suite.models), "limits": self.limits.to_dict(),
                "task_hash": _task_hash(t), "grader_hash": _trusted_hash(t), "objective": t.objective,
                "primary_metric": t.primary_metric, "train_metric": t.train_metric, "isolation": isolation,
                "config": {k: v for k, v in {
                    "implementer": self.config["implementer"], "control": bool(self.config["control"]),
                    "parallel": self.config["parallel"], "parallel_reason": self.config.get("parallel_reason"),
                    "endpoint": self.config["endpoint"], "red_team": [dict(r) for r in self.config["red_team"]],
                    "context": [dict(c) for c in self.config["context"]]}.items() if k not in optional or v}}

    # ---- finishing -------------------------------------------------------------------------------
    def _conclude_after_return(self) -> None:
        """The supervisor returned without an accepted finish: classify by why it stopped."""
        why = self.should_stop() or ""
        if why == "halt":
            self._conclude("halt", "the run was halted (HALT file)")
        elif why.startswith("budget"):
            self._conclude("budget_exhausted", f"stopped: {why}")
        else:
            self._conclude("error", "the supervisor ended without finishing the run")

    def _conclude_model_error(self, exc: ModelError) -> None:
        """The supervisor's model API call failed for good. Only the numeric HTTP status goes into the ledger (an error message
        may quote the request). 401, 402 and 403 (bad key, no credits, no access) mean the run lacks authority, like a budget
        stop; any other failure is a defect of the run."""
        status = exc.status if isinstance(exc.status, int) and not isinstance(exc.status, bool) else None
        LOG.error("supervisor stopped: the model API call failed (status %s)", status)
        if status is None:
            self._conclude("error", "stopped: the model API call failed; check the network, the API key and credits")
            return
        self._conclude("error", f"stopped: the model API rejected the request (HTTP {status}); check the API key and credits",
                       authority=status in AUTH_STATUSES)

    def _conclude(self, reason: str, summary: str, *, defects: bool = False, authority: bool = False) -> None:
        """Write ``run.finished`` for a run that did not end in an accepted finish.

        The verdict is ``missing_authority`` for a halt, a spent budget or (``authority=True``) a refused key, unless the
        ledger also holds a defect: an open block finding, an experiment still in flight or one left ``unresolved`` for any
        other reason than the stop itself (the engine's ``stopped by HALT`` / ``stopped by the budget`` records are the stop,
        not a defect). Any other reason, and ``defects=True``, is ``unresolved_defects``.
        """
        with self._lock:
            if self._finished:
                return
            if reason == "budget_exhausted":
                self._record_exhaustion()
            view = self.ledger.view()
            stopped = self._stop_unresolved_ids()
            unresolved = (defects or (reason == "error" and not authority) or bool(view.open_findings("block"))
                          or bool(view.in_flight())
                          or any(e.dev_outcome == Outcome.UNRESOLVED and e.exp_id not in stopped
                                 for e in view.experiments.values()))
            verdict = "unresolved_defects" if unresolved else "missing_authority"
            self._write_finished(reason, summary, verdict, [])

    def _stop_unresolved_ids(self) -> set[str]:
        """Experiments whose lane-phase record is the engine's own stop note: they were cut short, not found defective."""
        return {str(ev.data.get("exp_id")) for ev in self.ledger.events()
                if ev.kind == Kind.EXP_RECORDED and ev.data.get("phase", "lane") == "lane"
                and ev.data.get("outcome") == Outcome.UNRESOLVED
                and str(ev.data.get("note") or "").startswith(STOPPED_NOTES)}

    def _record_exhaustion(self) -> None:
        """Make the ledger say which budget ended the run (``budget.exhausted`` is written once per kind by the gate)."""
        stop = self.should_stop() or ""
        kind = stop.split(": ", 1)[1] if stop.startswith("budget: ") else ""
        if kind:
            try:
                self.gate.check(kind)
            except BudgetExceeded:
                pass

    def _write_finished(self, reason: str, summary: str, verdict: str, final_claims: Sequence[str]) -> None:
        with self._lock:
            if self._finished:
                return
            self.gate.charge_wall()
            seq, short = self.ledger.head_anchor()
            self.ledger.append(Kind.RUN_FINISHED, "engine", {
                "reason": reason, "summary": sanitize(summary, SUMMARY_CHARS), "verdict_class": verdict,
                "final_claims": list(final_claims), "head_anchor": f"{seq}:{short}"})
            self._finished = True
            self.ledger.seal()

    def _release(self) -> None:
        """Close the ledger writer (flock released). Idempotent."""
        try:
            self.ledger.close()
        except Exception:  # noqa: BLE001 - releasing must not mask the crash being propagated
            pass

    def _result(self) -> dict:
        view = self.ledger.view()
        fin = view.finished or {}
        inc = view.incumbent
        grade = inc.primary_grade if inc is not None else None
        seq, short = self.ledger.head_anchor()
        lane_exps = [e for e in view.experiments.values() if e.exp_id != view.baseline_id and e.lane != "baseline"
                     and not e.replicate_of]
        return {
            "run_id": self.ledger.run_id, "reason": fin.get("reason"), "verdict_class": fin.get("verdict_class"),
            "incumbent": {"exp_id": inc.exp_id if inc else None, "score": _num(grade.score) if grade else None,
                          "replicated": bool(inc.replicated) if inc else False},
            "counts": {"experiments": len(lane_exps), "replications": int(view.budget_used.get("replications", 0) or 0),
                       "findings_open": len(view.open_findings())},
            "head_anchor": f"{seq}:{short}"}

    # ---- SupervisorEnv: reading ------------------------------------------------------------------
    def tool_specs(self) -> list[ToolSpec]:
        """Exactly the six supervisor tools, each with a strict schema (``additionalProperties: false`` everywhere)."""
        return [ToolSpec(name=name, description=desc, parameters=copy.deepcopy(schema))
                for name, (desc, schema) in self._schemas.items()]

    def should_stop(self) -> Optional[str]:
        """None to continue, else ``finished``, ``halt`` or ``budget: <supervisor_turn|wall|llm>``. Cheap: one stat and one fold."""
        if self._finished:
            return "finished"
        try:
            if (self.run_dir / "HALT").exists():
                return "halt"
        except OSError:
            return "halt"
        exhausted = set(self.gate.snapshot()["exhausted"])
        for key, name in (("supervisor_turns", "supervisor_turn"), ("wall_sec", "wall"), ("llm_usd", "llm")):
            if key in exhausted:
                return f"budget: {name}"
        return None

    def state(self) -> dict:
        """The compact state of 7.9: dev numbers only; ``replicated`` is the only sealed-derived value.

        ``recent`` lists the last ``RECENT`` lane experiments with their dev ``score`` and, if it is older than those, the
        experiment whose replication was recorded last (so a supervisor can always report what it replicated).
        ``awaiting_replication`` lists only candidates that still beat the current incumbent on dev: a replication of a
        superseded one would waste one of the few sealed queries (``gates.dev_superseded``). ``stalled`` ignores hypotheses
        that were never tested (``gates.stall_streak``)."""
        view = self.ledger.view()
        base, inc = view.baseline, view.incumbent

        def dev_score(exp: Optional[ExperimentView]) -> Optional[float]:
            grade = exp.primary_grade if exp is not None else None
            return _num(grade.score) if grade is not None else None

        lane_ids = [i for i in view.order if i != view.baseline_id and view.experiments[i].lane != "baseline"
                    and not view.experiments[i].replicate_of]
        shown = lane_ids[-RECENT:]
        last_replicated = self._last_replicated(view, lane_ids)
        if last_replicated is not None and last_replicated not in shown:
            shown = [last_replicated, *shown]       # a supervisor must be able to report the replication it ran, however old
        recent = [self._recent_entry(view.experiments[i]) for i in shown]
        findings = sorted(view.open_findings(), key=lambda f: f.created_seq)[:20]
        pivots: list[str] = []
        if view.reflections:
            pivots = [fence(sanitize(p, 200), "pivot", max_chars=200) for p in _strings(view.reflections[-1].get("pivots"), 5)]
        stall_after = max(1, int(self.limits.stall_after))
        remaining = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.gate.remaining().items()}
        return {
            "budget": remaining,
            "baseline": {"exp_id": base.exp_id if base else None, "score": dev_score(base)},
            "incumbent": {"exp_id": inc.exp_id if inc else None, "score": dev_score(inc),
                          "replicated": bool(inc.replicated) if inc else False},
            "recent": recent,
            "awaiting_replication": [_ident(e.exp_id) for e in view.experiments.values()
                                     if e.awaiting_replication and not self._gates().dev_superseded(self.task, view, e.exp_id)],
            "open_findings": [self._finding_entry(f) for f in findings],
            "pivots": pivots,
            "in_flight": [_ident(e.exp_id) for e in view.in_flight()],
            "stalled": self._gates().stall_streak(view) >= stall_after,
            "turns_left": max(0, int(self.limits.max_supervisor_turns) - view.supervisor_turns),
        }

    def _last_replicated(self, view: Any, lane_ids: Sequence[str]) -> Optional[str]:
        """The lane experiment whose replication was recorded most recently (a replication that hit an environment error did
        not replicate anything and does not count), else None."""
        wanted = set(lane_ids)
        for ev in reversed(self.ledger.events()):
            if (ev.kind == Kind.EXP_RECORDED and ev.data.get("phase") == "replication"
                    and ev.data.get("outcome") != Outcome.INFRA and ev.data.get("exp_id") in wanted):
                return str(ev.data["exp_id"])
        return None

    @staticmethod
    def _recent_entry(exp: ExperimentView) -> dict:
        # dev_outcome, not outcome: a replication rewrites ``outcome`` with the sealed split's class. score: the dev grade only
        outcome = exp.dev_outcome or ("blocked" if exp.status == ExpStatus.BLOCKED else exp.status)
        eid = _ident(exp.exp_id)
        grade = exp.primary_grade
        return {"exp_id": eid, "lane": _label(exp.lane, 0) if exp.lane else "",
                "hypothesis": fence(sanitize(exp.hypothesis, 200), f"hyp_{eid}", max_chars=200),
                "outcome": sanitize(str(outcome), 24), "delta": _num(exp.delta),
                "score": _num(grade.score) if grade is not None and grade.valid else None, "replicated": bool(exp.replicated)}

    @staticmethod
    def _finding_entry(f: Any) -> dict:
        rationale = sanitize(f.rationale, 300)
        if f.source != "deterministic":      # an LLM judge wrote it from fenced experiment text: quote it as data
            rationale = fence(rationale, "finding", max_chars=300)
        return {"finding_id": _ident(f.finding_id, "?"), "invariant_id": _ident(f.invariant_id, "?"),
                "severity": sanitize(str(f.severity), 12), "subject": _ident(f.subject, "?"),
                "exp_id": _ident(f.exp_id) if f.exp_id else None, "rationale": rationale}

    # ---- SupervisorEnv: acting -------------------------------------------------------------------
    def _require_open(self) -> None:
        if self._finished:
            raise RunAlreadyFinished("run already finished")

    def launch(self, template: str, lanes: Sequence[LaneSpec], stages: Optional[Sequence[str]] = None) -> list[LaneResult]:
        """Run one batch of lanes through the engine and audit it. Results are in input order, dev numbers only.

        Raises ``InvalidComposition`` (after writing ``experiment.blocked``) for an unknown template or an invalid stage
        list. Lane labels are cleaned and made unique, hypotheses are capped at 600 characters, a lane with a bad ``base`` or
        beyond the 8th is returned blocked (and recorded as such) without reaching the engine.

        In a run started with red-team lanes, the scripted attacker lanes not yet injected are added after the supervisor's own
        lanes (as many as fit under the lane cap; ``RT-<behavior>`` labels, ``notes`` start with ``RED_TEAM_NOTE``) and their
        results are appended to the returned list.
        """
        self._require_open()
        tpl = self._templates()
        specs, local, truncated = self._prepare_lanes(lanes)
        names: Any = list(stages) if isinstance(stages, (list, tuple)) else stages
        errors: list[str] = []
        if not isinstance(template, str) or template not in tpl.TEMPLATES:
            errors.append(f"unknown template {ascii(str(template)[:24])}; known templates: {', '.join(tpl.TEMPLATES)}")
        else:
            if names is None:
                names = [s.name for s in tpl.TEMPLATES[template]]
            try:
                tpl.compose(names)
            except InvalidComposition as exc:
                errors = list(exc.errors)
        if errors:
            for spec in specs:
                self._write_block("invalid_composition", spec.lane, spec.hypothesis, "; ".join(errors)[:400])
            raise InvalidComposition(errors)
        known = {f.finding_id for f in self.ledger.view().open_findings()}
        injected = self._red_team_lanes(specs, local)
        self._announce_red_team(injected)
        run_specs = specs + injected
        engine_results = list(self.engine.run_lanes(list(names), run_specs)) if run_specs else []
        self.gate.charge_wall()
        if run_specs:
            self._audit_and_nudge(known)
        aligned = [self._clean_result(r) for r in self._align(engine_results, run_specs)]
        cleaned, planted = aligned[:len(specs)], aligned[len(specs):]
        for spec, result in zip(specs, cleaned):
            if spec.lane in truncated:
                result.notes = sanitize("; ".join(n for n in (result.notes, TRUNCATED_NOTE) if n), 400)
        for result in planted:
            result.notes = sanitize("; ".join(n for n in (RED_TEAM_NOTE, result.notes) if n), 400)
        return [local[i] if i in local else cleaned.pop(0) for i in range(len(local) + len(specs))] + planted

    def _red_team_done(self) -> set[str]:
        """Scripted ids of the configured red-team lanes that the ledger already holds: proposed (``experiment.proposed``), or
        blocked before a proposal existed (``experiment.blocked`` on an ``RT-`` lane with the bank's hypothesis)."""
        wanted = {r["scripted_id"] for r in self.config["red_team"]}
        done: set[str] = set()
        blocked: dict[str, str] = {}
        if wanted:
            from .agents.scripted import bank_proposal
            blocked = {sanitize(bank_proposal(self.task, sid).hypothesis, HYPOTHESIS_CHARS): sid for sid in wanted}
        for ev in self.ledger.events():
            if ev.kind == Kind.EXP_PROPOSED and ev.data.get("scripted_id") in wanted:
                done.add(ev.data["scripted_id"])
            elif ev.kind == Kind.EXP_BLOCKED and str(ev.data.get("lane") or "").startswith(RED_TEAM_PREFIX):
                sid = blocked.get(str(ev.data.get("hypothesis") or ""))
                if sid is not None:
                    done.add(sid)
        return done

    def _announce_red_team(self, specs: Sequence[LaneSpec]) -> None:
        """Say in the ledger (a ``supervisor.nudge``: labels the operator's bank chose, no model text) that these lanes are scripted
        attackers added by the operator, so that nobody reading the timeline or the supervisor's messages takes them for the
        supervisor's own hypotheses."""
        if specs:
            self.nudge("Operator notice: lane(s) " + ", ".join(s.lane for s in specs) + " in this launch are scripted red-team "
                       "attackers added by the operator to exercise the audit; they are not hypotheses of the supervisor.")

    def _flush_red_team(self) -> None:
        """Run the red-team lanes that no launch has carried yet, before a finish request is considered: after a resume, or when
        the supervisor's launches filled every lane slot, the supervisor may never launch again, and a planted fault that never
        ran would silently drop out of the run. They run as one operator launch of the default workflow; what they find is
        audited like any lane's result (new open findings reach the supervisor as the usual nudge and the finish gate)."""
        if self._finished or not self.config["red_team"]:
            return
        specs = self._red_team_lanes([], {})
        if not specs:
            return
        self._announce_red_team(specs)
        stages = [s.name for s in self._templates().TEMPLATES["explore"]]
        known = {f.finding_id for f in self.ledger.view().open_findings()}
        self.engine.run_lanes(stages, specs)
        self.gate.charge_wall()
        self._audit_and_nudge(known)

    def _red_team_lanes(self, specs: Sequence[LaneSpec], local: Mapping[int, LaneResult]) -> list[LaneSpec]:
        """The scripted attacker lanes to add to this launch: the configured ones that were not injected yet, as many as fit
        under ``MAX_LANES`` next to the supervisor's own lanes (which are never reduced). The rest wait for a later launch."""
        if not self.config["red_team"]:
            return []
        from .agents.scripted import bank_proposal
        with self._lock:
            done = self._red_team_done() | self._red_team_claimed
            room = MAX_LANES - len(specs)
            used = {s.lane for s in specs} | {r.lane for r in local.values()}
            out: list[LaneSpec] = []
            for record in self.config["red_team"]:
                sid = record["scripted_id"]
                if sid in done or len(out) >= room:
                    continue
                prop = bank_proposal(self.task, sid)
                label = _unique((RED_TEAM_PREFIX + record["behavior"])[:16], used)
                out.append(LaneSpec(lane=label, hypothesis=sanitize(prop.hypothesis, HYPOTHESIS_CHARS),
                                    approach=sanitize(prop.approach, APPROACH_CHARS), base="incumbent", replicate_of=None,
                                    scripted_id=sid))
                self._red_team_claimed.add(sid)
            return out

    def _prepare_lanes(self, lanes: Sequence[LaneSpec]) -> tuple[list[LaneSpec], dict[int, LaneResult], set[str]]:
        """Validated copies for the engine, blocked results (by input position) for lanes that cannot run, and the labels of
        the lanes whose hypothesis was cut to ``HYPOTHESIS_CHARS`` (their results say so in ``notes``)."""
        if not isinstance(lanes, (list, tuple)):
            raise TypeError("lanes must be a list of LaneSpec")
        used: set[str] = set()
        specs: list[LaneSpec] = []
        local: dict[int, LaneResult] = {}
        truncated: set[str] = set()
        for i, lane in enumerate(lanes):
            if not isinstance(lane, LaneSpec):
                raise TypeError("lanes must be a list of LaneSpec")
            label = _unique(_label(lane.lane, i), used)
            full = sanitize(lane.hypothesis if isinstance(lane.hypothesis, str) else "", 100_000)
            hypothesis = sanitize(full, HYPOTHESIS_CHARS)
            base = lane.base if isinstance(lane.base, str) and re.fullmatch(_BASE, lane.base) else None
            problem = ("too many lanes in one launch (at most %d)" % MAX_LANES if len(specs) >= MAX_LANES
                       else None if base else "the lane base must be 'incumbent', 'template' or an experiment id such as e003")
            if problem:
                self._write_block("invalid_composition", label, hypothesis, problem)
                local[i] = LaneResult(exp_id="", lane=label, status=ExpStatus.BLOCKED, outcome=None,
                                      blocked_reason="invalid_composition", notes=problem)
                continue
            scripted = lane.scripted_id if isinstance(lane.scripted_id, str) and _SCRIPTED_ID.fullmatch(lane.scripted_id) else None
            specs.append(LaneSpec(lane=label, hypothesis=hypothesis, approach=sanitize(lane.approach or "", APPROACH_CHARS),
                                  base=base, replicate_of=None, scripted_id=scripted))
            if len(full) > HYPOTHESIS_CHARS:
                truncated.add(label)
        return specs, local, truncated

    @staticmethod
    def _align(results: Sequence[Any], specs: Sequence[LaneSpec]) -> list[Any]:
        """One engine result per spec, in spec order, even if the engine returned a different number of them."""
        if len(results) == len(specs):
            return list(results)
        by_label = {getattr(r, "lane", None): r for r in results}
        return [by_label.get(s.lane) or LaneResult(exp_id="", lane=s.lane, status=ExpStatus.PROPOSED,
                                                   notes="the engine returned no result for this lane") for s in specs]

    @staticmethod
    def _clean_result(raw: Any) -> LaneResult:
        """A launch result with every free-text field sanitised and the numbers limited to the dev split's finite floats."""
        status = getattr(raw, "status", "")
        reason = getattr(raw, "blocked_reason", None)
        return LaneResult(
            exp_id=_ident(getattr(raw, "exp_id", ""), ""), lane=_label(getattr(raw, "lane", ""), 0) if getattr(raw, "lane", "") else "",
            status=sanitize(str(status), 24), outcome=sanitize(str(raw.outcome), 24) if getattr(raw, "outcome", None) else None,
            delta=_num(getattr(raw, "delta", None)), score=_num(getattr(raw, "score", None)),
            replicated=bool(getattr(raw, "replicated", False)),
            review_verdict=sanitize(str(raw.review_verdict), 16) if getattr(raw, "review_verdict", None) else None,
            blocked_reason=sanitize(str(reason), 24) if reason else None, notes=sanitize(str(getattr(raw, "notes", "") or ""), 400))

    def _write_block(self, reason: str, lane: str, hypothesis: str, detail: str) -> None:
        with self._lock:
            self.ledger.append(Kind.EXP_BLOCKED, "engine", {
                "exp_id": None, "lane": lane, "hypothesis": sanitize(hypothesis, HYPOTHESIS_CHARS),
                "hyp_fingerprint": self.memory.fingerprint(hypothesis) if hypothesis else "", "reason": reason,
                "detail": sanitize(detail, 400), "dup_of": None})

    def _audit_and_nudge(self, known: set[str]) -> None:
        """Run an audit window; if findings appeared that were not open before, tell the supervisor (ids and titles only).

        If the window's LLM budget runs out inside the audit the results already computed (and the experiments already in the
        ledger) are kept: the exhaustion is on record and the next ``should_stop`` reports it.
        """
        try:
            self.auditor.audit_window()
        except BudgetExceeded as exc:
            LOG.warning("audit window cut short: the %s budget was exhausted", _ident(exc.kind, "budget"))
            self._deterministic_audit(final=False)
        new = [f for f in self.ledger.view().open_findings() if f.finding_id not in known]
        if not new:
            return
        try:
            from .audit.invariants import BY_ID
        except ImportError:  # pragma: no cover
            BY_ID = {}
        parts = []
        for severity in ("block", "warn", "info"):
            items = []
            for f in sorted((f for f in new if f.severity == severity), key=lambda f: f.created_seq):
                title = getattr(BY_ID.get(f.invariant_id), "title", "")
                items.append(f"{_ident(f.finding_id, '<id>')} ({title})" if title else _ident(f.finding_id, "<id>"))
            if items:
                parts.append(f"Open {severity} findings: {', '.join(items[:10])}")
        if parts:
            self.nudge(". ".join(parts) + ". Use get_state to inspect them, fix the cause, then acknowledge_finding.")

    def _deterministic_audit(self, *, final: bool) -> None:
        """Record what the deterministic checks find after the real auditor's LLM judge ran the budget out.

        ``Auditor`` computes every finding of a pass before it writes any of them, so a ``BudgetExceeded`` from its judge
        loses the deterministic findings too (a protected-path violation, a dev/sealed disagreement) and leaves no
        ``audit.window``: the supervisor would be told nothing and a run that ends right after would not count the open
        block finding. A second ``Auditor`` without a judge writes exactly those findings and a window that says the LLM
        questions did not run. Only the real ``Auditor`` is handled; any other auditor is left alone.
        """
        aud = self.auditor
        if not isinstance(aud, Auditor):
            return
        fallback = Auditor(self.ledger, self.task, judge=None, final_judge=None, config=aud.config, clock=self._clock)
        with getattr(aud, "_lock", threading.RLock()):        # never interleave with another pass of the real auditor
            fallback.final_sweep() if final else fallback.audit_window()

    def replicate(self, exp_id: str) -> LaneResult:
        """Sealed re-grade through the engine, then an audit window. The result carries no score and no delta, only the
        boolean ``replicated``."""
        return self._replicate(exp_id)[0]

    def _replicate(self, exp_id: Any) -> tuple[LaneResult, Optional[bool]]:
        self._require_open()
        if not isinstance(exp_id, str) or not _EXP_ID.fullmatch(exp_id) or exp_id not in self.ledger.view().experiments:
            detail = "the replication target must be an existing experiment id such as e003"
            self._write_block("not_replicable", "", "", detail)
            return LaneResult(exp_id="", lane="", status=ExpStatus.BLOCKED, outcome="blocked", blocked_reason="not_replicable",
                              notes=detail), None
        known = {f.finding_id for f in self.ledger.view().open_findings()}
        raw = self.engine.replicate(exp_id)
        self.gate.charge_wall()
        self._audit_and_nudge(known)       # a failed replication is exactly when the dev/sealed findings (I02, I09) appear
        exp = self.ledger.view().experiments[exp_id]
        reason = getattr(raw, "blocked_reason", None)
        blocked = bool(reason) or getattr(raw, "status", "") == ExpStatus.BLOCKED
        if blocked:
            outcome, flag = "blocked", (True if exp.replicated else None)
        elif exp.replicated or getattr(raw, "replicated", False):
            outcome, flag = "replicated", True
        elif (exp.sealed_outcome not in (None, Outcome.INFRA)) or getattr(raw, "outcome", None) in (
                Outcome.IMPROVED, Outcome.NO_GAIN, Outcome.WORSE):
            outcome, flag = "not_replicated", False       # the sealed class itself is never reported, only this boolean
        elif exp.sealed_outcome == Outcome.INFRA or getattr(raw, "outcome", None) == Outcome.INFRA:
            outcome, flag = "infra_error", None
        else:
            outcome, flag = "unknown", None
        note = sanitize(str(getattr(raw, "notes", "") or ""), 200)
        if _has_digit(note):                              # replication notes are engine text; a number could be a sealed one
            note = "(details withheld: replication notes never carry numbers)"
        reason = reason if reason in BLOCK_REASONS else ("not_replicable" if blocked else None)
        result = LaneResult(exp_id=_ident(exp_id, ""), lane=_label(exp.lane, 0) if exp.lane else "",
                            status=ExpStatus.BLOCKED if blocked else exp.status, outcome=outcome, delta=None, score=None,
                            replicated=bool(flag), review_verdict=None, blocked_reason=reason, notes=note)
        return result, flag

    def reflect(self) -> dict:
        """Run the reflector; lessons and pivots are model text, so they come back sanitised and fenced."""
        self._require_open()
        out = self.engine.reflect("manual")
        out = out if isinstance(out, Mapping) else {}
        return {"lessons": [fence(sanitize(t, 400), "lesson", max_chars=400) for t in _strings(out.get("lessons"), 10)],
                "pivots": [fence(sanitize(t, 200), "pivot", max_chars=200) for t in _strings(out.get("pivots"), 5)]}

    def acknowledge(self, finding_id: str, action: str, reason: str) -> dict:
        """Ask the auditor to close a finding. It validates, checks containment and writes ``audit.ack`` itself."""
        self._require_open()
        raw = self.auditor.acknowledge(finding_id, action, sanitize(reason, REASON_CHARS) if isinstance(reason, str) else "",
                                       by="supervisor")
        raw = raw if isinstance(raw, Mapping) else {}
        out: dict[str, Any] = {}
        for key in ("ok", "error", "need", "finding_id", "action", "containment", "already_closed", "min_chars", "detail"):
            value = raw.get(key)
            if isinstance(value, bool) or isinstance(value, int):
                out[key] = value
            elif isinstance(value, str):
                out[key] = sanitize(value, 300)
        if isinstance(raw.get("allowed"), list):
            out["allowed"] = [sanitize(str(a), 20) for a in raw["allowed"][:4]]
        if any(k in out for k in ("containment", "need", "detail")) and self._replication_tested(
                finding_id, out.get("containment"), out.get("need"), out.get("detail")):
            # the auditor words these from the experiment's final outcome, which after a replication is the sealed split's class
            out.pop("containment", None)
            for key in ("need", "detail"):
                if key in out:
                    out[key] = "(details withheld: the experiment went through replication)"
        if not out:
            out = {"error": "the auditor gave no answer"}
        return out

    def _replication_tested(self, finding_id: Any, *texts: Any) -> bool:
        """True if the finding, or an experiment named in ``texts``, is about an experiment with a recorded replication."""
        view = self.ledger.view()
        found = view.findings.get(finding_id) if isinstance(finding_id, str) else None
        names = [t for t in texts if isinstance(t, str)] + ([found.subject, found.exp_id] if found is not None else [])
        if isinstance(finding_id, str):
            names.append(finding_id)
        ids = {m for n in names if isinstance(n, str) for m in re.findall(r"\be\d{3,6}\b", n)}
        return any(view.experiments[i].sealed_outcome is not None for i in ids if i in view.experiments)

    def claim(self, claim_type: str, text: str, exp_id: Optional[str] = None, cites: Sequence[int] = (),
              alias: Optional[str] = None) -> str:
        """Write an ``agent.claim`` as the supervisor and return its id (``run.c<n>`` or ``<exp_id>.c<n>``). Never evidence."""
        self._require_open()
        if claim_type not in CLAIM_TYPES:
            raise ValueError(f"claim_type must be one of {', '.join(CLAIM_TYPES)}")
        if exp_id is not None and (not isinstance(exp_id, str) or not _EXP_ID.fullmatch(exp_id)):
            raise ValueError("exp_id must look like e003")
        cite_list = [int(c) for c in list(cites or ())[:20] if isinstance(c, int) and not isinstance(c, bool) and c >= 0]
        with self._lock:
            view = self.ledger.view()
            if exp_id is not None and exp_id not in view.experiments:
                raise ValueError("unknown experiment")
            prefix = exp_id or "run"
            pattern = re.compile(rf"^{re.escape(prefix)}\.c(\d+)$")
            n = max((int(m.group(1)) for cid in view.claims if (m := pattern.match(cid))), default=0) + 1
            claim_id = f"{prefix}.c{n}"
            self.ledger.append(Kind.CLAIM, "supervisor", {
                "claim_id": claim_id, "exp_id": exp_id, "claim_type": claim_type, "text": sanitize(text, SUMMARY_CHARS),
                "cites": cite_list, "alias": _SLUG_BAD.sub("_", alias)[:40] if isinstance(alias, str) and alias else None})
            return claim_id

    def finish(self, summary: str, claim_ids: Sequence[str] = ()) -> dict:
        """Request to end the run: final audit sweep, then the finish gate. Returns ``{"accepted": bool, "reasons": [...]}``.

        Refusal writes ``supervisor.finish_refused``. Acceptance writes ``run.finished`` and seals the ledger before returning.
        The verdict is ``success`` only for a replicated incumbent that beats the baseline on the sealed split, else
        ``inconclusive``; the cited claims are what the finish gate checks, they do not decide the verdict.
        """
        self._flush_red_team()
        with self._lock:
            if self._finished:
                return {"accepted": False, "reasons": ["run already finished"]}
            text = sanitize(summary if isinstance(summary, str) else "", SUMMARY_CHARS) or "(no summary given)"
            ids = [sanitize(c, 60) for c in claim_ids[:MAX_CLAIM_IDS] if isinstance(c, str)] \
                if isinstance(claim_ids, (list, tuple)) else []
            self.ledger.append(Kind.SUP_FINISH_REQUESTED, "supervisor", {"summary": text, "claim_ids": ids})
            try:
                self.auditor.final_sweep()
            except BudgetExceeded:
                # no finish without the final audit: the run ends as budget_exhausted, but with the deterministic findings on record
                self._deterministic_audit(final=True)
                raise
            view = self.ledger.view()
            reasons = self._gates().finish_gate(view, self.task, claim_ids, limits=self.limits)
            if reasons:
                shown = [self._neutral_reason(r) for r in reasons[:10]]
                self.ledger.append(Kind.SUP_FINISH_REFUSED, "engine", {"reasons": shown})
                return {"accepted": False, "reasons": shown}
            self._write_finished("supervisor_finish", text, self._accepted_verdict(view), ids)
            return {"accepted": True, "reasons": []}

    @staticmethod
    def _neutral_reason(reason: Any) -> str:
        """A finish refusal reason as the supervisor may read it. The gate's wording for a replicated incumbent that does not
        beat the baseline says what the sealed comparison with the baseline came out as, one bit beyond ``replicated``;
        that bit is not for a model, so such a reason says only that the claim is not confirmed."""
        text = sanitize(str(reason), 300)
        if "sealed" in text.lower():
            m = re.match(r"(improved claim [A-Za-z0-9_.@:-]{1,60}):", text)
            return f"{m.group(1) if m else 'improved claim'}: not confirmed as an improvement over the baseline"
        return text

    def _accepted_verdict(self, view: Any) -> str:
        """``success`` iff the incumbent is not the baseline, is replicated and beats the baseline on the sealed split (as
        ``sealed_decision`` computes it from the ledger); every other accepted finish is ``inconclusive``.

        What the supervisor claimed or cited decides nothing here: an LLM supervisor has no tool that returns claim ids, so the
        verdict has to come from the evidence the gates already verified.
        """
        inc_id, base_id = view.incumbent_id, view.baseline_id
        inc = view.experiments.get(inc_id) if isinstance(inc_id, str) else None
        if inc is None or base_id is None or inc_id == base_id or not inc.replicated:
            return "inconclusive"
        decision = self._gates().sealed_decision(self.task, view, inc_id, base_id)
        return "success" if decision is not None and decision.outcome == Outcome.IMPROVED else "inconclusive"

    def note_turn(self, model: str, summary: str, *, say: Optional[str] = None, tool_calls: Optional[Sequence[dict]] = None,
                  reasoning_effort: Optional[str] = None, latency_ms: Optional[float] = None, text: Optional[str] = None,
                  source: str = "supervisor") -> None:
        """Write ``supervisor.turn`` (turn number = turns so far + 1). The turn cap is enforced through ``should_stop``."""
        self._require_open()
        calls = [{"name": sanitize(str(c.get("name", "")), 40), "args_summary": sanitize(str(c.get("args_summary", "")), 200)}
                 for c in list(tool_calls or []) if isinstance(c, Mapping)][:16]
        with self._lock:
            self.ledger.append(Kind.SUP_TURN, "supervisor", {
                "turn": self.ledger.view().supervisor_turns + 1, "model": sanitize(str(model), 80),
                "action_summary": sanitize(str(summary), 300), "say": sanitize(say, 1000) if isinstance(say, str) and say else None,
                "tool_calls": calls, "reasoning_effort": sanitize(reasoning_effort, 12) if isinstance(reasoning_effort, str) else None,
                "latency_ms": _num(latency_ms, 1), "text": sanitize(text, SUMMARY_CHARS) if isinstance(text, str) and text else None,
                "source": source if source in ("supervisor", "script") else "supervisor"})
            self.gate.charge_wall()      # keep the recorded wall time close to real time between launches (a crash loses less)

    def nudge(self, text: str) -> None:
        """Write ``supervisor.nudge``. Callers pass templates built from ids and static titles only; it is sanitised and capped."""
        self._require_open()
        with self._lock:
            self.ledger.append(Kind.SUP_NUDGE, "engine", {"text": sanitize(str(text), 500)})

    # ---- the tools -------------------------------------------------------------------------------
    def call_tool(self, name: str, arguments: dict) -> dict:
        """Dispatch one tool call. Bad input comes back as ``{"error": ...}``; only ``NightshiftError`` propagates."""
        if self._finished:
            return {"error": "run already finished"}
        if not isinstance(name, str) or name not in self._handlers:
            return {"error": "unknown_tool", "valid_tools": sorted(self._handlers)}
        args = {} if arguments is None else arguments
        problems = validate_arguments(args, self._schemas[name][1])
        if problems:
            return {"error": "invalid_arguments", "detail": "; ".join(problems[:5])}
        try:
            result = self._handlers[name](args)
        except InvalidComposition as exc:
            return {"error": "invalid_composition", "details": [sanitize(e, 200) for e in exc.errors[:5]]}
        except RunAlreadyFinished:
            return {"error": "run already finished"}
        except NightshiftError:
            raise
        except Exception as exc:  # noqa: BLE001 - a tool bug must not leave the call unanswered
            LOG.error("tool %s failed: %s", name, type(exc).__name__)
            return {"error": f"tool_failed: {type(exc).__name__}"}
        return _drop_sealed_keys(result)

    def _tool_launch(self, args: dict) -> dict:
        lanes, used, renames = [], set(), {}
        for i, item in enumerate(args["lanes"]):
            if not item["hypothesis"].strip():
                return {"error": "invalid_arguments", "detail": f"lanes[{i}].hypothesis must not be blank"}
            label = _unique(item["lane"], used)
            if label != item["lane"]:
                renames[label] = item["lane"]
            lanes.append(LaneSpec(lane=label, hypothesis=item["hypothesis"], approach=item.get("approach", ""),
                                  base=item.get("base") or "incumbent"))
        results = self.launch(args.get("template", "explore"), lanes, args.get("stages"))
        body: dict[str, Any] = {"results": [self._lane_dict(r) for r in results]}
        if renames:
            body["lane_renames"] = renames
        return body

    @staticmethod
    def _lane_dict(r: LaneResult) -> dict:
        d = r.to_dict()
        d["notes"] = fence(sanitize(d.get("notes") or "", 400), "notes", max_chars=400) if d.get("notes") else ""
        return d

    def _tool_replicate(self, args: dict) -> dict:
        result, flag = self._replicate(args["exp_id"])
        note = result.notes
        if result.blocked_reason:
            note = f"blocked ({result.blocked_reason}). {note}".strip()
        return {"exp_id": result.exp_id or None, "replicated": flag, "outcome": result.outcome, "note": note}

    def _tool_acknowledge(self, args: dict) -> dict:
        return self.acknowledge(args["finding_id"], args["action"], args["reason"])

    def _tool_finish(self, args: dict) -> dict:
        return self.finish(args["summary"], args.get("claim_ids") or [])


def _drop_sealed_keys(obj: Any) -> Any:
    """Defence in depth: no key containing ``sealed`` survives into a tool result."""
    if isinstance(obj, dict):
        return {k: _drop_sealed_keys(v) for k, v in obj.items() if not (isinstance(k, str) and "sealed" in k.lower())}
    if isinstance(obj, (list, tuple)):
        return [_drop_sealed_keys(v) for v in obj]
    return obj
