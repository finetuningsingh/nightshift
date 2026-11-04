"""The offline backend: a scripted supervisor, designer, implementer, reviewer and reflector that replay a task's
proposal bank (``tasks/<name>/scripted/manifest.json``, or ``manifest.control.json`` for the negative control).

This is a FAULT-INJECTION HARNESS that proves the audit machinery. It never pretends to be a language model:

* every narration it writes has ``source="script"`` and model ``"scripted"``; every identity string contains ``scripted``;
  a ``say`` states what the script is doing and why, never an outcome. Results appear only in the generated
  "Observed:" sentence, built from ``env.state()`` (dev outcomes and dev deltas; for a replication only the boolean);
* outcome numbers are never scripted: the engine really runs each proposal's files, only the faults are planted;
* it reads no grade except the boolean ``replicated`` and never reads ``planted_fault`` (the engine records it, using
  ``planted_fault_for`` below); the reviewer reads the diff text, the grade-summary flags and nothing else.

Guarantees
----------
* A malformed bank (unknown ``scripted_id``, missing file, unsafe path, repeated launch, duplicate defect marker, ...)
  raises ``ScriptedBankError`` (a ``ValueError``) when the suite is built, never half-way through a run.
* The supervisor's cursor lives in the ledger: exactly one ``note_turn`` per executed step, ``scripted_id -> exp_id``
  is rebuilt from ``experiment.proposed`` events, and a crash between a step's ``note_turn`` and its action re-runs the
  action only if its effect is not already in the ledger. Without a ledger (unit tests) the cursor is kept in memory.
* The implementer copies bank files into the workspace without enforcing the editable globs (tamper proposals must be
  able to write protected files: the engine and the sandbox are what catch them) but never writes outside the workspace.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from ..fence import sanitize
from ..types import (AgentSuite, DesignContext, ExperimentPlan, ImplementContext, ImplementResult, LaneSpec,
                     NightshiftError, ReflectContext, ReflectionResult, ReviewContext, ReviewVerdict,
                     SupervisorEnv, TaskSpec)

MANIFEST = "manifest.json"
CONTROL_MANIFEST = "manifest.control.json"
MODEL = "scripted"
SOURCE = "script"
ACTIONS = ("launch", "replicate", "acknowledge", "finish")
BEHAVIORS = ("normal", "tamper", "claim_without_run", "duplicate", "infra_flake", "hang", "overfit", "injection")
CLAIM_TYPES = ("improved", "no_effect", "fixed", "done", "safe", "retraction", "other")
MAX_REPAIR_ROUNDS = 2          # the reviewer asks for a repair only while repair_round is below this
STOP_PREFIX = "stop:"          # summary prefix of the turn written when the script stops early; not a step
SAY_CHARS = 1000               # the orchestrator's cap on one supervisor.turn say
MIN_OBSERVED_CHARS = 120       # the generated sentence is never shortened below this, even next to a very long scripted say
_MAX_FILE_BYTES = 1 << 20
_MAX_MANIFEST_BYTES = 2 << 20
_PID = re.compile(r"[A-Za-z0-9_]{1,40}")      # the engine records only ids of up to 40 characters
_LANE = re.compile(r"[A-Za-z0-9_-]{1,16}")
_TEMPLATE = re.compile(r"[a-z_]{1,32}")
_REL_PATH = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_FINDING_ID = re.compile(r"[A-Za-z0-9_.:@/-]{1,128}")
_EXP_ID = re.compile(r"e\d{3,}")


class ScriptedBankError(NightshiftError, ValueError):
    """The proposal bank (or a step of its script) is malformed. Raised at construction time."""


# --------------------------------------------------------------------------------------------------
# the bank
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Proposal:
    """One entry of ``manifest['proposals']``. ``files`` and ``repair_files`` map a workspace-relative destination to
    the absolute (already confined to ``scripted/``) source file."""

    pid: str
    hypothesis: str
    approach: str
    behavior: str
    planted: bool
    files: Mapping[str, Path]
    claim_type: str
    claim_text: str
    marker: Optional[str] = None
    repair_files: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Bank:
    """A validated manifest. Immutable by convention; shared by all five scripted roles."""

    control: bool
    steps: tuple[dict, ...]
    proposals: Mapping[str, Proposal]
    pivots: tuple[str, ...] = ()

    @property
    def markers(self) -> dict[str, str]:
        """defect marker -> proposal id, for every proposal that has a defect block."""
        return {p.marker: pid for pid, p in self.proposals.items() if p.marker}


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _signed(value: float) -> str:
    """``+0.728``; a delta that rounds to zero is ``+0.000``, never ``-0.000`` (which reads as a loss)."""
    text = f"{value:+.3f}"
    return "+0.000" if text == "-0.000" else text


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ScriptedBankError(message)


def _no_duplicate_keys(pairs: list) -> dict:
    """``json`` keeps the last of two equal keys; in a proposal bank that would silently drop a proposal or a file."""
    out: dict = {}
    for key, value in pairs:
        _need(key not in out, f"duplicate key {key!r} in the manifest")
        out[key] = value
    return out


def _text(value: Any, where: str, *, limit: int = 4000) -> str:
    _need(isinstance(value, str) and value.strip() != "", f"{where} must be a non-empty string")
    _need(len(value) <= limit, f"{where} is longer than {limit} characters")
    return value


def _rel(value: Any, where: str) -> str:
    _need(isinstance(value, str) and len(value) <= 128 and _REL_PATH.fullmatch(value) is not None
          and all(part not in (".", "..") for part in value.split("/")), f"{where} is not a safe relative path: {value!r}")
    return value


def _source(sdir: Path, value: Any, where: str) -> Path:
    rel = _rel(value, where)
    path = (sdir / rel).resolve()
    try:
        path.relative_to(sdir.resolve())
    except ValueError:
        raise ScriptedBankError(f"{where} escapes the scripted directory: {rel}") from None
    _need(path.is_file(), f"{where}: file {rel} does not exist in the bank")
    _need(path.stat().st_size <= _MAX_FILE_BYTES, f"{where}: file {rel} is larger than {_MAX_FILE_BYTES} bytes")
    return path


def _files(sdir: Path, value: Any, where: str, *, allow_empty: bool = False) -> dict[str, Path]:
    _need(isinstance(value, dict) and (value or allow_empty), f"{where} must be a non-empty object of dest -> source")
    out = {_rel(dest, f"{where} destination"): _source(sdir, src, f"{where}[{dest!r}]") for dest, src in value.items()}
    _need(not any(a != b and b.startswith(a + "/") for a in out for b in out),
          f"{where}: a destination is both a file and a directory of another destination")
    return out


def _proposal(sdir: Path, pid: Any, raw: Any) -> Proposal:
    _need(isinstance(pid, str) and _PID.fullmatch(pid) is not None, f"proposal id {pid!r} is not [A-Za-z0-9_]{{1,40}}")
    where = f"proposal {pid}"
    _need(isinstance(raw, dict), f"{where} must be an object")
    behavior = raw.get("behavior")
    _need(behavior in BEHAVIORS, f"{where}: behavior {behavior!r} is not one of {', '.join(BEHAVIORS)}")
    planted = raw.get("planted")
    _need(isinstance(planted, bool), f"{where}: planted must be a boolean")
    claim = raw.get("claim")
    _need(isinstance(claim, dict) and claim.get("type") in CLAIM_TYPES, f"{where}: claim.type must be one of {CLAIM_TYPES}")
    defect = raw.get("defect")
    marker, repair = None, {}
    if defect is not None:
        _need(isinstance(defect, dict), f"{where}: defect must be an object")
        marker = _text(defect.get("marker"), f"{where}: defect.marker", limit=200)
        repair = _files(sdir, defect.get("repair_files"), f"{where}: defect.repair_files")
    _need(planted or (behavior == "normal" and marker is None),
          f"{where}: a faulty proposal (behavior {behavior!r}, defect {marker is not None}) must have planted=true")
    return Proposal(pid=pid, hypothesis=_text(raw.get("hypothesis"), f"{where}: hypothesis", limit=1000),
                    approach=_text(raw.get("approach"), f"{where}: approach", limit=1000), behavior=behavior,
                    planted=planted, files=_files(sdir, raw.get("files"), f"{where}: files"), claim_type=claim["type"],
                    claim_text=_text(claim.get("text"), f"{where}: claim.text", limit=1000), marker=marker,
                    repair_files=repair)


def _steps(raw_steps: Any, proposals: Mapping[str, Proposal]) -> tuple[dict, ...]:
    _need(isinstance(raw_steps, list) and raw_steps, "steps must be a non-empty list")
    launched: set[str] = set()
    out: list[dict] = []

    def known(pid: Any, where: str) -> str:
        _need(isinstance(pid, str) and pid in proposals, f"{where}: scripted_id {pid!r} is not in the bank")
        _need(pid in launched, f"{where}: {pid!r} is referenced before any step launches it")
        return pid

    for i, raw in enumerate(raw_steps, 1):
        where = f"step {i}"
        _need(isinstance(raw, dict) and raw.get("action") in ACTIONS, f"{where}: action must be one of {ACTIONS}")
        step = dict(raw)
        action = step["action"]
        if isinstance(step.get("say"), str) and not step["say"].strip():
            step["say"] = None
        if step.get("say") is not None:
            _text(step["say"], f"{where}: say", limit=1000)
        if action == "launch":
            _need(_TEMPLATE.fullmatch(str(step.setdefault("template", "explore"))) is not None, f"{where}: bad template")
            stages = step.get("stages")
            _need(stages is None or (isinstance(stages, list) and all(isinstance(s, str) and s for s in stages)),
                  f"{where}: stages must be a list of stage names")
            lanes = step.get("lanes")
            _need(isinstance(lanes, list) and lanes, f"{where}: lanes must be a non-empty list")
            labels: set[str] = set()
            for lane in lanes:
                _need(isinstance(lane, dict) and isinstance(lane.get("lane"), str)
                      and _LANE.fullmatch(lane["lane"]) is not None, f"{where}: every lane needs a short label")
                _need(lane["lane"].lower() != "baseline", f"{where}: the lane label 'baseline' is reserved for the baseline")
                _need(lane["lane"] not in labels, f"{where}: lane label {lane['lane']!r} is used twice")
                labels.add(lane["lane"])
                pid = lane.get("scripted_id")
                _need(isinstance(pid, str) and pid in proposals, f"{where}: scripted_id {pid!r} is not in the bank")
                _need(pid not in launched, f"{where}: {pid!r} is launched more than once")
                launched.add(pid)
        elif action == "replicate":
            known(step.get("of"), f"{where} (replicate.of)")
        elif action == "acknowledge":
            known(step.get("finding_for"), f"{where} (acknowledge.finding_for)")
            _need(step.get("how") in ("addressed", "dismissed"), f"{where}: how must be 'addressed' or 'dismissed'")
            _text(step.get("reason"), f"{where}: reason", limit=500)
        else:
            _text(step.get("summary"), f"{where}: summary", limit=2000)
            claims = step.setdefault("claims", [])
            _need(isinstance(claims, list), f"{where}: claims must be a list of scripted ids")
            for alias in claims:
                known(alias, f"{where} (claims)")
            _need(step.setdefault("expect", "accepted") in ("accepted", "refused"),
                  f"{where}: expect must be 'accepted' or 'refused'")
        out.append(step)
    return tuple(out)


def _check_marker_placement(proposals: Mapping[str, Proposal]) -> None:
    """A defect marker must be in its own proposal's files (else the planted defect can never be caught) and in no repair
    file and no other proposal's file (else an honest proposal, or a repair, would be sent back to repair)."""
    def text(path: Path) -> str:
        return path.read_bytes().decode("utf-8", errors="replace")

    for pid, prop in proposals.items():
        if prop.marker is None:
            continue
        _need(any(prop.marker in text(src) for src in prop.files.values()),
              f"proposal {pid}: the defect marker {prop.marker!r} does not occur in any of its files")
        for other_id, other in proposals.items():
            for dest, src in {**(other.files if other_id != pid else {}), **other.repair_files}.items():
                _need(prop.marker not in text(src),
                      f"proposal {pid}: the defect marker {prop.marker!r} also occurs in {other_id}'s file {dest}")


def load_bank(task: TaskSpec, *, control: bool = False) -> Bank:
    """Parse and validate ``<task>/scripted/manifest[.control].json``. Raises ``ScriptedBankError`` (a ``ValueError``)."""
    _need(task.scripted_dir is not None and Path(task.scripted_dir).is_dir(),
          f"task {task.name!r} has no scripted/ proposal bank")
    sdir = Path(task.scripted_dir)
    path = sdir / (CONTROL_MANIFEST if control else MANIFEST)
    try:
        size = path.stat().st_size
        text = path.read_text(encoding="utf-8") if size <= _MAX_MANIFEST_BYTES else ""
    except OSError as exc:
        raise ScriptedBankError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ScriptedBankError(f"{path} is not valid text: {exc}") from exc
    _need(size <= _MAX_MANIFEST_BYTES, f"{path.name} is larger than {_MAX_MANIFEST_BYTES} bytes")
    try:
        raw = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, RecursionError) as exc:       # ScriptedBankError is a ValueError and passes through unchanged
        if isinstance(exc, ScriptedBankError):
            raise
        raise ScriptedBankError(f"{path} is not valid JSON: {type(exc).__name__}: {str(exc)[:120]}") from exc
    _need(isinstance(raw, dict) and raw.get("version") == 1, f"{path.name}: version must be 1")
    raw_props = raw.get("proposals")
    _need(isinstance(raw_props, dict) and raw_props, f"{path.name}: proposals must be a non-empty object")
    proposals = {pid: _proposal(sdir, pid, p) for pid, p in raw_props.items()}
    markers = [p.marker for p in proposals.values() if p.marker]
    _need(all(sum(1 for other in markers if m in other) == 1 for m in markers),
          "defect markers must be unique (no marker may equal or contain another)")
    _check_marker_placement(proposals)
    pivots: tuple[str, ...] = ()
    reflection = raw.get("reflection")
    if reflection is not None:
        raw_pivots = reflection.get("pivots") if isinstance(reflection, dict) else None
        _need(isinstance(raw_pivots, list) and all(isinstance(p, str) and p.strip() for p in raw_pivots),
              "reflection.pivots must be a list of non-empty strings")
        pivots = tuple(sanitize(p, 200) for p in raw_pivots[:8])
    return Bank(control=control, steps=_steps(raw.get("steps"), proposals), proposals=proposals, pivots=pivots)


_BANK_CACHE: dict[tuple[str, bool, int, int], Bank] = {}
_BANK_LOCK = threading.Lock()


def _cached_bank(task: TaskSpec, control: bool) -> Bank:
    """``load_bank`` memoised on (path, control, mtime, size) so the engine can ask once per lane cheaply."""
    path = Path(task.scripted_dir or ".") / (CONTROL_MANIFEST if control else MANIFEST)
    try:
        st = path.stat()
        key = (str(path.resolve()), control, st.st_mtime_ns, st.st_size)
    except OSError:
        return load_bank(task, control=control)       # raises a precise ScriptedBankError
    with _BANK_LOCK:
        bank = _BANK_CACHE.get(key)
    if bank is None:
        bank = load_bank(task, control=control)
        with _BANK_LOCK:
            _BANK_CACHE[key] = bank
    return bank


def planted_fault_for(task: TaskSpec, scripted_id: Optional[str], control: bool = False) -> Optional[str]:
    """The fault name the engine writes into ``experiment.proposed.planted_fault`` for a scripted lane.

    The proposal's ``behavior`` when it is planted and not ``normal``; ``"defect"`` for a ``normal`` proposal with a
    defect block (a reviewer-caught bug); else ``None``. ``None`` is also returned for no id, an id the bank does not
    know, or a task without a bank: the value is display metadata and must never break a lane.
    """
    if scripted_id is None or task.scripted_dir is None:
        return None
    prop = _cached_bank(task, control).proposals.get(scripted_id)
    if prop is None:
        return None
    if prop.planted and prop.behavior != "normal":
        return prop.behavior
    if prop.marker is not None and prop.behavior == "normal":
        return "defect"
    return None


def bank_proposal(task: TaskSpec, scripted_id: str, *, control: bool = False) -> Proposal:
    """The bank's proposal ``scripted_id`` (its hypothesis and approach become a red-team lane). ``ScriptedBankError`` if the
    task has no usable bank, ``KeyError`` if the bank has no such proposal."""
    return _cached_bank(task, control).proposals[scripted_id]


# behaviors an operator may plant into a meta run: the faulty ones (``normal`` is not a fault)
PLANTABLE = tuple(b for b in BEHAVIORS if b != "normal")


def planted_proposals(task: TaskSpec, behaviors: Sequence[str], *, control: bool = False) -> list[tuple[str, str]]:
    """``[(behavior, scripted_id)]`` for ``behaviors``, in the order given: each maps to the FIRST planted proposal of that
    behavior in the task's bank (manifest order). This is how ``nightshift run --plant`` picks its scripted attacker lanes.

    Raises ``ValueError`` (``ScriptedBankError`` when the bank itself is unusable) for an empty list, a name that is not a
    plantable task-bank behavior, a behavior listed twice (its second lane would only duplicate the first) and a behavior
    the task's bank has no planted proposal for. Nothing is returned for a half-valid list: all names are checked first.
    """
    names = list(behaviors)
    if not names:
        raise ValueError("no behavior to plant; choose from " + ", ".join(PLANTABLE))
    for name in names:
        if not isinstance(name, str) or name not in PLANTABLE:
            shown = ascii(name[:24]) if isinstance(name, str) else f"<{type(name).__name__}>"
            raise ValueError(f"unknown behavior {shown} to plant; choose from {', '.join(PLANTABLE)}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"behavior {repeated[0]!r} is listed more than once; each planted lane needs its own behavior")
    proposals = _cached_bank(task, control).proposals
    out: list[tuple[str, str]] = []
    for name in names:
        found = next((pid for pid, prop in proposals.items() if prop.planted and prop.behavior == name), None)
        if found is None:
            have = sorted({p.behavior for p in proposals.values() if p.planted and p.behavior != "normal"})
            raise ValueError(f"task {sanitize(task.name, 60)!r} has no planted {name!r} proposal in its scripted bank "
                             f"(it has: {', '.join(have) or 'none'})")
        out.append((name, found))
    return out


# --------------------------------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------------------------------
class ScriptedSupervisor:
    """Walks the manifest's steps through ``SupervisorEnv``. Not a model: it narrates only the script's own ``say``."""

    identity = "scripted-supervisor"

    def __init__(self, bank: Bank, *, ledger: Any = None):
        self.bank = bank
        self.ledger = ledger                       # a LedgerProtocol, or None: the cursor is then kept in memory
        self.log: list[dict] = []                  # what each step did this process (diagnostics and tests)
        self.stopped: Optional[str] = None         # why the script ended early, if it did
        self._turns = 0                            # in-memory cursor, used only when there is no ledger
        self._exp_ids: dict[str, str] = {}         # in-memory scripted_id -> exp_id, only without a ledger
        self._replicated: set[str] = set()         # in-memory, only without a ledger

    # ---- state derived from the ledger ---------------------------------------------------------
    def _view(self):
        return self.ledger.view() if self.ledger is not None else None

    def _step_turns(self) -> int:
        """Steps already started: supervisor turns in the ledger, not counting the script's own stop turns."""
        view = self._view()
        if view is None:
            return self._turns
        return sum(1 for t in view.turns if not str(t.get("action_summary", "")).startswith(STOP_PREFIX))

    def _turn_seq(self, step_index: int) -> int:
        """Ledger seq of the turn that started step ``step_index`` (0-based); 0 without a ledger."""
        view = self._view()
        if view is None:
            return 0
        turns = [t for t in view.turns if not str(t.get("action_summary", "")).startswith(STOP_PREFIX)]
        return int(turns[step_index]["seq"]) if step_index < len(turns) else 0

    def _experiments(self) -> dict[str, str]:
        view = self._view()
        if view is None:
            return dict(self._exp_ids)
        out: dict[str, str] = {}
        for exp_id in view.order:
            exp = view.experiments[exp_id]
            if exp.scripted_id and not exp.replicate_of and exp.lane != "baseline":
                out.setdefault(exp.scripted_id, exp_id)
        return out

    def _replication_done(self, exp_id: str) -> bool:
        view = self._view()
        if view is None:
            return exp_id in self._replicated
        exp = view.exp(exp_id)
        # only the boolean ``replicated`` and the fact that a sealed grade exists; never the sealed outcome or its numbers
        return exp is not None and (exp.replicated or exp.grade_for("replication") is not None)

    def _finish_settled_since(self, seq: int) -> bool:
        """True when a finish requested after ``seq`` already got its answer (refused, or the run finished). A request
        with no answer means the process died inside the gate: the finish has to be asked again."""
        view = self._view()
        if view is None:
            return False
        asked = [int(r.get("seq", 0)) for r in view.finish_requests if int(r.get("seq", 0)) > seq]
        if not asked:
            return False
        return view.finished is not None or any(int(r.get("seq", 0)) > max(asked) for r in view.finish_refusals)

    def _ack_requested_since(self, seq: int) -> bool:
        """True when the auditor already wrote an ack that this supervisor asked for after ``seq``."""
        view = self._view()
        if view is None:
            return False
        return any(a.get("by") == "supervisor" and int(a.get("seq", 0)) > seq for a in view.acks)

    def _existing_claim(self, alias: str, exp_id: str) -> Optional[str]:
        view = self._view()
        if view is None:
            return None
        for cid, claim in view.claims.items():
            if claim.get("actor") == "supervisor" and claim.get("alias") == alias and claim.get("exp_id") == exp_id:
                return cid
        return None

    # ---- the script ----------------------------------------------------------------------------
    def run(self, env: SupervisorEnv) -> None:
        steps = self.bank.steps
        done = self._step_turns()
        if self.ledger is not None and 0 < done <= len(steps):
            # a crash can fall between a step's note_turn and its action: finish that action if its effect is missing
            if self._stop_if_asked(env) or not self._run_action(env, steps[done - 1], done - 1, redo=True):
                return
        while done < len(steps):
            if self._stop_if_asked(env):
                return
            step = steps[done]
            env.note_turn(MODEL, self._summary(step, done + 1), say=self._say(env, step, steps[done - 1] if done else None),
                          tool_calls=[self._tool_call(step)], source=SOURCE)
            self._turns += 1
            done += 1
            if not self._run_action(env, step, done - 1, redo=False):
                return

    def _stop_if_asked(self, env: SupervisorEnv) -> bool:
        reason = env.should_stop()
        if not reason:
            return False
        self.stopped = sanitize(str(reason), 80)
        env.note_turn(MODEL, f"{STOP_PREFIX} {self.stopped}", source=SOURCE)
        return True

    def _say(self, env: SupervisorEnv, step: dict, previous: Optional[dict]) -> Optional[str]:
        """The turn's narration: the generated observation of the previous launch/replicate step (when there was one),
        then the step's own scripted ``say``. The observation is rebuilt from ``env.state()``, so a resumed process
        writes the same sentence the crashed one would have."""
        observed = self._observed(env, previous) if previous is not None else None
        own = step.get("say")
        if observed and own:
            # ``note_turn`` caps a say at SAY_CHARS: shorten the generated sentence so the scripted intent keeps its end
            room = max(SAY_CHARS - len(own) - 1, MIN_OBSERVED_CHARS)
            if len(observed) > room:
                observed = observed[:room - 1].rstrip() + "\u2026"
            return f"{observed} {own}"
        return observed or own

    def _observed(self, env: SupervisorEnv, step: dict) -> Optional[str]:
        """One factual sentence about what ``step`` (a launch or a replicate) produced, from the compact state only.

        Reports dev outcomes and dev deltas of the lanes just run; for a replication only whether it replicated (the
        state carries no other sealed-derived value). Anything the state does not show is said to be not visible.
        """
        action = step.get("action")
        if action not in ("launch", "replicate"):
            return None
        state = env.state()
        state = state if isinstance(state, dict) else {}
        recent = {e.get("exp_id"): e for e in state.get("recent") or [] if isinstance(e, dict)}
        incumbent = state.get("incumbent") if isinstance(state.get("incumbent"), dict) else {}
        exps = self._experiments()
        if action == "launch":
            parts = []
            for lane in step["lanes"]:
                pid, exp_id = lane["scripted_id"], exps.get(lane["scripted_id"])
                entry = recent.get(exp_id) if exp_id else None
                if entry is None:
                    parts.append(f"{pid} has no result in the state" if exp_id is None else f"{pid} ({exp_id}) is not visible in the state")
                    continue
                outcome = sanitize(str(entry.get("outcome") or "no outcome"), 24)
                delta = _finite(entry.get("delta"))
                parts.append(f"{pid} ({exp_id}) dev outcome {outcome}" + (f", dev delta {_signed(delta)}" if delta is not None else ""))
            base = _finite((state.get("baseline") or {}).get("score")) if isinstance(state.get("baseline"), dict) else None
            tail = f" (baseline dev score {base:.4f})" if base is not None else ""
            return "Observed: " + "; ".join(parts) + tail + "."
        pid = step["of"]
        exp_id = exps.get(pid)
        if exp_id is None:
            return f"Observed: {pid} has no experiment, so there was nothing to replicate."
        entry = recent.get(exp_id)
        if incumbent.get("exp_id") == exp_id:                 # the incumbent entry carries the same boolean
            flag: Optional[bool] = bool(incumbent.get("replicated"))
        elif entry is not None:
            flag = bool(entry.get("replicated"))
        else:
            flag = None
        if flag is None:
            said = f"{pid} ({exp_id}) is not visible in the state"
        elif flag:
            said = f"{pid} ({exp_id}) replicated"
        elif exp_id in (state.get("awaiting_replication") or []):
            said = f"{pid} ({exp_id}) is still awaiting replication"
        else:
            said = f"{pid} ({exp_id}) did not replicate"
        inc_id = sanitize(str(incumbent.get("exp_id") or "none"), 12)
        inc_score = _finite(incumbent.get("score"))
        inc = f"; the incumbent is {inc_id}" + (f" (dev score {inc_score:.4f})" if inc_score is not None else "")
        return f"Observed after the sealed replication: {said}{inc}."

    def _summary(self, step: dict, number: int) -> str:
        action = step["action"]
        if action == "launch":
            what = "launch " + ", ".join(f"{lane['lane']}={lane['scripted_id']}" for lane in step["lanes"])
        elif action == "replicate":
            what = f"replicate {step['of']}"
        elif action == "acknowledge":
            what = f"acknowledge ({step['how']}) the first open finding for {step['finding_for']}"
        else:
            what = "finish citing " + (", ".join(step["claims"]) or "no claims")
        return f"script step {number}/{len(self.bank.steps)}: {what}"

    @staticmethod
    def _tool_call(step: dict) -> dict:
        action = step["action"]
        if action == "launch":
            return {"name": "launch_experiments",
                    "args_summary": f"template={step['template']} lanes=" + ",".join(l["lane"] for l in step["lanes"])}
        if action == "replicate":
            return {"name": "replicate", "args_summary": f"of={step['of']}"}
        if action == "acknowledge":
            return {"name": "acknowledge_finding", "args_summary": f"for={step['finding_for']} action={step['how']}"}
        return {"name": "finish", "args_summary": "claims=" + ",".join(step["claims"])}

    def _run_action(self, env: SupervisorEnv, step: dict, index: int, *, redo: bool) -> bool:
        """Perform one step's action. ``redo`` means its turn was written by an earlier process. False stops the script."""
        action = step["action"]
        if action == "launch":
            self._launch(env, step)
        elif action == "replicate":
            self._replicate(env, step)
        elif action == "acknowledge":
            if redo and self._ack_requested_since(self._turn_seq(index)):
                self.log.append({"action": "acknowledge", "skipped": True})
                return True                      # already done: do not close the finding the NEXT step is meant for
            self._acknowledge(env, step)
        else:
            if redo and self._finish_settled_since(self._turn_seq(index)):
                return True                      # the finish already got its answer before the crash
            return self._finish(env, step, index)
        return True

    def _launch(self, env: SupervisorEnv, step: dict) -> None:
        known = self._experiments()
        lanes = [LaneSpec(lane=lane["lane"], hypothesis=self.bank.proposals[lane["scripted_id"]].hypothesis,
                          approach=self.bank.proposals[lane["scripted_id"]].approach, base="incumbent",
                          scripted_id=lane["scripted_id"])
                 for lane in step["lanes"] if lane["scripted_id"] not in known]
        if not lanes:
            self.log.append({"action": "launch", "launched": []})
            return
        results = env.launch(step["template"], lanes, stages=step.get("stages"))
        if self.ledger is None:                  # without a ledger the launch results are the only record
            by_lane = {lane.lane: lane.scripted_id for lane in lanes}
            for result in results if isinstance(results, (list, tuple)) else ():
                exp_id, label = getattr(result, "exp_id", None), getattr(result, "lane", None)
                if exp_id and label in by_lane:
                    self._exp_ids[by_lane[label]] = str(exp_id)
        self.log.append({"action": "launch", "launched": [lane.scripted_id for lane in lanes]})

    def _replicate(self, env: SupervisorEnv, step: dict) -> None:
        exp_id = self._experiments().get(step["of"])
        if exp_id is None or self._replication_done(exp_id):
            self.log.append({"action": "replicate", "skipped": True})
            return
        env.replicate(exp_id)
        self._replicated.add(exp_id)
        self.log.append({"action": "replicate", "exp_id": exp_id})

    def _acknowledge(self, env: SupervisorEnv, step: dict) -> None:
        exp_id = self._experiments().get(step["finding_for"])
        finding_id = None
        if exp_id is not None:
            for finding in (env.state() or {}).get("open_findings") or []:
                candidate = finding.get("finding_id") if isinstance(finding, dict) else None
                if (isinstance(finding, dict) and finding.get("exp_id") == exp_id and isinstance(candidate, str)
                        and _FINDING_ID.fullmatch(candidate)):
                    finding_id = candidate
                    break
        if finding_id is None:                   # the auditor may already have closed it: that is fine
            self.log.append({"action": "acknowledge", "skipped": True})
            return
        result = env.acknowledge(finding_id, step["how"], step["reason"])
        error = result.get("error") if isinstance(result, dict) else None
        self.log.append({"action": "acknowledge", "finding_id": finding_id, "error": error})

    def _finish(self, env: SupervisorEnv, step: dict, index: int) -> bool:
        exp_ids = self._experiments()
        claim_ids: list[str] = []
        for alias in step["claims"]:
            exp_id = exp_ids.get(alias)
            if exp_id is None:
                continue                         # the lane never became an experiment: there is nothing to claim
            claim_ids.append(self._existing_claim(alias, exp_id)
                             or env.claim("improved", self.bank.proposals[alias].claim_text, exp_id=exp_id, alias=alias))
        result = env.finish(step["summary"], claim_ids)
        accepted = bool(result.get("accepted")) if isinstance(result, dict) else False
        self.log.append({"action": "finish", "accepted": accepted, "expect": step["expect"], "claim_ids": claim_ids})
        got = "accepted" if accepted else "refused"
        if got != step["expect"]:
            self.stopped = f"finish_{got}_unexpectedly"
            try:
                env.nudge(f"Scripted harness mismatch at step {index + 1} of {len(self.bank.steps)}: the finish was "
                          f"{got} but the script expected it to be {step['expect']}. The script stops here.")
            except PermissionError:              # an accepted finish has already sealed the ledger
                pass
            return False
        return not accepted                      # an accepted finish ends the run


# --------------------------------------------------------------------------------------------------
# designer, implementer, reviewer, reflector
# --------------------------------------------------------------------------------------------------
class ScriptedDesigner:
    """Returns the proposal's own plan. ``files_to_touch`` lists every file the proposal writes, honestly."""

    identity = "scripted-designer"

    def __init__(self, bank: Bank):
        self.bank = bank

    def design(self, ctx: DesignContext) -> ExperimentPlan:
        prop = self.bank.proposals.get(ctx.lane.scripted_id or "")
        if prop is None:                         # not a bank lane: plan exactly what the lane asked for
            return ExperimentPlan(hypothesis=ctx.lane.hypothesis, approach=ctx.lane.approach)
        return ExperimentPlan(hypothesis=prop.hypothesis, approach=prop.approach, files_to_touch=list(prop.files))


def _write_file(root: Path, rel: str, source: Path) -> None:
    """Copy ``source`` to ``root/rel`` through a temp file and ``os.replace``; refuses a symlinked directory."""
    dest = root / rel
    current = root
    for part in PurePosixPath(rel).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ScriptedBankError(f"workspace path {rel!r} passes through a symlink")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.parent.resolve().relative_to(root.resolve())
    except ValueError:
        raise ScriptedBankError(f"workspace path {rel!r} resolves outside the workspace") from None
    data = source.read_bytes()
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".scripted-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)                    # replaces a read-only file; never follows a symlink at ``dest``
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class ScriptedImplementer:
    """Applies a proposal's files to the workspace. It is the thing the sandbox and the engine are tested against."""

    identity = "scripted-implementer"

    def __init__(self, bank: Bank):
        self.bank = bank

    def implement(self, ctx: ImplementContext) -> ImplementResult:
        pid = ctx.lane.scripted_id
        prop = self.bank.proposals.get(pid or "")
        if prop is None:
            raise ScriptedBankError(f"lane {ctx.lane.lane!r} has scripted_id {pid!r}, which is not in the bank")
        repairing = bool(ctx.repair_findings)
        files = prop.repair_files if repairing else prop.files
        root = Path(ctx.workspace)
        for rel, source in files.items():
            _write_file(root, rel, source)
        if not repairing and ctx.emit_claim is not None:
            ctx.emit_claim(prop.claim_type, prop.claim_text)
        verb = "repair of" if repairing else "implementation of"
        return ImplementResult(summary=f"scripted {verb} proposal {prop.pid}: wrote {len(files)} file(s)",
                               steps=len(files), worker=self.identity, raw_log="wrote: " + ", ".join(files),
                               self_report=f"scripted: copied {len(files)} file(s) from the proposal bank; "
                                           "no model was involved and no result is claimed here")


class ScriptedReviewer:
    """Rule-based reviewer: asks for a repair iff a bank defect marker occurs in the diff. Reads only the diff text,
    the repair round and the grade-summary flags; never claims, never the implementer's self-report."""

    identity = "scripted-reviewer"

    def __init__(self, bank: Bank):
        self.bank = bank

    def review(self, ctx: ReviewContext) -> ReviewVerdict:
        diff = ctx.diff_text if isinstance(ctx.diff_text, str) else ""
        hits = sorted((marker, pid) for marker, pid in self.bank.markers.items() if marker in diff)
        findings: list[dict] = []
        verdict = "accept"
        if hits:
            marker, pid = hits[0]
            if ctx.repair_round < MAX_REPAIR_ROUNDS:
                verdict = "repair"
                findings.append({"id": "defect_marker", "severity": "block", "evidence": marker,
                                 "text": f"The diff contains the known defect marker of scripted proposal {pid}; "
                                         "it must be repaired and re-run."})
            else:
                findings.append({"id": "defect_marker_unrepaired", "severity": "warn", "evidence": marker,
                                 "text": f"The defect marker of scripted proposal {pid} is still present after "
                                         f"{MAX_REPAIR_ROUNDS} repair rounds."})
        summary = ctx.grade_summary if isinstance(ctx.grade_summary, dict) else {}
        if summary.get("overfit_suspect") is True:
            findings.append({"id": "overfit_suspect", "severity": "warn", "evidence": "grade_summary.overfit_suspect",
                             "text": "The training metric improved without a held-out gain."})
        if summary.get("valid") is False:
            findings.append({"id": "grade_invalid", "severity": "warn", "evidence": "grade_summary.valid",
                             "text": "The grader reported the run's output as invalid."})
        return ReviewVerdict(verdict=verdict, findings=findings, reviewer=self.identity,
                             summary=f"scripted reviewer: {verdict} ({len(findings)} finding(s)); rule-based, not a model")


_HEAD = re.compile(r"(e\d{3,}) \| lane [^|]* \| ([^|]+)(?: \|.*)?")
_FENCE_OPEN = re.compile(r"<<<DATA:([A-Za-z0-9_-]+:[0-9a-f]{10})>>>")
_GENERIC_PIVOTS = (
    "Do not repeat an approach that already ended without a gain; change the model form, not its settings.",
    "Prefer a change aimed at the held-out error over one that only lowers the training error.",
    "Cite an improvement only after it has replicated on the independent re-check.",
)
_MODEL_FAILS = {"failed", "invalid", "unresolved"}


class ScriptedReflector:
    """Deterministic lessons from the tried table, with no numbers and no model-derived text."""

    identity = "scripted-reflector"

    def __init__(self, bank: Bank):
        self.bank = bank

    @staticmethod
    def _rows(table: str) -> list[tuple[str, str, bool]]:
        """(exp_id, status word, overfit_suspect) per entry. Lines inside a fenced block are data, never entries."""
        rows: list[tuple[str, str, bool]] = []
        closing: Optional[str] = None
        for line in str(table or "").splitlines():
            if closing is not None:
                closing = None if line == closing else closing
                continue
            fence = _FENCE_OPEN.fullmatch(line)
            if fence:
                closing = f"<<<END:{fence.group(1)}>>>"
                continue
            head = _HEAD.fullmatch(line)
            if head:
                rows.append((head.group(1), head.group(2).strip(), "overfit suspect" in line))
        return rows

    def reflect(self, ctx: ReflectContext) -> ReflectionResult:
        seen = {str(l.get("text", "")) for l in ctx.lessons if isinstance(l, dict)}
        lessons: list[dict] = []

        def add(text: str, exp_id: str, kind: str = "observation", classification: str = "model") -> None:
            if text not in seen:
                seen.add(text)
                lessons.append({"text": text, "kind": kind, "source_exps": [exp_id], "classification": classification})

        for exp_id, status, overfit in self._rows(ctx.experiments_table)[-12:]:
            word = status.split(" ", 1)[0]
            if status.startswith("no_gain (failed replication)"):
                add(f"{exp_id} improved on dev but did not replicate on the independent re-check; possible overfitting to the "
                    "evaluation it was tuned on.", exp_id)
            elif word in ("no_gain", "worse"):
                add(f"{exp_id} ended {'worse than' if word == 'worse' else 'without a gain over'} the incumbent on the "
                    "dev split; do not repeat the idea without a new angle.", exp_id)
                if overfit:
                    add(f"{exp_id} lowered the training-window error without a held-out gain; treat training-only "
                        "gains as overfitting.", exp_id, "correction")
            elif word in _MODEL_FAILS:
                add(f"{exp_id} did not produce a usable result ({word}); check the approach before retrying.", exp_id)
            elif word == "infra_error":
                add(f"{exp_id} hit an environment failure; its hypothesis was not tested.", exp_id, classification="infra")
            elif status.startswith("improved (awaiting replication)"):
                add(f"{exp_id} improved on dev; replicate it on the independent re-check before citing it.", exp_id, "strategy")
            elif status.startswith("improved"):
                add(f"{exp_id} improved on dev and replicated on the independent re-check.", exp_id)
        pivots = list(self.bank.pivots or _GENERIC_PIVOTS)
        return ReflectionResult(lessons=lessons, pivots=pivots,
                                summary=f"scripted reflection ({sanitize(str(ctx.trigger), 16)}): lessons derived from the "
                                        "tried table; rule-based, not a model")


# --------------------------------------------------------------------------------------------------
# suite
# --------------------------------------------------------------------------------------------------
def build_scripted_suite(task: TaskSpec, run_dir: Path | str, *, control: bool = False, ledger: Any = None) -> AgentSuite:
    """The offline ``AgentSuite``. ``ledger`` makes the supervisor resumable (its cursor and ``scripted_id -> exp_id``
    map are read from it); ``control=True`` replays ``manifest.control.json`` (honest proposals only).

    ``reviewer_hard`` is None: there is no second, different scripted reviewer, and the engine skips that stage with a note.
    """
    bank = load_bank(task, control=control)
    suite = AgentSuite(backend="scripted", supervisor=ScriptedSupervisor(bank, ledger=ledger),
                       designer=ScriptedDesigner(bank), implementer=ScriptedImplementer(bank),
                       reviewer=ScriptedReviewer(bank), reflector=ScriptedReflector(bank), judge=None,
                       reviewer_hard=None, final_judge=None,
                       models={role: MODEL for role in ("supervisor", "designer", "implementer", "reviewer", "reflector")})
    # AgentSuite has no field for it, but the engine asks ``getattr(suite, "control", False)`` to pick the manifest
    # that ``planted_fault_for`` reads, so a control run must say so.
    suite.control = bool(control)  # type: ignore[attr-defined]
    return suite
