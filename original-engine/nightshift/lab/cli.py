"""The lab command line and its composition root (docs/harness-design.md, "Ledger side", ``cli.py``).

    python -m nightshift.lab <lab_root> <command> [options]

``main(argv, *, world=None)`` is the whole program. With a ``LabWorld`` it uses the objects in it and builds nothing (tests pass a world of
fakes from ``tests/support/lab_world.py``); without one it builds the real objects from ``<lab_root>/lab.json`` (``build_world``). There is
no ``--fake`` flag and no fake in production code.

Exit codes: 0 done, 1 refused (a ``LabError``: an order rule, a gate, a guard, a frozen file) or ``verify`` found a problem, 2 usage
error, 3 unexpected failure. Output is one JSON object per command, except ``report`` (Markdown or HTML). During the search ``status`` and
``report`` show bits and costs only; selection rows and ground truth are never printed.

The score guards (``make_score_guard``) are built here from the ledger and injected into ``ScoreService``: the phase of a request is
derived from the ledger, never trusted from the request.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, TextIO

from nightshift.lab import stats
from nightshift.lab.contracts import (
    TEST_ASSIGNMENT_ID,
    AccessDenied,
    AdapterRef,
    ScorerError,
    Arm,
    AssignmentManifest,
    AssignmentRole,
    AssignmentWeight,
    BudgetSpec,
    ComparisonMethod,
    ContractError,
    DecisionStage,
    FailedTrialPolicy,
    FinalAnalysis,
    FinalComparisonRefused,
    FinalRequest,
    FinalReveal,
    FrozenFile,
    Gate,
    GateNotRecorded,
    GateOutcome,
    GateRecord,
    GenerationParams,
    LabConfig,
    LabError,
    LabInit,
    LabKind,
    LabLayout,
    LabState,
    LabWorld,
    LedgerError,
    MIN_FINAL_TRIALS_PER_PACK,
    OperatorChange,
    PackCommitted,
    PackRef,
    Phase,
    PlannedTrial,
    Rule,
    ScoreCost,
    ScoreFailure,
    ScoreGuard,
    ScoreRecord,
    ScoreRequest,
    SearchClosed,
    ServeEndpoint,
    Split,
    SpreadMeasured,
    SpreadPlan,
    TrialRecord,
    TrialScore,
    TrialSpec,
    TrialStatus,
    Verifier,
    check_trial_against_frozen,
    comparability_key,
    file_ref,
    load_frozen,
    loads_strict,
    read_record,
    read_regular_bytes,
    record_sha256,
    sha256_file,
    utc_iso,
    validate_analysis_against_manifest,
    validate_rule_against_manifest,
    write_record,
)

ACTOR = "coordinator"
EXIT_OK, EXIT_REFUSED, EXIT_USAGE, EXIT_FAILED = 0, 1, 2, 3
SCORED = (TrialStatus.COMPLETED, TrialStatus.BUDGET_EXHAUSTED)

COMMANDS = (
    "init", "freeze-manifest", "freeze-budget", "record-gate", "commit-pack", "plan-spread", "trial", "spread", "freeze-rule", "outer",
    "close-search", "freeze-final-analysis", "final", "reveal", "score", "abort-trial", "operator-change", "report", "status", "verify",
    "verify-isolation", "rentals", "anchor",
)


class UsageError(LabError):
    """The command line is wrong (exit code 2)."""


# --------------------------------------------------------------------------------------------------
# the world: what the production composition root hands the commands
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class CliWorld(LabWorld):
    """A ``LabWorld`` plus the shared objects the commands need that the contract container does not name: the lab-wide dollar cap shared by
    every trial meter, the rental manager (``rentals`` command), and a ``close`` hook. A plain ``LabWorld`` works too (the commands then
    build a ``LabCap`` themselves and ``rentals`` is unavailable)."""

    lab_cap: Any = None
    rentals: Any = None
    close: Optional[Callable[[], None]] = None
    challenger: Any = None  # outer_challenge.Challenger, or None (no challenge)


class LazyScorer:
    """A ``Scorer`` built on first use. ``ScoreService`` needs the frozen manifest and the verifiers, which do not exist when the world is
    built (``init`` and ``freeze-manifest`` run before). ``precheck`` is looked up with ``getattr`` by ``score_session``; it is always defined
    here and forwards when the real service has one."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._inner: Any = None
        self._lock = threading.Lock()

    def _get(self) -> Any:
        with self._lock:
            if self._inner is None:
                self._inner = self._factory()
            return self._inner

    @property
    def scorer_version(self) -> str:
        return self._get().scorer_version

    def score(self, request: ScoreRequest, generator: Any, model: str) -> ScoreRecord:
        return self._get().score(request, generator, model)

    def precheck(self, request: ScoreRequest) -> None:
        inner = self._get()
        check = getattr(inner, "precheck", None)
        if check is not None:
            check(request)


class _LazyBackend:
    """A verifier backend built on first use (the pinned third-party checkouts are only needed when something is scored)."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._inner: Any = None
        self._lock = threading.Lock()

    def _get(self) -> Any:
        with self._lock:
            if self._inner is None:
                self._inner = self._factory()
            return self._inner

    def score(self, rows: Any, responses: Any, *, mode: str = "both") -> Any:
        return self._get().score(rows, responses, mode=mode)

    def pins(self) -> dict:
        return self._get().pins()


# --------------------------------------------------------------------------------------------------
# score guards (built from the ledger)
# --------------------------------------------------------------------------------------------------
def make_score_guard(layout: LabLayout, reader: Any) -> ScoreGuard:
    """The ``ScoreGuard`` injected into ``ScoreService``. It is a pure read of the ledger (the service calls it once from ``precheck`` and
    once from ``score``) and derives the phase from the ledger. Search phase: selection scoring of an evolution assignment, once per
    (adapter, assignment), with the seeds, decoding and scorer version of the rule (or of the spread plan before the rule exists), a retry
    only after a recorded failure that revealed no number. Final phase: the ``final_guard`` of the design."""

    def guard(request: ScoreRequest) -> None:
        state: LabState = reader.state()
        closed = state.search_closed is not None
        if request.phase is Phase.SEARCH:
            if closed:
                raise AccessDenied("the search is closed: a search-phase score is refused")
            _search_guard(layout, state, request)
        else:
            if not closed:
                raise AccessDenied("a final-phase score is refused before lab.search_closed")
            _final_guard(layout, state, request)

    return guard


_FROZEN_STATE_ATTR = {
    LabKind.ASSIGNMENTS_FROZEN: "manifest", LabKind.BUDGET_FROZEN: "budget", LabKind.RULE_FROZEN: "rule", LabKind.FINAL_ANALYSIS_FROZEN: "final_analysis",
}


def _key_of(request_adapter_sha: Optional[str]) -> str:
    return request_adapter_sha or "stock"


def _prior_attempts(state: LabState, request: ScoreRequest, phase: Phase) -> tuple:
    """(records, failures) already in the ledger for the same (adapter, split, assignment, phase)."""
    sha = request.adapter.sha256 if request.adapter is not None else None
    records = [r for r in state.scores.values() if r.phase is phase and r.split is request.split and r.assignment_id == request.assignment_id
               and r.adapter_sha256 == sha]
    fails = [f for f in state.score_failures if f.split is request.split and f.assignment_id == request.assignment_id and f.adapter_sha256 == sha
             and (f.trial_id == request.trial_id or request.trial_id is None)]
    return records, fails


def _no_reroll(state: LabState, request: ScoreRequest, phase: Phase) -> None:
    records, fails = _prior_attempts(state, request, phase)
    if records:
        raise AccessDenied(f"a {request.split.value} score of this adapter on {request.assignment_id!r} already exists ({records[0].score_id}); it cannot be re-rolled")
    if any(f.result_revealed for f in fails):
        raise AccessDenied("an earlier attempt revealed a result; no further request is allowed")
    if len(fails) >= 2:
        raise AccessDenied("the one allowed retry after an infrastructure failure has been used")


def _search_guard(layout: LabLayout, state: LabState, request: ScoreRequest) -> None:
    if request.split is not Split.SELECTION:
        return  # dev on an evolution assignment never reaches the guard; nothing else is allowed by the scorer's table
    if state.manifest is None or state.budget is None:
        raise AccessDenied("selection scoring needs the frozen assignments and budget")
    if state.rule is not None:
        rule = load_frozen(layout, state.rule, Rule)
        if request.assignment_id not in rule.assignments:
            raise AccessDenied(f"{request.assignment_id!r} is not an assignment of the rule")
        if (request.scorer_version, request.seeds, request.params) != (rule.scorer_version, rule.score_seeds, rule.score_params):
            raise AccessDenied("the request's scorer version, seeds or decoding differ from the frozen rule")
    else:
        plan = state.spread_plan
        if plan is None:
            raise AccessDenied("selection scoring needs a declared spread plan (or a frozen rule)")
        if request.assignment_id not in plan.assignments or request.scorer_version != plan.scorer_version:
            raise AccessDenied("the request is not covered by the declared spread plan")
    _no_reroll(state, request, Phase.SEARCH)


def _bind_adapter_to_trial(state: LabState, request: ScoreRequest, trial_id: str) -> None:
    """The adapter of a final-phase score must be one this planned trial produced; once the trial names its adapter, or an earlier
    held-out or test attempt of the trial used one, it must be that one. Otherwise a different adapter hash would open a fresh no-reroll key."""
    if request.adapter is None:
        raise AccessDenied("a final-phase score of an arm with a pack needs the trial's adapter")
    sha = request.adapter.sha256
    fin = state.trials.get(trial_id)
    if fin is not None and fin.adapter is not None:
        allowed = {fin.adapter.sha256}
    else:
        allowed = {r.adapter_sha256 for r in state.scores.values() if r.trial_id == trial_id and r.adapter_sha256}
        used = {r.adapter_sha256 for r in state.scores.values() if r.trial_id == trial_id and r.phase is Phase.FINAL and r.split is not Split.DEV}
        used |= {f.adapter_sha256 for f in state.score_failures if f.trial_id == trial_id and f.split is not Split.DEV}
        used.discard(None)
        if used:
            allowed &= used
    if sha not in allowed:
        raise AccessDenied(f"the adapter is not the output of trial {trial_id!r}")


def _final_guard(layout: LabLayout, state: LabState, request: ScoreRequest) -> None:
    if state.search_closed is None or state.final_analysis is None:
        raise AccessDenied("a final-phase score needs lab.search_closed and the frozen final analysis")
    analysis = load_frozen(layout, state.final_analysis, FinalAnalysis)
    fr = next((f for f in state.finals if f.request_id == request.final_request_id), None)
    if fr is None:
        raise FinalComparisonRefused(f"no lab.final_requested {request.final_request_id!r}")
    if fr.arm is not request.arm and not (request.arm is None and request.split is Split.DEV):
        raise AccessDenied("the final request is for another arm")
    if fr.analysis_sha256 != analysis.sha256():
        raise AccessDenied("the final request names another analysis than the frozen one")
    pending = [t for t, s in state.trials_started.items() if s.phase is Phase.SEARCH and t not in state.trials]
    if pending:
        raise AccessDenied(f"search trial(s) {pending[:3]} are still pending")
    if request.split is not Split.DEV and request.split not in fr.splits:
        raise AccessDenied(f"the final request does not unlock the {request.split.value} split")
    if (request.seeds, request.params) != (analysis.score_seeds, analysis.score_params):
        raise AccessDenied("the request's seeds or decoding differ from the frozen final analysis")
    twin = [f for f in state.finals if f.arm is fr.arm and f.pack == fr.pack and f.repeat_of is None]
    if len(twin) > 1:
        raise FinalComparisonRefused("two final requests exist for the same arm and pack commit without repeat_of")
    if fr.arm is not Arm.STOCK:
        spec = state.trials_started.get(request.trial_id) if request.trial_id else None
        if spec is None:
            raise AccessDenied("a final-phase score of an arm with a pack needs its trial")
        if spec.pack != fr.pack or spec.final_request_id != fr.request_id:
            raise AccessDenied("the scored trial's pack or final request differs from the final request")
        try:
            planned = analysis.planned(spec.trial_id)
        except KeyError:
            raise AccessDenied(f"trial {spec.trial_id!r} is not in the frozen final plan") from None
        if planned.arm is not fr.arm or (request.split is not Split.TEST and planned.assignment_id != request.assignment_id):
            raise AccessDenied("the trial's planned arm or assignment differs from the request")
        if request.split is not Split.DEV:
            # LIVE 2026-11-04: a running final trial's job adapter is dev-scored before the trial names any adapter, so binding a DEV score
            # refused every final job. DEV scores are researcher feedback, never a held-out or test result; held-out and test stay bound.
            _bind_adapter_to_trial(state, request, spec.trial_id)
    elif request.split is not Split.TEST:
        raise AccessDenied("arm 1 (stock) is scored on the test split only")
    if request.split is not Split.DEV:
        _no_reroll(state, request, Phase.FINAL)


# --------------------------------------------------------------------------------------------------
# environment of one command
# --------------------------------------------------------------------------------------------------
class _Env:
    def __init__(self, layout: LabLayout, world: Optional[LabWorld], out: TextIO, err: TextIO, command: str):
        self.layout, self._world, self.out, self.err, self.command = layout, world, out, err, command
        self._ledger: Any = None
        self._reader: Any = None
        self._opened_here = False
        self._built: Optional[LabWorld] = None
        self._config: Optional[LabConfig] = None

    # config -----------------------------------------------------------------------------------
    @property
    def config(self) -> Optional[LabConfig]:
        if self._world is not None and self._world.config is not None:
            return self._world.config
        if self._config is None and self.layout.lab_config_path.exists():
            self._config = LabConfig.from_dict(loads_strict(read_regular_bytes(self.layout.root, "lab.json", max_bytes=1 << 20).decode("utf-8")))
        return self._config

    def need_config(self) -> LabConfig:
        cfg = self.config
        if cfg is None:
            raise ContractError(f"{self.layout.lab_config_path} is missing: run init --config FILE first")
        return cfg

    # light objects ----------------------------------------------------------------------------
    @property
    def clock(self) -> Callable[[], float]:
        return self._world.clock if self._world is not None else time.time

    def now(self) -> str:
        return utc_iso(self.clock())

    @property
    def ledger(self) -> Any:
        if self._world is not None:
            return self._world.ledger
        if self._ledger is None:
            from nightshift.lab.ledger import LabLedger, register_secrets

            cfg = self.config
            if cfg is not None:
                try:
                    register_secrets(*cfg.secret_env.values())
                except Exception:  # noqa: BLE001 - registering an unset variable must not stop a read-only command
                    pass
            if self.command == "init" and not self.layout.master_ledger_dir.joinpath("ledger.jsonl").exists():
                self.layout.root.mkdir(parents=True, exist_ok=True)
                self._ledger = LabLedger.create(self.layout.master_ledger_dir, clock=self.clock)
            else:
                if not self.layout.master_ledger_dir.joinpath("ledger.jsonl").exists():
                    raise ContractError(f"{self.layout.root} is not an initialized lab: run init first")
                self._ledger = LabLedger.open(self.layout.master_ledger_dir, clock=self.clock)
            self._opened_here = True
        return self._ledger

    @property
    def reader(self) -> Any:
        if self._world is not None:
            return self._world.state_reader
        if self._reader is None:
            from nightshift.lab.fold import FoldReader

            self._reader = FoldReader(self.layout, ledger=self.ledger)
        return self._reader

    def state(self) -> LabState:
        return self.reader.state()

    # the full world ---------------------------------------------------------------------------
    @property
    def world(self) -> LabWorld:
        if self._world is not None:
            return self._world
        if self._built is None:
            self._built = build_world(self.layout, self.need_config(), ledger=self.ledger)
        return self._built

    def lab_cap(self) -> Any:
        w = self.world
        cap = getattr(w, "lab_cap", None)
        if cap is None:
            from nightshift.lab.budget import LabCap, lab_spent_usd

            cfg = self.need_config()
            cap = LabCap(cfg.lab_budget_usd, lab_spent_usd(self.state(), self.layout), clock=w.clock)
            try:
                w.lab_cap = cap  # type: ignore[attr-defined]
            except AttributeError:
                pass
        return cap

    def close(self) -> None:
        if self._opened_here and self._ledger is not None:
            try:
                self._ledger.close()
            except Exception:  # noqa: BLE001
                pass
        built = self._built
        if built is not None:
            closer = getattr(built, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001
                    pass

    # output -----------------------------------------------------------------------------------
    def emit(self, obj: Any) -> None:
        self.out.write(json.dumps(obj, sort_keys=True, indent=1, default=str) + "\n")

    def note(self, text: str) -> None:
        self.err.write(text + "\n")

    # appending --------------------------------------------------------------------------------
    def append(self, kind: str, record: Any, actor: str = ACTOR) -> Any:
        return self.ledger.append(kind, actor, record)

    def freeze(self, kind: str, name: str, path: Path, record: Any) -> FrozenFile:
        """Write a frozen file (atomically, read-only afterwards) and append its ``FrozenFile`` event."""
        path.parent.mkdir(parents=True, exist_ok=True)
        attr = _FROZEN_STATE_ATTR.get(kind)
        if path.exists() and (attr is None or getattr(self.state(), attr) is not None):
            raise LedgerError(f"{path} exists already; a frozen file is written once")
        # a file the ledger does not name was left by a crash between the write and the event: it was never frozen, so the retry replaces it
        write_record(path, record)
        ff = FrozenFile(name, self.layout.file_ref(path))
        self.append(kind, ff)
        try:
            os.chmod(path, 0o444)
        except OSError:
            pass
        return ff

    def anchor(self, label: str, *, publish: bool = False) -> str:
        """Append the master head to the anchors file. ``publish`` (the design's milestones: rule freeze, every decision, final, reveal)
        also commits and pushes that file; a push that fails is reported on stderr and never undoes the milestone (the ledger already
        has it), and ``anchor --push`` repeats the push."""
        a = self.ledger.anchor()
        path = self.layout.anchors_path
        path.parent.mkdir(parents=True, exist_ok=True)
        head = f"{a.seq}:{a.hash16}"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{head} {label} {self.now()}\n")
        if publish:
            res = _publish_anchors(path, head)
            if not res.get("pushed"):
                self.note(f"warning: anchor {head} ({label}) is only local, not pushed: {res.get('push_error', 'unknown')}; fix the git remote and run: anchor --push")
        return head


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def _publish_anchors(path: Path, head: str) -> dict:
    """Commit only the anchors file and push it to ``origin main``. ``{"pushed": True}`` or ``{"push_error": ...}``; never raises."""
    git = lambda *args: subprocess.run(["git", *args], cwd=path.parent, capture_output=True, text=True, timeout=120)  # noqa: E731
    steps = [("add", ["add", "--", path.name]), ("commit", ["commit", "-m", f"lab anchor {head}", "--", path.name]),
             ("pull", ["pull", "--rebase", "origin", "main"]), ("push", ["push", "origin", "main"])]
    for name, args in steps:
        try:
            r = git(*args)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"push_error": f"git {name}: {type(exc).__name__}: {exc}"[:300]}
        if r.returncode != 0 and name == "pull" and "couldn't find remote ref" in (r.stderr or ""):
            continue  # an empty remote: nothing to rebase onto yet
        if r.returncode != 0:
            return {"push_error": f"git {name}: {(r.stderr or r.stdout).strip()[:300]}"}
    return {"pushed": True}


def _csv(text: Optional[str], cast: Callable[[str], Any] = str) -> tuple:
    return tuple(cast(p.strip()) for p in (text or "").split(",") if p.strip())


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, sort_keys=True, indent=1) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _need_frozen(env: _Env, state: LabState) -> tuple:
    """(manifest, budget, rule-or-None, analysis-or-None), each re-hashed against the ledger."""
    if state.manifest is None or state.budget is None:
        raise ContractError("assignments and budget must be frozen first (freeze-manifest, freeze-budget)")
    manifest = load_frozen(env.layout, state.manifest, AssignmentManifest)
    budget = load_frozen(env.layout, state.budget, BudgetSpec)
    rule = load_frozen(env.layout, state.rule, Rule) if state.rule is not None else None
    analysis = load_frozen(env.layout, state.final_analysis, FinalAnalysis) if state.final_analysis is not None else None
    return manifest, budget, rule, analysis


def _trial_mean(state: LabState, score_id: str, metric: str) -> float:
    vals = state.scores[score_id].metric_by_seed(metric)
    return math.fsum(vals) / len(vals)


def _seed_bits(state: LabState, pack_commit: str, assignment: str, phase: Phase, arm: Optional[Arm] = None) -> int:
    seeds = [s.seed for s in state.trials_started.values() if s.pack.commit == pack_commit and s.assignment_id == assignment and s.phase is phase and s.arm == arm]
    return (max(seeds) + 1) if seeds else 1


def _trial_summary(rec: TrialRecord) -> dict:
    """Bits and costs only: no score values."""
    u = rec.budget_used
    return {
        "trial_id": rec.spec.trial_id, "status": rec.status.value, "pack_version": rec.spec.pack_version, "assignment": rec.spec.assignment_id,
        "phase": rec.spec.phase.value, "arm": None if rec.spec.arm is None else rec.spec.arm.value, "scored": rec.selection_score_id is not None,
        "failure": rec.failure, "usage": {"researcher_tokens": u.researcher_tokens, "gpu_minutes": u.gpu_minutes, "dollars": round(u.dollars, 6),
                                            "wall_seconds": u.wall_seconds},
    }


# --------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------
def cmd_init(env: _Env, a: argparse.Namespace) -> None:
    layout = env.layout
    if a.config:
        text = Path(a.config).read_bytes().decode("utf-8")
        cfg = LabConfig.from_dict(loads_strict(text))
        if not layout.lab_config_path.exists():
            layout.root.mkdir(parents=True, exist_ok=True)
            write_record(layout.lab_config_path, cfg)
        env._config = cfg
    cfg = env.config
    if cfg is None:
        raise ContractError("lab.json is missing: pass --config FILE (a LabConfig record)")
    if env.state().initialized is not None:
        raise LedgerError("this lab is already initialized")
    if not layout.lab_config_path.exists():
        write_record(layout.lab_config_path, cfg)
    from nightshift.lab.ifdata import ensure_split_salt

    for d in (layout.frozen_dir, layout.anchors_path.parent, layout.reports_dir, layout.public_data_dir):
        d.mkdir(parents=True, exist_ok=True)
    ensure_split_salt(layout)
    env.append(LabKind.INITIALIZED, LabInit(cfg.lab_id, env.now(), a.host or cfg.control_host))
    env.emit({"initialized": cfg.lab_id, "root": str(layout.root)})


def _parse_plan(text: str) -> Any:
    from nightshift.lab.ifdata import AssignmentPlan

    parts = text.split(":")
    if len(parts) != 6:
        raise UsageError(f"--plan {text!r}: expected ID:ROLE:FAMILY,FAMILY:N_SELECTION:N_DEV:N_TRAIN")
    aid, role, fams, nsel, ndev, ntrain = parts
    try:
        return AssignmentPlan(aid, AssignmentRole(role), _csv(fams), int(nsel), int(ndev), int(ntrain))
    except ValueError as exc:
        raise UsageError(f"--plan {text!r}: {exc}") from None


def cmd_freeze_manifest(env: _Env, a: argparse.Namespace) -> None:
    from nightshift.lab.ifdata import build_assignment_manifest, build_splits, ensure_split_salt

    state = env.state()
    if state.initialized is None:
        raise ContractError("run init first")
    if state.manifest is not None:
        raise LedgerError("the assignments are frozen already")
    if state.candidates:
        raise ContractError("the assignment manifest is frozen before any candidate")
    cfg = env.need_config()
    plans = [_parse_plan(p) for p in a.plan]
    tests = [Path(t) for t in a.test_file]
    salt = ensure_split_salt(env.layout)
    evo, held = _csv(a.evolution_families), _csv(a.heldout_families)
    build_splits(a.rows, tests, env.layout.root, salt=salt, evolution_families=evo, heldout_families=held, plans=plans)
    dest = env.layout.test_data()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tests[0], dest)
    os.chmod(dest, 0o600)
    manifest = build_assignment_manifest(
        env.layout, plans, evolution_families=evo, heldout_families=held,
        ifrlvr_parquet_sha256=a.source_sha256 or sha256_file(a.rows),
        ifbench_commit=cfg.pins["ifbench_commit"], open_instruct_commit=cfg.pins["open_instruct_commit"], clock=env.clock,
    )
    env.freeze(LabKind.ASSIGNMENTS_FROZEN, "assignments", env.layout.manifest_path, manifest)
    env.emit({"assignments": [x.assignment_id for x in manifest.assignments], "item_mode": manifest.item_mode.value,
              "note": "item_mode salted_draw: the residual contamination risk is covered by the contamination scan and printed in the report"})


def cmd_freeze_budget(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    if state.initialized is None:
        raise ContractError("run init first")
    if state.candidates:
        raise ContractError("the budget is frozen before any candidate")
    budget = BudgetSpec(researcher_tokens=a.researcher_tokens, gpu_minutes=a.gpu_minutes, dollars=a.dollars, wall_seconds=a.wall_seconds)
    env.freeze(LabKind.BUDGET_FROZEN, "budget", env.layout.budget_path, budget)
    env.emit({"budget": budget.to_dict()})


def cmd_record_gate(env: _Env, a: argparse.Namespace) -> None:
    rec = GateRecord(Gate(a.gate), GateOutcome(a.outcome), a.statement, a.decided_by, env.now())
    env.append(LabKind.GATE_RECORDED, rec)
    env.emit({"gate": rec.gate.value, "outcome": rec.outcome.value, "gate_status": env.reader.gate_status(env.state(), rec.gate)})


def cmd_commit_pack(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    if "v0" in state.packs:
        raise LedgerError("v0 is committed already")
    packs = env.world.packs
    ref = packs.init_v0(a.template) if a.template else packs.init_v0(_default_template())
    env.append(LabKind.PACK_COMMITTED, PackCommitted("v0", ref, None, None))
    env.emit({"version": "v0", "commit": ref.commit})


def _default_template() -> Path:
    from nightshift.lab.packs import PACK_V0_TEMPLATE

    return PACK_V0_TEMPLATE


def cmd_plan_spread(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    manifest, _budget, rule, _an = _need_frozen(env, state)
    if rule is not None or state.spread_plan is not None:
        raise LedgerError("a spread plan exists already, or the rule is frozen")
    if "v0" not in state.packs:
        raise ContractError("commit v0 first (commit-pack)")
    assignments = _csv(a.assignments) or tuple(manifest.ids_with_role(AssignmentRole.EVOLUTION))
    for aid in assignments:
        if manifest.get(aid).role is not AssignmentRole.EVOLUTION:
            raise ContractError(f"{aid!r} is not an evolution assignment")
    plan = SpreadPlan(a.metric, env.world.scorer.scorer_version, assignments, a.trials, env.now())
    env.append(LabKind.SPREAD_PLANNED, plan)
    env.emit({"planned_trials": plan.planned_trials, "assignments": list(plan.assignments), "metric": plan.metric})


def _score_params(a: argparse.Namespace) -> tuple:
    return _csv(a.score_seeds, int) or (0, 1, 2), GenerationParams.recommended(a.max_tokens)


def cmd_trial(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    manifest, budget, rule, _an = _need_frozen(env, state)
    cfg = env.need_config()
    committed = state.packs.get(a.pack_version)
    if committed is None:
        raise ContractError(f"pack version {a.pack_version!r} is not committed")
    if a.pack_version == "v0" and rule is None:
        if state.spread_plan is None:
            raise ContractError("plan-spread precedes the first spread trial")
    elif rule is None:
        raise ContractError("a candidate's trial needs the frozen rule")
    if a.pack_version != "v0" and not env.reader.has_gate(state, Gate.TERMS_10_1_IX):
        raise GateNotRecorded("record the terms gate before a trial of a candidate pack")
    world = env.world
    if rule is not None:
        seeds, params = rule.score_seeds, rule.score_params
        scorer_version = rule.scorer_version
        assignment = a.assignment or rule.assignments[0]
    else:
        seeds, params = _score_params(a)
        scorer_version = world.scorer.scorer_version
        assignment = a.assignment or state.spread_plan.assignments[0]
    seed = a.seed if a.seed is not None else _seed_bits(state, committed.pack.commit, assignment, Phase.SEARCH)
    tid = a.trial_id or env.reader.next_id(state, "t")
    spec = TrialSpec(
        trial_id=tid, phase=Phase.SEARCH, arm=None, pack_version=a.pack_version, pack=committed.pack, assignment_id=assignment, seed=seed,
        budget=budget, shape=cfg.shape, base=cfg.base, data_revisions=dict(manifest.get(assignment).data_revisions), scorer_version=scorer_version,
        score_params=params, score_seeds=tuple(seeds), final_request_id=None,
    )
    check_trial_against_frozen(spec, manifest=manifest, budget=budget, rule=rule)
    rec = world.trial_executor.run(spec)
    env.emit(_trial_summary(rec))


def _spread_trials(state: LabState, plan: SpreadPlan) -> list:
    t0 = state.event_seq.get((LabKind.SPREAD_PLANNED, ""), 0)
    return [t for t, s in state.trials_started.items()
            if s.phase is Phase.SEARCH and s.pack_version == "v0" and s.assignment_id in plan.assignments
            and state.event_seq.get((LabKind.TRIAL_STARTED, t), 0) > t0]


def cmd_spread(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    plan = state.spread_plan
    if plan is None:
        raise ContractError("declare the plan first (plan-spread)")
    if state.spread is not None:
        raise LedgerError("the spread is measured already")
    if state.rule is not None:
        raise ContractError("the rule is frozen")
    ids = _spread_trials(state, plan)
    if len(ids) != plan.planned_trials:
        raise ContractError(f"the plan has {plan.planned_trials} trials; {len(ids)} were started after it (run exactly the planned number)")
    unfinished = [t for t in ids if t not in state.trials]
    if unfinished:
        raise ContractError(f"trial(s) {unfinished} have not finished")
    scores: list = []
    failed = 0
    for t in ids:
        rec = state.trials[t]
        sc = state.scores.get(rec.selection_score_id) if rec.selection_score_id else None
        hit = state.contamination.get(t)
        if rec.status not in SCORED or sc is None or (hit is not None and not hit.passed) or sc.scorer_version != plan.scorer_version \
                or any(plan.metric not in s.metrics for s in sc.per_seed):
            if getattr(a, "failed_as_zero", False):
                # the researcher pack's unreliability counts against it, as it will for every candidate under SCORED_ZERO
                spec = state.trials_started[t]
                scores.append(TrialScore(t, spec.assignment_id, None, plan.metric, plan.scorer_version, comparability_key(spec, plan.metric), 0.0))
                continue
            failed += 1
            continue
        spec = state.trials_started[t]
        scores.append(TrialScore(t, spec.assignment_id, sc.score_id, plan.metric, plan.scorer_version, comparability_key(spec, plan.metric),
                                 _trial_mean(state, sc.score_id, plan.metric)))
    if not scores:
        raise ContractError("no completed spread trial")
    est = stats.trial_spread(scores)
    meas = SpreadMeasured(plan.metric, plan.scorer_version, plan.planned_trials, failed, tuple(scores), est)
    env.append(LabKind.SPREAD_MEASURED, meas)
    env.emit({"planned": plan.planned_trials, "completed": len(scores), "failed": failed, "sigma": est.sigma, "df": est.df,
              "n_assignments": est.n_assignments})


def sidak_alpha(family_alpha: float, rounds: int) -> float:
    """Per-round alpha so that R rounds together allow ``family_alpha``: 1 - (1 - family)^(1/R)."""
    return family_alpha if rounds <= 1 else 1.0 - (1.0 - family_alpha) ** (1.0 / rounds)


def cmd_freeze_rule(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    manifest, budget, rule, _an = _need_frozen(env, state)
    cfg = env.need_config()
    if rule is not None:
        raise LedgerError("the rule is frozen already")
    if state.candidates:
        raise ContractError("the rule is frozen before the first candidate")
    plan, meas = state.spread_plan, state.spread
    if plan is None or meas is None:
        raise ContractError("measure the v0 spread first (plan-spread, trial, spread)")
    if meas.n_failed > 0:
        raise ContractError("the spread had failed trials: declare a new plan instead of freezing on a subset")
    if any(t.score_id is None for t in meas.trials) and FailedTrialPolicy(a.failed_trial_policy) is not FailedTrialPolicy.SCORED_ZERO:
        raise ContractError("the spread counts failed v0 trials as 0.0; the rule must then score failed trials as zero for every pack (--failed-trial-policy scored_zero)")
    if (meas.metric, meas.scorer_version, meas.planned_trials) != (plan.metric, plan.scorer_version, plan.planned_trials):
        raise ContractError("the measured spread does not match the declared plan")
    est = meas.estimate
    if est.df < 1:
        raise ContractError("the spread has no degrees of freedom: spread not estimated")
    started = [state.trials_started[t] for t in _spread_trials(state, plan)]
    pairs = {(s.score_seeds, record_sha256(s.score_params.to_dict())) for s in started}
    if len(pairs) != 1:
        raise ContractError("the spread trials used different seeds or decoding")
    seeds, params = started[0].score_seeds, started[0].score_params
    assignments = _csv(a.assignments) or plan.assignments
    n_rounds = -(-a.max_candidates // a.screen_batch_size)
    alpha = a.alpha if a.alpha is not None else sidak_alpha(a.family_alpha, n_rounds)
    pooled = est.df + (a.confirm_trials - 1) + (a.confirm_trials_incumbent - 1)
    mdg = stats.minimum_detectable_gain(est.sigma, pooled, a.confirm_trials, a.confirm_trials_incumbent, len(assignments), alpha, min_gain=a.min_gain)
    try:
        new = Rule(
            metric=plan.metric, split=Split.SELECTION, scorer_version=plan.scorer_version, assignments=tuple(assignments), spread=est, alpha=alpha,
            min_gain=a.min_gain, score_seeds=tuple(seeds), score_params=params, screen_batch_size=a.screen_batch_size,
            screen_trials_per_candidate=a.screen_trials, confirm_trials_per_candidate=a.confirm_trials,
            confirm_trials_per_incumbent=a.confirm_trials_incumbent, variance_guard_ratio=a.variance_guard_ratio,
            failed_trial_policy=FailedTrialPolicy(a.failed_trial_policy), assumed_gain=a.assumed_gain, minimum_detectable_gain=mdg,
            degenerate_override=a.degenerate_override, max_candidates=a.max_candidates, stop_after_rejections=a.stop_after_rejections,
            repeated_decisions_note=("Screen then confirm; the selection results are an adaptive search trace and the operator's choices steer "
                                      "the search. The held-out comparison is the claim."),
            frozen_utc=env.now(),
        )
    except ContractError as exc:
        if mdg > a.assumed_gain:
            raise ContractError(f"the lab cannot detect the gain you believe in: minimum detectable gain {mdg:.4f} exceeds assumed gain {a.assumed_gain}; "
                                f"use fewer candidates, bigger confirmations or a bigger effect target ({exc})") from None
        raise
    validate_rule_against_manifest(new, manifest)
    cap = cfg.affordable_trials
    if new.planned_search_trials() > cap / 2:
        raise ContractError(f"the rule plans {new.planned_search_trials()} search trials, more than half of the {cap} affordable trials (half is reserved for the final comparison)")
    env.freeze(LabKind.RULE_FROZEN, "rule", env.layout.rule_path, new)
    head = env.anchor("rule_frozen", publish=True)
    env.emit({"alpha": alpha, "rounds": n_rounds, "minimum_detectable_gain": mdg, "pooled_df": new.pooled_df(),
              "planned_search_trials": new.planned_search_trials(), "family_false_accept_bound": stats.false_accept_bound(alpha, n_rounds), "anchor": head})


def cmd_outer(env: _Env, a: argparse.Namespace) -> None:
    from nightshift.lab.outer_loop import OuterController

    state = env.state()
    manifest, budget, rule, _an = _need_frozen(env, state)
    if rule is None:
        raise ContractError("freeze the rule first")
    if not env.reader.has_gate(state, Gate.TERMS_10_1_IX):
        raise GateNotRecorded("record the terms gate (record-gate) before the first Wonder-directed revision")
    world, cfg = env.world, env.need_config()
    if world.proposal_source is None:
        raise ContractError("no proposal source is configured")
    set_cap = getattr(world.proposal_source, "set_call_cap", None)
    if callable(set_cap):  # the production source: its gate otherwise keeps the $5 default whatever --wonder-usd says
        set_cap(a.wonder_usd, rule.max_candidates)
    controller = OuterController(
        env.layout, world.ledger, world.state_reader, world.packs, world.proposal_source, world.trial_executor, world.decision_rule,
        clock=world.clock, lab_cap=env.lab_cap(), hosts=(cfg.control_host, cfg.researcher_host), proposal_usd_cap=a.wonder_usd,
        trial_llm_usd_cap=a.trial_llm_usd, scoring_usd_per_trial=a.scoring_usd_per_trial, shape=cfg.shape, base=cfg.base,
        after_decision=lambda d: env.anchor(f"decision:{d.candidate_id}:{d.stage.value}", publish=True),
        challenger=getattr(world, "challenger", None),
    )
    decisions = controller.run(a.rounds)
    env.emit({"decisions": [{"candidate": d.candidate_id, "stage": d.stage.value, "outcome": d.outcome.value, "round": d.round_index} for d in decisions]})


def cmd_close_search(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    if state.search_closed is not None:
        raise LedgerError("the search is closed already")
    _need_frozen(env, state)
    if state.rule is None:
        raise ContractError("there is no search to close: the rule is not frozen")
    pending = [t for t in state.trials_started if t not in state.trials]
    if pending:
        raise ContractError(f"search trial(s) {pending[:5]} have no lab.trial_finished (finish them or run abort-trial)")
    inc = env.reader.incumbent_version(state)
    reason = a.reason
    if getattr(a, "final_pack", None) and a.final_pack != inc:
        # an exploratory final pack that the keep rule did not accept (owner-approved design, e.g. Tier B "frozen v0 vs frozen best pack on
        # held-out"): it must be a committed pack and a recorded operator change must name it first, so the deviation is visible in the ledger
        if a.final_pack not in state.packs:
            raise ContractError(f"pack {a.final_pack!r} is not committed")
        if not any(c.target == "final_pack" and a.final_pack in c.detail for c in state.operator_changes):
            raise ContractError(f"record the exception first: operator-change --target final_pack --detail '... {a.final_pack} ...'")
        reason = f"{a.reason} [exploratory final pack {a.final_pack}: not accepted by the keep rule (rule incumbent {inc}); see lab.operator_change final_pack]"
        inc = a.final_pack
    n_trials = sum(1 for s in state.trials_started.values() if s.phase is Phase.SEARCH)
    rec = SearchClosed(inc, state.packs[inc].pack, len(state.candidates), n_trials, reason, env.now())
    env.append(LabKind.SEARCH_CLOSED, rec)
    env.emit({"incumbent_version": inc, "n_candidates": rec.n_candidates, "n_search_trials": rec.n_search_trials})


def cmd_freeze_final_analysis(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    manifest, budget, rule, analysis = _need_frozen(env, state)
    cfg = env.need_config()
    if state.search_closed is None:
        raise ContractError("close-search first")
    if analysis is not None:
        raise LedgerError("the final analysis is frozen already")
    assert rule is not None
    exploratory = bool(getattr(a, "exploratory", False))
    if exploratory and not any(c.target == "final_analysis_exploratory" for c in state.operator_changes):
        raise ContractError("an exploratory final analysis needs a recorded operator-change --target final_analysis_exploratory (owner-approved design) first")
    if a.trials_per_pack < (2 if exploratory else MIN_FINAL_TRIALS_PER_PACK):
        raise ContractError(f"at least {2 if exploratory else MIN_FINAL_TRIALS_PER_PACK} trials per pack")
    held = manifest.ids_with_role(AssignmentRole.HELDOUT)
    if a.assignments:
        weights = {}
        for item in _csv(a.assignments):
            aid, _, w = item.partition(":")
            weights[aid] = float(w) if w else None
        ids = list(weights)
        given = [w for w in weights.values() if w is not None]
        if len(given) == len(ids):
            ws = [weights[i] for i in ids]
        else:
            ws = [1.0 / len(ids)] * len(ids)
    else:
        ids = list(held)
        ws = [1.0 / len(ids)] * len(ids)
    total = sum(ws)
    ws = [w / total for w in ws]
    per_cell = a.trials_per_pack
    taken: list = []
    planned = []
    for arm in (Arm.V0, Arm.EVOLVED):
        for aid in ids:
            for k in range(per_cell):
                tid = env.reader.next_id(state, "t", taken=taken)
                taken.append(tid)
                planned.append(PlannedTrial(tid, arm, aid, a.seed_base + k))
    n_final = len(planned)
    if rule.planned_search_trials() + n_final > cfg.affordable_trials:
        raise ContractError(f"planned search trials ({rule.planned_search_trials()}) plus final trials ({n_final}) exceed the {cfg.affordable_trials} affordable trials")
    new = FinalAnalysis(
        heldout_metric=a.heldout_metric, test_metric=a.test_metric, assignments=tuple(AssignmentWeight(i, w) for i, w in zip(ids, ws)),
        planned_trials=tuple(planned), confidence=a.confidence, method=ComparisonMethod.BLOCKED_T if len(ids) >= 2 else ComparisonMethod.WELCH,
        score_seeds=rule.score_seeds, score_params=rule.score_params, failed_trial_policy=rule.failed_trial_policy,
        retry_policy="Retry only after an infrastructure failure that revealed no score (lab.score_failed with result_revealed false).",
        budget_sha256=record_sha256(budget), frozen_utc=env.now(), exploratory=exploratory,
    )
    validate_analysis_against_manifest(new, manifest)
    env.freeze(LabKind.FINAL_ANALYSIS_FROZEN, "final_analysis", env.layout.final_analysis_path, new)
    env.emit({"method": new.method.value, "assignments": ids, "trials_per_pack_per_assignment": per_cell, "planned_trials": n_final})


def _final_spec(env: _Env, state: LabState, plan: PlannedTrial, fr: FinalRequest, manifest: AssignmentManifest, budget: BudgetSpec,
                analysis: FinalAnalysis, scorer_version: str) -> TrialSpec:
    cfg = env.need_config()
    assert fr.pack is not None and fr.pack_version is not None
    return TrialSpec(
        trial_id=plan.trial_id, phase=Phase.FINAL, arm=plan.arm, pack_version=fr.pack_version, pack=fr.pack, assignment_id=plan.assignment_id,
        seed=plan.seed, budget=budget, shape=cfg.shape, base=cfg.base, data_revisions=dict(manifest.get(plan.assignment_id).data_revisions),
        scorer_version=scorer_version, score_params=analysis.score_params, score_seeds=analysis.score_seeds, final_request_id=fr.request_id,
    )


def cmd_final(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    manifest, budget, rule, analysis = _need_frozen(env, state)
    if state.search_closed is None or analysis is None:
        raise FinalComparisonRefused("final needs both lab.search_closed and lab.final_analysis_frozen")
    arm = Arm(a.arm)
    world = env.world
    if a.resume:
        fr = next((f for f in state.finals if f.request_id == a.resume), None)
        if fr is None:
            raise FinalComparisonRefused(f"no final request {a.resume!r}")
        if fr.arm is not arm:
            raise UsageError("--resume names a request of another arm")
    else:
        if arm is Arm.STOCK:
            pack_version, pack, splits = None, None, (Split.TEST,)
        elif arm is Arm.V0:
            pack_version, pack, splits = "v0", state.packs["v0"].pack, (Split.SELECTION, Split.TEST)
        else:
            pack_version, pack, splits = state.search_closed.incumbent_version, state.search_closed.incumbent, (Split.SELECTION, Split.TEST)
        if analysis.exploratory:
            if arm is Arm.STOCK:
                raise FinalComparisonRefused("the exploratory final analysis compares the two researcher packs on held-out selection only; no stock or test arm")
            splits = (Split.SELECTION,)  # no IFBench test scoring in the exploratory comparison
        prior = [f for f in state.finals if f.arm is arm and f.pack == pack]
        if prior and not a.repeat_of:
            raise FinalComparisonRefused(f"a final request for arm {arm.value} and this pack commit exists ({prior[0].request_id}); a repeat needs --repeat-of and --reason")
        if a.repeat_of and not a.reason:
            raise FinalComparisonRefused("--repeat-of needs --reason")
        if a.repeat_of and not any(f.request_id == a.repeat_of for f in state.finals):
            raise FinalComparisonRefused(f"--repeat-of names an unknown request {a.repeat_of!r}")
        used = [p.trial_id for p in analysis.planned_trials if p.arm is arm and p.trial_id in state.trials_started]
        if a.repeat_of and used:  # a trial id is used once: the repeat would skip every planned trial and report success having run nothing
            raise FinalComparisonRefused(f"the planned trial(s) {used[:5]} of arm {arm.value} have run already and a trial id is used once; a repeat needs a new frozen plan")
        rid = env.reader.next_id(state, "f")
        fr = FinalRequest(rid, arm, pack_version, pack, analysis.sha256(), splits, a.repeat_of, a.reason, env.now())
        env.append(LabKind.FINAL_REQUESTED, fr)
        env.anchor(f"final:{rid}", publish=True)
    scorer_version = world.scorer.scorer_version
    if arm is Arm.STOCK:
        rec = _score_stock(env, fr, analysis)
        env.emit({"final_request": fr.request_id, "arm": "stock", "scored": rec.score_id})
        return
    unfinished = [p.trial_id for p in analysis.planned_trials if p.arm is arm and p.trial_id in state.trials_started and p.trial_id not in state.trials]
    if unfinished:  # a trial is never resumed: record it with abort-trial rather than skip it and report success
        raise ContractError(f"planned trial(s) {unfinished[:5]} started but have no lab.trial_finished; run abort-trial for each, then final --resume {fr.request_id}")
    done = []
    for plan in analysis.planned_trials:
        if plan.arm is not arm:
            continue
        state = env.state()
        if plan.trial_id in state.trials_started:
            continue
        spec = _final_spec(env, state, plan, fr, manifest, budget, analysis, scorer_version)
        rec = world.trial_executor.run(spec)
        done.append({"trial_id": rec.spec.trial_id, "status": rec.status.value})  # no scores: they stay hidden until reveal
    env.emit({"final_request": fr.request_id, "arm": arm.value, "trials": done})


def _score_stock(env: _Env, fr: FinalRequest, analysis: FinalAnalysis) -> ScoreRecord:
    from nightshift.lab.score_session import score_adapter

    world, cfg = env.world, env.need_config()
    state = env.state()
    req = ScoreRequest(
        request_id=env.reader.next_id(state, "sc"), phase=Phase.FINAL, assignment_id=TEST_ASSIGNMENT_ID, split=Split.TEST, trial_id=None,
        arm=Arm.STOCK, base=cfg.base, adapter=None, seeds=analysis.score_seeds, params=analysis.score_params,
        scorer_version=world.scorer.scorer_version, reason=None, final_request_id=fr.request_id,
    )
    try:
        record, usage = score_adapter(req, world.model_server, world.scorer, world.generator_factory)
    except AccessDenied:
        raise
    except Exception as exc:
        usage = getattr(exc, "gpu_usage", None)
        env.append(LabKind.SCORE_FAILED, ScoreFailure(req.request_id, req.split, req.assignment_id, None, None, f"{type(exc).__name__}: {exc}"[:1500], False, env.now()), "scorer")
        if usage is not None:
            env.append(LabKind.SCORE_COST, ScoreCost(req.request_id, usage), "scorer")
        raise
    env.append(LabKind.SCORE_RECORDED, record, "scorer")
    env.append(LabKind.SCORE_COST, ScoreCost(record.score_id, usage), "scorer")
    return record


def cmd_reveal(env: _Env, a: argparse.Namespace) -> None:
    from nightshift.lab import report as report_mod

    state = env.state()
    _m, _b, _r, analysis = _need_frozen(env, state)
    if state.search_closed is None or analysis is None:
        raise FinalComparisonRefused("reveal needs lab.search_closed and the frozen final analysis")
    if state.reveal is not None:
        raise LedgerError("already revealed")
    warnings: list = []
    fin = report_mod.final_numbers(state, analysis, warnings)
    if not fin.complete:
        raise FinalComparisonRefused("reveal refuses until every planned trial has its held-out and test score: missing " + ", ".join(fin.missing[:8]))
    n_scores = sum(1 for r in state.scores.values() if r.phase is Phase.FINAL and r.split in (Split.SELECTION, Split.TEST))
    out: dict = {"schema_version": 1, "analysis_sha256": analysis.sha256(), "comparisons": []}
    for fc in (fin.heldout, fin.test):
        if fc is not None:
            out["comparisons"].append(fc.to_dict())
    for name in ("stock_vs_v0", "stock_vs_evolved"):
        v = getattr(fin, name)
        if v is not None:
            out[name] = dataclasses.asdict(v) if dataclasses.is_dataclass(v) else str(v)
    if fin.stock is None:
        out["note"] = "arm 1 (stock) has no test score: the stock comparison was not made"
    _atomic_json(env.layout.final_comparison_path, out)
    env.append(LabKind.FINAL_REVEALED, FinalReveal(analysis.sha256(), n_scores, env.now()))
    head = env.anchor("reveal", publish=True)
    env.emit({"revealed": True, "n_scores": n_scores, "comparisons": len(out["comparisons"]), "warnings": warnings, "anchor": head})


def cmd_score(env: _Env, a: argparse.Namespace) -> None:
    """A diagnostic score of an existing adapter through the same ScoreService and guard as everything else."""
    from nightshift.lab.score_session import score_adapter

    state = env.state()
    manifest, budget, rule, analysis = _need_frozen(env, state)
    world, cfg = env.world, env.need_config()
    if a.stock == bool(a.adapter):
        raise UsageError("give exactly one of --adapter ID or --stock")
    adapter = None if a.stock else read_record(env.layout.adapter_json_path(a.adapter), AdapterRef)
    phase = Phase.FINAL if state.search_closed is not None else Phase.SEARCH
    if phase is Phase.FINAL and not a.final_request:
        raise UsageError("after lab.search_closed a score needs --final-request")
    if rule is not None:
        seeds, params, version = rule.score_seeds, rule.score_params, rule.scorer_version
    else:
        seeds, params = _score_params(a)
        version = world.scorer.scorer_version
    split = Split(a.split)
    started = state.trials_started.get(a.trial) if a.trial else None
    req = ScoreRequest(  # the arm is the trial's own, as the trial runner sets it; the final guard compares it with the final request's
        request_id=env.reader.next_id(state, "sc"), phase=phase, assignment_id=TEST_ASSIGNMENT_ID if split is Split.TEST else a.assignment,
        split=split, trial_id=a.trial, arm=started.arm if started is not None and phase is Phase.FINAL else None, base=cfg.base, adapter=adapter, seeds=tuple(seeds), params=params, scorer_version=version,
        reason=a.reason, final_request_id=a.final_request if phase is Phase.FINAL else None,
    )
    try:
        record, usage = score_adapter(req, world.model_server, world.scorer, world.generator_factory)
    except AccessDenied:
        raise
    except Exception as exc:
        env.append(LabKind.SCORE_FAILED, ScoreFailure(req.request_id, req.split, req.assignment_id, req.trial_id, adapter.sha256 if adapter is not None else None, f"{type(exc).__name__}: {exc}"[:1500], False, env.now()), "scorer")
        if getattr(exc, "gpu_usage", None) is not None:
            env.append(LabKind.SCORE_COST, ScoreCost(req.request_id, exc.gpu_usage), "scorer")
        raise
    env.append(LabKind.SCORE_RECORDED, record, "scorer")
    env.append(LabKind.SCORE_COST, ScoreCost(record.score_id, usage), "scorer")
    out = {"score_id": record.score_id, "split": record.split.value, "n_prompts": record.n_prompts}
    if record.split is Split.DEV:  # dev numbers are public to researchers; selection and test numbers stay out of the terminal
        out["headline_mean"] = record.headline_mean()
    env.emit(out)


def cmd_abort_trial(env: _Env, a: argparse.Namespace) -> None:
    from nightshift.lab.budget import usage_from_events
    from nightshift.lab.ledger import LabLedger

    state = env.state()
    tid = a.trial_id
    spec = state.trials_started.get(tid)
    if spec is None:
        raise ContractError(f"{tid!r} has no lab.trial_started")
    if tid in state.trials:
        raise ContractError(f"{tid!r} has a lab.trial_finished already")
    tdir = env.layout.trial_ledger_dir(tid)
    events: list = []
    if (tdir / "ledger.jsonl").exists():
        events = LabLedger.read_only(tdir).events()
    used = usage_from_events(events, tid)
    ctx_file = env.layout.trial_dir(tid) / "initial_context.sha256"
    ctx = None
    if ctx_file.exists():
        text = ctx_file.read_text().strip()
        ctx = text if len(text) == 64 else None
    started = utc_iso(events[0].ts) if events else env.now()
    rec = TrialRecord(spec, TrialStatus.ABORTED, started, env.now(), ctx, used, None, None, None, None, None, None, None, {"aborted_by": "cli abort-trial"},
                      "aborted by the operator: the coordinator died or the trial was abandoned")
    if not env.layout.trial_record_path(tid).exists():
        write_record(env.layout.trial_record_path(tid), rec)
    env.append(LabKind.TRIAL_FINISHED, rec, f"trial:{tid}")
    env.emit({"aborted": tid, "usage": used.to_dict()})


def cmd_operator_change(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    rec = OperatorChange(a.target, a.detail, len(state.candidates), env.now())
    env.append(LabKind.OPERATOR_CHANGE, rec)
    env.emit({"recorded": a.target, "candidates_so_far": rec.candidates_so_far})


def cmd_report(env: _Env, a: argparse.Namespace) -> None:
    from nightshift.lab import report as report_mod

    state = env.state()
    rep = report_mod.build_report(state, env.layout, full=a.full)
    text = report_mod.render_html(rep) if a.format == "html" else report_mod.render_markdown(rep)
    if a.out:
        dest = (env.layout.reports_dir / a.out).resolve()
        if env.layout.reports_dir.resolve() not in dest.parents:
            raise UsageError("--out must be a path inside reports/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        env.emit({"written": str(dest.relative_to(env.layout.root.resolve())), "full": a.full})
    else:
        env.out.write(text if text.endswith("\n") else text + "\n")


def _candidate_bit(state: LabState, cid: str) -> str:
    d = state.decisions.get(cid)
    return "pending" if d is None else d.outcome.value


def cmd_status(env: _Env, a: argparse.Namespace) -> None:
    state = env.state()
    pending = [t for t in state.trials_started if t not in state.trials]
    spent = [r.budget_used for r in state.trials.values()]
    out: dict = {
        "initialized": state.initialized is not None,
        "frozen": {"assignments": state.manifest is not None, "budget": state.budget is not None, "rule": state.rule is not None,
                   "final_analysis": state.final_analysis is not None},
        "terms_gate": bool(state.gates), "terms_gate_status": env.reader.gate_status(state, Gate.TERMS_10_1_IX),
        "spread_planned": state.spread_plan is not None, "spread_measured": state.spread is not None,
        "packs": sorted(state.packs),
        "phase": "final" if state.search_closed is not None else "search",
        "search_closed": state.search_closed is not None, "revealed": state.reveal is not None,
        "trials": {"started": len(state.trials_started), "finished": len(state.trials), "pending": pending},
        "candidates": [{"id": c, "lever": rec.lever.value, "lint_passed": rec.lint_passed, "outcome": _candidate_bit(state, c)} for c, rec in sorted(state.candidates.items())],
        "operator_changes": len(state.operator_changes),
        "cost_usd": {"trials_dollars": round(math.fsum(u.dollars for u in spent), 6), "outer_proposals": round(math.fsum(c.propose_usd for c in state.candidates.values()), 6)},
        "malformed_events": len(state.malformed),
        "head": None if state.head is None else f"{state.head.seq}:{state.head.hash16}",
    }
    if state.rule is not None or state.candidates:
        out["incumbent"] = env.reader.incumbent_version(state)
    if state.final_analysis is not None:
        out["final"] = {"requests": len(state.finals), "revealed": state.reveal is not None}
    if pending:
        out["note"] = "trials without lab.trial_finished: finish them or run abort-trial"
    env.emit(out)


# ---- verify --------------------------------------------------------------------------------------
def _check(checks: list, name: str, ok: bool, detail: str = "") -> bool:
    checks.append({"name": name, "ok": bool(ok), "detail": "" if ok else detail})
    return bool(ok)


def _hash_at(events: Sequence[Any], seq: int) -> Optional[str]:
    for e in events:
        if e.seq == seq:
            return e.hash
    return None


def cmd_verify(env: _Env, a: argparse.Namespace) -> None:
    from nightshift.ledger import Ledger
    from nightshift.lab.ledger import LabLedger

    layout, checks = env.layout, []
    try:
        ledger = env.ledger
        env.state()
    except LabError:
        raise
    except Exception as exc:  # noqa: BLE001 - a corrupt ledger is the finding, not a crash
        env.emit({"ok": False, "checks": [{"name": "master ledger opens and folds", "ok": False, "detail": f"{type(exc).__name__}: {exc}"[:300]}]})
        raise _Failed("verify found a problem") from None
    ok, bad = ledger.verify()
    _check(checks, "master ledger hash chain", ok, "" if ok else f"first bad seq {bad}")
    state = env.state()
    events = ledger.events()
    # trial ledgers
    trials_dir = layout.root / "trials"
    present = {p.parent.name for p in trials_dir.glob("*/ledger.jsonl")} if trials_dir.is_dir() else set()
    # a trial the master ledger anchors must still have its ledger: deleting the file must not skip the anchor comparison
    for tid in sorted(present | {t for t, r in state.trials.items() if r.trial_ledger_anchor is not None}):
        if tid not in present:
            _check(checks, f"trial ledger {tid}", False, "lab.trial_finished anchors this trial ledger, but trials/<id>/ledger.jsonl is missing")
            continue
        try:
            tl = LabLedger.read_only(trials_dir / tid)
            tok, tbad = tl.verify()
        except Exception as exc:  # noqa: BLE001
            _check(checks, f"trial ledger {tid}", False, f"{type(exc).__name__}: {exc}"[:200])
            continue
        _check(checks, f"trial ledger {tid}", tok, "" if tok else f"first bad seq {tbad}")
        rec = state.trials.get(tid)
        if rec is not None and rec.trial_ledger_anchor is not None:
            h = _hash_at(tl.events(), rec.trial_ledger_anchor.seq)
            _check(checks, f"trial {tid} anchor", h is not None and h[:16] == rec.trial_ledger_anchor.hash16, "anchor in lab.trial_finished does not match the trial ledger")
    if (layout.outer_dir / "ledger.jsonl").exists():
        try:
            ook, obad = Ledger.read_only(layout.outer_dir).verify_chain()
            _check(checks, "outer ledger hash chain", ook, f"first bad seq {obad}")
        except Exception as exc:  # noqa: BLE001
            _check(checks, "outer ledger hash chain", False, f"{type(exc).__name__}: {exc}"[:200])
    # frozen files
    rule = manifest = budget = analysis = None
    try:
        # Each frozen file is re-hashed on its own: a missing sibling must not let an altered or deleted file pass unchecked.
        if state.rule is not None and state.manifest is not None and state.budget is not None:
            rule, manifest, budget = env.reader.load_frozen_all(state)
        else:
            if state.rule is not None:
                rule = load_frozen(layout, state.rule, Rule)
            if state.manifest is not None:
                manifest = load_frozen(layout, state.manifest, AssignmentManifest)
            if state.budget is not None:
                budget = load_frozen(layout, state.budget, BudgetSpec)
        if state.final_analysis is not None:
            analysis = load_frozen(layout, state.final_analysis, FinalAnalysis)
        _check(checks, "frozen files match the ledger", True)
    except (ContractError, OSError) as exc:  # a deleted frozen file is the finding, not a crash
        _check(checks, "frozen files match the ledger", False, str(exc))
    first_cand = min((state.event_seq.get((LabKind.CANDIDATE_PROPOSED, c), 1 << 60) for c in state.candidates), default=None)
    if first_cand is not None:
        rseq = state.event_seq.get((LabKind.RULE_FROZEN, ""))
        _check(checks, "rule frozen before the first candidate", rseq is not None and rseq < first_cand)
    # decisions recomputed from the ledger's scores and the rule
    if rule is not None:
        for cid, d in sorted(state.decisions.items()):
            _check(checks, f"decision {cid}", *_recompute_decision(state, rule, d, env.reader))
    # Only now: re-reading the decisions' scores through the fold can flag score records that differ from the rule, and a malformed
    # finding made there must fail verify like any other.
    _check(checks, "no malformed events", not state.malformed, "; ".join(f"seq {s}: {r}" for s, r in state.malformed[:5]))
    # anchors
    if layout.anchors_path.exists():
        bad_lines = []
        for line in layout.anchors_path.read_text().splitlines():
            head = line.split(" ", 1)[0]
            try:
                seq_s, h16 = head.split(":")
                h = _hash_at(events, int(seq_s))
            except ValueError:
                bad_lines.append(line[:60])
                continue
            if h is None or h[:16] != h16:
                bad_lines.append(line[:60])
        _check(checks, "anchors match the ledger", not bad_lines, "; ".join(bad_lines[:3]))
    # final plan
    if analysis is not None:
        wrong = [f.request_id for f in state.finals if f.analysis_sha256 != analysis.sha256()]
        _check(checks, "final requests name the frozen analysis", not wrong, ", ".join(wrong))
    allok = all(c["ok"] for c in checks)
    env.emit({"ok": allok, "head": None if state.head is None else f"{state.head.seq}:{state.head.hash16}", "checks": checks})
    if not allok:
        raise _Failed("verify found a problem")


class _Failed(LabError):
    """The command ran and reported a problem (exit code 1 with the output already printed)."""


def _recompute_decision(state: LabState, rule: Rule, d: Any, reader: Any = None) -> tuple:
    problems = []
    if d.rule_sha256 != rule.sha256():
        problems.append("rule_sha256 differs from the frozen rule")
    zeros: dict = {}  # pack_version -> trial ids the fold counts as 0.0 under SCORED_ZERO

    def zero_counted(ts: TrialScore) -> bool:
        """A TrialScore without a score id is legitimate only for a finished trial the fold itself counts as a failed 0.0."""
        spec = state.trials_started.get(ts.trial_id)
        if (rule.failed_trial_policy is not FailedTrialPolicy.SCORED_ZERO or reader is None or spec is None or ts.trial_id not in state.trials
                or ts.score != 0.0 or ts.assignment_id != spec.assignment_id):
            return False
        if spec.pack_version not in zeros:
            zeros[spec.pack_version] = {t.trial_id for t in reader.selection_trial_scores(state, rule, spec.pack_version) if t.score_id is None}
        return ts.trial_id in zeros[spec.pack_version] and comparability_key(spec, rule.metric) == ts.comparability_sha256

    eligible: dict = {}  # pack_version -> {trial_id: score_id} as the fold counts them

    def selection_evidence(ts: TrialScore, rec: Any, spec: Any) -> bool:
        """A scored entry is evidence only if its score is the selection score of a finished search trial that the fold itself counts."""
        fin = state.trials.get(ts.trial_id)
        if fin is None or fin.selection_score_id != ts.score_id or rec.trial_id != ts.trial_id:
            return False
        if rec.split is not Split.SELECTION or rec.phase is not Phase.SEARCH or spec.phase is not Phase.SEARCH:
            return False
        if reader is None:
            hit = state.contamination.get(ts.trial_id)
            return fin.status in (TrialStatus.COMPLETED, TrialStatus.BUDGET_EXHAUSTED) and (hit is None or hit.passed)
        if spec.pack_version not in eligible:
            eligible[spec.pack_version] = {t.trial_id: t.score_id for t in reader.selection_trial_scores(state, rule, spec.pack_version)}
        return eligible[spec.pack_version].get(ts.trial_id) == ts.score_id

    def check_scores(scores: Sequence[TrialScore], pack_version: str, whose: str) -> None:
        for ts in scores:
            owner = state.trials_started.get(ts.trial_id)
            if owner is not None and owner.pack_version != pack_version:
                problems.append(f"trial {ts.trial_id} is listed as a {whose} score but ran pack {owner.pack_version}, not {pack_version}")
            if ts.score_id is None:
                if not zero_counted(ts):
                    problems.append(f"trial {ts.trial_id} is scored 0.0 without a score id, but the ledger does not show it as a failed trial under scored_zero")
                continue
            rec = state.scores.get(ts.score_id)
            spec = state.trials_started.get(ts.trial_id)
            if rec is None or spec is None:
                problems.append(f"score {ts.score_id} or trial {ts.trial_id} is not in the ledger")
                continue
            if not selection_evidence(ts, rec, spec):
                problems.append(f"trial {ts.trial_id} is not a finished, eligible search trial whose selection score is {ts.score_id}")
                continue
            if not math.isclose(_trial_mean(state, ts.score_id, rule.metric), ts.score, rel_tol=0, abs_tol=1e-9):
                problems.append(f"score of {ts.trial_id} differs from the ledger")
            if comparability_key(spec, rule.metric) != ts.comparability_sha256:
                problems.append(f"comparability key of {ts.trial_id} differs from its spec")

    check_scores(d.incumbent_scores, d.incumbent_version, "incumbent")
    check_scores(d.candidate_scores, d.candidate_version, "candidate")
    if d.stage is DecisionStage.CONFIRM and d.outcome.value != "invalid":
        nums = stats.decide(rule, d.incumbent_scores, d.candidate_scores)
        for name, got, want in (("outcome", d.outcome, nums.outcome), ("delta", d.delta, nums.delta), ("margin", d.margin, nums.margin),
                                ("df", d.df, nums.df), ("sigma_used", d.sigma_used, nums.sigma_used)):
            same = got == want or (isinstance(got, float) and isinstance(want, float) and math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12))
            if not same:
                problems.append(f"{name}: ledger {got!r}, recomputed {want!r}")
    elif d.stage is DecisionStage.SCREEN and d.screen_delta is not None and d.candidate_scores:
        got = stats.screen(rule, d.incumbent_scores, {d.candidate_id: d.candidate_scores}).get(d.candidate_id)
        if got is None or not math.isclose(got, d.screen_delta, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"screen_delta: ledger {d.screen_delta!r}, recomputed {got!r}")
    return (not problems, "; ".join(problems))


# ---- verify-isolation ----------------------------------------------------------------------------
def cmd_verify_isolation(env: _Env, a: argparse.Namespace) -> None:
    """Checks that can be run anywhere (the full list runs as a trial user on the researcher host: UNVERIFIED until then). Link and token
    checks are not optional."""
    from nightshift.lab.contracts import file_ref as fref

    checks: list = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        canary = root / "private" / "canary.txt"
        canary.parent.mkdir()
        canary.write_text("CANARY-SELECTION-ROW")
        harvest = root / "harvest"
        harvest.mkdir()
        os.symlink(canary, harvest / "link.txt")
        os.link(canary, harvest / "hard.txt")
        (harvest / "plain.txt").write_text("fine")
        for name in ("link.txt", "hard.txt"):
            try:
                read_regular_bytes(root, f"harvest/{name}", max_bytes=1024)
                _check(checks, f"read_regular_bytes refuses {name}", False, "a planted link was read")
            except ContractError:
                _check(checks, f"read_regular_bytes refuses {name}", True)
            try:
                fref(root, harvest / name)
                _check(checks, f"file_ref refuses {name}", False, "a planted link was referenced")
            except ContractError:
                _check(checks, f"file_ref refuses {name}", True)
        _check(checks, "a plain file is accepted", read_regular_bytes(root, "harvest/plain.txt", max_bytes=1024) == b"fine")
    cfg = env.config
    names = list(cfg.secret_env.values()) if cfg is not None else []
    leaked = [n for n in names if os.environ.get(n)]
    if a.researcher_host:
        _check(checks, "no lab secret in this environment", not leaked, ", ".join(leaked))
        thunder = Path.home() / ".thunder"
        _check(checks, "no Thunder token file readable", not (thunder.exists() and os.access(thunder, os.R_OK)), str(thunder))
        _check(checks, "lab root is not readable from here", not (env.layout.root.exists() and os.access(env.layout.private_dir, os.R_OK)),
               "the control host's private/ directory is readable from this account")
        if cfg is not None and not a.skip_ssh:
            try:
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", cfg.control_host, "true"], capture_output=True, timeout=15)
                _check(checks, "ssh to the control host is refused", r.returncode != 0, "ssh to the control host succeeded")
            except (OSError, subprocess.TimeoutExpired):
                _check(checks, "ssh to the control host is refused", True)
    ok = all(c["ok"] for c in checks)
    env.emit({"ok": ok, "mode": "researcher-host" if a.researcher_host else "link-checks-only", "checks": checks,
              "note": "run with --researcher-host as the trial user on the researcher host for the token, lab-root and ssh checks"})
    if not ok:
        raise _Failed("verify-isolation found a problem")


# ---- rentals -------------------------------------------------------------------------------------
def cmd_rentals(env: _Env, a: argparse.Namespace) -> None:
    mgr = getattr(env.world, "rentals", None)
    if mgr is None:
        raise ContractError("this world has no rental manager")
    if a.action == "init":
        if a.hard_stop_usd is None or not a.why:
            raise UsageError("rentals init needs --hard-stop-usd and --why")
        mgr.init(a.hard_stop_usd, env.need_config().control_host if env.config else "control", a.why)
        env.emit({"hard_stop_usd": a.hard_stop_usd})
    elif a.action == "cost":
        env.emit(mgr.cost())
    elif a.action == "show":
        env.emit(mgr.read())
    elif a.action == "delete":
        if not a.session or not a.why:
            raise UsageError("rentals delete needs --session and --why")
        mgr.delete(a.session, a.why)
        env.emit({"deleted": a.session})
    else:  # pragma: no cover - argparse restricts the choices
        raise UsageError(a.action)


def cmd_anchor(env: _Env, a: argparse.Namespace) -> None:
    head = env.anchor(a.label)
    out: dict = {"anchor": head, "file": str(env.layout.anchors_path.relative_to(env.layout.root))}
    if a.push:
        out.update(_publish_anchors(env.layout.anchors_path, head))
    env.emit(out)


COMMAND_FUNCS: Mapping[str, Callable[[_Env, argparse.Namespace], None]] = {
    "init": cmd_init, "freeze-manifest": cmd_freeze_manifest, "freeze-budget": cmd_freeze_budget, "record-gate": cmd_record_gate,
    "commit-pack": cmd_commit_pack, "plan-spread": cmd_plan_spread, "trial": cmd_trial, "spread": cmd_spread, "freeze-rule": cmd_freeze_rule,
    "outer": cmd_outer, "close-search": cmd_close_search, "freeze-final-analysis": cmd_freeze_final_analysis, "final": cmd_final,
    "reveal": cmd_reveal, "score": cmd_score, "abort-trial": cmd_abort_trial, "operator-change": cmd_operator_change, "report": cmd_report,
    "status": cmd_status, "verify": cmd_verify, "verify-isolation": cmd_verify_isolation, "rentals": cmd_rentals, "anchor": cmd_anchor,
}


# --------------------------------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------------------------------
class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:  # argparse would call sys.exit
        raise UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -m nightshift.lab", description="Glimmer three-arm experiment: lab commands")
    p.add_argument("lab_root", help="the lab root directory on the control host")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help)

    s = add("init", "create the lab: salt, directories, master ledger")
    s.add_argument("--config", help="a LabConfig JSON file to install as lab.json")
    s.add_argument("--host", help="host name recorded in lab.initialized")
    s = add("freeze-manifest", "split the data, write the assignment manifest, freeze it")
    s.add_argument("--rows", required=True, help="source rows JSONL")
    s.add_argument("--test-file", action="append", required=True, help="IFBench test JSONL (the first one becomes private/test/IFBench_test.jsonl)")
    s.add_argument("--plan", action="append", required=True, help="ID:ROLE:FAMILY,FAMILY:N_SELECTION:N_DEV:N_TRAIN (repeatable)")
    s.add_argument("--evolution-families", required=True)
    s.add_argument("--heldout-families", required=True)
    s.add_argument("--source-sha256", help="sha256 of the source dataset file (default: of --rows)")
    s = add("freeze-budget", "freeze the per-trial budget")
    s.add_argument("--researcher-tokens", type=int, required=True)
    s.add_argument("--gpu-minutes", type=float, required=True)
    s.add_argument("--dollars", type=float, required=True)
    s.add_argument("--wall-seconds", type=float, required=True)
    s = add("record-gate", "record the terms gate outcome (clarified closes it; proceed_with_risk is a waiver and leaves it open)")
    s.add_argument("--gate", default=Gate.TERMS_10_1_IX.value, choices=[g.value for g in Gate])
    s.add_argument("--outcome", required=True, choices=[g.value for g in GateOutcome])
    s.add_argument("--statement", required=True)
    s.add_argument("--decided-by", required=True)
    s = add("commit-pack", "commit the v0 pack")
    s.add_argument("--template", help="pack template directory (default: assets/pack_v0)")
    s = add("plan-spread", "declare how many v0 trials measure the spread")
    s.add_argument("--trials", type=int, required=True)
    s.add_argument("--assignments", help="comma list (default: every evolution assignment)")
    s.add_argument("--metric", default="prompt_all")
    s = add("trial", "run one search-phase trial of a committed pack")
    s.add_argument("--pack-version", default="v0")
    s.add_argument("--assignment")
    s.add_argument("--seed", type=int)
    s.add_argument("--trial-id")
    s.add_argument("--score-seeds", help="comma list; before the rule exists (default 0,1,2)")
    s.add_argument("--max-tokens", type=int, default=8000, help="scoring max_tokens before the rule exists")
    s = add("spread", "measure the v0 spread from the planned trials")
    s.add_argument("--failed-as-zero", action="store_true",
                   help="count a planned v0 trial that produced no scored adapter as 0.0 (FailedTrialPolicy.SCORED_ZERO; freeze-rule then requires --failed-trial-policy scored_zero)")
    s = add("freeze-rule", "freeze the keep-or-reject rule")
    s.add_argument("--alpha", type=float, help="per-round one-sided alpha (default: the Sidak split of --family-alpha over the rounds)")
    s.add_argument("--family-alpha", type=float, default=0.05)
    s.add_argument("--min-gain", type=float, default=0.0)
    s.add_argument("--screen-batch-size", type=int, default=3)
    s.add_argument("--screen-trials", type=int, default=2)
    s.add_argument("--confirm-trials", type=int, default=3)
    s.add_argument("--confirm-trials-incumbent", type=int, default=3)
    s.add_argument("--variance-guard-ratio", type=float, default=1.5)
    s.add_argument("--failed-trial-policy", default=FailedTrialPolicy.INVALIDATES_CANDIDATE.value, choices=[f.value for f in FailedTrialPolicy])
    s.add_argument("--assumed-gain", type=float, required=True)
    s.add_argument("--max-candidates", type=int, required=True)
    s.add_argument("--stop-after-rejections", type=int)
    s.add_argument("--degenerate-override")
    s.add_argument("--assignments")
    s = add("outer", "run the outer loop (screen, confirm, decide)")
    s.add_argument("--rounds", type=int, default=1)
    s.add_argument("--wonder-usd", type=float, default=1.0, help="dollar cap of one proposal call; the Wonder budget gate allows this times rule.max_candidates in all")
    s.add_argument("--trial-llm-usd", type=float, help="researcher-inference dollars of one trial (worst case); default: the whole trial budget")
    s.add_argument("--scoring-usd-per-trial", type=float, default=0.0)
    s = add("close-search", "end the search")
    s.add_argument("--reason", required=True)
    s.add_argument("--final-pack", help="exploratory final pack not accepted by the keep rule; needs a prior operator-change --target final_pack naming it")
    s = add("freeze-final-analysis", "freeze the confirmatory plan")
    s.add_argument("--trials-per-pack", type=int, required=True, help="per held-out assignment")
    s.add_argument("--exploratory", action="store_true", help="owner-approved small comparison below the confirmatory floor: >= 2 trials per pack, held-out selection only; needs a prior operator-change --target final_analysis_exploratory")
    s.add_argument("--assignments", help="comma list of held-out ids, optionally id:weight")
    s.add_argument("--confidence", type=float, default=0.95)
    s.add_argument("--heldout-metric", default="prompt_all")
    s.add_argument("--test-metric", default="prompt_strict")
    s.add_argument("--seed-base", type=int, default=1000)
    s = add("final", "request and run the final comparison of one arm")
    s.add_argument("--arm", required=True, choices=[x.value for x in Arm])
    s.add_argument("--repeat-of")
    s.add_argument("--reason")
    s.add_argument("--resume", help="a final request id: run its planned trials that have not started")
    s = add("reveal", "reveal the final scores once the whole plan is scored")
    s = add("score", "diagnostic score of an existing adapter, or of the stock model with --stock (same guard as every score)")
    s.add_argument("--adapter")
    s.add_argument("--stock", action="store_true", help="score the unmodified base model instead of an adapter")
    s.add_argument("--split", choices=["dev", "selection", "test"], default="dev")
    s.add_argument("--assignment", default=None)
    s.add_argument("--trial")
    s.add_argument("--reason")
    s.add_argument("--final-request")
    s.add_argument("--score-seeds")
    s.add_argument("--max-tokens", type=int, default=8000)
    s = add("abort-trial", "record ABORTED for a trial that never finished")
    s.add_argument("trial_id")
    s = add("operator-change", "record an operator change after the first candidate")
    s.add_argument("--target", required=True)
    s.add_argument("--detail", required=True)
    s = add("report", "build the report (bits and costs during the search)")
    s.add_argument("--full", action="store_true", help="the full view; refused before lab.search_closed")
    s.add_argument("--format", choices=["md", "html"], default="md")
    s.add_argument("--out", help="write under reports/ instead of printing")
    s = add("status", "bits and costs")
    s = add("verify", "check the ledgers, frozen files, decisions and anchors")
    s = add("verify-isolation", "link and token checks")
    s.add_argument("--researcher-host", action="store_true", help="also run the token, lab-root and ssh checks (run on the researcher host)")
    s.add_argument("--skip-ssh", action="store_true")
    s = add("rentals", "rental manager")
    s.add_argument("action", choices=["init", "cost", "show", "delete"])
    s.add_argument("--hard-stop-usd", type=float)
    s.add_argument("--why")
    s.add_argument("--session")
    s = add("anchor", "append the master head to anchors/master.anchors")
    s.add_argument("--label", default="manual")
    s.add_argument("--push", action="store_true", help="commit and push the anchors file with git")
    return p


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
def main(argv: Sequence[str], *, world: Optional[LabWorld] = None, out: Optional[TextIO] = None, err: Optional[TextIO] = None) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    try:
        args = build_parser().parse_args(list(argv))
    except UsageError as exc:
        err.write(f"usage error: {exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:  # --help
        return int(exc.code or 0)
    layout = LabLayout(Path(args.lab_root).expanduser())
    if world is not None:
        try:
            same = layout.root.resolve() == world.layout.root.resolve()
        except OSError:
            same = False
        if not same:
            err.write("usage error: lab_root differs from the root of the given world\n")
            return EXIT_USAGE
        layout = world.layout
    env = _Env(layout, world, out, err, args.command)
    try:
        COMMAND_FUNCS[args.command](env, args)
        return EXIT_OK
    except _Failed as exc:
        err.write(f"{exc}\n")
        return EXIT_REFUSED
    except UsageError as exc:
        err.write(f"usage error: {exc}\n")
        return EXIT_USAGE
    except LabError as exc:
        err.write(f"refused: {type(exc).__name__}: {exc}\n")
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001 - a CLI ends with a message and a code, not a traceback, unless asked
        if os.environ.get("LAB_DEBUG"):
            raise
        err.write(f"failed: {type(exc).__name__}: {exc}\n")
        return EXIT_FAILED
    finally:
        env.close()


# --------------------------------------------------------------------------------------------------
# production composition root
# --------------------------------------------------------------------------------------------------
class _PortAllocator:
    """A free loopback port, never handed out twice by this process."""

    def __init__(self) -> None:
        self._issued: set = set()
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            for _ in range(50):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    port = s.getsockname()[1]
                if port not in self._issued:
                    self._issued.add(port)
                    return port
        raise LabError("no free port")


class _LazyProposalSource:
    """Muse Wonder through the reused ``MetaChatClient`` + ``MeteredClient`` + ``BudgetGate``, built on first use (the Meta key is read from
    the environment variable named in ``LabConfig.secret_env`` then)."""

    def __init__(self, layout: LabLayout, config: LabConfig, clock: Callable[[], float], model: str = "muse-wonder-1.0", max_usd: float = 5.0):
        self._layout, self._config, self._clock, self._model, self._max_usd = layout, config, clock, model, max_usd
        self._source: Any = None
        self._ledger: Any = None
        self._lock = threading.Lock()

    def set_call_cap(self, per_call_usd: float, max_calls: int) -> None:
        """Set the dollars the Wonder ``BudgetGate`` allows in all: ``per_call_usd`` for each of ``max_calls`` proposals (the ``outer
        --wonder-usd`` figure the controller budgets with). The gate sums every Wonder charge in ``outer/ledger.jsonl`` and refuses the next
        call once the total reaches this, so it must be set before the first call builds the gate."""
        with self._lock:
            if self._source is not None:
                raise LabError("the Wonder budget gate is built already; set its cap before the first proposal")
            self._max_usd = float(per_call_usd) * int(max_calls)

    def propose(self, context: Any) -> Any:
        with self._lock:
            if self._source is None:
                from nightshift.budget import BudgetGate, MeteredClient
                from nightshift.ledger import Ledger
                from nightshift.models.meta import MetaChatClient
                from nightshift.types import BudgetLimits
                from nightshift.lab.proposer import WonderProposalSource

                key = os.environ.get(self._config.secret_env["meta_api_key"])
                if not key:
                    raise LabError(f"the Meta key is not in ${self._config.secret_env['meta_api_key']}")
                d = self._layout.outer_dir
                d.mkdir(parents=True, exist_ok=True)
                self._ledger = Ledger.open(d, clock=self._clock) if (d / "ledger.jsonl").exists() else Ledger.create(d, clock=self._clock)
                client = MetaChatClient(self._model, base_url=self._config.upstream_base_url, api_key=key, fallback_model=None)
                gate = BudgetGate(BudgetLimits(max_llm_usd=self._max_usd), self._ledger, clock=self._clock)
                self._source = WonderProposalSource(MeteredClient(client, gate, "wonder"))
            source = self._source
        return source.propose(context)

    def close(self) -> None:
        if self._ledger is not None:
            self._ledger.close()


def researcher_model(environ: Optional[Mapping[str, str]] = None) -> str:
    """The Muse Code researcher model: ``LAB_RESEARCHER_MODEL`` (e.g. ``muse-wonder-full-1.0``), else the runtime default
    (``muse-spark-1.3``, unchanged for existing lab roots). The runtime validates it and checks what Muse reports serving."""
    from nightshift.lab.runtime_coral_cli import DEFAULT_MODEL

    env = os.environ if environ is None else environ
    return env.get("LAB_RESEARCHER_MODEL") or DEFAULT_MODEL


def outer_challenge_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """The outer challenge runs with the Muse Code proposer (``LAB_PROPOSER=muse_cli``) unless ``LAB_OUTER_CHALLENGE=off``. The API
    proposer path is unchanged."""
    env = os.environ if environ is None else environ
    return env.get("LAB_PROPOSER", "api") == "muse_cli" and env.get("LAB_OUTER_CHALLENGE", "on") != "off"


def build_world(layout: LabLayout, config: LabConfig, *, ledger: Any = None) -> CliWorld:
    """Build the real objects from ``lab.json``. Constructors only: nothing here touches the network, a GPU or a third-party checkout
    (verifiers, the Meta client and the scorer are built on first use)."""
    from nightshift.lab import contamination as contamination_mod
    from nightshift.lab.budget import LabCap, TrialBudgetMeter, lab_spent_usd
    from nightshift.lab.fold import FoldReader
    from nightshift.lab.generator import HttpGenerator
    from nightshift.lab.gpu_rentals import RentalManager
    from nightshift.lab.gpu_thunder import ThunderProvider
    from nightshift.lab.ledger import LabLedger
    from nightshift.lab.model_server import RentalModelServer
    from nightshift.lab.packs import PackRepo
    from nightshift.lab.runtime_coral import CoralRuntime
    from nightshift.lab.scorer import ScoreService
    from nightshift.lab.trainer import GlimmerTrainer
    from nightshift.lab.trial import TrialRunner

    clock = time.time
    if ledger is None:
        ledger = LabLedger.open(layout.master_ledger_dir, clock=clock)
    reader = FoldReader(layout, ledger=ledger)
    lab_cap = LabCap(config.lab_budget_usd, lab_spent_usd(reader.state(), layout), clock=clock)
    hostkit = Path(__file__).resolve().parent / "hostkit"
    ports = _PortAllocator()

    def new_trial_ledger(trial_id: str) -> Any:
        return LabLedger.create(layout.trial_ledger_dir(trial_id), clock=clock)

    def meter_factory(spec: TrialSpec, tledger: Any) -> Any:
        return TrialBudgetMeter(spec, tledger, spec.trial_id, clock=clock, lab_cap=lab_cap)

    def generator_factory(endpoint: ServeEndpoint) -> Any:
        # Model evaluation runs as an Inspect AI task (versioned task, log per seed, log hash in the score record). The plain HTTP loop is a
        # named, recorded choice (the record says ``generator: http``), never a silent fallback for a missing library.
        if os.environ.get("NIGHTSHIFT_LAB_GENERATOR", "inspect") == "http":
            return HttpGenerator(endpoint.base_url)
        try:
            from nightshift.lab.inspect_generator import InspectGenerator
        except ImportError as exc:  # pragma: no cover - inspect_ai itself is imported when the first score runs
            raise ScorerError(f"cannot load the Inspect generator: {exc}") from None
        try:
            import inspect_ai  # noqa: F401
        except ImportError:
            raise ScorerError("inspect_ai is not installed (pip install '.[lab-eval]'); set NIGHTSHIFT_LAB_GENERATOR=http to use the plain HTTP loop, which the score record will state") from None
        return InspectGenerator(endpoint.base_url, log_dir=layout.private_dir / "inspect_logs", workers=int(os.environ.get("NIGHTSHIFT_LAB_GEN_WORKERS", "128")))

    def make_scorer() -> Any:
        from nightshift.lab.verifiers import IfbenchUpstream, IfEvalG

        nltk = os.environ.get("NLTK_DATA")
        state = reader.state()
        manifest = load_frozen(layout, state.manifest, AssignmentManifest)
        verifiers = {
            Verifier.IFEVALG: _LazyBackend(lambda: IfEvalG(layout.third_party_dir / "open-instruct", nltk_data=nltk)),
            Verifier.IFBENCH_UPSTREAM: _LazyBackend(lambda: IfbenchUpstream(layout.third_party_dir / "IFBench", nltk_data=nltk)),
        }
        return ScoreService(layout, manifest, verifiers, guard=make_score_guard(layout, reader), clock=clock)

    scorer = LazyScorer(make_scorer)

    provider = ThunderProvider(config.secret_env["thunder_api_key"])
    rentals = RentalManager(layout.rental_ledger, provider, clock=clock, machine=config.control_host,
                            watchdog_provider_args=["--provider", "thunder", "--token-env", config.secret_env["thunder_api_key"]])
    # A Thunder instance gives a non-root user (no sudo), so /opt/lab, /work and /models cannot be created: everything lives under the user's
    # home (LIVE 2026-11-04: the first pilot job failed with "mkdir: cannot create directory '/opt/lab': Permission denied"). The data directory
    # is exported to recipes as GLIMMER_DATA_DIR, which the pack's task text names.
    from nightshift.lab.model_server import RemoteLayout

    rhome = os.environ.get("LAB_RENTAL_HOME", "/home/ubuntu/lab")
    remote_layout = RemoteLayout(f"{rhome}/opt", f"{rhome}/work", f"{rhome}/models", (("GLIMMER_DATA_DIR", f"{rhome}/work/data"),))
    trainer = GlimmerTrainer(rentals, config.shape, scorer, generator_factory, layout, hostkit, config.base, clock=clock, remote_layout=remote_layout)
    model_server = RentalModelServer(rentals, config.shape, hostkit, layout, clock=clock, remote_layout=remote_layout)
    coral_home = os.environ.get("LAB_CORAL_HOME", "/srv/lab/coral")
    if os.environ.get("LAB_RESEARCHER_RUNTIME", "coral_opencode") == "coral_muse":  # CLI-first proof: Muse Code agents in a container, subscription login, no API key
        from nightshift.lab.runtime_coral_cli import CoralMuseRuntime

        runtime = CoralMuseRuntime(
            layout, f"{config.researcher_user}@{config.researcher_host}", coral_home, config.pins, ports, clock=clock,
            muse_home=os.environ.get("LAB_MUSE_HOME", "/home/ankit/work/glimmer-lab/muse-home"),
            model=researcher_model(),
            remote_trials_root=os.environ.get("LAB_TRIALS_ROOT", "/srv/lab/trials"),
        )
    else:
        runtime = CoralRuntime(layout, f"{config.researcher_user}@{config.researcher_host}", coral_home, config.pins, ports, clock=clock)
    packs = PackRepo(layout, lint_ranges_path=layout.root / "lint_ranges.json")
    if os.environ.get("LAB_PROPOSER", "api") == "muse_cli":  # CLI-first proof: Muse Code with muse-wonder-full-1.0 on the subscription, no API key
        from nightshift.lab.proposer_muse_cli import MuseCliProposer

        proposal = MuseCliProposer(effort=os.environ.get("LAB_PROPOSER_EFFORT", "high"), precheck_allowance=True)
        proposal.close = lambda: None  # type: ignore[attr-defined]
    else:
        proposal = _LazyProposalSource(layout, config, clock)
    challenger = None
    if outer_challenge_enabled():  # the bounded challenge: rendered-config checker, one Wonder critic call, at most one revision
        from nightshift.lab.outer_challenge import CRITIC_MODEL, Challenger
        from nightshift.lab.proposer_muse_cli import MuseCliProposer

        challenger = Challenger(MuseCliProposer(model=CRITIC_MODEL, effort=os.environ.get("LAB_PROPOSER_EFFORT", "high"), precheck_allowance=True))

    scanner_box: dict = {}

    def contamination_scan(**kw: Any) -> Any:
        if "scanner" not in scanner_box:
            manifest = load_frozen(layout, reader.state().manifest, AssignmentManifest)
            scanner_box["manifest"] = manifest
            scanner_box["scanner"] = contamination_mod.build_scanner(layout, manifest)
        return contamination_mod.scan_job(layout, scanner_box["manifest"], clock=clock, scanner=scanner_box["scanner"], **kw)

    runner = TrialRunner(
        layout, ledger, new_trial_ledger, meter_factory, runtime, trainer, scorer, model_server, generator_factory, packs, reader,
        clock=clock, port_allocator=ports, upstream_base_url=config.upstream_base_url,
        api_key=(lambda: "unused-cli-route") if os.environ.get("LAB_RESEARCHER_RUNTIME") == "coral_muse" else (lambda: os.environ[config.secret_env["meta_api_key"]]),
        contamination_scan=contamination_scan,
    )
    return CliWorld(
        layout=layout, config=config, clock=clock, ledger=ledger, state_reader=reader, new_trial_ledger=new_trial_ledger,
        meter_factory=meter_factory, scorer=scorer, generator_factory=generator_factory, trainer=trainer, model_server=model_server,
        runtime=runtime, packs=packs, decision_rule=stats, trial_executor=runner, proposal_source=proposal, lab_cap=lab_cap, rentals=rentals,
        close=proposal.close, challenger=challenger,
    )


if __name__ == "__main__":  # pragma: no cover - python -m nightshift.lab is the entry point
    sys.exit(main(sys.argv[1:]))
