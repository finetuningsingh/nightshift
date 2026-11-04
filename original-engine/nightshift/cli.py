"""Command-line interface (docs/ARCHITECTURE.md section 6.9).

    nightshift run | resume | serve | report | audit | verify | doctor | demo | tasks

Exit codes: 0 success; 1 the run finished with open block findings, the budget ran out without an incumbent, or a check
failed; 2 usage error (bad arguments, missing directory or task, an invalid ``NIGHTSHIFT_ISOLATION``, bundled tasks not found).

Environment: ``NIGHTSHIFT_TASKS_DIR`` names the directory that holds the bundled tasks (default: ``tasks/`` of the checkout the
package was started from); ``NIGHTSHIFT_ISOLATION`` lowers the experiment isolation (``nightshift.sandbox``) and is validated
before ``run``, ``resume``, ``demo`` and ``doctor`` do anything else.

Modules written elsewhere (orchestrator, task loader, sandbox, auditor, models) are reached only through ``_api()`` so
tests can replace that one seam. A missing module is a clear error for the commands that need it, never an import-time
crash. The API key is registered with ``nightshift.secrets`` and is never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from . import secrets
from .agents.scripted import PLANTABLE, planted_proposals
from .dashboard.server import ReplayRefused, make_server
from .dashboard.state_api import format_anchor
from .ledger import LEDGER_NAME, Ledger, LedgerLocked
from .report import write_reports
from .fence import sanitize
from .types import BudgetLimits, LedgerCorrupt, LedgerView, NightshiftError, SimulatedCrash
from .view import build_view

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
KEY_ENVS = ("MODEL_API_KEY", "META_API_KEY")
DEMO_CRASH_AFTER_RUNS = 4        # baseline = 3 seed runs, so the 4th run dies inside the first candidate
DEFAULT_REPLAY_SPEED = 10.0
TASKS_ENV = "NIGHTSHIFT_TASKS_DIR"
TASKS_HINT = ("bundled tasks not found next to the nightshift package; run from a clone of the repository, install with "
              "pip install -e ., or pass --task /path/to/task")
SKIP_PREFIX = "missing_requirements:"      # a task problem that only says an optional dependency is absent: a skip, not a failure
CONTEXT_FILE_MAX_CHARS = 200_000     # --context: one file
CONTEXT_TOTAL_MAX_CHARS = 400_000    # --context: all files together
_BINARY_CHARS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")


class CliError(Exception):
    """A user-facing failure: the message is printed, ``code`` is the exit status."""

    def __init__(self, message: str, code: int = EXIT_FAIL):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------------------------------
# the seam
# --------------------------------------------------------------------------------------------------
def _opt(module: str, name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None


def _api() -> SimpleNamespace:
    """Everything the CLI needs from modules other engineers own. A missing piece is ``None``. Tests monkeypatch this."""
    return SimpleNamespace(
        load_task=_opt("nightshift.task", "load_task"), list_tasks=_opt("nightshift.task", "list_tasks"),
        validate_task=_opt("nightshift.task", "validate_task"),
        harden_parent=_opt("nightshift.sandbox", "harden_parent"),
        available_isolation=_opt("nightshift.sandbox", "available_isolation"),
        Orchestrator=_opt("nightshift.orchestrator", "Orchestrator"),
        Auditor=_opt("nightshift.audit.auditor", "Auditor"), LLMJudge=_opt("nightshift.audit.judge", "LLMJudge"),
        load_routing=_opt("nightshift.models.config", "load_routing"), client_for=_opt("nightshift.models.config", "client_for"),
        ModelConfigError=_opt("nightshift.models.base", "ModelConfigError") or ValueError,
        run_smoke=_opt("nightshift.models.live_smoke", "run_smoke"),
        build_client_factory=_opt("nightshift.models.live_smoke", "build_client_factory"),
        summarize=_opt("nightshift.models.live_smoke", "summarize"))


def _harden(api: SimpleNamespace) -> None:
    """First thing a command that runs experiments does: before any key is read or any child process exists."""
    if api.harden_parent is not None:
        api.harden_parent()


def _need(api: SimpleNamespace, name: str) -> Any:
    obj = getattr(api, name, None)
    if obj is None:
        raise CliError(f"{name} is not available: the module that provides it has not been installed in this checkout")
    return obj


# --------------------------------------------------------------------------------------------------
# paths and small helpers
# --------------------------------------------------------------------------------------------------
def _repo_root(start: Optional[Path] = None) -> Path:
    """The nearest ancestor of ``start`` (default: cwd) that holds ``pyproject.toml`` and ``nightshift/``; else the checkout
    this package lives in. Lets the demo work from any directory inside the repo."""
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").is_file() and (d / "nightshift").is_dir():
            return d
    return Path(__file__).resolve().parents[1]


def _tasks_dir(root: Optional[Path] = None) -> Optional[Path]:
    """The directory that holds the bundled tasks, or ``None`` when there is none (a non-editable ``pip install`` ships no
    tasks on purpose). ``$NIGHTSHIFT_TASKS_DIR`` wins when set; else ``tasks/`` of the repository root. A directory counts only
    when at least one ``*/task.toml`` is in it."""
    raw = os.environ.get(TASKS_ENV, "").strip()
    try:
        d = Path(raw).expanduser() if raw else (root or _repo_root()) / "tasks"
        return d if d.is_dir() and any(d.glob("*/task.toml")) else None
    except (OSError, RuntimeError, ValueError):
        return None


def _tasks_missing(spec: Optional[str] = None) -> CliError:
    """The one-line usage error for bundled tasks that cannot be found (``spec``: the task name that was asked for)."""
    raw = os.environ.get(TASKS_ENV, "").strip()
    where = f" ({TASKS_ENV}={ascii(raw[:80])} holds no task)" if raw else ""
    what = f"no task found for {ascii(spec[:80])}: " if spec else ""
    return CliError(f"{what}{TASKS_HINT}{where}", EXIT_USAGE)


def _check_isolation_env() -> None:
    """Refuse an invalid ``NIGHTSHIFT_ISOLATION`` now (exit 2, naming the allowed values) instead of deep inside a run, before
    any run directory or ledger exists."""
    validate = _opt("nightshift.sandbox", "isolation_override")
    if validate is None:
        return
    try:
        validate()
    except ValueError as exc:
        raise CliError(str(exc), EXIT_USAGE) from exc


def _run_dir(arg: str) -> Path:
    p = Path(arg)
    if not (p / LEDGER_NAME).is_file():
        raise CliError(f"{p} has no {LEDGER_NAME}; give a run directory (for example runs/demo)", EXIT_USAGE)
    return p


def _find_task_dir(spec: str, root: Optional[Path] = None) -> Path:
    """The task directory for ``spec``: a path that holds ``task.toml``, else a path relative to the repository root, else a
    name under the bundled tasks directory. Nothing found is a usage error; when there is no bundled tasks directory at all it
    is the one-line ``TASKS_HINT``."""
    root = root or _repo_root()
    tasks = _tasks_dir(root)
    cands = [Path(spec), root / spec] + ([tasks / spec] if tasks is not None else [])
    for c in cands:
        if (c / "task.toml").is_file():
            return c
    if tasks is None:
        raise _tasks_missing(spec)
    raise CliError(f"no task found for {spec!r} (looked for task.toml in: {', '.join(str(c) for c in cands)})", EXIT_USAGE)


def _load_task(api: SimpleNamespace, spec: str) -> Any:
    load = _need(api, "load_task")
    try:
        return load(_find_task_dir(spec))
    except ValueError as exc:
        raise CliError(f"bad task: {exc}", EXIT_USAGE) from exc


def _started(run_dir: Path) -> dict:
    for ev in Ledger.read_only(run_dir).events():
        if ev.kind == "run.started":
            return dict(ev.data)
    return {}


def _task_for_run(api: SimpleNamespace, run_dir: Path, explicit: Optional[str]) -> Any:
    """The task a run was started with: ``--task`` if given, else the task directory named by ``run.started``."""
    if explicit:
        return _load_task(api, explicit)
    name = str(_started(run_dir).get("task") or "")
    if not name:
        raise CliError("cannot tell which task this run used; pass --task", EXIT_USAGE)
    load = _need(api, "load_task")
    tasks = _tasks_dir()
    if tasks is None:
        raise _tasks_missing()
    for d in [tasks / name, *sorted(tasks.glob("*"))]:
        if (d / "task.toml").is_file():
            try:
                task = load(d)
            except ValueError:
                continue
            if getattr(task, "name", None) == name:
                return task
    raise CliError(f"no task named {name!r} under {tasks}; pass --task", EXIT_USAGE)


def _view(run_dir: Path) -> LedgerView:
    return build_view(Ledger.read_only(run_dir).events())


def _anchor(run_dir: Path) -> str:
    return format_anchor(Ledger.read_only(run_dir).events())


def _unique_dir(path: Path) -> Path:
    if not (path / LEDGER_NAME).exists():
        return path
    for i in range(2, 1000):
        cand = path.with_name(f"{path.name}-{i}")
        if not (cand / LEDGER_NAME).exists():
            return cand
    raise CliError(f"too many runs named {path.name}", EXIT_USAGE)


def _chain_warning(run_dir: Path) -> Optional[str]:
    """One line saying the hash chain of ``run_dir`` is broken (naming the first bad seq), or ``None`` when it verifies.

    Read-only and never raises: a ledger that cannot even be parsed is reported as broken too, because a caller that wants a
    warning must not fail on the very condition it is warning about."""
    try:
        ok, bad = Ledger.read_only(run_dir).verify_chain()
    except LedgerCorrupt as exc:
        return (f"warning: the ledger in {run_dir} is corrupt ({exc}); its hash chain does not verify and nothing derived from "
                "it can be trusted")
    except OSError as exc:
        return f"warning: the ledger in {run_dir} could not be read to check its hash chain ({exc.strerror or type(exc).__name__})"
    if ok:
        return None
    return (f"warning: the hash chain of {run_dir} is BROKEN at seq {bad}: the ledger was edited or damaged after it was written, "
            "so the numbers derived from it cannot be trusted (`nightshift verify` shows the same)")


def _hold(server: Any) -> None:
    """Keep the process (and its daemon server thread) alive until Ctrl-C. Tests replace this."""
    print("Dashboard is still running. Press Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print()


def _serve_blocking(server: Any) -> None:
    """Serve until Ctrl-C. Tests replace this."""
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


def _start_server(run_dir: Path, port: int, host: str = "127.0.0.1", **kw: Any) -> Any:
    try:
        server = make_server(run_dir, host, port, **kw)
    except (ValueError, OSError, ReplayRefused, LedgerCorrupt) as exc:
        raise CliError(f"cannot start the dashboard: {exc}",
                       EXIT_FAIL if isinstance(exc, (ReplayRefused, LedgerCorrupt)) else EXIT_USAGE) from exc
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True,
                     name="nightshift-dashboard").start()
    print(f"dashboard: {server.base_url}/", flush=True)
    return server


def _limits(args: argparse.Namespace) -> BudgetLimits:
    lim = BudgetLimits()
    for attr, field, cast in (("max_experiments", "max_experiments", int), ("max_usd", "max_llm_usd", float),
                              ("max_wall", "max_wall_sec", float), ("parallel", "max_parallel", int)):
        val = getattr(args, attr, None)
        if val is None:
            continue
        if not val > 0:
            raise CliError(f"--{attr.replace('_', '-')} must be positive", EXIT_USAGE)
        setattr(lim, field, cast(val))
    return lim


def _check_implementer(backend: str, implementer: Optional[str]) -> None:
    """The scripted backend has its own implementer and ignores ``--implementer``; accepting ``muse_code`` there would run
    something else than the command line says, so the combination is a usage error (exit 2)."""
    if backend == "scripted" and implementer == "muse_code":
        raise CliError("--implementer muse_code needs --backend meta: the scripted backend has its own implementer and would "
                       "ignore it", EXIT_USAGE)


def _parse_plant(raw: Optional[Sequence[str]], backend: str) -> list[str]:
    """``--plant NAME[,NAME...]`` (repeatable) as a list of behavior names. A usage error (exit 2) with the scripted backend (it
    plants its own faults), for an empty name, an unknown behavior or a repeated one. Whether the TASK's bank can serve each
    behavior is checked once the task is loaded (``_check_plant``)."""
    if raw is None:
        return []
    if backend != "meta":
        raise CliError("--plant needs --backend meta: red-team lanes are scripted attackers added to a run with REAL model roles, "
                       "and the scripted backend already plants its own faults", EXIT_USAGE)
    names: list[str] = []
    for chunk in raw:
        for name in chunk.split(","):
            name = name.strip()
            if not name:
                raise CliError("--plant takes behavior names separated by commas, for example --plant tamper,overfit", EXIT_USAGE)
            if name not in PLANTABLE:
                raise CliError(f"--plant: unknown behavior {ascii(name[:24])}; choose from {', '.join(PLANTABLE)}", EXIT_USAGE)
            if name in names:
                raise CliError(f"--plant: behavior {name!r} is listed more than once", EXIT_USAGE)
            names.append(name)
    return names


def _check_plant(task: Any, plant: Sequence[str]) -> list[tuple[str, str]]:
    """The ``(behavior, scripted_id)`` pairs for ``plant``, or a usage error naming what the task's bank lacks."""
    if not plant:
        return []
    try:
        return planted_proposals(task, plant)
    except ValueError as exc:
        raise CliError(f"--plant: {exc}", EXIT_USAGE) from exc


def _shown_path(text: str, limit: int = 80) -> str:
    """A path for an error message, quoted with ``ascii``. A long path keeps its END (the file name is what the operator
    needs), marked with a leading ellipsis, because temp directories on laptops are long."""
    return ascii(text if len(text) <= limit else "..." + text[-(limit - 3):])


def _read_context(raw: Optional[Sequence[Sequence[str]]], backend: str) -> tuple[str, list[dict]]:
    """``--context FILE [FILE ...]`` (repeatable): the files' text joined for the roles, and ``[{name, sha256, chars}]`` for the
    ledger (the basename only, the sha256 of the file's bytes and its length in characters; never the content).

    A usage error (exit 2) with the scripted backend, for a missing, unreadable, empty, binary or non-UTF-8 file, for a file
    longer than ``CONTEXT_FILE_MAX_CHARS`` and for a total beyond ``CONTEXT_TOTAL_MAX_CHARS``. Nothing is truncated here: a file
    is either accepted whole or refused, and the roles' own caps are reported by ``_context_note``."""
    paths = [p for group in (raw or []) for p in group]
    if not paths:
        return "", []
    if backend != "meta":
        raise CliError("--context needs --backend meta: the scripted backend has no language model to read background material",
                       EXIT_USAGE)
    parts: list[str] = []
    records: list[dict] = []
    seen: set[Path] = set()
    total = 0
    for given in paths:
        shown = _shown_path(given)
        try:
            path = Path(given).expanduser()
        except RuntimeError:        # "~nosuchuser/x" (or no home directory at all): a usage error, never a traceback
            raise CliError(f"--context: cannot resolve the home directory in {shown}", EXIT_USAGE) from None
        try:
            exists = path.exists()
            is_file = exists and path.is_file()
            resolved = path.resolve()
        except (OSError, ValueError, RuntimeError) as exc:      # a name too long for the filesystem, a symlink loop, a bad path
            raise CliError(f"--context: cannot use {shown}: {getattr(exc, 'strerror', None) or type(exc).__name__}", EXIT_USAGE) from exc
        if not exists:
            raise CliError(f"--context: file {shown} does not exist", EXIT_USAGE)
        if not is_file:
            raise CliError(f"--context: {shown} is not a file", EXIT_USAGE)
        if resolved in seen:
            raise CliError(f"--context: file {shown} is listed more than once", EXIT_USAGE)
        seen.add(resolved)
        try:
            size = path.stat().st_size
            if size > 4 * CONTEXT_FILE_MAX_CHARS:     # a UTF-8 character is at most 4 bytes: cannot fit, no need to read it all
                raise CliError(f"--context: file {shown} is larger than {CONTEXT_FILE_MAX_CHARS:,} characters", EXIT_USAGE)
            data = path.read_bytes()
        except OSError as exc:
            raise CliError(f"--context: cannot read {shown}: {exc.strerror or type(exc).__name__}", EXIT_USAGE) from exc
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise CliError(f"--context: file {shown} is not UTF-8 text (a binary file?)", EXIT_USAGE) from None
        if _BINARY_CHARS.search(text):
            raise CliError(f"--context: file {shown} contains control characters; only plain text or markdown is accepted", EXIT_USAGE)
        if not text.strip():
            raise CliError(f"--context: file {shown} is empty", EXIT_USAGE)
        if len(text) > CONTEXT_FILE_MAX_CHARS:
            raise CliError(f"--context: file {shown} has {len(text):,} characters; at most {CONTEXT_FILE_MAX_CHARS:,} per file",
                           EXIT_USAGE)
        total += len(text)
        if total > CONTEXT_TOTAL_MAX_CHARS:
            raise CliError(f"--context: the files together exceed {CONTEXT_TOTAL_MAX_CHARS:,} characters", EXIT_USAGE)
        name = sanitize(path.name, 80) or "file"
        records.append({"name": name, "sha256": hashlib.sha256(data).hexdigest(), "chars": len(text)})
        parts.append(f"===== {name} =====\n{text.strip()}\n")
    return "\n".join(parts), records


def _context_note(text: str) -> list[str]:
    """Lines to print about the background material: what was read, and, when it is longer than what the roles use, how much
    of it they will actually see."""
    sup = _opt("nightshift.agents.prompts", "BACKGROUND_SUPERVISOR_CHARS") or 60_000
    des = _opt("nightshift.agents.prompts", "BACKGROUND_DESIGNER_CHARS") or 8_000
    lines = [f"background: {len(text):,} characters; the supervisor reads up to {sup:,} of them and the designer up to {des:,}"]
    if len(text) > sup:
        lines.append(f"note: the background is longer than the supervisor uses: only its first {sup:,} characters reach the "
                     f"supervisor (and the first {des:,} the designer); the remaining {len(text) - sup:,} are not used")
    elif len(text) > des:
        lines.append(f"note: the designer reads only the first {des:,} characters of the background; the supervisor reads all of it")
    return lines


def _has_key() -> bool:
    return any(os.environ.get(k) for k in KEY_ENVS)


def _exit_status(run_dir: Path, result: Any) -> int:
    """1 when the run ended with open block findings, unresolved defects, an error stop (a refused API key included, which is
    ``missing_authority`` but not a success) or budget exhausted without an incumbent."""
    view = _view(run_dir)
    result = result if isinstance(result, dict) else {}
    fin = view.finished or {}      # the ledger is authoritative; the returned dict only fills gaps
    verdict = fin.get("verdict_class") or result.get("verdict_class")
    reason = fin.get("reason") or result.get("reason")
    if view.open_findings("block") or verdict == "unresolved_defects":
        return EXIT_FAIL
    if reason == "error":       # e.g. the model API refused the key (missing_authority): the run did not do its job
        return EXIT_FAIL
    if reason == "budget_exhausted" and view.incumbent_id in (None, view.baseline_id):
        return EXIT_FAIL
    return EXIT_OK


def _print_summary(run_dir: Path, result: Any) -> None:
    r = result if isinstance(result, dict) else {}
    print(f"run {r.get('run_id') or _view(run_dir).run.get('run_id', '')}: reason={r.get('reason')}, "
          f"verdict={r.get('verdict_class')}, incumbent={r.get('incumbent')}")
    print(f"head anchor: {_anchor(run_dir)}")
    print("Record the head anchor somewhere else; `nightshift verify RUN_DIR --expect-head ANCHOR` then detects truncation.")


def _orchestrate(api: SimpleNamespace, task: Any, *, run_dir: Path, backend: str, limits: Optional[BudgetLimits],
                 routing_path: Optional[str], implementer: Optional[str], control: bool, crash_after_runs: Optional[int],
                 serve_port: Optional[int], resume: bool = False, accept_broken_chain: bool = False,
                 hold: bool = True, plant: Sequence[str] = (), background: str = "",
                 context: Sequence[dict] = ()) -> int:
    """Shared by run, resume and demo: harden, register keys, create/resume, run, print the head anchor."""
    orchestrator = _need(api, "Orchestrator")
    secrets.register_env(*KEY_ENVS)
    if backend == "meta" and not _has_key():
        raise CliError("backend meta needs MODEL_API_KEY (or META_API_KEY) in the environment", EXIT_USAGE)
    routing = None
    if routing_path:
        try:
            routing = _need(api, "load_routing")(routing_path)
        except api.ModelConfigError as exc:
            raise CliError(f"bad config: {exc}", EXIT_USAGE) from exc
    kw: dict[str, Any] = {} if crash_after_runs is None else {"crash_after_runs": crash_after_runs}
    server = None
    try:
        try:
            if resume:
                extra = ({"control": True} if control else {}) | ({"implementer": implementer} if implementer else {})
                if background:
                    extra |= {"background": background, "context": [dict(c) for c in context]}
                orch = orchestrator.resume(run_dir, task, backend=backend, routing=routing,
                                           accept_broken_chain=accept_broken_chain, **extra, **kw)
            else:
                if plant:
                    kw["plant"] = list(plant)
                if background:
                    kw["background"], kw["context"] = background, [dict(c) for c in context]
                orch = orchestrator.create(run_dir, task, backend=backend, routing=routing, limits=limits,
                                           implementer=implementer or "patch", control=control, **kw)
        except FileExistsError as exc:
            raise CliError(f"{run_dir} already holds a run; choose a new --run-dir (`nightshift resume {run_dir}` only continues "
                           "a run that was killed before it finished)", EXIT_USAGE) from exc
        except (NightshiftError, LedgerLocked, ValueError) as exc:
            raise CliError(f"cannot {'resume' if resume else 'start'} the run: {exc}") from exc
        if serve_port is not None:
            server = _start_server(run_dir, serve_port, task=task)
        try:
            result = orch.run()
        except SimulatedCrash:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(137)   # mimics kill -9: nothing is closed or cleaned up on the way out
        except KeyboardInterrupt:
            print(f"\ninterrupted; the ledger is intact. An interrupted run is closed as halted and a finished run cannot "
                  f"be resumed: start a new run, or use `nightshift report {run_dir}` to see what was recorded", file=sys.stderr)
            return 130
        except NightshiftError as exc:
            raise CliError(f"run failed: {exc}") from exc
        _print_summary(run_dir, result)
        code = _exit_status(run_dir, result)
        if server is not None and hold:
            _hold(server)
        return code
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


# --------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    _check_isolation_env()
    _check_implementer(args.backend, args.implementer)
    plant = _parse_plant(args.plant, args.backend)
    background, context = _read_context(args.context, args.backend)
    api = _api()
    _harden(api)
    task = _load_task(api, args.task)
    red_team = _check_plant(task, plant)
    limits = _limits(args)
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"{task.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"task {task.name}, backend {args.backend}, run dir {run_dir}")
    if red_team:
        print("red team: " + ", ".join(f"{b} ({sid})" for b, sid in red_team) + " -- scripted attacker lanes injected by the "
              "operator; the supervisor, designer, reviewer and auditor stay real model calls")
    if context:
        print(f"context: {len(context)} file(s): " + ", ".join(f"{c['name']} ({c['chars']:,} characters)" for c in context))
        for line in _context_note(background):
            print(line)
    return _orchestrate(api, task, run_dir=run_dir, backend=args.backend, limits=limits, routing_path=args.config,
                        implementer=args.implementer, control=args.control, crash_after_runs=args.crash_after_runs,
                        serve_port=args.port if args.serve else None, plant=plant, background=background, context=context)


def _check_resume_context(saved: Any, given: Sequence[dict]) -> None:
    """``--context`` on ``resume`` must be exactly the files the run started with: the same set of ``(name, sha256)`` pairs as
    ``run.started.config.context``, in any order. Anything else is a usage error that names the file."""
    # pairs, not a name -> hash map: two files may share a basename (a/notes.md, b/notes.md) and each must be matched on its own
    recorded = {(str(c.get("name")), str(c.get("sha256"))) for c in saved if isinstance(c, dict)} if isinstance(saved, list) else set()
    if not recorded:
        raise CliError("--context: this run started without background files, so none can be given on resume", EXIT_USAGE)
    names = sorted({n for n, _ in recorded})
    by_name: dict[str, set[str]] = {}
    for n, digest in recorded:
        by_name.setdefault(n, set()).add(digest)
    have = {(c["name"], c["sha256"]) for c in given}
    for name, digest in sorted(have - recorded):
        if name not in by_name:
            raise CliError(f"--context: {ascii(name)} is not one of the background files this run started with "
                           f"({', '.join(ascii(n) for n in names)})", EXIT_USAGE)
        raise CliError(f"--context: {ascii(name)} has changed since the run started (sha256 {digest[:12]} instead of "
                       f"{sorted(by_name[name])[0][:12]}); a resumed run must read the same background", EXIT_USAGE)
    for name, _ in sorted(recorded - have):
        raise CliError(f"--context: the run started with the background file {ascii(name)}; give it again (every file of the "
                       "original --context is required)", EXIT_USAGE)


def cmd_resume(args: argparse.Namespace) -> int:
    _check_isolation_env()
    api = _api()
    _harden(api)
    run_dir = _run_dir(args.run_dir)
    task = _task_for_run(api, run_dir, args.task)
    started = _started(run_dir)
    backend = args.backend or str(started.get("backend") or "scripted")
    _check_implementer(backend, args.implementer)
    saved_config = started.get("config") if isinstance(started.get("config"), dict) else {}
    saved_context = saved_config.get("context")
    background, context = _read_context(args.context, backend)
    if context:
        _check_resume_context(saved_context, context)
        print(f"context: {len(context)} file(s), the same as at the start: "
              + ", ".join(f"{c['name']} ({c['chars']:,} characters)" for c in context))
        for line in _context_note(background):
            print(line)
    elif saved_context:
        print("warning: this run started with background files (--context) and none were given: the supervisor and designer "
              "will work WITHOUT the background. Pass the same files with --context FILE [FILE ...] to restore it",
              file=sys.stderr)
    return _orchestrate(api, task, run_dir=run_dir, backend=backend, limits=None, routing_path=args.config,
                        implementer=args.implementer, control=args.control, crash_after_runs=None, serve_port=None,
                        resume=True, accept_broken_chain=args.accept_broken_chain, background=background, context=context)


def cmd_serve(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_dir)
    host = args.host or ("0.0.0.0" if args.allow_remote else "127.0.0.1")
    try:
        server = make_server(run_dir, host, args.port, replay_speed=args.replay, replay_duration=args.replay_duration,
                             allow_remote=args.allow_remote)
    except ReplayRefused as exc:
        raise CliError(str(exc)) from exc
    except LedgerCorrupt as exc:
        raise CliError(f"cannot start the dashboard: {exc}") from exc
    except (ValueError, OSError) as exc:
        raise CliError(f"cannot start the dashboard: {exc}", EXIT_USAGE) from exc
    try:
        if not server.replay:       # a replay of a broken chain is refused by make_server; a live view is still served
            warning = _chain_warning(run_dir)
            if warning:
                print(warning, file=sys.stderr, flush=True)
        print(f"dashboard: {server.base_url}/" + (" (replay)" if server.replay else ""), flush=True)
        _serve_blocking(server)
    finally:
        server.server_close()
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_dir)
    try:
        md, html_path = write_reports(run_dir, args.out)
    except LedgerCorrupt as exc:
        raise CliError(f"ledger is corrupt, no report written: {exc}") from exc
    print(f"wrote {md}\nwrote {html_path}")
    warning = _chain_warning(run_dir)
    if warning:       # the report is written either way (it carries its own banner); the exit status says it is not trustworthy
        print(warning, file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


def anchor_matches(expected: str, events: Sequence[Any]) -> bool:
    """``expected`` is ``<seq>:<hash16>``, or a hex prefix (at least 8 characters) of the head hash."""
    exp = expected.strip().lower()
    if not events or not exp:
        return False
    head = events[-1]
    if exp == format_anchor(events):
        return True
    return len(exp) >= 8 and ":" not in exp and head.hash.startswith(exp)


def cmd_verify(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_dir)
    try:
        ledger = Ledger.read_only(run_dir)
    except LedgerCorrupt as exc:
        print(f"chain BROKEN: {exc}")
        return EXIT_FAIL
    events = ledger.events()
    ok, bad = ledger.verify_chain()
    print(f"events: {len(events)}")
    print(f"head anchor: {format_anchor(events)}")
    if ledger.recovered_torn_tail:
        print("note: the final line is incomplete (a write was interrupted); it is ignored")
    if not events:
        print("empty ledger: it holds no events, so there is no run to verify (a run is created with `nightshift run`)")
        return EXIT_FAIL
    if not ok:
        print(f"chain BROKEN at seq {bad}")
        return EXIT_FAIL
    print("chain consistent")
    if args.expect_head is not None:
        if not anchor_matches(args.expect_head, events):
            print(f"head MISMATCH: expected {args.expect_head}, found {format_anchor(events)} "
                  "(the ledger was truncated, rewritten or is a different run)")
            return EXIT_FAIL
        print("head matches the expected anchor")
    return EXIT_OK


def _judge(api: SimpleNamespace, kind: str, config: Optional[str]) -> Any:
    if kind == "none":
        return None
    secrets.register_env(*KEY_ENVS)
    if not _has_key():
        raise CliError("--judge meta needs MODEL_API_KEY (or META_API_KEY) in the environment", EXIT_USAGE)
    try:
        routing = _need(api, "load_routing")(config)
        return _need(api, "LLMJudge")(_need(api, "client_for")("auditor_final", routing))
    except api.ModelConfigError as exc:
        raise CliError(f"cannot build the judge: {exc}", EXIT_USAGE) from exc


def cmd_audit(args: argparse.Namespace) -> int:
    api = _api()
    run_dir = _run_dir(args.run_dir)
    task = _task_for_run(api, run_dir, args.task)
    auditor_cls = _need(api, "Auditor")
    judge = _judge(api, args.judge, args.config)
    before = set(_view(run_dir).findings)
    path = run_dir / LEDGER_NAME
    tmp: Optional[tempfile.TemporaryDirectory] = None
    restore_mode: Optional[int] = None
    if args.write:
        target = run_dir
        mode = path.stat().st_mode & 0o777
        if not mode & 0o200:           # a finished run is chmod 0444; lift it only while we append, then restore
            restore_mode = mode
            os.chmod(path, mode | 0o200)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="nightshift-audit-")
        target = Path(tmp.name)
        shutil.copyfile(path, target / LEDGER_NAME)
    try:
        try:
            ledger = Ledger.open(target, fsync=args.write)
        except LedgerLocked as exc:
            raise CliError(f"{exc}; a run is still writing it") from exc
        try:
            auditor = auditor_cls(ledger, task, judge=judge)
            auditor.audit_window()
            auditor.final_sweep()
            view = ledger.view()
        finally:
            ledger.close()
        new = [f for fid, f in view.findings.items() if fid not in before]
        print(f"re-audit of {run_dir} ({'written to the ledger' if args.write else 'on a temporary copy; the original is untouched'})")
        print(f"findings: {len(view.findings)} total, {len(new)} new, {len(view.open_findings('block'))} open block")
        for f in sorted(view.findings.values(), key=lambda f: f.finding_id):
            print(f"  {f.finding_id} [{f.severity}/{f.verdict}/{f.status}] {f.source}")
        return EXIT_FAIL if view.open_findings("block") else EXIT_OK
    finally:
        if restore_mode is not None:
            os.chmod(path, restore_mode)
        if tmp is not None:
            tmp.cleanup()


def _muse_code_probe(muse: Optional[str]) -> dict:
    """Run ``nightshift.agents.muse_code.probe_exec`` (the production command line against a throwaway workspace) when a muse
    binary is on PATH, print a summary and return the record that goes into the live probe file under ``muse_code``. A module
    without ``probe_exec`` or a probe that raises only produces a note: the Muse Code worker is optional."""
    if muse is None:
        print("muse binary: not on PATH, Muse Code worker not probed")
        return {"probed": False, "reason": "muse binary not on PATH"}
    probe = _opt("nightshift.agents.muse_code", "probe_exec")
    if not callable(probe):
        print("muse code probe: this build of nightshift.agents.muse_code has no probe_exec; the Muse Code worker was not probed")
        return {"probed": False, "reason": "nightshift.agents.muse_code has no probe_exec"}
    print("muse code probe: running the production muse exec command line once (a few model calls)...", flush=True)
    try:
        result = probe(timeout_sec=90.0)
    except Exception as exc:  # noqa: BLE001 - a probe must never stop the report
        print(f"muse code probe: FAIL, probe raised {type(exc).__name__}")
        return {"probed": True, "ok": False, "detail": f"probe raised {type(exc).__name__}"}
    result = dict(result) if isinstance(result, dict) else {"ok": False, "detail": "probe returned no result"}
    result = {"probed": True, **result}
    seen = lambda key: "yes" if result.get(key) else "no"  # noqa: E731
    print(f"muse code probe: {'PASS' if result.get('ok') else 'FAIL'} (exit {result.get('exit_code')}, model reported "
          f"{secrets.scrub_text(sanitize(str(result.get('model_reported') or 'none'), 60))}, usage line seen {seen('usage_line_seen')}, "
          f"trust stop {seen('found_trust_stop')})")
    detail = secrets.scrub_text(sanitize(str(result.get("detail") or ""), 300))
    if detail:
        print(f"      {detail}")
    tail = secrets.scrub_text(sanitize(str(result.get("stderr_tail") or ""), 300))
    if tail and not result.get("ok"):
        print(f"      stderr: {tail}")
    return secrets.scrub(result)


def _add_to_probe_file(path: Optional[str], report: dict) -> None:
    """Rewrite the live probe file (already written by the smoke run) with the ``muse_code`` section included. A file the smoke
    run did not write is not created here."""
    if not path:
        return
    target = Path(path)
    if not target.is_file():
        return
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        print(f"probe file: cannot add the muse_code section to {target}: {exc.strerror or type(exc).__name__}")


def cmd_doctor(args: argparse.Namespace) -> int:
    api = _api()
    secrets.register_env(*KEY_ENVS)      # anything a probe prints below is scrubbed of the key
    iso_problem: Optional[str] = None
    try:
        _check_isolation_env()
    except CliError as exc:
        iso_problem = str(exc)
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    if importlib.util.find_spec("numpy") is not None:
        try:
            from importlib.metadata import version
            print(f"numpy: present ({version('numpy')})")
        except Exception:  # noqa: BLE001 - version lookup is cosmetic
            print("numpy: present")
    else:
        print("numpy: missing (only the optional spiral task needs it)")
    try:
        if iso_problem is not None:
            iso = f"not run: {iso_problem}"
        else:
            iso = api.available_isolation() if api.available_isolation else "unavailable (sandbox module missing)"
    except Exception as exc:  # noqa: BLE001 - a failed probe must not stop the rest of the report
        iso = f"probe failed ({type(exc).__name__})"
    print(f"isolation probe: {iso}")
    muse = shutil.which("muse")
    if muse is None:
        print("muse binary: not on PATH")
    else:
        try:
            proc = subprocess.run([muse, "--version"], capture_output=True, text=True, timeout=5, check=False)
            text = (proc.stdout or proc.stderr).strip()
            line = secrets.scrub_text(text.splitlines()[0] if text else "")
            print(f"muse binary: {muse} ({line[:80] or 'no version output'}, exit {proc.returncode})")
        except subprocess.TimeoutExpired:
            print(f"muse binary: {muse} (--version timed out after 5 s)")
        except OSError as exc:
            print(f"muse binary: {muse} (cannot run: {type(exc).__name__})")
    for name in KEY_ENVS:
        print(f"{name}: {'set' if os.environ.get(name) else 'not set'}")
    if not _has_key():
        print("No API key: the offline demo works (python3 -m nightshift demo); live runs need MODEL_API_KEY, "
              "see docs/RUNNING_LIVE.md.")
    if iso_problem is not None:
        return EXIT_USAGE
    if not args.live:
        return EXIT_OK
    secrets.register_env(*KEY_ENVS)
    if api.run_smoke is None or api.build_client_factory is None:
        print("live smoke: nightshift.models.live_smoke is not available")
        return EXIT_FAIL
    try:
        factory = api.build_client_factory()
    except api.ModelConfigError as exc:
        print(f"live smoke: setup error: {exc}")
        return EXIT_USAGE
    report = api.run_smoke(factory, args.live_out)
    for line in (api.summarize(report) if api.summarize else []):
        print(line)
    # after the API probes: the Muse Code worker, only when a muse binary exists. Its outcome is reported and recorded, but
    # the exit status stays that of the API probes because the worker is optional (the default implementer is the patch worker)
    report["muse_code"] = _muse_code_probe(muse)
    _add_to_probe_file(args.live_out, report)
    return EXIT_OK if report.get("ok") else EXIT_FAIL


def cmd_tasks(args: argparse.Namespace) -> int:
    api = _api()
    list_tasks, validate = _need(api, "list_tasks"), _need(api, "validate_task")
    if args.root:
        root = Path(args.root)
    else:
        found = _tasks_dir(Path.cwd()) or _tasks_dir()      # ./tasks first (as before), then the checkout's
        if found is None:
            raise _tasks_missing()
        root = found
    bad: list[tuple[Path, ValueError]] = []
    tasks = list_tasks(root, on_error=lambda p, e: bad.append((p, e)))
    if not tasks and not bad:
        print(f"no tasks found under {root}")
        return EXIT_FAIL
    failed = 0
    for t in tasks:
        problems = list(validate(t, calibrate=True))
        # an absent optional dependency is a skip (the check could not run), any other problem is a failure
        real = [p for p in problems if not str(p).startswith(SKIP_PREFIX)]
        skipped = [p for p in problems if str(p).startswith(SKIP_PREFIX)]
        print(f"{'FAIL' if real else ('SKIP' if skipped else 'OK  ')} {t.name}: {t.description}")
        for p in real:
            print(f"       - {p}")
        for p in skipped:
            print(f"       skipped: {p}")
        failed += bool(real)
    for path, exc in bad:
        print(f"FAIL {path.name}: {exc}")
        failed += 1
    return EXIT_FAIL if failed else EXIT_OK


# --------------------------------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------------------------------
def _run_child(argv: Sequence[str], cwd: Path) -> int:
    """Run ``python -m nightshift ...`` in a child process and return its exit status."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-m", "nightshift", *argv], cwd=cwd, env=env, check=False).returncode


def _demo_crash(task_dir: Path, out: Path, root: Path) -> int:
    code = _run_child(["run", "--task", str(task_dir), "--backend", "scripted", "--run-dir", str(out),
                       "--crash-after-runs", str(DEMO_CRASH_AFTER_RUNS)], root)
    if code != 137:
        print(f"the first process exited with {code} instead of crashing (137); the run ended before the crash point")
        return code
    print("the first process was killed mid-experiment (exit 137). Resuming from the ledger in a second process...")
    return _run_child(["resume", str(out), "--task", str(task_dir)], root)


def _demo_replay(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.replay)
    server = _start_server(run_dir, args.port, replay_speed=args.replay_speed)
    try:
        print(f"replaying {run_dir} at {args.replay_speed}x (read-only)")
        _hold(server)
    finally:
        server.shutdown()
        server.server_close()
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    if args.replay:
        return _demo_replay(args)
    _check_isolation_env()
    api = _api()
    if not args.with_crash:
        _harden(api)             # the crash flow runs its work in child processes, which harden themselves
    root = _repo_root()          # used only to locate the bundled tasks; --out is the caller's, relative to the caller's cwd
    task_dir = _find_task_dir(args.task, root).absolute()   # child processes run elsewhere: a relative path would break
    task = _load_task(api, str(task_dir))
    try:
        wanted = Path(args.out).expanduser().absolute()
        out = _unique_dir(wanted)
    except RuntimeError:        # "~nosuchuser/x": no home directory to expand, a usage error and never a traceback
        raise CliError(f"--out: cannot resolve the home directory in {_shown_path(args.out)}", EXIT_USAGE) from None
    if out != wanted:
        print(f"the directory {wanted} already existed, using {out}")
    print(f"demo: task {task.name}, scripted backend (no language model), run dir {out}")
    server = _start_server(out, args.port, task=task) if args.serve and not args.no_serve else None
    try:
        if args.with_crash:
            code = _demo_crash(task_dir, out, root)
        else:
            code = _orchestrate(api, task, run_dir=out, backend="scripted", limits=None, routing_path=None, implementer=None,
                                control=False, crash_after_runs=None, serve_port=None)
        if (out / LEDGER_NAME).is_file():
            md, html_path = write_reports(out)
            print(f"report: {md}\nreport: {html_path}")     # the head anchor was printed once, by the run summary above
        if args.control:
            ctl_wanted = out.with_name(out.name + "-control")
            ctl = _unique_dir(ctl_wanted)
            if ctl != ctl_wanted:
                print(f"the directory {ctl_wanted} already existed, using {ctl}")
            print(f"negative control run dir: {ctl} (honest proposals only; it must find nothing)")
            ccode = _orchestrate(api, task, run_dir=ctl, backend="scripted", limits=None, routing_path=None, implementer=None,
                                 control=True, crash_after_runs=None, serve_port=None)
            n = len(_view(ctl).findings) if (ctl / LEDGER_NAME).is_file() else -1
            print(f"negative control findings: {n} (expected 0)")
            if (ctl / LEDGER_NAME).is_file():
                write_reports(ctl)
            code = code or ccode or (EXIT_FAIL if n != 0 else EXIT_OK)
        if server is not None:
            _hold(server)
        return code
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


# --------------------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------------------
def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not v > 0 or v != v or v == float("inf"):
        raise argparse.ArgumentTypeError("must be a positive, finite number")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nightshift", description="An auto-researcher you can leave running overnight.")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def cmd(name: str, fn: Any, help: str) -> argparse.ArgumentParser:  # noqa: A002
        sp = sub.add_parser(name, help=help, description=help)
        sp.set_defaults(func=fn)
        return sp

    r = cmd("run", cmd_run, "run a new research session")
    r.add_argument("--task", required=True, help="task directory (for example tasks/law-discovery)")
    r.add_argument("--backend", choices=("scripted", "meta"), default="scripted")
    r.add_argument("--run-dir")
    r.add_argument("--max-experiments", type=int)
    r.add_argument("--max-usd", type=float)
    r.add_argument("--max-wall", type=float, help="wall-clock budget in seconds")
    r.add_argument("--parallel", type=int)
    r.add_argument("--config", help="model routing file (nightshift.toml)")
    r.add_argument("--implementer", choices=("patch", "muse_code"), help="code-writing worker (meta backend only)")
    r.add_argument("--control", action="store_true", help="scripted negative control: honest proposals only")
    r.add_argument("--plant", action="append", metavar="NAME[,NAME...]",
                   help="meta backend only: add scripted red-team attacker lanes for these task-bank behaviors (%s) so the real "
                        "auditor and reviewer have faults to catch; the lanes are labelled as scripted everywhere"
                        % ", ".join(PLANTABLE))
    r.add_argument("--context", action="append", nargs="+", metavar="FILE",
                   help="meta backend only: UTF-8 text or markdown files (background material, for example related-work notes) "
                        "for the supervisor and designer; at most 200,000 characters per file and 400,000 in total. Only the "
                        "name, sha256 and length of each file are recorded in the ledger")
    r.add_argument("--serve", action="store_true", help="serve the live dashboard while the run goes")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--crash-after-runs", type=int, help=argparse.SUPPRESS)

    s = cmd("resume", cmd_resume, "continue a run whose process died before it finished (a finished run cannot be resumed)")
    s.add_argument("run_dir")
    s.add_argument("--task", help="task directory (default: the task named in the ledger, found under tasks/)")
    s.add_argument("--backend", choices=("scripted", "meta"))
    s.add_argument("--config")
    s.add_argument("--implementer", choices=("patch", "muse_code"))
    s.add_argument("--control", action="store_true")
    s.add_argument("--context", action="append", nargs="+", metavar="FILE",
                   help="meta backend only: the SAME background files the run was started with (checked by sha256 against "
                        "run.started; a different set is a usage error). Without it the supervisor and designer continue "
                        "without the background")
    s.add_argument("--accept-broken-chain", action="store_true",
                   help="continue although the ledger's hash chain is broken (at your own risk)")

    v = cmd("serve", cmd_serve, "serve the dashboard for a run (read-only)")
    v.add_argument("run_dir")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--host", help="bind address (default 127.0.0.1; 0.0.0.0 with --allow-remote)")
    grp = v.add_mutually_exclusive_group()
    grp.add_argument("--replay", type=_positive_float, metavar="SPEED", help="replay the ledger at SPEED x (0.1 to 100)")
    grp.add_argument("--replay-duration", type=_positive_float, metavar="SEC", help="replay the whole ledger in SEC seconds")
    v.add_argument("--allow-remote", action="store_true", help="bind a non-loopback address (no authentication!)")

    rp = cmd("report", cmd_report, "write report.md and report.html")
    rp.add_argument("run_dir")
    rp.add_argument("--out", help="output directory (default: the run directory)")

    a = cmd("audit", cmd_audit, "re-audit a run (on a temporary copy unless --write)")
    a.add_argument("run_dir")
    a.add_argument("--judge", choices=("meta", "none"), default="none")
    a.add_argument("--write", action="store_true", help="append the audit events to the real ledger")
    a.add_argument("--task")
    a.add_argument("--config")

    ve = cmd("verify", cmd_verify, "check the hash chain and print the head anchor")
    ve.add_argument("run_dir")
    ve.add_argument("--expect-head", help="head anchor recorded earlier (<seq>:<hash16> or a hash prefix)")

    d = cmd("doctor", cmd_doctor, "check python, numpy, isolation, the muse binary and the API key")
    d.add_argument("--live", action="store_true", help="also run the live API probes (spends a few cents)")
    d.add_argument("--live-out", help="write the live probe report here")

    t = cmd("tasks", cmd_tasks, "list and validate (with calibration) the tasks under ./tasks")
    t.add_argument("--root", help="tasks directory (default ./tasks)")

    de = cmd("demo", cmd_demo, "run the offline scripted demo")
    de.add_argument("--task", default="law-discovery")
    de.add_argument("--serve", action="store_true", help="serve the live dashboard in a background thread")
    de.add_argument("--no-serve", action="store_true", help="do not start the dashboard (the default)")
    de.add_argument("--with-crash", action="store_true", help="kill the first process mid-run, then resume in a second")
    de.add_argument("--control", action="store_true", help="also run the clean negative control (must have zero findings)")
    de.add_argument("--out", default="runs/demo", help="run directory (relative paths are relative to the current directory)")
    de.add_argument("--port", type=int, default=8765)
    de.add_argument("--replay", metavar="DIR", help="serve an existing ledger in replay mode instead of running")
    de.add_argument("--replay-speed", type=_positive_float, default=DEFAULT_REPLAY_SPEED)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns the exit status (``__main__`` passes it to ``SystemExit``)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"error: {secrets.scrub_text(str(exc))}", file=sys.stderr)
        return exc.code
    except (NightshiftError, LedgerLocked) as exc:
        # a corrupt or locked ledger, a refused resume, a budget error: a message and exit 1, never a traceback
        print(f"error: {type(exc).__name__}: {secrets.scrub_text(str(exc))}", file=sys.stderr)
        return EXIT_FAIL
    except BrokenPipeError:
        return EXIT_OK
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
