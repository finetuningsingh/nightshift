"""Task loading, validation and hashing (docs/ARCHITECTURE.md sections 4, 5, 6.1).

Guarantees
----------
* ``load_task`` raises ``ValueError`` naming the file, the table and the key for every malformed ``task.toml``.
* ``validate_task`` returns problem strings of the form ``code: detail`` (empty list = OK). Structural checks never run
  untrusted code; ``calibrate=True`` grades the unmodified template and reports ``seeds_vacuous`` when the seeds do not
  change the score.
* ``calibrate=True`` also estimates the unmodified template's per-seed standard deviation over at least
  ``CALIBRATION_MIN_SEEDS`` seeds on every split the task uses (``estimate_baseline_noise``), states it as the recommended
  ``sigma_floor`` and warns when the task's ``sigma_floor`` or ``noise_margin`` is well below it. These are notes, not
  problems: a task with a low floor is usable, it just has a weaker decision rule (docs/NULL_BENCHMARK.md). The estimate
  is skipped, with a note saying why, when requirements are missing or a baseline grade fails.
* ``trusted_hash`` is 12 hex over the sorted ``(relpath, bytes)`` of ``trusted/`` ignoring ``__pycache__`` and ``*.pyc``.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import math
import os
import re
import statistics
import stat
import sys
import tempfile
import tokenize
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .types import TaskSpec, canonical_json, sha256_hex, short_hash

_TOP_KEYS = {"name", "description", "brief", "objective", "primary_metric", "train_metric", "workspace", "run", "grade"}
_TABLE_KEYS = {
    "workspace": {"editable", "scratch"},
    "run": {"cmd", "timeout_sec", "max_output_bytes", "requires"},
    "grade": {"cmd", "timeout_sec", "seeds", "replication_seeds", "noise_margin", "sigma_floor", "sealed", "requires",
              "stall_sec"},
}
# Calls the trusted grader must never make (contract 4.2). Comments and docstrings are ignored; other strings are not.
_FORBIDDEN = (
    ("pickle", re.compile(r"pickle")),
    ("allow_pickle=True", re.compile(r"allow_pickle\s*=\s*True")),
    ("eval(", re.compile(r"(?<![A-Za-z0-9_])eval\s*\(")),
    ("exec(", re.compile(r"(?<![A-Za-z0-9_])exec\s*\(")),
    ("yaml.load", re.compile(r"yaml\s*\.\s*load\b")),
    ("importlib", re.compile(r"importlib")),
    ("marshal", re.compile(r"\bmarshal\b")),
    ("__import__(", re.compile(r"__import__\s*\(")),
)
_PICKLE_SUFFIXES = {".pkl", ".pickle"}


# --------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------
class _Reader:
    """Typed access to one TOML table with precise error messages."""

    def __init__(self, path: Path, where: str, table: Mapping[str, Any]):
        self.path, self.where, self.table = path, where, table

    def fail(self, key: str, msg: str) -> ValueError:
        loc = f"{self.where}.{key}" if self.where else key
        return ValueError(f"{self.path}: {loc}: {msg}")

    def has(self, key: str) -> bool:
        return key in self.table

    def string(self, key: str, default: Optional[str] = None, *, allow_empty: bool = False) -> str:
        if key not in self.table:
            if default is None:
                raise self.fail(key, "is required")
            return default
        v = self.table[key]
        if not isinstance(v, str) or (not allow_empty and not v.strip()):
            raise self.fail(key, f"must be {'a string' if allow_empty else 'a non-empty string'}, got {v!r}")
        return v

    def number(self, key: str, default: Optional[float] = None, *, minimum: Optional[float] = None,
               positive: bool = False) -> float:
        if key not in self.table:
            if default is None:
                raise self.fail(key, "is required")
            return default
        v = self.table[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise self.fail(key, f"must be a finite number, got {v!r}")
        if positive and v <= 0:
            raise self.fail(key, f"must be > 0, got {v!r}")
        if minimum is not None and v < minimum:
            raise self.fail(key, f"must be >= {minimum:g}, got {v!r}")
        return float(v)

    def boolean(self, key: str, default: bool) -> bool:
        v = self.table.get(key, default)
        if not isinstance(v, bool):
            raise self.fail(key, f"must be true or false, got {v!r}")
        return v

    def strings(self, key: str, default: Optional[list[str]] = None, *, non_empty: bool = False) -> list[str]:
        if key not in self.table:
            if default is None:
                raise self.fail(key, "is required")
            return list(default)
        v = self.table[key]
        if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
            raise self.fail(key, f"must be a list of non-empty strings, got {v!r}")
        if non_empty and not v:
            raise self.fail(key, "must not be empty")
        return list(v)

    def ints(self, key: str, default: Optional[list[int]] = None, *, non_empty: bool = False) -> list[int]:
        if key not in self.table:
            if default is None:
                raise self.fail(key, "is required")
            return list(default)
        v = self.table[key]
        if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            raise self.fail(key, f"must be a list of integers, got {v!r}")
        if non_empty and not v:
            raise self.fail(key, "must not be empty")
        return list(v)


def _table(path: Path, raw: Mapping[str, Any], name: str, required: bool) -> _Reader:
    v = raw.get(name, {} if not required else None)
    if v is None:
        raise ValueError(f"{path}: table [{name}] is required")
    if not isinstance(v, dict):
        raise ValueError(f"{path}: [{name}] must be a table, got {type(v).__name__}")
    return _Reader(path, name, v)


def _check_glob(r: _Reader, key: str, globs: Sequence[str]) -> None:
    for g in globs:
        parts = g.split("/")
        if g.startswith("/") or ".." in parts or "\\" in g:
            raise r.fail(key, f"pattern {g!r} must be workspace-relative with '/' separators and no '..'")


def load_task(path: str | os.PathLike) -> TaskSpec:
    """Parse ``<task_dir>/task.toml`` (``path`` may be the directory or the file) into a ``TaskSpec``."""
    p = Path(path)
    toml_path = p if p.is_file() else p / "task.toml"
    try:
        text = toml_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{toml_path}: cannot read task.toml: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{toml_path}: task.toml is not valid UTF-8") from exc
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{toml_path}: invalid TOML: {exc}") from exc

    task_dir = toml_path.parent
    top = _Reader(toml_path, "", raw)
    ws, run, grade = _table(toml_path, raw, "workspace", True), _table(toml_path, raw, "run", True), \
        _table(toml_path, raw, "grade", True)

    objective = top.string("objective")
    if objective not in ("minimize", "maximize"):
        raise top.fail("objective", f"must be 'minimize' or 'maximize', got {objective!r}")
    editable = ws.strings("editable", non_empty=True)
    scratch = ws.strings("scratch", ["outputs/**"])
    _check_glob(ws, "editable", editable)
    _check_glob(ws, "scratch", scratch)
    requires = grade.strings("requires", run.strings("requires", []))
    extra: dict[str, Any] = {}
    for name, table in (("", raw), *((n, raw.get(n)) for n in _TABLE_KEYS)):
        if not isinstance(table, dict):
            continue
        allowed = _TOP_KEYS if name == "" else _TABLE_KEYS[name]
        for key, value in table.items():
            if key not in allowed:
                extra[f"{name}.{key}" if name else key] = value

    scripted = task_dir / "scripted"
    return TaskSpec(
        name=top.string("name"),
        description=top.string("description", "", allow_empty=True),
        brief=top.string("brief").strip(),
        task_dir=task_dir,
        template_dir=task_dir / "template",
        trusted_dir=task_dir / "trusted",
        objective=objective,
        primary_metric=top.string("primary_metric"),
        train_metric=top.string("train_metric") if top.has("train_metric") else None,
        editable=editable,
        scratch=scratch,
        run_cmd=run.strings("cmd", non_empty=True),
        run_timeout_sec=run.number("timeout_sec", 20.0, positive=True),
        grade_cmd=grade.strings("cmd", non_empty=True),
        grade_timeout_sec=grade.number("timeout_sec", 20.0, positive=True),
        seeds=grade.ints("seeds", non_empty=True),
        replication_seeds=grade.ints("replication_seeds", []),
        noise_margin=grade.number("noise_margin", 0.0, minimum=0.0),
        requires=requires,
        scripted_dir=scripted if scripted.is_dir() else None,
        sealed=grade.boolean("sealed", True),
        sigma_floor=grade.number("sigma_floor", 0.0, minimum=0.0),
        max_output_bytes=int(run.number("max_output_bytes", 8 * 1024 * 1024, positive=True)),
        stall_sec=grade.number("stall_sec", 10.0, positive=True),
        extra=extra,
    )


def list_tasks(root: str | os.PathLike, *, on_error: Optional[Callable[[Path, ValueError], None]] = None) -> list[TaskSpec]:
    """Every ``<root>/<name>/task.toml`` as a ``TaskSpec``, sorted by directory name.

    A malformed task raises ``ValueError`` unless ``on_error`` is given, in which case it is reported there and skipped.
    """
    out: list[TaskSpec] = []
    base = Path(root)
    if not base.is_dir():
        return out
    for child in sorted(base.iterdir(), key=lambda c: c.name):
        if not (child / "task.toml").is_file():
            continue
        try:
            out.append(load_task(child))
        except ValueError as exc:
            if on_error is None:
                raise
            on_error(child, exc)
    return out


# --------------------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------------------
def _manifest(root: Path) -> dict[str, str]:
    """relpath -> sha256 of every file under ``root`` (links recorded by target), ignoring ``__pycache__`` and ``*.pyc``."""
    out: dict[str, str] = {}
    if not root.is_dir():
        return out

    def hash_file(full: str) -> str:
        # O_NONBLOCK + fstat: a FIFO planted in the tree must not hang the hash (open(2) on a FIFO blocks).
        fd = os.open(full, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return "special"
            h = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    return h.hexdigest()
                h.update(chunk)
        finally:
            os.close(fd)

    def walk(abs_dir: str, rel_dir: str) -> None:
        try:
            with os.scandir(abs_dir) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            out[rel_dir or "."] = "unreadable"
            return
        for entry in entries:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            try:
                mode = os.lstat(entry.path).st_mode
                if stat.S_ISDIR(mode):
                    if entry.name != "__pycache__":
                        walk(entry.path, rel)
                elif entry.name.endswith(".pyc"):
                    continue
                elif stat.S_ISLNK(mode):             # a link (to a file OR a directory) is recorded by its target
                    out[rel] = "link:" + os.readlink(entry.path)
                elif stat.S_ISREG(mode):
                    out[rel] = hash_file(entry.path)
                else:
                    out[rel] = "special"
            except OSError:
                out[rel] = "unreadable"

    walk(os.fspath(root), "")
    return out


def trusted_manifest(task: TaskSpec) -> dict[str, str]:
    """relpath -> sha256 for the trusted directory (same filter as ``trusted_hash``); used to name tampered files."""
    return _manifest(Path(task.trusted_dir))


def manifest_hash(manifest: Mapping[str, str]) -> str:
    return short_hash(canonical_json(sorted(manifest.items())))


def trusted_hash(task: TaskSpec) -> str:
    """12 hex over sorted (relpath, bytes) of ``trusted/``, ignoring ``__pycache__`` and ``*.pyc``."""
    return manifest_hash(trusted_manifest(task))


def task_hash(task: TaskSpec) -> str:
    """12 hex over ``task.toml``, the template tree, the trusted tree and (if present) the scripted bank."""
    toml_path = Path(task.task_dir) / "task.toml"
    try:
        spec: Any = sha256_hex(toml_path.read_bytes())
    except OSError:
        spec = [task.name, task.objective, task.primary_metric, task.train_metric, task.editable, task.scratch, task.run_cmd,
                task.run_timeout_sec, task.grade_cmd, task.grade_timeout_sec, task.seeds, task.replication_seeds,
                task.noise_margin, task.sigma_floor, task.sealed, task.requires, task.stall_sec, task.max_output_bytes]
    parts = {"spec": spec, "template": sorted(_manifest(Path(task.template_dir)).items()), "trusted": trusted_hash(task),
             "scripted": sorted(_manifest(Path(task.scripted_dir)).items()) if task.scripted_dir else []}
    return short_hash(canonical_json(parts))


def conditions_hash(task: TaskSpec, seeds: Sequence[int], split: str = "dev", *, grader_hash: Optional[str] = None) -> str:
    """``sha256(task_hash | grader_hash | split | seeds | run_cmd)[:12]``. Equal hashes mean comparable grades.

    ``grader_hash`` defaults to ``trusted_hash(task)``; pass it when it was already computed for the same evaluation."""
    g = trusted_hash(task) if grader_hash is None else grader_hash
    return short_hash("|".join([task_hash(task), g, split, canonical_json([int(s) for s in seeds]), canonical_json(task.run_cmd)]))


def missing_requirements(task: TaskSpec) -> list[str]:
    """Required modules that cannot be found (their top-level package is looked up without importing it)."""
    missing = []
    for mod in task.requires:
        try:
            found = importlib.util.find_spec(mod.split(".")[0]) is not None
        except (ImportError, ValueError, AttributeError):
            found = False
        if not found:
            missing.append(mod)
    return missing


# --------------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------------
def _comment_free(source: str) -> str:
    try:
        keep = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(keep)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source


def _code_for_scan(source: str) -> str:
    """The code of ``source`` without comments and docstrings, so prose such as "never use pickle" is not a finding."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _comment_free(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _forbidden_calls(trusted_dir: Path) -> list[str]:
    problems = []
    for dirpath, dirnames, filenames in os.walk(trusted_dir, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(trusted_dir).as_posix()
            if full.suffix in _PICKLE_SUFFIXES:
                problems.append(f"trusted_forbidden_file: trusted/{rel} is a pickle")
            if full.suffix != ".py":
                continue
            try:
                source = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                problems.append(f"trusted_unreadable: trusted/{rel}")
                continue
            code = _code_for_scan(source)
            for label, pattern in _FORBIDDEN:
                if pattern.search(code):
                    problems.append(f"forbidden_call: trusted/{rel} uses {label}")
    return problems


def _has_glob_chars(s: str) -> bool:
    return any(c in s for c in "*?[")


CALIBRATION_MIN_SEEDS = 12           # never estimate the baseline's per-seed std from fewer seeds (a 3-seed std is far too noisy)
CALIBRATION_SEEDS = 20               # seeds `validate_task(calibrate=True)` uses: the std is then known to about +-16%
CALIBRATION_FILLER_START = 1000      # extra calibration seeds are 1000, 1001, ... skipping the task's own seeds
FLOOR_WARN_RATIO = 0.75              # sigma_floor below this share of the measured std is "well below"
MARGIN_WARN_RATIO = 0.5              # noise_margin below this share of the measured std is "well below"


@dataclass(frozen=True)
class NoiseEstimate:
    """The unmodified template's per-seed scores on one split: ``std`` is the sample standard deviation (ddof 1)."""

    split: str
    n: int
    mean: float
    std: float
    seeds: tuple[int, ...]


def _calibration_seeds(task: TaskSpec, own: Sequence[int], n_seeds: int) -> list[int]:
    """``own`` (the task's seeds for the split) topped up with fresh seeds that no list of the task uses, to ``n_seeds``."""
    taken = set(task.seeds) | set(task.replication_seeds)
    out = [int(s) for s in own]
    nxt = CALIBRATION_FILLER_START
    while len(out) < n_seeds:
        if nxt not in taken:
            out.append(nxt)
        nxt += 1
    return out


def recommended_sigma_floor(std: float) -> float:
    """``std`` rounded to two significant digits: the value to write into ``sigma_floor``."""
    return float(f"{std:.2g}") if std > 0 else 0.0


def estimate_baseline_noise(task: TaskSpec, *, n_seeds: int = CALIBRATION_SEEDS) -> list[NoiseEstimate]:
    """Per-seed std of the UNMODIFIED template over ``max(n_seeds, CALIBRATION_MIN_SEEDS)`` seeds, on the dev split and,
    for a sealed task, on the sealed split.

    The estimate uses the task's own seeds first and tops up with fresh seeds, through ``grading.evaluate`` (so the
    same sandbox and grader as a run). One baseline evaluation per seed and split: a few seconds. The task's required
    modules must be importable (``missing_requirements``). Raises ``RuntimeError`` naming the split when a grade fails.
    """
    n = max(int(n_seeds), CALIBRATION_MIN_SEEDS)
    from . import grading, sandbox        # lazy: grading imports this module
    out: list[NoiseEstimate] = []
    with tempfile.TemporaryDirectory(prefix="nightshift-noise-") as tmp:
        run_dir = Path(tmp)
        ws = sandbox.create_workspace(task, run_dir, "noise")
        plans = [("baseline", task.seeds, "dev")]
        if task.sealed:
            plans.append(("replication_baseline", task.replication_seeds, "sealed"))
        start = 0
        for purpose, own, split in plans:
            seeds = _calibration_seeds(task, own, n)
            res = grading.evaluate(task, ws, seeds, purpose=purpose, run_dir=run_dir, expected_tree_hash=ws.tree_hash,
                                   isolation="none", run_idx_start=start)
            start += len(seeds)
            if not res.grade.valid:
                raise RuntimeError(f"{split} grading of the template failed ({res.failure_class}: "
                                   f"{(res.grade.error or '')[:200]})")
            scores = [float(x) for x in res.grade.per_seed]
            out.append(NoiseEstimate(split=split, n=len(scores), mean=statistics.fmean(scores),
                                     std=statistics.stdev(scores), seeds=tuple(seeds)))
    return out


def calibration_notes(task: TaskSpec, estimates: Sequence[NoiseEstimate]) -> list[str]:
    """Human-readable lines: the measured std per split, the recommended ``sigma_floor`` and warnings.

    The recommendation is the DEV std (it is the statistical prior of every comparison: the baseline's dev per-seed std).
    A warning is issued when ``sigma_floor`` is below ``FLOOR_WARN_RATIO`` of it (the prior is then inert and the dev gate
    falls back to ``noise_margin`` plus a bare t-test on the few seeds) or when ``noise_margin`` is below
    ``MARGIN_WARN_RATIO`` of it.
    """
    if not estimates:
        return []
    dev = next((e for e in estimates if e.split == "dev"), estimates[0])
    parts = ", ".join(f"{e.split} {e.std:.4f}" for e in estimates)
    rec = recommended_sigma_floor(dev.std)
    error = 100.0 / math.sqrt(2 * max(dev.n - 1, 1))          # standard error of a sample std of normal data, in percent
    notes = [f"calibration [{task.name}]: baseline per-seed std over {dev.n} seeds: {parts} (sampling error about +-{error:.0f}%). "
             f"Recommended sigma_floor = {rec:g} (task.toml has {task.sigma_floor:g}); noise_margin is {task.noise_margin:g}."]
    if rec > 0 and task.sigma_floor < FLOOR_WARN_RATIO * rec:
        notes.append(f"warning [{task.name}]: sigma_floor {task.sigma_floor:g} is well below the measured baseline std {rec:g}, so the "
                     "noise prior is inert: the dev margin falls back to noise_margin and, for a candidate whose paired noise is "
                     f"large, the dev gate approaches a plain 5% one-sided t-test on {len(task.seeds)} seeds (docs/NULL_BENCHMARK.md). "
                     f"Set sigma_floor = {rec:g}.")
    if rec > 0 and task.noise_margin < MARGIN_WARN_RATIO * rec:
        notes.append(f"warning [{task.name}]: noise_margin {task.noise_margin:g} is well below the measured baseline std {rec:g}: "
                     "an improvement smaller than the run-to-run noise could be reported. Raise it unless smaller effects "
                     "matter and you accept that.")
    return notes


def _calibrate(task: TaskSpec) -> list[str]:
    from . import grading, sandbox        # lazy: grading imports this module
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="nightshift-calibrate-") as tmp:
        run_dir = Path(tmp)
        ws = sandbox.create_workspace(task, run_dir, "calib")
        plans = [("baseline", task.seeds, "dev")]
        if task.sealed:
            plans.append(("replication_baseline", task.replication_seeds, "sealed"))
        for purpose, seeds, split in plans:
            try:
                res = grading.evaluate(task, ws, seeds, purpose=purpose, run_dir=run_dir,
                                       expected_tree_hash=ws.tree_hash, isolation="none",
                                       run_idx_start=0 if split == "dev" else len(task.seeds))
            except Exception as exc:  # noqa: BLE001 - a calibration failure is a reportable problem, not a crash
                problems.append(f"calibration_failed: {split} grading raised {type(exc).__name__}: {str(exc)[:200]}")
                continue
            if not res.grade.valid:
                problems.append(f"calibration_failed: {split} grading of the template failed "
                                f"({res.failure_class}: {(res.grade.error or '')[:200]})")
            elif len(set(res.grade.per_seed)) <= 1:
                problems.append(f"seeds_vacuous: per-seed {split} scores of the unmodified template are all equal "
                                f"({res.grade.per_seed[0]!r}); seeds must change the training sample and the evaluated rows")
    return problems


def _noise_notes(task: TaskSpec) -> list[str]:
    """The calibration notes for ``task``, or one note saying why the noise estimate was skipped."""
    try:
        return calibration_notes(task, estimate_baseline_noise(task))
    except Exception as exc:  # noqa: BLE001 - the estimate is advice; a failure must not fail validation
        return [f"calibration [{task.name}]: the baseline noise estimate was skipped ({type(exc).__name__}: {str(exc)[:200]})"]


def validate_task(task: TaskSpec, *, calibrate: bool = False, notes: Optional[list[str]] = None) -> list[str]:
    """Problems with ``task`` as ``code: detail`` strings; an empty list means the task is usable.

    Structural checks always run. ``calibrate=True`` additionally grades the unmodified template on all seeds (and on the
    replication seeds when sealed) through ``grading.evaluate`` and reports ``seeds_vacuous`` when the per-seed scores
    are all equal. Calibration is skipped when structural problems exist or required modules are missing.

    A calibrated, otherwise valid task also gets its baseline per-seed std estimated over at least
    ``CALIBRATION_MIN_SEEDS`` seeds (``estimate_baseline_noise``): the recommended ``sigma_floor`` and any warning are
    appended to ``notes`` when a list is given and printed to stdout, one line each, when it is not. They are advice
    and never problems. A baseline grade that fails here is a note, not a problem (the checks above already reported it).
    """
    problems: list[str] = []
    template, trusted = Path(task.template_dir), Path(task.trusted_dir)
    if not template.is_dir():
        problems.append(f"template_dir_missing: {template}")
    if not trusted.is_dir():
        problems.append(f"trusted_dir_missing: {trusted}")

    for key in sorted(task.extra):
        problems.append(f"unknown_key: {key} in task.toml is not part of the task format")
    if re.search(r"\b(seeds?|sealed)\b", task.brief, re.IGNORECASE):
        problems.append("brief_mentions_seed_or_sealed: the brief is shown to models and must not mention seeds or the sealed split")
    if task.train_metric and task.train_metric == task.primary_metric:
        problems.append("train_metric_equals_primary: train_metric must differ from primary_metric")

    if template.is_dir():
        files = sorted(p.relative_to(template).as_posix() for p in template.rglob("*") if p.is_file() or p.is_symlink())
        if any("__pycache__" in f.split("/") or f.endswith(".pyc") for f in files):
            problems.append("template_has_bytecode: template/ contains __pycache__ or .pyc files; delete them")
        for g in task.editable:
            if not _has_glob_chars(g) and g not in files:
                problems.append(f"editable_missing: {g} is editable but absent from template/")
        if task.run_cmd[:1] and Path(task.run_cmd[0]).name.startswith("python") and len(task.run_cmd) > 1 \
                and task.run_cmd[1].endswith(".py") and task.run_cmd[1] not in files:
            problems.append(f"run_script_missing: {task.run_cmd[1]} is not in template/")
        for link in (p for p in template.rglob("*") if p.is_symlink()):
            problems.append(f"template_symlink: template/{link.relative_to(template).as_posix()} is a symlink")
        from . import sandbox     # lazy: keeps this module importable without the sandbox
        try:
            _, special = sandbox.snapshot_hashes(template, task)
        except OSError as exc:
            special = [f"special:. ({exc.strerror or exc})"]
        for item in special:      # every workspace is a copy of the template, so one of these fails every experiment
            if not os.path.islink(template / item[len("special:"):]):
                problems.append(f"template_special: {item} (hard link, FIFO, unsafe file name or over the size/count caps)")

    if not task.run_cmd:
        problems.append("run_cmd_empty: [run] cmd is empty")
    for token in ("{workspace}", "{seed}"):
        if token not in task.grade_cmd:
            problems.append(f"grade_cmd_missing_token: grade cmd must contain {token}")
    if task.sealed and "{split}" not in task.grade_cmd:
        problems.append("grade_cmd_missing_token: a sealed task's grade cmd must contain {split}")
    script_tokens = [t for t in task.grade_cmd if "{task_dir}" in t]
    if not script_tokens:
        problems.append("grader_missing: grade cmd must run a script under {task_dir}/trusted/")
    else:
        target = Path(script_tokens[0].replace("{task_dir}", str(task.task_dir)))
        if not target.is_file():
            problems.append(f"grader_missing: {script_tokens[0]} does not exist")
        else:
            try:
                target.resolve().relative_to(trusted.resolve())
            except ValueError:
                problems.append("grader_outside_trusted: the grader script must live under trusted/")

    if len(task.seeds) < 3:
        problems.append(f"too_few_seeds: need at least 3 seeds, have {len(task.seeds)}")
    if len(task.replication_seeds) < 3:
        problems.append(f"too_few_replication_seeds: need at least 3 replication seeds, have {len(task.replication_seeds)}")
    if len(set(task.seeds)) != len(task.seeds) or len(set(task.replication_seeds)) != len(task.replication_seeds):
        problems.append("duplicate_seeds: a seed list repeats a value")
    overlap = sorted(set(task.seeds) & set(task.replication_seeds))
    if overlap:
        problems.append(f"seeds_not_disjoint: seeds and replication_seeds share {overlap}")
    if task.sealed and task.noise_margin <= 0:
        problems.append("noise_margin_not_positive: noise_margin must be > 0 when the task is sealed")
    if task.objective not in ("minimize", "maximize"):
        problems.append(f"bad_objective: {task.objective!r}")
    for what, value in (("run timeout_sec", task.run_timeout_sec), ("grade timeout_sec", task.grade_timeout_sec),
                        ("stall_sec", task.stall_sec)):
        if not value > 0:
            problems.append(f"bad_timeout: {what} must be > 0")

    if trusted.is_dir():
        problems.extend(_forbidden_calls(trusted))

    if calibrate and not problems:
        missing = missing_requirements(task)
        if missing:
            problems.append(f"missing_requirements: {', '.join(missing)} (calibration skipped)")
        else:
            cal = _calibrate(task)
            problems.extend(cal)
            if not cal:
                lines = _noise_notes(task)
                if notes is not None:
                    notes.extend(lines)
                else:
                    for line in lines:
                        print(line, file=sys.stdout, flush=True)
    return problems
