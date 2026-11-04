"""Researcher-pack git operations: init, checkout, strict apply, lint, accept, hashes (owner 4).

A pack is a directory of text files in a bare git repository (``layout.pack_repo``). Versions are commits. ``main`` is linear and holds
accepted versions only; every candidate commit is tagged ``pack/<version_label>`` (``pack/v0``, ``pack/cand-001``, ...) whether or not it
is accepted. No ref is ever deleted or moved backwards and no history is rewritten.

Everything is done with git plumbing against the bare repository (``hash-object``, ``mktree``, ``commit-tree``, ``update-ref``): no
working tree, no index, no hooks, a fixed author and a clock-derived commit date, and a scrubbed environment, so the commit ids depend
only on the content, the parent and the clock.

* ``apply_proposal_edits`` is the STRICT apply (pure, used by ``PackRepo.apply`` and by the in-memory test store): every edited path must
  be in the editable allowlist and not protected, ``search_replace`` blocks must each match exactly once and not overlap (OpenEvolve's
  parser silently skips a block that does not match; this one rejects the proposal), sizes are capped, files are text.
* ``PackRepo.lint`` = structure checks + ``compare_task_keys`` over JSON dumps of the parent's and the candidate's ``task.yaml``. The dump
  is made by a ``TaskDumper`` (default: the strict reader of ``assets/lint_pack.py`` in this process; in production pass a
  ``SubprocessDumper`` that runs the same script inside CORAL's virtual environment, which also asks CORAL to validate the file).
  ``compare_task_keys`` is pure and tested on JSON fixtures.
* The pack contract (which keys are editable and in which ranges) is ``lint_pack.DEFAULT_LINT_RANGES``; ``init_v0`` writes it to
  ``lint_ranges_path`` if that file does not exist yet. It must not change after the first candidate (an ``OperatorChange``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import time
import types
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional, Sequence

from nightshift.lab.contracts import (
    Clock,
    ContractError,
    EditOp,
    LabLayout,
    LintResult,
    PackRef,
    Proposal,
    ProposalInvalid,
    loads_strict,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LINT_SCRIPT = ASSETS_DIR / "lint_pack.py"
PACK_V0_TEMPLATE = ASSETS_DIR / "pack_v0"
ENTRYPOINT_JSON = ASSETS_DIR / "coral_grader" / "ENTRYPOINT.json"

# Paths (relative to the pack root) the outer loop may edit. An entry ending in "/" matches every file below it.
DEFAULT_EDITABLE_PATHS = ("task.yaml", "notes/", "skills/", "roles/", "seed/")
# Names that carry the provider route, the gateway or CORAL's own state (decision D9: the pack holds text levers only).
PROTECTED_BASENAMES = frozenset({"opencode.json", "opencode.jsonc", "litellm_config.yaml", "AGENTS.md", "CLAUDE.md"})
PROTECTED_DIRS = frozenset({".git", ".opencode", ".coral", ".claude", ".codex"})

# Context limits, not safeguards: the whole pack goes into the proposer's prompt.
MAX_FILE_BYTES = 128 * 1024
MAX_PACK_BYTES = 1024 * 1024
MAX_PACK_FILES = 200

_REGULAR_MODES = ("100644", "100755")
_MSG = 500  # LintResult messages are limited to 500 bytes


# --------------------------------------------------------------------------------------------------
# the lint script, loaded by path (it is also run as a script inside CORAL's virtual environment)
# --------------------------------------------------------------------------------------------------
_lint_module: Any = None


def _lint() -> Any:
    global _lint_module
    if _lint_module is None:
        # executed from source (no import machinery): no bytecode cache is written into the packaged assets directory
        mod = types.ModuleType("nightshift_lab_lint_pack")
        mod.__file__ = str(LINT_SCRIPT)
        exec(compile(LINT_SCRIPT.read_text(encoding="utf-8"), str(LINT_SCRIPT), "exec"), mod.__dict__)
        _lint_module = mod
    return _lint_module


def default_lint_ranges() -> dict:
    return json.loads(json.dumps(_lint().DEFAULT_LINT_RANGES))


def parse_task_yaml(text: str) -> dict:
    """The strict reader of ``lint_pack.py`` (the pack's task.yaml format). Raises ``ValueError`` (``YamlSubsetError``)."""
    return _lint().parse_strict_yaml(text)


# --------------------------------------------------------------------------------------------------
# paths, editable allowlist
# --------------------------------------------------------------------------------------------------
def check_pack_path(path: str) -> str:
    """A normalised relative POSIX path with no '..', '.git', empty or backslash part. Raises ProposalInvalid."""
    p = PurePosixPath(path)
    if not path or "\\" in path or p.is_absolute() or str(p) != path or any(part in ("", ".", "..") for part in p.parts):
        raise ProposalInvalid(f"path {path!r} is not a normalised relative path")
    return path


def is_protected(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return PurePosixPath(path).name in PROTECTED_BASENAMES or any(part in PROTECTED_DIRS for part in parts)


def make_is_editable(editable_paths: Sequence[str]) -> Callable[[str], bool]:
    entries = tuple(editable_paths)
    for e in entries:
        if not e or e.startswith("/") or ".." in e.split("/"):
            raise ContractError(f"editable path {e!r} must be a relative path")

    def is_editable(path: str) -> bool:
        if is_protected(path):
            return False
        return any(path.startswith(e) if e.endswith("/") else path == e for e in entries)

    return is_editable


# --------------------------------------------------------------------------------------------------
# strict apply (pure)
# --------------------------------------------------------------------------------------------------
def _apply_blocks(path: str, text: str, blocks: Sequence[Any]) -> str:
    spans: list[tuple[int, int, str]] = []
    for i, b in enumerate(blocks):
        n = text.count(b.search)
        if n != 1:
            raise ProposalInvalid(f"{path}: search block {i + 1} matches {n} times in the file (it must match exactly once)")
        start = text.index(b.search)
        spans.append((start, start + len(b.search), b.replace))
    spans.sort()
    for (s1, e1, _), (s2, _e2, _) in zip(spans, spans[1:]):
        if s2 < e1:
            raise ProposalInvalid(f"{path}: search blocks overlap")
    for start, end, repl in reversed(spans):
        text = text[:start] + repl + text[end:]
    return text


def apply_proposal_edits(
    files: Mapping[str, str],
    proposal: Proposal,
    *,
    is_editable: Callable[[str], bool],
    max_file_bytes: int = MAX_FILE_BYTES,
    max_pack_bytes: int = MAX_PACK_BYTES,
    max_pack_files: int = MAX_PACK_FILES,
) -> dict[str, str]:
    """The files of the new version (path -> text) after a strict apply of ``proposal`` to ``files``.

    Raises ProposalInvalid for: a path outside the editable allowlist or on the protected list, a ``search_replace`` block that does not
    match exactly once (or two that overlap), ``search_replace`` on a file that does not exist, a file or pack above the size caps,
    a file/directory name clash, and a proposal that changes nothing. Spends nothing.
    """
    new = dict(files)
    for edit in proposal.edits:
        path = check_pack_path(edit.path)
        if not is_editable(path):
            raise ProposalInvalid(f"{path}: not an editable pack path")
        if edit.op is EditOp.REPLACE_FILE:
            text = edit.content or ""
        else:
            if path not in new:
                raise ProposalInvalid(f"{path}: search_replace on a file that does not exist")
            text = _apply_blocks(path, new[path], edit.blocks)
        size = len(text.encode("utf-8"))
        if size > max_file_bytes:
            raise ProposalInvalid(f"{path}: {size} bytes exceeds the {max_file_bytes}-byte file limit")
        new[path] = text
    if new == dict(files):
        raise ProposalInvalid("the proposal changes nothing")
    paths = sorted(new)
    for p in paths:
        if any(p.startswith(q + "/") for q in paths):
            raise ProposalInvalid(f"{p}: a file cannot also be a directory")
    if len(new) > max_pack_files:
        raise ProposalInvalid(f"the pack would have {len(new)} files (limit {max_pack_files})")
    total = sum(len(t.encode("utf-8")) for t in new.values())
    if total > max_pack_bytes:
        raise ProposalInvalid(f"the pack would be {total} bytes (limit {max_pack_bytes})")
    return new


# --------------------------------------------------------------------------------------------------
# task.yaml key comparison (pure; the lint's second step)
# --------------------------------------------------------------------------------------------------
_MISSING = object()


def _escape_key(k: Any) -> str:
    """A mapping key as one path segment: a literal ``.`` (or ``\\``) inside a key is escaped, so a root-level ``grader.direction`` key can
    never collide with the nested ``grader: {direction: ...}`` path. A non-string key keeps its type in the segment."""
    text = k if isinstance(k, str) else f"{type(k).__name__}:{k}"
    return text.replace("\\", "\\\\").replace(".", "\\.")


def _leaves(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict) and obj:
        out: dict[str, Any] = {}
        for k in obj:
            seg = _escape_key(k)
            out.update(_leaves(obj[k], f"{prefix}.{seg}" if prefix else seg))
        return out
    return {prefix: obj}


def _under(path: str, key: str) -> bool:
    return path == key or path.startswith(key + ".")


def _same(a: Any, b: Any) -> bool:
    if a is _MISSING or b is _MISSING:
        return a is b
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _short(v: Any) -> str:
    s = "(absent)" if v is _MISSING else json.dumps(v, sort_keys=True)
    return s if len(s) <= 60 else s[:57] + "..."


def _get(cfg: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _in_range(v: Any, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def check_editable_values(cfg: Mapping[str, Any], ranges: Mapping[str, Any]) -> list[str]:
    """The editable keys of a task.yaml dump are inside the pack contract's ranges. Returns messages (empty = fine)."""
    msgs: list[str] = []
    cap = ranges["text_max_chars"]
    for key in ("task.description", "task.tips"):
        v = _get(cfg, key)
        if v is _MISSING:
            if key == "task.description":
                msgs.append("task.description is required")
            continue
        if not isinstance(v, str) or (key == "task.description" and not v.strip()):
            msgs.append(f"{key} must be a non-empty string" if key == "task.description" else f"{key} must be a string")
        elif len(v) > cap:
            msgs.append(f"{key} has {len(v)} characters (limit {cap})")
    for key, (lo, hi) in ranges["int_ranges"].items():
        v = _get(cfg, key)
        if v is not _MISSING and not (isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi):
            msgs.append(f"{key}={_short(v)} is outside the allowed range {lo}..{hi}")

    hb = _get(cfg, "agents.heartbeat")
    spec = ranges["heartbeat"]
    if hb is not _MISSING:
        if not isinstance(hb, list) or len(hb) > spec["max_entries"]:
            msgs.append(f"agents.heartbeat must be a list of at most {spec['max_entries']} entries")
        else:
            for i, h in enumerate(hb):
                where = f"agents.heartbeat[{i}]"
                allowed = {"name", "every", "global", "is_global", "trigger", "prompt", "options"}
                if not isinstance(h, dict) or set(h) - allowed:
                    msgs.append(f"{where}: unknown keys {sorted(set(h) - allowed) if isinstance(h, dict) else '(not an object)'}")
                    continue
                if h.get("name") not in spec["names"]:
                    msgs.append(f"{where}.name={_short(h.get('name'))} is not one of {spec['names']}")
                if not _in_range(h.get("every"), *spec["every"]) or not isinstance(h.get("every"), int):
                    msgs.append(f"{where}.every={_short(h.get('every'))} is outside {spec['every'][0]}..{spec['every'][1]}")
                if h.get("trigger", "interval") not in spec["triggers"]:
                    msgs.append(f"{where}.trigger={_short(h.get('trigger'))} is not one of {spec['triggers']}")
                if not isinstance(h.get("global", h.get("is_global", False)), bool):
                    msgs.append(f"{where}.global must be true or false")
                prompt = h.get("prompt", "")
                if not isinstance(prompt, str) or len(prompt) > spec["prompt_max_chars"]:
                    msgs.append(f"{where}.prompt must be a string of at most {spec['prompt_max_chars']} characters")
                opts = h.get("options", {})
                if not isinstance(opts, dict) or set(opts) - set(spec["option_keys"]):
                    msgs.append(f"{where}.options may only use {spec['option_keys']}")

    asg = _get(cfg, "agents.assignments")
    aspec = ranges["assignments"]
    if asg is not _MISSING:
        runtime, model = _get(cfg, "agents.runtime"), _get(cfg, "agents.model")
        if not isinstance(asg, list) or len(asg) > aspec["max_entries"]:
            msgs.append(f"agents.assignments must be a list of at most {aspec['max_entries']} entries")
        else:
            total = 0
            for i, a in enumerate(asg):
                where = f"agents.assignments[{i}]"
                if not isinstance(a, dict) or set(a) - set(aspec["keys"]):
                    msgs.append(f"{where}: allowed keys are {aspec['keys']}")
                    continue
                if a.get("runtime", "") not in ("", runtime):
                    msgs.append(f"{where}.runtime must stay empty or equal agents.runtime")
                if a.get("model", "") not in ("", model):
                    msgs.append(f"{where}.model must stay empty or equal agents.model")
                cnt = a.get("count", 1)
                if not (isinstance(cnt, int) and not isinstance(cnt, bool) and aspec["count"][0] <= cnt <= aspec["count"][1]):
                    msgs.append(f"{where}.count={_short(cnt)} is outside {aspec['count'][0]}..{aspec['count'][1]}")
                else:
                    total += cnt
                ro = a.get("runtime_options", {})
                if not isinstance(ro, dict) or set(ro) - {"role_file"}:
                    msgs.append(f"{where}.runtime_options may only carry role_file")
                elif "role_file" in ro and not _below(ro["role_file"], aspec["role_dir"]):
                    msgs.append(f"{where}.runtime_options.role_file must be a file below {aspec['role_dir']}")
            lo, hi = ranges["int_ranges"].get("agents.count", [1, 1 << 30])
            if total > hi:
                msgs.append(f"agents.assignments add up to {total} agents (limit {hi})")

    skills = _get(cfg, "agents.skills")
    sspec = ranges["skills"]
    if skills is not _MISSING:
        if not isinstance(skills, list) or len(skills) > sspec["max_entries"]:
            msgs.append(f"agents.skills must be a list of at most {sspec['max_entries']} entries")
        else:
            for i, s in enumerate(skills):
                if not _below(s, sspec["skill_dir"]):
                    msgs.append(f"agents.skills[{i}] must be a path below {sspec['skill_dir']}")
    return msgs


def _below(path: Any, directory: str) -> bool:
    if not isinstance(path, str) or not path.startswith(directory) or path == directory:
        return False
    return ".." not in path.split("/") and "\\" not in path and not path.startswith("/")


def compare_task_keys(parent_json: Optional[Mapping[str, Any]], candidate_json: Mapping[str, Any], lint_ranges: Mapping[str, Any]) -> list[str]:
    """Messages (empty = fine) for a candidate ``task.yaml`` dump against its parent's.

    Every key that is not under an editable key of the ranges must be equal to the parent's (presence included): grader.*, the model
    and runtime, the gateway, ``agents.research``, workspace, run, sharing and any key the parent does not have. The editable keys
    are checked against the ranges instead (``check_editable_values``). With no parent only the ranges are checked.
    """
    msgs: list[str] = []
    editable = tuple(lint_ranges["editable_keys"])
    if parent_json is not None:
        pmap, cmap = _leaves(parent_json), _leaves(candidate_json)
        for key in sorted(set(pmap) | set(cmap)):
            if any(_under(key, e) for e in editable):
                continue
            a, b = pmap.get(key, _MISSING), cmap.get(key, _MISSING)
            # an editable key whose parent value is a dict leaf of its own (e.g. an empty mapping) is compared by its ancestors below
            if not _same(a, b):
                msgs.append(f"{key} is not editable: parent {_short(a)}, candidate {_short(b)}")
    msgs.extend(check_editable_values(candidate_json, lint_ranges))
    return [m if len(m.encode("utf-8")) <= _MSG else m.encode("utf-8")[: _MSG - 3].decode("utf-8", "ignore") + "..." for m in msgs]


def compare_grader_values(candidate_json: Mapping[str, Any], entrypoint: Mapping[str, Any]) -> list[str]:
    """The grader section equals the runtime's ``ENTRYPOINT.json`` (keys entrypoint, setup, timeout)."""
    msgs = []
    for key in ("entrypoint", "setup", "timeout"):
        want, got = entrypoint.get(key, _MISSING), _get(candidate_json, f"grader.{key}")
        if not _same(want, got):
            msgs.append(f"grader.{key} must equal the runtime's value {_short(want)}, got {_short(got)}")
    return msgs


# --------------------------------------------------------------------------------------------------
# task.yaml dumpers
# --------------------------------------------------------------------------------------------------
TaskDumper = Callable[[Path], Mapping[str, Any]]  # task.yaml path -> {"config": {...}, "coral_validated": bool}; raises ContractError


def strict_dumper(path: Path) -> dict:
    """Dump with the strict reader in this process (no CORAL): syntax and the pack contract, not CORAL's own validation."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        config = parse_task_yaml(text)
    except (ValueError, OSError) as exc:
        raise ContractError(f"task.yaml: {exc}") from None
    if "preset" in config:
        raise ContractError("task.yaml: 'preset' is not allowed in a pack")
    return {"config": config, "coral_validated": False}


class SubprocessDumper:
    """Run ``<python_argv> -I lint_pack.py --dump <task.yaml>``; with CORAL's virtual-environment interpreter this also asks CORAL to
    validate the file. ``python_argv`` may be a wrapper (for example one that copies the file to the researcher host and runs it there)."""

    def __init__(self, python_argv: Sequence[str], *, script: Path = LINT_SCRIPT, timeout: float = 120.0):
        self.python_argv, self.script, self.timeout = tuple(python_argv), Path(script), timeout

    def __call__(self, path: Path) -> dict:
        try:
            proc = subprocess.run(
                [*self.python_argv, "-I", str(self.script), "--dump", str(path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContractError(f"task.yaml dump failed: {type(exc).__name__}") from None
        if proc.returncode != 0:
            raise ContractError(("task.yaml: " + proc.stderr.strip().splitlines()[-1]) if proc.stderr.strip() else "task.yaml dump failed")
        try:
            doc = json.loads(proc.stdout)
            if not isinstance(doc, dict) or not isinstance(doc.get("config"), dict):
                raise ValueError("no config")
        except ValueError:
            raise ContractError("task.yaml dump printed no JSON object") from None
        return doc


# --------------------------------------------------------------------------------------------------
# hashes
# --------------------------------------------------------------------------------------------------
def listing_sha256(files: Mapping[str, bytes]) -> str:
    """sha256 over the sorted listing ``<sha256 of content> <size> <path>`` (the form of the initial-context listing)."""
    lines = [f"{hashlib.sha256(files[p]).hexdigest()} {len(files[p])} {p}\n" for p in sorted(files)]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def dir_listing_sha256(root: str | os.PathLike[str]) -> str:
    """``listing_sha256`` of a directory on disk: regular files only (symbolic links and special files raise ContractError)."""
    base = Path(root)
    files: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(base):
        for name in dirnames + filenames:
            full = Path(dirpath) / name
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                raise ContractError(f"{full}: symbolic links are refused")
            if stat.S_ISREG(st.st_mode):
                files[full.relative_to(base).as_posix()] = full.read_bytes()
            elif not stat.S_ISDIR(st.st_mode):
                raise ContractError(f"{full}: not a regular file")
    return listing_sha256(files)


# --------------------------------------------------------------------------------------------------
# the repository
# --------------------------------------------------------------------------------------------------
class PackRepo:
    """``PackStore`` over the bare repository ``layout.pack_repo``. Thread-safe for concurrent reads; writers are serialised."""

    def __init__(
        self,
        layout: LabLayout,
        *,
        editable_paths: Sequence[str] = DEFAULT_EDITABLE_PATHS,
        lint_ranges_path: str | os.PathLike[str],
        task_dumper: Optional[TaskDumper] = None,
        entrypoint_json: Optional[str | os.PathLike[str]] = None,
        clock: Clock = time.time,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_pack_bytes: int = MAX_PACK_BYTES,
        max_pack_files: int = MAX_PACK_FILES,
    ):
        self.layout = layout
        self.repo = layout.pack_repo
        self.repo_rel = layout.pack_repo.relative_to(layout.root).as_posix()
        self.editable_paths = tuple(editable_paths)
        self._is_editable = make_is_editable(self.editable_paths)
        self.lint_ranges_path = Path(lint_ranges_path)
        self._dump = task_dumper or strict_dumper
        if entrypoint_json is not None:
            self.entrypoint_json: Optional[Path] = Path(entrypoint_json)
        else:
            self.entrypoint_json = ENTRYPOINT_JSON if ENTRYPOINT_JSON.is_file() else None
        self._clock = clock
        self._max_file, self._max_pack, self._max_files = max_file_bytes, max_pack_bytes, max_pack_files
        self._write_lock = threading.Lock()
        self._git = shutil.which("git")

    # -- git plumbing --------------------------------------------------------------------------
    def _env(self) -> dict[str, str]:
        epoch = int(self._clock())
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_DIR": str(self.repo),
            "GIT_AUTHOR_NAME": "nightshift-lab",
            "GIT_AUTHOR_EMAIL": "lab@localhost",
            "GIT_COMMITTER_NAME": "nightshift-lab",
            "GIT_COMMITTER_EMAIL": "lab@localhost",
            "GIT_AUTHOR_DATE": f"{epoch} +0000",
            "GIT_COMMITTER_DATE": f"{epoch} +0000",
            "LC_ALL": "C",
        }

    def _run(self, *args: str, input: Optional[bytes] = None, check: bool = True, env_extra: Optional[Mapping[str, Optional[str]]] = None) -> subprocess.CompletedProcess:
        if self._git is None:
            raise ContractError("git is not installed")
        env = self._env()
        for k, v in (env_extra or {}).items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        proc = subprocess.run(
            [self._git, "-c", f"core.hooksPath={os.devnull}", "-c", "core.quotepath=off", *args],
            input=input,
            capture_output=True,
            env=env,
            check=False,
        )
        if check and proc.returncode != 0:
            raise ContractError(f"git {args[0]} failed: {proc.stderr.decode('utf-8', 'replace').strip()[:300]}")
        return proc

    def _out(self, *args: str, input: Optional[bytes] = None) -> str:
        return self._run(*args, input=input).stdout.decode("utf-8").strip()

    def _check_pack(self, pack: PackRef) -> str:
        if pack.repo_path != self.repo_rel:
            raise ContractError(f"pack {pack.repo_path!r} is not this lab's pack repository ({self.repo_rel!r})")
        if not self.repo.is_dir():
            raise ContractError("the pack repository does not exist; run init_v0 first")
        proc = self._run("cat-file", "-e", pack.commit + "^{commit}", check=False)
        if proc.returncode != 0:
            raise ContractError(f"pack commit {pack.commit[:12]} is not in the pack repository")
        return pack.commit

    def _ref(self, name: str) -> Optional[str]:
        proc = self._run("rev-parse", "--verify", "-q", name + "^{commit}", check=False)
        return proc.stdout.decode().strip() if proc.returncode == 0 else None

    def _ref_pack(self, commit: str) -> PackRef:
        return PackRef(self.repo_rel, commit)

    # -- reading trees -------------------------------------------------------------------------
    def _tree_entries(self, commit: str) -> dict[str, tuple[str, str]]:
        raw = self._run("ls-tree", "-r", "-z", "--full-tree", commit).stdout
        out: dict[str, tuple[str, str]] = {}
        for rec in raw.split(b"\0"):
            if not rec:
                continue
            meta, _, name = rec.partition(b"\t")
            mode, typ, sha = meta.decode().split()
            path = name.decode("utf-8")
            if typ != "blob" or mode not in _REGULAR_MODES:
                raise ContractError(f"{path}: only regular files are allowed in a pack (mode {mode}, type {typ})")
            try:
                check_pack_path(path)
            except ProposalInvalid as exc:
                raise ContractError(str(exc)) from None
            if is_protected(path) and path != "":  # protected names never belong in a version, wherever they came from
                raise ContractError(f"{path}: protected path in a pack")
            out[path] = (mode, sha)
        return out

    def _blobs(self, shas: Sequence[str]) -> dict[str, bytes]:
        uniq = sorted(set(shas))
        if not uniq:
            return {}
        raw = self._run("cat-file", "--batch", input=("\n".join(uniq) + "\n").encode()).stdout
        out: dict[str, bytes] = {}
        pos = 0
        for sha in uniq:
            nl = raw.index(b"\n", pos)
            header = raw[pos:nl].decode().split()
            if len(header) != 3 or header[1] != "blob":
                raise ContractError(f"object {sha[:12]} is not a blob")
            size = int(header[2])
            out[sha] = raw[nl + 1 : nl + 1 + size]
            pos = nl + 1 + size + 1
        return out

    def _read_tree(self, commit: str) -> dict[str, tuple[str, bytes]]:
        entries = self._tree_entries(commit)
        if len(entries) > self._max_files:
            raise ContractError(f"the pack has {len(entries)} files (limit {self._max_files})")
        blobs = self._blobs([sha for _, sha in entries.values()])
        files = {p: (mode, blobs[sha]) for p, (mode, sha) in entries.items()}
        total = 0
        for p, (_, data) in files.items():
            if len(data) > self._max_file:
                raise ContractError(f"{p}: {len(data)} bytes exceeds the {self._max_file}-byte file limit")
            if b"\0" in data:
                raise ContractError(f"{p}: binary files are not allowed")
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                raise ContractError(f"{p}: not UTF-8 text") from None
            total += len(data)
        if total > self._max_pack:
            raise ContractError(f"the pack is {total} bytes (limit {self._max_pack})")
        return files

    def read_files(self, pack: PackRef) -> dict[str, str]:
        """The pack's files as text (path -> content), after the same checks as a checkout."""
        return {p: d.decode("utf-8") for p, (_, d) in self._read_tree(self._check_pack(pack)).items()}

    # -- writing trees -------------------------------------------------------------------------
    def _write_tree(self, entries: Mapping[str, tuple[str, str]]) -> str:
        """Tree object for ``path -> (mode, blob sha)``, built bottom-up with ``git mktree``."""
        tree: dict[str, Any] = {}
        for path, (mode, sha) in entries.items():
            node = tree
            parts = path.split("/")
            for part in parts[:-1]:
                nxt = node.setdefault(part, {})
                if not isinstance(nxt, dict):
                    raise ProposalInvalid(f"{path}: a file cannot also be a directory")
                node = nxt
            if parts[-1] in node:
                raise ProposalInvalid(f"{path}: a file cannot also be a directory")
            node[parts[-1]] = (mode, sha)

        def build(node: dict) -> str:
            lines = []
            for name in sorted(node):
                v = node[name]
                if isinstance(v, dict):
                    lines.append(f"040000 tree {build(v)}\t{name}\0")
                else:
                    lines.append(f"{v[0]} blob {v[1]}\t{name}\0")
            return self._run("mktree", "-z", input="".join(lines).encode("utf-8")).stdout.decode().strip()

        return build(tree)

    def _hash_blob(self, text: str) -> str:
        return self._run("hash-object", "-w", "--stdin", input=text.encode("utf-8")).stdout.decode().strip()

    def _commit(self, tree: str, parent: Optional[str], message: str) -> str:
        args = ["commit-tree", tree, "-m", message]
        if parent:
            args += ["-p", parent]
        return self._run(*args).stdout.decode().strip()

    def _create_ref(self, ref: str, commit: str) -> None:
        # an empty old value means "must not exist yet": a tag is never moved or replaced
        self._run("update-ref", ref, commit, "")

    @staticmethod
    def _check_label(label: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label):
            raise ContractError(f"version label {label!r} is not a safe identifier")
        return label

    def _check_tag_name(self, label: str) -> None:
        self._check_label(label)
        if self._run("check-ref-format", f"refs/tags/pack/{label}", check=False).returncode != 0:
            raise ContractError(f"version label {label!r} is not a valid git tag name")

    # -- PackStore -----------------------------------------------------------------------------
    def init_v0(self, template_dir: str | os.PathLike[str] = PACK_V0_TEMPLATE) -> PackRef:
        """Create the bare repository with the template's files as the first commit, tagged ``pack/v0``, ``main`` on it. Refuses if
        the repository already holds a ``main``. Writes ``lint_ranges_path`` if it does not exist."""
        with self._write_lock:
            if self.repo.exists() and self._ref("refs/heads/main"):
                raise ContractError("the pack repository already has a v0")
            files = self._template_files(Path(template_dir))
            if "task.yaml" not in files:
                raise ContractError("the pack template needs a task.yaml at its root")
            self.repo.parent.mkdir(parents=True, exist_ok=True)
            if not self.repo.exists():
                self._run("init", "-q", "--bare", "--initial-branch=main", "--template=", str(self.repo), env_extra={"GIT_DIR": None})
            entries = {p: ("100644", self._hash_blob(t)) for p, t in files.items()}
            commit = self._commit(self._write_tree(entries), None, "v0: initial researcher pack")
            self._create_ref("refs/heads/main", commit)
            self._create_ref("refs/tags/pack/v0", commit)
            if not self.lint_ranges_path.exists():
                self.lint_ranges_path.parent.mkdir(parents=True, exist_ok=True)
                self.lint_ranges_path.write_text(json.dumps(default_lint_ranges(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return self._ref_pack(commit)

    def _template_files(self, root: Path) -> dict[str, str]:
        files: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name == ".DS_Store" or name.endswith(".pyc"):
                    continue
                full = Path(dirpath) / name
                st = os.lstat(full)
                if not stat.S_ISREG(st.st_mode):
                    raise ContractError(f"{full}: the pack template may hold regular files only")
                rel = full.relative_to(root).as_posix()
                if is_protected(rel):
                    raise ContractError(f"{rel}: protected path in the pack template")
                try:
                    text = full.read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    raise ContractError(f"{rel}: not UTF-8 text") from None
                if "\0" in text or len(text.encode("utf-8")) > self._max_file:
                    raise ContractError(f"{rel}: binary or larger than {self._max_file} bytes")
                files[rel] = text
            for d in dirnames:
                if os.path.islink(Path(dirpath) / d):
                    raise ContractError(f"{Path(dirpath) / d}: symbolic links are refused")
        if len(files) > self._max_files:
            raise ContractError(f"the pack template has {len(files)} files (limit {self._max_files})")
        return files

    def checkout(self, pack: PackRef, dest: str | os.PathLike[str]) -> None:
        """Clean export of ``pack`` (no ``.git``) into ``dest``, which must not exist or must be an empty directory."""
        commit = self._check_pack(pack)
        files = self._read_tree(commit)
        d = Path(dest)
        if d.exists():
            if not d.is_dir() or any(d.iterdir()):
                raise ContractError(f"checkout destination {d} is not an empty directory")
        d.mkdir(parents=True, exist_ok=True)
        for path, (mode, data) in files.items():
            target = d / path
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o755 if mode == "100755" else 0o644)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)

    def tree_sha256(self, pack: PackRef) -> str:
        return listing_sha256({p: d for p, (_, d) in self._read_tree(self._check_pack(pack)).items()})

    def apply(self, proposal: Proposal, parent: PackRef, version_label: str) -> tuple[PackRef, str]:
        """Strict apply (see ``apply_proposal_edits``): returns the new commit, tagged ``pack/<version_label>``, and the diff text."""
        self._check_tag_name(version_label)
        with self._write_lock:
            parent_commit = self._check_pack(parent)
            if self._ref(f"refs/tags/pack/{version_label}"):
                raise ContractError(f"tag pack/{version_label} already exists")
            files = {p: d.decode("utf-8") for p, (_, d) in self._read_tree(parent_commit).items()}
            modes = {p: m for p, (m, _) in self._tree_entries(parent_commit).items()}
            new = apply_proposal_edits(
                files,
                proposal,
                is_editable=self._is_editable,
                max_file_bytes=self._max_file,
                max_pack_bytes=self._max_pack,
                max_pack_files=self._max_files,
            )
            entries = {p: (modes.get(p, "100644"), self._hash_blob(t)) for p, t in new.items()}
            commit = self._commit(self._write_tree(entries), parent_commit, _commit_message(version_label, proposal))
            self._create_ref(f"refs/tags/pack/{version_label}", commit)
            diff = self._run("diff", "--no-color", "--no-ext-diff", "-U3", parent_commit, commit).stdout.decode("utf-8", "replace")
            return self._ref_pack(commit), diff

    def accept(self, pack: PackRef, version_label: str) -> None:
        """Fast-forward ``main`` to ``pack`` (idempotent). The candidate's tag is created if missing and must not point elsewhere."""
        self._check_tag_name(version_label)
        with self._write_lock:
            commit = self._check_pack(pack)
            tag = self._ref(f"refs/tags/pack/{version_label}")
            if tag is None:
                self._create_ref(f"refs/tags/pack/{version_label}", commit)
            elif tag != commit:
                raise ContractError(f"tag pack/{version_label} points at {tag[:12]}, not at {commit[:12]}")
            head = self._ref("refs/heads/main")
            if head == commit:
                return
            if head is None:
                raise ContractError("the pack repository has no main; run init_v0 first")
            if self._run("merge-base", "--is-ancestor", head, commit, check=False).returncode != 0:
                raise ContractError(f"{version_label} is not a fast-forward of main ({head[:12]})")
            self._run("update-ref", "refs/heads/main", commit, head)

    def main_pack(self) -> PackRef:
        head = self._ref("refs/heads/main")
        if head is None:
            raise ContractError("the pack repository has no main")
        return self._ref_pack(head)

    def tagged(self, version_label: str) -> PackRef:
        c = self._ref(f"refs/tags/pack/{self._check_label(version_label)}")
        if c is None:
            raise ContractError(f"no tag pack/{version_label}")
        return self._ref_pack(c)

    # -- lint ----------------------------------------------------------------------------------
    def _load_ranges(self) -> dict:
        try:
            ranges = loads_strict(self.lint_ranges_path.read_text(encoding="utf-8"))
            for k in ("editable_keys", "text_max_chars", "int_ranges", "heartbeat", "assignments", "skills"):
                ranges[k]
        except (OSError, KeyError, TypeError, ContractError) as exc:
            raise ContractError(f"lint ranges file {self.lint_ranges_path} is missing or malformed: {type(exc).__name__}") from None
        return ranges

    def _dump_blob(self, data: bytes, tmp: Path, name: str) -> Mapping[str, Any]:
        path = tmp / name
        path.write_bytes(data)
        return self._dump(path)

    def lint(self, pack: PackRef) -> LintResult:
        import tempfile

        commit = self._check_pack(pack)
        ranges = self._load_ranges()
        try:
            files = self._read_tree(commit)
        except ContractError as exc:
            return LintResult(False, (str(exc)[:_MSG],), ())
        parent_proc = self._run("rev-parse", "--verify", "-q", commit + "^1", check=False)
        parent_commit = parent_proc.stdout.decode().strip() if parent_proc.returncode == 0 else None
        if parent_commit:
            raw = self._run("diff", "--name-only", "-z", parent_commit, commit).stdout
            changed = tuple(sorted(p for p in raw.decode("utf-8").split("\0") if p))
        else:
            changed = tuple(sorted(files))
        msgs: list[str] = []
        if parent_commit:
            for p in changed:
                if not self._is_editable(p):
                    msgs.append(f"{p}: not an editable pack path"[:_MSG])
        if "task.yaml" not in files:
            msgs.append("task.yaml is missing")
            return LintResult(False, tuple(msgs), changed)
        notes: list[str] = []
        with tempfile.TemporaryDirectory(prefix="pack-lint-") as td:
            tmp = Path(td)
            try:
                cand_doc = self._dump_blob(files["task.yaml"][1], tmp, "candidate.yaml")
            except ContractError as exc:
                msgs.append(str(exc)[:_MSG])
                return LintResult(False, tuple(msgs), changed)
            cand = cand_doc["config"]
            parent_cfg = None
            if parent_commit:
                parent_files = self._read_tree(parent_commit)
                try:
                    parent_cfg = self._dump_blob(parent_files["task.yaml"][1], tmp, "parent.yaml")["config"]
                except (ContractError, KeyError):
                    msgs.append("the parent's task.yaml cannot be read, so the candidate cannot be compared with it")
            msgs.extend(compare_task_keys(parent_cfg, cand, ranges) if (parent_cfg is not None or not parent_commit) else check_editable_values(cand, ranges))
            if self.entrypoint_json is not None:
                try:
                    ep = loads_strict(self.entrypoint_json.read_text(encoding="utf-8"))
                    msgs.extend(compare_grader_values(cand, ep))
                except (OSError, ContractError):
                    msgs.append("the runtime's ENTRYPOINT.json cannot be read")
            if not cand_doc.get("coral_validated"):
                notes.append("note: CORAL itself did not validate this task.yaml here (run the lint with CORAL's interpreter on the researcher host)")
        # referenced files exist
        have = set(files)
        for i, s in enumerate(_as_list(_get(cand, "agents.skills"))):
            if isinstance(s, str) and not any(p == s or p.startswith(s.rstrip("/") + "/") for p in have):
                msgs.append(f"agents.skills[{i}] {s!r} does not exist in the pack")
        for i, a in enumerate(_as_list(_get(cand, "agents.assignments"))):
            rf = a.get("runtime_options", {}).get("role_file") if isinstance(a, dict) and isinstance(a.get("runtime_options"), dict) else None
            if isinstance(rf, str) and rf not in have:
                msgs.append(f"agents.assignments[{i}] role_file {rf!r} does not exist in the pack")
        msgs = [m[:_MSG] for m in msgs]
        return LintResult(not msgs, tuple(msgs) if msgs else tuple(notes), changed)


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else []


def _commit_message(label: str, proposal: Proposal) -> str:
    def one_line(s: str, n: int) -> str:
        return " ".join(s.split())[:n]

    return (
        f"{label}: {one_line(proposal.change_summary, 120)}\n\n"
        f"lever: {proposal.lever.value}\nhypothesis: {one_line(proposal.hypothesis, 500)}\n"
        f"expected effect: {one_line(proposal.expected_effect, 300)}\nrisk: {one_line(proposal.risk, 300)}\n"
    )
