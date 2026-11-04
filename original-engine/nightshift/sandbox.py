"""Workspaces, throwaway copies, output staging and the contained untrusted run.

Guarantees
----------
* Nothing here follows a symlink inside a workspace. One walker (``_scan``, built on ``os.scandir`` + ``os.lstat``) yields
  only regular files; every symlink, FIFO, socket, device, hard-linked file or unsafe name becomes the violation string
  ``special:<path>``. Files are copied one by one with ``O_NOFOLLOW``; ``shutil.copytree`` is never used on a workspace.
* ``run_untrusted`` never uses ``preexec_fn``. It spawns ``[sys.executable, -I, -c, SHIM, *cmd]``; the shim applies rlimits
  (each one best effort) and ``execvpe``s the command with the scrubbed environment from contract section 4.3. The whole
  process group is SIGKILLed on every exit path, and on Linux leftovers (including daemons that called ``setsid``) are
  found by session, process group, parent chain, working directory and inherited pipe, then killed.
* ``available_isolation`` reports only what a real probe proved. This is best-effort containment of same-user code, not a
  security boundary: tampering is detected by hashes (docs/ARCHITECTURE.md section 11).

Isolation modes: what each one PREVENTS and what it only DETECTS
----------------------------------------------------------------
``auto`` (the default) applies the strongest mode a real probe proved on this machine. ``NIGHTSHIFT_ISOLATION`` (one of
``auto``, ``userns``, ``unshare-net``, ``none``) may only LOWER that: a mode the probe did not prove is never applied, the
strongest proven mode at or below the request is used instead, and the reason is written to ``RunResult.note``. A value
outside that set raises ``ValueError`` naming the allowed values at first use. The mode applied is always reported on
``RunResult.isolation`` (and from there on the ``experiment.run_started`` event); a machine with no namespaces still runs the
whole pipeline, with the weaker guarantees below.

===================================  ===============================  ===============================  ==========================
what the untrusted run can do        userns                           unshare-net                      none
===================================  ===============================  ===============================  ==========================
read ``trusted/`` (labels, grader)   PREVENTED (masked, empty)        succeeds; NOT detected (a read   succeeds; NOT detected
                                                                      leaves no trace)
read the ledger / other experiments  PREVENTED (masked)               succeeds; NOT detected           succeeds; NOT detected
change ``trusted/``, also when the   PREVENTED (cannot write or       DETECTED after the run (file     DETECTED after the run
bytes are put back before it exits   unmount; capabilities dropped)   content hash AND a stat snapshot
                                                                      of the whole tree: kind
                                                                      "trusted")
change the workspace's read-only     DETECTED (tree hash)             DETECTED (tree hash)             DETECTED (tree hash)
files, add links/FIFOs
open a network connection            PREVENTED (empty net namespace)  PREVENTED (empty net namespace)  succeeds; NOT detected
write elsewhere on the filesystem    succeeds; NOT detected           succeeds; NOT detected           succeeds; NOT detected
read the orchestrator's              PREVENTED (the pid namespace     PREVENTED when the probe proved  succeeds as root or with
``/proc/<pid>/environ`` (the         hides it)                        the pid-namespace variant (root   CAP_SYS_PTRACE; PREVENTED
``MODEL_API_KEY``)                                                    only); otherwise as ``none``      for a non-root same-uid
                                                                                                        reader (``harden_parent``)
leave a detached daemon behind       KILLED (pid namespace: all die   KILLED when the pid-namespace     best effort: found by
                                     with the namespace init)         variant is in use; otherwise      session, group, parent chain,
                                                                      best effort as ``none``           cwd, pipe; a double-forked
                                                                                                        daemon with stdio closed and
                                                                                                        cwd elsewhere can SURVIVE
===================================  ===============================  ===============================  ==========================

"Elsewhere on the filesystem" means paths outside the workspace and outside ``trusted/``/the masked paths: no mode makes the
filesystem read-only (there is no chroot), so what is checked is hashes of ``trusted/`` (by ``grading.evaluate``), of the
template (by the engine) and of the workspace. Reading ``trusted/`` or the ledger under ``unshare-net`` or ``none`` is what
those modes neither stop nor notice; the only trace is the diff tripwire, which flags text that names a trusted path in a
file the run changed. Only ``userns`` keeps the labels out of the run's reach; the other modes are for machines (laptops, CI
containers, macOS) where nothing stronger can be proven, and a run in them must not be described as sealed-safe.

How a change to ``trusted/`` is DETECTED (``unshare-net`` and ``none``)
-----------------------------------------------------------------------
A content hash compared after the run misses a file that is changed and then restored byte for byte before the process
exits. ``TrustedWatch`` therefore also takes a STAT snapshot (``trusted_stat``: for every file, directory and link under
``trusted/``: mode, size, inode, link count, owner, mtime, ctime and, for directories, the entry names; never any content, so
it is cheap) immediately before the untrusted process starts and compares it after the process group has been reaped and swept,
and again after the grader has run. Restoring the bytes and even ``mtime`` is possible for the run; restoring ``ctime`` is not
(no unprivileged call sets it), so a write-then-restore leaves a different ``st_ctime_ns`` and is reported. ``atime`` is ignored on
purpose: a read changes nothing but ``atime`` and reading is what the trusted grader does. Limits, stated plainly: a process
with ``CAP_SYS_TIME`` could step the system clock back to put an old ``ctime`` onto a fresh change; the check cannot see a
change that leaves every recorded field unchanged (none exists on a normal file system); and it is evidence about the window
between two snapshots, not a block.

The trusted grader does not write into ``trusted/`` (its environment has ``PYTHONDONTWRITEBYTECODE=1``, so importing a sibling
module leaves no ``__pycache__``). The content hash of ``trusted/`` ignores ``__pycache__`` and ``*.pyc``, which would hide a
``*.pyc`` planted next to the grader (Python imports a source-less ``random.pyc`` from the script's directory ahead of the
standard library); so the snapshot taken around the UNTRUSTED run includes them, and only the snapshot taken after the grader
ignores bytecode (a grader that writes its own bytecode has always been tolerated). That leaves one gap, in the modes where a
detached survivor can exist (``none``, ``unshare-net`` without the pid namespace): bytecode planted by a survivor after the
post-run check and before the grader imports it is not seen by the snapshots. A directory whose times moved without bytecode
being added or removed in it is still reported (a module that is planted and removed again), and a trusted directory that is
itself a link is resolved, so the tree behind it is what is watched.

Independent of any baseline, ``stray_bytecode`` names every source-less ``*.pyc`` (one outside a ``__pycache__`` directory)
in ``trusted/``: ``grading.evaluate`` refuses to start on top of one, looks again before the grader runs and after it, and so
a plant that outlived the window it was made in cannot silently poison the next evaluation of another lane.

Concurrent lanes and the classification of a difference. Lanes run in parallel threads and share one ``trusted/`` directory,
so a difference seen by one run may have been made by another lane's run. An in-process gate counts the untrusted runs. If
no other untrusted run was active at any moment of this run's window, the difference is attributable (to this run, to a
survivor of it, or to something outside Nightshift) and ``grading.evaluate`` raises ``ProtectedPathViolation(kind="trusted")``
naming the paths, exactly like a hash mismatch. If another run overlapped the window the difference is AMBIGUOUS: the run is
classified ``infra`` (``RunResult.trusted_contested``; the note says ``trusted_tree_changed_during_run`` and that the change may
come from a concurrent lane), never as tampering, so an honest experiment is not accused. The engine retries an ``infra``
failure once with ``attempt=2``; an attempt after the first runs ALONE (``solo``: the gate waits for every other untrusted run
and holds new ones back), so on the retry the difference can only be this run's own and is reported as a violation. The
trade-off: a hostile lane cannot hide a write-then-restore behind a busy machine (its retry is solo), an honest lane that
shared the window with a hostile one is retried once instead of failed, and the price is one serialised attempt. A change that
persists (a content-hash mismatch) is a tamper finding for every lane that sees it, as before, because the grader would then
use altered files. The gate never blocks forever: a solo request that cannot get the machine quiet within a bounded wait falls
back to a shared window (and an ambiguous result stays ``infra``), and a shared request that waited that long enters anyway
and makes the solo window contested again, so a solo run is never charged with a change another run made.

``harden_parent`` protects the orchestrator's memory and ``/proc/<pid>/environ`` from a non-root process of the SAME user. It
does NOT stop root, or any process holding ``CAP_SYS_PTRACE``: the kernel lets those read the file regardless of the dumpable
flag. So under ``unshare-net`` and ``none`` a Nightshift run that is itself root hands the API key to the experiment code
unless the pid-namespace variant of ``unshare-net`` was proven (it hides the orchestrator's /proc; capabilities are dropped so
the run cannot unmount it). ``none`` has no namespace at all. Run Nightshift as a non-root user, inside a container or VM, when
an API key or sealed numbers matter and ``userns`` is unavailable. A Mac has no user namespaces and no ``/proc``: the mode is
``none``, ``harden_parent`` does nothing there, and a same-user experiment can read whatever the user can (use a VM).

Off Linux (no ``/proc``, no ``setrlimit(RLIMIT_AS)``) the same pipeline runs with the isolation ``none``: process sweeping
and CPU accounting are skipped, every rlimit is applied best effort, and an honest run is still classified ``ok``.
"""
from __future__ import annotations

import difflib
import errno
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional, Sequence

from . import secrets
from .types import ProtectedPathViolation, TaskSpec, canonical_json, short_hash

_LINUX = sys.platform.startswith("linux")

SAFE_RELPATH = re.compile(r"[A-Za-z0-9._/-]{1,128}")      # always used with fullmatch (``$`` would accept a trailing \n)
SAFE_OUTPUT_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
TRIPWIRE = re.compile(r"trusted|labels\.|ledger\.jsonl|replication_seeds|tasks/")
_TRIPWIRE_BYTES = re.compile(TRIPWIRE.pattern.encode("ascii"))

MAX_FILES = 2000
MAX_DIRS = 2000                     # extra allowance for directories (and other entries) on top of MAX_FILES
MAX_TREE_BYTES = 64 * 1024 * 1024
MAX_SCAN_DEPTH = 32
MAX_DIFF_BYTES = 200 * 1024
MAX_DIFF_FILE_BYTES = 256 * 1024
MAX_TRIPWIRE_BYTES = 4 * 1024 * 1024
LOG_CAP_BYTES = 8 * 1024 * 1024
TAIL_BYTES = 64 * 1024
MAX_STAGED_FILES = 32
MAX_STAGE_DEPTH = 2
ISOLATION_LEVELS = ("none", "unshare-net", "userns")      # weakest first
_RANK = {name: i for i, name in enumerate(ISOLATION_LEVELS)}
ISOLATION_ENV = "NIGHTSHIFT_ISOLATION"
ISOLATION_CHOICES = ("auto", "userns", "unshare-net", "none")
MASK_FAILED_EXIT = 97
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


# --------------------------------------------------------------------------------------------------
# data types
# --------------------------------------------------------------------------------------------------
@dataclass
class Workspace:
    exp_id: str
    path: Path
    base_hashes: dict[str, str]
    tree_hash: str
    base_dir: Optional[Path] = None      # pristine base the diff is taken against; resolved from base.json when None


@dataclass
class WorkspaceDiff:
    changed: list[str]
    added: list[str]
    removed: list[str]
    unified: str
    diff_hash: str
    summary: str
    violations: list[str]
    tree_hash: str


@dataclass
class RunResult:
    exit_code: Optional[int]
    stdout: str                          # the LAST 64 KiB only
    stderr: str
    duration_sec: float
    timed_out: bool
    stalled: bool
    classification: str                  # ok | timeout | crash | infra
    output_bytes: int
    pid: Optional[int]
    note: str
    isolation: str = "none"              # isolation actually applied to this run
    # ``trusted/`` stat-snapshot difference seen when the run was reaped, as ``trusted/<path>`` (see ``TrustedWatch``). When
    # ``trusted_contested`` is also true another untrusted run overlapped, the difference is ambiguous and the run is
    # classified ``infra``; otherwise ``grading.evaluate`` raises a ``kind="trusted"`` violation for it.
    trusted_changed: list[str] = field(default_factory=list)
    trusted_contested: bool = False


# --------------------------------------------------------------------------------------------------
# paths, hashing, the one directory walker
# --------------------------------------------------------------------------------------------------
def _vpath(rel: str) -> str:
    """A log-safe rendering of an untrusted relative path (control characters and non-ASCII escaped, capped)."""
    text = rel.encode("unicode_escape").decode("ascii")
    return text if len(text) <= 160 else text[:157] + "..."


display_path = _vpath      # public name for log-safe rendering of untrusted paths


def _safe_rel(rel: str) -> bool:
    return SAFE_RELPATH.fullmatch(rel) is not None and ".." not in rel


@dataclass
class _Scan:
    files: dict[str, os.stat_result] = field(default_factory=dict)    # relpath -> lstat of a regular file
    violations: list[str] = field(default_factory=list)

    def flag(self, rel: str) -> None:
        item = f"special:{_vpath(rel)}"
        if item not in self.violations:
            self.violations.append(item)


def _scan(root: str | os.PathLike, *, max_files: Optional[int] = None, max_bytes: Optional[int] = None) -> _Scan:
    """Walk ``root`` without following links. Regular, singly-linked, safely named files are returned; anything else is a
    ``special:<path>`` violation. Stops (with a violation) at the file-count and byte caps."""
    max_files = MAX_FILES if max_files is None else max_files
    max_bytes = MAX_TREE_BYTES if max_bytes is None else max_bytes
    root_s = os.fspath(root)
    scan = _Scan()
    if not stat.S_ISDIR(os.lstat(root_s).st_mode):
        scan.flag(".")
        return scan
    total = 0
    seen = 0                                       # directory entries of any kind, so a directory bomb is stopped too
    entry_cap = max_files + MAX_DIRS
    queue: deque[tuple[str, str]] = deque([("", root_s)])
    while queue:
        rel_dir, abs_dir = queue.popleft()
        try:
            entries = []
            with os.scandir(abs_dir) as it:
                for entry in it:
                    seen += 1
                    if seen > entry_cap:
                        scan.flag("too_many_files")
                        return scan
                    entries.append(entry)
            entries.sort(key=lambda e: e.name)
        except OSError:
            scan.flag(rel_dir or ".")
            continue
        for entry in entries:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            try:
                st = os.lstat(entry.path)
            except OSError:
                scan.flag(rel)
                continue
            if stat.S_ISDIR(st.st_mode):
                if not _safe_rel(rel) or rel.count("/") >= MAX_SCAN_DEPTH:
                    scan.flag(rel)
                else:
                    queue.append((rel, entry.path))
            elif stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and _safe_rel(rel):
                if len(scan.files) >= max_files:
                    scan.flag("too_many_files")
                    return scan
                total += st.st_size
                if total > max_bytes:
                    scan.flag("too_large")
                    return scan
                scan.files[rel] = st
            else:
                scan.flag(rel)
    return scan


def _open_regular(path: str) -> tuple[int, os.stat_result]:
    """Open a plain regular file for reading, refusing symlinks, FIFOs, devices and hard-linked files."""
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise OSError(errno.EINVAL, "not a plain regular file", path)
        return fd, st
    except BaseException:
        os.close(fd)
        raise


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _sha_regular(path: str) -> str:
    fd, _ = _open_regular(path)
    try:
        h = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)
    finally:
        os.close(fd)


def _read_bounded(path: str, limit: int) -> bytes:
    fd, _ = _open_regular(path)
    try:
        chunks, got = [], 0
        while got <= limit:
            chunk = os.read(fd, min(1 << 20, limit + 1 - got))
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _is_scratch(rel: str, task: TaskSpec) -> bool:
    return any(fnmatch.fnmatchcase(rel, g) for g in task.scratch)


def _is_editable(rel: str, task: TaskSpec) -> bool:
    return any(fnmatch.fnmatchcase(rel, g) for g in task.editable)


def _digest(hashes: dict[str, str]) -> str:
    return short_hash(canonical_json(sorted(hashes.items())))


@dataclass
class _Snapshot:
    hashes: dict[str, str]        # sha256 of every NON-scratch regular file
    violations: list[str]         # special:<path> for anything that is not a plain regular file (scratch included)


def _snapshot(root: str | os.PathLike, task: TaskSpec) -> _Snapshot:
    scan = _scan(root)
    hashes: dict[str, str] = {}
    violations = list(scan.violations)
    for rel in sorted(scan.files):
        if _is_scratch(rel, task):
            continue
        try:
            hashes[rel] = _sha_regular(os.path.join(os.fspath(root), rel))
        except OSError:
            violations.append(f"special:{_vpath(rel)}")
    return _Snapshot(hashes, violations)


def tree_hash(root: Path, task: TaskSpec) -> str:
    """12 hex over the sorted (relpath, sha256) of the non-scratch regular files under ``root``."""
    return _digest(_snapshot(root, task).hashes)


def base_hash(ws: Workspace) -> str:
    """``short_hash(canonical_json(sorted(base_hashes.items())))`` (contract section 5)."""
    return _digest(ws.base_hashes)


# --------------------------------------------------------------------------------------------------
# copying
# --------------------------------------------------------------------------------------------------
def _prepare_dest(dst: str | os.PathLike) -> None:
    """Create ``dst`` (and parents) as a real empty directory. Refuses a symlink or a non-empty directory."""
    dst_s = os.fspath(dst)
    if os.path.lexists(dst_s):
        st = os.lstat(dst_s)
        if not stat.S_ISDIR(st.st_mode):
            raise FileExistsError(f"{dst_s} exists and is not a directory")
        if os.listdir(dst_s):
            raise FileExistsError(f"{dst_s} exists and is not empty")
        return
    os.makedirs(dst_s, mode=0o755)


def _copy_files(src_root: str, dst_root: str, rels: Iterable[str], kind: str) -> dict[str, str]:
    """Copy regular files one by one (O_NOFOLLOW on both sides). Returns the sha256 of the bytes actually copied.

    A file that stopped being plain between the scan and the copy is reported as ``special:<path>``."""
    out: dict[str, str] = {}
    total = 0
    for rel in rels:
        src, dst = os.path.join(src_root, rel), os.path.join(dst_root, rel)
        try:
            os.makedirs(os.path.dirname(dst), mode=0o755, exist_ok=True)
            fd, st = _open_regular(src)
            mode = 0o755 if st.st_mode & stat.S_IXUSR else 0o644
            try:
                out_fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC, mode)
                try:
                    h = hashlib.sha256()
                    while True:
                        chunk = os.read(fd, 1 << 20)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_TREE_BYTES:
                            raise ProtectedPathViolation(["special:too_large"], kind)
                        h.update(chunk)
                        _write_all(out_fd, chunk)
                finally:
                    os.close(out_fd)
            finally:
                os.close(fd)
            os.chmod(dst, mode)      # the umask may have stripped bits
        except OSError as exc:
            if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                raise
            raise ProtectedPathViolation([f"special:{_vpath(rel)}"], kind) from exc
        out[rel] = h.hexdigest()
    return out


def _copy_regular(src: str | os.PathLike, dst: str | os.PathLike, *, kind: str,
                  keep: Optional[Callable[[str], bool]] = None) -> dict[str, str]:
    scan = _scan(src)
    if scan.violations:
        raise ProtectedPathViolation(scan.violations, kind)
    rels = [r for r in sorted(scan.files) if keep is None or keep(r)]
    _prepare_dest(dst)
    try:
        return _copy_files(os.fspath(src), os.fspath(dst), rels, kind)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)       # a half-written copy is worse than none
        raise


def copy_regular(src: Path, dst: Path, task: TaskSpec, *, kind: str = "workspace") -> None:
    """Copy every regular file of ``src`` into the new directory ``dst``.

    Raises ``ProtectedPathViolation(kind)`` listing ``special:<path>`` for any symlink, FIFO, device, socket, hard-linked
    file, unsafe name, or a tree over 2000 files / 64 MiB; in that case nothing is written. ``task`` is accepted for
    signature stability.
    """
    _copy_regular(src, dst, kind=kind)


def copy_regular_hashes(src: Path, dst: Path, task: TaskSpec, *, kind: str = "workspace") -> dict[str, str]:
    """``copy_regular`` that also returns the sha256 map of the NON-scratch files actually copied (for tree-hash checks)."""
    copied = _copy_regular(src, dst, kind=kind)
    return {rel: sha for rel, sha in copied.items() if not _is_scratch(rel, task)}


def snapshot_hashes(root: Path, task: TaskSpec) -> tuple[dict[str, str], list[str]]:
    """(sha256 of every non-scratch regular file, ``special:<path>`` violations anywhere including scratch)."""
    snap = _snapshot(root, task)
    return snap.hashes, snap.violations


def hashes_digest(hashes: dict[str, str]) -> str:
    """The 12-hex tree hash of a relpath -> sha256 map."""
    return _digest(hashes)


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Log-safe names of files added, removed or modified between two hash maps."""
    names = set(before) ^ set(after) | {k for k in before if k in after and before[k] != after[k]}
    return [_vpath(n) for n in sorted(names)]


def _write_json_atomic(path: Path, obj: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _base_ref(base: Optional[Path]) -> str:
    if base is None:
        return "template"
    base = Path(base)
    if base.name == "frozen" and base.parent.parent.name == "exp":
        return base.parent.name
    return str(base)


def create_workspace(task: TaskSpec, run_dir: Path, exp_id: str, base: Optional[Path] = None) -> Workspace:
    """Create ``<run_dir>/exp/<exp_id>/workspace`` from the template (``base=None``) or a ``frozen/`` directory.

    Writes ``exp/<exp_id>/base.json``; ``tree_hash`` and ``base_hashes`` are recomputed from the bytes copied.
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", exp_id) or exp_id in (".", ".."):
        raise ValueError(f"unsafe exp_id {exp_id!r}")
    exp_dir = Path(run_dir) / "exp" / exp_id
    dest = exp_dir / "workspace"
    src = Path(base) if base is not None else Path(task.template_dir)
    hashes = _copy_regular(src, dest, kind="workspace")
    base_hashes = {rel: sha for rel, sha in hashes.items() if not _is_scratch(rel, task)}
    digest = _digest(base_hashes)
    _write_json_atomic(exp_dir / "base.json",
                       {"base_ref": _base_ref(base), "base_hashes": base_hashes, "tree_hash": digest})
    return Workspace(exp_id=exp_id, path=dest, base_hashes=base_hashes, tree_hash=digest, base_dir=src)


def _chmod_readonly(root: str) -> None:
    """Drop write permission from every file. Directories stay owner-writable so ``rm -rf`` / ``shutil.rmtree`` of a run
    directory keeps working for non-root users; integrity is enforced by tree-hash checks, not by directory modes."""
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in filenames:
            p = os.path.join(dirpath, name)
            os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) & ~0o222)


def freeze(ws_path: Path, dest: Path, task: TaskSpec) -> str:
    """Copy the non-scratch regular files of ``ws_path`` to ``dest``, make the files read-only, return the tree hash.

    Idempotent: if ``dest`` already holds an identical tree its hash is returned; a different one raises FileExistsError.
    """
    ws_path, dest = Path(ws_path), Path(dest)
    if os.path.lexists(dest) and os.listdir(dest):
        have, want = tree_hash(dest, task), tree_hash(ws_path, task)
        if have == want:
            return have
        raise FileExistsError(f"{dest} already holds a different frozen tree")
    hashes = _copy_regular(ws_path, dest, kind="workspace", keep=lambda r: not _is_scratch(r, task))
    _chmod_readonly(os.fspath(dest))
    return _digest(hashes)


# --------------------------------------------------------------------------------------------------
# diffing
# --------------------------------------------------------------------------------------------------
def _base_root(task: TaskSpec, ws: Workspace) -> Optional[Path]:
    if ws.base_dir is not None:
        return Path(ws.base_dir)
    exp_dir = Path(ws.path).parent
    try:
        ref = json.loads((exp_dir / "base.json").read_text(encoding="utf-8"))["base_ref"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if ref == "template":
        return Path(task.template_dir)
    if isinstance(ref, str):
        frozen = exp_dir.parent / ref / "frozen"
        if frozen.is_dir():
            return frozen
        return Path(ref) if Path(ref).is_dir() else None
    return None


def _read_base(root: Optional[Path], rel: str, want_sha: Optional[str]) -> Optional[bytes]:
    """Original bytes of ``rel``, or None when unavailable or no longer matching its recorded hash."""
    if root is None or want_sha is None:
        return None
    try:
        data = _read_bounded(os.path.join(os.fspath(root), rel), MAX_TRIPWIRE_BYTES)
    except OSError:
        return None
    return data if hashlib.sha256(data).hexdigest() == want_sha else None


def _as_text(data: bytes) -> Optional[str]:
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _file_diff(rel: str, old: Optional[bytes], new: Optional[bytes], *, base_unavailable: bool) -> tuple[str, int, int, bool]:
    """(unified text, added lines, removed lines, tripwire hit) for one file."""
    shown = _vpath(rel)
    if old is None and new is None:
        return f"removed file: {shown} (base unavailable)\n", 0, 0, False
    if (old is not None and len(old) > MAX_DIFF_FILE_BYTES) or (new is not None and len(new) > MAX_DIFF_FILE_BYTES):
        hit = new is not None and _TRIPWIRE_BYTES.search(new) is not None
        return f"large file changed: {shown}\n", 0, 0, hit
    old_text = "" if old is None else _as_text(old)
    new_text = "" if new is None else _as_text(new)
    if old_text is None or new_text is None:
        hit = new is not None and _TRIPWIRE_BYTES.search(new) is not None
        return f"binary file changed: {shown}\n", 0, 0, hit
    a, b = old_text.splitlines(), new_text.splitlines()
    fromfile = "/dev/null" if old is None else f"a/{shown}"
    tofile = "/dev/null" if new is None else f"b/{shown}"
    note = " (base unavailable)" if base_unavailable else ""
    if a == b:
        return f"--- {fromfile}\n+++ {tofile}{note}\n\\ whitespace or line-ending change only\n", 0, 0, False
    body = "\n".join(difflib.unified_diff(a, b, fromfile, tofile + note, lineterm=""))
    plus = minus = 0
    hit = False
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("replace", "delete"):
            minus += i2 - i1
        if tag in ("replace", "insert"):
            plus += j2 - j1
            hit = hit or any(TRIPWIRE.search(line) for line in b[j1:j2])
    return body + "\n", plus, minus, hit


def _cap_text(text: str, limit: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore") + f"\n[diff truncated at {limit} bytes]\n"


def diff_workspace(task: TaskSpec, ws: Workspace) -> WorkspaceDiff:
    """Compare the workspace's non-scratch files with ``ws.base_hashes``.

    ``violations`` holds ``special:<path>`` (links, FIFOs, hard links, unsafe names, caps), ``protected:<path>`` (a file
    outside ``task.editable`` changed, appeared or vanished) and ``probe:<path>`` (the tripwire regex matches text that was
    added or changed in a non-scratch file). ``unified`` is capped at 200 KB; binary files read ``binary file changed``.
    """
    snap = _snapshot(ws.path, task)
    cur, base = snap.hashes, ws.base_hashes
    added = sorted(set(cur) - set(base))
    removed = sorted(set(base) - set(cur))
    changed = sorted(r for r in cur if r in base and cur[r] != base[r])
    violations = list(snap.violations)
    for rel in sorted(changed + added + removed):
        if not _is_editable(rel, task):
            violations.append(f"protected:{_vpath(rel)}")
    base_root = _base_root(task, ws)
    parts: list[str] = []
    plus = minus = 0
    for rel in sorted(set(changed) | set(added) | set(removed)):
        old = None if rel in added else _read_base(base_root, rel, base.get(rel))
        new = None
        if rel not in removed:
            try:
                new = _read_bounded(os.path.join(os.fspath(ws.path), rel), MAX_TRIPWIRE_BYTES)
            except OSError:
                violations.append(f"special:{_vpath(rel)}")
                continue
        text, p, m, hit = _file_diff(rel, old, new, base_unavailable=(rel in changed and old is None))
        parts.append(text)
        plus, minus = plus + p, minus + m
        if hit:
            violations.append(f"probe:{_vpath(rel)}")
    ops = [["A", r, cur[r]] for r in added] + [["D", r, ""] for r in removed] + [["M", r, cur[r]] for r in changed]
    diff_hash = short_hash(canonical_json(sorted(ops, key=lambda o: o[1])))
    names = [f"{op} {_vpath(r)}" for op, r, _ in sorted(ops, key=lambda o: o[1])]
    summary = (", ".join(names[:8]) + (f", +{len(names) - 8} more" if len(names) > 8 else "")) if names else "no changes"
    if names:
        summary += f" (+{plus}/-{minus} lines)"
    return WorkspaceDiff(changed=changed, added=added, removed=removed, unified=_cap_text("".join(parts), MAX_DIFF_BYTES),
                         diff_hash=diff_hash, summary=summary[:400], violations=list(dict.fromkeys(violations)),
                         tree_hash=_digest(cur))


# --------------------------------------------------------------------------------------------------
# output staging
# --------------------------------------------------------------------------------------------------
def stage_outputs_report(run_ws: Path, dest: Path, task: TaskSpec) -> tuple[str, list[str]]:
    """``stage_outputs`` that also returns what was skipped, as ``kind:<path>`` strings."""
    skipped: list[str] = []
    staged: dict[str, str] = {}
    _prepare_dest(dest)
    src_root = os.path.join(os.fspath(run_ws), "outputs")
    try:
        root_stat = os.lstat(src_root)
    except OSError:
        return _digest(staged), skipped
    if not stat.S_ISDIR(root_stat.st_mode):            # e.g. `outputs -> /somewhere`: never followed
        return _digest(staged), [f"special:{_vpath('outputs')}"]

    candidates: list[tuple[str, str]] = []

    seen = [0]

    def visit(abs_dir: str, rel_dir: str, depth: int) -> None:
        try:
            entries = []
            with os.scandir(abs_dir) as it:
                for entry in it:
                    seen[0] += 1
                    if seen[0] > MAX_FILES + MAX_DIRS:      # never list an unbounded directory
                        skipped.append(f"toomany:{_vpath(rel_dir or 'outputs')}")
                        return
                    entries.append(entry)
            entries.sort(key=lambda e: e.name)
        except OSError:
            skipped.append(f"special:{_vpath(rel_dir or 'outputs')}")
            return
        for entry in entries:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            if SAFE_OUTPUT_NAME.fullmatch(entry.name) is None:
                skipped.append(f"badname:{_vpath(rel)}")
                continue
            try:
                st = os.lstat(entry.path)
            except OSError:
                skipped.append(f"special:{_vpath(rel)}")
                continue
            if stat.S_ISDIR(st.st_mode):
                if depth >= MAX_STAGE_DEPTH:
                    skipped.append(f"toodeep:{_vpath(rel)}")
                else:
                    visit(entry.path, rel, depth + 1)
            elif stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
                candidates.append((rel, entry.path))
            else:
                skipped.append(f"special:{_vpath(rel)}")

    visit(src_root, "", 1)
    limit = int(task.max_output_bytes)
    for rel, src in candidates:
        if len(staged) >= MAX_STAGED_FILES:
            skipped.append(f"toomany:{_vpath(rel)}")
            continue
        dst = os.path.join(os.fspath(dest), rel)
        try:
            fd, st = _open_regular(src)
        except OSError:
            skipped.append(f"special:{_vpath(rel)}")
            continue
        try:
            if st.st_size > limit:
                skipped.append(f"toolarge:{_vpath(rel)}")
                continue
            os.makedirs(os.path.dirname(dst), mode=0o755, exist_ok=True)
            out_fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC, 0o644)
            try:
                h, got, ok = hashlib.sha256(), 0, True
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        break
                    got += len(chunk)
                    if got > limit:                      # grew while we copied
                        ok = False
                        break
                    h.update(chunk)
                    _write_all(out_fd, chunk)
            finally:
                os.close(out_fd)
            if ok:
                staged[rel] = h.hexdigest()
            else:
                os.unlink(dst)
                skipped.append(f"toolarge:{_vpath(rel)}")
        finally:
            os.close(fd)
    return _digest(staged), skipped


def stage_outputs(run_ws: Path, dest: Path, task: TaskSpec) -> str:
    """Copy only regular files from ``<run_ws>/outputs`` into the new directory ``dest``; return ``outputs_hash`` (12 hex).

    A name must match ``^[A-Za-z0-9._-]{1,64}$``; depth <= 2; at most 32 files of at most ``task.max_output_bytes`` each;
    ``O_NOFOLLOW`` and ``st_nlink == 1``. Anything else (symlinks including ``outputs`` itself, FIFOs, hard links) is
    skipped, never followed.
    """
    return stage_outputs_report(run_ws, dest, task)[0]


# --------------------------------------------------------------------------------------------------
# trusted/ stat snapshots: a write-then-restore leaves a different ctime even when the bytes are identical again
# --------------------------------------------------------------------------------------------------
TrustedStat = dict[str, tuple]
MAX_TRUSTED_ENTRIES = 50_000
SOLO_WAIT_SEC = 120.0                # longest a solo run waits for the others before falling back to a shared window


def _stat_entry(st: os.stat_result, names: Optional[tuple[str, ...]]) -> tuple:
    return (st.st_mode, st.st_size, st.st_ino, st.st_nlink, st.st_uid, st.st_gid, st.st_mtime_ns, st.st_ctime_ns, names)


def trusted_stat(root: str | os.PathLike) -> TrustedStat:
    """relpath (``"."`` for the root) -> stat fingerprint of every file, directory and link under ``root``.

    Only ``lstat`` and directory listings are used: no content is read, no link below the root is followed (the root itself
    may be a link and is then resolved), ``atime`` is not part of the
    fingerprint (a read must not count as a change) and a tree of a few MB of labels costs a handful of syscalls. An entry
    that cannot be examined is recorded as ``("unreadable",)``, a missing root as ``("missing",)`` and a tree with more than
    ``MAX_TRUSTED_ENTRIES`` entries stops with ``("too_many",)``, so each of those differs from a normal snapshot."""
    root_s = os.fspath(root)
    out: TrustedStat = {}
    try:
        st = os.lstat(root_s)
        if stat.S_ISLNK(st.st_mode):
            # ``trusted`` may itself be a link to the real directory (the task loader and the content hash follow it): record
            # the link, then watch the directory it points to, or the whole tree would go unwatched. Swapping the link for
            # another directory changes the recorded target and the inode of the watched root.
            out["<link>"] = ("link", os.readlink(root_s), st.st_ino, st.st_ctime_ns)
            root_s = os.path.realpath(root_s)
            st = os.lstat(root_s)
    except OSError:
        return {".": ("missing",)}
    queue: deque[tuple[str, str, os.stat_result]] = deque([("", root_s, st)])
    seen = 0
    while queue:
        rel, path, st = queue.popleft()
        key = rel or "."
        names: Optional[tuple[str, ...]] = None
        children: list[tuple[str, str]] = []
        if stat.S_ISDIR(st.st_mode):
            try:
                with os.scandir(path) as it:
                    listing = sorted(entry.name for entry in it)
            except OSError:
                out[key] = ("unreadable",)
                continue
            names = tuple(listing)
            children = [(f"{rel}/{n}" if rel else n, os.path.join(path, n)) for n in listing]
        out[key] = _stat_entry(st, names)
        for child_rel, child_path in children:
            seen += 1
            if seen > MAX_TRUSTED_ENTRIES:
                out["."] = ("too_many",)
                return out
            try:
                queue.append((child_rel, child_path, os.lstat(child_path)))
            except OSError:
                out[child_rel] = ("unreadable",)
    return out


def _is_bytecode(rel: str) -> bool:
    return rel.endswith(".pyc") or "__pycache__" in rel.split("/")


def _is_dir_entry(entry: Optional[tuple]) -> bool:
    return entry is not None and len(entry) == 9 and stat.S_ISDIR(entry[0])


def _bytecode_names(entry: tuple) -> frozenset:
    return frozenset(n for n in entry[8] if _is_bytecode(n))


def _loose(entry: tuple) -> tuple:
    """A fingerprint that does not move when bytecode is added to or removed from a directory: for a directory only its
    mode, inode, owner and the names of its non-bytecode entries (its size, times and link count are dropped)."""
    if _is_dir_entry(entry):
        return (entry[0], entry[2], entry[4], entry[5], tuple(n for n in entry[8] if not _is_bytecode(n)))
    return entry


def trusted_stat_diff(before: TrustedStat, after: TrustedStat, *, ignore_bytecode: bool = False) -> list[str]:
    """Sorted ``trusted/<path>`` names (``trusted`` for the root itself) whose fingerprint differs, appears or vanishes.

    ``ignore_bytecode`` leaves out ``__pycache__`` and ``*.pyc`` entries, the way the content hash of ``trusted/`` does (and
    the times of a directory that gained or lost such an entry, which move when it is created). It is used after the grader
    has run, because a grader that writes bytecode into its own directory has always been tolerated; it is NOT used right
    after the untrusted run, where a new ``*.pyc`` in ``trusted/`` is exactly what a planted import shadow looks like."""
    names = set()
    for k in before.keys() | after.keys():
        if ignore_bytecode and k != "." and _is_bytecode(k):
            continue
        x, y = before.get(k), after.get(k)
        if ignore_bytecode and _is_dir_entry(x) and _is_dir_entry(y) and _bytecode_names(x) != _bytecode_names(y):
            # only a directory that gained or lost bytecode is excused its new times; any other change of its times means a
            # file came and went (e.g. a module planted for the grader's import and removed again), which is not bytecode
            x, y = _loose(x), _loose(y)
        if x != y:
            names.add(k)
    return [("trusted" if k == "." else f"trusted/{_vpath(k)}") for k in sorted(names)]


def stray_bytecode(snapshot: TrustedStat) -> list[str]:
    """``trusted/<path>`` of every ``*.pyc`` in a ``trusted_stat`` snapshot that is not inside a ``__pycache__`` directory.

    Python imports a source-less ``random.pyc`` from the script's directory ahead of the standard library, and the content
    hash of ``trusted/`` ignores every ``*.pyc``, so such a file would run inside the trusted grader (with the labels at hand)
    without changing ``grader_hash``. A trusted directory has no business containing one, so the mere presence is reported,
    whoever put it there and whenever: a plant that survived the window it was made in must not poison the next evaluation."""
    return [f"trusted/{_vpath(k)}" for k in sorted(snapshot)
            if k != "." and k.endswith(".pyc") and "__pycache__" not in k.split("/")[:-1]]


class _RunGate:
    """Counts the windows (untrusted runs, each with the grading that follows it) of this process so a ``trusted/``
    difference can be attributed, or declared ambiguous.

    A window's ticket records how many other windows were active when it entered and the start counter right after it; the
    window is ``contested`` when any other was active at entry or has started since. A ``solo`` entry waits until nothing
    else is registered and keeps new entries out until it leaves, so its window can never be contested."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._active = 0
        self._starts = 0                 # windows ever entered
        self._solo = False
        self._solo_waiting = 0

    def enter(self, *, solo: bool, wait_sec: Optional[float] = None) -> tuple[int, int, bool]:
        """Register a window; returns ``(windows active before, start counter after, solo granted)``. Waits at most
        ``wait_sec`` (default ``SOLO_WAIT_SEC``; a solo request that times out becomes an ordinary shared window), so it
        can never block for good. A shared request that waited that long enters anyway, which breaks the exclusivity of a
        solo window that lasted longer: ``TrustedWatch.contested`` then says so."""
        wait_sec = SOLO_WAIT_SEC if wait_sec is None else wait_sec
        with self._cond:
            granted = False
            if solo:
                self._solo_waiting += 1
                try:
                    granted = self._cond.wait_for(lambda: self._active == 0 and not self._solo, timeout=wait_sec)
                finally:
                    self._solo_waiting -= 1
                self._solo = self._solo or granted
            else:
                self._cond.wait_for(lambda: not self._solo and not self._solo_waiting, timeout=wait_sec)
            others = self._active
            self._active += 1
            self._starts += 1
            return others, self._starts, granted

    def leave(self, *, solo: bool) -> None:
        with self._cond:
            self._active -= 1
            if solo:
                self._solo = False
            self._cond.notify_all()

    def contested(self, others: int, starts: int) -> bool:
        with self._cond:
            return others > 0 or self._starts != starts


_GATE = _RunGate()


class TrustedWatch:
    """Context manager that detects a change to ``trusted/`` between two moments, restored bytes included.

    Entering registers the window with the in-process gate and takes the baseline ``trusted_stat``; ``arm()`` takes the
    baseline again (``run_untrusted`` calls it immediately before the process is spawned); ``check()`` returns the
    ``trusted/<path>`` names that differ from the baseline right now (empty when nothing changed); ``contested`` says whether
    another window overlapped this one, in which case a difference may be that run's doing (the module docstring says what
    the caller then does). ``solo`` first waits for a quiet process and then keeps others out. Exceptions are never
    swallowed."""

    def __init__(self, trusted_dir: str | os.PathLike, *, solo: bool = False):
        self.trusted_dir = os.fspath(trusted_dir)
        self._want_solo = solo
        self.solo = False                # whether the quiet process was really granted
        self.baseline: TrustedStat = {}
        self._others, self._starts = 0, 0
        self._entered = False

    def __enter__(self) -> "TrustedWatch":
        self._others, self._starts, self.solo = _GATE.enter(solo=self._want_solo)
        self._entered = True
        try:
            self.arm()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        if self._entered:
            self._entered = False
            _GATE.leave(solo=self.solo)

    def arm(self) -> None:
        """Take the baseline now (immediately before the untrusted process starts)."""
        self.baseline = trusted_stat(self.trusted_dir)

    def check(self, *, ignore_bytecode: bool = False) -> list[str]:
        """``trusted/<path>`` names that differ from the baseline now; never re-baselines (see ``trusted_stat_diff``)."""
        return trusted_stat_diff(self.baseline, trusted_stat(self.trusted_dir), ignore_bytecode=ignore_bytecode)

    @property
    def contested(self) -> bool:
        # also for a solo window: a waiter that gave up after SOLO_WAIT_SEC enters anyway, and that must show here, or a change
        # made by that other run would be charged to the (honest) solo run
        return _GATE.contested(self._others, self._starts)


# --------------------------------------------------------------------------------------------------
# /proc helpers (Linux): finding every process that belongs to a run
# --------------------------------------------------------------------------------------------------
class _Proc(NamedTuple):
    ppid: int
    pgrp: int
    sid: int
    state: str
    cpu_sec: float
    start: int


_TICKS = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") and "SC_CLK_TCK" in os.sysconf_names else 100.0


def _proc_table() -> dict[int, _Proc]:
    table: dict[int, _Proc] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                raw = fh.read()
            f = raw[raw.rindex(b")") + 2:].split()
            table[int(name)] = _Proc(ppid=int(f[1]), pgrp=int(f[2]), sid=int(f[3]), state=f[0].decode("ascii", "replace"),
                                     cpu_sec=(int(f[11]) + int(f[12])) / _TICKS, start=int(f[19]))
        except (OSError, ValueError, IndexError):
            continue
    return table


def _has_exited(proc: subprocess.Popen) -> bool:
    """True once the child has exited, without reaping it (so its pid and process group cannot be recycled yet)."""
    if proc.returncode is not None:
        return True
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except (AttributeError, ChildProcessError, OSError):
        return proc.poll() is not None


class _Tree:
    """Tracks the processes spawned by one run and kills all of them."""

    def __init__(self, leader: int, cwd: Optional[str], pipe_tags: set[str]):
        self.leader = leader
        self.cwd = os.path.realpath(cwd) if cwd else None
        self.pipes = pipe_tags
        self.known: dict[int, int] = {}          # pid -> start time (guards against pid reuse)
        self.enabled = _LINUX
        self.escaped = 0

    def _members(self, table: dict[int, _Proc], deep: bool) -> set[int]:
        me = os.getpid()
        seeds = {pid for pid, e in table.items() if pid != me and e.state != "Z"
                 and (pid == self.leader or e.sid == self.leader or e.pgrp == self.leader)}
        seeds |= {pid for pid, start in self.known.items() if pid in table and table[pid].start == start
                  and table[pid].state != "Z" and pid != me}
        if deep:
            seeds |= {pid for pid, e in table.items()
                      if pid != me and e.ppid != me and e.state != "Z" and self._holds(pid)}
        children: dict[int, list[int]] = {}
        for pid, e in table.items():
            children.setdefault(e.ppid, []).append(pid)
        out: set[int] = set()
        stack = list(seeds)
        while stack:
            pid = stack.pop()
            if pid in out or pid == me or pid not in table or table[pid].state == "Z":
                continue
            out.add(pid)
            stack.extend(children.get(pid, ()))
        return out

    def _holds(self, pid: int) -> bool:
        """Does ``pid`` have its working directory inside the run workspace or hold one of the run's output pipes?"""
        if self.cwd:
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd")
                if cwd == self.cwd or cwd.startswith(self.cwd + os.sep):
                    return True
            except OSError:
                pass
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    if os.readlink(f"/proc/{pid}/fd/{fd}") in self.pipes:
                        return True
                except OSError:
                    continue
        except OSError:
            pass
        return False

    def poll(self) -> Optional[float]:
        """Remember live descendants; return the CPU seconds used by the run so far (None off Linux)."""
        if not self.enabled:
            return None
        try:
            table = _proc_table()
        except OSError:
            self.enabled = False
            return None
        members = self._members(table, deep=False)
        for pid in members:
            self.known[pid] = table[pid].start
        return float(sum(table[p].cpu_sec for p in members))

    def kill_all(self, proc: subprocess.Popen) -> None:
        """SIGKILL the process group, then everything else that belongs to the run, until nothing is left; reap the leader."""
        for _ in range(40):
            try:
                os.killpg(self.leader, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            if not self.enabled:
                break
            try:
                table = _proc_table()
            except OSError:
                break
            targets = self._members(table, deep=True)
            if not targets:
                break
            for pid in targets:
                if table[pid].sid != self.leader and table[pid].pgrp != self.leader:
                    self.escaped += 1
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.005)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# --------------------------------------------------------------------------------------------------
# environment, rlimits, shim
# --------------------------------------------------------------------------------------------------
SHIM = r'''
import json, os, sys
cfg = json.loads(os.environ.pop("NIGHTSHIFT_SHIM_CONFIG", "{}"))
try:
    import resource
except Exception:
    resource = None
if resource is not None:
    for item in cfg.get("limits", ()):
        try:
            name, soft, hard = item
            resource.setrlimit(getattr(resource, name), (soft, hard))
        except Exception:
            pass
if cfg.get("drop_caps"):
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        for cap in range(64):
            libc.prctl(24, cap, 0, 0, 0)
        class _H(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
        class _D(ctypes.Structure):
            _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]
        if libc.capset(ctypes.byref(_H(0x20080522, 0)), (_D * 2)()) != 0:
            raise OSError("capset failed")
        libc.prctl(38, 1, 0, 0, 0)
    except Exception as exc:
        sys.stderr.write("nightshift: cannot drop capabilities: %s\n" % exc)
        os._exit(126)
try:
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
except OSError as exc:
    sys.stderr.write("nightshift: cannot execute %r: %s\n" % (sys.argv[1], exc))
    os._exit(127)
'''


def _resolve_cmd(cmd: Sequence[str]) -> list[str]:
    """``python``/``python3`` mean the interpreter running Nightshift, so the child sees the same packages."""
    out = [str(c) for c in cmd]
    if out and out[0] in ("python", "python3"):
        out[0] = sys.executable
    return out


def _requires_pythonpath(task: TaskSpec) -> list[str]:
    """Parent directory of each required module: HOME is the workspace, so user-site packages need an explicit path."""
    stdlib = sysconfig.get_path("stdlib")
    paths: list[str] = []
    for mod in task.requires:
        try:
            spec = importlib.util.find_spec(mod.split(".")[0])
        except (ImportError, ValueError, AttributeError):
            continue
        if spec is None:
            continue
        if spec.submodule_search_locations:
            parent = os.path.dirname(os.fspath(list(spec.submodule_search_locations)[0]).rstrip(os.sep))
        elif spec.origin and os.path.isabs(spec.origin):
            parent = os.path.dirname(spec.origin)
        else:
            continue
        if parent and parent != stdlib and parent not in paths:
            paths.append(parent)
    return paths


def _base_env(task: TaskSpec, home: str) -> dict[str, str]:
    path = os.pathsep.join(dict.fromkeys([os.path.dirname(sys.executable), "/usr/local/bin", "/usr/bin", "/bin"]))
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "HOME": home, "PATH": path}
    pp = _requires_pythonpath(task)
    if pp:
        env["PYTHONPATH"] = os.pathsep.join(pp)
    return env


def trusted_env(task: TaskSpec, home: str) -> dict[str, str]:
    """Scrubbed environment for trusted subprocesses (the grader): nothing is inherited from the parent."""
    return _base_env(task, home)


def _child_env(task: TaskSpec, run_ws: str, seed: int, attempt: int) -> dict[str, str]:
    env = _base_env(task, run_ws)
    env.update({"NIGHTSHIFT_SEED": str(seed), "NIGHTSHIFT_WORKSPACE": run_ws, "NIGHTSHIFT_ATTEMPT": str(attempt)})
    return env


def _task_count_for_uid() -> Optional[int]:
    """Threads and processes currently owned by this user (what RLIMIT_NPROC counts)."""
    uid, n = os.geteuid(), 0
    try:
        for name in os.listdir("/proc"):
            if name.isdigit():
                try:
                    if os.stat(f"/proc/{name}").st_uid == uid:
                        n += len(os.listdir(f"/proc/{name}/task"))
                except OSError:
                    continue
    except OSError:
        return None
    return n


def _limit_plan(task: TaskSpec, timeout_sec: float, *, trusted: bool = False) -> list[tuple[str, int, int]]:
    """(rlimit name, soft, hard) triples. Each is applied in its own try/except by the shim, so any may be skipped."""
    cpu = int(math.ceil(timeout_sec)) + 10
    plan: list[tuple[str, int, int]] = [("RLIMIT_CORE", 0, 0), ("RLIMIT_CPU", cpu, cpu + 5)]
    if not trusted:
        fsize = max(int(task.max_output_bytes), 1 << 20)
        plan.append(("RLIMIT_FSIZE", fsize, fsize))
    if _LINUX:
        plan.append(("RLIMIT_NOFILE", 1024 if trusted else 256, 1024 if trusted else 256))
        if not trusted:
            plan.append(("RLIMIT_AS", 4 << 30, 4 << 30))
            if os.geteuid() != 0:
                count = _task_count_for_uid()
                if count is not None:
                    plan.append(("RLIMIT_NPROC", count + 64, count + 64))
    return plan


def _shim_argv(cmd: Sequence[str]) -> list[str]:
    return [sys.executable, "-I", "-c", SHIM, *_resolve_cmd(cmd)]


# --------------------------------------------------------------------------------------------------
# isolation (user/mount/pid/net namespaces) chosen by really running a probe
# --------------------------------------------------------------------------------------------------
_iso_lock = threading.Lock()
_net_probe_lock = threading.Lock()
_iso_info: dict[str, object] = {}

_USERNS_PROBE = r'''
import json, os, socket, subprocess, sys
secret_dir, ledger = sys.argv[1], sys.argv[2]
out = {}
try:
    out["secret"] = sorted(os.listdir(secret_dir))
except OSError as exc:
    out["secret"] = ["!" + str(exc)]
try:
    out["ledger_size"] = os.path.getsize(ledger)
except OSError:
    out["ledger_size"] = -1
out["ifaces"] = sorted(n for _, n in socket.if_nameindex())
out["pid"] = os.getpid()
out["procs"] = len([d for d in os.listdir("/proc") if d.isdigit()])
cap = 1
with open("/proc/self/status") as fh:
    for line in fh:
        if line.startswith("CapEff:"):
            cap = int(line.split()[1], 16)
out["capeff"] = cap
try:
    out["umount_ok"] = subprocess.run(["umount", secret_dir], capture_output=True).returncode == 0
except OSError:
    out["umount_ok"] = False
print(json.dumps(out))
'''
_NET_PROBE = "import json, socket; print(json.dumps(sorted(n for _, n in socket.if_nameindex())))"
_PIDNS_PROBE = r'''
import json, os, socket, subprocess
out = {"ifaces": sorted(n for _, n in socket.if_nameindex()), "pid": os.getpid(),
       "procs": len([d for d in os.listdir("/proc") if d.isdigit()])}
cap = 1
with open("/proc/self/status") as fh:
    for line in fh:
        if line.startswith("CapEff:"):
            cap = int(line.split()[1], 16)
out["capeff"] = cap
try:
    out["umount_ok"] = subprocess.run(["umount", "/proc"], capture_output=True).returncode == 0
except OSError:
    out["umount_ok"] = False
print(json.dumps(out))
'''
_EXIT_PROBE = "import sys; sys.exit(7)"
PIDNS_FLAGS = ("--pid", "--fork", "--mount-proc")


def prefix_isolates_pids(prefix: Optional[Sequence[str]]) -> bool:
    """True for an ``unshare`` prefix that also gives the run its own pid namespace and ``/proc`` (the root-only variant)."""
    return bool(prefix) and "--pid" in prefix


def _mask_script(masks: Iterable[os.PathLike | str]) -> str:
    lines = []
    for m in masks:
        real = os.path.realpath(os.fspath(m))
        q = shlex.quote(real)
        if os.path.isdir(real):
            cmd = f"mount -t tmpfs -o size=64k,mode=0555,nosuid,nodev,noexec tmpfs {q}"
        elif os.path.isfile(real):
            cmd = f"mount --bind /dev/null {q}"
        else:
            continue
        lines.append(f"{cmd} || {{ echo NIGHTSHIFT_MASK_FAILED >&2; exit {MASK_FAILED_EXIT}; }}\n")
    return "".join(lines)


def _wrap_argv(level: str, inner: Sequence[str], masks: Sequence[os.PathLike | str], info: dict[str, object]) -> list[str]:
    if level == "userns":
        script = _mask_script(masks) + 'exec "$@"'
        return [str(info["unshare"]), "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--mount-proc",
                "--", str(info["sh"]), "-c", script, "sh", *inner]
    if level == "unshare-net":
        return [*info["net_prefix"], "--", *inner]       # type: ignore[misc]
    return list(inner)


def _probe_run(argv: list[str], cfg: dict) -> tuple[Optional[int], bytes]:
    env = {"PATH": os.pathsep.join(dict.fromkeys([os.path.dirname(sys.executable), "/usr/bin", "/bin"])),
           "NIGHTSHIFT_SHIM_CONFIG": json.dumps(cfg)}
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                env=env, start_new_session=True)
    except OSError:
        return None, b""
    try:
        out, _ = proc.communicate(timeout=20)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        return None, b""
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


def _probe_pidns_prefix(prefix: list[str], sh: str) -> bool:
    """Does ``prefix`` (an ``unshare --net --pid --fork --mount-proc`` prefix) really hide every other process?

    Run through the same shim that drops capabilities in a real run, the child must see only the empty network, be pid 1 with
    a nearly empty private ``/proc`` (so the orchestrator's ``/proc/<pid>/environ`` does not exist for it), hold no capability
    and be unable to unmount that ``/proc``; and the exit status of the command must come back unchanged."""
    info = {"net_prefix": prefix, "sh": sh}
    inner = [sys.executable, "-I", "-c", SHIM, sys.executable, "-I", "-c", _PIDNS_PROBE]
    rc, out = _probe_run(_wrap_argv("unshare-net", inner, [], info), {"drop_caps": True})
    try:
        seen = json.loads(out.decode("utf-8").strip().splitlines()[-1]) if rc == 0 else {}
    except (ValueError, IndexError):
        return False
    if not (seen.get("ifaces") == ["lo"] and seen.get("pid") == 1 and seen.get("procs", 99) <= 8
            and seen.get("capeff") == 0 and seen.get("umount_ok") is False):
        return False
    status_inner = [sys.executable, "-I", "-c", SHIM, sys.executable, "-I", "-c", _EXIT_PROBE]
    rc, _ = _probe_run(_wrap_argv("unshare-net", status_inner, [], info), {"drop_caps": True})
    return rc == 7


def _probe_net_prefix(unshare: str, sh: str) -> Optional[list[str]]:
    """The first ``unshare`` prefix whose child really sees an empty network (only ``lo``), or None.

    Candidates, strongest first: ``--net --pid --fork --mount-proc`` (only works as root; proven by ``_probe_pidns_prefix``:
    the run then cannot see the orchestrator's ``/proc/<pid>/environ`` even though root defeats ``harden_parent``), plain
    ``--net`` and ``--user --map-root-user --net``. A candidate that fails its probe is skipped, never assumed."""
    hardened = [unshare, "--net", *PIDNS_FLAGS]
    if os.geteuid() == 0 and _probe_pidns_prefix(hardened, sh):
        return hardened
    for prefix in ([unshare, "--net"], [unshare, "--user", "--map-root-user", "--net"]):
        argv = _wrap_argv("unshare-net", [sys.executable, "-I", "-c", _NET_PROBE], [], {"net_prefix": prefix, "sh": sh})
        rc, out = _probe_run(argv, {})
        if rc == 0 and out.decode("utf-8", "replace").strip() == '["lo"]':
            return prefix
    return None


def _probe_isolation() -> dict[str, object]:
    """Really run the wrappers. ``level`` is the strongest one whose probe showed its effect; ``reason`` says why a stronger
    one was not proven. ``net_prefix`` is only filled when ``unshare-net`` is the best level (see ``_net_prefix_for``)."""
    info: dict[str, object] = {"level": "none", "unshare": None, "sh": None, "net_prefix": None, "reason": ""}
    if not _LINUX:
        info["reason"] = "not Linux"
        return info
    unshare = shutil.which("unshare")
    sh = shutil.which("sh") or ("/bin/sh" if os.path.exists("/bin/sh") else None)
    if not unshare or not sh:
        info["reason"] = "unshare or sh not found on PATH"
        return info
    info.update(unshare=unshare, sh=sh)
    with tempfile.TemporaryDirectory(prefix="nightshift-probe-") as tmp:
        secret_dir = os.path.join(tmp, "secret")
        os.mkdir(secret_dir)
        with open(os.path.join(secret_dir, "s.txt"), "w") as fh:
            fh.write("x")
        ledger = os.path.join(tmp, "ledger.jsonl")
        with open(ledger, "w") as fh:
            fh.write("ledger")
        inner = [sys.executable, "-I", "-c", SHIM, sys.executable, "-I", "-c", _USERNS_PROBE, secret_dir, ledger]
        rc, out = _probe_run(_wrap_argv("userns", inner, [secret_dir, ledger], info), {"drop_caps": True})
        try:
            seen = json.loads(out.decode("utf-8").strip().splitlines()[-1]) if rc == 0 else {}
        except (ValueError, IndexError):
            seen = {}
        if (seen.get("secret") == [] and seen.get("ledger_size") == 0 and seen.get("ifaces") == ["lo"]
                and seen.get("pid") == 1 and seen.get("procs", 99) <= 8 and seen.get("capeff") == 0
                and seen.get("umount_ok") is False):
            info["level"] = "userns"
            return info
    info["reason"] = "user namespaces unavailable (probe failed)"
    prefix = _probe_net_prefix(unshare, sh)
    if prefix is not None:
        info.update(level="unshare-net", net_prefix=prefix)
    else:
        info["reason"] = "user and network namespaces unavailable (probes failed)"
        info["net_probed"] = True
    return info


def _net_prefix_for(info: dict[str, object]) -> Optional[list[str]]:
    """The proven ``unshare --net`` prefix, probing it on demand (and caching the answer, also a negative one).

    The ``userns`` probe returns early, so a machine whose best level is ``userns`` has not yet shown that ``unshare-net``
    works; forcing that weaker mode proves it first rather than assuming it."""
    prefix = info.get("net_prefix")
    if prefix is not None:
        return list(prefix)                                   # type: ignore[call-overload]
    if info.get("net_probed") or not info.get("unshare") or not info.get("sh"):
        return None
    with _net_probe_lock:                                     # concurrent first uses prove it once, not once each
        with _iso_lock:
            cached = dict(_iso_info)
        if cached.get("unshare") == info["unshare"] and (cached.get("net_probed") or cached.get("net_prefix") is not None):
            return list(cached["net_prefix"]) if cached.get("net_prefix") is not None else None   # type: ignore[call-overload]
        found = _probe_net_prefix(str(info["unshare"]), str(info["sh"]))
        with _iso_lock:
            if _iso_info.get("unshare") == info["unshare"]:
                _iso_info.update(net_prefix=found, net_probed=True)
        return found


def isolation_override() -> str:
    """The validated value of ``NIGHTSHIFT_ISOLATION``: ``auto`` (also when unset or blank), ``userns``, ``unshare-net`` or
    ``none``, case-insensitive. Any other value raises ``ValueError`` naming the allowed ones. Read on every call, never
    cached, so changing the variable takes effect at once."""
    raw = os.environ.get(ISOLATION_ENV)
    if raw is None or not raw.strip():
        return "auto"
    value = raw.strip().lower()
    if value not in ISOLATION_CHOICES:
        raise ValueError(f"{ISOLATION_ENV}={raw!r} is not valid; allowed values: {', '.join(ISOLATION_CHOICES)}")
    return value


def _probed(refresh: bool = False) -> dict[str, object]:
    """A copy of the cached probe result (run once per process; ``refresh`` runs it again). Independent of the override."""
    with _iso_lock:
        if refresh or not _iso_info:
            _iso_info.clear()
            _iso_info.update(_probe_isolation())
        return dict(_iso_info)


def _effective_isolation(requested: str, *, refresh: bool = False) -> tuple[str, dict[str, object], list[str]]:
    """(level to apply, probe info, notes) for an explicit ``requested`` mode combined with ``NIGHTSHIFT_ISOLATION``.

    Both only LOWER the isolation: the result is never stronger than what the probe proved, nor than either request. When a
    request names a mode the probe did not prove, the strongest proven mode below it is used and ``notes`` says why. A
    level of ``none`` demanded by the request or the environment is applied without running the probe at all."""
    if requested != "auto" and requested not in _RANK:
        raise ValueError(f"unknown isolation {requested!r}; allowed values: {', '.join(ISOLATION_CHOICES)}")
    override = isolation_override()
    caps = [(label, want) for label, want in ((f"isolation={requested}", requested), (f"{ISOLATION_ENV}={override}", override))
            if want != "auto"]
    if not refresh and any(want == "none" for _, want in caps):
        with _iso_lock:
            info = dict(_iso_info)
        # a caller that asked for ``none`` itself lost nothing to the variable, so the variable is not blamed for it
        notes = [f"isolation set to none by {ISOLATION_ENV}=none"] if override == "none" and requested != "none" else []
        return "none", info, notes
    info = _probed(refresh)
    best = str(info["level"])
    why = str(info.get("reason") or f"probe proved {best}")
    level = min([best, *(want for _, want in caps)], key=_RANK.__getitem__)
    if level == "unshare-net":
        prefix = _net_prefix_for(info)
        if prefix is None:
            level, why = "none", "network namespaces unavailable (probe failed)"
        else:
            info["net_prefix"] = prefix
    notes = []
    for label, want in caps:
        if _RANK[want] > _RANK[level]:
            notes.append(f"{label} not available ({why}); applied {level}")
    if override != "auto" and _RANK[override] < _RANK[best] and level == override:
        notes.append(f"isolation lowered from {best} to {level} by {ISOLATION_ENV}={override}")
    return level, info, notes


def available_isolation(*, refresh: bool = False) -> str:
    """``"userns"`` > ``"unshare-net"`` > ``"none"``: the best wrapper whose probe command really succeeded and showed the
    expected effect (masked directories, an empty network, a fresh pid namespace, no capabilities), lowered (never raised)
    by ``NIGHTSHIFT_ISOLATION``. ``"none"`` off Linux. The probe result is cached per process (``refresh=True`` repeats it)
    and the override is applied on every call, so the answer always matches the variable's current value."""
    return _effective_isolation("auto", refresh=refresh)[0]


def effective_isolation(requested: str = "auto") -> str:
    """The mode ``run_untrusted(isolation=requested)`` would apply right now (same rules, same ``ValueError``)."""
    return _effective_isolation(requested)[0]


def harden_parent() -> None:
    """prctl(PR_SET_DUMPABLE, 0): a NON-ROOT process of the same uid cannot read ``/proc/<pid>/environ`` or ``/mem`` of this
    process. Never raises.

    What it does NOT protect against: root, or any process with ``CAP_SYS_PTRACE``, reads those files whatever the dumpable
    flag says, so when Nightshift itself runs as root an experiment in the ``none`` mode (and in ``unshare-net`` unless its
    pid-namespace variant was proven, see ``prefix_isolates_pids``) can read the API key from the orchestrator's environment.
    Under ``userns`` the pid namespace hides the process altogether. On macOS there is no ``prctl`` and no ``/proc``: this
    does nothing, and a same-user experiment can read the environment of the orchestrator. Run Nightshift as a non-root
    user inside a container or VM when an API key or sealed numbers matter and ``userns`` is unavailable."""
    if not _LINUX:
        return
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)
    except Exception:  # noqa: BLE001 - hardening is best effort and must never stop a run
        pass


# --------------------------------------------------------------------------------------------------
# supervised execution
# --------------------------------------------------------------------------------------------------
class _Sink:
    """Receives one stream: optional capped log file plus a rolling tail."""

    def __init__(self, fd: Optional[int], cap: int, tail_cap: int = TAIL_BYTES):
        self.fd, self.cap, self.tail_cap = fd, cap, tail_cap
        self.written = 0
        self.total = 0
        self._tail = bytearray()

    def feed(self, chunk: bytes) -> bool:
        """Record ``chunk``; True once more than ``cap`` bytes have arrived in total."""
        self.total += len(chunk)
        self._tail += chunk
        if len(self._tail) > 2 * self.tail_cap:
            del self._tail[:-self.tail_cap]
        if self.fd is not None and self.written < self.cap:
            part = chunk[: self.cap - self.written]
            _write_all(self.fd, part)
            self.written += len(part)
        return self.total > self.cap

    def tail(self) -> bytes:
        return bytes(self._tail[-self.tail_cap:])


@dataclass
class _Outcome:
    exit_code: Optional[int]
    timed_out: bool
    stalled: bool
    flooded: bool
    duration_sec: float
    pid: Optional[int]
    escaped: int
    spawn_error: Optional[str]


def _pump(sel: selectors.BaseSelector, timeout: float) -> bool:
    """Read what is ready. True if any sink exceeded its cap."""
    flooded = False
    for key, _ in sel.select(timeout):
        try:
            chunk = os.read(key.fd, 1 << 16)
        except BlockingIOError:
            continue
        except OSError:
            chunk = b""
        if not chunk:
            sel.unregister(key.fileobj)
            continue
        flooded = key.data.feed(chunk) or flooded
    return flooded


def _execute(argv: list[str], *, env: dict[str, str], cwd: str, timeout_sec: float, out: _Sink, err: _Sink,
             heartbeat_sec: float = 2.0, stall_sec: Optional[float] = None, sweep_cwd: Optional[str] = None,
             on_start: Optional[Callable[[int], None]] = None,
             on_heartbeat: Optional[Callable[[int, Optional[float]], None]] = None,
             on_stalled: Optional[Callable[[float], None]] = None) -> _Outcome:
    """Run ``argv`` in its own session. The process group (and stragglers) is killed on EVERY exit path, hooks raising included."""
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
                                env=env, start_new_session=True, close_fds=True)
    except OSError as exc:
        return _Outcome(None, False, False, False, 0.0, None, 0, f"{type(exc).__name__}: {exc}")
    tags = {f"pipe:[{os.fstat(s.fileno()).st_ino}]" for s in (proc.stdout, proc.stderr) if s is not None}
    tree = _Tree(proc.pid, sweep_cwd, tags)
    sel = selectors.DefaultSelector()
    t0 = time.monotonic()
    timed_out = stalled = flooded = False
    duration = 0.0
    try:
        for stream, sink in ((proc.stdout, out), (proc.stderr, err)):
            os.set_blocking(stream.fileno(), False)
            sel.register(stream, selectors.EVENT_READ, sink)
        if on_start is not None:
            on_start(proc.pid)
        hb = max(heartbeat_sec, 0.01)
        track_every = min(hb, 0.25)
        next_track, next_hb = t0, t0 + hb
        last_bytes, last_growth, cpu = 0, t0, None
        while True:
            flooded = _pump(sel, 0.02) or flooded
            now = time.monotonic()
            if flooded or _has_exited(proc):
                break
            if now - t0 >= timeout_sec:
                timed_out = True
                break
            if now >= next_track:
                next_track = now + track_every
                cpu = tree.poll()
            total = out.total + err.total
            if total > last_bytes:
                last_bytes, last_growth = total, now
            elif stall_sec and now - last_growth >= stall_sec:
                stalled = True
                if on_stalled is not None:
                    on_stalled(now - last_growth)
                break
            if on_heartbeat is not None and now >= next_hb:
                next_hb = now + hb
                on_heartbeat(total, cpu)
        duration = time.monotonic() - t0
    finally:
        tree.kill_all(proc)
        try:
            end = time.monotonic() + 0.5
            while sel.get_map() and time.monotonic() < end:
                flooded = _pump(sel, 0.02) or flooded
        except OSError:
            pass
        sel.close()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
    return _Outcome(proc.returncode, timed_out, stalled, flooded, duration, proc.pid, tree.escaped, None)


@dataclass
class TrustedRun:
    exit_code: Optional[int]
    stdout: bytes                        # the last ``stdout_cap`` bytes
    stderr: bytes                        # the last 16 KiB; for diagnostics only, never shown to a model
    timed_out: bool
    overflow: bool                       # stdout exceeded ``stdout_cap``; the process was killed
    spawn_error: Optional[str]


def run_trusted(task: TaskSpec, argv: Sequence[str], *, cwd: str | os.PathLike, env: dict[str, str], timeout_sec: float,
                stdout_cap: int) -> TrustedRun:
    """Run a trusted command (the grader) in its own process group through the rlimit shim, with capped stdout."""
    cfg = {"limits": _limit_plan(task, timeout_sec, trusted=True)}
    full_env = dict(env)
    full_env["NIGHTSHIFT_SHIM_CONFIG"] = json.dumps(cfg)
    out, err = _Sink(None, stdout_cap, stdout_cap), _Sink(None, 1 << 30, 16 * 1024)
    res = _execute(_shim_argv(argv), env=full_env, cwd=os.fspath(cwd), timeout_sec=timeout_sec, out=out, err=err,
                   heartbeat_sec=3600.0)
    return TrustedRun(exit_code=res.exit_code, stdout=out.tail(), stderr=err.tail(), timed_out=res.timed_out,
                      overflow=res.flooded, spawn_error=res.spawn_error)


# --------------------------------------------------------------------------------------------------
# the untrusted run
# --------------------------------------------------------------------------------------------------
_INFRA_STDERR = ("MemoryError", "No space left on device", "Cannot allocate memory", "Temporary failure in name resolution",
                 "Resource temporarily unavailable")


def classify_failure(task: TaskSpec, exit_code: Optional[int], stderr: str, timed_out: bool) -> str:
    """``timeout`` | ``infra`` | ``crash`` | ``ok`` (contract section 6.2)."""
    if timed_out:
        return "timeout"
    if exit_code == 0:
        return "ok"
    if exit_code is None or exit_code in (-9, 137):
        return "infra"
    if any(p in stderr for p in _INFRA_STDERR):
        return "infra"
    for mod in task.requires:
        top = re.escape(mod.split(".")[0])
        if re.search(rf"(?:ModuleNotFoundError|ImportError)[^\n]*\b{top}\b", stderr):
            return "infra"
    return "crash"


def _scrub_file(path: str) -> None:
    """Rewrite a log file if it contains a registered secret."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        text = raw.decode("utf-8", errors="surrogateescape")
        clean = secrets.scrub_text(text)
        if clean != text:
            with open(path, "wb") as fh:
                fh.write(clean.encode("utf-8", errors="surrogateescape"))
    except OSError:
        pass


def _tail_text(sink: _Sink) -> str:
    return secrets.scrub_text(sink.tail().decode("utf-8", errors="replace"))


def run_untrusted(task: TaskSpec, run_ws: Path, seed: int, *, purpose: str = "primary", attempt: int = 1,
                  on_start: Optional[Callable[[Optional[int]], None]] = None,
                  on_heartbeat: Optional[Callable[[int, Optional[float]], None]] = None, heartbeat_sec: float = 2.0,
                  timeout_sec: Optional[float] = None, isolation: str = "auto", log_dir: Optional[Path] = None,
                  log_prefix: str = "run", on_stalled: Optional[Callable[[float], None]] = None,
                  mask_paths: Sequence[Path] = (), solo: bool = False,
                  watch: Optional[TrustedWatch] = None) -> RunResult:
    """Run ``task.run_cmd`` in ``run_ws`` as untrusted code.

    Guarantees: scrubbed environment (contract 4.3), rlimits applied best effort inside the shim, no ``preexec_fn``,
    stdout/stderr streamed to ``<log_dir>/<log_prefix>.stdout|stderr`` capped at 8 MiB each (exceeding kills the run:
    ``crash`` with note ``output_limit``), ``RunResult.stdout/stderr`` are the last 64 KiB, the wall timeout and the stall
    watchdog (no output growth for ``task.stall_sec`` -> ``on_stalled`` once, then kill, ``timeout`` + ``stalled``) are
    enforced, and the whole process group plus any stragglers are killed before this function returns or raises.
    ``mask_paths`` are hidden inside the namespace in addition to ``task.trusted_dir`` when isolation is ``userns``.
    ``isolation`` (``auto`` | ``userns`` | ``unshare-net`` | ``none``) and ``NIGHTSHIFT_ISOLATION`` can only lower the mode the
    probe proved; the mode applied is ``RunResult.isolation`` and a request that could not be honoured is explained in
    ``RunResult.note``.
    ``purpose`` is accepted for symmetry with the ledger and does not change containment.

    ``trusted/`` is watched in every mode (``TrustedWatch``): a stat snapshot is taken immediately before the process is
    spawned and compared once the process group has been reaped and swept. A difference (also one that put the original
    bytes back) is returned in ``RunResult.trusted_changed``; when another untrusted run overlapped, it is ambiguous, the
    run is classified ``infra`` and ``trusted_contested`` is set. ``solo`` makes this run wait for every other one and keeps
    them out (the retry of an ambiguous run, so that its result is decisive). A caller that wants the same watch to cover
    the grader passes its own entered ``watch``; otherwise one is created and released here.
    """
    ws = os.path.abspath(os.fspath(run_ws))
    timeout = float(task.run_timeout_sec if timeout_sec is None else timeout_sec)
    if timeout <= 0:
        raise ValueError("timeout_sec must be positive")
    level, info, iso_notes = _effective_isolation(isolation)
    masks = [task.trusted_dir, *mask_paths] if level == "userns" else []
    env = _child_env(task, ws, seed, attempt)
    # capabilities are dropped wherever a namespace is what keeps the run away from the host: the user namespace, or the pid
    # namespace of the root-only ``unshare-net`` variant (root could otherwise unmount the private /proc)
    drop_caps = level == "userns" or (level == "unshare-net" and prefix_isolates_pids(info.get("net_prefix")))   # type: ignore[arg-type]
    cfg = {"limits": _limit_plan(task, timeout), "drop_caps": drop_caps}
    env["NIGHTSHIFT_SHIM_CONFIG"] = json.dumps(cfg)
    argv = _wrap_argv(level, _shim_argv(task.run_cmd), masks, info)

    own_watch = watch is None
    if own_watch:
        watch = TrustedWatch(task.trusted_dir, solo=solo)
        watch.__enter__()
    assert watch is not None
    tmp: Optional[tempfile.TemporaryDirectory] = None
    out_fd = err_fd = -1
    try:
        if log_dir is None:
            tmp = tempfile.TemporaryDirectory(prefix="nightshift-logs-")
            log_root = tmp.name
        else:
            log_root = os.fspath(log_dir)
            os.makedirs(log_root, exist_ok=True)
        out_path, err_path = os.path.join(log_root, f"{log_prefix}.stdout"), os.path.join(log_root, f"{log_prefix}.stderr")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW | _O_CLOEXEC
        out_fd, err_fd = os.open(out_path, flags, 0o644), os.open(err_path, flags, 0o644)
        out, err = _Sink(out_fd, LOG_CAP_BYTES), _Sink(err_fd, LOG_CAP_BYTES)
        try:
            watch.arm()                                  # the snapshot is taken immediately before the process is spawned
            res = _execute(argv, env=env, cwd=ws, timeout_sec=timeout, out=out, err=err, heartbeat_sec=heartbeat_sec,
                           stall_sec=task.stall_sec, sweep_cwd=ws, on_start=on_start, on_heartbeat=on_heartbeat,
                           on_stalled=on_stalled)
            trusted_changed = watch.check()              # reaped and swept: nothing of this run is still alive (on Linux)
            trusted_contested = bool(trusted_changed) and watch.contested
        finally:
            os.close(out_fd)
            os.close(err_fd)
            _scrub_file(out_path)
            _scrub_file(err_path)
            if tmp is not None:
                tmp.cleanup()
    finally:
        if own_watch:
            watch.__exit__(None, None, None)
    stdout, stderr = _tail_text(out), _tail_text(err)
    notes: list[str] = list(iso_notes)
    if res.spawn_error:
        classification = "infra"
        notes.append(f"spawn_failed: {res.spawn_error}")
    elif res.flooded:
        classification = "crash"
        notes.append("output_limit")
    elif res.exit_code == MASK_FAILED_EXIT and "NIGHTSHIFT_MASK_FAILED" in stderr:
        classification = "infra"
        notes.append("isolation_mask_failed")
    elif res.stalled:
        classification = "timeout"
    else:
        classification = classify_failure(task, res.exit_code, stderr, res.timed_out)
    if res.stalled:
        notes.append(f"stalled: no output for {task.stall_sec:g}s")
    if res.timed_out:
        notes.append(f"timeout after {timeout:g}s")
    if res.escaped:
        notes.append(f"killed_stragglers:{res.escaped}")
    if trusted_changed:
        shown = ", ".join(trusted_changed[:5]) + (f" (+{len(trusted_changed) - 5} more)" if len(trusted_changed) > 5 else "")
        if trusted_contested:
            # another untrusted run shared the window: the difference may be theirs. Not an accusation: infra, retried alone.
            classification = "infra"
            notes.append(f"trusted_tree_changed_during_run: {shown}; the change may come from a concurrent lane, "
                         "the run is retried alone")
        else:
            notes.append(f"trusted_tree_changed: {shown}")
    return RunResult(exit_code=res.exit_code, stdout=stdout, stderr=stderr, duration_sec=res.duration_sec,
                     timed_out=res.timed_out, stalled=res.stalled, classification=classification,
                     output_bytes=out.total + err.total, pid=res.pid, note="; ".join(notes), isolation=level,
                     trusted_changed=trusted_changed, trusted_contested=trusted_contested)
