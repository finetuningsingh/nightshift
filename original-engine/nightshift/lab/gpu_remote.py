"""ssh helpers for a rented GPU: run, upload, fetch, detached start, port forward. Standard library only.

The rental is untrusted: a researcher recipe runs there as root. Everything that comes back is therefore read by privileged
control-host code through one rule, and nothing is ever extracted by ``tar -x`` or copied by ``rsync``/``scp`` on the control host:

* ``fetch`` and ``fetch_files`` ask the rental for ``tar -cf -`` of EXACT names, then parse the stream member by member with
  :mod:`tarfile` in streaming mode. A member that is not a regular file (symbolic link, hard link, device, fifo, directory), whose
  name is not exactly one that was asked for, or that is larger than the cap is refused; the bytes of an accepted member are written
  by this module into a file it creates itself (``O_EXCL``, ``O_NOFOLLOW``), so nothing on the rental can choose a local path or plant
  a link on the control host. Per-file and total size caps are enforced while reading, and the stream itself is capped.
* ``upload`` and ``upload_tree`` refuse local symbolic links, hard links and special files, and verify sha256 on the rental
  before the file is moved into place.

Remote paths are absolute POSIX paths without ``..``; every value placed into a remote command goes through ``shlex.quote``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import posixpath
import re
import shlex
import socket
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from nightshift.lab.contracts import GpuError

DEFAULT_TIMEOUT = 600.0
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SKIP_DIRS = ("__pycache__", ".git")

# Hash a file on the rental with the Python the rental already has (sha256sum is absent on macOS, where the tests run).
_REMOTE_SHA = (
    "import hashlib,sys\n"
    "h=hashlib.sha256()\n"
    "f=open(sys.argv[1],'rb')\n"
    "for c in iter(lambda:f.read(1<<20),b''):\n"
    "    h.update(c)\n"
    "print(h.hexdigest())\n"
)
# A manifest 'sha256  relative/path' of every regular file below a directory (symbolic links are listed as LINK).
_REMOTE_MANIFEST = (
    "import hashlib,os,sys\n"
    "root=sys.argv[1]\n"
    "for d,ds,fs in os.walk(root):\n"
    "    ds.sort()\n"
    "    for n in sorted(fs):\n"
    "        p=os.path.join(d,n)\n"
    "        rel=os.path.relpath(p,root)\n"
    "        if os.path.islink(p) or not os.path.isfile(p):\n"
    "            print('LINK  '+rel); continue\n"
    "        h=hashlib.sha256()\n"
    "        f=open(p,'rb')\n"
    "        for c in iter(lambda:f.read(1<<20),b''):\n"
    "            h.update(c)\n"
    "        print(h.hexdigest()+'  '+rel)\n"
)


class FetchRefused(GpuError):
    """The rental returned something a fetch refuses (a link, a device, an unexpected or oversize member, a hash mismatch). This is a
    property of the CONTENT, as opposed to an ssh failure, so callers can treat it as a failed job rather than an infrastructure fault."""


@dataclass(frozen=True)
class SshTarget:
    """Where to connect. ``known_hosts`` is a session-local file (Thunder addresses are dynamic, host keys are accepted on first use)."""

    host: str
    port: int
    user: str
    key_path: str
    known_hosts: Optional[str] = None
    connect_timeout: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not re.match(r"^[A-Za-z0-9._:-]+$", self.host):
            raise GpuError(f"SshTarget.host {self.host!r} is not a plain host name or address")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 0 < self.port < 65536:
            raise GpuError(f"SshTarget.port {self.port!r} is not a port")
        if not isinstance(self.user, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*$", self.user):
            raise GpuError(f"SshTarget.user {self.user!r} is not a plain user name")

    def options(self) -> list[str]:
        opts = [
            "-i", self.key_path, "-p", str(self.port),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-o", f"ConnectTimeout={int(self.connect_timeout)}",
        ]  # fmt: skip
        if self.known_hosts:
            opts += ["-o", f"UserKnownHostsFile={self.known_hosts}"]
        return opts

    def args(self, *extra: str) -> list[str]:
        """``ssh`` with the connection options, then ``extra`` ssh options, then ``user@host`` (a remote command goes AFTER this)."""
        return ["ssh", *self.options(), *extra, f"{self.user}@{self.host}"]

    def command_args(self, command: str) -> list[str]:
        return self.args() + [command]


@dataclass(frozen=True)
class RemoteResult:
    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class Fetched:
    name: str
    path: Path
    sha256: str
    size_bytes: int


def check_remote_path(path: str, what: str = "remote path") -> str:
    if not isinstance(path, str) or not path or "\x00" in path or "\n" in path:
        raise GpuError(f"{what}: empty or contains a control character")
    if not path.startswith("/") or (path != "/" and posixpath.normpath(path) != path):
        raise GpuError(f"{what}: {path!r} must be an absolute, normalised POSIX path")
    if ".." in path.split("/"):
        raise GpuError(f"{what}: {path!r} must not contain '..'")
    return path


def _file_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name) or name in (".", ".."):
        raise GpuError(f"file name {name!r} must be a plain name (no '/', no '..')")
    return name


MAX_OUTPUT_BYTES = 16 << 20  # stdout and stderr of one remote command are each kept at most this long: the rental is not trusted


def _drain(stream, limit: int, sink: bytearray, over: threading.Event) -> None:
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            room = limit - len(sink)
            sink += chunk[: max(0, room)]
            if len(chunk) > room:
                over.set()
                return
    except (OSError, ValueError):
        return


def _run(
    args: Sequence[str], *, timeout: float, stdin=None, input_bytes: Optional[bytes] = None, max_output: int = MAX_OUTPUT_BYTES, truncate: bool = False
) -> RemoteResult:
    """Run ``args`` keeping at most ``max_output`` bytes of each of stdout and stderr. A command that writes more is killed; that is a
    GpuError, or with ``truncate`` the first ``max_output`` bytes are returned."""
    try:
        # stdin is /dev/null unless data is given: an inherited stdin would make ssh wait for the caller's terminal or pipe.
        sin = subprocess.PIPE if input_bytes is not None else (stdin if stdin is not None else subprocess.DEVNULL)
        proc = subprocess.Popen(list(args), stdin=sin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise GpuError(f"cannot run ssh: {exc}") from None
    over = threading.Event()
    out, err = bytearray(), bytearray()
    threads = [threading.Thread(target=_drain, args=(proc.stdout, max_output, out, over), daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, max_output, err, over), daemon=True)]  # fmt: skip
    if input_bytes is not None:
        def feed() -> None:
            try:
                proc.stdin.write(input_bytes)
            except (OSError, ValueError):
                pass
            finally:
                with contextlib.suppress(OSError, ValueError):
                    proc.stdin.close()

        threads.append(threading.Thread(target=feed, daemon=True))
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    try:
        while proc.poll() is None:
            if over.is_set():
                break
            if time.monotonic() >= deadline:
                raise GpuError(f"ssh command timed out after {timeout:g} s")
            over.wait(0.05)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for t in threads:
            t.join(5.0)
        for f in (proc.stdout, proc.stderr):
            with contextlib.suppress(OSError, ValueError):
                f.close()
    if over.is_set() and not truncate:
        raise GpuError(f"the remote command wrote more than {max_output} bytes; it was stopped")
    return RemoteResult(proc.returncode, bytes(out), bytes(err))


def run(
    target: SshTarget, command: str, *, timeout: float = 120.0, input_bytes: Optional[bytes] = None, check: bool = True,
    max_output: int = MAX_OUTPUT_BYTES, truncate: bool = False,
) -> RemoteResult:  # fmt: skip
    """Run ``command`` (a shell command string, already quoted by the caller) on the rental. Raises GpuError on ssh failure or timeout;
    with ``check`` also on a non-zero exit status."""
    res = _run(target.command_args(command), timeout=timeout, input_bytes=input_bytes, max_output=max_output, truncate=truncate)
    if check and res.returncode != 0:
        raise GpuError(f"remote command failed (exit {res.returncode}): {res.stderr.decode('utf-8', 'replace')[-500:].strip()}")
    return res


def remote_sha256(target: SshTarget, remote: str, *, timeout: float = 600.0) -> str:
    check_remote_path(remote)
    res = run(target, f"python3 -I -c {shlex.quote(_REMOTE_SHA)} {shlex.quote(remote)}", timeout=timeout)
    out = res.text.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", out):
        raise GpuError(f"remote sha256 of {remote} gave {out[:80]!r}")
    return out


def read_remote_tail(target: SshTarget, remote: str, *, max_bytes: int = 20000, timeout: float = 60.0) -> str:
    """The last ``max_bytes`` of a remote text file (a log), or '' when it does not exist. For humans, never trusted."""
    check_remote_path(remote)
    res = run(
        target, f"tail -c {int(max_bytes)} {shlex.quote(remote)} 2>/dev/null || true", timeout=timeout, check=False,
        max_output=max(1, int(max_bytes)), truncate=True,
    )  # fmt: skip
    return res.stdout[-max(1, int(max_bytes)):].decode("utf-8", errors="replace")


def _local_regular(path: str | os.PathLike[str]) -> tuple[int, int]:
    """Open a local file read-only without following a link; refuse links, hard links and special files. Returns (fd, size)."""
    p = os.fspath(path)
    st = os.lstat(p)
    if stat.S_ISLNK(st.st_mode):
        raise GpuError(f"{p}: symbolic links are refused")
    if not stat.S_ISREG(st.st_mode):
        raise GpuError(f"{p}: not a regular file")
    if st.st_nlink > 1:
        raise GpuError(f"{p}: hard-linked files are refused")
    fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    return fd, st.st_size


def sha256_of_regular(path: str | os.PathLike[str]) -> str:
    fd, _ = _local_regular(path)
    h = hashlib.sha256()
    with os.fdopen(fd, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(target: SshTarget, local: str | os.PathLike[str], remote: str, *, mode: int = 0o644, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Copy one local regular file to ``remote``; the rental hashes what it received and the file is moved into place only if the hash
    equals the local one. Returns the sha256."""
    check_remote_path(remote, "upload destination")
    digest = sha256_of_regular(local)
    fd, _ = _local_regular(local)
    d, tmp = shlex.quote(posixpath.dirname(remote)), shlex.quote(remote + ".partial")
    try:
        with os.fdopen(fd, "rb") as fh:
            res = _run(target.command_args(f"mkdir -p {d} && cat > {tmp}"), timeout=timeout, stdin=fh)
    except OSError as exc:
        raise GpuError(f"upload of {local}: {exc}") from None
    if res.returncode != 0:
        raise GpuError(f"upload of {Path(local).name} failed: {res.stderr.decode('utf-8', 'replace')[-300:].strip()}")
    verify = (
        f"h=$(python3 -I -c {shlex.quote(_REMOTE_SHA)} {tmp}) && [ \"$h\" = {shlex.quote(digest)} ] "
        f"&& chmod {mode:o} {tmp} && mv -f {tmp} {shlex.quote(remote)} || {{ rm -f {tmp}; echo \"MISMATCH $h\" >&2; exit 3; }}"
    )  # fmt: skip
    res = run(target, verify, timeout=timeout, check=False)
    if res.returncode != 0:
        raise GpuError(f"upload of {Path(local).name}: sha256 verification on the rental failed ({res.stderr.decode('utf-8', 'replace')[-200:].strip()})")
    return digest


def _walk_tree(local_dir: Path) -> list[tuple[str, Path]]:
    """Regular files below ``local_dir`` as (relative posix path, path). Links and special files are refused."""
    root = Path(os.path.abspath(local_dir))
    if os.path.islink(root) or not root.is_dir():
        raise GpuError(f"{root} is not a real directory")
    out: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for d in dirnames:
            if os.path.islink(os.path.join(dirpath, d)):
                raise GpuError(f"{os.path.join(dirpath, d)}: symbolic links are refused")
        for name in sorted(filenames):
            if name.endswith((".pyc", ".pyo")):
                continue
            p = Path(dirpath) / name
            st = os.lstat(p)
            if not stat.S_ISREG(st.st_mode):
                raise GpuError(f"{p}: not a regular file (links and special files are refused)")
            if st.st_nlink > 1:
                raise GpuError(f"{p}: hard-linked files are refused")
            out.append((p.relative_to(root).as_posix(), p))
    return out


def upload_tree(target: SshTarget, local_dir: str | os.PathLike[str], remote_dir: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, str]:
    """Copy a local directory (regular files only) to ``remote_dir`` and verify every file's sha256 on the rental. Returns {relpath: sha256}.
    Modes are kept, owners are not."""
    check_remote_path(remote_dir, "upload destination")
    files = _walk_tree(Path(local_dir))
    manifest: dict[str, str] = {}
    buf = tempfile.TemporaryFile()
    try:
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for rel, p in files:
                info = tar.gettarinfo(str(p), arcname=rel)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o755 if info.mode & 0o111 else 0o644
                with open(p, "rb") as fh:
                    tar.addfile(info, fh)
                manifest[rel] = sha256_of_regular(p)
        buf.seek(0)
        d = shlex.quote(remote_dir)
        res = _run(target.command_args(f"mkdir -p {d} && tar -x -C {d} --no-same-owner -f -"), timeout=timeout, stdin=buf)
    finally:
        buf.close()
    if res.returncode != 0:
        raise GpuError(f"upload_tree failed: {res.stderr.decode('utf-8', 'replace')[-300:].strip()}")
    listing = run(target, f"python3 -I -c {shlex.quote(_REMOTE_MANIFEST)} {shlex.quote(remote_dir)}", timeout=timeout).text
    remote: dict[str, str] = {}
    for line in listing.splitlines():
        digest, _, rel = line.partition("  ")
        remote[rel] = digest
    for rel, digest in manifest.items():
        if remote.get(rel) != digest:
            raise GpuError(f"upload_tree: {rel} differs on the rental after the copy")
    return manifest


class _Capped:
    """Reads at most ``cap`` bytes from a pipe, then raises: an endless stream cannot fill the control host."""

    def __init__(self, raw, cap: int) -> None:
        self._raw, self._left = raw, cap

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0 or n > self._left + 1:
            n = self._left + 1
        data = self._raw.read(n)
        self._left -= len(data)
        if self._left < 0:
            raise FetchRefused("fetch: the tar stream is larger than the size caps allow")
        return data


def parse_tar_stream(
    stream,
    names: Sequence[str],
    local_dir: Path,
    *,
    max_file_bytes: int,
    max_total_bytes: int,
) -> dict[str, Fetched]:  # fmt: skip
    """Read a tar stream and write exactly the requested regular files into ``local_dir`` (created by this function, never by tar).

    Raises GpuError for any member that is not a regular file with exactly one of ``names``, for a duplicate or missing member, a
    member above ``max_file_bytes``, or a total above ``max_total_bytes``. Nothing is left behind on failure."""
    wanted = {_file_name(n) for n in names}
    cap = max_total_bytes + 65536 + 4096 * len(wanted)  # tar pads to 10240-byte records; extended headers take a block or two
    results: dict[str, Fetched] = {}
    temps: list[Path] = []
    total = 0
    try:
        with tarfile.open(fileobj=_Capped(stream, cap), mode="r|") as tar:
            for member in tar:
                kind = (
                    "symbolic link" if member.issym() else "hard link" if member.islnk() else "directory" if member.isdir()
                    else "device" if member.isdev() else "fifo" if member.isfifo() else None
                )  # fmt: skip
                if kind is not None:
                    raise FetchRefused(f"fetch: member {member.name!r} is a {kind}; only regular files are accepted")
                if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    raise FetchRefused(f"fetch: member {member.name!r} has tar type {member.type!r}; only regular files are accepted")
                if member.name not in wanted:
                    raise FetchRefused(f"fetch: unexpected member {member.name!r} (asked for {sorted(wanted)})")
                if member.name in results:
                    raise FetchRefused(f"fetch: duplicate member {member.name!r}")
                if member.size > max_file_bytes:
                    raise FetchRefused(f"fetch: {member.name} is {member.size} bytes, above the {max_file_bytes}-byte limit")
                total += member.size
                if total > max_total_bytes:
                    raise FetchRefused(f"fetch: total size above the {max_total_bytes}-byte limit")
                src = tar.extractfile(member)
                if src is None:
                    raise FetchRefused(f"fetch: cannot read member {member.name!r}")
                tmp = local_dir / f".{member.name}.fetch"
                temps.append(tmp)
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                h, written = hashlib.sha256(), 0
                with os.fdopen(fd, "wb") as out:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        written += len(chunk)
                        if written > max_file_bytes:
                            raise FetchRefused(f"fetch: {member.name} grew past the {max_file_bytes}-byte limit")
                        h.update(chunk)
                        out.write(chunk)
                if written != member.size:
                    raise FetchRefused(f"fetch: {member.name} is truncated ({written} of {member.size} bytes)")
                results[member.name] = Fetched(member.name, local_dir / member.name, h.hexdigest(), written)
        missing = sorted(wanted - set(results))
        if missing:
            raise FetchRefused(f"fetch: the rental did not return {missing}")
        for name, item in results.items():
            os.replace(local_dir / f".{name}.fetch", item.path)
        temps.clear()
        return results
    except tarfile.TarError as exc:
        raise FetchRefused(f"fetch: malformed tar stream ({exc})") from None
    finally:
        for t in temps:
            try:
                os.unlink(t)
            except OSError:
                pass


def fetch_files(
    target: SshTarget,
    remote_dir: str,
    names: Sequence[str],
    local_dir: str | os.PathLike[str],
    *,
    max_file_bytes: int,
    max_total_bytes: Optional[int] = None,
    expected_sha256: Optional[Mapping[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Fetched]:  # fmt: skip
    """Fetch the named regular files of ``remote_dir`` into the existing directory ``local_dir``. Only exact names, only regular files
    (see the module docstring). ``expected_sha256`` maps a name to the hash it must have, else the files are deleted and GpuError raised."""
    check_remote_path(remote_dir, "fetch source")
    wanted = [_file_name(n) for n in names]
    if not wanted:
        raise GpuError("fetch_files: no names")
    dest = Path(local_dir)
    if os.path.islink(dest) or not dest.is_dir():
        raise GpuError(f"fetch destination {dest} is not a real directory")
    total_cap = max_total_bytes if max_total_bytes is not None else max_file_bytes * len(wanted)
    # COPYFILE_DISABLE: bsdtar (macOS) would add AppleDouble "._name" members; GNU tar on the rental ignores it.
    cmd = f"COPYFILE_DISABLE=1 tar -C {shlex.quote(remote_dir)} -cf - -- " + " ".join(shlex.quote(n) for n in wanted)
    try:
        proc = subprocess.Popen(target.command_args(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise GpuError(f"cannot run ssh: {exc}") from None
    # stderr is drained as it arrives and only its last 2000 bytes are kept: a rental that writes it without end must not fill the control host's disk
    err_tail = bytearray()

    def drain_stderr() -> None:
        try:
            for chunk in iter(lambda: proc.stderr.read(1 << 14), b""):
                err_tail.extend(chunk)
                del err_tail[:-2000]
        except (OSError, ValueError):
            pass

    err_thread = threading.Thread(target=drain_stderr, daemon=True)
    err_thread.start()
    timer = threading.Timer(timeout, proc.kill)
    timer.start()

    def discard(items) -> None:
        for item in items:
            try:
                os.unlink(item.path)
            except OSError:
                pass

    try:
        try:
            results = parse_tar_stream(proc.stdout, wanted, dest, max_file_bytes=max_file_bytes, max_total_bytes=total_cap)
        except GpuError:
            proc.kill()
            raise
        proc.wait()
        if proc.returncode != 0:
            discard(results.values())
            err_thread.join(5)
            err = bytes(err_tail).decode("utf-8", "replace").strip()
            raise GpuError(f"fetch: remote tar exited {proc.returncode}: {err[-200:]}")
    finally:
        timer.cancel()
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        err_thread.join(5)
        try:
            proc.stderr.close()
        except Exception:
            pass
    for name, want in (expected_sha256 or {}).items():
        got = results.get(name)
        if got is None or got.sha256 != want:
            discard(results.values())
            raise FetchRefused(f"fetch: sha256 of {name} is {got.sha256 if got else 'missing'}, expected {want}")
    return results


def fetch(
    target: SshTarget,
    remote: str,
    local: str | os.PathLike[str],
    *,
    sha256: Optional[str] = None,
    max_bytes: int,
    timeout: float = DEFAULT_TIMEOUT,
) -> Fetched:  # fmt: skip
    """Fetch one remote regular file to the local path ``local`` (the file name may differ). Returns the verified Fetched."""
    check_remote_path(remote)
    rdir, rname = posixpath.split(remote)
    dest = Path(local)
    parent = dest.parent
    if os.path.islink(parent) or not parent.is_dir():
        raise GpuError(f"fetch destination directory {parent} is not a real directory")
    with tempfile.TemporaryDirectory(dir=parent, prefix=".fetch-") as td:
        got = fetch_files(
            target, rdir or "/", [rname], td, max_file_bytes=max_bytes, expected_sha256={rname: sha256} if sha256 else None, timeout=timeout
        )[rname]
        os.replace(got.path, dest)
    return Fetched(dest.name, dest, got.sha256, got.size_bytes)


def remote_exists(target: SshTarget, remote: str, *, timeout: float = 60.0) -> bool:
    check_remote_path(remote)
    q = shlex.quote(remote)
    return run(target, f"test -e {q} -o -L {q}", timeout=timeout, check=False).returncode == 0


def run_detached(
    target: SshTarget,
    argv: Sequence[str],
    log: str,
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: float = 120.0,
) -> str:  # fmt: skip
    """Start ``argv`` on the rental in its own session with output to ``log``; returns the remote pid (a string of digits).
    ssh returns at once: the command is not tied to this connection."""
    check_remote_path(log, "log path")
    if not argv:
        raise GpuError("run_detached: empty argv")
    prefix = ""
    if env:
        for k, v in env.items():
            if not re.match(r"^[A-Z][A-Z0-9_]*$", k):
                raise GpuError(f"run_detached: bad environment variable name {k!r}")
        prefix = "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
    cmd = " ".join(shlex.quote(a) for a in argv)
    cd = f"cd {shlex.quote(check_remote_path(cwd, 'cwd'))} && " if cwd else ""
    log_dir = shlex.quote(posixpath.dirname(log))
    script = (
        f"mkdir -p {log_dir} && {cd}S=$(command -v setsid || true); "
        f"nohup $S {prefix}{cmd} > {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
    )
    out = run(target, script, timeout=timeout).text.strip().splitlines()
    pid = out[-1].strip() if out else ""
    if not pid.isdigit():
        raise GpuError(f"run_detached: no pid came back ({pid[:80]!r})")
    return pid


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PortForward:
    """``ssh -N -L``: a control-host loopback port that reaches a loopback port on the rental. The control host never talks to a public port."""

    def __init__(self, proc: subprocess.Popen, local_port: int, remote_port: int) -> None:
        self._proc, self.local_port, self.remote_port = proc, local_port, remote_port

    def alive(self) -> bool:
        return self._proc.poll() is None

    def wait_ready(self, timeout: float = 30.0, *, sleep=time.sleep) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if not self.alive():
                raise GpuError(f"port forward exited ({self._proc.returncode}) before it listened")
            try:
                with socket.create_connection(("127.0.0.1", self.local_port), timeout=1.0):
                    return
            except OSError:
                if time.monotonic() > deadline:
                    raise GpuError("port forward did not start listening") from None
                sleep(0.1)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    def __enter__(self) -> "PortForward":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def forward_port(target: SshTarget, local_port: int, remote_port: int) -> PortForward:
    """Open ``127.0.0.1:local_port -> rental 127.0.0.1:remote_port``. Fails (ExitOnForwardFailure) instead of running without the forward."""
    for p in (local_port, remote_port):
        if not isinstance(p, int) or isinstance(p, bool) or not 0 < p < 65536:
            raise GpuError(f"bad port {p!r}")
    args = target.args("-N", "-o", "ExitOnForwardFailure=yes", "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}")
    try:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        raise GpuError(f"cannot run ssh: {exc}") from None
    return PortForward(proc, local_port, remote_port)


def write_ssh_config(path: str | os.PathLike[str], alias: str, target: SshTarget) -> None:
    """A session-local ssh_config for a human to reach the rental (mode 0600). The lab itself passes options on the command line."""
    lines = [
        f"Host {alias}", f"  HostName {target.host}", f"  Port {target.port}", f"  User {target.user}",
        f"  IdentityFile {target.key_path}", "  IdentitiesOnly yes", "  StrictHostKeyChecking accept-new",
        "  ServerAliveInterval 15", "  ServerAliveCountMax 4",
    ]  # fmt: skip
    if target.known_hosts:
        lines.append(f"  UserKnownHostsFile {target.known_hosts}")
    p = Path(path)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
