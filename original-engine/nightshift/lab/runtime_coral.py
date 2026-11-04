"""``ResearcherRuntime`` over CORAL: launch a pack's researcher team on the researcher host, watch it, stop it, harvest it.

What this module does (docs/harness-design.md "CORAL integration plan", D9):

* ``CoralRuntime.start(context)`` stages a clean trial directory on the researcher host (the pack checkout, the dev grader package, the
  assignment's public ``train.jsonl`` / ``dev.jsonl``, the ``glimmer`` tool, the opencode PWD shim, the runtime-owned provider and gateway
  config, the trial token), opens an SSH reverse tunnel to the control host's metering proxy and control API, and runs ``coral start``
  detached with the runtime-owned overrides on the command line. The pack holds only text levers; provider, upstream route, grader, ports,
  stop condition and environment cannot be changed by a pack edit.
* ``CoralHandle`` implements ``RuntimeHandle``: ``poll`` (running / finished / crashed / the reason ``stop`` was called with), ``stop``
  (process group, then a sweep of stragglers whose working directory is inside the trial directory), ``collect`` (harvest) and ``close``
  (kill leftovers, close the tunnel, delete the remote trial directory).
* Harvest reads only what CORAL writes under ``.coral/public`` through a tar stream that is extracted by this module's own safe extractor:
  symbolic links, hard links, devices, sparse files and escaping paths are skipped and counted, never followed. ``submission.json`` is read
  from the attempt's own commit with ``git cat-file blob`` (never from the live worktree), and the researcher's ``dev_score`` is discarded.

What it does NOT do: it has no knowledge of scores, budgets or the ledger (the trial runner owns them), it does not create throwaway unix
users (the isolation level is a label, ``shared_user_serial`` unless the caller says otherwise), and nothing here was run against a real
CORAL: the command line, the ``agents.runtime_options.mounts`` override and the harvest parsers are built from the probe evidence
(CORAL ``0123dfb9``, opencode 1.19.3, stub upstream) and are UNVERIFIED against Muse.

Standard library only. The remote host is reached through a small ``RemoteHost`` protocol (``SshRemote`` is the real implementation; the
tests use a local one), so every command line and parser is testable without SSH.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import tarfile
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NamedTuple, Optional, Protocol, Sequence

from nightshift.lab.contracts import (
    Clock,
    ExitReason,
    LabLayout,
    PortAllocator,
    RuntimeFailure,
    RuntimeResult,
    SubmittedAttempt,
    Submission,
    TrialContext,
    loads_strict,
    sha256_text,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
GRADER_DIR = ASSETS_DIR / "coral_grader"
SHIM_PATH = ASSETS_DIR / "shims" / "opencode"
TOOL_PATH = Path(__file__).resolve().parent / "researcher_tool.py"

PROVIDER_ID = "muse"
MODEL_ID = "muse-spark-1.3"
AGENT_MODEL = f"{PROVIDER_ID}/{MODEL_ID}"
OUTPUT_LIMIT = 16000  # opencode limit.output -> max_tokens (VERIFIED against the stub; 16000 is the safe ceiling until the real one is known)
CONTEXT_LIMIT = 1_048_576
DEFAULT_REMOTE_TRIALS_ROOT = "/srv/lab/trials"
ISOLATION_SHARED = "shared_user_serial"

MAX_HARVEST_BYTES = 128 * 1024 * 1024  # the whole tar stream
MAX_HARVEST_FILE_BYTES = 64 * 1024 * 1024
MAX_HARVEST_FILES = 20000
MAX_ATTEMPT_JSON_BYTES = 1 << 20
MAX_SUBMISSION_BYTES = 1 << 16
MAX_ENV_VALUE = 480
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_LOG_NAME = re.compile(r"^(?P<agent>.+)\.(?P<n>\d+)\.log$")

# The public parts of a CORAL run that are harvested. gateway/requests.jsonl is deliberately left out: it holds every full prompt and is
# unbounded; per-step token counts are in the opencode event logs.
HARVEST_PATHS = ("attempts", "logs", "notes", "skills", "roles", "diagnostics", "sessions.json", "auto_stop.json", "agent_state.json", "eval_count")


# --------------------------------------------------------------------------------------------------
# the remote host
# --------------------------------------------------------------------------------------------------
class RemoteResult(NamedTuple):
    returncode: int
    stdout: bytes
    stderr: bytes


class RemoteError(RuntimeFailure):
    """A remote command could not be run, timed out, or produced more output than allowed."""


class RemoteHost(Protocol):
    def run(self, argv: Sequence[str], *, input: Optional[bytes] = None, timeout: float = 60.0, max_output: int = 16 << 20) -> RemoteResult: ...

    def reverse_tunnel(self, remote_port: int, local_port: int) -> Any:
        """Forward ``127.0.0.1:remote_port`` on the remote host to ``127.0.0.1:local_port`` here. The result has ``alive()`` and ``close()``.
        Raises RemoteError if the forward cannot be set up."""


def run_capped(cmd: Sequence[str], *, input: Optional[bytes] = None, timeout: float = 60.0, max_output: int = 16 << 20, env: Optional[Mapping[str, str]] = None, cwd: Optional[str] = None) -> RemoteResult:
    """Run a command with a wall-clock timeout and a cap on captured stdout (a hostile remote tree cannot exhaust this host's memory)."""
    try:
        p = subprocess.Popen(
            list(cmd), stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=None if env is None else dict(env), cwd=cwd, start_new_session=True,
        )
    except OSError as exc:
        raise RemoteError(f"cannot run {cmd[0]}: {exc}") from None
    out, err = bytearray(), bytearray()
    over = threading.Event()

    def kill() -> None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def reader(stream: Any, buf: bytearray, cap: int) -> None:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            if len(buf) + len(chunk) > cap:
                over.set()
                kill()
                return
            buf += chunk

    def writer() -> None:
        try:
            p.stdin.write(input)  # type: ignore[union-attr]
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                p.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass

    threads = [threading.Thread(target=reader, args=(p.stdout, out, max_output), daemon=True), threading.Thread(target=reader, args=(p.stderr, err, 1 << 20), daemon=True)]
    if input is not None:
        threads.append(threading.Thread(target=writer, daemon=True))
    deadline = time.monotonic() + max(0.0, float(timeout))
    for t in threads:
        t.start()
    timed_out = False
    try:
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        if not timed_out:
            # The command exited, but a descendant that inherited stdout/stderr keeps the pipes open: the readers (and a stream.close() that
            # waits behind one) would block until it exits. The timeout covers the whole run, readers included.
            for t in threads:
                t.join(timeout=max(0.0, deadline - time.monotonic()))
            timed_out = any(t.is_alive() for t in threads)
        if timed_out:
            kill()  # the whole process group, descendants included: that closes the pipes the readers wait on
            p.wait()
            for t in threads:
                t.join(timeout=5)
    finally:
        for stream in (p.stdout, p.stderr):
            try:
                stream.close()
            except OSError:
                pass
    if timed_out:
        raise RemoteError(f"command timed out after {timeout:g}s")
    if over.is_set():
        raise RemoteError(f"command output exceeded {max_output} bytes")
    return RemoteResult(p.returncode, bytes(out), bytes(err))


class _SshTunnel:
    def __init__(self, proc: subprocess.Popen):
        self._p = proc

    def alive(self) -> bool:
        return self._p.poll() is None

    def close(self) -> None:
        if self._p.poll() is None:
            self._p.terminate()
            try:
                self._p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._p.kill()


class SshRemote:
    """``RemoteHost`` over the ``ssh`` binary (key-based, ``BatchMode``). ``target`` is ``user@host`` or an ssh config alias."""

    def __init__(self, target: str, *, port: Optional[int] = None, identity: Optional[str] = None, ssh_bin: str = "ssh", options: Sequence[str] = ()):
        if not target or target.startswith("-"):
            raise ValueError("invalid ssh target")
        self.target = target
        self._base = [ssh_bin, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
        if port is not None:
            self._base += ["-p", str(int(port))]
        if identity:
            self._base += ["-i", identity, "-o", "IdentitiesOnly=yes"]
        for o in options:
            self._base += ["-o", o]

    def run(self, argv: Sequence[str], *, input: Optional[bytes] = None, timeout: float = 60.0, max_output: int = 16 << 20) -> RemoteResult:
        return run_capped(self._base + ["--", self.target, shlex.join(list(argv))], input=input, timeout=timeout, max_output=max_output)

    def reverse_tunnel(self, remote_port: int, local_port: int) -> _SshTunnel:
        cmd = self._base + ["-N", "-o", "ExitOnForwardFailure=yes", "-R", f"127.0.0.1:{int(remote_port)}:127.0.0.1:{int(local_port)}", "--", self.target]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        except OSError as exc:
            raise RemoteError(f"cannot start ssh: {exc}") from None
        try:
            proc.wait(timeout=2.0)  # ExitOnForwardFailure makes a failed forward exit quickly
        except subprocess.TimeoutExpired:
            return _SshTunnel(proc)
        raise RemoteError("ssh reverse tunnel failed: " + (proc.stderr.read(500).decode("utf-8", "replace") if proc.stderr else ""))


# --------------------------------------------------------------------------------------------------
# rendered configuration (pure; tested)
# --------------------------------------------------------------------------------------------------
def render_opencode_json(gateway_port: int) -> dict:
    """The runtime-owned opencode provider config. Provider id ``muse`` (``meta`` collides with a built-in opencode provider)."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Meta Muse (via CORAL gateway)",
                "options": {"baseURL": f"http://127.0.0.1:{int(gateway_port)}/v1", "apiKey": "coral-gateway-placeholder"},
                "models": {MODEL_ID: {"name": "Muse Spark 1.3", "tool_call": True, "reasoning": False, "limit": {"context": CONTEXT_LIMIT, "output": OUTPUT_LIMIT}}},
            }
        },
    }


def render_litellm_config(proxy_port: int, token: str, reasoning_effort: str) -> str:
    """The gateway's upstream route: the trial's metering proxy (through the tunnel), the trial token as API key, ``reasoning_effort`` in
    ``extra_body`` (a top-level key is dropped by ``drop_params``). Values are JSON strings, which are valid YAML scalars."""
    q = json.dumps
    return (
        "model_list:\n"
        f"  - model_name: {q(MODEL_ID)}\n"
        "    litellm_params:\n"
        f"      model: {q('openai/' + MODEL_ID)}\n"
        f"      api_base: {q(f'http://127.0.0.1:{int(proxy_port)}/v1')}\n"
        f"      api_key: {q(token)}\n"
        "      extra_body:\n"
        f"        reasoning_effort: {q(reasoning_effort)}\n"
        "\n"
        "litellm_settings:\n"
        "  drop_params: true\n"
    )


def load_entrypoint_spec() -> dict:
    return json.loads((GRADER_DIR / "ENTRYPOINT.json").read_text(encoding="utf-8"))


def coral_overrides(root: str, gateway_port: int, *, max_real_attempts: Optional[int] = None, entry: Optional[Mapping[str, Any]] = None) -> list[str]:
    """The dotlist overrides appended to ``coral start``; none of these can be changed by a pack edit (docs: CORAL integration plan)."""
    entry = dict(entry or load_entrypoint_spec())
    mounts = json.dumps({f"{root}/runtime/opencode.json": "opencode.json"})
    out = [
        "run.session=local",
        "agents.runtime=opencode",
        f"agents.model={AGENT_MODEL}",
        f"workspace.results_dir={root}/results",
        f"workspace.repo_path={root}/no-such-dir",
        f"workspace.run_dir={root}/run",
        "agents.gateway.enabled=true",
        f"agents.gateway.port={int(gateway_port)}",
        f"agents.gateway.config={root}/runtime/litellm_config.yaml",
        f"grader.entrypoint={entry['entrypoint']}",
        "grader.setup=" + json.dumps(list(entry["setup"])),
        f"grader.timeout={int(entry['timeout'])}",
        "agents.runtime_options.mounts=" + mounts,
    ]
    if max_real_attempts is not None:
        out.append(f"run.stop.max_real_attempts={int(max_real_attempts)}")
    return out


def render_launch_script(root: str, coral_home: str, overrides: Sequence[str], *, control_url: str) -> str:
    """The script that starts CORAL detached. ``coral.pid`` is the session leader (its process group is the whole run); ``coral.exit`` holds
    the exit status once it ends. The environment of CORAL and everything under it is built here from scratch with ``env -i``: only the
    trial's own values, the shim first on PATH, XDG directories inside the trial directory (the user's global opencode config must not
    leak in), ``HOME`` inside the trial directory (a persistent home would carry notes between trials; it is deleted with the directory) and ``GLIMMER_TRIAL_ROOT``, which the stop script uses to find processes that left the trial directory. Nothing else of the
    remote shell's environment (credentials, canaries, agent sockets) reaches the researchers. The launching user's ``~/.local/bin`` stays on PATH
    for the tools CORAL needs; under shared_user_serial it is writable by the trial (only a separate unix user closes that, D2); which extra variables CORAL needs on a real
    host is UNVERIFIED."""
    q = shlex.quote
    cmd = " ".join(q(a) for a in ["coral", "start", "-c", "pack/task.yaml", *overrides])
    detached = cmd + '; echo $? > "$ROOT/coral.exit"'
    clean_env = " ".join(
        [
            'HOME="$ROOT/home"',
            'USER="$WHO"',
            'LOGNAME="$WHO"',
            'LANG="${LANG:-C.UTF-8}"',
            "TERM=dumb",
            'ROOT="$ROOT"',
            'GLIMMER_TRIAL_ROOT="$ROOT"',
            f'PATH="$ROOT/bin:{coral_home}/.venv/bin:$HOME/.local/bin:$PATH"',
            'XDG_CONFIG_HOME="$ROOT/xdg/config" XDG_DATA_HOME="$ROOT/xdg/data" XDG_CACHE_HOME="$ROOT/xdg/cache" XDG_STATE_HOME="$ROOT/xdg/state"',
            f"GLIMMER_CONTROL={q(control_url)}",
            'GLIMMER_TOKEN_FILE="$ROOT/.glimmer_token" GLIMMER_DATA_DIR="$ROOT/data"',
        ]
    )
    lines = [
        "#!/bin/bash",
        "set -u",
        f"export ROOT={q(root)}",
        'cd "$ROOT" || exit 1',
        'mkdir -p "$ROOT/home" || exit 1',
        'rm -f "$ROOT/coral.exit" "$ROOT/coral.pid"',
        'WHO="$(id -un)"',
        'PY="$(command -v python3)" || exit 1',
        # python3 does the setsid (the macOS test host has no setsid binary) and execs bash, so $! is the session leader and process-group id
        "nohup env -i " + clean_env + ' "$PY" -c \'import os,sys;os.setsid();os.execvp(sys.argv[1],sys.argv[1:])\' bash -c ' + q(detached) + ' > "$ROOT/coral.out" 2>&1 < /dev/null &',
        'echo $! > "$ROOT/coral.pid"',
        'echo started "$(cat "$ROOT/coral.pid")"',
    ]
    return "\n".join(lines) + "\n"


POLL_SCRIPT = r"""
ROOT="$1"
if [ -f "$ROOT/coral.exit" ]; then echo "exit:$(cat "$ROOT/coral.exit")"
elif kill -0 "$(cat "$ROOT/coral.pid" 2>/dev/null)" 2>/dev/null; then echo running
else
  sleep 0.3  # the shell writes coral.exit just after coral exits: do not report a fast exit as a vanished process
  if [ -f "$ROOT/coral.exit" ]; then echo "exit:$(cat "$ROOT/coral.exit")"; else echo dead; fi
fi
"""

# Stop: TERM the run's process group, wait, KILL it, then sweep stragglers (agents start their own sessions). A straggler is, in this
# order of evidence: a descendant of the run's session leader (taken BEFORE anything is killed, because orphans are re-parented to init),
# a process whose environment still carries the GLIMMER_TRIAL_ROOT marker the launch script set (it survives setsid, a double fork and a
# changed working directory; /proc/<pid>/environ on Linux, `ps eww` elsewhere), or a process whose working directory is inside the trial
# directory. pids written by the run itself (agent.pids) are only trusted after the working-directory check, because an agent can edit
# that file. What this cannot find: a process that was re-parented AND scrubbed its environment AND left the directory. Only a
# separate unix user per trial closes that (docs/harness-design.md D2); under shared_user_serial the report labels it.
STOP_SCRIPT = r"""
ORIG="$1"
ROOT="$(cd "$1" 2>/dev/null && pwd -P)" || ROOT="$1"
cd / || exit 1
cwd_of() { readlink "/proc/$1/cwd" 2>/dev/null || lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p'; }
inside() { case "$1" in "$ROOT"|"$ROOT"/*) return 0;; esac; return 1; }
descendants() {
  ps -A -o pid= -o ppid= 2>/dev/null | awk -v r="$1" '{ par[$1] = $2 } END { for (p in par) { x = p; d = 0; while ((x in par) && d < 64) { if (x == r) { print p; break } x = par[x]; d++ } } }'
}
marked() {
  for r in "$1" "$2"; do
    if [ -d /proc/self ]; then
      for d in /proc/[0-9]*; do
        [ -r "$d/environ" ] && tr '\0' '\n' < "$d/environ" 2>/dev/null | grep -qxF "GLIMMER_TRIAL_ROOT=$r" && echo "${d#/proc/}"
      done
    else
      out="$(ps eww -A -o pid= -o command= 2>/dev/null)"
      printf '%s\n' "$out" | sed 's/$/ /' | grep -F "GLIMMER_TRIAL_ROOT=$r " | awk '{print $1}'
    fi
  done
}
pid="$(cat "$ROOT/coral.pid" 2>/dev/null)"
case "$pid" in ''|*[!0-9]*) pid="";; esac
# coral.pid is writable by the researchers and a pid can be reused: it is only signalled (and its group, and its descendants) when the
# process is verifiably the trial's own, by the environment marker or a working directory inside the trial directory.
owned() {
  for m in $(marked "$ROOT" "$ORIG"); do [ "$m" = "$1" ] && return 0; done
  c="$(cwd_of "$1")" && inside "$c" && return 0
  return 1
}
if [ -n "$pid" ] && ! owned "$pid"; then pid=""; fi
known=""
[ -n "$pid" ] && known="$(descendants "$pid" | tr '\n' ' ')"
sweep() {
  sig="$1"
  pids="$known $(marked "$ROOT" "$ORIG" | tr '\n' ' ')"
  [ -n "$pid" ] && pids="$pids $(descendants "$pid" | tr '\n' ' ')"
  for pid_ in $pids; do
    [ "$pid_" = "$$" ] && continue
    kill "-$sig" "$pid_" 2>/dev/null
  done
  pids=""
  if [ -d /proc/self ]; then
    for d in /proc/[0-9]*; do pids="$pids ${d#/proc/}"; done
  fi
  for f in "$ROOT/run/.coral/public/agent.pids" "$ROOT/run/.coral/public/manager.pid"; do
    [ -f "$f" ] && pids="$pids $(grep -oE '^[0-9]+$' "$f" 2>/dev/null | head -200 | tr '\n' ' ')"
  done
  for pid_ in $pids; do
    [ "$pid_" = "$$" ] && continue
    c="$(cwd_of "$pid_")" || continue
    inside "$c" && kill "-$sig" "$pid_" 2>/dev/null
  done
}
if [ -n "$pid" ]; then kill -TERM -- "-$pid" 2>/dev/null; fi
sweep TERM
i=0
while [ $i -lt 10 ]; do
  alive=0
  [ -n "$pid" ] && kill -0 -- "-$pid" 2>/dev/null && alive=1
  [ "$alive" = 0 ] && break
  sleep 0.5; i=$((i+1))
done
if [ -n "$pid" ]; then kill -KILL -- "-$pid" 2>/dev/null; fi
sweep KILL
echo stopped
"""

REMOVE_SCRIPT = r"""
ROOT="$1"; BASE="$2"
case "$ROOT" in "$BASE"/?*) ;; *) echo refused; exit 3;; esac
case "$ROOT" in *..*|*' '*) echo refused; exit 3;; esac
rm -rf -- "$ROOT"; echo removed
"""

# shared_user_serial across coordinator restarts: the in-memory holder dies with the coordinator while a detached trial survives on the
# researcher host, so the claim also lives there, as a directory created atomically next to the trial directories. Only a clean close of
# the holding trial removes it; after a crash the operator cleans the host (the refusal names the holder and the path).
HOLDER_DIR = ".holder"
HOLD_SCRIPT = r"""
BASE="$1"; TID="$2"
mkdir -p -- "$BASE" || exit 5
if mkdir -- "$BASE/.holder" 2>/dev/null; then
  printf '%s' "$TID" > "$BASE/.holder/trial" || { rmdir -- "$BASE/.holder" 2>/dev/null; rm -f -- "$BASE/.holder/trial"; exit 5; }
  echo acquired
else
  echo "held $(cat "$BASE/.holder/trial" 2>/dev/null)"
  exit 9
fi
"""
UNHOLD_SCRIPT = r"""
BASE="$1"; TID="$2"
[ "$(cat "$BASE/.holder/trial" 2>/dev/null)" = "$TID" ] && rm -rf -- "$BASE/.holder"
echo released
"""

HARVEST_SCRIPT = r"""
export COPYFILE_DISABLE=1
D="$1"; shift
[ -d "$D" ] || { echo "no public dir" >&2; exit 4; }
cd "$D" || exit 4
set --
for p in %PATHS%; do [ -e "$p" ] && [ ! -L "$p" ] && set -- "$@" "$p"; done
[ "$#" -gt 0 ] || exit 0
# Only regular files with a single link are archived: symbolic links, hard links (tar would store the first one as a regular file with
# the target's content), fifos and devices never enter the stream. The control host's extractor refuses them anyway.
find "$@" -type f -links 1 -print0 | tar --null --no-recursion -czf - -T -
"""

LISTING_SCRIPT = r"""
cd "$1" || exit 4
find . -path ./.git -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum 2>/dev/null || find . -path ./.git -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256
"""


# --------------------------------------------------------------------------------------------------
# staging and harvest helpers (pure; tested)
# --------------------------------------------------------------------------------------------------
def listing_sha256(entries: Mapping[str, bytes]) -> str:
    """sha256 over the sorted listing ``<sha256> <size> <path>`` of ``entries`` (the initial-context hash)."""
    lines = "".join(f"{hashlib.sha256(data).hexdigest()} {len(data)} {path}\n" for path, data in sorted(entries.items()))
    return sha256_text(lines)


def read_tree(directory: Path, prefix: str, *, skip: frozenset = frozenset({".git", "__pycache__"})) -> dict[str, bytes]:
    """Every regular file under ``directory`` as ``{prefix/relpath: bytes}``. Symbolic links, hard-linked files and special files are an error."""
    out: dict[str, bytes] = {}
    root = Path(directory)
    if not root.is_dir():
        raise RuntimeFailure(f"{directory} is not a directory")

    def walk(d: Path, rel: str) -> None:
        for entry in sorted(os.scandir(d), key=lambda e: e.name):
            if entry.name in skip or entry.name.endswith(".pyc"):
                continue
            r = f"{rel}/{entry.name}" if rel else entry.name
            st = os.lstat(entry.path)
            if stat.S_ISLNK(st.st_mode):
                raise RuntimeFailure(f"{entry.path}: symbolic links are refused in a staged tree")
            if stat.S_ISDIR(st.st_mode):
                walk(Path(entry.path), r)
            elif stat.S_ISREG(st.st_mode):
                out[f"{prefix}/{r}" if prefix else r] = Path(entry.path).read_bytes()
            else:
                raise RuntimeFailure(f"{entry.path}: only regular files can be staged")

    walk(root, "")
    return out


def make_stage_tar(files: Mapping[str, tuple[bytes, int]]) -> bytes:
    """tar.gz of ``{path: (bytes, mode)}`` for ``tar -x`` on the remote host."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            dirs: set[str] = set()
            for path in sorted(files):
                pp = PurePosixPath(path)
                for i in range(1, len(pp.parts)):
                    d = "/".join(pp.parts[:i])
                    if d not in dirs:
                        dirs.add(d)
                        ti = tarfile.TarInfo(d)
                        ti.type, ti.mode, ti.mtime = tarfile.DIRTYPE, 0o755, 0
                        tar.addfile(ti)
                data, mode = files[path]
                ti = tarfile.TarInfo(path)
                ti.size, ti.mode, ti.mtime = len(data), mode, 0
                tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class HarvestStats:
    def __init__(self) -> None:
        self.files = 0
        self.bytes = 0
        self.n_skipped = 0
        self.skipped: list[str] = []

    def skip(self, name: str, why: str) -> None:
        self.n_skipped += 1
        if len(self.skipped) < 20:
            self.skipped.append(f"{why}:{name[:80]}")


class _CappedStream(io.RawIOBase):
    """Decompressed stream that refuses to yield more than ``limit`` bytes (a gzip bomb stops here)."""

    def __init__(self, raw: Any, limit: int):
        self._raw, self._left = raw, limit

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        data = self._raw.read(len(b))
        self._left -= len(data)
        if self._left < 0:
            raise OverflowError("harvest stream expands past its limit")
        b[: len(data)] = data
        return len(data)


def safe_extract(tar_gz: bytes, dest: Path, *, max_files: int = MAX_HARVEST_FILES, max_file_bytes: int = MAX_HARVEST_FILE_BYTES, max_total: int = MAX_HARVEST_BYTES) -> HarvestStats:
    """Extract a tar.gz written by an untrusted host into the (new, empty) directory ``dest``. Only regular files (and the directories
    needed for them) are created; links, devices, sparse files, absolute or escaping names, over-size files and everything past the
    totals are skipped and counted. Files are created ``O_EXCL | O_NOFOLLOW`` with mode 0600."""
    stats = HarvestStats()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        stream = io.BufferedReader(_CappedStream(gzip.GzipFile(fileobj=io.BytesIO(tar_gz), mode="rb"), 2 * max_total + max_files * 1536))
        tar = tarfile.open(fileobj=stream, mode="r|")
    except (tarfile.TarError, OSError, EOFError):
        raise RuntimeFailure("harvest stream is not a gzip tar archive") from None
    with tar:
        try:
            for member in tar:
                name = member.name
                path = PurePosixPath(name)
                if "\x00" in name or path.is_absolute() or not path.parts or any(p == ".." for p in path.parts):
                    stats.skip(name, "path")
                    continue
                if member.isdir():
                    continue
                if member.issym() or member.islnk():
                    stats.skip(name, "link")
                    continue
                if not member.isreg() or member.issparse():
                    stats.skip(name, "special")
                    continue
                if member.size > max_file_bytes:
                    stats.skip(name, "size")
                    continue
                if stats.files >= max_files or stats.bytes + member.size > max_total:
                    stats.skip(name, "total")
                    continue
                parts = [p for p in path.parts if p != "."]
                if not parts:
                    continue
                target = dest.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                fobj = tar.extractfile(member)
                if fobj is None:
                    stats.skip(name, "unreadable")
                    continue
                try:
                    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                except OSError:
                    stats.skip(name, "exists")
                    continue
                with os.fdopen(fd, "wb") as out:
                    remaining = member.size
                    while remaining > 0:
                        chunk = fobj.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        out.write(chunk)
                        remaining -= len(chunk)
                stats.files += 1
                stats.bytes += member.size
        except OverflowError:
            stats.skip("<stream>", "expands too far")
        except (tarfile.TarError, OSError, EOFError):
            stats.skip("<stream>", "truncated")
    return stats


def normalize_timestamp(value: Any) -> Optional[str]:
    """An ISO 8601 timestamp with a zone (naive is read as UTC); None when it cannot be parsed."""
    if not isinstance(value, str):
        return None
    try:
        t = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.isoformat()


def parse_attempt(raw: bytes) -> Optional[dict]:
    """One CORAL attempt JSON reduced to what the runtime needs: ``commit`` (40 hex), ``timestamp`` (UTC ISO), ``status``, ``score``.
    None when it is not a usable attempt."""
    try:
        doc = loads_strict(raw)
    except Exception:  # noqa: BLE001 - ContractError or UnicodeDecodeError: the file is not a usable attempt
        return None
    if not isinstance(doc, dict):
        return None
    commit, ts = doc.get("commit_hash"), normalize_timestamp(doc.get("timestamp"))
    if not isinstance(commit, str) or not _HEX40.match(commit) or ts is None:
        return None
    status = doc.get("status")
    return {"commit": commit, "timestamp": ts, "status": status if isinstance(status, str) else "", "agent_id": str(doc.get("agent_id", ""))[:64]}


def parse_submission(raw: bytes) -> Optional[tuple[str, str, str]]:
    """``(adapter_id, recipe_sha256, commit)`` from a committed ``submission.json``; the researcher's ``dev_score`` is ignored. None if invalid."""
    try:
        doc = loads_strict(raw)
        if not isinstance(doc, dict):
            return None
        s = Submission(doc.get("adapter_id"), doc.get("recipe_sha256"), doc.get("commit"), None)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None
    return s.adapter_id, s.recipe_sha256, s.commit


def concat_logs(public_dir: Path) -> bytes:
    """The opencode event logs ``logs/<agent>.<n>.log`` concatenated by agent then restart number, each preceded by a marker line."""
    logs = Path(public_dir) / "logs"
    found = []
    if logs.is_dir():
        for p in logs.iterdir():
            m = _LOG_NAME.match(p.name)
            if m and p.is_file() and not p.is_symlink():
                found.append((m.group("agent"), int(m.group("n")), p))
    out = bytearray()
    for agent, n, p in sorted(found, key=lambda t: (t[0], t[1])):
        out += json.dumps({"type": "lab", "subtype": "log_file", "agent": agent, "restart": n}).encode() + b"\n"
        data = p.read_bytes()
        out += data if data.endswith(b"\n") or not data else data + b"\n"
    return bytes(out)


def concat_notes(public_dir: Path) -> bytes:
    notes = Path(public_dir) / "notes"
    out = bytearray()
    if notes.is_dir():
        for p in sorted(notes.rglob("*")):
            if p.is_file() and not p.is_symlink():
                out += f"\n===== notes/{p.relative_to(notes).as_posix()} =====\n".encode() + p.read_bytes()
    return bytes(out)


def _stop_remote(remote: Any, root: str, *, attempts: int = 3, argv: Optional[Sequence[str]] = None) -> bool:
    """Run the stop script (or ``argv``, a variant runtime's own) until it reports ``stopped``; False when the researcher host could not be made to confirm it."""
    for _ in range(max(1, attempts)):
        try:
            res = remote.run(list(argv) if argv is not None else ["bash", "-c", STOP_SCRIPT, "_", root], timeout=120.0)
        except Exception:  # noqa: BLE001
            continue
        if res.returncode == 0 and res.stdout.strip().endswith(b"stopped"):
            return True
    return False


def _clip(text: str, n: int = MAX_ENV_VALUE) -> str:
    return text if len(text.encode("utf-8")) <= n else text.encode("utf-8")[:n].decode("utf-8", "ignore")


# --------------------------------------------------------------------------------------------------
# the runtime
# --------------------------------------------------------------------------------------------------
class CoralRuntime:
    """``ResearcherRuntime`` over CORAL. ``researcher_target`` is a ``RemoteHost`` or an ``user@host`` string (-> ``SshRemote``).

    The methods ``_shim_files``, ``_launch``, ``_make_handle`` and ``_stop_trial`` (and the class flag ``_uses_llm_endpoint``) are the points
    a variant runtime overrides (``runtime_coral_cli.CoralCliRuntime``); here they hold the opencode-through-a-gateway behaviour."""

    _uses_llm_endpoint = True

    def __init__(
        self,
        layout: LabLayout,
        researcher_target: Any,
        coral_home: str,
        pins: Mapping[str, str],
        port_allocator: PortAllocator,
        *,
        clock: Clock,
        remote_trials_root: str = DEFAULT_REMOTE_TRIALS_ROOT,
        reasoning_effort: str = "medium",
        max_real_attempts: Optional[int] = None,
        isolation: str = ISOLATION_SHARED,
    ):
        self._layout = layout
        self._remote: Any = SshRemote(researcher_target) if isinstance(researcher_target, str) else researcher_target
        for need in ("run", "reverse_tunnel"):
            if not callable(getattr(self._remote, need, None)):
                raise TypeError(f"researcher_target needs a {need}() method (RemoteHost)")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", coral_home or ""):
            raise ValueError("coral_home must be a plain absolute path")
        self._coral_home = coral_home.rstrip("/")
        self._pins = dict(pins)
        self._ports = port_allocator
        self._clock = clock
        self._base = remote_trials_root.rstrip("/")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", self._base) or ".." in self._base:
            raise ValueError("remote_trials_root must be a plain absolute path")
        self._effort = reasoning_effort
        self._max_attempts = max_real_attempts
        self._isolation = isolation
        # shared_user_serial: every trial runs as the same unix user, so a second trial alongside the first could read its token and use its
        # budget and adapters. At most one trial holds the runtime from start() until its handle's close() has stopped everything.
        self._serial_lock = threading.Lock()
        self._holder: Optional[str] = None
        self._holder_note = ""

    # ---- start ---------------------------------------------------------------------------------------
    def start(self, context: TrialContext) -> "CoralHandle":
        spec = context.spec
        tid = spec.trial_id
        root = f"{self._base}/{tid}"
        local_port = self._local_port(context.control_endpoint)
        if self._uses_llm_endpoint and self._local_port(context.llm_base_url) != local_port:
            raise RuntimeFailure("the LLM and control endpoints must be served on one port")
        tunnel = None
        handle: Optional[CoralHandle] = None
        staged = False
        claimed = False
        files, context_sha, grader_hashes = self._stage_files(context, root)  # local reads only: refuses before anything is sent
        self._acquire(tid)
        try:
            self._acquire_remote(tid)
            claimed = True
            remote_port = self._free_remote_port()
            gateway_port = self._free_remote_port(avoid=remote_port)
            files.update(self._rendered(context, remote_port, gateway_port))
            staged = True
            self._ok(self._remote.run(["bash", "-c", 'set -e; rm -rf -- "$1"; mkdir -p "$1"; tar -xzf - -C "$1" --no-same-owner; mkdir -p "$1/xdg/config" "$1/xdg/data" "$1/xdg/cache" "$1/xdg/state" "$1/results"; chmod 700 "$1"', "_", root], input=make_stage_tar(files), timeout=600.0), "stage trial directory")
            tunnel = self._remote.reverse_tunnel(remote_port, local_port)
            control_url = f"http://127.0.0.1:{remote_port}/control"
            res = self._launch(root, gateway_port, control_url, tid)
            self._ok(res, "launch coral")
            if not res.stdout.startswith(b"started"):
                raise RuntimeFailure("coral did not report a start")
            handle = self._make_handle(
                remote=self._remote, layout=self._layout, root=root, base=self._base, tunnel=tunnel, trial_id=tid, initial_context_sha256=context_sha,
                grader_hashes=grader_hashes, environment=self._environment(), clock=self._clock, on_closed=lambda clean: self._release(tid, clean),
            )
            return handle
        except BaseException:
            if handle is None:
                clean = True
                if staged:
                    # The launch may have succeeded without its acknowledgment reaching us (ssh timed out, the output was cut): stop whatever
                    # runs under the trial directory BEFORE the pid files that find it are deleted.
                    clean = self._stop_trial(root, tid)
                    if tunnel is not None:
                        try:
                            tunnel.close()
                        except Exception:  # noqa: BLE001
                            pass
                    if clean:
                        try:
                            self._remote.run(["bash", "-c", REMOVE_SCRIPT, "_", root, self._base], timeout=60.0)
                        except Exception:  # noqa: BLE001
                            pass
                self._release(tid, clean, remote=claimed)
            raise

    def _launch(self, root: str, gateway_port: int, control_url: str, trial_id: str) -> RemoteResult:
        """Start CORAL detached on the researcher host; the result's stdout starts with ``started``."""
        overrides = coral_overrides(root, gateway_port, max_real_attempts=self._max_attempts)
        script = render_launch_script(root, self._coral_home, overrides, control_url=control_url)
        return self._remote.run(["bash", "-c", script], timeout=120.0)

    def _make_handle(self, **kwargs: Any) -> "CoralHandle":
        return CoralHandle(**kwargs)

    def _stop_trial(self, root: str, trial_id: str) -> bool:
        """Stop whatever a failed or unacknowledged launch left under ``root``; True when the host confirmed it."""
        return _stop_remote(self._remote, root)

    def _shim_files(self) -> dict[str, bytes]:
        """Executable files placed in ``<trial>/bin`` ahead of everything else on PATH."""
        return {"bin/opencode": SHIM_PATH.read_bytes()}

    def _acquire(self, tid: str) -> None:
        if self._isolation != ISOLATION_SHARED:
            return
        with self._serial_lock:
            if self._holder is not None:
                raise RuntimeFailure(
                    f"{ISOLATION_SHARED} runs one trial at a time: trial {self._holder!r} still holds the researcher host{self._holder_note}; "
                    f"trial {tid!r} was not started"
                )
            self._holder, self._holder_note = tid, ""

    def _acquire_remote(self, tid: str) -> None:
        """Claim the researcher host itself, so a second coordinator, or this one after a restart, cannot start a trial beside a survivor."""
        if self._isolation != ISOLATION_SHARED:
            return
        try:
            res = self._remote.run(["bash", "-c", HOLD_SCRIPT, "_", self._base, tid], timeout=60.0)
        except RemoteError as exc:
            raise RuntimeFailure(f"could not claim the researcher host: {exc}") from None
        if res.returncode == 9:
            holder = res.stdout.decode("utf-8", "replace").strip()[len("held"):].strip()[:80] or "unknown"
            raise RuntimeFailure(
                f"{ISOLATION_SHARED} runs one trial at a time: trial {holder!r} still holds the researcher host ({self._base}/{HOLDER_DIR}); "
                f"trial {tid!r} was not started (if no trial runs there, stop its processes and remove that directory)"
            )
        self._ok(res, "claim the researcher host")

    def _release_remote(self, tid: str) -> None:
        try:
            self._remote.run(["bash", "-c", UNHOLD_SCRIPT, "_", self._base, tid], timeout=60.0)
        except Exception:  # noqa: BLE001 - the claim stays; the next start names it
            pass

    def _release(self, tid: str, clean: bool, *, remote: bool = True) -> None:
        """Called when a trial's run is over. A trial whose processes could not be confirmed stopped keeps the runtime held."""
        if self._isolation != ISOLATION_SHARED:
            return
        if clean:
            with self._serial_lock:
                mine = self._holder == tid
            if mine and remote:
                self._release_remote(tid)
        with self._serial_lock:
            if self._holder != tid:
                return
            if clean:
                self._holder, self._holder_note = None, ""
            else:
                self._holder_note = " (its processes could not be confirmed stopped; clean the researcher host and restart the coordinator)"

    # ---- pieces ----------------------------------------------------------------------------------------
    @staticmethod
    def _local_port(url: str) -> int:
        parts = urllib.parse.urlsplit(url)
        if parts.hostname not in ("127.0.0.1", "localhost") or not parts.port:
            raise RuntimeFailure("the trial endpoints must be loopback URLs with a port")
        return int(parts.port)

    @staticmethod
    def _ok(res: RemoteResult, what: str) -> None:
        if res.returncode != 0:
            raise RuntimeFailure(f"{what} failed (exit {res.returncode}): {res.stderr.decode('utf-8', 'replace')[-300:].strip()}")

    def _free_remote_port(self, avoid: Optional[int] = None) -> int:
        for _ in range(8):
            port = int(self._ports())
            if port == avoid:
                continue
            probe = "import socket,sys;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',int(sys.argv[1])))"
            res = self._remote.run(["python3", "-c", probe, str(port)], timeout=30.0)
            if res.returncode == 0:
                return port
        raise RuntimeFailure("no free port on the researcher host")

    def _stage_files(self, context: TrialContext, root: str) -> tuple[dict[str, tuple[bytes, int]], str, dict[str, str]]:
        spec, layout = context.spec, self._layout
        pack_dir = layout.trial_pack_dir(spec.trial_id)
        pack = read_tree(pack_dir, "pack")
        grader = {f"pack/grader/{k}": v for k, v in read_tree(GRADER_DIR, "").items() if not k.endswith("ENTRYPOINT.json")}
        for key, path in (("data/train.jsonl", layout.train_data(spec.assignment_id)), ("data/dev.jsonl", layout.dev_data(spec.assignment_id))):
            if not path.is_file() or path.is_symlink():
                raise RuntimeFailure(f"public data file missing: {path.name}")
        data = {"data/train.jsonl": layout.train_data(spec.assignment_id).read_bytes(), "data/dev.jsonl": layout.dev_data(spec.assignment_id).read_bytes()}
        tool = TOOL_PATH.read_bytes()
        wrapper = b'#!/bin/sh\nexec python3 -I "$(dirname "$0")/../tools/researcher_tool.py" "$@"\n'
        shims = self._shim_files()
        static: dict[str, bytes] = {**pack, **grader, **data, "tools/researcher_tool.py": tool, "bin/glimmer": wrapper, **shims}
        context_sha = listing_sha256(static)
        grader_hashes = {k[len("pack/grader/"):]: hashlib.sha256(v).hexdigest() for k, v in grader.items()}
        files: dict[str, tuple[bytes, int]] = {k: (v, 0o644) for k, v in static.items()}
        files["bin/glimmer"] = (wrapper, 0o755)
        for name, data_ in shims.items():
            files[name] = (data_, 0o755)
        files[".glimmer_token"] = (context.llm_token.encode("utf-8"), 0o600)
        return files, context_sha, grader_hashes

    def _rendered(self, context: TrialContext, remote_port: int, gateway_port: int) -> dict[str, tuple[bytes, int]]:
        opencode = (json.dumps(render_opencode_json(gateway_port), indent=2) + "\n").encode()
        litellm = render_litellm_config(remote_port, context.llm_token, self._effort).encode()
        return {
            "runtime/opencode.json": (opencode, 0o644),
            "runtime/litellm_config.yaml": (litellm, 0o600),
            "pack/seed/opencode.json": (opencode, 0o644),  # belt and braces: the mount re-applies it at every agent restart
        }

    def _environment(self) -> dict:
        env = {"isolation": self._isolation, "runtime": "coral+opencode", "model": AGENT_MODEL, "reasoning_effort": self._effort}
        for k in ("coral_commit", "opencode_version"):
            if k in self._pins:
                env[k] = self._pins[k]
        return {k: _clip(str(v)) for k, v in env.items()}


class CoralHandle:
    """``RuntimeHandle`` of one trial. ``close`` is not part of the protocol; the trial runner calls it when it exists."""

    def __init__(self, *, remote: Any, layout: LabLayout, root: str, base: str, tunnel: Any, trial_id: str, initial_context_sha256: str, grader_hashes: Mapping[str, str], environment: dict, clock: Clock, on_closed: Optional[Any] = None):
        self._on_closed = on_closed
        self._remote, self._layout, self._root, self._base, self._tunnel = remote, layout, root, base, tunnel
        self.trial_id = trial_id
        self.initial_context_sha256 = initial_context_sha256
        self._grader_hashes = dict(grader_hashes)
        self._env = dict(environment)
        self._clock = clock
        self._lock = threading.RLock()
        self._stop_reason: Optional[ExitReason] = None
        self._exit: Optional[ExitReason] = None
        self._poll_failures = 0
        self._collected: Optional[RuntimeResult] = None
        self._closed = False

    @property
    def remote_root(self) -> str:
        return self._root

    # ---- RuntimeHandle ---------------------------------------------------------------------------------
    def poll(self) -> Optional[ExitReason]:
        with self._lock:
            if self._exit is not None:
                return self._exit
            if self._stop_reason is not None:
                self._exit = self._stop_reason
                return self._exit
        if self._tunnel is not None and not self._tunnel.alive():
            self._env["tunnel"] = "died"
            self._do_stop(ExitReason.CRASHED)
            return self.poll()
        try:
            res = self._remote.run(self._poll_argv(), timeout=30.0)
        except RemoteError:
            res = None
        if res is None or res.returncode != 0:
            with self._lock:
                self._poll_failures += 1
                if self._poll_failures >= 10:
                    self._env["poll"] = "researcher host unreachable"
                    self._exit = ExitReason.CRASHED
                return self._exit
        with self._lock:
            self._poll_failures = 0
            text = res.stdout.decode("utf-8", "replace").strip()
            if text == "running":
                return None
            if text.startswith("exit:"):
                code = text[5:].strip()
                self._env["coral_exit"] = _clip(code)
                self._exit = ExitReason.FINISHED if code == "0" else ExitReason.CRASHED
            else:  # "dead": the process is gone without writing an exit status
                self._env["coral_exit"] = "vanished"
                self._exit = ExitReason.CRASHED
            return self._exit

    def stop(self, reason: ExitReason) -> None:
        """Stop the whole run and record ``reason`` as the exit reason, unless it had already ended on its own."""
        if self.poll() is not None:
            return
        self._do_stop(ExitReason(reason))

    def _do_stop(self, reason: ExitReason) -> None:
        with self._lock:
            if self._stop_reason is not None:
                return
            self._stop_reason = reason
        try:
            self._remote.run(self._stop_argv(), timeout=120.0)
        except RemoteError:
            pass  # close() tries again; the exit reason is already recorded

    def collect(self) -> RuntimeResult:
        with self._lock:
            if self._collected is not None:
                return self._collected
        exit_reason = self.poll()
        if exit_reason is None:
            raise RuntimeFailure("collect() called while the run is still going; stop it first")
        trace_dir = self._layout.trial_trace_dir(self.trial_id)
        public = trace_dir / "public"
        if public.exists():
            raise RuntimeFailure("this trial has already been harvested")
        trace_dir.mkdir(parents=True, exist_ok=True)
        pub_remote = f"{self._root}/run/.coral/public"
        script = HARVEST_SCRIPT.replace("%PATHS%", " ".join(HARVEST_PATHS))
        try:
            res = self._remote.run(["bash", "-c", script, "_", pub_remote], timeout=600.0, max_output=MAX_HARVEST_BYTES)
        except RemoteError as exc:
            raise RuntimeFailure(f"harvest failed: {exc}") from None
        if res.returncode == 4:
            raise RuntimeFailure("nothing to harvest: the run directory was never created")
        if res.returncode != 0:
            raise RuntimeFailure(f"harvest failed (exit {res.returncode})")
        stats = safe_extract(res.stdout, public) if res.stdout else HarvestStats()
        attempts = self._read_attempts(public)
        submissions = self._read_submissions(attempts)
        trace_file = trace_dir / "agent_events.jsonl"
        trace_file.write_bytes(concat_logs(public))
        notes_file = trace_dir / "notes.md"
        notes_file.write_bytes(concat_notes(public))
        changes = self._grader_changes()
        env = dict(self._env)
        env["attempts_total"] = str(len(attempts))
        env["harvest_skipped"] = str(stats.n_skipped)
        if stats.skipped:
            env["harvest_skipped_first"] = _clip("; ".join(stats.skipped[:5]))
        env["grader_changed_files"] = str(changes)
        result = RuntimeResult(
            exit_reason=exit_reason,
            attempts=len([a for a in attempts if a["status"] != "pending"]),
            submissions=tuple(submissions),
            trace=self._layout.file_ref(trace_file),
            notes=self._layout.file_ref(notes_file),
            initial_context_sha256=self.initial_context_sha256,
            environment={k: _clip(str(v)) for k, v in env.items()},
        )
        with self._lock:
            self._collected = result
        return result

    def close(self) -> None:
        """Kill anything left, close the tunnel, delete the remote trial directory. Idempotent; never raises."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        clean = _stop_remote(self._remote, self._root, argv=self._stop_argv())
        try:
            if self._tunnel is not None:
                self._tunnel.close()
        except Exception:  # noqa: BLE001
            pass
        if clean:  # an unconfirmed stop keeps the directory: its pid files are what a later cleanup needs
            clean = self._remove_resources()
        if clean:
            try:
                self._remote.run(["bash", "-c", REMOVE_SCRIPT, "_", self._root, self._base], timeout=120.0)
            except Exception:  # noqa: BLE001
                pass
        if self._on_closed is not None:
            try:
                self._on_closed(clean)
            except Exception:  # noqa: BLE001
                pass

    # ---- variant hooks -----------------------------------------------------------------------------------
    def _poll_argv(self) -> list[str]:
        return ["bash", "-c", POLL_SCRIPT, "_", self._root]

    def _stop_argv(self) -> list[str]:
        return ["bash", "-c", STOP_SCRIPT, "_", self._root]

    def _remove_resources(self) -> bool:
        """Release what lives outside the trial directory (a variant's container). False keeps the directory."""
        return True

    # ---- harvest internals -----------------------------------------------------------------------------
    def _read_attempts(self, public: Path) -> list[dict]:
        out = []
        d = public / "attempts"
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_ATTEMPT_JSON_BYTES:
                    continue
                a = parse_attempt(p.read_bytes())
                if a is not None:
                    out.append(a)
        return sorted(out, key=lambda a: (a["timestamp"], a["commit"]))

    def _read_submissions(self, attempts: list[dict]) -> list[SubmittedAttempt]:
        subs: list[SubmittedAttempt] = []
        repo = f"{self._root}/run/repo"
        for a in attempts:
            if a["status"] == "pending":
                continue
            try:
                res = self._remote.run(["git", "--no-pager", "-C", repo, "cat-file", "blob", f"{a['commit']}:submission.json"], timeout=60.0, max_output=MAX_SUBMISSION_BYTES)
            except RemoteError:
                continue  # too large or unreachable: not a usable submission
            if res.returncode != 0:
                continue  # that commit has no submission.json
            parsed = parse_submission(res.stdout)
            if parsed is None:
                continue
            adapter_id, recipe_sha, commit = parsed
            subs.append(SubmittedAttempt(Submission(adapter_id, recipe_sha, commit, None), a["timestamp"]))
        return subs

    def _grader_changes(self) -> int:
        """How many files of the staged dev grader package differ from what was staged. The grader source is writable by the agent on the
        default path (CORAL notes 5.1); a change means the dev numbers cannot be trusted (they never decide a keep, but the report says so)."""
        try:
            res = self._remote.run(["bash", "-c", LISTING_SCRIPT, "_", f"{self._root}/pack/grader"], timeout=60.0, max_output=1 << 22)
        except RemoteError:
            self._env["grader_modified"] = "unknown"
            return -1
        if res.returncode == 4:  # the directory is gone
            self._env["grader_modified"] = "true"
            return len(self._grader_hashes)
        if res.returncode != 0:
            self._env["grader_modified"] = "unknown"
            return -1
        actual: dict[str, str] = {}
        for line in res.stdout.decode("utf-8", "replace").splitlines():
            digest, _, path = line.partition("  ")
            if path.startswith("./"):
                p = PurePosixPath(path[2:])
                if "__pycache__" not in p.parts and p.suffix != ".pyc":
                    actual[p.as_posix()] = digest
        # a staged file that changed or vanished, or a new Python file; build metadata an editable install leaves behind is not a change
        changed = sum(1 for k, h in self._grader_hashes.items() if actual.get(k) != h) + sum(1 for k in actual if k not in self._grader_hashes and k.endswith(".py"))
        self._env["grader_modified"] = "true" if changed else "false"
        return changed
