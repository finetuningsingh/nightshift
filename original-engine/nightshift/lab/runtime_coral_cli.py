"""``ResearcherRuntime`` over CORAL with Muse Code (CLI) researchers, run inside a docker container.

CLI-backed variant of ``runtime_coral.CoralRuntime`` (blocker B1). Same trial: clean directory, pack checkout, dev grader package, public
train/dev data, the ``glimmer`` tool reaching the control API through the SSH reverse tunnel, harvest of ``.coral/public`` through the safe
extractor, submissions read from the attempt commits, stop/collect/close semantics. What differs:

* The researchers are **Muse Code** agents (``muse exec``), authenticated by the owner's Muse subscription login. No Meta/Muse API key, no
  LiteLLM gateway, no metering proxy (the trial runner still serves the control API; ``TrialContext.llm_base_url`` is ignored).
* CORAL at the pinned commit has no Muse runtime and is **not patched**: ``nightshift/lab/assets/coral_ext/glimmer_muse_runtime.py`` is a
  CORAL custom runtime, staged in ``<trial>/coral_ext/`` and selected with ``agents.runtime=glimmer_muse_runtime:MuseCodeRuntime`` (CORAL's
  registry imports ``module:Class`` entrypoints, ``coral/agent/registry.py::get_runtime``). Every agent keeps its own worktree and process, the
  shared notes/attempts/skills under ``.coral/public``, ``coral eval``, restart on exit and heartbeats, because the CORAL manager does those.
* The whole ``coral start`` process (manager, grader, agents) runs in one **docker container** on the researcher host, never directly in the
  shared home of the unix user. The container sees: the trial directory (read/write), the CORAL virtual environment, the Muse launcher and
  binary directory, and ``uv`` (all read-only), and the host network (Muse needs the internet; the reverse-tunnel port is a host loopback
  port). All capabilities dropped, ``no-new-privileges``, pid and memory limits, the unix user's own uid, no home directory.
* **One login file.** ``_AUTH_SCRIPT`` copies exactly ``<muse_home>/auth/auth.json`` byte for byte (never parsed, printed or logged) to
  ``<trial>/xdg/config/muse/auth.json`` (0600) and the container gets ``MUSE_AUTH_PATH`` pointing at the copy. It is deleted in ``close()``, never
  copied back, and ``collect()`` refuses a harvest in which any long token-like string of the copy appears. ``collect()`` also records
  whether Muse rewrote its copy (a refresh), so the operator knows if the original needs a new login.
* The agent environment is a strict allowlist (``glimmer_muse_runtime.build_agent_env``); the container environment is built from scratch here.
  ``MUSE_API_KEY``, ``BASETEN_API_KEY`` and ``META_API_KEY`` never reach either (``META_API_KEY`` would override the account login and bill the API).
* Researcher usage is **not metered**: ``environment`` says ``researcher_tokens = unmetered, wall-limited``. The wall limit (default 72 minutes)
  stops the team from ``poll()`` and, independently of the coordinator, from ``timeout`` around ``coral start``. Token counts Muse itself
  printed (``run.usage`` events in ``logs/<agent>.<n>.log``) are recorded only when present (``muse_usage_from_logs``), with the models its
  ``run.model.configured`` events reported.
* **The researcher model is verified, not assumed.** ``model`` (``LAB_RESEARCHER_MODEL`` in the CLI; default ``muse-spark-1.3``, e.g.
  ``muse-wonder-full-1.0``) goes to ``agents.model`` and from there to ``muse exec --model``. ``collect()`` records the requested, configured
  (``run.model.configured``) and served (``run.usage`` ``models[].model``) ids, and fails the trial with ``ResearcherModelMismatch`` when Muse
  reports any other model (a silent fallback). No reported model at all is recorded as ``unverified``, not as a pass.

CORAL settings verified in the pinned source (commit 0123dfb9, ``~/work/glimmer-coral-src/src`` on ax42):

* ``coral/config.py``: ``AgentConfig.runtime``/``model`` are plain dotlist keys (``CoralConfig.merge_dotlist``, OmegaConf); ``GatewayConfig.enabled``
  defaults to false and the pack's ``agents.gateway.enabled: true`` is overridden with ``false``; ``agents.runtime_options`` is a free dict
  passed to ``runtime.start``.
* ``coral/agent/registry.py``: an ``agents.runtime`` containing ``:`` is a custom entrypoint, instantiated with no arguments and checked
  against ``AgentRuntime`` (``start``, ``extract_session_id``, ``instruction_filename``, ``shared_dir_name``); ``classify_exit`` is optional.
* ``coral/agent/manager.py``: settings files are written only for shared dirs ``.claude/.opencode/.codex/.cursor``; ``.agents`` gets none, only
  the instruction file ``AGENTS.md`` and the shared-state symlinks. ``coral/workspace/worktree.py::setup_git_exclude`` does not know
  ``.agents/``; the runtime class adds it to ``info/exclude`` itself.
* ``coral/cli/start.py``: ``run.session=docker`` mounts the user's whole ``~/.claude`` and copies every ``*_API_KEY`` of the environment into
  the container; it is NOT used. This module runs ``coral start`` with ``run.session=local`` inside its own container.

Standard library only (reuses ``runtime_coral``). Every shell script here is built from validated values and tested with ``bash -n``.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shlex
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from nightshift.lab.contracts import Clock, ExitReason, LabLayout, PortAllocator, RuntimeFailure, RuntimeResult
from nightshift.lab.runtime_coral import (
    CoralHandle,
    CoralRuntime,
    DEFAULT_REMOTE_TRIALS_ROOT,
    ISOLATION_SHARED,
    RemoteError,
    RemoteResult,
    _clip,
    _stop_remote,
    coral_overrides,
)

DEFAULT_IMAGE = "glimmer-researcher:cli-1"
MUSE_MOUNT = "/opt/muse"
MUSE_LAUNCHER = f"{MUSE_MOUNT}/muse"
DEFAULT_WALL_MINUTES = 72
WALL_GRACE_SECONDS = 120  # the in-container timeout fires this long after the poll-based stop, so poll normally wins
AUTH_DIR_NAME = "xdg/config/muse"  # where the NATIVE CLI looks: $XDG_CONFIG_HOME/muse/auth.json (MUSE_AUTH_PATH is only read by the launcher)
DEFAULT_MODEL = "muse-spark-1.3"
RUNTIME_ENTRYPOINT = "glimmer_muse_runtime:MuseCodeRuntime"
UNMETERED = "unmetered, wall-limited"
EXT_DIR = Path(__file__).resolve().parent / "assets" / "coral_ext"
EXT_MODULE = EXT_DIR / "glimmer_muse_runtime.py"
CONTAINER_DOCKERFILE = Path(__file__).resolve().parent / "assets" / "researcher_container" / "Dockerfile"
FORBIDDEN_ENV = ("MUSE_API_KEY", "BASETEN_API_KEY", "META_API_KEY")

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\[\]-]{0,63}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")
_LIMIT = re.compile(r"^[1-9][0-9]{0,4}[kmg]?$")
_PLAIN_PATH = re.compile(r"^/[A-Za-z0-9_./+@-]+$")
_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def container_name(trial_id: str) -> str:
    if not _NAME.match(trial_id or ""):
        raise RuntimeFailure("trial id is not usable as a container name")
    return f"glimmer-{trial_id}"


# --------------------------------------------------------------------------------------------------
# usage reported by Muse itself
# --------------------------------------------------------------------------------------------------
_USAGE_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "calls")


def _count(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def muse_usage_from_logs(public_dir: Path) -> Optional[dict]:
    """Token counts and models that Muse wrote into CORAL's agent logs (``logs/<agent>.<n>.log``, one ``muse exec --json`` process each), or
    None when no log carries a ``run.usage`` event.

    Per run (``command_id``) the LAST ``run.usage`` event counts (the event is a status update; later ones supersede earlier ones); runs and
    logs add up. ``models`` are the ids of ``run.model.configured`` events, i.e. the model Muse says it ran, not the one requested. Nothing is
    estimated: a log without usage contributes nothing and is counted in ``logs_without_usage``. The numbers are what the CLI reported, not a
    meter: they are not a budget authority."""
    logs = Path(public_dir) / "logs"
    if not logs.is_dir():
        return None
    per_model: dict[str, dict[str, int]] = {}
    configured: dict[str, int] = {}
    with_usage = without = runs = 0
    window = weekly = None
    for path in sorted(logs.glob("*.log")):
        if path.is_symlink() or not path.is_file():
            continue
        last: dict[str, list] = {}
        try:
            with open(path, "rb") as fh:
                for raw in fh:
                    if b"run.usage" not in raw and b"run.model.configured" not in raw:
                        continue
                    try:
                        doc = json.loads(raw)
                    except ValueError:
                        continue
                    pl = doc.get("payload") if isinstance(doc, dict) else None
                    if not isinstance(pl, dict):
                        continue
                    kind = doc.get("payload_type")
                    if kind == "run.model.configured" and isinstance(pl.get("model_id"), str):
                        configured[pl["model_id"][:64]] = configured.get(pl["model_id"][:64], 0) + 1
                    elif kind == "run.usage" and isinstance(pl.get("models"), list):
                        last[str(pl.get("command_id"))] = [m for m in pl["models"] if isinstance(m, dict)]
                        al = pl.get("allowance")
                        if isinstance(al, dict):
                            for key in ("window", "weekly"):
                                part = al.get(key)
                                pct = part.get("used_percent") if isinstance(part, dict) else None
                                if isinstance(pct, (int, float)) and not isinstance(pct, bool):
                                    if key == "window":
                                        window = pct if window is None else max(window, pct)
                                    else:
                                        weekly = pct if weekly is None else max(weekly, pct)
        except OSError:
            continue
        if not last:
            without += 1
            continue
        with_usage += 1
        for models in last.values():
            runs += 1
            for m in models:
                name = str(m.get("model", "unknown"))[:64]
                acc = per_model.setdefault(name, {k: 0 for k in _USAGE_FIELDS})
                for k in _USAGE_FIELDS:
                    acc[k] += _count(m.get(k)) or 0
    if not with_usage:
        return None
    totals = {k: sum(m[k] for m in per_model.values()) for k in _USAGE_FIELDS}
    return {"models": per_model, "totals": totals, "runs": runs, "logs_with_usage": with_usage, "logs_without_usage": without,
            "configured_models": sorted(configured), "allowance_window_used_percent": window, "allowance_weekly_used_percent": weekly}


def configured_models_from_logs(public_dir: Path) -> list[str]:
    """Model ids from ``run.model.configured`` events, even in logs without usage."""
    u = muse_usage_from_logs(public_dir)
    if u is not None:
        return u["configured_models"]
    found: set[str] = set()
    logs = Path(public_dir) / "logs"
    for path in sorted(logs.glob("*.log")) if logs.is_dir() else []:
        if path.is_symlink() or not path.is_file():
            continue
        try:
            with open(path, "rb") as fh:
                for raw in fh:
                    if b"run.model.configured" in raw:
                        try:
                            mid = json.loads(raw).get("payload", {}).get("model_id")
                        except (ValueError, AttributeError):
                            continue
                        if isinstance(mid, str):
                            found.add(mid[:64])
        except OSError:
            continue
    return sorted(found)


class ResearcherModelMismatch(RuntimeFailure):
    """Muse reported running a researcher model other than the requested one (a silent fallback). The trial fails: work done by another
    model must not count as the requested model's."""


def served_models(usage: Optional[Mapping[str, Any]]) -> list[str]:
    """Model ids Muse named in ``run.usage`` events (``models[].model``); entries without a name are not an id."""
    return sorted(m for m in usage["models"] if m != "unknown") if usage else []


def model_verdict(requested: str, served: Sequence[str], configured: Sequence[str]) -> tuple[str, list[str]]:
    """``(verdict, other ids)``: ``ok`` when every reported id is ``requested``; ``MISMATCH`` when any other id was reported (configured or
    served); ``unverified`` when Muse reported none."""
    others = sorted({m for m in (*served, *configured) if m != requested})
    if others:
        return f"MISMATCH: requested {requested}, Muse reported {','.join(others)}", others
    if not served and not configured:
        return "unverified: Muse reported no model", []
    return f"ok: Muse reported only {requested}", []


def usage_environment(usage: Optional[Mapping[str, Any]], configured: Sequence[str] = ()) -> dict[str, str]:
    """The environment-record entries: always the honest label, the numbers only when Muse reported them."""
    served = served_models(usage)
    env = {"researcher_tokens": UNMETERED, "researcher_model_configured": ",".join(configured) if configured else "not reported",
           "researcher_model_served": ",".join(served) if served else "not reported"}
    if usage is None:
        env["researcher_usage_reported"] = "none: no Muse log carried a run.usage event"
    else:
        t = usage["totals"]
        text = (
            f"input={t['input_tokens']} cached={t['cached_tokens']} output={t['output_tokens']} reasoning={t['reasoning_tokens']} calls={t['calls']} "
            f"runs={usage['runs']} logs={usage['logs_with_usage']}/{usage['logs_with_usage'] + usage['logs_without_usage']}; reported by Muse, not metered"
        )
        env["researcher_usage_reported"] = text
        if usage["allowance_weekly_used_percent"] is not None or usage["allowance_window_used_percent"] is not None:
            env["researcher_allowance_used_percent"] = f"window={usage['allowance_window_used_percent']} weekly={usage['allowance_weekly_used_percent']}"
    return env


# --------------------------------------------------------------------------------------------------
# rendered files and scripts (pure; tested)
# --------------------------------------------------------------------------------------------------
# Run on the researcher host with ``python3 -I -c``: argv = source auth file, destination directory, muse bin directory. The file is
# copied as bytes and never parsed or printed. Prints one JSON line (size and Muse version only); errors go to stderr without content.
_AUTH_SCRIPT = r'''
import json, os, stat, sys
src, dest, bindir = sys.argv[1], sys.argv[2], sys.argv[3]
def fail(code, msg):
    sys.stderr.write(msg + "\n"); sys.exit(code)
try:
    st = os.lstat(src)
except OSError:
    fail(3, "no Muse login file on this host (log in first)")
if not stat.S_ISREG(st.st_mode): fail(3, "the Muse login path is not a regular file")
if st.st_size < 2 or st.st_size > 1 << 20: fail(3, "the Muse login file has an implausible size")
with open(src, "rb") as fh: data = fh.read()
os.makedirs(dest, mode=0o700, exist_ok=False)
fd = os.open(os.path.join(dest, "auth.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
with os.fdopen(fd, "wb") as out: out.write(data)
version = ""
try:
    with open(os.path.join(bindir, ".muse-version")) as fh: version = fh.read(64).strip()
except OSError:
    pass
launcher = os.path.join(bindir, "muse")
if not os.path.isfile(launcher): fail(3, "the Muse launcher is missing in " + bindir)
print(json.dumps({"auth_bytes": len(data), "muse_version": version}))
'''

# Run with ``python3 -I -c`` before a harvest: does any long token-like string of the auth copy (or any 31-character slice of it) appear in a file
# about to be harvested? The format of the file is not assumed. Prints only counts. Exit 0 clean (or nothing to scan), 7 found.
_SCAN_SCRIPT = r'''
import os, re, sys
root = sys.argv[1]
auth = os.path.join(root, "xdg", "config", "muse", "auth.json")
try:
    with open(auth, "rb") as fh: blob = fh.read(1 << 20)
except OSError:
    print("unavailable"); sys.exit(0)
needles = set()
for run in re.findall(rb"[A-Za-z0-9_\-.=+/]{24,}", blob):
    if len(run) < 32: needles.add(run)
    else:
        for i in range(0, len(run) - 23, 8): needles.add(run[i:i + 24])
        needles.add(run[-24:])  # any 31 contiguous characters of a token contain one of these windows
needles = sorted(needles)[:400]
if not needles: print("unavailable"); sys.exit(0)
pub = os.path.join(root, "run", ".coral", "public")
hits = scanned = 0
for base, dirs, files in os.walk(pub):
    for name in files:
        p = os.path.join(base, name)
        try:
            if os.path.islink(p) or not os.path.isfile(p): continue
            scanned += 1
            tail = b""
            with open(p, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk: break
                    buf = tail + chunk
                    if any(n in buf for n in needles): hits += 1; break
                    tail = buf[-64:]
        except OSError:
            continue
print("hits=%d scanned=%d needles=%d" % (hits, scanned, len(needles)))
sys.exit(7 if hits else 0)
'''

CLI_POLL_SCRIPT = r"""
ROOT="$1"; NAME="$2"
exitfile() { if [ -f "$ROOT/coral.exit" ]; then echo "exit:$(cat "$ROOT/coral.exit")"; return 0; fi; return 1; }
exitfile && exit 0
st="$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)" || {
  docker version >/dev/null 2>&1 || { echo "docker unavailable" >&2; exit 2; }
  st=missing
}
case "$st" in
  created|running|restarting|paused) echo running ;;
  exited|dead)
    sleep 0.3  # the shell writes coral.exit just after coral exits
    exitfile || echo "exit:$(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null || echo 1)" ;;
  *) sleep 0.3; exitfile || echo dead ;;
esac
"""

# Stop: TERM through the container's init, then KILL after 15 s. The container's pid namespace dies with its first process, so every
# agent and every straggler goes with it; no process sweep is needed. ``stopped`` is printed only when the docker daemon itself says the
# container is not running (a daemon that cannot be reached is not a confirmation).
CLI_STOP_SCRIPT = r"""
NAME="$2"
docker stop -t 15 "$NAME" >/dev/null 2>&1
running() { docker ps -q -f "name=^/${NAME}\$" 2>/dev/null; }
out="$(running)" || { echo "docker unavailable" >&2; exit 2; }
if [ -n "$out" ]; then
  docker kill "$NAME" >/dev/null 2>&1; sleep 1
  out="$(running)" || { echo "docker unavailable" >&2; exit 2; }
  [ -z "$out" ] || { echo running; exit 1; }
fi
echo stopped
"""

# After a confirmed stop: delete the container (no --rm, so its exit code stays readable until now) and the login copy.
CLI_REMOVE_SCRIPT = r"""
ROOT="$1"; BASE="$2"; NAME="$3"
case "$ROOT" in "$BASE"/?*) ;; *) echo refused; exit 3;; esac
case "$ROOT" in *..*|*' '*) echo refused; exit 3;; esac
docker rm -f "$NAME" >/dev/null 2>&1
left="$(docker ps -a -q -f "name=^/${NAME}\$" 2>/dev/null)" || { echo "docker unavailable" >&2; exit 2; }
[ -z "$left" ] || { echo "container remains" >&2; exit 1; }
rm -rf -- "$ROOT/xdg/config/muse"
echo removed
"""

# Did Muse rewrite its copy of the login (a token refresh)? cmp prints nothing; only the verdict is output.
AUTH_CMP_SCRIPT = r"""
if [ ! -f "$2/xdg/config/muse/auth.json" ] || [ ! -f "$1" ]; then echo unknown; exit 0; fi
if cmp -s "$1" "$2/xdg/config/muse/auth.json"; then echo no; else echo yes; fi
"""


def container_environment(root: str, coral_home: str, control_url: str) -> dict[str, str]:
    """The complete environment of the container's first process. Built from scratch; none of it is a secret (the trial token is a 0600 file
    in the trial directory, the Muse login a 0600 file at ``xdg/config/muse/auth.json``)."""
    return {
        "HOME": f"{root}/home",
        "USER": "glimmer",
        "LOGNAME": "glimmer",
        "LANG": "C.UTF-8",
        "TERM": "dumb",
        "ROOT": root,
        "GLIMMER_TRIAL_ROOT": root,
        "PATH": f"{root}/bin:{coral_home}/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "XDG_CONFIG_HOME": f"{root}/xdg/config",
        "XDG_DATA_HOME": f"{root}/xdg/data",
        "XDG_CACHE_HOME": f"{root}/xdg/cache",
        "XDG_STATE_HOME": f"{root}/xdg/state",
        "GLIMMER_CONTROL": control_url,
        "GLIMMER_TOKEN_FILE": f"{root}/.glimmer_token",
        "GLIMMER_DATA_DIR": f"{root}/data",
        "MUSE_AUTH_PATH": f"{root}/{AUTH_DIR_NAME}/auth.json",
        "MUSE_NO_AUTO_UPDATE": "1",
        "GLIMMER_MUSE_CMD": MUSE_LAUNCHER,
        "PYTHONPATH": f"{root}/coral_ext",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def coral_cli_overrides(root: str, *, model: str = DEFAULT_MODEL, max_real_attempts: Optional[int] = None, entry: Optional[Mapping[str, Any]] = None,
                        reasoning_effort: Optional[str] = None) -> list[str]:
    """Dotlist overrides for ``coral start``: the Muse runtime entrypoint, the model, no gateway, absolute workspace paths, the runtime-owned
    grader and the stop condition. A pack edit cannot change any of them (the pack lint compares the non-editable keys)."""
    if not _MODEL.match(model or ""):
        raise ValueError("model is not a plain model id")
    if reasoning_effort is not None and reasoning_effort not in _EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {_EFFORTS}")
    base = coral_overrides(root, 0, max_real_attempts=max_real_attempts, entry=entry)
    drop = ("agents.runtime=", "agents.model=", "agents.gateway.", "agents.runtime_options.mounts=")
    out = [o for o in base if not o.startswith(drop)]
    head = [f"agents.runtime={RUNTIME_ENTRYPOINT}", f"agents.model={model}", "agents.gateway.enabled=false"]
    if reasoning_effort:
        head.append(f"agents.runtime_options.reasoning_effort={reasoning_effort}")
    out[:0] = head
    return out


def render_container_launch_script(
    root: str,
    coral_home: str,
    overrides: Sequence[str],
    *,
    control_url: str,
    name: str,
    muse_bin_dir: str,
    image: str = DEFAULT_IMAGE,
    uv_path: Optional[str] = None,
    wall_seconds: int = DEFAULT_WALL_MINUTES * 60,
    memory: str = "16g",
    cpus: Optional[str] = None,
    pids_limit: int = 4096,
    network: str = "host",
) -> str:
    """The script that starts the container detached. The container's only process is ``coral start`` (under ``timeout`` and docker's init);
    it writes its exit status to ``<root>/coral.exit``. Everything interpolated is validated here or quoted."""
    q = shlex.quote
    if not _NAME.match(name):
        raise ValueError("bad container name")
    if not _LIMIT.match(memory):
        raise ValueError("bad memory limit")
    if cpus is not None and not re.fullmatch(r"[0-9]{1,3}(\.[0-9]{1,2})?", cpus):
        raise ValueError("bad cpu limit")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_./:@-]{0,200}", image):
        raise ValueError("bad image name")
    if network not in ("host", "bridge"):
        raise ValueError("network must be host or bridge")
    for p in (root, coral_home, muse_bin_dir) + ((uv_path,) if uv_path else ()):
        if not _PLAIN_PATH.match(p):
            raise ValueError(f"not a plain absolute path: {p!r}")
    timeout_s = int(wall_seconds) + WALL_GRACE_SECONDS
    inner = (
        f'cd {q(root)} && timeout -s TERM -k 30 {timeout_s} '
        + " ".join(q(a) for a in ["coral", "start", "-c", "pack/task.yaml", *overrides])
        + f' > {q(root + "/coral.out")} 2>&1 < /dev/null; echo $? > {q(root + "/coral.exit")}'
    )
    env = container_environment(root, coral_home, control_url)
    assert not any(k in env for k in FORBIDDEN_ENV)
    env_flags = " ".join(f"-e {q(k + '=' + v)}" for k, v in env.items())
    uv_expr = q(uv_path) if uv_path else '"$(readlink -f "$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")")"'
    run = [
        "docker run -d --init",
        f"--name {q(name)}",
        '--user "$(id -u):$(id -g)"',
        f"--network {network}",
        "--cap-drop ALL --security-opt no-new-privileges",
        f"--pids-limit {int(pids_limit)} --memory {memory} --memory-swap {memory}",
        *([f"--cpus {cpus}"] if cpus else []),
        "--tmpfs /tmp:rw,exec,nosuid,size=4g",
        f"-v {q(root + ':' + root + ':rw')}",
        f"-v {q(coral_home + ':' + coral_home + ':ro')}",
        f"-v {q(muse_bin_dir + ':' + MUSE_MOUNT + ':ro')}",
        '-v "$UV:/usr/local/bin/uv:ro"',
        f"-w {q(root)}",
        env_flags,
        q(image),
        "bash -c " + q(inner),
    ]
    lines = [
        "#!/bin/bash",
        "set -u",
        f"export ROOT={q(root)}",
        f"NAME={q(name)}",
        'cd "$ROOT" || exit 1',
        'mkdir -p "$ROOT/home" || exit 1',
        'rm -f "$ROOT/coral.exit" "$ROOT/coral.pid"',
        f"UV={uv_expr}",
        '[ -x "$UV" ] || { echo "uv not found on the researcher host" >&2; exit 1; }',
        f'docker image inspect {q(image)} >/dev/null 2>&1 || {{ echo "container image {image} is missing on the researcher host; build it from assets/researcher_container/Dockerfile" >&2; exit 1; }}',
        'docker rm -f "$NAME" >/dev/null 2>&1',
        " ".join(run) + " >/dev/null || exit 1",
        'echo started "$NAME"',
    ]
    return "\n".join(lines) + "\n"


# The two edits CORAL's own docker/claude/Dockerfile makes to pyproject.toml so that a source tree without git metadata builds (hatch-vcs
# needs a repository). They apply to the throwaway copy under coral_home, never to the pinned clone.
_PYPROJECT_SED = (
    r"""-e 's/"hatchling", "hatch-vcs"/"hatchling"/' -e 's/dynamic = \["version"\]/version = "0.3.1"/' """
    r"""-e '/^\[tool\.hatch\.version\]$/,+1d' -e '/^\[tool\.hatch\.build\.hooks\.vcs\]$/,+1d'"""
)


def render_install_script(coral_home: str, *, image: str = DEFAULT_IMAGE, uv_path: str = "uv") -> str:
    """One-time install of CORAL into ``coral_home/.venv`` THROUGH the container image (so the venv's interpreter is the image's python, not
    the host's). ``coral_home/src`` must hold the CORAL source at the pinned commit (e.g. ``git archive <commit> | tar -x -C coral_home/src``);
    the install is EDITABLE because CORAL replicates its own install into the grader venv from PEP 610 metadata and supports only editable
    or VCS origins (``coral/workspace/grader_env.py``), and the container sees ``coral_home`` at the same path, read-only. Needs the network
    for PyPI."""
    q = shlex.quote
    if not _PLAIN_PATH.match(coral_home):
        raise ValueError(f"not a plain absolute path: {coral_home!r}")
    src = coral_home + "/src"
    inner = (
        f"set -e; cd {q(src)}; sed -i {_PYPROJECT_SED} pyproject.toml; cd {q(coral_home)}; "
        f"[ -d .venv ] || uv venv .venv; VIRTUAL_ENV={q(coral_home + '/.venv')} uv pip install -e {q(src)}"
    )
    return (
        "#!/bin/bash\nset -eu\n"
        f'UV="$(readlink -f "$(command -v {q(uv_path)})")"\n'
        f'[ -f {q(src + "/pyproject.toml")} ] || {{ echo "put the CORAL source at {src} first" >&2; exit 1; }}\n'
        f'docker run --rm --init --user "$(id -u):$(id -g)" --network host --cap-drop ALL --security-opt no-new-privileges '
        f'-v {q(coral_home + ":" + coral_home + ":rw")} -v "$UV:/usr/local/bin/uv:ro" '
        f'-e HOME={q(coral_home)}/home -e UV_CACHE_DIR={q(coral_home)}/uv-cache -w {q(coral_home)} '
        f"{q(image)} bash -c {q(inner)}\n"
    )


# --------------------------------------------------------------------------------------------------
# the runtime
# --------------------------------------------------------------------------------------------------
class CoralMuseRuntime(CoralRuntime):
    """``ResearcherRuntime`` whose researchers are Muse Code agents in a docker container. ``muse_home`` is the directory holding ``bin/`` (the
    official launcher and its downloaded binary) and ``auth/auth.json`` (the login); only the copy of that one file and the read-only ``bin/`` are
    visible in the container. ``coral_home`` holds a ``.venv`` built through the image (``render_install_script``), mounted read-only. Other
    arguments are as in ``CoralRuntime``."""

    _uses_llm_endpoint = False

    def __init__(
        self,
        layout: LabLayout,
        researcher_target: Any,
        coral_home: str,
        pins: Mapping[str, str],
        port_allocator: PortAllocator,
        *,
        muse_home: str,
        clock: Clock,
        remote_trials_root: str = DEFAULT_REMOTE_TRIALS_ROOT,
        model: str = DEFAULT_MODEL,
        reasoning_effort: Optional[str] = None,
        max_real_attempts: Optional[int] = None,
        isolation: str = ISOLATION_SHARED,
        wall_limit_minutes: float = DEFAULT_WALL_MINUTES,
        image: str = DEFAULT_IMAGE,
        uv_path: Optional[str] = None,
        memory: str = "16g",
        cpus: Optional[str] = None,
        pids_limit: int = 4096,
        login_label: str = "muse-login",
    ):
        super().__init__(layout, researcher_target, coral_home, pins, port_allocator, clock=clock, remote_trials_root=remote_trials_root, max_real_attempts=max_real_attempts, isolation=isolation)
        if not isinstance(muse_home, str) or not _PLAIN_PATH.match(muse_home):
            raise ValueError("muse_home must be a plain absolute path")
        if not _MODEL.match(model or ""):
            raise ValueError("model must be a plain model id")
        if reasoning_effort is not None and reasoning_effort not in _EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {_EFFORTS}")
        if not isinstance(login_label, str) or not _LABEL.match(login_label):
            raise ValueError("login_label must be a plain label")
        if not (isinstance(wall_limit_minutes, (int, float)) and not isinstance(wall_limit_minutes, bool) and 1 <= wall_limit_minutes <= 24 * 60):
            raise ValueError("wall_limit_minutes must be between 1 and 1440")
        self._muse_home = muse_home.rstrip("/")
        self._model, self._effort_opt, self._label = model, reasoning_effort, login_label
        self._wall_seconds = int(round(float(wall_limit_minutes) * 60))
        self._image, self._uv, self._memory, self._cpus, self._pids = image, uv_path, memory, cpus, int(pids_limit)
        # validated now, so a bad value fails at construction rather than at the first launch
        render_container_launch_script("/r/t", coral_home, [], control_url="http://127.0.0.1:1/control", name="n", muse_bin_dir=f"{self._muse_home}/bin", image=image, uv_path=uv_path, memory=memory, cpus=cpus, pids_limit=self._pids)
        self._login_info: dict[str, Any] = {}

    @property
    def auth_source(self) -> str:
        return f"{self._muse_home}/auth/auth.json"

    # ---- hooks -----------------------------------------------------------------------------------------------
    def _shim_files(self) -> dict[str, bytes]:
        return {"coral_ext/glimmer_muse_runtime.py": EXT_MODULE.read_bytes()}

    def _rendered(self, context: Any, remote_port: int, gateway_port: int) -> dict[str, tuple[bytes, int]]:
        return {}  # no provider config, no gateway route, no token in a config file

    def _launch(self, root: str, gateway_port: int, control_url: str, trial_id: str) -> RemoteResult:
        name = container_name(trial_id)
        self._prepare_login(root)
        overrides = coral_cli_overrides(root, model=self._model, max_real_attempts=self._max_attempts, reasoning_effort=self._effort_opt, entry=self._grader_entry())
        script = render_container_launch_script(
            root, self._coral_home, overrides, control_url=control_url, name=name, muse_bin_dir=f"{self._muse_home}/bin", image=self._image, uv_path=self._uv,
            wall_seconds=self._wall_seconds, memory=self._memory, cpus=self._cpus, pids_limit=self._pids,
        )
        return self._remote.run(["bash", "-c", script], timeout=180.0)

    def _grader_entry(self) -> Optional[Mapping[str, Any]]:
        """The grader entrypoint spec; None means the runtime-owned dev grader package (``ENTRYPOINT.json``)."""
        return None

    def _prepare_login(self, root: str) -> None:
        try:
            res = self._remote.run(["python3", "-I", "-c", _AUTH_SCRIPT, self.auth_source, f"{root}/{AUTH_DIR_NAME}", f"{self._muse_home}/bin"], timeout=60.0)
        except RemoteError as exc:
            raise RuntimeFailure(f"could not prepare the Muse login: {exc}") from None
        if res.returncode != 0:
            raise RuntimeFailure("could not prepare the Muse login: " + res.stderr.decode("utf-8", "replace").strip()[-300:])
        try:
            info = json.loads(res.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise RuntimeFailure("could not prepare the Muse login: unreadable answer") from None
        self._login_info = info if isinstance(info, dict) else {}

    def _make_handle(self, **kwargs: Any) -> "CoralMuseHandle":
        return CoralMuseHandle(name=container_name(kwargs["trial_id"]), wall_seconds=self._wall_seconds, auth_source=self.auth_source, model=self._model, **kwargs)

    def _stop_trial(self, root: str, trial_id: str) -> bool:
        name = container_name(trial_id)
        if not _stop_remote(self._remote, root, argv=["bash", "-c", CLI_STOP_SCRIPT, "_", root, name]):
            return False
        try:
            res = self._remote.run(["bash", "-c", CLI_REMOVE_SCRIPT, "_", root, self._base, name], timeout=120.0)
        except RemoteError:
            return False
        return res.returncode == 0

    def _environment(self) -> dict:
        env = {
            "isolation": self._isolation,
            "runtime": "coral+muse_code+docker",
            "model": self._model,
            "researcher_login": self._label,
            "researcher_tokens": UNMETERED,
            "researcher_wall_limit_s": str(self._wall_seconds),
            "container_image": self._image,
            "container_network": "host",
            "muse_version": self._login_info.get("muse_version") or "unknown",
        }
        if self._effort_opt:
            env["reasoning_effort"] = self._effort_opt
        for k in ("coral_commit", "muse_version"):
            if k in self._pins:
                env[k] = self._pins[k]
        return {k: _clip(str(v)) for k, v in env.items()}


CoralCliRuntime = CoralMuseRuntime  # the name used while the researchers were still Claude Code


class CoralMuseHandle(CoralHandle):
    """``RuntimeHandle`` of one container-backed trial: polls and stops through docker, enforces the wall limit, scans the harvest for the
    Muse login, and records the usage and models Muse reported."""

    def __init__(self, *, name: str, wall_seconds: int, auth_source: str, model: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._name = name
        self._wall = int(wall_seconds)
        self._auth_source = auth_source
        self._model = model
        self._model_error: Optional[str] = None
        self._started = float(self._clock())

    @property
    def container(self) -> str:
        return self._name

    # ---- hooks of the base handle ------------------------------------------------------------------------
    def _poll_argv(self) -> list[str]:
        return ["bash", "-c", CLI_POLL_SCRIPT, "_", self._root, self._name]

    def _stop_argv(self) -> list[str]:
        return ["bash", "-c", CLI_STOP_SCRIPT, "_", self._root, self._name]

    def _remove_resources(self) -> bool:
        try:
            res = self._remote.run(["bash", "-c", CLI_REMOVE_SCRIPT, "_", self._root, self._base, self._name], timeout=120.0)
        except Exception:  # noqa: BLE001
            return False
        return res.returncode == 0

    # ---- RuntimeHandle -----------------------------------------------------------------------------------
    def poll(self) -> Optional[ExitReason]:
        reason = super().poll()
        if reason is None and self._clock() - self._started >= self._wall:
            with self._lock:
                self._env.setdefault("wall_limit", "reached")
            self._do_stop(ExitReason.STOPPED_WALL)
            reason = super().poll()
        with self._lock:
            if reason is ExitReason.CRASHED and self._exit is ExitReason.CRASHED and self._env.get("coral_exit") == "124":
                # the in-container timeout fired before the coordinator's poll did: still the wall limit, not a crash
                self._exit = reason = ExitReason.STOPPED_WALL
                self._env["wall_limit"] = "reached (in-container timeout)"
        return reason

    def collect(self) -> RuntimeResult:
        with self._lock:
            if self._model_error is not None:  # the harvest happened (the base handle cached its result); the verdict stands
                raise ResearcherModelMismatch(self._model_error)
            if self._collected is not None:
                return self._collected
        if self.poll() is None:
            raise RuntimeFailure("collect() called while the run is still going; stop it first")
        self._scan_for_login()
        self._record_auth_rewrite()
        result = super().collect()
        public = self._layout.trial_trace_dir(self.trial_id) / "public"
        usage = muse_usage_from_logs(public)
        configured = usage["configured_models"] if usage else configured_models_from_logs(public)
        env = {**result.environment, **usage_environment(usage, configured)}
        env["researcher_model_requested"] = self._model
        verdict, others = model_verdict(self._model, served_models(usage), configured)
        env["researcher_model_check"] = verdict
        result = dataclasses.replace(result, environment={k: _clip(str(v)) for k, v in env.items()})
        if others:
            msg = _clip(f"researcher model mismatch: requested {self._model}, Muse reported {', '.join(others)} "
                        f"(configured: {env['researcher_model_configured']}; served: {env['researcher_model_served']}); the trial fails")
            with self._lock:
                self._model_error = msg
            raise ResearcherModelMismatch(msg)
        with self._lock:
            self._collected = result
        return result

    def _scan_for_login(self) -> None:
        try:
            res = self._remote.run(["python3", "-I", "-c", _SCAN_SCRIPT, self._root], timeout=300.0)
        except RemoteError as exc:
            raise RuntimeFailure(f"harvest refused: the login scan could not run: {exc}") from None
        text = res.stdout.decode("utf-8", "replace").strip()
        if res.returncode == 7:
            self._env["login_scan"] = "FOUND in harvest paths; nothing harvested"
            raise RuntimeFailure("harvest refused: the Muse login (or a slice of it) appears in the files to be harvested; nothing was copied")
        if res.returncode != 0:
            raise RuntimeFailure(f"harvest refused: the login scan failed (exit {res.returncode})")
        self._env["login_scan"] = _clip("unavailable" if text == "unavailable" else "clean " + text)

    def _record_auth_rewrite(self) -> None:
        try:
            res = self._remote.run(["bash", "-c", AUTH_CMP_SCRIPT, "_", self._auth_source, self._root], timeout=30.0)
            verdict = res.stdout.decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001
            verdict = "unknown"
        self._env["login_rewritten_in_container"] = verdict if verdict in ("yes", "no", "unknown") else "unknown"
