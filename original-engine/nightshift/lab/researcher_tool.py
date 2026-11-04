#!/usr/bin/env python3
"""``glimmer``: the researcher's only handle on the lab. Standard library only; a single file, copied to the researcher host.

Run as ``python3 researcher_tool.py <command>`` (the runtime installs a ``glimmer`` wrapper script on PATH). Every command prints ONE JSON
object on stdout. Exit codes: 0 ok, 1 refused (budget, validation, unknown id, a failed job), 2 usage error, 3 control API unreachable,
4 ``wait`` timed out.

====================================================================  ===========================================================
``glimmer budget``                                                    tokens, GPU minutes, dollars and wall seconds left
``glimmer submit <recipe_dir> [--entrypoint "bash run.sh"] [--max-gpu-minutes N]``   pack the directory, start one training job
``glimmer status [<job_id>]``                                         state of a job (default: the latest)
``glimmer wait <job_id> [--timeout S]``                               poll until the job is terminal
``glimmer dev-score <adapter_id>``                                    the control-host DEV score of an adapter
``glimmer pointer <adapter_id>``                                      write ``submission.json`` here (commit it: the grader reads it)
====================================================================  ===========================================================

Configuration: ``GLIMMER_CONTROL`` (``http://127.0.0.1:<port>/control``) and the token from ``GLIMMER_TOKEN_FILE`` or, failing that, the
first ``.glimmer_token`` found in the current directory or any parent. The token is never printed.

The recipe tarball follows the rules of docs/harness-design.md "File formats" (the control API checks them again, and ``run_job.sh`` a third
time): regular files and directories only, relative paths, no ``.git``, at most 5,000 entries, 50 MB compressed and 200 MB unpacked. ``.git``
directories are skipped; a symbolic link or special file is an error. The bytes are deterministic, so the same directory gives the same hash.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shlex
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Optional, TextIO

EXIT_OK, EXIT_REFUSED, EXIT_USAGE, EXIT_UNREACHABLE, EXIT_TIMEOUT = 0, 1, 2, 3, 4

# Keep equal to nightshift.lab.control_api (a test asserts it): this file cannot import the lab package.
MAX_RECIPE_ENTRIES = 5000
MAX_RECIPE_COMPRESSED_BYTES = 50 * 1024 * 1024
MAX_RECIPE_UNPACKED_BYTES = 200 * 1024 * 1024
DEFAULT_ENTRYPOINT = "bash run.sh"
# What every submitted job pays on top of its recipe; keep equal to the trainer settings (GlimmerTrainer pull_margin_s 300, eval_reserve_s 900;
# a test asserts it). Learned the hard way on the first live pilot: researchers asked for 15 to 45 minute caps and every job failed on the reserves.
JOB_OVERHEAD = {
    "setup_minutes_about": 5,
    "adapter_pull_minutes": 5,
    "dev_score_reserve_minutes": 15,
    "rule": "the recipe may run only for max-gpu-minutes minus setup, pull and dev-score minutes (about 25 in all); ask for 25 plus the minutes your recipe needs, and keep the recipe well inside that window: a recipe still running at the deadline is stopped and leaves no adapter; the whole rental, overhead included, is charged against gpu_minutes_left",
}
DEFAULT_WAIT_TIMEOUT_S = 540.0
TERMINAL_STATES = ("ok", "failed", "timeout", "budget_exhausted")
_INTERPRETERS = frozenset({"bash", "sh", "python", "python3", "python3.11", "python3.12", "python3.13"})


class ToolError(Exception):
    """A refusal the researcher should read; becomes ``{"error": ...}`` and exit code 1."""


class Unreachable(Exception):
    pass


# ---- recipe packing ---------------------------------------------------------------------------------
def entrypoint_script(entrypoint: str) -> str:
    try:
        argv = shlex.split(entrypoint)
    except ValueError:
        raise ToolError("entrypoint cannot be parsed") from None
    if not argv:
        raise ToolError("entrypoint is empty")
    if argv[0] in _INTERPRETERS:
        rest = [a for a in argv[1:] if not a.startswith("-")]
        if not rest:
            raise ToolError("entrypoint names no script")
        script = rest[0]
    else:
        script = argv[0]
    parts = [p for p in script.split("/") if p not in ("", ".")]
    if script.startswith("/") or ".." in parts or not parts:
        raise ToolError("entrypoint must be a relative path inside the recipe directory")
    return "/".join(parts)


def pack_recipe(directory: str, entrypoint: str = DEFAULT_ENTRYPOINT) -> bytes:
    """Deterministic tar.gz of ``directory``; raises ToolError for anything the control API would refuse."""
    root = os.path.abspath(directory)
    if not os.path.isdir(root):
        raise ToolError(f"{directory} is not a directory")
    script = entrypoint_script(entrypoint)
    entries: list[tuple[str, str, bool]] = []  # (relpath, abs path, is_dir)
    total = 0

    def walk(abs_dir: str, rel: str) -> None:
        nonlocal total
        with os.scandir(abs_dir) as it:
            children = sorted(it, key=lambda e: e.name)
        for e in children:
            if e.name == ".git":
                continue
            r = f"{rel}/{e.name}" if rel else e.name
            st = os.lstat(e.path)
            if stat.S_ISLNK(st.st_mode):
                raise ToolError(f"{r} is a symbolic link; links are not allowed in a recipe")
            if stat.S_ISDIR(st.st_mode):
                entries.append((r, e.path, True))
                walk(e.path, r)
            elif stat.S_ISREG(st.st_mode):
                total += st.st_size
                entries.append((r, e.path, False))
            else:
                raise ToolError(f"{r} is not a regular file or directory")
            if len(entries) > MAX_RECIPE_ENTRIES:
                raise ToolError(f"recipe has more than {MAX_RECIPE_ENTRIES} entries")
            if total > MAX_RECIPE_UNPACKED_BYTES:
                raise ToolError(f"recipe is larger than {MAX_RECIPE_UNPACKED_BYTES} bytes unpacked")

    walk(root, "")
    if script not in {r for r, _p, d in entries if not d}:
        raise ToolError(f"entrypoint script {script!r} is not a file in {directory}")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for rel, path, is_dir in entries:
                ti = tarfile.TarInfo(rel)
                ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
                if is_dir:
                    ti.type, ti.mode = tarfile.DIRTYPE, 0o755
                    tar.addfile(ti)
                else:
                    st = os.lstat(path)
                    ti.size, ti.mode = st.st_size, 0o755 if st.st_mode & stat.S_IXUSR else 0o644
                    with open(path, "rb") as fh:
                        tar.addfile(ti, fh)
    data = buf.getvalue()
    if len(data) > MAX_RECIPE_COMPRESSED_BYTES:
        raise ToolError(f"recipe is larger than {MAX_RECIPE_COMPRESSED_BYTES} bytes compressed")
    return data


# ---- control API client -----------------------------------------------------------------------------
def find_token(env: Mapping[str, str], cwd: str) -> str:
    path = env.get("GLIMMER_TOKEN_FILE")
    if not path:
        d = os.path.abspath(cwd)
        while True:
            cand = os.path.join(d, ".glimmer_token")
            if os.path.isfile(cand):
                path = cand
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    if not path:
        raise ToolError("no .glimmer_token found here or in a parent directory (and GLIMMER_TOKEN_FILE is not set)")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            token = fh.read(4096).strip()
    except OSError:
        raise ToolError("the token file cannot be read") from None
    if not token:
        raise ToolError("the token file is empty")
    return token


class Client:
    def __init__(self, base: str, token: str):
        self.base, self.token = base.rstrip("/"), token

    def request(self, method: str, path: str, query: Optional[dict] = None, data: Optional[bytes] = None, timeout: float = 30.0) -> tuple[int, dict]:
        url = self.base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        headers = {"Authorization": "Bearer " + self.token, "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/gzip"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - loopback tunnel
                status, raw = resp.status, resp.read(1 << 20)
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read(1 << 20)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise Unreachable(str(getattr(exc, "reason", exc))) from None
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            doc = {"error": f"unexpected response from the control API (HTTP {status})"}
        if not isinstance(doc, dict):
            doc = {"error": f"unexpected response from the control API (HTTP {status})"}
        return status, doc


# ---- commands ---------------------------------------------------------------------------------------
def _emit(out: TextIO, doc: Mapping[str, Any]) -> None:
    out.write(json.dumps(doc, sort_keys=True) + "\n")
    out.flush()


def _git_head(cwd: str) -> str:
    try:
        r = subprocess.run(["git", "-C", cwd, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        raise ToolError("git is not available here") from None
    head = r.stdout.strip()
    if r.returncode != 0 or len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ToolError("this directory is not a git checkout with a commit; commit the recipe first")
    return head


def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glimmer", description="Training jobs, budget and dev scores for the Glimmer lab.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget", help="what is left of the trial budget")
    s = sub.add_parser("submit", help="pack a recipe directory and start a training job")
    s.add_argument("recipe_dir")
    s.add_argument("--entrypoint", default=DEFAULT_ENTRYPOINT)
    s.add_argument("--max-gpu-minutes", type=float, default=None)
    s = sub.add_parser("status", help="state of a job (default: the latest)")
    s.add_argument("job_id", nargs="?")
    s = sub.add_parser("wait", help="wait for a job to finish")
    s.add_argument("job_id")
    s.add_argument("--timeout", type=float, default=DEFAULT_WAIT_TIMEOUT_S)
    s.add_argument("--interval", type=float, default=5.0)
    s = sub.add_parser("dev-score", help="the dev score of an adapter")
    s.add_argument("adapter_id")
    s = sub.add_parser("pointer", help="write submission.json for an adapter")
    s.add_argument("adapter_id")
    return p


def main(
    argv: Optional[list] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    env = os.environ if env is None else env
    cwd = os.getcwd() if cwd is None else cwd
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    parser = build_parser()
    parser.exit_on_error = False
    real_stderr = sys.stderr
    try:
        sys.stderr = err
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
        except argparse.ArgumentError as exc:
            err.write(f"glimmer: {exc}\n")
            return EXIT_USAGE
    finally:
        sys.stderr = real_stderr
    try:
        return _run(args, env, cwd, out, sleep, monotonic)
    except ToolError as exc:
        _emit(out, {"error": str(exc)})
        return EXIT_REFUSED
    except Unreachable as exc:
        _emit(out, {"error": f"control API unreachable: {exc}"})
        return EXIT_UNREACHABLE


def _run(args: argparse.Namespace, env: Mapping[str, str], cwd: str, out: TextIO, sleep: Callable, monotonic: Callable) -> int:
    base = env.get("GLIMMER_CONTROL")
    if not base:
        raise ToolError("GLIMMER_CONTROL is not set")
    client = Client(base, find_token(env, cwd))
    cmd = args.cmd

    if cmd == "budget":
        status, doc = client.request("GET", "/budget")
        if status == 200 and isinstance(doc, dict):
            doc = dict(doc, every_job_costs=JOB_OVERHEAD)
        _emit(out, doc)
        return EXIT_OK if status == 200 else EXIT_REFUSED

    if cmd == "submit":
        if args.max_gpu_minutes is not None and not args.max_gpu_minutes > 0:
            raise ToolError("--max-gpu-minutes must be positive")
        data = pack_recipe(os.path.join(cwd, args.recipe_dir) if not os.path.isabs(args.recipe_dir) else args.recipe_dir, args.entrypoint)
        query = {"entrypoint": args.entrypoint}
        if args.max_gpu_minutes is not None:
            query["max_gpu_minutes"] = repr(float(args.max_gpu_minutes))
        status, doc = client.request("POST", "/submit", query, data, timeout=600.0)
        _emit(out, doc)
        return EXIT_OK if status == 200 and doc.get("status") == "queued" else EXIT_REFUSED

    if cmd == "status":
        status, doc = client.request("GET", "/status", {"job_id": args.job_id} if args.job_id else None)
        _emit(out, doc)
        return EXIT_OK if status == 200 else EXIT_REFUSED

    if cmd == "wait":
        deadline = monotonic() + max(0.0, args.timeout)
        interval = max(0.05, args.interval)
        while True:
            status, doc = client.request("GET", "/status", {"job_id": args.job_id})
            if status != 200:
                _emit(out, doc)
                return EXIT_REFUSED
            if doc.get("state") in TERMINAL_STATES:
                _emit(out, doc)
                return EXIT_OK if doc.get("state") == "ok" else EXIT_REFUSED
            if monotonic() >= deadline:
                doc["timed_out"] = True
                _emit(out, doc)
                return EXIT_TIMEOUT
            sleep(min(interval, max(0.0, deadline - monotonic())))

    if cmd == "dev-score":
        status, doc = client.request("GET", "/dev-score", {"adapter_id": args.adapter_id})
        _emit(out, doc)
        return EXIT_OK if status == 200 else EXIT_REFUSED

    if cmd == "pointer":
        status, doc = client.request("GET", "/pointer", {"adapter_id": args.adapter_id})
        if status != 200:
            _emit(out, doc)
            return EXIT_REFUSED
        submission = {
            "adapter_id": doc["adapter_id"],
            "recipe_sha256": doc["recipe_sha256"],
            "commit": _git_head(cwd),
            "dev_score": doc.get("dev_score"),
        }
        _atomic_write(os.path.join(cwd, "submission.json"), json.dumps(submission, sort_keys=True, indent=2) + "\n")
        _emit(out, submission)
        return EXIT_OK

    raise ToolError(f"unknown command {cmd!r}")  # pragma: no cover - argparse rejects it first


if __name__ == "__main__":
    sys.exit(main())
