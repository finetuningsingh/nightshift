"""CORAL agent runtime for Muse Code (``muse exec``), loaded by CORAL as a custom entrypoint.

Applies to CORAL commit 0123dfb939b35228cf2c1fde224cd0561e727408, which has no Muse runtime. Nothing in CORAL is patched: CORAL's
registry (``coral/agent/registry.py``, ``get_runtime``) imports any ``module.path:ClassName`` given as ``agents.runtime`` and checks that
the class satisfies the ``AgentRuntime`` protocol. The researcher runtime (``nightshift/lab/runtime_coral_cli.py``) stages this file in
``<trial>/coral_ext/``, puts that directory on ``PYTHONPATH`` inside the container and starts CORAL with
``agents.runtime=glimmer_muse_runtime:MuseCodeRuntime``.

What it preserves of CORAL's collaboration model, because the manager does all of it around this class: one git worktree and one process per
agent, ``<shared_dir>/notes|skills|attempts|roles`` symlinked to ``.coral/public``, the generated instruction file (``AGENTS.md``, which Muse
loads as workspace rules under ``--yolo``), ``coral eval`` commits, restart on exit with a new prompt, SIGINT for heartbeat interrupts.

Choices (each verified against ``muse exec --help`` of Muse Code 1.4.14 or marked UNVERIFIED until the live probe):

* ``--yolo``: no approval prompts and no OS sandbox, workspace trusted. The docker container is the containment. (Muse's own shell sandbox
  needs privileges the container drops.)
* ``--workspace <worktree>``, ``--json`` (JSONL events on stdout, one file per process: ``logs/<agent>.<n>.log``), ``--prompt-file``,
  ``--model``, ``--disable-request-user-input`` (nobody answers), ``--disable-web-tools`` unless ``runtime_options.web_tools`` is true,
  ``--no-foreign-personal-context``.
* ``--session-id <uuid>`` names the session so ``extract_session_id`` can find it. ``muse exec`` has no resume. A restart starts a NEW
  session with a "re-read your notes" prompt unless ``runtime_options.reuse_session`` is true, which passes the old id again (UNVERIFIED
  whether Muse continues the old conversation).
* ``shared_dir_name`` is ``.agents`` (where Muse discovers skills natively). CORAL's git exclude list does not know it, so ``start`` adds
  ``.agents/`` and ``.muse/`` to the repository's ``info/exclude`` before the agent runs; otherwise ``coral eval``'s ``git add -A`` would
  commit the symlinks.
* The child environment is an ALLOWLIST (``build_agent_env``). ``MUSE_API_KEY``, ``BASETEN_API_KEY`` and ``META_API_KEY`` can never reach
  it (``META_API_KEY`` would override the account login and bill the API).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from coral.agent.exit_classifier import classify_by_uptime
from coral.agent.process import open_agent_stderr_for_log_dir
from coral.agent.runtime import AgentHandle, write_coral_log_entry

logger = logging.getLogger(__name__)

ENV_ALLOW = frozenset(
    {
        "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "PATH", "TMPDIR", "TZ", "SHELL",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
        "MUSE_AUTH_PATH", "MUSE_NO_AUTO_UPDATE", "MUSE_CHANNEL",
        "GLIMMER_CONTROL", "GLIMMER_TOKEN_FILE", "GLIMMER_DATA_DIR", "GLIMMER_TRIAL_ROOT",
        "PYTHONDONTWRITEBYTECODE", "PYTHONPATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
        "UV_CACHE_DIR", "UV_LINK_MODE", "UV_PYTHON_PREFERENCE",
    }
)
ENV_NEVER = frozenset({"MUSE_API_KEY", "BASETEN_API_KEY", "META_API_KEY"})
EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_TERMINAL = re.compile(r'"payload_type"\s*:\s*"run\.terminal\.completed"')
DEFAULT_START = "Begin working on your task and iterating on the seed solution. There is no user in the loop: make decisions, run evals, accumulate knowledge, and iterate without waiting for input."
DEFAULT_RESTART = (
    "Your previous session ended and its conversation is gone. Re-read your instruction file, your notes, your recent attempts and your role in the shared "
    "directory, then continue evolving your solutions where you left off. There is no user in the loop: make decisions, run evals, accumulate knowledge, "
    "and iterate without waiting for input."
)


def build_agent_env(parent: Mapping[str, str], worktree: Path) -> dict[str, str]:
    """The environment of one ``muse exec`` process: allowlisted names only, plus the per-agent venv CORAL gives every runtime."""
    env = {k: v for k, v in parent.items() if k in ENV_ALLOW and k not in ENV_NEVER}
    venv = str(worktree / ".venv")
    env["UV_PROJECT_ENVIRONMENT"] = venv
    env["VIRTUAL_ENV"] = venv
    env["PATH"] = venv + "/bin" + os.pathsep + env.get("PATH", "")
    return env


def build_command(
    muse: str, model: str, workspace: Path, prompt_file: Path, *, session_id: str, options: Optional[Mapping[str, Any]] = None, max_turns: int = 0
) -> list[str]:
    opts = dict(options or {})
    cmd = [
        muse, "exec", "--json", "--yolo", "--no-foreign-personal-context", "--disable-request-user-input",
        "--model", model, "--workspace", str(workspace), "--session-id", session_id, "--prompt-file", str(prompt_file),
    ]
    if not opts.get("web_tools"):
        cmd.append("--disable-web-tools")
    effort = opts.get("reasoning_effort")
    if effort:
        if effort not in EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(EFFORTS)}")
        cmd += ["--reasoning-effort", str(effort)]
    if max_turns and max_turns > 0:
        cmd += ["--max-model-steps", str(int(max_turns))]
    return cmd


def ensure_git_exclude(worktree: Path, entries: tuple[str, ...] = (".agents/", ".muse/")) -> None:
    res = subprocess.run(["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True, cwd=worktree)
    if res.returncode != 0:
        raise RuntimeError(f"git rev-parse --git-common-dir failed in {worktree}: {res.stderr.strip()}")
    exclude = (worktree / res.stdout.strip()).resolve() / "info" / "exclude"
    have = set(exclude.read_text(encoding="utf-8").splitlines()) if exclude.exists() else set()
    missing = [e for e in entries if e not in have]
    if missing:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a", encoding="utf-8") as fh:
            for e in missing:
                fh.write(e + "\n")


def _events(log_path: Path):
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("{"):
                    try:
                        doc = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(doc, dict):
                        yield doc
    except OSError:
        return


class MuseCodeRuntime:
    """``AgentRuntime`` for ``muse exec``. The command is ``$GLIMMER_MUSE_CMD`` (the official launcher, run with MUSE_NO_AUTO_UPDATE=1) or ``muse``."""

    @property
    def instruction_filename(self) -> str:
        return "AGENTS.md"

    @property
    def shared_dir_name(self) -> str:
        return ".agents"

    def extract_session_id(self, log_path: Path) -> Optional[str]:
        """The session id this process was started with: the id the CORAL prompt entry recorded, else the first session stream in the events."""
        found: Optional[str] = None
        for doc in _events(log_path):
            if doc.get("type") == "coral" and isinstance(doc.get("session_id"), str) and _UUID.match(doc["session_id"]):
                found = doc["session_id"]
            stream = doc.get("stream")
            if isinstance(stream, dict) and stream.get("kind") == "session" and isinstance(stream.get("id"), str) and _UUID.match(stream["id"]):
                return stream["id"]
        return found

    def classify_exit(self, log_path: Path, exit_code: Optional[int], uptime_seconds: Optional[float], min_clean_runtime_seconds: int = 60) -> str:
        """``clean`` when the process exited 0 after Muse reported ``run.terminal.completed``; otherwise the uptime heuristic."""
        if exit_code == 0:
            try:
                tail = log_path.read_bytes()[-65536:].decode("utf-8", "replace")
            except OSError:
                tail = ""
            if _TERMINAL.search(tail):
                return "clean"
        return classify_by_uptime(exit_code, uptime_seconds, min_clean_runtime_seconds)

    def start(
        self,
        worktree_path: Path,
        coral_md_path: Path,
        model: str = "muse-spark-1.3",
        runtime_options: Optional[dict[str, Any]] = None,
        max_turns: int = 0,
        log_dir: Optional[Path] = None,
        verbose: bool = False,
        resume_session_id: Optional[str] = None,
        prompt: Optional[str] = None,
        prompt_source: Optional[str] = None,
        task_name: Optional[str] = None,
        task_description: Optional[str] = None,
        gateway_url: Optional[str] = None,
        gateway_api_key: Optional[str] = None,
        run_as_user: Optional[dict[str, Any]] = None,
        sandbox: Any = None,
    ) -> AgentHandle:
        if gateway_url or gateway_api_key:
            raise RuntimeError("the Muse runtime does not use a gateway; set agents.gateway.enabled=false")
        if run_as_user or sandbox:
            raise RuntimeError("the Muse runtime is contained by the docker container; agents.isolate_user and agents.sandbox are not supported")
        opts = dict(runtime_options or {})
        id_file = worktree_path / ".coral_agent_id"
        agent_id = id_file.read_text(encoding="utf-8").strip() if id_file.exists() else "unknown"
        if log_dir is None:
            log_dir = worktree_path / ".agents" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_idx = len(list(log_dir.glob(f"{agent_id}*.log")))
        log_path = log_dir / f"{agent_id}.{log_idx}.log"
        if prompt is None:
            prompt = DEFAULT_RESTART if resume_session_id else DEFAULT_START
        session_id = resume_session_id if (resume_session_id and opts.get("reuse_session") and _UUID.match(resume_session_id)) else str(uuid.uuid4())
        prompt_file = log_dir / f"{agent_id}.{log_idx}.prompt"
        prompt_file.write_text(prompt, encoding="utf-8")
        ensure_git_exclude(worktree_path)
        cmd = build_command(os.environ.get("GLIMMER_MUSE_CMD", "muse"), model, worktree_path, prompt_file, session_id=session_id, options=opts, max_turns=max_turns)
        env = build_agent_env(os.environ, worktree_path)
        logger.info("Starting Muse agent %s in %s (session %s)", agent_id, worktree_path, session_id)
        log_file = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
        err_path = err_file = None
        stderr_target: Any = subprocess.STDOUT
        opened = open_agent_stderr_for_log_dir(log_dir, agent_id)
        if opened is not None:
            err_path, err_file = opened
            stderr_target = err_file
        write_coral_log_entry(
            log_file, prompt=prompt, source=prompt_source or ("restart" if resume_session_id else "start"), agent_id=agent_id,
            session_id=session_id, task_name=task_name, task_description=task_description,
        )
        process = subprocess.Popen(cmd, cwd=str(worktree_path), stdin=subprocess.DEVNULL, stdout=log_file, stderr=stderr_target, start_new_session=True, env=env)
        logger.info("Agent %s started with PID %s", agent_id, process.pid)
        return AgentHandle(agent_id=agent_id, process=process, worktree_path=worktree_path, log_path=log_path, session_id=session_id, _log_file=log_file, err_file=err_file, err_path=err_path)
