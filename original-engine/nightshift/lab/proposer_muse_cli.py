"""``ProposalSource`` over Muse Code (``muse exec``), subscription login, no API key (blocker B3).

The API proposer (``nightshift.lab.proposer.WonderProposalSource``) sends the redacted ``ProposalContext`` to the Meta API with an API
key. This one gives the SAME context text to the ``muse`` CLI, logged in with the Muse Code subscription (OAuth), and reads the final
answer from its ``--json`` event stream. Everything after the model's text is shared with the API path (``parse_json_object``,
``proposal_from_dict``, ``UnparseableProposal``), so a reply is accepted or rejected exactly as there.

What the agent can and cannot see
* The prompt file holds ``context.system_prompt`` + ``context.user_prompt`` (the outer-visible projection the outer loop already ran
  through ``build_context`` and ``assert_no_forbidden``) and the proposal schema text, and nothing else. It is written OUTSIDE the
  workspace, in a sibling directory of the same private temporary root.
* The workspace passed as ``cwd`` and ``--workspace`` is a fresh, empty directory. Muse is started with the ``empty`` preset, shell,
  file-write, web and user-input tools withdrawn, no session log, no contributor-mode upload and no foreign personal context. Only the
  final TEXT of the run counts: the proposer never reads a file from the workspace; it only lists it afterwards (``last_run``) so an
  unexpected write is visible, then deletes the whole temporary root.

Billing guard (critical)
* The child environment is built from an allowlist (``PATH``, ``HOME``, ``LANG``, ``TMPDIR``) and nothing else, so ``MUSE_API_KEY``,
  ``BASETEN_API_KEY``, ``META_API_KEY`` and every other ``*_API_KEY`` never reach it, and the CLI falls back to the stored subscription
  login (``$HOME/.config/muse/auth.json``). ``build_child_env`` also asserts that no key ending in ``_API_KEY`` is present. Environment
  values are never logged or put into an exception message.

Accounting
* ``ProposalResult.usd`` is 0.0 (subscription). Tokens: ``prompt_tokens`` = input tokens, ``completion_tokens`` = output tokens summed
  over the models of the final ``run.usage`` event. The rest (cached and reasoning tokens, calls, the allowance window and weekly
  percentages) is in ``MuseCliProposer.last_usage`` after every call, also after a failed one. A base call costs about 29k input tokens
  of CLI overhead.

Critic
* ``critique(prompt)`` is the outer challenge's one critic call (``outer_challenge``): a separate ``MuseCliProposer`` instance, so its usage
  counters are its own, runs the given prompt once through the same ``_run_once`` (fresh temporary workspace and session, restrictive flags,
  environment allowlist) and returns the text with the requested and served model.

Stop conditions
* ``MuseAllowanceExhausted`` is a ``BudgetExhausted`` (axis ``muse_allowance``), which the outer loop already treats as "stop": the
  subscription window or weekly allowance is used up.
* ``MuseAuthMissing`` and ``MuseCliError`` are ``ProposerUnavailable``: no candidate exists, nothing is recorded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from nightshift import secrets
from nightshift.agents.prompts import lower_effort
from nightshift.lab.contracts import BudgetExhausted, ProposalContext, ProposalInvalid, ProposalResult
from nightshift.lab.proposer import (
    PROPOSAL_SCHEMA,
    ProposerUnavailable,
    UnparseableProposal,
    parse_json_object,
    proposal_from_dict,
)

DEFAULT_MODEL = "muse-wonder-full-1.0"
DEFAULT_EFFORT = "high"
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_MAX_MODEL_STEPS = 8
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "TMPDIR")
ALLOWANCE_AXIS = "muse_allowance"
_MAX_MESSAGE = 300

# Flags that withdraw everything except the model's text answer. The `empty` preset is Muse's no-tools profile; the --disable-* flags are
# belt and braces. The sandbox and approval stay ON (no --yolo, no --disable-sandbox, no --disable-approval, no --trust-workspace).
RESTRICTIVE_FLAGS: tuple[str, ...] = (
    "--preset", "empty",
    "--disable-shell",
    "--disable-write",
    "--disable-web-tools",
    "--disable-reminders",
    "--disable-request-user-input",
    "--approval-mode", "never",
    "--approval-judge", "off",
    "--no-foreign-personal-context",
    "--contributor-mode", "off",
    "--no-session-log",
)

_AUTH_PATTERNS = (
    "missing meta credentials", "muse login", "not logged in", "log in again", "login required", "unauthorized", "invalid credentials",
    "invalid_token", "token expired", "authentication",
)
_ALLOWANCE_PATTERNS = (
    "allowance", "usage limit", "limit reached", "quota", "out of credits", "exhausted", "exceeded your", "plan limit",
)


class MuseCliError(ProposerUnavailable):
    """``muse exec`` failed (nonzero exit, failed or cancelled run, timeout, unreadable stream). No candidate exists."""


class MuseAuthMissing(MuseCliError):
    """No usable Muse login (run ``muse login``). A stop condition for the caller: retrying cannot help."""


class MuseAllowanceExhausted(BudgetExhausted):
    """The Muse Code subscription allowance (window or weekly) is used up. A stop condition: the outer loop stops as for any budget wall."""

    def __init__(self, detail: str = "", *, used_percent: float = 100.0):
        super().__init__(ALLOWANCE_AXIS, 100.0, float(used_percent))
        self.detail = detail[:_MAX_MESSAGE]


@dataclass(frozen=True)
class MuseRun:
    """What one ``muse exec`` invocation produced (parsed, not yet interpreted)."""

    returncode: int
    text: str  # the final answer text ("" when none)
    terminal: Optional[str]  # "completed", "failed", ... or None when the stream had no terminal event
    terminal_reason: str
    usage: dict  # input_tokens, cached_tokens, output_tokens, reasoning_tokens, calls, served_model, window/weekly percent
    stderr: str
    workspace_files: tuple[str, ...]  # relative paths found in the workspace afterwards (ignored)
    timed_out: bool = False


def build_child_env(source: Optional[Mapping[str, str]] = None, *, tmpdir: Optional[str] = None) -> dict[str, str]:
    """The child environment: only ``ENV_ALLOWLIST`` names, copied from ``source`` (default ``os.environ``); ``tmpdir`` replaces TMPDIR.

    Never returns (or logs) anything else, so no ``*_API_KEY`` can reach the CLI and make it bill an API key instead of the subscription."""
    src = os.environ if source is None else source
    env = {k: src[k] for k in ENV_ALLOWLIST if k in src and src[k] != ""}
    env.setdefault("PATH", "/usr/bin:/bin")
    if tmpdir is not None:
        env["TMPDIR"] = tmpdir
    bad = [k for k in env if k.upper().endswith("_API_KEY") or k not in ENV_ALLOWLIST]
    if bad:  # unreachable by construction; fail closed rather than bill an API key
        raise MuseCliError("the child environment would carry a name outside the allowlist")
    return env


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def _pct(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_events(stdout: str) -> tuple[str, Optional[str], str, dict]:
    """``(final text, terminal, terminal reason, usage)`` from the JSONL event stream of ``muse exec --json``.

    Lines that are not JSON objects are skipped. The final text is the terminal event's ``text``; if it is empty the output deltas are
    joined. ``usage`` comes from the LAST ``run.usage`` event (token sums over its ``models`` and the allowance percentages)."""
    deltas: list[str] = []
    text, terminal, reason = "", None, ""
    usage: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        payload = ev.get("payload") if isinstance(ev, dict) else None
        if not isinstance(payload, dict):
            continue
        kind = payload.get("kind")
        if kind == "run_output_delta" and isinstance(payload.get("text"), str):
            deltas.append(payload["text"])
        elif kind == "run_terminal":
            terminal = str(payload.get("terminal") or "")
            reason = str(payload.get("reason") or "")
            text = payload["text"] if isinstance(payload.get("text"), str) else ""
        elif kind == "run_usage":
            usage = _usage_from_payload(payload)
    return (text if text.strip() else "".join(deltas)), terminal, reason, usage


def _usage_from_payload(payload: Mapping[str, Any]) -> dict:
    models = [m for m in (payload.get("models") or []) if isinstance(m, dict)]
    allowance = payload.get("allowance") if isinstance(payload.get("allowance"), dict) else {}
    window = allowance.get("window") if isinstance(allowance.get("window"), dict) else {}
    weekly = allowance.get("weekly") if isinstance(allowance.get("weekly"), dict) else {}
    served = next((str(m["model"]) for m in models if isinstance(m.get("model"), str) and m["model"]), "")
    return {
        "input_tokens": sum(_int(m.get("input_tokens")) for m in models),
        "cached_tokens": sum(_int(m.get("cached_tokens")) for m in models),
        "output_tokens": sum(_int(m.get("output_tokens")) for m in models),
        "reasoning_tokens": sum(_int(m.get("reasoning_tokens")) for m in models),
        "calls": _int(payload.get("calls")) or sum(_int(m.get("calls")) for m in models),
        "failed_calls": _int(payload.get("failed_calls")),
        "served_model": served,
        "allowance_window_used_percent": _pct(window.get("used_percent")),
        "allowance_weekly_used_percent": _pct(weekly.get("used_percent")),
        "allowance_window_resets_at_ms": window.get("resets_at_ms"),
        "allowance_weekly_resets_at_ms": weekly.get("resets_at_ms"),
    }


def _add_usage(a: dict, b: dict) -> dict:
    """Sum of the token counters of two runs (a retry); the later run's model and allowance win."""
    if not a:
        return dict(b)
    out = dict(b)
    for k in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "calls", "failed_calls"):
        out[k] = _int(a.get(k)) + _int(b.get(k))
    return out


def _clip(text: str) -> str:
    return secrets.scrub_text(re.sub(r"\s+", " ", text or "").strip())[:_MAX_MESSAGE]


def output_instructions(schema: Mapping[str, Any]) -> str:
    """What the API path passes out of band as ``response_format`` (the schema), and tells the model it has no tools."""
    return (
        "OUTPUT FORMAT\n"
        "Answer with ONE JSON object that follows this JSON schema exactly (no other keys) and nothing else: no prose, no code fence. "
        "This is a one-shot text answer. Do not call tools, do not read or write files; the workspace is empty and anything you write "
        "there is discarded. Only the text of your final message is read.\n\n"
        + json.dumps(schema, indent=2, sort_keys=True)
    )


def build_prompt(context: ProposalContext, schema: Mapping[str, Any]) -> str:
    """The prompt file text: exactly the system and user prompts the API proposer sends, then the schema and output rules."""
    return context.system_prompt + "\n\n" + context.user_prompt + "\n\n" + output_instructions(schema) + "\n"


def _list_files(root: Path) -> tuple[str, ...]:
    found: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in list(dirnames) + filenames:
                found.append(os.path.relpath(os.path.join(dirpath, name), root))
    except OSError:
        pass
    return tuple(sorted(found))


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


@dataclass
class MuseCliProposer:
    """Implements ``ProposalSource`` by running ``muse exec`` (see the module docstring).

    ``muse_path``: the executable (default: ``muse`` on PATH, else ``~/.local/bin/muse``). ``effort``: Muse reasoning effort
    (``--effort``). ``precheck_allowance``: run ``muse usage`` (no model call) before each proposal and raise ``MuseAllowanceExhausted``
    at 100 percent. ``base_env`` is the environment the allowlist is copied from (default: the process environment)."""

    muse_path: Optional[str] = None
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_model_steps: int = DEFAULT_MAX_MODEL_STEPS
    precheck_allowance: bool = False
    schema: Mapping[str, Any] = field(default_factory=lambda: PROPOSAL_SCHEMA)
    base_env: Optional[Mapping[str, str]] = None
    temp_parent: Optional[str] = None
    extra_args: Sequence[str] = ()
    last_usage: dict = field(default_factory=dict, init=False)
    last_run: Optional[MuseRun] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ---------------------------------------------------------------------------------------------
    def propose(self, context: ProposalContext) -> ProposalResult:
        if self.precheck_allowance:
            self._precheck()
        prompt = build_prompt(context, self.schema)
        total: dict = {}
        raw = ""
        for attempt in (0, 1):
            effort = self.effort if attempt == 0 else lower_effort(self.effort)
            run = self._run_once(prompt, effort)
            total = _add_usage(total, run.usage)
            self._publish(run, total)
            self._raise_for_failure(run)
            raw = run.text
            obj = parse_json_object(raw)
            if obj is not None:
                break
            # like chat_json: ONE retry (one notch lower effort) when the reply holds no JSON object; nothing else is asked twice
        facts = self._facts(total, raw)
        if obj is None:
            raise UnparseableProposal("the reply is not a JSON object", **facts)
        try:
            proposal = proposal_from_dict(obj)
        except ProposalInvalid as exc:
            raise UnparseableProposal(str(exc), **facts) from None
        return ProposalResult(proposal, self.model, facts["served_model"], facts["fallback_used"], 0.0, facts["prompt_tokens"],
                              facts["completion_tokens"], raw)

    # ---------------------------------------------------------------------------------------------
    def _executable(self) -> str:
        if self.muse_path:
            return str(self.muse_path)
        return shutil.which("muse") or str(Path.home() / ".local" / "bin" / "muse")

    def _facts(self, usage: Mapping[str, Any], raw: str) -> dict:
        served = usage.get("served_model") or self.model
        return dict(
            raw_response=raw, usd=0.0, model=self.model, served_model=served, fallback_used=served != self.model,
            prompt_tokens=_int(usage.get("input_tokens")), completion_tokens=_int(usage.get("output_tokens")),
        )

    def _publish(self, run: MuseRun, total: dict) -> None:
        with self._lock:
            self.last_run = run
            self.last_usage = dict(total)

    def _raise_for_failure(self, run: MuseRun) -> None:
        usage = run.usage
        pcts = [p for p in (usage.get("allowance_window_used_percent"), usage.get("allowance_weekly_used_percent")) if p is not None]
        failed = run.timed_out or run.returncode != 0 or run.terminal != "completed"
        if not failed:
            return
        evidence = " ".join((run.stderr, run.terminal_reason, run.text)).lower()
        detail = _clip(run.terminal_reason or run.stderr or run.text or f"exit code {run.returncode}")
        if run.timed_out:
            raise MuseCliError(f"muse exec timed out after {self.timeout_seconds:g}s")
        if any(p in evidence for p in _AUTH_PATTERNS):
            raise MuseAuthMissing(f"muse is not logged in or the login was refused: {detail}")
        if (pcts and max(pcts) >= 100) or any(p in evidence for p in _ALLOWANCE_PATTERNS):
            raise MuseAllowanceExhausted(detail, used_percent=max(pcts) if pcts else 100.0)
        raise MuseCliError(f"muse exec failed (exit {run.returncode}, terminal {run.terminal or 'none'}): {detail}")

    def _precheck(self) -> None:
        """``muse usage`` is a local/account query, not a model call. An unreadable answer is ignored (the run itself reports failures)."""
        try:
            done = subprocess.run([self._executable(), "usage"], capture_output=True, text=True, timeout=60, env=build_child_env(self.base_env),
                                  stdin=subprocess.DEVNULL, check=False)
        except (OSError, subprocess.SubprocessError):
            return
        for m in re.finditer(r"(?im)^\s*(window|weekly)\s*:\s*(\d+(?:\.\d+)?)\s*%\s*used", done.stdout or ""):
            if float(m.group(2)) >= 100:
                raise MuseAllowanceExhausted(f"{m.group(1).lower()} allowance used", used_percent=float(m.group(2)))

    def critique(self, prompt: str) -> Any:
        """One critic call for the outer challenge (``outer_challenge.Critic``): ``prompt`` is the complete prompt text (built and checked by
        ``outer_challenge``), sent once in a fresh ``muse exec`` with the same restrictive flags, environment allowlist, effort and model as a
        proposal. No retry. Raises like ``propose`` (``MuseAllowanceExhausted``, ``MuseAuthMissing``, ``MuseCliError``)."""
        from nightshift.lab.outer_challenge import CriticReply

        if self.precheck_allowance:
            self._precheck()
        run = self._run_once(prompt, self.effort)
        self._publish(run, dict(run.usage))
        self._raise_for_failure(run)
        served = str(run.usage.get("served_model") or "")
        return CriticReply(run.text, self.model, served or "not reported", bool(served) and served != self.model)

    def _run_once(self, prompt: str, effort: Optional[str]) -> MuseRun:
        root = Path(tempfile.mkdtemp(prefix="muse-propose-", dir=self.temp_parent))
        try:
            os.chmod(root, 0o700)
            workspace, prompt_dir, tmp = root / "workspace", root / "prompt", root / "tmp"
            for d in (workspace, prompt_dir, tmp):
                d.mkdir(mode=0o700)
            prompt_file = prompt_dir / "prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            argv = [
                self._executable(), "exec", "--json", "--model", self.model, "--effort", str(effort or self.effort),
                "--prompt-file", str(prompt_file), "--workspace", str(workspace), "--max-model-steps", str(int(self.max_model_steps)),
                *RESTRICTIVE_FLAGS, *self.extra_args,
            ]
            env = build_child_env(self.base_env, tmpdir=str(tmp))
            try:
                proc = subprocess.Popen(argv, cwd=str(workspace), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", start_new_session=True)
            except OSError as exc:
                raise MuseCliError(f"muse could not be started: {_clip(str(exc))}") from None
            timed_out = False
            try:
                out, err = proc.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                out, err = proc.communicate()
            text, terminal, reason, usage = parse_events(out or "")
            return MuseRun(proc.returncode if proc.returncode is not None else -1, text, terminal, reason, usage, err or "",
                           _list_files(workspace), timed_out)
        finally:
            shutil.rmtree(root, ignore_errors=True)
