"""PatchWorker: an implementer that edits the workspace through four small file tools.

Guarantees
----------
* Every path the model names is validated before it touches the disk: relative, only ``[A-Za-z0-9._/-]``, no ``..``,
  no absolute path, no symlink component, resolves inside the workspace. A write must also match ``task.editable``
  (``fnmatch`` against the workspace-relative path). Reads accept regular files of at most 200 KB (no FIFO, device,
  directory or symlink, opened with ``O_NOFOLLOW``).
* Writes go to a temporary file in the target's directory and are moved into place with ``os.replace``; a failed write
  leaves the old file intact and no temporary file behind.
* A rejected write is returned to the model as a tool error and counted (``rejected_writes`` in the log and summary); after
  ``max_rejected_writes`` the worker stops asking the model for more.
* At most ``max_steps`` (12) model calls. Every tool call in an assistant message is answered, in order.
* Muse Spark always reasons and reasoning tokens count against ``max_completion_tokens`` (``PATCH_STEP_MAX_TOKENS`` per
  step). A reply cut off at that limit (``finish_reason`` ``length``) that holds no usable tool call is asked for once more,
  with the cap doubled and the reasoning effort lowered a notch; the second call counts as a step. A tool call whose
  arguments were cut off is answered with an error that says so.
* If the last reply is still cut off and nothing was ever written, ``ModelError`` is raised: the hypothesis was never
  implemented, so the engine must record an infrastructure error, not grade the untouched workspace as a failed idea.
* File contents and listings go back to the model fenced as data. The model's final words are returned as ``self_report``
  and are never evidence. An ``agent.claim`` is emitted through ``ctx.emit_claim`` only when the final message plainly
  asserts an outcome (improved, fixed); claims are audited, not trusted.
* Every model call carries a ``charge_context`` (experiment id, ``implement`` or ``repair``, lane) when the client supports
  one (``budget.MeteredClient``), so the spend of one experiment can be read off the ledger; other clients are called
  without it.
* Nothing is caught from the model client. A ``ModelError`` (outage) propagates so the engine records the experiment as an
  infrastructure failure (``infra_error``, hypothesis not tested) instead of grading an unchanged or half-written workspace
  as if the hypothesis had failed; ``BudgetExceeded`` and every other ``NightshiftError`` propagate too.
"""
from __future__ import annotations

import fnmatch
import os
import re
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..fence import fence, sanitize
from ..types import ChatMessage, ImplementContext, ImplementResult, ModelClient, ModelError, ToolCall, ToolSpec
from .prompts import cache_key, charge_kwargs, implementer_messages, lower_effort, retry_cap, was_cut_off

MAX_READ_BYTES = 200_000
MAX_STEPS = 12
PATCH_STEP_MAX_TOKENS = 8000        # per model call; includes reasoning tokens
MAX_LIST_ENTRIES = 200
MAX_LIST_DEPTH = 4
MAX_LOG_CHARS = 60_000
_NAME_OK = re.compile(r"[A-Za-z0-9._/-]{1,128}")   # always used with fullmatch: "$" would also accept a trailing newline
_SKIP_DIRS = {"__pycache__", ".git"}


class PathError(ValueError):
    """A path was refused (escape, symlink, not editable, wrong kind of file, too large). The message is safe to show the model."""


class EditError(ValueError):
    """A content edit could not be applied (text not found, ambiguous). Not a security rejection."""


def _jsonschema(props: dict, required: Sequence[str]) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


PATCH_TOOLS: list[ToolSpec] = [
    ToolSpec("list_files", "List the regular files in the workspace with their sizes and whether they are editable. "
             "Optional path: a sub-directory to list.",
             _jsonschema({"path": {"type": "string", "description": "Sub-directory, relative to the workspace root."}}, [])),
    ToolSpec("read_file", "Read one text file (at most 200 KB) from the workspace.",
             _jsonschema({"path": {"type": "string", "description": "Relative path of the file."}}, ["path"])),
    ToolSpec("write_file", "Replace the whole content of an editable file (creating it if the editable list allows it).",
             _jsonschema({"path": {"type": "string", "description": "Relative path of an editable file."},
                          "content": {"type": "string", "description": "The complete new content."}}, ["path", "content"])),
    ToolSpec("replace_in_file", "Replace text in an editable file. old_str must occur exactly once unless replace_all is true.",
             _jsonschema({"path": {"type": "string", "description": "Relative path of an editable file."},
                          "old_str": {"type": "string", "description": "Exact text to replace, copied from the file."},
                          "new_str": {"type": "string", "description": "Replacement text (may be empty)."},
                          "replace_all": {"type": "boolean", "description": "Replace every occurrence."}},
                         ["path", "old_str", "new_str"])),
]
_WRITE_TOOLS = ("write_file", "replace_in_file")


# --------------------------------------------------------------------------------------------------
# confined file access
# --------------------------------------------------------------------------------------------------
class WorkspaceFiles:
    """File operations confined to ``root``. Every public method raises ``PathError`` (or ``EditError``) with a message
    that is safe to hand back to the model; nothing else escapes except programming errors."""

    def __init__(self, root: Path | str, editable: Sequence[str], *, max_read_bytes: int = MAX_READ_BYTES):
        self.root = Path(root).resolve()
        self.editable = tuple(editable)
        self.max_read_bytes = max_read_bytes

    # ---- path validation -------------------------------------------------------------------------
    @staticmethod
    def clean(rel: object, *, allow_empty: bool = False) -> str:
        """The normalised posix relative path, or ``PathError``."""
        if not isinstance(rel, str):
            raise PathError("path must be a string")
        if rel == "" and allow_empty:
            return ""
        if not rel:
            raise PathError("path must not be empty")
        if "\x00" in rel or "\\" in rel:
            raise PathError("path contains characters that are not allowed")
        if rel.startswith("/") or rel.startswith("~"):
            raise PathError("absolute paths are not allowed; use a path relative to the workspace root")
        if not _NAME_OK.fullmatch(rel):
            raise PathError("path may only contain letters, digits, '.', '_', '-' and '/' (at most 128 characters)")
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if ".." in parts:
            raise PathError("'..' is not allowed in paths")
        if not parts and not allow_empty:
            raise PathError("path must name a file")
        return "/".join(parts)

    def _locate(self, clean: str) -> Path:
        """The on-disk path for ``clean`` after proving that no existing component is a symlink or a non-directory."""
        cur = self.root
        for part in [p for p in clean.split("/") if p]:
            cur = cur / part
            try:
                mode = os.lstat(cur).st_mode
            except FileNotFoundError:
                break                     # the rest does not exist yet, so it cannot be a symlink
            except NotADirectoryError:
                raise PathError("a parent of that path is a file, not a directory") from None
            except OSError:
                raise PathError("that path cannot be accessed") from None
            if stat.S_ISLNK(mode):
                raise PathError("symlinks are not allowed in paths")
        target = self.root.joinpath(*[p for p in clean.split("/") if p])
        resolved = target.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PathError("path resolves outside the workspace")
        return target

    def is_editable(self, clean: str) -> bool:
        return any(fnmatch.fnmatchcase(clean, g) for g in self.editable)

    def _writable(self, rel: object) -> tuple[str, Path]:
        clean = self.clean(rel)
        if not self.is_editable(clean):
            raise PathError("that file is not editable; edit only: " + ", ".join(sanitize(g, 60) for g in self.editable[:10]))
        return clean, self._locate(clean)

    # ---- reading ---------------------------------------------------------------------------------
    def read_file(self, rel: object) -> tuple[str, str]:
        """``(clean_path, text)`` of a regular, non-binary file of at most ``max_read_bytes``."""
        clean = self.clean(rel)
        path = self._locate(clean)
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            raise PathError("file not found") from None
        except OSError:
            raise PathError("that path cannot be accessed") from None
        self._check_regular(st)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(path, flags)
        except OSError:
            raise PathError("that file cannot be opened") from None
        try:
            self._check_regular(os.fstat(fd))
            raw = b""
            while len(raw) <= self.max_read_bytes:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                raw += chunk
        finally:
            os.close(fd)
        if len(raw) > self.max_read_bytes:
            raise PathError(f"file is larger than {self.max_read_bytes} bytes")
        if b"\x00" in raw:
            raise PathError("binary files cannot be read")
        return clean, raw.decode("utf-8", errors="replace")

    def _check_regular(self, st: os.stat_result) -> None:
        if not stat.S_ISREG(st.st_mode):
            raise PathError("not a regular file")
        if st.st_size > self.max_read_bytes:
            raise PathError(f"file is larger than {self.max_read_bytes} bytes")

    # ---- writing ---------------------------------------------------------------------------------
    def write_file(self, rel: object, content: object) -> tuple[str, int]:
        """Atomically replace (or create) an editable file. Returns ``(clean_path, bytes_written)``."""
        clean, path = self._writable(rel)
        if not isinstance(content, str):
            raise PathError("content must be a string")
        if "\x00" in content:
            raise PathError("content must be text")
        encoded = content.encode("utf-8", errors="replace")
        if len(encoded) > self.max_read_bytes:
            raise PathError(f"content is larger than {self.max_read_bytes} bytes")
        mode = 0o644
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            st = None
        except OSError:
            raise PathError("that path cannot be accessed") from None
        if st is not None:
            if not stat.S_ISREG(st.st_mode):
                raise PathError("the target exists and is not a regular file")
            mode = stat.S_IMODE(st.st_mode) or mode
        self._atomic_write(path, encoded, mode)
        return clean, len(encoded)

    def replace_in_file(self, rel: object, old: object, new: object, replace_all: bool = False) -> tuple[str, int]:
        """Exact-text replacement in an editable file. ``old`` must occur once unless ``replace_all``. Returns ``(path, count)``."""
        clean, path = self._writable(rel)
        if not isinstance(old, str) or not isinstance(new, str):
            raise EditError("old_str and new_str must be strings")
        if not old:
            raise EditError("old_str must not be empty")
        _, text = self.read_file(clean)
        count = text.count(old)
        if count == 0 and "\r\n" in text and "\r\n" not in old:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
            count = text.count(old)
        if count == 0:
            raise EditError("old_str was not found in the file; read the file again and copy the text exactly")
        if count > 1 and not replace_all:
            raise EditError(f"old_str occurs {count} times; add surrounding context to make it unique, or set replace_all")
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        self.write_file(clean, updated)
        return clean, count if replace_all else 1

    def _atomic_write(self, path: Path, data: bytes, mode: int) -> None:
        tmp = path.with_name(f".{path.name[:60]}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fchmod(fd, mode)
            finally:
                os.close(fd)
            os.replace(tmp, path)
        except OSError as exc:
            raise PathError(f"the file could not be written ({sanitize(exc.strerror or 'I/O error', 60)})") from None
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ---- listing ---------------------------------------------------------------------------------
    def list_files(self, rel: object = "") -> str:
        """Plain-text listing of regular files under ``rel`` (default: the root), at most 200 entries and 4 levels deep."""
        clean = self.clean(rel if rel not in (None, ".") else "", allow_empty=True)
        base = self._locate(clean) if clean else self.root
        try:
            if not stat.S_ISDIR(os.lstat(base).st_mode):
                raise PathError("not a directory")
        except FileNotFoundError:
            raise PathError("directory not found") from None
        lines: list[str] = []
        skipped = 0

        def walk(directory: Path, depth: int) -> None:
            nonlocal skipped
            try:
                entries = sorted(os.scandir(directory), key=lambda e: e.name)
            except OSError:
                return
            for entry in entries:
                relpath = entry.path[len(str(self.root)) + 1:] if entry.path.startswith(str(self.root)) else entry.name
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISLNK(st.st_mode):
                    skipped += 1
                elif stat.S_ISDIR(st.st_mode):
                    if entry.name not in _SKIP_DIRS and depth < MAX_LIST_DEPTH:
                        walk(Path(entry.path), depth + 1)
                elif stat.S_ISREG(st.st_mode):
                    if len(lines) < MAX_LIST_ENTRIES:
                        flag = " [editable]" if self.is_editable(relpath) else ""
                        lines.append(f"{sanitize(relpath, 128)} ({st.st_size} bytes){flag}")
                    else:
                        skipped += 1

        walk(base, 0)
        if skipped:
            lines.append(f"[{skipped} further entries not shown (symlinks or over the listing limit)]")
        return "\n".join(lines) or "(no files)"


# --------------------------------------------------------------------------------------------------
# claim detection
# --------------------------------------------------------------------------------------------------
_HEDGE = re.compile(r"n't|\b(?:not|no|never|cannot|unable|fail\w*|without|hope\w*|expect\w*|should|may|might|could|would|"
                    r"aim\w*|attempt\w*|try|trying|tried|if|whether|unclear|untested|unmeasured|unknown|unsure)\b", re.I)
_IMPROVED = re.compile(r"\b(?:improv(?:ed|es|ement)|better than|outperform(?:s|ed)|reduc(?:ed|es) (?:the )?(?:error|rmse|loss))\b", re.I)
_FIXED = re.compile(r"\b(?:fixed|resolved|repaired|corrected)\b", re.I)


def detect_claim(text: str) -> tuple[str, str] | None:
    """``("improved"|"fixed", sentence)`` for the first sentence that plainly asserts an outcome, else ``None``.

    Deliberately conservative: a sentence with any hedge or negation ("should improve", "did not fix", "no improvement")
    asserts nothing. The claim is only a statement to be audited; it is never evidence.
    """
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        s = sentence.strip()
        if not s or _HEDGE.search(s):
            continue
        if _IMPROVED.search(s):
            return "improved", sanitize(s, 300)
        if _FIXED.search(s):
            return "fixed", sanitize(s, 300)
    return None


# --------------------------------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------------------------------
@dataclass
class _Tally:
    edited: list[str] = field(default_factory=list)
    writes: int = 0
    rejected: int = 0
    bad_calls: int = 0


class PatchWorker:
    """Implements ``types.Implementer`` with a bounded tool-calling loop over ``WorkspaceFiles``."""

    def __init__(self, client: ModelClient, *, reasoning_effort: str | None = "low", max_steps: int = MAX_STEPS,
                 max_completion_tokens: int | None = PATCH_STEP_MAX_TOKENS, max_rejected_writes: int = 5,
                 max_calls_per_step: int = 8):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.max_steps = max_steps
        self.max_completion_tokens = max_completion_tokens
        self.max_rejected_writes = max_rejected_writes
        self.max_calls_per_step = max_calls_per_step
        self.model = str(getattr(client, "model", "") or "model")
        self.identity = f"patch-worker[{self.model}]"

    def implement(self, ctx: ImplementContext) -> ImplementResult:
        files = WorkspaceFiles(ctx.workspace, ctx.task.editable)
        messages = implementer_messages(ctx)
        key = cache_key("implementer", ctx.task.name)
        lane = getattr(ctx.lane, "lane", None)
        charge = charge_kwargs(self.client, exp_id=ctx.exp_id, purpose="repair" if ctx.repair_findings else "implement",
                               actor=f"implementer:{lane}" if isinstance(lane, str) and lane else None)
        tally = _Tally()
        log: list[str] = []
        steps, final_text, finished, aborted = 0, "", False, False
        while steps < self.max_steps:
            resp = self.client.chat(messages, tools=PATCH_TOOLS, reasoning_effort=self.reasoning_effort,
                                    max_completion_tokens=self.max_completion_tokens, cache_key=key, **charge)
            steps += 1
            calls = list(resp.message.tool_calls or [])
            cut = was_cut_off(resp)
            if cut and not any(c.parse_error is None for c in calls) and steps < self.max_steps:
                log.append(f"step {steps}: reply cut off at the token limit with no usable tool call; asking once more")
                resp = self.client.chat(messages, tools=PATCH_TOOLS, reasoning_effort=lower_effort(self.reasoning_effort),
                                        max_completion_tokens=retry_cap(self.max_completion_tokens), cache_key=key, **charge)
                steps += 1
                calls = list(resp.message.tool_calls or [])
                cut = was_cut_off(resp)
            text = resp.message.content or ""
            log.append(f"step {steps}: model said {sanitize(text, 500)!r}" if text else f"step {steps}: model called tools")
            if not calls:
                final_text, finished = text, True
                break
            messages.append(ChatMessage(role="assistant", content=resp.message.content, tool_calls=calls))
            for index, call in enumerate(calls):
                if index >= self.max_calls_per_step:
                    result = f"error: at most {self.max_calls_per_step} tool calls per turn; this one was not run"
                else:
                    result = self._run_tool(files, call, tally, log, cut)
                messages.append(ChatMessage(role="tool", content=result, tool_call_id=call.id, name=call.name or None))
            if tally.rejected >= self.max_rejected_writes:
                log.append(f"stopped: {tally.rejected} rejected writes")
                aborted = True
                break
        if finished and cut and tally.writes == 0:
            # The last word was a reply cut off at the token limit and nothing was ever written: the hypothesis was never
            # tried. Grading the untouched workspace would record "no gain" (and a lesson) for an idea nobody implemented, so
            # report an unavailable model instead; the engine records an infrastructure error and the idea can be tried again.
            raise ModelError("the implementer's reply was cut off at the token limit before any edit was made", retryable=True)
        capped = not (finished or aborted)
        if capped:
            log.append(f"stopped: step cap of {self.max_steps} reached")
        self_report = sanitize(final_text, 2000)
        self._emit_claim(ctx, self_report)
        return ImplementResult(summary=self._summary(tally, steps, aborted, capped), steps=steps, worker=self.identity,
                               raw_log="\n".join(log)[:MAX_LOG_CHARS], self_report=self_report)

    # ---- internals -------------------------------------------------------------------------------
    @staticmethod
    def _emit_claim(ctx: ImplementContext, self_report: str) -> None:
        if ctx.emit_claim is None:
            return
        claim = detect_claim(self_report)
        if claim is None:
            return
        try:
            ctx.emit_claim(*claim)
        except (ValueError, TypeError, KeyError):
            pass                         # a malformed claim must not lose the implementation; other errors propagate

    @staticmethod
    def _summary(t: _Tally, steps: int, aborted: bool, capped: bool) -> str:
        edited = ", ".join(dict.fromkeys(t.edited)) or "no files"
        parts = [f"{t.writes} write(s) to {edited}", f"{steps} model step(s)", f"rejected_writes={t.rejected}"]
        if aborted:
            parts.append("stopped after repeated rejected writes")
        if capped:
            parts.append("step cap reached")
        return "; ".join(parts)

    def _run_tool(self, files: WorkspaceFiles, call: ToolCall, tally: _Tally, log: list[str], cut: bool = False) -> str:
        name = call.name
        if call.parse_error:
            tally.bad_calls += 1
            log.append(f"  {sanitize(name, 40)}: unparseable arguments")
            advice = ("Your reply was cut off at the output limit: make a smaller edit with replace_in_file, or write less."
                      if cut else "Send one JSON object with the documented fields.")
            return f"error: the arguments could not be parsed (parse_error: {sanitize(call.parse_error, 120)}). {advice}"
        args = call.arguments if isinstance(call.arguments, dict) else {}
        try:
            if name == "list_files":
                listing = files.list_files(args.get("path", ""))
                log.append(f"  list_files {sanitize(str(args.get('path', '')), 60)!r}")
                return fence(listing, "listing", max_chars=20_000)
            if name == "read_file":
                path, text = files.read_file(args.get("path"))
                log.append(f"  read_file {path} ({len(text)} chars)")
                return f"{path} ({len(text.encode('utf-8', errors='replace'))} bytes)\n" + fence(
                    text, f"file_{re.sub(r'[^A-Za-z0-9_-]', '_', path)[:30]}", max_chars=MAX_READ_BYTES)
            if name == "write_file":
                path, written = files.write_file(args.get("path"), args.get("content"))
                tally.writes += 1
                tally.edited.append(path)
                log.append(f"  write_file {path} ({written} bytes)")
                return f"ok: wrote {written} bytes to {path}"
            if name == "replace_in_file":
                path, count = files.replace_in_file(args.get("path"), args.get("old_str"), args.get("new_str"),
                                                    args.get("replace_all") is True)
                tally.writes += 1
                tally.edited.append(path)
                log.append(f"  replace_in_file {path} ({count} replacement(s))")
                return f"ok: replaced {count} occurrence(s) in {path}"
        except PathError as exc:
            if name in _WRITE_TOOLS:
                tally.rejected += 1
            log.append(f"  {sanitize(name, 40)} rejected: {exc}")
            return f"error: {exc}"
        except EditError as exc:
            log.append(f"  {sanitize(name, 40)} failed: {exc}")
            return f"error: {exc}"
        tally.bad_calls += 1
        log.append(f"  unknown tool {sanitize(name, 40)!r}")
        return "error: unknown tool; the tools are list_files, read_file, write_file and replace_in_file"
