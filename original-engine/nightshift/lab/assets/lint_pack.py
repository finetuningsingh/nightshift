#!/usr/bin/env python3
"""Pack lint support: the strict task.yaml reader and the editable-range contract. Standard library only.

Two jobs, one file, so the control host and the researcher host cannot disagree:

* ``python3 -I lint_pack.py --dump <task.yaml>`` prints the pack's ``task.yaml`` as one JSON object on stdout:
  ``{"config": {...}, "coral_validated": true|false}``. ``config`` is what the STRICT READER below made of the file (never what a
  full YAML parser would make of it), so the comparison in ``nightshift.lab.packs.compare_task_keys`` sees explicit keys only.
  When CORAL (``coral.config.CoralConfig``) is importable, the parsed mapping is also given to ``CoralConfig.from_dict`` and a
  rejection exits with status 3; that is the real "CORAL accepts this file" check and it only runs on the researcher host (inside
  CORAL's virtual environment). Without CORAL the dump still succeeds with ``coral_validated: false``.
  Always run it with ``-I``: the file is written by a language model.
* ``DEFAULT_LINT_RANGES`` is the pack contract: which keys the outer loop may change and inside which ranges. It is written to the
  lab's lint-ranges file when the v0 pack is created and must be fixed BEFORE the first candidate; changing it afterwards is an
  ``OperatorChange``.

THE STRICT READER (``parse_strict_yaml``). PyYAML is not installed on the control host and Python has no standard-library YAML
parser. A pack's ``task.yaml`` is therefore restricted to a small subset that this reader accepts and PyYAML reads the same way:

* block mappings (``key: value``) and block sequences (``- item``), nested by indentation with spaces (tabs refused); a sequence
  under a key is indented deeper than the key; a sequence item may start a mapping (``- name: x`` with its keys two columns in);
* keys match ``[A-Za-z_][A-Za-z0-9_.-]*``; duplicate keys are refused;
* scalars: double-quoted strings (JSON escapes), single-quoted strings, integers, decimals, ``true``, ``false``, ``null``, ``~``,
  plain strings, and flow collections only when they are valid JSON (so ``[]`` and ``{}`` work);
* literal block scalars ``|`` and ``|-`` (the indent of the first line sets the block indent);
* full-line ``#`` comments and blank lines. Trailing comments, anchors, aliases, tags, folded scalars (``>``), multi-document
  markers and plain scalars that PyYAML would turn into something other than a string (``yes``, ``no``, ``on``, ``off``, ``1e3``,
  ``2026-11-01``, ``0x10`` ...) are refused with a message that says to quote them.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------------------
# the pack contract: what the outer loop may change
# --------------------------------------------------------------------------------------------------
DEFAULT_LINT_RANGES: dict = {
    "version": 1,
    # Dotted key paths of task.yaml whose subtree the outer loop may change (each is validated below). Every other key of the
    # candidate's task.yaml must equal the parent's: grader.*, agents.model, agents.runtime, agents.gateway, agents.research,
    # agents.runtime_options, workspace.*, run.*, sharing.*, islands.* and any key the parent does not have.
    "editable_keys": [
        "task.description",
        "task.tips",
        "agents.count",
        "agents.heartbeat",
        "agents.assignments",
        "agents.skills",
    ],
    "text_max_chars": 20000,
    "int_ranges": {"agents.count": [1, 4]},
    "heartbeat": {
        "max_entries": 8,
        "names": ["reflect", "consolidate", "pivot", "lint_wiki"],
        "every": [1, 50],
        "triggers": ["interval", "plateau"],
        "option_keys": ["epsilon"],
        "prompt_max_chars": 4000,
    },
    # agents.assignments entries may carry only these keys; runtime and model must stay empty or equal agents.runtime / agents.model,
    # runtime_options may carry only role_file, which must point below roles/.
    "assignments": {"max_entries": 4, "keys": ["runtime", "model", "count", "runtime_options"], "count": [1, 4], "role_dir": "roles/"},
    "skills": {"max_entries": 8, "skill_dir": "skills/"},
}


# --------------------------------------------------------------------------------------------------
# the strict reader
# --------------------------------------------------------------------------------------------------
class YamlSubsetError(ValueError):
    """The text is outside the accepted subset (or malformed inside it)."""


_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*):(?: +(.*))?$")
_INT = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_FLOAT = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]+$")
_AMBIGUOUS = frozenset(
    "y Y n N yes Yes YES no No NO on On ON off Off OFF true True TRUE false False FALSE null Null NULL ~".split()
) - {"true", "false", "null", "~"}
_BAD_START = "&*!|>%@`#"


def _indent_of(line: str) -> int:
    n = len(line) - len(line.lstrip(" "))
    if line[n : n + 1] == "\t":
        raise YamlSubsetError("tabs are not allowed in indentation")
    return n


class _Reader:
    def __init__(self, text: str):
        if "\r" in text:
            raise YamlSubsetError("carriage returns are not allowed")
        if "\x00" in text:
            raise YamlSubsetError("NUL characters are not allowed")
        self.raw = text.split("\n")
        self.i = 0

    # -- navigation
    def _skip_blank(self) -> None:
        while self.i < len(self.raw):
            s = self.raw[self.i].strip(" ")
            if s == "" or s.startswith("#"):
                self.i += 1
            else:
                return

    def _peek(self) -> tuple[int, str] | None:
        self._skip_blank()
        if self.i >= len(self.raw):
            return None
        line = self.raw[self.i]
        return _indent_of(line), line.lstrip(" ")

    def _err(self, msg: str) -> YamlSubsetError:
        return YamlSubsetError(f"line {self.i + 1}: {msg}")

    # -- structure
    def parse_document(self) -> Any:
        head = self._peek()
        if head is None:
            return {}
        indent, content = head
        if indent != 0:
            raise self._err("the document must start in column 0")
        if content.startswith("---") or content.startswith("..."):
            raise self._err("document markers are not supported")
        value = self._block(0)
        if self._peek() is not None:
            raise self._err("unexpected content after the document")
        if not isinstance(value, dict):
            raise YamlSubsetError("the top level must be a mapping")
        return value

    def _block(self, indent: int) -> Any:
        head = self._peek()
        assert head is not None
        _, content = head
        if content == "-" or content.startswith("- "):
            return self._seq(indent)
        return self._map(indent)

    def _map(self, indent: int) -> dict:
        out: dict = {}
        while True:
            head = self._peek()
            if head is None:
                break
            ind, content = head
            if ind < indent:
                break
            if ind > indent:
                raise self._err("unexpected indentation")
            if content == "-" or content.startswith("- "):
                raise self._err("a sequence item where a key was expected (indent a sequence deeper than its key)")
            m = _KEY.match(content)
            if not m:
                raise self._err(f"expected 'key: value', got {content[:40]!r}")
            key, rest = m.group(1), (m.group(2) or "").strip()
            if key in out:
                raise self._err(f"duplicate key {key!r}")
            self.i += 1
            out[key] = self._value(rest, indent)
        return out

    def _value(self, rest: str, parent_indent: int) -> Any:
        if rest == "":
            head = self._peek()
            if head is None or head[0] <= parent_indent:
                return None
            return self._block(head[0])
        if rest in ("|", "|-"):
            return self._literal(parent_indent, strip=rest == "|-")
        return _scalar(rest, self.i)

    def _seq(self, indent: int) -> list:
        out: list = []
        while True:
            head = self._peek()
            if head is None:
                break
            ind, content = head
            if ind < indent:
                break
            if ind > indent:
                raise self._err("unexpected indentation")
            if not (content == "-" or content.startswith("- ")):
                raise self._err("expected '- ' in a sequence")
            rest = content[2:] if content != "-" else ""
            if rest.startswith(" "):
                raise self._err("exactly one space is allowed after '-'")
            rest = rest.rstrip(" ")
            if rest == "":
                self.i += 1
                nxt = self._peek()
                out.append(None if nxt is None or nxt[0] <= indent else self._block(nxt[0]))
            elif _KEY.match(rest) and not rest.startswith(('"', "'")):
                self.raw[self.i] = " " * (ind + 2) + rest
                out.append(self._map(ind + 2))
            else:
                self.i += 1
                out.append(_scalar(rest, self.i))
        return out

    def _literal(self, parent_indent: int, *, strip: bool) -> str:
        first = self.i
        block_indent = None
        lines: list[str] = []
        while self.i < len(self.raw):
            line = self.raw[self.i]
            if line.strip(" ") == "":
                lines.append("")
                self.i += 1
                continue
            ind = _indent_of(line)
            if block_indent is None:
                if ind <= parent_indent:
                    break
                block_indent = ind
            if ind < block_indent:
                break
            lines.append(line[block_indent:])
            self.i += 1
        if block_indent is None:
            raise YamlSubsetError(f"line {first}: an empty literal block")
        while lines and lines[-1] == "":
            lines.pop()
        # lines of blank content that preceded the next key were consumed above; give them back to the structure reader.
        text = "\n".join(lines)
        return text if strip else text + "\n"


def _scalar(text: str, lineno: int) -> Any:
    t = text.strip()
    where = f"line {lineno}: "
    if t == "":
        return None
    if t[0] == '"':
        try:
            value, end = json.JSONDecoder().raw_decode(t)
        except json.JSONDecodeError as exc:
            raise YamlSubsetError(where + f"bad double-quoted string ({exc.msg})") from None
        if not isinstance(value, str) or t[end:].strip() != "":
            raise YamlSubsetError(where + "text after a double-quoted string (trailing comments are not supported)")
        return value
    if t[0] == "'":
        if not re.fullmatch(r"'(?:[^']|'')*'", t):
            raise YamlSubsetError(where + "bad single-quoted string (trailing text or an unescaped quote)")
        return t[1:-1].replace("''", "'")
    if t[0] in "[{":
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            raise YamlSubsetError(where + "flow collections must be valid JSON") from None
    if t[0] in _BAD_START or t.startswith(("- ", "? ")):
        raise YamlSubsetError(where + f"a plain scalar cannot start with {t[0]!r}; quote it")
    if ": " in t or t.endswith(":") or " #" in t:
        raise YamlSubsetError(where + "a plain scalar cannot contain ': ' or ' #'; quote it")
    if t in ("true", "false"):
        return t == "true"
    if t in ("null", "~"):
        return None
    if t in _AMBIGUOUS:
        raise YamlSubsetError(where + f"{t!r} reads as a boolean or null in some YAML parsers; quote it")
    if _INT.match(t):
        return int(t)
    if _FLOAT.match(t):
        return float(t)
    if t[0].isdigit() or t[0] in "+-.":
        raise YamlSubsetError(where + f"{t!r} could be read as a number or date by a YAML parser; quote it")
    return t


def parse_strict_yaml(text: str) -> dict:
    """Parse ``text`` with the strict reader (see the module docstring). Raises ``YamlSubsetError``."""
    return _Reader(text).parse_document()


# --------------------------------------------------------------------------------------------------
# dump
# --------------------------------------------------------------------------------------------------
def dump_task(path: str | Path) -> dict:
    """``{"config": <strict parse>, "coral_validated": bool}``. Raises YamlSubsetError, or ValueError if CORAL rejects the mapping."""
    text = Path(path).read_text(encoding="utf-8")
    config = parse_strict_yaml(text)
    if "preset" in config:
        raise YamlSubsetError("'preset' is not allowed in a pack's task.yaml")
    try:
        from coral.config import CoralConfig  # type: ignore[import-not-found]
    except ImportError:
        return {"config": config, "coral_validated": False}
    CoralConfig.from_dict(json.loads(json.dumps(config)), base_dir=Path(path).parent)
    return {"config": config, "coral_validated": True}


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--dump":
        try:
            result = dump_task(argv[2])
        except (YamlSubsetError, OSError, UnicodeDecodeError) as exc:
            print(f"lint_pack: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # CORAL rejected the mapping
            print(f"lint_pack: CORAL rejects this task.yaml: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        print(json.dumps(result, sort_keys=True))
        return 0
    if len(argv) == 3 and argv[1] == "--write-ranges":
        Path(argv[2]).write_text(json.dumps(DEFAULT_LINT_RANGES, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    print("usage: lint_pack.py --dump <task.yaml> | --write-ranges <path>", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
