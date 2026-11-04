#!/usr/bin/env python3
"""FAKE 'muse' executable for testing nightshift.agents.muse_code. It is NOT Muse Code and says nothing about how Meta's agent behaves.

It emulates only the documented surface (docs/META_API_NOTES.md, https://dev.meta.ai/docs/muse-code and its changelog):
``muse --version`` and ``muse exec`` with ``--model M``, ``--disable-approval``, ``--yolo``, ``--disable-request-user-input``,
``--disable-reminders``, ``--no-session-log``, ``--sandbox-network MODE``, ``--trust-workspace``, ``-c key=value``,
``--max-model-steps N``, ``--json``, ``--prompt-file F``, ``--session-id UUID`` and an optional positional prompt. Exit codes:
0 done, 1 failed, 2 usage. The JSONL it prints is invented (the real schema is unverified; only the line name ``run.usage``
is documented, in changelog 1.4.11, and its fields here are a guess). It has NO sandbox and NO read confinement, which is
exactly why it can be used to prove what the WORKER does: it reads and writes wherever its cwd and the OS allow.

Install it in a test by copying this file to ``<bin dir>/muse`` (chmod +x; ``install()`` below does both) and writing the scenario
next to it as ``<bin dir>/muse.scenario.json``. The worker runs muse with a minimal environment, so the scenario cannot travel in an
environment variable; it is looked up in this order: ``$FAKE_MUSE_SCENARIO`` (a path), then ``<this file>.scenario.json``. Without a
scenario file ``exec`` prints one event and exits 0.

Scenario (JSON)::

    {"version": "muse 0.0.0-fake",          # first line of ``--version``
     "version_exit": 0,                     # exit code of ``--version`` (non-zero emulates a broken install)
     "version_hang": false,                 # ``--version`` never returns
     "record": "/abs/record.jsonl",         # every invocation appends JSON lines here (see below); omitted: nothing is recorded
     "actions": [...],                      # run in order by ``exec`` (see ACTIONS)
     "exit": 0,                             # exit code of ``exec`` when the actions finish (a "hang" action never finishes)
     "ignore_model": "muse-spark-1.2",      # emulate a muse that ignores ``--model``: the model events/usage name THIS model
     "report_model": true}                  # print ``run.model.configured`` first (with the model actually used)

ACTIONS (``op`` plus arguments)::

    edit        {path, content}             write a file relative to the cwd (parents created)
    append      {path, content}             append to a file relative to the cwd
    delete      {path}                      remove a file relative to the cwd
    event       {event: {...}}              print one JSON object as a JSONL line
    assistant   {text}                      print {"type": "assistant", "text": ...}
    raw         {text}                      print a raw line to stdout (garbage lines)
    bytes       {hex}                       write these raw bytes to stdout (invalid UTF-8)
    stderr      {text}                      write text to stderr
    print_key   {}                          print the value of META_API_KEY inside a JSONL event, and echo it to stderr
    spawn_child {pid_file}                  start a child in the same process group that sleeps for an hour; write its pid
    hang        {}                          sleep forever
    write       {path, content}             write to ANY path (absolute or ../): emulates an escape attempt; the outcome is recorded
    read        {path}                      read ANY path: emulates a peek at a protected area; the outcome is recorded
    flood       {bytes}                     print that many bytes of junk to stdout in one go
    sleep       {seconds}                   sleep that long
    peek        {paths: [...], parents: bool}  try to read each path (relative to the cwd or absolute; a directory is listed) and
                                            REPORT WHAT IT SAW in an assistant event (first 120 bytes, or the entry names);
                                            ``parents`` also lists ``..``, ``../..`` and ``../../..``; every outcome is recorded
    symlink     {path, target}              create a symlink relative to the cwd
    fifo        {path}                      create a named pipe relative to the cwd
    bigfile     {path, bytes}               create a file of that many bytes (zeros) relative to the cwd
    usage       {models: {id: {input_tokens, output_tokens, cached_tokens}}}
                                            print a ``run.usage`` line; without ``models`` it reports the ``--model`` value
    model_event {model}                     print ``run.model.configured`` naming ``model`` (default: the ``--model`` value)
    trust_stop  {}                          print the 1.4.0 trust message on stderr and stop with exit 1 (no further actions)
    chmod       {path, mode}                chmod a path relative to the cwd (mode is an octal string such as "000")
    swap_cwd    {}                          move the working directory aside and put a symlink to / in its place (a hostile agent)

Record lines (JSON objects, appended to ``record``)::

    {"type": "version", "argv": [...]}
    {"type": "exec", "argv", "cwd", "env", "pid", "pgid", "ppid", "prompt_path", "prompt_text", "prompt_in_cwd", "flags", "model_arg",
     "sandbox_network", "usage_error"}
    {"type": "attempt", "op": "write"|"read"|"peek", "path", "ok", "error", "saw"}
    {"type": "child", "pid"}
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

ACTIONS = ("edit", "append", "delete", "event", "assistant", "raw", "stderr", "print_key", "spawn_child", "hang", "write", "read",
           "flood", "sleep", "bytes", "peek", "symlink", "fifo", "bigfile", "usage", "model_event", "trust_stop", "chmod", "swap_cwd")
FLAGS_WITH_VALUE = ("--max-model-steps", "--prompt-file", "--session-id", "--model", "--sandbox-network", "-c", "--config",
                    "--effort", "--reasoning-effort")
FLAGS_BARE = ("--disable-approval", "--yolo", "--json", "--disable-request-user-input", "--disable-reminders", "--no-session-log",
              "--trust-workspace", "--disable-write", "--disable-shell")


def install(bin_dir: Path, scenario: dict | None = None) -> Path:
    """Copy this script to ``bin_dir/muse`` (executable) and write ``scenario``; returns the path of the executable."""
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "muse"
    target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    write_scenario(target, scenario or {})
    return target


def write_scenario(executable: Path, scenario: dict) -> None:
    """Replace the scenario of an installed fake."""
    Path(str(executable) + ".scenario.json").write_text(json.dumps(scenario), encoding="utf-8")


def read_records(path: Path) -> list[dict]:
    """Parse a record file written by the fake (an absent file is an empty list)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scenario() -> dict:
    path = os.environ.get("FAKE_MUSE_SCENARIO") or (os.path.abspath(__file__) + ".scenario.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class _Recorder:
    def __init__(self, path: str | None):
        self.path = path

    def add(self, obj: dict) -> None:
        if not self.path:
            return
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _parse_exec(args: list[str]) -> tuple[dict, str | None]:
    """Parse ``exec`` arguments like a CLI would. Returns (options, usage error)."""
    opts: dict = {"flags": [], "prompt": None}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in FLAGS_BARE:
            opts["flags"].append(arg)
        elif arg in FLAGS_WITH_VALUE:
            if i + 1 >= len(args):
                return opts, f"{arg} needs a value"
            if arg in opts and arg not in ("-c", "--config"):
                return opts, f"{arg} given more than once"      # changelog 1.4.4: repeated single-value options are rejected
            opts[arg] = args[i + 1]
            i += 1
        elif arg.startswith("-"):
            return opts, f"unknown option {arg}"
        elif opts["prompt"] is None:
            opts["prompt"] = arg
        else:
            return opts, "too many positional arguments"
        i += 1
    steps = opts.get("--max-model-steps")
    if steps is not None and not steps.isdigit():
        return opts, "--max-model-steps needs a positive integer"
    if opts["prompt"] is None and "--prompt-file" not in opts:
        return opts, "no prompt given"
    return opts, None


_RUN: dict = {"model": None, "scenario": {}}      # the model muse "used" in this exec, for the model/usage actions


def _used_model() -> str:
    """The model this fake reports having used: ``ignore_model`` if the scenario says so, else the ``--model`` value."""
    return str(_RUN["scenario"].get("ignore_model") or _RUN["model"] or "muse-spark-1.2")


def _peek(path: str) -> dict:
    try:
        if os.path.isdir(path):
            return {"ok": True, "dir": sorted(os.listdir(path))[:40]}
        with open(path, "rb") as fh:
            return {"ok": True, "head": fh.read(120).decode("utf-8", errors="replace")}
    except OSError as exc:
        return {"ok": False, "error": type(exc).__name__}


def _do_peek(act: dict, rec: _Recorder) -> None:
    paths = [str(p) for p in act.get("paths", [])]
    if act.get("parents"):
        paths += ["..", "../..", "../../.."]
    saw = {path: _peek(path) for path in paths}
    for path, outcome in saw.items():
        rec.add({"type": "attempt", "op": "peek", "path": path, "ok": outcome["ok"], "error": outcome.get("error"), "saw": outcome})
    _emit({"type": "assistant", "text": "peek: " + json.dumps(saw, sort_keys=True)})


def _do_action(act: dict, rec: _Recorder) -> None:
    op = act.get("op")
    if op not in ACTIONS:
        sys.stderr.write(f"fake muse: unknown action {op!r}\n")
        sys.exit(2)
    path, content = act.get("path", ""), act.get("content", "")
    if op in ("edit", "append"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a" if op == "append" else "w", encoding="utf-8") as fh:
            fh.write(content)
        _emit({"type": "tool_call", "tool": "write_file", "path": path})
    elif op == "delete":
        os.remove(path)
        _emit({"type": "tool_call", "tool": "delete_file", "path": path})
    elif op == "event":
        _emit(act.get("event", {}))
    elif op == "assistant":
        _emit({"type": "assistant", "text": act.get("text", "")})
    elif op == "raw":
        sys.stdout.write(str(act.get("text", "")) + "\n")
        sys.stdout.flush()
    elif op == "bytes":
        sys.stdout.flush()
        sys.stdout.buffer.write(bytes.fromhex(act.get("hex", "")) + b"\n")
        sys.stdout.buffer.flush()
    elif op == "stderr":
        sys.stderr.write(str(act.get("text", "")))
        sys.stderr.flush()
    elif op == "print_key":
        key = os.environ.get("META_API_KEY", "")
        _emit({"type": "debug", "auth": key, "header": f"Authorization: Bearer {key}"})
        sys.stderr.write(f"using key {key}\n")
        sys.stderr.flush()
    elif op == "spawn_child":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if act.get("pid_file"):
            Path(act["pid_file"]).write_text(str(child.pid), encoding="utf-8")
        rec.add({"type": "child", "pid": child.pid})
    elif op == "hang":
        while True:
            time.sleep(3600)
    elif op == "sleep":
        time.sleep(float(act.get("seconds", 0)))
    elif op == "flood":
        sys.stdout.write("x" * int(act.get("bytes", 0)) + "\n")
        sys.stdout.flush()
    elif op == "peek":
        _do_peek(act, rec)
    elif op == "symlink":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        os.symlink(str(act.get("target", "/dev/null")), path)
    elif op == "fifo":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        os.mkfifo(path)
    elif op == "bigfile":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.truncate(int(act.get("bytes", 0)))
    elif op == "usage":
        models = act.get("models") or {_used_model(): {"input_tokens": 1200, "output_tokens": 300, "cached_tokens": 200}}
        _emit({"type": "run.usage", "models": models, "subscription_allowance": {"remaining": "unknown"}})
    elif op == "model_event":
        _emit({"type": "run.model.configured", "model": act.get("model") or _used_model()})
    elif op == "chmod":
        os.chmod(path, int(str(act.get("mode", "700")), 8))
    elif op == "swap_cwd":
        here = os.getcwd()
        os.rename(here, here + ".moved")
        os.symlink("/", here)
    elif op == "trust_stop":
        sys.stderr.write("error: this workspace is not trusted and there is no interactive terminal to ask; "
                         "rerun with --trust-workspace to load its skills and rules\n")
        sys.stderr.flush()
        sys.exit(1)
    elif op == "write":
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            rec.add({"type": "attempt", "op": "write", "path": path, "ok": True, "error": None})
        except OSError as exc:
            rec.add({"type": "attempt", "op": "write", "path": path, "ok": False, "error": type(exc).__name__})
    elif op == "read":
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                fh.read(1000)
            rec.add({"type": "attempt", "op": "read", "path": path, "ok": True, "error": None})
        except OSError as exc:
            rec.add({"type": "attempt", "op": "read", "path": path, "ok": False, "error": type(exc).__name__})


def _exec(args: list[str], scenario: dict) -> int:
    rec = _Recorder(scenario.get("record"))
    opts, error = _parse_exec(args)
    prompt_path = opts.get("--prompt-file")
    prompt_text = opts.get("prompt")
    if error is None and prompt_path is not None:
        try:
            with open(prompt_path, encoding="utf-8") as fh:
                prompt_text = fh.read()
        except OSError:
            error = f"cannot read prompt file {prompt_path}"
    cwd = os.getcwd()
    inside = False
    if prompt_path:
        try:
            inside = os.path.commonpath([os.path.realpath(prompt_path), os.path.realpath(cwd)]) == os.path.realpath(cwd)
        except ValueError:
            inside = False
    _RUN["model"], _RUN["scenario"] = opts.get("--model"), scenario
    rec.add({"type": "exec", "argv": sys.argv, "cwd": cwd, "env": dict(os.environ), "pid": os.getpid(), "pgid": os.getpgid(0),
             "ppid": os.getppid(), "prompt_path": prompt_path, "prompt_text": prompt_text, "prompt_in_cwd": inside,
             "flags": opts["flags"], "model_arg": opts.get("--model"), "sandbox_network": opts.get("--sandbox-network"),
             "usage_error": error})
    if error is not None:
        sys.stderr.write(f"error: {error}\nusage: muse exec [options] <prompt>\n")
        return 2
    _emit({"type": "session_start", "session": "00000000-0000-0000-0000-00000000fake"})
    if scenario.get("report_model"):
        _emit({"type": "run.model.configured", "model": _used_model()})
    for act in scenario.get("actions", []):
        _do_action(act, rec)
    _emit({"type": "session_end", "status": "ok" if scenario.get("exit", 0) == 0 else "failed"})
    return int(scenario.get("exit", 0))


def main(argv: list[str]) -> int:
    scenario = _scenario()
    if argv[:1] == ["--version"]:
        _Recorder(scenario.get("record")).add({"type": "version", "argv": sys.argv})
        if scenario.get("version_hang"):
            time.sleep(3600)
        print(scenario.get("version", "muse 0.0.0-fake (test double)"))
        return int(scenario.get("version_exit", 0))
    if argv[:1] == ["exec"]:
        return _exec(argv[1:], scenario)
    sys.stderr.write("usage: muse [--version] exec [options] <prompt>\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
