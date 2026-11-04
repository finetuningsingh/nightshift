#!/usr/bin/env python3
"""Rehearse a LIVE run without a Meta API key: the real 'meta' backend against a SIMULATED model on loopback.

THIS IS A SIMULATION. Nothing here calls Meta and nothing here is Muse Spark output. The server is the repository's test double
(``tests/support/fake_meta_server.py``, which enforces the documented Chat Completions rules) driven by a scripted stand-in
brain (``tests/support/sim_muse.py``) that knows the answer key of the bundled task. What the run exercises is the plumbing: the
supervisor tool loop, the patch worker, the strict-schema roles, the audit judge, metering, the ledger, the dashboard and the
report, all over real HTTP. The ledger records the endpoint as loopback, so the dashboard and the report carry the SIMULATED
MODEL banner and say so. Use it to rehearse the screen recording and to learn the dashboard before spending real credits; never
show its output as a result of a Meta model.

    python3 scripts/simulate_live.py                                   # runs/simulated, about 7 seconds
    python3 scripts/simulate_live.py --plant tamper,injection,duplicate --out runs/sim-redteam
    python3 scripts/simulate_live.py --serve                           # watch it live at http://127.0.0.1:8765

A run takes 5 to 10 seconds (about 16 with ``hang``, which waits out a planted stall). ``--plant`` adds scripted red-team
attacker lanes (those lanes are scripted, the other roles are the simulated model). Which names take effect depends on the
simulator brain: measured by running this script once per name with the default brain (seven runs, all exit 0, chain
consistent), ``tamper``, ``injection``, ``hang``, ``infra_flake`` and ``duplicate`` each launched their planted lane (a
``RT-<name>`` experiment in the ledger; for ``duplicate`` the engine then blocks it as an identical change, which is the
point of that plant), while ``claim_without_run`` and ``overfit`` did NOT run: the default brain proposes the same ideas first,
so the planted lane is blocked as a duplicate of an experiment already in flight (``experiment.blocked``, reason duplicate)
before anything executes. The report's scoreboard lists the planted lanes that did not run. A bad ``--plant`` name exits 2
before anything starts, with the CLI's own message and list. A run that fails before it starts (a busy ``--port``, a bad
``--task``) is not verified and its empty run directory is removed so the same ``--out`` can be used again. Ctrl-C while
``--serve`` is active stops the run, still verifies the ledger, prints the summary and exits 130.

Afterwards: ``python3 -m nightshift serve runs/simulated --replay-duration 150`` replays the ledger as a screen recording and
``python3 -m nightshift report runs/simulated`` writes the report. Extra arguments after ``--`` go to ``nightshift run``.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nightshift.cli import CliError, _parse_plant  # noqa: E402  (the CLI's own --plant parser: same names, same wording)
from nightshift.ledger import LEDGER_NAME, Ledger  # noqa: E402
from nightshift.types import LedgerCorrupt  # noqa: E402
from tests.support.sim_muse import SimBrain, SimMuse  # noqa: E402

EXIT_INTERRUPTED = 130
CHILD_GRACE_SEC = 3.0        # after Ctrl-C: let a child that got the terminal's SIGINT finish closing the ledger by itself
CHILD_INT_SEC = 20.0         # then: SIGINT (the run closes as halted), and wait this long before terminate
CHILD_SETTLE_SEC = 1.0       # a run that has closed its ledger gets this long to print its summary before the next signal
CHILD_TERM_SEC = 5.0         # then: SIGTERM, and wait this long before kill
EMPTY_RUN_FILES = frozenset({LEDGER_NAME, LEDGER_NAME + ".lock"})     # all a run that never started leaves behind


def _ledger_state(run_dir: Path) -> tuple[Optional[int], bool]:
    """``(events, started)`` for a run directory: how many events its ledger holds (0 when there is none; ``None`` when it
    cannot be read) and whether ``run.started`` is among them. An unreadable ledger counts as started: it is neither deleted
    nor excused from ``verify``."""
    if not (run_dir / LEDGER_NAME).is_file():
        return 0, False
    try:
        events = Ledger.read_only(run_dir).events()
    except (LedgerCorrupt, OSError):
        return None, True
    return len(events), any(ev.kind == "run.started" for ev in events)


def _run_finished(run_dir: Path) -> bool:
    """True when the ledger of ``run_dir`` holds ``run.finished`` (the run closed itself). Never raises."""
    try:
        return any(ev.kind == "run.finished" for ev in Ledger.read_only(run_dir).events())
    except (LedgerCorrupt, OSError):
        return False


def _remove_if_empty(run_dir: Path) -> bool:
    """Remove ``run_dir`` when it holds no events and nothing but the ledger file and its lock (what a run that failed before
    it started leaves behind). True when it was removed; anything else in the directory is never touched."""
    events, _ = _ledger_state(run_dir)
    if events != 0 or not run_dir.is_dir() or {e.name for e in run_dir.iterdir()} - EMPTY_RUN_FILES:
        return False
    shutil.rmtree(run_dir, ignore_errors=True)
    return not run_dir.exists()


def _tee_stderr(proc: subprocess.Popen, lines: list[str]) -> threading.Thread:
    """Echo the child's stderr as it arrives and keep it, so a failed start can quote the run's own error afterwards."""
    def pump() -> None:
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            lines.append(line.rstrip("\n"))
    t = threading.Thread(target=pump, daemon=True, name="simulate-live-stderr")
    t.start()
    return t


def _wait_slices(proc: subprocess.Popen, patience: float, run_dir: Optional[Path], stop_when_finished: bool) -> int:
    """``proc.wait(timeout=patience)``, except that with ``stop_when_finished`` it gives up (``TimeoutExpired``) shortly after the
    ledger holds ``run.finished``: a run that closed after SIGINT only waits for a second Ctrl-C to leave its dashboard open."""
    deadline = time.monotonic() + patience
    finished_at: Optional[float] = None
    while True:
        remaining = deadline - time.monotonic()
        try:
            return proc.wait(timeout=max(0.0, min(0.5, remaining)))
        except subprocess.TimeoutExpired:
            pass
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, patience)
        if stop_when_finished and run_dir is not None:
            if finished_at is None and _run_finished(run_dir):
                finished_at = time.monotonic()
            if finished_at is not None and time.monotonic() - finished_at >= CHILD_SETTLE_SEC:
                raise subprocess.TimeoutExpired(proc.args, patience)


def _wait(proc: subprocess.Popen, run_dir: Optional[Path] = None) -> tuple[int, bool]:
    """Wait for the child. On Ctrl-C: give a child that already got the terminal's SIGINT a few seconds to close the ledger
    itself, then forward SIGINT (the run closes as halted), then SIGTERM, then SIGKILL. A child whose run is already closed
    (``run.finished`` in ``run_dir``) is not waited for in full: with ``--serve`` it only holds the dashboard open until a second
    Ctrl-C. Returns ``(exit code, interrupted)``."""
    try:
        return proc.wait(), False
    except KeyboardInterrupt:
        print("\ninterrupted: stopping the run (its ledger stays intact and is verified below)", file=sys.stderr, flush=True)
    for sig, patience in ((None, CHILD_GRACE_SEC), (signal.SIGINT, CHILD_INT_SEC), (signal.SIGTERM, CHILD_TERM_SEC),
                          (signal.SIGKILL, CHILD_TERM_SEC)):
        try:
            if sig is not None and proc.poll() is None:
                proc.send_signal(sig)
            return _wait_slices(proc, patience, run_dir, stop_when_finished=sig == signal.SIGINT), True
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:       # a second Ctrl-C: stop being polite
            proc.kill()
    return proc.wait(), True


def _exit_status(code: int) -> int:
    """A child killed by a signal has a negative return code; as a process exit status that must be the shell's 128 + signal."""
    return 128 - code if code < 0 else code


def _own_error(stderr_lines: list[str]) -> str:
    """The run's own error line: the last ``error:`` line of its stderr, else its last non-empty line."""
    lines = [ln.strip() for ln in stderr_lines if ln.strip()]
    errors = [ln for ln in lines if ln.lower().startswith("error")]
    line = (errors or lines or ["(the run printed no error message)"])[-1]
    return line[len("error:"):].strip() if line.lower().startswith("error:") else line


def main(argv: list[str] | None = None) -> int:
    """Run ``nightshift run --backend meta`` against the simulator, then ``verify`` the ledger and print a summary.

    Guarantees: a ``--plant`` name is checked up front with the CLI's own list and wording (exit 2, nothing started); an
    existing ``--out`` is never touched (exit 2); a run that fails before ``run.started`` (a busy ``--port``, a bad ``--task``)
    has its run directory removed when this invocation created it and it holds no events, is not verified, and its own error
    is repeated (exit status of the run); Ctrl-C stops the child, still verifies the ledger, prints the summary and exits 130."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=str(ROOT / "tasks" / "law-discovery"), help="task directory (default: law-discovery)")
    ap.add_argument("--out", type=Path, default=Path("runs") / "simulated", help="run directory to create (must not exist)")
    ap.add_argument("--plant", default="", help="red-team behaviors to inject, for example tamper,injection (see the module docstring "
                    "for the ones that take effect with the default simulator brain)")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--max-usd", type=float, default=2.0, help="budget cap; the simulated cost is a list-price estimate, nothing is billed")
    ap.add_argument("--max-experiments", type=int, default=12)
    ap.add_argument("--serve", action="store_true", help="serve the live dashboard while the run goes (loopback only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("rest", nargs="*", help="extra arguments for 'nightshift run' (put them after --)")
    args = ap.parse_args(argv)

    if args.plant:
        try:
            _parse_plant([args.plant], "meta")
        except CliError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return exc.code
    run_dir = args.out.resolve()
    if run_dir.exists():
        print(f"error: {run_dir} already exists; choose another --out", file=sys.stderr)
        return 2
    try:
        run_dir.parent.mkdir(parents=True, exist_ok=True)      # run_dir itself does not exist yet: whatever appears there is ours
    except OSError as exc:          # for example --out below a file, or somewhere that cannot be written: a usage error, no traceback
        print(f"error: cannot create the parent of {run_dir}: {exc.strerror or exc}; choose another --out", file=sys.stderr)
        return 2

    brain = SimBrain()
    with SimMuse(brain) as sim, tempfile.TemporaryDirectory(prefix="nightshift-sim-") as tmp:
        config = Path(tmp) / "nightshift.toml"
        config.write_text(f'[models.default]\nbase_url = "{sim.base_url}"\n', encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k not in ("META_API_KEY", "NIGHTSHIFT_BASE_URL", "NIGHTSHIFT_MODEL")}
        env["MODEL_API_KEY"] = sim.api_key          # the simulator's own dummy key, never a real one
        cmd = [sys.executable, "-m", "nightshift", "run", "--task", args.task, "--backend", "meta", "--run-dir", str(run_dir),
               "--config", str(config), "--max-usd", str(args.max_usd), "--max-experiments", str(args.max_experiments),
               "--parallel", str(args.parallel)]
        if args.plant:
            cmd += ["--plant", args.plant]
        if args.serve:
            cmd += ["--serve", "--port", str(args.port)]
        cmd += args.rest
        print("SIMULATED MODEL on loopback; nothing here is Meta model output.", flush=True)
        print("$ " + shlex.join(cmd), flush=True)
        stderr_lines: list[str] = []
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stderr=subprocess.PIPE, text=True, errors="replace")
        pump: Optional[threading.Thread] = None
        try:
            pump = _tee_stderr(proc, stderr_lines)
            code, interrupted = _wait(proc, run_dir)
        finally:
            if proc.poll() is None:        # any other way out of the wait: never leave the run (and its port) behind
                proc.kill()
                proc.wait()
            if pump is not None:
                pump.join(timeout=5)
        code = _exit_status(code)
        _, started = _ledger_state(run_dir)
        if not started:
            removed = _remove_if_empty(run_dir)
            why = "was interrupted before it started" if interrupted else f"failed before it started (exit {code})"
            print(f"error: the run {why}. Its own error: {_own_error(stderr_lines)}", file=sys.stderr)
            if removed:
                print(f"removed the empty run directory {run_dir}; the same --out can be used again", file=sys.stderr)
            elif run_dir.exists():
                print(f"left {run_dir} in place (it holds more than an empty ledger); remove it before reusing --out", file=sys.stderr)
            print(f"requests served by the simulator: {len(sim.requests)}", flush=True)
            return EXIT_INTERRUPTED if interrupted else (code or 1)
        try:
            verify = subprocess.run([sys.executable, "-m", "nightshift", "verify", str(run_dir)], cwd=ROOT, env=env).returncode
        except KeyboardInterrupt:
            interrupted, verify = True, 1
            print("\ninterrupted while verifying; run it again: python3 -m nightshift verify " + str(run_dir), file=sys.stderr)
        if interrupted:
            print("\nthe run was interrupted; the ledger above is what was recorded before it stopped", flush=True)
        print(f"\nrun directory: {run_dir}\nrequests served by the simulator: {len(sim.requests)}", flush=True)
    if interrupted:
        return EXIT_INTERRUPTED
    return code or verify


def _terminate(signum: int, frame: object) -> None:
    raise KeyboardInterrupt           # SIGTERM / SIGHUP take the Ctrl-C path: the child is stopped, never left running with its port


if __name__ == "__main__":
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _terminate)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:         # interrupted before or after the run itself (the run's own Ctrl-C is handled in main)
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(EXIT_INTERRUPTED)
