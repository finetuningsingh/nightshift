"""Run a command with a wall-clock deadline and clean up everything it started. Portable (macOS has no ``timeout``). ``python3 -I``.

    python3 -I run_with_timeout.py --deadline-epoch E [--grace 20] -- CMD ARG...

The command runs in its own session. At the deadline its process group gets SIGTERM, then SIGKILL after ``--grace`` seconds. After the
command exits, anything it left running in that process group is killed too (a recipe that starts a server and forgets it must not
keep the GPU). Exit status: the command's, 124 on a deadline kill, 127 if it cannot be started.
"""
import argparse
import os
import signal
import subprocess
import sys
import time


def kill_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--grace", type=float, default=20.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        print("no command", file=sys.stderr)
        return 2
    try:
        proc = subprocess.Popen(cmd, start_new_session=True)
    except OSError as exc:
        print(f"cannot start {cmd[0]}: {exc}", file=sys.stderr)
        return 127
    pgid, timed_out = proc.pid, False
    while True:
        try:
            rc = proc.wait(timeout=max(0.05, min(1.0, a.deadline_epoch - time.time())))
            break
        except subprocess.TimeoutExpired:
            if time.time() >= a.deadline_epoch:
                timed_out = True
                kill_group(pgid, signal.SIGTERM)
                try:
                    rc = proc.wait(timeout=a.grace)
                except subprocess.TimeoutExpired:
                    kill_group(pgid, signal.SIGKILL)
                    rc = proc.wait()
                break
    kill_group(pgid, signal.SIGTERM)  # leftovers of the recipe
    time.sleep(0.2)
    kill_group(pgid, signal.SIGKILL)
    return 124 if timed_out else (rc if rc >= 0 else 128 - rc)


if __name__ == "__main__":
    sys.exit(main())
