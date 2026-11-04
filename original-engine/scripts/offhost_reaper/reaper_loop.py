#!/usr/bin/env python3
"""Independent off-host money stop (runs on a worker, NOT the control host). Every INTERVAL seconds it runs
``python -m nightshift.lab.gpu_watchdog reap --no-record`` against a replicated copy of the rental ledger, with this machine's own Thunder
login (read from ~/.thunder/cli_config.json into the child's environment only; never printed or logged). The reaper deletes only ledger
rentals that are past their cap plus the grace period, by their recorded handle and ownership tag; a stale copy or an absent instance is a
no-op. Usage:  reaper_loop.py REPO_DIR REPLICA_LEDGER [INTERVAL_SECONDS] [STOP_EPOCH]"""
import json, os, subprocess, sys, time

repo, ledger = sys.argv[1], sys.argv[2]
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
stop_epoch = float(sys.argv[4]) if len(sys.argv) > 4 else float("inf")
while time.time() < stop_epoch:
    env = dict(os.environ)
    env["THUNDER_API_KEY"] = json.load(open(os.path.expanduser("~/.thunder/cli_config.json")))["token"]
    if os.path.isfile(ledger):
        r = subprocess.run([sys.executable, "-m", "nightshift.lab.gpu_watchdog", "reap", "--ledger", ledger, "--provider", "thunder",
                            "--token-env", "THUNDER_API_KEY", "--no-record"], cwd=repo, env=env, capture_output=True, text=True)
        print(time.strftime("%FT%TZ", time.gmtime()), "reap exit", r.returncode, (r.stdout.strip() or "-")[:400], flush=True)
        if r.returncode not in (0, 1):
            print("reap stderr:", r.stderr.strip()[-300:], flush=True)
    else:
        print(time.strftime("%FT%TZ", time.gmtime()), "no replica yet", flush=True)
    time.sleep(interval)
