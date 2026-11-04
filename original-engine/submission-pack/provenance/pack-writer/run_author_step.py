#!/usr/bin/env python3
"""Run one authoring step with Muse Code (muse-wonder-full-1.0) in the isolated workspace and record it. Usage: run_author_step.py N PROMPT_FILE"""
import hashlib, json, os, subprocess, sys, time, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); WS = os.path.join(os.path.dirname(HERE), "ws")
n, prompt = sys.argv[1], os.path.abspath(sys.argv[2])
env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR") if k in os.environ}  # no *_API_KEY variable reaches the child
sha = hashlib.sha256(open(prompt, "rb").read()).hexdigest()
ver = subprocess.run(["muse", "--version"], capture_output=True, text=True, env=env).stdout.strip()
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
out = os.path.join(HERE, f"step{n}.events.jsonl"); err = os.path.join(HERE, f"step{n}.stderr.txt")
cmd = ["muse", "exec", "--json", "--model", "muse-wonder-full-1.0", "--effort", "high", "--workspace", WS, "--prompt-file", prompt,
       "--disable-approval", "--disable-web-tools", "--disable-request-user-input", "--no-foreign-personal-context", "--contributor-mode", "off", "--max-model-steps", "300"]
with open(out, "w") as o, open(err, "w") as e:
    rc = subprocess.run(cmd, cwd=WS, env=env, stdout=o, stderr=e).returncode
rec = {"step": n, "prompt_file": os.path.basename(prompt), "prompt_sha256": sha, "muse_version": ver, "model": "muse-wonder-full-1.0", "effort": "high",
       "host": "ankit-mac", "workspace": WS, "command": " ".join(cmd), "started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "exit_code": rc, "events_file": os.path.basename(out), "events_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
open(os.path.join(HERE, "steps.jsonl"), "a").write(json.dumps(rec) + "\n")
print(json.dumps(rec))
