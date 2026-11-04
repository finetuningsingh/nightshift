#!/usr/bin/env python3
"""Run a ``python -m nightshift.lab`` command for the two-loop experiment with the CLI-first routes and no API keys.

    run_lab.py LAB_ROOT <lab command and arguments...>

Sets: researcher runtime = Muse Code in a container on ax42 (``coral_muse``), proposer = Muse Code ``muse-wonder-full-1.0`` (``muse_cli``),
the CORAL home / Muse home / trial root on ax42, and reads the Thunder token from this machine's Thunder CLI login into the child's
environment only (never printed). Removes MUSE_API_KEY, BASETEN_API_KEY, META_API_KEY and LAB_META_API_KEY from the child's environment."""
import json, os, sys

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
env = {k: v for k, v in os.environ.items() if k not in ("MUSE_API_KEY", "BASETEN_API_KEY", "META_API_KEY", "LAB_META_API_KEY")}
env.update(
    LAB_RESEARCHER_RUNTIME="coral_muse", LAB_PROPOSER="muse_cli", LAB_PROPOSER_EFFORT=os.environ.get("LAB_PROPOSER_EFFORT", "high"),
    LAB_MUSE_HOME="/home/ankit/work/glimmer-lab/muse-home", LAB_CORAL_HOME="/home/ankit/work/glimmer-lab/b1-probe-20261103-195322/coral",
    LAB_TRIALS_ROOT=os.environ.get("LAB_TRIALS_ROOT", "/home/ankit/work/glimmer-lab/trials-2l"), NLTK_DATA="/Users/ankitsingh/glimmer-lab-live/lab/nltk",
    LAB_THUNDER_TOKEN=json.load(open(os.path.expanduser("~/.thunder/cli_config.json")))["token"],
)
os.execvpe(sys.executable, [sys.executable, "-m", "nightshift.lab", *sys.argv[1:]], env)
