"""Start a job's entrypoint from ``job.json`` without any shell parsing: ``python3 -I job_entry.py JOB.json``.
``job.json`` is ``{"entrypoint": [argv...], "deadline_epoch": N}``. The entrypoint runs through ``run_with_timeout`` (own session, killed at the
deadline, leftovers killed on exit). Exit status as ``run_with_timeout``; 2 if job.json is malformed."""
import json
import os
import sys

sys.dont_write_bytecode = True  # the hostkit tree stays clean (python -I ignores PYTHONDONTWRITEBYTECODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_with_timeout  # noqa: E402


def main(argv):
    try:
        doc = json.load(open(argv[1]))
        entry, deadline = doc["entrypoint"], int(doc["deadline_epoch"])
        if not (isinstance(entry, list) and entry and all(isinstance(a, str) and a for a in entry)):
            raise ValueError("entrypoint must be a non-empty list of non-empty strings")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"job.json is malformed: {exc}", file=sys.stderr)
        return 2
    return run_with_timeout.main(["--deadline-epoch", str(deadline), "--", *entry])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
