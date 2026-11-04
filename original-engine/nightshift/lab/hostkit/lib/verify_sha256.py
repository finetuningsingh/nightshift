"""Verify files against a pin list: ``python3 -I verify_sha256.py ROOT PINS`` where PINS has lines ``<sha256>  <relative path>``.
Exit 1 and print every mismatch or missing file; a path that is absolute or contains ``..`` is refused. Portable (no sha256sum)."""
import hashlib
import os
import sys


def main(root, pins_file):
    bad = []
    with open(pins_file) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            digest, sep, rel = line.partition("  ")
            if not sep or len(digest) != 64 or not rel or rel.startswith("/") or ".." in rel.split("/"):
                bad.append(f"line {n}: malformed or unsafe pin {line[:80]!r}")
                continue
            path = os.path.join(root, rel)
            if os.path.islink(path) or not os.path.isfile(path):
                bad.append(f"{rel}: missing or not a regular file")
                continue
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                bad.append(f"{rel}: sha256 {h.hexdigest()} != pinned {digest}")
    for b in bad:
        print(b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
