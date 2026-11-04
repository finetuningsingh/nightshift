"""Canonical listing hash of an adapter directory (same value as ``contracts.canonical_listing_sha256``): sha256 over one line
``<sha256> <size> <name>\\n`` per file, sorted by name, for ``adapter_config.json`` and ``adapter_model.safetensors`` only.
``python3 -I adapter_listing.py DIR`` prints it; exit 1 if a file is missing or is not a regular file."""
import hashlib
import os
import stat
import sys

FILES = ("adapter_config.json", "adapter_model.safetensors")


def listing_sha256(directory):
    lines = []
    for name in sorted(FILES):
        p = os.path.join(directory, name)
        st = os.lstat(p)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
            raise SystemExit(f"{p}: not a regular, unlinked file")
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()} {st.st_size} {name}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


if __name__ == "__main__":
    print(listing_sha256(sys.argv[1]))
