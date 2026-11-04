"""Validate (and optionally extract) a recipe tarball made by ``glimmer submit``. Runs on the rental with ``python3 -I``; standard library only.

    python3 -I check_recipe_tar.py RECIPE.tar.gz [--extract DEST] [--max-entries 5000] [--max-unpacked 209715200] [--entrypoint PATH]

Rules (docs/harness-design.md, File formats): a gzip tar; regular files and directories only; relative paths with no ``..``, no leading
``/`` and no ``.git`` component; no symbolic links, hard links, devices or fifos; at most 5,000 entries and 200 MB unpacked. With
``--extract`` the files are written by this script (never by ``tar -x``), each created with O_EXCL under DEST, so nothing in the archive
can choose a path outside DEST. Exit 0 = valid, 1 = refused (reason on stderr).
"""
import argparse
import os
import posixpath
import sys
import tarfile

MAX_ENTRIES = 5000
MAX_UNPACKED = 200 * 1024 * 1024


def refuse(msg):
    print(f"recipe refused: {msg}", file=sys.stderr)
    sys.exit(1)


def check_name(name):
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        refuse(f"bad path {name!r}")
    norm = posixpath.normpath(name)
    parts = norm.split("/")
    if norm.startswith("..") or ".." in parts or ".git" in parts or norm == ".":
        refuse(f"path {name!r} is not an ordinary relative path")
    return norm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball")
    ap.add_argument("--extract")
    ap.add_argument("--max-entries", type=int, default=MAX_ENTRIES)
    ap.add_argument("--max-unpacked", type=int, default=MAX_UNPACKED)
    ap.add_argument("--entrypoint", help="a path inside the archive that must exist as a regular file")
    args = ap.parse_args(argv)
    if os.path.islink(args.tarball) or not os.path.isfile(args.tarball):
        refuse("the tarball is not a regular file")
    seen, files, total, n = set(), set(), 0, 0
    members = []
    try:
        with tarfile.open(args.tarball, mode="r:gz") as tar:
            for m in tar:
                n += 1
                if n > args.max_entries:
                    refuse(f"more than {args.max_entries} entries")
                if m.issym() or m.islnk() or m.isdev() or m.isfifo():
                    refuse(f"{m.name!r} is a link or a special file")
                if not (m.isreg() or m.isdir()):
                    refuse(f"{m.name!r} has an unsupported type")
                name = check_name(m.name)
                if name in seen:
                    refuse(f"duplicate entry {name!r}")
                seen.add(name)
                if m.isreg():
                    total += m.size
                    if total > args.max_unpacked:
                        refuse(f"more than {args.max_unpacked} bytes unpacked")
                    files.add(name)
                    if args.extract:
                        members.append((name, m.mode & 0o111 != 0, tar.extractfile(m).read() if m.size else b""))
                elif args.extract:
                    members.append((name, None, None))
    except (tarfile.TarError, OSError, EOFError) as exc:
        refuse(f"not a readable gzip tar ({exc})")
    if args.entrypoint:
        ep = posixpath.normpath(args.entrypoint)
        if ep not in files:
            refuse(f"the entrypoint {args.entrypoint!r} is not a regular file in the archive")
    if args.extract:
        dest = os.path.abspath(args.extract)
        os.makedirs(dest, exist_ok=True)
        for name, exe, data in members:
            target = os.path.join(dest, name)
            if os.path.commonpath([dest, os.path.abspath(target)]) != dest:
                refuse(f"{name!r} escapes the destination")
            if exe is None:
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o755 if exe else 0o644)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
    print(f"recipe ok: {len(files)} files, {total} bytes")


if __name__ == "__main__":
    main()
