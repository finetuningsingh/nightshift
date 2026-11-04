#!/usr/bin/env bash
# Fetch and pin the third-party verifier code the lab scorer uses. Nothing is vendored into the repository (decision D10).
#
#   scripts/lab_setup_verifiers.sh [--verify-only] [--nltk-data DIR] <third_party_dir>
#
# <third_party_dir> is <lab>/third_party (LabLayout.third_party_dir). It ends up holding
#   IFBench/        allenai/IFBench at the commit pinned in scripts/lab_pins.json (verifies the test split; its README says the package
#                   bundles the test file, so this directory lives on the control host only)
#   open-instruct/  allenai/open-instruct, sparse checkout of open_instruct/IFEvalG only, at the pinned commit (dev and selection verifier)
# then the sha256 of every file the scorer executes is checked against scripts/lab_pins.json. The scorer repeats that check before it
# imports anything. No downloaded code is run by this script.
#
# Python dependencies and NLTK data are NOT installed unless you ask: print the commands with --verify-only, or pass --nltk-data DIR to fetch
# the NLTK data (needs network). Dependencies: python3 -m pip install absl-py langdetect nltk immutabledict
#
# Environment: IFBENCH_URL and OPEN_INSTRUCT_URL override the clone sources (a local path works, for mirrors and for tests).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pins="$here/lab_pins.json"
verify_only=0
nltk_dir=""
dest=""
while [ $# -gt 0 ]; do
  case "$1" in
    --verify-only) verify_only=1 ;;
    --nltk-data) shift; nltk_dir="${1:?--nltk-data needs a directory}" ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) dest="$1" ;;
  esac
  shift
done
[ -n "$dest" ] || { echo "usage: $0 [--verify-only] [--nltk-data DIR] <third_party_dir>" >&2; exit 2; }
[ -f "$pins" ] || { echo "missing $pins" >&2; exit 2; }

pin() { python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]][sys.argv[3]])' "$pins" "$1" "$2"; }
ifbench_commit="$(pin ifbench commit)"
oi_commit="$(pin open_instruct commit)"
ifbench_url="${IFBENCH_URL:-$(pin ifbench repo)}"
oi_url="${OPEN_INSTRUCT_URL:-$(pin open_instruct repo)}"

at_commit() { [ -d "$1/.git" ] || [ -f "$1/.git" ] && [ "$(git -C "$1" rev-parse HEAD 2>/dev/null || true)" = "$2" ]; }

if [ "$verify_only" -eq 0 ]; then
  mkdir -p "$dest"
  if ! at_commit "$dest/IFBench" "$ifbench_commit"; then
    [ ! -e "$dest/IFBench" ] || { echo "$dest/IFBench exists but is not at $ifbench_commit; move it away" >&2; exit 1; }
    git clone --quiet "$ifbench_url" "$dest/IFBench"
    git -C "$dest/IFBench" checkout --quiet --detach "$ifbench_commit"
  fi
  if ! at_commit "$dest/open-instruct" "$oi_commit"; then
    [ ! -e "$dest/open-instruct" ] || { echo "$dest/open-instruct exists but is not at $oi_commit; move it away" >&2; exit 1; }
    git clone --quiet --filter=blob:none --no-checkout "$oi_url" "$dest/open-instruct"
    git -C "$dest/open-instruct" sparse-checkout set open_instruct/IFEvalG
    git -C "$dest/open-instruct" checkout --quiet --detach "$oi_commit"
  fi
fi

# verify every pinned file by content (reads files, runs nothing from them)
python3 -I - "$pins" "$dest" <<'PY'
import hashlib, json, os, stat, sys
pins = json.load(open(sys.argv[1]))
dest = sys.argv[2]
bad = []
def check(base, table):
    for rel, want in sorted(table.items()):
        p = os.path.join(base, rel)
        try:
            st = os.lstat(p)
            if not stat.S_ISREG(st.st_mode):
                bad.append(f"{p}: not a regular file"); continue
            got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        except OSError as e:
            bad.append(f"{p}: {e.strerror}"); continue
        if got != want:
            bad.append(f"{p}: sha256 {got} != pinned {want}")
check(os.path.join(dest, "IFBench"), pins["ifbench"]["file_sha256"])
check(os.path.join(dest, "open-instruct", pins["open_instruct"]["ifevalg_dir"]), pins["open_instruct"]["ifevalg_sha256"])
if bad:
    print("verifier pins NOT satisfied:", *bad, sep="\n  ", file=sys.stderr)
    sys.exit(1)
print("verifier pins ok: IFBench %s, open-instruct %s" % (pins["ifbench"]["commit"][:12], pins["open_instruct"]["commit"][:12]))
PY

if [ -n "$nltk_dir" ]; then
  python3 -m nltk.downloader -d "$nltk_dir" $(python3 -I -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["nltk_resources"]))' "$pins")
  echo "NLTK data in $nltk_dir: export NLTK_DATA=$nltk_dir"
else
  echo "next: python3 -m pip install absl-py langdetect nltk immutabledict"
  echo "      python3 -m nltk.downloader -d \$NLTK_DATA $(python3 -I -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["nltk_resources"]))' "$pins")"
fi
