#!/usr/bin/env bash
# Install the staged public files on the rental after verifying their sha256 pins:
#   $1 = a bundle directory (uploaded by the trainer) holding pins.sha256 ("<sha256>  <relative path>") and the pinned files.
# Installs ifevalg/* to $LAB_OPT/ifevalg, rewards.py to $LAB_OPT/rewards.py, optional weights.sha256 to $LAB_OPT, and writes the
# $LAB_OPT/bin/glimmer-* launchers. A file that is not in pins.sha256, or whose hash differs, stops the whole stage before anything is
# installed. Only public train-side files are ever staged: never selection, held-out or test data (the trainer enforces the file list).
set -euo pipefail
HOSTKIT="${HOSTKIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=lib/env.sh
source "$HOSTKIT/lib/env.sh"

BUNDLE="${1:?usage: stage.sh BUNDLE_DIR}"
[ -f "$BUNDLE/pins.sha256" ] || { echo "stage: $BUNDLE/pins.sha256 missing" >&2; exit 1; }
"$PY" -I "$HOSTKIT/lib/verify_sha256.py" "$BUNDLE" "$BUNDLE/pins.sha256" || { echo "stage: pin check failed; nothing installed" >&2; exit 1; }
# every regular file in the bundle must be pinned
"$PY" -I - "$BUNDLE" <<'PY' || { echo "stage: the bundle holds a file that is not pinned; nothing installed" >&2; exit 1; }
import os, sys
root = sys.argv[1]
pinned = {l.split("  ", 1)[1].rstrip("\n") for l in open(os.path.join(root, "pins.sha256")) if l.strip()}
extra = []
for d, _, fs in os.walk(root):
    for f in fs:
        rel = os.path.relpath(os.path.join(d, f), root)
        if rel != "pins.sha256" and rel not in pinned:
            extra.append(rel)
if extra:
    print("unpinned files:", extra, file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "$LAB_OPT/ifevalg" "$LAB_OPT/bin"
if [ -d "$BUNDLE/ifevalg" ]; then
  for f in "$BUNDLE"/ifevalg/*; do [ -f "$f" ] && install -m 0644 "$f" "$LAB_OPT/ifevalg/$(basename "$f")"; done
fi
[ -f "$BUNDLE/rewards.py" ] && install -m 0644 "$BUNDLE/rewards.py" "$LAB_OPT/rewards.py"
[ -f "$BUNDLE/weights.sha256" ] && install -m 0644 "$BUNDLE/weights.sha256" "$LAB_OPT/weights.sha256"
for name in glimmer-axolotl-train glimmer-vllm-up; do
  printf '#!/usr/bin/env bash\nexport HOSTKIT=%q\nexec bash "$HOSTKIT/bin/%s" "$@"\n' "$HOSTKIT" "$name" > "$LAB_OPT/bin/$name"
  chmod 0755 "$LAB_OPT/bin/$name"
done
echo "stage: ok"
