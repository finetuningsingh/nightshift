#!/usr/bin/env bash
# Serve Glimmer (bf16) with zero or more LoRA adapters behind an OpenAI-compatible endpoint (vLLM 0.32.0; UNVERIFIED end to end: nobody has
# published a Glimmer + LoRA vLLM run). Foreground; the caller runs it detached.
#   serve.sh --port P [--served-name muse-glimmer] [--max-model-len 8300] [--adapter NAME=DIR=LISTING_SHA256]...
# Each adapter directory must hash (canonical file listing) to LISTING_SHA256 or the server does not start. Weights are checked against
# $LAB_OPT/weights.sha256 when it exists. LAB_DRY_RUN=1 prints the command instead of running it (tests).
set -euo pipefail
HOSTKIT="${HOSTKIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=lib/env.sh
source "$HOSTKIT/lib/env.sh"

PORT=8000; NAME=muse-glimmer; MAXLEN=8300; ADAPTERS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=$2; shift 2;;
    --served-name) NAME=$2; shift 2;;
    --max-model-len) MAXLEN=$2; shift 2;;
    --adapter) ADAPTERS+=("$2"); shift 2;;
    *) echo "serve.sh: unknown argument $1" >&2; exit 2;;
  esac
done
[[ "$PORT" =~ ^[0-9]+$ && "$MAXLEN" =~ ^[0-9]+$ && "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "serve.sh: bad port, length or name" >&2; exit 2; }

LORA=()
for spec in ${ADAPTERS[@]+"${ADAPTERS[@]}"}; do
  IFS='=' read -r aname adir asha <<< "$spec"
  [[ "$aname" =~ ^[A-Za-z0-9._-]+$ && "$asha" =~ ^[0-9a-f]{64}$ && -n "$adir" ]] || { echo "serve.sh: bad --adapter $spec" >&2; exit 2; }
  actual=$("$PY" -I "$HOSTKIT/lib/adapter_listing.py" "$adir")
  [ "$actual" = "$asha" ] || { echo "serve.sh: adapter $aname at $adir hashes to $actual, expected $asha; refusing to serve it" >&2; exit 3; }
  LORA+=("$aname=$adir")
done

if [ -f "$LAB_OPT/weights.sha256" ] && [ "${LAB_SKIP_WEIGHT_CHECK:-0}" != "1" ]; then
  "$PY" -I "$HOSTKIT/lib/verify_sha256.py" "$GLIMMER_BASE" "$LAB_OPT/weights.sha256" || { echo "serve.sh: the base weights do not match their pins" >&2; exit 4; }
fi

CMD=("$VLLM_VENV/bin/vllm" serve "$GLIMMER_BASE" --served-model-name "$NAME" --port "$PORT" --host 127.0.0.1
     --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len "$MAXLEN" --enforce-eager
     --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer --generation-config auto)
if [ ${#LORA[@]} -gt 0 ]; then
  CMD+=(--enable-lora --max-loras "${#LORA[@]}" --max-lora-rank 32 --lora-modules "${LORA[@]}")
fi
if [ "${LAB_DRY_RUN:-0}" = "1" ]; then printf '%s\n' "${CMD[@]}"; exit 0; fi
mkdir -p "$LAB_WORK"
echo $$ > "$LAB_WORK/vllm.pid"
exec "${CMD[@]}"
