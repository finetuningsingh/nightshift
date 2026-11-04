#!/usr/bin/env bash
# The reference method that seeds v0: generate with the BASE model, filter with the pinned IFEvalG checkers, SFT the kept answers
# (Axolotl QLoRA), leave the adapter in $GLIMMER_OUT/adapter. Entrypoint of a recipe run by hostkit/run_job.sh:  bash run.sh
# Tunables (environment): RFT_N_PROMPTS (500), RFT_SAMPLES (4), RFT_MAX_TOKENS (6000), RFT_KEEP_PER_PROMPT (2), RFT_REASONING (high),
# RFT_INCLUDE_REASONING (1), RFT_PORT (8000). Records generation_s and train_s in $GLIMMER_OUT/timings.json, and the generation yield in
# $GLIMMER_OUT/rft_stats.json. UNVERIFIED on a GPU: this script has only run against stubs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTKIT="${HOSTKIT:-$(cd "$HERE/../.." && pwd)}"
# shellcheck source=../../lib/env.sh
source "$HOSTKIT/lib/env.sh"

TRAIN="${RFT_TRAIN:-$LAB_WORK/data/train.jsonl}"
PORT="${RFT_PORT:-8000}"
PIPE="$HERE/rft_pipeline.py"
CONFIG="${RFT_CONFIG:-$HERE/config.yaml}"
mkdir -p "$GLIMMER_OUT"

glimmer-vllm-up --port "$PORT"
export RFT_BASE_URL="${RFT_BASE_URL:-http://127.0.0.1:$PORT}"
python3 -I "$PIPE" generate --prompts "$TRAIN" --out "$GLIMMER_OUT/samples.jsonl"
glimmer-vllm-up --stop            # generation and training never share the GPU

python3 -I "$PIPE" filter --samples "$GLIMMER_OUT/samples.jsonl" --prompts "$TRAIN" --rewards "$REWARDS_PY" \
  --out "$GLIMMER_OUT/sft.jsonl" --stats "$GLIMMER_OUT/rft_stats.json"

T0=$(date +%s)
glimmer-axolotl-train "$CONFIG" --data "$GLIMMER_OUT/sft.jsonl"
python3 -I "$PIPE" timing --key train_s --seconds "$(( $(date +%s) - T0 ))"
