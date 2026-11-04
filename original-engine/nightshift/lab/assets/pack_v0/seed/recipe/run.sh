#!/usr/bin/env bash
# Starting recipe: sample from the base model, keep answers that pass every constraint, fine-tune on the kept answers.
# Runs on the rented GPU as `bash run.sh` with /work/data/train.jsonl present. Leaves the adapter in /work/out/adapter.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${GLIMMER_OUT:-/work/out}"
TRAIN="${TRAIN_JSONL:-/work/data/train.jsonl}"
N_PROMPTS="${N_PROMPTS:-400}"
SAMPLES="${SAMPLES:-4}"
MAX_TOKENS="${MAX_TOKENS:-3072}"
PER_PROMPT="${PER_PROMPT:-1}"
PORT="${VLLM_PORT:-8000}"
mkdir -p "$OUT"

server_pid=""
stop_server() {
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT

# 1. serve the base model (the helper blocks while serving, so run it in the background and wait until it answers)
glimmer-vllm-up --port "$PORT" >"$OUT/vllm.log" 2>&1 &
server_pid=$!
python3 generate.py --wait-only --base-url "http://127.0.0.1:$PORT/v1" --wait-seconds 1500

# 2. sample
python3 generate.py --data "$TRAIN" --base-url "http://127.0.0.1:$PORT/v1" --out "$OUT/samples.jsonl" \
  --n-prompts "$N_PROMPTS" --samples "$SAMPLES" --max-tokens "$MAX_TOKENS"

# the GPU is needed for training: stop the server first (serving and training never share the card)
stop_server

# 3. verify and keep
python3 build_sft.py --samples "$OUT/samples.jsonl" --out "$OUT/sft.jsonl" --per-prompt "$PER_PROMPT"

# 4. fine-tune (writes the adapter to /work/out/adapter and the processed dataset to /work/out/train_data.jsonl)
glimmer-axolotl-train "${GLIMMER_CONFIG_TEMPLATE:?GLIMMER_CONFIG_TEMPLATE is not set}" --data "$OUT/sft.jsonl"
