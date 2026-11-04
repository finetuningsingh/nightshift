#!/usr/bin/env bash
# Starter recipe: self-distillation with verifier filtering (RFT) for Muse Glimmer.
# Stages: preflight -> serve base model -> sample -> stop server -> verify/filter
# -> QLoRA train -> check adapter -> timings.json. See README.md for every knob.
set -Eeuo pipefail
cd "$(dirname "$0")"

OUT="${GLIMMER_OUT:-/work/out}"
DATA_DIR="${GLIMMER_DATA_DIR:-/work/data}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.jsonl}"   # dev.jsonl is never trained on
SMOKE="${SMOKE:-0}"

# --- tunables (all overridable by environment; defaults in README) ----------
PROMPT_LIMIT="${PROMPT_LIMIT:-250}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
CONCURRENCY="${CONCURRENCY:-128}"
MAX_TOKENS="${MAX_TOKENS:-6000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"                       # 0 = do not send top_k
REASONING_STRENGTH="${REASONING_STRENGTH:-high}"
GEN_TIMEBOX_SECONDS="${GEN_TIMEBOX_SECONDS:-2700}"
GEN_GRACE_SECONDS="${GEN_GRACE_SECONDS:-180}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1200}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8300}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
MODEL_NAME="${MODEL_NAME:-muse-glimmer}"
SEED="${SEED:-42}"
RESUME="${RESUME:-0}"                     # 1 = reuse generations already in $OUT
MAX_KEEP_PER_PROMPT="${MAX_KEEP_PER_PROMPT:-2}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-800}"
MIN_EXAMPLES="${MIN_EXAMPLES:-20}"
MAX_SFT_CHARS="${MAX_SFT_CHARS:-14000}"   # approx. guard for sequence_len=4096
KEEP_REASONING="${KEEP_REASONING:-0}"     # 1 = add reasoning_content to SFT targets

if [ "$SMOKE" = "1" ]; then
  # Whole pipeline, minimum spend: <=4 prompts, 1 sample each, small token
  # limit, low reasoning so an answer fits in the limit, 1 training step.
  PROMPT_LIMIT="${SMOKE_PROMPT_LIMIT:-4}"
  SAMPLES_PER_PROMPT=1
  MAX_TOKENS="${SMOKE_MAX_TOKENS:-1024}"
  REASONING_STRENGTH="${SMOKE_REASONING_STRENGTH:-low}"
  GEN_TIMEBOX_SECONDS="${SMOKE_GEN_TIMEBOX_SECONDS:-600}"
  MIN_EXAMPLES=1
  MAX_TRAIN_EXAMPLES=4
fi

GEN_JSONL="$OUT/generations.jsonl"
SAMPLES_JSONL="$OUT/glimmer_samples.jsonl"   # origin authority the harness checks (fixed name)
SFT_JSONL="$OUT/sft.jsonl"
GEN_STATS="$OUT/gen_stats.json"
SFT_STATS="$OUT/sft_stats.json"
SERVER_STARTED=0
T_START=$(date +%s)
T_GEN=0; T_FILTER=0; T_TRAIN=0

banner() { echo; echo "=== [$(date -u +%FT%TZ)] $* ==="; }
stop_server() {
  if [ "$SERVER_STARTED" = "1" ]; then
    SERVER_STARTED=0
    echo "[run.sh] stopping vLLM server (frees the GPU)..."
    glimmer-vllm-up --stop || echo "[run.sh] WARNING: glimmer-vllm-up --stop failed; GPU may still be held" >&2
  fi
}
trap stop_server EXIT   # serving and training must never share the GPU; stop on every exit path
die() { echo "FATAL: $*" >&2; exit 1; }

banner "Stage 0: preflight (no GPU spent yet)"
mkdir -p "$OUT"
[ -n "${GLIMMER_BASE:-}" ] || die "GLIMMER_BASE is not set (the harness must provide the base model path)."
[ -e "$GLIMMER_BASE" ] || die "GLIMMER_BASE=$GLIMMER_BASE does not exist on this machine."
[ -n "${REWARDS_PY:-}" ] || die "REWARDS_PY is not set (path of the verifier module is required)."
[ -f "$REWARDS_PY" ] || die "REWARDS_PY=$REWARDS_PY is not a file."
[ -f "$TRAIN_FILE" ] || die "training file not found: $TRAIN_FILE (set TRAIN_FILE or GLIMMER_DATA_DIR)."
[ -f config.yaml ] || die "config.yaml missing next to run.sh."
command -v glimmer-vllm-up >/dev/null || die "glimmer-vllm-up not on PATH (expected in /opt/lab/bin)."
command -v glimmer-axolotl-train >/dev/null || die "glimmer-axolotl-train not on PATH (expected in /opt/lab/bin)."
command -v python3 >/dev/null || die "python3 not found."
python3 build_sft.py --preflight-only --rewards-py "$REWARDS_PY" \
  || die "verifier preflight failed (see message above). Not starting the server."
echo "[run.sh] SMOKE=$SMOKE OUT=$OUT TRAIN_FILE=$TRAIN_FILE PROMPT_LIMIT=$PROMPT_LIMIT SAMPLES=$SAMPLES_PER_PROMPT MAX_TOKENS=$MAX_TOKENS"

banner "Stage 1: start vLLM server for the base model"
SERVER_STARTED=1
BASE_URL="$(glimmer-vllm-up --port "$VLLM_PORT" --max-model-len "$MAX_MODEL_LEN" --wait "$VLLM_WAIT_SECONDS" | tail -n 1 | tr -d '[:space:]')" \
  || die "glimmer-vllm-up failed to start the server. See $OUT/vllm.log."
case "$BASE_URL" in http://*|https://*) ;; *) die "glimmer-vllm-up did not print a base URL (got: '$BASE_URL'). See $OUT/vllm.log.";; esac
echo "[run.sh] server base URL: $BASE_URL"

banner "Stage 2: sample answers from the base model"
T0=$(date +%s)
RESUME_ARG=(); [ "$RESUME" = "1" ] && RESUME_ARG=(--resume)
python3 generate.py --base-url "$BASE_URL" --train-file "$TRAIN_FILE" \
  --out "$GEN_JSONL" --samples-out "$SAMPLES_JSONL" --stats "$GEN_STATS" --model "$MODEL_NAME" \
  --prompt-limit "$PROMPT_LIMIT" --samples "$SAMPLES_PER_PROMPT" \
  --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
  --reasoning-strength "$REASONING_STRENGTH" \
  --timebox-seconds "$GEN_TIMEBOX_SECONDS" --grace-seconds "$GEN_GRACE_SECONDS" \
  --request-timeout "$REQUEST_TIMEOUT_SECONDS" --seed "$SEED" \
  --exit-hard-on-abandon ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
  || die "generation failed (see message above). Partial samples, if any, are in $GEN_JSONL."
T_GEN=$(( $(date +%s) - T0 ))

banner "Stage 3: stop vLLM server before training"
stop_server

banner "Stage 4: verifier-filter samples into an SFT file + local origin check"
T0=$(date +%s)
KEEP_ARG=(); [ "$KEEP_REASONING" = "1" ] && KEEP_ARG=(--keep-reasoning)
python3 build_sft.py --generations "$GEN_JSONL" --out "$SFT_JSONL" --stats "$SFT_STATS" \
  --train-file "$TRAIN_FILE" --samples "$SAMPLES_JSONL" \
  --rewards-py "$REWARDS_PY" --max-keep-per-prompt "$MAX_KEEP_PER_PROMPT" \
  --max-train-examples "$MAX_TRAIN_EXAMPLES" --min-examples "$MIN_EXAMPLES" \
  --max-chars "$MAX_SFT_CHARS" --seed "$SEED" ${KEEP_ARG[@]+"${KEEP_ARG[@]}"} \
  || die "filtering failed / yield too low (see message above). Not training."
# Same origin rules the harness enforces inside glimmer-axolotl-train, checked
# now so a violation stops the run before any training starts.
python3 origin_check.py --sft "$SFT_JSONL" --train "$TRAIN_FILE" --samples "$SAMPLES_JSONL" \
  || die "local origin check failed (see message above). The harness would refuse this data; not training."
T_FILTER=$(( $(date +%s) - T0 ))

banner "Stage 5: QLoRA training"
TRAIN_CONFIG=config.yaml
if [ "$SMOKE" = "1" ]; then
  TRAIN_CONFIG="$OUT/config.smoke.yaml"
  grep -q '^# SMOKE_MAX_STEPS_ANCHOR' config.yaml || die "SMOKE_MAX_STEPS anchor missing in config.yaml; cannot build the 1-step smoke config."
  sed 's/^# SMOKE_MAX_STEPS_ANCHOR.*/max_steps: 1/' config.yaml > "$TRAIN_CONFIG"
  echo "[run.sh] smoke config (max_steps: 1): $TRAIN_CONFIG"
fi
T0=$(date +%s)
glimmer-axolotl-train "$TRAIN_CONFIG" --data "$SFT_JSONL" \
  || die "training command failed (glimmer-axolotl-train $TRAIN_CONFIG --data $SFT_JSONL). SFT data kept at $SFT_JSONL."
T_TRAIN=$(( $(date +%s) - T0 ))

banner "Stage 6: check the adapter is where the harness expects it"
if [ ! -f "$OUT/adapter/adapter_config.json" ] || [ ! -f "$OUT/adapter/adapter_model.safetensors" ]; then
  FOUND="$(find "$OUT" -name adapter_model.safetensors -not -path "$OUT/adapter/*" 2>/dev/null | head -n 1 || true)"
  if [ -n "$FOUND" ] && [ -f "$(dirname "$FOUND")/adapter_config.json" ]; then
    echo "[run.sh] adapter found at $(dirname "$FOUND"); copying to $OUT/adapter/"
    mkdir -p "$OUT/adapter"
    cp "$(dirname "$FOUND")/adapter_config.json" "$(dirname "$FOUND")/adapter_model.safetensors" "$OUT/adapter/"
  fi
fi
[ -f "$OUT/adapter/adapter_config.json" ] && [ -f "$OUT/adapter/adapter_model.safetensors" ] \
  || die "training finished but no adapter at $OUT/adapter/ (adapter_config.json + adapter_model.safetensors). Contents of $OUT: $(ls -R "$OUT" | head -n 50)"
echo "[run.sh] adapter OK: $OUT/adapter/ ($(du -h "$OUT/adapter/adapter_model.safetensors" | cut -f1))"

banner "Stage 7: timing/yield summary"
python3 summarize.py --gen-stats "$GEN_STATS" --sft-stats "$SFT_STATS" --out "$OUT/timings.json" \
  --gen-seconds "$T_GEN" --filter-seconds "$T_FILTER" --train-seconds "$T_TRAIN" \
  --total-seconds "$(( $(date +%s) - T_START ))" --smoke "$SMOKE" \
  || echo "[run.sh] WARNING: summarize.py failed; adapter is still valid, timings.json missing." >&2
banner "DONE in $(( $(date +%s) - T_START ))s - adapter: $OUT/adapter/ summary: $OUT/timings.json"
