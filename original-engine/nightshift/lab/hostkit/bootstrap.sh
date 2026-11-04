#!/usr/bin/env bash
# One-time setup of a fresh rental: the Axolotl 0.19.0 environment, the vLLM 0.32.0 environment and the pinned Glimmer weights.
# Idempotent (a marker per step). UNVERIFIED on a GPU: nothing here has run; the first rental measures it (docs/harness-design.md,
# "Real and tested offline, versus GPU or Meta key only"). Secrets: an optional HF_TOKEN in the environment is used for the download and
# never written to disk by this script. Writes $LAB_OPT/bootstrap.done (JSON with the seconds each step took).
set -euo pipefail
HOSTKIT="${HOSTKIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=lib/env.sh
source "$HOSTKIT/lib/env.sh"

mkdir -p "$LAB_OPT" "$LAB_WORK" "$LAB_MODELS"
T_TOTAL=$(date +%s)
step() { # step NAME COMMAND...: run once, remember it, time it (seconds go to .bootstrap.NAME.secs)
  local name=$1; shift
  if [ -f "$LAB_OPT/.bootstrap.$name.done" ]; then echo "[bootstrap] $name: already done"; return 0; fi
  echo "[bootstrap] $name: start"
  local t0; t0=$(date +%s)
  "$@"
  echo $(( $(date +%s) - t0 )) > "$LAB_OPT/.bootstrap.$name.secs"
  touch "$LAB_OPT/.bootstrap.$name.done"
  echo "[bootstrap] $name: done in $(cat "$LAB_OPT/.bootstrap.$name.secs") s"
}
secs() { cat "$LAB_OPT/.bootstrap.$1.secs" 2>/dev/null || echo 0; }

check_machine() {
  command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: not a GPU machine" >&2; return 1; }
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | tee "$LAB_OPT/gpu.txt"
  df -h "$LAB_OPT" "$LAB_MODELS" > "$LAB_OPT/disk.txt"
  command -v python3 >/dev/null || { echo "python3 missing" >&2; return 1; }
}

install_uv() { command -v uv >/dev/null || "$PY" -m pip install --quiet uv; }

arch_list() { # compute capability of GPU 0 as the CUDA arch list, e.g. 8.0 (A100) or 9.0 (H100)
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' '
}

make_axolotl_env() {
  uv venv --python 3.12 "$AXOLOTL_VENV"
  export UV_TORCH_BACKEND=cu130
  uv pip install --python "$AXOLOTL_VENV/bin/python" "torch==2.12.1" "torchvision==0.27.1"
  uv pip install --python "$AXOLOTL_VENV/bin/python" "axolotl==0.19.0"
  local arch; arch=$(arch_list)
  # LIVE 2026-11-02: a Thunder A100 has no sudo and no CUDA toolkit (no nvcc), so flash-attn cannot be built there. The configs use sdpa attention, which
  # needs no build; flash-attn is installed only where a compiler exists.
  if [ -x "${CUDA_HOME:-/usr/local/cuda-13.0}/bin/nvcc" ]; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}" CC=/usr/bin/gcc CXX=/usr/bin/g++ TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS="${MAX_JOBS:-4}" NVCC_THREADS=2 \
    uv pip install --python "$AXOLOTL_VENV/bin/python" --no-build-isolation "flash-attn==2.8.3"
  else
    echo "[bootstrap] no nvcc: skipping flash-attn==2.8.3 (the configs use sdpa attention)"
  fi
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}" CC=/usr/bin/gcc CXX=/usr/bin/g++ TORCH_CUDA_ARCH_LIST="$arch" \
    uv pip install --python "$AXOLOTL_VENV/bin/python" --no-build-isolation \
    "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@4dfa52212d0552805aec5a6a2176c0abbea277d2"
  uv pip check --python "$AXOLOTL_VENV/bin/python"
}

make_verify_env() { # the pinned IFEvalG checkers need nltk, langdetect, immutabledict, absl-py and the NLTK data; a recipe's plain `python3` must be able to run them
  uv venv --python 3.12 "$VERIFY_VENV"
  uv pip install --python "$VERIFY_VENV/bin/python" "nltk==3.10.4" "langdetect==1.0.9" "immutabledict==4.3.1" "absl-py==2.5.1"
  umask 022   # nltk refuses a download directory that is, or sits under, a group- or world-writable directory
  mkdir -p "$NLTK_DATA"
  chmod go-w "$LAB_OPT" "$NLTK_DATA" "$(dirname "$LAB_OPT")" 2>/dev/null || true
  "$VERIFY_VENV/bin/python" -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab stopwords averaged_perceptron_tagger_eng
  mkdir -p "$LAB_OPT/bin"
  printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$VERIFY_VENV/bin/python" > "$LAB_OPT/bin/python3"
  chmod 0755 "$LAB_OPT/bin/python3"
}

make_vllm_env() {
  uv venv --python 3.12 "$VLLM_VENV"
  uv pip install --python "$VLLM_VENV/bin/python" "vllm==0.32.0" "huggingface_hub"
}

download_weights() {
  if [ -f "$GLIMMER_BASE/model.safetensors.index.json" ] && [ -f "$GLIMMER_BASE/.download-complete" ]; then return 0; fi
  "$VLLM_VENV/bin/python" - "$GLIMMER_REPO" "$GLIMMER_REVISION" "$GLIMMER_BASE" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1:4]
snapshot_download(repo_id=repo, revision=rev, local_dir=dest, token=os.environ.get("HF_TOKEN") or None)
open(os.path.join(dest, ".download-complete"), "w").write(rev + "\n")
PY
}

verify_weights() { # the shard hashes are staged by the trainer as $LAB_OPT/weights.sha256 ("<sha256>  <relative path>"); absent = unchecked
  if [ -f "$LAB_OPT/weights.sha256" ]; then
    "$PY" -I "$HOSTKIT/lib/verify_sha256.py" "$GLIMMER_BASE" "$LAB_OPT/weights.sha256"
  else
    echo "[bootstrap] WARNING: no weights.sha256 was staged: the base weights are NOT verified by hash"
  fi
}

step machine check_machine
step uv install_uv
step axolotl_env make_axolotl_env
step vllm_env make_vllm_env
step verify_env make_verify_env
step weights download_weights
verify_weights
printf '{"download_s": %s, "env_s": %s, "total_s": %s}\n' "$(secs weights)" "$(( $(secs axolotl_env) + $(secs vllm_env) ))" "$(( $(date +%s) - T_TOTAL ))" > "$LAB_OPT/bootstrap.done"
echo "[bootstrap] ok"
