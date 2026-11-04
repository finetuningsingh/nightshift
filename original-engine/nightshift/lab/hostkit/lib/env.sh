# Sourced by every hostkit script. HOSTKIT must already be set. Everything is overridable so the scripts can be tested away from a rental.
export HOSTKIT
export LAB_OPT="${LAB_OPT:-/opt/lab}"
export LAB_WORK="${LAB_WORK:-/work}"
export LAB_MODELS="${LAB_MODELS:-/models}"
export GLIMMER_BASE="${GLIMMER_BASE:-$LAB_MODELS/glimmer}"
export GLIMMER_OUT="${GLIMMER_OUT:-$LAB_WORK/out}"
export GLIMMER_CONFIG_TEMPLATE="${GLIMMER_CONFIG_TEMPLATE:-$HOSTKIT/glimmer-qlora-text.yaml}"
export IFEVALG_PATH="${IFEVALG_PATH:-$LAB_OPT/ifevalg}"
export REWARDS_PY="${REWARDS_PY:-$LAB_OPT/rewards.py}"
export AXOLOTL_VENV="${AXOLOTL_VENV:-$LAB_OPT/venv-axolotl}"
export VLLM_VENV="${VLLM_VENV:-$LAB_OPT/venv-vllm}"
export VERIFY_VENV="${VERIFY_VENV:-$LAB_OPT/venv-verify}"   # python with the verifier packages (nltk, langdetect, immutabledict); fronted by $LAB_OPT/bin/python3
export NLTK_DATA="${NLTK_DATA:-$LAB_OPT/nltk_data}"
export PATH="$LAB_OPT/bin:$PATH"
# The pinned stack (docs notes: glimmer-training.md 2.2, 4.3). Changing a pin changes the scorer's comparability: do it on purpose.
GLIMMER_REPO="${GLIMMER_REPO:-meta-models/Muse-Glimmer-30B}"
GLIMMER_REVISION="${GLIMMER_REVISION:-a4e59da52a7bc87ae7251dd5545c0dd437c44b68}"
PY="${LAB_PYTHON:-python3}"
