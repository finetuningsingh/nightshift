#!/usr/bin/env bash
# generate.sh N_PROMPTS SAMPLES MAX_TOKENS WORKERS  - reference method step 1: Glimmer samples its own answers to public train prompts.
source ~/ref/env-lab.sh
export GLIMMER_OUT
python3 -I "$HOSTKIT/recipes/reference_rft/rft_pipeline.py" generate --prompts ~/lab/work/data/train.jsonl --out "$GLIMMER_OUT/samples.jsonl" \
  --n-prompts "$1" --samples "$2" --max-tokens "$3" --workers "$4" --reasoning-strength high --base-url http://127.0.0.1:8000
echo GEN_EXIT=$?
