#!/usr/bin/env bash
# train.sh - reference method step 3: QLoRA SFT of the kept answers (config recipes/reference_rft/config.yaml)
source ~/ref/env-lab.sh
export HF_HUB_OFFLINE=1 GLIMMER_OUT
T0=$(date +%s)
bash "$HOSTKIT/bin/glimmer-axolotl-train" "$HOSTKIT/recipes/reference_rft/config.yaml" --data ~/lab/work/data/sft.jsonl
RC=$?
echo "TRAIN_EXIT=$RC seconds=$(( $(date +%s) - T0 ))"
