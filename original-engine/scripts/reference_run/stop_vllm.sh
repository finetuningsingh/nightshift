#!/usr/bin/env bash
source ~/ref/env-lab.sh
bash "$HOSTKIT/bin/glimmer-vllm-up" --stop
sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader
