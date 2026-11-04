#!/usr/bin/env bash
# serve.sh [--adapter NAME=SHA]   starts vLLM detached (base only, or base + the adapter in $GLIMMER_OUT/adapter) and returns
source ~/ref/env-lab.sh
args=()
if [ "${1:-}" = "--adapter" ]; then args=(--adapter "${2%%=*}=$GLIMMER_OUT/adapter=${2#*=}"); fi
(nohup setsid bash "$HOSTKIT/serve.sh" --port 8000 "${args[@]}" > "$GLIMMER_OUT/vllm.log" 2>&1 < /dev/null &)
echo started
