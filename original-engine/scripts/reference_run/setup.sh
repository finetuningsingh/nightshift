#!/usr/bin/env bash
# Rental setup, the sequence that worked on 2026-11-02 (docs/live-smoke-2026-11-02.md): hostkit bootstrap without its Axolotl step (vLLM env +
# weights) and the hand-built Axolotl env, in parallel. Run detached; prints SETUP_DONE when both finish.
set -uo pipefail
mkdir -p ~/lab/opt ~/lab/work/out ~/lab/work/data ~/lab/models
source ~/ref/env-lab.sh
cd "$HOSTKIT" && grep -v "^step axolotl_env" bootstrap.sh > bootstrap-a.sh && chmod +x *.sh bin/*
(bash "$HOSTKIT/bootstrap-a.sh" > ~/bootstrap-a.log 2>&1; echo "BOOT_EXIT=$?" >> ~/bootstrap-a.log) &
(
  V=$HOME/lab/opt/venv-axolotl
  uv venv --python 3.12 "$V"
  export UV_TORCH_BACKEND=cu130
  uv pip install --python "$V/bin/python" "torch==2.12.1" "torchvision==0.27.1"
  uv pip install --python "$V/bin/python" "axolotl==0.19.0"
  uv pip install --python "$V/bin/python" "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@4dfa52212d0552805aec5a6a2176c0abbea277d2" || echo CCE_FAILED
  uv pip check --python "$V/bin/python" || true
  echo AXO_ENV_DONE
) > ~/axo-env.log 2>&1 &
wait
echo SETUP_DONE
