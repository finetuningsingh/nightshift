#!/usr/bin/env bash
# The job contract on the rental (docs/harness-design.md, Interface appendix). Run detached by the trainer:
#   bash run_job.sh            (reads $LAB_WORK/job.json and $LAB_WORK/recipe.tar.gz)
# Inputs : $LAB_WORK/recipe.tar.gz (re-checked here before extraction), $LAB_WORK/data/{train,dev}.jsonl (public data only),
#          $LAB_WORK/job.json {"entrypoint": [argv...], "deadline_epoch": N}
# Env for the recipe: GLIMMER_BASE, GLIMMER_CONFIG_TEMPLATE, GLIMMER_OUT=$LAB_WORK/out, IFEVALG_PATH, REWARDS_PY, and $LAB_OPT/bin on PATH.
# Outputs: $GLIMMER_OUT/adapter/{adapter_config.json,adapter_model.safetensors}, train_data.jsonl (written ONLY by the glimmer helper
#          commands), timings.json (optional, written by the recipe), job.log, reload_check.json, and LAST trial.done:
#          {"exit_code": N, "reason": "ok|recipe_failed|timeout|bad_recipe|no_adapter|no_train_data|reload_failed", ...}.
# A cap shorter than the job ends it by deleting the rental, not by asking the recipe to stop; deadline_epoch (cap minus the pull
# margin) is only the point after which this script kills the recipe so the adapter can still be pulled.
set -uo pipefail
HOSTKIT="${HOSTKIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=lib/env.sh
source "$HOSTKIT/lib/env.sh"

OUT="$GLIMMER_OUT"
mkdir -p "$OUT"
rm -rf "$OUT/trial.done" "$OUT/adapter" "$OUT/train_data.jsonl" "$OUT/timings.json" "$OUT/reload_check.json"
LOG="$OUT/job.log"
: > "$LOG"
T0=$(date +%s)

finish() { # finish REASON EXIT_CODE
  local tmp="$OUT/.trial.done.tmp"
  printf '{"reason": "%s", "exit_code": %s, "started_epoch": %s, "finished_epoch": %s}\n' "$1" "$2" "$T0" "$(date +%s)" > "$tmp"
  mv -f "$tmp" "$OUT/trial.done"
  exit 0
}

[ -f "$LAB_WORK/job.json" ] || { echo "no job.json" >> "$LOG"; finish bad_recipe 2; }
"$PY" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d["entrypoint"], list) and d["entrypoint"]; int(d["deadline_epoch"])' "$LAB_WORK/job.json" 2>> "$LOG" \
  || { echo "job.json is malformed" >> "$LOG"; finish bad_recipe 2; }

# re-check the tarball, then extract it with our own code (never tar -x)
rm -rf "$LAB_WORK/recipe"
"$PY" -I "$HOSTKIT/lib/check_recipe_tar.py" "$LAB_WORK/recipe.tar.gz" --extract "$LAB_WORK/recipe" >> "$LOG" 2>&1 || finish bad_recipe 2
mkdir -p "$OUT/adapter" "$LAB_WORK/cache"

echo "[run_job] starting the entrypoint from job.json" >> "$LOG"
(
  cd "$LAB_WORK/recipe" || exit 127
  exec "$PY" -I "$HOSTKIT/lib/job_entry.py" "$LAB_WORK/job.json"
) >> "$LOG" 2>&1
RC=$?
# anything still holding the GPU (a vLLM server the recipe forgot) is stopped before we look at the result
bash "$HOSTKIT/bin/glimmer-vllm-up" --stop >> "$LOG" 2>&1 || true

[ "$RC" -eq 124 ] && finish timeout 124
[ "$RC" -ne 0 ] && finish recipe_failed "$RC"
for f in adapter_config.json adapter_model.safetensors; do
  [ -f "$OUT/adapter/$f" ] && [ ! -L "$OUT/adapter/$f" ] || { echo "missing adapter file $f" >> "$LOG"; finish no_adapter 3; }
done
[ -s "$OUT/train_data.jsonl" ] || { echo "no train_data.jsonl: the recipe did not train through a glimmer helper command" >> "$LOG"; finish no_train_data 4; }

if [ "${LAB_SKIP_RELOAD_CHECK:-0}" != "1" ]; then
  if ! "$AXOLOTL_VENV/bin/python" "$HOSTKIT/lib/reload_check.py" "$OUT/adapter" "$GLIMMER_BASE" "$OUT/reload_check.json" >> "$LOG" 2>&1; then
    finish reload_failed 5
  fi
fi
finish ok 0
