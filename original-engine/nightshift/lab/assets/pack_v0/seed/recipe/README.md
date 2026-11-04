# Starting recipe: sample, verify, fine-tune

`bash run.sh` is the entrypoint. It runs on the rented GPU inside `glimmer submit`.

| Step | File | What it does |
|---|---|---|
| 1 | `glimmer-vllm-up --port 8000` | serves the base model (started by `run.sh`, stopped before training) |
| 2 | `generate.py` | samples `SAMPLES` answers for `N_PROMPTS` training prompts |
| 3 | `build_sft.py` | keeps answers whose constraints all pass (`/opt/lab/rewards.py`), writes chat-format training data |
| 4 | `glimmer-axolotl-train "$GLIMMER_CONFIG_TEMPLATE" --data sft.jsonl` | QLoRA fine-tune; leaves the adapter in `/work/out/adapter` |

Knobs (environment variables): `N_PROMPTS` (default 400), `SAMPLES` (4), `MAX_TOKENS` (3072), `PER_PROMPT` (1 answer kept per prompt).

Environment the job gives you: `GLIMMER_BASE`, `GLIMMER_CONFIG_TEMPLATE`, `GLIMMER_OUT` (/work/out), `IFEVALG_PATH`, `REWARDS_PY`.
Only the two `glimmer-*` helper commands may train or serve the model.
