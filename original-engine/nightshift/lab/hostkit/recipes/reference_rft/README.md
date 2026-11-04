Reference method (rejection-sampling fine-tuning) for the Glimmer harness. Seeds v0 (owner 4 copies this directory into
`assets/pack_v0/seed/recipe/` and writes `task.yaml` and `notes/seed.md`).

    glimmer submit nightshift/lab/hostkit/recipes/reference_rft --entrypoint "bash run.sh"

Steps: `glimmer-vllm-up` (base model in vLLM) -> `rft_pipeline.py generate` (K samples per public train prompt) -> stop vLLM ->
`rft_pipeline.py filter` (keep answers that pass every IFEvalG constraint; drop truncated ones) -> `glimmer-axolotl-train` (QLoRA SFT).
Records `generation_s`, `train_s` (timings.json) and the generation yield (rft_stats.json). Nothing here has run on a GPU.
