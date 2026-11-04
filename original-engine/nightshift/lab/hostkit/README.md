Scripts and configs that run on a rented GPU (Glimmer harness, owner 2). Uploaded to `/opt/lab/hostkit` by `trainer.py` / `model_server.py`.

* `bootstrap.sh` - one-time setup: Axolotl 0.19.0 and vLLM 0.32.0 environments, pinned Glimmer weights (idempotent, UNVERIFIED on a GPU).
* `stage.sh BUNDLE` - install the pinned public files (IFEvalG, rewards.py) after a sha256 check; writes the `glimmer-*` launchers.
* `run_job.sh` - the job contract (`$LAB_WORK/job.json`, `recipe.tar.gz`, `data/`; result in `$GLIMMER_OUT`, `trial.done` last).
* `serve.sh` - vLLM for the base and adapters (refuses an adapter whose listing hash differs from the control host's).
* `bin/glimmer-axolotl-train`, `bin/glimmer-vllm-up` - the only training entry points for recipes.
* `lib/` - small stdlib helpers (`check_recipe_tar`, `run_with_timeout`, `glimmer_data`, ...). `recipes/reference_rft/` - the reference method.
* `glimmer-qlora-text.yaml` - the harness QLoRA config.

Every script sources `lib/env.sh`; every directory is overridable by `LAB_OPT`, `LAB_WORK`, `LAB_MODELS` (the tests run the scripts against temp directories).
Nothing here selects, reads or stages selection, held-out or test data.
