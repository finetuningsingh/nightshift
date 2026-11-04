"""Run Axolotl QLoRA training on a registered dataset (runs in the Axolotl virtual environment, which has PyYAML).

    python axolotl_train.py CONFIG.yaml --data DATA.jsonl

The researcher's config is the base; the harness overrides what makes a trial comparable and safe: ``base_model`` is the local pinned
checkpoint (``GLIMMER_BASE``), the single dataset entry points at ``--data``, ``output_dir`` is ``$GLIMMER_OUT/adapter``, W&B is off,
validation is off. If the config sets no ``eot_tokens`` the harness sets ``["<|eot|>", "<|eom|>"]`` (the 0.19.0 example omits them, so the turn
terminator would not be trained). UNVERIFIED on a GPU: that Axolotl 0.19.0 accepts the text-only path (see glimmer-qlora-text.yaml).
"""
import argparse
import os
import subprocess
import sys

EOT = ["<|eot|>", "<|eom|>"]
DROP_KEYS = ("wandb_project", "wandb_entity", "wandb_watch", "wandb_name", "wandb_log_model", "wandb_mode", "hub_model_id", "push_to_hub")


def overlay(cfg, data, base, out_dir, cache_dir):
    """Return the harness config: ``cfg`` (a dict from yaml.safe_load) with the harness overrides applied."""
    cfg = dict(cfg)
    ds = cfg.get("datasets")
    if not isinstance(ds, list) or len(ds) != 1 or not isinstance(ds[0], dict):
        raise SystemExit("the config must define exactly one dataset entry (the harness fills in its path)")
    entry = dict(ds[0])
    entry.update(path=data, type=entry.get("type", "chat_template"), field_messages="messages")
    entry.setdefault("roles_to_train", ["assistant"])
    entry.setdefault("train_on_eos", "turn")
    cfg["datasets"] = [entry]
    cfg.update(base_model=base, output_dir=out_dir, dataset_prepared_path=cache_dir, val_set_size=0.0)
    cfg.pop("test_datasets", None)
    cfg.pop("model_revision", None)
    for k in DROP_KEYS:
        cfg.pop(k, None)
    cfg.setdefault("eot_tokens", EOT)
    return cfg


def train_command(config_path, python, environ):
    """The ``axolotl train`` argv and environment. ``python`` is the Axolotl virtual environment's interpreter: running a file with it does not
    put the environment's ``bin`` on PATH, so ``axolotl`` is addressed by its path there and the directory is added to PATH (torchrun)."""
    venv_bin = os.path.dirname(python)
    env = dict(environ, CUDA_VISIBLE_DEVICES=environ.get("CUDA_VISIBLE_DEVICES", "0"), WANDB_DISABLED="true", WANDB_MODE="disabled")
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    return [os.path.join(venv_bin, "axolotl"), "train", config_path, "--launcher", "torchrun", "--", "--nproc_per_node=1", "--nnodes=1"], env


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", required=True)
    ap.add_argument("--dry-run", action="store_true", help="write the harness config and print its path, do not train")
    a = ap.parse_args(argv)
    import yaml  # the Axolotl environment has it

    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    out = os.environ.get("GLIMMER_OUT", "/work/out")
    work = os.path.dirname(out.rstrip("/")) or "/work"
    final = overlay(cfg, os.path.abspath(a.data), os.environ["GLIMMER_BASE"], os.path.join(out, "adapter"), os.path.join(work, "cache", "prepared"))
    path = os.path.join(out, "harness-config.yaml")
    os.makedirs(out, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(final, fh, sort_keys=True)
    if a.dry_run:
        print(path)
        return 0
    argv, env = train_command(path, sys.executable, os.environ)
    return subprocess.call(argv, env=env)


if __name__ == "__main__":
    sys.exit(main())
