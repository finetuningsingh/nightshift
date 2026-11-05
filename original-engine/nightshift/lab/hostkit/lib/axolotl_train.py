"""Run Axolotl QLoRA training on a registered dataset (runs in the Axolotl virtual environment, which has PyYAML).

    python axolotl_train.py CONFIG.yaml --data DATA.jsonl

The researcher's config is the base; the harness overrides what makes a trial comparable and safe: ``base_model`` is the local pinned
checkpoint (``GLIMMER_BASE``), the single dataset entry points at ``--data``, ``output_dir`` is ``$GLIMMER_OUT/adapter``, W&B is off,
validation is off. If the config sets no ``eot_tokens`` the harness sets ``["<|eot|>", "<|eom|>"]`` (the 0.19.0 example omits them, so the turn
terminator would not be trained). Before training, the effective dose on the exact rows of ``--data`` (steps, examples seen, epochs;
``max_steps`` > 0 overrides ``num_epochs``) is printed and appended to ``$GLIMMER_OUT/dose.jsonl``; it is recorded, never a refusal. UNVERIFIED on a GPU: that Axolotl 0.19.0 accepts the text-only path (see glimmer-qlora-text.yaml).
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


def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def dose(cfg, rows):
    """The effective training dose of the harness config ``cfg`` on exactly ``rows`` accepted training rows (sample packing off, as the
    harness configs use): optimizer steps, examples seen and epochs. In Axolotl 0.19.0 a positive ``max_steps`` takes precedence over
    ``num_epochs``. ``checks`` are recorded facts, never a refusal."""
    micro, accum = _int(cfg.get("micro_batch_size", 1)) or 1, _int(cfg.get("gradient_accumulation_steps", 1)) or 1
    batch = micro * accum
    epochs_cfg = cfg.get("num_epochs", 1)
    max_steps = _int(cfg.get("max_steps"))
    per_epoch = -(-rows // batch) if rows > 0 else 0
    checks = []
    if max_steps is not None and max_steps > 0:
        steps = max_steps
        epochs = (steps * batch / rows) if rows else None
        checks.append(f"max_steps={max_steps} overrides num_epochs={epochs_cfg}")
        if rows and steps * batch < rows:
            checks.append(f"less than one epoch: {steps * batch} examples seen of {rows} rows")
    else:
        steps = int(-(-per_epoch * float(epochs_cfg) // 1)) if isinstance(epochs_cfg, (int, float)) and not isinstance(epochs_cfg, bool) else None
        epochs = float(epochs_cfg) if steps is not None else None
    if rows == 0:
        checks.append("no training rows")
    return {"rows": rows, "micro_batch_size": micro, "gradient_accumulation_steps": accum, "batch": batch, "num_epochs": epochs_cfg,
            "max_steps": max_steps, "steps_per_epoch": per_epoch, "steps": steps, "examples_seen": None if steps is None else steps * batch,
            "epochs": epochs, "checks": checks}


def count_rows(path):
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def record_dose(out_dir, record):
    """Append the dose of this call to ``$GLIMMER_OUT/dose.jsonl`` (one line per training call; the helper may be called once per round)."""
    import json

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dose.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


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
    d = dose(final, count_rows(a.data))  # from the exact rows about to be trained on, before the optimizer starts (recorded, not a refusal)
    record_dose(out, d)
    print(f"[glimmer-axolotl-train] dose: {d['rows']} rows, {d['steps']} steps x batch {d['batch']} = {d['examples_seen']} examples, "
          f"epochs {d['epochs'] if d['epochs'] is None else round(d['epochs'], 3)}" + (f"; {'; '.join(d['checks'])}" if d["checks"] else ""), flush=True)
    if a.dry_run:
        print(path)
        return 0
    argv, env = train_command(path, sys.executable, os.environ)
    return subprocess.call(argv, env=env)


if __name__ == "__main__":
    sys.exit(main())
