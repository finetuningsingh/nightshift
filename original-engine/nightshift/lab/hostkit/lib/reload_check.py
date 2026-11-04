"""Reload a QLoRA adapter through Transformers + PEFT the way Meta's guide requires (UNVERIFIED on a GPU).

    $AXOLOTL_VENV/bin/python reload_check.py ADAPTER_DIR BASE_DIR OUT_JSON

Loads the base in NF4 with ``dtype=bfloat16`` and ``bnb_4bit_quant_storage=bfloat16`` (otherwise the vision path loads in byte dtype),
attaches the adapter with ``PeftModel.from_pretrained(is_trainable=False)`` and writes ``{"ok": true, "seconds": s}``. A failure writes
``{"ok": false, "error": ...}`` and exits 1. Heavy imports are lazy so ``--help`` and the unit tests work without torch.
"""
import json
import sys
import time


def check_config(adapter_dir):
    """Cheap structural checks that need no GPU: adapter_config.json keys."""
    with open(f"{adapter_dir}/adapter_config.json") as fh:
        cfg = json.load(fh)
    if cfg.get("peft_type") != "LORA":
        raise SystemExit(f"peft_type is {cfg.get('peft_type')!r}, not LORA")
    r = cfg.get("r")
    if not isinstance(r, int) or not 1 <= r <= 32:
        raise SystemExit(f"rank r={r!r} is outside 1..32 (the serving stack runs --max-lora-rank 32)")
    return cfg


def main(argv):
    adapter, base, out = argv[1], argv[2], argv[3]
    t0 = time.time()
    try:
        check_config(adapter)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16,
        )  # fmt: skip
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, quantization_config=quant, device_map="auto")
        PeftModel.from_pretrained(model, adapter, is_trainable=False)
        result = {"ok": True, "seconds": round(time.time() - t0, 1)}
    except BaseException as exc:  # noqa: BLE001
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "seconds": round(time.time() - t0, 1)}
    with open(out, "w") as fh:
        json.dump(result, fh)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
