"""Merge a LoRA adapter into its base model and save HF safetensors (input for llama.cpp).

    python train/merge.py --adapter models/adapter --out models/merged
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "models" / "adapter"))
    ap.add_argument("--out", default=str(ROOT / "models" / "merged"))
    ap.add_argument("--base", default=None, help="override base model id/path")
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(a.adapter)
    if not (adapter / "adapter_config.json").exists():
        sys.exit(f"no adapter at {adapter} (run `make train` or unzip the Kaggle adapter.zip there)")
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    base = a.base or cfg["base_model_name_or_path"]
    local = ROOT / "models" / "base-qwen2.5-0.5b-instruct"
    if a.base is None and not Path(base).exists() and "0.5B" in base and local.exists():
        base = str(local)
    print(f"base={base}")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(a.out)
    print(f"merged model saved to {a.out}")


if __name__ == "__main__":
    main()
