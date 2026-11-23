"""LoRA / QLoRA fine-tuning of Qwen2.5-Instruct for MT103 extraction (TRL SFTTrainer).

    python train/train_lora.py                 # auto: full (GPU, QLoRA 4-bit) or smoke (CPU)
    python train/train_lora.py --mode smoke    # ~150 steps on CPU: pipeline proof, NOT fully trained

Outputs:
    models/adapter/                 LoRA adapter (+ tokenizer)
    results/train_log.json          loss per step, val loss per eval, val field accuracy
    results/loss.png                training / validation loss curve
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from prompts import ft_messages, parse_output, target_str  # noqa: E402


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_prompt_completion(rows):
    """Conversational prompt/completion format: TRL computes loss on the completion only."""
    return [{"prompt": ft_messages(r["text"]),
             "completion": [{"role": "assistant", "content": target_str(r["target"])}]} for r in rows]


def val_generation_accuracy(model, tok, rows, max_new_tokens=400):
    """Greedy-decode val examples and score them with the same scorer as evals/run_eval.py."""
    import torch
    from evals.run_eval import aggregate, score_one
    model.eval()
    scores, lat, toks, gen = [], [], [], []
    for r in rows:
        ids = tok.apply_chat_template(ft_messages(r["text"]), add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        dt_ = time.time() - t0
        new = out[0, ids["input_ids"].shape[1]:]
        raw = tok.decode(new, skip_special_tokens=True)
        scores.append(score_one(r["target"], parse_output(raw)))
        lat.append(dt_); toks.append(len(new)); gen.append(dt_)
    model.train()
    s = aggregate(scores, lat, toks, gen)
    return {k: s[k] for k in ("n", "valid_json_rate", "schema_valid_rate", "field_accuracy_macro",
                              "error_precision", "error_recall")}


def plot_loss(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    tr = [(e["step"], e["loss"]) for e in log["history"] if "loss" in e]
    va = [(e["step"], e["eval_loss"]) for e in log["history"] if "eval_loss" in e]
    if tr:
        ax.plot(*zip(*tr), color="#2a6fdb", lw=1.6, label="train loss")
    if va:
        ax.plot(*zip(*va), color="#d9534f", lw=1.6, marker="o", label="val loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss (completion tokens)")
    ax.set_title(f"LoRA on {log['base_model'].split('/')[-1]} — {log['mode']} run" +
                 (" (CPU smoke, not fully trained)" if log["mode"] == "smoke" else ""), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "train" / "config.yaml"))
    ap.add_argument("--mode", choices=["auto", "full", "smoke"], default="auto")
    ap.add_argument("--base-model", default=None, help="HF id or local path (overrides config)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "models" / "adapter"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--val-gen-examples", type=int, default=None)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(open(a.config))
    gpu = torch.cuda.is_available()
    mode = a.mode if a.mode != "auto" else ("full" if gpu else "smoke")
    mc = cfg[mode]
    if not gpu:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    base = a.base_model or (cfg["base_model_gpu"] if gpu else cfg["base_model_cpu"])
    local = ROOT / "models" / "base-qwen2.5-0.5b-instruct"
    if a.base_model is None and not gpu and local.exists():
        base = str(local)
    print(f"mode={mode} gpu={gpu} base={base}", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    kw = {}
    if gpu and mc.get("qlora"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16,
                                                       bnb_4bit_use_double_quant=True)
        kw["device_map"] = "auto"
    # CPU: bf16 weights (the VPS has avx512_bf16) to halve memory; LoRA weights stay fp32 (peft default).
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16 if gpu else torch.bfloat16, **kw)
    if gpu and mc.get("qlora"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    train_rows = load_jsonl(Path(a.data_dir) / "train.jsonl")
    val_rows = load_jsonl(Path(a.data_dir) / "val.jsonl")
    train_ds = Dataset.from_list(to_prompt_completion(train_rows)).shuffle(seed=cfg["train"]["seed"])
    val_ds = Dataset.from_list(to_prompt_completion(val_rows[: mc["val_loss_examples"]]))
    gen_rows = val_rows[: (a.val_gen_examples if a.val_gen_examples is not None else mc["val_gen_examples"])]

    lc = cfg["lora"]
    peft_cfg = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                          target_modules=lc["target_modules"], task_type="CAUSAL_LM")
    tc = cfg["train"]
    steps_kw = {"num_train_epochs": mc["epochs"]} if mode == "full" else {"max_steps": a.max_steps or mc["max_steps"]}
    eval_kw = ({"eval_strategy": "epoch"} if mode == "full" else
               {"eval_strategy": "steps", "eval_steps": mc["eval_every"]})
    sft = SFTConfig(
        output_dir=str(ROOT / "train" / "runs" / mode), per_device_train_batch_size=mc["batch_size"],
        per_device_eval_batch_size=mc["batch_size"], gradient_accumulation_steps=mc["grad_accum"],
        learning_rate=tc["lr"], lr_scheduler_type=tc["scheduler"], warmup_steps=tc["warmup_ratio"],
        max_length=tc["max_length"], logging_steps=5, save_strategy="no", report_to=[],
        seed=tc["seed"], fp16=gpu, bf16=not gpu, use_cpu=not gpu, dataloader_num_workers=0,
        gradient_checkpointing=bool(mc.get("gradient_checkpointing", False)),
        **steps_kw, **eval_kw)

    log = {"mode": mode, "base_model": base, "gpu": gpu, "lora": lc, "train": tc, "mode_cfg": mc,
           "n_train": len(train_rows), "history": [], "val_generation": []}

    class GenEval(TrainerCallback):
        """Full mode: generation-based val accuracy after every epoch."""
        def on_epoch_end(self, args, state, control, model=None, **k):
            if mode == "full":
                m = val_generation_accuracy(model, tok, gen_rows)
                m.update({"epoch": state.epoch, "step": state.global_step})
                log["val_generation"].append(m)
                print("val_generation", m, flush=True)

    trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=peft_cfg, callbacks=[GenEval()])
    t0 = time.time()
    trainer.train()
    log["train_seconds"] = round(time.time() - t0, 1)
    log["history"] = trainer.state.log_history
    if mode == "smoke":  # one generation-based val score at the end
        m = val_generation_accuracy(trainer.model, tok, gen_rows)
        m.update({"epoch": trainer.state.epoch, "step": trainer.state.global_step})
        log["val_generation"].append(m)
        print("val_generation", m, flush=True)

    out = Path(a.out)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "TRAINING_INFO.json").write_text(json.dumps(
        {"mode": mode, "base_model": base if mode == "full" else cfg["base_model_cpu"],
         "steps": trainer.state.global_step, "epochs": trainer.state.epoch,
         "label": "CPU smoke run, not fully trained" if mode == "smoke" else "full QLoRA run"}, indent=2))
    res = Path(a.results); res.mkdir(exist_ok=True)
    (res / "train_log.json").write_text(json.dumps(log, indent=2, default=str))
    plot_loss(log, res / "loss.png")
    print(f"saved adapter to {out} in {log['train_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
