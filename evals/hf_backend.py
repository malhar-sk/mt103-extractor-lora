"""Batched greedy generation with Hugging Face transformers (+ optional LoRA adapter), for GPU evals.

Used by `evals/run_eval.py --backend hf` (for example in the Kaggle kernel right after training), so the
fine-tuned adapter is scored by exactly the same scorer as the Ollama runs. Outputs are written to the
same results/cache/<name>.jsonl format.
"""
from __future__ import annotations

import time


def load(base: str, adapter: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16 if gpu else torch.float32,
                                                 device_map="auto" if gpu else None)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def generate(model, tok, conversations: list[list[dict]], batch_size: int = 16, max_new_tokens: int = 600):
    """Greedy-decode each conversation; returns [{raw, latency_s, out_tokens, gen_s}] in input order.
    latency_s is the batch wall time divided by the batch size (per-example latency is not comparable to
    the single-request CPU numbers)."""
    import torch
    out = []
    order = sorted(range(len(conversations)), key=lambda i: len(str(conversations[i])))  # less padding
    res = [None] * len(conversations)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        prompts = [tok.apply_chat_template(conversations[i], add_generation_prompt=True, tokenize=False) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        dt = time.time() - t0
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            ids = new[j]
            n = int((ids != tok.pad_token_id).sum())
            res[i] = {"raw": tok.decode(ids, skip_special_tokens=True), "latency_s": dt / len(idx),
                      "out_tokens": n, "gen_s": dt / len(idx), "prompt_tokens": int(enc["attention_mask"][j].sum())}
    out.extend(res)
    return out
