| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained** | 150 | 100.0% | 58.7% | 80.4% | 16.0% | 1.4% | 2.0% | 9.6 | 22.1 |
| Fine-tuned 0.5B LoRA (CPU smoke run) **+ rules** (`postprocess.py`, same raw outputs) | 150 | 100.0% | 98.0% | 82.0% | 18.0% | 36.4% | 48.0% | 9.6 | 22.1 |
| Fine-tuned 0.5B LoRA, **full run on v2 data** (Kaggle T4, fp16 adapter, short prompt; batched GPU, so latency is not comparable) | 600 | 100.0% | 97.2% | 98.8% | 86.8% | 85.9% | 85.5% | 0.8 | 208.2 |
| Fine-tuned 0.5B LoRA, full run on v2 **+ rules** (`postprocess.py`, same raw outputs) | 600 | 100.0% | 100.0% | 99.0% | 89.8% | 95.4% | 95.9% | 0.8 | 208.2 |

Per-field exact match:

| Field | ft-qwen2.5-0.5b-lora | ft-qwen2.5-0.5b-lora+pp | kaggle-ft-qwen2.5-0.5b-lora-v2 | kaggle-ft-qwen2.5-0.5b-lora-v2+pp |
|---|---|---|---|---|
| sender_reference | 82.7% | 82.7% | 100.0% | 100.0% |
| bank_operation_code | 80.7% | 80.7% | 100.0% | 100.0% |
| value_date | 88.0% | 94.0% | 96.5% | 97.2% |
| currency | 92.7% | 96.0% | 100.0% | 100.0% |
| amount | 88.7% | 91.3% | 99.5% | 99.8% |
| ordering_customer.account | 86.7% | 86.7% | 99.2% | 99.2% |
| ordering_customer.name | 84.7% | 84.7% | 99.0% | 99.0% |
| ordering_customer.address | 75.3% | 75.3% | 94.2% | 94.2% |
| ordering_institution | 63.3% | 64.0% | 97.7% | 99.5% |
| account_with_institution | 58.7% | 62.7% | 98.8% | 99.2% |
| beneficiary.account | 81.3% | 82.0% | 99.2% | 99.2% |
| beneficiary.name | 83.3% | 83.3% | 99.0% | 99.0% |
| beneficiary.address | 88.7% | 88.7% | 99.3% | 99.3% |
| remittance_info | 86.0% | 86.0% | 99.8% | 99.8% |
| charges | 66.0% | 71.3% | 99.7% | 99.8% |

Commands:

- `v2-ft-qwen2.5-0.5b-lora`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/v2/test.jsonl --limit 150 --name v2-ft-qwen2.5-0.5b-lora` (2026-11-24)
- `v2-ft-qwen2.5-0.5b-lora+pp`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/v2/test.jsonl --limit 150 --name v2-ft-qwen2.5-0.5b-lora --postprocess --cache-only` (2026-11-24)
- `v2-kaggle-ft-qwen2.5-0.5b-lora-v2`: `python evals/run_eval.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --prompt ft --data data/v2/test.jsonl --name v2-kaggle-ft-qwen2.5-0.5b-lora-v2 --batch-size 16` (2026-11-24)
- `v2-kaggle-ft-qwen2.5-0.5b-lora-v2+pp`: `python evals/run_eval.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --prompt ft --data data/v2/test.jsonl --name v2-kaggle-ft-qwen2.5-0.5b-lora-v2 --postprocess --cache-only` (2026-11-25)
