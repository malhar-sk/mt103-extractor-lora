| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Base Qwen2.5-0.5B-Instruct (Ollama q4_K_M, 4-shot prompt) | 300 | 100.0% | 59.0% | 62.6% | 1.0% | 0.0% | 0.0% | 13.0 | 13.5 |
| Base Qwen2.5-0.5B, 4-shot **+ rules** (`postprocess.py`, same raw outputs) | 300 | 100.0% | 99.0% | 69.4% | 3.0% | 49.0% | 68.1% | 13.0 | 13.5 |
| Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained** | 300 | 100.0% | 83.3% | 94.2% | 51.7% | 5.3% | 7.2% | 9.3 | 22.4 |
| Fine-tuned 0.5B LoRA (CPU smoke run) **+ rules** (`postprocess.py`, same raw outputs) | 300 | 100.0% | 100.0% | 95.7% | 63.3% | 59.7% | 53.6% | 9.3 | 22.4 |
| Fine-tuned 0.5B LoRA, **full run on v2 data** (Kaggle T4, fp16 adapter, short prompt; batched GPU, so latency is not comparable) | 300 | 100.0% | 98.0% | 99.3% | 89.3% | 78.2% | 88.4% | 0.8 | 216.6 |
| Fine-tuned 0.5B LoRA, full run on v2 **+ rules** (`postprocess.py`, same raw outputs) | 300 | 100.0% | 100.0% | 99.4% | 91.3% | 97.1% | 95.7% | 0.8 | 216.6 |
| Fine-tuned 0.5B LoRA, full run on v2, **served build** (GGUF q4_K_M via Ollama on CPU, short prompt) | 300 | 99.7% | 97.3% | 98.9% | 87.3% | 77.3% | 84.1% | 8.4 | 24.9 |
| Full run on v2, served CPU build **+ rules** (`postprocess.py`, same raw outputs) | 300 | 99.7% | 99.7% | 99.1% | 90.7% | 92.9% | 94.2% | 8.4 | 24.9 |
| Claude Haiku 5.5 via API (4-shot prompt), reference ceiling | 300 | 100.0% | 97.0% | 99.1% | 91.3% | 98.5% | 97.1% | 5.6 | 213.9 |
| Claude Haiku 5.5 **+ rules** (`postprocess.py`, same raw outputs) | 300 | 100.0% | 100.0% | 99.1% | 91.7% | 98.6% | 98.6% | 5.6 | 213.9 |

Per-field exact match:

| Field | base-qwen2.5-0.5b-fewshot | base-qwen2.5-0.5b-fewshot+pp | ft-qwen2.5-0.5b-lora | ft-qwen2.5-0.5b-lora+pp | kaggle-ft-qwen2.5-0.5b-lora-v2 | kaggle-ft-qwen2.5-0.5b-lora-v2+pp | cpu-ft-qwen2.5-0.5b-lora-v2 | cpu-ft-qwen2.5-0.5b-lora-v2+pp | api-claude-haiku-5-5-fewshot | api-claude-haiku-5-5-fewshot+pp |
|---|---|---|---|---|---|---|---|---|---|---|
| sender_reference | 96.7% | 96.7% | 99.3% | 99.3% | 100.0% | 100.0% | 99.7% | 99.7% | 99.7% | 99.7% |
| bank_operation_code | 49.7% | 96.0% | 89.7% | 89.7% | 99.7% | 99.7% | 99.3% | 99.3% | 99.3% | 99.3% |
| value_date | 60.7% | 68.3% | 91.0% | 97.3% | 93.0% | 94.3% | 94.3% | 96.0% | 99.3% | 99.3% |
| currency | 93.7% | 97.3% | 96.7% | 100.0% | 100.0% | 100.0% | 99.7% | 99.7% | 99.7% | 99.7% |
| amount | 82.0% | 89.7% | 90.0% | 93.3% | 97.7% | 97.7% | 97.0% | 97.0% | 97.0% | 97.0% |
| ordering_customer.account | 56.0% | 56.0% | 98.3% | 98.3% | 100.0% | 100.0% | 99.7% | 99.7% | 99.7% | 99.7% |
| ordering_customer.name | 63.7% | 63.7% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.7% | 99.7% | 99.7% |
| ordering_customer.address | 38.7% | 44.7% | 97.7% | 97.7% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.7% |
| ordering_institution | 41.3% | 45.0% | 87.7% | 88.0% | 99.7% | 100.0% | 98.7% | 99.3% | 99.7% | 99.7% |
| account_with_institution | 41.7% | 56.3% | 80.3% | 83.3% | 99.7% | 99.7% | 99.0% | 99.3% | 99.3% | 99.7% |
| beneficiary.account | 56.7% | 56.7% | 98.3% | 98.3% | 100.0% | 100.0% | 99.7% | 99.7% | 99.7% | 99.7% |
| beneficiary.name | 66.7% | 66.7% | 98.7% | 98.7% | 100.0% | 100.0% | 99.7% | 99.7% | 99.7% | 99.7% |
| beneficiary.address | 46.0% | 50.0% | 98.0% | 98.0% | 100.0% | 100.0% | 99.3% | 99.3% | 99.7% | 99.7% |
| remittance_info | 67.7% | 70.0% | 97.3% | 97.3% | 100.0% | 100.0% | 98.0% | 98.0% | 95.0% | 95.0% |
| charges | 78.7% | 83.7% | 90.0% | 96.0% | 99.7% | 100.0% | 99.3% | 99.7% | 99.7% | 99.7% |

Commands:

- `base-qwen2.5-0.5b-fewshot`: `python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --name base-qwen2.5-0.5b-fewshot` (2026-11-23)
- `base-qwen2.5-0.5b-fewshot+pp`: `python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --name base-qwen2.5-0.5b-fewshot --postprocess --cache-only` (2026-11-25)
- `ft-qwen2.5-0.5b-lora`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --name ft-qwen2.5-0.5b-lora --ollama-url http://127.0.0.1:21434` (2026-11-23)
- `ft-qwen2.5-0.5b-lora+pp`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --name ft-qwen2.5-0.5b-lora --postprocess --cache-only` (2026-11-25)
- `kaggle-ft-qwen2.5-0.5b-lora-v2`: `python evals/run_eval.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --prompt ft --data data/test.jsonl --name kaggle-ft-qwen2.5-0.5b-lora-v2 --batch-size 16` (2026-11-24)
- `kaggle-ft-qwen2.5-0.5b-lora-v2+pp`: `python evals/run_eval.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --prompt ft --data data/test.jsonl --name kaggle-ft-qwen2.5-0.5b-lora-v2 --postprocess --cache-only` (2026-11-25)
- `cpu-ft-qwen2.5-0.5b-lora-v2`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/test.jsonl --name cpu-ft-qwen2.5-0.5b-lora-v2 --ollama-url http://127.0.0.1:21434` (2026-11-25)
- `cpu-ft-qwen2.5-0.5b-lora-v2+pp`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/test.jsonl --name cpu-ft-qwen2.5-0.5b-lora-v2 --postprocess --cache-only` (2026-11-25)
- `api-claude-haiku-5-5-fewshot`: `python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --workers 3 --name api-claude-haiku-5-5-fewshot` (2026-11-23)
- `api-claude-haiku-5-5-fewshot+pp`: `python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --name api-claude-haiku-5-5-fewshot --postprocess --cache-only` (2026-11-25)
