| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Base Qwen2.5-0.5B-Instruct (Ollama q4_K_M, 4-shot prompt) | 100 | 97.0% | 62.0% | 47.5% | 0.0% | 0.0% | 0.0% | 11.2 | 19.4 |
| Base Qwen2.5-0.5B, 4-shot **+ rules** (`postprocess.py`, same raw outputs) | 100 | 97.0% | 93.0% | 53.8% | 0.0% | 63.0% | 39.5% | 11.2 | 19.4 |
| Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained** | 100 | 99.0% | 36.0% | 66.7% | 5.0% | 2.2% | 2.3% | 9.0 | 24.8 |
| Fine-tuned 0.5B LoRA (CPU smoke run) **+ rules** (`postprocess.py`, same raw outputs) | 100 | 99.0% | 88.0% | 69.5% | 6.0% | 42.9% | 48.8% | 9.0 | 24.8 |
| Claude Haiku 5.5 via API (4-shot prompt), reference ceiling | 100 | 100.0% | 100.0% | 98.1% | 78.0% | 90.5% | 88.4% | 5.8 | 230.8 |
| Claude Haiku 5.5 **+ rules** (`postprocess.py`, same raw outputs) | 100 | 100.0% | 100.0% | 98.3% | 80.0% | 92.9% | 90.7% | 5.8 | 230.8 |

Per-field exact match:

| Field | base-qwen2.5-0.5b-fewshot | base-qwen2.5-0.5b-fewshot+pp | ft-qwen2.5-0.5b-lora | ft-qwen2.5-0.5b-lora+pp | api-claude-haiku-5-5-fewshot | api-claude-haiku-5-5-fewshot+pp |
|---|---|---|---|---|---|---|
| sender_reference | 68.0% | 68.0% | 61.0% | 61.0% | 100.0% | 100.0% |
| bank_operation_code | 50.0% | 96.0% | 79.0% | 79.0% | 100.0% | 100.0% |
| value_date | 55.0% | 65.0% | 64.0% | 78.0% | 99.0% | 99.0% |
| currency | 75.0% | 82.0% | 85.0% | 93.0% | 99.0% | 100.0% |
| amount | 75.0% | 77.0% | 76.0% | 79.0% | 90.0% | 91.0% |
| ordering_customer.account | 39.0% | 39.0% | 59.0% | 59.0% | 97.0% | 97.0% |
| ordering_customer.name | 48.0% | 48.0% | 71.0% | 71.0% | 100.0% | 100.0% |
| ordering_customer.address | 38.0% | 43.0% | 76.0% | 76.0% | 100.0% | 100.0% |
| ordering_institution | 24.0% | 24.0% | 39.0% | 39.0% | 96.0% | 96.0% |
| account_with_institution | 25.0% | 43.0% | 46.0% | 53.0% | 94.0% | 95.0% |
| beneficiary.account | 32.0% | 33.0% | 67.0% | 68.0% | 100.0% | 100.0% |
| beneficiary.name | 36.0% | 36.0% | 67.0% | 67.0% | 100.0% | 100.0% |
| beneficiary.address | 42.0% | 45.0% | 75.0% | 75.0% | 100.0% | 100.0% |
| remittance_info | 52.0% | 52.0% | 73.0% | 73.0% | 100.0% | 100.0% |
| charges | 54.0% | 56.0% | 62.0% | 72.0% | 97.0% | 97.0% |

Commands:

- `ood-base-qwen2.5-0.5b-fewshot`: `python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --data data/ood_test.jsonl --name ood-base-qwen2.5-0.5b-fewshot --ollama-url http://127.0.0.1:21434` (2026-11-23)
- `ood-base-qwen2.5-0.5b-fewshot+pp`: `python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --data data/ood_test.jsonl --name ood-base-qwen2.5-0.5b-fewshot --postprocess --cache-only` (2026-11-24)
- `ood-ft-qwen2.5-0.5b-lora`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/ood_test.jsonl --name ood-ft-qwen2.5-0.5b-lora --ollama-url http://127.0.0.1:21434` (2026-11-23)
- `ood-ft-qwen2.5-0.5b-lora+pp`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/ood_test.jsonl --name ood-ft-qwen2.5-0.5b-lora --postprocess --cache-only` (2026-11-24)
- `ood-api-claude-haiku-5-5-fewshot`: `python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --workers 3 --data data/ood_test.jsonl --name ood-api-claude-haiku-5-5-fewshot` (2026-11-23)
- `ood-api-claude-haiku-5-5-fewshot+pp`: `python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --data data/ood_test.jsonl --name ood-api-claude-haiku-5-5-fewshot --postprocess --cache-only` (2026-11-24)
