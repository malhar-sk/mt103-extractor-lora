| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Base Qwen2.5-0.5B-Instruct (Ollama q4_K_M, 4-shot prompt) | 100 | 97.0% | 62.0% | 47.5% | 0.0% | 0.0% | 0.0% | 11.2 | 19.4 |
| Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained** | 100 | 99.0% | 36.0% | 66.7% | 5.0% | 2.2% | 2.3% | 9.0 | 24.8 |
| Claude Haiku 5.5 via API (4-shot prompt), reference ceiling | 100 | 100.0% | 100.0% | 98.1% | 78.0% | 90.5% | 88.4% | 5.8 | 230.8 |

Per-field exact match:

| Field | base-qwen2.5-0.5b-fewshot | ft-qwen2.5-0.5b-lora | api-claude-haiku-5-5-fewshot |
|---|---|---|---|
| sender_reference | 68.0% | 61.0% | 100.0% |
| bank_operation_code | 50.0% | 79.0% | 100.0% |
| value_date | 55.0% | 64.0% | 99.0% |
| currency | 75.0% | 85.0% | 99.0% |
| amount | 75.0% | 76.0% | 90.0% |
| ordering_customer.account | 39.0% | 59.0% | 97.0% |
| ordering_customer.name | 48.0% | 71.0% | 100.0% |
| ordering_customer.address | 38.0% | 76.0% | 100.0% |
| ordering_institution | 24.0% | 39.0% | 96.0% |
| account_with_institution | 25.0% | 46.0% | 94.0% |
| beneficiary.account | 32.0% | 67.0% | 100.0% |
| beneficiary.name | 36.0% | 67.0% | 100.0% |
| beneficiary.address | 42.0% | 75.0% | 100.0% |
| remittance_info | 52.0% | 73.0% | 100.0% |
| charges | 54.0% | 62.0% | 97.0% |

Commands:

- `ood-base-qwen2.5-0.5b-fewshot`: `python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --data data/ood_test.jsonl --name ood-base-qwen2.5-0.5b-fewshot --ollama-url http://127.0.0.1:21434` (2026-11-23)
- `ood-ft-qwen2.5-0.5b-lora`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/ood_test.jsonl --name ood-ft-qwen2.5-0.5b-lora --ollama-url http://127.0.0.1:21434` (2026-11-23)
- `ood-api-claude-haiku-5-5-fewshot`: `python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --workers 3 --data data/ood_test.jsonl --name ood-api-claude-haiku-5-5-fewshot` (2026-11-23)
