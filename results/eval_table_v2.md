| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained** | 150 | 100.0% | 58.7% | 80.4% | 16.0% | 1.4% | 2.0% | 9.6 | 22.1 |
| Fine-tuned 0.5B LoRA (CPU smoke run) **+ rules** (`postprocess.py`, same raw outputs) | 150 | 100.0% | 98.0% | 82.0% | 18.0% | 36.4% | 48.0% | 9.6 | 22.1 |

Per-field exact match:

| Field | ft-qwen2.5-0.5b-lora | ft-qwen2.5-0.5b-lora+pp |
|---|---|---|
| sender_reference | 82.7% | 82.7% |
| bank_operation_code | 80.7% | 80.7% |
| value_date | 88.0% | 94.0% |
| currency | 92.7% | 96.0% |
| amount | 88.7% | 91.3% |
| ordering_customer.account | 86.7% | 86.7% |
| ordering_customer.name | 84.7% | 84.7% |
| ordering_customer.address | 75.3% | 75.3% |
| ordering_institution | 63.3% | 64.0% |
| account_with_institution | 58.7% | 62.7% |
| beneficiary.account | 81.3% | 82.0% |
| beneficiary.name | 83.3% | 83.3% |
| beneficiary.address | 88.7% | 88.7% |
| remittance_info | 86.0% | 86.0% |
| charges | 66.0% | 71.3% |

Commands:

- `v2-ft-qwen2.5-0.5b-lora`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/v2/test.jsonl --limit 150 --name v2-ft-qwen2.5-0.5b-lora` (2026-11-24)
- `v2-ft-qwen2.5-0.5b-lora+pp`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/v2/test.jsonl --limit 150 --name v2-ft-qwen2.5-0.5b-lora --postprocess --cache-only` (2026-11-24)
