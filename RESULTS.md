# RESULTS.md

Every number here was measured on the build VPS on **2026-11-23** (2 vCPU shared, no GPU; Ollama container capped at
`cpus: 1.0`, `num_thread 1`). Dataset: **test split, 300 synthetic examples** (`data/test.jsonl`, seed 103) unless
stated otherwise. Raw model outputs are in `results/cache/*.jsonl`; full metrics are in `results/eval_*.json`.

## Headline numbers

| Metric | Base Qwen2.5-0.5B (4-shot) | Fine-tuned 0.5B LoRA (**CPU smoke run, not fully trained**) |
|---|---|---|
| Field exact-match accuracy (macro, 15 fields) | **62.6%** | **94.2%** |
| Valid-JSON rate | 100.0% | 100.0% |
| Schema-valid rate | 59.0% | 83.3% |
| Tokens/sec on CPU (Ollama q4_K_M, 1 thread) | 13.5 | 22.4 |

The tokens/sec gap is mostly the prompt, not the model. Both are the same 0.5B architecture at q4_K_M, but the base
model gets a long 4-shot prompt (`num_ctx 4096`) and the fine-tuned one a one-line prompt (`num_ctx 2048`).

## Full eval table (test split, n = 300)

| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Base Qwen2.5-0.5B-Instruct (Ollama q4_K_M, 4-shot prompt) | 300 | 100.0% | 59.0% | 62.6% | 1.0% | 0.0% | 0.0% | 13.0 | 13.5 |
| Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained** | 300 | 100.0% | 83.3% | 94.2% | 51.7% | 5.3% | 7.2% | 9.3 | 22.4 |
| Claude Haiku 5.5 via API (4-shot prompt), reference ceiling | 300 | 100.0% | 97.0% | 99.1% | 91.3% | 98.5% | 97.1% | 5.6 | 213.9 |

Per-field exact match:

| Field | base-qwen2.5-0.5b-fewshot | ft-qwen2.5-0.5b-lora | api-claude-haiku-5-5-fewshot |
|---|---|---|---|
| sender_reference | 96.7% | 99.3% | 99.7% |
| bank_operation_code | 49.7% | 89.7% | 99.3% |
| value_date | 60.7% | 91.0% | 99.3% |
| currency | 93.7% | 96.7% | 99.7% |
| amount | 82.0% | 90.0% | 97.0% |
| ordering_customer.account | 56.0% | 98.3% | 99.7% |
| ordering_customer.name | 63.7% | 100.0% | 99.7% |
| ordering_customer.address | 38.7% | 97.7% | 99.7% |
| ordering_institution | 41.3% | 87.7% | 99.7% |
| account_with_institution | 41.7% | 80.3% | 99.3% |
| beneficiary.account | 56.7% | 98.3% | 99.7% |
| beneficiary.name | 66.7% | 98.7% | 99.7% |
| beneficiary.address | 46.0% | 98.0% | 99.7% |
| remittance_info | 67.7% | 97.3% | 95.0% |
| charges | 78.7% | 90.0% | 99.7% |

Commands:

- `base-qwen2.5-0.5b-fewshot`: `python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --name base-qwen2.5-0.5b-fewshot` (2026-11-23)
- `ft-qwen2.5-0.5b-lora`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --name ft-qwen2.5-0.5b-lora --ollama-url http://127.0.0.1:21434` (2026-11-23)
- `api-claude-haiku-5-5-fewshot`: `python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --workers 3 --name api-claude-haiku-5-5-fewshot` (2026-11-23)

Commands are run from the repo root with `.venv/bin/python`, and the Ollama URL is `http://127.0.0.1:21434`. The table
is produced by `python evals/report.py`. Exact match is case- and whitespace-normalised, and amounts are compared to 2 dp.
"Err precision/recall" counts (code, field) pairs in the `errors` list against the generator's ground truth.

### Reading it honestly
- The smoke run (300 training examples seen, 0.125 epoch) already learnt the **format and field mapping** (62.6% → 94.2%),
  but it has **barely learnt error flagging** (precision 5.3%, recall 7.2%). It tends to "repair" impossible values,
  e.g. `USX` → `USD`, instead of flagging them (see `examples/extract_example.json`). The API still catches many of
  these, because pydantic validation reports them in `validation_errors`. A full GPU run is needed for this.
- The Claude Haiku row is a **reference ceiling**. It was called through the logged-in `claude` CLI (`--backend claude-cli`),
  because `ANTHROPIC_API_KEY` was empty on the host, so its latency includes CLI start-up (about 0.6 s) and network, and its
  tokens/sec is not comparable to local CPU numbers.

## Training (M3): CPU smoke run, not fully trained

| Item | Value |
|---|---|
| Mode | `smoke` (no GPU: `nvidia-smi` not present) |
| Base | Qwen2.5-0.5B-Instruct, bf16, LoRA r=16 alpha=32 dropout=0.05, all attention+MLP projections |
| Steps | 150 (batch 1 × grad-accum 2, so 300 of 2,400 train examples, 0.125 epoch), lr 2e-4 cosine, max len 1024 |
| Wall time | 5,280 s (88 min) |
| Train loss (completion tokens) | 1.158 (step 5) → 0.023 (step 150) |
| Val loss (40 val examples) | 0.063 (step 50), 0.037 (step 100), 0.030 (step 150) |
| Val generation (20 val examples, step 150) | valid JSON 100%, schema-valid 80%, field acc 94.0%, err P/R 0% / 0% |
| Command | `make train` (= `python train/train_lora.py`) |
| Artefacts | `results/loss.png`, `results/train_log.json`, `models/adapter/` (35 MB, committed) |

"Validation metric per epoch": the smoke run is shorter than one epoch, so val metrics are logged every 50 steps instead.
The full 3-epoch QLoRA run on Qwen2.5-1.5B (`notebooks/kaggle_train.ipynb`) is **NOT MEASURED**: there is no GPU on the
build host, and it is meant for Kaggle (TESTING.md step 6).

## Serving (M4)

- Merge, then GGUF f16, then q4_K_M (398 MB), then `ollama create mt103-extractor`: done on the VPS with `make merge`
  (llama.cpp built from source, `llama-quantize` 9.2 s).
- Recorded `/extract` calls with the fine-tuned model: `examples/extract_example.json` (latency 10.2 s and 8.2 s,
  22.4 and 23.4 tok/s, `fallback: false`).
- Fallback to the base model when the fine-tuned model is absent: covered by `tests/test_api.py` (mocked Ollama).

## Not measured / skipped
- **Base Qwen2.5-1.5B baseline:** NOT MEASURED. The plan asks for it only with a GPU, and there is none.
- **Fine-tuned 1.5B (full QLoRA):** NOT MEASURED, for the same reason. The notebook is provided.
- **M5 (investment-committee integration):** skipped: source not provided (`~/inputs/multi-invest-committee` does not exist).

## Tests
`make test` gives **33 passed** (2026-11-23). This covers schema validation, 100% of the 3,000 targets being
schema-valid, no split overlap, generator determinism, the eval script on 5 examples with a mock model, and the
API with a mocked Ollama.
