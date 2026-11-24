# TESTING.md

Copy-paste steps for a Windows laptop with Docker Desktop, Git and Python 3.11. Commands are for
**PowerShell**. `make` targets are shown too, so use them if you have make (WSL or Git Bash with make).
Without make, run the plain commands underneath each one.

> **Which container tool?** These steps use Docker: Docker Desktop on Windows or macOS, or Docker Engine
> on Linux. With **Podman**, install `podman-compose` and replace `docker compose` with `podman-compose`
> (and `docker` with `podman`). OrbStack on macOS works with the commands as written.
> Details: `docs/adr/0001-container-runtime.md`.

> **What happened on the build machine:** the VPS has **no GPU**, so `make train` ran a **CPU smoke
> training** (150 steps, about 0.125 epochs, Qwen2.5-0.5B, 88 min). That adapter was merged, converted to
> GGUF q4_K_M and evaluated. It is labelled "CPU smoke run, not fully trained" everywhere. The full
> QLoRA run (Qwen2.5-1.5B, 3 epochs) is in `notebooks/kaggle_train.ipynb` and is meant for Kaggle's free GPU
> (step 6). All data is **synthetic**.

## 1. Clone and set up Python

```powershell
git clone https://github.com/malhar-sk/mt103-extractor-lora.git
cd mt103-extractor-lora
copy .env.example .env
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
You should see pip finish without errors. If port 21434 or 18000 is taken on your laptop, change
`OLLAMA_HOST_PORT` / `API_HOST_PORT` in `.env`.

## 2. Run the tests (no model or GPU needed; the LLM is mocked)

```powershell
.venv\Scripts\python -m pytest -q tests          # or: make test
```
Expected: `50 passed` (schema, 100% target validity, split overlap, generator determinism, eval on
5 examples with a mock model, and the `/extract` API with a mocked Ollama).

## 3. Start Ollama + the API, pull the base model

```powershell
docker compose up -d --build                     # or: make up
docker compose exec ollama ollama pull qwen2.5:0.5b-instruct
docker compose ps
```
`mt103-ollama` should be `healthy` and `mt103-api` should be `Up`. Check it:
```powershell
curl.exe http://127.0.0.1:18000/health
```
You'll get `{"status":"ok", ... "fine_tuned_present": false}` (the fine-tuned model doesn't exist on your
laptop yet).

## 4. Regenerate the dataset (optional; it's committed) and run the base-model baseline

```powershell
.venv\Scripts\python data/generate.py --n 3000 --seed 103          # or: make data
git status data/          # nothing changed: the generator is deterministic
.venv\Scripts\python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --name base-qwen2.5-0.5b-fewshot --ollama-url http://127.0.0.1:21434
.venv\Scripts\python evals/report.py                                 # or: make eval-base (both lines)
```
The eval reads answers cached in `results/cache/`, so it finishes instantly and prints the summary JSON.
`results/eval_table.md` then holds the base vs fine-tuned vs API table. To re-run the model for real, add
`--no-cache` (about 15 s per example on a 1-CPU budget, so use `--limit 20`).

## 5. Try `/extract` with a messy email

The API falls back to the base model when the fine-tuned one is missing, and says so in the response:
```powershell
curl.exe -s -X POST http://127.0.0.1:18000/extract -H "Content-Type: application/json" -d '{\"text\": \"Hi team, pls send Rs 1,25,000.50 to Coromandel Logistics Pvt Ltd a/c 54150945237 at HDFCINBB, value 05/12/2026, our ref TXN778812, charges shared. Remitter: Helix Pharma Ltd a/c 726045614514. Invoice INV 2231. Thx\"}'
```
You'll see `fields` (amount `125000.5`, currency `INR`, `value_date` `2026-12-05`, ...), `errors`,
`validation_errors`, `model_used`, `fallback: true` with a `note`, and `latency_s`. A recorded run with the
fine-tuned model is in `examples/extract_example.json`.

## 6. Full training on Kaggle (free GPU)

1. kaggle.com, **Create, New Notebook**, then **File, Import Notebook**, and upload `notebooks/kaggle_train.ipynb`.
2. **Session options**: Accelerator **GPU T4 x2**, Internet **on**.
3. **Run All** (about 1 to 1.5 h). Download `adapter.zip` from the Output panel.
4. Unzip it so that `models\adapter\adapter_config.json` exists.

## 7. Merge, GGUF, serve, and evaluate the fine-tuned model

You need the training extras and llama.cpp (cmake and a C++ compiler; WSL is easiest on Windows):
```bash
.venv/bin/python -m pip install -r requirements-train.txt
make merge        # adapter -> models/merged -> llama.cpp f16 GGUF -> q4_K_M -> `ollama create mt103-extractor`
make serve        # API now reports "fine_tuned_present": true and "fallback": false
make eval-ft      # writes results/eval_ft-qwen2.5-0.5b-lora.json and refreshes results/eval_table.md
```
`make merge` prints `success` after `ollama create`. A Kaggle 1.5B adapter has a different base: train/merge.py
reads the base from `adapter_config.json`, so it pulls Qwen2.5-1.5B-Instruct from Hugging Face automatically.
Name the run differently (e.g. `--name ft-qwen2.5-1.5b-qlora`) so it doesn't overwrite the smoke row.

## 8. Compare

Open `results/eval_table.md` (or `RESULTS.md`): base vs fine-tuned vs Claude Haiku, on field accuracy,
valid-JSON rate, error-flag precision/recall and tokens/sec.

### 8b. Eval round 2: OOD set, per-field errors, error flags (optional, about 1 h on CPU)

```bash
python data/generate_ood.py --n 100 --seed 20261123
```
You should see a JSON manifest with `"n": 100`, 4 styles and 43 examples with errors. The file is identical to the
committed `data/ood_test.jsonl` (a test checks this). Then, with Ollama up and the fine-tuned model created (step 7):

```bash
make eval-ood     # base + fine-tuned via Ollama; Haiku needs a logged-in `claude` CLI (cached results are committed)
make analyze      # no model calls: reads results/cache/*.jsonl
```
You should see `results/eval_table_ood.md` (fine-tuned field accuracy about 66.7%) and `results/analysis_test.md` /
`results/analysis_ood.md` with per-field error types and a per-error-code table (ok / wrong / repaired / copied / null).
On Windows without `make`, copy the commands under `eval-ood:` and `analyze:` in the Makefile. `make analyze` alone
works straight after cloning, because the cached outputs are committed.

## 9. Clean up

```powershell
docker compose down
```

### Optional: reproduce the CPU smoke training (Linux/WSL, about 1.5 h on 2 vCPU)
```bash
make venv-train && make train      # writes models/adapter, results/loss.png, results/train_log.json
```
