# Linux/macOS/WSL. On Windows without make, run the commands under each target by hand (see TESTING.md).
PY ?= .venv/bin/python
OLLAMA_URL ?= http://127.0.0.1:21434
LLAMA_CPP ?= third_party/llama.cpp
export OMP_NUM_THREADS ?= 1

.PHONY: help venv up down data test eval eval-base eval-api eval-ft train merge gguf serve llama-cpp bench

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

venv:
	python3.11 -m venv .venv || python -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

up:                      ## start Ollama + API, pull the base model
	docker compose up -d --build
	docker compose exec ollama ollama pull qwen2.5:0.5b-instruct

down:
	docker compose down

data:                    ## regenerate the synthetic dataset (deterministic, seed 103)
	$(PY) data/generate.py --n 3000 --seed 103

test:
	$(PY) -m pytest -q tests

eval: eval-base eval-ft

eval-base:               ## base Qwen2.5-0.5B (Ollama q4_K_M) with the few-shot prompt, 300 test examples
	$(PY) evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --name base-qwen2.5-0.5b-fewshot --ollama-url $(OLLAMA_URL)
	$(PY) evals/report.py

eval-api:                ## reference ceiling via the `claude` CLI (cached in results/cache)
	$(PY) evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --workers 3 --name api-claude-haiku-5-5-fewshot
	$(PY) evals/report.py

eval-ft:                 ## fine-tuned model served by Ollama, short prompt
	$(PY) evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --name ft-qwen2.5-0.5b-lora --ollama-url $(OLLAMA_URL)
	$(PY) evals/report.py

bench:                   ## tokens/sec on this machine, 20 examples each, nothing else running
	$(PY) evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --limit 20 --no-cache --name speed-base --ollama-url $(OLLAMA_URL)
	$(PY) evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --limit 20 --no-cache --name speed-ft --ollama-url $(OLLAMA_URL)

train:                   ## LoRA: full QLoRA on GPU, ~150-step smoke run on CPU
	$(PY) train/train_lora.py

merge:                   ## adapter -> merged HF model -> GGUF q4_K_M -> `ollama create mt103-extractor`
	$(PY) train/merge.py --adapter models/adapter --out models/merged
	$(MAKE) gguf

llama-cpp:
	test -d $(LLAMA_CPP) || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $(LLAMA_CPP)
	test -x $(LLAMA_CPP)/build/bin/llama-quantize || (cmake -S $(LLAMA_CPP) -B $(LLAMA_CPP)/build -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF && cmake --build $(LLAMA_CPP)/build --target llama-quantize -j2)

gguf: llama-cpp
	mkdir -p models/gguf
	PYTHONPATH=$(LLAMA_CPP)/gguf-py $(PY) $(LLAMA_CPP)/convert_hf_to_gguf.py models/merged --outtype f16 --outfile models/gguf/mt103-extractor-f16.gguf
	$(LLAMA_CPP)/build/bin/llama-quantize models/gguf/mt103-extractor-f16.gguf models/gguf/mt103-extractor-q4_K_M.gguf Q4_K_M
	$(PY) -c "from prompts import FT_SYSTEM; t=open('serve/Modelfile.tmpl').read().replace('__FT_SYSTEM__', FT_SYSTEM); open('models/gguf/Modelfile','w').write(t)"
	docker compose exec ollama ollama create mt103-extractor -f /models/gguf/Modelfile

serve:                   ## API on http://127.0.0.1:18000 (uses fine-tuned model if present, else base)
	docker compose up -d --build api
	@echo "POST http://127.0.0.1:18000/extract"
