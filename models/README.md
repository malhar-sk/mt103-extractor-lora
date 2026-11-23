# models/

Large artefacts are git-ignored. Rebuild them with the Makefile.

| Path | In git? | How to get it |
|---|---|---|
| `adapter/` | **yes** (35 MB LoRA weights + config). This is the **CPU smoke run, not fully trained** (150 steps) | `make train`, or unzip the Kaggle `adapter.zip` here |
| `base-qwen2.5-0.5b-instruct/` | no (988 MB) | downloaded by `make train` / `train/merge.py` from Hugging Face (Apache-2.0) |
| `merged/` | no | `make merge` (step 1) |
| `gguf/mt103-extractor-q4_K_M.gguf` + `Modelfile` | no (398 MB) | `make gguf` (llama.cpp convert + quantize), then `ollama create mt103-extractor` |
