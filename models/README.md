# models/

Large artefacts are git-ignored. Rebuild them with the Makefile.

| Path | In git? | How to get it |
|---|---|---|
| `adapter/` | **yes** (35 MB LoRA weights + config). This is the **full run on v2** from Kaggle (2 epochs, fp16 LoRA, see `adapter/TRAINING_INFO.json`). The earlier CPU smoke adapter is in git history (`git show ea3d4a7:models/adapter/TRAINING_INFO.json`) | `bash kaggle/push.sh` then unzip `adapter.zip` here, or `make train` for a CPU smoke run |
| `base-qwen2.5-0.5b-instruct/` | no (988 MB) | downloaded by `make train` / `train/merge.py` from Hugging Face (Apache-2.0) |
| `merged/` | no | `make merge` (step 1) |
| `gguf/mt103-extractor-q4_K_M.gguf` + `Modelfile` | no (398 MB) | `make gguf` (llama.cpp convert + quantize), then `ollama create mt103-extractor` |
