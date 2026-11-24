"""Builds the Kaggle kernel notebook (kaggle/kaggle_train.ipynb, also copied to notebooks/kaggle_train.ipynb)
from the repo's own source files, so Kaggle trains and evaluates with exactly the code in the repo.

    python notebooks/build_notebook.py

The notebook: installs pinned deps, writes the project source, regenerates the SYNTHETIC data
(v1 test, v2 train/val/test, OOD) and checks each file's SHA-256 against the repo copy, trains the LoRA
on data/v2, then scores the adapter on the v1 test split, the v2 test split and the held-out OOD set,
with and without postprocess.py, and saves adapter + metrics to /kaggle/working.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["schema.py", "prompts.py", "postprocess.py", "data/__init__.py", "data/generate.py", "data/generate_v2.py",
         "data/generate_ood.py", "evals/__init__.py", "evals/run_eval.py", "evals/hf_backend.py",
         "train/config.yaml", "train/train_lora.py"]
DATA = ["data/test.jsonl", "data/ood_test.jsonl", "data/v2/train.jsonl", "data/v2/val.jsonl", "data/v2/test.jsonl"]
RUN = "kaggle-ft-qwen2.5-0.5b-lora-v2"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def main():
    sha = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in DATA}
    cells = [md("""
# mt103-extractor-lora: LoRA training + ID/OOD evals on a Kaggle GPU

Trains a LoRA adapter for **Qwen2.5-0.5B-Instruct** (the model the repo serves through Ollama) on the
format-diverse SYNTHETIC training set **v2** (`data/generate_v2.py`, 4,800 train examples), then scores it with
the repo's own scorer on:

* the v1 test split (300, in-distribution for the original templates),
* the v2 test split (600),
* the held-out OOD set (100, `data/generate_ood.py`; no v2 template is derived from it),

each with and without the deterministic rules in `postprocess.py`. Outputs (in `/kaggle/working`):
`adapter.zip`, `metrics.json`, `results.zip` (eval JSONs, raw outputs, train log, loss curve).

Generated from the repo by `notebooks/build_notebook.py`; do not edit by hand. Settings are in the next cell.
"""),
             code("""
# ---- settings (defaults = what the repo's docs/kaggle.md describes) ----
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # the served model; "Qwen/Qwen2.5-1.5B-Instruct" also works (slower)
EPOCHS = 2                                   # over 4,800 v2 train examples
EXTRA_TRAIN_ARGS = "--no-qlora --batch-size 4 --grad-accum 4"   # fp16 LoRA, effective batch 16 (fits T4/P100)
EVAL_BATCH = 16
"""),
             code("""
!pip install -q "transformers==5.24.0" "peft==0.22.0" "trl==1.18.2" "datasets" "accelerate" "pydantic>=2" pyyaml matplotlib
import os, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one GPU is enough; avoids naive model-parallel across 2 GPUs
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
import torch
assert torch.cuda.is_available(), "No GPU: in the notebook settings choose Accelerator = GPU (T4 or P100)"
"""),
             code("%cd /kaggle/working"),
             code("import os\nfor d in ['mt103/data/v2', 'mt103/evals', 'mt103/train', 'mt103/models', 'mt103/results']:\n"
                  "    os.makedirs(d, exist_ok=True)"),
             md("## Project source (written from the repo)")]
    for f in FILES:
        src = (ROOT / f).read_text(encoding="utf-8")
        cells.append(code(f"%%writefile mt103/{f}\n{src}"))
    cells += [md("## Data: SYNTHETIC, regenerated from the seeds, checked against the repo copies"),
              code(f"""
%cd /kaggle/working/mt103
!python data/generate.py --n 3000 --seed 103 --out data
!python data/generate_v2.py --n 6000 --seed 2026 --out data/v2
!python data/generate_ood.py --n 100 --seed 20261123 --out data/ood_test.jsonl
import hashlib
EXPECTED = {json.dumps(sha, indent=1)}
bad = [p for p, h in EXPECTED.items() if hashlib.sha256(open(p, "rb").read()).hexdigest() != h]
assert not bad, f"regenerated data differs from the repo copy: {{bad}}"
print("all data files match the repo (SHA-256)")
"""),
              md("## Train (LoRA on data/v2; val loss + val field accuracy logged every epoch)"),
              code("""
!python train/train_lora.py --mode full --base-model {BASE_MODEL} --data-dir data/v2 --epochs {EPOCHS} {EXTRA_TRAIN_ARGS} --out models/adapter --results results
import json
log = json.load(open("results/train_log.json"))
print("train seconds:", log["train_seconds"])
for v in log["val_generation"]:
    print(v)
from IPython.display import Image, display
display(Image("results/loss.png"))
"""),
              md("## Evaluate: v1 test, v2 test, OOD; raw and + postprocess.py"),
              code(f"""
RUN = "{RUN}"
SETS = {{"": "data/test.jsonl", "v2-": "data/v2/test.jsonl", "ood-": "data/ood_test.jsonl"}}
for prefix, path in SETS.items():
    base = f"python evals/run_eval.py --backend hf --model {{BASE_MODEL}} --adapter models/adapter --prompt ft --data {{path}} --name {{prefix}}{{RUN}}"
    !{{base}} --batch-size {{EVAL_BATCH}} 2>&1 | tail -n 3
    !{{base}} --postprocess --cache-only 2>&1 | tail -n 3
import json, glob
metrics = {{}}
for p in sorted(glob.glob(f"results/eval_*{{RUN}}*.json")):
    s = json.load(open(p))["summary"]
    metrics[s["name"]] = {{k: s[k] for k in ("n", "valid_json_rate", "schema_valid_rate", "field_accuracy_macro",
                                             "all_fields_correct_rate", "error_precision", "error_recall",
                                             "field_accuracy", "data")}}
    print(f"{{s['name']:55s}} n={{s['n']:4d}} field acc={{100 * s['field_accuracy_macro']:.1f}}%  "
          f"err P/R={{s['error_precision']}}/{{s['error_recall']}}")
json.dump({{"base_model": BASE_MODEL, "epochs": EPOCHS, "train_args": EXTRA_TRAIN_ARGS,
           "gpu": torch.cuda.get_device_name(0), "train_seconds": log["train_seconds"],
           "val_generation": log["val_generation"], "evals": metrics}},
          open("/kaggle/working/metrics.json", "w"), indent=2)
"""),
              md("## Package outputs"),
              code("""
!cd models && zip -qr /kaggle/working/adapter.zip adapter && cd ..
!zip -qr /kaggle/working/results.zip results
!ls -lh /kaggle/working/adapter.zip /kaggle/working/results.zip /kaggle/working/metrics.json
print("Download adapter.zip, results.zip and metrics.json from the Output tab (or `kaggle kernels output`).")
""")]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "kaggle" / "kaggle_train.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(out, ROOT / "notebooks" / "kaggle_train.ipynb")  # same notebook for manual upload
    print(f"wrote {out} and notebooks/kaggle_train.ipynb ({len(cells)} cells)")


if __name__ == "__main__":
    main()
