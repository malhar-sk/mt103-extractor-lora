"""Builds notebooks/kaggle_train.ipynb from the repo's own source files, so the notebook
always trains with exactly the same code/config as `make train`.

    python notebooks/build_notebook.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["schema.py", "prompts.py", "data/__init__.py", "data/generate.py", "evals/__init__.py",
         "evals/run_eval.py", "train/config.yaml", "train/train_lora.py"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def main():
    sha = {n: hashlib.sha256((ROOT / "data" / f"{n}.jsonl").read_bytes()).hexdigest() for n in ("train", "val", "test")}
    cells = [md("""
# mt103-extractor-lora: QLoRA fine-tuning on Kaggle (free GPU)

**What this does:** fine-tunes **Qwen2.5-1.5B-Instruct** with QLoRA (4-bit, rank 16, alpha 32, dropout 0.05,
lr 2e-4 cosine, max length 1024, 3 epochs) to turn messy payment text into MT103 JSON. Same code and config as
`make train` in the repo; this notebook is generated from the repo by `notebooks/build_notebook.py`.

**How to run**
1. kaggle.com, then **Create, New Notebook**, then **File, Import Notebook**, and upload this `.ipynb`.
2. Right sidebar: **Session options, Accelerator, GPU T4 x2**. Turn **Internet on** (needs phone verification).
3. **Run All**. It takes about 1 to 1.5 hours on a T4.
4. When it finishes, open the **Output** panel (`/kaggle/working`) and download **`adapter.zip`** (and
   `train_log.json`, `loss.png` if you want the curves).
5. In the repo: unzip into `models/adapter/` (so `models/adapter/adapter_config.json` exists), then run
   `make merge serve eval-ft` (see TESTING.md).

**Data:** the dataset is SYNTHETIC. It is regenerated here with the repo's seeded generator (seed 103), and the
SHA-256 of each split is checked against the repo's copy, so you don't need to upload anything. If you'd rather
upload `data/*.jsonl` as a Kaggle dataset, set `DATA_DIR` in the data cell to its `/kaggle/input/...` path.
"""),
             code("""
!pip install -q "transformers==5.24.0" "peft==0.22.0" "trl==1.18.2" "datasets" "accelerate" "bitsandbytes" "pydantic>=2" pyyaml matplotlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one T4 is enough; avoids naive model-parallel across 2 GPUs
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
             md("## Project source (written from the repo)")]
    for f in FILES:
        src = (ROOT / f).read_text(encoding="utf-8")
        d = str(Path(f).parent)
        cells.append(code((f"import os; os.makedirs('mt103/{d}', exist_ok=True)\n" if d != "." else "") +
                          f"%%writefile mt103/{f}\n{src}" if d == "." else
                          f"%%writefile mt103/{f}\n{src}"))
    # %%writefile must be the first line of a cell, so create directories up front instead
    cells.insert(3, code("import os\nfor d in ['mt103/data', 'mt103/evals', 'mt103/train', 'mt103/models', 'mt103/results']:\n    os.makedirs(d, exist_ok=True)"))
    cells += [md("## Data (synthetic, regenerated + checked)"),
              code(f"""
%cd /kaggle/working/mt103
!python data/generate.py --n 3000 --seed 103 --out data
import hashlib
EXPECTED = {json.dumps(sha)}
DATA_DIR = "data"
for n, h in EXPECTED.items():
    got = hashlib.sha256(open(f"{{DATA_DIR}}/{{n}}.jsonl", "rb").read()).hexdigest()
    print(n, "OK" if got == h else f"MISMATCH (got {{got[:12]}}): splits differ from the repo's copy")
"""),
              md("## Train (QLoRA, 3 epochs; val loss + val field accuracy logged every epoch)"),
              code("!python train/train_lora.py --mode full --data-dir $DATA_DIR --out models/adapter --results results"),
              code("""
import json
log = json.load(open("results/train_log.json"))
print("train seconds:", log["train_seconds"])
for v in log["val_generation"]:
    print(v)
from IPython.display import Image, display
display(Image("results/loss.png"))
"""),
              md("## Package the adapter for download"),
              code("""
!cd models && zip -qr /kaggle/working/adapter.zip adapter && cd ..
!cp results/train_log.json results/loss.png /kaggle/working/
!ls -lh /kaggle/working/adapter.zip /kaggle/working/train_log.json /kaggle/working/loss.png
print("Download adapter.zip from the Output panel, unzip into models/adapter/ in the repo.")
""")]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    # first source cell must cd to /kaggle/working before writing files
    nb["cells"].insert(2, code("%cd /kaggle/working"))
    out = ROOT / "notebooks" / "kaggle_train.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
