# Full training run on Kaggle (free GPU)

The build server has no GPU, so the real training run happens on Kaggle. Everything is prepared: one
command starts it. It runs on **your** Kaggle account, as a **private** notebook.

What the run does, top to bottom (`kaggle/kaggle_train.ipynb`, built from the repo by
`notebooks/build_notebook.py`):

1. Installs the pinned Python packages and checks that a GPU is attached.
2. Writes the project's source files, regenerates the synthetic data from its seeds (v1 test, v2
   train/val/test, OOD) and checks every file's SHA-256 against the repo copy. You upload nothing.
3. Trains a LoRA adapter for Qwen2.5-0.5B-Instruct on `data/v2/train.jsonl` (4,800 examples, 2 epochs,
   fp16 LoRA, effective batch 16). Validation loss and validation field accuracy are logged every epoch.
4. Scores the adapter with the repo's scorer on the v1 test split (300), the v2 test split (600) and the
   held-out OOD set (100), each raw and with `postprocess.py`.
5. Saves `adapter.zip`, `metrics.json` and `results.zip` (eval JSONs, raw outputs, train log, loss curve).

Run time: about 1.5 to 3 hours on a T4 or P100. **This is an estimate; it has not been measured.**

## Status on the build server (checked 2026-11-24)

- `~/.kaggle/kaggle.json`: **not present**.
- `KAGGLE_USERNAME` in `~/overnight-build/.env`: **not present**.
- Kaggle CLI: **not installed**.

So the kernel was not pushed. Do the steps below once, on any machine (the VPS or your laptop).

## One-time setup (about 5 minutes)

1. Sign in at kaggle.com. Open **Settings**, section **Phone verification**, and verify your phone.
   Kaggle needs this before a notebook may use a GPU or the internet.
2. Still in **Settings**, section **API**, click **Create New Token**. Your browser downloads `kaggle.json`.
3. Put the file in place and lock it (Linux / WSL / macOS):

   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```

   On Windows: move it to `C:\Users\<you>\.kaggle\kaggle.json`.
4. Install the Kaggle command-line tool into the project's virtual environment:

   ```bash
   .venv/bin/python -m pip install kaggle
   ```

## Start the run (one command)

From the repo root:

```bash
PATH=.venv/bin:$PATH bash kaggle/push.sh
```

It rebuilds the notebook from the current source, fills in your Kaggle username, and pushes it as the
private kernel `<your-username>/mt103-extractor-lora-train` with GPU and internet switched on. Kaggle starts
running it straight away.

Check on it:

```bash
PATH=.venv/bin:$PATH bash kaggle/push.sh status
```

When the status says `complete`, download the outputs into `kaggle/output/`:

```bash
PATH=.venv/bin:$PATH bash kaggle/push.sh output
```

You can also watch it in the browser: kaggle.com, **Code**, **Your Work**, `mt103-extractor-lora-train`.

### Without the command line

kaggle.com, **Create**, **New Notebook**, **File**, **Import Notebook**, upload `kaggle/kaggle_train.ipynb`.
In the right-hand panel set **Accelerator** to **GPU T4 x2** (or P100) and switch **Internet** on. Then
**Save Version**, **Save & Run All (Commit)**. The outputs appear under the version's **Output** tab.

## After the run: bring the results home

1. `unzip kaggle/output/adapter.zip -d models/` so that `models/adapter/adapter_config.json` exists.
   This replaces the CPU smoke adapter (it is in git, so `git checkout models/adapter` brings it back).
2. `unzip kaggle/output/results.zip -d kaggle/output/` and copy the `eval_*kaggle-ft*` files and
   `results/cache/*kaggle-ft*` into `results/`, then `make eval-pp` refreshes the tables in
   `results/eval_table*.md` (the Kaggle rows appear next to the smoke rows).
3. To serve it: `make merge serve`, then `make eval-ft eval-ood` re-measures the GGUF q4_K_M build on CPU.
   The Kaggle numbers are for the fp16 adapter on a GPU; the quantised CPU build can differ slightly.
4. Update the "pending retraining" rows in RESULTS.md and README.md with the measured numbers.

## If something fails

- `No GPU` assertion in the first cells: the notebook was run without an accelerator. Set it to GPU.
- `pip` cannot download: internet is off for the notebook, or the phone is not verified.
- Out of memory: in the settings cell, set `EXTRA_TRAIN_ARGS = "--no-qlora --batch-size 2 --grad-accum 8"`.
- The data check fails (`regenerated data differs`): the notebook was built from a different commit than the
  data files. Run `python notebooks/build_notebook.py` again, then push.
