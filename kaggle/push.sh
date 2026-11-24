#!/usr/bin/env bash
# Push and start the Kaggle training kernel (runs on YOUR Kaggle account; private kernel, GPU, internet on).
#   bash kaggle/push.sh            # push + start
#   bash kaggle/push.sh status     # is it still running?
#   bash kaggle/push.sh output     # download adapter.zip, results.zip, metrics.json into kaggle/output/
# Needs the Kaggle CLI (pip install kaggle) and an API token in ~/.kaggle/kaggle.json, or the
# KAGGLE_USERNAME + KAGGLE_KEY environment variables. The token is read by the CLI only; this script
# only reads the username, to fill in the kernel id.
set -euo pipefail
cd "$(dirname "$0")/.."
user="${KAGGLE_USERNAME:-}"
if [ -z "$user" ] && [ -f "$HOME/.kaggle/kaggle.json" ]; then
  user="$(python3 -c 'import json,os; print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')"
fi
[ -n "$user" ] || { echo "No Kaggle username: put kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME (see docs/kaggle.md)"; exit 1; }
command -v kaggle >/dev/null || { echo "Kaggle CLI missing: pip install kaggle"; exit 1; }
slug="$user/mt103-extractor-lora-train"
case "${1:-push}" in
  status) kaggle kernels status "$slug" ;;
  output) mkdir -p kaggle/output && kaggle kernels output "$slug" -p kaggle/output && ls -l kaggle/output ;;
  push)
    python3 notebooks/build_notebook.py
    tmp="$(mktemp -d)"
    cp kaggle/kaggle_train.ipynb "$tmp/"
    sed "s#KAGGLE_USERNAME/#$user/#" kaggle/kernel-metadata.json > "$tmp/kernel-metadata.json"
    kaggle kernels push -p "$tmp"
    rm -rf "$tmp"
    echo "Started. Check with: bash kaggle/push.sh status   (about 1.5 to 3 hours on a T4; estimate, not measured)"
    ;;
  *) echo "usage: bash kaggle/push.sh [push|status|output]"; exit 2 ;;
esac
