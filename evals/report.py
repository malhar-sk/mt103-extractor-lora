"""Collect results/eval_*.json into results/eval_table.md (pasted into RESULTS.md)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.run_eval import SCORED_FIELDS  # noqa: E402

ORDER = ["base-qwen2.5-0.5b-fewshot", "ft-qwen2.5-0.5b-lora", "api-claude-haiku-5-5-fewshot"]
LABEL = {"base-qwen2.5-0.5b-fewshot": "Base Qwen2.5-0.5B-Instruct (Ollama q4_K_M, 4-shot prompt)",
         "ft-qwen2.5-0.5b-lora": "Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained**",
         "api-claude-haiku-5-5-fewshot": "Claude Haiku 5.5 via API (4-shot prompt), reference ceiling"}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def num(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}"


def main(results=ROOT / "results", prefix="", out_name="eval_table.md"):
    """prefix="" -> main test-split table; prefix="ood-" -> out-of-distribution table (eval_table_ood.md)."""
    runs = {}
    for p in sorted(Path(results).glob("eval_*.json")):
        s = json.loads(p.read_text())["summary"]
        if prefix:
            if not s["name"].startswith(prefix):
                continue
            s = dict(s, name=s["name"][len(prefix):])
        elif s["name"].startswith("ood-"):
            continue
        runs[s["name"]] = s
    names = [n for n in ORDER if n in runs] + [n for n in runs if n not in ORDER and not n.startswith(("speed-", "mock", "probe"))]
    lines = ["| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = runs[n]
        lines.append(f"| {LABEL.get(n, n)} | {s['n']} | {pct(s['valid_json_rate'])} | {pct(s['schema_valid_rate'])} | "
                     f"{pct(s['field_accuracy_macro'])} | {pct(s['all_fields_correct_rate'])} | {pct(s['error_precision'])} | "
                     f"{pct(s['error_recall'])} | {num(s['latency_p50_s'])} | {num(s['tokens_per_sec'])} |")
    lines += ["", "Per-field exact match:", "", "| Field | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for f in SCORED_FIELDS:
        lines.append(f"| {f} | " + " | ".join(pct(runs[n]["field_accuracy"][f]) for n in names) + " |")
    lines += ["", "Commands:", ""] + [f"- `{prefix}{n}`: `python {runs[n]['command']}` ({runs[n]['date']})" for n in names]
    out = Path(results) / out_name
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
    if any(Path(ROOT / "results").glob("eval_ood-*.json")):
        main(prefix="ood-", out_name="eval_table_ood.md")
