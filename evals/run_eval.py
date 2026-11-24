"""Shared evaluation for every model variant (base / fine-tuned / API reference / mock).

Examples:
  python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --name base-0.5b
  python evals/run_eval.py --backend ollama --model mt103-extractor --name ft-0.5b-smoke
  python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --name api-haiku
  python evals/run_eval.py --backend mock --limit 5 --name mock

Metrics: valid-JSON rate, schema-valid rate, field-level exact match (per field + macro),
error-flag precision/recall (micro over "CODE:field" keys, and example-level detection),
latency per example, output tokens/sec. Raw outputs are cached on disk
(results/cache/<name>.jsonl) so re-runs are free and resumable.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prompts import FEWSHOT_IDS, chat_messages, ft_messages, parse_output, target_str  # noqa: E402
from postprocess import postprocess  # noqa: E402
from schema import Extraction, error_keys  # noqa: E402

SCORED_FIELDS = (
    "sender_reference", "bank_operation_code", "value_date", "currency", "amount",
    "ordering_customer.account", "ordering_customer.name", "ordering_customer.address",
    "ordering_institution", "account_with_institution",
    "beneficiary.account", "beneficiary.name", "beneficiary.address",
    "remittance_info", "charges",
)


# ---------------------------------------------------------------- scoring
def _get(fields, path):
    cur = fields
    for p in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _norm(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v).strip().strip(".,;").casefold()
        return s or None
    return v


def field_match(path, gold, pred) -> bool:
    if path == "amount":
        if gold is None or pred is None:
            return gold is None and pred is None
        try:
            return abs(float(pred) - float(gold)) < 0.005
        except (TypeError, ValueError):
            return False
    if isinstance(pred, (int, float)) and not isinstance(pred, bool):
        pred = str(pred)
    return _norm(gold) == _norm(pred)


def score_one(gold: dict, pred_obj) -> dict:
    valid_json = isinstance(pred_obj, dict)
    schema_ok = False
    if valid_json:
        try:
            Extraction.model_validate(pred_obj)
            schema_ok = True
        except Exception:
            pass
    pf = pred_obj.get("fields") if valid_json and isinstance(pred_obj.get("fields"), dict) else {}
    per = {p: field_match(p, _get(gold["fields"], p), _get(pf, p)) for p in SCORED_FIELDS}
    gk = error_keys(gold["errors"])
    pk = error_keys(pred_obj.get("errors")) if valid_json else set()
    return {"valid_json": valid_json, "schema_valid": schema_ok, "fields": per,
            "err_tp": len(gk & pk), "err_fp": len(pk - gk), "err_fn": len(gk - pk),
            "has_err_gold": bool(gk), "has_err_pred": bool(pk)}


def aggregate(scores: list[dict], lat: list[float], toks: list[int], gen_s: list[float]) -> dict:
    n = len(scores)
    per = {p: sum(s["fields"][p] for s in scores) / n for p in SCORED_FIELDS}
    tp = sum(s["err_tp"] for s in scores)
    fp = sum(s["err_fp"] for s in scores)
    fn = sum(s["err_fn"] for s in scores)
    dtp = sum(s["has_err_gold"] and s["has_err_pred"] for s in scores)
    dfp = sum((not s["has_err_gold"]) and s["has_err_pred"] for s in scores)
    dfn = sum(s["has_err_gold"] and not s["has_err_pred"] for s in scores)
    div = lambda a, b: (a / b) if b else None  # noqa: E731
    return {
        "n": n,
        "valid_json_rate": sum(s["valid_json"] for s in scores) / n,
        "schema_valid_rate": sum(s["schema_valid"] for s in scores) / n,
        "field_accuracy_macro": sum(per.values()) / len(per),
        "field_accuracy": per,
        "all_fields_correct_rate": sum(all(s["fields"].values()) for s in scores) / n,
        "error_precision": div(tp, tp + fp), "error_recall": div(tp, tp + fn),
        "error_detect_precision": div(dtp, dtp + dfp), "error_detect_recall": div(dtp, dtp + dfn),
        "latency_mean_s": statistics.mean(lat) if lat else None,
        "latency_p50_s": statistics.median(lat) if lat else None,
        "latency_p95_s": (sorted(lat)[max(0, int(round(0.95 * len(lat))) - 1)] if lat else None),
        "tokens_per_sec": (sum(toks) / sum(gen_s)) if gen_s and sum(gen_s) > 0 else None,
    }


# ---------------------------------------------------------------- backends
def call_ollama(msgs, model, url, num_ctx):
    body = json.dumps({"model": model, "messages": msgs, "stream": False,
                       "options": {"temperature": 0, "num_predict": 600, "num_ctx": num_ctx, "num_thread": 1,
                                   "seed": 0}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    lat = time.time() - t0
    return {"raw": out["message"]["content"], "latency_s": lat,
            "out_tokens": out.get("eval_count", 0), "gen_s": out.get("eval_duration", 0) / 1e9,
            "prompt_tokens": out.get("prompt_eval_count", 0)}


def call_claude_cli(msgs, model):
    """Uses the logged-in `claude` CLI (no API key on this host). Few-shot is inlined."""
    sys_prompt = msgs[0]["content"]
    convo = []
    for m in msgs[1:-1]:
        tag = "INPUT" if m["role"] == "user" else "OUTPUT"
        convo.append(f"<{tag}>\n{m['content']}\n</{tag}>")
    prompt = ("Examples:\n" + "\n".join(convo) + "\n\n" if convo else "") + \
             f"<INPUT>\n{msgs[-1]['content']}\n</INPUT>\nReply with the OUTPUT JSON only."
    env = dict(os.environ, ENABLE_CLAUDEAI_MCP_SERVERS="false", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--tools", "",
           "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--system-prompt", sys_prompt, prompt]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    lat = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {p.stderr[-300:]}")
    out = json.loads(p.stdout)
    usage = out.get("usage") or {}
    api_ms = out.get("duration_api_ms") or 0
    return {"raw": out.get("result", ""), "latency_s": lat, "out_tokens": usage.get("output_tokens", 0),
            "gen_s": api_ms / 1000.0, "prompt_tokens": usage.get("input_tokens", 0)}


def call_mock(msgs, ex):
    """Deterministic mock for CI: returns the gold target, except it corrupts every 5th example."""
    raw = target_str(ex["target"])
    h = int(hashlib.md5(ex["id"].encode()).hexdigest(), 16)
    if h % 5 == 0:
        raw = "Sure! " + raw[: len(raw) // 2]  # truncated -> invalid JSON
    return {"raw": raw, "latency_s": 0.001, "out_tokens": len(raw) // 4, "gen_s": 0.001, "prompt_tokens": 0}


# ---------------------------------------------------------------- main
def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(args) -> dict:
    test = load_jsonl(args.data)[: args.limit] if args.limit else load_jsonl(args.data)
    fewshot = []
    if args.fewshot:
        train = {r["id"]: r for r in load_jsonl(ROOT / "data" / "train.jsonl")}  # fixed v1 few-shot ids
        fewshot = [train[i] for i in FEWSHOT_IDS]
    cache_dir = Path(args.out_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_p = cache_dir / f"{args.name}.jsonl"
    cache = {}
    if cache_p.exists() and not args.no_cache:
        for r in load_jsonl(cache_p):
            cache[r["key"]] = r

    def key(ex, msgs):
        model_id = args.model + (f"+{args.adapter}" if args.adapter else "")
        return hashlib.sha1(json.dumps([args.backend, model_id, msgs], ensure_ascii=False).encode()).hexdigest()

    def msgs_for(ex):
        return ft_messages(ex["text"]) if args.prompt == "ft" else chat_messages(ex["text"], fewshot)

    if args.backend == "hf" and not args.cache_only:  # batched GPU generation up front, into the cache
        todo = [(ex, msgs_for(ex)) for ex in test]
        todo = [(ex, m, key(ex, m)) for ex, m in todo if key(ex, m) not in cache]
        if todo:
            from evals.hf_backend import generate, load
            model, tok = load(args.model, args.adapter)
            outs = generate(model, tok, [m for _, m, _ in todo], batch_size=args.batch_size)
            with open(cache_p, "a", encoding="utf-8") as fh:
                for (ex, _, k), o in zip(todo, outs):
                    cache[k] = {"key": k, "id": ex["id"], **o}
                    fh.write(json.dumps(cache[k], ensure_ascii=False) + "\n")

    def work(ex):
        msgs = msgs_for(ex)
        k = key(ex, msgs)
        if k in cache:
            return ex, cache[k], True
        if args.cache_only:  # scored as a failed call (no output), exactly like the original run's failures
            return ex, {"key": k, "id": ex["id"], "raw": None, "latency_s": None, "out_tokens": 0, "gen_s": 0,
                        "error": "not in cache (--cache-only)"}, True
        for attempt in range(3):
            try:
                if args.backend == "ollama":
                    res = call_ollama(msgs, args.model, args.ollama_url, args.num_ctx)
                elif args.backend == "claude-cli":
                    res = call_claude_cli(msgs, args.model)
                elif args.backend == "hf":
                    raise RuntimeError("hf output missing from the batched pre-pass")
                else:
                    res = call_mock(msgs, ex)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    res = {"raw": None, "latency_s": None, "out_tokens": 0, "gen_s": 0, "error": str(e)[:300]}
                time.sleep(2 * (attempt + 1))
        res = {"key": k, "id": ex["id"], **res}
        return ex, res, False

    rows = []
    workers = args.workers if args.backend == "claude-cli" else 1
    with open(cache_p, "a", encoding="utf-8") as cfh, cf.ThreadPoolExecutor(workers) as pool:
        for i, (ex, res, hit) in enumerate(pool.map(work, test), 1):
            if not hit and args.backend != "mock" and res.get("raw") is not None:
                cfh.write(json.dumps(res, ensure_ascii=False) + "\n")
                cfh.flush()
            rows.append((ex, res))
            if i % 10 == 0 or i == len(test):
                print(f"[{args.name}] {i}/{len(test)}", flush=True)

    scores, lat, toks, gen = [], [], [], []
    per_example = []
    for ex, res in rows:
        obj = parse_output(res.get("raw"))
        if args.postprocess:
            obj = postprocess(obj, ex["text"])
        s = score_one(ex["target"], obj)
        scores.append(s)
        if res.get("latency_s") is not None:
            lat.append(res["latency_s"])
        if res.get("gen_s"):
            toks.append(res.get("out_tokens", 0))
            gen.append(res["gen_s"])
        per_example.append({"id": ex["id"], "valid_json": s["valid_json"], "schema_valid": s["schema_valid"],
                            "fields_wrong": [k for k, v in s["fields"].items() if not v],
                            "err_fp": s["err_fp"], "err_fn": s["err_fn"]})
    summary = aggregate(scores, lat, toks, gen)
    name = args.name + ("+pp" if args.postprocess else "")
    summary.update({"name": name, "postprocess": bool(args.postprocess), "backend": args.backend, "model": args.model,
                    "fewshot": bool(args.fewshot), "prompt": args.prompt, "data": str(args.data),
                    "date": dt.date.today().isoformat(), "command": " ".join(sys.argv)})
    out = Path(args.out_dir) / f"eval_{name}.json"
    out.write_text(json.dumps({"summary": summary, "per_example": per_example}, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "field_accuracy"}, indent=2))
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "claude-cli", "mock", "hf"], required=True,
                    help="hf = transformers on GPU/CPU (--model = HF id or path, optional --adapter LoRA dir)")
    ap.add_argument("--model", default="")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--prompt", choices=["full", "ft"], default="full",
                    help="full = long instruction prompt (+ few-shot); ft = short prompt the LoRA model was trained with")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:21434"))
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--postprocess", action="store_true",
                    help="apply postprocess.py (deterministic rules) before scoring; reads the same cache, "
                         "writes eval_<name>+pp.json")
    ap.add_argument("--adapter", default="", help="hf backend: LoRA adapter directory")
    ap.add_argument("--batch-size", type=int, default=16, help="hf backend: generation batch size")
    ap.add_argument("--cache-only", action="store_true", help="never call a model; a cache miss is scored as a failed call")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
