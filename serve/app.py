"""FastAPI service: POST /extract {text} -> validated MT103 JSON + errors + latency.

Uses the fine-tuned Ollama model (FT_MODEL, default `mt103-extractor`). If it is not
present in Ollama, falls back to the base model with the few-shot prompt and says so in
the response (`model_used`, `fallback: true`, `note`).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from postprocess import postprocess as rule_postprocess  # noqa: E402
from prompts import FEWSHOT_IDS, chat_messages, ft_messages, parse_output  # noqa: E402
from schema import Extraction  # noqa: E402

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:21434")
# Deterministic rules from postprocess.py on top of the model output (POSTPROCESS=0 turns them off).
POSTPROCESS = os.environ.get("POSTPROCESS", "1").lower() not in ("0", "false", "no")
FT_MODEL = os.environ.get("FT_MODEL", "mt103-extractor")
BASE_MODEL = os.environ.get("BASE_MODEL", "qwen2.5:0.5b-instruct")
FEWSHOT_PATH = Path(os.environ.get("FEWSHOT_PATH", ROOT / "serve" / "fewshot.json"))

app = FastAPI(title="mt103-extractor", version="0.1.0")


class ExtractIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


def _fewshot():
    if FEWSHOT_PATH.exists():
        return json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))
    return []


def _ollama_models(client) -> set[str]:
    r = client.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    names = set()
    for m in r.json().get("models", []):
        names.add(m["name"])
        names.add(m["name"].split(":")[0] if m["name"].endswith(":latest") else m["name"])
    return names


def pick_model(client) -> tuple[str, bool]:
    names = _ollama_models(client)
    if FT_MODEL in names or f"{FT_MODEL}:latest" in names:
        return FT_MODEL, False
    if BASE_MODEL in names:
        return BASE_MODEL, True
    raise HTTPException(503, f"neither {FT_MODEL} nor {BASE_MODEL} is loaded in Ollama at {OLLAMA_URL}")


@app.get("/health")
def health():
    try:
        with httpx.Client() as c:
            names = sorted(_ollama_models(c))
        return {"status": "ok", "ollama": OLLAMA_URL, "models": names,
                "fine_tuned_present": FT_MODEL in names or f"{FT_MODEL}:latest" in names}
    except Exception as e:  # noqa: BLE001
        return {"status": "degraded", "ollama": OLLAMA_URL, "error": str(e)[:200]}


def postprocess(raw: str, text: str = "", rules: bool = False) -> dict:
    """Parse + validate the model output.

    valid_json / schema_valid / validation_errors always describe the RAW model output, so its mistakes
    stay visible. With rules=True, `fields` / `errors` come from the deterministic rules in postprocess.py
    (normalise, validate, check against the source text) and the raw output is kept as `model_output`.
    """
    obj = parse_output(raw)
    if not isinstance(obj, dict):
        return {"valid_json": False, "schema_valid": False, "fields": None, "errors": [],
                "validation_errors": ["model output was not valid JSON"], "postprocessed": False}
    try:
        ext = Extraction.model_validate(obj)
        res = {"valid_json": True, "schema_valid": True, "fields": ext.fields.model_dump(),
               "errors": ext.errors, "validation_errors": []}
    except ValidationError as e:
        msgs = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        res = {"valid_json": True, "schema_valid": False, "fields": obj.get("fields"),
               "errors": obj.get("errors", []), "validation_errors": msgs}
    res["postprocessed"] = False
    if rules:
        pp = rule_postprocess(obj, text)
        res["model_output"] = {"fields": res["fields"], "errors": res["errors"]}
        res.update({"fields": pp["fields"], "errors": pp["errors"], "postprocessed": True})
    return res


@app.post("/extract")
def extract(body: ExtractIn):
    with httpx.Client() as client:
        model, fallback = pick_model(client)
        msgs = chat_messages(body.text, _fewshot()) if fallback else ft_messages(body.text)
        t0 = time.time()
        r = client.post(f"{OLLAMA_URL}/api/chat", timeout=600, json={
            "model": model, "messages": msgs, "stream": False,
            "options": {"temperature": 0, "num_predict": 600, "num_thread": int(os.environ.get("OLLAMA_NUM_THREAD", "1")),
                        "num_ctx": 4096 if fallback else 2048}})
        latency = time.time() - t0
    r.raise_for_status()
    out = r.json()
    res = postprocess(out["message"]["content"], body.text, rules=POSTPROCESS)
    ev, ed = out.get("eval_count", 0), out.get("eval_duration", 0) / 1e9
    res.update({
        "model_used": model, "fallback": fallback,
        "note": (f"fine-tuned model '{FT_MODEL}' not found in Ollama; used base model '{BASE_MODEL}' "
                 "with a few-shot prompt") if fallback else "fine-tuned model",
        "latency_s": round(latency, 3), "output_tokens": ev,
        "tokens_per_sec": round(ev / ed, 2) if ed else None,
    })
    return res
