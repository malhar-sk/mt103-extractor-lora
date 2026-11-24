"""FastAPI /extract with a mocked Ollama (httpx MockTransport): fine-tuned path + fallback."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import serve.app as app_mod
from prompts import FT_SYSTEM, target_str

GOOD = {"fields": {"sender_reference": "TXN000001", "bank_operation_code": "CRED", "value_date": "2026-11-23",
                   "currency": "USD", "amount": 5000.0,
                   "ordering_customer": {"account": "1", "name": "Helix Pharma Ltd", "address": None},
                   "ordering_institution": None, "account_with_institution": "CHASUS33",
                   "beneficiary": {"account": "2", "name": "Orion Motors Inc", "address": None},
                   "remittance_info": "INV 0001", "charges": "SHA"}, "errors": []}


def make_transport(models, reply, seen):
    def handler(req: httpx.Request):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in models]})
        if req.url.path == "/api/chat":
            body = json.loads(req.content)
            seen.append(body)
            return httpx.Response(200, json={"message": {"content": reply}, "eval_count": 100,
                                             "eval_duration": 4_000_000_000})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


@pytest.fixture
def client_factory(monkeypatch):
    def make(models, reply):
        seen = []
        transport = make_transport(models, reply, seen)
        real = httpx.Client
        monkeypatch.setattr(app_mod.httpx, "Client", lambda *a, **k: real(transport=transport))
        return TestClient(app_mod.app), seen
    return make


def test_extract_uses_finetuned(client_factory):
    c, seen = client_factory(["mt103-extractor:latest", "qwen2.5:0.5b-instruct"], target_str(GOOD))
    r = c.post("/extract", json={"text": "pls send USD 5,000 to Orion Motors"})
    assert r.status_code == 200
    j = r.json()
    assert j["model_used"] == "mt103-extractor" and j["fallback"] is False
    assert j["schema_valid"] and j["fields"]["amount"] == 5000.0
    assert j["tokens_per_sec"] == 25.0
    assert seen[0]["messages"][0]["content"] == FT_SYSTEM


def test_extract_falls_back_to_base_and_says_so(client_factory):
    c, seen = client_factory(["qwen2.5:0.5b-instruct"], "```json\n" + target_str(GOOD) + "\n```")
    j = c.post("/extract", json={"text": "x"}).json()
    assert j["fallback"] is True and j["model_used"] == "qwen2.5:0.5b-instruct"
    assert "not found" in j["note"]
    assert len(seen[0]["messages"]) > 2  # few-shot prompt


def test_extract_reports_validation_errors(client_factory):
    bad = json.loads(target_str(GOOD)); bad["fields"]["currency"] = "USX"
    c, _ = client_factory(["mt103-extractor"], json.dumps(bad))
    j = c.post("/extract", json={"text": "x"}).json()
    assert j["valid_json"] and not j["schema_valid"]
    assert any("currency" in m for m in j["validation_errors"])


def test_extract_invalid_json(client_factory):
    c, _ = client_factory(["mt103-extractor"], "I cannot do that")
    j = c.post("/extract", json={"text": "x"}).json()
    assert j["valid_json"] is False


def test_no_models_503(client_factory):
    c, _ = client_factory([], "")
    assert c.post("/extract", json={"text": "x"}).status_code == 503


def test_empty_text_rejected(client_factory):
    c, _ = client_factory(["mt103-extractor"], "")
    assert c.post("/extract", json={"text": ""}).status_code == 422


def test_extract_applies_rules_and_keeps_raw_output(client_factory):
    bad = json.loads(target_str(GOOD))
    bad["fields"]["currency"] = "$"
    bad["fields"]["bank_operation_code"] = None
    bad["errors"] = ["INVALID_CHARGES: charges: SHA is not OUR/SHA/BEN"]  # a false alarm on a valid value
    c, _ = client_factory(["mt103-extractor"], json.dumps(bad))
    text = ("pls send $5,000.00 to Orion Motors Inc a/c 2 at CHASUS33, value 23/11/2026, ref TXN000001, "
            "charges SHA, from Helix Pharma Ltd a/c 1, INV 0001")
    j = c.post("/extract", json={"text": text}).json()
    assert j["postprocessed"] is True and not j["schema_valid"]  # schema_valid describes the raw output
    assert j["model_output"]["fields"]["currency"] == "$"
    assert j["fields"]["currency"] == "USD" and j["errors"] == []
