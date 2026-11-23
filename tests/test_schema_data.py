"""M1: schema validation, dataset targets, split hygiene, generator determinism."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data.generate import generate, indian_format, split_ids
from schema import Extraction, MT103Fields, error_keys

DATA = Path(__file__).resolve().parents[1] / "data"


def load(name):
    return [json.loads(l) for l in open(DATA / f"{name}.jsonl", encoding="utf-8")]


@pytest.fixture(scope="module")
def splits():
    return {n: load(n) for n in ("train", "val", "test")}


def test_split_sizes(splits):
    assert [len(splits[n]) for n in ("train", "val", "test")] == [2400, 300, 300]


def test_all_targets_schema_valid(splits):
    total = 0
    for rows in splits.values():
        for r in rows:
            Extraction.model_validate(r["target"])
            assert r["synthetic"] is True
            total += 1
    assert total == 3000


def test_splits_do_not_overlap(splits):
    ids = [set(r["id"] for r in rows) for rows in splits.values()]
    texts = [set(r["text"] for r in rows) for rows in splits.values()]
    for a in range(3):
        for b in range(a + 1, 3):
            assert not ids[a] & ids[b]
            assert not texts[a] & texts[b]


def test_errors_flagged_not_invented(splits):
    rows = [r for rs in splits.values() for r in rs]
    with_err = [r for r in rows if r["target"]["errors"]]
    assert 0.15 < len(with_err) / len(rows) < 0.3
    for r in with_err:
        for key in error_keys(r["target"]["errors"]):
            code, field = key.split(":")
            # the flagged field is null in the target (not invented)
            assert r["target"]["fields"][field] is None, (r["id"], key)


def test_generator_is_deterministic():
    a = generate(50, seed=7)
    b = generate(50, seed=7)
    c = generate(50, seed=8)
    assert a == b
    assert a != c
    assert split_ids(100, 7) == split_ids(100, 7)


def test_committed_data_matches_generator(splits):
    rows = generate(3000, seed=103)
    tr, va, te = split_ids(3000, 103)
    assert [rows[i] for i in sorted(te)] == splits["test"]


@pytest.mark.parametrize("bad", [
    {"currency": "USX"}, {"currency": "usd"}, {"value_date": "2026-02-30"}, {"value_date": "23/11/2026"},
    {"ordering_institution": "HDFC123"}, {"account_with_institution": "deutdedb"},
    {"amount": 0}, {"amount": -5.0}, {"charges": "FULL"}, {"bank_operation_code": "XXXX"},
    {"sender_reference": "THIS-REF-IS-WAY-TOO-LONG"},
])
def test_schema_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        MT103Fields(**bad)


def test_schema_accepts_valid():
    MT103Fields(sender_reference="TXN000123", currency="INR", amount=125000.5, value_date="2026-11-23",
                ordering_institution="SBININBB", account_with_institution="ICICINBBXXX", charges="SHA",
                beneficiary={"name": "Lotus Foods Ltd", "account": "123", "address": None})


def test_extraction_rejects_unknown_error_code():
    with pytest.raises(ValidationError):
        Extraction(fields=MT103Fields(), errors=["BAD_THING: amount: x"])


def test_indian_format():
    assert indian_format(125000.5) == "1,25,000.50"
    assert indian_format(98765432.1) == "9,87,65,432.10"
    assert indian_format(999.0, decimals=False) == "999"
