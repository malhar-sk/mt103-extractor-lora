"""OOD generator (synthetic, deterministic, really out-of-distribution) and the error-analysis classifiers."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from data.generate import BAD_BICS, BAD_CCY, BAD_CHARGES
from data.generate_ood import OOD_BAD_BICS, OOD_BAD_CCY, OOD_BAD_CHARGES, generate_ood
from evals.analyze import analyse_run, classify_field_error, flag_outcome
from schema import Extraction

ROOT = Path(__file__).resolve().parents[1]


def _load(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


@pytest.fixture(scope="module")
def ood():
    return _load(ROOT / "data" / "ood_test.jsonl")


def test_ood_is_synthetic_valid_and_committed_matches_generator(ood):
    assert len(ood) == 100
    assert all(r["synthetic"] is True for r in ood)
    for r in ood:
        Extraction.model_validate(r["target"])
    assert [r["text"] for r in generate_ood(100, 20261123)] == [r["text"] for r in ood]


def test_ood_does_not_overlap_training_data(ood):
    train = {hashlib.sha1(r["text"].encode()).hexdigest() for r in _load(ROOT / "data" / "train.jsonl")}
    assert not any(hashlib.sha1(r["text"].encode()).hexdigest() in train for r in ood)


def test_ood_bad_values_never_seen_in_training():
    assert not set(OOD_BAD_CCY) & set(BAD_CCY)
    assert not set(OOD_BAD_BICS) & set(BAD_BICS)
    assert not set(OOD_BAD_CHARGES) & set(BAD_CHARGES)


def test_ood_uses_new_styles_and_lowercase_tags(ood):
    assert {r["ood_style"] for r in ood} == {"swift_shuffled", "narrative", "pipe_table", "kv_mixed"}
    assert any(":32a:" in r["text"] for r in ood)


GOLD = {"fields": {"sender_reference": "TXN1", "bank_operation_code": None, "value_date": "2026-03-05",
                   "currency": None, "amount": 1250.0,
                   "ordering_customer": {"account": "111", "name": "A Ltd", "address": None},
                   "ordering_institution": "SBININBB", "account_with_institution": "HDFCINBB",
                   "beneficiary": {"account": "222", "name": "B GmbH", "address": "1 Road"},
                   "remittance_info": "INV 1", "charges": "SHA"},
        "errors": ["INVALID_CURRENCY: currency: USX is not an ISO-4217 code"]}


def _pred(**kw):
    p = copy.deepcopy(GOLD)
    for k, v in kw.items():
        p["fields"][k] = v
    return p


@pytest.mark.parametrize("path,pred,expected", [
    ("currency", _pred(currency="USD"), "repaired_impossible_value"),
    ("bank_operation_code", _pred(bank_operation_code="CRED"), "hallucinated_absent_field"),
    ("ordering_institution", _pred(ordering_institution=None), "missed_present_value"),
    ("ordering_institution", _pred(ordering_institution="HDFCINBB"), "value_from_other_field"),
    ("value_date", _pred(value_date="2026-05-03"), "date_day_month_swap"),
    ("value_date", _pred(value_date="2027-03-05"), "date_wrong_year"),
    ("amount", _pred(amount=1.25), "amount_scale"),
    ("amount", _pred(amount=999.0), "amount_misparsed"),
    ("charges", _pred(charges="OUR"), "enum_wrong_value"),
    ("remittance_info", _pred(remittance_info="INV 1 consulting"), "string_partial"),
    ("currency", None, "invalid_json"),
])
def test_classify_field_error(path, pred, expected):
    assert classify_field_error(path, GOLD["fields"], GOLD["errors"], pred) == expected


def test_flag_outcomes():
    k = "INVALID_CURRENCY:currency"
    assert flag_outcome(k, GOLD["fields"], GOLD) == "flagged_correctly"
    assert flag_outcome(k, GOLD["fields"], dict(_pred(), errors=[])) == "not_flagged_field_left_null"
    assert flag_outcome(k, GOLD["fields"], dict(_pred(currency="USD"), errors=[])) == "not_flagged_repaired_or_invented"
    assert flag_outcome(k, GOLD["fields"], dict(_pred(currency="USX"), errors=[])) == "not_flagged_copied_invalid_value"
    assert flag_outcome(k, GOLD["fields"], dict(_pred(), errors=["INVALID_BIC: currency: x"])) == "flagged_wrong_code_or_field"
    assert flag_outcome(k, GOLD["fields"], None) == "invalid_json"


def test_analyse_run_counts():
    data = [{"id": "a", "text": "t", "target": GOLD},
            {"id": "b", "text": "t", "target": dict(GOLD, errors=[], fields=dict(GOLD["fields"], currency="EUR"))}]
    preds = {"a": dict(_pred(currency="USD"), errors=[]),
             "b": dict(_pred(currency="EUR"), errors=["INVALID_BIC: ordering_institution: x"])}
    r = analyse_run(data, preds)
    e = r["error_flags"]["INVALID_CURRENCY"]
    assert (e["gold"], e["recall"], e["outcomes"]) == (1, 0.0, {"not_flagged_repaired_or_invented": 1})
    assert r["error_flags"]["INVALID_BIC"]["precision"] == 0.0
    assert r["false_alarm_rate_on_clean_examples"] == 1.0
    assert r["field_accuracy"]["currency"] == 0.5
