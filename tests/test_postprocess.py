"""postprocess.py: never damages a correct answer, and each rule does what it says on hand-built cases."""
import copy
import json
from pathlib import Path

import pytest

from postprocess import parse_number, postprocess, text_dates
from schema import Extraction

ROOT = Path(__file__).resolve().parents[1]


def _load(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


@pytest.mark.parametrize("split", ["train", "val", "test", "ood_test"])
def test_gold_targets_pass_through_unchanged(split):
    """The rules must never change a correct extraction (otherwise they could only hurt)."""
    rows = _load(ROOT / "data" / f"{split}.jsonl")
    bad = [r["id"] for r in rows if postprocess(r["target"], r["text"]) != r["target"]]
    assert bad == []


def test_gold_targets_v2_pass_through_unchanged():
    p = ROOT / "data" / "v2" / "test.jsonl"
    if not p.exists():
        pytest.skip("v2 data not generated")
    rows = _load(p)
    assert [r["id"] for r in rows if postprocess(r["target"], r["text"]) != r["target"]] == []


TEXT = ("Pls send Rs 1,25,000.50 to Coromandel Logistics Pvt Ltd a/c 54150945237 at HDFCINBB, value 05/12/2026, "
        "our ref TXN778812, charges shared. Remitter: Helix Pharma Ltd a/c 726045614514.")
GOLD = {"fields": {"sender_reference": "TXN778812", "bank_operation_code": None, "value_date": "2026-12-05",
                   "currency": "INR", "amount": 125000.5,
                   "ordering_customer": {"account": "726045614514", "name": "Helix Pharma Ltd", "address": None},
                   "ordering_institution": None, "account_with_institution": "HDFCINBB",
                   "beneficiary": {"account": "54150945237", "name": "Coromandel Logistics Pvt Ltd", "address": None},
                   "remittance_info": None, "charges": "SHA"}, "errors": []}


def pred(errors=None, **kw):
    p = copy.deepcopy(GOLD)
    p["fields"].update(kw)
    p["errors"] = errors or []
    return p


def test_normalises_strings_symbols_and_formats():
    out = postprocess(pred(currency="₹", amount="1,25,000.50", value_date="05/12/2026", charges="shared",
                           account_with_institution="hdfcinbb"), TEXT)
    assert out == GOLD
    Extraction.model_validate(out)


def test_invented_values_are_removed():
    out = postprocess(pred(bank_operation_code="CRED", ordering_institution="SBININBB", sender_reference="TXN1"), TEXT)
    f = out["fields"]
    assert f["bank_operation_code"] is None and f["ordering_institution"] is None and f["sender_reference"] is None


def test_repaired_impossible_date_is_nulled_and_flagged():
    text = TEXT.replace("05/12/2026", "31/04/2026")
    out = postprocess(pred(value_date="2026-04-30"), text)
    assert out["fields"]["value_date"] is None
    assert out["errors"] == ["INVALID_DATE: value_date: 31/04/2026 is not a real calendar date"]


def test_separator_misread_amount_is_reparsed():
    assert postprocess(pred(amount=1250.005), TEXT)["fields"]["amount"] == 125000.5


def test_negative_amount_in_text():
    text = TEXT.replace("Rs 1,25,000.50", "Rs -1,25,000.50")
    out = postprocess(pred(amount=125000.5), text)
    assert out["fields"]["amount"] is None and out["errors"][0].startswith("NON_POSITIVE_AMOUNT: amount:")


def test_error_on_valid_value_is_dropped_and_value_restored():
    out = postprocess(pred(errors=["INVALID_CURRENCY: currency: INR is not an ISO-4217 code",
                                   "INVALID_CHARGES: charges: SHA is not OUR/SHA/BEN"], currency=None), TEXT)
    assert out == GOLD


def test_invalid_value_copied_from_text_is_flagged():
    text = TEXT.replace("HDFCINBB", "HDFC123")
    out = postprocess(pred(account_with_institution="HDFC123"), text)
    assert out["fields"]["account_with_institution"] is None
    assert out["errors"] == ["INVALID_BIC: account_with_institution: HDFC123 is not a valid BIC"]


def test_unknown_codes_and_bad_fields_dropped():
    out = postprocess(pred(errors=["WRONG: currency: x", "INVALID_BIC: amount: x", "garbage"]), TEXT)
    assert out["errors"] == []


def test_missing_required_added_only_when_text_has_nothing():
    text = TEXT.replace(", value 05/12/2026", "").replace(", charges shared", "")
    out = postprocess(pred(value_date=None, charges=None), text)
    assert out["errors"] == ["MISSING_REQUIRED: value_date: not stated in the text",
                             "MISSING_REQUIRED: charges: not stated in the text"]
    # a date is in the text, the model just missed it: no flag (not a payment problem)
    assert postprocess(pred(value_date=None), TEXT)["errors"] == []


def test_not_a_dict_passes_through():
    assert postprocess(None, TEXT) is None


@pytest.mark.parametrize("s,x", [("1,25,000.50", 125000.5), ("2.5 lakh", 250000.0), ("1.2 crore", 12000000.0),
                                 ("1.234.567,89", 1234567.89), ("1'234.50", 1234.5), ("12 500,00", 12500.0),
                                 ("6.9e+06", 6900000.0), ("(500)", -500.0), ("-1,000", -1000.0)])
def test_parse_number(s, x):
    assert parse_number(s) == pytest.approx(x)


def test_text_dates():
    valid, bad = text_dates("on 5 March 2026, or March 6, 2026, 07/03/2026, 2026-03-08, :32A:260309USD1,0 and 31/04/2026")
    assert {d.isoformat() for d in valid} >= {"2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08", "2026-03-09"}
    assert bad == ["31/04/2026"]
