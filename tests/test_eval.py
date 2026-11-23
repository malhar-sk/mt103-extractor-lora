"""Eval script: parsing, scoring, and an end-to-end run on 5 examples with a mocked model."""
import json

from evals.run_eval import field_match, parse_args, run, score_one
from prompts import parse_output, target_str

GOLD = {"fields": {"sender_reference": "TXN000001", "bank_operation_code": None, "value_date": "2026-11-23",
                   "currency": "INR", "amount": 125000.5,
                   "ordering_customer": {"account": "123", "name": "Lotus Foods Ltd", "address": None},
                   "ordering_institution": "SBININBB", "account_with_institution": None,
                   "beneficiary": {"account": "9", "name": "Orion Steel GmbH", "address": None},
                   "remittance_info": None, "charges": None},
        "errors": ["INVALID_CHARGES: charges: FULL is not OUR/SHA/BEN"]}


def test_parse_output_variants():
    s = target_str(GOLD)
    assert parse_output(s) == GOLD
    assert parse_output("```json\n" + s + "\n```") == GOLD
    assert parse_output("Here you go: " + s + " hope it helps") == GOLD
    assert parse_output("no json here") is None
    assert parse_output(s[:-5]) is None


def test_perfect_score():
    s = score_one(GOLD, json.loads(target_str(GOLD)))
    assert s["valid_json"] and s["schema_valid"]
    assert all(s["fields"].values())
    assert (s["err_tp"], s["err_fp"], s["err_fn"]) == (1, 0, 0)


def test_wrong_fields_and_missed_error():
    pred = json.loads(target_str(GOLD))
    pred["fields"]["amount"] = 125000.0
    pred["fields"]["charges"] = "OUR"   # invented instead of flagged
    pred["errors"] = []
    s = score_one(GOLD, pred)
    assert not s["fields"]["amount"] and not s["fields"]["charges"]
    assert s["fields"]["currency"]
    assert (s["err_tp"], s["err_fn"]) == (0, 1)


def test_invalid_json_scores_zero_on_present_fields():
    s = score_one(GOLD, None)
    assert not s["valid_json"] and not s["schema_valid"]
    assert not s["fields"]["sender_reference"]
    assert s["fields"]["remittance_info"]  # null gold == null pred


def test_field_match_normalisation():
    assert field_match("beneficiary.name", "Orion Steel GmbH", " orion  steel gmbh. ")
    assert field_match("amount", 100.0, "100.00")
    assert not field_match("amount", 100.0, "abc")
    assert field_match("sender_reference", "12345", 12345)


def test_end_to_end_mock_5(tmp_path):
    args = parse_args(["--backend", "mock", "--limit", "5", "--name", "mock5", "--out-dir", str(tmp_path)])
    s = run(args)
    assert s["n"] == 5
    out = json.loads((tmp_path / "eval_mock5.json").read_text())
    assert len(out["per_example"]) == 5
    # mock returns gold except it truncates ~1 in 5 outputs
    assert 0.0 < s["valid_json_rate"] <= 1.0
    ok = [p for p in out["per_example"] if p["valid_json"]]
    assert all(not p["fields_wrong"] for p in ok)


def test_ft_prompt_is_short():
    from prompts import SYSTEM, FT_SYSTEM
    assert len(FT_SYSTEM) < len(SYSTEM) / 5
