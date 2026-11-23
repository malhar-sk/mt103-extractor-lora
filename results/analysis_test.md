# Error analysis: `test` (data/test.jsonl, n = 300, SYNTHETIC)

Generated 2026-11-23 by `python evals/analyze.py --data data/test.jsonl --runs base-qwen2.5-0.5b-fewshot ft-qwen2.5-0.5b-lora api-claude-haiku-5-5-fewshot --tag test` from cached model outputs (no new model calls).

## Per-field accuracy

| Field | base-qwen2.5-0.5b-fewshot | ft-qwen2.5-0.5b-lora | api-claude-haiku-5-5-fewshot |
|---|---|---|---|
| sender_reference | 96.7% | 99.3% | 99.7% |
| bank_operation_code | 49.7% | 89.7% | 99.3% |
| value_date | 60.7% | 91.0% | 99.3% |
| currency | 93.7% | 96.7% | 99.7% |
| amount | 82.0% | 90.0% | 97.0% |
| ordering_customer.account | 56.0% | 98.3% | 99.7% |
| ordering_customer.name | 63.7% | 100.0% | 99.7% |
| ordering_customer.address | 38.7% | 97.7% | 99.7% |
| ordering_institution | 41.3% | 87.7% | 99.7% |
| account_with_institution | 41.7% | 80.3% | 99.3% |
| beneficiary.account | 56.7% | 98.3% | 99.7% |
| beneficiary.name | 66.7% | 98.7% | 99.7% |
| beneficiary.address | 46.0% | 98.0% | 99.7% |
| remittance_info | 67.7% | 97.3% | 95.0% |
| charges | 78.7% | 90.0% | 99.7% |

## Field error types: base-qwen2.5-0.5b-fewshot (1681 wrong field values out of 4500)

| Error type | Count | Share | Worst fields | Meaning |
|---|---|---|---|---|
| value_from_other_field | 587 | 34.9% | ordering_customer.name (105), ordering_customer.address (99), ordering_customer.account (89) | the model put another field's value here (e.g. 52A/57A BICs swapped, account in address) |
| hallucinated_absent_field | 341 | 20.3% | bank_operation_code (139), ordering_customer.address (67), remittance_info (57) | the field is not in the text (gold null), the model invented a value |
| missed_present_value | 237 | 14.1% | ordering_institution (121), beneficiary.address (100), account_with_institution (13) | the value is in the text, the model returned null |
| string_wrong_value | 204 | 12.1% | beneficiary.account (63), ordering_customer.account (43), account_with_institution (38) | a different string altogether |
| date_misparsed | 86 | 5.1% | value_date (86) | date wrong in another way |
| repaired_impossible_value | 67 | 4.0% | value_date (23), charges (17), amount (12) | the source value was impossible (gold null + error), the model output a plausible value instead of null |
| enum_wrong_value | 65 | 3.9% | charges (44), bank_operation_code (12), currency (9) | wrong code (charges / op code / currency) |
| string_partial | 43 | 2.6% | remittance_info (27), ordering_customer.address (7), beneficiary.address (4) | right value with text missing or extra text attached |
| amount_scale | 21 | 1.2% | amount (21) | amount off by a power of ten (thousands/decimal separator misread) |
| amount_misparsed | 21 | 1.2% | amount (21) | amount wrong in another way |
| date_day_month_swap | 6 | 0.4% | value_date (6) | day and month swapped |
| date_wrong_year | 3 | 0.2% | value_date (3) | right day and month, wrong year |

## Field error types: ft-qwen2.5-0.5b-lora (261 wrong field values out of 4500)

| Error type | Count | Share | Worst fields | Meaning |
|---|---|---|---|---|
| missed_present_value | 100 | 38.3% | bank_operation_code (30), account_with_institution (27), ordering_institution (20) | the value is in the text, the model returned null |
| repaired_impossible_value | 51 | 19.5% | value_date (19), charges (16), currency (10) | the source value was impossible (gold null + error), the model output a plausible value instead of null |
| value_from_other_field | 37 | 14.2% | account_with_institution (15), ordering_institution (14), ordering_customer.address (3) | the model put another field's value here (e.g. 52A/57A BICs swapped, account in address) |
| string_wrong_value | 18 | 6.9% | ordering_customer.account (5), account_with_institution (4), beneficiary.name (3) | a different string altogether |
| amount_scale | 14 | 5.4% | amount (14) | amount off by a power of ten (thousands/decimal separator misread) |
| hallucinated_absent_field | 12 | 4.6% | account_with_institution (8), ordering_customer.address (2), remittance_info (2) | the field is not in the text (gold null), the model invented a value |
| date_misparsed | 8 | 3.1% | value_date (8) | date wrong in another way |
| enum_wrong_value | 7 | 2.7% | charges (6), bank_operation_code (1) | wrong code (charges / op code / currency) |
| amount_misparsed | 7 | 2.7% | amount (7) | amount wrong in another way |
| string_partial | 7 | 2.7% | account_with_institution (3), remittance_info (3), ordering_institution (1) | right value with text missing or extra text attached |

## Field error types: api-claude-haiku-5-5-fewshot (40 wrong field values out of 4500)

| Error type | Count | Share | Worst fields | Meaning |
|---|---|---|---|---|
| string_partial | 13 | 32.5% | remittance_info (13) | right value with text missing or extra text attached |
| string_wrong_value | 8 | 20.0% | remittance_info (2), sender_reference (1), ordering_customer.account (1) | a different string altogether |
| amount_misparsed | 8 | 20.0% | amount (8) | amount wrong in another way |
| missed_present_value | 5 | 12.5% | bank_operation_code (2), ordering_institution (1), beneficiary.address (1) | the value is in the text, the model returned null |
| date_misparsed | 2 | 5.0% | value_date (2) | date wrong in another way |
| repaired_impossible_value | 2 | 5.0% | amount (1), account_with_institution (1) | the source value was impossible (gold null + error), the model output a plausible value instead of null |
| enum_wrong_value | 1 | 2.5% | currency (1) | wrong code (charges / op code / currency) |
| hallucinated_absent_field | 1 | 2.5% | ordering_customer.address (1) | the field is not in the text (gold null), the model invented a value |

## Error-flag analysis by error code

Recall = gold errors flagged with the right CODE:field. Outcomes for each gold error: **ok** flagged correctly, **wrong** flagged with another code/field, **repaired** no flag and a plausible value invented (silent, nothing downstream catches it), **copied** no flag but the impossible value was copied through (the API's pydantic validation still catches it), **null** no flag, field left null, **json** invalid JSON.

| Code | Model | Gold n | Recall | Pred n | Precision | ok / wrong / repaired / copied / null / json |
|---|---|---|---|---|---|---|
| INVALID_CURRENCY | base-qwen2.5-0.5b-fewshot | 10 | 0.0% | 0 | n/a | 0 / 0 / 9 / 1 / 0 / 0 |
| INVALID_CURRENCY | ft-qwen2.5-0.5b-lora | 10 | 0.0% | 16 | 0.0% | 0 / 0 / 8 / 2 / 0 / 0 |
| INVALID_CURRENCY | api-claude-haiku-5-5-fewshot | 10 | 100.0% | 10 | 100.0% | 10 / 0 / 0 / 0 / 0 / 0 |
| INVALID_DATE | base-qwen2.5-0.5b-fewshot | 18 | 0.0% | 0 | n/a | 0 / 0 / 15 / 3 / 0 / 0 |
| INVALID_DATE | ft-qwen2.5-0.5b-lora | 18 | 16.7% | 22 | 13.6% | 3 / 0 / 9 / 6 / 0 / 0 |
| INVALID_DATE | api-claude-haiku-5-5-fewshot | 18 | 100.0% | 18 | 100.0% | 18 / 0 / 0 / 0 / 0 / 0 |
| INVALID_BIC | base-qwen2.5-0.5b-fewshot | 7 | 0.0% | 0 | n/a | 0 / 0 / 1 / 4 / 2 / 0 |
| INVALID_BIC | ft-qwen2.5-0.5b-lora | 7 | 0.0% | 4 | 0.0% | 0 / 0 / 1 / 2 / 4 / 0 |
| INVALID_BIC | api-claude-haiku-5-5-fewshot | 7 | 85.7% | 6 | 100.0% | 6 / 0 / 0 / 1 / 0 / 0 |
| NON_POSITIVE_AMOUNT | base-qwen2.5-0.5b-fewshot | 12 | 0.0% | 0 | n/a | 0 / 0 / 5 / 7 / 0 / 0 |
| NON_POSITIVE_AMOUNT | ft-qwen2.5-0.5b-lora | 12 | 0.0% | 0 | n/a | 0 / 0 / 3 / 0 / 9 / 0 |
| NON_POSITIVE_AMOUNT | api-claude-haiku-5-5-fewshot | 12 | 91.7% | 11 | 100.0% | 11 / 0 / 1 / 0 / 0 / 0 |
| INVALID_CHARGES | base-qwen2.5-0.5b-fewshot | 16 | 0.0% | 3 | 0.0% | 0 / 0 / 2 / 14 / 0 / 0 |
| INVALID_CHARGES | ft-qwen2.5-0.5b-lora | 16 | 12.5% | 53 | 3.8% | 2 / 0 / 0 / 14 / 0 / 0 |
| INVALID_CHARGES | api-claude-haiku-5-5-fewshot | 16 | 100.0% | 17 | 94.1% | 16 / 0 / 0 / 0 / 0 / 0 |
| MISSING_REQUIRED | base-qwen2.5-0.5b-fewshot | 6 | 0.0% | 0 | n/a | 0 / 0 / 6 / 0 / 0 / 0 |
| MISSING_REQUIRED | ft-qwen2.5-0.5b-lora | 6 | 0.0% | 0 | n/a | 0 / 3 / 1 / 0 / 2 / 0 |
| MISSING_REQUIRED | api-claude-haiku-5-5-fewshot | 6 | 100.0% | 6 | 100.0% | 6 / 0 / 0 / 0 / 0 / 0 |

False alarms (any error flagged on an example whose gold `errors` is empty):

- base-qwen2.5-0.5b-fewshot: 1.3% of 231 clean examples
- ft-qwen2.5-0.5b-lora: 31.2% of 231 clean examples
- api-claude-haiku-5-5-fewshot: 0.4% of 231 clean examples

## One example per error type (first occurrence)

### base-qwen2.5-0.5b-fewshot

- **value_from_other_field** `syn-00003` field `ordering_customer.address`: gold `null`, predicted `"Plot 7, MIDC, Chennai 600002 IN"`
- **hallucinated_absent_field** `syn-00027` field `bank_operation_code`: gold `null`, predicted `"CRED"`, gold errors ['INVALID_DATE: value_date: 2027-00-12 is not a real calendar date']
- **missed_present_value** `syn-00003` field `beneficiary.address`: gold `"Plot 7, MIDC, Chennai 600002 IN"`, predicted `null`
- **string_wrong_value** `syn-00003` field `ordering_customer.account`: gold `"726045614514"`, predicted `"260905INR2681837.84"`
- **date_misparsed** `syn-00003` field `value_date`: gold `"2026-09-05"`, predicted `"2026-11-02"`

### ft-qwen2.5-0.5b-lora

- **missed_present_value** `syn-00003` field `amount`: gold `2681837.84`, predicted `null`
- **repaired_impossible_value** `syn-00027` field `value_date`: gold `null`, predicted `"2027-01-12"`, gold errors ['INVALID_DATE: value_date: 2027-00-12 is not a real calendar date']
- **value_from_other_field** `syn-00003` field `ordering_customer.address`: gold `null`, predicted `"726045614514"`
- **string_wrong_value** `syn-00080` field `account_with_institution`: gold `"CHASUS33"`, predicted `"SWIFT"`
- **amount_scale** `syn-00351` field `amount`: gold `4565395.5`, predicted `4565.3955`

### api-claude-haiku-5-5-fewshot

- **string_partial** `syn-00382` field `remittance_info`: gold `"Software subscription 9763"`, predicted `"for software subscription 9763"`
- **string_wrong_value** `syn-00166` field `sender_reference`: gold `"TXN578958"`, predicted `"TXN610765"`
- **amount_misparsed** `syn-00166` field `amount`: gold `6052149.0`, predicted `5877.79`
- **missed_present_value** `syn-00046` field `ordering_institution`: gold `"DEUTDEFF"`, predicted `null`
- **date_misparsed** `syn-00114` field `value_date`: gold `"2027-01-24"`, predicted `"2024-01-27"`

