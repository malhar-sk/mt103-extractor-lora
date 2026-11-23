# Error analysis: `ood` (data/ood_test.jsonl, n = 100, SYNTHETIC)

Generated 2026-11-23 by `python evals/analyze.py --data data/ood_test.jsonl --runs ood-base-qwen2.5-0.5b-fewshot ood-ft-qwen2.5-0.5b-lora ood-api-claude-haiku-5-5-fewshot --tag ood` from cached model outputs (no new model calls).

## Per-field accuracy

| Field | ood-base-qwen2.5-0.5b-fewshot | ood-ft-qwen2.5-0.5b-lora | ood-api-claude-haiku-5-5-fewshot |
|---|---|---|---|
| sender_reference | 68.0% | 61.0% | 100.0% |
| bank_operation_code | 50.0% | 79.0% | 100.0% |
| value_date | 55.0% | 64.0% | 99.0% |
| currency | 75.0% | 85.0% | 99.0% |
| amount | 75.0% | 76.0% | 90.0% |
| ordering_customer.account | 39.0% | 59.0% | 97.0% |
| ordering_customer.name | 48.0% | 71.0% | 100.0% |
| ordering_customer.address | 38.0% | 76.0% | 100.0% |
| ordering_institution | 24.0% | 39.0% | 96.0% |
| account_with_institution | 25.0% | 46.0% | 94.0% |
| beneficiary.account | 32.0% | 67.0% | 100.0% |
| beneficiary.name | 36.0% | 67.0% | 100.0% |
| beneficiary.address | 42.0% | 75.0% | 100.0% |
| remittance_info | 52.0% | 73.0% | 100.0% |
| charges | 54.0% | 62.0% | 97.0% |

## Field error types: ood-base-qwen2.5-0.5b-fewshot (787 wrong field values out of 1500)

| Error type | Count | Share | Worst fields | Meaning |
|---|---|---|---|---|
| value_from_other_field | 278 | 35.3% | ordering_customer.name (46), beneficiary.name (46), account_with_institution (38) | the model put another field's value here (e.g. 52A/57A BICs swapped, account in address) |
| string_wrong_value | 131 | 16.6% | sender_reference (29), beneficiary.account (27), ordering_customer.account (22) | a different string altogether |
| hallucinated_absent_field | 99 | 12.6% | bank_operation_code (46), ordering_customer.address (19), account_with_institution (14) | the field is not in the text (gold null), the model invented a value |
| missed_present_value | 84 | 10.7% | ordering_institution (50), beneficiary.address (34) | the value is in the text, the model returned null |
| enum_wrong_value | 53 | 6.7% | charges (35), currency (15), bank_operation_code (3) | wrong code (charges / op code / currency) |
| repaired_impossible_value | 44 | 5.6% | value_date (11), charges (9), account_with_institution (7) | the source value was impossible (gold null + error), the model output a plausible value instead of null |
| invalid_json | 35 | 4.4% | sender_reference (3), value_date (3), ordering_customer.account (3) | output was not parseable JSON, so every present field is lost |
| date_misparsed | 28 | 3.6% | value_date (28) | date wrong in another way |
| amount_misparsed | 17 | 2.2% | amount (17) | amount wrong in another way |
| string_partial | 13 | 1.7% | remittance_info (4), ordering_customer.address (3), beneficiary.account (2) | right value with text missing or extra text attached |
| date_wrong_year | 2 | 0.3% | value_date (2) | right day and month, wrong year |
| amount_scale | 2 | 0.3% | amount (2) | amount off by a power of ten (thousands/decimal separator misread) |
| date_day_month_swap | 1 | 0.1% | value_date (1) | day and month swapped |

## Field error types: ood-ft-qwen2.5-0.5b-lora (500 wrong field values out of 1500)

| Error type | Count | Share | Worst fields | Meaning |
|---|---|---|---|---|
| value_from_other_field | 159 | 31.8% | account_with_institution (26), ordering_customer.name (24), beneficiary.name (23) | the model put another field's value here (e.g. 52A/57A BICs swapped, account in address) |
| missed_present_value | 125 | 25.0% | ordering_institution (49), bank_operation_code (18), account_with_institution (11) | the value is in the text, the model returned null |
| string_wrong_value | 73 | 14.6% | sender_reference (37), ordering_customer.account (18), account_with_institution (4) | a different string altogether |
| repaired_impossible_value | 37 | 7.4% | value_date (9), charges (7), currency (7) | the source value was impossible (gold null + error), the model output a plausible value instead of null |
| enum_wrong_value | 36 | 7.2% | charges (27), currency (6), bank_operation_code (3) | wrong code (charges / op code / currency) |
| date_misparsed | 24 | 4.8% | value_date (24) | date wrong in another way |
| hallucinated_absent_field | 14 | 2.8% | remittance_info (8), account_with_institution (5), ordering_customer.address (1) | the field is not in the text (gold null), the model invented a value |
| invalid_json | 11 | 2.2% | sender_reference (1), value_date (1), ordering_customer.account (1) | output was not parseable JSON, so every present field is lost |
| amount_misparsed | 9 | 1.8% | amount (9) | amount wrong in another way |
| string_partial | 7 | 1.4% | remittance_info (4), account_with_institution (1), ordering_customer.address (1) | right value with text missing or extra text attached |
| amount_scale | 3 | 0.6% | amount (3) | amount off by a power of ten (thousands/decimal separator misread) |
| date_day_month_swap | 1 | 0.2% | value_date (1) | day and month swapped |
| date_wrong_year | 1 | 0.2% | value_date (1) | right day and month, wrong year |

## Field error types: ood-api-claude-haiku-5-5-fewshot (28 wrong field values out of 1500)

| Error type | Count | Share | Worst fields | Meaning |
|---|---|---|---|---|
| missed_present_value | 12 | 42.9% | account_with_institution (4), ordering_customer.account (3), charges (3) | the value is in the text, the model returned null |
| amount_misparsed | 7 | 25.0% | amount (7) | amount wrong in another way |
| repaired_impossible_value | 5 | 17.9% | amount (3), account_with_institution (1), currency (1) | the source value was impossible (gold null + error), the model output a plausible value instead of null |
| value_from_other_field | 4 | 14.3% | ordering_institution (3), account_with_institution (1) | the model put another field's value here (e.g. 52A/57A BICs swapped, account in address) |

## Error-flag analysis by error code

Recall = gold errors flagged with the right CODE:field. Outcomes for each gold error: **ok** flagged correctly, **wrong** flagged with another code/field, **repaired** no flag and a plausible value invented (silent, nothing downstream catches it), **copied** no flag but the impossible value was copied through (the API's pydantic validation still catches it), **null** no flag, field left null, **json** invalid JSON.

| Code | Model | Gold n | Recall | Pred n | Precision | ok / wrong / repaired / copied / null / json |
|---|---|---|---|---|---|---|
| INVALID_CURRENCY | ood-base-qwen2.5-0.5b-fewshot | 7 | 0.0% | 0 | n/a | 0 / 0 / 4 / 3 / 0 / 0 |
| INVALID_CURRENCY | ood-ft-qwen2.5-0.5b-lora | 7 | 0.0% | 23 | 0.0% | 0 / 1 / 3 / 3 / 0 / 0 |
| INVALID_CURRENCY | ood-api-claude-haiku-5-5-fewshot | 7 | 85.7% | 6 | 100.0% | 6 / 0 / 1 / 0 / 0 / 0 |
| INVALID_DATE | ood-base-qwen2.5-0.5b-fewshot | 7 | 0.0% | 0 | n/a | 0 / 0 / 2 / 5 / 0 / 0 |
| INVALID_DATE | ood-ft-qwen2.5-0.5b-lora | 7 | 14.3% | 10 | 10.0% | 1 / 0 / 2 / 4 / 0 / 0 |
| INVALID_DATE | ood-api-claude-haiku-5-5-fewshot | 7 | 100.0% | 8 | 87.5% | 7 / 0 / 0 / 0 / 0 / 0 |
| INVALID_BIC | ood-base-qwen2.5-0.5b-fewshot | 9 | 0.0% | 0 | n/a | 0 / 0 / 5 / 2 / 1 / 1 |
| INVALID_BIC | ood-ft-qwen2.5-0.5b-lora | 9 | 0.0% | 2 | 0.0% | 0 / 0 / 2 / 4 / 3 / 0 |
| INVALID_BIC | ood-api-claude-haiku-5-5-fewshot | 9 | 88.9% | 8 | 100.0% | 8 / 0 / 1 / 0 / 0 / 0 |
| NON_POSITIVE_AMOUNT | ood-base-qwen2.5-0.5b-fewshot | 3 | 0.0% | 0 | n/a | 0 / 0 / 3 / 0 / 0 / 0 |
| NON_POSITIVE_AMOUNT | ood-ft-qwen2.5-0.5b-lora | 3 | 0.0% | 0 | n/a | 0 / 0 / 2 / 0 / 1 / 0 |
| NON_POSITIVE_AMOUNT | ood-api-claude-haiku-5-5-fewshot | 3 | 0.0% | 0 | n/a | 0 / 0 / 3 / 0 / 0 / 0 |
| INVALID_CHARGES | ood-base-qwen2.5-0.5b-fewshot | 5 | 0.0% | 3 | 0.0% | 0 / 0 / 2 / 2 / 0 / 1 |
| INVALID_CHARGES | ood-ft-qwen2.5-0.5b-lora | 5 | 0.0% | 9 | 0.0% | 0 / 1 / 0 / 4 / 0 / 0 |
| INVALID_CHARGES | ood-api-claude-haiku-5-5-fewshot | 5 | 100.0% | 7 | 71.4% | 5 / 0 / 0 / 0 / 0 / 0 |
| MISSING_REQUIRED | ood-base-qwen2.5-0.5b-fewshot | 12 | 0.0% | 0 | n/a | 0 / 0 / 12 / 0 / 0 / 0 |
| MISSING_REQUIRED | ood-ft-qwen2.5-0.5b-lora | 12 | 0.0% | 0 | n/a | 0 / 2 / 7 / 0 / 2 / 1 |
| MISSING_REQUIRED | ood-api-claude-haiku-5-5-fewshot | 12 | 100.0% | 13 | 92.3% | 12 / 0 / 0 / 0 / 0 / 0 |

## Field accuracy (macro) by input style

| Style | n | ood-base-qwen2.5-0.5b-fewshot | ood-ft-qwen2.5-0.5b-lora | ood-api-claude-haiku-5-5-fewshot |
|---|---|---|---|---|
| kv_mixed | 26 | 54.6% | 76.4% | 98.7% |
| narrative | 27 | 39.3% | 71.4% | 96.8% |
| pipe_table | 24 | 56.1% | 71.7% | 99.2% |
| swift_shuffled | 23 | 40.3% | 44.9% | 98.0% |

False alarms (any error flagged on an example whose gold `errors` is empty):

- ood-base-qwen2.5-0.5b-fewshot: 1.8% of 57 clean examples
- ood-ft-qwen2.5-0.5b-lora: 47.4% of 57 clean examples
- ood-api-claude-haiku-5-5-fewshot: 1.8% of 57 clean examples

## One example per error type (first occurrence)

### ood-base-qwen2.5-0.5b-fewshot

- **value_from_other_field** `ood-0000` field `ordering_customer.account`: gold `"DE457500407933791566"`, predicted `"844420567-96"`
- **string_wrong_value** `ood-0000` field `sender_reference`: gold `"FT2660655795"`, predicted `"TXN942429"`
- **hallucinated_absent_field** `ood-0002` field `bank_operation_code`: gold `null`, predicted `"CRED"`
- **missed_present_value** `ood-0001` field `ordering_institution`: gold `"CHASUS33"`, predicted `null`
- **enum_wrong_value** `ood-0000` field `bank_operation_code`: gold `"CRED"`, predicted `"SCH"`

### ood-ft-qwen2.5-0.5b-lora

- **value_from_other_field** `ood-0000` field `account_with_institution`: gold `"SCBLSGSG"`, predicted `"MHCBJPJT"`
- **missed_present_value** `ood-0000` field `ordering_institution`: gold `"MHCBJPJT"`, predicted `null`
- **string_wrong_value** `ood-0000` field `sender_reference`: gold `"FT2660655795"`, predicted `"DE457500407933791566"`
- **repaired_impossible_value** `ood-0005` field `charges`: gold `null`, predicted `"OUR"`, gold errors ['MISSING_REQUIRED: charges: not stated in the text']
- **enum_wrong_value** `ood-0001` field `charges`: gold `"BEN"`, predicted `"TXN247517"`

### ood-api-claude-haiku-5-5-fewshot

- **missed_present_value** `ood-0003` field `account_with_institution`: gold `"BNPAFRPP"`, predicted `null`
- **amount_misparsed** `ood-0011` field `amount`: gold `6949192.52`, predicted `6949190`
- **repaired_impossible_value** `ood-0021` field `amount`: gold `null`, predicted `9000`, gold errors ['NON_POSITIVE_AMOUNT: amount: (9000) is not a positive amount']
- **value_from_other_field** `ood-0003` field `ordering_institution`: gold `null`, predicted `"BNPAFRPP"`

