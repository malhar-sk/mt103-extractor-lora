# RESULTS.md

Every number here was measured on the build VPS on **2026-11-23** (2 vCPU shared, no GPU; Ollama container capped at
`cpus: 1.0`, `num_thread 1`). Dataset: **test split, 300 synthetic examples** (`data/test.jsonl`, seed 103) unless
stated otherwise. Raw model outputs are in `results/cache/*.jsonl`; full metrics are in `results/eval_*.json`.

## Headline numbers

| Metric | Base Qwen2.5-0.5B (4-shot) | Fine-tuned 0.5B LoRA (**CPU smoke run, not fully trained**) |
|---|---|---|
| Field exact-match accuracy (macro, 15 fields) | **62.6%** | **94.2%** |
| Valid-JSON rate | 100.0% | 100.0% |
| Schema-valid rate | 59.0% | 83.3% |
| Tokens/sec on CPU (Ollama q4_K_M, 1 thread) | 13.5 | 22.4 |

The tokens/sec gap is mostly the prompt, not the model. Both are the same 0.5B architecture at q4_K_M, but the base
model gets a long 4-shot prompt (`num_ctx 4096`) and the fine-tuned one a one-line prompt (`num_ctx 2048`).

## Full eval table (test split, n = 300)

| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Base Qwen2.5-0.5B-Instruct (Ollama q4_K_M, 4-shot prompt) | 300 | 100.0% | 59.0% | 62.6% | 1.0% | 0.0% | 0.0% | 13.0 | 13.5 |
| Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained** | 300 | 100.0% | 83.3% | 94.2% | 51.7% | 5.3% | 7.2% | 9.3 | 22.4 |
| Claude Haiku 5.5 via API (4-shot prompt), reference ceiling | 300 | 100.0% | 97.0% | 99.1% | 91.3% | 98.5% | 97.1% | 5.6 | 213.9 |

Per-field exact match:

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

Commands:

- `base-qwen2.5-0.5b-fewshot`: `python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --name base-qwen2.5-0.5b-fewshot` (2026-11-23)
- `ft-qwen2.5-0.5b-lora`: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --name ft-qwen2.5-0.5b-lora --ollama-url http://127.0.0.1:21434` (2026-11-23)
- `api-claude-haiku-5-5-fewshot`: `python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --workers 3 --name api-claude-haiku-5-5-fewshot` (2026-11-23)

Commands are run from the repo root with `.venv/bin/python`, and the Ollama URL is `http://127.0.0.1:21434`. The table
is produced by `python evals/report.py`. Exact match is case- and whitespace-normalised, and amounts are compared to 2 dp.
"Err precision/recall" counts (code, field) pairs in the `errors` list against the generator's ground truth.

### Reading it honestly
- The smoke run (300 training examples seen, 0.125 epoch) already learnt the **format and field mapping** (62.6% → 94.2%),
  but it has **barely learnt error flagging** (precision 5.3%, recall 7.2%). It tends to "repair" impossible values,
  e.g. `USX` → `USD`, instead of flagging them (see `examples/extract_example.json`). The API still catches many of
  these, because pydantic validation reports them in `validation_errors`. A full GPU run is needed for this.
- The Claude Haiku row is a **reference ceiling**. It was called through the logged-in `claude` CLI (`--backend claude-cli`),
  because `ANTHROPIC_API_KEY` was empty on the host, so its latency includes CLI start-up (about 0.6 s) and network, and its
  tokens/sec is not comparable to local CPU numbers.

## Training (M3): CPU smoke run, not fully trained

| Item | Value |
|---|---|
| Mode | `smoke` (no GPU: `nvidia-smi` not present) |
| Base | Qwen2.5-0.5B-Instruct, bf16, LoRA r=16 alpha=32 dropout=0.05, all attention+MLP projections |
| Steps | 150 (batch 1 × grad-accum 2, so 300 of 2,400 train examples, 0.125 epoch), lr 2e-4 cosine, max len 1024 |
| Wall time | 5,280 s (88 min) |
| Train loss (completion tokens) | 1.158 (step 5) → 0.023 (step 150) |
| Val loss (40 val examples) | 0.063 (step 50), 0.037 (step 100), 0.030 (step 150) |
| Val generation (20 val examples, step 150) | valid JSON 100%, schema-valid 80%, field acc 94.0%, err P/R 0% / 0% |
| Command | `make train` (= `python train/train_lora.py`) |
| Artefacts | `results/loss.png`, `results/train_log.json`, `models/adapter/` (35 MB, committed) |

"Validation metric per epoch": the smoke run is shorter than one epoch, so val metrics are logged every 50 steps instead.
The full 3-epoch QLoRA run on Qwen2.5-1.5B (`notebooks/kaggle_train.ipynb`) is **NOT MEASURED**: there is no GPU on the
build host, and it is meant for Kaggle (TESTING.md step 6).

## Serving (M4)

- Merge, then GGUF f16, then q4_K_M (398 MB), then `ollama create mt103-extractor`: done on the VPS with `make merge`
  (llama.cpp built from source, `llama-quantize` 9.2 s).
- Recorded `/extract` calls with the fine-tuned model: `examples/extract_example.json` (latency 10.2 s and 8.2 s,
  22.4 and 23.4 tok/s, `fallback: false`).
- Fallback to the base model when the fine-tuned model is absent: covered by `tests/test_api.py` (mocked Ollama).

## Not measured / skipped
- **Base Qwen2.5-1.5B baseline:** NOT MEASURED. The plan asks for it only with a GPU, and there is none.
- **Fine-tuned 1.5B (full QLoRA):** NOT MEASURED, for the same reason. The notebook is provided.
- **M5 (investment-committee integration):** skipped: source not provided (`~/inputs/multi-invest-committee` does not exist).

## Tests
`make test` gives **33 passed** (2026-11-23). This covers schema validation, 100% of the 3,000 targets being
schema-valid, no split overlap, generator determinism, the eval script on 5 examples with a mock model, and the
API with a mocked Ollama.

---

# Eval round 2 (added 2026-11-23)

Everything above this line is unchanged. All numbers below were measured on the same VPS on **2026-11-23**, with the
same models, the same eval script and metrics, Ollama capped at `cpus: 1.0`. Analyses are recomputed from the cached
raw outputs (`results/cache/*.jsonl`) by `evals/analyze.py`, which asserts that its per-field accuracy matches the
eval script's exactly.

## 1. Out-of-distribution test set (n = 100, SYNTHETIC)

`data/ood_test.jsonl` is **synthetic**, generated by `python data/generate_ood.py --n 100 --seed 20261123` (the training seed was 103).
It uses 4 renderers the model never saw in training: `swift_shuffled` (MT103 tags in random order, 60% lower-case `:32a:`/`:50k:`/`:71a:`, n=23),
`narrative` (free-text paragraph, facts in random order, n=27), `pipe_table` (`| key | value |`, unseen key names, n=24) and
`kv_mixed` (`key=value; ...`, n=26). It also adds OCR-style swaps on labels only (`Valuc`, `Credltor`, `Scnder ref`), extra
remarks with distractor amounts, dates and references, new date formats (`5th March 2026`, `2026/03/05`, `20260305`,
`Mon 17 Aug 26`, `09-AUG-2026`), and impossible values from new lists (`EURO`, `ICIC0INBB`, `2026-02-30`, `SPLIT`,
`(9000)`). 43 of the 100 examples have a gold error. Tests check that it has no text overlap with train, and that its bad values are disjoint from
the training lists (`tests/test_ood_analysis.py`).

| Model | n | Valid JSON | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall | Latency p50 (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Base Qwen2.5-0.5B-Instruct (Ollama q4_K_M, 4-shot prompt) | 100 | 97.0% | 62.0% | 47.5% | 0.0% | 0.0% | 0.0% | 11.2 | 19.4 |
| Fine-tuned Qwen2.5-0.5B LoRA (GGUF q4_K_M, short prompt). **CPU smoke run, not fully trained** | 100 | 99.0% | 36.0% | 66.7% | 5.0% | 2.2% | 2.3% | 9.0 | 24.8 |
| Claude Haiku 5.5 via `claude` CLI (4-shot prompt), reference ceiling | 100 | 100.0% | 100.0% | 98.1% | 78.0% | 90.5% | 88.4% | 5.8 | 230.8 |

In-distribution vs OOD, field accuracy (macro):

| Model | Test split (n=300) | OOD (n=100) | Drop |
|---|---|---|---|
| Base 0.5B, 4-shot | 62.6% | 47.5% | −15.1 pp |
| Fine-tuned 0.5B LoRA (smoke) | 94.2% | 66.7% | **−27.5 pp** |
| Claude Haiku 5.5 | 99.1% | 98.1% | −1.0 pp |

Field accuracy (macro) by OOD input style (from `results/analysis_ood.md`):

| Style | n | Base 0.5B | Fine-tuned 0.5B | Haiku |
|---|---|---|---|---|
| kv_mixed | 26 | 54.6% | 76.4% | 98.7% |
| narrative | 27 | 39.3% | 71.4% | 96.8% |
| pipe_table | 24 | 56.1% | 71.7% | 99.2% |
| swift_shuffled | 23 | 40.3% | **44.9%** | 98.0% |

Full per-field table: `results/eval_table_ood.md`. Commands (repo root, `.venv/bin/python`, Ollama on `http://127.0.0.1:21434`):

- `python evals/run_eval.py --backend ollama --model qwen2.5:0.5b-instruct --fewshot --data data/ood_test.jsonl --name ood-base-qwen2.5-0.5b-fewshot`
- `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/ood_test.jsonl --name ood-ft-qwen2.5-0.5b-lora`
- `python evals/run_eval.py --backend claude-cli --model claude-haiku-5-5 --fewshot --workers 3 --data data/ood_test.jsonl --name ood-api-claude-haiku-5-5-fewshot`
- or simply `make eval-ood` (tables: `python evals/report.py`, analysis: `make analyze`)

**What it shows:**
- **Most of the 94.2% is memorised format.** Fine-tuning still beats the base model by +19.2 pp on unseen formats, so
  some extraction skill does transfer. But the fine-tuned model loses 27.5 pp, nearly twice the base model's drop.
  Haiku barely moves.
- The worst case is `swift_shuffled`, at 44.9%. It is the format closest to the training data: the same MT103 tags,
  only reordered and sometimes lower-case. The smoke model seems to map fields by **position**, not by tag. Its main
  OOD error type is `value_from_other_field` (159 of 500 wrong values), e.g. the beneficiary name put in the ordering
  customer.
- Schema-valid falls to 36% for the fine-tuned model. Almost all of that comes from its error strings and invented values, not from broken JSON
  (99% valid JSON).
- 3 base-model outputs are counted as invalid JSON because Ollama aborted them with `token repeat limit reached`
  (the model looped). This reproduces on re-run, so it is a model failure and is scored as one.
- Haiku's one blind spot: 0 of 3 `NON_POSITIVE_AMOUNT` in the new notations (`(9000)` accounting parentheses,
  `- 86885.10` with a space). It returned the positive amount and no flag. Whether `(9000)` means negative is
  domain convention, but the generator labels it so, and the count is tiny (n=3).

## 2. Per-field error breakdown: fine-tuned model, test split (n = 300)

Source: `results/analysis_test.md` (command: `python evals/analyze.py --data data/test.jsonl --runs base-qwen2.5-0.5b-fewshot ft-qwen2.5-0.5b-lora api-claude-haiku-5-5-fewshot --tag test`).
The fine-tuned model got 261 of 4,500 field values (300 × 15) wrong.

Fields that still fail, worst first (accuracy, then main error types):

| Field | Accuracy | Main error types (count) |
|---|---|---|
| account_with_institution (57A) | 80.3% | missed_present_value 27, value_from_other_field 15 (52A/57A swap), hallucinated 8 |
| ordering_institution (52A) | 87.7% | missed_present_value 20, value_from_other_field 14 (52A/57A swap) |
| bank_operation_code (23B) | 89.7% | missed_present_value 30 |
| amount | 90.0% | amount_scale 14, amount_misparsed 7, missed 6, repaired_impossible 3 |
| charges | 90.0% | repaired_impossible_value 16, missed 8, enum_wrong 6 |
| value_date | 91.0% | repaired_impossible_value 19, date_misparsed 8 |
| currency | 96.7% | repaired_impossible_value 10 (all of its errors) |
| all other 8 fields | ≥ 97.3% | |

The 5 most common error types, with the first example of each:

| # | Error type | Count | Share | Example |
|---|---|---|---|---|
| 1 | missed_present_value: value in the text, model returns null | 100 | 38.3% | `syn-00003` amount: SWIFT `:32A:260905INR2681837,84`, gold `2681837.84`, predicted `null` |
| 2 | repaired_impossible_value: impossible source value "fixed" instead of nulled and flagged | 51 | 19.5% | `syn-00027` value_date: source `2027-00-12`, gold `null` + INVALID_DATE, predicted `"2027-01-12"` |
| 3 | value_from_other_field: another field's value put here | 37 | 14.2% | `syn-00003` ordering_customer.address: gold `null`, predicted the account number `"726045614514"` |
| 4 | string_wrong_value: a different string | 18 | 6.9% | `syn-00080` account_with_institution: gold `"CHASUS33"`, predicted `"SWIFT"` (the label) |
| 5 | amount_scale: off by a power of ten | 14 | 5.4% | `syn-00351` amount: gold `4565395.5`, predicted `4565.3955` (thousands separator read as decimal) |

**Which fields still fail?** The two BIC fields (52A/57A) and 23B: the model drops them or swaps 52A and 57A. Then
amount, charges and value_date, where the model rewrites impossible values instead of flagging them (section 3).

## 3. Error-flag analysis by error type (test split n = 300; OOD n = 100)

For each gold error (`CODE:field`), what the model did: **ok** = flagged correctly; **wrong** = flagged with another
code or field; **repaired** = no flag, a plausible value invented (silent, since nothing downstream can catch it);
**copied** = no flag, the impossible value copied through (the API's pydantic validation still reports it in
`validation_errors`); **null** = no flag, field left null; **json** = invalid output. Precision = correct flags / all flags
of that code.

Test split (n = 300; 69 gold errors in 69 examples; 231 clean examples):

| Code | Gold n | Base recall | FT recall | FT outcomes ok/wrong/repaired/copied/null | FT flags emitted (precision) | Haiku recall |
|---|---|---|---|---|---|---|
| INVALID_CURRENCY | 10 | 0% | 0% | 0 / 0 / 8 / 2 / 0 | 16 (0.0%) | 100% |
| INVALID_DATE | 18 | 0% | 16.7% | 3 / 0 / 9 / 6 / 0 | 22 (13.6%) | 100% |
| INVALID_BIC | 7 | 0% | 0% | 0 / 0 / 1 / 2 / 4 | 4 (0.0%) | 85.7% |
| NON_POSITIVE_AMOUNT | 12 | 0% | 0% | 0 / 0 / 3 / 0 / 9 | 0 (n/a) | 91.7% |
| INVALID_CHARGES | 16 | 0% | 12.5% | 2 / 0 / 0 / 14 / 0 | 53 (3.8%) | 100% |
| MISSING_REQUIRED | 6 | 0% | 0% | 0 / 3 / 1 / 0 / 2 | 0 (n/a) | 100% |
| **False alarms** (any flag on a clean example) | 231 clean | 1.3% | **31.2%** | | | 0.4% |

OOD (n = 100; 43 gold errors; 57 clean examples):

| Code | Gold n | Base recall | FT recall | FT outcomes ok/wrong/repaired/copied/null/json | Haiku recall |
|---|---|---|---|---|---|
| INVALID_CURRENCY | 7 | 0% | 0% | 0 / 1 / 3 / 3 / 0 / 0 | 85.7% |
| INVALID_DATE | 7 | 0% | 14.3% | 1 / 0 / 2 / 4 / 0 / 0 | 100% |
| INVALID_BIC | 9 | 0% | 0% | 0 / 0 / 2 / 4 / 3 / 0 | 88.9% |
| NON_POSITIVE_AMOUNT | 3 | 0% | 0% | 0 / 0 / 2 / 0 / 1 / 0 | 0% |
| INVALID_CHARGES | 5 | 0% | 0% | 0 / 1 / 0 / 4 / 0 / 0 | 100% |
| MISSING_REQUIRED | 12 | 0% | 0% | 0 / 2 / 7 / 0 / 2 / 1 | 100% |
| **False alarms** | 57 clean | 1.8% | **47.4%** | | 1.8% |

Full per-model outcome tables: `results/analysis_test.md`, `results/analysis_ood.md`.

**What the full Kaggle run needs to fix:**
1. **The error template is learnt, its trigger is not.** The smoke model writes well-formed error strings on valid
   values, for example `INVALID_CHARGES: charges: SHA is not OUR/SHA/BEN` (51 of its 53 INVALID_CHARGES flags are
   false) and `INVALID_CURRENCY: currency: INR is not an ISO-4217 code`. This gives a 31% false-alarm rate on clean
   examples. It needs more training steps: the smoke run saw 300 examples, so about 66 with errors at the generator's 22% error rate.
2. **Silent repairs** (`2027-00-12` → `2027-01-12`, `USX` → `USD`): 22 of 69 test errors (32%) and 16 of 43 OOD
   errors are rewritten into plausible values. This is the dangerous case, because pydantic validation cannot catch
   it. Fix: train to convergence, and oversample INVALID_DATE / INVALID_CURRENCY / NON_POSITIVE_AMOUNT examples.
3. **MISSING_REQUIRED is never flagged** (0/6 test, 0/12 OOD). Missing date or charges come out as `INVALID_DATE … null`
   or nothing at all; on OOD a missing beneficiary/date is often invented.
4. A cheap safety net, independent of training: **copied** cases (24 on test, 15 on OOD) are already caught by the API's
   pydantic validation. A rule-based post-check (ISO-4217 list, BIC regex, `date.fromisoformat`, amount > 0) could
   also turn them into proper `errors` entries.

## Tests (round 2)
`make test` gives **50 passed** (2026-11-23). The 17 new tests (`tests/test_ood_analysis.py`) cover: the OOD set is
synthetic, schema-valid, reproducible from its seed and disjoint from train; its bad values are disjoint from the
training lists; plus the field-error classifier and the error-flag outcome logic on hand-built cases.

---

# Eval round 3 (added 2026-11-24): OOD diagnosis, format-diverse data, post-processing

Everything above this line is unchanged. Below: **MEASURED** = computed on the build VPS on 2026-11-24. No model was
retrained in sections 1–4; section 5 (added 2026-11-25) is the retrained model from the Kaggle GPU run.

## 1. Which fields fail out of distribution (MEASURED, fine-tuned smoke model, OOD n = 100)

| Field | Test split | OOD | Main OOD failure (from `results/analysis_ood.md`) |
|---|---|---|---|
| ordering_institution (52A) | 87.7% | **39.0%** | missed 49×: when only one BIC is given, the model puts it in 57A (position, not label) |
| account_with_institution (57A) | 80.3% | **46.0%** | the 52A BIC put here (value_from_other_field 26×) |
| ordering_customer.account | 98.3% | 59.0% | wrong string / beneficiary account swapped in |
| sender_reference | 99.3% | 61.0% | wrong string 37×: takes the first ID-like token (an account, a cancelled "previous ref") |
| charges | 90.0% | 62.0% | wrong code 27× for unseen wordings, plus false INVALID_CHARGES flags |
| value_date | 91.0% | 64.0% | unseen date formats misparsed 24×; impossible dates "repaired" |
| beneficiary.name / account | 98.7% / 98.3% | 67.0% / 67.0% | ordering and beneficiary parties swapped |
| all other fields | ≥ 96% | 71–85% | |

By format: `swift_shuffled` 44.9%, `narrative` 71.4%, `pipe_table` 71.7%, `kv_mixed` 76.4%. The common thread is
that the smoke model learnt **where** a value sits in the v1 templates, not **what label** it carries: the two BIC
slots, the two parties and the reference are all decided by position.

## 2. What changed

1. **Training data v2** (`data/generate_v2.py`, 6,000 examples, seed 2026, committed in `data/v2/`): 10 new
   renderer families (chat logs with timestamps, SWIFT with distractor tags `:13C: :33B: :53A: :56A: :71F: :72:`,
   aligned forms, sectioned forms with multi-line parties, CSV, YAML-like blocks, formal letters, German / French /
   Spanish / Hindi-Latin labels, forwarded threads with quoted cancelled instructions), v1's styles kept for 16%;
   6 to 13 label synonyms per field and random field order; a lone 52A is as common as a lone 57A; distractor amounts,
   references, dates, IFSC codes and phone numbers; `1,25,000/-`, `12.5 lakh`, `1.2 crore`, Swiss and European
   grouping; typo noise; 30% of examples carry an error (v1: 22%). It also drops v1's lossy `%g` amount rendering
   (v1 has examples such as `INR 6.94919e+06` labelled 6949192.52, which the text does not support).
2. **Held-out guarantee.** The OOD generator is untouched and its file is hash-pinned. v2 deliberately does **not**
   use the OOD set's own perturbations, so a better OOD score after retraining would be real generalisation, not
   leakage. `tests/test_v2_heldout.py` checks every v2 text: no `| key | value |` tables, no `key=value;` pairs, no
   shuffled or lower-case SWIFT tags, none of the 28 OOD-only label names, none of the 7 OOD date formats, none of
   the OOD narrative/remark sentences, none of the OOD impossible values, and no text overlap with OOD or the v1
   test split. (The detectors were checked on the OOD file itself: they flag 50 to 84 of its 100 texts each.)
3. **`postprocess.py`**: deterministic rules applied to any model's output, using only the schema and the source
   text (no dataset lists): normalise symbols / lakh / dates; null and flag impossible values only if they literally
   occur in the text; null values that are not in the text (BIC, op code, currency, amount, date, account,
   reference); re-parse an amount whose digits match exactly one amount in the text; drop error flags that contradict
   a valid value (restoring the value); add INVALID_DATE / MISSING_REQUIRED only when the text has an impossible date
   or no date / charges wording at all. **Contract, tested:** every gold target of v1 train/val/test, v2 and OOD
   passes through unchanged. The `/extract` API applies it by default and returns the raw `model_output` too.
   Honesty note: the rules were written while looking at the test split's cached outputs; the OOD formats were
   already documented in round 2, so the date grammar covers them like any general date parser would. It has no
   OOD-specific lists.
4. **Kaggle kernel** (`kaggle/`): one command trains Qwen2.5-0.5B LoRA on v2 and scores v1 test, v2 test and OOD,
   raw and with rules. Run on 2026-11-24 on the owner's account: results in section 5.

## 3. Post-processing on the existing outputs (MEASURED, no new model calls)

Same cached raw outputs as rounds 1 and 2, re-scored with `make eval-pp` (`--postprocess --cache-only`).

| Model | Test split (n=300) | + rules | OOD (n=100) | + rules |
|---|---|---|---|---|
| Base Qwen2.5-0.5B, 4-shot: field acc | 62.6% | **69.4%** | 47.5% | **53.8%** |
| Fine-tuned 0.5B **CPU smoke**: field acc | 94.2% | **95.7%** | 66.7% | **69.5%** |
| Fine-tuned 0.5B CPU smoke: schema-valid | 83.3% | 100.0% | 36.0% | 88.0% |
| Fine-tuned 0.5B CPU smoke: error precision / recall | 5.3% / 7.2% | **59.7% / 53.6%** | 2.2% / 2.3% | **42.9% / 48.8%** |
| Claude Haiku 5.5: field acc | 99.1% | 99.1% | 98.1% | 98.3% |

Field values fixed / broken by the rules (`results/postprocess_delta.md`): fine-tuned 67 / **0** (test) and 43 / **0**
(OOD); base 303 / 0 and 94 / 0; Haiku 1 / 0 and 3 / 0. On OOD the rules fix value_date (64 → 78%), charges (62 → 72%),
currency (85 → 93%) and 57A (46 → 53%). They **cannot** fix the structural confusions: 52A stays at 39%, the
reference at 61%, the parties at 59–67%, and `swift_shuffled` only moves 44.9 → 49.3%. That part needs retraining.

## 4. The existing smoke adapter on the new v2 data (MEASURED, CPU, first 150 of the v2 test split)

This is the "before" row for the Kaggle run: the v1-trained smoke model (Ollama q4_K_M, same prompt) on the new formats.
Command: `python evals/run_eval.py --backend ollama --model mt103-extractor --prompt ft --data data/v2/test.jsonl --limit 150 --name v2-ft-qwen2.5-0.5b-lora` (table: `results/eval_table_v2.md`).

| Model | n | Schema-valid | Field acc (macro) | All fields right | Err precision | Err recall |
|---|---|---|---|---|---|---|
| Fine-tuned 0.5B **CPU smoke** (trained on v1) | 150 | 58.7% | **80.4%** | 16.0% | 1.4% | 2.0% |
| same + rules | 150 | 98.0% | **82.0%** | 18.0% | 36.4% | 48.0% |

Worst fields: 57A 58.7%, 52A 63.3%, charges 66.0%. The v2 test split is not the OOD set (it shares v2's own
templates), so after retraining it measures in-distribution v2 accuracy; OOD stays the generalisation test. The base
model on v2 was started but stopped to keep the shared CPU free: **not measured**.

## 5. Retrained on v2 (MEASURED 2026-11-24, Kaggle T4)

**Run:** `bash kaggle/push.sh` (kernel version 2; version 1 stopped in its first minute on a notebook bug, fixed in
`e59bdcb`). Qwen2.5-0.5B-Instruct, LoRA r16, **fp16 (no QLoRA)**, batch 4 × grad-accum 4, 2 epochs = 600 steps on
the 4,800 v2 training examples, **2,337 s (39 min) on one Tesla T4**. The notebook regenerated all data from the seeds
and checked it against the repo copies by SHA-256 before training. Validation during training (50 val examples):
field acc 98.1% after epoch 1, 99.1% after epoch 2. Training log: `results/kaggle_train_log.json`, `results/kaggle_loss.png`.

**Evaluation:** `evals/run_eval.py --backend hf --adapter models/adapter --prompt ft` on the GPU (fp16 adapter,
batched generation, so the latency column in the tables is not comparable with the CPU rows). The raw outputs are
cached in `results/cache/*kaggle-ft*.jsonl`, and re-scoring them on the build VPS with `make eval-pp` reproduces
every number below exactly (checked 2026-11-25).

| Model | Test split (n=300) | v2 test (n=600) | OOD (n=100) |
|---|---|---|---|
| Fine-tuned 0.5B **CPU smoke** (v1), field acc | 94.2% | 80.4% (first 150) | 66.7% |
| same + rules | 95.7% | 82.0% | 69.5% |
| **Fine-tuned 0.5B, full run on v2**, field acc | **99.3%** | **98.8%** | **89.1%** |
| **same + rules** | **99.4%** | **99.0%** | **90.3%** |
| Claude Haiku 5.5 + rules (reference) | 99.1% | not measured | 98.3% |

| Full run on v2 | Test split | + rules | v2 test | + rules | OOD | + rules |
|---|---|---|---|---|---|---|
| Schema-valid | 98.0% | 100.0% | 97.2% | 100.0% | 91.0% | 98.0% |
| All 15 fields right | 89.3% | 91.3% | 86.8% | 89.8% | 28.0% | 34.0% |
| Error precision | 78.2% | 97.1% | 85.9% | 95.4% | 43.4% | 78.6% |
| Error recall | 88.4% | 95.7% | 85.5% | 95.9% | 53.5% | 76.7% |

**Did v2 close the OOD gap?** Mostly. The fields that failed by position in round 3 (OOD, raw → full run raw, + rules):

| OOD field | Smoke | Smoke + rules | Full run | Full run + rules | Haiku + rules |
|---|---|---|---|---|---|
| sender_reference | 61% | 61% | 94% | 94% | 100% |
| value_date | 64% | 78% | 79% | 82% | 99% |
| ordering_customer.account | 59% | 59% | 86% | 86% | 97% |
| 52A ordering_institution | 39% | 39% | 62% | **70%** | 96% |
| 57A account_with_institution | 46% | 53% | 74% | **78%** | 95% |
| charges | 62% | 72% | 99% | 99% | 97% |

By OOD format (share of the 15 fields right, pooled over the examples of that format; recomputed from
`per_example`, so the smoke figures differ slightly from section 1's per-field average):

| OOD style | n | Smoke | Smoke + rules | Full run | Full run + rules | Haiku + rules |
|---|---|---|---|---|---|---|
| `kv_mixed` | 26 | 76.4% | 79.2% | 89.0% | 89.7% | 98.7% |
| `narrative` | 27 | 71.4% | 73.1% | 87.4% | 88.6% | 97.0% |
| `pipe_table` | 24 | 71.7% | 74.4% | 88.3% | 89.7% | 99.2% |
| `swift_shuffled` | 23 | 43.8% | 48.1% | **91.9%** | **93.6%** | 98.6% |

**Reading it honestly:**
- `swift_shuffled` going from about 44% to 92% is the clearest sign that the model now maps fields by tag, not by
  position. v2 never contains shuffled tags (the held-out test checks this), so this is generalisation.
- In distribution the 0.5B model now matches Haiku (99.4% vs 99.1% with rules). On OOD it is still 8 points behind,
  and only 34% of OOD messages are fully right (Haiku 80%). The two bank fields remain the weak spot: 52A at 70%.
- The rules add little to field accuracy now (+0.1 to +1.2 points) but a lot to error precision (43% → 79% on OOD),
  because they remove contradictory flags.
- Everything is still synthetic. n=100 for OOD means each style has 23–27 examples, so differences of a few points
  between styles are noise.

**The served CPU build (MEASURED 2026-11-25, build VPS).** The same adapter merged (`train/merge.py`), converted
with llama.cpp and quantised to **q4_K_M (398 MB)**, loaded as `mt103-extractor` in the project's Ollama container
(`cpus: 1.0`), same short prompt. Run names `cpu-ft-qwen2.5-0.5b-lora-v2` and `ood-cpu-ft-qwen2.5-0.5b-lora-v2`, so the
smoke model's rows are kept as the "before". Command: `python evals/run_eval.py --backend ollama --model mt103-extractor
--prompt ft --data <split> --name [ood-]cpu-ft-qwen2.5-0.5b-lora-v2` (rules: `make eval-pp`).

| Served CPU build (q4_K_M) | Test split (n=300) | + rules | OOD (n=100) | + rules |
|---|---|---|---|---|
| Field acc (macro) | 98.9% | **99.1%** | 87.0% | **89.0%** |
| fp16 adapter on GPU, for comparison | 99.3% | 99.4% | 89.1% | 90.3% |
| Valid JSON / schema-valid | 99.7% / 97.3% | 99.7% / 99.7% | 100% / 84.0% | 100% / 99.0% |
| All 15 fields right | 87.3% | 90.7% | 22.0% | 33.0% |
| Error precision / recall | 77.3% / 84.1% | 92.9% / 94.2% | 37.5% / 48.8% | 78.6% / 76.7% |
| OOD 52A / 57A | | | 58% / 75% | 68% / 81% |
| Latency p50 / tokens per s (1 CPU) | 8.4 s / 24.9 | | 8.4 s / 25.8 | |

Quantisation costs about 0.3 points in distribution and 1–2 points on OOD; with the rules the served model stays within
1.3 points of the fp16 adapter everywhere. Speed is about the same as the smoke model on the same CPU (8.4 s vs 9.3 s p50;
same architecture and quantisation, and the VPS is shared, so the difference is not meaningful).

## Tests (round 3)
`make test` gives **86 passed** (2026-11-24): the 50 earlier tests, plus post-processing (gold pass-through on every
split and hand-built cases for each rule), the API's rule layer, and the v2 held-out checks.
