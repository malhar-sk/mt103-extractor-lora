"""v2 training data: valid, reproducible, more diverse, and STRICTLY held out from the OOD test set.

Each test below is one part of the held-out guarantee documented in data/generate_v2.py and EVALS.md.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

from data.generate import BAD_BICS, BAD_CCY, BAD_CHARGES
from data.generate_ood import OOD_BAD_BICS, OOD_BAD_CCY, OOD_BAD_CHARGES, OOD_BAD_DATES, REMARKS, generate_ood
from data.generate_v2 import BAD_BICS_V2, BAD_CCY_V2, BAD_CHARGES_V2, generate_v2
from schema import Extraction

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
MON = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
MONTH_LONG = r"(January|February|March|April|May|June|July|August|September|October|November|December)"

# Label names used by the OOD renderers (copied from data/generate_ood.py OodGen._items / o_pipe_table).
OOD_LABELS = ["Sender ref", "Our ref#", "Txn reference", "Amount", "Sum", "Transfer amount", "Settle on", "Value",
              "Execution date", "Remitter", "Payer", "Debtor", "Payer bank BIC", "Sending bank", "Debtor agent",
              "Payee", "Creditor", "Recipient", "Payee bank BIC", "Receiving bank", "Creditor agent", "Narrative",
              "Payment details", "Memo", "Fees", "Charge bearer", "Costs", "Op", "Bank op code", "Instruction code",
              "Field"]
# Distinctive sentences of the OOD narrative renderer and remarks.
OOD_PHRASES = ["We would like to transfer", "make sure the funds arrive", "as our reference", "Debit us through",
               "SWIFT code is", "Use op code", " writing."] + [t.split("{")[0].strip() for t in REMARKS]
# Date formats only the OOD set uses (data/generate_ood.py ood_fmt_date).
OOD_DATE_RES = [
    re.compile(r"\b\d{1,2}(st|nd|rd|th) " + MONTH_LONG + r" \d{4}"),                 # 5th March 2026
    re.compile(r"(?<![\d/])\d{4}/\d{2}/\d{2}(?![\d/])"),                              # 2026/03/05
    re.compile(r"(?<![\w])20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?![\w])"),       # 20260305
    re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{1,2} " + MON + r" \d{2}\b"),       # Thu 5 Mar 26
    re.compile(r"(?<![\d.])(\d\.\d{1,2}|\d{1,2}\.\d)\.\d{4}(?!\d)"),                  # 5.3.2026
    re.compile(r"\b\d{2}-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-\d{4}\b"),  # 05-MAR-2026
    re.compile(r"\b" + MON + r"\. \d{1,2} \d{4}\b"),                                   # Mar. 5 2026
]
SWIFT_ORDER = ["20", "13C", "23B", "32A", "33B", "50K", "52A", "53A", "56A", "57A", "59", "70", "71A", "71F", "72"]


def _load(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


@pytest.fixture(scope="module")
def v2():
    if not (V2 / "train.jsonl").exists():
        pytest.skip("run `make data-v2` first")
    return {s: _load(V2 / f"{s}.jsonl") for s in ("train", "val", "test")}


@pytest.fixture(scope="module")
def v2_all(v2):
    return v2["train"] + v2["val"] + v2["test"]


@pytest.fixture(scope="module")
def v1_train_text():
    return "\n".join(r["text"] for r in _load(ROOT / "data" / "train.jsonl"))


def test_v2_valid_synthetic_and_reproducible(v2, v2_all):
    assert (len(v2["train"]), len(v2["val"]), len(v2["test"])) == (4800, 600, 600)
    for r in v2_all:
        assert r["synthetic"] is True
        Extraction.model_validate(r["target"])
    first = generate_v2(50, 2026)
    by_id = {r["id"]: r["text"] for r in v2_all}
    assert all(by_id[r["id"]] == r["text"] for r in first)


def test_v2_splits_disjoint_and_no_overlap_with_v1_test_or_ood(v2, v2_all):
    h = lambda rows: {hashlib.sha1(r["text"].encode()).hexdigest() for r in rows}  # noqa: E731
    assert not h(v2["train"]) & h(v2["val"]) and not h(v2["train"]) & h(v2["test"])
    assert not h(v2_all) & h(_load(ROOT / "data" / "ood_test.jsonl"))
    assert not h(v2_all) & h(_load(ROOT / "data" / "test.jsonl"))


def test_ood_file_unchanged():
    """generate_ood.py is untouched: the OOD set is the exact file scored in eval round 2."""
    ood = _load(ROOT / "data" / "ood_test.jsonl")
    assert [r["text"] for r in generate_ood(100, 20261123)] == [r["text"] for r in ood]
    assert hashlib.sha256((ROOT / "data" / "ood_test.jsonl").read_bytes()).hexdigest() == \
        "5d1e2a2a6d9ef6a23be2e19ac72cde03c54d35f5a57e1a5d86a3d9cb53b80ae7"


def test_no_ood_impossible_values(v2_all):
    for bad in (set(BAD_CCY_V2) & set(OOD_BAD_CCY), set(BAD_BICS_V2) & set(OOD_BAD_BICS),
                set(BAD_CHARGES_V2) & set(OOD_BAD_CHARGES)):
        assert not bad
    toks = OOD_BAD_CCY + OOD_BAD_BICS + OOD_BAD_CHARGES + OOD_BAD_DATES
    rx = re.compile("|".join(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])" for t in toks))
    assert [r["id"] for r in v2_all if rx.search(r["text"])] == []
    # v2 extends v1's lists, it does not drop them
    assert set(BAD_CCY) <= set(BAD_CCY_V2) and set(BAD_BICS) <= set(BAD_BICS_V2) and set(BAD_CHARGES) <= set(BAD_CHARGES_V2)


def test_no_ood_layouts(v2_all):
    pipe = re.compile(r"^\s*\|.*\|\s*$", re.M)
    kv = re.compile(r"[A-Za-z]=[^=\s]")
    lower_tag = re.compile(r"^:\d{2}[a-z]:", re.M)
    for r in v2_all:
        assert not pipe.search(r["text"]), r["id"]
        assert not kv.search(r["text"]), r["id"]
        assert not lower_tag.search(r["text"]), r["id"]


def test_swift_tags_always_in_standard_order(v2_all):
    tag = re.compile(r"^:(\d{2}[A-Z]?):", re.M)
    for r in v2_all:
        tags = tag.findall(r["text"])
        pos = [SWIFT_ORDER.index(t) for t in tags if t in SWIFT_ORDER]
        assert pos == sorted(pos), (r["id"], tags)


def test_no_ood_date_formats(v2_all):
    for r in v2_all:
        for rx in OOD_DATE_RES:
            assert not rx.search(r["text"]), (r["id"], rx.pattern, rx.search(r["text"]).group(0))


def test_no_ood_phrases(v2_all):
    for r in v2_all:
        low = r["text"].casefold()
        for p in OOD_PHRASES:
            assert p.casefold() not in low, (r["id"], p)


def test_no_ood_only_labels(v2_all, v1_train_text):
    """A label the OOD set uses and v1 training never used must not appear as a label in v2."""
    v1 = v1_train_text.casefold()
    ood_only = [lab for lab in OOD_LABELS if not re.search(re.escape(lab.casefold()) + r"\s*[:=|\t-]", v1)]
    assert ood_only, "expected some OOD-only labels"
    rx = re.compile(r"(?im)(^|[\s,;\"(\[•*-])(" + "|".join(re.escape(x) for x in ood_only) + r")\s*[:=|\t\"]")
    hits = [(r["id"], rx.search(r["text"]).group(2)) for r in v2_all if rx.search(r["text"])]
    assert hits == []


def test_v2_is_more_diverse_than_v1(v2):
    styles = {r["style"] for r in v2["train"]}
    assert len(styles) >= 10
    lone_52a = sum(1 for r in v2["train"] if r["target"]["fields"]["ordering_institution"]
                   and not r["target"]["fields"]["account_with_institution"])
    assert lone_52a > 300  # "only the sender's BIC" is common, so a lone BIC is not always 57A
