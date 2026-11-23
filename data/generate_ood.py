"""Out-of-distribution (OOD) SYNTHETIC test set: different seed, renderers and noise patterns.

Everything produced here is synthetic. The goal is to check whether the in-distribution test score
generalises or only reflects memorised surface format. Compared with data/generate.py, this set uses:

  * a different seed (20261123) and NEW renderers, none of which appear in training:
      - "swift_shuffled": MT103 tag blocks in a random order, tags often lower-case (:32a:, :50k:, :71a:)
      - "narrative": one free-text paragraph, with the facts in a random order
      - "pipe_table": a `| key | value |` table, rows in random order, unseen key names
      - "kv_mixed": a `key=value; ...` one-liner in random order, unseen key names
  * OCR-style character swaps (rn->m, l->1, O->0, e->c ...) applied ONLY to labels and filler words,
    never to the values being extracted, so the gold target stays unambiguous
  * extra free-text remarks, including distractor numbers, dates and amounts that must be ignored
  * date formats not used in training (5th March 2026, 2026/03/05, 20260305, Thu 5 Mar 26, 5.3.2026,
    05-MAR-2026), plus distractor dates in other formats
  * impossible values from NEW lists (currencies, BICs, dates, charge words that training never saw)

Gold targets follow the same schema and rules as the main set and are validated with schema.Extraction.

Usage: python data/generate_ood.py --n 100 --seed 20261123 --out data/ood_test.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate import BANKS, MONTHS, MONTHS_LONG, Gen  # noqa: E402
from schema import BIC_RE, ISO4217, Extraction  # noqa: E402

# Impossible values the training generator never produced (checked in tests/test_ood.py).
OOD_BAD_CCY = ["USDD", "EURO", "YEN", "RUP", "XXY", "GPB", "DLR", "INRS"]
OOD_BAD_BICS = ["ICIC0INBB", "CITI-US-33", "DEUT DE FF", "SCBL", "BOFAUS3NXXXX", "HDFCINB", "UBSW1234ZH"]
OOD_BAD_DATES = ["2026-02-30", "31st April 2026", "2026/06/31", "20260231", "0/05/2026", "31-SEP-2026",
                 "30th February 2027", "2026.11.31"]
OOD_BAD_CHARGES = ["SPLIT", "SHR", "BOTH", "OUR/BEN", "FREE"]
OOD_KINDS = ["ccy", "date", "bic", "amt", "charges", "missing"]

# OCR confusions, applied to labels/filler only.
OCR = [("rn", "m"), ("m", "rn"), ("l", "1"), ("O", "0"), ("e", "c"), ("i", "l"), ("S", "5"), ("cl", "d")]

REMARKS = [
    "Note: last month we sent {amt2} to them for a different invoice, ignore that one.",
    "Client called on {d2} to confirm, ticket closed after.",
    "Pls call me on +91 98{n7} if anything is unclear.",
    "Previous ref {ref2} was cancelled, do NOT reuse it.",
    "FYI compliance cleared this on {d2}.",
    "Remember the cut-off is 3pm IST; the {amt2} batch can wait.",
    "Attaching the invoice scan (page 2 of 3).",
    "This replaces the instruction from {d2}.",
]


def ood_fmt_date(r, d: dt.date) -> str:
    sfx = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return r.choice([
        f"{d.day}{sfx} {MONTHS_LONG[d.month - 1]} {d.year}",
        d.strftime("%Y/%m/%d"),
        d.strftime("%Y%m%d"),
        f"{d.strftime('%a')} {d.day} {MONTHS[d.month - 1]} {d.strftime('%y')}",
        f"{d.day}.{d.month}.{d.year}",
        f"{d.day:02d}-{MONTHS[d.month - 1].upper()}-{d.year}",
        f"{MONTHS[d.month - 1]}. {d.day} {d.year}",
    ])


class OodGen(Gen):
    """Reuses the primitive value generators from Gen; replaces errors, dates and renderers."""

    def ocr(self, s: str, p: float = 0.35) -> str:
        if self.r.random() >= p:
            return s
        cands = [(a, b) for a, b in OCR if a in s]
        if not cands:
            return s
        a, b = self.r.choice(cands)
        i = s.find(a)
        return s[:i] + b + s[i + len(a):]

    def lab(self, s: str) -> str:
        """A label: OCR noise + random case."""
        s = self.ocr(s)
        k = self.r.random()
        return s.lower() if k < 0.25 else s.upper() if k < 0.4 else s

    def remark(self) -> str:
        r = self.r
        d2 = self.date()
        return r.choice(REMARKS).format(
            amt2=f"{r.choice(['USD', 'EUR', 'INR'])} {r.randint(100, 99999):,}.00",
            d2=r.choice([d2.strftime("%d/%m/%Y"), d2.isoformat(), f"{d2.day} {MONTHS[d2.month - 1]}"]),
            n7=f"{r.randint(0, 99999999):08d}", ref2=self.ref())

    def example(self, idx: int) -> dict:
        r = self.r
        ccy = r.choice(["USD", "EUR", "GBP", "INR", "SGD", "JPY", "CHF", "AED", "HKD"])
        amt = self.amount(ccy)
        d = self.date()
        f = {
            "sender_reference": self.ref(),
            "bank_operation_code": "CRED" if r.random() < 0.85 else r.choice(["SPAY", "SPRI", "SSTD", "CRTS"]),
            "value_date": d.isoformat(), "currency": ccy, "amount": amt,
            "ordering_customer": self.party(), "ordering_institution": r.choice(BANKS),
            "account_with_institution": r.choice(BANKS), "beneficiary": self.party(),
            "remittance_info": self.remit(), "charges": r.choice(["OUR", "SHA", "BEN"]),
        }
        surf = {"date": ood_fmt_date(r, d), "ccy": self.fmt_ccy(ccy), "amt": self.fmt_amount(amt, ccy),
                "charges": r.choice({"OUR": ["OUR", "our", "all charges ours"],
                                     "SHA": ["SHA", "sha", "shared"],
                                     "BEN": ["BEN", "ben", "beneficiary pays"]}[f["charges"]])}
        for k, p in (("bank_operation_code", 0.45), ("ordering_institution", 0.3),
                     ("account_with_institution", 0.3), ("remittance_info", 0.2)):
            if r.random() < p:
                f[k] = None
        errors: list[str] = []
        # Higher error rate than training (35% vs 22%) so the per-code analysis has enough support.
        if r.random() < 0.35:
            kind = OOD_KINDS[idx % len(OOD_KINDS)]  # round-robin: balanced error codes
            if kind == "ccy":
                bad = r.choice(OOD_BAD_CCY)
                surf["ccy"] = bad; f["currency"] = None
                errors.append(f"INVALID_CURRENCY: currency: {bad} is not an ISO-4217 code")
            elif kind == "date":
                surf["date"] = r.choice(OOD_BAD_DATES); f["value_date"] = None
                errors.append(f"INVALID_DATE: value_date: {surf['date']} is not a real calendar date")
            elif kind == "bic":
                which = r.choice(["ordering_institution", "account_with_institution"])
                bad = r.choice(OOD_BAD_BICS); f[which] = None
                surf["badbic"] = (which, bad)
                errors.append(f"INVALID_BIC: {which}: {bad} is not a valid BIC")
            elif kind == "amt":
                bad = r.choice(["(" + self.fmt_amount(amt, ccy) + ")", "0.0", "- " + f"{amt:.2f}", "nil"])
                if bad == "nil":
                    bad = "0"
                surf["amt"] = bad; f["amount"] = None
                errors.append(f"NON_POSITIVE_AMOUNT: amount: {bad} is not a positive amount")
            elif kind == "charges":
                bad = r.choice(OOD_BAD_CHARGES); surf["charges"] = bad; f["charges"] = None
                errors.append(f"INVALID_CHARGES: charges: {bad} is not OUR/SHA/BEN")
            else:
                miss = r.choice(["charges", "beneficiary", "value_date"])
                f[miss] = None; surf["missing"] = miss
                errors.append(f"MISSING_REQUIRED: {miss}: not stated in the text")
        style = r.choice(["swift_shuffled", "narrative", "pipe_table", "kv_mixed"])
        text = getattr(self, f"o_{style}")(f, surf).strip()
        target = {"fields": f, "errors": errors}
        Extraction.model_validate(target)
        return {"id": f"ood-{idx:04d}", "synthetic": True, "ood_style": style, "text": text, "target": target}

    # ---------- renderers (all new) ----------
    def _items(self, f, s):
        """(label_variants, value) items common to the table-like renderers; None values skipped."""
        oc, bn = f["ordering_customer"], f["beneficiary"]
        it = [(["Sender ref", "Our ref#", "Txn reference"], f["sender_reference"]),
              (["Amount", "Sum", "Transfer amount"], self.money(s))]
        if s.get("missing") != "value_date":
            it.append((["Settle on", "Value", "Execution date"], s["date"]))
        it.append((["Remitter", "Payer", "Debtor"], self.party_s(oc, " / ")))
        if f["ordering_institution"] or s.get("badbic", ("",))[0] == "ordering_institution":
            it.append((["Payer bank BIC", "Sending bank", "Debtor agent"], self.bic_s(f, s, "ordering_institution")))
        if s.get("missing") != "beneficiary":
            it.append((["Payee", "Creditor", "Recipient"], self.party_s(bn, " / ")))
        if f["account_with_institution"] or s.get("badbic", ("",))[0] == "account_with_institution":
            it.append((["Payee bank BIC", "Receiving bank", "Creditor agent"], self.bic_s(f, s, "account_with_institution")))
        if f["remittance_info"]:
            it.append((["Narrative", "Payment details", "Memo"], f["remittance_info"]))
        if s.get("missing") != "charges":
            it.append((["Fees", "Charge bearer", "Costs"], s["charges"]))
        if f["bank_operation_code"]:
            it.append((["Op", "Bank op code", "Instruction code"], f["bank_operation_code"]))
        self.r.shuffle(it)
        return [(self.lab(self.r.choice(labels)), v) for labels, v in it]

    def o_pipe_table(self, f, s):
        rows = ["| " + self.lab("Field") + " | " + self.lab("Value") + " |", "|---|---|"]
        rows += [f"| {k} | {v} |" for k, v in self._items(f, s)]
        if self.r.random() < 0.7:
            rows.append("")
            rows.append(self.remark())
        return "\n".join(rows)

    def o_kv_mixed(self, f, s):
        parts = [f"{k}={v}" for k, v in self._items(f, s)]
        out = "; ".join(parts)
        if self.r.random() < 0.6:
            out = self.remark() + " " + out if self.r.random() < 0.5 else out + " -- " + self.remark()
        return out

    def o_narrative(self, f, s):
        r = self.r
        oc, bn = f["ordering_customer"], f["beneficiary"]
        sents = [f"{self.ocr('This is')} {self.party_s(oc, ', ')} {self.ocr('writing')}."]
        sents.append(f"We would like to transfer {self.money(s)}"
                     + (f" to {self.party_s(bn, ', ')}" if s.get("missing") != "beneficiary" else "") + ".")
        if s.get("missing") != "value_date":
            sents.append(f"{self.ocr('Please make sure the funds arrive on')} {s['date']}.")
        sents.append(f"Quote {f['sender_reference']} as {self.ocr('our reference')}.")
        if f["ordering_institution"] or s.get("badbic", ("",))[0] == "ordering_institution":
            sents.append(f"Debit us through {self.bic_s(f, s, 'ordering_institution')}.")
        if f["account_with_institution"] or s.get("badbic", ("",))[0] == "account_with_institution":
            sents.append(f"Their bank's SWIFT code is {self.bic_s(f, s, 'account_with_institution')}.")
        if f["remittance_info"]:
            sents.append(f"It is for \"{f['remittance_info']}\".")
        if s.get("missing") != "charges":
            sents.append(f"{self.ocr('Charges')}: {s['charges']}.")
        if f["bank_operation_code"]:
            sents.append(f"Use op code {f['bank_operation_code']}.")
        head, rest = sents[:1], sents[1:]
        r.shuffle(rest)
        rest.insert(r.randint(0, len(rest)), self.remark())
        if r.random() < 0.5:
            rest.insert(r.randint(0, len(rest)), self.remark())
        return " ".join(head + rest)

    def o_swift_shuffled(self, f, s):
        r = self.r
        oc, bn = f["ordering_customer"], f["beneficiary"]
        t = (lambda tag: tag.lower()) if r.random() < 0.6 else (lambda tag: tag)
        ccy = s["ccy"].upper() if s["ccy"].isalpha() else (f["currency"] or s["ccy"])
        date = s["date"]
        if f["value_date"] and r.random() < 0.5:
            date = dt.date.fromisoformat(f["value_date"]).strftime("%y%m%d")
        blocks = [f":20:{f['sender_reference']}"]
        if f["bank_operation_code"]:
            blocks.append(f":{t('23B')}:{f['bank_operation_code']}")
        amt = s["amt"]
        if s.get("missing") != "value_date":
            blocks.append(f":{t('32A')}:{date} {ccy} {amt}")
        else:
            blocks.append(f":{t('32A')}:{ccy} {amt}")
        blocks.append(f":{t('50K')}:" + "\n".join(x for x in [("/" + oc["account"]) if oc["account"] else None, oc["name"], oc["address"]] if x))
        if f["ordering_institution"] or s.get("badbic", ("",))[0] == "ordering_institution":
            blocks.append(f":{t('52A')}:{self.bic_s(f, s, 'ordering_institution')}")
        if f["account_with_institution"] or s.get("badbic", ("",))[0] == "account_with_institution":
            blocks.append(f":{t('57A')}:{self.bic_s(f, s, 'account_with_institution')}")
        if s.get("missing") != "beneficiary":
            blocks.append(":59:" + "\n".join(x for x in [("/" + bn["account"]) if bn["account"] else None, bn["name"], bn["address"]] if x))
        if f["remittance_info"]:
            blocks.append(f":70:{f['remittance_info']}")
        if s.get("missing") != "charges":
            blocks.append(f":{t('71A')}:{s['charges'].upper()}")
        r.shuffle(blocks)
        if r.random() < 0.6:
            blocks.insert(r.randint(0, len(blocks)), "/* " + self.remark() + " */")
        return "\n".join(blocks)


def generate_ood(n: int, seed: int, exclude_texts: set[str] = frozenset()) -> list[dict]:
    g = OodGen(seed)
    rows, seen = [], set(exclude_texts)
    while len(rows) < n:
        ex = g.example(len(rows))
        h = hashlib.sha1(ex["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append(ex)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--out", default=str(Path(__file__).parent / "ood_test.jsonl"))
    a = ap.parse_args()
    rows = generate_ood(a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    styles = {}
    for row in rows:
        styles[row["ood_style"]] = styles.get(row["ood_style"], 0) + 1
    codes = {}
    for row in rows:
        for e in row["target"]["errors"]:
            codes[e.split(":")[0]] = codes.get(e.split(":")[0], 0) + 1
    manifest = {"synthetic": True, "note": "SYNTHETIC out-of-distribution test set (data/generate_ood.py); not real payments.",
                "seed": a.seed, "n": len(rows), "styles": styles, "error_codes": codes,
                "examples_with_errors": sum(1 for r in rows if r["target"]["errors"])}
    Path(a.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    assert all(c not in ISO4217 for c in OOD_BAD_CCY)
    assert all(not BIC_RE.match(b) for b in OOD_BAD_BICS)
    main()
