"""Seeded generator of SYNTHETIC "messy payment text -> MT103 JSON" examples.

Everything produced here is synthetic. Names, accounts, references and amounts are
random; BICs are reused from the swift-mt103-monitor simulator's bank list plus a few
other well-known public BICs. No real payment data is used.

Usage:
    python data/generate.py --n 3000 --seed 103 --out data/
Writes data/{train,val,test}.jsonl (80/10/10) and data/MANIFEST.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import ISO4217, Extraction  # noqa: E402

# Bank list from swift-mt103-monitor/swift/producer.go, extended.
BANKS_SRC = ["DEUTDEDB", "CHASUS33", "BARCGB22", "BNPAFRPP", "HSBCHKHH"]
BANKS_EXTRA = ["SBININBB", "HDFCINBB", "ICICINBBXXX", "UTIBINBB", "CITIUS33", "DEUTDEFF",
               "BOFAUS3N", "SCBLSGSG", "MHCBJPJT", "UBSWCHZH80A", "INGBNL2A", "COBADEFFXXX"]
BANKS = BANKS_SRC + BANKS_EXTRA
BAD_BICS = ["HDFC123", "CHASUS", "BARC-GB22", "1234INBB", "DEUTDEDBX", "SBIN IN BB", "HSBCHK"]

# currency -> (weight, typical amount range)
CCY = {"USD": (40, (10, 10000)), "EUR": (14, (50, 250000)), "GBP": (8, (50, 150000)),
       "INR": (20, (1000, 9000000)), "SGD": (4, (100, 200000)), "JPY": (4, (10000, 9000000)),
       "CHF": (3, (100, 100000)), "AED": (4, (500, 400000)), "HKD": (3, (500, 500000))}
CCY_SYMBOL = {"USD": ["$", "US$"], "EUR": ["€"], "GBP": ["£"], "INR": ["Rs.", "₹", "INR", "Rs"],
              "JPY": ["¥"]}
BAD_CCY = ["USX", "EUD", "RSS", "GBR", "IND", "DOL", "UDS", "ERU"]

FIRST = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Meera", "John", "Sarah", "Lukas", "Chloe",
         "Kenji", "Fatima", "Omar", "Elena", "Arjun", "Divya", "Marco", "Aisha", "Wei", "Nikhil"]
LAST = ["Sharma", "Iyer", "Patel", "Reddy", "Kapoor", "Smith", "Müller", "Dubois", "Tanaka",
        "Khan", "Rossi", "Nair", "Chen", "Gupta", "Fernandes", "Menon", "Brown", "Schmidt"]
CO_A = ["Arcadia", "Bluepeak", "Coromandel", "Deccan", "Everest", "Falcon", "Ganges", "Helix",
        "Indus", "Juniper", "Kestrel", "Lotus", "Meridian", "Nilgiri", "Orion", "Saffron",
        "Tata-Vale", "Zenith", "Northwind", "Harbor"]
CO_B = ["Textiles", "Logistics", "Pharma", "Steel", "Software", "Exports", "Agro", "Capital",
        "Trading", "Components", "Foods", "Shipping", "Analytics", "Motors"]
CO_C = ["Pvt Ltd", "Ltd", "GmbH", "LLC", "Inc", "SA", "Pte Ltd", "LLP", "BV", "PLC"]
CITY = ["Mumbai 400001 IN", "Bengaluru 560001 IN", "Pune 411001 IN", "Chennai 600002 IN",
        "Frankfurt 60311 DE", "London EC2V 7HN GB", "New York NY 10005 US", "Paris 75009 FR",
        "Singapore 048583 SG", "Hong Kong HK", "Dubai AE", "Zurich 8001 CH", "Tokyo 100-0005 JP"]
STREET = ["12 MG Road", "4th Floor, Nariman Point", "221 Park Street", "Plot 7, MIDC",
          "Taunusanlage 12", "1 Churchill Place", "270 Park Avenue", "16 Bd des Italiens",
          "8 Marina Blvd", "Unit 5B, Hi-Tech Park"]
REMIT = ["INV {n} consulting fees", "Invoice {n} / spare parts", "PO {n} advance payment",
         "Salary {mon} 2026", "Rent {mon} 2026 office", "Refund order {n}",
         "Freight charges BL {n}", "Licence fee Q{q} 2026", "Settlement of INV {n} and INV {n2}",
         "Tuition fees semester {q}", "Dividend payout {mon}", "Software subscription {n}"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
CHARGE_WORDS = {"OUR": ["OUR", "our", "ours", "OUR (we pay all charges)", "all charges on us"],
                "SHA": ["SHA", "sha", "shared", "SHA - shared", "split charges (SHA)"],
                "BEN": ["BEN", "ben", "beneficiary pays", "BEN (charges to beneficiary)"]}
BAD_CHARGES = ["FULL", "NONE", "OURS+BEN", "XYZ", "HALF"]
TYPOS = {"beneficiary": ["benificiary", "beneficary", "bene", "benef."], "amount": ["amt", "amout", "amnt"],
         "please": ["pls", "plz", "please"], "account": ["acct", "a/c", "acc"],
         "reference": ["ref", "refrence", "ref no"], "payment": ["pymt", "paymnt", "payment"],
         "charges": ["charges", "chrgs", "charge"], "value date": ["value date", "val dt", "VD", "value dt"]}


def indian_format(x: float, decimals: bool = True) -> str:
    s = f"{x:.2f}"
    whole, frac = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    return f"{whole}.{frac}" if decimals or frac != "00" else whole


class Gen:
    def __init__(self, seed: int):
        self.r = random.Random(seed)

    # ---------- primitives ----------
    def w(self, key: str) -> str:
        return self.r.choice(TYPOS[key]) if self.r.random() < 0.5 else key

    def company(self) -> str:
        r = self.r
        return f"{r.choice(CO_A)} {r.choice(CO_B)} {r.choice(CO_C)}"

    def person(self) -> str:
        return f"{self.r.choice(FIRST)} {self.r.choice(LAST)}"

    def account(self) -> str:
        r = self.r
        k = r.random()
        if k < 0.4:
            return "".join(r.choice("0123456789") for _ in range(r.choice([11, 12, 14, 15])))
        if k < 0.75:
            cc = r.choice(["DE", "GB", "FR", "NL", "CH"])
            return cc + "".join(r.choice("0123456789") for _ in range(r.choice([18, 20])))
        return "".join(r.choice("0123456789") for _ in range(9)) + "-" + str(r.randint(10, 99))

    def party(self, need_account: bool = True) -> dict:
        r = self.r
        name = self.company() if r.random() < 0.7 else self.person()
        acct = self.account() if (need_account or r.random() < 0.7) else None
        addr = f"{r.choice(STREET)}, {r.choice(CITY)}" if r.random() < 0.55 else None
        return {"account": acct, "name": name, "address": addr}

    def ref(self) -> str:
        r = self.r
        k = r.random()
        if k < 0.4:
            return f"TXN{r.randint(0, 999999):06d}"  # same pattern as swift-mt103-monitor
        if k < 0.7:
            return f"FT26{r.randint(10000000, 99999999)}"[:16]
        if k < 0.9:
            return f"PAY-{r.randint(2025, 2026)}-{r.randint(1, 99999):05d}"
        return f"OPS/{r.choice(MONTHS).upper()}/{r.randint(1, 999)}"

    def amount(self, ccy: str) -> float:
        lo, hi = CCY[ccy][1]
        x = round(self.r.uniform(lo, hi), 2)
        if ccy == "JPY":
            x = float(int(x))
        if self.r.random() < 0.25:
            x = float(round(x, -2)) or 100.0
        return x

    def fmt_amount(self, x: float, ccy: str) -> str:
        r = self.r
        k = r.random()
        if ccy == "INR" and k < 0.55:
            if x >= 100000 and r.random() < 0.2 and round(x / 100000, 2) * 100000 == x:
                return f"{x / 100000:g} lakh"
            return indian_format(x, decimals=r.random() < 0.7)
        if ccy == "EUR" and k < 0.25:
            s = f"{x:,.2f}"
            return s.replace(",", "_").replace(".", ",").replace("_", ".")
        if k < 0.6:
            return f"{x:,.2f}"
        if k < 0.85:
            return f"{x:.2f}" if r.random() < 0.6 else f"{x:g}"
        return f"{x:,.2f}".replace(",", " ")

    def fmt_ccy(self, ccy: str) -> str:
        r = self.r
        if ccy in CCY_SYMBOL and r.random() < 0.35:
            return r.choice(CCY_SYMBOL[ccy])
        return ccy if r.random() < 0.8 else ccy.lower()

    def date(self) -> dt.date:
        return dt.date(2026, 1, 1) + dt.timedelta(days=self.r.randint(0, 420))

    def fmt_date(self, d: dt.date) -> str:
        r = self.r
        k = r.randint(0, 7)
        return [d.isoformat(), d.strftime("%d/%m/%Y"), d.strftime("%d-%m-%Y"), d.strftime("%d.%m.%Y"),
                f"{d.day} {MONTHS[d.month - 1]} {d.year}", f"{MONTHS_LONG[d.month - 1]} {d.day}, {d.year}",
                f"{d.day:02d}-{MONTHS[d.month - 1]}-{d.strftime('%y')}", d.strftime("%y%m%d")][k]

    def bad_date(self) -> str:
        r = self.r
        y = r.choice([2026, 2027])
        return r.choice([f"31/04/{y}", f"30/02/{y}", f"{y}-13-05", f"32.01.{y}", f"31 Jun {y}",
                         f"29/02/{y}" if y == 2026 else f"31/09/{y}", f"{y}-00-12", f"31-11-{y}"])

    def remit(self) -> str:
        r = self.r
        return r.choice(REMIT).format(n=f"{r.randint(1, 9999):04d}", n2=f"{r.randint(1, 9999):04d}",
                                      mon=r.choice(MONTHS), q=r.randint(1, 4))

    # ---------- one example ----------
    def example(self, idx: int) -> dict:
        r = self.r
        ccy = r.choices(list(CCY), weights=[v[0] for v in CCY.values()])[0]
        amt = self.amount(ccy)
        d = self.date()
        f = {
            "sender_reference": self.ref(),
            "bank_operation_code": "CRED" if r.random() < 0.9 else r.choice(["SPAY", "SPRI", "SSTD", "CRTS"]),
            "value_date": d.isoformat(), "currency": ccy, "amount": amt,
            "ordering_customer": self.party(), "ordering_institution": r.choice(BANKS),
            "account_with_institution": r.choice(BANKS), "beneficiary": self.party(),
            "remittance_info": self.remit(), "charges": r.choice(["OUR", "SHA", "SHA", "BEN"]),
        }
        # surface strings (what appears in the text)
        surf = {"date": self.fmt_date(d), "ccy": self.fmt_ccy(ccy), "amt": self.fmt_amount(amt, ccy),
                "charges": r.choice(CHARGE_WORDS[f["charges"]])}
        # optional fields absent from text -> null in target, no error
        for k, p in (("bank_operation_code", 0.45), ("ordering_institution", 0.3),
                     ("account_with_institution", 0.3), ("remittance_info", 0.2)):
            if r.random() < p:
                f[k] = None
        errors: list[str] = []
        # impossible values (~22% of examples have at least one)
        if r.random() < 0.22:
            kind = r.choice(["ccy", "date", "bic", "amt", "charges", "missing"])
            if kind == "ccy":
                bad = r.choice(BAD_CCY); assert bad not in ISO4217
                surf["ccy"] = bad; f["currency"] = None
                errors.append(f"INVALID_CURRENCY: currency: {bad} is not an ISO-4217 code")
            elif kind == "date":
                surf["date"] = self.bad_date(); f["value_date"] = None
                errors.append(f"INVALID_DATE: value_date: {surf['date']} is not a real calendar date")
            elif kind == "bic":
                which = r.choice(["ordering_institution", "account_with_institution"])
                bad = r.choice(BAD_BICS); f[which] = None
                surf["badbic"] = (which, bad)
                errors.append(f"INVALID_BIC: {which}: {bad} is not a valid BIC")
            elif kind == "amt":
                bad = r.choice(["0.00", "-" + self.fmt_amount(amt, ccy), "0", "-" + f"{amt:.2f}"])
                surf["amt"] = bad; f["amount"] = None
                errors.append(f"NON_POSITIVE_AMOUNT: amount: {bad} is not a positive amount")
            elif kind == "charges":
                bad = r.choice(BAD_CHARGES); surf["charges"] = bad; f["charges"] = None
                errors.append(f"INVALID_CHARGES: charges: {bad} is not OUR/SHA/BEN")
            else:
                miss = r.choice(["charges", "beneficiary", "value_date"])
                f[miss] = None; surf["missing"] = miss
                errors.append(f"MISSING_REQUIRED: {miss}: not stated in the text")
        text = self.render(f, surf)
        target = {"fields": f, "errors": errors}
        Extraction.model_validate(target)  # every target must be schema-valid
        return {"id": f"syn-{idx:05d}", "synthetic": True, "text": text, "target": target}

    # ---------- renderers ----------
    def bic_s(self, f, surf, key):
        if "badbic" in surf and surf["badbic"][0] == key:
            return surf["badbic"][1]
        return f[key]

    def party_s(self, p, sep=", "):
        bits = []
        if p.get("account"):
            bits.append(p["account"] if self.r.random() < 0.5 else f"{self.w('account')} {p['account']}")
        bits.append(p["name"])
        if p.get("address"):
            bits.append(p["address"])
        return sep.join(bits)

    def render(self, f, surf) -> str:
        style = self.r.choices(["email", "chat", "swift", "ticket", "forward"], weights=[30, 25, 20, 15, 10])[0]
        return getattr(self, f"r_{style}")(f, surf).strip()

    def money(self, surf):
        c, a = surf["ccy"], surf["amt"]
        if c in ("$", "US$", "€", "£", "₹", "¥", "Rs.", "Rs"):
            return f"{c}{a}" if self.r.random() < 0.6 else f"{c} {a}"
        return f"{c} {a}" if self.r.random() < 0.7 else f"{a} {c}"

    def r_email(self, f, s):
        r, w = self.r, self.w
        oc, bn = f["ordering_customer"], f["beneficiary"]
        lines = [r.choice(["Hi team,", "Dear Ops,", "Hello,", "Good morning,", "Hi"]), ""]
        lines.append(f"{w('please').capitalize()} process the below {w('payment')} today.")
        lines.append(f"Our {w('reference')}: {f['sender_reference']}")
        lines.append(f"{w('amount').capitalize()}: {self.money(s)}")
        if s.get("missing") != "value_date":
            lines.append(f"{w('value date').capitalize()}: {s['date']}")
        lines.append(f"Debit {w('account')} / remitter: {self.party_s(oc)}")
        if f["ordering_institution"] or "badbic" in s and s["badbic"][0] == "ordering_institution":
            lines.append(f"Our bank: {self.bic_s(f, s, 'ordering_institution')}")
        if s.get("missing") != "beneficiary":
            lines.append(f"{w('beneficiary').capitalize()}: {self.party_s(bn)}")
        if f["account_with_institution"] or "badbic" in s and s["badbic"][0] == "account_with_institution":
            lines.append(f"{w('beneficiary').capitalize()} bank SWIFT: {self.bic_s(f, s, 'account_with_institution')}")
        if f["remittance_info"]:
            lines.append(f"Purpose: {f['remittance_info']}")
        if s.get("missing") != "charges":
            lines.append(f"{w('charges').capitalize()}: {s['charges']}")
        if f["bank_operation_code"]:
            lines.append(f"Op code {f['bank_operation_code']}")
        body = lines[:2] + r.sample(lines[2:], len(lines) - 2) if r.random() < 0.3 else lines
        body += ["", r.choice(["Thanks,", "Regards,", "Thx", "Best,"]), self.person()]
        return "\n".join(body)

    def r_chat(self, f, s):
        r, w = self.r, self.w
        oc, bn = f["ordering_customer"], f["beneficiary"]
        parts = [r.choice(["hey", "urgent:", "pls rush -", "@ops", "quick one"]),
                 f"send {self.money(s)} to {self.party_s(bn, ' ') if s.get('missing') != 'beneficiary' else 'the usual guys'}"]
        if f["account_with_institution"] or "badbic" in s and s["badbic"][0] == "account_with_institution":
            parts.append(f"at {self.bic_s(f, s, 'account_with_institution')}")
        parts.append(f"from {self.party_s(oc, ' ')}")
        if f["ordering_institution"] or "badbic" in s and s["badbic"][0] == "ordering_institution":
            parts.append(f"via {self.bic_s(f, s, 'ordering_institution')}")
        if s.get("missing") != "value_date":
            parts.append(f"{w('value date')} {s['date']}")
        parts.append(f"{w('reference')} {f['sender_reference']}")
        if f["remittance_info"]:
            parts.append(f"for {f['remittance_info']}")
        if s.get("missing") != "charges":
            parts.append(f"{w('charges')} {s['charges']}")
        if f["bank_operation_code"]:
            parts.append(f"({f['bank_operation_code']})")
        txt = " ".join(parts)
        return txt.lower() if r.random() < 0.3 else txt

    def r_swift(self, f, s):
        r = self.r
        oc, bn = f["ordering_customer"], f["beneficiary"]
        ccy = s["ccy"] if len(s["ccy"]) == 3 and s["ccy"].isalpha() else f["currency"] or s["ccy"]
        ccy = ccy.upper()
        amt = s["amt"]
        if not amt.startswith("-") and "lakh" not in amt:
            amt = f"{f['amount']:.2f}".replace(".", ",") if f["amount"] else amt
        date = s["date"]
        if f["value_date"] and r.random() < 0.7:
            date = dt.date.fromisoformat(f["value_date"]).strftime("%y%m%d")
        out = []
        if r.random() < 0.4 and f["ordering_institution"] and f["account_with_institution"]:
            # header only when both BICs are real, so it never leaks a BIC the target says is absent
            out.append(f"{{1:F01{f['ordering_institution'][:8]}AXXX0000000000}}{{2:I103{f['account_with_institution'][:8]}XN}}{{4:")
        out.append(f":20:{f['sender_reference']}")
        if f["bank_operation_code"]:
            out.append(f":23B:{f['bank_operation_code']}")
        if s.get("missing") != "value_date":
            out.append(f":32A:{date}{ccy}{amt}")
        else:
            out.append(f":32A:{ccy}{amt}  (date tbc)")
        out.append(":50K:" + "\n".join(x for x in [("/" + oc["account"]) if oc["account"] else None, oc["name"], oc["address"]] if x))
        if f["ordering_institution"] or "badbic" in s and s["badbic"][0] == "ordering_institution":
            out.append(f":52A:{self.bic_s(f, s, 'ordering_institution')}")
        if f["account_with_institution"] or "badbic" in s and s["badbic"][0] == "account_with_institution":
            out.append(f":57A:{self.bic_s(f, s, 'account_with_institution')}")
        if s.get("missing") != "beneficiary":
            out.append(":59:" + "\n".join(x for x in [("/" + bn["account"]) if bn["account"] else None, bn["name"], bn["address"]] if x))
        if f["remittance_info"]:
            out.append(f":70:{f['remittance_info']}")
        if s.get("missing") != "charges":
            ch = s["charges"] if s["charges"] in BAD_CHARGES else f["charges"]
            out.append(f":71A:{ch}")
        if r.random() < 0.3:  # partial paste: chopped header / trailing junk
            out.insert(0, r.choice(["pasted from SWIFT Alliance:", "msg below, pls check", "---"]))
        out.append("-}" if out[0].startswith("{1:") else r.choice(["", "-}", "...(rest cut off)"]))
        return "\n".join(out)

    def r_ticket(self, f, s):
        r = self.r
        oc, bn = f["ordering_customer"], f["beneficiary"]
        rows = [("Ticket", f"OPS-{r.randint(1000, 9999)}"), ("Ref", f["sender_reference"]),
                ("Ccy/Amt", f"{s['ccy']} | {s['amt']}")]
        if s.get("missing") != "value_date":
            rows.append(("VD", s["date"]))
        rows.append(("Ordering cust", self.party_s(oc, " ; ")))
        if f["ordering_institution"] or "badbic" in s and s["badbic"][0] == "ordering_institution":
            rows.append(("Ord inst (52A)", self.bic_s(f, s, "ordering_institution")))
        if s.get("missing") != "beneficiary":
            rows.append(("Bene", self.party_s(bn, " ; ")))
        if f["account_with_institution"] or "badbic" in s and s["badbic"][0] == "account_with_institution":
            rows.append(("AWI (57A)", self.bic_s(f, s, "account_with_institution")))
        if f["remittance_info"]:
            rows.append(("Details", f["remittance_info"]))
        if s.get("missing") != "charges":
            rows.append(("Chg", s["charges"]))
        if f["bank_operation_code"]:
            rows.append(("23B", f["bank_operation_code"]))
        sep = r.choice([": ", " = ", " - ", "\t"])
        return "\n".join(f"{k}{sep}{v}" for k, v in rows)

    def r_forward(self, f, s):
        head = "\n".join(["---------- Forwarded message ---------",
                          f"From: {self.person()} <ops{self.r.randint(1, 99)}@example.com>",
                          f"Subject: {self.r.choice(['RE: payment', 'FW: wire instructions', 'Urgent remittance', 'pymt req'])}", ""])
        inner = self.r_email(f, s) if self.r.random() < 0.5 else self.r_chat(f, s)
        return head + inner + "\n\n> earlier: pls ignore the draft I sent yesterday"


def split_ids(n: int, seed: int):
    ids = list(range(n))
    random.Random(seed + 1).shuffle(ids)
    a, b = int(n * 0.8), int(n * 0.9)
    return ids[:a], ids[a:b], ids[b:]


def generate(n: int, seed: int) -> list[dict]:
    g = Gen(seed)
    rows, seen = [], set()
    i = 0
    while len(rows) < n:
        ex = g.example(len(rows))
        i += 1
        h = hashlib.sha1(ex["text"].encode()).hexdigest()
        if h in seen:  # avoid duplicate texts across splits
            continue
        seen.add(h)
        rows.append(ex)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=103)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    a = ap.parse_args()
    rows = generate(a.n, a.seed)
    tr, va, te = split_ids(len(rows), a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for i in sorted(idx):
                fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        counts[name] = len(idx)
    n_err = sum(1 for r in rows if r["target"]["errors"])
    manifest = {"synthetic": True, "note": "SYNTHETIC data generated by data/generate.py; not real payments.",
                "seed": a.seed, "n": len(rows), "splits": counts, "examples_with_errors": n_err}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
