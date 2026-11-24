"""Format-diverse SYNTHETIC training data (v2): many more templates, phrasings and orderings.

Everything produced here is synthetic (random names, accounts, references and amounts; public BICs).

Why: eval round 2 showed the CPU smoke model scores 94.2% on the v1 test split but 66.7% on unseen
formats (data/ood_test.jsonl): it learnt the v1 templates, e.g. it maps SWIFT fields by position and
puts a lone BIC into 57A. v2 attacks that with variety, and with cases that a template-matcher gets
wrong:
  * 10 renderer families (v1's 5 are kept for ~25% of rows): email, chat log with timestamps, SWIFT
    with extra/distractor tags (:13C: :33B: :36: :53A: :56A: :71F: :72:), aligned forms, sectioned
    forms with multi-line parties, CSV export, YAML-like block, formal letter, multilingual
    (de/fr/es/Hindi-Latin labels), long forwarded thread with quoted old instructions
  * many label synonyms per field, random field order everywhere except SWIFT, optional fields
    dropped independently, both BICs or only one of them (labelled, never positional)
  * distractors: other amounts (GST total, fee estimate, old cancelled instruction), other
    references (invoice, ticket, old ref), other dates (mail headers, call dates), IFSC codes,
    phone numbers, VAT numbers, disclaimers, signatures
  * dates: v1 formats plus "05 March 2026", "Thursday, 5 March 2026", "Mar 5, 2026", "5/3/2026",
    "05/03/26", "05.03.26", "5-Mar-2026"
  * amounts: v1 formats (without v1's lossy %g rendering) plus Indian "1,25,000/-", "12.5 lakh",
    "1.2 crore", Swiss "1'234.50", "1 234 567,89", unicode minus for negatives
  * noise: typos (transposed / dropped / doubled letters), random case, extra blank lines and spaces
  * error rate 30% (v1 22%), more MISSING_REQUIRED, and impossible values from lists that extend v1's

HELD-OUT GUARANTEE (tests/test_v2_heldout.py checks each point on the generated files):
  the OOD set (data/generate_ood.py) is NOT used or imitated. v2 has no `| key | value |` tables, no
  `key=value;` one-liners, no shuffled or lower-case SWIFT tags, no OCR letter/digit swaps, none of
  the OOD-only label names, none of the OOD date formats (ordinal days, 2026/03/05, 20260305,
  "Thu 5 Mar 26", 5.3.2026, 05-MAR-2026, "Mar. 5 2026"), none of the OOD remark sentences and none
  of the OOD impossible values. data/generate_ood.py is not modified, so data/ood_test.jsonl is
  byte-identical to the one scored in eval round 2.

Usage:
    python data/generate_v2.py --n 6000 --seed 2026 --out data/v2
Writes data/v2/{train,val,test}.jsonl (80/10/10) and data/v2/MANIFEST.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate import (BAD_BICS, BAD_CCY, BAD_CHARGES, BANKS, CCY, CCY_SYMBOL, MONTHS,  # noqa: E402
                           MONTHS_LONG, Gen, indian_format, split_ids)
from schema import BIC_RE, ISO4217, Extraction  # noqa: E402

DAYS_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

BANKS_V2 = BANKS + ["KKBKINBB", "YESBINBB", "PUNBINBB", "BKIDINBB", "CNRBINBB", "NWBKGB2L", "LOYDGB2L",
                    "SOGEFRPP", "CRLYFRPP", "RABONL2U", "ABNANL2A", "DBSSSGSG", "OCBCSGSG", "BOTKJPJT",
                    "SMBCJPJT", "ANZBAU3M", "NATAAU33", "ROYCCAT2", "TDOMCATT", "NBADAEAA", "EBILAEAD",
                    "CRESCHZZ80A", "WFBIUS6S", "IRVTUS3N", "HDFCINBBXXX", "SBININBB104"]
INDIAN_BANKS = {"SBININBB", "HDFCINBB", "ICICINBBXXX", "UTIBINBB", "KKBKINBB", "YESBINBB", "PUNBINBB",
                "BKIDINBB", "CNRBINBB", "HDFCINBBXXX", "SBININBB104"}
CCY_V2 = dict(CCY, CAD=(3, (100, 200000)), AUD=(3, (100, 200000)), SAR=(2, (500, 400000)),
              ZAR=(1, (500, 900000)), SEK=(1, (500, 900000)), CNY=(2, (500, 900000)))

# impossible values: v1's lists plus new ones; all disjoint from the OOD lists (tested)
BAD_CCY_V2 = BAD_CCY + ["SGP", "JYP", "HKG", "CHN", "AUS", "UAE", "EUU", "USS"]
BAD_BICS_V2 = BAD_BICS + ["SBININ", "HDFC0001234", "CITIUS3", "BARCGB2", "DEUTDEDB1", "ABNA-NL2A", "12345678",
                          "BNP@FRPP"]
BAD_CHARGES_V2 = BAD_CHARGES + ["ALL", "NIL", "SHARE", "BENOUR", "TBD"]

FIRST_V2 = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Meera", "John", "Sarah", "Lukas", "Chloe",
            "Kenji", "Fatima", "Omar", "Elena", "Arjun", "Divya", "Marco", "Aisha", "Wei", "Nikhil",
            "Siddharth", "Kavya", "Rahul", "Sneha", "Ishaan", "Pooja", "Hannah", "Mateo", "Sofia",
            "Yuki", "Ahmed", "Grace", "Tomás", "Ingrid", "Ravi", "Lakshmi", "Daniel", "Amara"]
LAST_V2 = ["Sharma", "Iyer", "Patel", "Reddy", "Kapoor", "Smith", "Müller", "Dubois", "Tanaka", "Khan",
           "Rossi", "Nair", "Chen", "Gupta", "Fernandes", "Menon", "Brown", "Schmidt", "Banerjee",
           "Chatterjee", "Joshi", "Desai", "Kulkarni", "Okafor", "García", "Lindqvist", "Nakamura",
           "O'Brien", "Hassan", "Pillai", "Verma", "Agarwal", "Novak", "Silva"]
CO_A_V2 = ["Arcadia", "Bluepeak", "Coromandel", "Deccan", "Everest", "Falcon", "Ganges", "Helix", "Indus",
           "Juniper", "Kestrel", "Lotus", "Meridian", "Nilgiri", "Orion", "Saffron", "Tata-Vale", "Zenith",
           "Northwind", "Harbor", "Sahyadri", "Konkan", "Vindhya", "Aravalli", "Brightline", "Cobalt",
           "Silverleaf", "Redwood", "Alpine", "Monsoon", "Quartz", "Riverstone"]
CO_B_V2 = ["Textiles", "Logistics", "Pharma", "Steel", "Software", "Exports", "Agro", "Capital", "Trading",
           "Components", "Foods", "Shipping", "Analytics", "Motors", "Solar", "Ceramics", "Healthcare",
           "Infra", "Chemicals", "Retail", "Fintech", "Polymers"]
CO_C_V2 = ["Pvt Ltd", "Ltd", "GmbH", "LLC", "Inc", "SA", "Pte Ltd", "LLP", "BV", "PLC", "Private Limited",
           "AG", "SARL", "Corp", "OPC Pvt Ltd", "KK"]
CITY_V2 = ["Mumbai 400001 IN", "Bengaluru 560001 IN", "Pune 411001 IN", "Chennai 600002 IN",
           "Hyderabad 500081 IN", "Kolkata 700001 IN", "Ahmedabad 380009 IN", "Gurugram 122002 IN",
           "Frankfurt 60311 DE", "Munich 80331 DE", "London EC2V 7HN GB", "Manchester M1 1AE GB",
           "New York NY 10005 US", "Chicago IL 60603 US", "Paris 75009 FR", "Lyon 69002 FR",
           "Singapore 048583 SG", "Hong Kong HK", "Dubai AE", "Zurich 8001 CH", "Tokyo 100-0005 JP",
           "Amsterdam 1012 NL", "Toronto ON M5H 2N2 CA", "Sydney NSW 2000 AU", "Madrid 28013 ES"]
STREET_V2 = ["12 MG Road", "4th Floor, Nariman Point", "221 Park Street", "Plot 7, MIDC", "Taunusanlage 12",
             "1 Churchill Place", "270 Park Avenue", "16 Bd des Italiens", "8 Marina Blvd",
             "Unit 5B, Hi-Tech Park", "Flat 302, Sai Residency", "No. 45, 2nd Cross, Indiranagar",
             "Tower B, Cyber City", "Shop 9, Linking Road", "77 Bishopsgate", "Calle de Alcalá 48",
             "Herengracht 210", "Suite 1200, 100 King St W", "Level 20, 1 Martin Place",
             "Gala 14, Andheri East", "SCO 112, Sector 17"]
REMIT_V2 = ["INV {n} consulting fees", "Invoice {n} / spare parts", "PO {n} advance payment", "Salary {mon} 2026",
            "Rent {mon} 2026 office", "Refund order {n}", "Freight charges BL {n}", "Licence fee Q{q} 2026",
            "Settlement of INV {n} and INV {n2}", "Tuition fees semester {q}", "Dividend payout {mon}",
            "Software subscription {n}", "AMC renewal {n}", "Import duty BOE {n}", "Royalty Q{q} 2026",
            "Retainer {mon} 2026", "Milestone {q} payment PO {n}", "GST refund ref {n}", "Loan EMI {mon} 2026",
            "Security deposit flat {n}", "Commission {mon} 2026", "Container demurrage {n}"]

# label synonyms per field (none of the OOD-only labels; see tests/test_v2_heldout.py)
L = {
    "ref": ["Our reference", "Our ref", "Reference", "Ref no", "Ref #", "Sender's reference", "Payment ref",
            "Transaction ref", "TRN", "Instruction ref", "Your ref (pls quote)", "Client ref", "Ref."],
    "amount": ["Amount", "Amt", "Amount to pay", "Remit amount", "Payment amount", "Total to remit",
               "Instructed amount", "Amount (ccy)", "Net amount"],
    "date": ["Value date", "Val dt", "VD", "Payment date", "Value dt", "Credit on", "Funds on", "Pay on",
             "Date of payment", "Requested value date", "Process on"],
    "oc": ["Ordering customer", "Remitter", "Debit account", "From", "Sender", "Ordering party",
           "Account holder (debit)", "Pay from", "Remitting party", "Applicant"],
    "oi": ["Our bank", "Ordering bank", "Remitter bank SWIFT", "Sender's bank BIC", "Debit bank", "Bank (debit side)",
           "Ordering institution", "Ord inst (52A)", "From bank", "Remitting bank SWIFT code"],
    "bn": ["Beneficiary", "Bene", "Pay to", "Beneficiary name & a/c", "Credit to", "To", "Bene name",
           "Beneficiary details", "In favour of", "Receiver"],
    "awi": ["Beneficiary bank", "Bene bank SWIFT", "Beneficiary bank BIC", "Account with institution",
            "AWI (57A)", "Credit bank", "Bene bank SWIFT code", "Beneficiary's bank", "To bank"],
    "remit": ["Purpose", "Details", "Remittance info", "Narration", "Info for beneficiary",
              "Description", "Reason for payment", "Invoice details", "Particulars", "Remarks to bene"],
    "charges": ["Charges", "Chrgs", "Charge type", "Bank charges", "Charges borne by", "Charge code", "Charges paid by",
                "Who pays charges", "71A charges"],
    "op": ["Op code", "23B", "Operation", "Operation code", "Type", "Op. code"],
}
ML = {  # multilingual-ish labels (German, French, Spanish, Hindi in Latin script)
    "ref": ["Referenz", "Référence", "Referencia", "Sandarbh sankhya"],
    "amount": ["Betrag", "Montant", "Importe", "Rakam"],
    "date": ["Valutadatum", "Date de valeur", "Fecha valor", "Tareekh"],
    "oc": ["Auftraggeber", "Donneur d'ordre", "Ordenante", "Bhejne wala"],
    "oi": ["Bank des Auftraggebers", "Banque émettrice", "Banco ordenante", "Bhejne wale ka bank"],
    "bn": ["Empfänger", "Bénéficiaire", "Beneficiario", "Paane wala"],
    "awi": ["Empfängerbank", "Banque du bénéficiaire", "Banco beneficiario", "Paane wale ka bank"],
    "remit": ["Verwendungszweck", "Motif", "Concepto", "Uddeshya"],
    "charges": ["Gebühren", "Frais", "Gastos", "Shulk"],
    "op": ["Op code", "Code opération", "Código", "Op code"],
}
ML_INTRO = ["Bitte folgende Zahlung ausführen:", "Merci d'exécuter le virement suivant :",
            "Por favor procesar la siguiente transferencia:", "Kripya yeh payment aaj kar dijiye:",
            "bhai ye payment jaldi kar do pls", "Zahlungsauftrag / payment order:"]
CHARGE_WORDS_V2 = {"OUR": ["OUR", "our", "ours", "OUR (we pay all charges)", "all charges on us", "remitter pays all",
                           "charges for our account", "we bear all charges"],
                   "SHA": ["SHA", "sha", "shared", "SHA - shared", "split charges (SHA)", "shared equally",
                           "each side pays its own bank"],
                   "BEN": ["BEN", "ben", "beneficiary pays", "BEN (charges to beneficiary)", "deduct from beneficiary",
                           "all charges to beneficiary"]}
GREET = ["Hi team,", "Dear Ops,", "Hello,", "Good morning,", "Hi", "Dear Sir/Madam,", "Hi Treasury team,",
         "Team,", "Dear Payments desk,", "Namaste,", "Hey all,", "Good afternoon team,"]
ASK = ["{please} process the below {payment} today.", "Kindly arrange the following transfer.",
       "Requesting an outward remittance as per details below.", "Can you release this wire asap?",
       "Please debit our account and remit as follows.", "Need this one out before cut-off pls.",
       "Below are the {payment} details, kindly execute.", "Find the transfer details below:"]
SIGN = ["Thanks,", "Regards,", "Thx", "Best,", "Warm regards,", "Cheers,", "Thanks & regards,", "Sincerely,"]
DISCLAIMER = ["This e-mail is confidential and may be privileged. If you are not the intended recipient, delete it.",
              "Please consider the environment before printing this email.",
              "Sent from my iPhone", "Sent from Outlook for Android",
              "CONFIDENTIALITY NOTICE: the information in this message is intended for the addressee only."]
BULLETS = ["", "", "- ", "* ", "• "]  # never "> ": that marks quoted (old) text in thread2


class GenV2(Gen):
    """Same target logic as v1; new surface formats, renderers, noise and distractors."""

    # ---------- primitives ----------
    def company(self) -> str:
        r = self.r
        return f"{r.choice(CO_A_V2)} {r.choice(CO_B_V2)} {r.choice(CO_C_V2)}"

    def person(self) -> str:
        return f"{self.r.choice(FIRST_V2)} {self.r.choice(LAST_V2)}"

    def account(self) -> str:
        r = self.r
        k = r.random()
        if k < 0.8:
            return super().account()
        if k < 0.9:
            return "".join(r.choice("0123456789") for _ in range(r.choice([14, 16])))  # Indian savings / current
        return f"{r.choice(['SB', 'CA', 'OD'])}-{r.randint(10 ** 9, 10 ** 10 - 1)}"

    def party(self, need_account: bool = True) -> dict:
        r = self.r
        name = self.company() if r.random() < 0.65 else self.person()
        acct = self.account() if (need_account or r.random() < 0.7) else None
        addr = f"{r.choice(STREET_V2)}, {r.choice(CITY_V2)}" if r.random() < 0.55 else None
        return {"account": acct, "name": name, "address": addr}

    def ref(self) -> str:
        r = self.r
        k = r.random()
        if k < 0.6:
            return super().ref()
        if k < 0.7:
            return f"RTGS{r.randint(10 ** 8, 10 ** 9 - 1)}"
        if k < 0.8:
            return f"WIRE-{r.randint(100000, 999999)}"
        if k < 0.9:
            return f"{r.choice(['ACME', 'HLX', 'ZEN', 'ORN'])}/PO/{r.randint(1, 9999):04d}"
        return f"TT{r.randint(26, 27)}{r.randint(1, 12):02d}{r.randint(1000, 9999)}"

    def remit(self) -> str:
        r = self.r
        return r.choice(REMIT_V2).format(n=f"{r.randint(1, 9999):04d}", n2=f"{r.randint(1, 9999):04d}",
                                         mon=r.choice(MONTHS), q=r.randint(1, 4))

    def amount(self, ccy: str) -> float:
        lo, hi = CCY_V2[ccy][1]
        r = self.r
        x = round(r.uniform(lo, hi), 2)
        if ccy == "JPY":
            x = float(int(x))
        k = r.random()
        if k < 0.2:
            x = float(round(x, -2)) or 100.0
        elif k < 0.27 and ccy == "INR":
            x = float(r.randint(1, 400) * 25000)  # round lakh amounts, so "x lakh" renders exactly
        return x

    def fmt_amount(self, x: float, ccy: str) -> str:
        """Every format renders the exact value (v1's f"{x:g}" was lossy above 6 digits)."""
        r = self.r
        k = r.random()
        cents = x != int(x)
        if ccy == "INR":
            if k < 0.15 and x >= 100000 and abs(x / 100000 - round(x / 100000, 2)) < 1e-9:
                v = round(x / 100000, 2)
                return f"{v:g} {r.choice(['lakh', 'lakhs', 'lac'])}"
            if k < 0.2 and x >= 10 ** 7 and abs(x / 10 ** 7 - round(x / 10 ** 7, 2)) < 1e-9:
                return f"{round(x / 10 ** 7, 2):g} crore"
            if k < 0.55:
                s = indian_format(x, decimals=cents or r.random() < 0.5)
                return s + "/-" if r.random() < 0.25 else s
        if ccy in ("EUR", "SEK") and k < 0.3:
            return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        if ccy == "CHF" and k < 0.3:
            return f"{x:,.2f}".replace(",", "'")
        if k < 0.6:
            return f"{x:,.2f}" if cents or r.random() < 0.6 else f"{x:,.0f}"
        if k < 0.8:
            return f"{x:.2f}"
        if k < 0.9:
            return f"{x:,.2f}".replace(",", " ")
        return f"{x:,.2f}".replace(",", " ").replace(".", ",")  # 1 234 567,89

    def fmt_ccy(self, ccy: str) -> str:
        r = self.r
        if ccy in CCY_SYMBOL and r.random() < 0.35:
            return r.choice(CCY_SYMBOL[ccy])
        return ccy if r.random() < 0.85 else ccy.lower()

    def fmt_date(self, d: dt.date) -> str:
        r = self.r
        if r.random() < 0.4:
            return super().fmt_date(d)
        return r.choice([
            f"{d.day:02d} {MONTHS_LONG[d.month - 1]} {d.year}",
            f"{DAYS_LONG[d.weekday()]}, {d.day} {MONTHS_LONG[d.month - 1]} {d.year}",
            f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
            f"{d.day}/{d.month}/{d.year}",
            d.strftime("%d/%m/%y"),
            d.strftime("%d.%m.%y"),
            f"{d.day}-{MONTHS[d.month - 1]}-{d.year}",
            f"{d.day} {MONTHS_LONG[d.month - 1].lower()} {d.year}",
        ])

    def bad_date(self) -> str:
        r = self.r
        if r.random() < 0.5:
            return super().bad_date()
        return r.choice(["31/06/2026", "2026-04-31", "29 Feb 2026", "31-04-2026", "00/12/2026", "2026-02-29",
                         "31.11.2026", "April 31, 2026", "30/02/2027", "2027-13-01"])

    def w2(self, key: str, ml: bool = False) -> str:
        """A label for a field: synonym + typo/case noise."""
        lab = self.r.choice(ML[key] if ml else L[key])
        return self.noise(lab)

    def noise(self, s: str, p: float = 0.18) -> str:
        """Typing noise: transposed / dropped / doubled letter, or random case. (No OCR letter-digit swaps.)"""
        r = self.r
        if r.random() < p:
            idx = [i for i, c in enumerate(s) if c.isalpha()]
            if len(idx) >= 4:
                i = r.choice(idx[1:-1])
                k = r.random()
                if k < 0.4 and s[i + 1].isalpha():
                    s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
                elif k < 0.7:
                    s = s[:i] + s[i + 1:]
                else:
                    s = s[:i] + s[i] + s[i:]
        k = r.random()
        return s.upper() if k < 0.1 else s.lower() if k < 0.2 else s

    def distractor(self, f) -> str:
        r = self.r
        d2 = self.date()
        ccy = f["currency"] or "USD"
        amt2 = self.fmt_amount(self.amount(ccy if ccy in CCY_V2 else "USD"), ccy if ccy in CCY_V2 else "USD")
        opts = [
            f"Invoice total incl. GST: {ccy} {amt2} (we pay part now)",
            f"Bank charges estimate approx {ccy} {r.randint(5, 60)}.00",
            f"Ticket #{r.randint(10000, 99999)} raised on {self.fmt_date(d2)}",
            f"Call me at +44 20 {r.randint(1000, 9999)} {r.randint(1000, 9999)} if needed",
            f"Old instruction {self.ref()} for {ccy} {amt2} stands cancelled",
            f"Our VAT no DE{r.randint(10 ** 8, 10 ** 9 - 1)}",
            f"PAN on file, GSTIN 27AAAC{r.randint(1000, 9999)}F1Z{r.randint(1, 9)}",
            f"Cut-off today is {r.randint(1, 4)}:30 pm.",
            f"Last payment to them went out on {self.fmt_date(d2)}.",
            f"(Supersedes draft ref {self.ref()})",
        ]
        if f["account_with_institution"] in INDIAN_BANKS or f["ordering_institution"] in INDIAN_BANKS:
            opts.append(f"IFSC: {r.choice(['HDFC', 'SBIN', 'ICIC', 'UTIB'])}0{r.randint(100000, 999999)}")
        return r.choice(opts)

    # ---------- one example ----------
    def example(self, idx: int) -> dict:
        r = self.r
        ccy = r.choices(list(CCY_V2), weights=[v[0] for v in CCY_V2.values()])[0]
        amt = self.amount(ccy)
        d = self.date()
        f = {
            "sender_reference": self.ref(),
            "bank_operation_code": "CRED" if r.random() < 0.85 else r.choice(["SPAY", "SPRI", "SSTD", "CRTS"]),
            "value_date": d.isoformat(), "currency": ccy, "amount": amt,
            "ordering_customer": self.party(), "ordering_institution": r.choice(BANKS_V2),
            "account_with_institution": r.choice(BANKS_V2), "beneficiary": self.party(),
            "remittance_info": self.remit(), "charges": r.choice(["OUR", "SHA", "SHA", "BEN"]),
        }
        surf = {"date": self.fmt_date(d), "ccy": self.fmt_ccy(ccy), "amt": self.fmt_amount(amt, ccy),
                "charges": r.choice(CHARGE_WORDS_V2[f["charges"]])}
        # optional fields absent from the text (independently, so "only 52A" happens as often as "only 57A")
        for k, p in (("bank_operation_code", 0.45), ("ordering_institution", 0.35),
                     ("account_with_institution", 0.35), ("remittance_info", 0.2)):
            if r.random() < p:
                f[k] = None
        errors: list[str] = []
        if r.random() < 0.30:
            kind = r.choices(["ccy", "date", "bic", "amt", "charges", "missing"], weights=[1, 1, 1, 1, 1, 1.4])[0]
            if kind == "ccy":
                bad = r.choice(BAD_CCY_V2); assert bad not in ISO4217
                surf["ccy"] = bad; f["currency"] = None
                errors.append(f"INVALID_CURRENCY: currency: {bad} is not an ISO-4217 code")
            elif kind == "date":
                surf["date"] = self.bad_date(); f["value_date"] = None
                errors.append(f"INVALID_DATE: value_date: {surf['date']} is not a real calendar date")
            elif kind == "bic":
                which = r.choice(["ordering_institution", "account_with_institution"])
                bad = r.choice(BAD_BICS_V2); f[which] = None
                surf["badbic"] = (which, bad)
                errors.append(f"INVALID_BIC: {which}: {bad} is not a valid BIC")
            elif kind == "amt":
                bad = r.choice(["0.00", "-" + self.fmt_amount(amt, ccy), "0", "-" + f"{amt:.2f}",
                                "−" + f"{amt:,.2f}", "-0.00"])
                surf["amt"] = bad; f["amount"] = None
                errors.append(f"NON_POSITIVE_AMOUNT: amount: {bad} is not a positive amount")
            elif kind == "charges":
                bad = r.choice(BAD_CHARGES_V2); surf["charges"] = bad; f["charges"] = None
                errors.append(f"INVALID_CHARGES: charges: {bad} is not OUR/SHA/BEN")
            else:
                miss = r.choice(["charges", "beneficiary", "value_date"])
                f[miss] = None; surf["missing"] = miss
                errors.append(f"MISSING_REQUIRED: {miss}: not stated in the text")
        text = self.render(f, surf)
        target = {"fields": f, "errors": errors}
        Extraction.model_validate(target)
        return {"id": f"v2-{idx:05d}", "synthetic": True, "style": self._style, "text": text, "target": target}

    # ---------- renderers ----------
    STYLES = {"email2": 14, "chat2": 10, "swift2": 12, "form2": 9, "sections2": 7, "csv2": 5, "yaml2": 5,
              "letter2": 7, "multi2": 7, "thread2": 8, "v1": 16}

    def render(self, f, surf) -> str:
        style = self.r.choices(list(self.STYLES), weights=list(self.STYLES.values()))[0]
        if style == "v1" and surf.get("charges") in set(BAD_CHARGES_V2) - set(BAD_CHARGES):
            style = "email2"  # v1's SWIFT renderer only knows v1's bad charge words
        self._style = style
        if style == "v1":
            s = super().render(f, surf)
            if self.r.random() < 0.3:
                s = s + "\n\n" + self.distractor(f)
            return s
        return getattr(self, f"r_{style}")(f, surf).strip()

    def _has_bic(self, f, s, key):
        return bool(f[key]) or s.get("badbic", ("",))[0] == key

    def _money2(self, s):
        c, a = s["ccy"], s["amt"]
        if c in ("$", "US$", "€", "£", "₹", "¥", "Rs.", "Rs"):
            return f"{c}{a}" if self.r.random() < 0.5 else f"{c} {a}"
        return f"{c} {a}" if self.r.random() < 0.7 else f"{a} {c}"

    def _party_txt(self, p, sep=", "):
        r = self.r
        bits = []
        if p.get("account"):
            bits.append(p["account"] if r.random() < 0.4 else f"{r.choice(['a/c', 'A/C', 'acct', 'account', 'Acc no', 'IBAN' if p['account'][:2].isalpha() else 'a/c no'])} {p['account']}")
        bits.append(p["name"])
        if p.get("address"):
            bits.append(p["address"])
        if r.random() < 0.3 and len(bits) > 1:  # name first, then account
            bits[0], bits[1] = bits[1], bits[0]
        return sep.join(bits)

    def _items(self, f, s, ml=False):
        """(field key, label, value) for every field present in the text, in random order."""
        it = [("ref", f["sender_reference"]), ("amount", self._money2(s))]
        if s.get("missing") != "value_date":
            it.append(("date", s["date"]))
        it.append(("oc", self._party_txt(f["ordering_customer"])))
        if self._has_bic(f, s, "ordering_institution"):
            it.append(("oi", self.bic_s(f, s, "ordering_institution")))
        if s.get("missing") != "beneficiary":
            it.append(("bn", self._party_txt(f["beneficiary"])))
        if self._has_bic(f, s, "account_with_institution"):
            it.append(("awi", self.bic_s(f, s, "account_with_institution")))
        if f["remittance_info"]:
            it.append(("remit", f["remittance_info"]))
        if s.get("missing") != "charges":
            it.append(("charges", s["charges"]))
        if f["bank_operation_code"]:
            it.append(("op", f["bank_operation_code"]))
        self.r.shuffle(it)
        return [(k, self.w2(k, ml), v) for k, v in it]

    def r_email2(self, f, s):
        r = self.r
        sep = r.choice([": ", " : ", " - ", ":  ", " – "])
        b = r.choice(BULLETS)
        lines = [r.choice(GREET), "", r.choice(ASK).format(please=self.w("please").capitalize(), payment=self.w("payment"))]
        if r.random() < 0.3:
            lines.append("")
        lines += [f"{b}{lab}{sep}{v}" for _, lab, v in self._items(f, s)]
        if r.random() < 0.5:
            lines += ["", self.distractor(f)]
        lines += ["", r.choice(SIGN), self.person()]
        if r.random() < 0.4:
            lines.append(r.choice(["Treasury Ops", "Accounts Payable", "Finance team", f"Ext. {r.randint(100, 999)}"]))
        if r.random() < 0.3:
            lines += ["", r.choice(DISCLAIMER)]
        return "\n".join(lines)

    def r_chat2(self, f, s):
        r = self.r
        who = [self.r.choice(FIRST_V2), self.r.choice(FIRST_V2)]
        h, m = r.randint(8, 18), r.randint(0, 50)
        msgs = []
        chunks = []
        for k, lab, v in self._items(f, s):
            if k in ("amount", "bn") and r.random() < 0.5:
                chunks.append(("send " if k == "amount" else "to ") + v)
            elif r.random() < 0.5:
                chunks.append(f"{lab.lower()} {v}")
            else:
                chunks.append(f"{lab} {v}")
        opener = r.choice(["pls do a payment", "urgent pymt", "need one wire today", "ek payment karna hai",
                           "quick one", "can u release a transfer"])
        msgs.append(opener)
        i = 0
        while i < len(chunks):
            n = r.randint(1, 3)
            msgs.append(", ".join(chunks[i:i + n]))
            i += n
        if r.random() < 0.4:
            msgs.insert(r.randint(1, len(msgs)), self.distractor(f))
        out = []
        for j, mtxt in enumerate(msgs):
            m += r.randint(0, 3)
            stamp = r.choice([f"[{h:02d}:{m % 60:02d}] {who[0]}: ", f"{who[0]} {h:02d}:{m % 60:02d} - ", ""])
            out.append(stamp + mtxt)
        if r.random() < 0.5:
            out.append(r.choice([f"[{h:02d}:{(m + 2) % 60:02d}] {who[1]}: ok will do", f"{who[1]}: noted 👍",
                                 f"{who[1]}: done?"]))
        return "\n".join(out)

    def r_swift2(self, f, s):
        """Standard tag ORDER (never shuffled), upper-case tags; extra and distractor tags vary positions."""
        r = self.r
        oc, bn = f["ordering_customer"], f["beneficiary"]
        ccy = s["ccy"] if len(s["ccy"]) == 3 and s["ccy"].isalpha() else (f["currency"] or s["ccy"])
        ccy = ccy.upper()
        amt = s["amt"]
        if not amt.startswith(("-", "−")) and not any(w in amt for w in ("lakh", "lac", "crore")) and f["amount"]:
            amt = f"{f['amount']:.2f}".replace(".", ",")
        date = s["date"]
        if f["value_date"] and r.random() < 0.75:
            date = dt.date.fromisoformat(f["value_date"]).strftime("%y%m%d")
        out = []
        if r.random() < 0.4 and f["ordering_institution"] and f["account_with_institution"]:
            out.append(f"{{1:F01{f['ordering_institution'][:8]}AXXX{r.randint(10 ** 9, 10 ** 10 - 1)}}}"
                       f"{{2:I103{f['account_with_institution'][:8]}XN}}{{3:{{108:{r.randint(1000, 9999)}}}}}{{4:")
        out.append(f":20:{f['sender_reference']}")
        if r.random() < 0.15:
            out.append(f":13C:/SNDTIME/{r.randint(1000, 2359):04d}+0530")
        if f["bank_operation_code"]:
            out.append(f":23B:{f['bank_operation_code']}")
        if s.get("missing") != "value_date":
            out.append(f":32A:{date}{ccy}{amt}")
        else:
            out.append(f":32A:{ccy}{amt}")
        if r.random() < 0.2 and f["amount"] and f["currency"]:
            out.append(f":33B:{ccy}{amt}")  # instructed amount = same amount (common in real MT103s)
        out.append(":50K:" + "\n".join(x for x in [("/" + oc["account"]) if oc["account"] else None, oc["name"], oc["address"]] if x))
        if self._has_bic(f, s, "ordering_institution"):
            out.append(f":52A:{self.bic_s(f, s, 'ordering_institution')}")
        if r.random() < 0.15:
            out.append(f":53A:{r.choice(BANKS_V2)}")  # sender's correspondent: NOT 52A/57A
        if r.random() < 0.15:
            out.append(f":56A:{r.choice(BANKS_V2)}")  # intermediary: NOT 52A/57A
        if self._has_bic(f, s, "account_with_institution"):
            out.append(f":57A:{self.bic_s(f, s, 'account_with_institution')}")
        if s.get("missing") != "beneficiary":
            out.append(":59:" + "\n".join(x for x in [("/" + bn["account"]) if bn["account"] else None, bn["name"], bn["address"]] if x))
        if f["remittance_info"]:
            out.append(f":70:{f['remittance_info']}")
        if s.get("missing") != "charges":
            ch = s["charges"] if s["charges"] in BAD_CHARGES_V2 else f["charges"]
            out.append(f":71A:{ch}")
        if r.random() < 0.2:
            out.append(f":71F:{ccy}{r.randint(5, 40)},00")  # sender's charges: a distractor amount
        if r.random() < 0.2:
            out.append(f":72:/ACC/{r.choice(['PLS CREDIT SAME DAY', 'URGENT', 'TEL BEFORE PAYMENT'])}")
        if out[0].startswith("{1:"):
            out.append("-}")
        wrap = r.random()
        if wrap < 0.3:
            return "\n".join([r.choice(GREET), "", r.choice(["MT103 below, pls check before release:",
                                                              "Copy of the swift we got:", "Draft MT103:"]), ""] + out
                             + ["", r.choice(SIGN), self.person()])
        if wrap < 0.45:
            return "\n".join(out) + "\n\n" + self.distractor(f)
        return "\n".join(out)

    def r_form2(self, f, s):
        r = self.r
        items = self._items(f, s)
        width = max(len(lab) for _, lab, _ in items) + r.randint(2, 6)
        fill = r.choice([" ", ".", "_"])
        title = r.choice(["PAYMENT INSTRUCTION FORM", "Outward Remittance Request", "WIRE TRANSFER REQUEST",
                          "Fund Transfer Form (Treasury)", "Payment Request #" + str(r.randint(1000, 9999))])
        lines = [title, r.choice(["=" * len(title), "-" * len(title), ""])]
        for _, lab, v in items:
            lines.append(f"{lab.ljust(width, fill)} {v}" if fill != " " else f"{lab}:".ljust(width + 1) + f" {v}")
        if r.random() < 0.4:
            lines += ["", f"Authorised by: {self.person()}", f"Checked by: {self.person()}"]
        return "\n".join(lines)

    def r_sections2(self, f, s):
        r = self.r
        oc, bn = f["ordering_customer"], f["beneficiary"]
        hdr = r.choice([lambda t: f"[{t}]", lambda t: f"## {t}", lambda t: f"{t.upper()}:", lambda t: f"--- {t} ---"])

        def block(title, p, bic_key, bic_label):
            out = [hdr(title), f"{r.choice(['Name', 'Name', 'Account name'])}: {p['name']}"]
            if p.get("account"):
                out.append(f"{r.choice(['Account', 'A/c no', 'Account number', 'IBAN / a/c'])}: {p['account']}")
            if p.get("address"):
                out.append(f"{r.choice(['Address', 'Addr'])}: {p['address']}")
            if self._has_bic(f, s, bic_key):
                out.append(f"{bic_label}: {self.bic_s(f, s, bic_key)}")
            return out

        secs = [block(r.choice(["Ordering customer", "Remitter", "Debit side", "From"]), oc, "ordering_institution",
                      r.choice(["Bank SWIFT", "BIC", "Bank BIC"]))]
        if s.get("missing") != "beneficiary":
            secs.append(block(r.choice(["Beneficiary", "Credit side", "Bene", "To"]), bn, "account_with_institution",
                              r.choice(["Bank SWIFT", "BIC", "Bank BIC"])))
        elif self._has_bic(f, s, "account_with_institution"):
            secs.append([hdr("Beneficiary bank"), f"BIC: {self.bic_s(f, s, 'account_with_institution')}"])
        pay = [hdr(r.choice(["Payment", "Transfer", "Details"])), f"{self.w2('ref')}: {f['sender_reference']}",
               f"{self.w2('amount')}: {self._money2(s)}"]
        if s.get("missing") != "value_date":
            pay.append(f"{self.w2('date')}: {s['date']}")
        if f["remittance_info"]:
            pay.append(f"{self.w2('remit')}: {f['remittance_info']}")
        if s.get("missing") != "charges":
            pay.append(f"{self.w2('charges')}: {s['charges']}")
        if f["bank_operation_code"]:
            pay.append(f"{self.w2('op')}: {f['bank_operation_code']}")
        secs.insert(r.randint(0, len(secs)), pay)
        return "\n\n".join("\n".join(x) for x in secs)

    def r_csv2(self, f, s):
        r = self.r
        sep = r.choice([",", "\t", ";"])
        items = self._items(f, s)
        q = lambda v: f'"{v}"' if (sep in v or "," in v or r.random() < 0.3) else v  # noqa: E731
        head = sep.join(q(lab) for _, lab, _ in items)
        row = sep.join(q(v) for _, _, v in items)
        pre = r.choice(["", "Exported from ERP, 1 row:", "payments_batch.csv (row 1 of 1)", "Upload file:"])
        return "\n".join(x for x in [pre, head, row] if x)

    def r_yaml2(self, f, s):
        r = self.r
        ind = r.choice(["  ", "    "])
        out = [r.choice(["payment:", "transfer:", "instruction:"])]
        for k, lab, v in self._items(f, s):
            key = lab.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("'", "")
            if k in ("oc", "bn"):
                p = f["ordering_customer"] if k == "oc" else f["beneficiary"]
                out.append(f"{ind}{key}:")
                for pk in ("name", "account", "address"):
                    if p.get(pk):
                        out.append(f"{ind * 2}{pk}: {p[pk]}")
            else:
                out.append(f"{ind}{key}: {v}")
        return "\n".join(out)

    def r_letter2(self, f, s):
        r = self.r
        oc, bn = f["ordering_customer"], f["beneficiary"]
        sents = [f"We, {oc['name']}" + (f" (account {oc['account']})" if oc.get("account") else "")
                 + ", hereby request you to remit the amount below."]
        sents.append(f"The sum is {self._money2(s)}"
                     + (f", payable to {self._party_txt(bn)}" if s.get("missing") != "beneficiary" else "") + ".")
        if s.get("missing") != "value_date":
            sents.append(r.choice(["Value date requested: ", "Kindly credit the funds with value ",
                                   "The payment should be effected on "]) + s["date"] + ".")
        sents.append(r.choice(["Please mention reference ", "Our internal reference is ", "Kindly quote "])
                     + f["sender_reference"] + (" in all correspondence." if r.random() < 0.5 else "."))
        if self._has_bic(f, s, "ordering_institution"):
            sents.append(f"Please debit our account held with your branch, BIC {self.bic_s(f, s, 'ordering_institution')}.")
        if self._has_bic(f, s, "account_with_institution"):
            sents.append(f"The beneficiary banks with SWIFT {self.bic_s(f, s, 'account_with_institution')}.")
        if f["remittance_info"]:
            sents.append(f"This payment is towards {f['remittance_info']}.")
        if s.get("missing") != "charges":
            sents.append(f"Charges: {s['charges']}.")
        if f["bank_operation_code"]:
            sents.append(f"Bank operation code {f['bank_operation_code']}.")
        head, rest = sents[:1], sents[1:]
        r.shuffle(rest)
        if r.random() < 0.4:
            rest.insert(r.randint(0, len(rest)), self.distractor(f) + ".")
        body = " ".join(head + rest)
        return "\n".join([r.choice(["To,\nThe Branch Manager", "Dear Sir/Madam,", "To the Payments Department,"]), "",
                          r.choice(["Sub: Request for outward remittance", "Re: wire transfer", "Subject: Payment"]), "",
                          body, "", r.choice(["Yours faithfully,", "Yours sincerely,", "Regards,"]),
                          f"For {oc['name']}", "(Authorised Signatory)"])

    def r_multi2(self, f, s):
        r = self.r
        sep = r.choice([": ", " : ", " - "])
        lines = [r.choice(ML_INTRO), ""]
        for _, lab, v in self._items(f, s, ml=r.random() < 0.8):
            lines.append(f"{lab}{sep}{v}")
        lines += ["", r.choice(["Danke", "Merci", "Gracias", "Dhanyavaad", "Thanks / Danke", "Shukriya"])]
        return "\n".join(lines)

    def r_thread2(self, f, s):
        r = self.r
        sent = self.date()
        hdr = [r.choice(["-----Original Message-----", "---------- Forwarded message ---------",
                         "Begin forwarded message:", "________________________________"]),
               f"From: {self.person()} <{r.choice(['ops', 'treasury', 'ap', 'finance'])}{r.randint(1, 99)}@example.com>",
               f"Sent: {DAYS_LONG[sent.weekday()][:3]}, {sent.day:02d} {MONTHS[sent.month - 1]} {sent.year} "
               f"{r.randint(8, 19):02d}:{r.randint(0, 59):02d}",
               f"Subject: {r.choice(['RE: RE: payment', 'FW: wire instructions', 'Fwd: urgent remittance', 'RE: pymt req – revised'])}",
               ""]
        new_part = self.r_email2(f, s) if r.random() < 0.6 else self.r_chat2(f, s)
        old_ccy = f["currency"] or "USD"
        old_amt = self.fmt_amount(self.amount(old_ccy if old_ccy in CCY_V2 else "USD"), old_ccy if old_ccy in CCY_V2 else "USD")
        old = [f"> {r.choice(['Earlier draft (do NOT use)', 'Previous instruction (cancelled)', 'Older mail below, ignore'])}:",
               f"> {self.w2('ref')}: {self.ref()}",
               f"> {self.w2('amount')}: {old_ccy} {old_amt}",
               f"> {self.w2('date')}: {self.fmt_date(self.date())}"]
        top = [r.choice(["Pls action the latest one below.", "Revised details below, thanks.", "FYI - see below",
                         "Forwarding for processing."]), ""]
        parts = top + hdr + [new_part, ""] + (old if r.random() < 0.7 else [])
        if r.random() < 0.3:
            parts += ["", r.choice(DISCLAIMER)]
        return "\n".join(parts)


def generate_v2(n: int, seed: int, exclude_texts: frozenset = frozenset()) -> list[dict]:
    g = GenV2(seed)
    rows, seen = [], set(exclude_texts)
    while len(rows) < n:
        ex = g.example(len(rows))
        h = hashlib.sha1(ex["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append(ex)
    return rows


def ood_text_hashes() -> frozenset:
    p = Path(__file__).parent / "ood_test.jsonl"
    if not p.exists():
        return frozenset()
    return frozenset(hashlib.sha1(json.loads(x)["text"].encode()).hexdigest()
                     for x in open(p, encoding="utf-8") if x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(Path(__file__).parent / "v2"))
    a = ap.parse_args()
    rows = generate_v2(a.n, a.seed, ood_text_hashes())
    tr, va, te = split_ids(len(rows), a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for i in sorted(idx):
                fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        counts[name] = len(idx)
    styles, codes = {}, {}
    for row in rows:
        styles[row["style"]] = styles.get(row["style"], 0) + 1
        for e in row["target"]["errors"]:
            codes[e.split(":")[0]] = codes.get(e.split(":")[0], 0) + 1
    manifest = {"synthetic": True, "note": "SYNTHETIC format-diverse training data (data/generate_v2.py); not real payments.",
                "seed": a.seed, "n": len(rows), "splits": counts, "styles": dict(sorted(styles.items())),
                "error_codes": dict(sorted(codes.items())),
                "examples_with_errors": sum(1 for r in rows if r["target"]["errors"])}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    assert all(c not in ISO4217 for c in BAD_CCY_V2)
    assert all(not BIC_RE.match(b) for b in BAD_BICS_V2)
    main()
