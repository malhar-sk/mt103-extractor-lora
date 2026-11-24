"""Deterministic post-processing of a model's extraction, checked against the source text.

It is model-agnostic (base, fine-tuned or API output) and uses no dataset-specific lists: only the
schema's own rules (ISO-4217 list, BIC shape, real calendar dates, amount > 0, OUR/SHA/BEN) and the
source text. Every rule either normalises a value, or removes something that cannot be right.

  R1 shape      keep only schema keys; "", "null", "n/a", "none" -> null; strip strings.
  R2 normalise  currency symbol / case -> ISO code; amount strings -> number (Indian lakh/crore and
                1,25,000 grouping, European 1.234,56, Swiss 1'234.50, spaces); date strings -> ISO
                (numeric dates are day-first, as in the prompt); charges words -> OUR/SHA/BEN;
                BICs and op codes upper-case.
  R3 validate   an impossible value is set to null. An error is added only if that impossible value
                literally occurs in the source text (so the model's own garbage is not reported as a
                problem with the payment).
  R4 ground     a BIC, op code, currency, amount, date, account or reference that does not occur in
                the source text is an invention and is set to null. An amount whose digit string
                matches exactly one amount in the text is replaced by that amount, parsed properly
                (fixes thousands/decimal separator misreads such as 4565.3955 for 4,565,395.50).
                An amount that only occurs negated in the text (-500, (500)) is null + NON_POSITIVE_AMOUNT.
  R5 errors     canonical "CODE: field: detail"; unknown codes and impossible code/field pairs are
                dropped; an error on a field that holds a valid value is dropped (by the schema
                contract a flagged field is null); an error that calls a valid, grounded value
                impossible is dropped and the value restored (unless another field already holds it).
  R6 add flags  value_date null with no error: INVALID_DATE if the text contains an impossible date,
                else MISSING_REQUIRED if the text contains no date at all. charges null with no error
                and no charges wording in the text: MISSING_REQUIRED.

Contract (tested): postprocess(gold, text) == gold for every gold target, i.e. the rules never
damage a correct answer.
"""
from __future__ import annotations

import calendar
import copy
import datetime as dt
import re

from schema import BANK_OP_CODES, BIC_RE, ERROR_CODES, FIELD_KEYS, ISO4217

PARTY_KEYS = ("account", "name", "address")
NULLISH = {"", "null", "none", "n/a", "na", "nil", "-", "unknown", "not stated", "not provided"}

SYMBOLS = [("US$", "USD"), ("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("₹", "INR"), ("¥", "JPY")]
CCY_WORDS = {"RS": "INR", "RS.": "INR", "RUPEES": "INR", "RUPEE": "INR", "DOLLARS": "USD", "EUROS": None}
CHARGE_MAP = {"OUR": "OUR", "OURS": "OUR", "SHA": "SHA", "SHARED": "SHA", "BEN": "BEN",
              "BENEFICIARY": "BEN", "BENEFICIARY PAYS": "BEN"}
FIXED_FIELD = {"INVALID_CURRENCY": "currency", "INVALID_DATE": "value_date",
               "NON_POSITIVE_AMOUNT": "amount", "INVALID_CHARGES": "charges"}
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MON_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"


# ----------------------------------------------------------------------------- helpers
def _clean(v):
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v).strip()
        return None if s.casefold() in NULLISH else s
    return v


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.casefold())


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).casefold()


def _token_in(token: str, text_upper: str) -> bool:
    if re.search(r"(?<![A-Z0-9])" + re.escape(token.upper()) + r"(?![A-Z0-9])", text_upper):
        return True
    # SWIFT 32A packs date, currency and amount together: ':32A:260905INR2681837,84'
    return any(m.group("ccy") == token.upper() for m in SWIFT_32A_FULL.finditer(text_upper))


def _ccy_grounded(code: str, text_upper: str) -> bool:
    """The ISO code, or a symbol / word that means it, occurs in the text."""
    return (_token_in(code, text_upper) or any(c == code and sym in text_upper for sym, c in SYMBOLS)
            or (code == "INR" and re.search(r"\bRS\.?(?![A-Z])", text_upper) is not None))


# ----------------------------------------------------------------------------- dates
def _mk(y: int, m: int, d: int):
    if y < 100:
        y += 2000
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


_B = r"(?<![\w./-])"  # not inside a longer number (1.234.567) or a reference (PAY-2026-01)
_E = r"(?![\w/-]|\.\d)"
DATE_PATTERNS = [
    # (regex, groups -> (y, m, d)); numeric dates are day-first, 4-digit-year-first are Y-M-D
    (re.compile(_B + r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})" + _E), lambda g: (g[0], g[1], g[2])),
    (re.compile(_B + r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})" + _E), lambda g: (g[2], g[1], g[0])),
    (re.compile(_B + r"(20\d{2})(\d{2})(\d{2})" + _E), lambda g: (g[0], g[1], g[2])),
    (re.compile(r"(?i)(?<![\w])(\d{1,2})(?:st|nd|rd|th)?[\s-]+" + MON_RE + r",?[\s-]+(\d{4}|\d{2})(?!\d)"),
     lambda g: (g[2], g[1], g[0])),
    (re.compile(r"(?i)(?<![\w])" + MON_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4}|\d{2})(?!\d)"),
     lambda g: (g[2], g[0], g[1])),
]
SWIFT_32A = re.compile(r"(?i):32A:\s*(\d{2})(\d{2})(\d{2})(?!\d)")
SWIFT_32A_FULL = re.compile(r"(?im):32A:[ \t]*(?P<date>[^\n]*)(?<![A-Z])(?P<ccy>[A-Z]{3,5})[ \t]*[-−(]?\d")
YYMMDD = re.compile(_B + r"(2\d)(\d{2})(\d{2})" + _E)


def _month(tok: str):
    t = tok.lower().rstrip(".")[:3]
    return MONTHS.get(t)


def text_dates(text: str):
    """(valid dates, impossible date strings) mentioned in the text."""
    valid, bad = set(), []
    for rx, order in DATE_PATTERNS:
        for m in rx.finditer(text):
            y, mo, d = order(m.groups())
            mo = int(mo) if str(mo).isdigit() else _month(mo)
            if mo is None:
                continue
            x = _mk(int(y), mo, int(d))
            if x:
                valid.add(x)
            else:
                bad.append(m.group(0).strip())
    for m in SWIFT_32A.finditer(text):
        x = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        (valid.add(x) if x else bad.append(m.group(1) + m.group(2) + m.group(3)))
    for m in SWIFT_32A_FULL.finditer(text):  # date glued to the currency: ':32A:17/01/2026USD1749,31'
        if m.group("date") and not re.fullmatch(r"\d{6}", m.group("date")):
            v2, b2 = text_dates(m.group("date") + " ")
            valid |= v2
            bad += [b for b in b2 if b not in bad]
    for m in YYMMDD.finditer(text):  # SWIFT-style YYMMDD, only counted when it is a real date
        x = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if x:
            valid.add(x)
    return valid, bad


def parse_date_value(v):
    """Model's value_date -> (iso or None, impossible: bool)."""
    if not isinstance(v, str):
        return None, False
    s = v.strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        x = _mk(*map(int, m.groups()))
        return (x.isoformat(), False) if x else (None, True)
    valid, bad = text_dates(s)
    if len(valid) == 1 and not bad:
        return next(iter(valid)).isoformat(), False
    return None, bool(bad)


# ----------------------------------------------------------------------------- amounts
_MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7}


def parse_number(s: str):
    """'1,25,000.50' / '1.234,56' / "1'234.50" / '12 500,00' / '2.5 lakh' / '6.9e+06' -> float."""
    if s is None:
        return None
    s = str(s).strip().replace("\u00a0", " ").replace("’", "'")
    neg = s.startswith(("-", "−")) or (s.startswith("(") and s.endswith(")"))
    s = s.strip("()-− ")
    mult = 1.0
    m = re.search(r"(?i)\s*(lakhs?|lacs?|crores?|cr)\.?$", s)
    if m:
        mult = _MULT[m.group(1).lower()]
        s = s[: m.start()]
    s = re.sub(r"[^\d.,' eE+-]", "", s).strip()
    if re.fullmatch(r"\d+(\.\d+)?[eE][+-]?\d+", s):
        x = float(s)
    else:
        s = s.replace("'", "").replace(" ", "")
        if not re.fullmatch(r"[\d.,]+", s or "x"):
            return None
        if "," in s and "." in s:
            dec = "," if s.rfind(",") > s.rfind(".") else "."
            th = "." if dec == "," else ","
            s = s.replace(th, "").replace(dec, ".")
        elif "," in s:
            parts = s.split(",")
            if len(parts) == 2 and len(parts[1]) in (1, 2):
                s = s.replace(",", ".")  # European decimal comma
            else:
                s = s.replace(",", "")
        elif s.count(".") > 1:
            s = s.replace(".", "")
        try:
            x = float(s)
        except ValueError:
            return None
    x *= mult
    return -x if neg else x


AMOUNT_RE = re.compile(
    r"(?P<neg>\(|-|−)?(?P<num>\d+(?:\.\d+)?[eE][+-]?\d+|\d{1,3}(?:[ ,.'’]\d{2,3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d+)?)"
    r"(?P<mult>\s*(?:lakhs?|lacs?|crores?|cr)\b)?(?P<close>\))?", re.I)


def _tolerance(num: str) -> float:
    """Half a unit of the last digit written: 256933 -> 0.5, 6.94919e+06 -> 5.0, 1749.31 -> 0.005."""
    m = re.fullmatch(r"(\d+)(?:\.(\d+))?[eE]([+-]?\d+)", num)
    if m:
        return max(0.005, 0.5 * 10 ** (int(m.group(3)) - len(m.group(2) or "")))
    m = re.search(r"[.,](\d{1,2})$", num)
    return max(0.005, 0.5 * 10 ** -len(m.group(1))) if m else 0.5


def text_amounts(text: str):
    """[(value, digits, negative, tolerance)] for every number-like token in the text."""
    out = []
    for m in AMOUNT_RE.finditer(text):
        prev = text[m.start() - 1] if m.start() > 0 else " "
        if prev.isalnum() and not m.group("neg") and not _after_ccy(text, m.start()):
            continue  # part of a word / reference such as TXN123
        x = parse_number(m.group("num") + (m.group("mult") or ""))
        if x is None:
            continue
        neg = bool(m.group("neg") == "(" and m.group("close")) or m.group("neg") in ("-", "−")
        if m.group("neg") in ("-", "−") and prev.isalnum():
            neg = False  # a hyphen inside a token such as 844420567-96
        tol = 0.005 if m.group("mult") else _tolerance(m.group("num"))
        out.append((x, re.sub(r"\D", "", m.group("num")).rstrip("0"), neg, tol))
    return out


def _after_ccy(text: str, i: int) -> bool:
    """True when the number is glued to a currency: 'Rs26,589.91', 'INR2681837,84' (SWIFT 32A)."""
    m = re.search(r"([A-Za-z]+)\.?$", text[max(0, i - 8):i])
    # 'rs', an ISO code, or any 3-letter upper-case code (an invalid currency such as 'GBR' is still a currency slot)
    return bool(m) and (m.group(1).lower() == "rs" or m.group(1)[-3:].upper() in ISO4217
                        or re.fullmatch(r"[A-Z]{3}", m.group(1)[-3:]) is not None)


def _digits_of(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return re.sub(r"\D", "", s).rstrip("0")


# ----------------------------------------------------------------------------- errors
def parse_error(e):
    if not isinstance(e, str) or ":" not in e:
        return None
    parts = [p.strip() for p in e.split(":", 2)]
    code = parts[0].upper().replace(" ", "_")
    if code not in ERROR_CODES:
        return None
    raw_field = parts[1].lower() if len(parts) > 1 else ""
    detail = parts[2] if len(parts) > 2 else ""
    if code in FIXED_FIELD:
        field = FIXED_FIELD[code]
    elif code == "INVALID_BIC":
        if "order" in raw_field or "52" in raw_field or "sender" in raw_field:
            field = "ordering_institution"
        elif "account_with" in raw_field or "57" in raw_field or "benef" in raw_field or "receiv" in raw_field:
            field = "account_with_institution"
        else:
            return None
    else:  # MISSING_REQUIRED
        if "date" in raw_field:
            field = "value_date"
        elif "benef" in raw_field or raw_field.startswith("59"):
            field = "beneficiary"
        elif "charge" in raw_field or "71" in raw_field:
            field = "charges"
        else:
            return None
    return code, field, detail


def _fmt_error(code, field, detail):
    return f"{code}: {field}: {detail}" if detail else f"{code}: {field}: not stated in the text"


# ----------------------------------------------------------------------------- main
def postprocess(obj, text: str):
    """Return a new extraction dict (or obj unchanged if it is not a dict). See module docstring."""
    if not isinstance(obj, dict):
        return obj
    text = text or ""
    tu, tn, ta = text.upper(), _norm_text(text), _alnum(text)
    fin = obj.get("fields") if isinstance(obj.get("fields"), dict) else {}
    f = {k: _clean(copy.deepcopy(fin.get(k))) for k in FIELD_KEYS}
    added = []  # (code, field, detail) produced by R3/R4

    # parties (R1, R4 accounts)
    for pk in ("ordering_customer", "beneficiary"):
        p = f[pk]
        if isinstance(p, str):
            p = {"account": None, "name": p, "address": None}
        if isinstance(p, dict):
            p = {k: _clean(p.get(k)) for k in PARTY_KEYS}
            for k in PARTY_KEYS:
                if p[k] is not None and not isinstance(p[k], str):
                    p[k] = str(p[k])
            if p["account"] and _alnum(p["account"]) not in ta:
                p["account"] = None
            f[pk] = p if any(p.values()) else None
        elif p is not None:
            f[pk] = None

    # sender_reference / remittance_info
    for k in ("sender_reference", "remittance_info"):
        if f[k] is not None and not isinstance(f[k], str):
            f[k] = str(f[k])
    if f["sender_reference"] and _norm_text(f["sender_reference"]) not in tn:
        f["sender_reference"] = None

    # bank_operation_code
    if f["bank_operation_code"] is not None:
        c = str(f["bank_operation_code"]).upper()
        f["bank_operation_code"] = c if c in BANK_OP_CODES and _token_in(c, tu) else None

    # currency
    c = f["currency"]
    if c is not None:
        c = str(c).strip()
        cu = c.upper()
        c = dict(SYMBOLS).get(c, CCY_WORDS.get(cu, cu))
        if c is None or c not in ISO4217:
            if c and _token_in(c, tu):
                added.append(("INVALID_CURRENCY", "currency", f"{c} is not an ISO-4217 code"))
            c = None
        elif not _ccy_grounded(c, tu):
            c = None
        f["currency"] = c

    # amount
    a = f["amount"]
    if a is not None:
        a = parse_number(a) if isinstance(a, str) else (float(a) if isinstance(a, (int, float)) and not isinstance(a, bool) else None)
    if a is not None:
        # whole text, plus each CSV-like cell (a comma after a date must not glue it to the next amount)
        cands = text_amounts(text) + [c for cell in re.split(r'[,;\t\n"|]', text) for c in text_amounts(cell)]
        pos = [x for x, _, neg, tol in cands if not neg and abs(x - a) <= tol]
        negs = [x for x, _, neg, tol in cands if neg and abs(abs(x) - abs(a)) <= tol]
        if a <= 0:
            if any(abs(x) < 0.005 for x, _, _, _ in cands) or negs:
                added.append(("NON_POSITIVE_AMOUNT", "amount", f"{a:g} is not a positive amount"))
            a = None
        elif pos:
            pass
        elif negs:
            added.append(("NON_POSITIVE_AMOUNT", "amount", f"-{a:g} is not a positive amount"))
            a = None
        else:
            d = _digits_of(a)
            same = {x for x, digits, neg, _ in cands if digits == d and not neg and x > 0}
            a = same.pop() if len(same) == 1 else None
    f["amount"] = a

    # value_date
    dates, bad_dates = text_dates(text)
    v = f["value_date"]
    if v is not None:
        iso, impossible = parse_date_value(str(v))
        if iso is None:
            if impossible and any(_alnum(b) == _alnum(str(v)) for b in bad_dates):
                added.append(("INVALID_DATE", "value_date", f"{v} is not a real calendar date"))
            elif impossible and bad_dates:
                added.append(("INVALID_DATE", "value_date", f"{bad_dates[0]} is not a real calendar date"))
        elif dt.date.fromisoformat(iso) not in dates:
            iso = None
        f["value_date"] = iso

    # BICs
    for k in ("ordering_institution", "account_with_institution"):
        b = f[k]
        if b is None:
            continue
        b = str(b).strip().upper()
        if not BIC_RE.match(b):
            if _token_in(b, tu):
                added.append(("INVALID_BIC", k, f"{b} is not a valid BIC"))
            b = None
        elif not (_token_in(b, tu) or (len(b) == 8 and _token_in(b + "XXX", tu))):
            b = None
        f[k] = b

    # charges
    ch = f["charges"]
    if ch is not None:
        cu = re.sub(r"\s+", " ", str(ch)).strip().upper()
        mapped = CHARGE_MAP.get(re.sub(r"\s*\(.*\)$", "", cu))
        if mapped is None and not re.search(r"[+/&]", cu):
            hits = {CHARGE_MAP[t] for t in re.findall(r"[A-Z]+", cu) if t in ("OUR", "OURS", "SHA", "BEN")}
            mapped = hits.pop() if len(hits) == 1 else None
        if mapped is None:
            if _token_in(cu, tu):
                added.append(("INVALID_CHARGES", "charges", f"{ch} is not OUR/SHA/BEN"))
        f["charges"] = mapped

    # R5: model errors
    def valid_value(field, val):
        if val is None:
            return False
        if field == "currency":
            return val in ISO4217
        if field == "value_date":
            return parse_date_value(val)[0] is not None and not parse_date_value(val)[1]
        if field == "amount":
            return isinstance(val, (int, float)) and val > 0
        if field in ("ordering_institution", "account_with_institution"):
            return bool(BIC_RE.match(val))
        if field == "charges":
            return val in ("OUR", "SHA", "BEN")
        return True

    errs, seen = [], set()
    raw_errors = obj.get("errors") if isinstance(obj.get("errors"), list) else []
    for e in [parse_error(x) for x in raw_errors] + added:
        if e is None:
            continue
        code, field, detail = e
        if (code, field) in seen:
            continue
        if f.get(field) is not None:  # a flagged field must be null; the value wins if it survived R2-R4
            continue
        if code != "MISSING_REQUIRED":
            quoted = detail.split(" ")[0].strip("'\"") if detail else ""
            restored = _restore(field, quoted, f, tu, dates)
            if restored is not None:
                f[field] = restored
                continue
            if quoted and valid_value(field, quoted):
                continue  # the model called a valid value impossible; it is not in the text as such
        seen.add((code, field))
        errs.append(_fmt_error(code, field, detail))

    # R6: required fields with nothing at all in the text
    flagged = {fld for _, fld in seen}
    if f["value_date"] is None and "value_date" not in flagged:
        if bad_dates:
            errs.append(_fmt_error("INVALID_DATE", "value_date", f"{bad_dates[0]} is not a real calendar date"))
        elif not dates:
            errs.append(_fmt_error("MISSING_REQUIRED", "value_date", ""))
    if f["charges"] is None and "charges" not in flagged and not (
            re.search(r"(?i)charg|chrg|\bfees?\b|\bcosts?\b|bearer|shared", text)
            or re.search(r"\b(OUR|SHA|BEN)\b", text)):
        errs.append(_fmt_error("MISSING_REQUIRED", "charges", ""))
    return {"fields": f, "errors": errs}


def _restore(field, quoted, f, tu, dates):
    """An error that quotes a valid value which is in the text: restore the value instead."""
    if not quoted:
        return None
    q = quoted.strip(".,;")
    if field == "currency" and q.upper() in ISO4217 and _ccy_grounded(q.upper(), tu):
        return q.upper()
    if field in ("ordering_institution", "account_with_institution"):
        other = "account_with_institution" if field == "ordering_institution" else "ordering_institution"
        if BIC_RE.match(q.upper()) and _token_in(q, tu) and f.get(other) != q.upper():
            return q.upper()
    if field == "charges" and q.upper() in ("OUR", "SHA", "BEN") and _token_in(q, tu):
        return q.upper()
    return None
