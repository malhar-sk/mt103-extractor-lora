"""Prompt construction and output parsing shared by training, eval and serving."""
from __future__ import annotations

import json
import re

SYSTEM = """You convert messy payment instructions (emails, chats, partial SWIFT text, tickets) into SWIFT MT103 JSON.
Return ONLY one JSON object, no prose, with exactly this shape:
{"fields":{"sender_reference":str|null,"bank_operation_code":"CRED"|"CRTS"|"SPAY"|"SPRI"|"SSTD"|null,"value_date":"YYYY-MM-DD"|null,"currency":ISO-4217 str|null,"amount":number|null,"ordering_customer":{"account":str|null,"name":str,"address":str|null}|null,"ordering_institution":BIC|null,"account_with_institution":BIC|null,"beneficiary":{"account":str|null,"name":str,"address":str|null}|null,"remittance_info":str|null,"charges":"OUR"|"SHA"|"BEN"|null},"errors":[str]}
Rules:
- 20=sender_reference, 23B=bank_operation_code, 32A=value_date+currency+amount, 50K=ordering_customer, 52A=ordering_institution (sender's bank BIC), 57A=account_with_institution (beneficiary's bank BIC), 59=beneficiary, 70=remittance_info, 71A=charges.
- Numeric dates are day-first (dd/mm/yyyy). SWIFT 6-digit dates are YYMMDD. Output dates as YYYY-MM-DD.
- Amounts: plain number. Indian grouping 1,25,000.50 = 125000.5; "2.5 lakh" = 250000; European 1.234,56 = 1234.56. Symbols: $=USD, €=EUR, £=GBP, Rs/₹=INR, ¥=JPY.
- Charges: ours/we pay=OUR, shared=SHA, beneficiary pays=BEN.
- Copy names, accounts, addresses and references exactly as written. A field not in the text is null. Never invent values.
- If a value is impossible, set the field to null and add an error "CODE: field: detail". Codes: INVALID_CURRENCY (not ISO-4217), INVALID_DATE (not a real date), INVALID_BIC (not 8 or 11 chars AAAABBCC[XXX]), NON_POSITIVE_AMOUNT (zero or negative), INVALID_CHARGES (not OUR/SHA/BEN), MISSING_REQUIRED (value_date, beneficiary or charges not stated).
- errors is [] when everything is fine."""

# Fixed few-shot ids from the TRAIN split (email, chat with Indian amount format, partial SWIFT, ticket with an error).
FEWSHOT_IDS = ("syn-00004", "syn-00012", "syn-00000", "syn-00005")


def target_str(target: dict) -> str:
    """Canonical compact JSON for a target (what the fine-tuned model is trained to emit)."""
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def chat_messages(text: str, fewshot: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in fewshot or []:
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": target_str(ex["target"])})
    msgs.append({"role": "user", "content": text})
    return msgs


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_output(raw: str):
    """Return (obj or None). Accepts fenced code blocks and leading/trailing prose."""
    if raw is None:
        return None
    s = raw.strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:
                    return None
    return None


# Short system prompt used for the FINE-TUNED model (training, Ollama Modelfile, eval with
# --prompt ft). The format and error rules are learned from data, so the prompt stays tiny:
# that is what makes the fine-tuned model much cheaper per request than few-shot prompting.
FT_SYSTEM = "Extract the payment as MT103 JSON {fields, errors}. Null for absent fields; flag impossible values in errors."


def ft_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": FT_SYSTEM}, {"role": "user", "content": text}]
