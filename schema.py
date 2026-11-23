"""Pydantic schema for the subset of SWIFT MT103 fields used by this project.

Fields (tag -> key):
  20  sender_reference        23B bank_operation_code
  32A value_date / currency / amount
  50K ordering_customer       52A ordering_institution (BIC)
  57A account_with_institution (BIC)
  59  beneficiary             70  remittance_info
  71A charges (OUR/SHA/BEN)

An extraction result is {"fields": MT103Fields-like dict, "errors": [str, ...]}.
Fields that are missing or impossible in the source text are null, and impossible
values are reported in `errors` (never invented).
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ISO 4217 active alphabetic codes (complete list as of 2024, minus funds/precious metals).
ISO4217 = frozenset("""
AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL BSD BTN BWP BYN
BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS
GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW
KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MYR MZN NAD
NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD
SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VES
VND VUV WST XAF XCD XOF XPF YER ZAR ZMW ZWL
""".split())

BIC_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")
REF_RE = re.compile(r"^[A-Za-z0-9/\-?:().,'+ ]{1,16}$")
BANK_OP_CODES = ("CRED", "CRTS", "SPAY", "SPRI", "SSTD")
FIELD_KEYS = (
    "sender_reference", "bank_operation_code", "value_date", "currency", "amount",
    "ordering_customer", "ordering_institution", "account_with_institution",
    "beneficiary", "remittance_info", "charges",
)
REQUIRED_KEYS = ("sender_reference", "value_date", "currency", "amount",
                 "ordering_customer", "beneficiary", "charges")
# Machine-readable error codes. Each `errors` entry is "<CODE>: <field>: <detail>", e.g.
# "INVALID_CURRENCY: currency: USX is not ISO-4217". Scoring uses (CODE, field) only.
ERROR_CODES = ("INVALID_CURRENCY", "INVALID_DATE", "INVALID_BIC", "NON_POSITIVE_AMOUNT",
               "INVALID_CHARGES", "MISSING_REQUIRED")


class Party(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account: Optional[str] = None
    name: str = Field(min_length=1, max_length=140)
    address: Optional[str] = None


class MT103Fields(BaseModel):
    """All fields optional: an extractor may legitimately leave a field null."""
    model_config = ConfigDict(extra="forbid")

    sender_reference: Optional[str] = Field(None, description=":20:")
    bank_operation_code: Optional[Literal["CRED", "CRTS", "SPAY", "SPRI", "SSTD"]] = Field(None, description=":23B:")
    value_date: Optional[str] = Field(None, description=":32A: date, ISO YYYY-MM-DD")
    currency: Optional[str] = Field(None, description=":32A: ISO-4217")
    amount: Optional[float] = Field(None, description=":32A: amount, > 0")
    ordering_customer: Optional[Party] = Field(None, description=":50K:")
    ordering_institution: Optional[str] = Field(None, description=":52A: BIC")
    account_with_institution: Optional[str] = Field(None, description=":57A: BIC")
    beneficiary: Optional[Party] = Field(None, description=":59:")
    remittance_info: Optional[str] = Field(None, max_length=140, description=":70:")
    charges: Optional[Literal["OUR", "SHA", "BEN"]] = Field(None, description=":71A:")

    @field_validator("sender_reference")
    @classmethod
    def _ref(cls, v):
        if v is not None and not REF_RE.match(v):
            raise ValueError("sender_reference must be 1-16 SWIFT x-chars")
        return v

    @field_validator("value_date")
    @classmethod
    def _date(cls, v):
        if v is not None:
            dt.date.fromisoformat(v)  # raises on bad format / impossible date
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                raise ValueError("value_date must be YYYY-MM-DD")
        return v

    @field_validator("currency")
    @classmethod
    def _ccy(cls, v):
        if v is not None and v not in ISO4217:
            raise ValueError(f"{v} is not an ISO-4217 currency")
        return v

    @field_validator("amount")
    @classmethod
    def _amt(cls, v):
        if v is not None and not v > 0:
            raise ValueError("amount must be positive")
        return v

    @field_validator("ordering_institution", "account_with_institution")
    @classmethod
    def _bic(cls, v):
        if v is not None and not BIC_RE.match(v):
            raise ValueError(f"{v} is not a valid BIC")
        return v


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fields: MT103Fields
    errors: list[str] = Field(default_factory=list)

    @field_validator("errors")
    @classmethod
    def _codes(cls, v):
        for e in v:
            if e.split(":", 1)[0].strip() not in ERROR_CODES:
                raise ValueError(f"unknown error code in {e!r}")
        return v


def error_keys(errors) -> set[str]:
    """Normalise an errors list to a set of "CODE:field" keys (detail text ignored)."""
    out = set()
    if not isinstance(errors, list):
        return out
    for e in errors:
        if not isinstance(e, str):
            continue
        parts = [x.strip() for x in e.split(":")]
        code = parts[0].upper().replace(" ", "_")
        field = parts[1].split()[0].lower() if len(parts) > 1 and parts[1] else ""
        out.add(f"{code}:{field}")
    return out
