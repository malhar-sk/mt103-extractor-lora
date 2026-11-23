"""Error analysis on top of cached eval outputs (no model calls).

1. Per-field accuracy and a taxonomy of field errors (what kind of mistake), with one example each.
2. Error-flag analysis per error code: for every gold error (CODE:field), did the model flag it, flag it
   with the wrong code/field, silently "repair" the impossible value into a plausible one, copy the
   impossible value through (the API's pydantic validation would still catch that), or just leave it null?

Usage (from the repo root):
  python evals/analyze.py --data data/test.jsonl --runs base-qwen2.5-0.5b-fewshot ft-qwen2.5-0.5b-lora \
      api-claude-haiku-5-5-fewshot --tag test
Writes results/analysis_<tag>.json and results/analysis_<tag>.md.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.run_eval import SCORED_FIELDS, _get, _norm, field_match, load_jsonl, score_one  # noqa: E402
from prompts import FEWSHOT_IDS, chat_messages, ft_messages, parse_output  # noqa: E402
from schema import ERROR_CODES, MT103Fields, error_keys  # noqa: E402

SIBLING = {  # values that legitimately live in another field and get confused with this one
    "ordering_institution": ["account_with_institution"],
    "account_with_institution": ["ordering_institution"],
    "ordering_customer.account": ["beneficiary.account", "ordering_customer.address"],
    "ordering_customer.name": ["beneficiary.name"],
    "ordering_customer.address": ["ordering_customer.account", "beneficiary.address"],
    "beneficiary.account": ["ordering_customer.account", "beneficiary.address"],
    "beneficiary.name": ["ordering_customer.name"],
    "beneficiary.address": ["beneficiary.account", "ordering_customer.address"],
    "sender_reference": ["remittance_info"],
    "remittance_info": ["sender_reference"],
}

ERROR_TYPE_HELP = {
    "invalid_json": "output was not parseable JSON, so every present field is lost",
    "repaired_impossible_value": "the source value was impossible (gold null + error), the model output a plausible value instead of null",
    "hallucinated_absent_field": "the field is not in the text (gold null), the model invented a value",
    "missed_present_value": "the value is in the text, the model returned null",
    "value_from_other_field": "the model put another field's value here (e.g. 52A/57A BICs swapped, account in address)",
    "date_day_month_swap": "day and month swapped",
    "date_wrong_year": "right day and month, wrong year",
    "date_misparsed": "date wrong in another way",
    "amount_scale": "amount off by a power of ten (thousands/decimal separator misread)",
    "amount_misparsed": "amount wrong in another way",
    "enum_wrong_value": "wrong code (charges / op code / currency)",
    "string_partial": "right value with text missing or extra text attached",
    "string_wrong_value": "a different string altogether",
}


def _cache_key(backend, model, msgs):
    return hashlib.sha1(json.dumps([backend, model, msgs], ensure_ascii=False).encode()).hexdigest()


def load_predictions(name: str, data: list[dict], results: Path, data_path: Path) -> dict:
    """id -> parsed prediction (or None), looked up with the same cache key run_eval used."""
    summ = json.loads((results / f"eval_{name}.json").read_text())["summary"]
    cache = {r["key"]: r for r in load_jsonl(results / "cache" / f"{name}.jsonl")}
    fewshot = []
    if summ.get("fewshot"):
        train = {r["id"]: r for r in load_jsonl(data_path.parent / "train.jsonl")}
        fewshot = [train[i] for i in FEWSHOT_IDS]
    out = {}
    for ex in data:
        msgs = ft_messages(ex["text"]) if summ.get("prompt") == "ft" else chat_messages(ex["text"], fewshot)
        hit = cache.get(_cache_key(summ["backend"], summ["model"], msgs))
        out[ex["id"]] = parse_output(hit["raw"]) if hit else None
    return out


def _date_kind(gold: str, pred) -> str:
    try:
        g, p = dt.date.fromisoformat(gold), dt.date.fromisoformat(str(pred))
    except (TypeError, ValueError):
        return "date_misparsed"
    if (p.day, p.month) == (g.month, g.day) and p.year == g.year:
        return "date_day_month_swap"
    if (p.day, p.month) == (g.day, g.month):
        return "date_wrong_year"
    return "date_misparsed"


def _amount_kind(gold: float, pred) -> str:
    try:
        ratio = float(pred) / float(gold)
    except (TypeError, ValueError, ZeroDivisionError):
        return "amount_misparsed"
    for k in (10, 100, 1000, 1e5):
        if abs(ratio - k) < 1e-6 * k or abs(ratio - 1 / k) < 1e-6 / k:
            return "amount_scale"
    return "amount_misparsed"


def classify_field_error(path: str, gold_fields: dict, gold_errors: list, pred_obj) -> str:
    """Type of mistake for a field that did not match. Caller guarantees it is wrong."""
    if not isinstance(pred_obj, dict):
        return "invalid_json"
    pf = pred_obj.get("fields") if isinstance(pred_obj.get("fields"), dict) else {}
    g, p = _get(gold_fields, path), _get(pf, path)
    top = path.split(".")[0]
    gold_err_fields = {k.split(":", 1)[1] for k in error_keys(gold_errors)}
    if p is None or (isinstance(p, str) and not p.strip()):
        return "missed_present_value"
    if g is None and top in gold_err_fields:
        return "repaired_impossible_value"
    for sib in SIBLING.get(path, []):
        sv = _get(gold_fields, sib)
        if sv is not None and field_match(path, sv, p):
            return "value_from_other_field"
    if g is None:
        return "hallucinated_absent_field"
    if path == "value_date":
        return _date_kind(g, p)
    if path == "amount":
        return _amount_kind(g, p)
    if path in ("charges", "bank_operation_code", "currency"):
        return "enum_wrong_value"
    ng, np_ = _norm(g), _norm(str(p))
    if ng and np_ and (ng in np_ or np_ in ng):
        return "string_partial"
    return "string_wrong_value"


def _field_invalid(top: str, value) -> bool:
    """True if pydantic would reject this value for this field (the API would report it)."""
    try:
        MT103Fields.model_validate({top: value})
        return False
    except Exception:
        return True


def flag_outcome(gold_key: str, gold_fields: dict, pred_obj) -> str:
    code, field = gold_key.split(":", 1)
    if not isinstance(pred_obj, dict):
        return "invalid_json"
    pk = error_keys(pred_obj.get("errors"))
    if gold_key in pk:
        return "flagged_correctly"
    if any(k.split(":", 1)[1] == field for k in pk) or any(k.split(":", 1)[0] == code for k in pk):
        return "flagged_wrong_code_or_field"
    pf = pred_obj.get("fields") if isinstance(pred_obj.get("fields"), dict) else {}
    pv = pf.get(field)
    if pv is None or pv == "":
        return "not_flagged_field_left_null"
    if code != "MISSING_REQUIRED" and _field_invalid(field, pv):
        return "not_flagged_copied_invalid_value"  # pydantic validation in the API would still catch it
    return "not_flagged_repaired_or_invented"      # silent: a plausible value, nothing downstream catches it


OUTCOMES = ["flagged_correctly", "flagged_wrong_code_or_field", "not_flagged_repaired_or_invented",
            "not_flagged_copied_invalid_value", "not_flagged_field_left_null", "invalid_json"]


def analyse_run(data: list[dict], preds: dict) -> dict:
    per_field_ok = collections.Counter()
    err_types = collections.Counter()
    err_by_field = collections.defaultdict(collections.Counter)
    examples = {}
    flag = {c: {"gold": 0, "outcomes": collections.Counter(), "pred": 0, "pred_tp": 0} for c in ERROR_CODES}
    false_alarm_examples = 0
    clean_examples = 0
    for ex in data:
        gold, pred = ex["target"], preds.get(ex["id"])
        s = score_one(gold, pred)
        for path, ok in s["fields"].items():
            if ok:
                per_field_ok[path] += 1
                continue
            t = classify_field_error(path, gold["fields"], gold["errors"], pred)
            err_types[t] += 1
            err_by_field[path][t] += 1
            if t not in examples:
                pf = pred.get("fields", {}) if isinstance(pred, dict) and isinstance(pred.get("fields"), dict) else {}
                examples[t] = {"id": ex["id"], "field": path, "gold": _get(gold["fields"], path),
                               "pred": _get(pf, path) if isinstance(pred, dict) else "<invalid JSON>",
                               "gold_errors": gold["errors"], "text": ex["text"][:300]}
        gk = error_keys(gold["errors"])
        pk = error_keys(pred.get("errors")) if isinstance(pred, dict) else set()
        for k in gk:
            c = k.split(":", 1)[0]
            if c in flag:
                flag[c]["gold"] += 1
                flag[c]["outcomes"][flag_outcome(k, gold["fields"], pred)] += 1
        for k in pk:
            c = k.split(":", 1)[0]
            if c in flag:
                flag[c]["pred"] += 1
                flag[c]["pred_tp"] += k in gk
        if not gk:
            clean_examples += 1
            false_alarm_examples += bool(pk)
    n = len(data)
    by_style = collections.defaultdict(list)
    for ex in data:
        if "ood_style" in ex:
            sc = score_one(ex["target"], preds.get(ex["id"]))["fields"]
            by_style[ex["ood_style"]].append(sum(sc.values()) / len(sc))
    return {
        "n": n,
        "field_accuracy_macro_by_style": {k: (len(v), sum(v) / len(v)) for k, v in sorted(by_style.items())},
        "field_accuracy": {p: per_field_ok[p] / n for p in SCORED_FIELDS},
        "field_error_types": dict(err_types.most_common()),
        "field_error_types_by_field": {p: dict(c.most_common()) for p, c in err_by_field.items()},
        "field_error_examples": examples,
        "error_flags": {c: {"gold": v["gold"], "pred": v["pred"],
                            "recall": (v["outcomes"]["flagged_correctly"] / v["gold"]) if v["gold"] else None,
                            "precision": (v["pred_tp"] / v["pred"]) if v["pred"] else None,
                            "outcomes": dict(v["outcomes"])} for c, v in flag.items()},
        "false_alarm_rate_on_clean_examples": (false_alarm_examples / clean_examples) if clean_examples else None,
        "clean_examples": clean_examples,
    }


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_md(tag: str, data_path: str, out: dict, cmd: str) -> str:
    runs = out["runs"]
    names = list(runs)
    L = [f"# Error analysis: `{tag}` ({data_path}, n = {out['n']}, SYNTHETIC)", "",
         f"Generated {out['date']} by `{cmd}` from cached model outputs (no new model calls).", "",
         "## Per-field accuracy", "", "| Field | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for p in SCORED_FIELDS:
        L.append(f"| {p} | " + " | ".join(pct(runs[n]["field_accuracy"][p]) for n in names) + " |")
    for n in names:
        r = runs[n]
        tot = sum(r["field_error_types"].values())
        L += ["", f"## Field error types: {n} ({tot} wrong field values out of {out['n'] * len(SCORED_FIELDS)})", "",
              "| Error type | Count | Share | Worst fields | Meaning |", "|---|---|---|---|---|"]
        for t, c in r["field_error_types"].items():
            worst = sorted(((f, d.get(t, 0)) for f, d in r["field_error_types_by_field"].items()), key=lambda x: -x[1])
            worst = ", ".join(f"{f} ({k})" for f, k in worst[:3] if k)
            L.append(f"| {t} | {c} | {pct(c / tot if tot else None)} | {worst} | {ERROR_TYPE_HELP.get(t, '')} |")
    L += ["", "## Error-flag analysis by error code", "",
          "Recall = gold errors flagged with the right CODE:field. Outcomes for each gold error: "
          "**ok** flagged correctly, **wrong** flagged with another code/field, **repaired** no flag and a plausible "
          "value invented (silent, nothing downstream catches it), **copied** no flag but the impossible value was "
          "copied through (the API's pydantic validation still catches it), **null** no flag, field left null, "
          "**json** invalid JSON.", ""]
    L += ["| Code | Model | Gold n | Recall | Pred n | Precision | ok / wrong / repaired / copied / null / json |",
          "|---|---|---|---|---|---|---|"]
    for c in ERROR_CODES:
        for n in names:
            e = runs[n]["error_flags"][c]
            o = e["outcomes"]
            L.append(f"| {c} | {n} | {e['gold']} | {pct(e['recall'])} | {e['pred']} | {pct(e['precision'])} | "
                     + " / ".join(str(o.get(k, 0)) for k in OUTCOMES) + " |")
    styles = sorted({k for n in names for k in runs[n].get("field_accuracy_macro_by_style", {})})
    if styles:
        L += ["", "## Field accuracy (macro) by input style", "", "| Style | n | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
        for st in styles:
            cnt = runs[names[0]]["field_accuracy_macro_by_style"][st][0]
            L.append(f"| {st} | {cnt} | " + " | ".join(pct(runs[n]["field_accuracy_macro_by_style"][st][1]) for n in names) + " |")
    L += ["", "False alarms (any error flagged on an example whose gold `errors` is empty):", ""]
    for n in names:
        r = runs[n]
        L.append(f"- {n}: {pct(r['false_alarm_rate_on_clean_examples'])} of {r['clean_examples']} clean examples")
    L += ["", "## One example per error type (first occurrence)", ""]
    for n in names:
        L += [f"### {n}", ""]
        for t in list(runs[n]["field_error_types"])[:5]:
            e = runs[n]["field_error_examples"][t]
            L.append(f"- **{t}** `{e['id']}` field `{e['field']}`: gold `{json.dumps(e['gold'], ensure_ascii=False)}`, "
                     f"predicted `{json.dumps(e['pred'], ensure_ascii=False)}`"
                     + (f", gold errors {e['gold_errors']}" if e["gold_errors"] else ""))
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    results, data_path = Path(a.results), Path(a.data)
    data = load_jsonl(data_path)
    out = {"tag": a.tag, "data": str(a.data), "n": len(data), "date": dt.date.today().isoformat(), "runs": {}}
    for name in a.runs:
        preds = load_predictions(name, data, results, data_path)
        missing = sum(1 for ex in data if preds.get(ex["id"]) is None)
        r = analyse_run(data, preds)
        r["unparsed_or_missing"] = missing
        # sanity: must agree with the eval script's own numbers
        summ = json.loads((results / f"eval_{name}.json").read_text())["summary"]
        for p in SCORED_FIELDS:
            assert abs(r["field_accuracy"][p] - summ["field_accuracy"][p]) < 1e-9, (name, p)
        out["runs"][name] = r
    cmd = "python evals/analyze.py " + " ".join(argv if argv is not None else sys.argv[1:])
    (results / f"analysis_{a.tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    md = render_md(a.tag, a.data, out, cmd)
    (results / f"analysis_{a.tag}.md").write_text(md)
    print(md)
    return out


if __name__ == "__main__":
    main()
