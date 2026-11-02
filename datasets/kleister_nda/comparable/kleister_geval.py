"""COMPARABLE view for kleister_nda: a stdlib re-implementation of the Kleister-NDA GEval headline metric.

Official metric (raw/repo/config.txt at commit c2c7bf06): `--metric MultiLabel-F1:uN<F1(UC)>`, i.e. GEval
MultiLabel-F1 on upper-cased `key=value` labels. GEval's own description (src/GEval/MetricsMeta.hs, saved as
raw/pages/geval_MetricsMeta.hs): "Counts for precision and recall are calculated for the whole test set, in other
words it is a micro-average". Per-key rows mirror config.txt's m<effective_date=\\S+\\s*> ... selections.

NOT the GEval binary: we do not execute downloaded binaries. Two behaviours are our choices and are UNKNOWN in
GEval: (1) duplicate labels within one line are de-duplicated; (2) the per-key filter is case-insensitive.
Verified only by --selftest (expected lines scored against themselves = 1.0; empty output = 0.0).

Output adapter (ours, pre-registered, the same for every arm), harness prediction -> Kleister line:
  effective_date  parse_date(dayfirst=False) -> YYYY-MM-DD (unparseable -> raw text, which will miss)
  jurisdiction    value as returned
  party           every element of the `party` list; an arm without list output (Sage 2) uses its single
                  `party_any` answer instead (recall is then capped; report that row separately)
  term            the arm's OWN unit, never converted: P2Y / "2 years" -> 2_years, "24 months" -> 24_months,
                  singular for 1 ("1_year"). Kleister keeps the document's original unit, so an arm that returns
                  months (Clairic with unit=months) misses "N_years" golds here; the TYPED view is unit-agnostic.
  all values      commas removed (Kleister gold has none: 0/334 values), whitespace collapsed,
                  spaces and colons -> "_" (README "Normalization").

Usage:
  python -I comparable/kleister_geval.py --selftest
  python -I comparable/kleister_geval.py --preds-dir ../../runs/RUN/preds/ARM/kleister_nda [--exclude-holdout]
"""
from __future__ import annotations

import argparse, json, re, sys
from collections import Counter
from pathlib import Path

DS = Path(__file__).resolve().parents[1]
EXB = DS.parents[1]
from extract_bench.score import parse_date  # noqa: E402

KEYS = ("effective_date", "jurisdiction", "party", "term")
UNITS = {"y": "year", "year": "year", "yr": "year", "m": "month", "mo": "month", "month": "month",
         "w": "week", "wk": "week", "week": "week", "d": "day", "day": "day"}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "eighteen": 18, "twenty": 20, "twenty-four": 24, "thirty": 30,
         "thirty-six": 36, "sixty": 60, "ninety": 90}


def kval(s) -> str:
    s = re.sub(r"\s+", " ", str(s).replace(",", " ")).strip()
    return re.sub(r"[ :]", "_", s)


def kterm(v) -> str | None:
    """Duration -> '{n}_{units}' in the unit the arm used. None if unparseable."""
    if v is None:
        return None
    s = str(v).strip().casefold()
    m = re.fullmatch(r"p(\d+(?:\.\d+)?)([ymwd])", s)                       # single-unit ISO 8601
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)?)\s*(?:\(\s*\d+\s*\)\s*)?(years?|yrs?|months?|mos?|weeks?|wks?|days?)\b", s)
    if not m:
        return None
    n = m[1]
    n = WORDS.get(n) if not n[0].isdigit() else float(n)
    if n is None:
        return None
    unit = UNITS[m[2].rstrip("s") if m[2] not in ("m", "d", "w", "y") else m[2]]
    n = int(n) if float(n).is_integer() else n
    return f"{n}_{unit}" + ("" if n == 1 else "s")


def adapter_line(item: dict, preds: dict) -> str:
    def val(name):
        p = preds.get(name) or {}
        return None if p.get("unsupported") else p.get("value")
    labels = []
    ed = val("effective_date")
    if ed not in (None, ""):
        labels.append("effective_date=" + (parse_date(ed, dayfirst=False) or kval(ed)))
    ju = val("jurisdiction")
    if ju not in (None, ""):
        labels.append("jurisdiction=" + kval(ju))
    parties = val("party")
    if parties in (None, [], "") or (preds.get("party") or {}).get("unsupported"):
        pa = val("party_any")
        parties = [pa] if pa not in (None, "") else []
    if isinstance(parties, str):
        parties = [parties]
    labels += ["party=" + kval(p) for p in parties if p not in (None, "")]
    te = kterm(val("term"))
    if te:
        labels.append("term=" + te)
    return " ".join(sorted(labels))


def geval_f1(expected: list[str], out: list[str], key: str | None = None, upper: bool = True) -> dict:
    tp = n_exp = n_out = 0
    for e, o in zip(expected, out, strict=True):
        def labs(line):
            ls = [x for x in line.split(" ") if x]
            if key:
                ls = [x for x in ls if x.casefold().startswith(key + "=")]
            return set(x.upper() if upper else x for x in ls)
        E, O = labs(e), labs(o)
        tp += len(E & O); n_exp += len(E); n_out += len(O)
    p = tp / n_out if n_out else 0.0
    r = tp / n_exp if n_exp else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"F1": round(100 * f, 2), "P": round(100 * p, 2), "R": round(100 * r, 2), "tp": tp, "n_expected": n_exp,
            "n_output": n_out}


def report(items, out_lines) -> dict:
    exp = [it["meta"]["kleister_expected"] for it in items]
    res = {"n_docs": len(items), "ALL_UC": geval_f1(exp, out_lines), "ALL_truecase": geval_f1(exp, out_lines, upper=False)}
    for k in KEYS:
        res[k] = geval_f1(exp, out_lines, key=k)
    return res


def load_items(exclude_holdout=False):
    its = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]
    return [it for it in its if not (exclude_holdout and it["meta"]["holdout"])]


def selftest() -> dict:
    items = load_items()
    exp = [it["meta"]["kleister_expected"] for it in items]
    out = {"expected_vs_itself": report(items, exp)["ALL_UC"]["F1"],
           "empty_output": report(items, [""] * len(items))["ALL_UC"]["F1"]}
    # gold re-expressed as harness predictions (ISO dates, ISO durations, surface-form parties with commas)
    def gold_preds(it, surface: bool):
        pr = {}
        for f in it["fields"]:
            g = f["gold"]
            if f["name"] == "party":
                g = (g[-1] if surface else g[0]) if g and isinstance(g[0], list) else g
            elif isinstance(g, list):
                g = g[-1] if surface else g[0]
            pr[f["name"]] = {"value": g}
        return pr
    for name, surface in (("gold_kleister_form_via_adapter", False), ("gold_surface_form_via_adapter", True)):
        out[name] = report(items, [adapter_line(it, gold_preds(it, surface)) for it in items])["ALL_UC"]["F1"]
    sage_like = []
    for it in items:
        pr = gold_preds(it, False)
        pr["party"] = {"value": None, "unsupported": "sage2_no_list_type"}
        sage_like.append(adapter_line(it, pr))
    out["gold_but_single_party_any (Sage-like ceiling)"] = report(items, sage_like)
    months = []
    for it in items:
        pr = gold_preds(it, False)
        g = pr["term"]["value"]
        if g:
            n, u = int(g[1:-1]), g[-1]
            pr["term"]["value"] = f"{n * 12 if u == 'Y' else n} months"
        months.append(adapter_line(it, pr))
    out["gold_with_term_in_months (Clairic unit=months ceiling)"] = report(items, months)["term"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preds-dir", type=Path)
    ap.add_argument("--exclude-holdout", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=1))
        return
    items = load_items(a.exclude_holdout)
    lines, missing = [], 0
    for it in items:
        f = a.preds_dir / f"{it['id']}.json"
        if not f.exists():
            missing += 1
            lines.append("")
            continue
        lines.append(adapter_line(it, json.loads(f.read_text()).get("preds") or {}))
    res = report(items, lines)
    res["missing_pred_files"] = missing
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
