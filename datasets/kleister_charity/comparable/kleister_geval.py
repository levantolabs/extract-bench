"""COMPARABLE view for kleister_charity: a stdlib re-implementation of the Kleister-Charity GEval headline metric.

Official metric (raw/repo/config.txt at commit 2309f486): the first metric is `MultiLabel-F1:uN<F1(UC)>`, i.e.
GEval MultiLabel-F1 on upper-cased (`u` flag, raw/pages/geval_README.md "Metric flags") `key=value` labels,
micro-averaged over the whole set (raw/pages/geval_MetricsMeta.hs). Per-key rows mirror config.txt's
m<address__\\S+>, m<(income|spending)_...>, m<charity_name=...> ... selections.

NOT the GEval binary: downloaded binaries are not executed. Our choices, UNKNOWN in GEval: (1) duplicate labels in
one line are de-duplicated; (2) the per-key filter is case-insensitive. Verified only by --selftest.

Output adapter (ours, pre-registered, the same for every arm), harness prediction -> Kleister line:
  report_date        parse_date(dayfirst=True) -> YYYY-MM-DD (unparseable -> raw text, which will miss)
  income / spending  parse_number -> "%.2f" (e.g. 10348000.00); unparseable -> raw text
  text fields        value as returned: whitespace collapsed, spaces and colons -> "_" (README "Normalization").
                     Commas and trailing periods are KEPT (Kleister gold keeps them: "St. Andrew,_Enfield",
                     59/3,423 dev-0 values are names ending in "Ltd.").
  null / unsupported no label (an absent key in gold + no label = no penalty, as in GEval)

Aggregates: unweighted micro-F1 on the sampled docs; per stratum (long > 15k / short <= 15k o200k tokens); and a
dev-0-reweighted micro-F1 (each doc's tp / n_expected / n_output counts x meta.stratum_weight), which is the only
number to put next to a published row, and even then only as a reference (published rows are test-A).

Usage:
  python -I comparable/kleister_geval.py --selftest
  python -I comparable/kleister_geval.py --preds-dir ../../runs/RUN/preds/ARM/kleister_charity [--exclude-holdout]
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

DS = Path(__file__).resolve().parents[1]
EXB = DS.parents[1]
from extract_bench.score import parse_date, parse_number  # noqa: E402

KEYS = ("address__post_town", "address__postcode", "address__street_line", "charity_name", "charity_number",
        "income_annually_in_british_pounds", "report_date", "spending_annually_in_british_pounds")
MONEY = ("income_annually_in_british_pounds", "spending_annually_in_british_pounds")
# config.txt per-key selections (name -> label-prefix regex)
GROUPS = {"address": r"address__\S+", "money": r"(income|spending)_annually_in_british_pounds=\S+",
          "town": r"address__post_town=", "postcode": r"address__postcode=", "street": r"address__street_line=",
          "name": r"charity_name=", "number": r"charity_number=", "income": r"income_annually_in_british_pounds=",
          "spending": r"spending_annually_in_british_pounds=", "date": r"report_date="}


def kval(s) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return re.sub(r"[ :]", "_", s)


def adapter_line(item: dict, preds: dict) -> str:
    labels = []
    for k in KEYS:
        p = preds.get(k) or {}
        v = None if p.get("unsupported") else p.get("value")
        if isinstance(v, list):
            v = v[0] if v else None
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if k == "report_date":
            out = parse_date(v, dayfirst=True) or kval(v)
        elif k in MONEY:
            x = parse_number(v)
            out = f"{x:.2f}" if x is not None else kval(v)
        else:
            out = kval(v)
        labels.append(f"{k}={out}")
    return " ".join(sorted(labels))


def _labs(line: str, group: str | None, upper: bool) -> set:
    ls = [x for x in line.split(" ") if x]
    if group:
        ls = [x for x in ls if re.match(GROUPS[group], x, re.I)]
    return set(x.upper() if upper else x for x in ls)


def geval_f1(expected: list[str], out: list[str], group: str | None = None, upper: bool = True,
             weights: list[float] | None = None) -> dict:
    tp = n_exp = n_out = 0.0
    for i, (e, o) in enumerate(zip(expected, out, strict=True)):
        w = weights[i] if weights else 1.0
        E, O = _labs(e, group, upper), _labs(o, group, upper)
        tp += w * len(E & O); n_exp += w * len(E); n_out += w * len(O)
    p = tp / n_out if n_out else 0.0
    r = tp / n_exp if n_exp else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"F1": round(100 * f, 2), "P": round(100 * p, 2), "R": round(100 * r, 2), "tp": round(tp, 2),
            "n_expected": round(n_exp, 2), "n_output": round(n_out, 2)}


def report(items, out_lines) -> dict:
    exp = [it["meta"]["kleister_expected"] for it in items]
    w = [it["meta"]["stratum_weight"] for it in items]
    res = {"n_docs": len(items),
           "ALL_UC_dev0_reweighted": geval_f1(exp, out_lines, weights=w),
           "ALL_UC_sample": geval_f1(exp, out_lines),
           "ALL_truecase_sample": geval_f1(exp, out_lines, upper=False)}
    for s in ("long", "short"):
        idx = [i for i, it in enumerate(items) if it["meta"]["stratum"] == s]
        res[f"ALL_UC_{s}"] = geval_f1([exp[i] for i in idx], [out_lines[i] for i in idx])
    res["per_group_dev0_reweighted"] = {g: geval_f1(exp, out_lines, group=g, weights=w)["F1"] for g in GROUPS}
    return res


def load_items(exclude_holdout=False):
    its = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]
    return [it for it in its if not (exclude_holdout and it["meta"]["holdout"])]


def selftest() -> dict:
    items = load_items()
    exp = [it["meta"]["kleister_expected"] for it in items]
    out = {"expected_vs_itself": report(items, exp)["ALL_UC_dev0_reweighted"]["F1"],
           "empty_output": report(items, [""] * len(items))["ALL_UC_dev0_reweighted"]["F1"]}

    def gold_preds(it, pick):
        pr = {}
        for f in it["fields"]:
            g = f["gold"]
            if isinstance(g, list):
                g = g[0] if pick == "first" else g[-1]
            pr[f["name"]] = {"value": g}
        return pr
    r = report(items, [adapter_line(it, gold_preds(it, "first")) for it in items])
    out["typed_gold_first_value_via_adapter"] = {"ALL_UC_dev0_reweighted": r["ALL_UC_dev0_reweighted"]["F1"],
                                                 "per_group": r["per_group_dev0_reweighted"]}
    # verbatim-span ceiling on the primary (Textract) text: a perfect extractor that copies text. Per text field:
    # the Kleister value if it occurs (casefolded, whitespace-collapsed) in the text, else the most frequent
    # surface variant that occurs in the text, else the Kleister value (unreachable; counted as correct here).
    def norm_ws(s):
        return re.sub(r"\s+", " ", s).casefold()
    span = []
    for it in items:
        pr = gold_preds(it, "first")
        t = norm_ws(it["text"])
        for f in it["fields"]:
            if f["type"] == "text" and isinstance(f["gold"], list) and norm_ws(f["gold"][0]) not in t:
                hit = [v for v in f["gold"][1:] if norm_ws(v) in t]
                if hit:
                    pr[f["name"]] = {"value": hit[0]}
        span.append(adapter_line(it, pr))
    r = report(items, span)
    out["verbatim_span_ceiling_textract_via_adapter"] = {"ALL_UC_dev0_reweighted": r["ALL_UC_dev0_reweighted"]["F1"],
                                                         "per_group": r["per_group_dev0_reweighted"]}
    # money as Clairic-style floats and as written amounts; dates as UK text
    alt = []
    for it in items:
        pr = gold_preds(it, "first")
        for k in MONEY:
            if pr[k]["value"] is not None:
                pr[k]["value"] = float(pr[k]["value"])
        if pr["report_date"]["value"]:
            y, m, d = pr["report_date"]["value"].split("-")
            pr["report_date"]["value"] = f"{d}/{m}/{y}"
        alt.append(adapter_line(it, pr))
    out["gold_money_float_date_ddmmyyyy_via_adapter"] = report(items, alt)["ALL_UC_dev0_reweighted"]["F1"]
    # a 'null-gold hallucinator': gold everywhere plus a made-up value for every null-gold key (precision hit)
    hall = []
    for it in items:
        pr = gold_preds(it, "first")
        for f in it["fields"]:
            if f["gold"] is None:
                pr[f["name"]] = {"value": "1.00" if f["type"] == "money" else "ATLANTIS"}
        hall.append(adapter_line(it, pr))
    out["gold_plus_value_for_every_null_key"] = report(items, hall)["ALL_UC_dev0_reweighted"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preds-dir", type=Path)
    ap.add_argument("--exclude-holdout", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        res = selftest()
        (DS / "comparable" / "selftest.out.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
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
