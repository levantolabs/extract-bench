"""Self-check for kleister_charity/items.jsonl against the common item format and the harness scorers ($0, no network).

  /opt/homebrew/bin/uv run --no-project --with tiktoken python -I extract-bench/datasets/kleister_charity/scripts/selfcheck.py
Writes scripts/selfcheck.out.json. Exit code 1 if any hard check fails.
Independently re-derives the strata, the seed-20 sample and the holdout from raw/ (does not import build.py).
"""
from __future__ import annotations

import csv, json, lzma, random, re, statistics, sys
from pathlib import Path

DS = Path(__file__).resolve().parents[1]
EXB = DS.parents[1]
sys.path.insert(0, str(EXB))
from harness import clients as C, score as S  # noqa: E402
from harness.common import validate_item, windows  # noqa: E402

TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
METRICS = {"exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim"}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
REQ = ("id", "dataset", "use_case", "modality", "text", "text_source", "images", "fields", "metric", "meta")
FIELDS = ["charity_name", "charity_number", "report_date", "income_annually_in_british_pounds",
          "spending_annually_in_british_pounds", "address__post_town", "address__postcode", "address__street_line"]


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def unescape(s):
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "f": "\f", "\\": "\\"}.get(m[1], "\\" + m[1]), s)


def main():
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    ntok = lambda s: len(enc.encode(s, disallowed_special=()))  # noqa: E731
    fails, out = [], {}
    lines = (DS / "items.jsonl").read_text(encoding="utf-8").splitlines()
    items = []
    for i, l in enumerate(lines, 1):
        try:
            items.append(json.loads(l))
        except json.JSONDecodeError as e:
            fails.append(f"line {i}: {e}")
    out["n_lines"], out["n_parsed"] = len(lines), len(items)
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        fails.append("duplicate ids")

    # ---- format
    n_fields = n_null = 0
    empty_alt = {"tesseract_layer": 0, "djvu_layer": 0, "best_layer": 0}
    gold_self = {"values_scored": 0, "values_perfect": 0}
    for it in items:
        try:
            validate_item(it)
        except ValueError as e:
            fails.append(str(e))
        for k in REQ:
            if k not in it:
                fails.append(f"{it.get('id')}: missing {k}")
        if it["dataset"] != "kleister_charity" or it["modality"] != "text" or it["metric"] != "exact_norm":
            fails.append(f"{it['id']}: dataset/modality/metric")
        if it["text_source"] != "official_ocr" or not it["text"].strip():
            fails.append(f"{it['id']}: text")
        if it["images"]:
            fails.append(f"{it['id']}: images should be []")
        for k, v in it["meta"]["alt_texts"].items():
            p = DS / v
            if not p.is_file():
                fails.append(f"{it['id']}: alt text missing {v}")
            elif p.stat().st_size == 0:
                empty_alt[k] += 1
                if k != "djvu_layer":   # djvu is empty on scans with no PDF text layer; others must not be
                    fails.append(f"{it['id']}: empty alt text {v}")
        if [f["name"] for f in it["fields"]] != FIELDS:
            fails.append(f"{it['id']}: field order/names")
        for f in it["fields"]:
            n_fields += 1
            if not NAME.match(f["name"]) or f["type"] not in TYPES or "gold" not in f or not f.get("instructions"):
                fails.append(f"{it['id']}: field {f['name']}")
            g = f["gold"]
            if g is None:
                n_null += 1
                if f.get("null_semantics") != "key_absent_in_kleister_gold":
                    fails.append(f"{it['id']}/{f['name']}: null gold without null_semantics")
                if S.outcome(None, None, None, f["type"], it["metric"], f, "a")[0] != "correct_abstain":
                    fails.append(f"{it['id']}/{f['name']}: null+null")
                if S.outcome("x", 0.9, None, f["type"], it["metric"], f, "a")[0] != "hallucinated":
                    fails.append(f"{it['id']}/{f['name']}: null+value")
                continue
            for a in (g if isinstance(g, list) else [g]):
                s = S.score_value(a, g, f["type"], it["metric"], f)
                gold_self["values_scored"] += 1
                gold_self["values_perfect"] += s == 1.0
                if s != 1.0:
                    fails.append(f"{it['id']}/{f['name']}: gold-as-pred {a!r} scored {s}")
            wrong = {"text": "Atlantis Trust", "money": "1.23", "date": "1999-12-31"}[f["type"]]
            if S.score_value(wrong, g, f["type"], it["metric"], f) != 0.0:
                fails.append(f"{it['id']}/{f['name']}: wrong value scored > 0")
    out["n_fields"], out["n_null_gold_fields"], out["gold_as_pred"] = n_fields, n_null, gold_self
    out["empty_alt_texts"] = empty_alt

    # ---- typed equivalences the arms will produce
    eq = {"money_float": 0, "money_written_gbp": 0, "money_n": 0, "date_uk_words": 0, "date_ddmmyyyy": 0, "date_n": 0}
    mon = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
    for it in items:
        for f in it["fields"]:
            g = f["gold"]
            if g is None:
                continue
            if f["type"] == "money":
                x = float(g)
                eq["money_n"] += 1
                eq["money_float"] += S.exact_norm(x, g, "money", f) == 1.0
                eq["money_written_gbp"] += S.exact_norm(("-" if x < 0 else "") + f"£{abs(x):,.2f}", g, "money", f) == 1.0
            if f["type"] == "date":
                y, m, d = g.split("-")
                eq["date_n"] += 1
                eq["date_uk_words"] += S.exact_norm(f"{int(d)} {mon[int(m) - 1]} {y}", g, "date", f) == 1.0
                eq["date_ddmmyyyy"] += S.exact_norm(f"{d}/{m}/{y}", g, "date", f) == 1.0
    out["typed_equivalence"] = eq
    if eq["money_float"] != eq["money_n"] or eq["money_written_gbp"] != eq["money_n"] or \
            eq["date_uk_words"] != eq["date_n"] or eq["date_ddmmyyyy"] != eq["date_n"]:
        fails.append(f"typed equivalence {eq}")

    # ---- request bodies (no network)
    it = items[0]
    cb = C.clairic_body(it, it["text"])
    sb = C.sage2_body(it, it["text"], "precise")
    js = C.json_schema(it)
    out["request_shapes"] = {"clairic_fields": {k: v["type"] + (f"/{v['unit']}" if "unit" in v else "")
                                                for k, v in cb["fields"].items()},
                             "sage_fields": {k: v["type"] for k, v in sb["extract"].items()},
                             "openrouter_required": js["required"]}
    if set(sb["extract"]) != set(FIELDS) or cb["fields"]["income_annually_in_british_pounds"].get("unit") != "GBP" \
            or sb["extract"]["income_annually_in_british_pounds"]["type"] != "money":
        fails.append("request shapes")

    # ---- independent re-derivation of strata, sample and holdout from raw/
    csv.field_size_limit(sys.maxsize)
    with lzma.open(DS / "raw/repo/dev-0/in.tsv.xz", "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    tok = {r[0].removesuffix(".pdf"): ntok(unescape(r[4])) for r in rows}
    long_ids = sorted(m for m, t in tok.items() if t > 15000)
    short_all = sorted(m for m, t in tok.items() if t <= 15000)
    short_ids = sorted(random.Random(20).sample(short_all, 100))
    want = {f"kleister_charity-{m}" for m in long_ids + short_ids}
    if want != set(ids):
        fails.append(f"sample mismatch: {len(want ^ set(ids))} ids differ")
    hold = set()
    for grp in (long_ids, short_ids):
        hold |= set(random.Random(20).sample(grp, round(0.2 * len(grp))))
    if {f"kleister_charity-{m}" for m in hold} != {it["id"] for it in items if it["meta"]["holdout"]}:
        fails.append("holdout mismatch")
    out["strata"] = {"dev0_long": len(long_ids), "dev0_short": len(short_all), "sample_long": len(long_ids),
                     "sample_short": len(short_ids), "holdout_long": len(hold & set(long_ids)),
                     "holdout_short": len(hold & set(short_ids))}
    for it in items:
        m = it["meta"]
        st = "long" if m["doc_id"] in long_ids else "short"
        w = {"long": 1.0, "short": len(short_all) / 100}[st]
        if m["stratum"] != st or abs(m["stratum_weight"] - w) > 1e-9 or m["text_tokens_o200k"] != tok[m["doc_id"]]:
            fails.append(f"{it['id']}: stratum/weight/tokens")

    # ---- tokens per layer (o200k) and Sage window counts via the harness's own windows()
    toks = {"textract_primary": [ntok(x["text"]) for x in items]}
    for layer in ("tesseract_layer", "djvu_layer", "best_layer"):
        toks[layer] = [ntok((DS / x["meta"]["alt_texts"][layer]).read_text()) for x in items]
    out["tokens_o200k"] = {k: {"p50": statistics.median(v), "p95": round(pct(v, 0.95), 1), "max": max(v),
                               "total": sum(v)} for k, v in toks.items()}
    for s in ("long", "short"):
        v = [ntok(x["text"]) for x in items if x["meta"]["stratum"] == s]
        out["tokens_o200k"][f"textract_{s}"] = {"n": len(v), "p50": statistics.median(v), "p95": round(pct(v, 0.95), 1),
                                                "max": max(v)}
    win_bad = 0
    for x in items:
        defs = len(enc.encode(json.dumps(C.sage2_body(x, "", "precise")["extract"])))
        nw = len(windows(x["text"], max(1000, C.SAGE_WINDOW - defs)))
        win_bad += nw != x["meta"]["sage_windows_est"]
    out["sage_windows_mismatch_vs_harness"] = win_bad
    out["sage_windows_total_harness"] = sum(x["meta"]["sage_windows_est"] for x in items)
    out["holdout_n"] = sum(x["meta"]["holdout"] for x in items)
    out["null_by_field"] = {n: sum(1 for x in items for f in x["fields"] if f["name"] == n and f["gold"] is None)
                            for n in FIELDS}
    out["pdf_annex_keys"] = sum(1 for x in items if x["meta"].get("pdf_bytes"))
    out["fails"] = fails[:50]
    out["n_fails"] = len(fails)
    out["ok"] = not fails
    (DS / "scripts" / "selfcheck.out.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
