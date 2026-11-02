"""Self-check for kleister_nda/items.jsonl against the common item format and the harness scorers ($0, no network).

  /opt/homebrew/bin/uv run --no-project --with tiktoken python -I extract-bench/datasets/kleister_nda/scripts/selfcheck.py
Writes scripts/selfcheck.out.json. Exit code 1 if any hard check fails.
"""
from __future__ import annotations

import json, re, statistics, sys
from pathlib import Path

DS = Path(__file__).resolve().parents[1]
EXB = DS.parents[1]
sys.path.insert(0, str(EXB))
from harness import clients as C, score as S  # noqa: E402
from harness.common import validate_item  # noqa: E402

TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
METRICS = {"exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim"}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
REQ = ("id", "dataset", "use_case", "modality", "text", "text_source", "images", "fields", "metric", "meta")


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
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
    n_fields = n_null = 0
    gold_self = {"scored": 0, "perfect": 0, "variants_scored": 0, "variants_perfect": 0}
    for it in items:
        try:
            validate_item(it)
        except ValueError as e:
            fails.append(str(e))
        for k in REQ:
            if k not in it:
                fails.append(f"{it.get('id')}: missing {k}")
        if it["dataset"] != "kleister_nda" or it["modality"] != "text" or it["metric"] not in METRICS:
            fails.append(f"{it['id']}: dataset/modality/metric")
        if it["text_source"] != "native" or not it["text"]:
            fails.append(f"{it['id']}: text")
        for p in it["images"]:
            if not (DS / p).is_file():
                fails.append(f"{it['id']}: missing image {p}")
        for v in it["meta"]["alt_texts"].values():
            if not (DS / v).is_file() or (DS / v).stat().st_size == 0:
                fails.append(f"{it['id']}: alt text {v}")
        for f in it["fields"]:
            n_fields += 1
            if not NAME.match(f["name"]) or f["type"] not in TYPES or "gold" not in f or not f.get("instructions"):
                fails.append(f"{it['id']}: field {f}")
            g = f["gold"]
            if g is None:
                n_null += 1
                o = S.outcome(None, None, None, f["type"], it["metric"], f, "a")
                if o[0] != "correct_abstain":
                    fails.append(f"{it['id']}/{f['name']}: null gold + null pred -> {o}")
                continue
            # gold as prediction must score 1.0 (first acceptable value, then every acceptable value)
            first = g[0] if isinstance(g, list) and (f["type"] != "list" or isinstance(g[0], list)) else g
            s = S.score_value(first, g, f["type"], it["metric"], f)
            gold_self["scored"] += 1
            gold_self["perfect"] += s == 1.0
            if s != 1.0:
                fails.append(f"{it['id']}/{f['name']}: gold-as-pred scored {s}")
            alts = g if isinstance(g, list) and (f["type"] != "list" or isinstance(g[0], list)) else [g]
            for a in alts:
                s = S.score_value(a, g, f["type"], it["metric"], f)
                gold_self["variants_scored"] += 1
                gold_self["variants_perfect"] += s == 1.0
            # wrong values must score 0
            wrong = {"date": "1999-12-31", "text": "Atlantis", "duration": "P9D", "list": ["Atlantis Corp."]}[f["type"]]
            if S.score_value(wrong, g, f["type"], it["metric"], f) != 0.0:
                fails.append(f"{it['id']}/{f['name']}: wrong value scored > 0")
    out["n_fields"], out["n_null_gold_fields"], out["gold_as_pred"] = n_fields, n_null, gold_self

    # term: Kleister forms, word forms and Clairic numeric months all equal the ISO gold
    term_eq = {"kleister_words": 0, "clairic_months": 0, "n": 0}
    for it in items:
        f = next(x for x in it["fields"] if x["name"] == "term")
        if f["gold"] is None:
            continue
        kg = it["meta"]["kleister_gold"]["term"][0]                      # e.g. 2_years
        n, u = int(f["gold"][1:-1]), f["gold"][-1]
        months = n * 12 if u == "Y" else n
        cl = C.parse_clairic(it, {"result": {"answers": {"term": {"value": months, "confidence": 0.9}}}})["term"]["value"]
        term_eq["n"] += 1
        term_eq["kleister_words"] += S.exact_norm(kg.replace("_", " "), f["gold"], "duration", f) == 1.0
        term_eq["clairic_months"] += S.exact_norm(cl, f["gold"], "duration", f) == 1.0
    out["term_equivalence"] = term_eq
    if term_eq["kleister_words"] != term_eq["n"] or term_eq["clairic_months"] != term_eq["n"]:
        fails.append(f"term equivalence {term_eq}")

    # party in Kleister form (no commas) and in surface form both hit
    party_forms = {"kleister_list_f1_1": 0, "n": 0}
    for it in items:
        f = next(x for x in it["fields"] if x["name"] == "party")
        kl = [p.replace("_", " ") for p in it["meta"]["kleister_gold"]["party"]]
        party_forms["n"] += 1
        party_forms["kleister_list_f1_1"] += S.score_value(kl, f["gold"], "list", it["metric"], f) == 1.0
    out["party_forms"] = party_forms
    if party_forms["kleister_list_f1_1"] != party_forms["n"]:
        fails.append(f"party forms {party_forms}")

    # request bodies build (no network): Clairic all 5 fields, Sage 4 (no list), OpenRouter strict schema
    it = items[0]
    cb = C.clairic_body(it, it["text"])
    sb = C.sage2_body(it, it["text"], "precise")
    js = C.json_schema(it)
    out["request_shapes"] = {"clairic_fields": {k: v["type"] + (f"/{v['unit']}" if "unit" in v else "")
                                                for k, v in cb["fields"].items()},
                             "sage_fields": {k: v["type"] for k, v in sb["extract"].items()},
                             "openrouter_required": js["required"]}
    if set(sb["extract"]) != {"effective_date", "jurisdiction", "term", "party_any"} or cb["fields"]["term"].get("unit") != "months":
        fails.append("request shapes")

    # tokens (o200k) recomputed independently of build.py
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        tok = {"text_best": [len(enc.encode(x["text"], disallowed_special=())) for x in items]}
        for layer in ("tesseract_layer", "textract_layer"):
            tok[layer] = [len(enc.encode((DS / x["meta"]["alt_texts"][layer]).read_text(), disallowed_special=()))
                          for x in items]
        out["tokens_o200k"] = {k: {"p50": statistics.median(v), "p95": round(pct(v, 0.95), 1), "max": max(v),
                                   "total": sum(v)} for k, v in tok.items()}
        if tok["text_best"] != [x["meta"]["text_tokens_o200k"] for x in items]:
            fails.append("token counts differ from meta")
    except ImportError:
        out["tokens_o200k"] = "tiktoken unavailable"
        fails.append("tiktoken unavailable")
    pages = [x["meta"]["page_count"] for x in items]
    out["pages"] = {"p50": statistics.median(pages), "p95": round(pct(pages, 0.95), 1), "max": max(pages),
                    "total": sum(pages)}
    out["holdout_n"] = sum(x["meta"]["holdout"] for x in items)
    out["null_by_field"] = {n: sum(1 for x in items for f in x["fields"] if f["name"] == n and f["gold"] is None)
                            for n in ("effective_date", "jurisdiction", "term", "party", "party_any")}
    out["fails"] = fails
    out["ok"] = not fails
    (DS / "scripts" / "selfcheck.out.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
