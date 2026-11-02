"""Self-check for datasets/realdoc/items.jsonl (+ token estimates and scorer sanity). $0, no network.

Checks: every line parses; harness validate_item(); field names/types; every field has a 'gold' key (value or
null); ids unique; images exist, are PNG/JPEG by magic, <= 4 MiB each, <= 4 per item, <= 8 MiB total; the
vendored RealDoc scorer is verbatim (verify()) and passes the upstream unit-test asserts (transcribed from
tests/test_evaluate_scoring.py @ fb26a687); gold-vs-gold scores 1.0 under the official score_typed; our field
golds equal the official gold_dict values under deep_equal.

Token estimates (o200k_base): `text` is null for every item (no official OCR), so this reports the PDF text
layer where one exists plus the field instructions; local Tesseract OCR tokens come from scripts/ocr_proxy.py
(the harness OCR stage fails on this machine, see that script). Usage (from extract-bench/):
  /opt/homebrew/bin/uv run --no-project --with tiktoken==0.12.0 --with 'rapidfuzz>=3.9' python -I \
      datasets/realdoc/scripts/selfcheck.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

DS = Path(__file__).resolve().parent.parent
EXB = DS.parent.parent
sys.path.insert(0, str(DS / "scripts"))
sys.path.insert(0, str(EXB))
import realdoc_official as R  # noqa: E402
sys.path.insert(0, str(DS))
from build import reachable  # noqa: E402  (our build.py; same gold_reachable rule as meta.gold_reachable)
from harness.common import FIELD_NAME, FIELD_TYPES, validate_item  # noqa: E402

GH_SHA = "fb26a6876481de76dc293f722ab4efa71279904d"
UP = DS / "raw" / "github" / "src" / f"realdoc-bench-{GH_SHA}" / "realdoc_bench" / "evaluate"


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1)))))
    return xs[k]


def upstream_asserts() -> int:
    """Transcribed from realdoc-bench tests/test_evaluate_scoring.py (same inputs, same expectations)."""
    S = R
    cases = [
        S.deep_equal(1234, "$1,234"), S.deep_equal(1234.0, "$1,234.00"), S.deep_equal(5.875, "5.875%"),
        not S.deep_equal(1234, "$1,235"),
        S.deep_equal("2024-01-15", "2024-01-15"), not S.deep_equal("01/15/2024", "2024-01-15"),
        S.deep_equal("0.00", "0 . 0 0"), S.deep_equal("CAT", "C A T"),
        S.deep_equal("can't", "can’t"), S.deep_equal("12-34", "12–34"),
        S.deep_equal("hello world", "**hello** world"), S.deep_equal("a b", "a<br/>b"),
        S.deep_equal(True, True), not S.deep_equal(True, 1), not S.deep_equal(False, 0),
        S.deep_equal([{"k": 1}], {"k": 1}), S.deep_equal({"k": 1}, [{"k": 1}]),
        not S.fuzzy_equal("4001884081", "4001844081"), S.fuzzy_equal("4001884081", "4001884081"),
        S.fuzzy_equal("the quick brown fox jumps over the lazy dog",
                      "the quick brown fox jumped over the lazy dogs"),
        S.fuzzy_equal(1234, "$1,234"),
        S.string_keys('{\n  "name": <string>,\n  "amount": <number>,\n  "active": <boolean>,\n'
                      '  "category": <one of: A | B>\n}') == {"name"},
        S.score_typed({"name": "Acme Corp.", "amount": 100}, {"name": "Acme Corp", "amount": 100}, {"name"})
        == ({"name": True, "amount": True}, True),
        '"amount"' in S.build_template("Return exactly: amount=<number>; date=<date>",
                                       {"amount": 100, "date": "2024-01-15"}),
        "<number" in S.build_template("Return exactly: amount=<number>; date=<date>",
                                      {"amount": 100, "date": "2024-01-15"}),
        '"date"' in S.build_template("Return exactly: amount=<number>; date=<date>",
                                     {"amount": 100, "date": "2024-01-15"}),
        "list of strings" in S.build_template("describe the items", {"items": ["a", "b"]}),
        # null handling we rely on (not in upstream tests): null==null True, null vs value False
        S.deep_equal(None, None), not S.deep_equal(None, "x"), not S.fuzzy_equal(None, "Acme Corp"),
    ]
    bad = [i for i, ok in enumerate(cases) if not ok]
    if bad:
        raise AssertionError(f"upstream asserts failed at {bad}")
    return len(cases)


def magic(p: Path) -> str:
    b = p.read_bytes()[:8]
    return "png" if b == b"\x89PNG\r\n\x1a\n" else ("jpeg" if b[:3] == b"\xff\xd8\xff" else "other")


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    ntok = lambda s: len(enc.encode(s or "", disallowed_special=()))  # noqa: E731

    out = {"vendored_defs_verbatim": R.verify(UP), "upstream_asserts_passed": upstream_asserts()}
    lines = (DS / "items.jsonl").read_text().splitlines()
    items, ids = [], set()
    for i, l in enumerate(lines, 1):
        it = json.loads(l)  # raises on a bad line
        validate_item(it)
        assert it["dataset"] == "realdoc" and it["id"] not in ids, it["id"]
        ids.add(it["id"])
        assert it["modality"] == "image" and it["text"] is None and it["text_source"] == "none"
        assert 1 <= len(it["images"]) <= 4
        tot = 0
        for p in it["images"]:
            f = DS / p
            assert f.is_file(), p
            assert magic(f) in ("png", "jpeg"), p
            assert f.stat().st_size <= 4 * 1024 * 1024, p
            tot += f.stat().st_size
        assert tot <= 8 * 1024 * 1024, it["id"]
        assert len(it["images"]) == it["meta"]["page_count"] <= 4
        names = [f["name"] for f in it["fields"]]
        assert len(names) == len(set(names)), it["id"]
        og, sk = it["meta"]["official_gold_dict"], set(it["meta"]["official_str_keys"])
        fm, allc = R.score_typed(og, og, sk)
        assert allc, it["id"]
        for f in it["fields"]:
            assert FIELD_NAME.match(f["name"]) and f["type"] in FIELD_TYPES and "gold" in f
            assert f["type"] in ("text", "number", "date"), f
            assert R.deep_equal(f["gold"], og[f["realdoc_key"]]), (it["id"], f["name"])
            assert it["meta"]["name_map"][f["name"]] == f["realdoc_key"]
            # an empty answer must not score as correct on a non-null gold
            if f["gold"] is not None:
                assert not R.fuzzy_equal(None, f["gold"]) and not R.deep_equal(None, f["gold"])
        assert set(og) == {f["realdoc_key"] for f in it["fields"]} | set(it["meta"]["excluded_fields"])
        items.append(it)
    out["n_items"] = len(items)
    out["n_lines"] = len(lines)

    fields = [f for it in items for f in it["fields"]]
    out["n_fields_scored"] = len(fields)
    out["n_fields_total_incl_excluded"] = sum(it["meta"]["n_gold_fields_total"] for it in items)
    out["n_excluded"] = Counter(v["reason"].split(" ")[0] for it in items
                                for v in it["meta"]["excluded_fields"].values())
    out["n_null_gold"] = sum(f["gold"] is None for f in fields)
    out["fields_by_type"] = Counter(f["type"] for f in fields)
    out["fields_by_hint"] = Counter(("enum" if f["realdoc_hint"].startswith("one of") else
                                     f["realdoc_hint"].replace(" | null", "")) for f in fields)
    out["enum_state_fields"] = sum(bool(f.get("enum_state")) for f in fields)
    out["integer_flag_fields"] = sum(bool(f.get("integer")) for f in fields)
    out["fuzzy_fields"] = sum(bool(f.get("fuzzy")) for f in fields)
    out["sanitized_names"] = sum(f["name"] != f["realdoc_key"] for f in fields)
    out["by_domain"] = Counter(it["meta"]["domain"] for it in items)
    out["null_by_domain"] = Counter(it["meta"]["domain"] for it in items for f in it["fields"] if f["gold"] is None)
    out["holdout"] = sum(it["meta"]["holdout"] for it in items)
    out["unique_docs"] = len({it["meta"]["doc_id"] for it in items})
    out["page_counts"] = Counter(it["meta"]["page_count"] for it in items)
    out["questions_per_doc_max"] = max(Counter(it["meta"]["doc_id"] for it in items).values())
    out["ai_status"] = Counter(it["meta"]["provenance"]["ai_status"] for it in items)
    out["capabilities_top"] = Counter(c for it in items for c in it["meta"]["capabilities"]).most_common(12)
    out["fields_per_item"] = Counter(len(it["fields"]) for it in items)
    docs_tl = {it["meta"]["doc_id"]: it["meta"]["text_layer"]["has_text_layer"] for it in items}
    out["docs_with_text_layer"] = f"{sum(docs_tl.values())}/{len(docs_tl)}"
    sizes = [(DS / p).stat().st_size for it in items for p in it["images"]]
    out["image_bytes_p50_p95_max"] = [pct(sizes, 50), pct(sizes, 95), max(sizes)]
    out["image_formats"] = Counter(magic(DS / p) for it in items for p in it["images"])

    # tokens
    instr = [sum(ntok(f["instructions"]) for f in it["fields"]) for it in items]
    tl_tok = [ntok((DS / it["meta"]["alt_texts"]["pdf_text_layer"]).read_text()) for it in items
              if it["meta"]["alt_texts"].get("pdf_text_layer")]
    out["tokens_text_field"] = {"p50": 0, "p95": 0, "note": "text is null on every item"}
    out["tokens_field_instructions_per_item"] = {"p50": pct(instr, 50), "p95": pct(instr, 95), "max": max(instr)}
    out["tokens_pdf_text_layer"] = {"n": len(tl_tok), "p50": pct(tl_tok, 50), "p95": pct(tl_tok, 95),
                                    "max": max(tl_tok) if tl_tok else None}
    # reachability in the PDF text layer (span ceiling preview), non-null, non-enum golds
    rr = [(it["meta"]["gold_reachable"]["pdf_text_layer"][f["name"]]) for it in items for f in it["fields"]
          if it["meta"]["gold_reachable"] and it["meta"]["gold_reachable"]["pdf_text_layer"].get(f["name"]) is not None]
    out["gold_reachable_text_layer"] = f"{sum(rr)}/{len(rr)}"

    try:
        import rapidfuzz
        out["rapidfuzz_version"] = rapidfuzz.__version__
    except Exception:  # noqa: BLE001
        pass
    out["tiktoken_version"] = tiktoken.__version__ if hasattr(tiktoken, "__version__") else "UNKNOWN"
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
