#!/usr/bin/env python3
"""Self-check for datasets/sroie/items.jsonl (+ sanity run of the unmodified CC-OCR KIE scorer).

Run:
  /opt/homebrew/bin/uv run --no-project --with pillow==11.3.0 --with tiktoken==0.12.0 \
     --with zss==1.2.0 --with nltk==3.9.1 --with tabulate==0.9.0 python -I \
     datasets/sroie/scripts/selfcheck.py
Writes scripts/selfcheck.out.json.
Only kie_evaluator.py + common.py from the CC-OCR repo are loaded (both read in full first);
the package __init__ (which imports the other evaluators) is NOT executed.
"""
import importlib.util
import json
import os
import re
import statistics
import sys
import types

import tiktoken
from PIL import Image

DS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(DS, "items.jsonl")
MIRROR = os.path.join(DS, "raw", "jsdnrs-mirror", "test_text.jsonl")
EVAL_DIR = os.path.join(DS, "raw", "cc-ocr-github", "evaluation", "evaluator")
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
METRICS = {"exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim"}
MAX_B = 4 * 1024 * 1024


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def load_ccocr_kie():
    pkg = types.ModuleType("ccocr_eval")
    pkg.__path__ = [EVAL_DIR]
    sys.modules["ccocr_eval"] = pkg
    for mod in ("common", "kie_evaluator"):
        spec = importlib.util.spec_from_file_location(f"ccocr_eval.{mod}",
                                                      os.path.join(EVAL_DIR, f"{mod}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"ccocr_eval.{mod}"] = m
        spec.loader.exec_module(m)
    return sys.modules["ccocr_eval.kie_evaluator"]


def main():
    errors, n = [], 0
    ids = set()
    enc = tiktoken.get_encoding("o200k_base")
    text_tok, prompt_tok, gold_tok, proxy_tok, img_bytes, holdout = [], [], [], [], [], 0
    typed_list_gold = 0
    items = []
    with open(ITEMS) as f:
        for ln, line in enumerate(f, 1):
            try:
                it = json.loads(line)
            except Exception as e:  # noqa
                errors.append(f"line {ln}: json {e}")
                continue
            items.append(it)
            n += 1
            for k in ("id", "dataset", "use_case", "modality", "text", "text_source", "images",
                      "fields", "metric", "meta"):
                if k not in it:
                    errors.append(f"{ln}: missing {k}")
            if it["id"] in ids:
                errors.append(f"{ln}: dup id")
            ids.add(it["id"])
            if it["dataset"] != "sroie":
                errors.append(f"{ln}: dataset")
            if it["modality"] not in ("text", "image", "text+image"):
                errors.append(f"{ln}: modality")
            if it["text_source"] not in ("native", "official_ocr", "none"):
                errors.append(f"{ln}: text_source")
            if (it["text"] is None) != (it["text_source"] == "none"):
                errors.append(f"{ln}: text/text_source mismatch")
            if it["metric"] not in METRICS:
                errors.append(f"{ln}: metric")
            if not (1 <= len(it["images"]) <= 4):
                errors.append(f"{ln}: images count")
            for p in it["images"]:
                fp = os.path.join(DS, p)
                if not os.path.isfile(fp):
                    errors.append(f"{ln}: missing image {p}")
                    continue
                sz = os.path.getsize(fp)
                img_bytes.append(sz)
                if sz > MAX_B:
                    errors.append(f"{ln}: image > 4 MiB {p}")
                with Image.open(fp) as im:
                    if im.format not in ("JPEG", "PNG", "WEBP"):
                        errors.append(f"{ln}: image format {im.format}")
            names = set()
            for fd in it["fields"]:
                if not NAME_RE.match(fd.get("name", "")):
                    errors.append(f"{ln}: bad field name {fd.get('name')}")
                if fd["name"] in names:
                    errors.append(f"{ln}: dup field {fd['name']}")
                names.add(fd["name"])
                if fd.get("type") not in TYPES:
                    errors.append(f"{ln}: bad type {fd.get('type')}")
                if "gold" not in fd:
                    errors.append(f"{ln}: gold key missing {fd['name']}")
                if not fd.get("instructions"):
                    errors.append(f"{ln}: empty instructions {fd['name']}")
                g = fd.get("gold")
                if g is None:
                    errors.append(f"{ln}: null gold on SROIE {fd['name']} (spec: no nulls)")
                if isinstance(g, list):
                    typed_list_gold += 1
                if fd["type"] == "date" and isinstance(g, str) and \
                        not re.fullmatch(r"\d{4}-\d{2}-\d{2}", g):
                    errors.append(f"{ln}: date gold not ISO {g}")
                if fd["type"] == "money" and isinstance(g, str) and \
                        not re.fullmatch(r"-?\d+\.\d{2}", g):
                    errors.append(f"{ln}: money gold not %.2f {g}")
            if names != {"company", "date", "address", "total"}:
                errors.append(f"{ln}: field set {names}")
            text_tok.append(len(enc.encode(it["text"])) if it["text"] else 0)
            prompt_tok.append(len(enc.encode(it["meta"]["comparable"]["prompt"])))
            gold_tok.append(len(enc.encode(json.dumps(it["meta"]["gold_verbatim"],
                                                      ensure_ascii=False))))
            holdout += bool(it["meta"]["holdout"])

    # proxy document-text size: mirror Task 1/2 OCR lines (truncated at first comma => lower bound)
    mirror = {}
    with open(MIRROR) as f:
        for line in f:
            r = json.loads(line)
            mirror["sroie-" + r["key"]] = r
    for it in items:
        r = mirror.get(it["id"])
        if r:
            proxy_tok.append(len(enc.encode("\n".join(r["words"]))))

    # unmodified CC-OCR scorer sanity: gold-as-prediction => 1.0; empty => 0.0
    kie = load_ccocr_kie()
    labels = json.load(open(os.path.join(DS, "comparable", "label.json")))
    ev = kie.KieEvaluator("kie")
    perfect = {k: v for k, v in labels.items()}  # raw JSON strings, as model responses
    _, e1 = ev(perfect, dict(labels))
    _, e0 = ev({k: "{}" for k in labels}, dict(labels))
    # typed-vs-comparable conflict demo: ISO dates fail the comparable scorer
    iso = {}
    for it in items:
        g = dict(it["meta"]["gold_verbatim"])
        d = next(fd["gold"] for fd in it["fields"] if fd["name"] == "date")
        if isinstance(d, str):
            g["date"] = d
        iso[it["meta"]["image_name"]] = json.dumps(g)
    _, e2 = ev(iso, dict(labels))

    out = {
        "n_items": n, "errors": errors[:50], "n_errors": len(errors), "holdout_n": holdout,
        "n_fields_total": sum(len(it["fields"]) for it in items),
        "n_null_gold_fields": sum(1 for it in items for fd in it["fields"] if fd["gold"] is None),
        "typed_fields_with_list_gold": typed_list_gold,
        "text_tokens_o200k": {"p50": pct(text_tok, 50), "p95": pct(text_tok, 95),
                              "note": "text is null for every item (harness OCRs later)"},
        "proxy_ocr_tokens_o200k_lower_bound": {
            "n": len(proxy_tok), "p50": pct(proxy_tok, 50), "p95": pct(proxy_tok, 95),
            "max": max(proxy_tok) if proxy_tok else None,
            "note": "jsdnrs mirror Task1/2 OCR lines joined by \\n; lines truncated at first comma, "
                    "so true OCR text is somewhat longer"},
        "ccocr_prompt_tokens_o200k": {"p50": pct(prompt_tok, 50), "p95": pct(prompt_tok, 95)},
        "gold_json_tokens_o200k": {"p50": pct(gold_tok, 50), "p95": pct(gold_tok, 95)},
        "image_bytes": {"p50": pct(img_bytes, 50), "p95": pct(img_bytes, 95),
                        "max": max(img_bytes) if img_bytes else None},
        "ccocr_scorer_sanity": {
            "gold_as_prediction_f1": e1["summary"]["f1_score"],
            "gold_as_prediction_acc": e1["summary"]["acc"],
            "empty_prediction_f1": e0["summary"]["f1_score"],
            "iso_date_prediction_f1": e2["summary"]["f1_score"],
            "iso_date_class_f1": {k: round(v["acc"], 4) for k, v in e2["class_f1_score"].items()},
        },
    }
    with open(os.path.join(DS, "scripts", "selfcheck.out.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
