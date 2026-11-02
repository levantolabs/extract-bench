#!/usr/bin/env python3
"""Self-check datasets/cord/items.jsonl against the COMMON ITEM FORMAT and count o200k tokens.

Run:
  /opt/homebrew/bin/uv run --no-project --with pillow==11.3.0 --with tiktoken==0.9.0 python -I \
      datasets/cord/scripts/selfcheck.py
Writes scripts/selfcheck.out.json next to this script.
"""
import collections
import json
import os
import re
import statistics

import tiktoken
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.dirname(HERE)
ITEMS = os.path.join(DS, "items.jsonl")
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
METRICS = {"exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim"}
TOP_KEYS = {"id", "dataset", "use_case", "modality", "text", "text_source", "images", "fields",
            "metric", "meta"}
MAX_IMG = 4 * 1024 * 1024


def pct(xs):
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=100, method="inclusive")
    return {"n": len(xs), "p50": q[49], "p95": q[94], "min": xs[0], "max": xs[-1],
            "mean": round(sum(xs) / len(xs), 1)}


def main():
    enc = tiktoken.get_encoding("o200k_base")
    errors, ids = [], set()
    n = 0
    split = collections.Counter()
    holdout = collections.Counter()
    gold_null = collections.Counter()
    gold_present = collections.Counter()
    field_types = collections.Counter()
    oracle_tok, fielddef_tok, ccocr_prompt_tok, img_bytes = [], [], [], []
    with open(ITEMS) as f:
        for ln, line in enumerate(f, 1):
            try:
                it = json.loads(line)
            except Exception as e:
                errors.append(f"line {ln}: json {e}")
                continue
            n += 1
            if set(it) != TOP_KEYS:
                errors.append(f"{ln}: keys {sorted(set(it) ^ TOP_KEYS)}")
            iid = it.get("id")
            if not isinstance(iid, str) or iid in ids:
                errors.append(f"{ln}: bad/duplicate id {iid}")
            ids.add(iid)
            if it["dataset"] != "cord":
                errors.append(f"{iid}: dataset {it['dataset']}")
            if it["modality"] not in ("text", "image", "text+image"):
                errors.append(f"{iid}: modality")
            if it["text_source"] not in ("native", "official_ocr", "none"):
                errors.append(f"{iid}: text_source")
            if it["text_source"] == "none" and it["text"] is not None:
                errors.append(f"{iid}: text_source none but text set")
            if it["metric"] not in METRICS:
                errors.append(f"{iid}: metric {it['metric']}")
            if not isinstance(it["images"], list) or len(it["images"]) > 4:
                errors.append(f"{iid}: images list")
            for rel in it["images"]:
                p = os.path.join(DS, rel)
                if not os.path.isfile(p):
                    errors.append(f"{iid}: missing image {rel}")
                    continue
                sz = os.path.getsize(p)
                img_bytes.append(sz)
                if sz > MAX_IMG:
                    errors.append(f"{iid}: image > 4 MiB {rel} {sz}")
                with Image.open(p) as im:
                    if im.format not in ("PNG", "JPEG", "WEBP"):
                        errors.append(f"{iid}: image format {im.format}")
            names = set()
            for fd in it["fields"]:
                if set(fd) != {"name", "type", "instructions", "gold"}:
                    errors.append(f"{iid}: field keys {sorted(fd)}")
                if not NAME_RE.match(fd.get("name", "")):
                    errors.append(f"{iid}: field name {fd.get('name')}")
                if fd["name"] in names:
                    errors.append(f"{iid}: duplicate field {fd['name']}")
                names.add(fd["name"])
                if fd["type"] not in TYPES:
                    errors.append(f"{iid}: type {fd['type']}")
                if not isinstance(fd["instructions"], str) or not fd["instructions"].strip():
                    errors.append(f"{iid}: empty instructions {fd['name']}")
                g = fd["gold"]
                if fd["type"] == "list":
                    if g is not None and (not isinstance(g, list) or not g
                                          or not all(isinstance(x, str) for x in g)):
                        errors.append(f"{iid}: list gold {fd['name']} {g}")
                else:
                    ok = g is None or isinstance(g, (int, float)) or (
                        isinstance(g, list) and g and all(isinstance(x, (int, float)) for x in g))
                    if not ok:
                        errors.append(f"{iid}: scalar gold {fd['name']} {g}")
                (gold_null if g is None else gold_present)[fd["name"]] += 1
                field_types[fd["type"]] += 1
            m = it["meta"]
            split[m["split"]] += 1
            holdout[(m["split"], m["holdout"])] += 1
            if m["page_count"] != 1:
                errors.append(f"{iid}: page_count")
            oracle_tok.append(len(enc.encode(m["oracle_text"])))
            fielddef_tok.append(len(enc.encode(json.dumps(
                {fd["name"]: {"type": fd["type"], "instructions": fd["instructions"]}
                 for fd in it["fields"]}))))
            if m["split"] == "test":
                c = m["comparable"]
                if not c or not os.path.isfile(os.path.join(DS, c["image"])):
                    errors.append(f"{iid}: comparable view missing")
                else:
                    ccocr_prompt_tok.append(len(enc.encode(c["prompt"])))
                    if os.path.getsize(os.path.join(DS, c["image"])) > MAX_IMG:
                        errors.append(f"{iid}: comparable image > 4 MiB")
            elif m["comparable"] is not None:
                errors.append(f"{iid}: dev item has comparable view")
    out = {
        "n_items": n, "errors": errors, "by_split": dict(split),
        "holdout": {f"{k[0]}:{k[1]}": v for k, v in sorted(holdout.items())},
        "field_types": dict(field_types),
        "n_fields_total": sum(field_types.values()),
        "gold_null": dict(gold_null), "gold_present": dict(gold_present),
        "n_null_gold_fields": sum(gold_null.values()),
        "n_null_gold_scalar_fields": sum(v for k, v in gold_null.items()
                                         if not k.startswith("menu_")),
        "oracle_text_tokens_o200k": pct(oracle_tok),
        "field_defs_tokens_o200k": pct(fielddef_tok),
        "oracle_plus_field_defs_tokens_o200k": pct([a + b for a, b in zip(oracle_tok, fielddef_tok)]),
        "ccocr_prompt_tokens_o200k_test": pct(ccocr_prompt_tok) if ccocr_prompt_tok else None,
        "image_bytes": pct(img_bytes),
    }
    with open(os.path.join(HERE, "selfcheck.out.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
