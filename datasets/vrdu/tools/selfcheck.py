"""Self-check for datasets/vrdu/items.jsonl (offline). Writes tools/selfcheck.out.json.

  /opt/homebrew/bin/uv run --no-project --with tiktoken --with pillow python -I extract-bench/datasets/vrdu/tools/selfcheck.py
Checks: JSON parses; harness validate_item; field names/types; gold key present; images exist, <=4, JPEG, decodable,
<4 MiB each, <=8 MiB total; o200k tokens of text (p50/p95/max) vs the Sage 2 16,384-token cap; gold parseability
under the harness's own parsers (date dayfirst=False, money, list items); null-gold counts; gold reachability.
"""
from __future__ import annotations

import io, json, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

DS = Path(__file__).resolve().parent.parent
EXB = DS.parent.parent
sys.path.insert(0, str(EXB))
from harness import common as C, score as S  # noqa: E402
import tiktoken  # noqa: E402
from PIL import Image  # noqa: E402

enc = tiktoken.get_encoding("o200k_base")
FIELD_TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def main():
    lines = (DS / "items.jsonl").read_text().splitlines()
    items, errors = [], []
    for i, l in enumerate(lines):
        try:
            items.append(json.loads(l))
        except Exception as e:  # noqa: BLE001
            errors.append(f"line {i}: {e}")
    ids = Counter(it["id"] for it in items)
    errors += [f"dup id {k}" for k, v in ids.items() if v > 1]
    tok = defaultdict(list)
    nulls, golds = Counter(), Counter()
    parse_fail = defaultdict(list)
    reach = Counter(); reach_n = Counter()
    img_stats = {"n_images": 0, "max_bytes": 0, "max_total": 0, "pages_hist": Counter()}
    sage_over = []
    for it in items:
        try:
            C.validate_item(it)
        except ValueError as e:
            errors.append(str(e))
        corp = "registration" if it["id"].startswith("vrdu-reg-") else "adbuy"
        if it["dataset"] != "vrdu" or it["text_source"] != "official_ocr" or not it["text"]:
            errors.append(f"{it['id']}: dataset/text_source/text")
        n = len(enc.encode(it["text"], disallowed_special=()))
        if n != it["meta"]["n_tokens_o200k"]:
            errors.append(f"{it['id']}: token count mismatch")
        tok[corp].append(n); tok["all"].append(n)
        defs = sum(len(enc.encode(f["name"] + " " + f["instructions"])) + 8 for f in it["fields"] if f["type"] != "list")
        if n + defs > 16384:
            sage_over.append((it["id"], n + defs))
        # images
        if not (1 <= len(it["images"]) <= 4):
            errors.append(f"{it['id']}: {len(it['images'])} images")
        tot = 0
        for rel in it["images"]:
            p = DS / rel
            if not p.is_file():
                errors.append(f"{it['id']}: missing {rel}"); continue
            b = p.read_bytes(); tot += len(b)
            if b[:3] != b"\xff\xd8\xff":
                errors.append(f"{rel}: not JPEG")
            if len(b) >= 4 * 1024 * 1024:
                errors.append(f"{rel}: >= 4 MiB")
            Image.open(io.BytesIO(b)).verify()
            img_stats["n_images"] += 1; img_stats["max_bytes"] = max(img_stats["max_bytes"], len(b))
        img_stats["max_total"] = max(img_stats["max_total"], tot)
        if tot > 8 * 1024 * 1024:
            errors.append(f"{it['id']}: total images {tot} > 8 MiB")
        img_stats["pages_hist"][f"{corp}:{it['meta']['pdf_pages']}"] += 1
        if it["meta"]["pdf_pages"] != it["meta"]["ocr_pages"]:
            errors.append(f"{it['id']}: pdf pages {it['meta']['pdf_pages']} != ocr pages {it['meta']['ocr_pages']}")
        # fields
        for f in it["fields"]:
            key = f"{corp}.{f['name']}"
            if not NAME.match(f["name"]) or f["type"] not in FIELD_TYPES or "gold" not in f:
                errors.append(f"{it['id']}: bad field {f}")
            g = f["gold"]
            if g is None:
                nulls[key] += 1; continue
            golds[key] += 1
            vals = g if isinstance(g, list) else [g]
            if f["type"] == "list":
                if not isinstance(g, list):
                    errors.append(f"{it['id']}: list gold not list")
                alts = g if (g and isinstance(g[0], list)) else [g]
                if len(alts) > 1 and len({len(a) for a in alts}) != 1:
                    errors.append(f"{it['id']}: {f['name']} acceptable lists differ in length")
                for v in alts[-1]:  # the most-repaired acceptable list
                    if f.get("items") == "date" and S.parse_date(v, False) is None:
                        parse_fail[key].append(v)
                    if f.get("items") == "number" and S.parse_number(v) is None:
                        parse_fail[key].append(v)
                continue
            if f["type"] == "date" and not any(S.parse_date(v, False) for v in vals):
                parse_fail[key].append(g)
            if f["type"] == "money" and not any(S.parse_number(v) is not None for v in vals):
                parse_fail[key].append(g)
            if f["type"] == "text":
                reach_n[key] += 1
                reach[key] += bool(it["meta"]["gold_reachable"]["official_ocr"].get(f["name"]))
        oc = it["meta"]["gold_offsets_check"]
        if oc["mismatch"]:
            parse_fail["_offset_mismatch_docs"].append(it["id"])
    out = {
        "n_items": len(items), "errors": errors[:50], "n_errors": len(errors),
        "tokens_o200k": {k: {"n": len(v), "p50": statistics.median(v), "p95": pct(v, 95), "max": max(v),
                             "mean": round(statistics.mean(v), 1)} for k, v in tok.items()},
        "sage_cap_16384_over": sage_over,
        "images": {**img_stats, "pages_hist": dict(sorted(img_stats["pages_hist"].items()))},
        "gold_present": dict(sorted(golds.items())), "gold_null": dict(sorted(nulls.items())),
        "n_fields_total": sum(len(it["fields"]) for it in items),
        "n_null_gold_fields": sum(nulls.values()),
        "n_null_gold_list_fields": sum(v for k, v in nulls.items() if k.split(".")[1].startswith("line_")),
        "gold_parse_fail": {k: {"n": len(v), "examples": v[:8]} for k, v in parse_fail.items()},
        "text_gold_reachable": {k: f"{reach[k]}/{reach_n[k]}" for k in sorted(reach_n)},
        "gold_adaptations": Counter(a["field"] + ":" + a["rule"] for it in items for a in it["meta"]["gold_adaptations"]),
        "line_elements_repaired": sum(a.get("n_repaired", 0) for it in items for a in it["meta"]["gold_adaptations"]),
        "gold_issues": Counter(i["field"] + ":" + i["issue"] for it in items for i in it["meta"]["gold_issues"]),
        "line_elements_unrepaired": sum(i.get("n", 0) for it in items for i in it["meta"]["gold_issues"]),
    }
    (DS / "tools" / "selfcheck.out.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str)[:6000])


if __name__ == "__main__":
    main()
