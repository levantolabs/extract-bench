#!/usr/bin/env python3
"""Build datasets/cord/items.jsonl from raw/ (no network, no metered API).

Source: CORD v2, https://huggingface.co/datasets/naver-clova-ix/cord-v2 at revision
        7f0115a4b758a71d6473b8d085751692da2fef98 (CC BY 4.0), files
        data/test-00000-of-00001-9c204eb3f4e11791.parquet        (100 receipts)
        data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet  (100 receipts, "dev")
        Columns: image (struct bytes/path), ground_truth (JSON string with gt_parse,
        meta, valid_line, roi, repeating_symbol, dontcare).
COMPARABLE view (test only): CC-OCR KIE track CORD subset,
        https://huggingface.co/datasets/wulipc/CC-OCR revision
        c64517e92179991d509776064174776700cdd5a2, kie/constrained_category/CORD_100.tsv
        (VLMEvalKit TSV: index, image b64, image_name test_<id>.jpg, question, answer, ...),
        scored by the unmodified CC-OCR kie_evaluator.py
        (AlibabaResearch/AdvancedLiterateMachinery@851672c8, raw/cc-ocr-github/).

Sampling: sample_n = 200 = every receipt in test (100) + dev (100); nothing is subsampled.
Seed 20 only (a) marks the sealed 20% holdout (SURVEY L13), stratified by split
(20 test + 20 dev), and (b) is recorded for reproducibility.

Two scoring views per item (never printed in one table):
  * `fields` (TYPED, "our adaptation, not comparable"): 9 scalar leaves of sub_total/total
    as money/number (gold null when the key is absent from gt_parse) + 3 list fields
    (menu nm/cnt/price, list-capable arms only, scored list_f1).
  * `meta.comparable` (test only; COMPARABLE with CC-OCR Table 5 CORD field F1 and, as a
    reference, LMDX Table 3): CC-OCR prompt verbatim (per-image key set), gold = CC-OCR
    answer JSON (checked equal to CORD v2 gt_parse), CC-OCR's own JPEG bytes.

Text: items carry text=null / text_source="none": CORD ships no OCR-engine output, only human
word transcripts, and the suite's primary text for Sage 2 / Clairic-text is the harness OCR
(SURVEY L6). The human transcript is kept as a labelled ORACLE arm in
meta.oracle_text ("gold-word oracle", never "OCR ceiling"): valid_line words grouped by the
annotators' row_id (physical line), lines ordered by median quad y-centre, words by quad
x-min. CORD v2 `dontcare` entries are polygons with NO transcription, so they cannot be added.

Run (from anywhere; never from inside raw/):
  /opt/homebrew/bin/uv run --no-project --with pyarrow==20.0.0 --with pillow==11.3.0 \
      --with tiktoken==0.9.0 python -I \
      datasets/cord/build.py
"""
import base64
import csv
import hashlib
import io
import json
import os
import random
import re
import statistics
import sys

import pyarrow.parquet as pq
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
IMG_DIR = os.path.join(HERE, "img")
CMP_DIR = os.path.join(HERE, "comparable")
OUT = os.path.join(HERE, "items.jsonl")
DATASET = "cord"
USE_CASE = "Receipt photos (expenses; Clairic's own launch demo case)"
SEED = 20
HOLDOUT_FRAC = 0.20
MAX_IMG_BYTES = 4 * 1024 * 1024
HF_REV = "7f0115a4b758a71d6473b8d085751692da2fef98"
PARQUETS = {  # split label -> (file, HF LFS sha256 from raw/hf_api.json)
    "test": ("test-00000-of-00001-9c204eb3f4e11791.parquet",
             "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7"),
    "dev": ("validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
            "0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15"),
}
CCOCR_TSV = os.path.join(RAW, "ccocr", "CORD_100.tsv")
CCOCR_REV = "c64517e92179991d509776064174776700cdd5a2"
CCOCR_COMMIT = "851672c8005bef8b7cb38c79c51a4fbf9a66d970"

# ---------------------------------------------------------------------------
# TYPED fields. Same instructions for every arm; written before any model output exists.
# (name, gt_parse section, gt_parse key, type, instructions)
# ---------------------------------------------------------------------------
CUR = "Amount in Indonesian rupiah (IDR) as a number, without the currency symbol."
NULL_NOTE = " If the receipt does not show it, return null."
SCALARS = [
    ("subtotal_price", "sub_total", "subtotal_price", "money",
     "Subtotal of the receipt: the sum of the items before tax, service charge and other "
     "adjustments. " + CUR + NULL_NOTE),
    ("discount_price", "sub_total", "discount_price", "money",
     "Total discount applied to the whole bill. " + CUR + NULL_NOTE),
    ("service_price", "sub_total", "service_price", "money",
     "Service charge amount. " + CUR + NULL_NOTE),
    ("tax_price", "sub_total", "tax_price", "money",
     "Tax amount (e.g. PB1, PPN or VAT). " + CUR + NULL_NOTE),
    ("total_price", "total", "total_price", "money",
     "Total amount to pay for the receipt (the grand total). " + CUR + NULL_NOTE),
    ("cashprice", "total", "cashprice", "money",
     "Amount paid in cash (cash tendered). " + CUR + NULL_NOTE),
    ("changeprice", "total", "changeprice", "money",
     "Change given back to the customer. " + CUR + NULL_NOTE),
    ("creditcardprice", "total", "creditcardprice", "money",
     "Amount paid by credit or debit card. " + CUR + NULL_NOTE),
    ("menuqty_cnt", "total", "menuqty_cnt", "number",
     "Total quantity of items purchased, as printed on the receipt (e.g. 'Total Qty' or "
     "'Items'). A count, not money." + NULL_NOTE),
]
LISTS = [
    ("menu_nm", "nm", "text",
     "Name of each purchased menu item, in the order printed (top to bottom), one entry per "
     "item line. Do not include sub-items/modifiers." + NULL_NOTE),
    ("menu_cnt", "cnt", "number",
     "Quantity of each purchased menu item, in the order printed (top to bottom), one entry "
     "per item line that shows a quantity." + NULL_NOTE),
    ("menu_price", "price", "money",
     "Line total price of each purchased menu item, in the order printed (top to bottom), "
     "one entry per item line that shows a price. " + CUR + NULL_NOTE),
]
# Keys in the same sections that are NOT asked; their presence is recorded so null golds of
# neighbouring fields can be hand-audited (SURVEY C7: audit 30 null golds per source).
AUDIT_KEYS = ["sub_total.etc", "sub_total.othersvc_price", "total.total_etc",
              "total.emoneyprice", "total.menutype_cnt"]


# ---------------------------------------------------------------------------
# Pre-registered gold conversion (SURVEY L2: "CORD Indonesian thousands rule").
# MONEY, in order:
#   1. strip currency tokens Rp / Rp. / IDR (any case), spaces and the wrappers * [ ] : =
#   2. "(x)" or a leading "-" marks a negative amount
#   3. the remaining string must be digits with optional '.'/',' separators, else UNPARSED
#   4. no separator -> integer
#   5. last group has 1-2 digits -> that separator is the decimal mark; every earlier
#      separator is a thousands separator (each later group exactly 3 digits)
#   6. last group has exactly 3 digits -> every separator is a thousands separator
#      (first group 1-3 digits, every later group 3 digits), e.g. "5.455" -> 5455
#   7. anything else (e.g. a 4-digit group after a separator) -> UNPARSED
# COUNT (menuqty_cnt, menu.cnt): the first number in the string ("1.00xITEMs" -> 1,
#   "(Qty=4.00)" -> 4, "x2" -> 2); '.'/',' followed by 1-2 digits is a decimal mark
#   ("2.00" -> 2); any other separator pattern -> UNPARSED.
# DISCOUNT: sign-insensitive (receipts print "-60.000", "(0)" or "19,400" for the same thing),
#   so gold lists both signs.
# UNPARSED gold (e.g. tax printed as "-") is never guessed: the field is left out of that
#   item's `fields` and recorded in meta.dropped_fields.
# A key with several values in gt_parse (e.g. tax ["6,818", "0"]) -> gold is the list of the
#   distinct parsed values (any one accepted), flagged in meta.multi_value_fields.
# ---------------------------------------------------------------------------
_CUR_RE = re.compile(r"(?i)rp\.?|idr")


def _num(v):
    return int(v) if float(v).is_integer() else round(float(v), 4)


def typed_money(raw):
    s = _CUR_RE.sub("", raw).replace(" ", "").strip("*[]:=")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.startswith("-"):
        neg, s = True, s[1:]
    if not re.fullmatch(r"\d+([.,]\d+)*", s):
        return None, "unparsed"
    parts = re.split(r"[.,]", s)
    if len(parts) == 1:
        val, rule = int(s), "plain"
    elif len(parts[-1]) in (1, 2):
        ints = parts[:-1]
        if len(ints) > 1 and not (1 <= len(ints[0]) <= 3 and all(len(p) == 3 for p in ints[1:])):
            return None, "unparsed"
        val, rule = float("".join(ints) + "." + parts[-1]), "decimal_mark"
    elif len(parts[-1]) == 3 and 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
        val, rule = int("".join(parts)), "thousands"
    else:
        return None, "unparsed"
    val = -val if neg else val
    return _num(val) if val != 0 else 0, rule


def typed_count(raw):
    m = re.search(r"\d+(?:[.,]\d+)?", raw)
    if not m:
        return None, "unparsed"
    t = m.group(0)
    rule = "plain" if t == raw.strip() else "first_number"
    if re.fullmatch(r"\d+", t):
        return int(t), rule
    whole, frac = re.split(r"[.,]", t)
    if len(frac) in (1, 2):
        return _num(float(whole + "." + frac)), rule + "+decimal_mark"
    return None, "unparsed"


# Hand audit of null golds (SURVEY C7), recorded by scripts/null_gold_audit.py. Optional input:
# merged into meta.null_gold_audit.hand_audit; `wrong_null` slots also go to
# meta.gold_adjudicated. `fields[].gold` always stays the raw CORD gold.
AUDIT_FILE = os.path.join(HERE, "audit", "null_gold_audit-2026-10-23.json")
HAND_AUDIT = {}
if os.path.exists(AUDIT_FILE):
    with open(AUDIT_FILE) as _f:
        for _s in json.load(_f)["slots"]:
            HAND_AUDIT.setdefault(_s["id"], []).append(
                {k: _s[k] for k in ("field", "verdict", "value_on_image", "note")})
# Independent verifier pass (targeted, not random; see the file's "method"). Merged into
# meta.null_gold_audit.verifier_audit; `wrong_null` and `wrong_value` slots also go to
# meta.gold_adjudicated. `fields[].gold` still stays the raw CORD gold.
VERIFY_FILE = os.path.join(HERE, "audit", "verifier_gold_audit-2026-10-23.json")
VERIFY_AUDIT = {}
if os.path.exists(VERIFY_FILE):
    with open(VERIFY_FILE) as _f:
        for _s in json.load(_f)["slots"]:
            VERIFY_AUDIT.setdefault(_s["id"], []).append(
                {k: _s[k] for k in ("field", "verdict", "value_on_image", "note")})


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def word_box(q):
    xs = [q["x1"], q["x2"], q["x3"], q["x4"]]
    ys = [q["y1"], q["y2"], q["y3"], q["y4"]]
    return min(xs), min(ys), max(xs), max(ys)


def oracle_text(valid_line):
    """Gold-word ORACLE text (human transcription, not OCR). See module docstring."""
    rows = {}
    for vl in valid_line:
        for w in vl["words"]:
            x0, y0, x1, y1 = word_box(w["quad"])
            key = w.get("row_id")
            key = ("row", key) if key is not None else ("solo", len(rows), x0, y0)
            rows.setdefault(key, []).append((x0, (y0 + y1) / 2.0, w["text"]))
    lines = []
    for ws in rows.values():
        ws.sort(key=lambda t: t[0])
        lines.append((statistics.median(t[1] for t in ws), ws[0][0],
                      " ".join(t[2] for t in ws if t[2] is not None)))
    lines.sort()
    return "\n".join(l[2] for l in lines)


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def load_ccocr():
    csv.field_size_limit(1 << 30)
    with open(CCOCR_TSV, "rb") as f:
        b = f.read()
    rows = list(csv.DictReader(io.StringIO(b.decode("utf-8")), delimiter="\t", quotechar='"'))
    assert len(rows) == 100, len(rows)
    out = {}
    for r in rows:
        m = re.fullmatch(r"test_(\d+)\.jpg", r["image_name"])
        assert m, r["image_name"]
        out[int(m.group(1))] = r
    assert sorted(out) == list(range(100))
    return out, sha256(b)


def write_image(b, stem, report):
    im = Image.open(io.BytesIO(b))
    im.load()
    fmt = im.format
    w, h = im.size
    exif_orient = None
    try:
        exif_orient = im.getexif().get(0x0112)
    except Exception:
        pass
    reenc = False
    if fmt in ("JPEG", "PNG", "WEBP") and len(b) <= MAX_IMG_BYTES:
        ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
        out_b = b
    else:
        reenc, ext = True, ".jpg"
        im2 = im.convert("RGB")
        q = 92
        while True:
            buf = io.BytesIO()
            im2.save(buf, "JPEG", quality=q)
            if buf.tell() <= MAX_IMG_BYTES or q <= 50:
                break
            q -= 7
        out_b = buf.getvalue()
        assert len(out_b) <= MAX_IMG_BYTES, (stem, len(out_b))
        report["reencoded"] += 1
    rel = f"img/{stem}{ext}"
    with open(os.path.join(HERE, rel), "wb") as f:
        f.write(out_b)
    report["img_formats"][fmt] = report["img_formats"].get(fmt, 0) + 1
    return rel, {"format": fmt, "width": w, "height": h, "bytes": len(out_b),
                 "sha256": sha256(out_b), "reencoded": reenc, "source_sha256": sha256(b),
                 "source_bytes": len(b), "exif_orientation": exif_orient}


def main():
    # ---- verify raw files against the pinned HF LFS hashes
    rows = []
    for split, (fn, want) in PARQUETS.items():
        p = os.path.join(RAW, fn)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            sys.exit(f"sha256 mismatch for {fn}: {h.hexdigest()} != {want}")
        t = pq.read_table(p, columns=["image", "ground_truth"]).to_pylist()
        assert len(t) == 100, (fn, len(t))
        for i, r in enumerate(t):
            rows.append((split, i, r))
    ccocr, ccocr_sha = load_ccocr()

    tok = None
    try:
        import tiktoken
        tok = tiktoken.get_encoding("o200k_base")
    except Exception as e:  # token counts are optional for the build
        print("tiktoken unavailable:", e, file=sys.stderr)

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(os.path.join(CMP_DIR, "images"), exist_ok=True)

    # ---- ids + holdout (seed 20, 20% per split)
    ids = {}
    for split, i, r in rows:
        gt = json.loads(r["ground_truth"])
        img_id = gt["meta"]["image_id"]
        ids[(split, i)] = f"cord-{split}-{img_id:03d}"
    assert len(set(ids.values())) == len(ids), "duplicate ids"
    rng = random.Random(SEED)
    holdout = set()
    for split in ("test", "dev"):
        pool = sorted(v for (s, _), v in ids.items() if s == split)
        holdout |= set(rng.sample(pool, round(HOLDOUT_FRAC * len(pool))))

    report = {"n": 0, "by_split": {}, "reencoded": 0, "img_formats": {}, "money_rules": {},
              "count_rules": {}, "null_gold": {}, "present_gold": {}, "dropped_fields": [],
              "multi_value_fields": [], "nonnumeric_raw": [], "list_value_joined": [],
              "ccocr_answer_ne_gt_parse": [], "ccocr_image_dims_differ": [],
              "audit_flags": 0, "oracle_tokens_o200k": [], "meta_split_values": {},
              "gold_reachable_oracle": {}}
    items, labels, qa_lines = [], {}, []
    for split, i, r in rows:
        iid = ids[(split, i)]
        gt = json.loads(r["ground_truth"])
        meta_gt = gt["meta"]
        report["meta_split_values"][meta_gt.get("split")] = \
            report["meta_split_values"].get(meta_gt.get("split"), 0) + 1
        p = gt["gt_parse"]
        img_b = r["image"]["bytes"]
        rel, img_info = write_image(img_b, iid, report)
        assert (img_info["width"], img_info["height"]) == (
            meta_gt["image_size"]["width"], meta_gt["image_size"]["height"]), iid
        otext = oracle_text(gt["valid_line"])
        otext_n = norm_ws(otext)

        fields, gold_raw, dropped, multi, rules, field_metrics = [], {}, [], [], {}, {}
        reach = {}
        for name, sec, key, ftype, instr in SCALARS:
            secs = [d for d in as_list(p.get(sec)) if isinstance(d, dict)]
            raws = []
            for d in secs:
                raws += [str(x) for x in as_list(d.get(key))]
            if not raws:
                gold = None
                report["null_gold"][name] = report["null_gold"].get(name, 0) + 1
            else:
                report["present_gold"][name] = report["present_gold"].get(name, 0) + 1
                gold_raw[name] = raws if len(raws) > 1 else raws[0]
                fn = typed_count if ftype == "number" else typed_money
                parsed = [fn(x) for x in raws]
                for x, (v, rule) in zip(raws, parsed):
                    bucket = "count_rules" if ftype == "number" else "money_rules"
                    report[bucket][rule] = report[bucket].get(rule, 0) + 1
                    if v is not None and not re.fullmatch(r"[\d.,]+", x.strip()):
                        report["nonnumeric_raw"].append({"id": iid, "field": name, "raw": x,
                                                         "typed": v})
                if any(v is None for v, _ in parsed):
                    dropped.append({"field": name, "raw": gold_raw[name],
                                    "reason": "gold not parseable by the pre-registered rule"})
                    report["dropped_fields"].append({"id": iid, "field": name,
                                                     "raw": gold_raw[name]})
                    continue
                vals = []
                for v, _ in parsed:
                    if v not in vals:
                        vals.append(v)
                if name == "discount_price":
                    for v in list(vals):
                        if v != 0 and -v not in vals:
                            vals.append(-v)
                gold = vals[0] if len(vals) == 1 else vals
                rules[name] = [rule for _, rule in parsed]
                if len(raws) > 1:
                    multi.append(name)
                    report["multi_value_fields"].append({"id": iid, "field": name, "raw": raws})
                reach[name] = any(norm_ws(x) in otext_n for x in raws)
                rr = report["gold_reachable_oracle"].setdefault(name, [0, 0])
                rr[0] += int(reach[name])
                rr[1] += 1
            fields.append({"name": name, "type": ftype, "instructions": instr, "gold": gold})
            field_metrics[name] = "exact_norm"

        menu = [d for d in as_list(p.get("menu")) if isinstance(d, dict)]
        list_typed = {}
        for name, key, item_type, instr in LISTS:
            vals = []
            for d in menu:
                v = d.get(key)
                if v is None:
                    continue
                if isinstance(v, list):  # several lines of one menu item: join in order
                    report["list_value_joined"].append({"id": iid, "field": name, "raw": v})
                    v = " ".join(str(x) for x in v)
                vals.append(str(v))
            gold = vals if vals else None
            if gold is None:
                report["null_gold"][name] = report["null_gold"].get(name, 0) + 1
            else:
                report["present_gold"][name] = report["present_gold"].get(name, 0) + 1
            if item_type != "text" and vals:
                fn = typed_count if item_type == "number" else typed_money
                list_typed[name] = [fn(v)[0] for v in vals]
            fields.append({"name": name, "type": "list", "instructions": instr, "gold": gold})
            field_metrics[name] = "list_f1"

        present_audit = {}
        for k in AUDIT_KEYS:
            sec, key = k.split(".")
            v = [x for d in as_list(p.get(sec)) if isinstance(d, dict) for x in as_list(d.get(key))]
            if v:
                present_audit[k] = v
        null_scalars = [f["name"] for f in fields if f["type"] != "list" and f["gold"] is None]
        audit = {"null_scalar_fields": null_scalars, "other_keys_present": present_audit,
                 "needs_review": bool(null_scalars and present_audit),
                 "hand_audit": HAND_AUDIT.get(iid),
                 "verifier_audit": VERIFY_AUDIT.get(iid)}
        adjudicated = {s["field"]: {"gold": s["value_on_image"], "verdict": s["verdict"],
                                    "note": s["note"]}
                       for s in (HAND_AUDIT.get(iid) or []) + (VERIFY_AUDIT.get(iid) or [])
                       if s["verdict"] in ("wrong_null", "wrong_value")}
        report["audit_flags"] += int(audit["needs_review"])
        report["hand_audited_items"] = report.get("hand_audited_items", 0) + int(iid in HAND_AUDIT)
        report["gold_adjudicated_fields"] = report.get("gold_adjudicated_fields", 0) + len(adjudicated)

        n_tok = len(tok.encode(otext)) if tok else None
        if n_tok is not None:
            report["oracle_tokens_o200k"].append(n_tok)

        comparable = None
        if split == "test":
            cr = ccocr[meta_gt["image_id"]]
            ans = json.loads(cr["answer"])
            same = ans == p
            if not same:
                report["ccocr_answer_ne_gt_parse"].append(iid)
            cb = base64.b64decode(cr["image"])
            cim = Image.open(io.BytesIO(cb))
            cim.load()
            crel = f"comparable/images/{cr['image_name']}"
            with open(os.path.join(HERE, crel), "wb") as f:
                f.write(cb)
            if cim.size != (img_info["width"], img_info["height"]):
                report["ccocr_image_dims_differ"].append(
                    {"id": iid, "ccocr": list(cim.size),
                     "hf": [img_info["width"], img_info["height"]]})
            comparable = {
                "label": "COMPARABLE with CC-OCR Table 5 CORD field F1 (test 100, menu included)",
                "scorer": "raw/cc-ocr-github/evaluation/evaluator/kie_evaluator.py "
                          f"(AlibabaResearch/AdvancedLiterateMachinery@{CCOCR_COMMIT[:8]}, "
                          "unmodified; micro field F1 over flattened leaves)",
                "source": f"wulipc/CC-OCR@{CCOCR_REV} kie/constrained_category/CORD_100.tsv",
                "tsv_index": int(cr["index"]),
                "image_name": cr["image_name"],
                "image": crel,
                "image_info": {"format": cim.format, "width": cim.size[0],
                               "height": cim.size[1], "bytes": len(cb), "sha256": sha256(cb)},
                "prompt": cr["question"],
                "gold_json": ans,
                "gold_equals_cord_v2_gt_parse": same,
                "image_bytes_equal_cord_v2": sha256(cb) == img_info["source_sha256"],
            }
            report["ccocr_image_bytes_equal_cord_v2"] = \
                report.get("ccocr_image_bytes_equal_cord_v2", 0) + int(sha256(cb) == img_info["source_sha256"])
            labels[cr["image_name"]] = cr["answer"]
            qa_lines.append({"url": f"images/{cr['image_name']}", "prompt": cr["question"]})

        item = {
            "id": iid,
            "dataset": DATASET,
            "use_case": USE_CASE,
            "modality": "image",
            "text": None,
            "text_source": "none",
            "images": [rel],
            "fields": fields,
            "metric": "exact_norm",
            "meta": {
                "split": split,
                "cord_meta": meta_gt,
                "source": f"naver-clova-ix/cord-v2@{HF_REV} {PARQUETS[split][0]}",
                "parquet_row": i,
                "page_count": 1,
                "image": img_info,
                "holdout": iid in holdout,
                "currency": "IDR",
                "unit_hint": {f[0]: "IDR" for f in SCALARS if f[3] == "money"} | {"menu_price": "IDR"},
                "field_metrics": field_metrics,
                "field_source_keys": {**{f[0]: f"{f[1]}.{f[2]}" for f in SCALARS},
                                      **{f[0]: f"menu.{f[1]}" for f in LISTS}},
                "list_gold_semantics": "type=list: gold is the list value itself (ordered, "
                                       "multiset), not alternatives; null = key absent",
                "list_item_types": {f[0]: f[2] for f in LISTS},
                "list_gold_typed": list_typed,
                "gold_raw": gold_raw,
                "typed_gold": {"label": "our adaptation, not comparable to CC-OCR/LMDX",
                               "rules": rules, "multi_value_fields": multi,
                               "discount_sign_insensitive": True},
                "dropped_fields": dropped,
                "null_gold_audit": audit,
                "gold_adjudicated": adjudicated,  # raw gold stays in `fields` (SURVEY L7)
                "gt_parse": p,
                "oracle_text": otext,
                "oracle_text_label": "gold-word oracle: human word transcripts (valid_line) "
                                     "in row_id lines sorted by quad (y, x); NOT OCR, no "
                                     "unlabelled/distractor text; dontcare has no text",
                "oracle_text_tokens_o200k": n_tok,
                "gold_reachable_oracle": reach,
                "gold_reachable": None,  # computed after the harness OCR exists
                "comparable": comparable,
                "n_dontcare_regions": len(gt.get("dontcare") or []),
            },
        }
        items.append(item)
        report["n"] += 1
        report["by_split"][split] = report["by_split"].get(split, 0) + 1

    items.sort(key=lambda it: it["id"])
    # ---- near-duplicate receipts (same menu multiset + same typed scalar golds). Found while
    # building: 3 pairs share an identical gt_parse (two are the same photo framing), 2 more are
    # the same purchase on a second receipt. Kept (spec: all of test + dev) but grouped, so the
    # bootstrap resamples groups and effective n is stated honestly.
    def dup_key(it):
        menu = sorted((str(d.get("nm", "")).replace(" ", "").lower(), str(d.get("price", "")))
                      for d in as_list(it["meta"]["gt_parse"].get("menu")) if isinstance(d, dict))
        sc = [(f["name"], f["gold"]) for f in it["fields"] if f["type"] != "list"]
        return json.dumps([menu, sc], sort_keys=True)
    groups = {}
    for it in items:
        groups.setdefault(dup_key(it), []).append(it["id"])
    dups = sorted(sorted(v) for v in groups.values() if len(v) > 1)
    gid = {i: f"dup{n + 1}" for n, g in enumerate(dups) for i in g}
    for it in items:
        it["meta"]["dup_group"] = gid.get(it["id"])
        it["meta"]["cluster_id"] = gid.get(it["id"], it["id"])
    report["dup_groups"] = dups
    report["effective_independent_receipts"] = len(items) - sum(len(g) - 1 for g in dups)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(CMP_DIR, "label.json"), "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    with open(os.path.join(CMP_DIR, "qa.jsonl"), "w") as f:
        for q in qa_lines:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    report["holdout_n"] = len(holdout)
    report["ccocr_tsv_sha256"] = ccocr_sha
    toks = sorted(report.pop("oracle_tokens_o200k"))
    if toks:
        q = statistics.quantiles(toks, n=100, method="inclusive")
        report["oracle_tokens_o200k"] = {"n": len(toks), "p50": q[49], "p95": q[94],
                                         "max": toks[-1], "mean": round(sum(toks) / len(toks), 1)}
    with open(os.path.join(HERE, "build_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    brief = {k: v for k, v in report.items()
             if k not in ("nonnumeric_raw", "multi_value_fields", "list_value_joined")}
    print(json.dumps(brief, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
