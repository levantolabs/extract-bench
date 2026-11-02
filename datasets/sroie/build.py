#!/usr/bin/env python3
"""Build datasets/sroie/items.jsonl from raw/ (no network, no metered API).

Source: CC-OCR KIE track, SROIE subset (`kie/constrained_category/sroie2019_word_347.tsv`)
        from https://huggingface.co/datasets/wulipc/CC-OCR at revision
        c64517e92179991d509776064174776700cdd5a2 (VLMEvalKit TSV: index, image (base64),
        image_name, question, answer, category, l2-category, split).
Cross-check only (never used as item text): jsdnrs/ICDAR2019-SROIE test rows
        (datasets-server, revision bffe40c26759f3376ec2b3ae9031dbba54cd587c), whose
        Task 1/2 OCR lines are truncated at the first comma (see CARD.md).

All 347 receipts are kept (sample_n = 347 = the whole CC-OCR SROIE subset), in TSV order.
Seed 20 is used only to mark the sealed 20% holdout (SURVEY L13).

Two scoring views per item (never printed in one table):
  * `fields` (TYPED, "our adaptation"): company text, date date (ISO), address text,
    total money ("%.2f"). Gold converted from the verbatim CC-OCR gold by the
    pre-registered rules in `typed_date()` / `typed_money()` below.
  * `meta.comparable` (COMPARABLE with CC-OCR Table 5): all four keys as text, the
    CC-OCR prompt verbatim, gold verbatim; scored with the unmodified
    CC-OCR `kie_evaluator.py` (raw/cc-ocr-github/..., commit 851672c8).
    `comparable/label.json` + `comparable/qa.jsonl` are written in CC-OCR's own layout.

Run (from anywhere):
  /opt/homebrew/bin/uv run --no-project --with pillow==11.3.0 python -I \
      datasets/sroie/build.py
"""
import base64
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
from datetime import date

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
TSV = os.path.join(RAW, "cc-ocr", "sroie2019_word_347.tsv")
TSV_SHA256 = "93b5440238227b46f0eb0a4992a7ba10ece9afd1f09974d18715e91d2122755a"  # HF LFS oid
MIRROR = os.path.join(RAW, "jsdnrs-mirror", "test_text.jsonl")
IMG_DIR = os.path.join(HERE, "img")
OUT = os.path.join(HERE, "items.jsonl")
CMP_DIR = os.path.join(HERE, "comparable")
DATASET = "sroie"
USE_CASE = "Scanned receipts / tax invoices (expenses)"
SEED = 20
HOLDOUT_FRAC = 0.20
MAX_IMG_BYTES = 4 * 1024 * 1024
KEYS = ["company", "date", "address", "total"]

# TYPED view instructions (same text for every arm; written before any model output exists).
TYPED_FIELDS = [
    ("company", "text",
     "The name of the company or shop that issued the receipt, exactly as printed "
     "(usually the top line, e.g. '... SDN BHD'). Not the customer."),
    ("date", "date",
     "The date of the receipt / transaction (date only, no time), as YYYY-MM-DD. "
     "These receipts are from Malaysia: numeric dates are written day-first (DD/MM/YYYY)."),
    ("address", "text",
     "The full postal address of the company or shop as printed on the receipt, all "
     "address lines in reading order (no phone, fax, email or GST/registration numbers)."),
    ("total", "money",
     "The total amount of the receipt (the final amount payable, after tax and rounding), "
     "as a number without the currency symbol. Currency is Malaysian ringgit (RM)."),
]


# ---------------------------------------------------------------------------
# Pre-registered gold conversion rules for the TYPED view (SURVEY L2).
# DATE: Malaysia => day-first. Rules, in order:
#   1. YYYY<sep>MM<sep>DD (4-digit first)              -> year-month-day
#   2. DD<sep>MM<sep>YYYY or DD<sep>MM<sep>YY (sep / - . or space) -> day-month-year, YY -> 20YY
#   3. DD<sep?>MonthName<sep?>YYYY|YY  (e.g. "26 May 2018", "26-MAY-18", "26MAY2018")
#   4. MonthName DD, YYYY (e.g. "May 26, 2018")
#   5. DDMMYYYY / YYYYMMDD 8 digits: only if exactly one reading is a valid date in 2000-2030
# Anything else, or an invalid calendar date under the day-first reading, is NOT guessed:
# the field's typed gold is set to the list of every valid reading (if any) and the item is
# flagged in meta.typed_gold.notes for blind review; never silently re-ordered.
# MONEY: strip currency tokens (RM, MYR, $) and spaces; ',' followed by exactly 3 digits is a
#   thousands separator; a single ',' followed by 1-2 digits at the end is a decimal comma;
#   result formatted "%.2f". Unparseable -> flagged, and the typed gold becomes the
#   one-element list [verbatim string] (gold is never null on SROIE). Negative totals are
#   kept as is and flagged (informational).
# ---------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MONTHS.update({"sept": 9, "june": 6, "july": 7, "march": 3, "april": 4, "january": 1,
               "february": 2, "august": 8, "september": 9, "october": 10, "november": 11,
               "december": 12})
SEP = r"[\/\-\. ]"


def _mk(y, m, d):
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except ValueError:
        return None


def _yy(y):
    y = int(y)
    return y + 2000 if y < 100 else y


def typed_date(raw):
    """Return (iso_or_list, rule_name, note)."""
    s = raw.strip()
    m = re.fullmatch(r"(\d{4})" + SEP + r"(\d{1,2})" + SEP + r"(\d{1,2})", s)
    if m:
        iso = _mk(m.group(1), m.group(2), m.group(3))
        return (iso, "ymd", None) if iso else (None, "ymd_invalid", "invalid calendar date")
    m = re.fullmatch(r"(\d{1,2})\s*" + SEP + r"\s*(\d{1,2})\s*" + SEP + r"\s*(\d{4}|\d{2})", s)
    if m:
        d, mo, y = m.group(1), m.group(2), _yy(m.group(3))
        iso = _mk(y, mo, d)
        if iso:
            alt = _mk(y, d, mo)
            if alt and alt != iso and (len(d) == 1 or len(mo) == 1):
                # Informational only: gold stays day-first (the rule); unpadded components
                # hint at a different POS date format, so list for blind review.
                return iso, "dmy_numeric", ("ambiguous_unpadded: day-first kept by rule; "
                                            f"month-first reading would be {alt}")
            return iso, "dmy_numeric", None
        alt = _mk(y, d, mo)
        return ([alt] if alt else None), "dmy_numeric_invalid", \
            "day-first reading is not a valid date; month-first reading listed for review"
    m = re.fullmatch(r"(\d{1,2})\s*[\/\-\. ]?\s*([A-Za-z]{3,9})\.?\s*[\/\-\. ,]?\s*(\d{4}|\d{2})", s)
    if m and m.group(2).lower() in MONTHS:
        iso = _mk(_yy(m.group(3)), MONTHS[m.group(2).lower()], m.group(1))
        return (iso, "d_monname_y", None) if iso else (None, "d_monname_y_invalid", "invalid")
    m = re.fullmatch(r"([A-Za-z]{3,9})\.?\s*(\d{1,2}),?\s*(\d{4}|\d{2})", s)
    if m and m.group(1).lower() in MONTHS:
        iso = _mk(_yy(m.group(3)), MONTHS[m.group(1).lower()], m.group(2))
        return (iso, "monname_d_y", None) if iso else (None, "monname_d_y_invalid", "invalid")
    m = re.fullmatch(r"(\d{8})", s)
    if m:
        a = _mk(s[4:], s[2:4], s[:2])      # DDMMYYYY
        b = _mk(s[:4], s[4:6], s[6:])      # YYYYMMDD
        cands = [x for x in (a, b) if x and "2000" <= x[:4] <= "2030"]
        if len(cands) == 1:
            return cands[0], "8digit", None
        return (cands or None), "8digit_ambiguous", "8-digit date with 0 or 2 valid readings"
    return None, "unparsed", "no pre-registered date pattern matched"


def typed_money(raw):
    s = raw.strip()
    s2 = re.sub(r"(?i)\b(RM|MYR)\b|RM|\$", "", s).replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s2):
        s2 = s2.replace(",", "")
        rule = "thousands_comma"
    elif re.fullmatch(r"-?\d+,\d{1,2}", s2):
        s2 = s2.replace(",", ".")
        rule = "decimal_comma"
    else:
        rule = "plain" if s2 == s else "currency_stripped"
    if re.fullmatch(r"-?\d+(\.\d+)?", s2):
        return f"{float(s2):.2f}", rule, None
    return None, "unparsed", "no pre-registered money pattern matched"


# ---------------------------------------------------------------------------
# COMPARABLE view: field instructions derived only from the CC-OCR SROIE prompt sentences.
# ---------------------------------------------------------------------------
def comparable_fields(prompt, gold):
    base = ("Fill this key of the receipt's JSON schema with the information in the image. "
            "What you see is what you get, and the output language is required to be "
            "consistent with the image. If the value is pure English characters, use uppercase.")
    extra = {"date": " The date field is designed to display only the date, without including "
                     "the specific time."}
    out = []
    for k in KEYS:
        out.append({"name": k, "type": "text",
                    "instructions": f"Key '{k}'. " + base + extra.get(k, ""),
                    "gold": gold.get(k)})
    return out


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def main():
    with open(TSV, "rb") as f:
        tsv_bytes = f.read()
    got = sha256(tsv_bytes)
    if got != TSV_SHA256:
        sys.exit(f"TSV sha256 mismatch: {got} != {TSV_SHA256}")
    csv.field_size_limit(1 << 30)
    rows = list(csv.DictReader(io.StringIO(tsv_bytes.decode("utf-8")), delimiter="\t",
                               quotechar='"'))
    assert len(rows) == 347, len(rows)

    mirror = {}
    if os.path.exists(MIRROR):
        with open(MIRROR) as f:
            for line in f:
                r = json.loads(line)
                mirror[r["key"]] = r

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(CMP_DIR, exist_ok=True)
    ids = []
    for r in rows:
        stem = os.path.splitext(r["image_name"])[0]
        assert re.fullmatch(r"[A-Za-z0-9_]+", stem), stem
        ids.append(f"sroie-{stem}")
    assert len(set(ids)) == len(ids), "duplicate image names"
    holdout = set(random.Random(SEED).sample(sorted(ids), round(HOLDOUT_FRAC * len(ids))))

    items, labels, qa_lines, report = [], {}, [], {
        "n": 0, "reencoded": 0, "img_formats": {}, "date_rules": {}, "money_rules": {},
        "typed_flags": [], "mirror_gold_diff": [], "mirror_missing": [], "mirror_dims_diff": [],
        "prompts_distinct": 0}
    prompts = set()
    for r, iid in zip(rows, ids):
        name = r["image_name"]
        stem = os.path.splitext(name)[0]
        gold = json.loads(r["answer"])
        assert set(gold) == set(KEYS), (name, gold.keys())
        for k in KEYS:
            assert isinstance(gold[k], str) and gold[k].strip(), (name, k, gold[k])
        prompts.add(r["question"])

        # --- image: keep CC-OCR's exact bytes when allowed (same pixels as the published rows)
        b = base64.b64decode(r["image"])
        im = Image.open(io.BytesIO(b))
        im.load()
        fmt = im.format
        w, h = im.size
        reenc = False
        if fmt in ("JPEG", "PNG", "WEBP") and len(b) <= MAX_IMG_BYTES:
            ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
            out_b = b
        else:
            reenc = True
            ext = ".jpg"
            im2 = im.convert("RGB")
            q = 90
            while True:
                buf = io.BytesIO()
                im2.save(buf, "JPEG", quality=q)
                if buf.tell() <= MAX_IMG_BYTES or q <= 50:
                    break
                q -= 10
            out_b = buf.getvalue()
            assert len(out_b) <= MAX_IMG_BYTES
            report["reencoded"] += 1
        rel = f"img/{stem}{ext}"
        with open(os.path.join(HERE, rel), "wb") as f:
            f.write(out_b)
        report["img_formats"][fmt] = report["img_formats"].get(fmt, 0) + 1

        # --- TYPED gold
        notes = []
        d_gold, d_rule, d_note = typed_date(gold["date"])
        report["date_rules"][d_rule] = report["date_rules"].get(d_rule, 0) + 1
        if d_note:
            notes.append({"field": "date", "raw": gold["date"], "rule": d_rule, "note": d_note})
        m_gold, m_rule, m_note = typed_money(gold["total"])
        report["money_rules"][m_rule] = report["money_rules"].get(m_rule, 0) + 1
        if m_note:  # parse failure: keep the verbatim string as the only acceptable value
            notes.append({"field": "total", "raw": gold["total"], "rule": m_rule, "note": m_note})
            m_gold = [gold["total"]]
        elif m_gold.startswith("-"):
            notes.append({"field": "total", "raw": gold["total"], "rule": m_rule,
                          "note": "negative total in gold (credit/refund receipt?); kept as is"})
        if d_gold is None:  # parse failure: keep the verbatim string as the only acceptable value
            d_gold = [gold["date"]]
        if notes:
            report["typed_flags"].append({"id": iid, "notes": notes})

        fields = []
        for name_f, ftype, instr in TYPED_FIELDS:
            g = {"company": gold["company"], "address": gold["address"],
                 "date": d_gold, "total": m_gold}[name_f]
            fields.append({"name": name_f, "type": ftype, "instructions": instr, "gold": g})

        # --- mirror cross-check (gold + image dims); never used as text
        mrow = mirror.get(stem)
        mirror_info = None
        if mrow is None:
            report["mirror_missing"].append(stem)
        else:
            diff = {k: [gold[k], mrow["entities"].get(k)] for k in KEYS
                    if gold[k] != mrow["entities"].get(k)}
            if diff:
                report["mirror_gold_diff"].append({"id": iid, "diff": diff})
            dims = mrow.get("image_size") or {}
            if (dims.get("width"), dims.get("height")) != (w, h):
                report["mirror_dims_diff"].append({"id": iid, "ccocr": [w, h],
                                                   "mirror": [dims.get("width"), dims.get("height")]})
            mirror_info = {"gold_identical": not diff, "dims_identical":
                           (dims.get("width"), dims.get("height")) == (w, h)}

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
                "split": "test",
                "source": "wulipc/CC-OCR kie/constrained_category/sroie2019_word_347.tsv",
                "source_revision": "c64517e92179991d509776064174776700cdd5a2",
                "tsv_index": int(r["index"]),
                "image_name": name,
                "category": r.get("category"),
                "l2_category": r.get("l2-category"),
                "tsv_split": r.get("split"),
                "page_count": 1,
                "image": {"format": fmt, "width": w, "height": h, "bytes": len(out_b),
                          "sha256": sha256(out_b), "reencoded": reenc,
                          "source_sha256": sha256(b)},
                "holdout": iid in holdout,
                "gold_verbatim": gold,
                "typed_gold": {"date_rule": d_rule, "money_rule": m_rule, "notes": notes,
                               "label": "our adaptation, not comparable to CC-OCR"},
                "comparable": {
                    "label": "COMPARABLE with CC-OCR Table 5 SROIE field F1",
                    "scorer": "raw/cc-ocr-github/evaluation/evaluator/kie_evaluator.py "
                              "(AlibabaResearch/AdvancedLiterateMachinery@851672c8, unmodified)",
                    "prompt": r["question"],
                    "fields": comparable_fields(r["question"], gold),
                },
                "mirror_crosscheck": mirror_info,
                "gold_reachable": None,  # computed after the harness OCR exists
            },
        }
        items.append(item)
        labels[name] = r["answer"]
        qa_lines.append({"url": f"../{rel}", "prompt": r["question"]})
        report["n"] += 1

    report["prompts_distinct"] = len(prompts)
    report["holdout_n"] = len(holdout)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(CMP_DIR, "label.json"), "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    with open(os.path.join(CMP_DIR, "qa.jsonl"), "w") as f:
        for q in qa_lines:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "build_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("typed_flags", "mirror_gold_diff")}, indent=1))
    print("typed_flags:", len(report["typed_flags"]), "mirror_gold_diff:",
          len(report["mirror_gold_diff"]))


if __name__ == "__main__":
    main()
