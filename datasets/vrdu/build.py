"""Build extract-bench/datasets/vrdu/items.jsonl from raw/ (VRDU, google-research-datasets/vrdu @ 04a4df7).

Run (from anywhere; network only with --fetch):
  U="/opt/homebrew/bin/uv run --no-project --with tiktoken --with pillow python -I"
  $U extract-bench/datasets/vrdu/build.py --fetch    # 1st time: download the sampled PDFs (pinned commit, blob-SHA checked)
  $U extract-bench/datasets/vrdu/build.py            # rebuild items.jsonl + img/ from raw/ only (deterministic)

What it does:
  * Registration (FARA): ALL 300 docs of the official mixed-template (lv2) test split. The test list is identical in all
    12 FARA-lv2-mixed_template-* split files, so this is the exact set LMDX's zero-shot "Mixed Template" rows ran on.
  * Ad-buy (DeepForm): 60 of the 300 mixed-template test docs, seed 20, stratified 30 x 1-page + 30 x 2-4-page.
    Docs with >4 pages (vision cap = 4 images) are dropped and counted. meta.stratum_weight reweights to the <=4-page
    test population.
  * text = the official OCR text (dataset.jsonl ocr.text, Google Cloud Vision OCR per the VRDU paper); gold spans index it.
  * images = PDF pages 1..<=4 rendered with poppler pdftoppm as JPEG; each < 4 MiB and total <= 7.5 MiB (8 MiB cap).
  * gold = whitespace-collapsed annotated text; several distinct mentions -> list of acceptable values; entity not
    annotated in the doc -> null. Line items -> 5 parallel list fields (document order; a sub-entity that no line item
    carries -> null, not []). The untouched official
    annotations are re-read from raw/ by tools/official_eval.py (official VRDU matchers) for LMDX-comparable numbers.
"""
from __future__ import annotations

import argparse, gzip, hashlib, io, json, random, re, subprocess, sys, tempfile, urllib.parse, urllib.request
from collections import Counter
from pathlib import Path

DS = Path(__file__).resolve().parent
RAW = DS / "raw"
IMG = DS / "img"
OUT = DS / "items.jsonl"
SEED = 20
REPO_SHA = "04a4df7ac5cc694e51c48115c03fe428f0e240cc"
RAW_URL = "https://raw.githubusercontent.com/google-research-datasets/vrdu/{sha}/{path}"
TREE = RAW / "repo_tree_04a4df7.json"

CORPORA = {
    "registration": {"dir": "registration-form", "split": "FARA-lv2-mixed_template-train_10-test_300-valid_100-SD_0.json"},
    "adbuy": {"dir": "ad-buy-form", "split": "DeepForm-mixed_template-train_10-test_300-valid_100-SD_0.json"},
}
ADBUY_PER_STRATUM = 30
MAX_PAGES = 4
IMG_CAP = 4 * 1024 * 1024 - 64 * 1024        # per image, margin under 4 MiB
TOTAL_CAP = int(7.5 * 1024 * 1024)            # all images of one item, margin under 8 MiB decoded
RENDER_LADDER = [(200, 85), (150, 80), (110, 75), (90, 70)]  # (max DPI, JPEG quality), first that fits wins
MAX_LONG_SIDE_PX = 2200                       # cap DPI for oversized pages

USE_CASE = {
    "registration": "Government / regulatory forms (US DOJ FARA registration filings)",
    "adbuy": "Purchase orders / TV ad-buy orders and invoices (FCC political ad files)",
}

US_DATE = " US document: numeric dates are month-first (MM/DD/YY)."
REG_FIELDS = [  # (name, type, official match func, instructions)
    ("file_date", "date", "DateMatch",
     "The date the filing was signed or dated by the registrant (e.g. the 'Date of signature' or 'Date of report' "
     "entry), not a form-revision date or a received/stamp date." + US_DATE),
    ("foreign_principle_name", "text", "NameMatch",
     "Name of the foreign principal: the foreign government, party, company or person on whose behalf the "
     "registrant acts, as written on the form."),
    ("registrant_name", "text", "NameMatch",
     "Name of the registrant (the person or firm registered under the Foreign Agents Registration Act), as written."),
    ("registration_num", "text", "NumericalStringMatch", "The registrant's FARA registration number."),
    ("signer_name", "text", "NameMatch", "Name of the person who signed the form."),
    ("signer_title", "text", "GeneralStringMatch", "Title or position of the person who signed the form."),
]
ADBUY_FIELDS = [
    ("advertiser", "text", "GeneralStringMatch",
     "The advertiser / client buying the airtime (e.g. the campaign or committee), as written."),
    ("agency", "text", "GeneralStringMatch", "The advertising or media-buying agency that placed the order."),
    ("contract_num", "text", "NumericalStringMatch", "The contract or order number of this ad buy."),
    ("product", "text", "GeneralStringMatch",
     "The product / campaign being advertised (e.g. the 'Product' or 'Product Desc' entry)."),
    ("property", "text", "GeneralStringMatch",
     "The TV station (property) selling the airtime: station call letters or station name."),
    ("tv_address", "text", "AddressMatch",
     "The postal address of the TV station / seller printed on the order (may be a remit-to address)."),
    ("flight_from", "date", "DateMatch", "Start date of the advertising flight (the schedule's run dates)." + US_DATE),
    ("flight_to", "date", "DateMatch", "End date of the advertising flight (the schedule's run dates)." + US_DATE),
    ("gross_amount", "money", "PriceMatch", "The gross total amount of the order (before agency commission), in US dollars."),
]
LINE_FIELDS = [  # (list field name, official sub-entity, items type, official match func, instructions)
    ("line_channel", "channel", "text", "GeneralStringMatch",
     "For each line item (one row per program / spot schedule), its channel or station code as printed; one entry "
     "per line item, in document order."),
    ("line_program_desc", "program_desc", "text", "GeneralStringMatch",
     "For each line item, the program description or daypart (e.g. 'M-F 6-7a News'); one entry per line item, in "
     "document order."),
    ("line_program_start_date", "program_start_date", "date", "DateMatch",
     "For each line item, its start (air) date; one entry per line item, in document order." + US_DATE),
    ("line_program_end_date", "program_end_date", "date", "DateMatch",
     "For each line item, its end date; one entry per line item, in document order." + US_DATE),
    ("line_sub_amount", "sub_amount", "number", "PriceMatch",
     "For each line item, its dollar amount (line total); one entry per line item, in document order."),
]


_PARSE = None


def harness_parses(v: str) -> bool:
    """Date repair triggers only where the suite's own parser (harness/score.py parse_date, dayfirst=False) fails."""
    global _PARSE
    if _PARSE is None:
        sys.path.insert(0, str(DS.parent.parent))
        from harness.score import parse_date
        _PARSE = parse_date
    return _PARSE(v, False) is not None


def ws(s: str) -> str:
    """The official remove_redundant_whitespace (vrdu/match_utils.py)."""
    return " ".join(s.strip().split())


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def slug(s: str, n: int = 48) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n].strip("-")


def load_split(corpus: str) -> list[str]:
    c = CORPORA[corpus]
    sdir = RAW / c["dir"] / "few_shot-splits"
    tests = {frozenset(json.loads(f.read_text())["test"]) for f in sdir.glob("*mixed_template*.json")}
    if len(tests) != 1:
        sys.exit(f"{corpus}: expected one shared mixed-template test set across split files, found {len(tests)}")
    test = json.loads((sdir / c["split"]).read_text())["test"]
    assert len(test) == 300 and len(set(test)) == 300
    return sorted(test)


def load_records(corpus: str, names: set[str]) -> dict[str, dict]:
    out = {}
    with gzip.open(RAW / CORPORA[corpus]["dir"] / "main" / "dataset.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["filename"] in names:
                out[r["filename"]] = r
    missing = names - set(out)
    if missing:
        sys.exit(f"{corpus}: {len(missing)} test docs missing from dataset.jsonl")
    return out


def sample(recs_reg: dict, recs_ad: dict) -> tuple[list[str], list[str], dict]:
    reg = sorted(recs_reg)
    pages = {f: len(r["ocr"]["pages"]) for f, r in recs_ad.items()}
    one = sorted(f for f, p in pages.items() if p == 1)
    multi = sorted(f for f, p in pages.items() if 2 <= p <= MAX_PAGES)
    over = sorted(f for f, p in pages.items() if p > MAX_PAGES)
    rng = random.Random(SEED)
    s1 = sorted(rng.sample(one, ADBUY_PER_STRATUM))
    s2 = sorted(rng.sample(multi, ADBUY_PER_STRATUM))
    elig = len(one) + len(multi)
    info = {"adbuy_test_docs": len(pages), "adbuy_dropped_over_4_pages": len(over),
            "adbuy_eligible": elig, "adbuy_pop_1page": len(one), "adbuy_pop_2to4page": len(multi),
            "stratum_weight": {"1-page": (len(one) / elig) / 0.5, "2-4-page": (len(multi) / elig) / 0.5},
            "adbuy_page_hist_test": dict(sorted(Counter(pages.values()).items()))}
    return reg, s1 + s2, info


def pdf_rel(corpus: str, filename: str) -> str:
    return f"{CORPORA[corpus]['dir']}/main/pdfs/{filename}"


def fetch_pdfs(todo: list[tuple[str, str]]) -> None:
    tree = {t["path"]: t for t in json.loads(TREE.read_text())["tree"] if t["type"] == "blob"}
    got = 0
    for corpus, fn in todo:
        rel = pdf_rel(corpus, fn)
        dst = RAW / rel
        want = tree[rel]["sha"]
        if dst.exists() and blob_sha(dst.read_bytes()) == want:
            continue
        url = RAW_URL.format(sha=REPO_SHA, path=urllib.parse.quote(rel))
        req = urllib.request.Request(url, headers={"User-Agent": "extract-bench-vrdu-build/1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if blob_sha(data) != want:
            sys.exit(f"blob sha mismatch for {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        got += 1
    print(f"fetched {got} PDFs ({len(todo) - got} already present and verified)")


def pdf_pages(pdf: Path) -> list[tuple[float, float]]:
    out = subprocess.run(["pdfinfo", "-f", "1", "-l", "9999", str(pdf)], capture_output=True, text=True, check=True).stdout
    sizes = [(float(a), float(b)) for a, b in re.findall(r"Page\s+\d+ size:\s+([\d.]+) x ([\d.]+) pts", out)]
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    n = int(m.group(1)) if m else len(sizes)
    assert len(sizes) == n, (pdf, n, len(sizes))
    return sizes


def render(pdf: Path, n_pages: int, sizes, stem: str) -> tuple[list[str], dict]:
    from PIL import Image  # verification only
    for dpi_max, q in RENDER_LADDER:
        blobs, dpis, dims = [], [], []
        with tempfile.TemporaryDirectory() as td:
            for p in range(1, n_pages + 1):
                w, h = sizes[p - 1]
                dpi = min(dpi_max, int(MAX_LONG_SIDE_PX * 72 / max(w, h)))
                base = Path(td) / f"p{p}"
                subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p), "-r", str(dpi), "-jpeg", "-jpegopt",
                                f"quality={q}", "-singlefile", str(pdf), str(base)], check=True, capture_output=True)
                data = (base.with_suffix(".jpg")).read_bytes()
                im = Image.open(io.BytesIO(data)); im.verify()
                blobs.append(data); dpis.append(dpi); dims.append(Image.open(io.BytesIO(data)).size)
        if max(len(b) for b in blobs) <= IMG_CAP and sum(len(b) for b in blobs) <= TOTAL_CAP:
            rels = []
            for p, data in enumerate(blobs, 1):
                rel = f"img/{stem}_p{p}.jpg"
                (DS / rel).write_bytes(data)
                rels.append(rel)
            return rels, {"dpi": dpis, "jpeg_quality": q, "px": dims, "bytes": [len(b) for b in blobs],
                          "total_bytes": sum(len(b) for b in blobs)}
    raise RuntimeError(f"cannot fit {pdf} under the image caps")


def entity_gold(ments: list) -> str | list[str]:
    vals = []
    for m in ments:
        v = ws(m[0])
        if v and v not in vals:
            vals.append(v)
    return vals[0] if len(vals) == 1 else vals


# ---- declared date-gold repair (adds an acceptable alternative; the verbatim official gold always stays first) ----
NUM_DATE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-](\d{2}|\d{4})$")
RANGE_TAIL = re.compile(r"^\d{1,4}-\d{1,2}/")  # year fragment + '-' + next date, e.g. '20-01/02/20'
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}
DAY_OF = re.compile(r"^(?:this\s+)?(\d{1,2})(?:st|nd|rd|th)?\s*day\s+of\s+([A-Za-z]+)\s*[,.]?\s*,?\s*(\d{4})\W*$", re.I)
MON_D_Y = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2})[.,]\s*(\d{4})$")
D_MON_Y = re.compile(r"^(\d{1,2})\.?\s+([A-Za-z]{3,})\.?\s+(\d{4})$")


def _iso(y: int, mon: str, d: int) -> str | None:
    import datetime
    m = MONTHS.get(mon.lower()) or next((v for k, v in MONTHS.items() if len(mon) >= 3 and k.startswith(mon.lower())), None)
    if not m:
        return None
    try:
        return datetime.date(y, m, d).isoformat()
    except ValueError:
        return None


def looks_numeric_date(s: str) -> bool:
    if not NUM_DATE.match(s):
        return False
    a, b = (int(x) for x in re.split(r"[/-]", s)[:2])
    return 1 <= a <= 12 and 1 <= b <= 31


def repair_date(v: str, mention: list | None, text: str) -> tuple[str | None, str | None]:
    """Return (alternative, rule) for a date gold that is not a plain numeric date, else (None, None).
    rule 'token_expand': gold is a character slice of one OCR token such as '02/27/20-03/02/20' -> expand the span to
      the whitespace-delimited token, split ranges on '-', keep the piece overlapping the gold span most, accept only
      if it is a complete numeric date (MM/DD/YY[YY]).
    rule 'day_of_month' / 'month_d_y' / 'd_mon_y': written forms ('28th day of December 1976', 'June 11.2008',
      '5. Jul 2013') -> ISO date."""
    if looks_numeric_date(v):
        return None, None
    for rx, rule, order in ((DAY_OF, "day_of_month", "dmy"), (MON_D_Y, "month_d_y", "mdy"), (D_MON_Y, "d_mon_y", "dmy")):
        m = rx.match(v)
        if m:
            g = m.groups()
            iso = _iso(int(g[2]), g[1], int(g[0])) if order == "dmy" else _iso(int(g[2]), g[0], int(g[1]))
            if iso:
                return iso, rule
    if mention and mention[2]:
        s, e = mention[2][0][0], mention[2][-1][1]
        ts = s
        while ts > 0 and not text[ts - 1].isspace():
            ts -= 1
        te = e
        while te > s and text[te - 1].isspace():
            te -= 1
        while te < len(text) and not text[te].isspace():
            te += 1
        tok = text[ts:te]
        best, best_ov, pos = None, 0, ts
        for piece in tok.split("-"):
            ps, pe = pos, pos + len(piece)
            ov = min(pe, e) - max(ps, s)
            if ov > best_ov:
                best, best_ov = piece, ov
            pos = pe + 1
        if best and best != v and looks_numeric_date(best):
            return best, "token_expand"
    return None, None


def build_item(corpus: str, rec: dict, stem: str, tok, info: dict) -> dict:
    text = rec["ocr"]["text"]
    ann = rec["annotations"]
    scalars = {name: ments for name, ments in ann if not isinstance(name, list)}
    lines = [(name, ments[0]) for name, ments in ann if isinstance(name, list)]
    # every gold mention's offsets point at its text in the official OCR (verified here, counted in meta)
    off_ok = off_bad = 0
    for name, ments in ann:
        ents = ments[0] if isinstance(name, list) else ments
        for m in ents:
            seg = "".join(text[s:e] for s, e in m[2])
            if ws(seg) == ws(m[0]) or ws(m[0]) in ws(seg):
                off_ok += 1
            else:
                off_bad += 1
    nt = " ".join(text.split()).casefold()
    fields, reach, adapt, issues = [], {}, [], []
    spec = REG_FIELDS if corpus == "registration" else ADBUY_FIELDS
    for name, ftype, match, instr in spec:
        gold = entity_gold(scalars[name]) if name in scalars else None
        if ftype == "date" and gold is not None:
            vals = [gold] if isinstance(gold, str) else list(gold)
            if not any(harness_parses(v) for v in vals):
                for m in scalars[name]:
                    alt, rule = repair_date(ws(m[0]), m, text)
                    if alt and alt not in vals and harness_parses(alt):
                        vals.append(alt)
                        adapt.append({"field": name, "official": ws(m[0]), "added": alt, "rule": rule})
                if len(vals) > (1 if isinstance(gold, str) else len(gold)):
                    gold = vals
                else:
                    issues.append({"field": name, "issue": "date gold unparseable, no repair", "gold": gold})
        f = {"name": name, "type": ftype, "instructions": instr, "gold": gold, "official_match": match}
        if ftype == "date":
            f["dayfirst"] = False
        if ftype == "money":
            f["unit"] = "USD"
        fields.append(f)
        if gold is not None:
            reach[name] = any(g.casefold() in nt for g in ([gold] if isinstance(gold, str) else gold))
    official_lines = []
    if corpus == "adbuy":
        rows = sorted(lines, key=lambda nm: min(s for m in nm[1] for s, _ in m[2]))  # document order
        for names, ents in rows:
            official_lines.append({"order": list(names), "values": {n: ws(m[0]) for n, m in zip(names, ents)},
                                   "mentions": {n: m for n, m in zip(names, ents)}})
        for lname, sub, items, match, instr in LINE_FIELDS:
            gold = [r["values"][sub] for r in official_lines if sub in r["values"]]
            if items == "date":
                rep, n_rep, n_bad = [], 0, 0
                for r in official_lines:
                    if sub not in r["values"]:
                        continue
                    v = r["values"][sub]
                    # RANGE_TAIL (verifier fix 2026-10-23): '20-01/02/20' is the tail of '01/01/20-01/02/20'; the
                    # harness parses it, wrongly, as 2002-01-20, so it must go through the repair too.
                    if harness_parses(v) and not RANGE_TAIL.match(v):
                        rep.append(v); continue
                    alt, rule = repair_date(v, r["mentions"][sub], text)
                    if alt and harness_parses(alt):
                        rep.append(alt); n_rep += 1
                    else:
                        rep.append(v); n_bad += 1
                if n_rep:
                    adapt.append({"field": lname, "rule": "token_expand (per element)", "n_repaired": n_rep,
                                  "n_elements": len(gold)})
                    gold = [gold, rep]  # two acceptable lists: verbatim official, repaired
                if n_bad:
                    issues.append({"field": lname, "issue": "date elements unparseable after repair", "n": n_bad,
                                   "n_elements": len(rep)})
            if not gold:  # no line item carries this sub-entity -> absent -> null (common format), not []
                gold = None
            fields.append({"name": lname, "type": "list", "items": items, "instructions": instr, "gold": gold,
                           "official_match": match, "official_sub_entity": sub,
                           **({"dayfirst": False} if items == "date" else {})})
    ocr_pages = len(rec["ocr"]["pages"])
    pdf = RAW / pdf_rel(corpus, rec["filename"])
    if not pdf.exists():
        sys.exit(f"missing PDF {pdf}; run build.py --fetch once")
    sizes = pdf_pages(pdf)
    n_render = min(MAX_PAGES, len(sizes))
    images, imeta = render(pdf, n_render, sizes, stem)
    meta = {
        "corpus": "registration (FARA)" if corpus == "registration" else "ad-buy (DeepForm)",
        "split": "mixed_template test (identical in all 12 lv2/mixed split files)",
        "split_file": f"{CORPORA[corpus]['dir']}/few_shot-splits/{CORPORA[corpus]['split']}",
        "source_revision": REPO_SHA,
        "filename": rec["filename"], "doc_id": rec["filename"],
        "pdf_path": f"raw/{pdf_rel(corpus, rec['filename'])}",
        "ocr_pages": ocr_pages, "pdf_pages": len(sizes), "rendered_pages": n_render,
        "page_stratum": "1-page" if len(sizes) == 1 else "2-4-page",
        "images": imeta,
        "n_chars": len(text), "n_tokens_o200k": len(tok.encode(text, disallowed_special=())),
        "gold_reachable": {"official_ocr": reach},
        "gold_offsets_check": {"ok": off_ok, "mismatch": off_bad},
        "n_gold_scalar": sum(f["gold"] is not None for f in fields if f["type"] != "list"),
        "n_null_scalar": sum(f["gold"] is None for f in fields if f["type"] != "list"),
        "n_null_list": sum(f["gold"] is None for f in fields if f["type"] == "list"),
        "lmdx_comparable": corpus == "registration",
        "gold_adaptations": adapt,
        "gold_issues": issues,
    }
    if corpus == "registration":
        meta["template"] = rec["filename"][:-4].rsplit("_", 1)[-1]
    else:
        meta["stratum"] = meta["page_stratum"]
        meta["stratum_weight"] = round(info["stratum_weight"][meta["page_stratum"]], 6)
        meta["n_line_items"] = len(official_lines)
        meta["official_line_items"] = [{"order": r["order"], "values": r["values"]} for r in official_lines]
    return {"id": f"vrdu-{'reg' if corpus == 'registration' else 'adbuy'}-{stem}", "dataset": "vrdu",
            "use_case": USE_CASE[corpus], "modality": "text+image", "text": text, "text_source": "official_ocr",
            "images": images, "fields": fields, "metric": "exact_norm", "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download missing sampled PDFs from the pinned commit")
    args = ap.parse_args()
    import tiktoken
    tok = tiktoken.get_encoding("o200k_base")

    reg_names = load_split("registration")
    ad_names = load_split("adbuy")
    recs_reg = load_records("registration", set(reg_names))
    recs_ad = load_records("adbuy", set(ad_names))
    reg, ad, info = sample(recs_reg, recs_ad)
    print(json.dumps(info))
    todo = [("registration", f) for f in reg] + [("adbuy", f) for f in ad]
    if args.fetch:
        fetch_pdfs(todo)

    IMG.mkdir(exist_ok=True)
    for old in IMG.glob("*.jpg"):
        old.unlink()
    items, seen = [], set()
    for corpus, fn in todo:
        rec = recs_reg[fn] if corpus == "registration" else recs_ad[fn]
        h = hashlib.sha1(fn.encode()).hexdigest()[:8]
        stem = f"{slug(fn[:-4], 40)}-{h}" if corpus == "registration" else fn[:-4].split("-")[0] + f"-{h}"
        assert stem not in seen
        seen.add(stem)
        items.append(build_item(corpus, rec, stem, tok, info))
    with OUT.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    (DS / "build_info.json").write_text(json.dumps({**info, "n_items": len(items),
                                                    "n_registration": len(reg), "n_adbuy": len(ad)}, indent=1))
    print(f"wrote {len(items)} items -> {OUT}")


if __name__ == "__main__":
    main()
