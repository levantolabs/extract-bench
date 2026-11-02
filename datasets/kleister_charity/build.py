"""Build extract-bench/datasets/kleister_charity/items.jsonl from raw/ (Kleister-Charity dev-0).

Run (no network; stdlib + tiktoken, which is REQUIRED because the long/short strata are defined in o200k tokens):
  /opt/homebrew/bin/uv run --no-project --with tiktoken python -I extract-bench/datasets/kleister_charity/build.py

Inputs (fetched once from GitHub at the pinned commit; build.py never downloads and never executes downloaded code):
  raw/repo/{README.md,config.txt,in-header.tsv,dev-0/in.tsv.xz,dev-0/expected.tsv}
  raw/pages/tree.json          GitHub git tree of the pinned commit; every raw file's git blob sha1 is checked against it
  raw/annex_keys_dev0.tsv      OPTIONAL: git-annex key per PDF (byte size + SHA-256), from scripts/fetch_annex_keys.py

Outputs: items.jsonl, alt/{tesseract,djvu,best}/<md5>.txt (official alternative text layers, sampled docs only),
build_report.json.

Pre-registered rules (written before any model output exists):
  * Primary text = in.tsv column `text_textract` (AWS Textract, API version of 1 March 2020, per README), README
    escapes decoded (\\n, \\t, \\f, \\\\). text_source = "official_ocr". Alternative official layers go to alt/:
    tesseract (4.1.1, --oem 2 -l eng --dpi 300), djvu (pdf2djvu/djvu2hocr 0.9.8: the PDF's own text layer, empty
    for 21/440 dev-0 scans), best (README's combination: djvu unless tesseract is >= 1000 chars longer).
  * Strata (SURVEY C8), on o200k tokens of the PRIMARY (Textract) text:
      long  = tokens > 15,000 (does not fit one Sage 2 call: 16,384 cap minus field defs) -> ALL dev-0 docs taken
      short = tokens <= 15,000 -> random.Random(20).sample(sorted(md5s), 100)
    meta.stratum_weight = (dev-0 docs in stratum) / (sampled docs in stratum), for reweighting to dev-0.
  * Sealed holdout (SURVEY L13), stratified: per stratum random.Random(20).sample(sorted(md5s), round(0.2 * n)).
  * Fields: all 8 Kleister keys on every item, in a fixed order. A key missing from expected.tsv -> gold null.
    Kleister's gold is register-derived (and manually checked on dev/test), so a null here is NOT claimed to mean
    "absent from the document": field.null_semantics = "key_absent_in_kleister_gold"; no abstention claim (C7).
  * Gold: Kleister value with "_" -> " " (README: spaces and colons were replaced with underscores).
    money: "1234.00" strings, unit GBP. report_date: ISO, dayfirst true (UK documents).
    Text fields (name, number, town, postcode, street): acceptable values = the Kleister value + every surface
    form in ANY of the four official layers whose alphanumerics (casefolded) equal the gold's, with <= 3
    non-alphanumerics between characters (same rule as kleister_nda party). Only punctuation / spacing / case can
    differ, never a letter or digit; at most 4 values after harness norm_text dedupe. No hand edits.
    charity_name only: a gold ending in "Ltd." also accepts "... Limited" (and vice versa), because the Kleister
    paper (§3.2) says Limited/LTD name differences were normalised; surface forms are searched for both.
  * meta.gold_reachable[layer][field] (SURVEY C9): is some acceptable gold visible in that layer?
    text: norm_text(gold) substring of norm_text(layer); money: the amount written with or without thousands
    separators (pence only if non-zero), plus a separate 'thousands' flag for amounts stated in GBP 000;
    date: any day/month/year surface form in the layer that parses to the gold date.
"""
from __future__ import annotations

import csv, hashlib, json, lzma, math, random, re, statistics, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
REPO = RAW / "repo"
KEY = "kleister_charity"
SEED = 20
HOLDOUT_FRAC = 0.20
N_SHORT = 100
LONG_OVER = 15000                  # o200k tokens of the primary text; > this = long stratum
SAGE_LIMIT = 16384                 # Sage 2 413 threshold (content + field defs), 20 Oct spec
SAGE_WINDOW = 15000                # harness/clients.py SAGE_WINDOW (content tokens per call, minus defs)
CLAIRIC_CTX = 65536                # Clairic context (Cloudflare model page)
CLAIRIC_HARNESS_WINDOW = 45000     # harness/clients.py CLAIRIC_WINDOW
WIN_OVERLAP = 200                  # harness/common.py windows() default
COMMIT = "2309f4861cc49864ac653a7fa108efc2144400b3"   # applicaai/kleister-charity master HEAD, 2021-08-18
SOURCE = "https://github.com/applicaai/kleister-charity"
FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
USE_CASE = "Financial reports (UK charity annual reports and accounts; scanned, long, OCR text)"
LAYERS = {"textract": 4, "tesseract": 3, "djvu": 2, "best": 5}     # column index in in.tsv
ALT_LAYERS = ("tesseract", "djvu", "best")

FIELDS = [  # (name, type, instructions, extra hints) - instructions paraphrase README "Information to be extracted"
    ("charity_name", "text",
     "The name of the charitable organisation that this annual report and accounts is about.", {}),
    ("charity_number", "text",
     "The charity's registered charity number (its number in the Charity Commission register for England and "
     "Wales), not a company number.", {}),
    ("report_date", "date",
     "The reporting date of this annual report: the end date of the financial year or period that the report "
     "and accounts cover.", {"dayfirst": True}),
    ("income_annually_in_british_pounds", "money",
     "The charity's total income (total incoming resources) for the reporting year, in British pounds. If the "
     "accounts are stated in thousands of pounds, give the full amount.", {"unit": "GBP"}),
    ("spending_annually_in_british_pounds", "money",
     "The charity's total spending (total expenditure / resources expended) for the reporting year, in British "
     "pounds. If the accounts are stated in thousands of pounds, give the full amount.", {"unit": "GBP"}),
    ("address__post_town", "text",
     "The post town of the charity's own address (its registered or principal office), not the address of an "
     "auditor, bank, adviser or other organisation.", {}),
    ("address__postcode", "text",
     "The UK postcode of the charity's own address (its registered or principal office).", {}),
    ("address__street_line", "text",
     "The street line of the charity's own address (building number and street or road name), without the "
     "town or postcode.", {}),
]
TEXT_FIELDS = [f[0] for f in FIELDS if f[1] == "text"]
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                          "august", "september", "october", "november", "december"])}


def git_blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def unescape(s: str) -> str:
    """README 'Escaping special characters': \\f page break, \\n end of line, \\t tab, \\\\ backslash."""
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "f": "\f", "\\": "\\"}.get(m[1], "\\" + m[1]), s)


def norm_text(s) -> str:
    """Same as harness/score.py norm_text (copied so build.py stays standalone)."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:!?\"'`()[]{}")


def dedupe_norm(vals: list[str]) -> list[str]:
    out, seen = [], set()
    for v in vals:
        k = norm_text(v)
        if k and k not in seen:
            seen.add(k)
            out.append(v)
    return out


def surface_forms(gold: str, text: str) -> dict[str, int]:
    """Strings in `text` whose casefolded alphanumerics equal the gold's (<= 3 non-alphanumerics between chars)."""
    chars = re.findall(r"[0-9a-z]", gold.casefold())
    if not chars:
        return {}
    pat = r"(?<![0-9a-z])" + r"[^0-9a-z]{0,3}".join(map(re.escape, chars)) + r"(?![0-9a-z])"
    seen: dict[str, int] = {}
    for m in re.finditer(pat, text, re.I):
        s = re.sub(r"\s+", " ", m.group(0)).strip()
        seen[s] = seen.get(s, 0) + 1
    return seen


def money_reachable(gold: str, text: str) -> dict:
    """Is the GBP amount written in the text? exact: full amount with/without thousands separators (comma, dot,
    space; pence only when non-zero). thousands: amount / 1000 written the same way (accounts in GBP 000), only
    for whole thousands >= 100 (smaller numbers are too common to mean anything)."""
    x = float(gold)
    pence = round(abs(x) * 100) % 100

    def pat(n: int, p: int | None) -> str:
        s = str(n)
        groups = [s[max(0, i - 3):i] for i in range(len(s), 0, -3)][::-1]
        body = groups[0] + "".join(r"[,. ]?" + g for g in groups[1:])
        tail = (r"[.,]" + f"{p:02d}") if p else r"(?:[.,]00)?"
        return r"(?<![\d.,])" + body + tail + r"(?![\d])"
    exact = re.search(pat(int(abs(x)), pence), text) is not None
    thousands = None
    if pence == 0 and abs(x) % 1000 == 0 and abs(x) / 1000 >= 100:
        thousands = re.search(pat(int(abs(x) // 1000), None), text) is not None
    return {"exact": exact, "thousands": thousands}


_DATE_PATS = [
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[ .\-/]*(january|february|march|april|may|june|july|august|september|"
               r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?[ ,.\-/]*(\d{4}|\d{2})\b"),
    re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|"
               r"mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?[ .\-]*(\d{1,2})(?:st|nd|rd|th)?[ ,.\-]*(\d{4})\b"),
    re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{4}|\d{2})\b"),
]


def _mon(s: str) -> int | None:
    s = s.casefold()
    for name, i in MONTHS.items():
        if name.startswith(s[:3]):
            return i
    return None


def _yr(s: str) -> int:
    y = int(s)
    return y + 2000 if y < 100 else y


def date_reachable(gold: str, text_cf: str) -> bool:
    """Any day/month/year surface form in the (casefolded) text equal to the ISO gold date (UK day-first)."""
    gy, gm, gd = map(int, gold.split("-"))
    for m in _DATE_PATS[0].finditer(text_cf):
        if (int(m[1]), _mon(m[2]), _yr(m[3])) == (gd, gm, gy):
            return True
    for m in _DATE_PATS[1].finditer(text_cf):
        if (int(m[2]), _mon(m[1]), _yr(m[3])) == (gd, gm, gy):
            return True
    for m in _DATE_PATS[2].finditer(text_cf):
        if (int(m[1]), int(m[2]), _yr(m[3])) == (gd, gm, gy):
            return True
    return False


def n_windows(ntok: int, size: int, overlap: int = WIN_OVERLAP) -> int:
    """Number of windows harness/common.py windows() makes for a text of ntok tokens."""
    if ntok <= size:
        return 1
    return math.ceil(ntok / (size - overlap)) if ntok > 0 else 1


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(xs):
    return ({"n": len(xs), "p50": statistics.median(xs), "p95": round(pct(xs, 0.95), 1), "max": max(xs),
             "min": min(xs), "mean": round(statistics.mean(xs), 1), "total": sum(xs)} if xs else None)


def main():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"tiktoken is required (strata are defined in o200k tokens): {e}")

    def ntok(s: str) -> int:
        return len(enc.encode(s, disallowed_special=()))

    report: dict = {"dataset": KEY, "source": SOURCE, "commit": COMMIT, "tokenizer": "tiktoken o200k_base"}
    # ---- 1. provenance
    tree = {e["path"]: e for e in json.loads((RAW / "pages" / "tree.json").read_text())["tree"]}
    prov = {}
    for rel in ["README.md", "config.txt", "in-header.tsv", "dev-0/in.tsv.xz", "dev-0/expected.tsv"]:
        b = (REPO / rel).read_bytes()
        if git_blob_sha1(b) != tree[rel]["sha"]:
            sys.exit(f"blob mismatch for {rel}")
        prov[rel] = {"git_blob_sha1": tree[rel]["sha"], "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    report["provenance"] = prov
    header = (REPO / "in-header.tsv").read_text().rstrip("\n").split("\t")
    assert header == ["filename", "keys", "text_djvu", "text_tesseract", "text_textract", "text_best"], header

    csv.field_size_limit(sys.maxsize)
    with lzma.open(REPO / "dev-0" / "in.tsv.xz", "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    exp_lines = (REPO / "dev-0" / "expected.tsv").read_text(encoding="utf-8").split("\n")
    if exp_lines and exp_lines[-1] == "":
        exp_lines.pop()
    assert len(rows) == len(exp_lines) == 440, (len(rows), len(exp_lines))
    assert all(len(r) == 6 for r in rows)

    annex = {}
    ak = RAW / "annex_keys_dev0.tsv"
    if ak.exists():
        for line in ak.read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) == 4 and f[2]:
                annex[f[0]] = {"annex_key": f[1], "pdf_bytes": int(f[2]), "pdf_sha256": f[3]}

    # ---- 2. strata on the whole dev-0 split (primary = Textract)
    dev = []
    for ln, (r, e) in enumerate(zip(rows, exp_lines), 1):
        md5 = r[0].removesuffix(".pdf")
        if not re.fullmatch(r"[0-9a-f]{32}", md5):
            sys.exit(f"bad filename {r[0]}")
        prim = unescape(r[LAYERS["textract"]])
        dev.append({"ln": ln, "row": r, "exp": e, "md5": md5, "tok": ntok(prim)})
    long_ids = sorted(d["md5"] for d in dev if d["tok"] > LONG_OVER)
    short_all = sorted(d["md5"] for d in dev if d["tok"] <= LONG_OVER)
    short_ids = sorted(random.Random(SEED).sample(short_all, N_SHORT))
    picked = set(long_ids) | set(short_ids)
    # Charity-level bootstrap cluster (added by the 2026-10-23 verify pass). The paper (§3.2 "Dataset split") says
    # Kleister Charity holds several documents per charity from different years, and splits train/dev/test BY
    # charity, so dev-0 docs of one charity share name, number and address. Key = charity_number from gold; a doc
    # whose number key is absent borrows the number of a same-name (upper-cased) doc, else falls back to its name.
    def _kv(e: str, key: str):
        for tok in e.split(" "):
            if tok.startswith(key + "="):
                return tok.split("=", 1)[1]
        return None
    name2num = {}
    for d in dev:
        nm, num = _kv(d["exp"], "charity_name"), _kv(d["exp"], "charity_number")
        if nm and num:
            name2num.setdefault(nm.upper(), num)
    for d in dev:
        nm, num = _kv(d["exp"], "charity_name"), _kv(d["exp"], "charity_number")
        num = num or name2num.get((nm or "").upper())
        d["cluster"] = f"charity-{num}" if num else (f"name-{nm.upper()}" if nm else f"doc-{d['md5']}")
    weight ={"long": len(long_ids) / len(long_ids), "short": len(short_all) / len(short_ids)}
    holdout = set()
    for ids in (long_ids, short_ids):
        holdout |= set(random.Random(SEED).sample(ids, round(HOLDOUT_FRAC * len(ids))))
    report["strata"] = {
        "rule": f"o200k tokens of text_textract > {LONG_OVER} = long (all taken); <= {LONG_OVER} = short "
                f"(random.Random({SEED}).sample(sorted(md5s), {N_SHORT}))",
        "dev0_long": len(long_ids), "dev0_short": len(short_all), "sample_long": len(long_ids),
        "sample_short": len(short_ids), "stratum_weight": weight,
        "dev0_long_share": round(len(long_ids) / len(dev), 4),
        "dev0_over_15k_by_layer": {}, "dev0_tokens_textract": stats([d["tok"] for d in dev]),
    }
    for name, col in LAYERS.items():
        if name != "textract":
            report["strata"]["dev0_over_15k_by_layer"][name] = sum(ntok(unescape(d["row"][col])) > LONG_OVER for d in dev)
    report["strata"]["dev0_over_15k_by_layer"]["textract"] = len(long_ids)

    # ---- 3. items
    for lay in ALT_LAYERS:
        (ROOT / "alt" / lay).mkdir(parents=True, exist_ok=True)
    items, flags = [], []
    names = [f[0] for f in FIELDS]
    presence = {n: 0 for n in names}
    reach_tot = {lay: {n: [0, 0] for n in names} for lay in LAYERS}           # [reachable, with gold]
    money_thousands = {lay: [0, 0] for lay in LAYERS}
    tok_by = {lay: [] for lay in LAYERS}
    variant_counts = []
    for d in dev:
        if d["md5"] not in picked:
            continue
        r, md5 = d["row"], d["md5"]
        iid = f"{KEY}-{md5}"
        texts = {lay: unescape(r[col]) for lay, col in LAYERS.items()}
        text = texts["textract"]
        for lay in ALT_LAYERS:
            (ROOT / "alt" / lay / f"{md5}.txt").write_text(texts[lay], encoding="utf-8")
        toks = {lay: (d["tok"] if lay == "textract" else ntok(t)) for lay, t in texts.items()}
        for lay in LAYERS:
            tok_by[lay].append(toks[lay])

        gold: dict[str, list[str]] = {}
        for kv in d["exp"].split(" "):
            if kv:
                k, v = kv.split("=", 1)
                gold.setdefault(k, []).append(v)
        if not set(gold) <= set(names):
            sys.exit(f"{iid}: unexpected keys {set(gold) - set(names)}")
        for k, vs in gold.items():
            if len(vs) != 1:
                sys.exit(f"{iid}: {k} has {len(vs)} values")
            presence[k] += 1
        if r[1].split() != sorted(names):
            sys.exit(f"{iid}: keys column {r[1]!r}")

        cf = {lay: t.casefold() for lay, t in texts.items()}
        nt = {lay: norm_text(t) for lay, t in texts.items()}
        fields, reach, variants = [], {lay: {} for lay in LAYERS}, {}
        for name, ftype, instr, extra in FIELDS:
            kv = gold.get(name, [None])[0]
            f = {"name": name, "type": ftype, "instructions": instr, **extra}
            if kv is None:
                f.update(gold=None, null_semantics="key_absent_in_kleister_gold",
                         note="absent key in expected.tsv (register-derived gold); not claimed as abstention")
                fields.append(f)
                continue
            if ftype == "text":
                g = kv.replace("_", " ")
                bases = [g]
                if name == "charity_name":   # paper §3.2: Kleister normalised "Limited" <-> "LTD" name differences
                    m = re.fullmatch(r"(.*\S)\s+(Ltd\.?|Limited)", g, re.I)
                    if m:
                        bases.append(f"{m[1]} {'Limited' if m[2].lower().startswith('ltd') else 'Ltd.'}")
                forms: dict[str, int] = {}
                for b in bases:
                    for t in texts.values():
                        for s, c in surface_forms(b, t).items():
                            forms[s] = forms.get(s, 0) + c
                ranked = [s for s, _ in sorted(forms.items(), key=lambda kv2: (-kv2[1], kv2[0]))]
                acc = dedupe_norm(bases + ranked)[:4]
                variants[name] = acc
                variant_counts.append(len(acc))
                f["gold"] = acc if len(acc) > 1 else acc[0]
                for lay in LAYERS:
                    reach[lay][name] = any(norm_text(a) in nt[lay] for a in acc)
            elif ftype == "money":
                if not re.fullmatch(r"-?\d+\.\d\d", kv):
                    sys.exit(f"{iid}: money {kv!r}")
                f["gold"] = kv
                for lay in LAYERS:
                    mr = money_reachable(kv, texts[lay])
                    reach[lay][name] = mr["exact"]
                    if mr["thousands"] is not None:
                        reach[lay][name + "__thousands"] = mr["thousands"]
                        money_thousands[lay][0] += mr["thousands"] and not mr["exact"]
                        money_thousands[lay][1] += 1
            elif ftype == "date":
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", kv):
                    sys.exit(f"{iid}: date {kv!r}")
                f["gold"] = kv
                for lay in LAYERS:
                    reach[lay][name] = date_reachable(kv, cf[lay])
            fields.append(f)
            for lay in LAYERS:
                reach_tot[lay][name][0] += bool(reach[lay].get(name))
                reach_tot[lay][name][1] += 1
        for f in fields:
            assert FIELD_NAME.match(f["name"]), f["name"]

        defs = json.dumps({f["name"]: {"type": f["type"], "instructions": f["instructions"]} for f in fields})
        defs_tok = ntok(defs)
        sage_tok = toks["textract"] + defs_tok
        stratum = "long" if md5 in long_ids else "short"
        pdf = annex.get(r[0], {})
        items.append({
            "id": iid, "dataset": KEY, "use_case": USE_CASE, "modality": "text",
            "text": text, "text_source": "official_ocr", "images": [],
            "fields": fields, "metric": "exact_norm",
            "meta": {
                "split": "dev-0", "source": SOURCE, "commit": COMMIT, "doc_id": md5,
                "source_file": f"dev-0/in.tsv.xz line {d['ln']}", "pdf_filename": r[0],
                "pdf_bytes": pdf.get("pdf_bytes"), "pdf_sha256": pdf.get("pdf_sha256"),
                "pdf_annex_key": pdf.get("annex_key"),
                "page_count": None,
                "page_count_note": "UNKNOWN: PDFs not downloaded (text-only set); no layer carries reliable page "
                                   "breaks (Textract none, Tesseract \\f count disagrees with paper's ~22 pages/doc)",
                "text_layer": "text_textract (AWS Textract, API version 2020-03-01, per README)",
                "alt_texts": {f"{lay}_layer": f"alt/{lay}/{md5}.txt" for lay in ALT_LAYERS},
                "stratum": stratum, "stratum_weight": weight[stratum],
                "text_tokens_o200k": toks["textract"],
                "alt_tokens_o200k": {f"{lay}_layer": toks[lay] for lay in ALT_LAYERS},
                "sage_tokens_est": sage_tok, "fits_sage_16k": sage_tok <= SAGE_LIMIT,
                "sage_windows_est": n_windows(toks["textract"], max(1000, SAGE_WINDOW - defs_tok)),
                "fits_clairic_65k_est": toks["textract"] <= CLAIRIC_CTX,
                "clairic_harness_windows_est": n_windows(toks["textract"], CLAIRIC_HARNESS_WINDOW),
                "clairic_w15k_windows_est": n_windows(toks["textract"], SAGE_WINDOW),
                "official_keys": r[1].split(),
                "kleister_expected": d["exp"],
                "kleister_gold": gold,
                "gold_variants": variants,
                "gold_reachable": reach["textract"],
                "gold_reachable_by_layer": {("primary_textract" if lay == "textract" else f"{lay}_layer"): reach[lay]
                                            for lay in LAYERS},
                "null_fields": [f["name"] for f in fields if f["gold"] is None],
                "null_semantics": "absent key in Kleister gold; not evidence the value is absent from the document",
                "headline_fields": names,
                "holdout": md5 in holdout,
                "charity_cluster": d["cluster"],
                "charity_cluster_note": "use as the bootstrap cluster: dev-0 has several yearly reports per charity "
                                        "(paper §3.2); doc_id clusters only by document",
                "comparable": {"scorer": "comparable/kleister_geval.py",
                               "metric": "GEval MultiLabel-F1:u (micro F1 over key=value labels, upper-cased)"},
            },
        })

    with (ROOT / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    nulls = {n: sum(1 for it in items for x in it["fields"] if x["name"] == n and x["gold"] is None) for n in names}
    dev_absent = {n: sum(1 for d in dev if f"{n}=" not in d["exp"]) for n in names}
    by_stratum = {}
    for s in ("long", "short"):
        its = [it for it in items if it["meta"]["stratum"] == s]
        by_stratum[s] = {"n": len(its), "holdout": sum(it["meta"]["holdout"] for it in its),
                         "tokens_textract": stats([it["meta"]["text_tokens_o200k"] for it in its]),
                         "sage_windows": stats([it["meta"]["sage_windows_est"] for it in its]),
                         "null_gold_fields": sum(len(it["meta"]["null_fields"]) for it in its)}
    report.update({
        "n_items": len(items), "n_fields_total": sum(len(it["fields"]) for it in items),
        "n_null_gold_fields": sum(nulls.values()), "null_gold_by_field": nulls,
        "dev0_absent_by_field": dev_absent, "gold_key_presence_in_sample": presence,
        "holdout_n": len(holdout), "by_stratum": by_stratum,
        "charity_clusters": {
            "n_clusters": len({it["meta"]["charity_cluster"] for it in items}),
            "n_items_in_multi_doc_clusters": sum(
                1 for it in items
                if sum(1 for o in items if o["meta"]["charity_cluster"] == it["meta"]["charity_cluster"]) > 1),
            "n_clusters_split_across_holdout": sum(
                1 for c in {it["meta"]["charity_cluster"] for it in items}
                if len({it["meta"]["holdout"] for it in items if it["meta"]["charity_cluster"] == c}) > 1)},
        "tokens_o200k_by_layer": {lay: stats(v) for lay, v in tok_by.items()},
        "n_over_sage_16k": sum(1 for it in items if not it["meta"]["fits_sage_16k"]),
        "n_over_clairic_65k_est": sum(1 for it in items if not it["meta"]["fits_clairic_65k_est"]),
        "n_over_clairic_harness_45k": sum(1 for it in items if it["meta"]["clairic_harness_windows_est"] > 1),
        "dev0_over_clairic_65k_est": sum(d["tok"] > CLAIRIC_CTX for d in dev),
        "dev0_over_45k": sum(d["tok"] > CLAIRIC_HARNESS_WINDOW for d in dev),
        "sage_windows_total": sum(it["meta"]["sage_windows_est"] for it in items),
        "gold_reachable_rate_by_layer": {
            lay: {n: {"reachable": a, "with_gold": b, "rate": round(a / b, 4) if b else None}
                  for n, (a, b) in reach_tot[lay].items()} for lay in LAYERS},
        "money_only_in_thousands_by_layer": {lay: {"n": a, "of_whole_thousand_golds": b}
                                             for lay, (a, b) in money_thousands.items()},
        "text_gold_variants": {"n_text_golds": len(variant_counts),
                               "with_extra_surface_forms": sum(c > 1 for c in variant_counts),
                               "max_values": max(variant_counts) if variant_counts else None},
        "pdf_annex_keys_found": sum(1 for it in items if it["meta"]["pdf_bytes"] is not None),
        "pdf_bytes_sample_total": sum(it["meta"]["pdf_bytes"] or 0 for it in items),
        "gold_flags": flags,
    })
    (ROOT / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("n_items", "n_fields_total", "n_null_gold_fields", "null_gold_by_field",
                                             "holdout_n", "n_over_sage_16k", "n_over_clairic_65k_est",
                                             "n_over_clairic_harness_45k", "sage_windows_total")}))
    print(json.dumps(report["strata"]))
    print(json.dumps(report["by_stratum"]))


if __name__ == "__main__":
    main()
