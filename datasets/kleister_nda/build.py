"""Build extract-bench/datasets/kleister_nda/items.jsonl from raw/ (Kleister-NDA dev-0, all 83 documents).

Run (from anywhere; stdlib + optional tiktoken for o200k token counts):
  /opt/homebrew/bin/uv run --no-project --with tiktoken python -I extract-bench/datasets/kleister_nda/build.py

Inputs (fetched once from GitHub at the pinned commit; build.py never downloads, never executes downloaded code):
  raw/repo/{README.md,config.txt,in-header.tsv,dev-0/in.tsv.xz,dev-0/expected.tsv,documents/<md5>.pdf}
  raw/pages/tree.json   (GitHub git tree of the pinned commit; every raw file's git blob sha1 is checked against it)

Outputs: items.jsonl, alt/{tesseract,textract}/<md5>.txt (official alternative text layers), build_report.json.

Pre-registered rules (written before any model output exists):
  * n = 83 = the whole dev-0 split, in in.tsv order. Seed 20 only marks the sealed 20% holdout (SURVEY L13):
    random.Random(20).sample(sorted(ids), round(0.2 * 83)) -> meta.holdout (17 items).
  * text = in.tsv column `text_best` with the README escapes decoded (\\n, \\t, \\f, \\\\). text_best equals text_djvu
    (pdf2djvu/djvu2hocr 0.9.8 = the born-digital PDF's own text layer) on 83/83, so text_source = "native".
    The Tesseract 4.1.1 and Textract (March 2020) OCR layers go to alt/ for an OCR-noise arm (meta.alt_texts).
  * Fields (every item gets all five; a key missing from expected.tsv -> gold null):
      effective_date (date, ISO gold), jurisdiction (text), term (duration, Kleister {n}_{units} -> ISO 8601
      PnY/PnM/PnW/PnD; Clairic unit hint fixed to "months" for every item), party (list of every party),
      party_any (text; gold = any party; the Sage-expressible stand-in for the list, reported separately).
  * Party surface variants: Kleister gold drops commas ("Jda Software Group Inc." for "JDA Software Group, Inc.").
    For each gold party, acceptable values = the Kleister value (underscores -> spaces) + every surface form in
    the document text whose alphanumerics equal the gold's (case-insensitive, <=3 non-alphanumerics between
    characters). Gold-derived only; no hand edits. party (list) gold = list of acceptable lists (product of
    the per-party variants, deduped by the harness's norm_text).
"""
from __future__ import annotations

import csv, hashlib, io, itertools, json, lzma, random, re, statistics, subprocess, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
REPO = RAW / "repo"
KEY = "kleister_nda"
SEED = 20
HOLDOUT_FRAC = 0.20
COMMIT = "c2c7bf069b919bfb618b268fda2a2c079c0db316"   # applicaai/kleister-nda master HEAD, 2021-08-18
SOURCE = "https://github.com/applicaai/kleister-nda"
SAGE_LIMIT = 16384
FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
USE_CASE = "Contracts / legal (EDGAR NDAs)"
UNIT_ISO = {"year": "Y", "years": "Y", "month": "M", "months": "M", "week": "W", "weeks": "W", "day": "D", "days": "D"}

INSTR = {
    "effective_date": "The effective date of the agreement: the date from which the contract is legally binding.",
    "jurisdiction": "The US state or country whose jurisdiction (governing law) the agreement is made under. "
                    "Give the state or country name only, e.g. New York.",
    "term": "The length of the agreement's term (its duration) as stated in the document, e.g. two years.",
    "party": "Every party to the agreement (company or person names), one name per entry.",
    "party_any": "The name of one party to the agreement (a company or a person).",
}


def git_blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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


def iso_duration(kleister: str) -> str:
    m = re.fullmatch(r"(\d+)_([a-z]+)", kleister)
    if not m or m[2] not in UNIT_ISO:
        raise ValueError(f"unexpected term value {kleister!r}")
    return f"P{int(m[1])}{UNIT_ISO[m[2]]}"


def party_surface_forms(gold: str, text: str) -> list[str]:
    """Surface strings in `text` whose alphanumerics (casefolded) equal the gold's, most frequent first."""
    chars = re.findall(r"[0-9a-z]", gold.casefold())
    if not chars:
        return []
    pat = r"(?<![0-9a-z])" + r"[^0-9a-z]{0,3}".join(map(re.escape, chars)) + r"(?![0-9a-z])\.?"
    seen: dict[str, int] = {}
    for m in re.finditer(pat, text, re.I):
        s = re.sub(r"\s+", " ", m.group(0)).strip()
        seen[s] = seen.get(s, 0) + 1
    return [s for s, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))]


def dedupe_norm(vals: list[str]) -> list[str]:
    out, seen = [], set()
    for v in vals:
        k = norm_text(v)
        if k and k not in seen:
            seen.add(k)
            out.append(v)
    return out


def pdf_pages(path: Path) -> int | None:
    try:
        r = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=30)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
        return int(m[1]) if m else None
    except Exception:  # noqa: BLE001  (poppler missing -> page count unknown, reported)
        return None


def tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001
        return None


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    report: dict = {"dataset": KEY, "source": SOURCE, "commit": COMMIT}
    # ---- 1. provenance: every raw file must match the git blob sha1 in the pinned commit's tree
    tree = {e["path"]: e for e in json.loads((RAW / "pages" / "tree.json").read_text())["tree"]}
    files = ["README.md", "config.txt", "in-header.tsv", "dev-0/in.tsv.xz", "dev-0/expected.tsv"]
    prov = {}
    for rel in files:
        b = (REPO / rel).read_bytes()
        if git_blob_sha1(b) != tree[rel]["sha"]:
            sys.exit(f"blob mismatch for {rel}")
        prov[rel] = {"git_blob_sha1": tree[rel]["sha"], "sha256": sha256(b), "bytes": len(b)}
    report["provenance"] = prov
    header = (REPO / "in-header.tsv").read_text().rstrip("\n").split("\t")
    assert header == ["filename", "keys", "text_djvu", "text_tesseract", "text_textract", "text_best"], header

    csv.field_size_limit(sys.maxsize)
    with lzma.open(REPO / "dev-0" / "in.tsv.xz", "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    exp_lines = (REPO / "dev-0" / "expected.tsv").read_text(encoding="utf-8").split("\n")
    if exp_lines and exp_lines[-1] == "":
        exp_lines.pop()
    assert len(rows) == len(exp_lines) == 83, (len(rows), len(exp_lines))

    enc = tokenizer()
    report["tokenizer"] = "tiktoken o200k_base" if enc else "UNAVAILABLE (token counts null)"
    alt_dir = {"tesseract_layer": ROOT / "alt" / "tesseract", "textract_layer": ROOT / "alt" / "textract"}
    for d in alt_dir.values():
        d.mkdir(parents=True, exist_ok=True)

    ids = [f"{KEY}-{r[0].removesuffix('.pdf')}" for r in rows]
    holdout = set(random.Random(SEED).sample(sorted(ids), round(HOLDOUT_FRAC * len(ids))))

    items, flags = [], []
    stats = {"best_eq_djvu": 0, "pdf_blob_ok": 0, "pdf_md5_name_ok": 0}
    presence = {k: 0 for k in ("effective_date", "jurisdiction", "party", "term")}
    keys_col_vs_gold = {}
    tok_list, page_list, reach = [], [], {"party_any": 0, "party_values_with_surface_form": 0, "party_values": 0}
    for ln, (r, e) in enumerate(zip(rows, exp_lines), 1):
        assert len(r) == 6, (ln, len(r))
        fname, keys_col = r[0], r[1].split()
        md5 = fname.removesuffix(".pdf")
        iid = f"{KEY}-{md5}"
        text = unescape(r[5])
        stats["best_eq_djvu"] += r[5] == r[2]
        for (variant, d), col in zip(alt_dir.items(), (3, 4)):
            (d / f"{md5}.txt").write_text(unescape(r[col]), encoding="utf-8")

        pdf = REPO / "documents" / fname
        pb = pdf.read_bytes()
        blob_ok = git_blob_sha1(pb) == tree[f"documents/{fname}"]["sha"]
        md5_ok = hashlib.md5(pb).hexdigest() == md5
        stats["pdf_blob_ok"] += blob_ok
        stats["pdf_md5_name_ok"] += md5_ok
        if not blob_ok:
            sys.exit(f"PDF blob mismatch {fname}")
        pages = pdf_pages(pdf)
        if pages:
            page_list.append(pages)

        gold: dict[str, list[str]] = {}
        for kv in e.split(" "):
            if kv:
                k, v = kv.split("=", 1)
                gold.setdefault(k, []).append(v)
        assert set(gold) <= set(presence), gold.keys()
        for k in gold:
            presence[k] += 1
        kk = (" ".join(keys_col), " ".join(sorted(gold)))
        keys_col_vs_gold[kk] = keys_col_vs_gold.get(kk, 0) + 1
        for k, vs in gold.items():
            if k != "party" and len(vs) != 1:
                sys.exit(f"{iid}: {k} has {len(vs)} values")

        # ---- fields
        fields, gold_reachable, party_variants = [], {}, {}
        ed = gold.get("effective_date", [None])[0]
        if ed is not None:
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", ed), ed
        fields.append({"name": "effective_date", "type": "date", "instructions": INSTR["effective_date"],
                       "gold": ed, "dayfirst": False})
        ju = gold.get("jurisdiction", [None])[0]
        ju = ju.replace("_", " ") if ju is not None else None
        fields.append({"name": "jurisdiction", "type": "text", "instructions": INSTR["jurisdiction"], "gold": ju})
        if ju is not None:
            gold_reachable["jurisdiction"] = norm_text(ju) in norm_text(text)
        te = gold.get("term", [None])[0]
        fields.append({"name": "term", "type": "duration", "instructions": INSTR["term"],
                       "gold": iso_duration(te) if te else None, "unit": "months"})

        parties = [p.replace("_", " ") for p in gold.get("party", [])]
        per_party = []
        for p in parties:
            forms = party_surface_forms(p, text)
            reach["party_values"] += 1
            reach["party_values_with_surface_form"] += bool(forms)
            if not forms:
                flags.append({"id": iid, "field": "party", "gold": p,
                              "note": "no surface form in text_best with the same alphanumerics; gold kept as is"})
            v = dedupe_norm([p] + forms)[:4]
            party_variants[p] = v
            per_party.append(v)
        if per_party:
            combos = [list(c) for c in itertools.product(*per_party)]
            assert len(combos) <= 256, (iid, len(combos))
            party_list_gold = combos if len(combos) > 1 else combos[0]
            party_any_gold = dedupe_norm([v for vs in per_party for v in vs])
        else:
            party_list_gold, party_any_gold = None, None
        fields.append({"name": "party", "type": "list", "items": "text", "instructions": INSTR["party"],
                       "gold": party_list_gold, "headline": False,
                       "note": "list-capable arms only (Sage 2: no list type -> unscored); multiset F1"})
        fields.append({"name": "party_any", "type": "text", "instructions": INSTR["party_any"],
                       "gold": party_any_gold, "headline": False,
                       "note": "hit if the answer equals any gold party (or its surface form); reported separately"})
        gold_reachable["party_any"] = any(norm_text(v) in norm_text(text) for v in (party_any_gold or []))
        reach["party_any"] += gold_reachable["party_any"]
        for f in fields:
            assert FIELD_NAME.match(f["name"])

        ntok = len(enc.encode(text, disallowed_special=())) if enc else None
        sage_defs = json.dumps({f["name"]: {"type": f["type"], "instructions": f["instructions"]}
                                for f in fields if f["type"] != "list"})
        sage_tok = (ntok + len(enc.encode(sage_defs))) if enc else None
        if ntok is not None:
            tok_list.append(ntok)

        items.append({
            "id": iid, "dataset": KEY, "use_case": USE_CASE, "modality": "text",
            "text": text, "text_source": "native", "images": [],
            "fields": fields, "metric": "exact_norm",
            "meta": {
                "split": "dev-0", "source": SOURCE, "commit": COMMIT, "doc_id": md5,
                "source_file": f"dev-0/in.tsv.xz line {ln}", "pdf": f"raw/repo/documents/{fname}",
                "page_count": pages, "pdf_bytes": len(pb),
                "text_layer": "text_best (identical to text_djvu: pdf2djvu/djvu2hocr 0.9.8 text layer of the born-digital PDF)",
                "alt_texts": {k: f"alt/{k.split('_')[0]}/{md5}.txt" for k in alt_dir},
                "text_tokens_o200k": ntok, "sage_tokens_est": sage_tok,
                "fits_sage_16k": (sage_tok <= SAGE_LIMIT) if sage_tok is not None else None,
                "official_keys": keys_col,
                "kleister_expected": e,
                "kleister_gold": gold,
                "party_variants": party_variants,
                "gold_reachable": gold_reachable,
                "null_fields": [f["name"] for f in fields if f["gold"] is None],
                # verifier 2026-10-23: null golds on keys the official `keys` column does not ask for. Kleister
                # never asked for these, so "absent" is our inference, not an annotation -> exclude from the C7
                # abstention slice unless hand-audited.
                "null_outside_official_keys": [f["name"] for f in fields
                                               if f["gold"] is None and f["name"] not in keys_col],
                "headline_fields": ["effective_date", "jurisdiction", "term"],
                "holdout": iid in holdout,
                "comparable": {"scorer": "comparable/kleister_geval.py",
                               "metric": "GEval MultiLabel-F1:u (micro F1 over key=value labels, upper-cased)"},
            },
        })

    # ---- write
    with (ROOT / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    nulls = {f: sum(1 for it in items for x in it["fields"] if x["name"] == f and x["gold"] is None)
             for f in ("effective_date", "jurisdiction", "term", "party", "party_any")}
    report.update({
        "n_items": len(items), "n_fields_total": sum(len(it["fields"]) for it in items),
        "n_null_gold_fields": sum(nulls.values()), "null_gold_by_field": nulls,
        "gold_key_presence": presence,
        "term_values": sorted({it["meta"]["kleister_gold"]["term"][0] for it in items if "term" in it["meta"]["kleister_gold"]}),
        "n_party_values": reach["party_values"],
        "party_values_with_surface_form": reach["party_values_with_surface_form"],
        "party_any_reachable_verbatim": reach["party_any"],
        "jurisdiction_reachable_verbatim": sum(it["meta"]["gold_reachable"].get("jurisdiction", False) for it in items),
        "max_party_list_combos": max((len(x["gold"]) if x["gold"] and isinstance(x["gold"][0], list) else 1)
                                     for it in items for x in it["fields"] if x["name"] == "party"),
        "keys_column_vs_gold_keys": [{"keys_column": a, "gold_keys": b, "n": n}
                                     for (a, b), n in sorted(keys_col_vs_gold.items(), key=lambda kv: -kv[1])],
        "checks": stats,
        "holdout_n": len(holdout),
        "tokens_o200k": ({"p50": statistics.median(tok_list), "p95": round(pct(tok_list, 0.95), 1),
                          "max": max(tok_list), "min": min(tok_list), "mean": round(statistics.mean(tok_list), 1),
                          "total": sum(tok_list)} if tok_list else None),
        "n_over_sage_16k": sum(1 for it in items if it["meta"]["fits_sage_16k"] is False),
        "pages": ({"p50": statistics.median(page_list), "p95": round(pct(page_list, 0.95), 1),
                   "max": max(page_list), "min": min(page_list), "n_known": len(page_list)} if page_list else None),
        "gold_flags": flags,
        "n_null_outside_official_keys": sum(len(it["meta"]["null_outside_official_keys"]) for it in items),
    })
    (ROOT / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("n_items", "n_fields_total", "n_null_gold_fields", "null_gold_by_field",
                                             "checks", "holdout_n", "tokens_o200k", "pages", "n_over_sage_16k")}))


if __name__ == "__main__":
    main()
