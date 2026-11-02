"""FATURA (Zenodo 10371464 v2, FATURA2.zip) -> extract-bench items.jsonl.

Two modes:
  --fetch   range-reads only what is needed from FATURA2.zip into raw/ (no full download):
            the split files, the Annotations block (Original_Format for all 10,000 docs and
            layoutlm_HF_format for the 10 held-out templates), and the white-background JPGs
            of the sampled items. Also writes raw/zip_index.json and raw/fetch_manifest.json.
  (default) builds items.jsonl + img/ from raw/ only. Deterministic (seed 20).

Run (from anywhere; never from inside raw/):
  uv run --no-project --with remotezip python -I build.py --fetch
  uv run --no-project --with pillow python -I build.py

Sampling: Strat2 (inter-template) held-out templates = dev_inds = test_inds in
raw/Strat2_Split.txt. 15 instances per held-out template, random.Random(20), templates in
the file's order, instances drawn from Instance0..Instance199. The shipped split file gives
template ids only (no image-level dev/test partition), so all 200 instances of each held-out
template are eligible.

Gold: from Original_Format JSON "text" of each class, with a declared label-stripping adapter
(our code, see "label-stripping adapter" below). A class absent from the document's annotation
gives gold null (each case checked on the page images). Exceptions, all from the hand audit:
  - tax also reads a single "GST(n%)" class (T2, T7);
  - amount_due reads the TOTAL region when it is labelled "BALANCE DUE" (T49);
  - buyer_name reads BUYER and/or BILL_TO; when both are on the page (T49) either name is accepted;
  - EXCLUDE lists fields left out of an item because a null gold would be ambiguous on the page;
  - amount_due is left out (not scored) where no due label is printed but a TOTAL is (deviation
    from the spec's field plan, which had gold null there; see CARD.md).
The OTHER class (a full-page text) is NOT used: it does not match the v2 images (e.g. it names
a seller that is not on the page), so items carry text=null and the harness OCRs the image.
"""
import argparse, binascii, hashlib, io, json, os, random, re, struct, sys, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
IMG = HERE / "img"
URL = "https://zenodo.org/records/10371464/files/FATURA2.zip?download=1"
ZIP_MD5 = "4c9404462f22c5241eb1a290a02eb2a2"
ZIP_SIZE = 690717658
PREFIX = "invoices_dataset_final/"
SEED = 20
PER_TEMPLATE = 15
KEY = "fatura"
USE_CASE = "Invoices / AP"


# ----------------------------------------------------------------------------- sampling
def heldout_templates():
    txt = (RAW / "Strat2_Split.txt").read_text()
    m = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = [s.strip() for s in line.split("=", 1)]
        if v.startswith("["):
            m[k] = [int(x) for x in re.findall(r"\d+", v)]
        else:
            m[k] = m[v]
    assert m["test_inds"] == m["dev_inds"], "spec says Strat2 test == dev"
    assert m["dev_inds"] == [50, 7, 32, 39, 2, 12, 4, 49, 10, 25], m["dev_inds"]
    return m["dev_inds"], m


def sample_ids():
    tpls, _ = heldout_templates()
    rng = random.Random(SEED)
    out = []
    for t in tpls:
        inst = sorted(rng.sample(range(200), PER_TEMPLATE))
        out += [(t, i) for i in inst]
    return out


# ----------------------------------------------------------------------------- fetch (range reads)
def _range(url, start, end):
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "extract-bench/0.1 (research; range read)"})
    with urllib.request.urlopen(req, timeout=300) as r:
        assert r.status == 206, r.status
        return r.read()


def _member_bytes(blob, blob_start, info):
    off = info["header_offset"] - blob_start
    sig, = struct.unpack("<I", blob[off:off + 4])
    assert sig == 0x04034B50, hex(sig)
    n, m = struct.unpack("<HH", blob[off + 26:off + 30])
    d0 = off + 30 + n + m
    comp = blob[d0:d0 + info["csize"]]
    if info["method"] == 0:
        data = comp
    elif info["method"] == 8:
        data = zlib.decompressobj(-15).decompress(comp)
    else:
        raise ValueError(info["method"])
    assert len(data) == info["size"] and (binascii.crc32(data) & 0xFFFFFFFF) == info["crc"], info["name"]
    return data


def _write_member(info, data):
    rel = info["name"][len(PREFIX):]
    p = RAW / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return rel


def fetch():
    from remotezip import RemoteZip
    RAW.mkdir(exist_ok=True)
    with RemoteZip(URL) as z:
        infos = [{"name": i.filename, "size": i.file_size, "csize": i.compress_size, "crc": i.CRC,
                  "method": i.compress_type, "header_offset": i.header_offset} for i in z.infolist()]
    (RAW / "zip_index.json").write_text(json.dumps(infos))
    by = {i["name"]: i for i in infos}
    order = sorted(infos, key=lambda i: i["header_offset"])
    nxt = {a["name"]: (order[k + 1]["header_offset"] if k + 1 < len(order) else None) for k, a in enumerate(order)}
    written = []

    def fetch_one(name):
        i = by[name]
        end = (nxt[name] or ZIP_SIZE) - 1
        blob = _range(URL, i["header_offset"], end)
        return _write_member(i, _member_bytes(blob, i["header_offset"], i))

    for n in ["Strat2_Split.txt", "strat1_train.csv", "strat1_dev.csv", "strat1_test.csv"]:
        written.append(fetch_one(PREFIX + n))

    # one contiguous range over the annotation blocks (HF + COCO + Original, ~21 MB compressed)
    ann = [i for i in infos if i["name"].startswith(PREFIX + "Annotations/") and i["size"] > 0
           and i["name"].endswith(".json") and ("/Original_Format/" in i["name"] or "/layoutlm_HF_format/" in i["name"])]
    lo = min(i["header_offset"] for i in ann)
    last = max(ann, key=lambda i: i["header_offset"])
    hi = (nxt[last["name"]] or ZIP_SIZE) - 1
    print(f"annotation span {lo}-{hi} ({(hi - lo) / 1e6:.1f} MB)", file=sys.stderr)
    blob = _range(URL, lo, hi)
    tpls, _ = heldout_templates()
    for i in ann:
        if "/layoutlm_HF_format/" in i["name"]:
            m = re.search(r"Template(\d+)_Instance(\d+)", i["name"])
            if not m or int(m.group(1)) not in tpls:
                continue
        written.append(_write_member(i, _member_bytes(blob, lo, i)))

    # sampled white-background images, one range each
    for t, k in sample_ids():
        written.append(fetch_one(f"{PREFIX}images/Template{t}_Instance{k}.jpg"))
    man = {"source": URL, "zenodo_record": 10371464, "zenodo_version": "v2", "zip_md5": ZIP_MD5,
           "zip_size": ZIP_SIZE, "n_files": len(written),
           "files_sha256": {w: hashlib.sha256((RAW / w).read_bytes()).hexdigest() for w in written
                            if not w.startswith("Annotations/")},
           "note": "Annotations/ files are CRC-checked against the zip central directory on extraction."}
    (RAW / "fetch_manifest.json").write_text(json.dumps(man, indent=1))
    print("fetched", len(written), "files", file=sys.stderr)



# ----------------------------------------------------------------------------- label-stripping adapter
# FATURA region texts carry their on-page label ("Invoice Date: 14-Apr-2022",
# "SUB_TOTAL : 885.25  USD", "INVOICE ID INV52959005"). This adapter is ours (not the authors').
# The label/value patterns it relies on were enumerated over all 2,000 held-out docs with
# tools/labels_scan.py (output: tools/labels_scan.out.txt); build() asserts every gold parses.
NUMBER_LABEL = re.compile(r"^\s*invoice\s*(?:#|id|number)\s*", re.I)
DATE_RX = re.compile(r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})\b")
MONEY_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(USD|EUR|\$)?\s*$")
PCT_RX = re.compile(r"\(\s*\d+(?:\.\d+)?\s*%\s*\)")
BUYER_LABEL = re.compile(r"^\s*(?:buyer|bill[\s_]*to)\s*:?\s*", re.I)
GST_CLASS = re.compile(r"^GST\(\d+(?:\.\d+)?%\)$")


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def parse_date(text):
    from datetime import datetime
    m = DATE_RX.search(norm_ws(text))
    if not m:
        return None, None
    return datetime.strptime(m.group(0), "%d-%b-%Y").strftime("%Y-%m-%d"), m.group(0)


def parse_money(text):
    """Value after the last ':' (labels hold rates such as 'VAT (3.22%)' and '(-)')."""
    s = norm_ws(text)
    tail = s.rsplit(":", 1)[-1].replace("(-)", " ")
    tail = PCT_RX.sub(" ", tail)
    m = MONEY_RX.search(tail.strip())
    if not m:
        return None, None, None
    cur = {"USD": "USD", "EUR": "EUR", "$": "$"}.get(m.group(2) or "", None)
    return round(float(m.group(1)), 2), m.group(0).strip(), cur


def parse_name_block(text):
    """First line of a 'Buyer :' / 'Bill to:' / 'BILL_TO:' block, label removed."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    first = BUYER_LABEL.sub("", lines[0]).strip()
    if first:
        return first
    return lines[1].strip() if len(lines) > 1 else None


def region_text(doc, cls):
    v = doc.get(cls)
    if isinstance(v, dict) and isinstance(v.get("text"), str) and v["text"].strip():
        return v["text"]
    return None


# Field set from the spec. Instructions describe the field only (no hint about absence);
# null-handling is left to each system's native abstention.
FIELDS = [
    ("invoice_number", "text", "The invoice number (invoice ID) printed on the invoice, without its label."),
    ("invoice_date", "date", "The invoice issue date."),
    ("due_date", "date", "The payment due date."),
    ("po_number", "text", "The purchase order (PO) number, without its label."),
    ("seller_name", "text", "The name of the seller (the company issuing the invoice)."),
    ("buyer_name", "text", "The name of the buyer / bill-to party (the customer being invoiced), without its label."),
    ("subtotal", "money", "The subtotal: the amount before tax and discount."),
    ("tax", "money", "The tax amount charged on the invoice (the amount, not the rate)."),
    ("discount", "money", "The discount amount (the amount, not the rate)."),
    ("total", "money", "The invoice total after tax and discount."),
    ("amount_due", "money", "The amount due / balance due."),
]

# Per-template exclusions from the hand audit of the page images (CARD.md, "Gold / null audit").
# A field excluded here is left out of the item (not scored) because a null gold would be
# ambiguous on the page. Every other absent class -> gold null (verified absent on the page).
EXCLUDE = {
    (10, "tax"): "page has a per-line TAX column showing $0.00 and no tax total; null vs 0.00 is ambiguous",
    (32, "tax"): "page has a per-line TAX column showing $0.00 and no tax total; null vs 0.00 is ambiguous",
    (25, "tax"): "page shows five separate GST(n%) lines and no single tax amount",
    (39, "seller_name"): "no seller-name text, but the template logo contains the letters 'LCSR', which a reader may take as the seller",
}


def gold_for(name, doc, t):
    """Returns (gold, info). gold None = absent on the page."""
    info = {}
    if name == "invoice_number":
        raw = region_text(doc, "NUMBER")
        info.update(src_class="NUMBER" if raw else None, src_text=raw)
        if raw is None:
            return None, info
        v = NUMBER_LABEL.sub("", norm_ws(raw)).strip()
        assert v and v != norm_ws(raw), (t, raw)
        return v, info
    if name in ("invoice_date", "due_date"):
        cls = "DATE" if name == "invoice_date" else "DUE_DATE"
        raw = region_text(doc, cls)
        info.update(src_class=cls if raw else None, src_text=raw)
        if raw is None:
            return None, info
        iso, span = parse_date(raw)
        assert iso, (t, raw)
        info["span"] = span
        return iso, info
    if name == "po_number":
        raw = region_text(doc, "PO_NUMBER")
        info.update(src_class="PO_NUMBER" if raw else None, src_text=raw)
        assert raw is None, "no held-out template has PO_NUMBER; adapter needs a rule if this fires"
        return None, info
    if name == "seller_name":
        raw = region_text(doc, "SELLER_NAME")
        info.update(src_class="SELLER_NAME" if raw else None, src_text=raw)
        return (norm_ws(raw) if raw else None), info
    if name == "buyer_name":
        names, srcs = [], []
        for cls in ("BUYER", "BILL_TO"):
            raw = region_text(doc, cls)
            if raw:
                n = parse_name_block(raw)
                assert n, (t, raw)
                names.append(n)
                srcs.append({"class": cls, "text": raw})
        info.update(src_class=[s["class"] for s in srcs] or None, src_text=[s["text"] for s in srcs] or None)
        if not names:
            return None, info
        if len(names) > 1:
            info["note"] = "page shows both a 'Buyer' block and a 'BILL_TO' block with different names; either is accepted"
            return names, info
        return names[0], info
    if name in ("subtotal", "discount", "total", "tax", "amount_due"):
        if name == "tax":
            raw_cls = "TAX" if region_text(doc, "TAX") else None
            if raw_cls is None:
                gst = [c for c in doc if GST_CLASS.match(c) and region_text(doc, c)]
                if len(gst) == 1:
                    raw_cls = gst[0]
                elif len(gst) > 1:
                    raise AssertionError(f"T{t} tax: multiple GST lines {sorted(gst)}; must be listed in EXCLUDE")
        elif name == "amount_due":
            raw_cls = "AMOUNT_DUE" if region_text(doc, "AMOUNT_DUE") else None
            tot = region_text(doc, "TOTAL")
            if raw_cls is None and tot and re.match(r"^\s*(balance|amount)\s+due\b", tot, re.I):
                raw_cls = "TOTAL"
                info["note"] = "the TOTAL region is labelled 'BALANCE DUE' on the page"
            elif raw_cls is None and tot:
                # No 'amount due' label, but a TOTAL is printed: in AP practice the total is the
                # amount due, so a null gold would punish a reasonable answer. Not scored.
                info["exclude_reason"] = ("no 'amount due'/'balance due' label, but a TOTAL is printed; "
                                          "total-as-amount-due vs null is ambiguous")
                return "__EXCLUDE__", info
        else:
            raw_cls = {"subtotal": "SUB_TOTAL", "discount": "DISCOUNT", "total": "TOTAL"}[name]
            if region_text(doc, raw_cls) is None:
                raw_cls = None
        raw = region_text(doc, raw_cls) if raw_cls else None
        info.update(src_class=raw_cls, src_text=raw)
        if raw is None:
            return None, info
        v, span, cur = parse_money(raw)
        assert v is not None, (t, name, raw)
        info.update(span=span, currency=cur)
        if name == "discount":
            info["note"] = "printed as '(-) x'; +x and -x both accepted"
            return [v, -v], info
        return v, info
    raise KeyError(name)


def load_doc(t, k):
    return json.loads((RAW / "Annotations" / "Original_Format" / f"Template{t}_Instance{k}.json").read_text())


def build():
    from PIL import Image
    IMG.mkdir(exist_ok=True)
    tpls, split = heldout_templates()
    items = []
    for t, k in sample_ids():
        doc = load_doc(t, k)
        src = RAW / "images" / f"Template{t}_Instance{k}.jpg"
        im = Image.open(src)
        im.load()
        w, h = im.size
        dst_rel = f"img/Template{t}_Instance{k}.jpg"
        data = src.read_bytes()
        reencoded = False
        if len(data) > 4 * 1024 * 1024 or im.format != "JPEG":
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=90)
            data, reencoded = buf.getvalue(), True
        (HERE / dst_rel).write_bytes(data)
        fields, gmeta, excluded = [], {}, {}
        for name, ftype, instr in FIELDS:
            if (t, name) in EXCLUDE:
                excluded[name] = EXCLUDE[(t, name)]
                continue
            g, info = gold_for(name, doc, t)
            if g == "__EXCLUDE__":
                excluded[name] = info["exclude_reason"]
                continue
            fields.append({"name": name, "type": ftype, "instructions": instr, "gold": g})
            gmeta[name] = info
        items.append({
            "id": f"fatura-T{t}-I{k}",
            "dataset": KEY,
            "use_case": USE_CASE,
            "modality": "image",
            "text": None,
            "text_source": "none",
            "images": [dst_rel],
            "fields": fields,
            "metric": "exact_norm",
            "meta": {
                "split": "strat2_heldout_templates (Strat2_Split.txt: test_inds = dev_inds)",
                "template": t, "instance": k,
                "source_record": "https://zenodo.org/records/10371464 (v2, FATURA2.zip md5 " + ZIP_MD5 + ")",
                "source_image": f"{PREFIX}images/Template{t}_Instance{k}.jpg",
                "source_annotation": f"{PREFIX}Annotations/Original_Format/Template{t}_Instance{k}.json",
                "colored_variant": f"{PREFIX}colored_images/Template{t}_Instance{k}.jpg (not downloaded)",
                "page_count": 1, "image_px": [w, h], "image_bytes": len(data), "reencoded": reencoded,
                "classes_present": sorted(c for c, v in doc.items() if isinstance(v, dict) and v.get("text") and c != "OTHER"),
                "excluded_fields": excluded,
                "gold_source": gmeta,
                "synthetic": True,
                "gold_conventions": "date = ISO YYYY-MM-DD; money = number (2 dp, currency dropped; currency in gold_source); "
                                    "list gold = any listed value is correct; null = verified absent on the page",
            },
        })
    with open(HERE / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("items", len(items), file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
