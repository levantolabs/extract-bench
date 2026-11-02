"""Self-check for datasets/fatura/items.jsonl (+ adapter run over all 2,000 held-out docs,
+ a local Tesseract sanity pass for gold reachability and o200k token estimates).

Run: uv run --no-project --with pillow --with tiktoken python -I tools/selfcheck.py <dataset_dir>
Writes <dataset_dir>/tools/selfcheck.out.json and <dataset_dir>/tools/tesseract_txt/*.txt
Tesseract output is a sanity check only. It is NOT the harness OCR and NOT stored in items.jsonl.
"""
import collections, importlib.util, io, json, re, statistics, subprocess, sys, tempfile
from pathlib import Path

D = Path(sys.argv[1]).resolve()
NAME_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
METRICS = {"exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim"}
TOP = {"id", "dataset", "use_case", "modality", "text", "text_source", "images", "fields", "metric", "meta"}

out = {"errors": []}
items = []
for ln, line in enumerate(open(D / "items.jsonl"), 1):
    try:
        it = json.loads(line)
    except Exception as e:
        out["errors"].append(f"line {ln}: parse {e}")
        continue
    items.append(it)
    if set(it) != TOP:
        out["errors"].append(f"{ln}: keys {sorted(set(it) ^ TOP)}")
    if it["dataset"] != "fatura" or it["modality"] not in ("text", "image", "text+image"):
        out["errors"].append(f"{ln}: dataset/modality")
    if it["text_source"] not in ("native", "official_ocr", "none") or (it["text"] is None) != (it["text_source"] == "none"):
        out["errors"].append(f"{ln}: text/text_source")
    if it["metric"] not in METRICS:
        out["errors"].append(f"{ln}: metric")
    if not (0 < len(it["images"]) <= 4):
        out["errors"].append(f"{ln}: n images")
    for p in it["images"]:
        f = D / p
        if not f.exists():
            out["errors"].append(f"{ln}: missing {p}")
            continue
        b = f.read_bytes()
        if len(b) > 4 * 1024 * 1024 or not (b[:3] == b"\xff\xd8\xff" or b[:8] == b"\x89PNG\r\n\x1a\n"):
            out["errors"].append(f"{ln}: image size/format {p}")
    names = [f["name"] for f in it["fields"]]
    if len(set(names)) != len(names):
        out["errors"].append(f"{ln}: dup field names")
    for f in it["fields"]:
        if set(f) != {"name", "type", "instructions", "gold"}:
            out["errors"].append(f"{ln}: field keys {f.get('name')}")
        if not NAME_RX.match(f["name"]) or f["type"] not in TYPES or not f["instructions"]:
            out["errors"].append(f"{ln}: field {f['name']}/{f['type']}")
        g = f["gold"]
        vals = g if isinstance(g, list) else [g]
        for v in vals:
            if v is None and isinstance(g, list):
                out["errors"].append(f"{ln}: null inside list gold {f['name']}")
            if f["type"] == "date" and v is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                out["errors"].append(f"{ln}: date gold {v}")
            if f["type"] == "money" and v is not None and not isinstance(v, (int, float)):
                out["errors"].append(f"{ln}: money gold {v}")
ids = [i["id"] for i in items]
if len(set(ids)) != len(ids):
    out["errors"].append("duplicate ids")

out["n_items"] = len(items)
out["per_template"] = dict(collections.Counter(i["meta"]["template"] for i in items))
fs = collections.defaultdict(lambda: collections.Counter())
for it in items:
    for f in it["fields"]:
        fs[f["name"]]["null" if f["gold"] is None else ("multi" if isinstance(f["gold"], list) else "value")] += 1
    for n in it["meta"]["excluded_fields"]:
        fs[n]["excluded"] += 1
out["field_gold_counts"] = {k: dict(v) for k, v in fs.items()}
out["n_fields_total"] = sum(len(i["fields"]) for i in items)
out["n_null_gold_fields"] = sum(1 for i in items for f in i["fields"] if f["gold"] is None)
out["n_null_gold_fields_excl_po"] = sum(1 for i in items for f in i["fields"] if f["gold"] is None and f["name"] != "po_number")
out["null_by_template_field"] = dict(collections.Counter(f"T{i['meta']['template']}:{f['name']}" for i in items for f in i["fields"] if f["gold"] is None))

# ---- adapter over all 2,000 held-out docs (catches label variants outside the sample)
spec = importlib.util.spec_from_file_location("fatura_build", D / "build.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
tpls, _ = B.heldout_templates()
adapter_fail, n_docs = [], 0
for t in tpls:
    for k in range(200):
        doc = B.load_doc(t, k)
        n_docs += 1
        for name, _, _ in B.FIELDS:
            if (t, name) in B.EXCLUDE:
                continue
            try:
                g, _ = B.gold_for(name, doc, t)
                if g == "__EXCLUDE__" and name != "amount_due":
                    adapter_fail.append((t, k, name, "unexpected exclusion"))
            except AssertionError as e:
                adapter_fail.append((t, k, name, str(e)[:200]))
out["adapter_docs_checked"] = n_docs
out["adapter_failures"] = adapter_fail[:50]
out["adapter_failure_count"] = len(adapter_fail)

# ---- Tesseract sanity pass (local, free): gold reachability + o200k token estimate
import tiktoken
from PIL import Image
enc = tiktoken.get_encoding("o200k_base")
TD = D / "tools" / "tesseract_txt"
TD.mkdir(parents=True, exist_ok=True)
ver = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0]


def ocr(img_path, scale):
    im = Image.open(img_path).convert("RGB")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        im.save(tf.name)
        r = subprocess.run(["tesseract", tf.name, "-", "--psm", "3", "-l", "eng"], capture_output=True, text=True)
    return r.stdout


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def reachable(field, ocr_norm, gsrc):
    span = gsrc.get("span")
    g = field["gold"]
    cands = []
    if field["type"] == "money":
        v = g[0] if isinstance(g, list) else g
        cands = [f"{v:.2f}", f"{abs(v):.2f}", str(abs(v))]
        if span:
            cands.append(re.sub(r"\s*(usd|eur|\$)$", "", span, flags=re.I))
    elif field["type"] == "date":
        cands = [span] if span else []
    else:
        cands = g if isinstance(g, list) else [g]
    ocr_sq = re.sub(r"\s+", "", ocr_norm)
    return any(c and (norm(str(c)) in ocr_norm or re.sub(r"\s+", "", norm(str(c))) in ocr_sq) for c in cands)


reach = {1: collections.Counter(), 2: collections.Counter()}
tot = collections.Counter()
toks = {1: [], 2: []}
for it in items:
    img = D / it["images"][0]
    for sc in (1, 2):
        txt = ocr(img, sc)
        (TD / f"{it['id']}_x{sc}.txt").write_text(txt)
        toks[sc].append(len(enc.encode(txt)))
        on = norm(txt)
        for f in it["fields"]:
            if f["gold"] is None:
                continue
            ok = reachable(f, on, it["meta"]["gold_source"][f["name"]])
            reach[sc][f["name"]] += ok
            if sc == 1:
                tot[f["name"]] += 1


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


out["tesseract"] = {
    "version": ver, "psm": 3, "lang": "eng",
    "note": "sanity check only; not the harness OCR engine; text not written to items.jsonl",
    "gold_reachable_by_field_native": {k: f"{reach[1][k]}/{tot[k]}" for k in tot},
    "gold_reachable_by_field_x2_lanczos": {k: f"{reach[2][k]}/{tot[k]}" for k in tot},
    "gold_reachable_all_native": f"{sum(reach[1].values())}/{sum(tot.values())}",
    "gold_reachable_all_x2": f"{sum(reach[2].values())}/{sum(tot.values())}",
    "o200k_tokens_native": {"p50": statistics.median(toks[1]), "p95": pct(toks[1], 95), "min": min(toks[1]), "max": max(toks[1])},
    "o200k_tokens_x2": {"p50": statistics.median(toks[2]), "p95": pct(toks[2], 95), "min": min(toks[2]), "max": max(toks[2])},
}
out["ok"] = not out["errors"] and not adapter_fail
(D / "tools" / "selfcheck.out.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("null_by_template_field",)}, indent=1))
