# Validate items.jsonl against the extract-bench common item format; token stats (o200k). Usage: selfcheck.py <dataset_dir>
import json, os, re, sys, statistics, collections
import tiktoken
D = sys.argv[1]
enc = tiktoken.get_encoding("o200k_base")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
METRICS = {"exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim"}
KEYS = {"id", "dataset", "use_case", "modality", "text", "text_source", "images", "fields", "metric", "meta"}
errs, ids, toks, toks_q, n_fields, n_null = [], set(), [], [], 0, 0
img_sizes = {}
def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]
with open(os.path.join(D, "items.jsonl")) as f:
    lines = f.read().splitlines()
for i, ln in enumerate(lines):
    try:
        it = json.loads(ln)
    except Exception as e:
        errs.append(f"line {i}: json {e}"); continue
    if set(it) != KEYS: errs.append(f"{i}: keys {sorted(set(it) ^ KEYS)}")
    if it["id"] in ids: errs.append(f"{i}: dup id")
    ids.add(it["id"])
    if it["modality"] not in ("text", "image", "text+image"): errs.append(f"{i}: modality")
    if it["text_source"] not in ("native", "official_ocr", "none"): errs.append(f"{i}: text_source")
    if it["metric"] not in METRICS: errs.append(f"{i}: metric")
    if not isinstance(it["images"], list) or len(it["images"]) > 4: errs.append(f"{i}: images list")
    for p in it["images"]:
        fp = os.path.join(D, p)
        if not os.path.isfile(fp): errs.append(f"{i}: missing {p}"); continue
        sz = os.path.getsize(fp); img_sizes[p] = sz
        if sz > 4 * 1024 * 1024: errs.append(f"{i}: {p} > 4 MiB")
        head = open(fp, "rb").read(12)
        if not (head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8\xff") or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")):
            errs.append(f"{i}: {p} not PNG/JPEG/WebP")
    if it["text"] is None and it["text_source"] != "none": errs.append(f"{i}: text null but source")
    for fd in it["fields"]:
        n_fields += 1
        if not NAME.match(fd.get("name", "")): errs.append(f"{i}: field name {fd.get('name')}")
        if fd.get("type") not in TYPES: errs.append(f"{i}: type {fd.get('type')}")
        if "gold" not in fd: errs.append(f"{i}: no gold key")
        if fd.get("gold") is None: n_null += 1
        if not isinstance(fd.get("instructions"), str) or not fd["instructions"].strip(): errs.append(f"{i}: instructions")
    t = it["text"] or ""
    toks.append(len(enc.encode(t, disallowed_special=())))
    toks_q.append(toks[-1] + sum(len(enc.encode(fd["instructions"], disallowed_special=())) for fd in it["fields"]))
out = {"n_items": len(lines), "n_fields_total": n_fields, "n_null_gold_fields": n_null, "errors": errs[:50], "n_errors": len(errs),
       "text_o200k": {"p50": pct(toks, 50), "p95": pct(toks, 95), "max": max(toks), "mean": round(statistics.mean(toks), 1)},
       "text_plus_instructions_o200k": {"p50": pct(toks_q, 50), "p95": pct(toks_q, 95), "max": max(toks_q)},
       "n_images": len(img_sizes), "image_bytes": {"p50": pct(list(img_sizes.values()), 50), "max": max(img_sizes.values()) if img_sizes else None,
                                                   "total_mb": round(sum(img_sizes.values()) / 1e6, 1)}}
json.dump(out, open(os.path.join(D, "selfcheck.out.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
