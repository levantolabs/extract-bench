"""DocVQA single-page val -> extract-bench items.jsonl (common item format).

Sources (pinned HF revisions, free downloads, no loading scripts executed):
  * pixparse/docvqa-single-page-questions @ 33136ef456fa5a3fe68568d6e31dda4eeff95b9b
      -> question, answers, official commercial OCR (`ocr_results`), image filename, doc ids
  * lmms-lab-encoder/DocVQA (config DocVQA) @ 539088ef8a8ada01ac8e2e6d4e372586748a265e
      -> question_types (incl. 'handwritten') and the page PNG bytes

Stages (run with: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with pyarrow --with fsspec --with aiohttp --with requests
        --with pillow --with tiktoken --with numpy python -I build.py <stage>; `build` also needs the local `tesseract` binary):
  fetch-meta    network: non-image columns of both mirrors' val split -> raw/{pixparse,lmms}_val_meta.parquet
  fetch-images  network: only the lmms row groups holding a sampled page; byte ranges of the image column are
                range-read into a temporary sparse parquet, the sampled pages' PNG bytes are written to
                raw/lmms_png/<docId>.png (original bytes, untouched) and the temp file is deleted
  build         offline (default): re-creates items.jsonl + img/ + build_report.json from raw/ only
                Alignment checks (abort on failure): mirror metadata identical; PNG sha256 == manifest; PNG size == OCR
                page size; per-page content check = shift-profile peak OR tesseract word recall OR checks/manual_verified.json.

Sampling (seed 20, deterministic; all candidate lists sorted by questionId before drawing):
  * Population: the 5,349 val questions present in both mirrors (join on questionId; question text, answer set,
    docId, UCSF id/page and image filename must agree, else abort).
  * primary_type: 'handwritten' if the question carries that tag, else its rarest tag (val-wide tag frequency).
  * Stratum P (240): proportional allocation (largest remainder) of 240 over primary_type, SRS within each.
  * Stratum H (110): SRS of 110 from the handwritten questions not already drawn into P.
  * weight_val: post-stratification weight on the handwritten flag (sum over the 350 items = 5,349), for quoting a
    val-representative aggregate over all 350 items. P alone is the plain proportional sample.
"""
import collections, hashlib, io, json, math, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
KEY = "docvqa"
SEED = 20
N_PROP, N_HW = 240, 110
PIX_REV = "33136ef456fa5a3fe68568d6e31dda4eeff95b9b"
LMM_REV = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
PIX = f"https://huggingface.co/datasets/pixparse/docvqa-single-page-questions/resolve/{PIX_REV}/data/validation-%05d-of-00004.parquet"
LMM = f"https://huggingface.co/datasets/lmms-lab-encoder/DocVQA/resolve/{LMM_REV}/DocVQA/validation-%05d-of-00006.parquet"
MAX_IMG_BYTES = 4 * 1024 * 1024
UA = {"User-Agent": "extract-bench-docvqa/1.0 (research; free HF download)"}
USE_CASE = "Read a value off a scanned business page (forms, tables, letters); handwriting"


# ---------------------------------------------------------------- helpers
def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_alnum(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())


def load_meta():
    import pyarrow.parquet as pq
    pix = pq.read_table(os.path.join(RAW, "pixparse_val_meta.parquet")).to_pylist()
    lmm = pq.read_table(os.path.join(RAW, "lmms_val_meta.parquet")).to_pylist()
    return pix, lmm


def join_mirrors(pix, lmm):
    """Join on questionId; every other shared key must agree, else abort."""
    P = {int(r["question_id"]): r for r in pix}
    L = {int(r["questionId"]): r for r in lmm}
    if len(P) != len(pix) or len(L) != len(lmm):
        sys.exit("ABORT: duplicate questionIds inside a mirror")
    if set(P) != set(L):
        sys.exit(f"ABORT: questionId sets differ: only-pix={len(set(P) - set(L))} only-lmms={len(set(L) - set(P))}")
    rows, answer_order_diffs = [], 0
    for q in sorted(P):
        p, l = P[q], L[q]
        om = p["other_metadata"]
        checks = {
            "question": p["question"] == l["question"],
            "answers_set": sorted(p["answers"]) == sorted(l["answers"]),
            "docId": om["doc_id"] == l["docId"],
            "ucsf_document_id": om["ucsf_document_id"] == l["ucsf_document_id"],
            "ucsf_document_page_no": om["ucsf_document_page_no"] == l["ucsf_document_page_no"],
            "image_filename": om["image"] == l["image"]["path"] == p["image"]["path"],
            "split": p["data_split"] == l["data_split"] == "val",
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            sys.exit(f"ABORT: mirror mismatch on questionId {q}: {bad}")
        answer_order_diffs += list(p["answers"]) != list(l["answers"])
        rows.append({"qid": q, "pix": p, "lmm": l})
    return rows, answer_order_diffs


def ocr_text(ocr):
    return "\n".join(line["text"] for line in (ocr.get("lines") or []))


def largest_remainder(counts, n):
    tot = sum(counts.values())
    quota = {k: n * v / tot for k, v in counts.items()}
    alloc = {k: int(math.floor(q)) for k, q in quota.items()}
    left = n - sum(alloc.values())
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[:left]:
        alloc[k] += 1
    return alloc, quota


def draw_sample(rows):
    tagfreq = collections.Counter(t for r in rows for t in (r["lmm"]["question_types"] or []))

    def primary(tags):
        tags = list(tags or [])
        if "handwritten" in tags:
            return "handwritten"
        if not tags:
            return "(none)"
        return sorted(tags, key=lambda t: (tagfreq[t], t))[0]

    for r in rows:
        r["primary_type"] = primary(r["lmm"]["question_types"])
        r["handwritten"] = "handwritten" in (r["lmm"]["question_types"] or [])
    strata = collections.defaultdict(list)
    for r in rows:
        strata[r["primary_type"]].append(r)
    alloc, quota = largest_remainder({k: len(v) for k, v in strata.items()}, N_PROP)
    rng = random.Random(SEED)
    chosen_p = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["qid"])
        chosen_p += rng.sample(pool, alloc[k])
    p_ids = {r["qid"] for r in chosen_p}
    hw_pool = sorted((r for r in rows if r["handwritten"] and r["qid"] not in p_ids), key=lambda r: r["qid"])
    if len(hw_pool) < N_HW:
        sys.exit(f"ABORT: only {len(hw_pool)} handwritten questions left for the H stratum")
    chosen_h = rng.sample(hw_pool, N_HW)
    # order: P (seeded shuffle) then H (seeded shuffle) -> any prefix of the P block is a proportional val sample
    chosen_p.sort(key=lambda r: r["qid"]); rng.shuffle(chosen_p)
    chosen_h.sort(key=lambda r: r["qid"]); rng.shuffle(chosen_h)
    for r in chosen_p:
        r["stratum"] = "proportional"
    for r in chosen_h:
        r["stratum"] = "handwritten_extra"
    sample = chosen_p + chosen_h
    n_hw_pop = sum(r["handwritten"] for r in rows)
    n_hw_s = sum(r["handwritten"] for r in sample)
    w_hw = n_hw_pop / n_hw_s
    w_non = (len(rows) - n_hw_pop) / (len(sample) - n_hw_s)
    for r in sample:
        r["weight_val"] = w_hw if r["handwritten"] else w_non
    design = {
        "population_questions": len(rows),
        "population_docs": len({r["lmm"]["docId"] for r in rows}),
        "tag_frequency_val": dict(tagfreq.most_common()),
        "primary_type_counts_val": {k: len(v) for k, v in sorted(strata.items())},
        "proportional_allocation": dict(sorted(alloc.items())),
        "proportional_quota": {k: round(v, 3) for k, v in sorted(quota.items())},
        "handwritten_pool_for_H": len(hw_pool),
        "handwritten_val": n_hw_pop,
        "handwritten_in_sample": n_hw_s,
        "weight_handwritten": w_hw,
        "weight_non_handwritten": w_non,
    }
    return sample, design


# ---------------------------------------------------------------- stage: fetch-meta
def fetch_meta():
    import fsspec, pyarrow as pa, pyarrow.parquet as pq
    fs = fsspec.filesystem("http", headers=UA)
    os.makedirs(RAW, exist_ok=True)

    def read(u, cols):
        with fs.open(u, "rb", block_size=2 ** 16, cache_type="none") as f:
            return pq.ParquetFile(f, pre_buffer=False).read(columns=cols)

    for name, pat, n, cols in (
        ("pixparse", PIX, 4, ["image.path", "question_id", "question", "answers", "data_split", "ocr_results", "other_metadata"]),
        ("lmms", LMM, 6, ["questionId", "question", "question_types", "image.path", "docId", "ucsf_document_id",
                          "ucsf_document_page_no", "answers", "data_split"]),
    ):
        parts = []
        for i in range(n):
            t = read(pat % i, cols)
            t = t.append_column("src_file", pa.array([pat % i] * t.num_rows))
            t = t.append_column("src_row", pa.array(list(range(t.num_rows)), pa.int64()))
            parts.append(t)
            print(name, i, t.num_rows, flush=True)
        pq.write_table(pa.concat_tables(parts), os.path.join(RAW, f"{name}_val_meta.parquet"), compression="zstd")


# ---------------------------------------------------------------- stage: fetch-images
def _get_range(url, start, end, tries=8):
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, headers={**UA, "Range": f"bytes={start}-{end}"}, timeout=(30, 300))
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
            err = f"status {r.status_code} len {len(r.content)}"
        except Exception as e:  # network hiccup -> retry
            err = repr(e)
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"range {start}-{end} of {url} failed: {err}")


def fetch_images(workers=16, piece=4 * 1024 * 1024):
    import fsspec, pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pix, lmm = load_meta()
    rows, _ = join_mirrors(pix, lmm)
    sample, _ = draw_sample(rows)
    need_docs = {r["lmm"]["docId"] for r in sample}
    outdir = os.path.join(RAW, "lmms_png")
    os.makedirs(outdir, exist_ok=True)
    have = {int(f[:-4]) for f in os.listdir(outdir) if f.endswith(".png")}
    need_docs -= have
    print("sampled docs", len({r['lmm']['docId'] for r in sample}), "still needed", len(need_docs), flush=True)
    if not need_docs:
        return
    # doc -> candidate (file, rowgroup); lmms val row groups are 100 rows each (checked against the footer below)
    fs = fsspec.filesystem("http", headers=UA)
    footers = {}
    by_file = collections.defaultdict(list)
    for r in lmm:
        by_file[r["src_file"]].append(r)
    plan = collections.defaultdict(set)  # (file, rg) -> docs wanted
    cover = collections.defaultdict(set)
    for u, rs in by_file.items():
        with fs.open(u, "rb", block_size=2 ** 16, cache_type="none") as f:
            md = pq.ParquetFile(f, pre_buffer=False).metadata
        bounds, s = [], 0
        for g in range(md.num_row_groups):
            n = md.row_group(g).num_rows
            bounds.append((s, s + n)); s += n
        footers[u] = md
        for r in rs:
            g = next(i for i, (a, b) in enumerate(bounds) if a <= r["src_row"] < b)
            if r["docId"] in need_docs:
                cover[(u, g)].add(r["docId"])
    # greedy set cover over row groups
    left = set(need_docs)
    while left:
        best = max(cover, key=lambda k: (len(cover[k] & left), k))
        plan[best] = cover[best] & left
        left -= cover[best]
    tot = 0
    jobs = []
    for (u, g), docs in sorted(plan.items()):
        md = footers[u]
        rg = md.row_group(g)
        for c in range(rg.num_columns):
            cc = rg.column(c)
            if cc.path_in_schema in ("image.bytes", "image.path", "docId", "questionId"):
                start = cc.dictionary_page_offset if cc.has_dictionary_page else cc.data_page_offset
                jobs.append((u, g, start, start + cc.total_compressed_size - 1))
                tot += cc.total_compressed_size
    print(f"row groups to read: {len(plan)}; bytes to fetch ~{tot / 1e6:.1f} MB", flush=True)
    tmpdir = os.path.join(HERE, ".tmp_sparse")
    os.makedirs(tmpdir, exist_ok=True)
    # sparse local copies: footer + needed column chunks at their original offsets
    local = {}
    for u in {u for (u, g) in plan}:
        size = int(fs.info(u)["size"])
        p = os.path.join(tmpdir, os.path.basename(u))
        with open(p, "wb") as f:
            f.truncate(size)
        tail = _get_range(u, size - 8, size - 1)
        flen = int.from_bytes(tail[:4], "little")
        foot = _get_range(u, size - 8 - flen, size - 1)
        fd = os.open(p, os.O_WRONLY)
        os.pwrite(fd, foot, size - 8 - flen)
        os.close(fd)
        local[u] = p
    pieces = []
    for (u, g, a, b) in jobs:
        s = a
        while s <= b:
            e = min(b, s + piece - 1)
            pieces.append((u, s, e)); s = e + 1
    done_bytes, t0 = 0, time.time()

    def work(job):
        u, s, e = job
        data = _get_range(u, s, e)
        fd = os.open(local[u], os.O_WRONLY)
        os.pwrite(fd, data, s)
        os.close(fd)
        return len(data)

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, j) for j in pieces]
        for i, f in enumerate(as_completed(futs)):
            done_bytes += f.result()
            if i % 10 == 0 or i == len(futs) - 1:
                el = time.time() - t0
                print(f"  {i + 1}/{len(futs)} pieces, {done_bytes / 1e6:.1f} MB, {done_bytes / 1e6 / max(el, 1e-9):.2f} MB/s", flush=True)
    manifest_p = os.path.join(outdir, "manifest.json")
    manifest = json.load(open(manifest_p)) if os.path.exists(manifest_p) else {}
    for (u, g), docs in sorted(plan.items()):
        t = pq.ParquetFile(local[u]).read_row_group(g, columns=["questionId", "docId", "image"]).to_pylist()
        for r in t:
            d = r["docId"]
            if d in docs and str(d) not in manifest:
                b = r["image"]["bytes"]
                with open(os.path.join(outdir, f"{d}.png"), "wb") as f:
                    f.write(b)
                manifest[str(d)] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "lmms_file": u,
                                    "lmms_row_group": g, "lmms_questionId": r["questionId"], "image_path": r["image"]["path"]}
    json.dump(manifest, open(manifest_p, "w"), indent=1, sort_keys=True)
    for p in local.values():
        os.remove(p)
    os.rmdir(tmpdir)
    missing = {r["lmm"]["docId"] for r in sample} - {int(k) for k in manifest}
    print("missing after fetch:", sorted(missing), flush=True)


# ---------------------------------------------------------------- stage: build (offline)
SHIFTS = list(range(-60, 61, 2))
COMMON_DF = 0.05     # tokens present in >= 5% of sampled pages' OCR are ignored by the tesseract check
TESS_MIN_RECALL = 0.25
SHIFT_MAX_DY = 8


def shift_profile_peak(img, ocr):
    """Content alignment test 1 (image <-> pixparse OCR). Mean darkness inside the OCR word boxes is computed with the
    boxes shifted vertically by -60..+60 px; on an aligned page the profile peaks at ~0 (boxes sit on the text lines).
    Returns the dy of the peak, or None if the page has no word boxes. On wrong images the peak lands within 8 px
    only ~6% of the time (control in build_report.json)."""
    import numpy as np
    a = 255.0 - np.asarray(img.convert("L"), dtype=np.float64)
    H, W = a.shape
    integ = np.zeros((H + 1, W + 1)); integ[1:, 1:] = a.cumsum(0).cumsum(1)
    bx = []
    for line in ocr.get("lines") or []:
        for w in line.get("words") or []:
            b = w.get("bounding_box") or []
            if len(b) == 8:
                xs, ys = b[0::2], b[1::2]
                bx.append((max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))))
    bx = np.array([b for b in bx if b[2] > b[0] and b[3] > b[1]], dtype=np.int64)
    if len(bx) == 0:
        return None
    prof = []
    for dy in SHIFTS:
        y0 = np.clip(bx[:, 1] + dy, 0, H); y1 = np.clip(bx[:, 3] + dy, 0, H)
        x0, x1 = bx[:, 0], bx[:, 2]
        s = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        prof.append(s.sum() / max(1, ((y1 - y0) * (x1 - x0)).sum()))
    return SHIFTS[int(np.argmax(prof))]


def tesseract_text(png_path, doc_id):
    """Content alignment test 2: local tesseract OCR of the page (cached in checks/tesseract/)."""
    import subprocess
    out_dir = os.path.join(HERE, "checks", "tesseract")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{doc_id}.txt")
    if not os.path.exists(p):
        subprocess.run(["tesseract", png_path, p[:-4], "--psm", "3", "-l", "eng"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return open(p, encoding="utf-8", errors="replace").read()


def toks4(s):
    return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4)


def build():
    from PIL import Image
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    pix, lmm = load_meta()
    rows, order_diffs = join_mirrors(pix, lmm)
    sample, design = draw_sample(rows)
    png_dir = os.path.join(RAW, "lmms_png")
    manifest = json.load(open(os.path.join(png_dir, "manifest.json")))
    img_dir = os.path.join(HERE, "img")
    os.makedirs(img_dir, exist_ok=True)
    items, img_info, problems = [], {}, []
    manual = json.load(open(os.path.join(HERE, "checks", "manual_verified.json")))
    # ---- pass 1: per-page alignment evidence (image from lmms vs OCR from pixparse)
    page_ocr, page_img, align = {}, {}, {}
    for r in sample:
        d = r["lmm"]["docId"]
        if d in page_img:
            continue
        src = os.path.join(png_dir, f"{d}.png")
        if not os.path.exists(src):
            sys.exit(f"ABORT: missing page image for doc {d}; run fetch-images")
        b = open(src, "rb").read()
        m = manifest[str(d)]
        if hashlib.sha256(b).hexdigest() != m["sha256"] or m["image_path"] != r["lmm"]["image"]["path"]:
            sys.exit(f"ABORT: image hash / filename mismatch for doc {d}")
        im = Image.open(io.BytesIO(b)); im.load()
        ocr = r["pix"]["ocr_results"]
        if (im.width, im.height) != (ocr["width"], ocr["height"]):
            sys.exit(f"ABORT: image size {im.size} != OCR page size {(ocr['width'], ocr['height'])} for doc {d}")
        page_img[d], page_ocr[d] = (b, im, src), ocr
    docs = sorted(page_img)
    otok = {d: toks4(ocr_text(page_ocr[d])) for d in docs}
    ttok = {d: toks4(tesseract_text(page_img[d][2], d)) for d in docs}
    df = collections.Counter(w for d in docs for w in otok[d])
    common = {w for w, c in df.items() if c >= COMMON_DF * len(docs)}
    ctrl_shift, ctrl_tess = [], []
    for i, d in enumerate(docs):
        o, t = otok[d] - common, ttok[d] - common
        rec = len(o & t) / len(o) if o else None
        dy = shift_profile_peak(page_img[d][1], page_ocr[d])
        ok_shift = dy is not None and abs(dy) <= SHIFT_MAX_DY
        ok_tess = rec is not None and rec >= TESS_MIN_RECALL
        ok_manual = str(d) in manual
        if not (ok_shift or ok_tess or ok_manual):
            sys.exit(f"ABORT: page {d} fails all alignment checks (shift dy={dy}, tesseract recall={rec}); verify by eye")
        align[d] = {"shift_peak_dy": dy, "tesseract_recall": None if rec is None else round(rec, 3),
                    "pass_shift": ok_shift, "pass_tesseract": ok_tess, "manual_verified": ok_manual,
                    "passed_by": "shift" if ok_shift else ("tesseract" if ok_tess else "manual")}
        e = docs[(i + 1) % len(docs)]  # control: this page's OCR against the next (wrong) page image
        cdy = shift_profile_peak(page_img[e][1], page_ocr[d])
        ctrl_shift.append(cdy is not None and abs(cdy) <= SHIFT_MAX_DY)
        ctrl_tess.append(bool(o) and len(o & (ttok[e] - common)) / len(o) >= TESS_MIN_RECALL)
    align_summary = {
        "pages": len(docs),
        "passed_by": dict(collections.Counter(a["passed_by"] for a in align.values())),
        "pass_shift": sum(a["pass_shift"] for a in align.values()),
        "pass_tesseract": sum(a["pass_tesseract"] for a in align.values()),
        "pass_shift_or_tesseract": sum(a["pass_shift"] or a["pass_tesseract"] for a in align.values()),
        "manual_only": sorted(d for d, a in align.items() if a["passed_by"] == "manual"),
        "control_wrong_image_false_pass_rate_shift": round(sum(ctrl_shift) / max(1, len(ctrl_shift)), 3),
        "control_wrong_image_false_pass_rate_tesseract": round(sum(ctrl_tess) / max(1, len(ctrl_tess)), 3),
        "control_wrong_image_false_pass_rate_either": None,
        "common_tokens_ignored": len(common),
        "size_unique_in_val": None,
    }
    sizes = collections.Counter((rr["pix"]["ocr_results"]["width"], rr["pix"]["ocr_results"]["height"])
                                for rr in {x["lmm"]["docId"]: x for x in rows}.values())
    align_summary["size_unique_in_val"] = sum(1 for d in docs if sizes[(page_ocr[d]["width"], page_ocr[d]["height"])] == 1)
    both = [a or b for a, b in zip(ctrl_shift, ctrl_tess)]
    align_summary["control_wrong_image_false_pass_rate_either"] = round(sum(both) / len(both), 3)
    same_size_ctrl = sum(1 for d in docs if sizes[(page_ocr[d]["width"], page_ocr[d]["height"])] > 1)
    align_summary["sampled_pages_sharing_their_size_with_another_val_page"] = same_size_ctrl
    # ---- pass 2: items
    for r in sample:
        p, l = r["pix"], r["lmm"]
        d = l["docId"]
        if d not in img_info:
            b, im, src = page_img[d]
            m = manifest[str(d)]
            fmt = im.format
            out_name = f"img/{d}.png"
            if fmt in ("PNG", "JPEG") and len(b) <= MAX_IMG_BYTES:
                data, reenc = b, None
                if fmt == "JPEG":
                    out_name = f"img/{d}.jpg"
            else:  # re-encode: PNG first, JPEG q90 fallback
                bio = io.BytesIO(); im.save(bio, "PNG", optimize=True); data, reenc = bio.getvalue(), "png_optimize"
                if len(data) > MAX_IMG_BYTES:
                    bio = io.BytesIO(); im.convert("L" if im.mode in ("L", "1", "LA") else "RGB").save(bio, "JPEG", quality=90)
                    data, reenc, out_name = bio.getvalue(), "jpeg_q90", f"img/{d}.jpg"
            with open(os.path.join(HERE, out_name), "wb") as f:
                f.write(data)
            img_info[d] = {"path": out_name, "sha256_original": m["sha256"], "bytes_original": len(b), "bytes": len(data),
                           "format_original": fmt, "reencoded": reenc, "width": im.width, "height": im.height, "mode": im.mode,
                           "alignment": align[d]}
        info = img_info[d]
        ocr = p["ocr_results"]
        text = ocr_text(ocr)
        answers = list(dict.fromkeys(p["answers"]))
        nt, na = norm_ws(text), norm_alnum(text)
        reach = any(norm_ws(a) and norm_ws(a) in nt for a in answers)
        reach_alnum = any(norm_alnum(a) and norm_alnum(a) in na for a in answers)
        words = [w for line in ocr["lines"] for w in (line.get("words") or [])]
        items.append({
            "id": f"{KEY}-val-{r['qid']}",
            "dataset": KEY,
            "use_case": USE_CASE,
            "modality": "image",
            "text": text,
            "text_source": "official_ocr",
            "images": [info["path"]],
            "fields": [{"name": "answer", "type": "text", "instructions": p["question"], "gold": answers}],
            "metric": "anls",
            "meta": {
                "split": "validation",
                "question_id": r["qid"],
                "question": p["question"],
                "question_types": list(l["question_types"] or []),
                "primary_type": r["primary_type"],
                "handwritten": r["handwritten"],
                "stratum": r["stratum"],
                "weight_val": round(r["weight_val"], 6),
                "doc_id": l["docId"],
                "ucsf_document_id": l["ucsf_document_id"],
                "ucsf_document_page_no": l["ucsf_document_page_no"],
                "image_filename": l["image"]["path"],
                "page_count": 1,
                "image": info,
                "ocr": {"engine": "commercial OCR shipped with DocVQA (docvqa.org: 'a commercial OCR solution')",
                        "width": ocr["width"], "height": ocr["height"], "unit": ocr["unit"],
                        "clockwise_orientation": ocr["clockwise_orientation"], "n_lines": len(ocr["lines"]),
                        "n_words": len(words), "n_low_confidence_words": sum(1 for w in words if w.get("confidence") == "Low"),
                        "serialisation": "line texts in shipped order joined by \\n"},
                "text_o200k_tokens": len(enc.encode(text, disallowed_special=())),
                "question_o200k_tokens": len(enc.encode(p["question"], disallowed_special=())),
                "gold_reachable": reach,
                "gold_reachable_alnum": reach_alnum,
                "anls_threshold": 0.5,
                "answer_order_differs_between_mirrors": list(p["answers"]) != list(l["answers"]),
                "sources": {"pixparse": {"file": p["src_file"], "row": p["src_row"]},
                            "lmms": {"file": l["src_file"], "row": l["src_row"]}},
                "mode_hint": "Sage bare-question mode (extract = the question string); Clairic: one text field 'answer'",
            },
        })
    with open(os.path.join(HERE, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # population-level reachability (all 5,349) for the CARD
    pop_reach = sum(any(norm_ws(a) and norm_ws(a) in norm_ws(ocr_text(r["pix"]["ocr_results"])) for a in r["pix"]["answers"]) for r in rows)
    pop_reach_alnum = sum(any(norm_alnum(a) and norm_alnum(a) in norm_alnum(ocr_text(r["pix"]["ocr_results"])) for a in r["pix"]["answers"]) for r in rows)
    tagfreq = design["tag_frequency_val"]
    p_items = [it for it in items if it["meta"]["stratum"] == "proportional"]
    tag_p = collections.Counter(t for it in p_items for t in it["meta"]["question_types"])
    tag_all = collections.Counter(t for it in items for t in it["meta"]["question_types"])
    marg = {t: {"val": n, "expected_in_P": round(len(p_items) * n / len(rows), 1), "in_P": tag_p.get(t, 0), "in_all_350": tag_all.get(t, 0)}
            for t, n in tagfreq.items()}

    def frac(sel, key):
        sel = list(sel)
        return {"n": len(sel), "pct": round(100 * sum(it["meta"][key] for it in sel) / max(1, len(sel)), 1)}
    reach = {
        "all_350": frac(items, "gold_reachable"),
        "proportional": frac(p_items, "gold_reachable"),
        "handwritten_all": frac((it for it in items if it["meta"]["handwritten"]), "gold_reachable"),
        "non_handwritten": frac((it for it in items if not it["meta"]["handwritten"]), "gold_reachable"),
        "alnum_all_350": frac(items, "gold_reachable_alnum"),
    }
    report = {"design": design, "n_items": len(items), "n_docs": len(img_info),
              "tag_marginals": marg, "gold_reachable_sample": reach,
              "gold_reachable_alnum_val_pct": round(100 * pop_reach_alnum / len(rows), 2),
              "image_bytes_total_mb": round(sum(v["bytes"] for v in img_info.values()) / 1e6, 1),
              "image_bytes_max": max(v["bytes"] for v in img_info.values()),
              "images_reencoded": sum(1 for v in img_info.values() if v["reencoded"]),
              "alignment": align_summary,
              "answer_order_diffs_between_mirrors_val": order_diffs,
              "gold_reachable_val_pct": round(100 * pop_reach / len(rows), 2),
              "problems": problems}
    json.dump(report, open(os.path.join(HERE, "build_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "design"}, indent=1))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"fetch-meta": fetch_meta, "fetch-images": fetch_images, "build": build}[stage]()
