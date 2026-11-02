import sys, collections, json, pyarrow.parquet as pq
pix = pq.read_table("raw/pixparse_val_meta.parquet").to_pylist()
lmm = pq.read_table("raw/lmms_val_meta.parquet").to_pylist()
print(len(pix), len(lmm))
print({k: (v if k != "ocr_results" else "...") for k, v in pix[0].items()})
print(lmm[0])
P = {r["question_id"]: r for r in pix}; L = {int(r["questionId"]): r for r in lmm}
print("ids equal", set(P) == set(L), len(P), len(L))
mm = collections.Counter()
for q, p in P.items():
    l = L[q]
    if p["question"] != l["question"]: mm["question"] += 1
    if list(p["answers"]) != list(l["answers"]): mm["answers"] += 1
    if p["other_metadata"]["doc_id"] != l["docId"]: mm["docId"] += 1
    if p["other_metadata"]["ucsf_document_id"] != l["ucsf_document_id"]: mm["ucsf"] += 1
    if p["other_metadata"]["ucsf_document_page_no"] != l["ucsf_document_page_no"]: mm["page"] += 1
    if p["other_metadata"]["image"] != l["image"]["path"]: mm["imgpath_meta_vs_lmms"] += 1
    if p["image"]["path"] != l["image"]["path"]: mm["imgpath"] += 1
print("mismatches", dict(mm))
ex = [(q, P[q]["other_metadata"]["image"], P[q]["image"]["path"], L[q]["image"]["path"]) for q in list(P)[:3]]; print(ex)
qt = collections.Counter(tuple(sorted(r["question_types"] or [])) for r in lmm)
print("n combos", len(qt)); print(qt.most_common(40))
single = collections.Counter(t for r in lmm for t in (r["question_types"] or []))
print(single.most_common())
print("n types per q", collections.Counter(len(r["question_types"] or []) for r in lmm))
hw = [r for r in lmm if "handwritten" in (r["question_types"] or [])]
print("hw", len(hw), "docs", len({r["docId"] for r in hw}))
print("docs", len({r["docId"] for r in lmm}))
print("ocr keys", list(pix[0]["ocr_results"].keys()), "n lines", len(pix[0]["ocr_results"]["lines"]), pix[0]["ocr_results"]["lines"][0]["text"], pix[0]["ocr_results"]["unit"], pix[0]["ocr_results"]["width"], pix[0]["ocr_results"]["height"])
print("null ocr", sum(1 for r in pix if not r["ocr_results"] or not r["ocr_results"]["lines"]))
print("data_split", collections.Counter(r["data_split"] for r in pix), collections.Counter(r["data_split"] for r in lmm))
# per-doc consistency of OCR across questions on same doc
by = collections.defaultdict(set)
for r in pix: by[r["other_metadata"]["doc_id"]].add("\n".join(l["text"] for l in r["ocr_results"]["lines"]))
print("docs with >1 distinct OCR", sum(1 for v in by.values() if len(v) > 1))
print("answers sizes", collections.Counter(len(r["answers"]) for r in pix).most_common(8))
print("empty answer strings", sum(1 for r in pix for a in r["answers"] if not a.strip()))
