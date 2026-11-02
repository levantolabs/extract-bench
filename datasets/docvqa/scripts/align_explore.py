import os, re, sys, pyarrow.parquet as pq
D = sys.argv[1]
pix = pq.read_table(os.path.join(D, "raw/pixparse_val_meta.parquet"), columns=["other_metadata", "ocr_results"]).to_pylist()
ocr = {}
for r in pix: ocr.setdefault(r["other_metadata"]["doc_id"], r["ocr_results"])
tok = lambda s: set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4)
docs = sorted(int(f[:-4]) for f in os.listdir(os.path.join(D, "checks/tesseract")))
T = {d: tok(open(os.path.join(D, "checks/tesseract", f"{d}.txt")).read()) for d in docs}
O = {d: tok("\n".join(l["text"] for l in ocr[d]["lines"])) for d in docs}
res = []
for i, d in enumerate(docs):
    o = O[d]; t = T[d]
    rec = len(o & t) / max(1, len(o))
    ctrl = max(len(O[e] & t) / max(1, len(O[e])) for e in docs if e != d)  # best wrong page
    res.append((d, round(rec, 3), round(ctrl, 3), len(o), len(t)))
res.sort(key=lambda x: x[1])
print("lowest recall", res[:10])
print("max control", max(r[2] for r in res), "min recall", res[0][1], "min margin", min(r[1] - r[2] for r in res))
print("pages where recall <= best wrong page:", [r for r in res if r[1] <= r[2]])
