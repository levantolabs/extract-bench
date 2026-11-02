import os, sys, collections, numpy as np, pyarrow.parquet as pq
from PIL import Image
D = sys.argv[1]
pix = pq.read_table(os.path.join(D, "raw/pixparse_val_meta.parquet"), columns=["other_metadata", "ocr_results"]).to_pylist()
ocr = {}
for r in pix: ocr.setdefault(r["other_metadata"]["doc_id"], r["ocr_results"])
sizes = collections.Counter((o["width"], o["height"]) for o in ocr.values())
docs = sorted(int(f[:-4]) for f in os.listdir(os.path.join(D, "raw/lmms_png")) if f.endswith(".png"))
uniq = sum(1 for d in docs if sizes[(ocr[d]["width"], ocr[d]["height"])] == 1)
print("val pages", len(ocr), "distinct sizes", len(sizes), "sampled pages with a size unique in val", uniq, "/", len(docs),
      "largest same-size group in val", sizes.most_common(3))
SH = list(range(-60, 61, 2))
def profile(img, o):
    a = 255.0 - np.asarray(img.convert("L"), dtype=np.float64)
    H, W = a.shape
    I = np.zeros((H + 1, W + 1)); I[1:, 1:] = a.cumsum(0).cumsum(1)
    bx = []
    for l in o["lines"]:
        for w in l["words"]:
            b = w["bounding_box"]; xs, ys = b[0::2], b[1::2]
            bx.append((max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))))
    bx = np.array([b for b in bx if b[2] > b[0] and b[3] > b[1]])
    if len(bx) == 0: return None
    out = []
    for dy in SH:
        y0 = np.clip(bx[:, 1] + dy, 0, H); y1 = np.clip(bx[:, 3] + dy, 0, H)
        x0, x1 = bx[:, 0], bx[:, 2]
        s = I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
        area = (y1 - y0) * (x1 - x0)
        out.append(s.sum() / max(1, area.sum()))
    return np.array(out)
res = []
for d in docs:
    img = Image.open(os.path.join(D, "raw/lmms_png", f"{d}.png"))
    p = profile(img, ocr[d])
    if p is None: res.append((d, None)); continue
    k = int(np.argmax(p)); s0 = p[SH.index(0)]
    prom = (p.max() - np.median(p)) / max(1e-9, np.median(p))
    res.append((d, SH[k], round(float(s0 / p.max()), 3), round(float(prom), 3)))
bad = [r for r in res if r[1] is None or abs(r[1]) > 8]
print("argmax |dy| distribution", collections.Counter(min(abs(r[1]) // 4 * 4, 40) for r in res if r[1] is not None))
print("pages with |argmax dy| > 8:", bad)
print("lowest prominence", sorted([r for r in res if r[1] is not None], key=lambda r: r[3])[:8])
# control: profile of each page's OCR boxes on a different page image of the same size (if any) or the next page
ctrl = []
for i, d in enumerate(docs[:80]):
    e = docs[(i + 1) % len(docs)]
    img = Image.open(os.path.join(D, "raw/lmms_png", f"{e}.png"))
    p = profile(img, ocr[d])
    if p is None: continue
    ctrl.append(SH[int(np.argmax(p))])
print("control (wrong image) argmax dy:", collections.Counter(min(abs(x) // 4 * 4, 40) for x in ctrl), "share within 8px", round(sum(abs(x) <= 8 for x in ctrl) / len(ctrl), 3))
