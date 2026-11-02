import json, os, sys, statistics
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFilter, ImageStat, ImageChops
D = sys.argv[1]
pix = pq.read_table(os.path.join(D, "raw/pixparse_val_meta.parquet"), columns=["other_metadata", "ocr_results"]).to_pylist()
ocr = {}
for r in pix: ocr.setdefault(r["other_metadata"]["doc_id"], r["ocr_results"])
res = []
for f in sorted(os.listdir(os.path.join(D, "raw/lmms_png"))):
    if not f.endswith(".png"): continue
    d = int(f[:-4]); o = ocr[d]
    g = Image.open(os.path.join(D, "raw/lmms_png", f)).convert("L")
    e = g.filter(ImageFilter.FIND_EDGES)
    mask = Image.new("L", g.size, 0); dr = ImageDraw.Draw(mask)
    for line in o["lines"]:
        for w in line["words"]:
            bb = w["bounding_box"]; dr.polygon([(bb[0], bb[1]), (bb[2], bb[3]), (bb[4], bb[5]), (bb[6], bb[7])], fill=255)
    shifted = ImageChops.offset(mask, g.width // 2, g.height // 3)
    ein = ImageStat.Stat(e, mask).mean[0]; esh = ImageStat.Stat(e, shifted).mean[0]; epage = ImageStat.Stat(e).mean[0]
    res.append((d, round(ein / max(epage, 1e-6), 2), round(ein / max(esh, 1e-6), 2), round(ein, 1), round(epage, 1)))
r1 = sorted(x[1] for x in res); r2 = sorted(x[2] for x in res)
print("n", len(res), "edge in/page min..p5..p50", r1[0], r1[len(r1)//20], r1[len(r1)//2], "| in/shifted min..p5..p50", r2[0], r2[len(r2)//20], r2[len(r2)//2])
print("lowest by in/page", sorted(res, key=lambda x: x[1])[:6])
print("lowest by in/shifted", sorted(res, key=lambda x: x[2])[:6])
