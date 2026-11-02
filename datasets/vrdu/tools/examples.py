"""Print gold values with OCR context for a few test docs (read-only)."""
import gzip, json, sys, random
from pathlib import Path
raw = Path(sys.argv[1]); corpus = sys.argv[2]; split = sys.argv[3]; k = int(sys.argv[4])
test = set(json.loads((raw / corpus / "few_shot-splits" / split).read_text())["test"])
recs = []
with gzip.open(raw / corpus / "main" / "dataset.jsonl.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        if r["filename"] in test: recs.append(r)
rng = random.Random(1)
for r in rng.sample(recs, k):
    t = r["ocr"]["text"]
    print("###", r["filename"], "pages", len(r["ocr"]["pages"]), "chars", len(t))
    for name, ments in r["annotations"]:
        if isinstance(name, list): continue
        m = ments[0]; s, e = m[2][0][0], m[2][-1][1]
        print(f"  {name:24s} {m[0].strip()!r:45s} ctx: {t[max(0,s-70):s]!r} >> {t[e:e+30]!r}")
    nl = [a for a in r["annotations"] if isinstance(a[0], list)]
    for name, ments in nl[:2]:
        print("  LINE", {n: m[0].strip() for n, m in zip(name, ments[0])})
    if nl: print("  ...line items:", len(nl))
