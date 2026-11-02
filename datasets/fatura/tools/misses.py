"""List gold values not found in the x2 Tesseract text, with the nearest OCR line (for the audit)."""
import json, re, sys, difflib
from pathlib import Path
D = Path(sys.argv[1])
def norm(s): return re.sub(r"\s+", " ", s).strip().lower()
for line in open(D / "items.jsonl"):
    it = json.loads(line)
    txt = (D / "tools" / "tesseract_txt" / f"{it['id']}_x2.txt").read_text()
    on = norm(txt); sq = on.replace(" ", "")
    lines = [l for l in txt.splitlines() if l.strip()]
    for f in it["fields"]:
        g = f["gold"]
        if g is None: continue
        gs = it["meta"]["gold_source"][f["name"]]
        if f["type"] == "money":
            v = g[0] if isinstance(g, list) else g
            c = [f"{v:.2f}", str(abs(v))]
        elif f["type"] == "date":
            c = [gs["span"]]
        else:
            c = g if isinstance(g, list) else [g]
        if any(norm(str(x)) in on or norm(str(x)).replace(" ", "") in sq for x in c):
            continue
        src = gs["src_text"] if isinstance(gs["src_text"], str) else " | ".join(gs["src_text"])
        best = difflib.get_close_matches(norm(src.split("\n")[0]), [norm(l) for l in lines], n=1, cutoff=0)
        print(f"{it['id']:22s} {f['name']:15s} gold={g!r:28s} ann={src.splitlines()[0]!r:45s} ocr~{best[0] if best else ''!r}")
