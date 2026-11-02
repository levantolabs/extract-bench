"""Exploration only: spot-check money golds not reachable in the Textract layer (exact or in GBP 000)."""
import json, re
from pathlib import Path
DS = Path(__file__).resolve().parents[1]
items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines()]
n = both = 0
miss = []
for it in items:
    r = it["meta"]["gold_reachable"]
    for k in ("income_annually_in_british_pounds", "spending_annually_in_british_pounds"):
        g = next(f["gold"] for f in it["fields"] if f["name"] == k)
        if g is None:
            continue
        n += 1
        ok = r.get(k) or r.get(k + "__thousands")
        both += bool(ok)
        if not ok:
            miss.append((it["id"][-8:], k[:6], g, it["text"]))
print("money golds", n, "reachable exact-or-thousands", both, round(both / n, 3))
for mid, k, g, t in miss[:6]:
    x = int(float(g)); s = str(x)
    # any occurrence of the leading 4 digits of the amount, with context
    lead = s[:3]
    hits = [m.start() for m in re.finditer(re.escape(lead), t.replace(",", ""))][:3]
    tt = t.replace(",", "")
    print("---", mid, k, g, [tt[max(0, h - 60):h + 40].replace("\n", " | ") for h in hits])
