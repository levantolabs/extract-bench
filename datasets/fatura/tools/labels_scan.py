"""Count label prefixes (text with digits/values masked) per class over the 2,000 held-out docs."""
import json, sys, re, collections
from pathlib import Path
RAW = Path(sys.argv[1])
held = [50, 7, 32, 39, 2, 12, 4, 49, 10, 25]
OF = RAW / "Annotations" / "Original_Format"
cls = ["NUMBER","DATE","DUE_DATE","SELLER_NAME","BUYER","BILL_TO","SUB_TOTAL","TAX","DISCOUNT","TOTAL"]
pat = collections.defaultdict(collections.Counter)
gst = collections.Counter()
for t in held:
    for k in range(200):
        d = json.loads((OF / f"Template{t}_Instance{k}.json").read_text())
        for c in cls:
            v = d.get(c)
            if isinstance(v, dict) and v.get("text"):
                s = v["text"].split("\n")[0]
                if c in ("SELLER_NAME",):
                    s = re.sub(r"[A-Za-z]", "a", s); s = re.sub(r"a+", "A", s)
                elif c in ("BUYER","BILL_TO"):
                    s = re.sub(r"[:\s]*[A-Z][a-z]+.*$", " <NAME...>", s)
                else:
                    s = re.sub(r"\d", "9", s)
                    s = re.sub(r"9+", "9", s)
                    s = re.sub(r"(?<=9)-[A-Za-z]{3}-", "-Mon-", s)
                pat[c][(t, s)] += 1
        g = [c for c in d if c.startswith("GST")]
        gst[(t, tuple(sorted(g)))] += 1
for c in cls:
    print("==", c)
    for (t, s), n in sorted(pat[c].items()):
        print(f"   T{t:<3} {n:4d}  {s!r}")
print(gst)
