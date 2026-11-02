import json, sys, re, collections
from pathlib import Path
RAW = Path(sys.argv[1])
held = [50, 7, 32, 39, 2, 12, 4, 49, 10, 25]
OF = RAW / "Annotations" / "Original_Format"
want = ["NUMBER","DATE","DUE_DATE","PO_NUMBER","SELLER_NAME","BUYER","BILL_TO","SEND_TO","SUB_TOTAL","TAX","DISCOUNT","TOTAL","AMOUNT_DUE"]
allcls = collections.Counter()
for t in held:
    pres = collections.Counter()
    ex = {}
    for k in range(200):
        d = json.loads((OF / f"Template{t}_Instance{k}.json").read_text())
        for c, v in d.items():
            if isinstance(v, dict) and v.get("text"):
                pres[c] += 1
                ex.setdefault(c, v["text"])
            elif isinstance(v, list) and v:
                pres[c+"[list]"] += 1
    allcls.update(pres.keys())
    print(f"== Template{t}: ", dict(pres))
    for c in want:
        if c in ex:
            print(f"   {c}: {ex[c]!r}")
print(allcls)
