"""Exploration only: gold value formats in dev-0 expected.tsv."""
import calendar, re
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
exp = (ROOT / "raw/repo/dev-0/expected.tsv").read_text(encoding="utf-8").rstrip("\n").split("\n")
vals = {}
for e in exp:
    for kv in e.split(" "):
        if kv:
            k, v = kv.split("=", 1); vals.setdefault(k, []).append(v)
for k, vs in sorted(vals.items()):
    shapes = Counter(re.sub(r"[A-Z]", "A", re.sub(r"[a-z]", "a", re.sub(r"\d", "9", v))) for v in vs)
    print(k, len(vs), "examples", vs[:3], "top shapes", shapes.most_common(4) if k not in ("charity_name", "address__street_line") else "")
cn = vals["charity_number"]; print("charity_number all digits", sum(v.isdigit() for v in cn), [v for v in cn if not v.isdigit()][:10])
rd = vals["report_date"]
me = sum(int(v[8:]) == calendar.monthrange(int(v[:4]), int(v[5:7]))[1] for v in rd)
print("report_date month-end", me, "/", len(rd), Counter(v[5:7] for v in rd).most_common(5), "years", Counter(v[:4] for v in rd).most_common(8))
for k in ("income_annually_in_british_pounds", "spending_annually_in_british_pounds"):
    vs = vals[k]; print(k, "non-.00", sum(not v.endswith(".00") for v in vs), "neg", sum(v.startswith("-") for v in vs), "min", min(vs, key=float), "max", max(vs, key=float), "div1000", sum(float(v) % 1000 == 0 for v in vs))
special = Counter(c for k in vals for v in vals[k] for c in v if not c.isalnum() and c not in "_")
print("special chars", special)
print("name case", Counter("upper" if v.isupper() else "title/mixed" for v in vals["charity_name"]))
print("town case", Counter("upper" if v.replace("_","").replace("-","").isupper() else "mixed" for v in vals["address__post_town"]))
print("street samples", vals["address__street_line"][:12])
