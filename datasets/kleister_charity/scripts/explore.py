"""Exploration only: layer lengths, page breaks, token counts, gold key presence (Kleister-Charity dev-0)."""
import csv, json, lzma, re, statistics, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "raw" / "repo"
csv.field_size_limit(sys.maxsize)
def unescape(s):
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "f": "\f", "\\": "\\"}.get(m[1], "\\" + m[1]), s)
with lzma.open(REPO / "dev-0/in.tsv.xz", "rt", encoding="utf-8", newline="") as fh:
    rows = list(csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
exp = (REPO / "dev-0/expected.tsv").read_text(encoding="utf-8").rstrip("\n").split("\n")
print("rows", len(rows), "exp", len(exp), "cols", Counter(len(r) for r in rows))
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
names = ["djvu", "tesseract", "textract", "best"]
out = {}
toks = {n: [] for n in names}
ff = {n: [] for n in names}
empty = Counter()
best_eq = Counter()
for r in rows:
    for i, n in enumerate(names):
        t = unescape(r[2 + i])
        toks[n].append(len(enc.encode(t, disallowed_special=())))
        ff[n].append(t.count("\f"))
        empty[n] += len(t.strip()) == 0
    best_eq["djvu"] += r[5] == r[2]
    best_eq["tesseract"] += r[5] == r[3]
for n in names:
    xs = sorted(toks[n])
    print(n, "p50", statistics.median(xs), "p95", xs[int(0.95 * (len(xs) - 1))], "max", max(xs), ">15k", sum(x > 15000 for x in xs), ">45k", sum(x > 45000 for x in xs), ">65536", sum(x > 65536 for x in xs), "empty", empty[n], "ff p50", statistics.median(ff[n]), "ff max", max(ff[n]))
print("best==", best_eq)
# page-break agreement
agree = sum(ff["textract"][i] == ff["tesseract"][i] for i in range(len(rows)))
print("ff textract==tesseract", agree)
keys = Counter(); multi = Counter(); keyscol = Counter(); absent = Counter()
K = ["address__post_town", "address__postcode", "address__street_line", "charity_name", "charity_number", "income_annually_in_british_pounds", "report_date", "spending_annually_in_british_pounds"]
for r, e in zip(rows, exp):
    g = {}
    for kv in e.split(" "):
        if kv:
            k, v = kv.split("=", 1); g.setdefault(k, []).append(v)
    for k, vs in g.items():
        keys[k] += 1
        if len(vs) > 1: multi[k] += 1
    keyscol[r[1]] += 1
    for k in K:
        if k not in g: absent[k] += 1
print("gold key presence", dict(keys)); print("multi", dict(multi)); print("absent", dict(absent))
print("keys col", keyscol.most_common(5))
