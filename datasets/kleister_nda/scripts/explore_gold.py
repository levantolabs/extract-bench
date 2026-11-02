"""Exploration only (not used by build.py): how do Kleister-NDA dev-0 gold values appear in the text layer?"""
import csv, lzma, re, sys, collections, json
csv.field_size_limit(sys.maxsize)
RAW = sys.argv[1]
rows = list(csv.reader(lzma.open(f"{RAW}/repo/dev-0/in.tsv.xz", "rt"), delimiter="\t", quoting=csv.QUOTE_NONE))
exp = [l.rstrip("\n") for l in open(f"{RAW}/repo/dev-0/expected.tsv")]

def unesc(s):
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "f": "\f", "\\": "\\"}.get(m[1], "\\" + m[1]), s)

def squash(s):
    return re.sub(r"[^0-9a-z]", "", s.casefold())

out = collections.defaultdict(list)
for r, e in zip(rows, exp):
    text = unesc(r[5]); tl = re.sub(r"\s+", " ", text).casefold(); ts = squash(text)
    for kv in e.split(" "):
        if not kv:
            continue
        k, v = kv.split("=", 1)
        vs = v.replace("_", " ")
        verb = vs.casefold() in tl
        sq = squash(vs) in ts
        out[k].append({"doc": r[0][:8], "gold": vs, "verbatim_ci": verb, "squashed": sq})
for k, L in out.items():
    print(k, len(L), "verbatim_ci", sum(x["verbatim_ci"] for x in L), "squashed", sum(x["squashed"] for x in L))
    for x in L:
        if not x["squashed"]:
            print("   MISS", x)
