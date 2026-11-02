"""Exploration only: surface forms in the text that alnum-match each gold party (and term phrases)."""
import csv, lzma, re, sys, collections
csv.field_size_limit(sys.maxsize)
RAW = sys.argv[1]
rows = list(csv.reader(lzma.open(f"{RAW}/repo/dev-0/in.tsv.xz", "rt"), delimiter="\t", quoting=csv.QUOTE_NONE))
exp = [l.rstrip("\n") for l in open(f"{RAW}/repo/dev-0/expected.tsv")]
def unesc(s):
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "f": "\f", "\\": "\\"}.get(m[1], "\\" + m[1]), s)
diff = collections.Counter()
for r, e in zip(rows, exp):
    text = re.sub(r"\s+", " ", unesc(r[5]))
    for kv in e.split(" "):
        if not kv: continue
        k, v = kv.split("=", 1)
        if k == "party":
            g = v.replace("_", " ")
            toks = re.findall(r"[0-9A-Za-z]+", g)
            pat = r"(?<![0-9A-Za-z])" + r"[\W_]*".join(map(re.escape, toks)) + r"(?![0-9A-Za-z])\.?"
            forms = collections.Counter(m.group(0) for m in re.finditer(pat, text, re.I))
            exact = [f for f in forms if f.casefold() == g.casefold()]
            if not exact:
                print("PARTY", r[0][:8], repr(g), "->", forms.most_common(3))
        if k == "term":
            pass
