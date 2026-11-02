import collections, pyarrow.parquet as pq
lmm = pq.read_table("raw/lmms_val_meta.parquet", columns=["questionId","docId","src_file","src_row"]).to_pylist()
rg = collections.defaultdict(set); docrg = collections.defaultdict(set)
for r in lmm:
    k = (r["src_file"][-30:], r["src_row"] // 100); rg[k].add(r["docId"]); docrg[r["docId"]].add(k)
print("RGs", len(rg), "docs/RG", sorted(len(v) for v in rg.values())[::5])
print("RGs per doc", collections.Counter(len(v) for v in docrg.values()))
# contiguity of docs in row order
prev=None; runs=0
for r in lmm:
    if r["docId"]!=prev: runs+=1; prev=r["docId"]
print("doc runs in row order", runs, "docs", len(docrg))
