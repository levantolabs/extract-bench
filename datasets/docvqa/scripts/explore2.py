import collections, pyarrow.parquet as pq
pix = pq.read_table("raw/pixparse_val_meta.parquet", columns=["question_id","answers"]).to_pylist()
lmm = pq.read_table("raw/lmms_val_meta.parquet", columns=["questionId","answers"]).to_pylist()
L = {int(r["questionId"]): r["answers"] for r in lmm}
kinds = collections.Counter(); ex = []
for r in pix:
    a, b = list(r["answers"]), list(L[r["question_id"]])
    if a == b: continue
    if sorted(a) == sorted(b): kinds["order"] += 1
    elif set(a) == set(b): kinds["dup"] += 1
    else: kinds["content"] += 1; ex.append((r["question_id"], a, b))
print(kinds); print(ex[:10])
