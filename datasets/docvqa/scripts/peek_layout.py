# Print parquet schema + row-group layout of the DocVQA val shards of both mirrors (range reads, footer only).
import json, sys, fsspec, pyarrow.parquet as pq
PIX = "https://huggingface.co/datasets/pixparse/docvqa-single-page-questions/resolve/33136ef456fa5a3fe68568d6e31dda4eeff95b9b/data/validation-%05d-of-00004.parquet"
LMM = "https://huggingface.co/datasets/lmms-lab-encoder/DocVQA/resolve/539088ef8a8ada01ac8e2e6d4e372586748a265e/DocVQA/validation-%05d-of-00006.parquet"
fs = fsspec.filesystem("http")
out = {}
for name, pat, n in (("pixparse", PIX, 4), ("lmms", LMM, 6)):
    for i in range(n):
        u = pat % i
        with fs.open(u, "rb", block_size=2**16, cache_type="none") as f:
            pf = pq.ParquetFile(f, pre_buffer=False)
            md = pf.metadata
            if i == 0: print(name, pf.schema_arrow)
            rgs = []
            for g in range(md.num_row_groups):
                rg = md.row_group(g)
                cols = {rg.column(c).path_in_schema: rg.column(c).total_compressed_size for c in range(rg.num_columns)}
                img = sum(v for k, v in cols.items() if k.startswith("image"))
                rgs.append((rg.num_rows, img, sum(cols.values())))
            print(name, i, "rows", md.num_rows, "row_groups", md.num_row_groups, "first rgs", rgs[:3], "img bytes total", sum(r[1] for r in rgs))
            out[u] = rgs
json.dump(out, open(sys.argv[1], "w"))
