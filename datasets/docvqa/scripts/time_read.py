import time, sys, fsspec, pyarrow.parquet as pq
u = "https://huggingface.co/datasets/pixparse/docvqa-single-page-questions/resolve/33136ef456fa5a3fe68568d6e31dda4eeff95b9b/data/validation-00000-of-00004.parquet"
fs = fsspec.filesystem("http")
for cols in (["question_id"], ["image.path"]):
    t0 = time.time(); n = [0]
    with fs.open(u, "rb", block_size=2**16, cache_type="none") as f:
        orig = f.read
        def rd(*a, **k):
            b = orig(*a, **k); n[0] += len(b); return b
        f.read = rd
        t = pq.ParquetFile(f, pre_buffer=False).read(columns=cols)
    print(cols, t.num_rows, "bytes read", n[0], "secs", round(time.time() - t0, 1), flush=True)
