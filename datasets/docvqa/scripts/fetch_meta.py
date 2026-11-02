# Stage 1 (network, free HF downloads): read only the non-image columns of DocVQA val from both mirrors, pinned revisions.
import sys, fsspec, pyarrow as pa, pyarrow.parquet as pq
RAW = sys.argv[1]
PIX_REV = "33136ef456fa5a3fe68568d6e31dda4eeff95b9b"; LMM_REV = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
PIX = f"https://huggingface.co/datasets/pixparse/docvqa-single-page-questions/resolve/{PIX_REV}/data/validation-%05d-of-00004.parquet"
LMM = f"https://huggingface.co/datasets/lmms-lab-encoder/DocVQA/resolve/{LMM_REV}/DocVQA/validation-%05d-of-00006.parquet"
fs = fsspec.filesystem("http")
def read(u, cols):
    with fs.open(u, "rb", block_size=2**16, cache_type="none") as f:
        t = pq.ParquetFile(f, pre_buffer=False).read(columns=cols)
    return t
pix = []
for i in range(4):
    t = read(PIX % i, ["image.path", "question_id", "question", "answers", "data_split", "ocr_results", "other_metadata"])
    t = t.append_column("src_file", pa.array([PIX % i] * t.num_rows)).append_column("src_row", pa.array(list(range(t.num_rows)), pa.int64()))
    pix.append(t); print("pix", i, t.num_rows, flush=True)
pq.write_table(pa.concat_tables(pix), f"{RAW}/pixparse_val_meta.parquet", compression="zstd")
lmm = []
for i in range(6):
    t = read(LMM % i, ["questionId", "question", "question_types", "image.path", "docId", "ucsf_document_id", "ucsf_document_page_no", "answers", "data_split"])
    t = t.append_column("src_file", pa.array([LMM % i] * t.num_rows)).append_column("src_row", pa.array(list(range(t.num_rows)), pa.int64()))
    lmm.append(t); print("lmm", i, t.num_rows, flush=True)
pq.write_table(pa.concat_tables(lmm), f"{RAW}/lmms_val_meta.parquet", compression="zstd")
