import fsspec, pyarrow.parquet as pq, pyarrow
print(pyarrow.__version__)
PIX = "https://huggingface.co/datasets/pixparse/docvqa-single-page-questions/resolve/33136ef456fa5a3fe68568d6e31dda4eeff95b9b/data/validation-00000-of-00004.parquet"
LMM = "https://huggingface.co/datasets/lmms-lab-encoder/DocVQA/resolve/539088ef8a8ada01ac8e2e6d4e372586748a265e/DocVQA/validation-00000-of-00006.parquet"
fs = fsspec.filesystem("http")
for u in (PIX, LMM):
    with fs.open(u, "rb", block_size=2**16, cache_type="none") as f:
        pf = pq.ParquetFile(f, pre_buffer=False)
        md = pf.metadata
        print(u.split('/')[4], md.created_by, md.format_version)
        rg = md.row_group(0)
        for c in range(rg.num_columns):
            cc = rg.column(c)
            d = cc.to_dict()
            print("  ", cc.path_in_schema, "comp", cc.compression, "enc", cc.encodings, "dict_off", cc.dictionary_page_offset, "data_off", cc.data_page_offset, "size", cc.total_compressed_size,
                  "has_offset_index", getattr(cc, "has_offset_index", "?"), "has_column_index", getattr(cc, "has_column_index", "?"))
