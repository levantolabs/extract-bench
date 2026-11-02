"""Read only the parquet footer of one tasksource extraction file (row groups, column chunk sizes). No data rows.
Run: /opt/homebrew/bin/uv run --no-project --with pyarrow --with fsspec --with aiohttp --with requests python -I \
       scripts/probe_parquet_footer.py <url>"""
import sys, time
import fsspec, pyarrow.parquet as pq

t0 = time.time()
fs = fsspec.filesystem("http")
with fs.open(sys.argv[1], "rb", block_size=1 << 16) as fh:
    md = pq.ParquetFile(fh).metadata
    print("rows", md.num_rows, "row_groups", md.num_row_groups, "secs", round(time.time() - t0, 1))
    names = [md.schema.column(i).path for i in range(md.num_columns)]
    for rg in range(min(md.num_row_groups, 3)):
        r = md.row_group(rg)
        print(rg, r.num_rows, {names[i]: r.column(i).total_compressed_size for i in range(r.num_columns)})
