"""Fetch FULL rows of source 'tatqa/text-span' from the tasksource extraction corpus (free HF download).

Why: the spec says the build drops any TAT-QA test item whose full context appears in tasksource extraction rows.
The survey kept only a 300-char state_head; this keeps the full `state` and `metadata` (question_ids) per row.
Only 'tatqa/text-span' is TAT-QA-derived (train 1,507 / validation 188 / test 0), per the source frequencies in
survey/critique-legitimacy-2026-10-23/raw/tasksource_extraction_stats_*.json, so the test split is skipped.

Method: HTTP range reads of only the `source`, `id`, `state`, `metadata` column chunks of each converted parquet
file (refs/convert/parquet; one row group per file), filtered locally with pyarrow; per-file results are written to
raw/tasksource/parts/ so a rerun resumes. Nothing from the dataset is executed.
Earlier attempts (23 Oct 2026, ~06:00-07:00 SGT), all abandoned:
  duckdb httpfs over all files: no output after >25 min.
  datasets-server /filter: HTTP 500 "the dataset index is loading".
  whole-file curl: ~118 KB/s (machine-wide), ~2 h for 845 MB.
  datasets-server /rows per row (rows are shuffled: 187 runs for 188 validation rows): HTTP 429 on every thread.

Run: /opt/homebrew/bin/uv run --no-project --with pyarrow --with fsspec --with aiohttp --with requests \
       python -I scripts/fetch_tasksource_tatqa.py <dataset_dir>
Output: <dataset_dir>/raw/tasksource/tatqa_text_span_rows.json
"""
import datetime, json, sys, time, urllib.request
from pathlib import Path

import fsspec
import pyarrow.compute as pc
import pyarrow.parquet as pq

ds = Path(sys.argv[1])
out = ds / "raw" / "tasksource"
parts = out / "parts"
parts.mkdir(parents=True, exist_ok=True)
REPO = "tasksource/tasksource-jev-typed-decisions"
SRC = "tatqa/text-span"
SPLITS = ("validation", "train")
COLS = ["source", "id", "state", "metadata"]
meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60))
listing = json.load(urllib.request.urlopen(
    f"https://datasets-server.huggingface.co/parquet?dataset={REPO}&config=extraction", timeout=60))
fs = fsspec.filesystem("http")
files, res = [], {}
for split in SPLITS:
    for f in sorted((f for f in listing["parquet_files"] if f["split"] == split), key=lambda f: f["filename"]):
        part = parts / f"{split}_{f['filename']}.json"
        if part.exists():
            rec = json.loads(part.read_text())
        else:
            t0 = time.time()
            with fs.open(f["url"], "rb", block_size=4 << 20, cache_type="readahead") as fh:
                pf = pq.ParquetFile(fh)
                md = pf.metadata
                names = [md.schema.column(i).path for i in range(md.num_columns)]
                chunk = {names[i]: md.row_group(0).column(i).total_compressed_size for i in range(md.num_columns)}
                t = pf.read(columns=COLS)
            sub = t.filter(pc.equal(t["source"], SRC))
            idx = pc.indices_nonzero(pc.equal(t["source"], SRC)).to_pylist()
            rows = [r | {"_file_row": i} for r, i in zip(sub.to_pylist(), idx)]
            rec = {"file": {"split": split, "filename": f["filename"], "url": f["url"], "size": f["size"],
                            "num_rows": md.num_rows, "bytes_read_cols": sum(chunk[c] for c in COLS),
                            "tatqa_rows": len(rows), "secs": round(time.time() - t0, 1)}, "rows": rows}
            part.write_text(json.dumps(rec, ensure_ascii=False, default=str))
        files.append(rec["file"])
        res.setdefault(split, []).extend(rec["rows"])
        print(rec["file"], flush=True)
prov = {"repo": REPO, "config": "extraction", "source": SRC, "sha": meta.get("sha"),
        "lastModified": meta.get("lastModified"), "parquet_ref": "refs/convert/parquet", "columns": COLS,
        "files": files, "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
json.dump({"provenance": prov, "splits": res}, open(out / "tatqa_text_span_rows.json", "w"), ensure_ascii=False,
          default=str)
print({s: len(v) for s, v in res.items()})
