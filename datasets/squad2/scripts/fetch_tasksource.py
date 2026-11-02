"""Fetch every `squad-v2` row (full state) of the tasksource extraction config via the HF datasets-server /filter API.

(Range-reading the parquet with duckdb ran at ~25 KB/s on 23 Oct, so this pulls only the matching rows.)
The datasets-server serves refs/convert/parquet; its commit is recorded in raw/pages/hf_tasksource_convert_commit.json.
Writes <out_dir>/squad_v2_rows.<split>.jsonl (one row per line: source, id, metadata, state, fields, answers, split).
Usage: uv run --no-project python -I fetch_tasksource.py <out_dir> [split ...]
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

out_dir = Path(sys.argv[1])
splits = sys.argv[2:] or ["validation", "test", "train"]
DS = "tasksource/tasksource-jev-typed-decisions"
KEEP = ("source", "id", "metadata", "state", "fields", "answers", "license")


def get(params):
    url = "https://datasets-server.huggingface.co/filter?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, str(e)[:120], flush=True)
            time.sleep(10 + 10 * attempt)
    raise SystemExit("gave up on " + url)


for split in splits:
    f = out_dir / f"squad_v2_rows.{split}.jsonl"
    done = sum(1 for _ in f.open()) if f.exists() else 0
    off, total = done, None
    with f.open("a") as fo:
        while total is None or off < total:
            d = get({"dataset": DS, "config": "extraction", "split": split, "where": "\"source\"='squad-v2'",
                     "offset": off, "length": 100})
            total = d.get("num_rows_total", 0)
            for r in d["rows"]:
                row = {k: r["row"].get(k) for k in KEEP if k in r["row"]}
                row["split"] = split
                row["row_idx"] = r.get("row_idx")
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            off += len(d["rows"])
            print(split, off, "/", total, flush=True)
            if not d["rows"]:
                break
    print("done", split, off, flush=True)
