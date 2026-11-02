"""Does the datasets-server (refs/convert/parquet 8614e633) serve the same squad-v2 rows as main 8b1afcc6?
Compares the 98 rows fetched via /filter (raw/tasksource/squad_v2_rows.validation.jsonl) with the main parquet.
Usage: uv run --no-project --with pyarrow python -I convert_vs_main.py <squad2_dir>"""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
d = Path(sys.argv[1])
t = {r["id"]: r for r in pq.read_table(d / "raw/tasksource/extraction_validation-00000-of-00001.parquet",
                                         columns=["source", "id", "metadata", "state"]).to_pylist()}
rows = [json.loads(l) for l in (d / "raw/tasksource/squad_v2_rows.validation.jsonl").read_text().splitlines() if l.strip()]
same = sum(1 for r in rows if r["id"] in t and t[r["id"]]["state"] == r["state"] and t[r["id"]]["metadata"] == r["metadata"])
out = {"filter_rows": len(rows), "identical_in_main_parquet": same,
       "main_parquet_squad_v2_rows": sum(1 for r in t.values() if r["source"] == "squad-v2")}
(d / "derived/convert_vs_main.out.json").write_text(json.dumps(out, indent=1)); print(out)
