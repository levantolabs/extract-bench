"""Export the repo's published per-email committee predictions (8 reference models) keyed by our item ids.

Join = matcher.py's 8 merge columns (lower+strip). Writes comparable/published_predictions.jsonl:
  {"id", "csv_row", "model_file", "committee", "in_sample"}   for all 1,000 rows x 8 models,
and comparable/preds_<model_stem>.sample.jsonl (100 sample ids only) in score_emails.py's PREDS format.
These are published model outputs (MIT repo, commit 72b8d5fe), not our runs.

Usage: uv run --no-project --with pandas python -I scripts/export_published_preds.py <dataset_dir>
"""
import json, sys
from pathlib import Path

import pandas as pd

D = Path(sys.argv[1]).resolve()
FE = D / "raw" / "repo" / "fundraising-emails"
MERGE = ["email", "subject", "year", "month", "day", "hour", "minute", "domain"]
MODELS = [
    "gpt-4o_november_2024_prompt2.json", "gpt-4o-mini_november_2024.json",
    "claude37_sonnet_november_2024_prompt2.json", "gemini_25_november_2024_prompt2.json",
    "claude35_haiku_november_2024_prompt2.json", "gpt_41_mini_november_2024_prompt2.json",
    "claude_haiku_45_november_2024_prompt2.json", "gpt_41_nano_november_2024_prompt2.json",
]

g = pd.read_csv(FE / "training.csv")
g["csv_row"] = range(len(g))
for c in MERGE:
    g[c] = g[c].astype(str).str.strip().str.lower()
sample = {json.loads(l)["meta"]["csv_row"] for l in (D / "items.jsonl").read_text().splitlines() if l.strip()}
out_all = (D / "comparable" / "published_predictions.jsonl").open("w", encoding="utf-8")
for fn in MODELS:
    pj = pd.DataFrame(json.loads((FE / fn).read_text()))
    for c in MERGE:
        pj[c] = pj[c].astype(str).str.strip().str.lower()
    m = pd.merge(g[MERGE + ["csv_row"]], pj[MERGE + ["committee"]], on=MERGE, how="inner")
    assert m["csv_row"].is_unique and len(m) == 1000, (fn, len(m))
    stem = fn.removesuffix(".json")
    with (D / "comparable" / f"preds_{stem}.sample.jsonl").open("w", encoding="utf-8") as fs:
        for r in m.sort_values("csv_row").itertuples():
            c = None if pd.isna(r.committee) else r.committee
            rec = {"id": f"emails-r{r.csv_row:04d}", "csv_row": int(r.csv_row), "model_file": fn,
                   "committee": c, "in_sample": int(r.csv_row) in sample}
            out_all.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["in_sample"]:
                fs.write(json.dumps({"id": rec["id"], "committee": c}, ensure_ascii=False) + "\n")
out_all.close()
print("ok")
