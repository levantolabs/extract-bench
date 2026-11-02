"""List every date gold the harness parser (dayfirst=False) cannot parse; count line-item date fragments."""
import json, sys
from pathlib import Path
DS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS.parent.parent))
from harness import score as S
items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]
for it in items:
    for f in it["fields"]:
        if f["type"] == "date" and f["gold"] is not None:
            vals = f["gold"] if isinstance(f["gold"], list) else [f["gold"]]
            if not any(S.parse_date(v, False) for v in vals):
                print(f"{f['name']:12s} {it['id'][:44]:44s} {vals}")
tot = bad = 0
for it in items:
    for f in it["fields"]:
        if f["type"] == "list" and f.get("items") == "date":
            for v in f["gold"]:
                tot += 1; bad += S.parse_date(v, False) is None
print(f"line-item date golds: {bad}/{tot} unparseable")
