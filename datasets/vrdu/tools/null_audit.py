"""Print OCR context for a seeded sample of null-gold slots so a human can judge 'truly absent' vs 'missed label'.
  /opt/homebrew/bin/uv run --no-project python -I extract-bench/datasets/vrdu/tools/null_audit.py 30 > tools/null_audit.out.txt
"""
import json, random, re, sys
from pathlib import Path

DS = Path(__file__).resolve().parent.parent
LABELS = {  # label regex whose neighbourhood would hold the value if present
    "signer_title": r"title",
    "signer_name": r"signature|signed|name and title|/s/",
    "foreign_principle_name": r"foreign principal",
    "file_date": r"date of (signature|report)|dated|date\b",
    "registrant_name": r"name of registrant|registrant",
    "registration_num": r"registration n(o|umber)",
    "agency": r"agency",
    "flight_from": r"flight|schedule dates|period",
    "flight_to": r"flight|schedule dates|period",
    "tv_address": r"address|remit",
    "product": r"product",
    "property": r"station|property",
    "contract_num": r"contract|order",
    "gross_amount": r"gross|total",
    "advertiser": r"advertiser|client",
}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]
slots = [(it, f) for it in items for f in it["fields"] if f["type"] != "list" and f["gold"] is None]
print(f"null scalar slots: {len(slots)}")
rng = random.Random(20)
for it, f in sorted(rng.sample(slots, min(n, len(slots))), key=lambda x: (x[1]["name"], x[0]["id"])):
    t = it["text"]
    ms = [m.start() for m in re.finditer(LABELS.get(f["name"], f["name"]), t, re.I)][:2]
    print(f"\n### {f['name']} | {it['id']} | template={it['meta'].get('template', it['meta'].get('stratum'))}")
    for s in ms:
        print("   ...", repr(t[max(0, s - 80): s + 160]))
    if not ms:
        print("   (label not found in OCR)")
