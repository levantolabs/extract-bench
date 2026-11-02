"""Per-template gold status table + o200k tokens of the field definitions (what Sage/Clairic bill per call)."""
import json, sys, collections, statistics
from pathlib import Path
import tiktoken
D = Path(sys.argv[1]); enc = tiktoken.get_encoding("o200k_base")
items = [json.loads(l) for l in open(D / "items.jsonl")]
names = [f["name"] for f in items[0]["fields"]] + [n for n in items[0]["meta"]["excluded_fields"]]
order = ["invoice_number","invoice_date","due_date","po_number","seller_name","buyer_name","subtotal","tax","discount","total","amount_due"]
tab = {}
for it in items:
    t = it["meta"]["template"]; row = tab.setdefault(t, {})
    for f in it["fields"]:
        g = f["gold"]; row[f["name"]] = "null" if g is None else ("alt" if isinstance(g, list) else "val")
    for n in it["meta"]["excluded_fields"]:
        row[n] = "EXCL"
lines = ["| template | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
for t in [50, 7, 32, 39, 2, 12, 4, 49, 10, 25]:
    lines.append(f"| T{t} | " + " | ".join(tab[t].get(n, "?") for n in order) + " |")
ft = [len(enc.encode(json.dumps({f["name"]: {"type": f["type"], "instructions": f["instructions"]} for f in it["fields"]}))) for it in items]
out = {"table_md": "\n".join(lines), "field_def_tokens_o200k": {"p50": statistics.median(ft), "max": max(ft), "min": min(ft)}}
(D / "tools" / "field_tokens.out.json").write_text(json.dumps(out, indent=1))
print(out["table_md"]); print(out["field_def_tokens_o200k"])
