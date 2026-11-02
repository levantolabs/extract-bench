"""How internally consistent are the synthetic invoices? (sampled 150 items)
- total vs subtotal + tax - discount (where all printed)
- due_date before invoice_date"""
import json, sys
from pathlib import Path
D = Path(sys.argv[1])
items = [json.loads(l) for l in open(D / "items.jsonl")]
arith = {"checked": 0, "match_0.02": 0, "examples": []}
dates = {"checked": 0, "due_before_issue": 0}
for it in items:
    g = {f["name"]: f["gold"] for f in it["fields"]}
    # dates first: the arithmetic block below `continue`s on tax-excluded items, which used to
    # skip the date check too (fixed by the verifier, 2026-10-23: 45/90 -> 54/105).
    if g.get("due_date") and g.get("invoice_date"):
        dates["checked"] += 1
        dates["due_before_issue"] += g["due_date"] < g["invoice_date"]
    st, tot = g.get("subtotal"), g.get("total")
    if isinstance(st, (int, float)) and isinstance(tot, (int, float)):
        tax = g.get("tax") if isinstance(g.get("tax"), (int, float)) else 0.0
        disc = g["discount"][0] if isinstance(g.get("discount"), list) else 0.0
        if "tax" in it["meta"]["excluded_fields"]:
            continue
        arith["checked"] += 1
        ok = abs(st + tax - disc - tot) <= 0.02
        arith["match_0.02"] += ok
        if not ok and len(arith["examples"]) < 5:
            arith["examples"].append({"id": it["id"], "subtotal": st, "tax": tax, "discount": disc, "total": tot})
out = {"arith": arith, "dates": dates}
(D / "tools" / "consistency.out.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
