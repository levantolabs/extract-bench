#!/usr/bin/env python3
"""Record the 2026-10-23 hand audit of CORD null golds (SURVEY C7) as data.

Sample: all 7 receipts with total_price gold null + 12 other receipts drawn with
random.Random(20).sample(sorted(remaining ids), 12). Every null scalar slot on each receipt
was checked against the image (downscaled previews, ~500 px wide).
Auditor: Claude (Opus 5.5), NOT blind to gold, single pass: [MY JUDGEMENT], not adjudication
by a human. Verdicts:
  absent      - no such value printed on the visible receipt
  masked      - a privacy-blurred line could hold it; unreadable, so null is right for a reader
  wrong_null  - the value is printed and readable; CORD files it under another key
  unclear     - partly cut off / not legible enough to decide
  ambiguous   - printed, but whether it is this field is a labelling choice
Run: python3 -I scripts/null_gold_audit.py  (reads items.jsonl, writes audit/null_gold_audit-2026-10-23.json)
"""
import json, os, random

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "items.jsonl"))}
total_null = sorted(i for i, it in items.items()
                    if any(f["name"] == "total_price" and f["gold"] is None for f in it["fields"]))
rest = sorted(i for i in items if i not in total_null)
rand = sorted(random.Random(20).sample(rest, 12))
EXCEPT = {
    ("cord-test-037", "total_price"): ("wrong_null", 20000,
        "'TOTAL 20.000' printed under the item line; CORD labels it total.total_etc"),
    ("cord-test-039", "total_price"): ("unclear", None,
        "a 'TOTAL HRG' line is cut off at the bottom edge of the photo; amount not legible"),
    ("cord-test-009", "discount_price"): ("ambiguous", 0,
        "'Discount (0 %)' printed with no amount; CORD labels only the key word (is_key=1), "
        "so gt_parse has no discount_price. 0 is a defensible reading"),
    ("cord-test-076", "menuqty_cnt"): ("ambiguous", None,
        "'ITEMS 5' printed; CORD labels it total.menutype_cnt ('50 -')"),
    ("cord-dev-049", "service_price"): ("masked", None,
        "a blurred line between SubTTL and TAX"),
    ("cord-dev-025", "cashprice"): ("masked", None, "blurred payment line after Total"),
    ("cord-dev-025", "creditcardprice"): ("masked", None, "blurred payment line after Total"),
    ("cord-test-092", "subtotal_price"): ("masked", None, "two blurred lines between item and TOTAL"),
    ("cord-test-092", "tax_price"): ("masked", None, "two blurred lines between item and TOTAL"),
}
slots, counts = [], {}
for iid in total_null + rand:
    for f in items[iid]["fields"]:
        if f["type"] == "list" or f["gold"] is not None:
            continue
        v, val, note = EXCEPT.get((iid, f["name"]), ("absent", None, ""))
        slots.append({"id": iid, "field": f["name"], "verdict": v, "value_on_image": val,
                      "note": note, "stratum": "total_price_null" if iid in total_null else "random_seed20"})
        counts[v] = counts.get(v, 0) + 1
assert all(any(s["id"] == i and s["field"] == k for s in slots) for i, k in EXCEPT), "exception not in sample"
out = {"date": "2026-10-23", "auditor": "Claude Opus 5.5, not blind to gold, single pass [MY JUDGEMENT]",
       "receipts": total_null + rand, "n_receipts": len(total_null) + len(rand),
       "n_null_slots": len(slots), "verdict_counts": counts, "slots": slots}
os.makedirs(os.path.join(HERE, "audit"), exist_ok=True)
with open(os.path.join(HERE, "audit", "null_gold_audit-2026-10-23.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "slots"}, indent=1))
