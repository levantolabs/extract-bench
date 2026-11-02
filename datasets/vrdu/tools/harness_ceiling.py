"""Harness-side ceilings (offline): score gold-as-prediction with harness/score.py.
  verbatim: each field returns its first acceptable gold (lists: verbatim official list)
  typed:    dates returned as ISO (harness parse_date of the best parseable gold, else verbatim), money as a float,
            list dates as ISO - what a typed-output arm gets if it reads every value correctly."""
import json, sys
from collections import defaultdict
from pathlib import Path
DS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS.parent.parent))
from harness import score as S
items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]

def first(g):
    return g if isinstance(g, str) else g[0]

def typed(f):
    g = f["gold"]
    if f["type"] == "list":
        lst = g[-1] if (g and isinstance(g[0], list)) else g
        if f.get("items") == "date":
            return [S.parse_date(v, False) or v for v in lst]
        if f.get("items") == "number":
            return [S.parse_number(v) if S.parse_number(v) is not None else v for v in lst]
        return lst
    vals = [g] if isinstance(g, str) else g
    if f["type"] == "date":
        for v in vals:
            if S.parse_date(v, False):
                return S.parse_date(v, False)
        return S.parse_date(first(g), False) or "1900-01-01"  # unparseable gold: a typed arm cannot echo it
    if f["type"] == "money":
        n = S.parse_number(first(g))
        return n if n is not None else first(g)
    return first(g)

out = {}
for mode in ("verbatim", "typed"):
    acc = defaultdict(list)
    for it in items:
        corp = "registration" if it["id"].startswith("vrdu-reg") else "adbuy"
        preds = {}
        for f in it["fields"]:
            if f["gold"] is None:
                preds[f["name"]] = {"value": None, "conf": 0.99}
            elif mode == "verbatim":
                g = f["gold"]
                v = (g[0] if (g and isinstance(g[0], list)) else g) if f["type"] == "list" else first(g)
                preds[f["name"]] = {"value": v, "conf": 0.99}
            else:
                preds[f["name"]] = {"value": typed(f), "conf": 0.99}
        sc = S.score_item(it, preds)
        for fr in sc["fields"]:
            if fr["gold_null"]:
                continue
            acc[f"{corp}.{fr['name']}"].append(fr.get("score_a", fr.get("score")))
    out[mode] = {k: round(sum(v) / len(v), 4) for k, v in sorted(acc.items()) if v and v[0] is not None}
print(json.dumps(out, indent=1))
