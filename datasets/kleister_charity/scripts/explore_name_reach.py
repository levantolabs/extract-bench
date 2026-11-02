"""Exploration only: why charity_name golds are unreachable in the Textract layer (Ltd./Limited, 'The', etc.)."""
import json, re, sys
from pathlib import Path
DS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DS.parents[1]))
from harness.score import norm_text, squash  # noqa: E402
items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines()]
n = unreach = 0
fix = {"ltd->limited": 0, "drop_ltd": 0, "drop_the": 0, "add_the": 0, "none": 0}
ex = []
for it in items:
    f = next(x for x in it["fields"] if x["name"] == "charity_name")
    n += 1
    if it["meta"]["gold_reachable"]["charity_name"]:
        continue
    unreach += 1
    g = f["gold"] if isinstance(f["gold"], str) else f["gold"][0]
    t = squash(it["text"])
    cands = {
        "ltd->limited": re.sub(r"\bLtd\.?$", "Limited", g),
        "drop_ltd": re.sub(r"\s*\b(Ltd\.?|Limited)$", "", g),
        "drop_the": re.sub(r"^The\s+", "", g),
        "add_the": "The " + g,
    }
    hit = [k for k, v in cands.items() if v != g and squash(v) in t]
    for k in hit[:1]:
        fix[k] += 1
    if not hit:
        fix["none"] += 1
        ex.append(g)
print({"n": n, "unreachable": unreach, **fix})
print(ex[:15])
