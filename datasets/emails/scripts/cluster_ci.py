"""Committee clusters in the 100-item sample, plus a committee-clustered bootstrap CI (seed 20, 10,000 resamples)
for two published prediction files under the repo matcher. Output: scripts/cluster_ci.out.txt.

Usage (from the dataset dir): python3 -I scripts/cluster_ci.py
"""
import json, random
from collections import Counter

its = [json.loads(l) for l in open("items.jsonl", encoding="utf-8")]
c = Counter(it["meta"]["committee_cluster"] for it in its)
print("clusters", len(c), "top", c.most_common(6))
print("singletons", sum(1 for v in c.values() if v == 1))


def load(f):
    return {json.loads(l)["id"]: json.loads(l)["committee"] for l in open(f, encoding="utf-8")}


for name in ["gpt-4o_november_2024_prompt2", "claude_haiku_45_november_2024_prompt2"]:
    P = load(f"comparable/preds_{name}.sample.jsonl")
    hit = {it["id"]: float(str(P[it["id"]]).lower() == it["fields"][0]["gold"].lower()) for it in its}
    cl = {}
    for it in its:
        cl.setdefault(it["meta"]["committee_cluster"], []).append(hit[it["id"]])
    keys, r, ms = list(cl), random.Random(20), []
    for _ in range(10000):
        s = [cl[keys[r.randrange(len(keys))]] for _ in keys]
        ms.append(sum(sum(x) for x in s) / sum(len(x) for x in s))
    ms.sort()
    print(name, round(sum(hit.values()) / 100, 3), "committee-clustered CI95", round(ms[250], 3), round(ms[9749], 3))
