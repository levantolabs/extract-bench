"""Gold-noise candidates (SURVEY L7 rule): items where the 8 published reference models' committee predictions
agree with each other (all 8 identical after the view's normalisation) but not with the gold. Gold is NOT changed;
this list is the blind-review queue for adjudication, applied to all arms equally.

Views: repo matcher (lower) and alnum-squash. Usage: python3 -I scripts/gold_disagreement.py <dataset_dir>
"""
import json, re, sys, unicodedata
from pathlib import Path

D = Path(sys.argv[1]).resolve()


def squash(s):
    s = unicodedata.normalize("NFKC", str(s or "")).casefold()
    return re.sub(r"[^0-9a-z]", "", s)


def lower(s):
    return "none" if s is None else str(s).lower()


anchor = {json.loads(l)["id"]: json.loads(l) for l in (D / "anchor" / "items_all1000.jsonl").open(encoding="utf-8")}
sample = {json.loads(l)["id"] for l in (D / "items.jsonl").open(encoding="utf-8")}
P = {}
for l in (D / "comparable" / "published_predictions.jsonl").open(encoding="utf-8"):
    r = json.loads(l)
    P.setdefault(r["id"], {})[r["model_file"]] = r["committee"]

out = {"n_models": 8}
for view, fn in (("repo_matcher", lower), ("squashed", squash)):
    rows = []
    for iid, it in anchor.items():
        g = it["fields"][0]["gold"]
        ps = list(P[iid].values())
        assert len(ps) == 8
        if len({fn(p) for p in ps}) == 1 and fn(ps[0]) != fn(g):
            rows.append({"id": iid, "in_sample": iid in sample, "gold": g, "all8_pred": ps[0]})
    out[view] = {"n_all1000": len(rows), "n_sample": sum(r["in_sample"] for r in rows), "items": rows}
    out[view]["any_model_wrong_sample"] = sum(
        1 for iid in sample if any(fn(p) != fn(anchor[iid]["fields"][0]["gold"]) for p in P[iid].values()))
print(json.dumps(out, indent=1, ensure_ascii=False))
