"""Eligible vs excluded dev paragraphs: questions per paragraph, length, unanswerable share. -> derived/pool_compare.out.json
Usage: python -I pool_compare.py <squad2_dir>"""
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
dev = json.loads((d / "raw/official/dev-v2.0.json").read_text())
e = json.loads((d / "derived/exclusion.json").read_text())
ctx = {f"{x['title']}#{pi}": p for x in dev["data"] for pi, p in enumerate(x["paragraphs"])}
def st(keys):
    nq = [len(ctx[k]["qas"]) for k in keys]
    return {"n": len(keys), "mean_questions": round(statistics.mean(nq), 2),
            "median_chars": statistics.median(len(ctx[k]["context"]) for k in keys),
            "mean_unanswerable_share": round(statistics.mean(sum(q["is_impossible"] for q in ctx[k]["qas"]) / len(ctx[k]["qas"]) for k in keys), 3)}
out = {"eligible": st(e["eligible"]), "excluded": st(list(e["excluded"]))}
(d / "derived/pool_compare.out.json").write_text(json.dumps(out, indent=1)); print(out)
