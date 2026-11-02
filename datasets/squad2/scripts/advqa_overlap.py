"""Do AdversarialQA (a tasksource extraction source) passages repeat SQuAD 2.0 dev contexts? Exact + 120-char head match.
Usage: uv run --no-project --with pyarrow python -I advqa_overlap.py <squad2_dir>"""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
d = Path(sys.argv[1])
dev = json.loads((d / "raw/official/dev-v2.0.json").read_text())
ctx = {}
for a in dev["data"]:
    for pi, p in enumerate(a["paragraphs"]):
        ctx[p["context"]] = f"{a['title']}#{pi}"
heads = {c[:120]: k for c, k in ctx.items()}
out = {}
for s in ("train", "validation", "test"):
    t = pq.read_table(d / f"raw/adversarial_qa/adversarialQA_{s}.parquet", columns=["title", "context"]).to_pylist()
    cs = {r["context"] for r in t}
    exact = sorted({ctx[c] for c in cs if c in ctx})
    head = sorted({heads[c[:120]] for c in cs if c[:120] in heads})
    titles = sorted({r["title"] for r in t} & {k.rsplit("#", 1)[0] for k in ctx.values()})
    out[s] = {"rows": len(t), "contexts": len(cs), "exact_dev_contexts": len(exact), "head_dev_contexts": len(head),
              "shared_titles": titles, "dev_contexts_hit": sorted(set(exact) | set(head))}
(d / "derived/advqa_overlap.out.json").write_text(json.dumps(out, indent=1))
print(json.dumps({s: {k: (v if k != "dev_contexts_hit" else len(v)) for k, v in o.items()} for s, o in out.items()}, indent=1))
