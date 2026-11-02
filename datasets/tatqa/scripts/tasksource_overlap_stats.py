"""Describe the tasksource 'tatqa/text-span' rows vs TAT-QA train/dev/test, and the sampled items' overlap.

Run: /opt/homebrew/bin/uv run --no-project python -I extract-bench/datasets/tatqa/scripts/tasksource_overlap_stats.py
Writes scripts/tasksource_overlap_stats.out.json.
"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

DS = Path(__file__).resolve().parent.parent
RAW = DS / "raw"
norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "")).strip()

ctx, para_split = {}, {}
docs = {}
for sp in ("train", "dev", "test_gold"):
    for doc in json.loads((RAW / f"tatqa_dataset_{sp}.json").read_text()):
        ps = [norm(p["text"]) for p in sorted(doc["paragraphs"], key=lambda p: int(p["order"]))]
        ctx.setdefault(" ".join(ps), []).append((sp, doc["table"]["uid"]))
        for p in ps:
            para_split.setdefault(p, set()).add(sp)
        if sp == "test_gold":
            docs[doc["table"]["uid"]] = (doc, ps)
ts = json.loads((RAW / "tasksource" / "tatqa_text_span_rows.json").read_text())
states = [(sp, norm(r["state"])) for sp, rows in ts["splits"].items() for r in rows]
exact = Counter()
for sp, s in states:
    hit = ctx.get(s)
    exact[(sp, "+".join(sorted({h[0] for h in hit})) if hit else "no_exact_context")] += 1

report = json.loads((DS / "build_report.json").read_text())
items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]

# excluded contexts (recompute with the build's rule)
sys.path.insert(0, str(DS))
import build  # noqa: E402

ts_states = [(sp, s) for sp, s in states]
excluded = []
for uid, (doc, ps) in docs.items():
    ov = build.overlap(doc, ts_states, set())
    if ov["excluded"]:
        cov_rows = [sp for sp, s in ts_states if any(p in s for p in ps if len(p) >= build.MIN_PARA)]
        excluded.append({"table_uid": uid, "n_questions": len(doc["questions"]), "paragraphs": ps,
                         "paragraph_tatqa_splits": [sorted(para_split[p]) for p in ps],
                         "covering_ts_rows_by_split": dict(Counter(cov_rows)), **ov})

part = Counter((it["meta"]["tasksource"]["paras_covered"], it["meta"]["tasksource"]["paras_long"])
               for it in items if it["meta"]["tasksource"]["partial_overlap"])
part_types = Counter(it["meta"]["answer_type"] for it in items if it["meta"]["tasksource"]["partial_overlap"])
# are partially-overlapping paragraphs ones TAT-QA itself reuses across splits?
shared_with_trdev = 0
cov_total = 0
for it in items:
    if not it["meta"]["tasksource"]["partial_overlap"]:
        continue
    doc, ps = docs[it["meta"]["doc_id"]]
    for p in ps:
        if len(p) >= build.MIN_PARA and any(p in s for _, s in ts_states):
            cov_total += 1
            shared_with_trdev += bool(para_split[p] & {"train", "dev"})
out = {"tasksource_rows": {f"{a}|{b}": n for (a, b), n in sorted(exact.items())},
       "n_tasksource_rows": len(states), "excluded_contexts": excluded,
       "sample_partial_overlap_items": sum(part.values()), "sample_partial_by_type": dict(part_types),
       "sample_partial_covered_of_long": {f"{a}/{b}": n for (a, b), n in sorted(part.items())},
       "covered_paragraphs_in_sample_also_in_tatqa_train_or_dev": f"{shared_with_trdev}/{cov_total}",
       "sample_clean_items (no tasksource paragraph)": sum(not it["meta"]["tasksource"]["partial_overlap"]
                                                           for it in items)}
(DS / "scripts" / "tasksource_overlap_stats.out.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False)[:4000])
