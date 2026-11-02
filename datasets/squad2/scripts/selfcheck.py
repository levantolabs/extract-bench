"""Self-check for datasets/squad2/items.jsonl. Writes scripts/selfcheck.out.json.

- every line parses; harness validate_item passes; field names / types / metric / modality valid
- gold key present on every field; gold null <=> meta.is_impossible; images [] (no paths to check)
- text == the official context; every reference answer is a span of text at its answer_start
- TYPED view (harness score.outcome, rules a/b): gold as prediction -> 1.0; null on null-gold -> correct_abstain;
  the plausible (wrong) answer on null-gold -> hallucinated; null on answerable -> false_abstain
- COMPARABLE view (comparable/squad_v2_eval.py): gold -> 100/100; all-null -> NoAns 100, HasAns 0, overall 50;
  plausible answers on unanswerable + gold on answerable -> HasAns 100, NoAns 0
- o200k tokens (tiktoken) of text and of text+question: p50 / p95 / max; Sage 16,384 cap; question <= 256 chars
Usage: uv run --no-project --with tiktoken python -I scripts/selfcheck.py
"""
import json, statistics, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXB = HERE.parent.parent
sys.path.insert(0, str(EXB))
sys.path.insert(0, str(HERE / "comparable"))
from harness.common import FIELD_NAME, FIELD_TYPES, METRICS, MODALITIES, validate_item  # noqa: E402
from harness import score  # noqa: E402
import squad_v2_eval  # noqa: E402
import tiktoken  # noqa: E402

enc = tiktoken.get_encoding("o200k_base")
nt = lambda s: len(enc.encode(s or "", disallowed_special=()))
dev = json.loads((HERE / "raw/official/dev-v2.0.json").read_text())
ctx = {f"{a['title']}#{pi}": p["context"] for a in dev["data"] for pi, p in enumerate(a["paragraphs"])}

lines = (HERE / "items.jsonl").read_text().splitlines()
items, c = [], Counter()
for l in lines:
    it = json.loads(l)
    validate_item(it)
    items.append(it)
    assert it["dataset"] == "squad2" and it["modality"] in MODALITIES and it["metric"] in METRICS
    assert it["modality"] == "text" and it["text_source"] == "native" and it["images"] == []
    assert it["text"] == ctx[it["meta"]["doc_id"]], it["id"]
    assert len(it["fields"]) == 1
    f = it["fields"][0]
    assert FIELD_NAME.match(f["name"]) and f["type"] in FIELD_TYPES and "gold" in f
    assert (f["gold"] is None) == it["meta"]["is_impossible"], it["id"]
    for a in it["meta"]["answers"] + it["meta"]["plausible_answers"]:
        ok = it["text"][a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
        c["span_at_offset" if ok else "span_offset_mismatch"] += 1
        c["span_in_text" if a["text"] in it["text"] else "span_not_in_text"] += 1
    if len(f["instructions"]) > 256:
        c["question_over_256"] += 1
    if '"' in f["instructions"] or "\n" in f["instructions"]:
        c["question_has_quote_or_newline"] += 1
    if f["gold"] is not None and len(f["gold"]) > 1:
        c["answerable_with_multiple_distinct_golds"] += 1

ids = [it["id"] for it in items]
assert len(set(ids)) == len(ids)
assert len({it["meta"]["doc_id"] for it in items}) == len(items), "one question per context"

# ---- TYPED view
typed = Counter()
for it in items:
    f = it["fields"][0]
    g = f["gold"]
    for rule in ("a", "b"):
        if g is None:
            typed[score.outcome(None, None, g, "text", "token_f1", f, rule)] += 1
            pl = it["meta"]["plausible_answers"]
            if pl:
                typed[("plausible_on_null",) + score.outcome(pl[0]["text"], 0.9, g, "text", "token_f1", f, rule)] += 1
            typed[("low_conf_answer_on_null",) + score.outcome("x", 0.3, g, "text", "token_f1", f, rule)] += 1
        else:
            for gv in g:
                typed[("gold_as_pred",) + score.outcome(gv, 1.0, g, "text", "token_f1", f, rule)] += 1
            typed[("null_on_answerable",) + score.outcome(None, None, g, "text", "token_f1", f, rule)] += 1
typed = {" | ".join(map(str, k)) if isinstance(k, tuple) else str(k): v for k, v in typed.items()}

# ---- COMPARABLE view
gold_preds = {it["id"]: {"value": (it["fields"][0]["gold"] or [None])[0], "conf": 1.0} for it in items}
null_preds = {it["id"]: {"value": None} for it in items}
plaus = {it["id"]: {"value": (it["fields"][0]["gold"][0] if it["fields"][0]["gold"] else
                              (it["meta"]["plausible_answers"][0]["text"] if it["meta"]["plausible_answers"] else "x")),
                    "conf": 0.9} for it in items}
comp = {k: {kk: vv for kk, vv in squad_v2_eval.evaluate(items, p).items() if kk != "note"}
        for k, p in (("gold", gold_preds), ("all_null", null_preds), ("gold_plus_plausible_on_null", plaus))}
assert comp["gold"]["exact"] == 100.0 and comp["gold"]["f1"] == 100.0
assert comp["all_null"]["NoAns_f1"] == 100.0 and comp["all_null"]["HasAns_f1"] == 0.0

# ---- tokens
def stats(v):
    v = sorted(v)
    q = statistics.quantiles(v, n=100, method="inclusive")
    return {"p50": statistics.median(v), "p95": round(q[94], 1), "max": v[-1], "min": v[0], "total": sum(v)}

tt = [nt(it["text"]) for it in items]
tq = [nt(it["text"]) + nt(it["fields"][0]["instructions"]) for it in items]
out = {
    "n_lines": len(lines), "n_items": len(items),
    "strata": dict(Counter(it["meta"]["stratum"] for it in items)),
    "holdout": {f"{s}:{h}": n for (s, h), n in Counter((it["meta"]["stratum"], it["meta"]["holdout"]) for it in items).items()},
    "articles": len({it["meta"]["article"] for it in items}),
    "max_items_per_article": max(Counter(it["meta"]["article"] for it in items).values()),
    "contexts_distinct": len({it["meta"]["doc_id"] for it in items}),
    "gold_null_fields": sum(1 for it in items if it["fields"][0]["gold"] is None),
    "checks": dict(c),
    "typed_view": typed,
    "comparable_view": comp,
    "tokens_o200k_text": stats(tt),
    "tokens_o200k_text_plus_question": stats(tq),
    "n_over_sage_16384": sum(1 for x in tq if x > 16384),
    "n_over_clairic_8192_pass": sum(1 for x in tq if x > 8192),
    "question_chars": stats([len(it["fields"][0]["instructions"]) for it in items]),
}
(HERE / "scripts/selfcheck.out.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
