"""Build extract-bench/datasets/squad2/items.jsonl from raw/ (SQuAD 2.0 dev, minus tasksource overlap). $0, offline.

Pre-registered sampling (seed 20), written before the first run:
  1. Source: raw/official/dev-v2.0.json (SQuAD-explorer commit eee5fdbf, blob c6fa1a89), 35 articles, 1,204 contexts,
     11,873 questions. raw/hf/validation-00000-of-00001.parquet (rajpurkar/squad_v2 @ 3ffb306f) is only cross-checked.
  2. Exclusion (SURVEY L1), against tasksource/tasksource-jev-typed-decisions `extraction` @ main 8b1afcc6.
     A dev context is ineligible if ANY of:
     a. one of its question ids appears in the metadata of any tasksource `squad-v2` row (validation/test from the
        parquet; train from the survey pull of the same revision);
     b. its full context string equals, contains or is contained in any tasksource `squad-v2` validation/test state,
        or shares the first 120 characters with any `squad-v2` state/head of any split (this catches the one
        SQuAD-train paragraph that repeats a dev paragraph);
     c. a probe (80 chars at start/middle/end, whitespace-collapsed) of it occurs in ANY validation/test row of
        ANY source, squad-v2 included (derived/xsource_hits.json from scripts/xsource_probe.py; required). This
        also drops dev paragraphs that only share a sentence with a tasksource row (e.g. Economic_inequality#20
        repeats a sentence of #29). Added after the 23 Oct finding that AdversarialQA val/test reuse SQuAD 1.1
        dev passages (derived/advqa_overlap.out.json) and MuSiQue validation reuses SQuAD paragraphs.
     Whole contexts are dropped, not just the listed question ids, so no sampled passage is in a tasksource eval
     split. Train-side coverage is partial: squad-v2 qids + 120-char heads, and the AdversarialQA original train
     split (0 dev passages); other train sources were not scanned (780 MB parquet, ~25-250 KB/s on 23 Oct).
  3. One question per context (independent passages; bootstrap cluster = context).
  4. Allocation: 60 answerable + 60 unanswerable (is_impossible). rng = random.Random(20).
     - articles with >=1 eligible context, sorted by title, then rng.shuffle;
     - within each article, eligible contexts sorted by paragraph index, then rng.shuffle;
     - round-robin over articles, one context per visit. For each context the wanted stratum is the one with
       fewer picks so far (ties -> answerable); if the context has no question of that stratum the other stratum
       is used if it still needs items, else the context is skipped. The question is rng.choice over the
       context's question ids of that stratum, sorted.
     - stop when both strata hold 60.
  5. Sealed 20% holdout, stratified: random.Random(20).sample(sorted(ids_of_stratum), 12) for each stratum
     (answerable first) -> meta.holdout (24 items).
Item: one `text` field `answer`; instructions = the question verbatim (Sage bare-question mode uses it as the
question); gold = de-duplicated reference answer texts whose SQuAD normalisation is non-empty (as the official
evaluate-v2.0 does), or null for unanswerable. Metric token_f1 (SQuAD v2 normalisation).

Usage: uv run --no-project --with tiktoken --with pyarrow python -I build.py
"""
from __future__ import annotations

import hashlib, json, random, re, string, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
KEY = "squad2"
SEED = 20
N_PER_STRATUM = 60
HOLDOUT_PER_STRATUM = 12
USE_CASE = "Q&A over a passage where the answer may be absent (abstention)"
DEV_JSON = RAW / "official" / "dev-v2.0.json"
HF_PARQUET = RAW / "hf" / "validation-00000-of-00001.parquet"
TS_PARQUET = {s: RAW / "tasksource" / f"extraction_{s}-00000-of-00001.parquet" for s in ("validation", "test")}
TS_SURVEY = RAW / "tasksource" / "survey_tasksource_extraction_tatqa_squad_rows.json"
XSOURCE = HERE / "derived" / "xsource_hits.json"
ADVQA = HERE / "derived" / "advqa_overlap.out.json"
EXPECT_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_blob_sha1(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def squad_norm(s: str) -> str:
    """normalize_answer() of the official evaluate-v2.0.py (read, not executed)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def load_dev():
    d = json.loads(DEV_JSON.read_text())
    ctxs = []          # one dict per context
    for a in d["data"]:
        for pi, p in enumerate(a["paragraphs"]):
            ctxs.append({"title": a["title"], "pi": pi, "context": p["context"], "qas": p["qas"]})
    return d["version"], ctxs


def _qids(m) -> list:
    m = json.loads(m) if isinstance(m, str) else (m or {})
    return m.get("question_ids") or []


def load_tasksource():
    """squad-v2 rows of the tasksource extraction config.
    validation/test: full rows from the main-8b1afcc6 parquet. train: question ids + 300-char state heads from the
    survey pull (refs/convert/parquet 8614e633, 23 Oct; the 780 MB train parquet was not downloaded)."""
    import pyarrow.parquet as pq
    qids, states, heads = set(), [], set()
    per_split = Counter()
    for split, f in TS_PARQUET.items():
        t = pq.read_table(f, columns=["source", "id", "metadata", "state"]).to_pylist()
        for r in t:
            if r["source"] != "squad-v2":
                continue
            per_split[split] += 1
            qids.update(_qids(r["metadata"]))
            states.append(r["state"])
            heads.add(r["state"][:120])
    survey = json.loads(TS_SURVEY.read_text())
    survey_eval_qids = set()
    for split, rows in survey.items():
        for r in rows:
            if r["source"] != "squad-v2":
                continue
            if split == "train":
                per_split["train(survey heads)"] += 1
                qids.update(_qids(r["metadata"]))
                heads.add(r["state_head"][:120])
            else:
                survey_eval_qids.update(_qids(r["metadata"]))
    eval_qids_now = set()
    for f in TS_PARQUET.values():
        for r in pq.read_table(f, columns=["source", "metadata"]).to_pylist():
            if r["source"] == "squad-v2":
                eval_qids_now.update(_qids(r["metadata"]))
    drift = {"survey_eval_qids": len(survey_eval_qids), "parquet_eval_qids": len(eval_qids_now),
             "identical": survey_eval_qids == eval_qids_now}
    return qids, states, heads, dict(per_split), drift


def hf_crosscheck(ctxs) -> dict:
    """The HF parquet must carry the same ids, questions, contexts and answers as the official JSON."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"checked": False, "why": "pyarrow not installed"}
    t = pq.read_table(HF_PARQUET).to_pylist()
    off = {}
    for c in ctxs:
        for q in c["qas"]:
            off[q["id"]] = (c["title"], c["context"], q["question"], [a["text"] for a in q["answers"]],
                            [a["answer_start"] for a in q["answers"]])
    mism = 0
    for r in t:
        o = off.get(r["id"])
        h = (r["title"], r["context"], r["question"], list(r["answers"]["text"]), list(r["answers"]["answer_start"]))
        if o != h:
            mism += 1
    return {"checked": True, "hf_rows": len(t), "official_questions": len(off),
            "ids_equal": set(off) == {r["id"] for r in t}, "row_mismatches": mism}


def main() -> None:
    report: dict = {"dataset": KEY, "seed": SEED}
    assert sha256(DEV_JSON) == EXPECT_SHA256, "dev-v2.0.json sha256 changed"
    report["dev_json"] = {"sha256": sha256(DEV_JSON), "git_blob_sha1": git_blob_sha1(DEV_JSON)}
    version, ctxs = load_dev()
    report["dev"] = {"version": version, "articles": len({c["title"] for c in ctxs}), "contexts": len(ctxs),
                     "questions": sum(len(c["qas"]) for c in ctxs),
                     "answerable": sum(1 for c in ctxs for q in c["qas"] if not q.get("is_impossible")),
                     "unanswerable": sum(1 for c in ctxs for q in c["qas"] if q.get("is_impossible"))}
    report["hf_crosscheck"] = hf_crosscheck(ctxs)

    # ---- exclusion
    ts_qids, ts_states, ts_heads, ts_split, ts_drift = load_tasksource()
    dev_qids = {q["id"] for c in ctxs for q in c["qas"]}
    ts_state_set = set(ts_states)
    assert XSOURCE.exists(), "run scripts/xsource_probe.py first (exclusion rule c is required)"
    xs = json.loads(XSOURCE.read_text())
    xs_hits = (xs or {}).get("hits", {})
    why = Counter()
    eligible = []
    for c in ctxs:
        reasons = []
        if any(q["id"] in ts_qids for q in c["qas"]):
            reasons.append("qid_in_tasksource_squad_v2")
        cx = c["context"]
        if cx in ts_state_set or cx[:120] in ts_heads or any(cx in s or (len(s) > 200 and s in cx) for s in ts_states):
            reasons.append("context_text_in_tasksource_squad_v2")
        h = xs_hits.get(f"{c['title']}#{c['pi']}", [])
        if any(x["source"] != "squad-v2" for x in h):
            reasons.append("probe_in_tasksource_eval_other_source")
        if any(x["source"] == "squad-v2" for x in h):
            reasons.append("probe_in_tasksource_eval_squad_v2")
        for r in reasons:
            why[r] += 1
        c["excluded"] = reasons
        if not reasons:
            eligible.append(c)
    report["tasksource"] = {"commit": "8b1afcc6ad12554b347134a91b5f0dc6ae2361db",
                            "squad_v2_rows_by_split": ts_split, "squad_v2_qids_in_metadata": len(ts_qids),
                            "qids_that_are_dev": len(ts_qids & dev_qids), "survey_vs_parquet_eval_qids": ts_drift,
                            "xsource_probe_applied": xs is not None,
                            "xsource_summary": {k: v for k, v in (xs or {}).items()
                                                if k not in ("hits", "hit_contexts", "hit_contexts_not_in_squad_v2_rows")},
                            "advqa_original_overlap": ({s: {k: v for k, v in o.items() if k != "dev_contexts_hit"}
                                                        for s, o in json.loads(ADVQA.read_text()).items()}
                                                       if ADVQA.exists() else None)}
    excl_q = sum(len(c["qas"]) for c in ctxs if c["excluded"])
    (HERE / "derived").mkdir(exist_ok=True)
    (HERE / "derived" / "exclusion.json").write_text(json.dumps(
        {"excluded": {f"{c['title']}#{c['pi']}": c["excluded"] for c in ctxs if c["excluded"]},
         "eligible": [f"{c['title']}#{c['pi']}" for c in eligible]}, indent=0, ensure_ascii=False))
    report["exclusion"] = {"contexts_excluded": len(ctxs) - len(eligible), "by_reason": dict(why),
                           "questions_in_excluded_contexts": excl_q,
                           "eligible_contexts": len(eligible),
                           "eligible_articles": len({c["title"] for c in eligible}),
                           "eligible_answerable_q": sum(1 for c in eligible for q in c["qas"] if not q.get("is_impossible")),
                           "eligible_unanswerable_q": sum(1 for c in eligible for q in c["qas"] if q.get("is_impossible"))}

    # ---- sampling (see docstring)
    rng = random.Random(SEED)
    by_art = defaultdict(list)
    for c in eligible:
        by_art[c["title"]].append(c)
    arts = sorted(by_art)
    rng.shuffle(arts)
    for a in arts:
        by_art[a].sort(key=lambda c: c["pi"])
        rng.shuffle(by_art[a])
    picks = {"ans": [], "unans": []}
    cursor = {a: 0 for a in arts}
    while len(picks["ans"]) < N_PER_STRATUM or len(picks["unans"]) < N_PER_STRATUM:
        progressed = False
        for a in arts:
            if len(picks["ans"]) >= N_PER_STRATUM and len(picks["unans"]) >= N_PER_STRATUM:
                break
            if cursor[a] >= len(by_art[a]):
                continue
            c = by_art[a][cursor[a]]
            cursor[a] += 1
            progressed = True
            pools = {"ans": sorted(q["id"] for q in c["qas"] if not q.get("is_impossible")),
                     "unans": sorted(q["id"] for q in c["qas"] if q.get("is_impossible"))}
            want = "ans" if len(picks["ans"]) <= len(picks["unans"]) else "unans"
            other = "unans" if want == "ans" else "ans"
            for s in (want, other):
                if len(picks[s]) < N_PER_STRATUM and pools[s]:
                    picks[s].append((c, rng.choice(pools[s])))
                    break
        if not progressed:
            raise SystemExit(f"ran out of eligible contexts: {len(picks['ans'])} ans, {len(picks['unans'])} unans")

    hrng = random.Random(SEED)
    holdout = set()
    for s in ("ans", "unans"):
        ids = sorted(f"{KEY}-{qid}" for _, qid in picks[s])
        holdout |= set(hrng.sample(ids, HOLDOUT_PER_STRATUM))

    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        ntok = lambda s: len(enc.encode(s, disallowed_special=()))
    except ImportError:
        ntok = None

    items, checks = [], Counter()
    for s in ("ans", "unans"):
        for c, qid in picks[s]:
            q = next(x for x in c["qas"] if x["id"] == qid)
            ans = q["answers"]
            for a in ans + q.get("plausible_answers", []):
                ok = c["context"][a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
                checks["offset_ok" if ok else "offset_mismatch"] += 1
            if s == "ans":
                gold, seen = [], set()
                for a in ans:
                    if squad_norm(a["text"]) and a["text"] not in seen:
                        seen.add(a["text"])
                        gold.append(a["text"])
                if not gold:
                    checks["answerable_all_gold_norm_empty"] += 1
                    gold = [""]
            else:
                gold = None
            question = q["question"].strip()
            iid = f"{KEY}-{qid}"
            meta = {
                "split": "dev", "squad_version": version, "qid": qid, "article": c["title"],
                "paragraph_index": c["pi"], "doc_id": f"{c['title']}#{c['pi']}",
                "is_impossible": bool(q.get("is_impossible")), "stratum": "answerable" if s == "ans" else "unanswerable",
                "answers": [{"text": a["text"], "answer_start": a["answer_start"]} for a in ans],
                "plausible_answers": [{"text": a["text"], "answer_start": a["answer_start"]}
                                      for a in q.get("plausible_answers", [])],
                "n_questions_in_context": len(c["qas"]), "question_raw": q["question"],
                "sage_shape": "question",
                "sage_question_needs_sanitising": bool(re.search(r'["\n]', question)) or len(question) > 256,
                "tasksource_excluded": False, "holdout": iid in holdout,
                "comparable": "official evaluate-v2.0.py EM/F1 (+ na_prob for best_f1 oracle)",
            }
            if ntok:
                meta["tokens_o200k"] = {"text": ntok(c["context"]), "question": ntok(question)}
            items.append({
                "id": iid, "dataset": KEY, "use_case": USE_CASE, "modality": "text",
                "text": c["context"], "text_source": "native", "images": [],
                "fields": [{"name": "answer", "type": "text", "instructions": question, "gold": gold}],
                "metric": "token_f1", "meta": meta,
            })
    out = HERE / "items.jsonl"
    with out.open("w") as fo:
        for it in items:
            fo.write(json.dumps(it, ensure_ascii=False) + "\n")

    report["sample"] = {
        "n": len(items), "answerable": len(picks["ans"]), "unanswerable": len(picks["unans"]),
        "articles": len({it["meta"]["article"] for it in items}),
        "per_article": dict(sorted(Counter(it["meta"]["article"] for it in items).items())),
        "contexts_distinct": len({it["meta"]["doc_id"] for it in items}),
        "holdout_n": len(holdout), "holdout_ids": sorted(holdout),
        "checks": dict(checks), "tokens_counted": bool(ntok),
        "items_sha256": sha256(out),
    }
    (HERE / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("dev", "hf_crosscheck", "tasksource", "exclusion")}, indent=1))
    print(json.dumps({k: v for k, v in report["sample"].items() if k not in ("holdout_ids", "per_article")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
