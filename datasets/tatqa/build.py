"""Build extract-bench/datasets/tatqa/items.jsonl from raw/ (TAT-QA test gold, proportional sample, Tier C).

Run (from anywhere; nothing here executes dataset code):
  /opt/homebrew/bin/uv run --no-project --with tiktoken python -I extract-bench/datasets/tatqa/build.py

Inputs (all under raw/, see CARD.md for URLs + hashes):
  tatqa_dataset_test_gold.json       TAT-QA test with gold (GitHub NExTplusplus/TAT-QA @ 870accc4; file added 12fd7c8b)
  tatqa_dataset_train.json / _dev    only to flag paragraph overlap between our test items and TAT-QA train/dev
  tasksource/tatqa_text_span_rows.json  every 'tatqa/text-span' row of tasksource/tasksource-jev-typed-decisions
                                     (config 'extraction'), fetched by scripts/fetch_tasksource_tatqa.py

Pre-declared rules (spec + SURVEY L1/L11/C15):
  * Unit = one TAT-QA question -> one item with one field `answer`. id = TAT-QA question uid.
  * Exclusion (before sampling): drop every question of a test context (table+paragraphs) if
      (a) its whole paragraph text (whitespace-normalised, in order) is a substring of a tasksource state, or
      (b) every one of its paragraphs (>= 20 chars) appears inside some tasksource state, or
      (c) any of its question uids appears in a tasksource row's metadata.question_ids.
    tasksource states are TAT-QA paragraphs only (no table), so "full context" is tested on the paragraph text.
    Partial overlap (some paragraphs) is NOT excluded; it is flagged in meta.tasksource for slicing.
  * Sample n=150, seed 20, proportional to the FULL test answer types (span 714 / arithmetic 699 / multi-span 210 /
    count 40) by largest remainder -> 64 / 63 / 19 / 4; random.Random(20).sample within each stratum of the eligible
    pool, strata in that order.
  * Sealed holdout: 20% per stratum, random.Random(20) (a second instance), meta.holdout.
  * Field mapping:
      span       -> text,  gold = [official answer string]
      multi-span -> list,  gold = official answer list; items "number" if every answer is a plain number else "text"
      arithmetic -> number, decimals 2, gold = [value in the report's units] + [fully expanded value] (thousand/
                    million/billion) or + [value/100] (percent). The scale is NOT sent to the arm (no `unit`):
                    the instruction asks for the report's units and the scorer also accepts the expanded number.
      count      -> number, integer, gold = [int]
    Computed fields (arithmetic, count) carry na_arms=["sage2"]: Sage 2 returns spans of the document and has no
    compute capability in the 20 Oct spec, so those cells are reported n/a (spec field_plan), never averaged in.
    Groups (meta.group extractive/computed, meta.answer_type) are reported separately and never averaged.
  * Text = table rendered as a markdown table (row 0 = header, newlines in cells -> space) + paragraphs in TAT-QA
    order, the same table->markdown + paragraphs composition TAT-LLM used (arXiv 2401.13223 sec. 3).
  * metric = numeric_tol (OURS). The comparable number is the official scale-aware tatqa_eval, run through
    comparable/run_official.py on the same items (see CARD.md for its two scale bounds).
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, re, statistics, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
KEY = "tatqa"
SEED = 20
HOLDOUT_FRAC = 0.20
FULL_TEST_COUNTS = {"span": 714, "arithmetic": 699, "multi-span": 210, "count": 40}   # verified in build_report
STRATA = ("span", "multi-span", "arithmetic", "count")
SHA256 = {
    "tatqa_dataset_test_gold.json": "c4d08418359c1d76468dec420ee748a37f48c06b63cb8ec2766f19d5d314b597",
    "tatqa_dataset_dev.json": "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5",
    "tatqa_dataset_train.json": "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69",
}
USE_CASE = "Financial-report tables + text; computed values with scale (Clairic-only capability)"
SCALE_FACTOR = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
PLAIN_NUM = re.compile(r"^\(?-?[$€£¥]?\s?-?\d[\d,]*(\.\d+)?\s?%?\)?$")
MIN_PARA = 20


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "")).strip()


# ---------------------------------------------------------------- rendering
def cell(c) -> str:
    return str(c).replace("\r", " ").replace("\n", " ").strip()


def render(doc: dict) -> tuple[str, dict]:
    rows = [[cell(c) for c in r] for r in doc["table"]["table"]]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * ncol) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    paras = [p["text"].strip() for p in sorted(doc["paragraphs"], key=lambda p: int(p["order"]))]
    text = "Table:\n" + "\n".join(md) + "\n\nText:\n" + "\n\n".join(paras)
    return text, {"table_shape": [len(rows), ncol], "n_paragraphs": len(paras)}


def in_text(ans: str, text: str) -> str:
    a = str(ans)
    if a in text:
        return "exact"
    if norm(a).casefold() in norm(text).casefold():
        return "normalized"
    return "absent"


# ---------------------------------------------------------------- fields
def num(x) -> float | int:
    x = float(x)
    return int(x) if x.is_integer() else x


def field_for(q: dict) -> dict:
    t, ans, scale, question = q["answer_type"], q["answer"], q["scale"], q["question"].strip()
    if t == "span":
        assert isinstance(ans, list) and len(ans) == 1, q["uid"]
        return {"name": "answer", "type": "text", "instructions": question, "gold": [str(ans[0])]}
    if t == "multi-span":
        assert isinstance(ans, list) and len(ans) >= 2, q["uid"]
        items = "number" if all(PLAIN_NUM.match(str(a).strip()) for a in ans) else "text"
        return {"name": "answer", "type": "list", "items": items,
                "instructions": f"{question} Return each answer as a separate list item, as written in the document.",
                "gold": [str(a) for a in ans]}
    if t == "arithmetic":
        v = num(ans)
        gold = [v]
        if scale in SCALE_FACTOR:
            gold.append(num(round(v * SCALE_FACTOR[scale], 6)))
        elif scale == "percent":
            gold.append(num(round(v / 100, 8)))
        return {"name": "answer", "type": "number", "decimals": 2, "computed": True, "na_arms": ["sage2"],
                "instructions": (f"{question} Compute the value from the table and text. Give a percentage as the "
                                 "percent number (12.5 for 12.5%) and an amount in the units the document uses for it "
                                 "(1234 when the table shows 1,234 in thousands). Round to 2 decimal places."),
                "gold": gold}
    if t == "count":
        return {"name": "answer", "type": "number", "integer": True, "computed": True, "na_arms": ["sage2"],
                "instructions": f"{question} Return the count as an integer.", "gold": [int(str(ans).strip())]}
    raise ValueError(t)


# ---------------------------------------------------------------- tasksource overlap
def load_tasksource(path: Path):
    d = json.loads(path.read_text())
    states, qids, by_split = [], set(), {}
    for split, rows in d["splits"].items():
        by_split[split] = len(rows)
        for r in rows:
            states.append((split, norm(r["state"])))
            try:
                m = json.loads(r.get("metadata") or "{}")
            except json.JSONDecodeError:
                m = {}
            qids.update(m.get("question_ids") or [])
    return d["provenance"], states, qids, by_split


def overlap(doc: dict, states, qids) -> dict:
    paras = [norm(p["text"]) for p in sorted(doc["paragraphs"], key=lambda p: int(p["order"]))]
    full = " ".join(paras)
    full_hit = sorted({sp for sp, s in states if full and full in s})
    long_paras = [p for p in paras if len(p) >= MIN_PARA]
    covered = [p for p in long_paras if any(p in s for _, s in states)]
    q_hit = sorted(q["uid"] for q in doc["questions"] if q["uid"] in qids)
    all_cov = bool(long_paras) and len(covered) == len(long_paras)
    return {"full_context_in": full_hit, "paras_long": len(long_paras), "paras_covered": len(covered),
            "all_paras_covered": all_cov, "qid_hits": q_hit,
            "excluded": bool(full_hit) or all_cov or bool(q_hit)}


def tatqa_split_paras(paths: list[Path]) -> set[str]:
    s = set()
    for p in paths:
        for doc in json.loads(p.read_text()):
            s.update(norm(x["text"]) for x in doc["paragraphs"])
    return s


# ---------------------------------------------------------------- allocation
def largest_remainder(counts: dict, n: int) -> dict:
    tot = sum(counts.values())
    raw = {k: n * v / tot for k, v in counts.items()}
    alloc = {k: math.floor(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--allow-missing-tasksource", action="store_true",
                    help="build without the tasksource exclusion (recorded as NOT APPLIED in build_report)")
    a = ap.parse_args(argv)

    for f, h in SHA256.items():
        got = sha256(RAW / f)
        if got != h:
            sys.exit(f"{f}: sha256 {got} != pinned {h}")
    gold = json.loads((RAW / "tatqa_dataset_test_gold.json").read_text())
    qs_all = [(doc, q) for doc in gold for q in doc["questions"]]
    counts = Counter(q["answer_type"] for _, q in qs_all)
    assert dict(counts) == FULL_TEST_COUNTS, counts

    ts_path = RAW / "tasksource" / "tatqa_text_span_rows.json"
    if ts_path.exists():
        ts_prov, states, qids, ts_by_split = load_tasksource(ts_path)
        ts_applied = True
    elif a.allow_missing_tasksource:
        ts_prov, states, qids, ts_by_split, ts_applied = None, [], set(), {}, False
    else:
        sys.exit(f"missing {ts_path}; run scripts/fetch_tasksource_tatqa.py (or --allow-missing-tasksource)")

    trdev = tatqa_split_paras([RAW / "tatqa_dataset_train.json", RAW / "tatqa_dataset_dev.json"])

    ov_by_doc = {doc["table"]["uid"]: overlap(doc, states, qids) for doc in gold}
    excl_docs = {k for k, v in ov_by_doc.items() if v["excluded"]}
    pool = defaultdict(list)
    for doc, q in qs_all:
        if doc["table"]["uid"] not in excl_docs:
            pool[q["answer_type"]].append(q["uid"])

    alloc = largest_remainder(FULL_TEST_COUNTS, a.n)
    rng = random.Random(SEED)
    chosen = {}
    for t in STRATA:
        ids = sorted(pool[t])
        if len(ids) < alloc[t]:
            sys.exit(f"stratum {t}: pool {len(ids)} < {alloc[t]}")
        chosen[t] = rng.sample(ids, alloc[t])
    rng_h = random.Random(SEED)
    holdout = set()
    for t in STRATA:
        holdout |= set(rng_h.sample(sorted(chosen[t]), round(HOLDOUT_FRAC * len(chosen[t]))))

    by_uid = {q["uid"]: (doc, q) for doc, q in qs_all}
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        ntok = lambda s: len(enc.encode(s, disallowed_special=()))
    except Exception:  # noqa: BLE001
        enc, ntok = None, None

    items, sample_docs = [], defaultdict(list)
    verb = Counter()
    for t in STRATA:
        for uid in chosen[t]:
            doc, q = by_uid[uid]
            text, shape = render(doc)
            f = field_for(q)
            golds_txt = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
            vt = [in_text(g, text) for g in golds_txt] if t in ("span", "multi-span") else None
            if vt:
                verb[(t, "all_exact" if all(v == "exact" for v in vt) else
                      "all_found" if all(v != "absent" for v in vt) else "some_absent")] += 1
            ov = ov_by_doc[doc["table"]["uid"]]
            paras_n = [norm(p["text"]) for p in doc["paragraphs"]]
            meta = {
                "split": "test", "tier": "C", "tatqa_question_uid": uid, "doc_id": doc["table"]["uid"],
                "table_uid": doc["table"]["uid"], "paragraph_uids": [p["uid"] for p in doc["paragraphs"]],
                "answer_type": t, "answer_from": q["answer_from"], "scale": q["scale"],
                "group": "computed" if t in ("arithmetic", "count") else "extractive",
                "sage2": "n/a (computed)" if t in ("arithmetic", "count") else
                         ("unscored (no list type)" if t == "multi-span" else "scored"),
                "official_answer": q["answer"], "derivation": q["derivation"], "rel_paragraphs": q["rel_paragraphs"],
                "req_comparison": q["req_comparison"], "question": q["question"],
                "gold_in_text": vt, **shape, "pages": None,
                "tasksource": {"paras_long": ov["paras_long"], "paras_covered": ov["paras_covered"],
                               "partial_overlap": ov["paras_covered"] > 0, "applied": ts_applied},
                "tatqa_train_dev_para_overlap": sum(p in trdev for p in paras_n if len(p) >= MIN_PARA),
                "holdout": uid in holdout,
                "o200k_tokens": ntok(text) if ntok else None,
                "o200k_tokens_instructions": ntok(f["instructions"]) if ntok else None,
            }
            items.append({"id": uid, "dataset": KEY, "use_case": USE_CASE, "modality": "text", "text": text,
                          "text_source": "native", "images": [], "fields": [f], "metric": "numeric_tol",
                          "meta": meta})
            sample_docs[doc["table"]["uid"]].append(uid)

    with open(HERE / "items.jsonl", "w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    # gold subset in the official format (only sampled questions) for comparable/run_official.py
    comp = HERE / "comparable"
    comp.mkdir(exist_ok=True)
    sub = []
    for doc in gold:
        keep = set(sample_docs.get(doc["table"]["uid"], []))
        if keep:
            sub.append({**doc, "questions": [q for q in doc["questions"] if q["uid"] in keep]})
    (comp / "tatqa_gold_sample.json").write_text(json.dumps(sub, ensure_ascii=False, indent=1))

    toks = sorted(it["meta"]["o200k_tokens"] for it in items) if ntok else []
    pct = lambda p: toks[min(len(toks) - 1, max(0, math.ceil(p * len(toks)) - 1))] if toks else None
    n_excl_q = sum(len(doc["questions"]) for doc in gold if doc["table"]["uid"] in excl_docs)
    report = {
        "n_items": len(items), "alloc": alloc, "full_test_counts": dict(counts),
        "pool_after_exclusion": {t: len(pool[t]) for t in STRATA},
        "tasksource": {"applied": ts_applied, "provenance": {k: v for k, v in (ts_prov or {}).items() if k != "files"},
                       "rows_by_split": ts_by_split, "n_states": len(states), "n_question_ids": len(qids),
                       "test_contexts": len(gold), "contexts_excluded": len(excl_docs),
                       "questions_excluded": n_excl_q,
                       "why": Counter(("full" if v["full_context_in"] else "") + ("+allparas" if v["all_paras_covered"]
                                      else "") + ("+qid" if v["qid_hits"] else "") for v in ov_by_doc.values()
                                      if v["excluded"]),
                       "contexts_partial_overlap": sum(1 for v in ov_by_doc.values()
                                                       if v["paras_covered"] and not v["excluded"]),
                       "qid_hits_total": sum(len(v["qid_hits"]) for v in ov_by_doc.values())},
        "sample_partial_tasksource_overlap": sum(it["meta"]["tasksource"]["partial_overlap"] for it in items),
        "sample_tatqa_train_dev_para_overlap_items": sum(it["meta"]["tatqa_train_dev_para_overlap"] > 0 for it in items),
        "holdout_n": len(holdout), "holdout_by_type": dict(Counter(it["meta"]["answer_type"] for it in items
                                                                  if it["meta"]["holdout"])),
        "distinct_contexts": len(sample_docs),
        "by_type_scale": {f"{k[0]}|{k[1] or 'none'}": v for k, v in
                          sorted(Counter((it["meta"]["answer_type"], it["meta"]["scale"]) for it in items).items())},
        "by_answer_from": dict(Counter(it["meta"]["answer_from"] for it in items)),
        "span_gold_verbatim": {f"{k[0]}|{k[1]}": v for k, v in sorted(verb.items())},
        "multi_span_items_type": dict(Counter(it["fields"][0].get("items") for it in items
                                              if it["fields"][0]["type"] == "list")),
        "o200k_text_tokens": {"exact": bool(ntok), "p50": pct(0.5), "p95": pct(0.95), "max": toks[-1] if toks else None,
                              "mean": round(statistics.mean(toks), 1) if toks else None},
        "sha256": {**SHA256, "items.jsonl": sha256(HERE / "items.jsonl"),
                   "comparable/tatqa_gold_sample.json": sha256(comp / "tatqa_gold_sample.json")},
        "seed": SEED,
    }
    (HERE / "build_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
