"""Score VRDU items with the OFFICIAL VRDU evaluator (google-research vrdu/benchmark_utils.py + match_utils.py,
vendored verbatim in tools/vrdu_official/vrdu/, commit 600df90d). Gold = the untouched official annotations read from
raw/<corpus>/main/dataset.jsonl.gz; only the docs present in items.jsonl are scored.

  U="/opt/homebrew/bin/uv run --no-project --with editdistance python -I"
  $U extract-bench/datasets/vrdu/tools/official_eval.py --gold-roundtrip
  $U extract-bench/datasets/vrdu/tools/official_eval.py --run RUN --arm ARM [--rule a|b] [--line-order official|insensitive]

Prediction -> official extraction (adapter; declared in CARD.md):
  * scalar: null value (rule a), or null / conf < 0.5 (rule b) -> no extraction (official: not counted as a prediction).
    Otherwise the arm's verbatim text (pred.text, else state[span] when the state was the official OCR), else str(value).
    ISO 'YYYY-MM-DD' -> 'YYYY/MM/DD' because official DateMatch has no ISO pattern (format only, same date).
  * ad-buy line items: rows rebuilt by index from the 5 parallel list fields; each row is emitted as flat sub-entities
    and the official code regroups them into nested line items. --emit reading (default) orders a row's sub-entities by
    reading position (the arm's own list spans if returned, else the value's next occurrence in the official OCR; gold is
    never used); --emit schema uses a fixed schema order.
  * --line-order official   : official matching, which keys nested entities by the ORDERED tuple of sub-entity names.
                              The gold uses 68 different orders, so a row only matches if its order equals the gold's.
  * --line-order insensitive: gold and (after official grouping) extraction tuples both put in schema order (our fix,
                              NOT official).
  --gold-roundtrip emits gold three ways: gold-order rows, schema-order rows, and the items.jsonl gold pushed through
  the same parallel-list adapter predictions use (= the ceiling for a perfect list-capable arm).
"""
from __future__ import annotations

import argparse, gzip, json, re, sys
from pathlib import Path

DS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "vrdu_official"))
from vrdu import benchmark_utils as bu  # noqa: E402  (vendored official code)

RAW = DS / "raw"
CORP = {"registration": "registration-form", "adbuy": "ad-buy-form"}
LINE_ORDER = ["channel", "program_desc", "program_start_date", "program_end_date", "sub_amount"]
LINE_FIELD = {"line_channel": "channel", "line_program_desc": "program_desc",
              "line_program_start_date": "program_start_date", "line_program_end_date": "program_end_date",
              "line_sub_amount": "sub_amount"}
BOX = (0, 0.0, 0.0, 0.0, 0.0)


def corpus_of(item):
    return "registration" if item["id"].startswith("vrdu-reg-") else "adbuy"


def load_items():
    return [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]


def canon_nested(entity_item):
    names, ents = entity_item
    if not bu.is_nested_entity(names):
        return entity_item
    pairs = sorted(zip(names, ents[0]), key=lambda p: LINE_ORDER.index(p[0]))
    return (tuple(p[0] for p in pairs), [tuple(p[1] for p in pairs)])


def benchmark(corpus, filenames, insensitive=False):
    b = object.__new__(bu.DataUtils)
    b.entity_name_to_match_func, b.appearance_pattern_to_entity_names = bu.DataUtils.load_entity_name_info(
        b, str(RAW / CORP[corpus] / "main" / "meta.json"))
    want, ann = set(filenames), {}
    with gzip.open(RAW / CORP[corpus] / "main" / "dataset.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["filename"] in want:
                items = [bu.convert_list_to_tuple_for_entity_items(a) for a in r["annotations"]]
                ann[r["filename"]] = [canon_nested(a) for a in items] if insensitive else items
    assert set(ann) == want, want - set(ann)
    b.annotations, b.test_filenames = ann, list(filenames)
    b.train_filenames = b.valid_filenames = []
    return b


def to_text(v):
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    return s.replace("-", "/") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else s


def pred_text(p, state):
    if p.get("text"):
        return to_text(p["text"])
    sp = p.get("span")
    if state and isinstance(sp, (list, tuple)) and len(sp) == 2 and all(isinstance(x, int) for x in sp) and sp[1] > sp[0]:
        return state[sp[0]:sp[1]]
    return to_text(p["value"])


def abstained(p, rule):
    if p is None or p.get("value") in (None, "", []):
        return True
    return rule == "b" and p.get("conf") is not None and p["conf"] < 0.5


def _occ(v, doc_text, cap=500):
    out, k = [], doc_text.find(v) if v else -1
    while k >= 0 and len(out) < cap:
        out.append(k)
        k = doc_text.find(v, k + 1)
    return out


def _row_positions(rows, doc_text):
    """No arm spans: place each predicted row in the official OCR without gold. Anchor = the row's value with the
    fewest occurrences at/after the previous row's anchor; the other values take their occurrence nearest the anchor."""
    out, cursor = [], 0
    for row in rows:
        occ = {s: _occ(v, doc_text) or _occ(" ".join(v.split()[:2]), doc_text) for s, v in row.items()}
        cand = {s: ([k for k in o if k >= cursor] or o) for s, o in occ.items() if o}
        if not cand:
            out.append({}); continue
        anchor = min(cand, key=lambda s: (len(cand[s]), LINE_ORDER.index(s)))
        apos = cand[anchor][0]
        pos = {s: min(o, key=lambda k: abs(k - apos)) for s, o in cand.items()}
        cursor = apos + 1
        out.append(pos)
    return out


def extraction_from_preds(item, preds, rule, state, emit="reading"):
    out = []
    lists, pos = {}, {}
    for f in item["fields"]:
        p = preds.get(f["name"])
        if f["type"] == "list":
            if not abstained(p, rule):
                keep = [(i, x) for i, x in enumerate(p["value"]) if x not in (None, "")]
                vals = [to_text(x) for _, x in keep]
                sp = p.get("span") if state else None
                if isinstance(sp, list) and len(sp) == len(p["value"]):
                    sp = [sp[i] for i, _ in keep]
                else:
                    sp = None
                sub = LINE_FIELD[f["name"]]
                lists[sub] = vals
                ok = (isinstance(sp, list) and all(isinstance(x, (list, tuple)) and len(x) == 2
                                                   and isinstance(x[0], int) for x in sp))
                pos[sub] = [x[0] for x in sp] if ok else None  # the arm's own span starts
            continue
        if abstained(p, rule):
            continue
        out.append((f["name"], (pred_text(p, state), BOX, [])))
    n = max((len(v) for v in lists.values()), default=0)
    rows = [{sub: lists[sub][i] for sub in LINE_ORDER if sub in lists and i < len(lists[sub])} for i in range(n)]
    guess = _row_positions(rows, item["text"]) if emit == "reading" else None
    big = 10 ** 9
    for i, row in enumerate(rows):
        order = list(row)
        if emit == "reading":  # official tuples are in reading order (verified 4,238/4,238 gold line items)
            def key(sub):
                if pos.get(sub) is not None:
                    return (pos[sub][i], 0)
                return (guess[i].get(sub, big), LINE_ORDER.index(sub))
            order.sort(key=key)
        out.extend((sub, (row[sub], BOX, [])) for sub in order)
    return out


def extraction_from_gold(item, order):
    """Gold-as-prediction from items.jsonl gold (first acceptable value) + meta.official_line_items."""
    out = []
    for f in item["fields"]:
        if f["type"] == "list" or f["gold"] is None:
            continue
        g = f["gold"] if isinstance(f["gold"], str) else f["gold"][0]
        out.append((f["name"], (g, BOX, [])))
    for row in item["meta"].get("official_line_items", []):
        names = row["order"] if order == "gold" else [n for n in LINE_ORDER if n in row["values"]]
        out.extend((n, (row["values"][n], BOX, [])) for n in names)
    return out


def canon_nested_ex(entity_item):
    names, ent = entity_item
    if not bu.is_nested_entity(names):
        return entity_item
    pairs = sorted(zip(names, ent), key=lambda p: LINE_ORDER.index(p[0]))
    return (tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))


def gold_preds(item):
    """items.jsonl gold as if an arm had returned it (scalar: first acceptable value; lists: as stored)."""
    out = {}
    for f in item["fields"]:
        g = f["gold"]
        if g is None:
            continue
        if f["type"] == "list":
            v = g[0] if (g and isinstance(g[0], list)) else g  # verbatim official list (repaired list is 2nd)
        else:
            v = g if isinstance(g, str) else g[0]
        out[f["name"]] = {"value": v, "conf": 1.0}
    return out


def score(corpus, items, extractions, insensitive):
    b = benchmark(corpus, [it["meta"]["filename"] for it in items], insensitive)
    me = {"meta": {}, "results": extractions}
    if insensitive:  # group with the official code first, then put both sides' sub-entity tuples in schema order
        me = bu.process_model_extractions(me, b.appearance_pattern_to_entity_names)
        me["results"] = {k: [canon_nested_ex(e) for e in v] for k, v in me["results"].items()}
    res = bu.evaluate(me, b)  # idempotent on pre-grouped input (nested names are never re-grouped)
    keep = {k: round(v, 4) for k, v in res.items() if k.endswith("f1")}
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--arm")
    ap.add_argument("--rule", default="a", choices=["a", "b"])
    ap.add_argument("--line-order", default="official", choices=["official", "insensitive"])
    ap.add_argument("--emit", default="reading", choices=["reading", "schema"],
                    help="order of each predicted line item's sub-entities (official tuples are in reading order)")
    ap.add_argument("--gold-roundtrip", action="store_true")
    ap.add_argument("--corpus", choices=["registration", "adbuy"])
    a = ap.parse_args()
    items = load_items()
    report = {}
    for corpus in ("registration", "adbuy"):
        if a.corpus and corpus != a.corpus:
            continue
        its = [it for it in items if corpus_of(it) == corpus]
        if a.gold_roundtrip:
            for order in ("gold", "schema", "parallel-lists-schema", "parallel-lists-reading"):
                for lo in ("official", "insensitive"):
                    if order.startswith("parallel-lists"):  # items.jsonl gold fed through the prediction adapter
                        em = order.rsplit("-", 1)[1]
                        ex = {it["meta"]["filename"]: extraction_from_preds(it, gold_preds(it), "a", None, em)
                              for it in its}
                    else:
                        ex = {it["meta"]["filename"]: extraction_from_gold(it, order) for it in its}
                    r = score(corpus, its, ex, lo == "insensitive")
                    report[f"{corpus}|emit={order}|line-order={lo}"] = {
                        k: r[k] for k in ("micro_f1", "macro_f1", "unrepeated_f1", "line_item_f1") if k in r}
                if corpus == "registration":
                    break
            continue
        if not (a.run and a.arm):
            ap.error("--run and --arm are required unless --gold-roundtrip")
        pdir = DS.parent.parent / "runs" / a.run / "preds" / a.arm / "vrdu"
        ex, missing = {}, 0
        for it in its:
            f = pdir / f"{it['id']}.json"
            if not f.exists():
                missing += 1
                continue
            pr = json.loads(f.read_text())
            state = it["text"] if pr.get("text_source") == "official_ocr" else None
            ex[it["meta"]["filename"]] = extraction_from_preds(it, pr.get("preds") or {}, a.rule, state, a.emit)
        scored = [it for it in its if it["meta"]["filename"] in ex]
        if scored:
            report[corpus] = {"n_docs": len(scored), "missing_preds": missing,
                              **score(corpus, scored, ex, a.line_order == "insensitive")}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
