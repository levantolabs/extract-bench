"""COMPARABLE view: a line-by-line re-implementation of the official SQuAD 2.0 evaluate-v2.0.py
(raw/official/evaluate-v2.0.py.txt, SQuAD-explorer commit eee5fdbf, blob 21f52c17; read, NOT executed),
restricted to the items in items.jsonl (the "dataset" is our 120 questions, so num_no_ans counts our items only).

preds: {item_id: {"value": str|None, "conf": float|None, "p_null": float|None}}
  official pred string = value or "" (null -> "" = no answer)
  na_prob = p_null if given, else 1-conf if conf given, else 1.0 for null / 0.0 for an answer.
Outputs exact/f1 at na_prob_thresh (default 1.0, i.e. the arm's own nulls only), HasAns_/NoAns_ splits, and
best_exact/best_f1 (+ thresholds) = the ORACLE-threshold upper bound (L17: never the headline).
Usage: python -I squad_v2_eval.py <items.jsonl> <preds.json> [na_prob_thresh]
"""
import collections, json, re, string, sys


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(re.compile(r'\b(a|an|the)\b', re.UNICODE), ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s):
    return normalize_answer(s).split() if s else []


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks, pred_toks = get_tokens(a_gold), get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    p, r = num_same / len(pred_toks), num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def evaluate(items, preds, na_prob_thresh=1.0):
    has_ans, exact_raw, f1_raw, pstr, na = {}, {}, {}, {}, {}
    for it in items:
        qid = it["id"]
        gold = it["fields"][0]["gold"]
        has_ans[qid] = gold is not None
        # official: every reference answer whose normalisation is non-empty; unanswerable -> [""]
        golds = [x["text"] for x in it["meta"]["answers"] if normalize_answer(x["text"])]
        if not golds:
            golds = [""]
        if qid not in preds:
            continue
        p = preds[qid] or {}
        v = p.get("value")
        pstr[qid] = "" if v is None else str(v)
        if p.get("p_null") is not None:
            na[qid] = float(p["p_null"])
        elif p.get("conf") is not None:
            na[qid] = 1.0 - float(p["conf"])
        else:
            na[qid] = 1.0 if v is None else 0.0
        exact_raw[qid] = max(compute_exact(a, pstr[qid]) for a in golds)
        f1_raw[qid] = max(compute_f1(a, pstr[qid]) for a in golds)

    def thresh(scores):
        return {q: (float(not has_ans[q]) if na[q] > na_prob_thresh else s) for q, s in scores.items()}

    def ev(e, f, ql=None):
        ql = ql if ql is not None else list(e)
        n = len(ql)
        return {"exact": 100.0 * sum(e[k] for k in ql) / n, "f1": 100.0 * sum(f[k] for k in ql) / n, "total": n} if n else {}

    e_t, f_t = thresh(exact_raw), thresh(f1_raw)
    out = ev(e_t, f_t)
    for name, ql in (("HasAns", [q for q in e_t if has_ans[q]]), ("NoAns", [q for q in e_t if not has_ans[q]])):
        for k, v in ev(e_t, f_t, ql).items():
            out[f"{name}_{k}"] = v

    def best(scores):
        cur = best_s = sum(1 for q in has_ans if not has_ans[q])
        best_t = 0.0
        for q in sorted(na, key=lambda k: na[k]):
            if q not in scores:
                continue
            diff = scores[q] if has_ans[q] else (-1 if pstr[q] else 0)
            cur += diff
            if cur > best_s:
                best_s, best_t = cur, na[q]
        return 100.0 * best_s / len(scores), best_t

    out["best_exact"], out["best_exact_thresh"] = best(exact_raw)
    out["best_f1"], out["best_f1_thresh"] = best(f1_raw)
    out["note"] = "best_* = oracle threshold chosen on these same items (upper bound, not a headline)"
    return out


if __name__ == "__main__":
    items = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    preds = json.load(open(sys.argv[2]))
    print(json.dumps(evaluate(items, preds, float(sys.argv[3]) if len(sys.argv) > 3 else 1.0), indent=1))
