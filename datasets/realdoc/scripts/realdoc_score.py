"""Official-style RealDoc scoring for our items (adapter for the harness; $0, no network).

Per field: upstream `fuzzy_equal` if the key is in meta.official_str_keys (plain <string> template fields), else
`deep_equal` (scripts/realdoc_official.py, verbatim from extend-hq/realdoc-bench@fb26a687). A null prediction
equals a null gold (deep_equal(None, None) is True upstream), so abstention on null-gold fields counts as correct,
exactly as in the published protocol.

Reported numbers (all per arm):
  field_acc        correct / scored fields                         (scored = the item's `fields`)
  field_acc_cov    correct / ALL gold keys incl. excluded booleans/lists (= coverage-adjusted; excluded = miss)
  question_acc     items whose scored fields are all correct / items        (NOT comparable to the paper's
                   per-question number when booleans were dropped from an item)
  question_acc_strict  items with no excluded keys and all fields correct / all items

Usage as a library: score_item(item, {field_name: value}) -> {name: bool}; summarize(items, preds_by_id).
CLI self-test: uv run --no-project --with 'rapidfuzz>=3.9' python -I datasets/realdoc/scripts/realdoc_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realdoc_official as R  # noqa: E402


def score_item(item: dict, preds: dict) -> dict:
    sk = set(item["meta"]["official_str_keys"])
    out = {}
    for f in item["fields"]:
        k, p = f["realdoc_key"], preds.get(f["name"])
        gold = item["meta"]["official_gold_dict"][k]
        out[f["name"]] = bool(R.fuzzy_equal(p, gold) if k in sk else R.deep_equal(p, gold))
    return out


def summarize(items: list[dict], preds_by_id: dict) -> dict:
    n_f = n_ok = n_all = q_ok = q_strict = 0
    for it in items:
        fm = score_item(it, preds_by_id.get(it["id"], {}))
        n_f += len(fm)
        n_ok += sum(fm.values())
        n_all += it["meta"]["n_gold_fields_total"]
        allc = bool(fm) and all(fm.values())
        q_ok += allc
        q_strict += allc and not it["meta"]["excluded_fields"]
    n = len(items)
    return {"n_items": n, "n_fields": n_f, "n_gold_keys_all": n_all,
            "field_acc": n_ok / n_f if n_f else None, "field_acc_cov": n_ok / n_all if n_all else None,
            "question_acc": q_ok / n if n else None, "question_acc_strict": q_strict / n if n else None}


if __name__ == "__main__":
    items = [json.loads(l) for l in (HERE.parent / "items.jsonl").read_text().splitlines() if l.strip()]
    gold = {it["id"]: {f["name"]: f["gold"] for f in it["fields"]} for it in items}
    empty = {it["id"]: {} for it in items}
    print("gold as prediction:", json.dumps(summarize(items, gold)))
    print("all-null prediction:", json.dumps(summarize(items, empty)))
