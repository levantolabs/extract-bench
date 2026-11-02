"""Profile the RealDoc QA bank with the OFFICIAL template builder (vendored verbatim): type hints per field,
gold Python types, enum fields, field-name validity, eligibility (>=1 non-bool, non-list field) per domain.

Usage: uv run --no-project python -I datasets/realdoc/scripts/profile_bank.py > scripts/profile_bank.out.txt
"""
import collections
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realdoc_official as R  # noqa: E402

bank = json.loads((HERE.parent / "raw" / "hf" / "qa_bank.json").read_text())["items"]
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
hint_c, gtype_c, pair_c, dom_all, dom_elig = (collections.Counter() for _ in range(5))
bad_names, widened, gold_changed, enum_examples, n_fields_q = [], [], [], [], collections.Counter()
nulls_by_hint = collections.Counter()
for it in bank:
    g0 = it["gold_dict"]
    g = copy.deepcopy(g0)
    tpl = R.build_template(it["response_format"], g)
    if g != g0:
        gold_changed.append((it["question_id"], g0, g))
    hints = dict(R._TYPE_RE.findall(tpl))
    dom_all[it["domain"]] += 1
    elig = False
    for k, v in g.items():
        h = hints.get(k, "MISSING").replace(" | null", "").strip()
        hb = "enum" if h.startswith("one of:") else h
        hint_c[hb] += 1
        gt = "null" if v is None else type(v).__name__
        gtype_c[gt] += 1
        pair_c[(hb, gt)] += 1
        if v is None:
            nulls_by_hint[hb] += 1
        if not NAME.match(k):
            bad_names.append((it["question_id"], k))
        if hb == "enum" and len(enum_examples) < 15:
            enum_examples.append((it["question_id"], k, h, v))
        if hb not in ("boolean",) and not isinstance(v, (bool, list)):
            elig = True
    n_fields_q[len(g)] += 1
    if elig:
        dom_elig[it["domain"]] += 1

print("questions", len(bank), "fields", sum(hint_c.values()))
print("template hint (base) counts", hint_c.most_common())
print("gold python types", gtype_c.most_common())
print("(hint, gold type) pairs", sorted(pair_c.items(), key=lambda x: -x[1]))
print("null golds by hint", nulls_by_hint.most_common())
print("fields per question", sorted(n_fields_q.items()))
print("domains all", dom_all.most_common())
print("domains eligible (>=1 non-bool non-list field)", dom_elig.most_common(), "total", sum(dom_elig.values()))
print("bad field names", len(bad_names), bad_names[:10])
print("gold changed by widen_types_from_gold", len(gold_changed), gold_changed[:10])
print("enum examples")
for e in enum_examples:
    print("  ", e)
