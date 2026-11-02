"""Sanity check of the scalar gold mapping, at $0: score the vendor-run Feb-2025 predictions that ship in
results.csv (OCR -> azure-gpt-4o; ground-truth md -> gpt-4o) on OUR scalar fields with a simple normaliser.

This is NOT the suite's official exact_norm (the harness owns that). It exists to catch mapping bugs:
if gpt-4o on gold markdown scores far below ~0.95 on our scalar fields, the mapping is wrong.

Usage: uv run --no-project python -I scripts/sanity_published_scalar.py <dataset_dir> <out.json>
"""
import collections
import json
import os
import re
import sys

D, OUT = sys.argv[1], sys.argv[2]
items = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(D, 'items.jsonl'), encoding='utf-8')}
preds = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(D, 'published_predictions.jsonl'), encoding='utf-8')}
MISSING = object()


def nav(obj, path):
    cur = obj
    for key in path.split('.'):
        m = re.fullmatch(r'(.+)\[(\d+)\]', key)
        if m:
            cur = cur.get(m[1], MISSING) if isinstance(cur, dict) else MISSING
            if isinstance(cur, list) and int(m[2]) < len(cur):
                cur = cur[int(m[2])]
            else:
                return MISSING
        else:
            cur = cur.get(key, MISSING) if isinstance(cur, dict) else MISSING
        if cur is MISSING:
            return MISSING
    return cur


def norm(v):
    if v is None or v is MISSING:
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return ('num', round(float(v), 4))
    s = str(v).strip()
    t = s.replace(',', '').replace('$', '')
    if re.fullmatch(r'-?\d+(\.\d+)?', t):
        return ('num', round(float(t), 4))
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s.lower().replace('_', ' '))).strip()


res = collections.defaultdict(lambda: collections.Counter())
by_type = collections.defaultdict(lambda: collections.Counter())
for iid, it in items.items():
    pp = preds[iid]['pipelines']
    for pipe, p in pp.items():
        try:
            pj = json.loads(p['predicted_json']) if isinstance(p['predicted_json'], str) else p['predicted_json']
        except Exception:  # noqa: BLE001
            pj = None
        for f in it['fields']:
            if f['type'] == 'list':
                continue
            fm = it['meta']['field_meta'][f['name']]
            pv = nav(pj, fm['orig_path']) if isinstance(pj, dict) else MISSING
            gold = f['gold']
            if gold is None:
                ok = pv is None or pv is MISSING
            else:
                accs = gold if isinstance(gold, list) else [gold]
                ok = norm(pv) is not None and any(norm(pv) == norm(a) for a in accs)
            res[pipe]['n'] += 1
            res[pipe]['ok'] += ok
            by_type[(pipe, f['type'])]['n'] += 1
            by_type[(pipe, f['type'])]['ok'] += ok
out = {'note': 'sanity check only; simple normaliser; vendor-run predictions; scalar fields only',
       'scalar_accuracy': {p: {'n': c['n'], 'acc': round(c['ok'] / c['n'], 4)} for p, c in sorted(res.items())},
       'by_type': {f'{k[0]}|{k[1]}': {'n': c['n'], 'acc': round(c['ok'] / c['n'], 4)} for k, c in sorted(by_type.items())
                   if k[0] in ('ground-truth', 'azure-document-intelligence')}}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
