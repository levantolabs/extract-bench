"""Strict overlap: OmniAI PHOTO_RECEIPT (store name + date + total) vs SROIE test entities (company, date, total).
Usage: python -I overlap_sroie_strict.py <omniai metadata.jsonl> <sroie test_text.jsonl> <out.json>"""
import json, sys, re, ast
meta = [json.loads(l) for l in open(sys.argv[1])]
sroie = [json.loads(l) for l in open(sys.argv[2])]
def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
def numnorm(s):
    try: return round(float(str(s).replace(',', '')), 2)
    except Exception: return None
ents = []
for r in sroie:
    e = ast.literal_eval(r['entities']) if isinstance(r['entities'], str) else r['entities']
    ents.append((r['key'], e))
out = []
for m in meta:
    if json.loads(m['metadata']).get('format', '').strip() != 'PHOTO_RECEIPT': continue
    g = json.loads(m['true_json_output'])
    name = (g.get('store_info') or {}).get('name'); date = (g.get('cashier_info') or {}).get('date'); tot = (g.get('totals') or {}).get('total')
    matches = []
    for k, e in ents:
        sc = (norm(name) and norm(name) in norm(e.get('company', '')), numnorm(tot) is not None and numnorm(tot) == numnorm(e.get('total')),
              re.sub(r'\D', '', str(date)) and re.sub(r'\D', '', str(date))[:4] == re.sub(r'\D', '', str(e.get('date', '')))[:4])
        if sc[0] and (sc[1] or sc[2]): matches.append({'sroie_key': k, 'name': sc[0], 'total': sc[1], 'date_prefix': bool(sc[2]), 'sroie': e})
    out.append({'omniai_id': m['id'], 'name': name, 'date': date, 'total': tot, 'sroie_matches': matches})
    print(m['id'], name, date, tot, '->', [(x['sroie_key'], x['total'], x['date_prefix']) for x in matches])
json.dump(out, open(sys.argv[3], 'w'), indent=1, ensure_ascii=False)
