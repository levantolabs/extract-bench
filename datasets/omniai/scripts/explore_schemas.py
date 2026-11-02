"""Print example schema + gold per format, plus schema 'format'/'enum'/'emum' values and array shapes. Run with -I."""
import json, sys, collections
raw, = sys.argv[1:2]
meta = [json.loads(l) for l in open(f'{raw}/metadata.jsonl')]
show = sys.argv[2].split(',') if len(sys.argv) > 2 else []
seen = set()
fmtvals = collections.Counter(); arrlen = collections.defaultdict(collections.Counter); arrkinds = collections.Counter()
def walk(s, g, p, f):
    t = s.get('type')
    if 'format' in s: fmtvals[(t, s['format'])] += 1
    if t == 'object':
        for k, v in (s.get('properties') or {}).items(): walk(v, g.get(k) if isinstance(g, dict) else None, p + (k,), f)
    elif t == 'array':
        it = s.get('items') or {}
        arrkinds[(f, it.get('type'))] += 1
        arrlen[f][len(g) if isinstance(g, list) else 'none'] += 1
        if it.get('type') == 'object':
            for k, v in (it.get('properties') or {}).items():
                if v.get('type') in ('array', 'object'): arrkinds[(f, 'nested:' + v.get('type'))] += 1
for m in meta:
    f = json.loads(m['metadata']).get('format', '').strip()
    s = json.loads(m['json_schema']); g = json.loads(m['true_json_output'])
    walk(s, g, (), f)
    if f in show and f not in seen:
        seen.add(f)
        print('=====', f, m['id'], m['metadata'])
        print(json.dumps(s)[:2500]); print('GOLD', json.dumps(g)[:1500])
print('format values', fmtvals.most_common())
for f in ['BANK_CHECK','SHIPPING_INVOICE','DELIVERY_NOTE','PHOTO_RECEIPT','ACCOUNT_STATEMENT','CREDIT_CARD_STATEMENT','FORM_1040','PATIENT_INTAKE','PETITION_FORM','FORM','SCANNED_FORM']:
    print(f, 'array lens', dict(arrlen[f]), {k[1]: v for k, v in arrkinds.items() if k[0] == f})
