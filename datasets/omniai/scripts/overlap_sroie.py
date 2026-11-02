"""Check whether OmniAI PHOTO_RECEIPT / FORM / SCANNED_FORM docs overlap SROIE test (company+date+total) or
FUNSD test (text snippets). Data-only reads. Usage: python -I overlap_sroie.py <omniai_raw_metadata.jsonl> <sroie test_text.jsonl> <funsd test.parquet>"""
import json, sys, re
meta = [json.loads(l) for l in open(sys.argv[1])]
sroie = [json.loads(l) for l in open(sys.argv[2])]
print('sroie rows', len(sroie), 'keys', list(sroie[0].keys())[:10])
print({k: str(v)[:200] for k, v in sroie[0].items()})
blob = '\n'.join(json.dumps(r, ensure_ascii=False).lower() for r in sroie)
def norm(s): return re.sub(r'\s+', ' ', str(s).lower()).strip()
for m in meta:
    f = json.loads(m['metadata']).get('format', '').strip()
    if f != 'PHOTO_RECEIPT': continue
    g = json.loads(m['true_json_output'])
    name = (g.get('store_info') or {}).get('name'); date = (g.get('cashier_info') or {}).get('date'); tot = (g.get('totals') or {}).get('total')
    hit = name and norm(name) in blob
    print(m['id'], repr(name), date, tot, 'name_in_sroie_test' if hit else '-')
try:
    import pyarrow.parquet as pq
    t = pq.read_table(sys.argv[3]).to_pylist()
    print('funsd rows', len(t), list(t[0].keys()))
    fblob = '\n'.join(norm(' '.join(map(str, r.get('words') or r.get('tokens') or []))) for r in t)
    for m in meta:
        f = json.loads(m['metadata']).get('format', '').strip()
        if f not in ('FORM', 'SCANNED_FORM'): continue
        md = m['true_markdown_output']; md = json.loads(md) if md.startswith('"') else md
        words = [w for w in re.findall(r'[A-Za-z]{5,}', md)][:40]
        probe = ' '.join(words[3:9]).lower()
        print(m['id'], f, repr(probe[:60]), 'IN_FUNSD_TEST' if probe and probe in fblob else '-')
except Exception as e:
    print('funsd check skipped', e)
