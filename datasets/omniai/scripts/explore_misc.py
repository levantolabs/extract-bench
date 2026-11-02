"""Misc checks: docs lacking Azure DI rows, empty OCR text, emum leaves, nested arrays in arrays. Run with -I."""
import csv, json, sys, collections
csv.field_size_limit(1 << 30)
raw = sys.argv[1]
def canon(s): return json.dumps(json.loads(s), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
meta = [json.loads(l) for l in open(f'{raw}/metadata.jsonl')]
key2id = {(canon(m['true_json_output']), canon(m['json_schema'])): m['id'] for m in meta}
fmt = {m['id']: json.loads(m['metadata']).get('format', '').strip() for m in meta}
prov = collections.defaultdict(dict)
for r in csv.DictReader(open(f'{raw}/results.csv', newline='')):
    i = key2id[(canon(r['true_json']), canon(r['json_schema']))]
    prov[i][r['ocr_model']] = r
for p in ['azure-document-intelligence', 'aws-textract', 'gemini-2.0-flash-001', 'ground-truth']:
    missing = [i for i in fmt if p not in prov[i]]
    empty = [i for i in fmt if p in prov[i] and not prov[i][p]['predicted_markdown'].strip()]
    print(p, 'missing', len(missing), [(i, fmt[i]) for i in missing][:15], 'empty_md', len(empty), [(i, fmt[i]) for i in empty][:10])
# extraction model per provider row; json_accuracy blank?
c = collections.Counter((r['ocr_model'], r['extraction_model'], r['json_accuracy'] == '') for d in prov.values() for r in d.values())
print(c)
# ground-truth predicted_markdown == true_markdown ?
print('gt pred==true md', sum(prov[i]['ground-truth']['predicted_markdown'] == prov[i]['ground-truth']['true_markdown'] for i in prov))
def walk(s, p, in_arr, f, out):
    t = s.get('type')
    if 'emum' in s: out['emum'].append((f, p, s.get('type'), s['emum'][:5]))
    if t == 'object':
        for k, v in (s.get('properties') or {}).items(): walk(v, p + '.' + k, in_arr, f, out)
    elif t == 'array':
        if in_arr: out['nested_arr'].append((f, p, (s.get('items') or {}).get('type')))
        walk(s.get('items') or {}, p + '[]', True, f, out)
out = collections.defaultdict(list)
for m in meta: walk(json.loads(m['json_schema']), '', False, fmt[m['id']], out)
print('emum', len(out['emum']), collections.Counter((a, b) for a, b, *_ in out['emum']).most_common(10), out['emum'][:3])
print('nested arrays', collections.Counter((a, b, c) for a, b, c in out['nested_arr']).most_common(20))
# azure usage/latency per row sample
r = prov['17']['azure-document-intelligence']; print('azure md sample id17:', r['predicted_markdown'][:600])
