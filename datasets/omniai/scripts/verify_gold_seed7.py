import json, random, re, sys, csv, datetime, hashlib
D = sys.argv[1]
items = [json.loads(l) for l in open(f'{D}/items.jsonl')]
raw = {}
for l in open(f'{D}/raw/metadata.jsonl'):
    r = json.loads(l); raw[int(r['id'])] = r
rng = random.Random(7)
sample = rng.sample(items, 15)
print('sample ids', [it['id'] for it in sample])
MISSING = object()
def resolve(gold, path):
    """path like 'a.b', 'checks[0].x', 'items[].price', 'a[]', 'a[].b.c[]'"""
    # split on '[]' segments
    parts = re.split(r'\.', path)
    cur = [gold]; is_list = False
    for p in parts:
        nxt = []
        m = re.fullmatch(r'(\w+)\[0\]', p)
        if m:
            for c in cur:
                v = c.get(m[1], MISSING) if isinstance(c, dict) else MISSING
                nxt.append(v[0] if isinstance(v, list) and v else MISSING)
            cur = nxt; continue
        arr = p.endswith('[]'); key = p[:-2] if arr else p
        for c in cur:
            v = c.get(key, MISSING) if isinstance(c, dict) else (None if c is None else MISSING)
            if arr:
                is_list = True
                if isinstance(v, list): nxt.extend(v)
                else: nxt.append(v)  # None / MISSING
            else:
                nxt.append(v)
        cur = nxt
    return cur, is_list

def iso_ok(verbatim, iso_list):
    return True
nfields = 0; mism = []
for it in sample:
    r = raw[it['meta']['orig_id']]
    gold = json.loads(r['true_json_output'])
    md = json.loads(r['metadata'])
    assert md['format'] == it['meta']['format'], it['id']
    for f in it['fields']:
        nfields += 1
        fm = it['meta']['field_meta'][f['name']]
        path = it['meta']['name_map'][f['name']]
        vals, is_list = resolve(gold, path)
        g = f['gold']
        if f['type'] == 'list':
            if vals and all(v is None or v is MISSING for v in vals) and (len(vals) == 1):
                exp = None
            else:
                exp = [v for v in vals if v is not None and v is not MISSING]
            if fm.get('item_type') == 'date':
                cmp_g = fm.get('gold_verbatim')
            else:
                cmp_g = g
            if cmp_g != exp:
                mism.append((it['id'], f['name'], path, 'list', g, exp))
            continue
        assert len(vals) == 1, (it['id'], path, vals)
        v = vals[0]
        if v is None or v is MISSING:
            if g is not None: mism.append((it['id'], f['name'], path, 'null', g, v))
            continue
        accs = g if isinstance(g, list) else [g]
        if f['type'] == 'number':
            if not (isinstance(g, (int, float)) and float(g) == float(str(v).replace(',', ''))):
                mism.append((it['id'], f['name'], path, 'num', g, v))
        else:
            if accs[0] != v and str(v) not in accs:
                mism.append((it['id'], f['name'], path, f['type'], g, v))
            if f['type'] == 'date':
                # independent check: ISO alternative must be a real date & consistent with verbatim digits
                isos = [a for a in accs if re.fullmatch(r'\d{4}-\d{2}-\d{2}', a)]
                if not isos: mism.append((it['id'], f['name'], path, 'date_no_iso', g, v))
                for a in isos:
                    y, m_, d_ = map(int, a.split('-'))
                    nums = [int(x) for x in re.findall(r'\d+', v)]
                    if not ((d_ in nums) and (y in nums or y % 100 in nums)):
                        # month-name dates: day and year
                        mism.append((it['id'], f['name'], path, 'date_iso_suspect', g, v))
    # booleans
print('fields checked', nfields)
print('mismatches', len(mism))
for m in mism[:60]: print(m)
# text check vs results.csv
csv.field_size_limit(1 << 30)
def canon(s): return json.dumps(json.loads(s), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
want = {(canon(raw[it['meta']['orig_id']]['true_json_output']), canon(raw[it['meta']['orig_id']]['json_schema'])): it for it in sample}
ok = 0; seen = 0
for row in csv.DictReader(open(f'{D}/raw/results.csv', newline='')):
    if row['ocr_model'] != 'azure-document-intelligence': continue
    k = (canon(row['true_json']), canon(row['json_schema']))
    if k in want:
        seen += 1; it = want[k]
        ok += row['predicted_markdown'] == it['text']
        if row['file_url'] != it['meta']['results_file_url']: print('file_url mismatch', it['id'])
print('azure text rows matched', seen, 'identical', ok)
# image check vs raw
for it in sample:
    r = raw[it['meta']['orig_id']]
    src = f"{D}/raw/images/{r['file_name'].split('/')[-1]}"
    a = open(src, 'rb').read(); b = open(f"{D}/{it['images'][0]}", 'rb').read()
    if a != b: print('image differs (reencoded?)', it['id'], it['meta']['image'].get('reencoded'))
