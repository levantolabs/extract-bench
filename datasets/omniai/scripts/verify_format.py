import json, re, sys, os, collections
D = sys.argv[1]
items = [json.loads(l) for l in open(f'{D}/items.jsonl')]
NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
TYPES = {"text","number","money","date","time","datetime","duration","list"}
METRICS = {"exact_norm","anls","token_f1","numeric_tol","date_exact","list_f1","edit_sim"}
REQ = ["id","dataset","use_case","modality","text","text_source","images","fields","metric","meta"]
err = collections.Counter(); ex = []
ids = set()
def e(k, x):
    err[k] += 1
    if len(ex) < 40: ex.append((k, x))
for it in items:
    if list(it.keys()) != REQ: e('keys', (it['id'], list(it.keys())))
    if it['id'] in ids: e('dup_id', it['id'])
    ids.add(it['id'])
    if it['dataset'] != 'omniai': e('dataset', it['id'])
    if it['modality'] not in ('text','image','text+image'): e('modality', it['id'])
    if it['text_source'] not in ('native','official_ocr','none'): e('ts', it['id'])
    if it['text_source'] == 'official_ocr' and not it['text']: e('empty_text', it['id'])
    if it['metric'] not in METRICS: e('metric', it['id'])
    if not (1 <= len(it['images']) <= 4): e('nimg', it['id'])
    for p in it['images']:
        fp = os.path.join(D, p)
        if not os.path.isfile(fp): e('img_missing', p); continue
        sz = os.path.getsize(fp)
        if sz > 4*1024*1024: e('img_big', (p, sz))
        b = open(fp,'rb').read(16)
        kind = 'png' if b.startswith(b'\x89PNG') else 'jpg' if b.startswith(b'\xff\xd8') else 'webp' if b[8:12]==b'WEBP' else '?'
        if kind == '?' or not p.endswith('.'+kind): e('img_ext', (p, kind))
    names = set()
    for f in it['fields']:
        if set(f.keys()) != {'name','type','instructions','gold'}: e('fkeys', (it['id'], f.keys()))
        if not NAME.match(f['name']): e('fname', f['name'])
        if f['name'] in names: e('fname_dup', (it['id'], f['name']))
        names.add(f['name'])
        if f['type'] not in TYPES: e('ftype', f['type'])
        if not f['instructions'].strip(): e('instr_empty', (it['id'], f['name']))
        g = f['gold']
        if f['type'] == 'list':
            if g is not None and not isinstance(g, list): e('list_gold', (it['id'], f['name']))
        elif f['type'] == 'number':
            accs = g if isinstance(g, list) else [g]
            if g is not None and not all(isinstance(a,(int,float)) and not isinstance(a,bool) for a in accs): e('num_gold', (it['id'], f['name'], g))
        elif f['type'] in ('text','date','time'):
            accs = g if isinstance(g, list) else [g]
            if g is not None and not all(isinstance(a,str) for a in accs): e('str_gold', (it['id'], f['name'], g))
            if isinstance(g, list) and len(g) < 2: e('list_of_1', (it['id'], f['name']))
            if f['type']=='date' and g is not None and not any(re.fullmatch(r'\d{4}-\d{2}-\d{2}',a) for a in accs): e('date_no_iso',(it['id'],f['name'],g))
            if f['type']=='time' and g is not None and not any(re.fullmatch(r'\d{2}:\d{2}(:\d{2})?',a) for a in accs): e('time_no_iso',(it['id'],f['name'],g))
        if f['name'] not in it['meta']['field_meta']: e('no_fmeta', f['name'])
        if isinstance(g, str) and g.strip()=='' : e('empty_str_gold',(it['id'],f['name']))
    # completeness: every leaf of the gold JSON is a field / boolean / skipped / null
    gold = json.loads(it['meta']['true_json'])
    covered = set(it['meta']['name_map'].values()) | set(it['meta']['booleans_excluded']['gold'].keys()) | set(it['meta']['skipped_paths'])
    def leaves(x, p):
        if isinstance(x, dict):
            for k, v in x.items(): yield from leaves(v, p + [k])
        elif isinstance(x, list):
            for v in x:
                yield from leaves(v, p[:-1] + [p[-1] + '[]'] if p else ['[]'])
        else:
            yield '.'.join(p), x
    for path, v in leaves(gold, []):
        if it['meta']['format']=='BANK_CHECK' and path.startswith('checks[].') and any(k.startswith('checks[0]') for k in covered):
            path = path.replace('checks[].', 'checks[0].', 1)
        # path for array-of-scalars leaf: 'a[]' ; nested: 'a[].b[]'
        if path in covered: continue
        e('leaf_uncovered', (it['id'], path, v))
print('items', len(items), 'errors', dict(err))
for x in ex: print(x)
# text_source honesty: compare with ocr file
for it in items:
    alt = it['meta']['ocr_alternatives']
    if open(os.path.join(D, alt['azure-document-intelligence'])).read() != it['text']: print('azure alt mismatch', it['id'])
c = collections.Counter(f['type'] for it in items for f in it['fields'])
print(c, sum(c.values()))
print('null scalar', sum(1 for it in items for f in it['fields'] if f['type']!='list' and f['gold'] is None),
      'null list', sum(1 for it in items for f in it['fields'] if f['type']=='list' and f['gold'] is None))
print('empty list golds', sum(1 for it in items for f in it['fields'] if f['type']=='list' and f['gold']==[]))
