"""Check content-join between results.csv rows and benchmark metadata ids. Run with -I."""
import csv, json, sys, collections
csv.field_size_limit(1 << 30)
raw = sys.argv[1]
def canon(s):
    return json.dumps(json.loads(s), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
def md_of(s):
    return json.loads(s) if s.startswith('"') else s
meta = [json.loads(l) for l in open(f'{raw}/metadata.jsonl')]
k1 = collections.defaultdict(list); k2 = collections.defaultdict(list)
for m in meta:
    k1[(canon(m['true_json_output']), canon(m['json_schema']))].append(m['id'])
    k2[(canon(m['true_json_output']), canon(m['json_schema']), md_of(m['true_markdown_output']))].append(m['id'])
print('meta', len(meta), 'unique key json+schema', len(k1), 'dupe groups', sum(len(v) > 1 for v in k1.values()))
print('unique key json+schema+md', len(k2), 'dupe groups', sum(len(v) > 1 for v in k2.values()))
for k, v in k2.items():
    if len(v) > 1: print(' dupe ids', v)
rows = list(csv.DictReader(open(f'{raw}/results.csv', newline='')))
url2ids = collections.defaultdict(set); miss = collections.Counter(); hit = collections.Counter()
for r in rows:
    key = (canon(r['true_json']), canon(r['json_schema']), r['true_markdown'])
    ids = k2.get(key)
    if ids is None:
        key1 = (canon(r['true_json']), canon(r['json_schema']))
        ids = k1.get(key1)
        if ids is None: miss[r['ocr_model']] += 1; continue
        hit['json+schema only'] += 1
    else:
        hit['full'] += 1
    url2ids[r['file_url']].update(ids)
print('hits', hit, 'misses', miss)
print('file_urls', len(url2ids), 'ambiguous', sum(len(v) > 1 for v in url2ids.values()))
id2urls = collections.defaultdict(set)
for u, ids in url2ids.items():
    if len(ids) == 1: id2urls[next(iter(ids))].add(u)
print('ids covered', len(id2urls), 'ids with >1 url', sum(len(v) > 1 for v in id2urls.values()))
# metadata json equality
m_by_id = {m['id']: m for m in meta}
fm = 0
for r in rows:
    pass
# provider coverage per id
prov = collections.defaultdict(set)
for r in rows:
    ids = url2ids.get(r['file_url'])
    if ids and len(ids) == 1: prov[next(iter(ids))].add(r['ocr_model'])
c = collections.Counter()
for i in m_by_id:
    for p in ['azure-document-intelligence', 'ground-truth', 'aws-textract', 'gemini-2.0-flash-001']:
        c[p] += p in prov.get(i, set())
print('coverage by provider over 1000 ids', dict(c))
