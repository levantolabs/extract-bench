"""Explore results.csv structure: id/file_url mapping to benchmark ids, provider coverage. Run with -I."""
import csv, json, sys, collections, re
csv.field_size_limit(1 << 30)
raw = sys.argv[1]
meta = {json.loads(l)['id']: json.loads(l) for l in open(f'{raw}/metadata.jsonl')}
rows = list(csv.DictReader(open(f'{raw}/results.csv', newline='')))
print('rows', len(rows))
r0 = rows[0]
for k, v in r0.items():
    print(k, '=>', repr(v[:160]))
urls = collections.Counter()
for r in rows[:5]:
    print(r['id'], r['file_url'], r['ocr_model'], r['extraction_model'])
# map file_url -> benchmark file_name
pat = re.compile(r'(\d+)\.(png|jpg|jpeg)$', re.I)
ids = collections.Counter()
unmatched = 0
for r in rows:
    m = pat.search(r['file_url'])
    if not m: unmatched += 1
    else: ids[m.group(1)] += 1
print('unmatched file_url', unmatched, 'distinct numeric ids', len(ids))
# check true_markdown / true_json equality with metadata for matched ids
mismatch_md = mismatch_json = 0
for r in rows:
    if r['ocr_model'] != 'azure-document-intelligence': continue
    m = pat.search(r['file_url']); bid = m.group(1) if m else None
    if bid not in meta: print('no meta', r['file_url']); continue
    md = meta[bid]
    tj = json.loads(md['true_json_output'])
    if json.loads(r['true_json']) != tj: mismatch_json += 1
    tm = md['true_markdown_output']; tm = json.loads(tm) if tm.startswith('"') else tm
    if r['true_markdown'] != tm: mismatch_md += 1
print('azure rows: true_json mismatches vs metadata', mismatch_json, 'true_md mismatches', mismatch_md)
