"""Per-format / per-type stats for CARD.md. Usage: python -I scripts/card_stats.py <dataset_dir> <out.json>"""
import collections, json, os, sys
D, OUT = sys.argv[1], sys.argv[2]
items = [json.loads(l) for l in open(os.path.join(D, 'items.jsonl'), encoding='utf-8')]
pf = collections.defaultdict(collections.Counter)
null_reason = collections.Counter(); date_rules = collections.Counter(); list_item = collections.Counter()
reenc = []; strata = collections.Counter(); photo_rot = collections.Counter(); dup = []
span = collections.defaultdict(collections.Counter)
for it in items:
    f = it['meta']['format']; m = it['meta']
    pf[f]['items'] += 1
    pf[f]['photo'] += m['document_quality'] == 'PHOTO'
    pf[f]['rotated'] += bool(m['rotation'])
    if m['image']['reencoded']: reenc.append((it['id'], m['image']))
    if m.get('cross_dataset_duplicate'): dup.append((it['id'], m['cross_dataset_duplicate']))
    pf[f]['bools_excluded'] += m['booleans_excluded']['n']
    for fl in it['fields']:
        fm = m['field_meta'][fl['name']]
        k = 'list' if fl['type'] == 'list' else 'scalar'
        pf[f][k] += 1
        if fl['gold'] is None:
            pf[f][k + '_null'] += 1
            null_reason[(k, fm.get('gold_absent_reason'))] += 1
        if fl['type'] == 'date': date_rules[fm.get('date_rule')] += 1
        if fl['type'] == 'list': list_item[fm['item_type']] += 1
        strata[fm['stratum']] += 1
        if 'span_in_primary_ocr' in fm:
            span[f]['n'] += 1; span[f]['in_azure'] += fm['span_in_primary_ocr']; span[f]['in_gold_md'] += fm['span_in_gold_markdown']
out = {'per_format': {k: dict(v) for k, v in pf.items()}, 'null_reason': {f'{a}|{b}': c for (a, b), c in null_reason.items()},
       'date_rules': dict(date_rules), 'list_item_types': dict(list_item), 'strata': dict(strata),
       'reencoded': reenc, 'cross_dataset_duplicates': dup,
       'text_span_presence_per_format': {k: dict(v) for k, v in span.items()}}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
