"""Profile the 13 use-case formats: scalar leaves (outside arrays), types, nulls, date/time hints. Run with -I."""
import json, sys, collections, re
raw = sys.argv[1]
FULL = ['SHIPPING_INVOICE', 'DELIVERY_NOTE', 'PHOTO_RECEIPT', 'DRIVER_LICENSE', 'FORM', 'SCANNED_FORM']
EVEN = ['BANK_CHECK', 'ACCOUNT_STATEMENT', 'CREDIT_CARD_STATEMENT', 'FORM_1040', 'PATIENT_INTAKE', 'COMMERCIAL_LEASE_AGREEMENT', 'PETITION_FORM']
meta = [json.loads(l) for l in open(f'{raw}/metadata.jsonl')]
MISSING = object()
def leaves(schema, gold, path=()):
    t = schema.get('type')
    if t == 'object' or (t is None and 'properties' in schema):
        for k, v in (schema.get('properties') or {}).items():
            g = gold.get(k, MISSING) if isinstance(gold, dict) else MISSING
            yield from leaves(v, g, path + (k,))
    elif t == 'array':
        yield (path, 'array', schema, gold)
    else:
        yield (path, t if t else ('enum' if 'enum' in schema else str(t)), schema, gold)
stat = collections.defaultdict(collections.Counter)
tyc = collections.Counter(); fmtkeys = collections.Counter(); desc_samples = collections.defaultdict(list)
for m in meta:
    md = json.loads(m['metadata']); f = md.get('format', '').strip()
    if f not in FULL + EVEN: continue
    s = json.loads(m['json_schema']); g = json.loads(m['true_json_output'])
    L = list(leaves(s, g))
    sc = [x for x in L if x[1] not in ('array', 'boolean')]
    st = stat[f]; st['docs'] += 1
    st['docs_with_scalar'] += bool(sc)
    st['scalars'] += len(sc)
    st['bool'] += sum(x[1] == 'boolean' for x in L)
    st['arrays'] += sum(x[1] == 'array' for x in L)
    st['null_or_missing'] += sum(x[3] is None or x[3] is MISSING for x in sc)
    st['missing_key'] += sum(x[3] is MISSING for x in sc)
    st['photo'] += md.get('documentQuality') == 'PHOTO'
    st['rotated'] += bool(md.get('rotation'))
    st['nested_scalar'] += sum(len(x[0]) > 1 for x in sc)
    for p, t, sch, gv in sc:
        tyc[(t, type(gv).__name__ if gv is not MISSING else 'MISSING')] += 1
        for k in sch:
            fmtkeys[k] += 1
        name = p[-1]
        d = sch.get('description', '')
        if re.search(r'date|dob|birth|expir|issued', name + ' ' + d, re.I) and len(desc_samples['date']) < 25:
            desc_samples['date'].append((f, '.'.join(p), t, d[:80], gv if gv is not MISSING else 'MISSING'))
        if re.search(r'time|hour|duration|minute|period|term|length|days|months|years', name + ' ' + d, re.I) and len(desc_samples['time']) < 60:
            desc_samples['time'].append((f, '.'.join(p), t, d[:80], gv if gv is not MISSING else 'MISSING'))
tot = collections.Counter()
for f in FULL + EVEN:
    print(f, dict(stat[f])); tot.update(stat[f])
print('TOTAL', dict(tot))
print('type x gold pytype', tyc.most_common())
print('schema keys on scalar leaves', fmtkeys.most_common())
for k, v in desc_samples.items():
    print('==', k)
    for x in v: print('  ', x)
