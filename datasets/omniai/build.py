#!/usr/bin/env python3
"""Build extract-bench/datasets/omniai/items.jsonl (OmniAI OCR Benchmark, use-case-weighted).

Inputs (all under raw/, downloaded at pinned revisions, see CARD.md):
  raw/metadata.jsonl          getomni-ai/ocr-benchmark @ 4ed0d952  test/metadata.jsonl (1,000 docs)
  raw/results.csv             getomni-ai/ocr-leaderboard @ 5bedb06b results.csv (8,932 per-item provider rows)
  raw/hf_tree_images.json     size + LFS sha256 of every test/images/* file at 4ed0d952
  raw/images/<file>           only the sampled page images (fetched by this script if missing)

Outputs (dataset dir):
  items.jsonl                       240 items in the extract-bench COMMON ITEM FORMAT
  img/<id>.<png|jpg>                page image, <= 4 MiB (verbatim copy unless re-encoding was needed)
  ocr/<source>/<id>.md              gold markdown + Azure DI / Textract / Gemini-2.0-Flash OCR text
  published_predictions.jsonl       per-item vendor-run predictions + json_accuracy for all 9 pipelines
  build_stats.json                  counts reported in CARD.md

Usage:
  /opt/homebrew/bin/uv run --no-project --with pillow --with tiktoken python -I build.py [--offline]

Never executes anything from raw/. Reads raw files as data only.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import datetime
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
KEY = 'omniai'
BENCH_REPO = 'getomni-ai/ocr-benchmark'
BENCH_REV = '4ed0d95271ca00107726230f7a0944ed9e90d897'
LB_REPO = 'getomni-ai/ocr-leaderboard'
LB_REV = '5bedb06b13b42743cbb4e2f088ec24da6cc10ca4'
IMG_URL = 'https://huggingface.co/datasets/{repo}/resolve/{rev}/test/{file}'
SEED = 20
N_TOTAL = 240
MAX_IMG_BYTES = 4 * 1024 * 1024  # 4 MiB (Clairic per-image limit)

# Spec C4: take ALL eligible docs of FULL formats, then an even share of EVEN formats.
FULL = ['SHIPPING_INVOICE', 'DELIVERY_NOTE', 'PHOTO_RECEIPT', 'DRIVER_LICENSE', 'FORM', 'SCANNED_FORM']
EVEN = ['BANK_CHECK', 'ACCOUNT_STATEMENT', 'CREDIT_CARD_STATEMENT', 'FORM_1040', 'PATIENT_INTAKE',
        'COMMERCIAL_LEASE_AGREEMENT', 'PETITION_FORM']
USE_CASE = {
    'SHIPPING_INVOICE': 'invoices / AP (shipping invoice)',
    'DELIVERY_NOTE': 'logistics (delivery note)',
    'PHOTO_RECEIPT': 'receipts / expenses (photographed receipt)',
    'DRIVER_LICENSE': 'ID / KYC (driver licence)',
    'FORM': 'forms (photographed business form)',
    'SCANNED_FORM': 'forms (scanned business form)',
    'BANK_CHECK': 'banking (bank check)',
    'ACCOUNT_STATEMENT': 'banking (account statement)',
    'CREDIT_CARD_STATEMENT': 'banking (credit-card statement)',
    'FORM_1040': 'tax forms (IRS Form 1040)',
    'PATIENT_INTAKE': 'clinical / medical (patient intake form)',
    'COMMERCIAL_LEASE_AGREEMENT': 'contracts / legal (commercial lease agreement)',
    'PETITION_FORM': 'government / regulatory forms (petition form)',
}
# Same physical receipt as an item of the suite's `sroie` set (company + date + total all equal; found by
# scripts/overlap_sroie_strict.py against the SROIE test entities). Flagged, not dropped: pooled suite stats
# must count each receipt once.
CROSS_DATASET_DUP = {9: {'dataset': 'sroie', 'sroie_test_key': 'X51006329183'},
                     19: {'dataset': 'sroie', 'sroie_test_key': 'X51005746203'}}
PRIMARY_OCR = 'azure-document-intelligence'
ALT_OCR = ['aws-textract', 'gemini-2.0-flash-001']
NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
MISSING = object()

# Heuristic Sage type map for `number` fields (CARD.md "Type map"); a hint for the harness only.
MONEY_TOK = {'amount', 'amounts', 'price', 'cost', 'total', 'subtotal', 'balance', 'rent', 'deposit', 'deposits',
             'fee', 'fees', 'tax', 'payment', 'payments', 'paid', 'due', 'charge', 'charges', 'wage', 'wages',
             'salary', 'income', 'credit', 'credits', 'debit', 'debits', 'withdrawal', 'withdrawals', 'gain',
             'loss', 'benefits', 'dividends', 'interest', 'deduction', 'refund', 'premium', 'tip', 'owed',
             'overpaid', 'withholding', 'cash', 'earnings', 'compensation', 'proceeds', 'contribution',
             'contributions', 'purchases', 'adjustments', 'value', 'tooling'}
COUNT_TOK = {'items', 'item', 'count', 'qty', 'quantity', 'number', 'num', 'sn', 'no', 'id', 'year', 'years',
             'month', 'months', 'days', 'weeks', 'hours', 'minutes', 'zip', 'percent', 'rate', 'pct', 'weight',
             'watts', 'feet', 'length', 'size', 'page', 'holes', 'tips', 'phone', 'units', 'digits'}
DURATION_TOK = {'years', 'months', 'weeks', 'days', 'hours', 'minutes', 'duration', 'term'}
DATE_TOK = {'date', 'dob', 'birth', 'birthday', 'expiration', 'expiry'}
MONTHS = {m: i + 1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
                                         'nov', 'dec'])}


# ----------------------------------------------------------------------------- helpers
def tokens(s):
    """camelCase / snake_case / digits -> lower-case word tokens."""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return [t.lower() for t in re.split(r'[^A-Za-z]+', s) if t]


def md_text(s):
    return json.loads(s) if s.startswith('"') else s


def canon(s):
    return json.dumps(json.loads(s), sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def year4(y):
    y = int(y)
    if y < 100:  # pre-registered 2-digit-year pivot
        return 2000 + y if y <= 30 else 1900 + y
    return y


def valid(y, m, d):
    try:
        return datetime.date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(s):
    """Return (iso, alt_iso, rule) for a full calendar date string, or None. alt_iso is set only when the
    numeric day/month order is ambiguous (both <= 12 and different): primary = month-first (US docs)."""
    if not isinstance(s, str):
        return None
    t = s.strip().rstrip('.').strip()
    m = re.fullmatch(r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', t)
    if m:
        iso = valid(int(m[1]), int(m[2]), int(m[3]))
        return (iso, None, 'ymd') if iso else None
    m = re.fullmatch(r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})', t)
    if m:
        a, b, y = int(m[1]), int(m[2]), year4(m[3])
        rule = 'yy_pivot30' if len(m[3]) == 2 else ''
        if a > 12 and b <= 12:
            iso = valid(y, b, a)
            return (iso, None, ('dmy ' + rule).strip()) if iso else None
        if b > 12 and a <= 12:
            iso = valid(y, a, b)
            return (iso, None, ('mdy ' + rule).strip()) if iso else None
        if a <= 12 and b <= 12:
            iso = valid(y, a, b)
            alt = valid(y, b, a) if a != b else None
            return (iso, alt, ('mdy_ambiguous ' + rule).strip() if alt else ('mdy ' + rule).strip()) if iso else None
        return None
    m = re.fullmatch(r'([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', t)
    if m and m[1][:3].lower() in MONTHS:
        iso = valid(int(m[3]), MONTHS[m[1][:3].lower()], int(m[2]))
        return (iso, None, 'month_name') if iso else None
    m = re.fullmatch(r'(\d{1,2})(?:st|nd|rd|th)?[\s-]+([A-Za-z]{3,9})\.?,?[\s-]+(\d{2}|\d{4})', t)
    if m and m[2][:3].lower() in MONTHS:
        iso = valid(year4(m[3]), MONTHS[m[2][:3].lower()], int(m[1]))
        return (iso, None, 'month_name') if iso else None
    return None


def parse_time(s):
    """Return list of acceptable ISO clock forms for a time string, or None."""
    if not isinstance(s, str):
        return None
    m = re.fullmatch(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:([AaPp])\.?\s*[Mm]\.?)?', s.strip())
    if not m:
        # bare HHMM (one PHOTO_RECEIPT gold is '1223'); only reached for fields whose name says 'time'
        m = re.fullmatch(r'(\d{2})(\d{2})()()', s.strip())
        if m:
            m = [None, m[1], m[2], None, None]
    if not m:
        return None
    h, mi, se, ap = int(m[1]), int(m[2]), m[3], m[4]
    if ap:
        if not 1 <= h <= 12:
            return None
        h = (h % 12) + (12 if ap.lower() == 'p' else 0)
    if h > 23 or mi > 59 or (se and int(se) > 59):
        return None
    full = f'{h:02d}:{mi:02d}:{int(se or 0):02d}'
    out = [full]
    if not se or int(se) == 0:
        out.append(f'{h:02d}:{mi:02d}')
    return out


def dedupe(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def norm_span(s):
    return re.sub(r'\s+', ' ', re.sub(r'[*|#`]', ' ', str(s).lower())).strip()


def sanitize(raw_name, used):
    s = re.sub(r'[^A-Za-z0-9_]', '_', raw_name)
    if not re.match(r'[A-Za-z_]', s[:1]):
        s = '_' + s
    if len(s) > 64:
        s = s[:55] + '_' + hashlib.sha1(raw_name.encode()).hexdigest()[:8]
    base, k = s, 2
    while s in used:
        suf = f'_{k}'
        s = base[:64 - len(suf)] + suf
        k += 1
    assert NAME_RE.match(s), s
    used.add(s)
    return s


def leaf_kind(s):
    """Normalise a schema leaf's type: string | number | integer | boolean | enum | object | array."""
    t = s.get('type')
    if isinstance(t, list):
        t = next((x for x in t if x != 'null'), None)
    if t == 'enum' or 'enum' in s or 'emum' in s:  # 'emum' is a typo present in 28 PATIENT_INTAKE schemas
        if t in (None, 'enum', 'string'):
            return 'enum'
    if t is None and 'properties' in s:
        return 'object'
    return t


def enum_options(s):
    return s.get('enum') or s.get('emum') or []


# ----------------------------------------------------------------------------- schema walk
def child(gold, k):
    if isinstance(gold, dict):
        return gold.get(k, MISSING)
    if gold is None:
        return None
    return MISSING


def walk(schema, gold, path, descs, out, bools):
    """Collect scalar leaves and arrays outside arrays. path: list of keys. descs: parallel object descriptions."""
    k = leaf_kind(schema)
    if k == 'object':
        for key, sub in (schema.get('properties') or {}).items():
            walk(sub, child(gold, key), path + [key], descs + [schema.get('description', '') if path else ''], out, bools)
    elif k == 'array':
        out.append(('array', path, schema, gold, descs))
    elif k == 'boolean':
        bools.append(('.'.join(path), None if gold is MISSING else gold))
    else:
        out.append(('scalar', path, schema, gold, descs))


def columns(item_schema, rel, descs, out):
    """Columns of an array-of-objects: (rel_path, leaf_schema, kind, descs). Nested scalar arrays -> 'nested_list'."""
    k = leaf_kind(item_schema)
    if k == 'object':
        for key, sub in (item_schema.get('properties') or {}).items():
            columns(sub, rel + [key], descs + [item_schema.get('description', '') if rel else ''], out)
    elif k == 'array':
        it = item_schema.get('items') or {}
        ik = leaf_kind(it)
        out.append((rel, it, 'nested_list' if ik not in ('object', 'array') else 'nested_records', descs))
    else:
        out.append((rel, item_schema, 'col', descs))


def get_rel(elem, rel):
    cur = elem
    for key in rel:
        if isinstance(cur, dict):
            cur = cur.get(key, MISSING)
        else:
            return MISSING
    return cur


def raw_fields(doc):
    """Return (raw_fields, bools, skipped) for one doc. Each raw field is a dict with kind scalar|list."""
    fmt = doc['format']
    schema, gold = doc['schema'], doc['gold']
    out, bools = [], []
    flatten = False
    if fmt == 'BANK_CHECK':
        arr = gold.get('checks') if isinstance(gold, dict) else None
        if isinstance(arr, list) and len(arr) == 1:
            flatten = True  # declared adaptation: single-check page -> scalar fields
            it = (schema.get('properties') or {}).get('checks', {}).get('items') or {}
            walk(it, arr[0], ['checks[0]'], [''], out, bools)
            for key, sub in (schema.get('properties') or {}).items():
                if key != 'checks':
                    walk(sub, child(gold, key), [key], [''], out, bools)
    if not flatten:
        walk(schema, gold, [], [], out, bools)
    fields, skipped = [], []
    for kind, path, s, g, descs in out:
        if kind == 'scalar':
            fields.append(dict(kind='scalar', path=path, schema=s, gold=g, descs=descs,
                               path_key='.'.join(path), flattened=flatten and path[0] == 'checks[0]'))
            continue
        # arrays -> list fields
        it = s.get('items') or {}
        ik = leaf_kind(it)
        arr_desc = s.get('description', '')
        if ik == 'boolean':
            n = len(g) if isinstance(g, list) else 0
            bools.append(('.'.join(path) + '[]', g if g is not MISSING else None))
            continue
        if ik not in ('object', 'array'):
            vals = None if (g is None or g is MISSING) else [v for v in g if v is not None]
            fields.append(dict(kind='list', path=path, schema=it, gold=vals, descs=descs, arr_desc=arr_desc,
                               col=None, path_key='.'.join(path) + '[]', nested=False))
            continue
        if ik == 'array':
            skipped.append('.'.join(path) + '[][]')
            continue
        cols = []
        columns(it, [], [], cols)
        for rel, cs, ckind, cdescs in cols:
            ck = leaf_kind(cs)
            pk = '.'.join(path) + '[].' + '.'.join(rel)
            if ckind == 'nested_records':
                skipped.append(pk + '[]')
                continue
            if ck == 'boolean' or (ckind == 'nested_list' and ck == 'boolean'):
                bools.append((pk, None if (g is None or g is MISSING) else
                              [get_rel(e, rel) if get_rel(e, rel) is not MISSING else None for e in g]))
                continue
            if g is None or g is MISSING:
                vals = None
            else:
                vals = []
                for e in g:
                    v = get_rel(e, rel)
                    if ckind == 'nested_list':
                        if isinstance(v, list):
                            vals.extend(x for x in v if x is not None)
                    elif v is not None and v is not MISSING:
                        vals.append(v)
            fields.append(dict(kind='list', path=path, schema=cs, gold=vals, descs=descs, arr_desc=arr_desc,
                               col=rel, col_descs=cdescs, path_key=pk + ('[]' if ckind == 'nested_list' else ''),
                               nested=ckind == 'nested_list'))
    return fields, bools, skipped


# ----------------------------------------------------------------------------- typing (dataset level)
def is_dateish(s, key):
    if s.get('format') == 'date':
        return True
    if DATE_TOK & set(tokens(key)):
        return True
    return 'date' in tokens(s.get('description', ''))


def decide_types(docs):
    """Per (format, path_key) decide text/date/time for string leaves and int-ness for numbers."""
    vals = collections.defaultdict(list)
    meta = {}
    for d in docs:
        fields, _, _ = raw_fields(d)
        for f in fields:
            key = (d['format'], f['path_key'])
            last = (f['col'] or f['path'])[-1] if f['kind'] == 'list' and f['col'] else f['path'][-1]
            meta[key] = (leaf_kind(f['schema']), f['schema'], last)
            g = f['gold']
            if g is None or g is MISSING:
                continue
            vals[key].extend(g if f['kind'] == 'list' else [g])
    decided = {}
    for key, (k, s, last) in meta.items():
        v = vals.get(key, [])
        if k == 'string':
            if is_dateish(s, last) and v and all(parse_date(x) for x in v):
                decided[key] = ('date', 'schema format=date' if s.get('format') == 'date' else 'name/description says date')
            elif s.get('format') == 'date' or is_dateish(s, last):
                decided[key] = ('text', 'date-ish but not every gold value is a full calendar date')
            elif 'time' in tokens(last) and v and all(parse_time(x) for x in v):
                decided[key] = ('time', 'name says time; every gold value is a clock time')
            else:
                decided[key] = ('text', '')
        elif k in ('number', 'integer'):
            all_int = bool(v) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
            decided[key] = ('number', 'all_int' if (all_int or k == 'integer') else 'has_float')
        elif k == 'enum':
            decided[key] = ('enum', '')
        else:
            decided[key] = ('text', f'unusual schema type {k!r} -> text')
    return decided


def sage_type(ftype, key_last, desc, int_flag):
    if ftype in ('text', 'enum'):
        return 'text'
    if ftype in ('date', 'time', 'datetime', 'duration'):
        return ftype
    kt = set(tokens(key_last))
    dt = set(tokens(desc))
    if not (kt & COUNT_TOK) and ((kt & MONEY_TOK) or ((dt & MONEY_TOK) and not (dt & COUNT_TOK))):
        return 'money'
    return 'integer' if int_flag == 'all_int' else 'decimal'


# ----------------------------------------------------------------------------- field building
def human(path, descs):
    """'a [desc of a] > b > leaf'. descs[j] = description of the object at path[:j] (descs[0] = root, unused)."""
    parts = []
    for i, key in enumerate(path):
        key = 'check' if key == 'checks[0]' else key
        d = descs[i + 1] if i + 1 < len(descs) else ''
        parts.append(f'{key} [{d.strip()}]' if (d and d.strip() and i < len(path) - 1) else key)
    return ' > '.join(parts)


def build_fields(doc, decided, azure_text, gold_md):
    fields_raw, bools, skipped = raw_fields(doc)
    used, fields, fmeta = set(), [], {}
    stats = collections.Counter()
    for f in fields_raw:
        s = f['schema']
        key = (doc['format'], f['path_key'])
        dtype, why = decided[key]
        desc = (s.get('description') or '').strip()
        suffix = ''
        if f['kind'] == 'scalar':
            last = f['path'][-1]
            raw_name = '__'.join(['check' if p == 'checks[0]' else p for p in f['path']])
            orig = '.'.join(f['path'])
            instr = human(f['path'], f['descs'])
            if desc:
                instr += f': {desc}'
        else:
            last = (f['col'] or f['path'])[-1]
            raw_name = '__'.join(f['path'] + (f['col'] or []))
            orig = f['path_key']
            if f['col'] is None:
                instr = human(f['path'], f['descs'])
                d2 = desc or (f.get('arr_desc') or '').strip()
                if d2:
                    instr += f': {d2}'
                suffix = ' Return every value, in document order.'
            else:
                arr = human(f['path'], f['descs'])
                if (f.get('arr_desc') or '').strip():
                    arr += f' [{f["arr_desc"].strip()}]'
                instr = f'For every entry of {arr}: ' + human(f['col'], f['col_descs'])
                if desc:
                    instr += f': {desc}'
                if f['nested']:
                    suffix = ' Return all values of this nested list across entries, in document order.'
                else:
                    suffix = ' Return one value per entry, in document order.'
        name = sanitize(raw_name, used)
        opts = enum_options(s) if dtype == 'enum' else []
        if opts:
            instr += ('' if instr.endswith('.') else '.') + ' One of: ' + ' | '.join(str(o) for o in opts) + '.'
        instr = (instr if instr.endswith('.') else instr + '.') + suffix
        g = f['gold']
        fm = {'orig_path': orig, 'schema_type': leaf_kind(s), 'kind': f['kind']}
        if f['kind'] == 'scalar' and f.get('flattened'):
            fm['adaptation'] = 'bank_check_single_record_flatten'
        if g is MISSING:
            fm['gold_absent_reason'] = 'key_missing_in_gold_json'
        elif g is None:
            fm['gold_absent_reason'] = 'null_in_gold_json'
        if f['kind'] == 'scalar':
            if dtype == 'enum':
                ftype = 'text'
                gold = None if g in (None, MISSING) else dedupe([str(g), str(g).replace('_', ' ')])
                fm['enum_options'] = opts
                stratum = 'enum'
            elif dtype == 'date':
                ftype = 'date'
                if g in (None, MISSING):
                    gold = None
                else:
                    iso, alt, rule = parse_date(g)
                    gold = dedupe([g, iso] + ([alt] if alt else []))
                    fm['date_rule'] = rule
                    if alt:
                        stats['date_ambiguous'] += 1
                stratum = 'date'
                fm['type_rule'] = why
            elif dtype == 'time':
                ftype = 'time'
                gold = None if g in (None, MISSING) else dedupe([g] + parse_time(g))
                stratum = 'time'
                fm['type_rule'] = why
            elif dtype == 'number':
                ftype = 'number'
                if g in (None, MISSING):
                    gold = None
                elif isinstance(g, bool):
                    gold = g
                    stats['number_gold_bool'] += 1
                elif isinstance(g, (int, float)):
                    gold = g
                else:
                    try:
                        gold = float(str(g).replace(',', ''))
                        gold = int(gold) if gold.is_integer() else gold
                        fm['gold_coerced_from_string'] = g
                        stats['number_gold_coerced'] += 1
                    except ValueError:
                        gold = g
                        stats['number_gold_non_numeric'] += 1
                fm['sage_type'] = sage_type('number', last, desc, why)
                stratum = 'duration_component' if set(tokens(last)) & DURATION_TOK else fm['sage_type']
            else:
                ftype = 'text'
                gold = None if g in (None, MISSING) else (g if isinstance(g, str) else str(g))
                if g not in (None, MISSING) and not isinstance(g, str):
                    fm['gold_coerced_from'] = type(g).__name__
                stratum = 'text'
                if why:
                    fm['type_rule'] = why
            fm.setdefault('sage_type', {'text': 'text', 'date': 'date', 'time': 'time'}.get(ftype))
            fm['metric'] = 'exact_norm'
            if ftype == 'text' and gold is not None:
                accs = gold if isinstance(gold, list) else [gold]
                fm['span_in_primary_ocr'] = any(norm_span(a) in norm_span(azure_text) for a in accs)
                fm['span_in_gold_markdown'] = any(norm_span(a) in norm_span(gold_md) for a in accs)
            if isinstance(gold, list) and len(gold) == 1:
                gold = gold[0]
        else:
            ftype = 'list'
            item_type = {'number': 'number', 'integer': 'number', 'date': 'date'}.get(dtype, 'text')
            fm['item_type'] = item_type
            fm['metric'] = 'list_f1'
            fm['sage_type'] = None
            stratum = 'list'
            if f['nested']:
                fm['adaptation'] = 'nested_list_flattened_across_entries'
            if dtype == 'enum':
                fm['enum_options'] = opts
            if g is None:
                gold = None
            elif item_type == 'date':
                parsed = [parse_date(x) for x in g]
                gold = [p[0] for p in parsed]
                fm['gold_verbatim'] = g
                fm['date_ambiguous_elements'] = sum(1 for p in parsed if p[1])
            else:
                gold = list(g)
        fm['stratum'] = stratum
        fields.append({'name': name, 'type': ftype, 'instructions': instr, 'gold': gold})
        fmeta[name] = fm
        stats['fields_' + f['kind']] += 1
        stats['null_gold_' + f['kind']] += gold is None
    return fields, fmeta, bools, skipped, stats


# ----------------------------------------------------------------------------- images
def fetch_image(file_name, tree, offline):
    rel = 'test/' + file_name
    dst = os.path.join(RAW, 'images', os.path.basename(file_name))
    exp = tree['files'][rel]
    if not os.path.exists(dst):
        if offline:
            raise SystemExit(f'missing {dst} (offline)')
        url = IMG_URL.format(repo=BENCH_REPO, rev=BENCH_REV, file=file_name)
        req = urllib.request.Request(url, headers={'User-Agent': 'extract-bench-builder/0.1'})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    raise
                print('retry', file_name, e, file=sys.stderr)
        tmp = dst + '.part'
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, dst)
    data = open(dst, 'rb').read()
    sha = hashlib.sha256(data).hexdigest()
    if len(data) != exp['size'] or sha != exp['lfs_sha256']:
        raise SystemExit(f'checksum mismatch for {dst}: {len(data)} {sha} vs {exp}')
    return dst, sha


def prepare_image(src, item_id):
    from PIL import Image  # noqa: PLC0415
    data = open(src, 'rb').read()
    with Image.open(io.BytesIO(data)) as im:
        fmt, (w, h), mode = im.format, im.size, im.mode
        info = {'orig_file': os.path.relpath(src, HERE), 'orig_bytes': len(data), 'orig_format': fmt,
                'width': w, 'height': h, 'mode': mode, 'orig_sha256': hashlib.sha256(data).hexdigest()}
        # MPO = phone JPEG with an extra frame; its first frame is a baseline JPEG (FFD8 magic), copied verbatim.
        if fmt in ('PNG', 'JPEG', 'MPO') and len(data) <= MAX_IMG_BYTES:
            ext = 'png' if fmt == 'PNG' else 'jpg'
            out = os.path.join(HERE, 'img', f'{item_id}.{ext}')
            shutil.copyfile(src, out)
            info.update(reencoded=False, out_bytes=len(data), out_format='PNG' if fmt == 'PNG' else 'JPEG')
            return f'img/{item_id}.{ext}', info
        im.load()
        if im.mode in ('RGBA', 'LA', 'P', 'PA'):
            rgba = im.convert('RGBA')
            work = Image.new('RGB', rgba.size, (255, 255, 255))
            work.paste(rgba, mask=rgba.split()[-1])
        elif im.mode not in ('RGB', 'L'):
            work = im.convert('RGB')
        else:
            work = im.copy()
    scale, q = 1.0, 90
    while True:
        cur = work if scale == 1.0 else work.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        cur.save(buf, 'JPEG', quality=q, optimize=True)
        if buf.tell() <= MAX_IMG_BYTES:
            break
        scale *= 0.85
    out = os.path.join(HERE, 'img', f'{item_id}.jpg')
    with open(out, 'wb') as fh:
        fh.write(buf.getvalue())
    info.update(reencoded=True, out_bytes=buf.tell(), out_format='JPEG', jpeg_quality=q, scale=round(scale, 4),
                out_width=cur.size[0], out_height=cur.size[1],
                reencode_reason='over 4 MiB' if len(data) > MAX_IMG_BYTES else f'format {fmt} not PNG/JPEG')
    return f'img/{item_id}.jpg', info


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading missing images')
    args = ap.parse_args()
    import tiktoken  # noqa: PLC0415
    enc = tiktoken.get_encoding('o200k_base')

    csv.field_size_limit(1 << 30)
    meta_rows = [json.loads(l) for l in open(os.path.join(RAW, 'metadata.jsonl'))]
    docs = []
    for r in meta_rows:
        md = json.loads(r['metadata'])
        docs.append({'orig_id': int(r['id']), 'format': (md.get('format') or '').strip(), 'meta': md,
                     'schema': json.loads(r['json_schema']), 'gold': json.loads(r['true_json_output']),
                     'schema_raw': r['json_schema'], 'gold_raw': r['true_json_output'],
                     'gold_md': md_text(r['true_markdown_output']), 'file_name': r['file_name']})
    by_id = {d['orig_id']: d for d in docs}
    key2id = {(canon(d['gold_raw']), canon(d['schema_raw'])): d['orig_id'] for d in docs}
    assert len(key2id) == len(docs)

    # join results.csv rows to benchmark ids by (gold JSON, schema) content; file_url is an S3 UUID
    prov = collections.defaultdict(dict)
    unjoined = 0
    for r in csv.DictReader(open(os.path.join(RAW, 'results.csv'), newline='')):
        i = key2id.get((canon(r['true_json']), canon(r['json_schema'])))
        if i is None:
            unjoined += 1
            continue
        assert r['ocr_model'] not in prov[i], ('duplicate provider row', i, r['ocr_model'])
        prov[i][r['ocr_model']] = r
    assert unjoined == 0, unjoined

    scope = [d for d in docs if d['format'] in FULL + EVEN]
    decided = decide_types(scope)

    # eligibility: Azure DI text present AND >= 1 non-boolean scalar field after mapping
    elig = collections.defaultdict(list)
    inelig = collections.Counter()
    for d in scope:
        if PRIMARY_OCR not in prov[d['orig_id']]:
            inelig[(d['format'], 'no_azure_di_text')] += 1
            continue
        fr, _, _ = raw_fields(d)
        if not any(f['kind'] == 'scalar' for f in fr):
            inelig[(d['format'], 'no_non_bool_scalar')] += 1
            continue
        elig[d['format']].append(d['orig_id'])

    rng = random.Random(SEED)
    chosen, how = [], {}
    for f in FULL:
        for i in sorted(elig[f]):
            chosen.append(i)
            how[i] = 'full_format_all'
    remaining = N_TOTAL - len(chosen)
    base, extra = divmod(remaining, len(EVEN))
    quotas = {f: base + (1 if k < extra else 0) for k, f in enumerate(EVEN)}
    for f in EVEN:
        pool = sorted(elig[f])
        q = quotas[f]
        assert len(pool) >= q, (f, len(pool), q)
        forced = [i for i in pool if by_id[i]['meta'].get('documentQuality') == 'PHOTO' or by_id[i]['meta'].get('rotation')]
        if len(forced) > q:
            forced = sorted(rng.sample(forced, q))
        rest = [i for i in pool if i not in forced]
        fill = sorted(rng.sample(rest, q - len(forced)))
        for i in forced:
            chosen.append(i)
            how[i] = 'even_share_forced_photo_or_rotated'
        for i in fill:
            chosen.append(i)
            how[i] = 'even_share_random_seed20'
    assert len(chosen) == N_TOTAL and len(set(chosen)) == N_TOTAL, len(chosen)

    tree = json.load(open(os.path.join(RAW, 'hf_tree_images.json')))
    assert tree['revision'] == BENCH_REV
    os.makedirs(os.path.join(RAW, 'images'), exist_ok=True)
    for sub in ['img', 'ocr/gold_markdown', f'ocr/{PRIMARY_OCR}'] + [f'ocr/{p}' for p in ALT_OCR]:
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    with cf.ThreadPoolExecutor(8) as ex:
        fetched = dict(zip(chosen, ex.map(lambda i: fetch_image(by_id[i]['file_name'], tree, args.offline), chosen)))

    items, preds = [], []
    tot = collections.Counter()
    tok_text, tok_gold, tok_fields = [], [], []
    for i in chosen:
        d = by_id[i]
        item_id = f'{KEY}-{i:04d}'
        rows = prov[i]
        azure = rows[PRIMARY_OCR]['predicted_markdown']
        fields, fmeta, bools, skipped, st = build_fields(d, decided, azure, d['gold_md'])
        tot.update(st)
        img_rel, img_info = prepare_image(fetched[i][0], item_id)
        ocr_paths = {}
        for src, text in [('gold_markdown', d['gold_md']), (PRIMARY_OCR, azure)] + \
                         [(p, rows[p]['predicted_markdown'] if p in rows else None) for p in ALT_OCR]:
            if text is None:
                ocr_paths[src] = None
                continue
            rel = f'ocr/{src}/{item_id}.md'
            with open(os.path.join(HERE, rel), 'w', encoding='utf-8') as fh:
                fh.write(text)
            ocr_paths[src] = rel
        n_text = len(enc.encode(azure, disallowed_special=()))
        n_gold = len(enc.encode(d['gold_md'], disallowed_special=()))
        n_fdef = len(enc.encode(json.dumps([{k: f[k] for k in ('name', 'type', 'instructions')} for f in fields]),
                                disallowed_special=()))
        tok_text.append(n_text)
        tok_gold.append(n_gold)
        tok_fields.append(n_fdef)
        md = d['meta']
        published = {p: float(r['json_accuracy']) for p, r in sorted(rows.items())}
        item = {
            'id': item_id,
            'dataset': KEY,
            'use_case': USE_CASE[d['format']],
            'modality': 'image',
            'text': azure,
            'text_source': 'official_ocr',
            'images': [img_rel],
            'fields': fields,
            'metric': 'exact_norm',
            'meta': {
                'split': 'test',
                'source': f'{BENCH_REPO}@{BENCH_REV}',
                'ocr_source': f'{LB_REPO}@{LB_REV} results.csv ocr_model={PRIMARY_OCR}',
                'orig_id': i,
                'file_name': d['file_name'],
                'results_file_url': rows[PRIMARY_OCR]['file_url'],
                'format': d['format'],
                'document_quality': md.get('documentQuality'),
                'font_family': md.get('fontFamily'),
                'rotation': md.get('rotation', 0),
                'page_count': 1,
                'sample_rule': how[i],
                'cross_dataset_duplicate': CROSS_DATASET_DUP.get(i),
                'text_note': 'text = Azure Document Intelligence markdown exactly as shipped in results.csv '
                             '(realistic commercial OCR). Gold markdown (ceiling arm) and Textract / Gemini-2.0-Flash '
                             'OCR text are in ocr_alternatives.',
                'ocr_alternatives': ocr_paths,
                'tokens_o200k': {'text': n_text, 'gold_markdown': n_gold, 'field_defs': n_fdef},
                'image': img_info,
                'name_map': {n: fm['orig_path'] for n, fm in fmeta.items()},
                'field_meta': fmeta,
                'n_fields': {'scalar': st['fields_scalar'], 'list': st['fields_list'],
                             'null_gold_scalar': st['null_gold_scalar'], 'null_gold_list': st['null_gold_list']},
                'booleans_excluded': {'n': len(bools), 'gold': dict(bools)},
                'skipped_paths': skipped,
                'published_json_accuracy': published,
                'published_note': 'Per-item json_accuracy from vendor-run Feb-2025 pipelines (OCR -> azure-gpt-4o '
                                  'extraction; ground-truth row = gold markdown -> gpt-4o), scored by OmniAI json.ts '
                                  'on the FULL gold JSON (incl. arrays and booleans). Not comparable to scalar exact_norm.',
                'json_schema': d['schema_raw'],
                'true_json': d['gold_raw'],
            },
        }
        items.append(item)
        preds.append({'id': item_id, 'orig_id': i, 'pipelines': {
            p: {'extraction_model': r['extraction_model'], 'json_accuracy': float(r['json_accuracy']),
                'predicted_json': r['predicted_json']} for p, r in sorted(rows.items())}})
        tot['items_' + d['format']] += 1
        tot['booleans_excluded'] += len(bools)
        tot['skipped_paths'] += len(skipped)
        tot['images_reencoded'] += img_info['reencoded']
        tot['photo'] += md.get('documentQuality') == 'PHOTO'
        tot['rotated'] += bool(md.get('rotation'))

    with open(os.path.join(HERE, 'items.jsonl'), 'w', encoding='utf-8') as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + '\n')
    with open(os.path.join(HERE, 'published_predictions.jsonl'), 'w', encoding='utf-8') as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + '\n')

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

    # field-level stats
    fstat = collections.Counter()
    span = collections.Counter()
    for it in items:
        for f in it['fields']:
            fm = it['meta']['field_meta'][f['name']]
            fstat[(f['type'], fm['stratum'])] += 1
            if f['type'] != 'list':
                fstat['scalar_total'] += 1
                fstat['scalar_null'] += f['gold'] is None
                if fm.get('sage_type'):
                    fstat['sage_' + fm['sage_type']] += 1
            else:
                fstat['list_total'] += 1
                fstat['list_null'] += f['gold'] is None
            if 'span_in_primary_ocr' in fm:
                span['text_fields'] += 1
                span['in_primary_ocr'] += fm['span_in_primary_ocr']
                span['in_gold_markdown'] += fm['span_in_gold_markdown']
    pub = collections.defaultdict(list)
    for it in items:
        for p, v in it['meta']['published_json_accuracy'].items():
            pub[p].append(v)
    stats = {
        'n_items': len(items),
        'per_format': {f: tot['items_' + f] for f in FULL + EVEN},
        'quotas_even': quotas,
        'eligible_per_format': {f: len(elig[f]) for f in FULL + EVEN},
        'ineligible': {f'{k[0]}:{k[1]}': v for k, v in sorted(inelig.items())},
        'sample_rule_counts': dict(collections.Counter(how[i] for i in chosen)),
        'photo_items': tot['photo'], 'rotated_items': tot['rotated'],
        'images_reencoded': tot['images_reencoded'],
        'fields': {('|'.join(k) if isinstance(k, tuple) else k): v for k, v in sorted(fstat.items(), key=str)},
        'text_span_presence': dict(span),
        'date_ambiguous_scalar': tot['date_ambiguous'],
        'number_gold_coerced': tot['number_gold_coerced'], 'number_gold_non_numeric': tot['number_gold_non_numeric'],
        'booleans_excluded': tot['booleans_excluded'], 'skipped_paths': tot['skipped_paths'],
        'tokens_o200k_text': {'p50': pct(tok_text, 50), 'p95': pct(tok_text, 95), 'max': max(tok_text),
                              'mean': round(sum(tok_text) / len(tok_text))},
        'tokens_o200k_gold_markdown': {'p50': pct(tok_gold, 50), 'p95': pct(tok_gold, 95), 'max': max(tok_gold)},
        'tokens_o200k_field_defs': {'p50': pct(tok_fields, 50), 'p95': pct(tok_fields, 95), 'max': max(tok_fields)},
        'published_json_accuracy_mean_on_sample': {p: {'n': len(v), 'mean': round(sum(v) / len(v), 4)}
                                                   for p, v in sorted(pub.items())},
        'type_decisions': {f'{k[0]}:{k[1]}': v for k, v in sorted(decided.items()) if v[0] in ('date', 'time') or v[1]},
    }
    json.dump(stats, open(os.path.join(HERE, 'build_stats.json'), 'w'), indent=1, sort_keys=True)
    print(json.dumps({k: stats[k] for k in ['n_items', 'per_format', 'sample_rule_counts', 'images_reencoded',
                                            'tokens_o200k_text', 'text_span_presence']}, indent=1))


if __name__ == '__main__':
    main()
