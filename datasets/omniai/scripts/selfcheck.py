"""Self-check extract-bench/datasets/omniai/items.jsonl against the COMMON ITEM FORMAT.

Checks: every line parses; required keys; dataset/modality/text_source/metric values; field names match
^[A-Za-z_][A-Za-z0-9_]{0,63}$ and are unique per item; field types in the allowed set; every field has a
'gold' key (value, list of acceptable values, or explicit null); list golds are lists; image paths exist,
<= 4 per item, <= 4 MiB, magic bytes PNG/JPEG/WebP; o200k token counts of text (p50/p95).

Usage: uv run --no-project --with tiktoken python -I scripts/selfcheck.py <dataset_dir> <out.json>
"""
import collections
import json
import os
import re
import sys

import tiktoken

D, OUT = sys.argv[1], sys.argv[2]
NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
TYPES = {'text', 'number', 'money', 'date', 'time', 'datetime', 'duration', 'list'}
METRICS = {'exact_norm', 'anls', 'token_f1', 'numeric_tol', 'date_exact', 'list_f1', 'edit_sim'}
REQ = ['id', 'dataset', 'use_case', 'modality', 'text', 'text_source', 'images', 'fields', 'metric', 'meta']
enc = tiktoken.get_encoding('o200k_base')
errors, warn = [], []
toks, ids = [], set()
c = collections.Counter()
per_item_fields = []


def magic(b):
    if b[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if b[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if b[:4] == b'RIFF' and b[8:12] == b'WEBP':
        return 'webp'
    return None


for ln, line in enumerate(open(os.path.join(D, 'items.jsonl'), encoding='utf-8'), 1):
    try:
        it = json.loads(line)
    except Exception as e:  # noqa: BLE001
        errors.append(f'line {ln}: parse error {e}')
        continue
    c['items'] += 1
    for k in REQ:
        if k not in it:
            errors.append(f'line {ln}: missing key {k}')
    if it.get('id') in ids:
        errors.append(f'line {ln}: duplicate id {it.get("id")}')
    ids.add(it.get('id'))
    if it.get('dataset') != 'omniai':
        errors.append(f'line {ln}: dataset {it.get("dataset")}')
    if it.get('modality') not in ('text', 'image', 'text+image'):
        errors.append(f'line {ln}: modality {it.get("modality")}')
    if it.get('text_source') not in ('native', 'official_ocr', 'none'):
        errors.append(f'line {ln}: text_source')
    if it.get('metric') not in METRICS:
        errors.append(f'line {ln}: metric {it.get("metric")}')
    if it.get('text_source') == 'official_ocr' and not isinstance(it.get('text'), str):
        errors.append(f'line {ln}: official_ocr but text not str')
    if isinstance(it.get('text'), str):
        toks.append(len(enc.encode(it['text'], disallowed_special=())))
        if not it['text'].strip():
            errors.append(f'line {ln}: empty text')
    imgs = it.get('images') or []
    if len(imgs) > 4:
        errors.append(f'line {ln}: {len(imgs)} images')
    for p in imgs:
        fp = os.path.join(D, p)
        if not os.path.isfile(fp):
            errors.append(f'line {ln}: image missing {p}')
            continue
        b = open(fp, 'rb').read()
        if len(b) > 4 * 1024 * 1024:
            errors.append(f'line {ln}: image > 4 MiB {p} {len(b)}')
        m = magic(b)
        if m is None:
            errors.append(f'line {ln}: image not PNG/JPEG/WebP {p}')
        c['img_' + str(m)] += 1
    names = set()
    fields = it.get('fields') or []
    if not fields:
        errors.append(f'line {ln}: no fields')
    n_scalar = 0
    for f in fields:
        for k in ('name', 'type', 'instructions', 'gold'):
            if k not in f:
                errors.append(f'line {ln}: field missing {k}: {f.get("name")}')
        n = f.get('name', '')
        if not NAME_RE.match(n):
            errors.append(f'line {ln}: bad field name {n!r}')
        if n in names:
            errors.append(f'line {ln}: duplicate field {n}')
        names.add(n)
        if f.get('type') not in TYPES:
            errors.append(f'line {ln}: bad type {f.get("type")} for {n}')
        if not isinstance(f.get('instructions'), str) or not f['instructions'].strip():
            errors.append(f'line {ln}: empty instructions {n}')
        g = f.get('gold', '__ABSENT__')
        if f.get('type') == 'list':
            if g is not None and not isinstance(g, list):
                errors.append(f'line {ln}: list gold not list {n}')
            c['list_fields'] += 1
            c['list_null'] += g is None
            c['list_empty'] += g == []
        else:
            n_scalar += 1
            c['scalar_fields'] += 1
            c['scalar_null'] += g is None
            c['scalar_multi_acceptable'] += isinstance(g, list)
            c['scalar_type_' + f['type']] += 1
            if isinstance(g, list) and not g:
                errors.append(f'line {ln}: empty acceptable list {n}')
            if f['type'] == 'number' and g is not None and not (isinstance(g, (int, float)) and not isinstance(g, bool)):
                warn.append(f'line {ln}: number gold not numeric {n}={g!r}')
        fm = (it.get('meta') or {}).get('field_meta', {}).get(n)
        if fm is None:
            errors.append(f'line {ln}: no field_meta for {n}')
    if n_scalar == 0:
        errors.append(f'line {ln}: no scalar field')
    per_item_fields.append(len(fields))
    c['fmt_' + it['meta']['format']] += 1


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


res = {'ok': not errors, 'n_errors': len(errors), 'errors': errors[:50], 'warnings': warn[:50], 'n_warnings': len(warn),
       'counts': dict(sorted(c.items())),
       'text_tokens_o200k': {'n': len(toks), 'p50': pct(toks, 50), 'p95': pct(toks, 95), 'max': max(toks),
                             'over_16384': sum(t > 16384 for t in toks)},
       'fields_per_item': {'p50': pct(per_item_fields, 50), 'p95': pct(per_item_fields, 95),
                           'max': max(per_item_fields), 'total': sum(per_item_fields)}}
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
