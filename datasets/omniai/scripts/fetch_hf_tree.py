"""Save the HF file tree (size + LFS sha256) of getomni-ai/ocr-benchmark test/images at a pinned revision.
Usage: python -I fetch_hf_tree.py <out.json>"""
import json, sys, urllib.request
REV = '4ed0d95271ca00107726230f7a0944ed9e90d897'
url = f'https://huggingface.co/api/datasets/getomni-ai/ocr-benchmark/tree/{REV}/test/images?expand=true'
out = {}
while url:
    req = urllib.request.Request(url, headers={'User-Agent': 'extract-bench-builder/0.1'})
    with urllib.request.urlopen(req, timeout=120) as r:
        page = json.load(r)
        link = r.headers.get('Link') or ''
    for e in page:
        if e.get('type') == 'file':
            out[e['path']] = {'size': e.get('size'), 'lfs_sha256': (e.get('lfs') or {}).get('oid'), 'git_oid': e.get('oid')}
    url = None
    for part in link.split(','):
        if 'rel="next"' in part:
            url = part[part.find('<') + 1:part.find('>')]
json.dump({'revision': REV, 'files': out}, open(sys.argv[1], 'w'), indent=0, sort_keys=True)
print('files', len(out), 'with lfs sha256', sum(1 for v in out.values() if v['lfs_sha256']))
