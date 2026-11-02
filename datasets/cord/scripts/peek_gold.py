"""Peek at CORD gt_parse structure for the planned scalar keys (reads a jsonl of rows with ground_truth)."""
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
KEYS = {'sub_total': ['subtotal_price', 'discount_price', 'service_price', 'tax_price', 'othersvc_price', 'etc'],
        'total': ['total_price', 'cashprice', 'changeprice', 'creditcardprice', 'menuqty_cnt', 'emoneyprice', 'menutype_cnt', 'total_etc']}
vals = collections.defaultdict(list); types = collections.Counter(); topkeys = collections.Counter(); gtkeys = collections.Counter()
for r in rows:
    gt = json.loads(r['ground_truth'])
    gtkeys.update(gt.keys())
    p = gt['gt_parse']; topkeys.update(p.keys())
    for sec, ks in KEYS.items():
        v = p.get(sec)
        types[(sec, type(v).__name__)] += 1
        secs = v if isinstance(v, list) else ([v] if v else [])
        for d in secs:
            for k, x in d.items():
                vals[sec + '.' + k].append(x)
print('gt keys', gtkeys); print('top', topkeys); print('section types', types)
for k, v in sorted(vals.items()):
    print(k, len(v), [x for x in v][:60])
