"""How are nested line-item name tuples ordered in the gold? (read-only)"""
import gzip, json, sys
from collections import Counter
from pathlib import Path
raw = Path(sys.argv[1])
test = set(json.loads((raw / "ad-buy-form/few_shot-splits/DeepForm-mixed_template-train_10-test_300-valid_100-SD_0.json").read_text())["test"])
ordered, sets, per_doc_orders = Counter(), Counter(), Counter()
docs_with_mixed_order = 0
with gzip.open(raw / "ad-buy-form/main/dataset.jsonl.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        if r["filename"] not in test: continue
        by_set = {}
        for name, ments in r["annotations"]:
            if not isinstance(name, list): continue
            ordered[tuple(name)] += 1; sets[frozenset(name)] += 1
            by_set.setdefault(frozenset(name), set()).add(tuple(name))
            assert len(ments) == 1, len(ments)
        if any(len(v) > 1 for v in by_set.values()): docs_with_mixed_order += 1
print("distinct ordered tuples:", len(ordered), " distinct sets:", len(sets))
print("docs where the same sub-entity set appears in >1 order:", docs_with_mixed_order)
for t, c in ordered.most_common(12): print(c, t)
