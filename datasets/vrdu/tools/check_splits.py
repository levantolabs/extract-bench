"""Compare VRDU mixed-template test sets across train sizes and seeds (read-only on raw/splits)."""
import json, sys, itertools
from pathlib import Path
d = Path(sys.argv[1])
for corpus in ("FARA-lv2-mixed_template", "DeepForm-mixed_template"):
    tests = {}
    for f in sorted(d.glob(corpus + "-*.json")):
        j = json.loads(f.read_text())
        tests[f.name] = j
    keys = list(tests)
    print(corpus, "files", len(keys), "split keys", list(tests[keys[0]].keys()))
    sets = {k: frozenset(tests[k]["test"]) for k in keys}
    uniq = set(sets.values())
    print("  distinct test sets:", len(uniq), "sizes", sorted({len(s) for s in uniq}))
    for k in keys:
        t = tests[k]
        print("  ", k, "train", len(t["train"]), "valid", len(t["valid"]), "test", len(t["test"]),
              "test∩train", len(set(t["test"]) & set(t["train"])), "test∩valid", len(set(t["test"]) & set(t["valid"])))
    ex = next(iter(uniq)); print("  example test ids:", sorted(ex)[:3])
