"""Which SQuAD 2.0 dev contexts appear (any source) in the tasksource extraction validation/test splits?

Reads raw/tasksource/extraction_{validation,test}-00000-of-00001.parquet (main 8b1afcc6) with pyarrow.
Whitespace is collapsed in both the states and the dev contexts. For each dev context, three 80-char probes
(start, middle, end) are searched with Aho-Corasick over every state; a context is a hit if ANY probe occurs in
ANY row. Writes derived/xsource_hits.json: hit_contexts ("Title#paragraph_index"), hits per context
(split, source, row id, probe), and counts by source.
Usage: uv run --no-project --with pyarrow --with pyahocorasick python -I xsource_probe.py <squad2_dir>
"""
import json, re, sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
import pyarrow.parquet as pq

d = Path(sys.argv[1])
ws = lambda s: re.sub(r"\s+", " ", s).strip()
dev = json.loads((d / "raw/official/dev-v2.0.json").read_text())
probes = {}   # probe -> set of ctx keys
for a in dev["data"]:
    for pi, p in enumerate(a["paragraphs"]):
        c = ws(p["context"])
        key = f"{a['title']}#{pi}"
        L = 80 if len(c) >= 80 else len(c)
        for pos in sorted({0, max(0, len(c) // 2 - L // 2), max(0, len(c) - L)}):
            probes.setdefault(c[pos:pos + L], set()).add(key)
A = ahocorasick.Automaton()
for i, pr in enumerate(probes):
    A.add_word(pr, pr)
A.make_automaton()

hits = defaultdict(list)
by_source = defaultdict(set)
rows_scanned = {}
for split in ("validation", "test"):
    t = pq.read_table(d / f"raw/tasksource/extraction_{split}-00000-of-00001.parquet", columns=["source", "id", "state"])
    src, ids, states = t.column("source").to_pylist(), t.column("id").to_pylist(), t.column("state").to_pylist()
    rows_scanned[split] = len(states)
    starts, buf, off = [], [], 0
    for s in states:
        s = ws(s or "")
        starts.append(off)
        buf.append(s)
        off += len(s) + 1
    big = "\x00".join(buf)
    seen = set()
    for end, pr in A.iter(big):
        r = bisect_right(starts, end - len(pr) + 1) - 1
        for key in probes[pr]:
            if (key, split, r) in seen:
                continue
            seen.add((key, split, r))
            hits[key].append({"split": split, "source": src[r], "row_id": ids[r]})
            by_source[f"{split}:{src[r]}"].add(key)
    del big, buf
out = {
    "tasksource_commit": "8b1afcc6ad12554b347134a91b5f0dc6ae2361db",
    "rows_scanned": rows_scanned,
    "dev_contexts": sum(len(a["paragraphs"]) for a in dev["data"]),
    "n_probes": len(probes),
    "hit_contexts": sorted(hits),
    "n_hit_contexts": len(hits),
    "contexts_by_split_source": {k: len(v) for k, v in sorted(by_source.items())},
    "hit_contexts_not_in_squad_v2_rows": sorted(k for k, v in hits.items() if all(h["source"] != "squad-v2" for h in v)),
    "hits": hits,
}
out["n_hit_contexts_not_in_squad_v2_rows"] = len(out["hit_contexts_not_in_squad_v2_rows"])
(d / "derived").mkdir(exist_ok=True)
(d / "derived/xsource_hits.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k not in ("hits", "hit_contexts", "hit_contexts_not_in_squad_v2_rows")}, indent=1))
