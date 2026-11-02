"""Parallel prefetch of the source PDFs build.py will walk (HF downloads are ~12 s each, latency-bound).

Replicates build.py's seeded order (same functions, imported from build.py) and downloads the PDFs of the first
quota + MARGIN candidates per domain with 8 threads. build.py stays the source of truth: it re-verifies sha256
against manifest.json and page counts, and fetches anything still missing with --fetch.

Usage (from extract-bench/): /opt/homebrew/bin/uv run --no-project python -I datasets/realdoc/scripts/prefetch.py
"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS))
import build as B  # noqa: E402  (our own build.py; main() is guarded)

MARGIN = 15
bank = json.loads((B.RAW / "hf" / "qa_bank.json").read_text())["items"]
eligible = {}
for it in bank:
    if B.field_plan(it)["fields"]:
        eligible.setdefault(it["domain"], []).append(it)
tot = sum(len(v) for v in eligible.values())
raw_q = {d: B.N * len(v) / tot for d, v in eligible.items()}
quota = {d: int(x) for d, x in raw_q.items()}
for d in sorted(raw_q, key=lambda d: (-(raw_q[d] - quota[d]), d))[: B.N - sum(quota.values())]:
    quota[d] += 1


def qkey(it):
    m = re.search(r"_(\d+)$", it["source_file"]), re.search(r"_q(\d+)$", it["question_id"])
    return (int(m[0].group(1)) if m[0] else 0, int(m[1].group(1)) if m[1] else 0, it["question_id"])


rng = random.Random(B.SEED)
stems = []
for d in sorted(eligible):
    order = sorted(eligible[d], key=qkey)
    rng.shuffle(order)
    for it in order[: quota[d] + MARGIN]:
        if it["source_file"] not in stems:
            stems.append(it["source_file"])
B.DOCS.mkdir(parents=True, exist_ok=True)
todo = [s for s in stems if not (B.DOCS / f"{s}.pdf").exists()]
print("quota", quota, "candidate docs", len(stems), "to fetch", len(todo), flush=True)
with ThreadPoolExecutor(8) as ex:
    for i, p in enumerate(ex.map(B.fetch_pdf, todo), 1):
        if i % 20 == 0:
            print(i, p.name, flush=True)
print("done", len(list(B.DOCS.glob("*.pdf"))), "pdfs")
