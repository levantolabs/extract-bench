"""Self-check of datasets/tatqa/items.jsonl against the common item format and the harness's own validator/scorers.

Run: /opt/homebrew/bin/uv run --no-project --with tiktoken python -I extract-bench/datasets/tatqa/scripts/selfcheck.py
Writes scripts/selfcheck.out.json. Exit 1 on any failure.
"""
import json, math, re, sys
from collections import Counter
from pathlib import Path

DS = Path(__file__).resolve().parent.parent
EXB = DS.parent.parent
sys.path.insert(0, str(EXB))
from harness import common, score  # noqa: E402

import tiktoken  # noqa: E402

enc = tiktoken.get_encoding("o200k_base")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
METRICS = {"exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim"}
fails, lines = [], (DS / "items.jsonl").read_text().splitlines()
items = []
for i, l in enumerate(lines, 1):
    try:
        items.append(json.loads(l))
    except json.JSONDecodeError as e:
        fails.append(f"line {i}: {e}")
ids = Counter(it["id"] for it in items)
fails += [f"duplicate id {k}" for k, v in ids.items() if v > 1]
tok, req_tok, gold_self, perturb = [], [], [], []
for it in items:
    try:
        common.validate_item(it)
    except ValueError as e:
        fails.append(str(e))
    if it["dataset"] != "tatqa" or it["modality"] != "text" or it["text_source"] != "native" or it["images"]:
        fails.append(f"{it['id']}: dataset/modality/text_source/images")
    if not isinstance(it["text"], str) or not it["text"]:
        fails.append(f"{it['id']}: empty text")
    if it["metric"] not in METRICS:
        fails.append(f"{it['id']}: metric")
    for f in it["fields"]:
        if not NAME.match(f["name"]) or f["type"] not in TYPES or "gold" not in f or "instructions" not in f:
            fails.append(f"{it['id']}: field {f}")
        if f["gold"] is None:
            continue                       # explicit null gold (none expected in TAT-QA; counted below)
        golds = [f["gold"]] if f["type"] == "list" else (f["gold"] if isinstance(f["gold"], list) else [f["gold"]])
        for g in golds:                    # every acceptable gold must score 1.0 under the harness's own scorer
            s = score.score_value(g, f["gold"], f["type"], it["metric"], f)
            gold_self.append(s)
            if s != 1.0:
                fails.append(f"{it['id']}: gold {g!r} self-scores {s}")
        if f["type"] == "number":          # a 1% / +1 perturbation must score 0 (tolerance sanity)
            g0 = float(golds[0])
            bad = g0 * 1.01 if abs(g0) >= 1 else g0 + 1
            s = score.score_value(bad, f["gold"], f["type"], it["metric"], f)
            perturb.append(s)
            if s != 0.0:
                fails.append(f"{it['id']}: perturbed {bad} scores {s}")
    n = len(enc.encode(it["text"], disallowed_special=()))
    tok.append(n)
    req_tok.append(n + sum(len(enc.encode(f["instructions"])) for f in it["fields"]))
    if it["meta"].get("o200k_tokens") != n:
        fails.append(f"{it['id']}: meta.o200k_tokens mismatch")
    if req_tok[-1] > 16384 - 256:
        fails.append(f"{it['id']}: over Sage 2 16,384 cap")
    if n > 8192 - 512:
        fails.append(f"{it['id']}: state near Clairic 8,192-token pass")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))]


out = {
    "n_lines": len(lines), "n_items": len(items), "n_fields": sum(len(it["fields"]) for it in items),
    "n_null_gold_fields": sum(f["gold"] is None for it in items for f in it["fields"]),
    "field_types": dict(Counter(f["type"] for it in items for f in it["fields"])),
    "answer_types": dict(Counter(it["meta"]["answer_type"] for it in items)),
    "groups": dict(Counter(it["meta"]["group"] for it in items)),
    "sage2_status": dict(Counter(it["meta"]["sage2"] for it in items)),
    "holdout": sum(it["meta"]["holdout"] for it in items),
    "distinct_contexts": len({it["meta"]["doc_id"] for it in items}),
    "images_checked": 0, "image_paths_missing": 0,
    "o200k_text": {"p50": pct(tok, .5), "p95": pct(tok, .95), "max": max(tok), "min": min(tok),
                   "mean": round(sum(tok) / len(tok), 1)},
    "o200k_text_plus_instructions": {"p50": pct(req_tok, .5), "p95": pct(req_tok, .95), "max": max(req_tok)},
    "gold_self_score_min": min(gold_self), "n_gold_values_checked": len(gold_self),
    "perturbed_number_max_score": max(perturb) if perturb else None,
    "harness_validate_item": "ok" if not any("validate" in x for x in fails) else "see failures",
    "failures": fails[:50], "n_failures": len(fails), "pass": not fails,
}
(DS / "scripts" / "selfcheck.out.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
sys.exit(0 if not fails else 1)
