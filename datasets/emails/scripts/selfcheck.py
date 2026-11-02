"""Self-check for datasets/emails: parse, schema, gold, images, harness validate_item, o200k token stats.

Usage: uv run --no-project --with tiktoken python -I scripts/selfcheck.py <dataset_dir>
"""
import json, re, sys
from pathlib import Path

D = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(D.parents[1]))  # extract-bench/ so `harness` imports (our own code)
from harness.common import validate_item  # noqa: E402

import tiktoken  # noqa: E402

enc = tiktoken.get_encoding("o200k_base")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
TYPES = {"text", "number", "money", "date", "time", "datetime", "duration", "list"}
METRICS = {"exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim"}


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def check(path):
    errs, toks, ids = [], [], set()
    n_fields = n_null = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    for ln, l in enumerate(lines, 1):
        try:
            it = json.loads(l)
        except Exception as e:  # noqa: BLE001
            errs.append(f"line {ln}: parse {e}")
            continue
        try:
            validate_item(it)
        except ValueError as e:
            errs.append(str(e))
        if it["id"] in ids:
            errs.append(f"dup id {it['id']}")
        ids.add(it["id"])
        if it["dataset"] != "emails" or it["modality"] != "text" or it["text_source"] != "native":
            errs.append(f"{it['id']}: dataset/modality/text_source")
        if it["metric"] not in METRICS:
            errs.append(f"{it['id']}: metric")
        if not isinstance(it["text"], str) or not it["text"].strip():
            errs.append(f"{it['id']}: empty text")
        for im in it["images"]:
            if not (path.parent / im).exists():
                errs.append(f"{it['id']}: missing image {im}")
        for f in it["fields"]:
            n_fields += 1
            if not NAME.match(f["name"]) or f["type"] not in TYPES or "gold" not in f or "instructions" not in f:
                errs.append(f"{it['id']}: field {f}")
            n_null += f["gold"] is None
        toks.append(len(enc.encode(it["text"], disallowed_special=())))
    return {"file": str(path.relative_to(D)), "lines": len(lines), "errors": errs[:20], "n_errors": len(errs),
            "n_fields": n_fields, "n_null_gold": n_null, "n_images": 0,
            "tokens_o200k": {"p50": pct(toks, .5), "p95": pct(toks, .95), "max": max(toks), "min": min(toks),
                             "total": sum(toks)}}


out = {"items": check(D / "items.jsonl"), "anchor": check(D / "anchor" / "items_all1000.jsonl")}
print(json.dumps(out, indent=1))
sys.exit(1 if out["items"]["n_errors"] or out["anchor"]["n_errors"] else 0)
