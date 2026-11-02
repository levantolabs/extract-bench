"""`extract-bench validate DIR`: check a submission before opening a pull request.

DIR has the layout of results/<version>/<system>/:
  metadata.json            who and what was run (keys below)
  preds/<dataset>.jsonl    one line per item: {"id", "dataset", "preds": {field: {"value": ..., "conf": ...}}}
                           or {"id", "dataset", "error": "..."} for a read that failed

Checks: required metadata keys; every line parses and names a known item; preds only name fields the item has;
coverage of the version's split (every item, both halves); then scores the submission on the eval half per dataset
(the same scorer as `extract-bench report`). Exit code 1 on any error; coverage gaps are errors unless --partial.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from .common import BENCH_VERSION, load_items, load_split
from .report import DATASET_ORDER, IMAGE_SETS, RECEIPTS, field_scores

REQUIRED = {"system": str, "display_name": str, "org": str, "model": str, "input": str, "run_date": str,
            "open_weights": bool, "benchmark_version": str}
INPUTS = ("text", "image", "image+text")


def validate(d: Path, version: str, partial: bool) -> tuple[list[str], list[str], dict]:
    errors, warnings, summary = [], [], {}
    mf = d / "metadata.json"
    if not mf.exists():
        return [f"{mf}: missing"], warnings, summary
    try:
        meta = json.loads(mf.read_text())
    except json.JSONDecodeError as e:
        return [f"{mf}: not JSON ({e})"], warnings, summary
    for k, t in REQUIRED.items():
        if k not in meta:
            errors.append(f"metadata.json: missing key {k!r}")
        elif not isinstance(meta[k], t):
            errors.append(f"metadata.json: {k!r} should be {t.__name__}")
    if meta.get("input") not in INPUTS:
        errors.append(f"metadata.json: input must be one of {INPUTS}")
    if meta.get("benchmark_version") not in (None, version):
        errors.append(f"metadata.json: benchmark_version {meta.get('benchmark_version')!r} != {version!r}")
    split = load_split(version)
    wanted = [ds for ds in DATASET_ORDER if ds in split and (meta.get("input") == "text" or ds in IMAGE_SETS)]
    for ds in wanted:
        items = {it["id"]: it for it in load_items(ds)}
        f = d / "preds" / f"{ds}.jsonl"
        ids = split[ds]["eval"] + split[ds]["fit"]
        if not f.exists():
            (warnings if partial else errors).append(f"preds/{ds}.jsonl: missing ({len(ids)} items expected)")
            continue
        recs = {}
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"preds/{ds}.jsonl:{n}: not JSON ({e})"); continue
            iid = r.get("id")
            if iid not in items:
                errors.append(f"preds/{ds}.jsonl:{n}: unknown item id {iid!r}"); continue
            if r.get("dataset") != ds:
                errors.append(f"preds/{ds}.jsonl:{n}: dataset {r.get('dataset')!r} != {ds!r}")
            if iid in recs:
                errors.append(f"preds/{ds}.jsonl:{n}: duplicate id {iid}")
            if not r.get("error"):
                p = r.get("preds")
                if not isinstance(p, dict):
                    errors.append(f"preds/{ds}.jsonl:{n}: 'preds' must be an object (or set 'error')"); continue
                names = {fl["name"] for fl in items[iid]["fields"]}
                extra = set(p) - names
                if extra:
                    errors.append(f"preds/{ds}.jsonl:{n}: unknown fields {sorted(extra)[:5]}")
                for k, v in p.items():
                    if not isinstance(v, dict) or "value" not in v:
                        errors.append(f"preds/{ds}.jsonl:{n}: field {k!r} must be {{\"value\": ..., \"conf\": ...}}")
                        break
            recs[iid] = r
        missing = [i for i in ids if i not in recs]
        if missing:
            (warnings if partial else errors).append(f"preds/{ds}.jsonl: {len(missing)} of {len(ids)} split items missing")
        tot = cor = 0.0; n_err = 0
        for iid in split[ds]["eval"]:
            fs = field_scores(items[iid], recs.get(iid)) if iid in recs else None
            if fs is None:
                n_err += iid in recs; continue
            tot += len(fs); cor += sum(v[0] for v in fs.values())
        summary[ds] = {"lines": len(recs), "eval_items": len(split[ds]["eval"]), "errored_eval": n_err,
                       "eval_field_acc": (100 * cor / tot) if tot else None}
    return errors, warnings, summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="extract-bench validate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--version", default=BENCH_VERSION)
    ap.add_argument("--partial", action="store_true", help="missing datasets/items are warnings, not errors")
    a = ap.parse_args(argv)
    errors, warnings, summary = validate(a.dir, a.version, a.partial)
    if summary:
        print(f"{'dataset':<17} {'lines':>6} {'eval':>5} {'errored':>8} {'eval acc*':>9}")
        for ds, s in summary.items():
            acc = "–" if s["eval_field_acc"] is None else f"{s['eval_field_acc']:.1f}"
            print(f"{ds:<17} {s['lines']:>6} {s['eval_items']:>5} {s['errored_eval']:>8} {acc:>9}"
                  f"{'  (receipts)' if ds in RECEIPTS else ''}")
        print("* Eval half, scored on this submission's own fields. The leaderboard keeps only the fields that\n"
              "  every system answered, so `extract-bench report` can differ slightly.")
    for w in warnings:
        print("WARNING", w)
    for e in errors[:50]:
        print("ERROR", e)
    if len(errors) > 50:
        print(f"... and {len(errors) - 50} more errors")
    print("OK" if not errors else f"FAILED: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
