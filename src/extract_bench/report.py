"""`extract-bench report`: score published predictions and print / write the leaderboard.

Scores every system in results/<version>/ from its predictions (the same `score.score_item` as `run score`), on the
items in splits/<version>.json, and writes results/<version>/board.json + board.md.

Method (v0.1):
- Items: the version's split; `--split eval` (default; the half no system was tuned on) or `all`.
- Fields: headline fields only, intersected across the systems in a comparison, so every system answers the same
  fields of the same documents. A read that errored drops that item from the comparison; output that did not parse
  (`unparsed`) is kept and every field scores 0.
- Score: rule (a) field accuracy (a correct "not in the document" is correct), micro-averaged per dataset; the
  headline is the mean of datasets, receipts (SROIE, CORD) excluded.
- Uncertainty: item-paired bootstrap per dataset (2,000 resamples, seed 20) for each system minus the reference;
  for the mean of datasets, items are resampled within each dataset (stratified), same count and seed.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

from . import score as S
from .common import BENCH_VERSION, RESULTS, load_items, load_split

RECEIPTS = {"sroie", "cord"}
IMAGE_SETS = {"fatura", "omniai", "docvqa", "vrdu", "realdoc", "sroie", "cord"}
DATASET_ORDER = ["fatura", "omniai", "docvqa", "vrdu", "realdoc", "kleister_nda", "kleister_charity", "emails", "tatqa",
                 "squad2", "sroie", "cord"]
# board name -> (systems, reference, dataset filter)
BOARDS = {
    "headline": (["sage2", "clairic"], "sage2", None),
    "text": (["sage2", "clairic", "nuextract3", "schematron3b", "gliner25"], "sage2", None),
    "image": (["sage2", "clairic_img"], "sage2", IMAGE_SETS),
}
N_BOOT, SEED = 2000, 20


def load_system(results: Path, system: str) -> tuple[dict, dict]:
    """-> (metadata, {(dataset, id): pred record})"""
    d = results / system
    meta = json.loads((d / "metadata.json").read_text())
    recs = {}
    for f in sorted((d / "preds").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[(r["dataset"], r["id"])] = r
    return meta, recs


def field_scores(item: dict, rec: dict | None) -> dict | None:
    """{field: (score, gold_null)} over headline fields; None if the system has no usable read of this item."""
    if rec is None or rec.get("error"):
        return None
    sc = S.score_item(item, rec.get("preds") or {})
    out = {}
    for x in sc["fields"]:
        if not x.get("headline", True):
            continue
        if x["scored"]:
            out[x["name"]] = (x["score_a"], x["gold_null"])
        elif x.get("reason") == "missing":        # no answer for a field the system can express: wrong
            out[x["name"]] = (0.0, x.get("gold_null", False))
    return out


def _acc(rows, j):
    return 100 * sum(r[j] for it in rows for r in it) / sum(len(it) for it in rows)


def compare(systems, ref, items, split_ids, recs):
    rows, absent = [], {s: [0, 0] for s in systems}
    for iid in split_ids:
        it = items.get(iid)
        if it is None:
            continue
        fs = [field_scores(it, recs[s].get((it["dataset"], iid))) for s in systems]
        if any(f is None for f in fs):
            continue
        keys = sorted(set.intersection(*(set(f) for f in fs)))
        if not keys:
            continue
        rows.append([tuple(f[k][0] for f in fs) for k in keys])
        for s, f in zip(systems, fs):
            for k in keys:
                if f[k][1]:
                    absent[s][0] += f[k][0]; absent[s][1] += 1
    if not rows:
        return None
    rng = random.Random(SEED)
    res = {"items": len(rows), "fields": sum(len(r) for r in rows),
           "acc": {s: _acc(rows, j) for j, s in enumerate(systems)},
           "absent_fields": absent[systems[0]][1],
           "absent_correct": {s: (100 * absent[s][0] / absent[s][1] if absent[s][1] else None) for s in systems},
           "diff_vs_ref": {}, "_rows": rows}
    r0 = systems.index(ref)
    boots = [[rows[rng.randrange(len(rows))] for _ in rows] for _ in range(N_BOOT)]
    lo_i, hi_i = int(0.025 * N_BOOT), int(0.975 * N_BOOT) - 1
    for j, s in enumerate(systems):
        if s == ref:
            continue
        d = sorted(_acc(z, j) - _acc(z, r0) for z in boots)
        res["diff_vs_ref"][s] = {"diff": res["acc"][s] - res["acc"][ref], "ci95": [d[lo_i], d[hi_i]],
                                 "sig": d[lo_i] > 0 or d[hi_i] < 0}
    return res


def board(results: Path, version: str, split: str) -> dict:
    sp = load_split(version)
    names = sorted({s for v in BOARDS.values() for s in v[0]})
    meta, recs = {}, {}
    for s in names:
        if (results / s / "metadata.json").exists():
            meta[s], recs[s] = load_system(results, s)
    out = {"version": version, "split": split, "systems": meta, "boards": {}}
    for bname, (systems, ref, only) in BOARDS.items():
        if not all(s in recs for s in systems):
            continue
        B = out["boards"][bname] = {"systems": systems, "reference": ref, "datasets": {}}
        for ds in DATASET_ORDER:
            if ds not in sp or (only and ds not in only):
                continue
            ids = sp[ds]["eval"] if split == "eval" else sp[ds]["eval"] + sp[ds]["fit"]
            items = {it["id"]: it for it in load_items(ds)}
            r = compare(systems, ref, items, sorted(ids), recs)
            if r:
                B["datasets"][ds] = r
        sets = [d for d in B["datasets"] if d not in RECEIPTS]
        B["mean_sets"] = sets
        B["mean"] = {s: sum(B["datasets"][d]["acc"][s] for d in sets) / len(sets) for s in systems}
        B["mean_all_sets"] = {s: sum(r["acc"][s] for r in B["datasets"].values()) / len(B["datasets"]) for s in systems}
        ab = [d for d in sets if B["datasets"][d]["absent_fields"]]
        B["absent_correct_mean"] = {s: sum(B["datasets"][d]["absent_correct"][s] for d in ab) / len(ab) for s in systems}
        B["absent_sets"] = ab
        # stratified bootstrap of the mean of datasets: resample items within each dataset
        rng = random.Random(SEED); r0 = systems.index(ref); boots = []
        for _ in range(N_BOOT):
            per = []
            for d in sets:
                rows = B["datasets"][d]["_rows"]; z = [rows[rng.randrange(len(rows))] for _ in rows]
                per.append([_acc(z, j) for j in range(len(systems))])
            boots.append([sum(x[j] for x in per) / len(sets) for j in range(len(systems))])
        lo_i, hi_i = int(0.025 * N_BOOT), int(0.975 * N_BOOT) - 1
        B["mean_diff_vs_ref"] = {}
        for j, s in enumerate(systems):
            if s != ref:
                dd = sorted(b[j] - b[r0] for b in boots)
                B["mean_diff_vs_ref"][s] = {"diff": B["mean"][s] - B["mean"][ref], "ci95": [dd[lo_i], dd[hi_i]],
                                            "sig": dd[lo_i] > 0 or dd[hi_i] < 0}
        for d in B["datasets"].values():
            d.pop("_rows", None)
    return out


def _f(v):
    return "–" if v is None else f"{v:.1f}"


def _sg(v):
    return f"{v:+.1f}"


def to_markdown(b: dict) -> str:
    md = [f"# Extract Bench {b['version']} board ({b['split']} split)", "",
          "`extract-bench report` generates this file. Each cell is field accuracy in percent under rule (a), on "
          "headline fields. A ✱ means the 95% bootstrap interval of the difference from the reference excludes 0. "
          "The means leave out the receipt sets.", ""]
    for bname, B in b["boards"].items():
        names = [b["systems"][s]["display_name"] for s in B["systems"]]
        ref = b["systems"][B["reference"]]["display_name"]
        md += [f"## {bname}", "", "| Dataset | Items | Fields | " + " | ".join(names) + f" | Δ vs {ref} [95% CI] |",
               "|---|---:|---:|" + "---:|" * len(names) + "---|"]
        for ds, r in B["datasets"].items():
            best = max(r["acc"].values())
            cells = [f"**{_f(r['acc'][s])}**" if r["acc"][s] == best else _f(r["acc"][s]) for s in B["systems"]]
            diffs = "; ".join(f"{b['systems'][s]['display_name']} {_sg(v['diff'])} [{_sg(v['ci95'][0])}, {_sg(v['ci95'][1])}]"
                              f"{' ✱' if v['sig'] else ''}" for s, v in r["diff_vs_ref"].items())
            md.append(f"| {ds}{' (receipts)' if ds in RECEIPTS else ''} | {r['items']} | {r['fields']:,} | "
                      + " | ".join(cells) + f" | {diffs} |")
        md.append(f"| **Mean of {len(B['mean_sets'])} sets** | | | " + " | ".join(f"**{_f(B['mean'][s])}**" for s in B["systems"])
                  + " | " + "; ".join(f"{b['systems'][s]['display_name']} {_sg(v['diff'])} [{_sg(v['ci95'][0])}, {_sg(v['ci95'][1])}]"
                                      f"{' ✱' if v['sig'] else ''}" for s, v in B["mean_diff_vs_ref"].items()) + " |")
        md.append(f"| \"Not in the document\" correct ({len(B['absent_sets'])} sets) | | | "
                  + " | ".join(_f(B["absent_correct_mean"][s]) for s in B["systems"]) + " | |")
        md.append("")
    return "\n".join(md) + "\n"


def to_terminal(b: dict, board_name: str = "text") -> str:
    B = b["boards"][board_name]
    rows = sorted(B["systems"], key=lambda s: -B["mean"][s])
    w = max(len(b["systems"][s]["display_name"]) for s in rows)
    out = [f"Extract Bench {b['version']} · {board_name} board · {b['split']} split · mean of {len(B['mean_sets'])} datasets", ""]
    out.append(f"{'#':>2}  {'system':<{w}}  {'acc':>5}  {'Δ vs ' + b['systems'][B['reference']]['display_name']:>22}  "
               f"{'not-in-doc':>10}")
    for i, s in enumerate(rows, 1):
        d = B["mean_diff_vs_ref"].get(s)
        dd = f"{_sg(d['diff'])} [{_sg(d['ci95'][0])}, {_sg(d['ci95'][1])}]{'*' if d['sig'] else ' '}" if d else "reference "
        bar = "█" * round(B["mean"][s] / 4)
        out.append(f"{i:>2}  {b['systems'][s]['display_name']:<{w}}  {B['mean'][s]:5.1f}  {dd:>22}  "
                   f"{B['absent_correct_mean'][s]:9.1f}%  {bar}")
    out += ["", "per dataset:", f"{'dataset':<17} {'n':>4} " + " ".join(f"{b['systems'][s]['display_name'][:12]:>12}" for s in B["systems"])]
    for ds, r in B["datasets"].items():
        out.append(f"{ds:<17} {r['items']:>4} " + " ".join(f"{r['acc'][s]:12.1f}" for s in B["systems"]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="extract-bench report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=BENCH_VERSION)
    ap.add_argument("--results", type=Path, default=None, help="default: results/<version>/")
    ap.add_argument("--split", choices=("eval", "all"), default="eval")
    ap.add_argument("--board", default="text", help="board to print (headline | text | image)")
    ap.add_argument("--no-write", action="store_true", help="print only; don't write board.json / board.md")
    a = ap.parse_args(argv)
    results = a.results or RESULTS / a.version
    b = board(results, a.version, a.split)
    print(to_terminal(b, a.board))
    if not a.no_write:
        suffix = "" if a.split == "eval" else f".{a.split}"
        (results / f"board{suffix}.json").write_text(json.dumps(b, indent=1) + "\n")
        (results / f"board{suffix}.md").write_text(to_markdown(b))
        print(f"\nwrote {results / f'board{suffix}.json'} and board{suffix}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
