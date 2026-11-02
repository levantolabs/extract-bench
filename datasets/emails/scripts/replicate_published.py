"""Re-score the repo's published prediction files with our re-implementation of matcher.py.

Purpose: (1) check our matcher re-implementation reproduces summary_all_json.csv exactly on all 1,000;
(2) give the same models' accuracy on OUR 100-item sample (same items, same predictions: a reference
row computed by us from published outputs, not a number published by the repo).

matcher.py (commit 72b8d5fe) logic, re-implemented without running downloaded code:
  - lower+strip the 8 merge columns in both CSV and JSON, inner-join on them;
  - Accuracy = mean(committee_csv.fillna('none').astype(str).lower() == committee_json.fillna('none').astype(str).lower())
    (note: the Accuracy line lower-cases but does NOT strip; committee_match strips; we report the Accuracy line).
Also reports the alnum-squashed score (harness score.squash) for the same predictions.

Usage: uv run --no-project --with pandas python -I scripts/replicate_published.py <dataset_dir>
"""
import json, re, sys, unicodedata
from pathlib import Path

import pandas as pd

D = Path(sys.argv[1]).resolve()
REPO = D / "raw" / "repo" / "fundraising-emails"
MERGE = ["email", "subject", "year", "month", "day", "hour", "minute", "domain"]
MODELS = [
    "gpt-4o_november_2024_prompt2.json",
    "gpt-4o-mini_november_2024.json",
    "claude37_sonnet_november_2024_prompt2.json",
    "gemini_25_november_2024_prompt2.json",
    "claude35_haiku_november_2024_prompt2.json",
    "gpt_41_mini_november_2024_prompt2.json",
    "claude_haiku_45_november_2024_prompt2.json",
    "gpt_41_nano_november_2024_prompt2.json",
]


def norm_text(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:!?\"'`()[]{}")


def squash(s):
    return re.sub(r"[^0-9a-z]", "", norm_text(s))


def main():
    gold = pd.read_csv(REPO / "training.csv")
    gold["row"] = range(len(gold))
    summary = pd.read_csv(REPO / "summary_all_json.csv").set_index("JSON Filename")
    items = [json.loads(l) for l in (D / "items.jsonl").read_text().splitlines() if l.strip()]
    sample_rows = {it["meta"]["csv_row"] for it in items}
    out = {"n_sample": len(sample_rows), "models": {}}
    g = gold.copy()
    for c in MERGE:
        g[c] = g[c].astype(str).str.strip().str.lower()
    for fn in MODELS:
        p = REPO / fn
        if not p.exists():
            out["models"][fn] = "missing"
            continue
        pj = pd.DataFrame(json.loads(p.read_text()))
        for c in MERGE:
            pj[c] = pj[c].astype(str).str.strip().str.lower()
        m = pd.merge(g, pj, on=MERGE, how="inner", suffixes=("_csv", "_json"))
        yt = m["committee_csv"].fillna("none").astype(str).str.lower()
        yp = m["committee_json"].fillna("none").astype(str).str.lower()
        hit = (yt == yp)
        sq = [squash(a) == squash(b) and squash(a) != "" for a, b in zip(m["committee_csv"], m["committee_json"].fillna(""))]
        m["hit"], m["sq"] = hit.values, sq
        ins = m[m["row"].isin(sample_rows)]
        pub = summary.loc[fn] if fn in summary.index else None
        out["models"][fn] = {
            "all_merged_rows": int(len(m)),
            "all_acc_ours": round(float(hit.mean()), 4),
            "all_acc_published": None if pub is None else float(pub["Accuracy"]),
            "all_records_published": None if pub is None else int(pub["Total Records"]),
            "reproduces_published": None if pub is None else bool(abs(float(hit.mean()) - float(pub["Accuracy"])) < 1e-9 and len(m) == int(pub["Total Records"])),
            "all_acc_squashed": round(float(m["sq"].mean()), 4),
            "sample_rows_merged": int(len(ins)),
            "sample_rows_distinct": int(ins["row"].nunique()),
            "sample_acc_repo_matcher": round(float(ins["hit"].mean()), 4) if len(ins) else None,
            "sample_acc_squashed": round(float(ins["sq"].mean()), 4) if len(ins) else None,
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
