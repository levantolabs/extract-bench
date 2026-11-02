"""Official TAT-QA scoring (scale-aware EM / numeracy-focused F1) for an extract-bench run, on the items an arm answered.

Usage (from ..):
  U="/opt/homebrew/bin/uv run --no-project --with numpy --with pandas --with scipy python -I"
  $U extract-bench/datasets/tatqa/comparable/run_official.py --selftest
  $U extract-bench/datasets/tatqa/comparable/run_official.py --run RUN --arm clairic_text [--scale both]

Prediction -> official format {question_uid: [answer, scale]}:
  * value: number -> str(number); list -> [str(x), ...]; text -> str; null/missing -> None (scores 0, as tatqa_eval).
  * scale: no arm has a scale output channel (Clairic number fields return a bare number, and the item sends no
    `unit`). The item instruction asks for the report's units, which is what TAT-QA's `scale` annotates, so:
      oracle: pred_scale = gold scale. Upper bound: credits the arm for naming the scale the instruction implied.
      none:   pred_scale = "".         Lower bound: scaled answers only score if the arm expanded them (1,234
              thousand -> 1234000) or gave a percent as a fraction (official add_percent_pred rule).
    Published systems (GPT-4 in arXiv 2401.13223, the leaderboard) predicted the scale themselves, so neither bound
    is strictly comparable: print both, next to the published row, labelled.
  * gold = comparable/tatqa_gold_sample.json restricted to the uids this arm has a pred record for (e.g. Sage runs
    without --final skip the sealed holdout). n is printed. Missing records are NOT silently counted as 0.
Sage 2 is n/a on computed items (arithmetic, count) and has no list type (multi-span): report Sage on span only.
official/*.py are byte-identical copies of NExTplusplus/TAT-QA @ 870accc4 (sha256 in CARD.md), unmodified.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent
EXB = DS.parent.parent
sys.path.insert(0, str(HERE / "official"))
from tatqa_metric import TaTQAEmAndF1  # noqa: E402  (vendored official scorer)


def to_official(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    if isinstance(v, list):
        return [to_official(x) for x in v if x is not None] or None
    if isinstance(v, dict):
        for k in ("amount", "value", "number"):
            if k in v:
                return to_official(v[k])
        return None
    s = str(v).strip()
    return s or None


def score(gold_docs, preds: dict, scale_mode: str) -> dict:
    m = TaTQAEmAndF1()
    n = 0
    for doc in gold_docs:
        for q in doc["questions"]:
            if q["uid"] not in preds:
                continue
            n += 1
            ans = preds[q["uid"]]
            sc = q["scale"] if scale_mode == "oracle" else ""
            m(ground_truth=q, prediction=ans, pred_scale=sc)
    em, f1, scale_acc, _ = m.get_overall_metric()
    by = defaultdict(lambda: {"n": 0, "em": 0.0, "f1": 0.0})
    for d in m.get_raw():
        b = by[d["answer_type"]]
        b["n"] += 1; b["em"] += d["em"]; b["f1"] += d["f1"]
    by_type = {t: {"n": b["n"], "em": round(100 * b["em"] / b["n"], 2), "f1": round(100 * b["f1"] / b["n"], 2)}
               for t, b in sorted(by.items())}
    return {"scale_mode": scale_mode, "n": n, "em": round(100 * em, 2), "f1": round(100 * f1, 2),
            "scale_acc": round(100 * scale_acc, 2), "by_answer_type": by_type}


def load_gold():
    return json.loads((HERE / "tatqa_gold_sample.json").read_text())


def selftest():
    gold = load_gold()
    items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]
    # through to_official like real preds: the official `if not prediction` check scores a raw numeric 0 as wrong
    # (1 sampled gold is 0 million); as the string "0" it is creditable. The official sample_prediction.json passes
    # raw ints/floats, so published systems may have hit this quirk; it affects <=1/150 items here.
    gold_pred = {it["id"]: to_official(it["meta"]["official_answer"]) for it in items}
    res_raw0 = score(gold, {it["id"]: it["meta"]["official_answer"] for it in items}, "oracle")
    alt = {}
    for it in items:   # the 2nd acceptable gold of computed fields (expanded value / percent fraction)
        g = it["fields"][0]["gold"]
        alt[it["id"]] = to_official(g[1] if it["meta"]["group"] == "computed" and len(g) > 1 else g[0]
                                    if it["meta"]["group"] == "computed" else it["meta"]["official_answer"])
    wrong = {it["id"]: "zzz-wrong" for it in items}
    res = {"gold_as_pred": {m: score(gold, gold_pred, m) for m in ("oracle", "none")},
           "expanded_or_fraction_as_pred": {m: score(gold, alt, m) for m in ("oracle", "none")},
           "wrong_text_as_pred": score(gold, wrong, "oracle"),
           "gold_raw_numbers_as_pred_oracle (official zero quirk)": res_raw0}
    ok = res["gold_as_pred"]["oracle"]["em"] == 100.0 and res["wrong_text_as_pred"]["em"] == 0.0
    res["pass"] = ok
    (HERE / "selftest.out.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--arm")
    ap.add_argument("--scale", choices=("oracle", "none", "both"), default="both")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.run and a.arm):
        ap.error("--run and --arm (or --selftest)")
    d = EXB / "runs" / a.run / "preds" / a.arm.replace("/", "_").replace(":", "~") / "tatqa"
    if not d.is_dir():
        sys.exit(f"no predictions at {d}")
    preds, errors = {}, 0
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("error"):
            errors += 1
            continue
        v = ((r.get("preds") or {}).get("answer") or {})
        if v.get("unsupported"):
            continue
        preds[r["id"]] = to_official(v.get("value"))
    modes = ("oracle", "none") if a.scale == "both" else (a.scale,)
    out = {"run": a.run, "arm": a.arm, "n_pred_records": len(preds), "n_error_records": errors,
           "scores": [score(load_gold(), preds, m) for m in modes]}
    print(json.dumps(out, indent=1))
    (d.parent.parent.parent / f"tatqa_official_{a.arm.replace('/', '_')}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
