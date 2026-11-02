"""Score committee predictions on the emails set three ways (stdlib only).

  repo_matcher : the repo's matcher.py Accuracy line (commit 72b8d5fe): fillna('none') -> str -> lower(), exact.
                 It does NOT strip on the Accuracy line, and an arm's null becomes 'none' (always wrong here: 0 null golds).
                 ONLY this number may sit next to the published leaderboard (and only for prompt2-style runs).
  exact_norm   : harness/score.py norm_text (NFKC, casefold, whitespace collapsed, outer punctuation stripped). Our view.
  squashed     : alphanumerics only (harness/score.py squash). Fair to span extractors: the body has line-join
                 artefacts ("Harris forPresident"), so 71/1,000 golds are not verbatim in the body; 0/1,000 after squash.

Usage:
  python3 -I score_emails.py ITEMS.jsonl PREDS.jsonl
PREDS.jsonl lines: {"id": "emails-r0007", "committee": "End Citizens United Nonfederal" | null}
Missing ids count as null (wrong). Prints JSON with per-view accuracy and 95% bootstrap CIs (seed 20, 10,000 resamples).
"""
import json, random, re, sys, unicodedata


def norm_text(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:!?\"'`()[]{}")


def squash(s):
    return re.sub(r"[^0-9a-z]", "", norm_text(s))


def repo_norm(v):
    return "none" if v is None else str(v).lower()


def ci(xs, seed=20, b=10000):
    if not xs:
        return [None, None]
    r = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[r.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [round(ms[int(0.025 * b)], 4), round(ms[int(0.975 * b) - 1], 4)]


def main(items_path, preds_path):
    items = [json.loads(l) for l in open(items_path, encoding="utf-8") if l.strip()]
    preds = {}
    for l in open(preds_path, encoding="utf-8"):
        if l.strip():
            p = json.loads(l)
            preds[p["id"]] = p.get("committee")
    views = {"repo_matcher": [], "exact_norm": [], "squashed": []}
    reach = {"verbatim": [], "non_verbatim": []}
    for it in items:
        g = it["fields"][0]["gold"]
        p = preds.get(it["id"])
        views["repo_matcher"].append(float(repo_norm(g) == repo_norm(p)))
        views["exact_norm"].append(float(bool(norm_text(g)) and norm_text(p) == norm_text(g)))
        views["squashed"].append(float(bool(squash(g)) and squash(p) == squash(g)))
        reach["verbatim" if it["meta"]["gold_reachable"]["committee"] else "non_verbatim"].append(
            views["repo_matcher"][-1])
    out = {"n": len(items), "n_pred": sum(it["id"] in preds for it in items),
           "n_pred_null": sum(1 for it in items if preds.get(it["id"]) is None)}
    for k, xs in views.items():
        out[k] = {"acc": round(sum(xs) / len(xs), 4), "ci95": ci(xs)}
    out["repo_matcher_by_reachability"] = {k: {"n": len(v), "acc": round(sum(v) / len(v), 4) if v else None}
                                           for k, v in reach.items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
