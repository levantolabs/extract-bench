"""Hints for the SURVEY C7 hand-audit of null golds (term, effective_date). Heuristic only; gold is NOT changed.

  python3 -I extract-bench/datasets/kleister_nda/scripts/null_audit_hints.py   -> scripts/null_audit_hints.out.json
term: a null-term doc is a candidate if a '<number|word> [(n)] year(s)/month(s)' phrase sits within 120 chars after
'term' / 'period' / 'expire' / 'terminate'. (Survival / confidentiality periods also match, so candidates are
not errors: the Kleister term is the contract's length.)
effective_date: a null-date doc is a candidate if 'effective' / 'dated' / 'as of' is followed within 60 chars by a
full written date with a 4-digit year (blank forms such as 'dated as of [ ]' do not match).
"""
import json, re
from pathlib import Path

DS = Path(__file__).resolve().parents[1]
items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines()]
NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|eighteen|twenty-four|thirty-six)"
TERM = re.compile(rf"\b(?:term|period|expire|terminate)\w*\b[^.]{{0,120}}?\b{NUM}\s*(?:\(\s*\d+\s*\)\s*)?(?:years?|months?)\b", re.I | re.S)
MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE = re.compile(rf"\b(?:effective|dated|as of)\b[^.]{{0,60}}?(?:{MON}\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{MON},?\s+\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{4}})", re.I | re.S)
out = {"term": {"n_null": 0, "candidates": []}, "effective_date": {"n_null": 0, "candidates": []}}
for it in items:
    t = re.sub(r"\s+", " ", it["text"])
    g = {f["name"]: f["gold"] for f in it["fields"]}
    for name, rx in (("term", TERM), ("effective_date", DATE)):
        if g[name] is None:
            out[name]["n_null"] += 1
            m = rx.search(t)
            if m:
                out[name]["candidates"].append({"id": it["id"], "snippet": t[max(0, m.start() - 40): m.end() + 40]})
for k in out:
    out[k]["n_candidates"] = len(out[k]["candidates"])
(DS / "scripts" / "null_audit_hints.out.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print({k: (v["n_null"], v["n_candidates"]) for k, v in out.items()})
