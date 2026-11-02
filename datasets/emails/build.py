"""Build extract-bench/datasets/emails/items.jsonl from raw/ (Willis LLM-Extraction-Challenge, fundraising emails).

Source: https://github.com/dwillis/LLM-Extraction-Challenge (MIT), pinned to commit 72b8d5fe (HEAD, 2026-08-06).
Gold: fundraising-emails/training.csv (1,000 real Nov 2024 political fundraising emails; `committee` = the committee
named in the "Paid for by" disclaimer; `name` = sender). This is the file the repo's matcher.py grades against.

What this script does (stdlib only; tiktoken optional for token counts):
  1. Verifies every raw/repo file against the git blob sha1 in raw/pages/tree.json (aborts on mismatch).
  2. Parses training.csv with the stdlib csv module (no pandas NaN coercion) and checks 1,000 rows, 0 empty committees.
  3. Samples 100 of 1,000 rows with random.Random(20).sample(range(1000), 100) (simple random; the spec asks for no
     stratification). Items are written in CSV row order.
  4. Marks the sealed 20% holdout like the other sets: random.Random(20).sample(sorted(ids), 20) -> meta.holdout.
  5. Writes one field per item: committee (text), gold = the official training.csv string (no invented variants).
     meta.gold_reachable records whether the gold is in the body verbatim (whitespace-collapsed, lower-cased) and after
     alnum-squash; meta.gold_body_span is the body substring whose alnum-squash equals the gold's (span-extractor view).
  6. Writes anchor/items_all1000.jsonl: the same format for ALL 1,000 rows (replication anchor; never mixed with the
     100-item sample) and comparable/prompt2.txt + comparable/prompt2_schema.json (verbatim from email_llm.py).
  7. Writes build_report.json (counts, token stats, checks).

Run:  /opt/homebrew/bin/uv run --no-project --with tiktoken python -I build.py   (from anywhere; paths are file-relative)
"""
from __future__ import annotations

import csv, hashlib, json, random, re, statistics, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
REPO = RAW / "repo"
FE = REPO / "fundraising-emails"
KEY = "emails"
SOURCE = "https://github.com/dwillis/LLM-Extraction-Challenge"
COMMIT = "72b8d5fe9a8b44702503005559438868bc73e000"
GOLD_COMMIT = "c181a6da75"  # last commit that changed training.csv (2025-07-10, "more evals")
USE_CASE = "Email / message field extraction (committee named in the disclaimer)"
SEED, N_SAMPLE, HOLDOUT_FRAC = 20, 100, 0.20
SAGE_LIMIT = 16384

# Field instruction = the committee clause of the repo's prompt2 (email_llm.py lines 26-27), verbatim.
INSTR = ("The name of the committee in the disclaimer that begins with \"Paid for by\" but do not include the "
         "\"Paid for by\" text itself. If no committee is present, use null.")

# prompt2 (email_llm.py lines 26-31) and its JSON schema (lines 9-22), verbatim. The smevals README in the same repo
# (commit 60400b7) says this is the "prompt2" text behind the *_prompt2.json leaderboard rows.
PROMPT2 = """Extract the following information from this email:
1. 'committee': The name of the committee in the disclaimer that begins with "Paid for by" but do not include the "Paid for by" text itself. If no committee is present, use null.
2. 'sender': The name of the person, if any, mentioned as the author of the email. If there is no person named, use null.

Email body:
{body}"""
PROMPT2_SCHEMA = {
    "type": "object",
    "properties": {
        "committee": {"type": ["string", "null"],
                      "description": "The name of the committee from the disclaimer (without 'Paid for by' prefix)"},
        "sender": {"type": ["string", "null"],
                   "description": "The name of the person mentioned as the author of the email"},
    },
    "required": ["committee", "sender"],
}
MERGE_COLS = ["email", "subject", "year", "month", "day", "hour", "minute", "domain"]  # matcher.py join key


def git_blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def ws_lower(s: str) -> str:  # the span-ceiling check in survey/critique-legitimacy-2026-10-23/emails_span_ceiling.py
    return re.sub(r"\s+", " ", str(s)).lower()


def squash(s: str) -> str:  # same as harness/score.py squash(): NFKC casefold, keep [0-9a-z]
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return re.sub(r"[^0-9a-z]", "", s)


def body_span_for(gold: str, body: str) -> tuple[str | None, int | None, int | None]:
    """Shortest body substring whose alnum-squash equals squash(gold) (first occurrence). None if absent."""
    target = squash(gold)
    if not target:
        return None, None, None
    # map squashed-character positions back to body offsets
    keep, pos = [], []
    for i, ch in enumerate(body):
        for c in squash(ch):
            keep.append(c)
            pos.append(i)
    sq = "".join(keep)
    k = sq.find(target)
    if k < 0:
        return None, None, None
    start, end = pos[k], pos[k + len(target) - 1] + 1
    return body[start:end], start, end


def verify_raw() -> dict:
    tree = {e["path"]: e["sha"] for e in json.loads((RAW / "pages" / "tree.json").read_text())["tree"]
            if e["type"] == "blob"}
    n = 0
    for p in sorted(REPO.rglob("*")):
        if p.is_file():
            rel = p.relative_to(REPO).as_posix()
            if tree.get(rel) != git_blob_sha1(p.read_bytes()):
                sys.exit(f"git blob mismatch: {rel}")
            n += 1
    return {"files_verified": n, "training_csv_blob": tree["fundraising-emails/training.csv"],
            "training_csv_sha256": hashlib.sha256((FE / "training.csv").read_bytes()).hexdigest()}


def token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s, disallowed_special=()))
    except Exception:  # tiktoken missing or offline: counts become None
        return None


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def make_item(i: int, r: dict, ntok, holdout: set[str] | None, anchor: bool) -> dict:
    body, gold = r["body"], r["committee"]
    iid = f"{KEY}-r{i:04d}"
    span, s0, s1 = body_span_for(gold, body)
    t = ntok(body) if ntok else None
    sage_defs = json.dumps({"committee": {"type": "text", "instructions": INSTR}})
    sage_tok = (t + ntok(sage_defs)) if ntok else None
    return {
        "id": iid, "dataset": KEY, "use_case": USE_CASE, "modality": "text",
        "text": body, "text_source": "native", "images": [],
        "fields": [{"name": "committee", "type": "text", "instructions": INSTR, "gold": gold}],
        "metric": "exact_norm",
        "meta": {
            "split": "training.csv (the repo's only labelled file; it is what matcher.py scores against)",
            "source": SOURCE, "commit": COMMIT, "gold_last_changed_commit": GOLD_COMMIT,
            "csv_row": i, "doc_id": iid,
            "merge_key": {c: r[c] for c in MERGE_COLS},
            "from_email": r["email"], "subject": r["subject"], "date": r["date"], "domain": r["domain"],
            "party": r["party"] or None, "disclaimer": r["disclaimer"],
            "repo_sender_unscored": r["name"],
            "committee_cluster": squash(gold),
            "text_tokens_o200k": t, "sage_tokens_est": sage_tok,
            "fits_sage_16k": (sage_tok <= SAGE_LIMIT) if sage_tok is not None else None,
            "paid_for_by_in_body": "paid for by" in body.lower(),
            "gold_reachable": {"committee": ws_lower(gold) in ws_lower(body)},
            "gold_reachable_squashed": {"committee": squash(gold) in squash(body)},
            "gold_body_span": {"text": span, "start": s0, "end": s1},
            "holdout": (iid in holdout) if holdout is not None else None,
            "anchor_all1000": anchor,
            "comparable": {"scorer": "comparable/score_emails.py",
                           "prompt": "comparable/prompt2.txt", "schema": "comparable/prompt2_schema.json",
                           "metric": "repo matcher.py Accuracy: fillna('none').astype(str).lower() exact match "
                                     "on committee (no strip on the Accuracy line)"},
        },
    }


def main():
    report = {"source": SOURCE, "commit": COMMIT}
    report.update(verify_raw())

    csv.field_size_limit(10**9)
    with (FE / "training.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1000, len(rows)
    assert all(r["committee"].strip() for r in rows), "empty committee gold"
    report["n_total"] = len(rows)
    report["committee_with_outer_whitespace"] = sum(r["committee"] != r["committee"].strip() for r in rows)
    report["distinct_committees"] = len({r["committee"] for r in rows})

    ntok = token_counter()
    sample_rows = sorted(random.Random(SEED).sample(range(len(rows)), N_SAMPLE))
    ids = [f"{KEY}-r{i:04d}" for i in sample_rows]
    holdout = set(random.Random(SEED).sample(sorted(ids), round(HOLDOUT_FRAC * len(ids))))

    items = [make_item(i, rows[i], ntok, holdout, anchor=False) for i in sample_rows]
    anchor = [make_item(i, r, ntok, None, anchor=True) for i, r in enumerate(rows)]

    with (ROOT / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    (ROOT / "anchor").mkdir(exist_ok=True)
    with (ROOT / "anchor" / "items_all1000.jsonl").open("w", encoding="utf-8") as fh:
        for it in anchor:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    (ROOT / "comparable").mkdir(exist_ok=True)
    (ROOT / "comparable" / "prompt2.txt").write_text(PROMPT2, encoding="utf-8")
    (ROOT / "comparable" / "prompt2_schema.json").write_text(json.dumps(PROMPT2_SCHEMA, indent=4) + "\n",
                                                             encoding="utf-8")
    # the verbatim prompt must still be in the pinned email_llm.py
    src = (FE / "email_llm.py").read_text(encoding="utf-8")
    report["prompt2_verbatim_in_email_llm_py"] = PROMPT2.replace("{body}", "{email['body']}") in src
    if not report["prompt2_verbatim_in_email_llm_py"]:
        sys.exit("prompt2 text no longer matches email_llm.py")

    def stats(its):
        toks = [it["meta"]["text_tokens_o200k"] for it in its if it["meta"]["text_tokens_o200k"] is not None]
        return {
            "n": len(its),
            "gold_verbatim_ws_lower": sum(it["meta"]["gold_reachable"]["committee"] for it in its),
            "gold_in_body_squashed": sum(it["meta"]["gold_reachable_squashed"]["committee"] for it in its),
            "paid_for_by_in_body": sum(it["meta"]["paid_for_by_in_body"] for it in its),
            "null_gold": sum(f["gold"] is None for it in its for f in it["fields"]),
            "distinct_committees": len({it["fields"][0]["gold"] for it in its}),
            "tokens_o200k": {"p50": pct(toks, .5), "p95": pct(toks, .95), "max": max(toks) if toks else None,
                             "mean": round(statistics.mean(toks), 1) if toks else None, "total": sum(toks)},
            "n_over_sage_16k": sum(1 for it in its if it["meta"]["fits_sage_16k"] is False),
            "party": {str(p): sum(1 for it in its if it["meta"]["party"] == p)
                      for p in sorted({it["meta"]["party"] for it in its}, key=str)},
        }

    report["sample"] = stats(items)
    report["all1000"] = stats(anchor)
    report["sample_rows"] = sample_rows
    report["holdout_n"] = len(holdout)
    report["holdout_ids"] = sorted(holdout)
    report["non_verbatim_sample"] = [
        {"id": it["id"], "gold": it["fields"][0]["gold"], "body_span": it["meta"]["gold_body_span"]["text"]}
        for it in items if not it["meta"]["gold_reachable"]["committee"]]
    (ROOT / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_total", "files_verified", "prompt2_verbatim_in_email_llm_py",
                                             "sample", "all1000", "holdout_n")}, indent=1))


if __name__ == "__main__":
    main()
