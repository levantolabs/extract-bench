#!/usr/bin/env python3
"""$0 pre-scan for SROIE gold-noise CANDIDATES (no gold is changed here).

Evidence source: the jsdnrs mirror's Task 1/2 OCR lines (human transcripts; truncated at the
first comma, so only *positive* evidence is used: a gold token that is absent from the OCR
while a same-length OCR token differs from it by exactly one character, e.g. gold "B1750"
vs OCR "81750"). Also flags company/date/total gold strings that are not substrings of the
whitespace-collapsed OCR lines (weaker evidence; truncation can cause misses for company).

The pre-registered adjudication rule (SURVEY L7) is applied only AFTER arms run: items where
all arms agree with each other but not with gold get a blind review against the image, for
all arms equally; raw and adjudicated scores are reported side by side. This file just lists
where to look. Output: scripts/gold_noise_candidates.jsonl + summary on stdout.

Run: /opt/homebrew/bin/uv run --no-project python -I .../scripts/gold_noise_scan.py
"""
import json
import os
import re

DS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one_char_diff(a, b):
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    mirror = {}
    with open(os.path.join(DS, "raw", "jsdnrs-mirror", "test_text.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            mirror[r["key"]] = r
    out, counts = [], {"items": 0, "with_candidates": 0, "one_char_token": 0,
                       "not_substring": {"company": 0, "date": 0, "total": 0}}
    with open(os.path.join(DS, "items.jsonl")) as f:
        for line in f:
            it = json.loads(line)
            counts["items"] += 1
            stem = it["meta"]["image_name"].rsplit(".", 1)[0]
            r = mirror.get(stem)
            if not r:
                continue
            gold = it["meta"]["gold_verbatim"]
            words = r["words"]
            ocr_join = norm(" ".join(words))
            ocr_tokens = set(t for w in words for t in re.split(r"[\s,]+", w) if t)
            cands = []
            for k in ("company", "address", "date", "total"):
                for tok in re.split(r"[\s,]+", gold[k]):
                    if len(tok) < 3 or tok in ocr_tokens:
                        continue
                    near = sorted(t for t in ocr_tokens if one_char_diff(tok, t))
                    if near:
                        cands.append({"field": k, "kind": "one_char_token", "gold_token": tok,
                                      "ocr_tokens": near})
                        counts["one_char_token"] += 1
            for k in ("company", "date", "total"):
                if norm(gold[k]) not in ocr_join:
                    cands.append({"field": k, "kind": "not_substring_of_ocr", "gold": gold[k]})
                    counts["not_substring"][k] += 1
            if cands:
                counts["with_candidates"] += 1
                out.append({"id": it["id"], "image": it["images"][0], "candidates": cands})
    with open(os.path.join(DS, "scripts", "gold_noise_candidates.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(os.path.join(DS, "scripts", "gold_noise_scan.out.json"), "w") as f:
        json.dump(counts, f, indent=1)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
