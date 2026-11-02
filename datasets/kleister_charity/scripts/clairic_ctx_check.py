"""Exploration: dev-0 docs over Clairic's 65,536-token context per text layer, with o200k and the harness's
o200k x 1.12 Clairic-token assumption (harness/estimate.py docstring). Writes scripts/clairic_ctx_check.out.json."""
import csv, json, lzma, re, sys
from pathlib import Path
import tiktoken
DS = Path(__file__).resolve().parents[1]
enc = tiktoken.get_encoding("o200k_base")
csv.field_size_limit(sys.maxsize)
un = lambda s: re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "f": "\f", "\\": "\\"}.get(m[1], "\\" + m[1]), s)  # noqa: E731
with lzma.open(DS / "raw/repo/dev-0/in.tsv.xz", "rt", encoding="utf-8", newline="") as fh:
    rows = list(csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
out = {}
for lay, col in {"djvu": 2, "tesseract": 3, "textract": 4, "best": 5}.items():
    t = [len(enc.encode(un(r[col]), disallowed_special=())) for r in rows]
    out[lay] = {"n": len(t), "max_o200k": max(t), "over_65536_o200k": sum(x > 65536 for x in t),
                "over_65536_x1.12": sum(x * 1.12 > 65536 for x in t),
                "over_58514_o200k(=65536/1.12)": sum(x > 65536 / 1.12 for x in t),
                "within_5pct_of_limit_x1.12": sum(0.95 * 65536 < x * 1.12 <= 65536 for x in t)}
(DS / "scripts/clairic_ctx_check.out.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
