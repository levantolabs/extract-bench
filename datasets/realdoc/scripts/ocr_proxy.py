"""$0 local Tesseract OCR proxy for token estimates + gold reachability (NOT the harness OCR cache).

Mirrors harness/ocr.py settings (pdftoppm -r 200 -png, then tesseract -l eng --oem 1 --psm 3) with two
differences: the page image goes to tesseract on stdin, and output is decoded from bytes. On this machine
harness/ocr.py fails on every realdoc item: tesseract spawned from Python cannot open the temp-dir PNG by path,
Leptonica's error message echoes raw PNG bytes (0x89) on stderr, and text=True raises UnicodeDecodeError
(reported, not fixed here; the harness is not ours to edit). One OCR per unique source PDF, cached in scripts/ocr_proxy/<doc>.txt.

Usage (from extract-bench/):
  /opt/homebrew/bin/uv run --no-project --with tiktoken==0.12.0 python -I datasets/realdoc/scripts/ocr_proxy.py
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS / "scripts"))
sys.path.insert(0, str(DS))
from build import reachable  # noqa: E402

OUT = DS / "scripts" / "ocr_proxy"
TESS, PDFTOPPM = "/opt/homebrew/bin/tesseract", "/opt/homebrew/bin/pdftoppm"


def ocr_doc(stem: str) -> dict:
    f = OUT / f"{stem}.txt"
    if f.exists():
        return {"doc": stem, "cached": True, "stderr_nonutf8": None}
    pdf = DS / "raw" / "docs" / f"{stem}.pdf"
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rdb-ocr-") as td:
        subprocess.run([PDFTOPPM, "-r", "200", "-png", str(pdf), str(Path(td) / "p")], check=True,
                       capture_output=True, timeout=300)
        texts, bad = [], False
        for pg in sorted(Path(td).glob("p*.png")):
            # image via stdin: in this sandbox, tesseract spawned from Python cannot open files under the temp
            # dir by path (Leptonica "failed to open locally ... image file not found"), stdin works
            r = subprocess.run([TESS, "stdin", "stdout", "-l", "eng", "--oem", "1", "--psm", "3"],
                               input=pg.read_bytes(), capture_output=True, timeout=300)
            if not r.stdout.strip():
                raise RuntimeError(f"{stem}: empty OCR; stderr={r.stderr[:200]!r}")
            try:
                r.stderr.decode("utf-8")
            except UnicodeDecodeError:
                bad = True
            texts.append(r.stdout.decode("utf-8", errors="replace").strip())
    OUT.mkdir(exist_ok=True)
    f.write_text("\n\n".join(texts))
    return {"doc": stem, "cached": False, "stderr_nonutf8": bad, "ms": round((time.perf_counter() - t0) * 1000)}


def main():
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    items = [json.loads(l) for l in (DS / "items.jsonl").read_text().splitlines() if l.strip()]
    docs = sorted({it["meta"]["doc_id"] for it in items})
    with ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(ocr_doc, docs))
    ver = subprocess.run([TESS, "--version"], capture_output=True).stdout.decode(errors="replace").splitlines()[0]
    toks, reach, reach_enum, by_kind = [], [], [], Counter()
    for it in items:
        t = (OUT / f"{it['meta']['doc_id']}.txt").read_text()
        toks.append(len(enc.encode(t, disallowed_special=())))
        for f in it["fields"]:
            if f["gold"] is None:
                continue
            r = reachable(f["gold"], t)
            if "enum_options" in f:
                reach_enum.append(r)
                by_kind[("enum_state" if f["enum_state"] else "enum_label", r)] += 1
            else:
                reach.append(r)
                by_kind[(f["type"], r)] += 1
    xs = sorted(toks)
    q = lambda p: xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]  # noqa: E731
    out = {"engine": ver, "settings": "pdftoppm -r 200 -png; tesseract -l eng --oem 1 --psm 3", "docs": len(docs),
           "docs_stderr_nonutf8": [r["doc"] for r in recs if r.get("stderr_nonutf8")],
           "tokens_per_item_o200k": {"p50": q(50), "p95": q(95), "max": xs[-1], "mean": round(statistics.mean(xs), 1)},
           "gold_reachable_nonenum_nonnull": f"{sum(reach)}/{len(reach)}",
           "enum_gold_found_in_text": f"{sum(reach_enum)}/{len(reach_enum)}",
           "reachable_by_kind": {f"{k[0]}:{'yes' if k[1] else 'no'}": v for k, v in sorted(by_kind.items())}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
