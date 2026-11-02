"""Render the README's raw-output figures as standalone HTML (docs/examples/*.html).

  python scripts/render_examples.py                       # example document + terminal captures
  python scripts/render_examples.py --item fatura fatura-T10-I6

1. example-<id>.html: one document, its gold fields, and every system's raw answer from results/<version>/, marked
   correct / wrong with the benchmark's own scorer (rule a).
2. terminal-report.html / terminal-validate.html: the real stdout of `extract-bench report` and
   `extract-bench validate results/v0.1/nuextract3`.
The PNGs in docs/images/ are 2x screenshots of these pages taken with a headless browser.
"""
from __future__ import annotations

import argparse, html, io, json, shutil
from contextlib import redirect_stdout
from pathlib import Path

from extract_bench import score as S
from extract_bench.common import BENCH_VERSION, RESULTS, ROOT, load_items
from extract_bench.report import load_system
from extract_bench.report import main as report_main
from extract_bench.validate import main as validate_main

OUT = ROOT / "docs" / "examples"
SYSTEMS = ["sage2", "clairic", "nuextract3", "schematron3b", "gliner25"]
CSS = """
:root { --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e; --muted:#8a8984; --rule:#e6e5e1; --good:#0ca30c; --bad:#d03b3b;
        --chip:#f3f2ee; }
* { box-sizing:border-box; } body { margin:0; background:var(--surface); color:var(--text);
  font:14px/1.45 "Helvetica Neue", Helvetica, Arial, sans-serif; }
.wrap { padding:28px 32px; display:flex; gap:28px; align-items:flex-start; width:max-content; }
.doc img { width:360px; border:1px solid var(--rule); border-radius:6px; display:block; }
.doc p { color:var(--muted); font-size:12px; margin:8px 0 0; max-width:360px; }
h1 { font-size:17px; margin:0 0 4px; } .sub { color:var(--text2); margin:0 0 16px; font-size:13px; }
table { border-collapse:collapse; font-size:13px; } th, td { padding:7px 10px; border-bottom:1px solid var(--rule);
  text-align:left; vertical-align:top; white-space:nowrap; }
th { color:var(--text2); font-weight:600; font-size:12px; } td.f { font-family:Menlo, Consolas, monospace; font-size:12px; }
td.gold { background:var(--chip); font-weight:600; }
.v { font-family:Menlo, Consolas, monospace; font-size:12px; } .null { color:var(--muted); font-style:italic; }
.ok::before { content:"✓ "; color:var(--good); font-weight:700; } .no::before { content:"✗ "; color:var(--bad); font-weight:700; }
.part::before { content:"◐ "; color:var(--text2); }
tfoot td { color:var(--text2); font-weight:600; border-bottom:none; }
.term { margin:28px 32px; background:#141413; color:#e8e6df; border-radius:10px; padding:18px 22px 20px;
  font:12.5px/1.5 Menlo, Consolas, monospace; white-space:pre; width:max-content; box-shadow:0 1px 0 #0003; }
.term .bar { color:#8f8e86; margin-bottom:10px; } .term .cmd { color:#7fb6f5; }
"""


def _val(v):
    if v is None:
        return '<span class="null">null</span>'
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return f'<span class="v">{html.escape(s if len(s) <= 44 else s[:43] + "…")}</span>'


def example(ds: str, iid: str, version: str) -> Path:
    it = next(i for i in load_items(ds) if i["id"] == iid)
    res = RESULTS / version
    rows, meta = [], {}
    for s in SYSTEMS:
        m, recs = load_system(res, s)
        meta[s] = m
        r = recs.get((ds, iid)) or {}
        sc = {x["name"]: x for x in S.score_item(it, r.get("preds") or {})["fields"]}
        rows.append((s, r.get("preds") or {}, sc))
    head = "".join(f"<th>{html.escape(meta[s]['display_name'])}</th>" for s in SYSTEMS)
    body, totals = [], {s: [0.0, 0] for s in SYSTEMS}
    for f in it["fields"]:
        if f.get("headline") is False:
            continue
        g = f.get("gold")
        gold = '<span class="null">not in document</span>' if g is None else _val(g if not isinstance(g, list) or f["type"] == "list" else " | ".join(map(str, g)))
        cells = []
        for s, preds, sc in rows:
            x = sc.get(f["name"]) or {}
            if not x.get("scored"):
                cells.append(f'<td>{_val((preds.get(f["name"]) or {}).get("value"))}</td>'); continue
            k = "ok" if x["score_a"] >= 1 else ("no" if x["score_a"] <= 0 else "part")
            totals[s][0] += x["score_a"]; totals[s][1] += 1
            cells.append(f'<td class="{k}">{_val((preds.get(f["name"]) or {}).get("value"))}</td>')
        body.append(f'<tr><td class="f">{f["name"]}</td><td class="gold">{gold}</td>{"".join(cells)}</tr>')
    foot = "".join(f"<td>{t[0]:g} / {t[1]}</td>" for t in totals.values())
    img = it["images"][0] if it.get("images") else None
    img_html = ""
    if img:
        src = ROOT / "datasets" / ds / img
        dst = ROOT / "docs" / "images" / f"example-{iid}{Path(img).suffix}"
        if src.exists():
            shutil.copyfile(src, dst)
        img_html = (f'<div class="doc"><img src="../images/{dst.name}" alt="{iid}">'
                    f'<p>FATURA invoice {html.escape(iid)} (CC BY 4.0, Limam et al. 2023). Every system read the same '
                    f'Tesseract OCR of this page.</p></div>')
    page = (f'<!doctype html><meta charset="utf-8"><title>{iid}</title><style>{CSS}</style><div class="wrap">{img_html}'
            f'<div><h1>Raw answers on one invoice</h1><p class="sub">The gold value and each system\'s own answer for '
            f'every field of {iid}. ✓ marks a correct answer and ✗ a wrong one. Under rule (a), a correct '
            f'"not in the document" counts.</p>'
            f'<table><thead><tr><th>field</th><th>gold</th>{head}</tr></thead><tbody>{"".join(body)}</tbody>'
            f'<tfoot><tr><td>correct</td><td></td>{foot}</tr></tfoot></table></div></div>')
    out = OUT / f"example-{iid}.html"
    out.write_text(page)
    return out


def terminal(name: str, argv: list[str], fn) -> Path:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(argv[1:])
    text = "\n".join(l for l in buf.getvalue().splitlines() if not l.startswith("wrote "))
    page = (f'<!doctype html><meta charset="utf-8"><title>{name}</title><style>{CSS}</style><div class="term">'
            f'<div class="bar">● ● ●</div><span class="cmd">$ extract-bench {html.escape(" ".join(argv))}</span>\n'
            f'{html.escape(text.rstrip())}</div>')
    out = OUT / f"terminal-{name}.html"
    out.write_text(page)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item", nargs=2, default=["fatura", "fatura-T10-I6"], metavar=("DATASET", "ID"))
    ap.add_argument("--version", default=BENCH_VERSION)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(example(*a.item, a.version))
    print(terminal("report", ["report", "--no-write"], report_main))
    print(terminal("validate", ["validate", f"results/{a.version}/nuextract3"], validate_main))


if __name__ == "__main__":
    main()
