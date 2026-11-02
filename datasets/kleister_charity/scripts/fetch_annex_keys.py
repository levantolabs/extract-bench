"""One-off fetch (free, GitHub raw): the git-annex symlink target of every dev-0 PDF at the pinned commit.
The target names the annex key SHA256E-s<bytes>--<sha256>.pdf, i.e. the PDF's byte size and SHA-256, without
downloading the PDF (PDFs are excluded from this text-only dataset). Output: raw/annex_keys_dev0.tsv
(filename, annex_key, pdf_bytes, pdf_sha256). Also saves the git-annex branch remote.log (S3 remote config) if reachable.
Run: uv run --no-project --with httpx python -I scripts/fetch_annex_keys.py"""
import csv, lzma, re, sys, time
from pathlib import Path
import httpx
ROOT = Path(__file__).resolve().parents[1]
COMMIT = "2309f4861cc49864ac653a7fa108efc2144400b3"
csv.field_size_limit(sys.maxsize)
with lzma.open(ROOT / "raw/repo/dev-0/in.tsv.xz", "rt", encoding="utf-8", newline="") as fh:
    names = [r[0] for r in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)]
out = ROOT / "raw/annex_keys_dev0.tsv"
done = {}
if out.exists():
    for line in out.read_text().splitlines()[1:]:
        f = line.split("\t"); done[f[0]] = line
rows = []
with httpx.Client(http2=False, timeout=30, headers={"User-Agent": "extract-bench-provenance"}) as c:
    for i, n in enumerate(names):
        if n in done:
            rows.append(done[n]); continue
        for attempt in range(4):
            try:
                r = c.get(f"https://raw.githubusercontent.com/applicaai/kleister-charity/{COMMIT}/documents/{n}")
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2 * (attempt + 1))
        t = r.text.strip()
        m = re.search(r"(SHA256E-s(\d+)--([0-9a-f]{64})\.pdf)$", t)
        rows.append("\t".join([n, m[1], m[2], m[3]]) if m else "\t".join([n, "UNKNOWN:" + t[:80], "", ""]))
out.write_text("filename\tannex_key\tpdf_bytes\tpdf_sha256\n" + "\n".join(rows) + "\n")
print("rows", len(rows), "unknown", sum("UNKNOWN" in r for r in rows))
try:
    r = httpx.get("https://raw.githubusercontent.com/applicaai/kleister-charity/git-annex/remote.log", timeout=30)
    (ROOT / "raw/pages/git-annex_remote.log").write_text(r.text)
    print("remote.log", r.status_code, r.text[:300])
except httpx.HTTPError as e:
    print("remote.log unreachable", e)
