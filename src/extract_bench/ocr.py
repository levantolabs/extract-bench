"""Free local OCR for image items that ship no official OCR (fatura, sroie, cord, realdoc).

Engine: Tesseract 5 (the shipped OCR caches were made with 5.5.0 / leptonica 1.85.0), `-l eng --oem 1 --psm 3`
(LSTM, fully automatic page segmentation), with psm 6 retried on pages where psm 3 finds no text. PDF inputs are
rendered with poppler `pdftoppm -r 200 -png`. Neither tool is installed by this package.

The OCR text used for the published results ships in datasets/<key>/ocr/, so the text track needs no OCR at all.
Other engines (olmOCR, marker, docling, PaddleOCR) can plug in via ENGINES.

Cache: datasets/<key>/ocr/<engine_tag>/<id>.txt plus <id>.json (version, per-image ms, args). Outputs are data.
"""
from __future__ import annotations

import json, shutil, subprocess, tempfile, time
from pathlib import Path

from .common import write_json

TESSERACT = shutil.which("tesseract") or "tesseract"
PDFTOPPM = shutil.which("pdftoppm") or "pdftoppm"
PAGE_SEP = "\n\n"


class OCRUnavailable(RuntimeError):
    ...


def tesseract_version() -> str | None:
    try:
        out = subprocess.run([TESSERACT, "--version"], capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return None


# Pre-registered default (2026-10-23): psm 3, and psm 6 (single uniform block) for a page where psm 3 finds no text.
# psm 3 returned an EMPTY page for 49/200 CORD and 14/347 SROIE receipts (small or odd-DPI photos; Tesseract's
# layout analysis found no text block), which would silently drop those items from every text arm.
DEFAULT_OCR = {"engine": "tesseract", "psm": 3, "lang": "eng", "fallback_psm": 6}


def engine_tag(engine: str = "tesseract", psm: int = 3, lang: str = "eng", fallback_psm: int | None = None) -> str:
    return f"{engine}-{lang}-psm{psm}" + (f"-fb{fallback_psm}" if fallback_psm else "")


def _pdf_pages(pdf: Path, tmp: Path, dpi: int = 200) -> list[Path]:
    if not Path(PDFTOPPM).exists():
        raise OCRUnavailable("pdftoppm not found; `brew install poppler`")
    subprocess.run([PDFTOPPM, "-r", str(dpi), "-png", str(pdf), str(tmp / "p")], check=True, capture_output=True,
                   timeout=300)
    return sorted(tmp.glob("p*.png"))


def tesseract_image(path: Path, psm: int = 3, lang: str = "eng", timeout: int = 180) -> str:
    if not Path(TESSERACT).exists():
        raise OCRUnavailable("tesseract not found; install with `brew install tesseract`")
    # Resolve symlinks: on this Mac /tmp -> /private/tmp and Leptonica cannot open the /tmp/... spelling
    # (it then echoes raw image bytes on stderr). Decode leniently so such stderr can never crash the run.
    out = subprocess.run([TESSERACT, str(Path(path).resolve()), "stdout", "-l", lang, "--oem", "1", "--psm", str(psm)],
                         capture_output=True, timeout=timeout)
    stdout = out.stdout.decode("utf-8", errors="replace")
    if out.returncode != 0 or ("Error in" in (err := out.stderr.decode("utf-8", errors="replace")) and not stdout.strip()):
        raise RuntimeError(f"tesseract failed on {path.name}: {err[:300]!r}")
    return stdout


def ocr_paths(paths: list[Path], psm: int = 3, lang: str = "eng", fallback_psm: int | None = None) -> dict:
    texts, per = [], []
    with tempfile.TemporaryDirectory(prefix="exb-ocr-") as td:
        td = Path(td).resolve()  # /tmp symlink, see tesseract_image
        for p in paths:
            pages = _pdf_pages(p, td) if p.suffix.lower() == ".pdf" else [p]
            for pg in pages:
                t0 = time.perf_counter()
                t, used = tesseract_image(pg, psm=psm, lang=lang).strip(), psm
                if not t and fallback_psm:
                    t, used = tesseract_image(pg, psm=fallback_psm, lang=lang).strip(), fallback_psm
                texts.append(t)
                per.append({"file": p.name, "page": pg.name, "psm_used": used, "chars": len(t),
                            "ms": round((time.perf_counter() - t0) * 1000, 1)})
    return {"text": PAGE_SEP.join(texts), "per_image": per}


ENGINES = {"tesseract": ocr_paths}


def ocr_item(item: dict, ds_dir: Path, engine: str = "tesseract", psm: int = 3, lang: str = "eng",
             fallback_psm: int | None = None, use_cache: bool = True) -> dict:
    """Return {'text', 'engine', 'version', 'cached', ...} for an image item, OCRing and caching if needed."""
    tag = engine_tag(engine, psm, lang, fallback_psm)
    cdir = ds_dir / "ocr" / tag
    ft, fj = cdir / f"{item['id']}.txt", cdir / f"{item['id']}.json"
    if use_cache and ft.exists() and fj.exists():
        return {**json.loads(fj.read_text()), "text": ft.read_text(), "cached": True}
    if engine not in ENGINES:
        raise OCRUnavailable(f"engine {engine} not implemented (see module TODO)")
    srcs = [ds_dir / p for p in item.get("images") or []]
    if item.get("meta", {}).get("pdf"):  # builders may point at the source PDF for multi-page docs
        srcs = [ds_dir / item["meta"]["pdf"]]
    if not srcs:
        raise ValueError(f"{item['id']}: no images to OCR")
    t0 = time.perf_counter()
    r = ENGINES[engine](srcs, psm=psm, lang=lang, fallback_psm=fallback_psm)
    rec = {"id": item["id"], "engine": engine, "engine_tag": tag,
           "version": tesseract_version() if engine == "tesseract" else None,
           "args": {"psm": psm, "lang": lang, "oem": 1, "fallback_psm": fallback_psm}, "ms": round((time.perf_counter() - t0) * 1000, 1),
           "per_image": r["per_image"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    cdir.mkdir(parents=True, exist_ok=True)
    ft.write_text(r["text"])
    write_json(fj, rec)
    return {**rec, "text": r["text"], "cached": False}


def available() -> bool:
    return Path(TESSERACT).exists() and tesseract_version() is not None
