"""Shared helpers for the extraction-bench harness: paths, item loading, text resolution, tokens, holdout.

Item format: see datasets/<key>/items.jsonl (one JSON object per line). Field dicts may carry
optional hints beyond the common format; the harness reads them when present:
  unit (str), integer (bool), decimals (int), items ("text"|"number"|"date"), dayfirst (bool),
  number_locale ("dot_thousands" for "1.234" = 1234, e.g. CORD Indonesian amounts).
"""
from __future__ import annotations

import hashlib, json, os, re
from pathlib import Path

def _find_root() -> Path:
    """The benchmark checkout: $EXTRACT_BENCH_ROOT, else the nearest parent of the cwd (or of this package, for an
    editable install) that has a datasets/ directory next to a pyproject.toml."""
    env = os.environ.get("EXTRACT_BENCH_ROOT")
    if env:
        return Path(env).resolve()
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for d in (start, *start.parents):
            if (d / "datasets").is_dir() and (d / "pyproject.toml").is_file():
                return d
    return Path.cwd().resolve()


ROOT = _find_root()
DATASETS = Path(os.environ.get("EXB_DATASETS", ROOT / "datasets"))
RUNS = Path(os.environ.get("EXB_RUNS", ROOT / "runs"))
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
BENCH_VERSION = "v0.1"
SEED = 20
HOLDOUT_FRAC = 0.20

FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
FIELD_TYPES = ("text", "number", "money", "date", "time", "datetime", "duration", "list")
METRICS = ("exact_norm", "anls", "token_f1", "numeric_tol", "date_exact", "list_f1", "edit_sim")
MODALITIES = ("text", "image", "text+image")


def base_key(key: str) -> str:
    """'emails~anchor' -> 'emails' (an alternative item set inside a dataset dir; see items_file)."""
    return key.split("~", 1)[0]


def dataset_dir(key: str, root: Path | None = None) -> Path:
    return Path(root or DATASETS) / base_key(key)


def items_file(key: str, root: Path | None = None) -> Path:
    """KEY -> datasets/KEY/items.jsonl. KEY~SUB -> the single *.jsonl in datasets/KEY/SUB/ (e.g. emails~anchor =
    anchor/items_all1000.jsonl, the 1,000-email replication anchor). Its preds live under runs/RUN/preds/ARM/KEY~SUB/,
    so they never mix with the sample's."""
    d = dataset_dir(key, root)
    if "~" not in key:
        return d / "items.jsonl"
    sub = d / key.split("~", 1)[1]
    fs = sorted(sub.glob("*.jsonl")) if sub.is_dir() else []
    if len(fs) != 1:
        raise FileNotFoundError(f"{key}: expected exactly one .jsonl in {sub}, found {len(fs)}")
    return fs[0]


def load_items(key: str, root: Path | None = None, limit: int | None = None) -> list[dict]:
    f = items_file(key, root)
    items = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    for it in items:
        validate_item(it)
        apply_hints(it)
    return items[:limit] if limit else items


# Dataset-level number locale for money fields (CARD rule, applied in memory; items.jsonl is never edited).
# CORD: Indonesian receipts print 45.500 = 45500 (cord CARD "Indonesian thousands rule"); without the hint a verbatim
# prediction "45.500" parses as 45.5 and a correct answer scores 0 (critic pass 2026-10-23).
NUMBER_LOCALE_DEFAULT = {"cord": "dot_thousands"}
_LIST_ITEM_MAP = {"text": "text", "number": "number", "money": "number", "date": "date"}


def apply_hints(it: dict) -> dict:
    """Fill field hints that a builder recorded in meta but not on the field (in memory, never overwriting a key the
    field already has). Requests (Clairic `items`, OpenRouter schema) and scoring both read the same fields.
      list `items`:  meta.field_meta[name].item_type (omniai: 458 number lists) or meta.list_item_types[name]
                     (cord: menu_cnt number, menu_price money -> number). Without it a correct typed answer 16500 is
                     scored as text against the gold '16,500' and fails, although the instruction asks for a number.
      number_locale: NUMBER_LOCALE_DEFAULT[dataset] on money fields and on number/money list fields."""
    m = it.get("meta") or {}
    fm, lit = m.get("field_meta") or {}, m.get("list_item_types") or {}
    loc = NUMBER_LOCALE_DEFAULT.get(base_key(it.get("dataset", "")))
    for f in it["fields"]:
        if f["type"] == "list" and not f.get("items"):
            src = (fm.get(f["name"]) or {}).get("item_type") if isinstance(fm.get(f["name"]), dict) else None
            src = src or lit.get(f["name"])
            if _LIST_ITEM_MAP.get(src):
                f["items"] = _LIST_ITEM_MAP[src]
        if loc and "number_locale" not in f and (f["type"] == "money" or (f["type"] == "list" and f.get("items") == "number")):
            f["number_locale"] = loc
        uh = (m.get("unit_hint") or {}).get(f["name"]) if isinstance(m.get("unit_hint"), dict) else None
        if f["type"] == "money" and not f.get("unit") and isinstance(uh, str):
            f["unit"] = uh        # cord CARD: "Pass it as Clairic's unit" (IDR); otherwise Clairic gets "currency"
    return it


def list_datasets(root: Path | None = None) -> list[str]:
    r = Path(root or DATASETS)
    return sorted(p.name for p in r.iterdir() if (p / "items.jsonl").exists()) if r.exists() else []


def validate_item(it: dict) -> None:
    """Cheap structural check; raises ValueError naming the item and the problem."""
    def bad(msg):
        raise ValueError(f"{it.get('dataset')}/{it.get('id')}: {msg}")
    for k in ("id", "dataset", "use_case", "modality", "text", "text_source", "images", "fields", "metric", "meta"):
        if k not in it:
            bad(f"missing key {k}")
    if it["modality"] not in MODALITIES:
        bad(f"modality {it['modality']}")
    if it["metric"] not in METRICS:
        bad(f"metric {it['metric']}")
    if len(it["images"]) > 4:
        bad("more than 4 images")
    if not it["fields"]:
        bad("no fields")
    for f in it["fields"]:
        if not FIELD_NAME.match(f.get("name", "")):
            bad(f"field name {f.get('name')!r}")
        if f.get("type") not in FIELD_TYPES:
            bad(f"field {f['name']} type {f.get('type')!r}")


def load_split(version: str = BENCH_VERSION) -> dict:
    """splits/VERSION.json -> {dataset: {"eval": [ids], "fit": [ids]}}: the items a board version is scored on."""
    return json.loads((SPLITS / f"{version}.json").read_text())["datasets"]


def is_holdout(item: dict) -> bool:
    """Sealed 20% holdout per set: meta.holdout if the builder set it, else a seed-20 hash of the id."""
    h = item.get("meta", {}).get("holdout")
    if h is not None:
        return bool(h)
    d = hashlib.sha256(f"{SEED}:{item['dataset']}:{item['id']}".encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64 < HOLDOUT_FRAC


# Bootstrap cluster key per dataset, from each CARD (verify passes 2026-10-23). Default: meta.doc_id (the source
# document; several DocVQA questions share a page, several RealDoc questions share a PDF), else the item.
CLUSTER_KEY = {
    "kleister_charity": "charity_cluster",   # several yearly reports per charity share name/number/address gold
    "cord": "cluster_id",                    # 5 duplicate receipt pairs -> 195 independent receipts
    "fatura": "template",                    # 10 held-out templates; CARD: template-clustered CIs
}


def cluster_of(item: dict) -> str:
    m = item.get("meta", {})
    k = CLUSTER_KEY.get(item.get("dataset"))
    v = m.get(k) if k else None
    if v is None:
        v = m.get("doc_id") or m.get("doc") or item["id"]
    return f"{k}:{v}" if k and m.get(k) is not None else str(v)


ALT_META_TEXT = ("oracle_text",)   # CORD gold-word oracle (never call it an OCR ceiling; CARD)


def alt_text(item: dict, ds_dir: Path, variant: str) -> str | None:
    """Named alternative text. Looked up in meta.alt_texts (kleister @tesseract_layer/@textract_layer/@djvu_layer/
    @best_layer, realdoc @pdf_text_layer), then meta.ocr_alternatives (omniai @gold_markdown / @aws-textract /
    @gemini-2.0-flash-001 / @azure-document-intelligence), then meta[variant] for ALT_META_TEXT (cord @oracle_text).
    The value is the text itself or a path relative to the dataset dir. None = this item has no such text."""
    m = item.get("meta", {})
    v = (m.get("alt_texts") or {}).get(variant)
    if v is None:
        v = (m.get("ocr_alternatives") or {}).get(variant)
    if v is None and variant in ALT_META_TEXT and isinstance(m.get(variant), str):
        return m[variant]
    if v is None:
        return None
    p = ds_dir / v
    if len(v) < 300 and "\n" not in v and p.is_file():
        return p.read_text()
    return v


# ---------------------------------------------------------------- tokens
_ENC = None


def enc():
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001  (no tiktoken: fall back to ~4 chars/token, flagged in outputs)
            _ENC = False
    return _ENC


def n_tokens(text: str | None) -> int:
    if not text:
        return 0
    e = enc()
    return len(e.encode(text, disallowed_special=())) if e else (len(text) + 3) // 4


def tokens_exact() -> bool:
    return bool(enc())


def windows(text: str, size: int, overlap: int = 200) -> list[str]:
    """Token windows (o200k) with a 200-token overlap."""
    e = enc()
    if not e:
        cs, co = size * 4, overlap * 4
        return [text] if len(text) <= cs else [text[i:i + cs] for i in range(0, len(text), cs - co)]
    toks = e.encode(text, disallowed_special=())
    if len(toks) <= size:
        return [text]
    out, i = [], 0
    while i < len(toks):
        out.append(e.decode(toks[i:i + size]))
        i += size - overlap
    return out


def write_json(f: Path, obj) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=str, ensure_ascii=False))
    tmp.replace(f)
