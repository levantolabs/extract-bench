"""VERBATIM copies of RealDoc-Bench template building + scoring (generated; do not edit by hand).

Source: https://github.com/extend-hq/realdoc-bench at commit fb26a6876481de76dc293f722ab4efa71279904d (Apache-2.0).
Regenerate with scripts/vendor_official.py; check with `verify()` below. Only the header, the guarded
rapidfuzz import and verify() are ours. fuzzy_equal() needs rapidfuzz (scoring time only).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

try:
    from rapidfuzz import fuzz
except ImportError:  # build-time only; fuzzy_equal needs it
    fuzz = None

# ---- verbatim from realdoc_bench/evaluate/jsonify.py (sha256 fdff87223fb89475de8aa945cf26327d585321969c687de94c8e1090ec50e58a)
_INT_LIKE = {"digits", "digit", "year", "qty", "quantity", "count", "integer", "int"}

_NUMBER_LIKE = {"amount", "money", "cost", "price", "fee", "balance", "total",
                "subtotal", "premium", "payment", "deposit", "percent",
                "percentage", "rate", "decimal", "float", "number"}

_DATE_LIKE = {"date", "mm/dd/yyyy", "mm-dd-yyyy", "yyyy-mm-dd", "yyyy/mm/dd"}

_STRING_LIKE = {"string", "str", "text", "value", "val", "name"}

_BOOL_LIKE = {"boolean", "bool"}

_BOOL_PAIRS = {("true", "false")}

def _placeholder_to_type_hint(inner: str) -> str:
    raw = inner.strip()
    options = [o.strip() for o in raw.split("|") if o.strip()]
    lower = [o.lower() for o in options]
    nullable = "null" in lower
    non_null = [o for o, lo in zip(options, lower) if lo != "null"]
    non_null_lo = [o.lower() for o in non_null]

    def with_null(t: str) -> str:
        return f"{t} | null" if nullable else t

    if len(non_null) == 1:
        tok = non_null_lo[0]
        if tok in _BOOL_LIKE:
            return with_null("boolean")
        if tok in _INT_LIKE:
            return with_null("integer")
        if tok in _NUMBER_LIKE:
            return with_null("number")
        if tok in _DATE_LIKE:
            return with_null("date string YYYY-MM-DD")
        if tok in _STRING_LIKE:
            return with_null("string")
        return with_null("string")
    if non_null:
        tup = tuple(non_null_lo)
        if tup in _BOOL_PAIRS or tuple(reversed(tup)) in _BOOL_PAIRS:
            return with_null("boolean")
        seen, uniq = set(), []
        for o in non_null:
            if o.lower() not in seen:
                seen.add(o.lower()); uniq.append(o)
        return with_null("one of: " + " | ".join(uniq))
    return with_null("string")

def render_template(key_types: dict[str, str]) -> str:
    lines = ["{"]
    visible = [(k, t) for k, t in key_types.items() if not k.startswith("__")]
    for i, (k, t) in enumerate(visible):
        comma = "," if i < len(visible) - 1 else ""
        lines.append(f'  {json.dumps(k)}: <{t}>{comma}')
    lines.append("}")
    return "\n".join(lines)

_DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y", "%m-%d-%Y",
                 "%m.%d.%Y", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%y", "%m-%d-%y",
                 "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
                 "%B %d %Y", "%b %d %Y"]

def _to_iso_date(text: str) -> str | None:
    t = re.sub(r"\bSept\b", "Sep", text.strip().rstrip(".,;:"))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def widen_types_from_gold(key_types: dict[str, str], gold: dict[str, Any]) -> None:
    """Widen a key's type when the gold value doesn't fit the declared type so
    the template never forces the model into a format the gold can't match."""
    for k, v in list(gold.items()):
        ht = key_types.get(k, "")
        base = ht.replace(" | null", "").strip()
        nullable = "null" in ht
        if base in {"integer", "number"} and isinstance(v, str) and v.strip():
            iso = _to_iso_date(v)
            if iso:
                key_types[k] = "date string YYYY-MM-DD" + (" | null" if nullable else "")
                gold[k] = iso
            else:
                key_types[k] = "string | null" if nullable else "string"
        elif base.startswith("date") and isinstance(v, str) and \
                not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            key_types[k] = "string | null" if nullable else "string"
        elif base.startswith("one of:") and isinstance(v, str):
            toks = {t.strip().lower()
                    for t in base[len("one of:"):].split("|") if t.strip()}
            if v.lower() not in toks:
                key_types[k] = "string | null" if nullable else "string"

# ---- verbatim from realdoc_bench/evaluate/score.py (sha256 337d432e52177892bcd03e1cadbdd8283b6f39e31646b2fa18a171db73f05ffb)
DEFAULT_MODEL = "gemini-3-flash-preview"

FUZZ_THRESHOLD = 92

FUZZ_MIN_WORDS = 5

SYSTEM_PROMPT = (
    "You are extracting structured answers from document-parser markdown. "
    "Read the markdown only -- never use outside knowledge. "
    "Return ONE JSON object whose keys and value types exactly match the answer "
    "template provided. The TYPE HINT inside `< >` is the source of truth for "
    "each field; the KEY NAME does not change typing or formatting. Rules: "
    "(1) numbers must be JSON numbers (no commas, no '$', no '%', no quotes); "
    "(2) only fields whose template says `<boolean>` may be JSON true / false; "
    "(3) fields whose template says `<one of: a | b | ...>` MUST return one of "
    "the listed tokens as a JSON string, copied verbatim including case; "
    "(4) ONLY fields whose type hint is `<date string YYYY-MM-DD>` get "
    "normalized to ISO YYYY-MM-DD; "
    "(5) all other fields -- including any `<string>` -- copy the page text "
    "verbatim, trimmed (do not normalize dates, numbers, capitalization or "
    "separators), EXCEPT collapse spurious OCR character-spacing: if a value "
    "is printed with stray spaces between its characters (e.g. '0 . 0 0', "
    "'1 2 3 4', 'C A T'), return its natural un-spaced form; "
    "(6) for any field whose value is absent, blank, illegible, or genuinely "
    "ambiguous in the markdown, use JSON null; "
    "(6a) treat redaction markers like `[Redacted]` as ABSENT -- return JSON "
    "null if a cell holds only a marker; drop the marker if mixed with real "
    "content. "
    "Do not include keys outside the template. Do not wrap in an 'answer' "
    "parent key beyond what the template shows. Output the JSON object only, "
    "no prose or fences."
)

_WS_AROUND_SEP = re.compile(r"\s*([/\-,:])\s*")

_WS_RUN = re.compile(r"\s+")

_DASH_RE = re.compile(r"[‐‑‒–—―−]")

_SMART_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"'})

_TAG_RE = re.compile(r"<br\s*/?>|</?[a-z][^>]*>", re.I)

_STYLE_RE = re.compile(r"\*\*|__|`")

def _norm_str(s: str) -> str:
    s = s.translate(_SMART_QUOTES)
    s = _DASH_RE.sub("-", s)
    s = _TAG_RE.sub(" ", s)
    s = _STYLE_RE.sub("", s)
    s = _WS_AROUND_SEP.sub(r"\1", s)
    return _WS_RUN.sub(" ", s).strip()

def deep_equal(a: Any, b: Any) -> bool:
    """Type-tolerant equality used as the base comparison for every key.

    Tolerates: parser-style 1-element list wrapping a dict, smart-quote /
    dash variants, parser markdown styling (`<br>`, `**`, ``), OCR
    character-spacing on numeric strings, and number-vs-numeric-string with
    currency / percent / comma formatting (`"$1,234" == 1234`).
    """
    if isinstance(a, list) and len(a) == 1 and isinstance(b, dict) and isinstance(a[0], dict):
        return deep_equal(a[0], b)
    if isinstance(b, list) and len(b) == 1 and isinstance(a, dict) and isinstance(b[0], dict):
        return deep_equal(a, b[0])
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        na, nb = _norm_str(a), _norm_str(b)
        if na == nb:
            return True
        # OCR character-spacing — equal once ALL whitespace removed.
        return re.sub(r"\s+", "", na) == re.sub(r"\s+", "", nb)
    if isinstance(a, (int, float)) and isinstance(b, str):
        num, s = a, b
    elif isinstance(b, (int, float)) and isinstance(a, str):
        num, s = b, a
    else:
        return a == b
    try:
        return float(re.sub(r"[,$%\s]", "", s)) == float(num)
    except ValueError:
        return False

def fuzzy_equal(a: Any, b: Any) -> bool:
    """Conservative rapidfuzz comparison — ONLY for plain <string> fields.
    Falls back to deep_equal first; then punctuation-insensitive equality;
    then rapidfuzz ratio above FUZZ_THRESHOLD, but only when both sides are
    multi-word (short strings are IDs/amounts where 90+ ratio would forgive
    a single wrong digit)."""
    if deep_equal(a, b):
        return True
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    ca = re.sub(r"[^0-9a-z]+", " ", a.lower()).strip()
    cb = re.sub(r"[^0-9a-z]+", " ", b.lower()).strip()
    if ca and ca == cb:
        return True
    if min(len(ca.split()), len(cb.split())) < FUZZ_MIN_WORDS:
        return False
    return bool(ca) and fuzz.ratio(ca, cb) >= FUZZ_THRESHOLD

_TYPE_RE = re.compile(r'"([^"]+)":\s*<([^>]+)>')

def string_keys(template: str) -> set[str]:
    """Keys whose template type is plain <string> — the fuzzy-eligible ones.
    Enums, numbers, booleans and dates stay exact."""
    return {k for k, h in _TYPE_RE.findall(template)
            if h.replace("| null", "").strip() == "string"}

def score_typed(answer: Any, gold_dict: dict,
                str_keys: set[str]) -> tuple[dict, bool]:
    """Per-field score: fuzzy_equal for <string> fields, deep_equal otherwise."""
    fm = {}
    for k, gv in gold_dict.items():
        av = answer.get(k) if isinstance(answer, dict) else None
        fm[k] = fuzzy_equal(av, gv) if k in str_keys else deep_equal(av, gv)
    return fm, (bool(fm) and all(fm.values()))

_RF_PREFIX = re.compile(r"^\s*Return exactly[^:]*:\s*")

_PAIR_RE = re.compile(r"([^=<>;|]+)=<([^>]+)>")

def build_template(rf: str, gold_dict: dict) -> str:
    """Turn a freeform `response_format` into a typed JSON template.

    Handles three dialects observed in the bank:
      'Return exactly: k=<tok>; k2=<tok>'            semicolon-separated
      'Return exactly (... \" || \"): k=<tok> || ...'   pipe-pair-separated
      'Return ... separated by semicolon-space'      list-style; gold is list

    If neither pattern matches, derives the shape from the gold (list → list of
    strings, scalar → string).
    """
    body = _RF_PREFIX.sub("", rf or "")
    pairs = [(k.strip(), tok) for k, tok in _PAIR_RE.findall(body)]
    if not pairs:
        kt = {k: ("list of strings" if isinstance(v, list) else "string")
              for k, v in (gold_dict or {}).items()}
        return render_template(kt or {"answer": "string"})
    kt = {k: _placeholder_to_type_hint(tok) for k, tok in pairs}
    try:
        widen_types_from_gold(kt, gold_dict or {})
    except Exception:  # noqa: BLE001 — keep going on a bad widen
        pass
    return render_template(kt)

def verify(upstream_dir) -> int:
    """Re-check that every vendored definition is a verbatim substring of the upstream files."""
    import ast as _ast, inspect as _inspect, pathlib as _pl, sys as _sys
    me = _pl.Path(__file__).read_text()
    up = ''.join((_pl.Path(upstream_dir) / f).read_text() for f in ('jsonify.py', 'score.py'))
    n = 0
    for node in _ast.parse(me).body:
        if isinstance(node, (_ast.FunctionDef, _ast.Assign)) and getattr(node, 'name', '') != 'verify':
            seg = _ast.get_source_segment(me, node)
            if seg.startswith('fuzz = None'):
                continue
            if seg not in up:
                raise AssertionError('not verbatim: ' + seg[:80])
            n += 1
    return n
