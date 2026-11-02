"""Scorers for the extraction bench (stdlib only).

These are the harness's OWN adapted scorers ("exact_norm" etc.). Any number printed next to a published
baseline must come from that benchmark's official scorer (CC-OCR kie_evaluator, VRDU matchers, Kleister
normalisation, tatqa_eval, SQuAD v2 eval, OmniAI json.ts, RealDoc deep_equal); see each dataset's CARD.md.

Field outcome per abstention rule (pre-registered before the first live run; see docs/METHODOLOGY.md):
  rule a: the arm's value is null -> abstained.   rule b: null OR confidence < 0.5 -> abstained.
  gold null & abstained  -> correct_abstain (1)   gold null & answered  -> hallucinated (0)
  gold set  & abstained  -> false_abstain (0)     gold set  & answered  -> answered (metric score)
A field the arm cannot express (e.g. Sage 2 has no list type) is `unscored`: excluded from the arm's accuracy,
counted as 0 in the coverage-adjusted score, and reported as '% gold fields scored'.
"""
from __future__ import annotations

import math, random, re, string, unicodedata
from collections import Counter, defaultdict

CONF_FLOOR = 0.5
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MONTH_FULL = {"january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7, "august": 8,
               "september": 9, "sept": 9, "october": 10, "november": 11, "december": 12}
MONTHS.update(_MONTH_FULL)


# ================================================================ normalisers
# Unicode dashes -> '-', curly quotes -> ASCII (fix 2026-10-26, audit scoring #4: RealDoc en dash vs '-').
_PUNCT_MAP = str.maketrans({"–": "-", "—": "-", "‒": "-", "−": "-", "‐": "-", "‑": "-",
                            "‘": "'", "’": "'", "‚": "'", "‛": "'",
                            "“": '"', "”": '"', "„": '"', "‟": '"'})


def norm_text(s) -> str:
    """NFKC, unicode dashes/curly quotes -> ASCII, casefold, collapse whitespace, strip surrounding punctuation/space."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).translate(_PUNCT_MAP).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:!?\"'`()[]{}")


def squash(s) -> str:
    """Alphanumerics only (the emails repo's 'alnum-squashed' comparison)."""
    return re.sub(r"[^0-9a-z]", "", norm_text(s))


_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def _strip_article(s: str) -> str:
    """Drop one leading article from an already-normalised string ('the nrsc' -> 'nrsc'; audit scoring #5)."""
    return _ARTICLE.sub("", s, count=1)


def _usquash(s: str) -> str:
    """Unicode-alphanumerics only. Unlike `squash` (ASCII [0-9a-z]) it keeps accented letters, so 'josé' != 'jos'."""
    return "".join(ch for ch in s if ch.isalnum())


def text_eq(pred, gold) -> bool:
    """Primary text equality for a text field (fix 2026-10-26, audit scoring #1/#4/#5, validity F2-F4): equal after
    norm_text, or after dropping a leading article, or squashed to alphanumerics (both non-empty). Squash keeps every
    letter and digit in order, so 'SG18 9NR' != 'SG18 9N4' and '#1884' == '1884'. The digit runs must also be the same,
    so digit groups never merge: '9341 12' != '9341 1/2', '2.03(b)' != '203(b)', while '365 94 0248' == '365-94-0248'."""
    p, g = norm_text(pred), norm_text(gold)
    if not g or not p:
        return False
    if p == g:
        return True
    p, g = _strip_article(p), _strip_article(g)
    if p == g:
        return True
    sp, sg = _usquash(p), _usquash(g)
    return bool(sp) and sp == sg and re.findall(r"\d+", p) == re.findall(r"\d+", g)


def _tokens(s) -> list[str]:
    return re.findall(r"\w+", norm_text(s))


LENIENT_MIN_TOKENS, LENIENT_RATIO = 6, 3


def text_contains(pred, gold) -> bool:
    """Lenient (secondary) text credit: the gold's word tokens appear contiguously in the prediction, and the
    prediction is at most max(6, 3 x gold) tokens. 'State of Washington' ~ 'Washington' passes; a whole governing-law
    clause does not (audit scoring #2, validity F13)."""
    pt, gt = _tokens(pred), _tokens(_strip_article(norm_text(gold)))
    if not pt or not gt or len(pt) > max(LENIENT_MIN_TOKENS, LENIENT_RATIO * len(gt)):
        return False
    n = len(gt)
    return any(pt[i:i + n] == gt for i in range(len(pt) - n + 1))


def parse_number(v, locale: str | None = None):
    """Number from int/float/str such as '$1,234.56', 'RM 193.00', '1.234,56', '(12.5)', '12.5%', '-3'.
    locale='dot_thousands' reads '1.234' as 1234 (Indonesian receipts). Returns float or None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, dict):  # e.g. {"amount": 12.5, "currency": "USD"} money shapes
        for k in ("amount", "value", "number"):
            if k in v:
                return parse_number(v[k], locale)
        return None
    s = unicodedata.normalize("NFKC", str(v)).strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")") or bool(re.match(r"^\s*-|^[^\d]*-\s*\d", s))
    s = re.sub(r"[^\d.,]", "", s)
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(",", ".")
    elif "," in s:
        parts = s.split(",")
        if len(parts) > 2 or (len(parts[-1]) == 3 and locale != "comma_decimal"):
            s = s.replace(",", "")          # 1,234 / 1,234,567 -> thousands
        else:
            s = s.replace(",", ".")         # 12,5 -> decimal comma
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2 or (locale == "dot_thousands" and len(parts[-1]) == 3):
            s = s.replace(".", "")          # 1.234.567 -> thousands
    try:
        x = float(s)
    except ValueError:
        return None
    return -abs(x) if neg else x


def _year(y: str) -> int:
    y = int(y)
    return y + (2000 if y < 50 else 1900) if y < 100 else y


def _valid(y, m, d):
    import datetime as _dt
    try:
        return _dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(v, dayfirst: bool | None = None) -> str | None:
    """ISO 'YYYY-MM-DD' from common written forms; None if unparseable. dayfirst decides 01/02/2019
    (default: day-first, unless only month-first is a valid date)."""
    if v is None:
        return None
    s = unicodedata.normalize("NFKC", str(v)).strip().casefold()
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)                  # 2019-01-15, 2019.01.15
    if m:
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", s)               # 15/01/2019, 01/15/19
    if m:
        a, b, y = int(m[1]), int(m[2]), _year(m[3])
        first = (y, b, a) if dayfirst is not False else (y, a, b)
        second = (y, a, b) if dayfirst is not False else (y, b, a)
        return _valid(*first) or _valid(*second)
    mon = r"([a-z]{3,9})\.?"
    m = re.search(rf"\b(\d{{1,2}})[ -]{mon}[ ,-]*(\d{{2,4}})\b", s)               # 15 jan 2019, 15-Jan-19
    if m and m[2][:3] in MONTHS:
        return _valid(_year(m[3]), MONTHS.get(m[2], MONTHS[m[2][:3]]), int(m[1]))
    m = re.search(rf"\b{mon}[ -](\d{{1,2}}),?[ -](\d{{2,4}})\b", s)               # january 15, 2019
    if m and m[1][:3] in MONTHS:
        return _valid(_year(m[3]), MONTHS.get(m[1], MONTHS[m[1][:3]]), int(m[2]))
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s)                                 # 20190115
    if m:
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    return None


def parse_time(v) -> str | None:
    """'HH:MM:SS' (24h) from '14:05', '2:05 pm', '14:05:09', '1405'."""
    if v is None:
        return None
    s = unicodedata.normalize("NFKC", str(v)).strip().casefold().replace(".", "")
    m = re.search(r"\b(\d{1,2})[:h](\d{2})(?::(\d{2}))?\s*(am|pm)?", s) or \
        re.search(r"\b(\d{1,2})()()\s*(am|pm)\b", s)
    if not m:
        m2 = re.fullmatch(r"(\d{2})(\d{2})", s)
        if not m2:
            return None
        h, mi, se, ap = int(m2[1]), int(m2[2]), 0, None
    else:
        h, mi, se, ap = int(m[1]), int(m[2] or 0), int(m[3] or 0), m[4]
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return f"{h:02d}:{mi:02d}:{se:02d}" if h < 24 and mi < 60 and se < 60 else None


def parse_datetime(v, dayfirst=None) -> str | None:
    if v is None:
        return None
    s = str(v)
    d = parse_date(s, dayfirst)
    rest = re.sub(r"^\s*\d{4}-\d{2}-\d{2}[t ]?", "", s, flags=re.I) if d else s
    t = parse_time(rest) or "00:00:00"
    return f"{d}T{t}" if d else None


_DUR_UNITS = {"s": 1, "sec": 1, "second": 1, "min": 60, "minute": 60, "h": 3600, "hr": 3600, "hour": 3600,
              "d": 86400, "day": 86400, "w": 604800, "wk": 604800, "week": 604800,
              "mo": 2592000, "month": 2592000, "y": 31536000, "yr": 31536000, "year": 31536000}


def parse_duration(v, unit: str | None = None) -> float | None:
    """Seconds (month = 30 d, year = 365 d) from ISO 8601 ('P2Y', 'P1Y6M', 'PT90M'), '2 years',
    'two (2) years', '18 months', or a bare number in `unit` (Clairic durations are numbers in a unit)."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) * _DUR_UNITS.get((unit or "seconds").rstrip("s").casefold(), 1)
    s = unicodedata.normalize("NFKC", str(v)).strip().casefold()
    m = re.fullmatch(r"p(?:(\d+(?:\.\d+)?)y)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)w)?(?:(\d+(?:\.\d+)?)d)?"
                     r"(?:t(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?)?", s)
    if m and any(m.groups()):
        f = [float(x or 0) for x in m.groups()]
        return f[0] * 31536000 + f[1] * 2592000 + f[2] * 604800 + f[3] * 86400 + f[4] * 3600 + f[5] * 60 + f[6]
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "eighteen": 18, "twenty": 20, "thirty": 30, "sixty": 60,
             "ninety": 90}
    tot, hit = 0.0, False
    for num, u in re.findall(r"(\d+(?:\.\d+)?|[a-z]+)\s*(?:\(\s*\d+\s*\)\s*)?([a-z]+)", s):
        n = float(num) if num[0].isdigit() else words.get(num)
        uu = u.rstrip("s") if u not in ("s",) else u
        if n is not None and uu in _DUR_UNITS:
            tot += n * _DUR_UNITS[uu]
            hit = True
    if hit:
        return tot
    x = parse_number(s)
    return None if x is None else x * _DUR_UNITS.get((unit or "seconds").rstrip("s").casefold(), 1)


# ================================================================ string metrics
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _golds(gold) -> list:
    return list(gold) if isinstance(gold, list) else [gold]


def anls(pred, gold, tau: float = 0.5) -> float:
    """DocVQA ANLS for one question: max over golds of 1 - NL if NL < tau else 0 (lower-cased, stripped)."""
    p = " ".join(str(pred or "").lower().split())
    best = 0.0
    for g in _golds(gold):
        g = " ".join(str(g).lower().split())
        nl = levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


def _squad_norm(s) -> list[str]:
    s = str(s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def token_f1(pred, gold) -> float:
    """SQuAD token F1, max over golds (SQuAD v2 normalisation)."""
    pt = _squad_norm(pred)
    best = 0.0
    for g in _golds(gold):
        gt = _squad_norm(g)
        if not pt or not gt:
            best = max(best, float(pt == gt))
            continue
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if same:
            pr, rc = same / len(pt), same / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def edit_sim(pred, gold) -> float:
    best = 0.0
    p = norm_text(pred)
    for g in _golds(gold):
        g = norm_text(g)
        best = max(best, 1 - levenshtein(p, g) / max(len(p), len(g), 1))
    return best


_NUMERIC_GOLD = re.compile(r"[(\-+]?\s*(?:[$€£¥]|rm|rp|usd|eur|gbp)?\s*[(\-+]?\s*\d[\d.,\s]*\s*"
                           r"(?:%|percent|thousand|million|billion|mn|bn|m|k)?\s*\)?", re.I)


def is_numeric_answer(s) -> bool:
    """True if `s` is a number (optionally with sign, currency, % or scale word), not a span that merely contains one:
    '$80 million', '(1,512)', '6%' -> True; 'a $2.43 billion decrease', 'July 3, 2018', 'FY18' -> False."""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return True
    return s is not None and bool(_NUMERIC_GOLD.fullmatch(norm_text(s).replace("−", "-")))


def numeric_tol(pred, gold, rel: float = 1e-4, abs_: float = 5e-3, locale=None) -> float:
    """Numbers equal within max(abs_, rel*|gold|) (percent signs dropped: '20.5%' == 20.5).
    A non-numeric gold (TAT-QA span answer) is scored with SQuAD token F1, as tatqa_eval reports F1 (fix 2026-10-26,
    audit scoring #6); a numeric gold with a non-numeric prediction falls back to text equality. Max over golds."""
    best = 0.0
    for g in _golds(gold):
        if not is_numeric_answer(g):
            best = max(best, token_f1(pred, g))
            continue
        gx, px = parse_number(g, locale), parse_number(pred, locale)
        if gx is not None and px is not None:
            if abs(px - gx) <= max(abs_, rel * abs(gx)):
                return 1.0
        elif text_eq(pred, g):
            return 1.0
    return best


def list_f1(pred, gold, item_type: str = "text", field: dict | None = None) -> float:
    """Multiset F1 over normalised list items. gold [] and pred [] -> 1."""
    def canon(x):
        if item_type == "number":
            n = parse_number(x, (field or {}).get("number_locale"))
            return round(n, 6) if n is not None else norm_text(x)
        if item_type == "date":
            return parse_date(x, (field or {}).get("dayfirst")) or norm_text(x)
        return norm_text(x)
    golds = gold if gold and isinstance(gold[0], list) else [gold]  # list of acceptable lists, or one list
    p = Counter(canon(x) for x in (pred or []) if x is not None)
    best = 0.0
    for g in golds:
        gc = Counter(canon(x) for x in (g or []))
        if not p and not gc:
            return 1.0
        same = sum((p & gc).values())
        if same:
            pr, rc = same / sum(p.values()), same / sum(gc.values())
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


# ================================================================ typed exact match
YEAR_MIN, YEAR_MAX = 1800, 2199


def _plausible(iso: str | None) -> str | None:
    """Scoring-time year sanity bound. parse_date itself is left unchanged on purpose (vrdu/build.py imports it to
    decide gold repairs, so changing it would change items.jsonl); OCR-garbage golds such as 'April 27, 193' ->
    0193-04-27 are treated as unparseable here and fall back to the text comparison (vrdu CARD verifier note)."""
    return iso if iso and YEAR_MIN <= int(iso[:4]) <= YEAR_MAX else None


_LEGAL_DAY_OF = re.compile(r"(\d)(?:st|nd|rd|th)?\s+day\s+of\s+", re.I)


def _pred_date(pred, dayfirst=None) -> str | None:
    """parse_date for a PREDICTION, with one extra written form retried when parse_date fails: the contract phrasing
    '15th day of January, 2019' / 'the 1st day of March 2010' (Kleister-NDA effective dates; parse_date returns None on
    it, so a correct verbatim span scored 0). Gold parsing is unchanged (parse_date is imported by vrdu/build.py and
    the VRDU typed ceilings depend on it). Critic pass 2026-10-23."""
    d = parse_date(pred, dayfirst)
    if d is None and pred is not None and _LEGAL_DAY_OF.search(str(pred)):
        s = re.sub(r"\s+", " ", _LEGAL_DAY_OF.sub(r"\1 ", str(pred)))
        d = parse_date(s, dayfirst)
    return d


def exact_norm(pred, gold, ftype: str = "text", field: dict | None = None) -> float:
    """1.0 if pred equals any acceptable gold after type-aware normalisation."""
    field = field or {}
    loc, df = field.get("number_locale"), field.get("dayfirst")
    for g in _golds(gold):
        if ftype in ("number", "money"):
            gx, px = parse_number(g, loc), parse_number(pred, loc)
            if gx is not None and px is not None:
                if ftype == "money":   # cents: both rounded half-up to 2 dp, then < 0.005 (fix 2026-10-26, audit #9)
                    if abs(_cents(px) - _cents(gx)) < 0.005:
                        return 1.0
                elif abs(px - gx) <= max(1e-9, 1e-9 * abs(gx)):
                    return 1.0
                continue
        elif ftype == "date":
            gd, pd = _plausible(parse_date(g, df)), _plausible(_pred_date(pred, df))
            if gd and pd:
                if gd == pd:
                    return 1.0
                continue
        elif ftype == "time":
            gt, pt = parse_time(g), parse_time(pred)
            if gt and pt:
                if gt == pt or (gt.endswith(":00") and gt[:5] == pt[:5]):
                    return 1.0
                continue
        elif ftype == "datetime":
            gd, pd = _plausible(parse_datetime(g, df)), _plausible(parse_datetime(pred, df))
            if gd and pd:
                if gd == pd:
                    return 1.0
                continue
        elif ftype == "duration":
            gs, ps = parse_duration(g, field.get("unit")), parse_duration(pred, field.get("unit"))
            if gs is not None and ps is not None:
                if abs(gs - ps) <= max(1.0, 0.02 * abs(gs)):  # 2%: 12 months (360 d) == 1 year (365 d)
                    return 1.0
                continue
        if ftype == "text":
            if text_eq(pred, g):
                return 1.0
        elif norm_text(pred) == norm_text(g) and norm_text(g):
            return 1.0
    return 0.0


def _cents(x: float) -> float:
    """Round half-up to 2 dp via the shortest repr, so 914.845 -> 914.85 (float round() gives 914.84)."""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def date_exact(pred, gold, field=None) -> float:
    df = (field or {}).get("dayfirst")
    return float(any(parse_date(g, df) is not None and parse_date(g, df) == parse_date(pred, df) for g in _golds(gold)))


def score_value(pred, gold, ftype: str, metric: str, field: dict | None = None, lenient: bool = False) -> float:
    """Score a non-null prediction against a non-null gold. List fields always use list_f1.
    lenient=True (secondary score only): an exact_norm text field also scores 1 when `text_contains` holds."""
    field = field or {}
    if lenient:
        s = score_value(pred, gold, ftype, metric, field)
        if s < 1 and metric == "exact_norm" and ftype == "text" and not isinstance(pred, list):
            s = max(s, float(any(text_contains(pred, g) for g in _golds(gold))))
        return s
    if ftype == "list":
        return list_f1(pred if isinstance(pred, list) else [pred], gold, field.get("items", "text"), field)
    if isinstance(pred, list):
        pred = pred[0] if len(pred) == 1 else " ".join(map(str, pred))
    if metric == "exact_norm":
        return exact_norm(pred, gold, ftype, field)
    if metric == "anls":
        return anls(pred, gold)
    if metric == "token_f1":
        return token_f1(pred, gold)
    if metric == "numeric_tol":
        return numeric_tol(pred, gold, locale=field.get("number_locale"))
    if metric == "date_exact":
        return date_exact(pred, gold, field)
    if metric == "list_f1":
        return list_f1(pred if isinstance(pred, list) else [pred], gold, field.get("items", "text"), field)
    if metric == "edit_sim":
        return edit_sim(pred, gold)
    raise KeyError(metric)


# ================================================================ abstention + per field / item
def _is_null(v, ftype: str | None = None) -> bool:
    """None or blank string; for list fields also [] / a list of nulls (VRDU/Kleister: empty list gold = null)."""
    if ftype == "list" and isinstance(v, list):
        return all(_is_null(x) for x in v)
    return v is None or (isinstance(v, str) and not v.strip())


def outcome(pred, conf, gold, ftype, metric, field, rule: str, lenient: bool = False):
    gold_null = gold is None or (ftype == "list" and isinstance(gold, list) and not gold)
    abstained = _is_null(pred, ftype) or (rule == "b" and conf is not None and conf < CONF_FLOOR)
    if gold_null:
        return ("correct_abstain", 1.0) if abstained else ("hallucinated", 0.0)
    if abstained:
        return "false_abstain", 0.0
    return "answered", score_value(pred, gold, ftype, metric, field, lenient)


# Harness-side date-order default for sets whose builder set no `dayfirst` and whose documents are US-format
# (RealDoc: US court filings / forms, e.g. OCR 'Filed 05/21/20'). Gold is ISO either way; this only decides how a
# verbatim prediction such as '02/03/25' is read. Declared in PLAN-2026-10-23.md.
DAYFIRST_DEFAULT = {"realdoc": False}


def gold_unparseable(f: dict) -> bool:
    """A date/datetime field whose (non-null) gold has no acceptable value that parses to a plausible date, e.g. VRDU
    OCR debris '10121116', 'Auguit9.2016'. No typed answer can match it, so it is unscored (fix 2026-10-26, audit #8)."""
    g, df = f.get("gold"), f.get("dayfirst")
    if f["type"] not in ("date", "datetime") or g is None:
        return False
    parse = parse_date if f["type"] == "date" else parse_datetime
    return not any(_plausible(parse(x, df)) for x in _golds(g) if x is not None)


def score_item(item: dict, preds: dict) -> dict:
    """preds: {field_name: {'value', 'conf', 'unsupported'?}} as returned by clients.parse_*.
    Missing field or unsupported=True -> unscored."""
    out = []
    head = (item.get("meta") or {}).get("headline_fields")
    df = DAYFIRST_DEFAULT.get(item.get("dataset"))
    for f in item["fields"]:
        if df is not None and "dayfirst" not in f and f["type"] in ("date", "datetime"):
            f = {**f, "dayfirst": df}
        p = preds.get(f["name"])
        rec = {"name": f["name"], "type": f["type"], "gold_null": f["gold"] is None or (f["type"] == "list" and f["gold"] == []),
               "headline": (f["name"] in head) if head else f.get("headline", True) is not False}
        if p is None or p.get("unsupported"):
            rec.update(scored=False, reason=(p or {}).get("unsupported") or "missing")
        elif gold_unparseable(f):
            rec.update(scored=False, reason="gold_unparseable", pred=p.get("value"))
        else:
            rec.update(scored=True, pred=p.get("value"), conf=p.get("conf"))
            for rule in ("a", "b"):
                o, s = outcome(p.get("value"), p.get("conf"), f["gold"], f["type"], item["metric"], f, rule)
                rec[f"outcome_{rule}"], rec[f"score_{rule}"] = o, s
                rec[f"score_lenient_{rule}"] = outcome(p.get("value"), p.get("conf"), f["gold"], f["type"],
                                                       item["metric"], f, rule, lenient=True)[1] if s < 1 else s
            # Forced answer (full-range coverage curves): the model's value, or when it returned null its best guess
            # (Sage: top candidate in `probabilities`; Clairic: `guess`). Answering a field with no gold scores 0.
            fv = p["forced"] if "forced" in p else (p.get("value") if not _is_null(p.get("value"), f["type"]) else p.get("guess"))
            fc = p.get("forced_conf") if "forced" in p else p.get("conf")
            rec["forced_pred"], rec["forced_conf"] = fv, fc
            rec["outcome_forced"], rec["score_forced"] = outcome(fv, fc, f["gold"], f["type"], item["metric"], f, "a")
            adj = ((item.get("meta") or {}).get("gold_adjudicated") or {}).get(f["name"])
            if isinstance(adj, dict) and "gold" in adj:   # CORD audit: raw gold stays primary, adjudicated beside it
                rec["adjudicated"] = adj.get("verdict") or True
                for rule in ("a", "b"):
                    rec[f"score_adj_{rule}"] = outcome(p.get("value"), p.get("conf"), adj["gold"], f["type"],
                                                       item["metric"], f, rule)[1]
        out.append(rec)
    sc = [r for r in out if r["scored"]]
    res = {"id": item["id"], "dataset": item["dataset"], "metric": item["metric"], "fields": out,
           "n_fields": len(out), "n_scored": len(sc)}
    for rule in ("a", "b"):
        res[f"item_acc_{rule}"] = sum(r[f"score_{rule}"] for r in sc) / len(sc) if sc else None
        res[f"item_cov_adj_{rule}"] = sum(r[f"score_{rule}"] for r in sc) / len(out) if out else None
        res[f"item_all_correct_{rule}"] = bool(sc) and len(sc) == len(out) and all(r[f"score_{rule}"] >= 1.0 for r in sc)
    return res


def _len(f: dict, rule: str) -> float:
    """Lenient score of a scored field (score files written before 2026-10-26 have none: strict is used)."""
    return f.get(f"score_lenient_{rule}", f[f"score_{rule}"])


def aggregate(scored_items: list[dict]) -> dict:
    """Arm x dataset summary: field accuracy, coverage, coverage-adjusted score, abstention table, per type."""
    fields = [f for it in scored_items for f in it["fields"]]
    sc = [f for f in fields if f["scored"]]
    agg = {"n_items": len(scored_items), "n_fields": len(fields), "n_scored": len(sc),
           "pct_gold_fields_scored": len(sc) / len(fields) if fields else None}
    for rule in ("a", "b"):
        agg[f"field_acc_{rule}"] = sum(f[f"score_{rule}"] for f in sc) / len(sc) if sc else None
        agg[f"cov_adj_{rule}"] = sum(f[f"score_{rule}"] for f in sc) / len(fields) if fields else None
        agg[f"field_acc_lenient_{rule}"] = sum(_len(f, rule) for f in sc) / len(sc) if sc else None
        vals = [it[f"item_acc_{rule}"] for it in scored_items if it[f"item_acc_{rule}"] is not None]
        agg[f"item_acc_{rule}"] = sum(vals) / len(vals) if vals else None
        agg[f"all_correct_{rule}"] = sum(it[f"item_all_correct_{rule}"] for it in scored_items) / len(scored_items) if scored_items else None
        agg[f"outcomes_{rule}"] = dict(Counter(f[f"outcome_{rule}"] for f in sc))
        by = defaultdict(list)
        for f in sc:
            by[f["type"]].append(f[f"score_{rule}"])
        agg[f"by_type_{rule}"] = {t: {"n": len(v), "acc": sum(v) / len(v)} for t, v in sorted(by.items())}
    agg["unscored_reasons"] = dict(Counter(f.get("reason") for f in fields if not f["scored"]))
    hf = [f for f in fields if f.get("headline", True)]
    if len(hf) < len(fields):           # CARD-declared headline subset (kleister_nda: effective_date/jurisdiction/term)
        hs = [f for f in hf if f["scored"]]
        agg["headline"] = {"n_fields": len(hf), "n_scored": len(hs)}
        for rule in ("a", "b"):
            agg["headline"][f"field_acc_{rule}"] = sum(f[f"score_{rule}"] for f in hs) / len(hs) if hs else None
            agg["headline"][f"cov_adj_{rule}"] = sum(f[f"score_{rule}"] for f in hs) / len(hf) if hf else None
            agg["headline"][f"field_acc_lenient_{rule}"] = sum(_len(f, rule) for f in hs) / len(hs) if hs else None
    if any(f.get("adjudicated") for f in sc):
        for rule in ("a", "b"):
            agg[f"field_acc_adj_{rule}"] = sum(f.get(f"score_adj_{rule}", f[f"score_{rule}"]) for f in sc) / len(sc)
        agg["n_adjudicated"] = sum(bool(f.get("adjudicated")) for f in sc)
    nulls = [f for f in sc if f["gold_null"]]
    pres = [f for f in sc if not f["gold_null"]]
    for rule in ("a", "b"):             # never report only the pooled number (CORD: 56% of scalar golds are null)
        agg[f"present_gold_acc_{rule}"] = sum(f[f"score_{rule}"] for f in pres) / len(pres) if pres else None
        agg[f"null_gold_acc_{rule}"] = sum(f[f"score_{rule}"] for f in nulls) / len(nulls) if nulls else None
        agg[f"present_gold_acc_lenient_{rule}"] = sum(_len(f, rule) for f in pres) / len(pres) if pres else None
    agg["n_null_gold_scored"] = len(nulls)
    return agg


# ================================================================ statistics
def paired_bootstrap(a: dict, b: dict, clusters: dict | None = None, n: int = 10000, seed: int = 20,
                     alpha: float = 0.05) -> dict:
    """Doc-clustered paired bootstrap of mean(a) - mean(b) over the ids present in both.
    a, b: {id: score}; clusters: {id: cluster_id} (default: each id its own cluster)."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    cl = defaultdict(lambda: [0.0, 0.0, 0])
    for i in ids:
        c = cl[(clusters or {}).get(i, i)]
        c[0] += a[i]; c[1] += b[i]; c[2] += 1
    rows = list(cl.values())
    sa, sb, cnt = (sum(r[k] for r in rows) for k in range(3))
    obs = (sa - sb) / cnt
    rng = random.Random(seed)
    k = len(rows)
    diffs = []
    for _ in range(n):
        xa = xb = xn = 0.0
        for _ in range(k):
            r = rows[rng.randrange(k)]
            xa += r[0]; xb += r[1]; xn += r[2]
        diffs.append((xa - xb) / xn)
    diffs.sort()
    lo, hi = diffs[int(alpha / 2 * n)], diffs[min(n - 1, int((1 - alpha / 2) * n))]
    p = min(1.0, 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / n)
    return {"n": len(ids), "n_clusters": k, "mean_a": sa / cnt, "mean_b": sb / cnt, "diff": obs,
            "ci_lo": lo, "ci_hi": hi, "p_two_sided": p, "n_boot": n, "seed": seed}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm step-down: {name: p} -> {name: {'p', 'p_adj', 'reject'}}."""
    order = sorted(pvals, key=pvals.get)
    m, run, out, stop = len(order), 0.0, {}, False
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        stop = stop or run > alpha
        out[k] = {"p": pvals[k], "p_adj": run, "reject": not stop}
    return out
