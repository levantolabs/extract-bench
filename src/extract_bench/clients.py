"""Adapters: item -> exact request body -> (mock or live) call -> normalised per-field predictions.

Three systems:
  * Clairic  (@cf/cloudflare/clairic, Workers AI)    clairic_body / call_clairic
  * Sage 2   (POST {SAGE2_BASE_URL}/v1/extract)       sage2_body / call_sage2   (text only, scalar fields only)
  * OpenRouter JSON-schema LLM/VLM control            openrouter_body / call_openrouter

SAFETY: every call_* defaults to mock=True and returns a fake but well-formed response without touching the
network or reading keys. A real call needs mock=False (`run read --live`) AND env EXB_LIVE=1 AND an explicit
spend cap EXB_CAP_USD (no default). Keys come from the environment (see .env.example): CLOUDFLARE_ACCOUNT_ID +
CLOUDFLARE_API_TOKEN (Clairic), SAGE_API_KEY + SAGE2_BASE_URL (Sage 2), OPENROUTER_API_KEY (OpenRouter). Every
live call is appended to the spend ledger runs/_ledger.jsonl; the cap is checked against it before each call.

Normalised prediction per field: {"value", "conf", "p_null"?, "span"?, "text"?, "unsupported"?: reason}.
"""
from __future__ import annotations

import base64, hashlib, json, os, random, threading, time
from pathlib import Path

from .common import RUNS, n_tokens, windows

# ---------------------------------------------------------------- prices / limits
CLAIRIC_PER_TOK = 0.10e-6       # $/input token; https://developers.cloudflare.com/workers-ai/platform/pricing/
# Sage 2 extraction prices are not published yet: cost_usd is None unless both env prices are set ($ per token).
SAGE2_IN_PER_TOK = float(os.environ["SAGE2_PRICE_IN_PER_TOK"]) if os.environ.get("SAGE2_PRICE_IN_PER_TOK") else None
SAGE2_OUT_PER_TOK = float(os.environ["SAGE2_PRICE_OUT_PER_TOK"]) if os.environ.get("SAGE2_PRICE_OUT_PER_TOK") else None
SAGE_LIMIT = 16384              # Sage 2 returns 413 above this (content + field definitions)
SAGE_WINDOW = 15000             # o200k content tokens per Sage 2 call
CLAIRIC_WINDOW = 45000          # o200k content tokens per Clairic call
CLAIRIC_WHOLE = 58500           # o200k; "whole doc" arm: one window for every item in the suite (largest: a
                                # Kleister-Charity report, 58,423 o200k ~ 65.4k Clairic tokens at x1.12, right at the
                                # 65,536 context). Clairic's `truncated` flag is logged per call.
CLAIRIC_MAX_IMAGES, CLAIRIC_IMG_MAX, CLAIRIC_IMG_TOTAL = 4, 4 * 2**20, 8 * 2**20  # clairic_schema-input.json
CLAIRIC_DURATION_UNITS = ("seconds", "minutes", "hours", "days", "weeks", "months", "years")
IMG_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
LIVE_ENV = "EXB_LIVE"
CAP_ENV = "EXB_CAP_USD"
UA = "extract-bench/0.1"        # Sage 2 rejects default HTTP-library user agents
LEDGER = RUNS / "_ledger.jsonl"


class LiveCallBlocked(RuntimeError):
    ...


class BudgetExceeded(RuntimeError):
    ...


class ImageRejected(ValueError):
    ...


# ================================================================ field type maps
def clairic_field(f: dict) -> dict:
    """Item field -> Clairic field spec. money -> number, decimals 2, unit (field unit, else 'currency')."""
    t = f["type"]
    spec: dict = {"instructions": f.get("instructions") or f["name"]}
    if t == "money":
        spec.update(type="number", decimals=2, unit=f.get("unit") or "currency")
    elif t == "number":
        spec["type"] = "number"
        if f.get("integer"):
            spec["integer"] = True
        if f.get("decimals") is not None:
            spec["decimals"] = int(f["decimals"])
        if f.get("unit"):
            spec["unit"] = f["unit"]
    elif t == "duration":
        spec["type"] = "duration"
        if f.get("unit") in CLAIRIC_DURATION_UNITS:
            spec["unit"] = f["unit"]
    elif t == "list":
        spec.update(type="list", items=f.get("items") or "text")
    else:  # text, date, time, datetime map 1:1
        spec["type"] = t
    return spec


# Sage 2 typed-field spec (API 2.0.0):
#   extract = {name: {"type": text|integer|decimal|money|date|time|datetime|duration, "definition": str}}
#   every-mention list field = [ {"type": ..., "definition": ...} ]; <= 64 fields; definitions <= 4,000 chars.
#   values[name] = value | null | [values]; results[name] = FieldResult {value, text, start, end, probability,
#   probabilities} or MentionsResult {values: [{value, text, start, end, probability}]}. money = {amount, currency}.
SAGE_INSTR_KEY = "definition"
SAGE_MAX_FIELDS = 64
SAGE_SHAPE = os.environ.get("EXB_SAGE_SHAPE", "typed")   # "typed" (one call, all fields) | "question" (one call per field)
# Pre-registered per dataset in the CARDs (independent of the unknown typed format): one free-text question per item,
# sent as Sage's bare-question `extract` string. Item meta.sage_shape wins over this; EXB_SAGE_SHAPE is the default.
SAGE_SHAPE_BY_DATASET = {"docvqa": "question", "squad2": "question"}


def sage_shape_for(item: dict) -> str:
    return (item.get("meta") or {}).get("sage_shape") or SAGE_SHAPE_BY_DATASET.get(item.get("dataset")) or SAGE_SHAPE


def na_for(f: dict, system: str) -> bool:
    """Field-level opt-out declared by the builder (e.g. TAT-QA computed answers: na_arms=['sage2'])."""
    return system in (f.get("na_arms") or [])


def sage_unsupported(f: dict) -> str | None:
    if na_for(f, "sage2"):
        return "na_arms:sage2"
    return None                        # 2.0.0 (23 Oct) has every-mention lists, so lists are scored too


def sage_type(f: dict) -> str | None:
    t = f["type"]
    if sage_unsupported(f):
        return None
    if t == "number":
        return "integer" if f.get("integer") else "decimal"
    if t == "list":
        it = f.get("items", "text")
        return "list:" + ({"number": "integer" if f.get("integer") else "decimal"}.get(it) or
                          (it if it in ("text", "money", "date", "time", "datetime", "duration") else "text"))
    return t                           # text, money, date, time, datetime, duration 1:1


def sage_spec(f: dict):
    """One field's Sage 2.0.0 spec: {type, definition}, or [{type, definition}] for an every-mention list."""
    st = sage_type(f)
    d = (f.get("instructions") or f["name"])[:4000]
    if st.startswith("list:"):
        return [{"type": st[5:], SAGE_INSTR_KEY: d}]
    return {"type": st, SAGE_INSTR_KEY: d}


def sage_question(f: dict) -> str:
    """Bare-question form: <=256 chars, no quotes / line breaks (API rule seen 20 Oct, readers.py)."""
    q = f.get("instructions") or f["name"]
    return q.replace('"', "'").replace("\n", " ").strip()[:256]


# ================================================================ images
def _downscale(b: bytes, max_edge: int) -> tuple[bytes, str] | None:
    """Longest edge -> max_edge (JPEG q92) when larger; None if already small enough. Needs Pillow (uv --with pillow)."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(b))
    if max(im.size) <= max_edge:
        return None
    s = max_edge / max(im.size)
    im = im.convert("RGB").resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    out = io.BytesIO(); im.save(out, "JPEG", quality=92)
    return out.getvalue(), "image/jpeg"


def load_image(path: Path, max_edge: int | None = None) -> dict:
    ct = IMG_TYPES.get(path.suffix.lower())
    if not ct:
        raise ImageRejected(f"{path.name}: not PNG/JPEG/WebP")
    b = path.read_bytes()
    if max_edge:   # Clairic rejects > ~4096^2 px with code 3030 (23 Oct live run); declared retry only
        r = _downscale(b, max_edge)
        if r:
            b, ct = r
    if len(b) > CLAIRIC_IMG_MAX:
        raise ImageRejected(f"{path.name}: {len(b)} bytes > 4 MiB")
    magic = {"image/png": b"\x89PNG", "image/jpeg": b"\xff\xd8", "image/webp": b"RIFF"}[ct]
    if not b.startswith(magic):
        raise ImageRejected(f"{path.name}: content does not match {ct}")
    return {"content_type": ct, "base64": base64.b64encode(b).decode(), "bytes": len(b)}


def load_images(item: dict, ds_dir: Path, max_edge: int | None = None) -> list[dict]:
    paths = item.get("images") or []
    if len(paths) > CLAIRIC_MAX_IMAGES:
        raise ImageRejected(f"{len(paths)} images > {CLAIRIC_MAX_IMAGES}")
    imgs = [load_image(ds_dir / p, max_edge) for p in paths]
    if sum(i["bytes"] for i in imgs) > CLAIRIC_IMG_TOTAL:
        raise ImageRejected("images > 8 MiB total")
    return imgs


# ================================================================ request bodies
def clairic_body(item: dict, state: str | None, images: list[dict] | None = None,
                 return_state_text: bool = False) -> dict:
    body = {"model": "clairic", "state": state or "",
            "fields": {f["name"]: clairic_field(f) for f in item["fields"]}}
    if images:
        body["images"] = [{"content_type": i["content_type"], "base64": i["base64"]} for i in images]
    if not body["state"] and not images:
        raise ValueError(f"{item['id']}: Clairic needs state text or images")
    rd = item.get("meta", {}).get("reference_date")
    if rd:
        body["reference_date"] = rd
    if return_state_text:
        body["options"] = {"return_state_text": True}
    return body


def sage2_body(item: dict, content: str, mode: str, field: dict | None = None) -> dict:
    """typed shape: extract = {name: {type, <SAGE_INSTR_KEY>}} for every Sage-expressible field.
    question shape (field given): extract = the bare question string for that one field."""
    if mode not in ("fast", "precise"):
        raise ValueError(mode)
    if field is not None:   # bare question; a one-question list asks for every answer (2.0.0)
        q = sage_question(field)
        return {"content": content, "extract": [q] if field["type"] == "list" else q, "mode": mode}
    ext = {f["name"]: sage_spec(f) for f in item["fields"] if sage_type(f)}
    if not ext:
        raise ValueError(f"{item['id']}: no Sage-expressible fields")
    if len(ext) > SAGE_MAX_FIELDS:
        raise ValueError(f"{item['id']}: {len(ext)} fields > {SAGE_MAX_FIELDS}")
    return {"content": content, "extract": ext, "mode": mode}


def json_schema(item: dict) -> dict:
    """Strict JSON schema: every field required, nullable; null = not in the document."""
    def prop(f):
        t, d = f["type"], (f.get("instructions") or f["name"])
        if t == "list":
            it = {"number": "number"}.get(f.get("items"), "string")
            return {"type": ["array", "null"], "items": {"type": it}, "description": d}
        if t in ("number", "money"):
            extra = " Number only, no currency symbol or thousands separators." if t == "money" else ""
            return {"type": ["number", "null"], "description": d + extra}
        fmt = {"date": " Format YYYY-MM-DD.", "time": " Format HH:MM:SS (24h).",
               "datetime": " Format YYYY-MM-DDTHH:MM:SS.", "duration": " ISO 8601 duration, e.g. P2Y or P18M."}
        return {"type": ["string", "null"], "description": d + fmt.get(t, " Copy the text exactly as written.")}
    return {"type": "object", "additionalProperties": False, "required": [f["name"] for f in item["fields"]],
            "properties": {f["name"]: prop(f) for f in item["fields"]}}


OR_SYSTEM = ("You extract typed values from a document. Return only JSON matching the schema. "
             "Use null when the document does not contain the value. Do not guess.")


def native_prompt(item: dict, ds_dir: Path | None) -> tuple[str, dict | None] | None:
    """The benchmark's own prompt for an OpenRouter replication arm (view '#comparable'): meta.comparable.prompt is
    the prompt text (CC-OCR SROIE/CORD, per image) or a path to it relative to the dataset dir (emails prompt2, with
    '{body}' filled from the text); meta.comparable.schema optionally points at a JSON schema. None if absent."""
    c = (item.get("meta") or {}).get("comparable")
    if not isinstance(c, dict) or not isinstance(c.get("prompt"), str):
        return None
    pr, schema = c["prompt"], None
    if ds_dir is not None and len(pr) < 300 and "\n" not in pr and (ds_dir / pr).is_file():
        pr = (ds_dir / pr).read_text()
    if ds_dir is not None and isinstance(c.get("schema"), str) and (ds_dir / c["schema"]).is_file():
        schema = json.loads((ds_dir / c["schema"]).read_text())
    return pr, schema


def openrouter_native_body(item: dict, model: str, text: str | None, images: list[dict] | None,
                           prompt: str, schema: dict | None) -> dict:
    """Verbatim benchmark prompt, no harness system prompt. '{body}' -> text; otherwise text (if any) is appended.
    A given schema goes in as a non-strict json_schema (prompt2's schema has no additionalProperties:false)."""
    parts = [{"type": "image_url", "image_url": {"url": f"data:{im['content_type']};base64,{im['base64']}"}}
             for im in images or []]
    if "{body}" in prompt:
        msg = prompt.replace("{body}", text or "")
    else:
        msg = prompt + (f"\n\n{text}" if text else "")
    parts.append({"type": "text", "text": msg})
    body = {"model": model, "messages": [{"role": "user", "content": parts}], "max_tokens": 2000,
            "usage": {"include": True}}
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "native", "strict": False, "schema": schema}}
    if not model.startswith("anthropic/"):
        body["temperature"] = 0
    return body


def openrouter_body(item: dict, model: str, text: str | None, images: list[dict] | None = None) -> dict:
    parts = []
    for im in images or []:
        parts.append({"type": "image_url", "image_url": {"url": f"data:{im['content_type']};base64,{im['base64']}"}})
    parts.append({"type": "text", "text": f"Document:\n{text}" if text else "The document is the image(s) above."})
    body = {"model": model, "messages": [{"role": "system", "content": OR_SYSTEM}, {"role": "user", "content": parts}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "extraction", "strict": True, "schema": json_schema(item)}},
            "max_tokens": int(os.environ.get("EXB_OR_MAX_TOKENS", "2000")), "usage": {"include": True}}
    if not model.startswith("anthropic/"):  # Haiku 5.5 on OpenRouter lists no temperature parameter
        body["temperature"] = 0
    return body


# ================================================================ response parsers
def parse_clairic(item: dict, resp: dict) -> dict:
    res = resp.get("result", resp) if isinstance(resp, dict) else {}
    ans = (res or {}).get("answers") or {}
    out = {}
    for f in item["fields"]:
        a = ans.get(f["name"])
        if a is None:
            continue
        v = a.get("value")
        if f["type"] == "duration" and isinstance(v, (int, float)):
            v = f"{v} {clairic_field(f).get('unit', 'seconds')}"   # number in the field's unit -> scorer string
        out[f["name"]] = {"value": v, "conf": a.get("confidence"), "p_null": a.get("p_null"),
                          "guess": a.get("guess"), "span": a.get("span") or a.get("spans")}
    return out


def _sage_one(r) -> dict:
    r = r or {}
    if isinstance(r.get("values"), list):          # MentionsResult (every-mention list field)
        ms = [m for m in r["values"] if isinstance(m, dict)]
        ps = [m.get("probability") for m in ms if m.get("probability") is not None]
        return {"value": [m.get("value") for m in ms], "conf": (sum(ps) / len(ps)) if ps else None, "p_absent": None,
                "text": [m.get("text") for m in ms], "span": [[m.get("start"), m.get("end")] for m in ms],
                "probs": [m.get("probability") for m in ms]}
    pa = next((c.get("probability") for c in r.get("probabilities") or [] if c.get("value") is None), None)
    # Every non-null candidate Sage ranked, kept even when it returns null: the top one is its forced answer, which the
    # full-range coverage curves need (scripts/forced_coverage_2026-10-25.py). Fast mode lists its candidates here
    # without any diagnostics header; precise returns only the null entry, so it has no forced answer.
    cands = [{"value": c.get("value"), "probability": c.get("probability"), "text": c.get("text"),
              "span": [c.get("start"), c.get("end")]} for c in r.get("probabilities") or [] if c.get("value") is not None]
    top = max(cands, key=lambda c: c["probability"] or 0) if cands else None
    forced, fconf = ((r.get("value"), r.get("probability")) if r.get("value") is not None
                     else ((top["value"], top["probability"]) if top else (None, None)))
    return {"value": r.get("value"), "conf": r.get("probability"), "p_absent": pa, "text": r.get("text"),
            "span": [r.get("start"), r.get("end")], "parse_error": r.get("parse_error"),
            "candidates": cands, "forced": forced, "forced_conf": fconf}


def parse_sage2(item: dict, resp: dict, field: dict | None = None) -> dict:
    """Typed: values {name: v}, results {name: {...}} (shape ASSUMED for the new version).
    Question: values = scalar, results = one {...} (seen 20 Oct)."""
    vals, res = resp.get("values"), resp.get("results")
    if field is not None:
        one = _sage_one(res if isinstance(res, dict) else None)
        one["value"] = vals if not isinstance(vals, dict) else one["value"]
        return {field["name"]: one}
    out = {}
    for f in item["fields"]:
        if not sage_type(f):
            out[f["name"]] = {"value": None, "conf": None, "unsupported": sage_unsupported(f)}
            continue
        r = (res or {}).get(f["name"]) if isinstance(res, dict) else None
        one = _sage_one(r)
        if isinstance(vals, dict) and f["name"] in vals:
            one["value"] = vals[f["name"]]
        if r is not None or (isinstance(vals, dict) and f["name"] in vals):
            out[f["name"]] = one
    return out


def parse_openrouter(item: dict, text: str | None) -> dict:
    try:
        s = (text or "").strip()
        if s.startswith("```"):
            s = s.strip("`").split("\n", 1)[1] if "\n" in s else s
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
    except Exception:  # noqa: BLE001
        return {}
    return {f["name"]: {"value": obj.get(f["name"]), "conf": None} for f in item["fields"] if f["name"] in obj}


# ================================================================ mock responses (deterministic, labelled)
def _rng(*key) -> random.Random:
    return random.Random(int(hashlib.sha256(":".join(map(str, key)).encode()).hexdigest()[:12], 16))


def _mock_value(f, rng):
    """70% first gold, 15% null, 15% a wrong value. Mock scores are meaningless; they exercise the plumbing."""
    g = f["gold"][0] if isinstance(f["gold"], list) and f["type"] != "list" and f["gold"] else f["gold"]
    u = rng.random()
    if u < 0.15:
        return None
    if u < 0.30 or g is None:
        return {"list": ["MOCK"], "number": 7, "money": 7.77, "date": "1999-12-31", "time": "12:34:00",
                "datetime": "1999-12-31T12:34:00", "duration": "P9D"}.get(f["type"], "MOCK-WRONG")
    return g


def mock_clairic(item, body, arm):
    rng = _rng("clairic", arm, item["id"])
    ans = {}
    for f in item["fields"]:
        v = _mock_value(f, rng)
        if f["type"] == "duration" and v is not None:
            v = 1.0
        ans[f["name"]] = {"type": body["fields"][f["name"]]["type"], "value": v, "guess": v,
                          "p_null": 0.9 if v is None else 0.01, "confidence": round(rng.uniform(0.3, 1.0), 3)}
    st = n_tokens(body.get("state"))
    n_img = len(body.get("images") or [])
    per = int(st * 1.12) + 1000 * n_img + 130
    usage = {"state_tokens": int(st * 1.12), "image_tokens": 1000 * n_img, "schema_tokens": 130 * len(ans),
             "input_tokens": per * len(ans), "output_tokens": 0, "passes": len(ans)}
    return {"success": True, "result": {"model": "clairic", "answers": ans, "truncated": False, "usage": usage},
            "_mock": True}


def mock_sage2(item, body, arm):
    rng = _rng("sage2", arm, item["id"], json.dumps(body["extract"], sort_keys=True)[:200])
    if isinstance(body["extract"], list) and body["extract"] and isinstance(body["extract"][0], str):
        f = next(f for f in item["fields"] if sage_question(f) == body["extract"][0])
        v = _mock_value(f, rng)
        v = v if isinstance(v, list) else ([] if v is None else [v])
        p = round(rng.uniform(0.3, 1.0), 3)
        vals, res = v, {"values": [{"value": x, "text": str(x), "start": 0, "end": 0, "probability": p} for x in v]}
    elif isinstance(body["extract"], str):
        f = next(f for f in item["fields"] if sage_question(f) == body["extract"])
        v = _mock_value(f, rng)
        p = round(rng.uniform(0.3, 1.0), 3)
        vals, res = v, {"value": v, "text": None if v is None else str(v), "start": 0, "end": 0, "probability": p,
                        "probabilities": [{"value": v, "probability": p}, {"value": None, "probability": 1 - p}]}
    else:
        vals, res = {}, {}
        for name in body["extract"]:
            f = next(x for x in item["fields"] if x["name"] == name)
            v = _mock_value(f, rng)
            p = round(rng.uniform(0.3, 1.0), 3)
            if isinstance(body["extract"][name], list):       # every-mention list
                v = v if isinstance(v, list) else ([] if v is None else [v])
                vals[name] = v
                res[name] = {"values": [{"value": x, "text": str(x), "start": 0, "end": 0, "probability": p} for x in v]}
                continue
            vals[name] = v
            res[name] = {"value": v, "text": None if v is None else str(v), "start": 0, "end": 0, "probability": p,
                         "probabilities": [{"value": v, "probability": p}, {"value": None, "probability": 1 - p}]}
    tin = n_tokens(body["content"]) + 180 + n_tokens(json.dumps(body["extract"]))
    return {"values": vals, "results": res, "_mock": True,
            "meta": {"model": "levanto-sage-2", "mode": body["mode"],
                     "usage": {"input_tokens": tin, "output_tokens": 164 * max(1, len(vals) if isinstance(vals, dict) else 1)},
                     "latency_ms": 250}}


def mock_openrouter(item, body, arm):
    rng = _rng("or", arm, item["id"])
    obj = {f["name"]: _mock_value(f, rng) for f in item["fields"]}
    user = body["messages"][-1]["content"]
    txt = "".join(p.get("text", "") for p in user if p["type"] == "text")
    n_img = sum(p["type"] == "image_url" for p in user)
    return {"_mock": True, "model": body["model"], "choices": [{"message": {"content": json.dumps(obj)}}],
            "usage": {"prompt_tokens": n_tokens(txt) + 300 + 1100 * n_img, "completion_tokens": 30 * len(obj),
                      "cost": None}}


# ================================================================ live plumbing (lazy; never imported in mock)
_spend = {"usd": None}
_lock = threading.Lock()


def require_live():
    if os.environ.get(LIVE_ENV) != "1":
        raise LiveCallBlocked(f"live call refused: set {LIVE_ENV}=1 (and pass --live)")
    if not os.environ.get(CAP_ENV):
        raise LiveCallBlocked(f"live call refused: set {CAP_ENV}=<$ ceiling for this run>")


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise LiveCallBlocked(f"live call refused: {name} is not set (see .env.example)")
    return v


_HTTP: dict = {"client": None}
_OR_SEM = threading.Semaphore(int(os.environ.get("EXB_OR_CONC", "4")))   # concurrent OpenRouter requests


def _client():
    if _HTTP["client"] is None:
        import httpx
        _HTTP["client"] = httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0),
                                       limits=httpx.Limits(max_connections=32))
    return _HTTP["client"]


def lane_spend() -> float:
    """Total live spend recorded in runs/_ledger.jsonl (all runs: the cap is cumulative)."""
    with _lock:
        if _spend["usd"] is None:
            tot = 0.0
            if LEDGER.exists():
                with open(LEDGER) as fh:
                    for line in fh:
                        try:
                            tot += float(json.loads(line).get("cost_usd") or 0)
                        except Exception:  # noqa: BLE001
                            pass
            _spend["usd"] = tot
        return _spend["usd"]


def _check_cap():
    cap = float(os.environ[CAP_ENV])
    if lane_spend() >= cap:
        raise BudgetExceeded(f"spend ${lane_spend():.3f} >= cap ${cap:.2f} (runs/_ledger.jsonl); stopping")


def _ledger(provider, model, endpoint, ok, status, in_tok, out_tok, cost, ms, n_q=1, n_images=0, tag="",
            server_ms=None, version=None):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "provider": provider, "model": model, "endpoint": endpoint,
           "ok": bool(ok), "status": status, "in_tok": in_tok, "out_tok": out_tok, "cost_usd": cost, "wall_ms": ms,
           "server_ms": server_ms, "n_questions": n_q, "n_images": n_images, "tag": tag, "version": version}
    with _lock:
        _spend["usd"] = (_spend["usd"] or 0.0) + (cost or 0.0)
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


VERSION_HEADERS = ("x-model-version", "x-build", "x-sage-version", "cf-ray", "x-request-id", "x-generation-id", "server")


def _post(url, body, headers, timeout=120.0, tries=4, retry=(429, 500, 502, 503, 504, 520, 522, 524)):
    """POST with retries on transient statuses; also returns version-ish response headers (logged per call)."""
    last = None
    for a in range(tries):
        t0 = time.perf_counter()
        try:
            r = _client().post(url, json=body, headers=headers, timeout=timeout)
            ms = (time.perf_counter() - t0) * 1000
            hv = {k: r.headers[k] for k in VERSION_HEADERS if k in r.headers}
            if r.status_code in retry:
                last = (r.status_code, r.text[:500], ms, hv)
                time.sleep(2 * (a + 1) + a * a)
                continue
            try:
                js = r.json()
            except Exception:  # noqa: BLE001
                js = r.text[:800]
            return r.status_code, js, ms, hv
        except Exception as e:  # noqa: BLE001
            last = ("EXC", repr(e)[:300], (time.perf_counter() - t0) * 1000, {})
            time.sleep(2 * (a + 1))
    return last


# ================================================================ calls
def call_clairic(item, body, arm="clairic", mock=True, tag=""):
    n_img = len(body.get("images") or [])
    if mock:
        js, st, ms, hv = mock_clairic(item, body, arm), 200, 0.0, {"mock": "1"}
    else:
        require_live(); _check_cap()
        url = f"https://api.cloudflare.com/client/v4/accounts/{_env('CLOUDFLARE_ACCOUNT_ID')}/ai/run/@cf/cloudflare/clairic"
        st, js, ms, hv = _post(url, body, {"Authorization": f"Bearer {_env('CLOUDFLARE_API_TOKEN')}", "User-Agent": UA})
    res = js.get("result") if isinstance(js, dict) else None
    ok = st == 200 and isinstance(res, dict) and "answers" in res
    u = (res or {}).get("usage") or {}
    cost = (u.get("input_tokens") or 0) * CLAIRIC_PER_TOK
    if not mock:
        _ledger("clairic", "@cf/cloudflare/clairic", "ai/run", ok, st, u.get("input_tokens"), 0, cost, ms,
                n_q=len(body["fields"]), n_images=n_img, tag=tag, version=hv)
    return {"ok": ok, "status": st, "mock": mock, "preds": parse_clairic(item, js) if ok else {},
            "truncated": (res or {}).get("truncated"), "state_text": (res or {}).get("state_text"),
            "usage": u, "cost_usd": cost, "wall_ms": ms, "headers": hv,
            "raw_responses": [{"call": 0, "status": st, "response": strip_b64(js)}], "harness_rev": HARNESS_REV,
            "error": None if ok else str(js)[:500]}


def _sage_raw(body, mock, item, arm, tag):
    if mock:
        return 200, mock_sage2(item, body, arm), 0.0, {"mock": "1"}
    require_live(); _check_cap()
    hdr = {"Authorization": f"Bearer {_env('SAGE_API_KEY')}", "User-Agent": UA}
    return _post(f"{_env('SAGE2_BASE_URL').rstrip('/')}/v1/extract", body, hdr, timeout=90)


HARNESS_REV = "2026-10-26"   # span-based list union + raw_responses; run.py --force skips preds already at this rev


def strip_b64(obj, _min=200):
    """Copy of an API response with any base64 echo removed (keys 'base64'/'b64_json', data: URLs, long base64-only
    strings) so raw_responses stay small. Everything else is kept verbatim."""
    if isinstance(obj, dict):
        return {k: (f"<base64 {len(v)} chars omitted>" if k in ("base64", "b64_json") and isinstance(v, str)
                    else strip_b64(v, _min)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_b64(v, _min) for v in obj]
    if isinstance(obj, str) and len(obj) >= _min:
        if obj.startswith("data:") and ";base64," in obj[:100]:
            return f"<data-url {len(obj)} chars omitted>"
        if "\n" not in obj and " " not in obj and all(c.isalnum() or c in "+/=" for c in obj[:2000]):
            return f"<base64? {len(obj)} chars omitted>"
    return obj


def _locate(hay: str, needle: str, start: int = 0) -> int | None:
    """Code-point offset of window `needle` in `hay` (searching from `start`). Token-window decode can turn a split
    multi-byte character at either edge into U+FFFD, so retry with those stripped. None = not found."""
    i = hay.find(needle, start)
    if i >= 0:
        return i
    core = needle.lstrip("�")
    lead = len(needle) - len(core)
    core = core.rstrip("�")
    if core:
        i = hay.find(core, start)
        if i >= 0:
            return i - lead
    return None


def merge_mentions(cs: list[dict], wins: list[dict], content: str | None) -> dict:
    """Every-mention union across windows (fix 2026-10-26, audit fairness #2). Within a window every mention is kept,
    repeats included (the scorer is multiset F1). A mention from a later window is dropped only when it is the same
    span as a mention from an earlier window:
      (1) same (start, end) after mapping window offsets back into `content` (span trusted only when
          content[start:end] == the mention's text, or the mention has no text), or
      (2) otherwise, same text and it lies in the two windows' overlap region (by span if trusted, else the text
          occurs in the overlap substring; windows that could not be located count as wholly overlapping).
    Each earlier mention absorbs at most one duplicate per later window. cs[i]['_win'] indexes `wins`
    ({'offset': int|None, 'len': int}). Mentions with value None are dropped (as before)."""
    seen, vals, texts, spans, confs, n_drop = [], [], [], [], [], 0
    for c in cs:
        wi = c.get("_win", 0)
        w = wins[wi] if wi < len(wins) else {"offset": None, "len": 0}
        off = w.get("offset")
        n = len(c.get("value") or [])
        tx = (c.get("text") if isinstance(c.get("text"), list) else None) or [None] * n
        sp = (c.get("span") if isinstance(c.get("span"), list) else None) or [[None, None]] * n
        pr = c.get("probs") or [None] * n
        for v, t, (s, e), p in zip(c.get("value") or [], tx, sp, pr):
            if v is None:
                continue
            g = None
            if off is not None and isinstance(s, int) and isinstance(e, int) and e > s:
                g = (off + s, off + e)
                if t is not None and content is not None and content[g[0]:g[1]] != t:
                    g = None                       # offsets do not index the content as documented: don't trust
            dup = None
            for k in seen:
                if k["win"] == wi or wi in k["absorbed"]:
                    continue
                if g is not None and k["g"] is not None:
                    if g == k["g"]:
                        dup = k; break
                    continue                       # both spans trusted and different: distinct mentions
                if t is None or t != k["t"]:
                    continue
                kw = wins[k["win"]] if k["win"] < len(wins) else {"offset": None, "len": 0}
                if off is None or kw.get("offset") is None or content is None:
                    dup = k; break                 # unlocated window: treat as wholly overlapping
                lo, hi = max(off, kw["offset"]), min(off + w["len"], kw["offset"] + kw["len"])
                pos = g or k["g"]
                if hi > lo and (pos[0] >= lo and pos[1] <= hi if pos else t in content[lo:hi]):
                    dup = k; break
            if dup is not None:
                dup["absorbed"].add(wi); n_drop += 1
                continue
            seen.append({"win": wi, "g": g, "t": t, "absorbed": set()})
            vals.append(v); texts.append(t); spans.append(list(g) if g else [s, e]); confs.append(p)
    return {"value": vals, "conf": max((c["conf"] or 0) for c in cs) if cs else None, "p_absent": None,
            "n_windows": len({c.get("_win", 0) for c in cs}), "text": texts, "span": spans, "mention_conf": confs,
            "n_cross_window_dups_dropped": n_drop, "merge": "span_union_v2"}


def call_sage2(item, content, mode, arm="sage2", mock=True, tag="", shape=None):
    """Windows the content (<= SAGE_WINDOW o200k tokens, minus field defs) and splits again on 413.
    Scalar field: best non-null by probability across windows, else the most confident null.
    List field: every mention, unioned across windows by span (merge_mentions). Every response is kept in
    raw_responses (with the window's offset into the content) so lists can be re-merged / re-scored offline."""
    shape = shape or sage_shape_for(item)
    groups = [None] if shape == "typed" else [f for f in item["fields"] if sage_type(f)]
    cands: dict = {}
    calls, cost, ms, sms, usages, hv_all, errors = 0, 0.0, 0.0, 0.0, [], [], []
    wins, raws = [], []
    for fld in groups:
        defs = n_tokens(json.dumps(sage2_body(item, "", mode, fld)["extract"]))
        todo, cur = [], 0
        for w in windows(content, max(1000, SAGE_WINDOW - defs)):
            o = _locate(content, w, cur)
            todo.append((w, o)); cur = o if o is not None else cur
        while todo:
            w, woff = todo.pop(0)
            body = sage2_body(item, w, mode, fld)
            st, js, wms, hv = _sage_raw(body, mock, item, arm, tag)
            calls += 1; ms += wms; hv_all.append(hv)
            raws.append({"call": calls - 1, "field": fld["name"] if fld else None, "window_offset": woff,
                         "window_chars": len(w), "status": st, "response": strip_b64(js)})
            if st == 413 and n_tokens(w) > 1000:
                n = n_tokens(w)
                sub, cur = [], 0
                for sw in windows(w, n // 2 + 1, overlap=100):
                    so = _locate(w, sw, cur)
                    sub.append((sw, None if so is None or woff is None else woff + so)); cur = so if so is not None else cur
                todo = sub + todo
                continue
            ok = st == 200 and isinstance(js, dict) and "values" in js
            meta = (js or {}).get("meta") if isinstance(js, dict) else None
            u = (meta or {}).get("usage") or {}
            c = ((u.get("input_tokens") or 0) * SAGE2_IN_PER_TOK + (u.get("output_tokens") or 0) * SAGE2_OUT_PER_TOK
                 if SAGE2_IN_PER_TOK is not None and SAGE2_OUT_PER_TOK is not None else None)
            cost = None if c is None or cost is None else cost + c
            sms += (meta or {}).get("latency_ms") or 0; usages.append(u)
            if not mock:
                _ledger("sage2", f"levanto-sage-2:{mode}", "v1/extract", ok, st, u.get("input_tokens"),
                        u.get("output_tokens"), c, wms, n_q=len(body["extract"]) if isinstance(body["extract"], dict) else 1,
                        tag=tag, server_ms=(meta or {}).get("latency_ms"), version={**hv, "meta_model": (meta or {}).get("model")})
            if not ok:
                errors.append(f"{st}: {str(js)[:300]}")
                break
            for name, p in parse_sage2(item, js, fld).items():
                p["_win"] = len(wins)
                cands.setdefault(name, []).append(p)
            wins.append({"offset": woff, "len": len(w)})
    preds = {}
    for f in item["fields"]:
        if not sage_type(f):
            preds[f["name"]] = {"value": None, "conf": None, "unsupported": sage_unsupported(f)}
        elif f["name"] in cands:
            cs = cands[f["name"]]
            if f["type"] == "list":                 # every-mention: span union across windows only (no value dedupe)
                if not all(isinstance(c.get("value"), list) or c.get("value") is None for c in cs):
                    cs = [{**c, "value": [c["value"]], "text": [c.get("text")], "span": [c.get("span") or [None, None]],
                           "probs": [c.get("conf")]} if not isinstance(c.get("value"), (list, type(None))) else c
                          for c in cs]       # bare-question list answered with a scalar: one mention
                preds[f["name"]] = merge_mentions(cs, wins, content)
                continue
            nn = [c for c in cs if c["value"] is not None]
            best = max(nn, key=lambda c: c["conf"] or 0) if nn else max(cs, key=lambda c: c["conf"] or 0)
            preds[f["name"]] = {k: v for k, v in best.items() if k != "_win"} | {"window": best.get("_win")}
    ok = not errors
    return {"ok": ok, "status": 200 if ok else "ERR", "mock": mock, "preds": preds if ok else {}, "n_calls": calls,
            "usage": usages, "cost_usd": cost, "wall_ms": ms, "server_ms": sms, "headers": hv_all,
            "shape": shape, "windows": wins, "raw_responses": raws, "harness_rev": HARNESS_REV,
            "error": "; ".join(errors) or None}


def call_openrouter(item, body, arm="or", mock=True, tag=""):
    n_img = sum(p["type"] == "image_url" for p in body["messages"][-1]["content"])
    if mock:
        st, js, ms, hv = 200, mock_openrouter(item, body, arm), 0.0, {"mock": "1"}
    else:
        require_live(); _check_cap()
        with _OR_SEM:
            st, js, ms, hv = _post("https://openrouter.ai/api/v1/chat/completions", body,
                                   {"Authorization": f"Bearer {_env('OPENROUTER_API_KEY')}", "User-Agent": UA,
                                    "X-Title": "extract-bench"}, timeout=180)
    ok = st == 200 and isinstance(js, dict) and bool(js.get("choices"))
    u = (js.get("usage") if isinstance(js, dict) else None) or {}
    cost = u.get("cost") or 0.0
    text = (js["choices"][0]["message"].get("content") or "") if ok else None
    if not mock:
        _ledger("openrouter", js.get("model") if isinstance(js, dict) else body["model"], "chat", ok, st,
                u.get("prompt_tokens"), u.get("completion_tokens"), cost, ms, n_q=len(item["fields"]),
                n_images=n_img, tag=tag, version=hv)
    return {"ok": ok, "status": st, "mock": mock, "preds": parse_openrouter(item, text) if ok else {},
            "raw_text": text, "usage": u, "cost_usd": cost, "wall_ms": ms, "headers": hv,
            "raw_responses": [{"call": 0, "status": st, "response": strip_b64(js)}], "harness_rev": HARNESS_REV,
            "error": None if ok else str(js)[:500]}


def clairic_read(item, text, images=None, window=CLAIRIC_WINDOW, arm="clairic", mock=True, tag="",
                 return_state_text=False):
    """Clairic over text windows (one window unless the text exceeds `window` o200k tokens; images ride on
    every window). Per field: best non-null by confidence across windows, else the most confident null.
    window=SAGE_WINDOW gives the 'Clairic-windowed' control for long Kleister-Charity docs."""
    ws = windows(text, window) if text else [""]
    cands, calls, cost, ms, usages, hv_all, trunc = {}, 0, 0.0, 0.0, [], [], False
    state_text, raws, cur = None, [], 0
    for w in ws:
        body = clairic_body(item, w, images, return_state_text=return_state_text)
        r = call_clairic(item, body, arm=arm, mock=mock, tag=tag)
        calls += 1; cost += r["cost_usd"]; ms += r["wall_ms"]; usages.append(r["usage"]); hv_all.append(r["headers"])
        woff = _locate(text, w, cur) if text else None
        cur = woff if woff is not None else cur
        raws += [{**x, "call": calls - 1, "window_offset": woff, "window_chars": len(w)} for x in r["raw_responses"]]
        trunc = trunc or bool(r["truncated"])
        state_text = state_text or r.get("state_text")
        if not r["ok"]:
            return {**r, "n_calls": calls, "cost_usd": cost, "wall_ms": ms, "usage": usages, "headers": hv_all,
                    "raw_responses": raws}
        for name, p in r["preds"].items():
            cands.setdefault(name, []).append(p)
    preds = {}
    for name, cs in cands.items():
        nn = [c for c in cs if c["value"] is not None]
        preds[name] = max(nn, key=lambda c: c["conf"] or 0) if nn else max(cs, key=lambda c: c["conf"] or 0)
    return {"ok": True, "status": 200, "mock": mock, "preds": preds, "n_calls": calls, "n_windows": len(ws),
            "truncated": trunc, "state_text": state_text, "usage": usages, "cost_usd": cost, "wall_ms": ms,
            "headers": hv_all, "raw_responses": raws, "harness_rev": HARNESS_REV, "error": None}
