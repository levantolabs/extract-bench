"""Open-weight extraction models on the bench: NuExtract3 (NuMind), Schematron-3B (Inference.net), GLiNER2.5-base (Fastino).

The requests are built HERE, from the same item text (`run.item_text`) and the same field definitions every other arm gets.
A GPU box only runs inference (`python -m extract_bench.pod.infer_open`). The raw outputs come back and are parsed HERE
into ordinary pred files, then scored by `extract-bench run score` like any other arm.

  extract-bench open export --out reqs/                       # requests JSONL per model (the v0.1 item split)
  python -m extract_bench.pod.infer_open nuextract3 reqs/nuextract3.jsonl out/nuextract3.results.jsonl   # on the GPU
  extract-bench open import --run RUN --results out/          # -> runs/RUN/preds/MODEL/

Item set = every item in splits/<version>.json (both halves).

Field mapping (the same definition text as `instructions` for Clairic / `definition` for Sage):
- NuExtract3: JSON template, keys = field names, leaves = NuExtract types (text -> verbatim-string, number -> number or
  integer, money -> number, date/time/date-time/duration -> the ISO types, list -> [type]); every field's definition goes
  into the `instructions` chat-template kwarg; thinking off; greedy.
- Schematron-3B: JSON Schema (draft-07), one nullable property per field with the definition as `description`; the text
  wrapped as minimal HTML (<pre>), because Schematron is trained on HTML; the model card's own prompt; output
  constrained to the schema; greedy.
- GLiNER2.5-base: one entity label per field with the definition as its description; scalar field = the most confident
  span, list field = every span; documents longer than the encoder window are split into overlapping chunks.
"""
from __future__ import annotations

import argparse, html, json, re, time
from pathlib import Path

from .common import BENCH_VERSION, RUNS, dataset_dir, is_holdout, load_items, load_split, write_json
from .run import item_text

DATASETS = ["fatura", "omniai", "docvqa", "vrdu", "realdoc", "kleister_nda", "kleister_charity", "emails", "tatqa",
            "squad2", "sroie", "cord"]
MODELS = {
    "nuextract3": {"hf": "numind/NuExtract3", "kind": "nuextract"},
    "schematron3b": {"hf": "inference-net/Schematron-3B", "kind": "schematron"},
    "gliner25": {"hf": "fastino/gliner2.5-base-v1", "kind": "gliner"},
}
NU_TYPES = {"text": "verbatim-string", "money": "number", "date": "date", "time": "time", "datetime": "date-time",
            "duration": "duration"}
NULL_RULE = "If the document does not state a field, return null for it (an empty list for list fields). Do not guess."


def definition(f):
    return (f.get("instructions") or f["name"]).strip()


def nu_type(f):
    if f["type"] == "number":
        return "integer" if f.get("integer") else "number"
    if f["type"] == "list":
        it = f.get("items", "text")
        return [("integer" if f.get("integer") else "number") if it == "number" else NU_TYPES.get(it, "verbatim-string")]
    return NU_TYPES.get(f["type"], "verbatim-string")


def json_type(f):
    if f["type"] == "list":
        it = f.get("items", "text")
        inner = {"type": "number"} if it == "number" else {"type": "string"}
        return {"type": ["array", "null"], "items": inner}
    if f["type"] in ("number", "money"):
        return {"type": ["integer", "null"]} if f.get("integer") else {"type": ["number", "null"]}
    return {"type": ["string", "null"]}


FMT = {"date": " Format: YYYY-MM-DD.", "time": " Format: HH:MM (24h).", "datetime": " Format: ISO 8601 date-time.",
       "duration": " Format: ISO 8601 duration, e.g. P2Y.", "money": " A plain number, no currency symbol."}


def build(item, text, model):
    fields = item["fields"]
    if model == "nuextract3":
        template = {f["name"]: nu_type(f) for f in fields}
        instructions = "Field definitions:\n" + "\n".join(f"- {f['name']}: {definition(f)}" for f in fields) + "\n" + NULL_RULE
        return {"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
                "template": json.dumps(template, ensure_ascii=False), "instructions": instructions}
    if model == "schematron3b":
        schema = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "additionalProperties": False,
                  "required": [f["name"] for f in fields],
                  "properties": {f["name"]: {**json_type(f), "description": definition(f) + FMT.get(f["type"], "")
                                             + " Null when the page does not state it."} for f in fields}}
        page = f"<html><body><pre>{html.escape(text)}</pre></body></html>"
        prompt = ("You are going to be given a JSON schema following the standardized JSON Schema format. You are going to "
                  "be given a HTML page and you are going to apply the schema to the HTML page however you see it as "
                  "applicable and return the results in a JSON object. The schema is as follows:\n\n"
                  + json.dumps(schema, ensure_ascii=False) + "\n\nHere is the HTML page:\n\n" + page + "\n\nMAKE SURE ITS VALID JSON.")
        return {"messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}],
                "json_schema": schema}
    if model == "gliner25":
        return {"text": text, "labels": {f["name"]: definition(f) for f in fields},
                "list_fields": [f["name"] for f in fields if f["type"] == "list"]}
    raise KeyError(model)


def export(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fhs = {m: open(out / f"{m}.jsonl", "w") for m in a.models}
    n = 0
    split = load_split(a.split)
    for ds in DATASETS:
        ids = set(split[ds]["eval"]) | set(split[ds]["fit"])
        dsd = dataset_dir(ds)
        for it in load_items(ds):
            if it["id"] not in ids:
                continue
            text, src = item_text(it, dsd)
            for m in a.models:
                rec = {"id": it["id"], "dataset": ds, "arm": m, "hf": MODELS[m]["hf"], "text_source": src,
                       "empty": not (text or "").strip()}
                if not rec["empty"]:
                    rec["request"] = build(it, text, m)
                fhs[m].write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    for fh in fhs.values():
        fh.close()
    print(f"exported {n} items x {len(a.models)} models -> {out}")


def _json_obj(s):
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    return json.loads(s[i:j + 1]) if i >= 0 and j > i else None


def parse(item, model, result):
    """-> harness preds {field: {value, conf}}; None if the output is not JSON (imported as an OK read with no answers)."""
    names = [f["name"] for f in item["fields"]]
    if model in ("nuextract3", "schematron3b"):
        try:
            obj = _json_obj(result.get("text"))
        except Exception:  # noqa: BLE001
            obj = None
        if not isinstance(obj, dict):
            return None
        out = {}
        for f in item["fields"]:
            v = obj.get(f["name"])
            if f["type"] == "list":
                v = v if isinstance(v, list) else ([] if v in (None, "") else [v])
                v = [x for x in v if x not in (None, "")]
            elif v == "" or v == []:
                v = None
            out[f["name"]] = {"value": v, "conf": None}
        return out
    if model == "gliner25":
        spans = result.get("spans") or {}
        out = {}
        for f in item["fields"]:
            ss = sorted(spans.get(f["name"]) or [], key=lambda s: -(s.get("confidence") or 0))
            if f["type"] == "list":
                seen, vals = set(), []
                for s in sorted(spans.get(f["name"]) or [], key=lambda s: s.get("start") or 0):
                    k = (s.get("start"), s.get("end"))
                    if k not in seen:
                        seen.add(k); vals.append(s["text"])
                out[f["name"]] = {"value": vals, "conf": ss[0]["confidence"] if ss else None}
            else:
                out[f["name"]] = {"value": ss[0]["text"] if ss else None, "conf": ss[0].get("confidence") if ss else None}
        return out
    raise KeyError(model)


def do_import(a):
    res_dir = Path(a.results)
    items = {ds: {it["id"]: it for it in load_items(ds)} for ds in DATASETS}
    for m in MODELS:
        rf = res_dir / f"{m}.results.jsonl"
        if not rf.exists():
            continue
        ok = err = empty = bad = 0
        for line in open(rf):
            r = json.loads(line)
            it = items[r["dataset"]][r["id"]]
            spec = {"system": "open", "input": "text", "model": MODELS[m]["hf"], "name": m, "variant": None, "view": None}
            base = {"id": it["id"], "dataset": r["dataset"], "arm": m, "spec": spec, "text_source": r.get("text_source"),
                    "n_images": 0, "holdout": is_holdout(it), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mock": False,
                    "cost_usd": 0.0, "wall_ms": r.get("wall_ms"), "n_calls": r.get("n_calls", 1),
                    "raw_responses": [{"call": 0, "response": r.get("text") if m != "gliner25" else r.get("spans")}],
                    "pod": r.get("pod")}
            if r.get("empty"):
                preds = {f["name"]: {"value": None, "conf": None, "input_empty": True} for f in it["fields"]}
                rec = {**base, "ok": True, "input_empty": True, "error": None, "preds": preds}; empty += 1
            else:
                preds = parse(it, m, r) if not r.get("error") else None
                if r.get("error"):
                    rec = {**base, "ok": False, "error": r["error"], "preds": {}}; err += 1
                elif preds is None:
                    # the model ran but its output is not JSON (repetition loop cut off at max_tokens, say): an OK read
                    # with no answers, so every field scores 'missing' = wrong, like an unparsed LLM response
                    rec = {**base, "ok": True, "error": None, "unparsed": True, "finish": r.get("finish"), "preds": {}}
                    bad += 1
                else:
                    rec = {**base, "ok": True, "error": None, "preds": preds}; ok += 1
            write_json(RUNS / a.run / "preds" / m / r["dataset"] / f"{it['id']}.json", rec)
        print(f"{m}: ok {ok}, unparsed {bad}, errors {err}, empty-input {empty}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="extract-bench open", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--split", default=BENCH_VERSION)
    e.add_argument("--out", required=True); e.add_argument("--models", default=",".join(MODELS))
    i = sub.add_parser("import"); i.add_argument("--run", required=True); i.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        a.models = [m.strip() for m in a.models.split(",")]
        export(a)
    else:
        do_import(a)


if __name__ == "__main__":
    main()
