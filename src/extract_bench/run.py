"""Extraction-bench runner (`extract-bench run STAGE ...`). Resumable: one JSON per (arm, dataset, item); existing OK
files are skipped.

  extract-bench run ocr   --datasets sroie,cord
  extract-bench run read  --run dryrun-x --arms clairic_text,sage_fast
  extract-bench run score --run dryrun-x --pair clairic_text sage_fast

Stages: ocr (free local Tesseract for items without official OCR) -> read (arms) -> score (per item + summary).
MOCK is the default (also spelled --dry-run): no network, no keys, run name must start with "dryrun-".
--live needs env EXB_LIVE=1 and EXB_CAP_USD=<your $ ceiling> as well (see clients.require_live).
Sage arms skip the sealed 20% holdout unless --final. Layout: runs/RUN/{preds,scores}/ARM/DS/ID.json.
Re-runs (2026-10-26): --ids-file PATH restricts any stage to those item ids; read --force re-reads OK preds (moving
the old one to runs/RUN/preds_prev_2026-10-26/ARM/DS/ID.json); --final still gates the holdout. Every pred written
since then carries harness_rev and raw_responses (one per API call, base64 echo stripped) for $0 re-scoring.

Arm name = BASE[@variant][#view]
  @variant  read meta.alt_texts[variant] instead of the default text (e.g. clairic_text@gold_md, @tesseract_layer)
  #comparable  the benchmark's own view: fields from meta.comparable.fields (SROIE: all four keys as verbatim text),
            image from meta.comparable.image (CORD: CC-OCR's JPEG bytes); OpenRouter arms send the benchmark's own
            prompt verbatim (meta.comparable.prompt: CC-OCR prompt, emails prompt2 + schema). Items without a
            comparable view are n/a. Score these with the official scorer only (`run export`), never next to TYPED.
Item sets: --datasets KEY~SUB reads the single .jsonl in datasets/KEY/SUB/ (emails~anchor = all 1,000 emails).
Stages also include `export` (per-arm prediction files for the official/comparable scorers, see README).

Arms:
"""
from __future__ import annotations

import argparse, json, os, random, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import clients as C, ocr, score as S
from .common import (DATASETS, RUNS, SEED, alt_text, cluster_of, dataset_dir, is_holdout,
                            list_datasets, load_items, write_json)

HAIKU = "anthropic/claude-haiku-5.5"
GEMINI_FLASH = "google/gemini-3.8-flash"
ARMS = {
    "clairic_text":       {"system": "clairic", "input": "text"},
    "clairic_text_w15k":  {"system": "clairic", "input": "text", "window": C.SAGE_WINDOW},   # windowed control
    "clairic_text_whole": {"system": "clairic", "input": "text", "window": C.CLAIRIC_WHOLE},  # literal whole doc
    "clairic_vision":     {"system": "clairic", "input": "image"},
    "clairic_vision_ocr": {"system": "clairic", "input": "image+text"},
    "sage_precise":       {"system": "sage2", "input": "text", "mode": "precise"},
    "sage_fast":          {"system": "sage2", "input": "text", "mode": "fast"},
    "haiku_json":         {"system": "openrouter", "input": "text", "model": HAIKU},
    "gemini_vision_json": {"system": "openrouter", "input": "image", "model": GEMINI_FLASH},
    # open-weight extraction models, run on a GPU box via `extract-bench open` (export -> GPU inference -> import)
    "nuextract3":         {"system": "open", "input": "text", "model": "numind/NuExtract3"},
    "schematron3b":       {"system": "open", "input": "text", "model": "inference-net/Schematron-3B"},
    "gliner25":           {"system": "open", "input": "text", "model": "fastino/gliner2.5-base-v1"},
}
__doc__ += "".join(f"  {k:<20} {v}\n" for k, v in ARMS.items()) + \
    "  or:<model>:text|image   any OpenRouter model with the JSON-schema prompt\n"
SAGE_SEM = threading.Semaphore(int(os.environ.get("EXB_SAGE_CONC", "2")))  # concurrent Sage 2 requests
_print = threading.Lock()
OCR_OPTS = dict(ocr.DEFAULT_OCR)  # psm 3 with psm-6 fallback on empty pages (see ocr.DEFAULT_OCR)


def log(*a):
    with _print:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def arm_spec(name: str) -> dict:
    rest, _, view = name.partition("#")
    if view not in ("", "comparable"):
        raise KeyError(f"unknown view #{view} in {name}")
    base, _, variant = rest.partition("@")
    if base.startswith("or:"):
        model, _, inp = base[3:].rpartition(":")
        if inp not in ("text", "image", "image+text"):
            model, inp = base[3:], "text"
        spec = {"system": "openrouter", "input": inp, "model": model}
    else:
        spec = dict(ARMS[base])
    return {**spec, "name": name, "variant": variant or None, "view": view or None}


def view_item(item: dict, spec: dict) -> dict | None:
    """Apply an arm's #view. None = the item has no such view (n/a)."""
    if not spec.get("view"):
        return item
    c = (item.get("meta") or {}).get("comparable")
    if not isinstance(c, dict):
        return None
    out = dict(item)
    if isinstance(c.get("fields"), list) and c["fields"]:
        out["fields"] = c["fields"]
    elif spec["system"] != "openrouter":
        return None                     # e.g. CORD: CC-OCR prompt only, no typed field list for Clairic/Sage
    if c.get("image"):
        out["images"] = [c["image"]]
    return out


def safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "~")


def item_text(item, ds_dir, variant=None, allow_ocr=True):
    """Text column for an item: alt variant > item.text (native / official OCR) > cached-or-fresh harness OCR."""
    if variant:
        t = alt_text(item, ds_dir, variant)
        return t, f"alt:{variant}" if t is not None else None
    if item.get("text") is not None:
        return item["text"], item["text_source"]
    if item.get("images") and allow_ocr and ocr.available():
        r = ocr.ocr_item(item, ds_dir, **OCR_OPTS)
        return r["text"], f"harness_ocr:{r['engine_tag']}"
    return None, None


def p(run, *parts) -> Path:
    return RUNS / run / Path(*parts)


# ------------------------------------------------------------------ stages
def do_ocr(run, key, item, args):
    if item.get("text") is not None or not item.get("images"):
        return "skip"
    r = ocr.ocr_item(item, dataset_dir(key, args.root), **OCR_OPTS)
    return "cached" if r["cached"] else "ocr"


PREV_DIR = "preds_prev_2026-10-26"   # --force moves the pred it replaces here (same ARM/DS/ID layout)


def _move_prev(run, f: Path) -> Path | None:
    """--force: keep the old pred as runs/RUN/preds_prev_2026-10-26/ARM/DS/ID.json (ID.N.json if taken)."""
    if not f.exists():
        return None
    rel = f.relative_to(p(run, "preds"))
    dst = p(run, PREV_DIR, rel)
    n = 1
    while dst.exists():
        dst = p(run, PREV_DIR, rel.parent, f"{rel.stem}.{n}.json"); n += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    f.replace(dst)
    return dst


def read_one(run, key, item, arm, args):
    spec = arm_spec(arm)
    f = p(run, "preds", safe(arm), key, f"{item['id']}.json")
    replace = False
    if f.exists():
        old = json.loads(f.read_text())
        if not old.get("error"):
            # --force re-reads an OK pred unless it was already written by this harness revision (so an interrupted
            # forced re-run resumes instead of paying twice)
            if not getattr(args, "force", False) or old.get("harness_rev") == C.HARNESS_REV:
                return "skip"
            replace = True
    if spec["system"] == "sage2" and is_holdout(item) and not args.final:
        return "holdout"
    item = view_item(item, spec)
    if item is None:
        return "n/a"
    if all(C.na_for(fl, spec["system"]) for fl in item["fields"]):
        return "n/a"                    # every field opted out for this system (TAT-QA computed items for Sage)
    if spec["system"] == "sage2" and not any(C.sage_type(fl) for fl in item["fields"]):
        return "n/a"                    # nothing Sage can express (list-only items)
    ds_dir = dataset_dir(key, args.root)
    want_img, want_txt = "image" in spec["input"], "text" in spec["input"]
    if want_img and not item.get("images"):
        return "n/a"
    text = src = None
    if want_txt:
        text, src = item_text(item, ds_dir, spec["variant"])
        if text is None and spec["variant"]:
            return "n/a"                # this item has no such alternative text (e.g. realdoc @pdf_text_layer)
        if not (text or "").strip():
            # The pipeline's own text is empty (Tesseract found nothing on the page). Not dropped: a text-only arm
            # is recorded as abstaining on every field at $0 (no call), so the miss counts against the pipeline,
            # identically for Clairic-text and Sage; an image+text arm falls back to the image alone.
            if not want_img:
                write_json(f, {"id": item["id"], "dataset": key, "arm": arm, "spec": spec, "text_source": src,
                               "n_images": 0, "holdout": is_holdout(item), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               "ok": True, "mock": not args.live, "input_empty": True, "cost_usd": 0.0, "wall_ms": 0.0,
                               "error": None, "preds": {fl["name"]: (
                                   {"value": None, "conf": None, "unsupported": C.sage_unsupported(fl)}
                                   if spec["system"] == "sage2" and C.sage_unsupported(fl) else
                                   {"value": None, "conf": None, "input_empty": True}) for fl in item["fields"]}})
                return "input_empty"
            text = None
    tag = f"exb:{run}:{arm}:{key}"
    mock = not args.live
    t0 = time.time()
    try:
        images = C.load_images(item, ds_dir) if want_img else None
    except C.ImageRejected as e:
        res = {"ok": False, "error": f"image_rejected: {e}", "preds": {}}
    else:
        if spec["system"] == "clairic":
            res = C.clairic_read(item, text, images, window=spec.get("window", C.CLAIRIC_WINDOW), arm=arm,
                                 mock=mock, tag=tag)
            if images and not res.get("ok") and "image dimensions are too large" in str(res.get("error")):
                # Clairic code 3030 (undocumented pixel cap, ~4096^2): retry once downscaled to 4,096 px longest edge.
                # Declared in RESULTS; the OpenRouter arms keep the original images.
                images = C.load_images(item, ds_dir, max_edge=4096)
                res = C.clairic_read(item, text, images, window=spec.get("window", C.CLAIRIC_WINDOW), arm=arm,
                                     mock=mock, tag=tag)
                res["image_downscaled_max_edge"] = 4096
        elif spec["system"] == "sage2":
            with SAGE_SEM:
                res = C.call_sage2(item, text, spec["mode"], arm=arm, mock=mock, tag=tag)
        else:
            nat = C.native_prompt(item, ds_dir) if spec.get("view") == "comparable" else None
            body = (C.openrouter_native_body(item, spec["model"], text, images, *nat) if nat
                    else C.openrouter_body(item, spec["model"], text, images))
            res = C.call_openrouter(item, body, arm=arm, mock=mock, tag=tag)
            if not nat and not res.get("ok") and "too many parameters with union" in str(res.get("error")):
                # Anthropic strict-schema cap (many nullable fields, e.g. OmniAI). Declared fallback: same schema,
                # written into the system prompt instead of enforced; output parsed the same way.
                sch = body["response_format"]["json_schema"]["schema"]
                body = {k: v for k, v in body.items() if k != "response_format"}
                body["messages"] = [{"role": "system", "content": C.OR_SYSTEM + " JSON schema (every key required, "
                                     "null when absent): " + json.dumps(sch)}] + body["messages"][1:]
                res = C.call_openrouter(item, body, arm=arm, mock=mock, tag=tag)
                res["schema_in_prompt_fallback"] = True
            if nat:
                res["native_prompt"] = True
    for fl in item["fields"]:           # builder-declared opt-outs (field.na_arms) -> unscored for this arm
        if C.na_for(fl, spec["system"]) and res.get("preds") is not None:
            res["preds"][fl["name"]] = {"value": None, "conf": None, "unsupported": f"na_arms:{spec['system']}"}
    prev = _move_prev(run, f) if replace else None   # moved only now, after the new read, so a crash keeps the old
    if prev is not None:
        res["replaced_prev"] = str(prev.relative_to(p(run)))
    write_json(f, {"id": item["id"], "dataset": key, "arm": arm, "spec": spec, "text_source": src,
                   "n_images": len(item.get("images") or []) if want_img else 0,
                   "holdout": is_holdout(item), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "elapsed_s": round(time.time() - t0, 3), **res})
    return "error" if res.get("error") else "ok"


def do_read(run, key, item, args):
    order = list(args.arms)
    random.Random(f"{SEED}:{item['id']}:read").shuffle(order)  # interleave arms per item
    return [read_one(run, key, item, a, args) for a in order]


def do_score(run, key, item, args):
    out = []
    for af in sorted(p(run, "preds").glob(f"*/{key}/{item['id']}.json")):
        a = json.loads(af.read_text())
        if a.get("error"):
            out.append("error"); continue
        spec = arm_spec(a["arm"])
        vit = view_item(item, spec)
        if vit is None:
            out.append("n/a"); continue
        if spec.get("view") and vit["fields"] is item["fields"]:
            out.append("export_only"); continue   # CORD #comparable: CC-OCR nested JSON, official scorer only
        sc = S.score_item(vit, a.get("preds") or {})
        sc["view"] = (a.get("spec") or {}).get("view")
        write_json(p(run, "scores", af.parent.parent.name, key, f"{item['id']}.json"),
                   {"arm": a["arm"], "mock": a.get("mock"), "cluster": cluster_of(item), "holdout": a.get("holdout"),
                    "cost_usd": a.get("cost_usd"), "wall_ms": a.get("wall_ms"), **sc})
        out.append("ok")
    return out


def summarize(run, keys, pairs):
    base = p(run, "scores")
    summary = {"run": run, "mock": False, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "arms": {}, "pairs": {}}
    per = {}  # (arm, key) -> {id: scored item}
    for d in sorted(base.glob("*/*")):
        arm_dir, key = d.parent.name, d.name
        if keys and key not in keys:
            continue
        its = [json.loads(x.read_text()) for x in sorted(d.glob("*.json"))]
        if not its:
            continue
        arm = its[0]["arm"]
        summary["mock"] |= any(i.get("mock") for i in its)
        per[(arm, key)] = {i["id"]: i for i in its}
        agg = S.aggregate(its)
        agg["cost_usd"] = sum(i.get("cost_usd") or 0 for i in its)
        summary["arms"].setdefault(arm, {})[key] = agg
    # Failures are NOT in the accuracies above: an errored read is dropped from scoring (like OmniAI's published rows),
    # and an OK read whose output did not parse (OpenRouter JSON cut off by max_tokens, say) leaves every field
    # 'missing' = unscored. Count both per arm x dataset so they are reported, never silently absorbed (critic 23 Oct).
    fails = {}
    for d in sorted(p(run, "preds").glob("*/*")) if p(run, "preds").exists() else []:
        key = d.name
        if keys and key not in keys:
            continue
        ne = nu = 0
        arm = None
        for x in d.glob("*.json"):
            r = json.loads(x.read_text())
            arm = arm or r.get("arm")
            if r.get("error"):
                ne += 1
            elif not r.get("preds") and not r.get("input_empty"):
                nu += 1
        if arm and (ne or nu):
            fails[f"{arm} | {key}"] = {"errors_dropped": ne, "empty_preds_unscored": nu}
            if arm in summary["arms"] and key in summary["arms"][arm]:
                summary["arms"][arm][key].update(n_errors_dropped=ne, n_empty_preds=nu)
    summary["failures"] = fails
    pv = {}
    for a, b in pairs:
        for key in sorted({k for (_, k) in per}):
            A, B = per.get((a, key)), per.get((b, key))
            if not A or not B:
                continue
            for rule in ("a", "b"):
                sa, sb, cl = {}, {}, {}
                for i in set(A) & set(B):
                    # headline fields only (CARD-declared subsets, e.g. kleister_nda), scored by both arms
                    fa = {f["name"]: f for f in A[i]["fields"] if f["scored"] and f.get("headline", True)}
                    fb = {f["name"]: f for f in B[i]["fields"] if f["scored"] and f.get("headline", True)}
                    common = sorted(set(fa) & set(fb))   # compare only fields both arms could express
                    if common:
                        sa[i] = sum(fa[n][f"score_{rule}"] for n in common) / len(common)
                        sb[i] = sum(fb[n][f"score_{rule}"] for n in common) / len(common)
                        cl[i] = A[i].get("cluster", i)
                bs = S.paired_bootstrap(sa, sb, cl, n=10000, seed=SEED)
                name = f"{a} vs {b} | {key} | rule {rule}"
                summary["pairs"][name] = bs
                if bs.get("n"):
                    pv[name] = bs["p_two_sided"]
    if pv:
        summary["holm"] = S.holm(pv)
    write_json(p(run, "summary.json"), summary)
    tag = "  [MOCK: fake responses, scores are meaningless]" if summary["mock"] else ""
    print(f"\nrun {run}{tag}")
    w = max([24] + [len(a) + 2 for a in summary["arms"]])
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else "-"  # noqa: E731
    for title, sel in (("TYPED view (harness scorers; the head-to-head)", lambda a: "#" not in a),
                       ("#comparable view: harness exact_norm shown for plumbing only; publishable numbers come from "
                        "the official scorer (run.py export)", lambda a: "#" in a)):
        rows = [(a, k, g) for a, ds in sorted(summary["arms"].items()) if sel(a) for k, g in ds.items()]
        if not rows:
            continue
        print(f"\n{title}")
        print(f"{'arm':<{w}}{'dataset':<18}{'items':>6}{'%scored':>9}{'acc(a)':>8}{'acc(b)':>8}{'covadj(a)':>10}"
              f"{'present(a)':>11}{'null(a)':>8}{'head(a)':>8}{'$':>9}")
        for arm, key, g in rows:
            print(f"{arm:<{w}}{key:<18}{g['n_items']:>6}{fmt(g['pct_gold_fields_scored']):>9}{fmt(g['field_acc_a']):>8}"
                  f"{fmt(g['field_acc_b']):>8}{fmt(g['cov_adj_a']):>10}{fmt(g.get('present_gold_acc_a')):>11}"
                  f"{fmt(g.get('null_gold_acc_a')):>8}{fmt((g.get('headline') or {}).get('field_acc_a')):>8}"
                  f"{g['cost_usd']:>9.4f}")
    for name, c in summary["failures"].items():
        print(f"  FAILURES {name}: {c['errors_dropped']} errored reads dropped, {c['empty_preds_unscored']} unparsed "
              f"outputs left unscored (not counted as wrong in acc; see %scored)")
    for name, bs in summary["pairs"].items():
        if bs.get("n"):
            h = summary.get("holm", {}).get(name, {})
            print(f"  {name}: diff {bs['diff']:+.3f} [{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}] n={bs['n']} "
                  f"clusters={bs['n_clusters']} p={bs['p_two_sided']:.4f} holm={h.get('p_adj', float('nan')):.4f}")
    return summary


def _plain(v):
    return "" if v is None else v


def export(run, keys, arms, root=None):
    """Per-arm prediction files for the official / comparable scorers (all local, $0). Written under
    runs/RUN/export/<arm dir>/:
      KEY.preds.jsonl          every dataset: {id, holdout, preds: {field: value}, conf: {field: conf}, raw_text?}
      KEY.items.jsonl          (emails, squad2, realdoc) the items those records cover: use it as ITEMS below
      emails.score_emails.jsonl   {id, committee}  -> datasets/emails/comparable/score_emails.py ITEMS PREDS
      squad2.squad_preds.json     {id: {value, conf, p_null}} -> datasets/squad2/comparable/squad_v2_eval.py
      realdoc.realdoc_preds.json  {id: {field: value}} -> datasets/realdoc/scripts/realdoc_score.summarize()
      KEY.ccocr.json (sroie, cord) {image_name: {key: value}} = CC-OCR KieEvaluator().evaluate(response_info, gt)
        (#comparable arms: the OpenRouter raw response text is parsed as CC-OCR's post_process_to_json would see it)
    Kleister (comparable/kleister_geval.py --preds-dir), TAT-QA (comparable/run_official.py --run --arm) and VRDU
    (tools/official_eval.py --run --arm) read runs/RUN/preds/ directly."""
    n = 0
    for arm in arms:
        adir = safe(arm)
        for key in keys:
            pdir = p(run, "preds", adir, key)
            if not pdir.exists():
                continue
            items = {it["id"]: it for it in load_items(key, root)}
            recs = []
            for f in sorted(pdir.glob("*.json")):
                r = json.loads(f.read_text())
                if r.get("error") or r["id"] not in items:
                    continue
                pr = r.get("preds") or {}
                recs.append({"id": r["id"], "holdout": r.get("holdout"), "view": (r.get("spec") or {}).get("view"),
                             "preds": {k: v.get("value") for k, v in pr.items() if not v.get("unsupported")},
                             "conf": {k: v.get("conf") for k, v in pr.items() if not v.get("unsupported")},
                             "p_null": {k: v.get("p_null", v.get("p_absent")) for k, v in pr.items()
                                        if not v.get("unsupported")},
                             **({"raw_text": r["raw_text"]} if r.get("raw_text") else {})})
            if not recs:
                continue
            out = p(run, "export", adir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{key}.preds.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs))
            # the items this arm actually has records for (Sage dev runs skip the sealed holdout): pass THIS file,
            # not datasets/KEY/items.jsonl, to scorers that count a missing prediction as wrong (score_emails.py)
            bk = key.split("~")[0]
            if bk in ("emails", "squad2", "realdoc"):
                (out / f"{key}.items.jsonl").write_text("".join(json.dumps(items[x["id"]], ensure_ascii=False) + "\n"
                                                                for x in recs))
            if bk == "emails":
                (out / f"{key}.score_emails.jsonl").write_text("".join(
                    json.dumps({"id": x["id"], "committee": x["preds"].get("committee")}, ensure_ascii=False) + "\n"
                    for x in recs))
            if bk == "squad2":
                (out / f"{key}.squad_preds.json").write_text(json.dumps(
                    {x["id"]: {"value": x["preds"].get("answer"), "conf": x["conf"].get("answer"),
                               "p_null": x["p_null"].get("answer")} for x in recs}, ensure_ascii=False))
            if bk == "realdoc":
                (out / f"{key}.realdoc_preds.json").write_text(json.dumps({x["id"]: x["preds"] for x in recs},
                                                                          ensure_ascii=False))
            if bk in ("sroie", "cord"):
                cc = {}
                for x in recs:
                    m = items[x["id"]]["meta"]
                    name = (m.get("comparable") or {}).get("image_name") or m.get("image_name")
                    if not name or (bk == "cord" and not m.get("comparable")):
                        continue
                    obj = None
                    if x.get("raw_text"):
                        try:
                            t = x["raw_text"]
                            obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
                        except Exception:  # noqa: BLE001
                            obj = {}
                    cc[name] = obj if obj is not None else {k: _plain(v) for k, v in x["preds"].items()}
                (out / f"{key}.ccocr.json").write_text(json.dumps(cc, ensure_ascii=False))
            n += len(recs)
            print(f"export {arm:<28} {key:<18} {len(recs):>5} records -> {out.relative_to(RUNS.parent)}")
    return n


def load_ids(path: Path) -> set[str]:
    return {ln.split("#", 1)[0].strip() for ln in path.read_text().splitlines() if ln.split("#", 1)[0].strip()}


STAGES = {"ocr": do_ocr, "read": do_read, "score": do_score, "export": None}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="extract-bench run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--run", default=None)
    ap.add_argument("--datasets", help="comma list (default: every datasets/<key>/items.jsonl)")
    ap.add_argument("--arms", default=None, help="comma list; see above (read default: clairic_text,sage_precise,"
                    "sage_fast; export default: every arm in the run)")
    ap.add_argument("--limit", type=int, help="first N items per dataset (after --ids-file)")
    ap.add_argument("--ids-file", type=Path, help="read/score only these item ids (one per line, '#' comments)")
    ap.add_argument("--force", action="store_true",
                    help="read: re-read items even if an OK pred exists (unless it is already harness_rev "
                         f"{C.HARNESS_REV}); the old pred is moved to runs/RUN/{PREV_DIR}/ARM/DS/ID.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="mock responses (the default unless --live)")
    ap.add_argument("--live", action="store_true", help="real metered calls; also needs WSB_EXTRACT_LIVE=1 + EXB_CAP_USD")
    ap.add_argument("--final", action="store_true", help="include the sealed holdout for Sage arms (final run only)")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="score: paired bootstrap A - B on fields both arms scored (repeatable; Holm across all)")
    ap.add_argument("--root", type=Path, default=None, help=f"datasets root (default {DATASETS})")
    ap.add_argument("--psm", type=int, default=ocr.DEFAULT_OCR["psm"])
    ap.add_argument("--fallback-psm", type=int, default=ocr.DEFAULT_OCR["fallback_psm"],
                    help="psm retried on pages where --psm finds no text (0 = off)")
    a = ap.parse_args(argv)
    if a.live and a.dry_run:
        ap.error("--live and --dry-run are exclusive")
    if a.live:
        try:
            C.require_live()
        except C.LiveCallBlocked as e:
            ap.error(str(e))
    OCR_OPTS["psm"], OCR_OPTS["fallback_psm"] = a.psm, (a.fallback_psm or None)
    if a.stage != "ocr":
        if not a.run:
            ap.error("--run is required")
        if not a.live and a.stage != "export" and not a.run.startswith("dryrun-"):
            ap.error("mock runs must be named dryrun-*, so fake predictions never mix with live ones")
    a.arms_given = a.arms is not None
    a.arms = [x.strip() for x in (a.arms or "clairic_text,sage_precise,sage_fast").split(",") if x.strip()]
    for x in a.arms:
        if a.stage != "export":
            arm_spec(x)  # KeyError early on a typo
    keys = [k.strip() for k in a.datasets.split(",")] if a.datasets else list_datasets(a.root)
    if a.stage == "export":
        if not p(a.run, "preds").exists():
            ap.error(f"no preds under runs/{a.run}")
        arms = a.arms if a.arms_given else sorted({d.name for d in p(a.run, "preds").iterdir() if d.is_dir()})
        export(a.run, keys, arms, a.root)
        return 0
    if a.stage == "ocr" and not ocr.available():
        sys.exit("tesseract not available: install Tesseract 5 (e.g. `brew install tesseract`, `apt-get install tesseract-ocr`)")
    ids = load_ids(a.ids_file) if a.ids_file else None
    if a.force and a.stage != "read":
        ap.error("--force applies to the read stage only")
    work = [(k, it) for k in keys
            for it in [x for x in load_items(k, a.root) if ids is None or x["id"] in ids][:a.limit or None]]
    log(f"{a.stage}: {len(work)} items in {len(keys)} datasets; arms {a.arms if a.stage == 'read' else '-'}; "
        f"{'LIVE' if a.live else 'MOCK'}")
    counts, fail = {}, 0
    fn = STAGES[a.stage]
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fn, a.run, k, it, a): (k, it["id"]) for k, it in work}
        for fu in as_completed(futs):
            try:
                r = fu.result()
                for x in (r if isinstance(r, list) else [r]):
                    counts[x] = counts.get(x, 0) + 1
            except (C.BudgetExceeded, C.LiveCallBlocked) as e:
                log("STOP:", e); ex.shutdown(cancel_futures=True); break
            except Exception:  # noqa: BLE001
                fail += 1
                log("FAIL", futs[fu], traceback.format_exc(limit=3).splitlines()[-1])
    log(f"{a.stage} done: {counts} fail {fail}")
    if a.stage == "score":
        summarize(a.run, keys, [tuple(x) for x in a.pair])
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
