"""Fix pass 2026-10-26 (audit fairness #2): span-based Sage list union, raw_responses, run.py --ids-file / --force."""
import json

from conftest import FIXTURES

from extract_bench import clients as C, common
from extract_bench import run as R

ITEMS = {it["id"]: it for it in common.load_items("demo", FIXTURES)}


def _mentions(content, woff, needles):
    """MentionsResult for the given (text, occurrence) mentions, offsets relative to the window at woff."""
    out = []
    for t, occ in needles:
        i = -1
        for _ in range(occ + 1):
            i = content.find(t, i + 1)
        assert i >= 0, t
        out.append({"value": t, "text": t, "start": i - woff, "end": i - woff + len(t), "probability": 0.9})
    return {"values": out}


def test_within_window_repeats_kept():
    content = "Paid to Acme. Later paid to Acme again. And Acme once more."
    c = C._sage_one(_mentions(content, 0, [("Acme", 0), ("Acme", 1), ("Acme", 2)]))
    c["_win"] = 0
    m = C.merge_mentions([c], [{"offset": 0, "len": len(content)}], content)
    assert m["value"] == ["Acme", "Acme", "Acme"] and m["n_cross_window_dups_dropped"] == 0


def test_cross_window_overlap_duplicates_removed_by_span():
    content = "A: Acme. B: Beta. C: Acme. D: Delta. E: Acme."
    w0, w1 = (0, 30), (18, len(content))          # overlap = content[18:30] holds the 2nd "Acme"
    c0 = C._sage_one(_mentions(content, w0[0], [("Acme", 0), ("Beta", 0), ("Acme", 1)]))
    c1 = C._sage_one(_mentions(content, w1[0], [("Acme", 1), ("Delta", 0), ("Acme", 2)]))
    c0["_win"], c1["_win"] = 0, 1
    wins = [{"offset": w0[0], "len": w0[1] - w0[0]}, {"offset": w1[0], "len": w1[1] - w1[0]}]
    m = C.merge_mentions([c0, c1], wins, content)
    assert m["value"] == ["Acme", "Beta", "Acme", "Delta", "Acme"]   # 3 real Acmes; the overlap copy dropped once
    assert m["n_cross_window_dups_dropped"] == 1
    assert [content[s:e] for s, e in m["span"]] == m["text"]          # spans mapped into the full content


def test_cross_window_text_rule_only_inside_overlap():
    """No usable offsets (start/end missing): same text is a duplicate only if it occurs in the overlap region."""
    content = "Acme here. " + "x" * 40 + " shared Zeta tail. " + "y" * 40 + " Acme there."
    o1 = content.find(" shared")
    wins = [{"offset": 0, "len": o1 + 20}, {"offset": o1, "len": len(content) - o1}]
    c0 = {"value": ["Acme", "Zeta"], "text": ["Acme", "Zeta"], "span": [[None, None]] * 2, "conf": 0.9, "_win": 0}
    c1 = {"value": ["Zeta", "Acme"], "text": ["Zeta", "Acme"], "span": [[None, None]] * 2, "conf": 0.9, "_win": 1}
    m = C.merge_mentions([c0, c1], wins, content)
    assert m["value"] == ["Acme", "Zeta", "Acme"]   # Zeta (in overlap) deduped; Acme (outside overlap) kept twice


def test_untrusted_span_falls_back_to_text_rule():
    content = "Acme " * 3
    c0 = {"value": ["Acme"], "text": ["Acme"], "span": [[100, 104]], "conf": 0.5, "_win": 0}   # offsets don't index
    c1 = {"value": ["Acme"], "text": ["Acme"], "span": [[0, 4]], "conf": 0.5, "_win": 1}
    wins = [{"offset": 0, "len": 10}, {"offset": 5, "len": 10}]
    m = C.merge_mentions([c0, c1], wins, content)
    assert m["value"] == ["Acme"]                       # "Acme" occurs in content[5:10] = overlap


def test_call_sage2_merges_across_real_windows(monkeypatch):
    """End to end through call_sage2: forced 2 windows, a fake server that answers with real per-window offsets."""
    it = {**ITEMS["demo-inv-1"], "fields": [f for f in ITEMS["demo-inv-1"]["fields"] if f["name"] == "contacts"]}
    monkeypatch.setattr(C, "SAGE_WINDOW", 1500)
    size = max(1000, C.SAGE_WINDOW - common.n_tokens(json.dumps(C.sage2_body(it, "", "precise")["extract"])))
    e = common.enc()
    filler = lambda n: e.decode(e.encode(" ".join(f"w{i}" for i in range(4 * n)))[:n])   # noqa: E731
    # mentions ~120 tokens before the end of window 0, i.e. inside the 200-token overlap with window 1
    content = filler(size - 120) + " Dana Ruiz signs. Dana Ruiz again. " + filler(300)

    def fake(body, mock, item, arm, tag):
        w = body["content"]
        ms = [{"value": "Dana Ruiz", "text": "Dana Ruiz", "start": i, "end": i + 9, "probability": 0.8}
              for i in range(len(w)) if w.startswith("Dana Ruiz", i)]
        return 200, {"values": {"contacts": [m["value"] for m in ms]}, "results": {"contacts": {"values": ms}},
                     "meta": {"usage": {"input_tokens": 10, "output_tokens": 5}}}, 0.0, {}
    monkeypatch.setattr(C, "_sage_raw", fake)
    r = C.call_sage2(it, content, "precise", mock=True)
    assert r["n_calls"] >= 2 and all(w["offset"] is not None for w in r["windows"])
    assert r["preds"]["contacts"]["value"] == ["Dana Ruiz", "Dana Ruiz"]
    raw_n = sum(len(x["response"]["results"]["contacts"]["values"]) for x in r["raw_responses"])
    assert raw_n == 4 and r["preds"]["contacts"]["n_cross_window_dups_dropped"] == 2   # both seen twice, kept once
    assert len(r["raw_responses"]) == r["n_calls"] and r["harness_rev"] == C.HARNESS_REV


def test_raw_responses_all_systems_no_base64():
    it = ITEMS["demo-rcpt-1"]
    imgs = C.load_images(it, FIXTURES / "demo")
    rc = C.clairic_read(it, "ACME STORE total 193.00", imgs, mock=True)
    assert rc["raw_responses"] and rc["raw_responses"][0]["response"]["result"]["answers"]
    assert imgs[0]["base64"][:200] not in json.dumps(rc["raw_responses"])
    rs = C.call_sage2(ITEMS["demo-inv-1"], ITEMS["demo-inv-1"]["text"], "fast", mock=True)
    assert len(rs["raw_responses"]) == rs["n_calls"] and rs["raw_responses"][0]["response"]["values"] is not None
    ro = C.call_openrouter(it, C.openrouter_body(it, "x/y", None, imgs), mock=True)
    assert ro["raw_responses"][0]["response"]["choices"]
    echo = {"images": [{"content_type": "image/png", "base64": imgs[0]["base64"]}],
            "u": "data:image/png;base64," + imgs[0]["base64"], "keep": "short text"}
    s = json.dumps(C.strip_b64(echo))
    assert imgs[0]["base64"][:100] not in s and "short text" in s


def _run(tmp_path, *extra):
    return R.main(["read", "--run", "dryrun-f", "--root", str(FIXTURES), "--datasets", "demo",
                   "--arms", "sage_precise,clairic_text", "--workers", "1", *extra])


def test_ids_file_force_and_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path)
    ids = tmp_path / "ids.txt"
    ids.write_text("# re-run list\ndemo-inv-1\n")
    assert _run(tmp_path, "--ids-file", str(ids), "--final") == 0
    got = {p.stem for p in tmp_path.glob("dryrun-f/preds/*/demo/*.json")}
    assert got == {"demo-inv-1"}
    f = tmp_path / "dryrun-f/preds/sage_precise/demo/demo-inv-1.json"
    rec = json.loads(f.read_text())
    assert rec["harness_rev"] == C.HARNESS_REV and rec["raw_responses"]
    # an old (pre-fix) OK pred: --force re-reads it and moves the old one aside; without --force it is kept
    old = {**rec, "harness_rev": None, "marker": "old"}
    f.write_text(json.dumps(old))
    assert _run(tmp_path, "--ids-file", str(ids), "--final") == 0
    assert json.loads(f.read_text()).get("marker") == "old"
    assert _run(tmp_path, "--ids-file", str(ids), "--final", "--force") == 0
    prev = tmp_path / "dryrun-f" / R.PREV_DIR / "sage_precise/demo/demo-inv-1.json"
    assert json.loads(prev.read_text())["marker"] == "old"
    new = json.loads(f.read_text())
    assert "marker" not in new and new["replaced_prev"].startswith(R.PREV_DIR)
    # a second --force does not re-pay for preds already at this revision
    mt = f.stat().st_mtime_ns
    assert _run(tmp_path, "--ids-file", str(ids), "--final", "--force") == 0
    assert f.stat().st_mtime_ns == mt


def test_force_still_gates_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path)
    monkeypatch.setattr(R, "is_holdout", lambda it: True)
    assert _run(tmp_path, "--force") == 0
    assert not list(tmp_path.glob("dryrun-f/preds/sage_precise/demo/*.json"))
    assert list(tmp_path.glob("dryrun-f/preds/clairic_text/demo/*.json"))      # Clairic has no holdout
