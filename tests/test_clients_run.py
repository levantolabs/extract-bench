"""Request-body shapes, the live gate, and an end-to-end mock run on the fixture dataset (no network)."""
import json
import os

import pytest
from conftest import FIXTURES

from extract_bench import clients as C, common, ocr
from extract_bench import run as R

ITEMS = {it["id"]: it for it in common.load_items("demo", FIXTURES)}


def test_clairic_body_types():
    it = ITEMS["demo-inv-1"]
    b = C.clairic_body(it, it["text"])
    f = b["fields"]
    assert b["model"] == "clairic" and b["state"] == it["text"] and "images" not in b
    assert f["total"] == {"instructions": "Amount due including shipping", "type": "number", "decimals": 2, "unit": "USD"}
    assert f["pallets"]["type"] == "number" and f["pallets"]["integer"] is True
    assert f["contacts"] == {"instructions": "Contact people", "type": "list", "items": "text"}
    assert f["issue_date"]["type"] == "date"
    with pytest.raises(ValueError):
        C.clairic_body(it, "", None)


def test_clairic_images():
    it = ITEMS["demo-rcpt-1"]
    imgs = C.load_images(it, FIXTURES / "demo")
    b = C.clairic_body(it, None, imgs)
    assert b["state"] == "" and b["images"][0]["content_type"] == "image/png" and "base64" in b["images"][0]


def test_sage2_body_types():
    it = ITEMS["demo-inv-1"]
    b = C.sage2_body(it, it["text"], "precise")
    ext = b["extract"]
    assert b["mode"] == "precise" and isinstance(ext["contacts"], list)  # 2.0.0: every-mention list
    assert ext["contacts"][0]["type"] == "text" and "definition" in ext["contacts"][0]
    assert "definition" in ext["total"] and "instructions" not in ext["total"]
    assert ext["total"]["type"] == "money" and ext["pallets"]["type"] == "integer"
    assert ext["issue_date"]["type"] == "date" and ext["invoice_number"]["type"] == "text"
    q = C.sage2_body(it, "x", "fast", field={"name": "q", "type": "text", "instructions": 'say "hi"\nnow'})
    assert q["extract"] == "say 'hi' now"
    with pytest.raises(ValueError):
        C.sage2_body(it, "x", "slow")


def test_openrouter_body_schema():
    it = ITEMS["demo-inv-1"]
    b = C.openrouter_body(it, "anthropic/claude-haiku-5.5", it["text"])
    sch = b["response_format"]["json_schema"]["schema"]
    assert set(sch["required"]) == {f["name"] for f in it["fields"]} and sch["additionalProperties"] is False
    assert sch["properties"]["total"]["type"] == ["number", "null"]
    assert sch["properties"]["contacts"]["type"] == ["array", "null"]
    assert "temperature" not in b
    assert C.openrouter_body(it, "google/gemini-3.8-flash", "t")["temperature"] == 0


def test_mock_parsers_roundtrip():
    it = ITEMS["demo-inv-1"]
    rc = C.clairic_read(it, it["text"], mock=True)
    assert rc["ok"] and set(rc["preds"]) == {f["name"] for f in it["fields"]} and rc["cost_usd"] > 0
    rs = C.call_sage2(it, it["text"], "precise", mock=True)
    assert rs["ok"] and isinstance(rs["preds"]["contacts"]["value"], list)
    rq = C.call_sage2(it, it["text"], "fast", mock=True, shape="question")
    assert rq["ok"] and rq["n_calls"] == 6 and isinstance(rq["preds"]["contacts"]["value"], list)
    ro = C.call_openrouter(it, C.openrouter_body(it, "x/y", it["text"]), mock=True)
    assert ro["ok"] and "total" in ro["preds"]


def test_sage_windows_split_long_content():
    it = ITEMS["demo-inv-1"]
    long = ("word " * 40000).strip()
    r = C.call_sage2(it, long, "fast", mock=True)
    assert r["n_calls"] >= 3


def test_live_gate(monkeypatch):
    monkeypatch.delenv(C.LIVE_ENV, raising=False)
    it = ITEMS["demo-inv-1"]
    with pytest.raises(C.LiveCallBlocked):
        C.call_clairic(it, C.clairic_body(it, it["text"]), mock=False)
    monkeypatch.setenv(C.LIVE_ENV, "1")
    monkeypatch.delenv(C.CAP_ENV, raising=False)
    with pytest.raises(C.LiveCallBlocked):
        C.call_sage2(it, it["text"], "fast", mock=False)
    with pytest.raises(SystemExit):     # --live without the env var is refused by the CLI too
        monkeypatch.delenv(C.LIVE_ENV)
        R.main(["read", "--run", "x", "--live", "--root", str(FIXTURES)])


def test_mock_run_names(tmp_path):
    with pytest.raises(SystemExit):
        R.main(["read", "--run", "not-dry", "--root", str(FIXTURES)])


def test_end_to_end_mock(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path)
    arms = "clairic_text,clairic_vision,clairic_vision_ocr,sage_precise,haiku_json,gemini_vision_json,clairic_text@gold_md"
    rc = R.main(["read", "--run", "dryrun-t", "--root", str(FIXTURES), "--datasets", "demo", "--arms", arms,
                 "--final", "--workers", "2"])
    assert rc == 0
    preds = list(tmp_path.glob("dryrun-t/preds/*/demo/*.json"))
    assert preds and all(json.loads(p.read_text())["mock"] for p in preds)
    # vision arms only touch the image item; harness OCR fills the image item's text for text arms
    assert {p.stem for p in tmp_path.glob("dryrun-t/preds/clairic_vision/demo/*.json")} == {"demo-rcpt-1"}
    if ocr.available():
        rec = json.loads((tmp_path / "dryrun-t/preds/clairic_text/demo/demo-rcpt-1.json").read_text())
        assert rec["text_source"].startswith("harness_ocr:tesseract")
    rc = R.main(["score", "--run", "dryrun-t", "--root", str(FIXTURES), "--datasets", "demo",
                 "--pair", "clairic_text", "sage_precise"])
    s = json.loads((tmp_path / "dryrun-t/summary.json").read_text())
    assert s["mock"] is True and "clairic_text" in s["arms"] and s["pairs"]


def test_ocr_fixture():
    if not ocr.available():
        pytest.skip("tesseract not installed")
    r = ocr.ocr_item(ITEMS["demo-rcpt-1"], FIXTURES / "demo", use_cache=False)
    assert "193.00" in r["text"] and r["version"].startswith("tesseract")


def test_sage_one_keeps_candidates_and_forced_answer():
    """A null Sage result keeps its ranked candidates; the top one is the forced answer (no diagnostics header needed)."""
    from extract_bench import clients as C
    r = {"value": None, "probability": 0.62, "probabilities": [
        {"value": "Beacon", "text": "Beacon", "start": 38, "end": 44, "probability": 0.30},
        {"value": "Acme", "text": "Acme", "start": 0, "end": 4, "probability": 0.08},
        {"value": None, "probability": 0.62}]}
    one = C._sage_one(r)
    assert one["value"] is None and one["p_absent"] == 0.62
    assert [c["value"] for c in one["candidates"]] == ["Beacon", "Acme"]
    assert one["forced"] == "Beacon" and one["forced_conf"] == 0.30
    hit = C._sage_one({"value": "Beacon", "probability": 0.9, "probabilities": [{"value": "Beacon", "probability": 0.9}, {"value": None, "probability": 0.1}]})
    assert hit["forced"] == "Beacon" and hit["forced_conf"] == 0.9
    empty = C._sage_one({"value": None, "probability": 1.0, "probabilities": [{"value": None, "probability": 1.0}]})
    assert empty["forced"] is None and empty["candidates"] == []
