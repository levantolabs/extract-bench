"""Suite-integration features added 2026-10-23: clusters, item sets, views, native prompts, na_arms, Sage shapes,
list abstention, year bound, headline subsets, adjudicated gold, empty-OCR handling, export. No network."""
import json

import pytest
from conftest import FIXTURES

from extract_bench import clients as C, common, score as S
from extract_bench import run as R


def _it(**kw):
    base = {"id": "x-1", "dataset": "demo", "use_case": "t", "modality": "text", "text": "Total 5", "text_source": "native",
            "images": [], "metric": "exact_norm", "meta": {},
            "fields": [{"name": "total", "type": "money", "instructions": "Total", "gold": 5}]}
    base.update(kw)
    return base


def test_cluster_keys_per_dataset():
    assert common.cluster_of(_it(dataset="kleister_charity", meta={"doc_id": "d1", "charity_cluster": "charity-9"})) \
        == "charity_cluster:charity-9"
    assert common.cluster_of(_it(dataset="fatura", meta={"template": 49})) == "template:49"
    assert common.cluster_of(_it(dataset="cord", meta={"cluster_id": "c3"})) == "cluster_id:c3"
    assert common.cluster_of(_it(dataset="docvqa", meta={"doc_id": 14814})) == "14814"
    assert common.cluster_of(_it(dataset="sroie")) == "x-1"


def test_item_set_spec(tmp_path):
    d = tmp_path / "ds" / "anchor"
    d.mkdir(parents=True)
    (d / "all.jsonl").write_text(json.dumps(_it(dataset="ds")) + "\n")
    assert common.base_key("ds~anchor") == "ds"
    assert common.items_file("ds~anchor", tmp_path) == d / "all.jsonl"
    assert common.load_items("ds~anchor", tmp_path)[0]["id"] == "x-1"


def test_alt_text_sources(tmp_path):
    (tmp_path / "ocr").mkdir()
    (tmp_path / "ocr" / "g.md").write_text("GOLD MD")
    it = _it(meta={"ocr_alternatives": {"gold_markdown": "ocr/g.md"}, "oracle_text": "1 TART 13,000"})
    assert common.alt_text(it, tmp_path, "gold_markdown") == "GOLD MD"
    assert common.alt_text(it, tmp_path, "oracle_text") == "1 TART 13,000"
    assert common.alt_text(it, tmp_path, "pdf_text_layer") is None


def test_sage_shape_and_na_arms():
    assert C.sage_shape_for(_it(dataset="docvqa")) == "question"
    assert C.sage_shape_for(_it(dataset="squad2")) == "question"
    assert C.sage_shape_for(_it(dataset="x", meta={"sage_shape": "question"})) == "question"
    assert C.sage_shape_for(_it(dataset="fatura")) == C.SAGE_SHAPE
    f = {"name": "a", "type": "number", "na_arms": ["sage2"], "gold": 1}
    assert C.sage_type(f) is None and C.sage_unsupported(f) == "na_arms:sage2"
    it = _it(fields=[f, {"name": "b", "type": "text", "gold": "x"}])
    r = C.call_sage2(it, "content", "precise", mock=True)
    assert r["preds"]["a"]["unsupported"] == "na_arms:sage2" and "b" in r["preds"]
    q = C.call_sage2(_it(dataset="docvqa", fields=[{"name": "answer", "type": "text", "instructions": "Who?",
                                                    "gold": ["Bo"]}]), "content", "fast", mock=True)
    assert q["shape"] == "question" and q["n_calls"] == 1 and "answer" in q["preds"]


def test_views_and_native_prompt(tmp_path):
    comp_fields = [{"name": "total", "type": "text", "instructions": "Key 'total'", "gold": "5.00"}]
    sro = _it(dataset="sroie", images=["img/a.jpg"], meta={"comparable": {"prompt": "Fill {\"total\": \"\"}",
                                                                          "fields": comp_fields}})
    cord = _it(dataset="cord", meta={"comparable": {"prompt": "P", "image": "comparable/images/t.jpg"}})
    for arm, item, want in [("clairic_vision#comparable", sro, "fields"), ("sage_precise#comparable", cord, None),
                            ("or:openai/gpt-4o:image#comparable", cord, "image"), ("clairic_text#comparable", _it(), None)]:
        v = R.view_item(item, R.arm_spec(arm))
        if want is None:
            assert v is None
        elif want == "fields":
            assert v["fields"] == comp_fields and v["images"] == ["img/a.jpg"]
        else:
            assert v["images"] == ["comparable/images/t.jpg"] and v["fields"] == item["fields"]
    with pytest.raises(KeyError):
        R.arm_spec("clairic_text#nope")
    (tmp_path / "comparable").mkdir()
    (tmp_path / "comparable" / "p.txt").write_text("Extract:\nEmail body:\n{body}")
    (tmp_path / "comparable" / "s.json").write_text(json.dumps({"type": "object", "properties": {"total": {}}}))
    em = _it(meta={"comparable": {"prompt": "comparable/p.txt", "schema": "comparable/s.json"}})
    pr, sch = C.native_prompt(em, tmp_path)
    b = C.openrouter_native_body(em, "anthropic/claude-haiku-5.5", "BODY", None, pr, sch)
    assert len(b["messages"]) == 1 and b["messages"][0]["content"][-1]["text"].endswith("Email body:\nBODY")
    assert b["response_format"]["json_schema"]["strict"] is False and "temperature" not in b
    r = C.call_openrouter(em, b, mock=True)
    assert r["ok"] and "total" in r["preds"]


def test_list_abstention_and_year_bound():
    assert S._is_null([], "list") and S._is_null([None, " "], "list") and not S._is_null([], "text")
    assert S.outcome([], None, None, "list", "exact_norm", {}, "a")[0] == "correct_abstain"
    assert S.outcome([], None, ["A"], "list", "exact_norm", {}, "a")[0] == "false_abstain"
    # OCR-garbage gold 'April 27, 193' parses to year 193: no longer a date match for anything
    assert S.parse_date("April 27, 193") == "0193-04-27"           # parse_date unchanged (vrdu/build.py uses it)
    assert S.exact_norm("0193-04-27", "April 27, 193", "date") == 0.0
    assert S.exact_norm("1993-04-27", "April 27, 1993", "date") == 1.0


def test_headline_and_adjudicated_aggregate():
    it = _it(meta={"headline_fields": ["total"], "gold_adjudicated": {"total": {"gold": 7, "verdict": "wrong_value"}}},
             fields=[{"name": "total", "type": "money", "gold": 5}, {"name": "party", "type": "list", "gold": ["A"]}])
    sc = S.score_item(it, {"total": {"value": 7, "conf": 0.9}, "party": {"value": ["A"], "conf": None}})
    agg = S.aggregate([sc])
    assert agg["field_acc_a"] == 0.5 and agg["headline"]["field_acc_a"] == 0.0 and agg["headline"]["n_fields"] == 1
    assert agg["field_acc_adj_a"] == 1.0 and agg["n_adjudicated"] == 1
    assert agg["present_gold_acc_a"] == 0.5 and agg["null_gold_acc_a"] is None


def test_empty_ocr_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path / "runs")
    root = tmp_path / "datasets"
    (root / "emails").mkdir(parents=True)
    its = [_it(id="e1", dataset="emails", fields=[{"name": "committee", "type": "text", "gold": "NRSC"}],
               text="Paid for by NRSC", meta={"holdout": False}),
           _it(id="e2", dataset="emails", fields=[{"name": "committee", "type": "text", "gold": "DSCC"}],
               text="   ", meta={"holdout": False})]
    (root / "emails" / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in its))
    assert R.main(["read", "--run", "dryrun-e", "--root", str(root), "--arms", "clairic_text,sage_precise"]) == 0
    rec = json.loads((tmp_path / "runs/dryrun-e/preds/clairic_text/emails/e2.json").read_text())
    assert rec["input_empty"] and rec["preds"]["committee"]["value"] is None and rec["cost_usd"] == 0
    assert R.main(["score", "--run", "dryrun-e", "--root", str(root)]) == 0
    sc = json.loads((tmp_path / "runs/dryrun-e/scores/clairic_text/emails/e2.json").read_text())
    assert sc["fields"][0]["outcome_a"] == "false_abstain"
    assert R.main(["export", "--run", "dryrun-e", "--root", str(root)]) == 0
    lines = (tmp_path / "runs/dryrun-e/export/clairic_text/emails.score_emails.jsonl").read_text().splitlines()
    assert {json.loads(x)["id"] for x in lines} == {"e1", "e2"}
