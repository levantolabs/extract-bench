"""Leaderboard, validator and degenerate-baseline checks on the real v0.1 data (no network)."""
import json

import pytest

from extract_bench import cli, report as R, score as S, validate as V
from extract_bench.common import RESULTS, load_items, load_split

SPLIT = load_split("v0.1")


def _gold_preds(it):
    preds = {}
    for f in it["fields"]:
        g = f.get("gold")
        if f["type"] != "list" and isinstance(g, list):   # list of acceptable alternatives: answer the first
            g = g[0] if g else None
        preds[f["name"]] = {"value": g, "conf": 1.0}
    return preds


@pytest.mark.parametrize("ds", ["fatura", "emails", "kleister_nda", "sroie", "squad2"])
def test_gold_answers_score_100(ds):
    items = {it["id"]: it for it in load_items(ds)}
    for iid in SPLIT[ds]["eval"]:
        fs = R.field_scores(items[iid], {"preds": _gold_preds(items[iid])})
        assert all(v[0] == 1.0 for v in fs.values()), (iid, fs)


@pytest.mark.parametrize("ds", ["fatura", "kleister_nda", "squad2", "cord"])
def test_all_null_scores_exactly_the_absent_share(ds):
    items = {it["id"]: it for it in load_items(ds)}
    for iid in SPLIT[ds]["eval"]:
        it = items[iid]
        fs = R.field_scores(it, {"preds": {f["name"]: {"value": None, "conf": None} for f in it["fields"]}})
        assert all(score == (1.0 if gold_null else 0.0) for score, gold_null in fs.values()), iid


def test_errored_read_drops_item_and_unparsed_scores_zero():
    it = next(i for i in load_items("emails") if i["id"] in SPLIT["emails"]["eval"])
    assert R.field_scores(it, {"error": "HTTP 500"}) is None
    assert R.field_scores(it, None) is None
    fs = R.field_scores(it, {"preds": {}, "unparsed": True})
    assert fs and all(v[0] == 0.0 for k, v in fs.items() if not v[1])


def test_report_reproduces_published_board():
    b = R.board(RESULTS / "v0.1", "v0.1", "eval")
    text = b["boards"]["text"]
    assert {k: round(v, 1) for k, v in text["mean"].items()} == {
        "sage2": 77.8, "clairic": 75.3, "nuextract3": 76.1, "schematron3b": 59.9, "gliner25": 35.7}
    assert len(text["mean_sets"]) == 10 and "sroie" not in text["mean_sets"]
    d = text["mean_diff_vs_ref"]["clairic"]
    assert (round(d["ci95"][0], 1), round(d["ci95"][1], 1), d["sig"]) == (-5.4, 0.2, False)
    committed = json.loads((RESULTS / "v0.1" / "board.json").read_text())
    assert committed["boards"]["text"]["mean"] == pytest.approx(text["mean"])


def test_validate_accepts_published_results():
    errors, warnings, summary = V.validate(RESULTS / "v0.1" / "nuextract3", "v0.1", partial=False)
    assert errors == [] and warnings == [] and len(summary) == 12


def test_validate_rejects_bad_submissions(tmp_path):
    assert V.validate(tmp_path, "v0.1", partial=False)[0]           # no metadata.json
    (tmp_path / "metadata.json").write_text(json.dumps({"system": "x", "input": "text"}))
    (tmp_path / "preds").mkdir()
    (tmp_path / "preds" / "emails.jsonl").write_text('{"id": "nope", "dataset": "emails", "preds": {}}\nnot json\n')
    errors, _, _ = V.validate(tmp_path, "v0.1", partial=True)
    joined = "\n".join(errors)
    assert "missing key 'display_name'" in joined and "unknown item id" in joined and "not JSON" in joined


def test_cli_entry_points(capsys):
    assert cli.main(["--version"]) == 0
    assert cli.main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "kleister_nda" in out and "v0.1 eval" in out
    assert cli.main(["nope"]) == 2


def test_scorer_is_the_published_one():
    """score_item is the one function every number comes from; spot-check its abstention rule (a)."""
    it = {"id": "x", "dataset": "emails", "metric": "exact_norm", "meta": {},
          "fields": [{"name": "a", "type": "text", "gold": None}, {"name": "b", "type": "text", "gold": "ACME"}]}
    sc = S.score_item(it, {"a": {"value": None}, "b": {"value": "acme"}})
    assert [f["score_a"] for f in sc["fields"]] == [1.0, 1.0]
    sc = S.score_item(it, {"a": {"value": "ACME"}, "b": {"value": None}})
    assert [f["score_a"] for f in sc["fields"]] == [0.0, 0.0]
