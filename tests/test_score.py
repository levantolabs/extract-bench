"""Hand-made cases for every scorer. Values are constructed here, not taken from any dataset."""
import math

import pytest

from extract_bench import score as S


@pytest.mark.parametrize("s,exp", [
    ("$1,234.56", 1234.56), ("RM 193.00", 193.0), ("1.234,56", 1234.56), ("12,5", 12.5), ("1,234", 1234.0),
    ("(12.50)", -12.5), ("-3", -3.0), ("20.5%", 20.5), (7, 7.0), ("1.234.567", 1234567.0), ("abc", None),
    ({"amount": 9.99, "currency": "USD"}, 9.99), (None, None), (True, None)])
def test_parse_number(s, exp):
    assert S.parse_number(s) == exp


def test_parse_number_dot_thousands():
    assert S.parse_number("Rp 1.234", "dot_thousands") == 1234.0
    assert S.parse_number("1.234") == 1.234


@pytest.mark.parametrize("s,df,exp", [
    ("2019-01-15", None, "2019-01-15"), ("15/01/2019", True, "2019-01-15"), ("01/15/2019", None, "2019-01-15"),
    ("03/04/2019", True, "2019-04-03"), ("03/04/2019", False, "2019-03-04"), ("15 Jan 2019", None, "2019-01-15"),
    ("January 15, 2019", None, "2019-01-15"), ("15-Jan-19", None, "2019-01-15"), ("2019.01.15", None, "2019-01-15"),
    ("20190115", None, "2019-01-15"), ("1st of nothing", None, None), ("31/02/2019", True, None),
    ("Sept 3rd, 2021", None, "2021-09-03")])
def test_parse_date(s, df, exp):
    assert S.parse_date(s, df) == exp


@pytest.mark.parametrize("s,exp", [("14:05", "14:05:00"), ("2:05 pm", "14:05:00"), ("12:00 am", "00:00:00"),
                                   ("14:05:09", "14:05:09"), ("1405", "14:05:00"), ("noon", None)])
def test_parse_time(s, exp):
    assert S.parse_time(s) == exp


@pytest.mark.parametrize("s,unit,days", [("P2Y", None, 730), ("two (2) years", None, 730), ("18 months", None, 540),
                                         ("PT90M", None, 90 / 1440), (3, "days", 3), ("1 year 6 months", None, 545)])
def test_parse_duration(s, unit, days):
    assert math.isclose(S.parse_duration(s, unit) / 86400, days, rel_tol=1e-9)


def test_exact_norm_types():
    assert S.exact_norm("  ACME Store. ", "acme store", "text") == 1
    assert S.exact_norm("ACME Stores", "acme store", "text") == 0
    assert S.exact_norm("$1,080.00", 1080, "money") == 1
    assert S.exact_norm(1080.004, "1080.00", "money") == 1
    assert S.exact_norm(1080.01, "1080.00", "money") == 0
    assert S.exact_norm("4", 4, "number") == 1
    assert S.exact_norm("15/01/2019", "2019-01-15", "date", {"dayfirst": True}) == 1
    assert S.exact_norm("2019-01-16", "2019-01-15", "date") == 0
    assert S.exact_norm("2:05 PM", "14:05", "time") == 1
    assert S.exact_norm("2019-01-15T14:05:00", "15 Jan 2019 2:05 pm", "datetime") == 1
    assert S.exact_norm("12 months", "P1Y", "duration") == 1        # within 2%
    assert S.exact_norm("2 years", "P1Y", "duration") == 0
    assert S.exact_norm("France", ["Paris", "france"], "text") == 1  # any acceptable gold
    assert S.exact_norm("", "", "text") == 0                         # empty never matches


def test_anls():
    assert S.anls("Paris", "paris") == 1
    assert math.isclose(S.anls("Pariss", "paris"), 1 - 1 / 6)
    assert S.anls("London", "paris") == 0       # NL >= 0.5 -> 0
    assert S.anls("x", ["abc", "x "]) == 1


def test_token_f1():
    assert S.token_f1("the France", "France") == 1
    assert math.isclose(S.token_f1("in northern France", "France"), 2 * (1 / 3 * 1) / (1 / 3 + 1))
    assert S.token_f1("", "") == 1
    assert S.token_f1("Spain", ["France", "in France"]) == 0


def test_numeric_tol():
    assert S.numeric_tol("20.5%", 20.5) == 1
    assert S.numeric_tol(20.52, 20.5) == 0
    assert S.numeric_tol(1000.05, 1000) == 1     # rel 1e-4
    assert S.numeric_tol(1000.9, 1000) == 0
    assert S.numeric_tol("Q4 sales", "q4 sales") == 1  # span answers fall back to text


def test_list_f1():
    assert S.list_f1(["Dana Ruiz", "priya shah"], ["Dana Ruiz", "Priya Shah"]) == 1
    assert math.isclose(S.list_f1(["Dana Ruiz"], ["Dana Ruiz", "Priya Shah"]), 2 / 3)
    assert S.list_f1([], []) == 1
    assert S.list_f1(["1,000", "2"], [1000, 2.0], "number") == 1
    assert S.list_f1(["x"], [["a"], ["x"]]) == 1   # list of acceptable lists


def test_edit_sim_and_date_exact():
    assert S.edit_sim("abcd", "abce") == 0.75
    assert S.date_exact("Jan 15 2019", "2019-01-15") == 1
    assert S.date_exact(None, "2019-01-15") == 0


def test_outcomes_rules():
    f = {"name": "x", "type": "text"}
    assert S.outcome(None, None, None, "text", "exact_norm", f, "a") == ("correct_abstain", 1.0)
    assert S.outcome("v", 0.9, None, "text", "exact_norm", f, "a") == ("hallucinated", 0.0)
    assert S.outcome(None, 0.9, "v", "text", "exact_norm", f, "a") == ("false_abstain", 0.0)
    assert S.outcome("v", 0.9, "v", "text", "exact_norm", f, "a") == ("answered", 1.0)
    # rule b: low confidence counts as abstaining
    assert S.outcome("v", 0.3, None, "text", "exact_norm", f, "b") == ("correct_abstain", 1.0)
    assert S.outcome("v", 0.3, "v", "text", "exact_norm", f, "b") == ("false_abstain", 0.0)
    assert S.outcome("v", None, "v", "text", "exact_norm", f, "b") == ("answered", 1.0)  # no conf (LLM) -> rule a


def test_score_item_and_aggregate():
    item = {"id": "i1", "dataset": "d", "metric": "exact_norm", "fields": [
        {"name": "a", "type": "text", "gold": "X"}, {"name": "b", "type": "money", "gold": 10},
        {"name": "c", "type": "text", "gold": None}, {"name": "d", "type": "list", "gold": ["p", "q"]}]}
    preds = {"a": {"value": "x", "conf": 0.9}, "b": {"value": "10.00", "conf": 0.4}, "c": {"value": "Y", "conf": 0.2},
             "d": {"value": None, "conf": None, "unsupported": "sage2_no_list_type"}}
    r = S.score_item(item, preds)
    assert r["n_scored"] == 3
    assert r["item_acc_a"] == pytest.approx(2 / 3)          # a ok, b ok, c hallucinated
    assert r["item_acc_b"] == pytest.approx(2 / 3)          # a ok, b false-abstain, c correct-abstain
    assert r["item_cov_adj_a"] == pytest.approx(2 / 4)
    g = S.aggregate([r])
    assert g["pct_gold_fields_scored"] == 0.75
    assert g["outcomes_a"] == {"answered": 2, "hallucinated": 1}
    assert g["outcomes_b"] == {"answered": 1, "false_abstain": 1, "correct_abstain": 1}
    assert g["unscored_reasons"] == {"sage2_no_list_type": 1}


def test_paired_bootstrap_and_holm():
    a = {f"i{k}": 1.0 for k in range(50)}
    b = {f"i{k}": float(k % 2) for k in range(50)}
    bs = S.paired_bootstrap(a, b, n=2000, seed=20)
    assert bs["diff"] == pytest.approx(0.5) and bs["ci_lo"] > 0.3 and bs["p_two_sided"] < 0.01
    same = S.paired_bootstrap(a, dict(a), n=500)
    assert same["diff"] == 0 and same["ci_lo"] == 0 == same["ci_hi"]
    # clustering: 5 docs x 10 questions -> 5 clusters
    cl = {f"i{k}": f"doc{k // 10}" for k in range(50)}
    assert S.paired_bootstrap(a, b, cl, n=200)["n_clusters"] == 5
    h = S.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert h["x"]["reject"] and not h["y"]["reject"] and not h["z"]["reject"]
    assert h["x"]["p_adj"] == pytest.approx(0.03) and h["z"]["p_adj"] == pytest.approx(0.06)


def test_forced_answer_scoring():
    """score_forced: the forced guess is scored like an answer; a guess on a field with no gold scores 0."""
    item = {"id": "x", "dataset": "t", "metric": "exact_norm", "fields": [
        {"name": "a", "type": "text", "gold": "Beacon"}, {"name": "b", "type": "text", "gold": None},
        {"name": "c", "type": "text", "gold": "Acme"}]}
    preds = {"a": {"value": None, "conf": 0.6, "forced": "Beacon", "forced_conf": 0.3},
             "b": {"value": None, "conf": 0.9, "forced": "Zeta", "forced_conf": 0.1},
             "c": {"value": None, "conf": 0.2, "guess": "Acme"}}            # Clairic-style: guess field
    rec = {r["name"]: r for r in S.score_item(item, preds)["fields"]}
    assert rec["a"]["score_a"] == 0.0 and rec["a"]["score_forced"] == 1.0
    assert rec["b"]["score_a"] == 1.0 and rec["b"]["score_forced"] == 0.0
    assert rec["c"]["forced_pred"] == "Acme" and rec["c"]["score_forced"] == 1.0
