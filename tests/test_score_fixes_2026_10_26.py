"""Scorer fixes of 26 Oct 2026 (audit-2026-10-25/scoring.md #1-#9, validity.md F2-F4, F11). Hand-made values."""
import pytest

from extract_bench import score as S


def test_norm_text_dashes_and_quotes():
    assert S.norm_text("Anxiety – panic") == "anxiety - panic"
    assert S.norm_text("A—B−C‒D") == "a-b-c-d"
    assert S.norm_text("“O’Brien”") == "o'brien"
    assert S.exact_norm("Smith – Jones", "smith - jones", "text") == 1


def test_leading_article_stripped_for_text():
    assert S.exact_norm("the NRSC", "NRSC", "text") == 1
    assert S.exact_norm("The Pennsylvania Democratic Party", "Pennsylvania Democratic Party", "text") == 1
    assert S.exact_norm("An Owl", "owl", "text") == 1
    assert S.exact_norm("theatre", "atre", "text") == 0              # whole word only
    assert S.exact_norm("the", "the", "text") == 1                   # identical still matches
    assert S.exact_norm("A", "B", "text") == 0


def test_squash_fallback():
    assert S.exact_norm("Hung Cao forVirginia", "Hung Cao for Virginia", "text") == 1
    assert S.exact_norm("STEVE GARVEYFOR U.S. SENATE", "Steve Garvey for US Senate", "text") == 1
    assert S.exact_norm("12, JALAN TAMPOI 7/4, KAWASAN", "12, JALAN TAMPOI 7/4,KAWASAN", "text") == 1
    assert S.exact_norm("#1884", "1884", "text") == 1
    assert S.exact_norm("Anxiety /panic disorder", "Anxiety/panic disorder", "text") == 1
    assert S.exact_norm("05749- 7335", "05749-7335", "text") == 1
    # IDs / postcodes must not merge when any letter or digit differs
    assert S.exact_norm("SG18 9NR", "SG18 9N4", "text") == 0
    assert S.exact_norm("CB3 ODS", "CB3 0DS", "text") == 0
    assert S.exact_norm("INV-1001", "INV-1010", "text") == 0
    assert S.exact_norm("José", "Jos", "text") == 0                  # unicode-aware: accents kept
    # digit groups must not merge (hand-check 26 Oct: omniai-0371/0386, realdoc-mortgage_q229, omniai-0365)
    assert S.exact_norm("9341 12 Ash Way", "9341 1/2 Ash Way", "text") == 0
    assert S.exact_norm("57213/4 New Wolf Dr", "5721 3/4 New Wolf Dr", "text") == 0
    assert S.exact_norm("2.03(b)", "203(b)", "text") == 0
    assert S.exact_norm("2:4", "24", "text") == 0
    assert S.exact_norm("365 94 0248", "365-94-0248", "text") == 1
    assert S.exact_norm("(212) 841-8787", "212-841-8787", "text") == 1
    assert S.exact_norm("70139- 3623", "70139-3623", "text") == 1
    # both must be non-empty after squashing
    assert S.exact_norm("--", "..", "text") == 0
    assert S.exact_norm("***", "-", "text") == 0
    # squash is for text fields only; typed fields keep their own comparison
    assert S.exact_norm("12.50", "1250", "money") == 0


def test_text_contains_lenient():
    assert S.text_contains("State of Washington", "Washington")
    assert S.text_contains("the State of Delaware", "Delaware")
    assert not S.text_contains("the laws of the State of Delaware", "Delaware")  # 7 tokens > cap 6 for 1-token gold
    assert S.text_contains("TURNING POINT ACTION (TPACTION.COM)", "TURNING POINT ACTION")
    assert not S.text_contains("This Agreement shall be governed by and construed in accordance with the laws of "
                               "the State of Delaware", "Delaware")       # a whole clause
    assert not S.text_contains("Washingtonian Trust", "Washington")     # word boundary
    assert not S.text_contains("New York", "York City")
    assert not S.text_contains("", "x") and not S.text_contains("x", "")
    # cap = max(6, 3 x gold tokens): 9 tokens allowed for a 3-token gold, 10 rejected
    assert S.text_contains("a b c d e f Big Red Barn", "Big Red Barn")
    assert not S.text_contains("a b c d e f g Big Red Barn", "Big Red Barn")


def test_lenient_score_in_score_value_and_item():
    f = {"name": "j", "type": "text"}
    assert S.score_value("State of Delaware", "Delaware", "text", "exact_norm", f) == 0
    assert S.score_value("State of Delaware", "Delaware", "text", "exact_norm", f, lenient=True) == 1
    # lenient only touches exact_norm text fields
    assert S.score_value("12 apples", "12", "number", "exact_norm", f, lenient=True) == 1  # (strict number parse)
    assert S.score_value("x Paris y", "Paris", "text", "anls", f, lenient=True) == S.anls("x Paris y", "Paris")
    item = {"id": "i", "dataset": "d", "metric": "exact_norm", "fields": [
        {"name": "j", "type": "text", "gold": "Delaware"}, {"name": "k", "type": "text", "gold": "X"},
        {"name": "n", "type": "text", "gold": None}]}
    r = S.score_item(item, {"j": {"value": "State of Delaware", "conf": 0.9}, "k": {"value": "X", "conf": 0.9},
                            "n": {"value": "State of Y", "conf": 0.9}})
    by = {x["name"]: x for x in r["fields"]}
    assert by["j"]["score_a"] == 0 and by["j"]["score_lenient_a"] == 1 and by["j"]["score_lenient_b"] == 1
    assert by["k"]["score_lenient_a"] == 1
    assert by["n"]["score_lenient_a"] == 0                    # hallucination on a null gold stays wrong
    g = S.aggregate([r])
    assert g["field_acc_a"] == pytest.approx(1 / 3)
    assert g["field_acc_lenient_a"] == pytest.approx(2 / 3)
    assert g["present_gold_acc_lenient_a"] == pytest.approx(1.0)
    # old score files without lenient keys fall back to strict
    old = {**r, "fields": [{k: v for k, v in x.items() if not k.startswith("score_lenient")} for x in r["fields"]]}
    assert S.aggregate([old])["field_acc_lenient_a"] == pytest.approx(1 / 3)


def test_lenient_headline():
    item = {"id": "i", "dataset": "kleister_nda", "metric": "exact_norm",
            "meta": {"headline_fields": ["j"]}, "fields": [
                {"name": "j", "type": "text", "gold": "Delaware"}, {"name": "p", "type": "text", "gold": "Q"}]}
    r = S.score_item(item, {"j": {"value": "State of Delaware", "conf": 1}, "p": {"value": "Q", "conf": 1}})
    h = S.aggregate([r])["headline"]
    assert h["field_acc_a"] == 0 and h["field_acc_lenient_a"] == 1


def test_money_cents_tolerance():
    # audit #9: gold 914.84; 914.845 and 914.85 are now treated the same (both round to 914.85)
    assert S.exact_norm(914.845, 914.84, "money") == 0
    assert S.exact_norm(914.85, 914.84, "money") == 0
    assert S.exact_norm(914.845, 914.85, "money") == 1
    assert S.exact_norm(914.84, 914.84, "money") == 1
    assert S.exact_norm("1,080.00", 1080, "money") == 1
    assert S.exact_norm(1080.004, "1080.00", "money") == 1      # rounds to 1080.00
    assert S.exact_norm(1080.005, "1080.00", "money") == 0      # rounds half-up to 1080.01
    assert S.exact_norm(0.1 + 0.2, 0.3, "money") == 1
    assert S.exact_norm(-12.345, -12.35, "money") == 1


def test_gold_unparseable_date_unscored():
    item = {"id": "v", "dataset": "vrdu", "metric": "exact_norm", "fields": [
        {"name": "d1", "type": "date", "gold": "10121116"}, {"name": "d2", "type": "date", "gold": "Auguit9.2016"},
        {"name": "d3", "type": "date", "gold": "2016-08-09"}, {"name": "d4", "type": "date", "gold": None},
        {"name": "d5", "type": "date", "gold": ["junk", "9 Aug 2016"]}, {"name": "d6", "type": "date", "gold": "April 27, 193"}]}
    preds = {k: {"value": "2016-08-09", "conf": 0.9} for k in ("d1", "d2", "d3", "d4", "d5", "d6")}
    r = S.score_item(item, preds)
    by = {x["name"]: x for x in r["fields"]}
    for k in ("d1", "d2", "d6"):
        assert not by[k]["scored"] and by[k]["reason"] == "gold_unparseable"
    assert by["d3"]["scored"] and by["d3"]["score_a"] == 1
    assert by["d4"]["scored"] and by["d4"]["outcome_a"] == "hallucinated"   # null gold is not 'unparseable'
    assert by["d5"]["scored"] and by["d5"]["score_a"] == 1                  # one acceptable gold parses
    assert S.aggregate([r])["unscored_reasons"] == {"gold_unparseable": 3}
    # unsupported keeps its own reason
    r2 = S.score_item(item, {"d1": {"value": None, "unsupported": "no_type"}})
    assert {x["name"]: x.get("reason") for x in r2["fields"]}["d1"] == "no_type"


def test_parse_date_unchanged():
    # vrdu/build.py imports parse_date: its signature and behaviour must not move
    assert S.parse_date("10121116") == "1012-11-16"   # parses, but _plausible rejects year 1012 at scoring time
    assert S.parse_date("Auguit9.2016") is None
    assert S.parse_date("15/01/2019") == "2019-01-15"
    assert S.parse_date("01/02/2019", dayfirst=False) == "2019-01-02"


@pytest.mark.parametrize("s,exp", [
    ("$80 million", True), ("(1,512)", True), ("6%", True), ("43.24", True), ("2019", True), ("$5,053", True),
    ("-3.6", True), (12, True), ("a $2.43 billion decrease", False), ("July 3, 2018", False), ("FY18", False),
    ("12.5 years as of the acquisition date", False), ("brands and trademarks", False), (None, False)])
def test_is_numeric_answer(s, exp):
    assert S.is_numeric_answer(s) is exp


def test_tatqa_span_answers_token_f1():
    assert S.numeric_tol("Q4 sales", "q4 sales") == 1
    gold = "repurchase stock under our stock repurchase program"
    pred = "to repurchase stock under our stock repurchase program"
    assert S.numeric_tol(pred, gold) == pytest.approx(S.token_f1(pred, gold))
    assert 0.9 < S.numeric_tol(pred, gold) < 1
    assert S.numeric_tol("FY19", ["FY19"]) == 1
    assert S.numeric_tol("$80 million", "80") == 1 and S.numeric_tol("80", "$80 million") == 1
    assert S.numeric_tol("81", "$80 million") == 0                     # numeric gold stays tolerance-scored
    assert S.numeric_tol("an increase of $3", "a $2.43 billion decrease") == 0
