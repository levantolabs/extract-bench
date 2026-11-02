"""Regression tests for the 2026-10-23 critic-pass fixes. Values are constructed here, not taken from any dataset."""
from extract_bench import clients as C, common, score as S


def _item(dataset, fields, meta=None):
    return {"id": "x", "dataset": dataset, "use_case": "t", "modality": "text", "text": "t", "text_source": "native",
            "images": [], "fields": fields, "metric": "exact_norm", "meta": meta or {}}


def test_legal_day_of_prediction_parses():
    assert S.exact_norm("15th day of January, 2019", "2019-01-15", "date", {"dayfirst": False}) == 1.0
    assert S.exact_norm("the 1st day of March 2010", "2010-03-01", "date", {}) == 1.0
    assert S.exact_norm("15th day of January, 2019", "2019-01-16", "date", {}) == 0.0


def test_gold_date_parsing_unchanged():
    # parse_date (used on gold and by vrdu/build.py) is untouched by the prediction-side leniency
    assert S.parse_date("15th day of January, 2019") is None


def test_cord_money_locale_hint():
    it = common.apply_hints(_item("cord", [{"name": "total_price", "type": "money", "gold": 45500}]))
    f = it["fields"][0]
    assert f["number_locale"] == "dot_thousands"
    assert S.exact_norm("45.500", 45500, "money", f) == 1.0
    assert S.exact_norm("Rp 45,500", 45500, "money", f) == 1.0
    assert S.exact_norm("45.50", 45500, "money", f) == 0.0
    u = common.apply_hints(_item("cord", [{"name": "total_price", "type": "money", "gold": 1}],
                                 {"unit_hint": {"total_price": "IDR"}}))["fields"][0]
    assert C.clairic_field(u)["unit"] == "IDR"


def test_list_items_from_meta_and_locale():
    it = common.apply_hints(_item("cord", [{"name": "menu_price", "type": "list", "gold": ["16,500", "13.000"]}],
                                  {"list_item_types": {"menu_price": "money"}}))
    f = it["fields"][0]
    assert f["items"] == "number" and f["number_locale"] == "dot_thousands"
    assert S.score_value([16500, 13000], f["gold"], "list", "exact_norm", f) == 1.0
    assert C.clairic_field(f)["items"] == "number"
    om = common.apply_hints(_item("omniai", [{"name": "a__amount", "type": "list", "gold": [6.2, 1500]}],
                                  {"field_meta": {"a__amount": {"item_type": "number"}}}))
    g = om["fields"][0]
    assert g["items"] == "number" and "number_locale" not in g
    assert S.score_value(["6.20", "$1,500.00"], g["gold"], "list", "exact_norm", g) == 1.0
    assert C.json_schema(om)["properties"]["a__amount"]["items"]["type"] == "number"


def test_hints_never_overwrite():
    it = common.apply_hints(_item("cord", [{"name": "l", "type": "list", "items": "text", "gold": ["a"],
                                            "number_locale": "x"}], {"list_item_types": {"l": "money"}}))
    assert it["fields"][0]["items"] == "text" and it["fields"][0]["number_locale"] == "x"
