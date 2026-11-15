import pandas as pd
from src.features.build_features import build_features


def test_lags_use_past_sales_only():
    dates = pd.date_range("2025-01-01", periods=20)
    data = pd.DataFrame({
        "date": list(dates) * 2,
        "store_id": ["S1"] * 20 + ["S2"] * 20,
        "product_id": ["P1"] * 40,
        "price": [9.] * 40,
        "promo": [0] * 40,
        "units_sold": list(range(1, 21)) + list(range(101, 121)),
    })
    result = build_features(data)
    s1 = result[(result.store_id == "S1") & (result.product_id == "P1")]
    row = s1.iloc[0]
    assert row["units_sold"] == 15
    assert row["lag_1"] == 14
    assert row["lag_7"] == 8
    assert row["lag_14"] == 1
    assert row["rolling_7"] == 11
    s2 = result[result.store_id == "S2"].iloc[0]
    assert s2["lag_1"] == 114


def test_missing_columns_raise():
    try:
        build_features(pd.DataFrame({"date": []}))
    except ValueError as exc:
        assert "Missing columns" in str(exc)
    else:
        raise AssertionError("Expected missing-column error")
