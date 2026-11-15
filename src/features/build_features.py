"""Time-series features with all sales history shifted into the past."""
import pandas as pd

CATEGORICAL = ["store_id", "product_id"]
NUMERIC = ["price", "promo", "day_of_week", "day_of_month", "month", "is_weekend", "lag_1", "lag_7", "lag_14", "rolling_7"]
FEATURES = CATEGORICAL + NUMERIC


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "store_id", "product_id", "units_sold", "price", "promo"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = add_calendar_features(df)
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    series = df.groupby(["store_id", "product_id"], sort=False)["units_sold"]
    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = series.shift(lag)
    df["rolling_7"] = series.transform(lambda values: values.shift(1).rolling(7, min_periods=7).mean())
    return df.dropna(subset=["lag_1", "lag_7", "lag_14", "rolling_7"]).reset_index(drop=True)
