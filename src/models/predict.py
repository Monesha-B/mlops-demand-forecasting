"""Iterative multi-day forecasts using a persisted scikit-learn pipeline."""
from datetime import timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_FILE
from src.features.build_features import FEATURES


def forecast(df: pd.DataFrame, store_id: str, product_id: str, days: int = 7, model_path: Path = MODEL_FILE) -> list[dict]:
    if not (1 <= days <= 14):
        raise ValueError("days must be from 1 to 14")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("Model missing. Train first with: python -m src.pipeline.run --no-mlflow")
    series = df.loc[(df["store_id"] == store_id) & (df["product_id"] == product_id)].copy()
    if len(series) < 14:
        raise ValueError("Unknown store/product or fewer than 14 days of sales history")
    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values("date")
    values = [float(v) for v in series["units_sold"]]
    date = series["date"].iloc[-1]
    price = float(series["price"].iloc[-1])
    model = joblib.load(model_path)
    result = []
    for _ in range(days):
        date += timedelta(days=1)
        row = pd.DataFrame([{
            "store_id": store_id,
            "product_id": product_id,
            "price": price,
            "promo": 0,  # unknown future promotion: assume none
            "day_of_week": date.dayofweek,
            "day_of_month": date.day,
            "month": date.month,
            "is_weekend": int(date.dayofweek >= 5),
            "lag_1": values[-1],
            "lag_7": values[-7],
            "lag_14": values[-14],
            "rolling_7": float(np.mean(values[-7:])),
        }])[FEATURES]
        estimate = max(0.0, float(model.predict(row)[0]))
        result.append({"date": date.date().isoformat(), "units_sold": round(estimate, 2)})
        values.append(estimate)
    return result
