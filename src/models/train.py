"""Train a global RandomForest demand model; evaluate on a date-based holdout."""
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.config import METRICS_FILE, MODEL_FILE, ROOT
from src.features.build_features import FEATURES, CATEGORICAL, NUMERIC, build_features


def train_model(df: pd.DataFrame, log_mlflow: bool = True, model_path: Path = MODEL_FILE, metrics_path: Path = METRICS_FILE) -> dict:
    feature_df = build_features(df)
    dates = np.sort(feature_df["date"].unique())
    if len(dates) < 35:
        raise ValueError("Not enough dated history after the lag-14 warmup; use at least 49 days")
    cutoff = dates[-28]
    train = feature_df.loc[feature_df["date"] < cutoff]
    test = feature_df.loc[feature_df["date"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Chronological holdout is empty")
    transform = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", "passthrough", NUMERIC),
    ])
    pipeline = Pipeline([
        ("preprocess", transform),
        ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42)),
    ])
    pipeline.fit(train[FEATURES], train["units_sold"])
    prediction = np.maximum(0, pipeline.predict(test[FEATURES]))
    actual = test["units_sold"].to_numpy()
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - prediction[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None
    metrics = {
        "mae": round(float(mean_absolute_error(actual, prediction)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, prediction))), 4),
        "mape_nonzero_pct": round(mape, 4) if mape is not None else None,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "holdout_start": str(pd.Timestamp(cutoff).date()),
        "max_data_date": str(pd.to_datetime(df["date"]).max().date()),
        "model": "RandomForestRegressor",
        "evaluation": "one-step-ahead with observed historical lag features",
    }
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    if log_mlflow:
        try:
            import mlflow
            uri = os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("demand-forecasting")
            with mlflow.start_run(run_name="random-forest-chronological-holdout"):
                mlflow.log_params({"model": "RandomForestRegressor", "n_estimators": 100, "min_samples_leaf": 2, "horizon_days": 28})
                mlflow.log_metrics({k: v for k, v in metrics.items() if k in ("mae", "rmse", "mape_nonzero_pct") and v is not None})
                mlflow.log_artifact(str(metrics_path))
                # Don't log potentially sensitive raw sales data to the tracking server.
        except Exception as exc:
            print(f"WARNING: MLflow tracking unavailable; model still saved: {exc}")
    return metrics
