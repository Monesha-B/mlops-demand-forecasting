import json
from pathlib import Path
import pandas as pd
import pytest
from src.ingestion.generate_data import generate_data
from src.models.predict import forecast
from src.models.train import train_model


@pytest.fixture(scope="module")
def sample_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    model_path = root / "model.joblib"
    metrics_path = root / "metrics.json"
    df = generate_data(days=75)
    result = train_model(df, log_mlflow=False, model_path=model_path, metrics_path=metrics_path)
    return df, model_path, metrics_path, result


def test_training_and_forecast(sample_model):
    df, model_path, metrics_path, metrics = sample_model
    assert model_path.is_file()
    assert json.loads(metrics_path.read_text())["test_rows"] > 0
    assert metrics["mae"] >= 0
    predictions = forecast(df, "S1", "P1", 7, model_path=model_path)
    assert len(predictions) == 7
    assert all(item["units_sold"] >= 0 for item in predictions)
    assert predictions[0]["date"] == (pd.to_datetime(df["date"]).max() + pd.Timedelta(days=1)).date().isoformat()


def test_invalid_forecast_arguments(sample_model):
    df, model_path, _, _ = sample_model
    with pytest.raises(ValueError):
        forecast(df, "S1", "P1", 15, model_path=model_path)
    with pytest.raises(ValueError):
        forecast(df, "NOT-A-STORE", "P1", 7, model_path=model_path)
