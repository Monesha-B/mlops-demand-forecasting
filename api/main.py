"""Forecasting REST API. Keep the retrain endpoint private; serve read-only forecasts publicly."""
import os
import secrets
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from src.config import MODEL_FILE, METRICS_FILE
from src.ingestion.generate_data import load_or_create_data
from src.models.predict import forecast
from src.models.train import train_model

app = FastAPI(title="MLOps Demand Forecasting", version="1.0.0")


def load_input_data():
    if os.getenv("DATABASE_URL"):
        from src.ingestion.database import load_from_database
        return load_from_database()
    return load_or_create_data()


class ForecastRequest(BaseModel):
    store_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    days: int = Field(default=7, ge=1, le=14)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_FILE.exists()}


@app.get("/catalog")
def catalog():
    df = load_input_data()
    return {"stores": sorted(df["store_id"].unique().tolist()), "products": sorted(df["product_id"].unique().tolist())}


@app.get("/metrics")
def metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="Train model first")
    import json
    return json.loads(METRICS_FILE.read_text())


@app.post("/forecast")
def predict(request: ForecastRequest):
    try:
        data = load_input_data()
        predictions = forecast(data, request.store_id, request.product_id, request.days)
        return {"store_id": request.store_id, "product_id": request.product_id, "days": request.days, "predictions": predictions}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/admin/retrain")
def retrain(x_training_token: str | None = Header(default=None)):
    expected = os.getenv("TRAINING_TOKEN")
    if not expected or not x_training_token or not secrets.compare_digest(x_training_token, expected):
        raise HTTPException(status_code=403, detail="Invalid or unset training token")
    metrics = train_model(load_input_data(), log_mlflow=True)
    return {"trained": True, "metrics": metrics}
