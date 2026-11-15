"""Streamlit dashboard, supports Docker API and self-contained public demo mode."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import streamlit as st
from src.config import METRICS_FILE, MODEL_FILE
from src.ingestion.generate_data import load_or_create_data
from src.models.predict import forecast
from src.models.train import train_model

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")
st.title("Demand Forecasting | MLOps Demo")
st.caption("Synthetic retail sales data · scikit-learn forecasts · 1–14 day horizon")

api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def api_active():
    try:
        status = requests.get(f"{api_url}/health", timeout=2)
        return status.ok and status.json().get("model_ready") is True
    except requests.RequestException:
        return False


use_api = api_active()
if not use_api and not MODEL_FILE.exists():
    with st.spinner("Preparing first-run demo model using synthetic sales data..."):
        train_model(load_or_create_data(), log_mlflow=False)

history = load_or_create_data()
stores = sorted(history["store_id"].unique().tolist())
products = sorted(history["product_id"].unique().tolist())
left, middle, right = st.columns(3)
with left:
    store = st.selectbox("Store", stores)
with middle:
    product = st.selectbox("Product", products)
with right:
    horizon = st.slider("Forecast horizon (days)", 1, 14, 7)

if use_api:
    try:
        response = requests.post(f"{api_url}/forecast", json={"store_id": store, "product_id": product, "days": horizon}, timeout=20)
        response.raise_for_status()
        preds = response.json()["predictions"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        st.error(f"API forecast unavailable: {exc}")
        st.stop()
else:
    preds = forecast(history, store, product, horizon)

actual = history.loc[(history["store_id"] == store) & (history["product_id"] == product)].sort_values("date").tail(30)
future = pd.DataFrame(preds)
future["date"] = pd.to_datetime(future["date"])

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Last actual day", int(actual["units_sold"].iloc[-1]))
metric2.metric("Next-day forecast", f"{future['units_sold'].iloc[0]:.1f}")
metric3.metric("Forecast sum", f"{future['units_sold'].sum():.1f}")

st.subheader("Sales and projected demand")
actual_plot = actual[["date", "units_sold"]].rename(columns={"units_sold": "Actual sales"}).set_index("date")
forecast_plot = future[["date", "units_sold"]].rename(columns={"units_sold": "Forecast sales"}).set_index("date")
st.line_chart(actual_plot.join(forecast_plot, how="outer"), use_container_width=True)
st.subheader("Forecast table")
st.dataframe(future.rename(columns={"units_sold": "predicted_units"}), hide_index=True, use_container_width=True)

if METRICS_FILE.exists():
    metrics = json.loads(METRICS_FILE.read_text())
    st.caption(f"Validation MAE: {metrics['mae']:.2f} · RMSE: {metrics['rmse']:.2f} · One-step test, not multi-day backtesting")
st.info("Demo assumptions: future promotions are off and prices stay constant. Forecasts are recursive and based on synthetic data. This is a learning project, not production planning advice.")
st.caption("Running mode: " + ("FastAPI service" if use_api else "Standalone demo (no hosted API required)"))
