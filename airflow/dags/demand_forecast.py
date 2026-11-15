"""Example Airflow 2.x DAG: trigger a private training run in the API service."""
import os
import urllib.request
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def trigger_retraining():
    token = os.getenv("TRAINING_TOKEN")
    if not token:
        raise RuntimeError("TRAINING_TOKEN is required")
    url = os.getenv("API_URL", "http://api:8000").rstrip("/") + "/admin/retrain"
    request = urllib.request.Request(url, data=b"{}", headers={"X-Training-Token": token, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"Retraining returned HTTP {response.status}")
        print(response.read().decode("utf-8")[:1500])


with DAG(
    dag_id="demand_forecasting_retrain",
    description="Daily retraining orchestration for the demo forecasting API",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"owner": "demo", "retries": 1, "retry_delay": timedelta(minutes=3)},
    tags=["mlops", "forecasting"],
) as dag:
    PythonOperator(task_id="train_and_log", python_callable=trigger_retraining)
