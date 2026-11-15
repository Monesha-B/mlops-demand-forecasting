# MLOps Demand Forecasting Pipeline

**[Try the live Streamlit demo](https://mlops-demand-forecasting-demo.streamlit.app/)** · [GitHub Actions](https://github.com/Monesha-B/mlops-demand-forecasting/actions)

> The public demo is self-contained and uses synthetic retail data. PostgreSQL, FastAPI, MLflow, and Airflow run in the full local Docker stack; the hosted demo does not expose those backend services.

A portfolio-ready end-to-end demand forecasting example using **Python, Pandas, NumPy, scikit-learn, MLflow, PostgreSQL, FastAPI, Streamlit, Docker, Pytest, GitHub Actions, and Apache Airflow**.

**Demo workflow:** generate realistic *synthetic* daily store/product sales, build leakage-aware lag and calendar features, fit a Random Forest, evaluate on the most recent 28 days, log MLflow metrics, serve forecasts over FastAPI, and visualize an interactive 1–14-day forecast in Streamlit. Data and predictions are illustrative, not business forecasts.

## Directory structure

```text
src/ingestion/       synthetic data generation + PostgreSQL import
src/features/        lag/rolling and calendar features
src/models/          model fitting, evaluation, persistence
src/pipeline/        run end-to-end pipeline
api/                 FastAPI service (includes protected retraining route)
dashboard/           Streamlit UI (with local-demo fallback)
airflow/dags/        scheduled API orchestration (optional Docker profile)
tests/               Pytest
.github/workflows/   GitHub Actions CI
artifacts/           local model files (created when training; gitignored)
data/raw/            generated demand CSV (gitignored)
```

## Option A — local Mac, fastest first run

Requires Conda (Python 3.11); Docker not required for the first run.

```bash
cd ~/Downloads/mlops-demand-forecasting
conda create -n demand-forecast python=3.11 -y   # skip if it already exists
conda activate demand-forecast
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.pipeline.run --no-mlflow
python -m pytest -q
```

Open **two separate Terminal tabs**, activate the same Conda environment in both, then:

```bash
# Terminal 1
cd ~/Downloads/mlops-demand-forecasting
conda activate demand-forecast
uvicorn api.main:app --reload --port 8000
```

```bash
# Terminal 2
cd ~/Downloads/mlops-demand-forecasting
conda activate demand-forecast
streamlit run dashboard/app.py
```

Visit http://localhost:8501 (dashboard), http://localhost:8000/docs (interactive API), and http://localhost:8000/health.

For full local MLflow tracking, use the PostgreSQL-backed Docker Compose stack below; the filesystem-based `mlruns/` backend is deprecated in recent MLflow versions.

## Option B — run PostgreSQL, MLflow, API, Streamlit with Docker

Requires Docker Desktop running (allow several GB of memory for container images).

```bash
cd ~/Downloads/mlops-demand-forecasting
cp .env.example .env  # first run only; choose a private TRAINING_TOKEN
# edit .env to change TRAINING_TOKEN from the example before exposing API anywhere
docker compose up -d --build postgres mlflow api dashboard
```

Open:

- Streamlit: http://localhost:8522
- FastAPI docs: http://localhost:8002/docs
- MLflow: http://localhost:5002
- PostgreSQL: `localhost:5433` (container port `5432`), credentials from `.env` or Compose defaults; local development only

The API service initializes the dataset in PostgreSQL and trains the model on startup. To run SQL ingestion and training from your host Conda environment instead:

```bash
export DATABASE_URL=postgresql+psycopg2://forecast:forecast@localhost:5433/forecast
python -m src.pipeline.run --sync-db --use-db
```

To start **Airflow** as well (optional, much larger image):

```bash
# Create an 'airflow' database in the same PostgreSQL container first.
docker compose --profile orchestration up --build -d
# UI: http://localhost:8081
# Look in Airflow container logs for standalone admin password:
docker compose logs airflow
```

In Airflow, enable/run DAG `demand_forecasting_retrain`. Its task calls the API's token-protected retraining endpoint. Airflow runs are local-development demonstration only; it is not secured for production internet exposure.

To shut down: `docker compose --profile orchestration down` (or `docker compose down` without Airflow). Add `-v` only if you intend to delete persisted PostgreSQL data.

## Manual API example

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/catalog
curl -s -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"store_id":"S1", "product_id":"P1", "days":7}'
```

Retraining (only on trusted localhost environments):

```bash
curl -X POST http://localhost:8000/admin/retrain \
  -H "X-Training-Token: YOUR_PRIVATE_TOKEN"
```

## How the model works

- Generated daily synthetic sales for 3 stores, 4 products and ~240 days. Demand varies with weekly seasonality, product, promotions, prices and noise.
- Features: `lag_1`, `lag_7`, `lag_14`, 7-day rolling mean **shifted by one day**, calendar fields, promo, price, store ID and product ID. This avoids directly using same-day target values.
- The last 28 dates are **held out** for evaluation; `MAE`, `RMSE` and `MAPE` are saved in `artifacts/metrics.json` and optionally logged to MLflow. **Important:** holdout metrics are for one-step-ahead predictions with observed historical lags; they do not measure fully recursive multi-day rollout accuracy.
- Forecasts use recursive prediction: previous predicted demand is fed into future lag/rolling features; unknown future promotions are assumed off and future prices held at the latest observed price. Forecasts become more uncertain further out.
- This is a demonstrator, **not** a production-validated forecasting solution. Evaluate recursive rolling-origin forecasts and add real input data, data contracts, model monitoring, feature pipelines, security and scheduling reliability before real-world use.

## Tests and GitHub CI

```bash
python -m pytest -q
```

`.github/workflows/ci.yml` runs unit tests and trains a model without MLflow on every push/PR. The workflow deliberately uses the lean demo requirements to keep CI lightweight.

## GitHub (do **not** push secrets)

```bash
git init
git add .
git commit -m "Build demand forecasting MLOps MVP"
git branch -M main
# Create an empty repository on GitHub and replace USERNAME and REPO:
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

`.gitignore` excludes `.env`, generated sales CSV, model binaries, MLflow runs, caches and DB state. Check `git status` before pushing.

## Deploy public demo for free (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. On Streamlit Community Cloud create an app using your repository, branch `main` and entrypoint `dashboard/app.py`.
3. `dashboard/requirements.txt` installs only the lean demo stack, rather than heavyweight server components.
4. The dashboard generates synthetic data and trains a local demo model on first launch if no API is configured. **No PostgreSQL, API hosting or paid model keys required** for the public demo.
5. Test different store/product combinations and forecast horizons in the public URL.

Live app: https://mlops-demand-forecasting-demo.streamlit.app/

Public demo mode is a lightweight subset; the full Docker stack (PostgreSQL, MLflow, Airflow and API) runs locally. For a fully public API, deploy the API separately with protected retraining, persistent model storage, HTTPS and proper authorization.

## Common troubleshooting

- `conda: command not found`: install/initialize Miniconda, restart Terminal, run `conda init zsh` if necessary.
- `No module named ...`: activate `demand-forecast`, then `python -m pip install -r requirements.txt`.
- `port already in use`: check with `lsof -nP -iTCP:8000 -sTCP:LISTEN` (or `:8501`) and stop the existing service, or change the port.
- `Cannot connect to Docker daemon`: start Docker Desktop.
- PostgreSQL port 5432 already used: stop an existing local PostgreSQL or change host mapping in `docker-compose.yml`.
- `Model not found`: run `python -m src.pipeline.run --no-mlflow`; Streamlit demo mode can train on first launch.
- Airflow image too heavy on Mac: skip `--profile orchestration` until core app works.
- `MLflow unavailable`: model artifacts remain available; rerun with `--no-mlflow` if starting without the tracking service.

## Interview explanation (short)

"I built an end-to-end forecasting prototype. Pandas creates leakage-safe lagged features and scikit-learn trains a global demand model with a chronological holdout. MLflow tracks metrics and parameters. PostgreSQL stores the generated sales history, FastAPI exposes forecasts, and Streamlit provides interactive exploration. Docker Compose reproduces the local services, an optional Airflow DAG triggers retraining, and Pytest plus GitHub Actions validate the pipeline. The public demo uses local, synthetic-data inference to avoid paid infrastructure."
