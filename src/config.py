"""Filesystem configuration for local and Docker execution."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = Path(os.getenv("DATA_FILE", str(ROOT / "data" / "raw" / "demand.csv")))
MODEL_FILE = Path(os.getenv("MODEL_FILE", str(ROOT / "artifacts" / "demand_model.joblib")))
METRICS_FILE = Path(os.getenv("METRICS_FILE", str(ROOT / "artifacts" / "metrics.json")))
