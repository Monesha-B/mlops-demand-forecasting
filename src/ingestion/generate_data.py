"""Deterministic synthetic retail demand data; not real customer information."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import DATA_FILE


def generate_data(days: int = 240, seed: int = 42) -> pd.DataFrame:
    if days < 45:
        raise ValueError("At least 45 days are required for training")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for store_idx in range(1, 4):
        for product_idx in range(1, 5):
            base = 14 + store_idx * 4 + product_idx * 5
            base_price = 8.0 + product_idx * 2.5
            for day_num, date in enumerate(dates):
                promo = int(rng.random() < 0.13)
                price = round(base_price * (0.9 if promo else 1.0), 2)
                weekday_effect = 5 if date.dayofweek in (4, 5) else -2 if date.dayofweek == 0 else 0
                seasonal = 4 * np.sin(2 * np.pi * day_num / 30)
                trend = day_num * 0.015
                noise = rng.normal(0, 3.4)
                units = max(0, round(base + weekday_effect + seasonal + trend + 7 * promo + noise))
                rows.append((date, f"S{store_idx}", f"P{product_idx}", price, promo, units))
    return pd.DataFrame(rows, columns=["date", "store_id", "product_id", "price", "promo", "units_sold"])


def load_or_create_data(path: Path = DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_data().to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"])
