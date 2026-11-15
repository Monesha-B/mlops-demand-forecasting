"""Minimal PostgreSQL ingestion. Database use is optional for standalone demo."""
import os
import pandas as pd


def _engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL before requesting PostgreSQL operations")
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)


def sync_to_database(df: pd.DataFrame) -> None:
    # Deterministic sample-data reset; production should use migrations + idempotent upserts.
    with _engine().begin() as conn:
        df.to_sql("daily_demand", conn, if_exists="replace", index=False, chunksize=500, method="multi")


def load_from_database() -> pd.DataFrame:
    from sqlalchemy import text
    with _engine().connect() as conn:
        df = pd.read_sql(text("SELECT date, store_id, product_id, price, promo, units_sold FROM daily_demand"), conn)
    df["date"] = pd.to_datetime(df["date"])
    return df
