"""Run generation -> optional PostgreSQL sync/read -> train -> save metrics."""
import argparse
import json
from src.ingestion.generate_data import load_or_create_data, generate_data
from src.config import DATA_FILE
from src.models.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true", help="Overwrite existing synthetic sample CSV")
    parser.add_argument("--sync-db", action="store_true", help="Replace example sales table in PostgreSQL")
    parser.add_argument("--use-db", action="store_true", help="Train using PostgreSQL table, not CSV")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow tracking (useful for first local run/CI)")
    args = parser.parse_args()
    if args.regenerate:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        generate_data().to_csv(DATA_FILE, index=False)
    data = load_or_create_data()
    print(f"Loaded {len(data)} synthetic rows")
    if args.sync_db or args.use_db:
        from src.ingestion.database import sync_to_database, load_from_database
        if args.sync_db:
            sync_to_database(data)
            print("Synchronized sample demand into PostgreSQL")
        if args.use_db:
            data = load_from_database()
            print(f"Loaded {len(data)} rows from PostgreSQL")
    metrics = train_model(data, log_mlflow=not args.no_mlflow)
    print(json.dumps(metrics, indent=2))
    print("Saved model and metrics under artifacts/")


if __name__ == "__main__":
    main()
