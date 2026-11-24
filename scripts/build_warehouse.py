#!/usr/bin/env python3
"""Load the clean SMARD snapshot into the PostgreSQL reporting warehouse.

Run after ingestion, and again after scripts/benchmark.py if you want the
forecast fact table populated. Needs a running Postgres (`docker compose up -d postgres`,
or any server named by DATABASE_URL):

    PYTHONPATH=src python scripts/build_warehouse.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from energy_forecast.data import load_clean_demand
from energy_forecast.warehouse import build_warehouse, query

POOLED_PATH = Path("reports/benchmark_pooled_hourly.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-forecasts", action="store_true",
        help="build only fact_demand and dim_calendar, ignoring the benchmark output",
    )
    args = parser.parse_args()

    demand = load_clean_demand()
    forecasts = None
    if not args.skip_forecasts and POOLED_PATH.exists():
        forecasts = pd.read_csv(POOLED_PATH, parse_dates=["origin", "timestamp"])

    target = build_warehouse(demand, forecasts=forecasts)
    counts = query(
        "SELECT 'fact_demand' AS table_name, count(*) AS rows FROM fact_demand "
        "UNION ALL SELECT 'dim_calendar', count(*) FROM dim_calendar "
        "UNION ALL SELECT 'fact_forecast', count(*) FROM fact_forecast",
        target,
    )
    print(f"Wrote schema {target.schema!r}")
    print(counts.to_string(index=False))
    if forecasts is None:
        print(f"\nfact_forecast is empty: run scripts/benchmark.py to produce {POOLED_PATH}.")


if __name__ == "__main__":
    main()
