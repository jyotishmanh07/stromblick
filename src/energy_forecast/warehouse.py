"""DuckDB analytical layer over the canonical demand snapshot.

This is a *reporting* warehouse, not part of the ingest path: `canonicalize_demand()`
remains the single funnel every source passes through. Here the canonical frame is
loaded into a small star schema so the EDA and error-slice aggregations can be
expressed as SQL joins rather than pandas groupbys.

`dim_calendar` is materialised from the same helpers the model features use
(`german_holidays`, `_dst_transition_flags`), so a SQL slice and a model feature
can never disagree about what counts as a holiday.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .features import _dst_transition_flags, german_holidays

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
DEFAULT_DB_PATH = Path("data/clean/warehouse.duckdb")


def _connect(db_path: Path | str, read_only: bool = False):
    """Open a connection pinned to UTC.

    DuckDB renders TIMESTAMPTZ in the session timezone, which defaults to the host's.
    Without this pin the same query prints different offsets on a developer laptop and
    a UTC CI runner -- and the project's rule is that timestamps crossing a boundary
    are UTC, with local time used only for calendar attributes and display.
    """
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute("SET TimeZone = 'UTC'")
    return con


def calendar_frame(timestamps: pd.Series) -> pd.DataFrame:
    """Local-calendar attributes for each UTC timestamp, matching features.add_features."""
    stamps = pd.to_datetime(pd.Series(timestamps), utc=True).sort_values().reset_index(drop=True)
    local = stamps.dt.tz_convert("Europe/Berlin")
    holidays = german_holidays(set(local.dt.year.tolist()))
    is_holiday = local.dt.date.isin(holidays)
    is_weekend = local.dt.weekday >= 5
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "local_timestamp": local.dt.tz_localize(None),
            "hour": local.dt.hour.astype(int),
            "weekday": local.dt.weekday.astype(int),
            "weekday_name": local.dt.day_name(),
            "month": local.dt.month.astype(int),
            "year_month": local.dt.strftime("%Y-%m"),
            "is_weekend": is_weekend.astype(int),
            "is_public_holiday": is_holiday.astype(int),
            "is_dst_transition": _dst_transition_flags(stamps).astype(int),
            "day_type": pd.Series(
                ["Public holiday" if h else "Weekend" if w else "Weekday"
                 for h, w in zip(is_holiday, is_weekend)]
            ),
        }
    )


def build_warehouse(
    demand: pd.DataFrame, db_path: Path | str = DEFAULT_DB_PATH,
    forecasts: pd.DataFrame | None = None,
) -> Path:
    """Materialise fact_demand, dim_calendar and (optionally) fact_forecast.

    `forecasts` is the pooled per-hour backtest output: origin, timestamp, model,
    demand_mw, predicted.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fact_demand = demand.copy()
    fact_demand["timestamp"] = pd.to_datetime(fact_demand.timestamp, utc=True)
    fact_demand = fact_demand[["timestamp", "demand_mw"]].sort_values("timestamp")
    dim_calendar = calendar_frame(fact_demand.timestamp)

    with _connect(path) as con:
        con.register("_demand", fact_demand)
        con.register("_calendar", dim_calendar)
        con.execute("CREATE OR REPLACE TABLE fact_demand AS SELECT * FROM _demand")
        con.execute("CREATE OR REPLACE TABLE dim_calendar AS SELECT * FROM _calendar")
        if forecasts is not None:
            pooled = forecasts.copy()
            pooled["timestamp"] = pd.to_datetime(pooled.timestamp, utc=True)
            pooled["origin"] = pd.to_datetime(pooled.origin, utc=True)
            con.register("_forecast", pooled)
            con.execute("CREATE OR REPLACE TABLE fact_forecast AS SELECT * FROM _forecast")
        else:
            con.execute(
                "CREATE TABLE IF NOT EXISTS fact_forecast "
                "(origin TIMESTAMPTZ, timestamp TIMESTAMPTZ, model VARCHAR, "
                "demand_mw DOUBLE, predicted DOUBLE)"
            )
    return path


def query(
    sql: str, db_path: Path | str = DEFAULT_DB_PATH, params: list | None = None
) -> pd.DataFrame:
    """Run a query against the warehouse. `sql` is a statement or a name in sql/."""
    text = sql
    if not sql.strip().lower().startswith(("select", "with")):
        candidate = SQL_DIR / (sql if sql.endswith(".sql") else f"{sql}.sql")
        if not candidate.exists():
            raise FileNotFoundError(f"no SQL statement or file named {sql!r} ({candidate})")
        text = candidate.read_text(encoding="utf-8")
    with _connect(db_path, read_only=True) as con:
        return con.execute(text, params or []).df()
