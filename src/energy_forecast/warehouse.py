"""PostgreSQL analytical layer over the canonical demand snapshot.

This is a *reporting* warehouse, not part of the ingest path: `canonicalize_demand()`
remains the single funnel every source passes through. Here the canonical frame is
loaded into a small star schema so the EDA and error-slice aggregations can be
expressed as SQL joins rather than pandas groupbys.

`dim_calendar` is materialised from the same helpers the model features use
(`german_holidays`, `_dst_transition_flags`), so a SQL slice and a model feature
can never disagree about what counts as a holiday.

The connection comes from `DATABASE_URL` (default: the `postgres` service in
docker-compose.yml). Every table is rebuilt from the snapshot, so the database holds
nothing that cannot be regenerated.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from .features import _dst_transition_flags, german_holidays

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
DEFAULT_URL = "postgresql+psycopg://stromblick:stromblick@localhost:5432/stromblick"
_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class Warehouse:
    """Where the warehouse lives: a database URL and the schema holding its tables."""

    url: str
    schema: str = "public"

    def __post_init__(self) -> None:
        if not _SCHEMA_NAME.match(self.schema):
            raise ValueError(f"invalid schema name {self.schema!r}")


def default_warehouse() -> Warehouse:
    return Warehouse(os.environ.get("DATABASE_URL", DEFAULT_URL))


def _engine(target: Warehouse):
    """An engine pinned to UTC and to the target schema.

    Postgres renders TIMESTAMPTZ in the session timezone, which defaults to the server's.
    Without this pin the same query could print different offsets on a developer laptop
    and a UTC CI runner, and the project's rule is that timestamps crossing a boundary
    are UTC, with local time used only for calendar attributes and display.
    """
    options = f"-c timezone=UTC -c search_path={target.schema}"
    return create_engine(target.url, connect_args={"options": options})


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


def _replace(frame: pd.DataFrame, name: str, con, schema: str) -> None:
    frame.to_sql(
        name, con, schema=schema, if_exists="replace", index=False,
        method="multi", chunksize=2000,
    )


def build_warehouse(
    demand: pd.DataFrame, target: Warehouse | None = None,
    forecasts: pd.DataFrame | None = None,
) -> Warehouse:
    """Materialise fact_demand, dim_calendar and (optionally) fact_forecast.

    `forecasts` is the pooled per-hour backtest output: origin, timestamp, model,
    demand_mw, predicted. Everything runs in one transaction, so a failed build leaves
    the previous tables in place.
    """
    target = target or default_warehouse()
    fact_demand = demand.copy()
    fact_demand["timestamp"] = pd.to_datetime(fact_demand.timestamp, utc=True)
    fact_demand = fact_demand[["timestamp", "demand_mw"]].sort_values("timestamp")
    dim_calendar = calendar_frame(fact_demand.timestamp)

    engine = _engine(target)
    try:
        with engine.begin() as con:
            con.execute(text(f"CREATE SCHEMA IF NOT EXISTS {target.schema}"))
            _replace(fact_demand, "fact_demand", con, target.schema)
            _replace(dim_calendar, "dim_calendar", con, target.schema)
            if forecasts is not None:
                pooled = forecasts.copy()
                pooled["timestamp"] = pd.to_datetime(pooled.timestamp, utc=True)
                pooled["origin"] = pd.to_datetime(pooled.origin, utc=True)
                _replace(pooled, "fact_forecast", con, target.schema)
            else:
                con.execute(text(
                    "CREATE TABLE IF NOT EXISTS fact_forecast "
                    "(origin TIMESTAMPTZ, timestamp TIMESTAMPTZ, model TEXT, "
                    "demand_mw DOUBLE PRECISION, predicted DOUBLE PRECISION)"
                ))
    finally:
        engine.dispose()
    return target


def query(
    sql: str, target: Warehouse | None = None, params: dict | None = None
) -> pd.DataFrame:
    """Run a query against the warehouse. `sql` is a statement or a name in sql/.

    Parameters are named (`:model` in the SQL, `{"model": ...}` here).
    """
    text_sql = sql
    if not sql.strip().lower().startswith(("select", "with")):
        candidate = SQL_DIR / (sql if sql.endswith(".sql") else f"{sql}.sql")
        if not candidate.exists():
            raise FileNotFoundError(f"no SQL statement or file named {sql!r} ({candidate})")
        text_sql = candidate.read_text(encoding="utf-8")
    engine = _engine(target or default_warehouse())
    try:
        with engine.connect() as con:
            return pd.read_sql(text(text_sql), con, params=params or {})
    finally:
        engine.dispose()
