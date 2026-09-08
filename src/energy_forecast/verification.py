"""Forecast verification: how did the forecast we published actually do?

`ForecastService.forecast` answers "what happens next". This module answers the
retrospective question, which needs a different mechanism: `forecast_with_interval`
refits on the *full* history regardless of `as_of`, and the GBM reads observed lags
straight out of that history, so pointing the service at a past `as_of` leaks the
future into the answer. `replay_origins` instead refits a fresh model per origin on
rows dated at or before it.

It lives as a module of pure functions rather than as `ForecastService` methods for
the same reason `anomaly.py` and `inference.py` do: a dashboard tab, a script, and a
test can all call these without constructing a service.

`walk_forward` in `scripts/benchmark.py` is the positional-origin twin of
`replay_origins` — same refit-per-origin shape, same NaN-actual masking rule.
Consolidating the two is deliberate future work, not part of this change.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from .evaluation import metrics
from .live import last_observed
from .models import HistGradientBoostingForecast

if TYPE_CHECKING:  # service.py imports `hindcast` from here, so this would be a cycle.
    from .service import ForecastService

HINDCAST_COLUMNS = ["origin", "timestamp", "demand_mw", "prediction", "lower_bound", "upper_bound"]
LOG_COLUMNS = [
    "issued_at", "origin", "timestamp", "prediction", "lower_bound", "upper_bound", "model_version",
]
DEFAULT_LOG_PATH = Path("data/forecasts/forecast_log.csv")
VALIDATION_HOURS = 24 * 7  # residual window, same as the service's
MIN_FIT_ROWS = 368  # 168h of lags + 200 complete rows, per HistGradientBoostingForecast.fit


def replay_origins(
    history: pd.DataFrame, origins: list[pd.Timestamp],
    model_factory: Callable[[], object] = HistGradientBoostingForecast,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Refit a fresh model at each origin and forecast the following `horizon_hours`."""
    data = history.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    actuals = data[["timestamp", "demand_mw"]]
    parts: list[pd.DataFrame] = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        # Time-based, never .iloc: interior NaN rows and the live merge make row
        # positions and hours diverge. NaN rows stay in — model.fit drops them itself.
        train = data[data.timestamp <= origin]
        target = pd.date_range(
            origin + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h", tz="UTC"
        )
        predicted = model_factory().fit(train).predict(target)
        window = pd.DataFrame({"origin": origin, "timestamp": target})
        # Left merge: hours SMARD has not published yet arrive as NaN and stay that way.
        window = window.merge(actuals, on="timestamp", how="left")
        window["prediction"] = np.asarray(predicted, dtype=float)
        parts.append(window[["origin", "timestamp", "demand_mw", "prediction"]])
    if not parts:
        return pd.DataFrame(columns=["origin", "timestamp", "demand_mw", "prediction"])
    return pd.concat(parts, ignore_index=True)


def hindcast(
    history: pd.DataFrame, days: int = 7, horizon_hours: int = 24,
    end: pd.Timestamp | None = None,
    model_factory: Callable[[], object] = HistGradientBoostingForecast,
    residual_quantile: float = 0.95,
) -> pd.DataFrame:
    """Replay the last `days` daily origins, refitting strictly before each one."""
    empty = pd.DataFrame(columns=HINDCAST_COLUMNS)
    end = end if end is not None else last_observed(history)
    if end is None:
        return empty

    end = pd.Timestamp(end)
    origins = [end - pd.Timedelta(hours=k * horizon_hours) for k in range(days, 0, -1)]

    data = history.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Count rows up to `end`, not the whole frame: every fit below is anchored to the
    # origins, so rows after `end` cannot feed one. Too little to fit is a silent
    # degrade, like ForecastService.detect_recent_anomalies, never an exception.
    if (data.timestamp <= end).sum() < days * horizon_hours + VALIDATION_HOURS + MIN_FIT_ROWS:
        return empty

    # A fresh fit, not `service.residuals`: those come from the trailing 168h, which is
    # exactly the window replayed below, so reusing them would leak the scored hours
    # into the band that judges them.
    band_end = origins[0] - pd.Timedelta(hours=VALIDATION_HOURS)
    validation = data[(data.timestamp > band_end) & (data.timestamp <= origins[0])]
    residual_model = model_factory().fit(data[data.timestamp <= band_end])
    residuals = validation.demand_mw.to_numpy(dtype=float) - residual_model.predict(
        pd.DatetimeIndex(validation.timestamp)
    )
    residuals = residuals[~np.isnan(residuals)]  # hours with no observed demand
    if len(residuals):
        spread = float(np.quantile(np.abs(residuals), residual_quantile))
    else:
        spread = float(np.std(data.demand_mw.dropna()) * 0.15)  # forecast_with_interval's fallback

    frame = replay_origins(data, origins, model_factory, horizon_hours)
    frame["lower_bound"] = frame.prediction - spread
    frame["upper_bound"] = frame.prediction + spread
    return frame[HINDCAST_COLUMNS]


def daily_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-origin MAE and interval coverage over the hours that have been published."""
    columns = ["origin", "hours_scored", "mae", "coverage"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for origin, window in frame.groupby("origin", sort=True):
        actual = window.demand_mw.to_numpy(dtype=float)
        observed = ~np.isnan(actual)
        if not observed.any():
            records.append(
                {"origin": origin, "hours_scored": 0, "mae": float("nan"),
                 "coverage": float("nan")}
            )
            continue
        predicted = window.prediction.to_numpy(dtype=float)[observed]
        inside = (
            (window.lower_bound.to_numpy(dtype=float)[observed] <= actual[observed])
            & (actual[observed] <= window.upper_bound.to_numpy(dtype=float)[observed])
        )
        records.append(
            {
                "origin": origin,
                "hours_scored": int(observed.sum()),
                "mae": metrics(actual[observed], predicted)["mae"],
                "coverage": float(np.mean(inside)),
            }
        )
    return pd.DataFrame(records, columns=columns).sort_values("origin").reset_index(drop=True)


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issued_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "origin": pd.Series(dtype="datetime64[ns, UTC]"),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "prediction": pd.Series(dtype=float),
            "lower_bound": pd.Series(dtype=float),
            "upper_bound": pd.Series(dtype=float),
            "model_version": pd.Series(dtype=object),
        }
    )[LOG_COLUMNS]


def load_forecast_log(path: str | Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    """Read the published-forecast log, or an empty typed frame when there is none."""
    path = Path(path)
    if not path.exists():
        return _empty_log()
    frame = pd.read_csv(path)
    if frame.empty:
        return _empty_log()
    for column in ("issued_at", "origin", "timestamp"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame[LOG_COLUMNS]


def append_forecast_log(
    path: str | Path, rows: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    """Merge `rows` into the log at `path`, keeping the latest issue per (origin, hour).

    Returns the merged frame and whether the file was written. A re-run that adds no
    new information leaves the file untouched byte-for-byte, so a scheduled workflow
    running twice in a day produces no commit.
    """
    path = Path(path)
    existing = load_forecast_log(path)
    merged = pd.concat([existing, rows], ignore_index=True)
    merged["issued_at"] = pd.to_datetime(merged.issued_at, utc=True)
    merged["origin"] = pd.to_datetime(merged.origin, utc=True)
    merged["timestamp"] = pd.to_datetime(merged.timestamp, utc=True)
    merged = (
        merged.sort_values(["origin", "timestamp", "issued_at"])
        .drop_duplicates(["origin", "timestamp"], keep="last")
        .sort_values(["origin", "timestamp"])
        .reset_index(drop=True)[LOG_COLUMNS]
    )
    if _log_payload(merged).equals(_log_payload(existing)):
        return existing, False
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged, True


def _log_payload(frame: pd.DataFrame) -> pd.DataFrame:
    """Everything but `issued_at`, rounded to the 3 dp that survive the CSV round-trip.

    `ForecastService.forecast` already rounds there, so this compares like with like:
    a re-issued but identical forecast is not a change.
    """
    payload = frame.drop(columns=["issued_at"]).reset_index(drop=True)
    for column in ("prediction", "lower_bound", "upper_bound"):
        payload[column] = payload[column].astype(float).round(3)
    return payload


def join_log_to_actuals(log: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach observed demand to logged forecasts; unpublished hours stay NaN."""
    actuals = history[["timestamp", "demand_mw"]].copy()
    actuals["timestamp"] = pd.to_datetime(actuals.timestamp, utc=True)
    joined = log.copy()
    joined["timestamp"] = pd.to_datetime(joined.timestamp, utc=True)
    return joined.merge(actuals, on="timestamp", how="left")


def forecast_record(
    service: "ForecastService", issued_at: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Take the forecast the service would publish now and shape it for the log."""
    origin = service.last_observed
    if origin is None:
        raise ValueError("no published hour to forecast from")
    issued_at = (
        pd.Timestamp(issued_at) if issued_at is not None
        else pd.Timestamp.now(tz="UTC").floor("s")
    )
    response = service.forecast(origin, 24)
    rows = pd.DataFrame(response["forecast"])
    rows["timestamp"] = pd.to_datetime(rows.timestamp, utc=True)
    rows["issued_at"] = issued_at
    rows["origin"] = origin
    rows["model_version"] = response["model_version"]
    return rows[LOG_COLUMNS]
