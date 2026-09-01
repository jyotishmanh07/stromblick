"""Chronological rolling-origin evaluation and error slices."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .features import german_holidays


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    ratio = np.where(denominator == 0, 0, 200 * np.abs(actual - predicted) / denominator)
    return float(np.mean(ratio))


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape": smape(actual, predicted),
    }


def rolling_origin_backtest(
    frame: pd.DataFrame, model_factories: dict[str, Callable[[], object]],
    initial_train_hours: int = 24 * 28, horizon_hours: int = 24, step_hours: int = 24,
) -> pd.DataFrame:
    """Evaluate future windows chronologically; no random split is used."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    records: list[dict[str, object]] = []
    for origin in range(initial_train_hours, len(data) - horizon_hours + 1, step_hours):
        train, test = data.iloc[:origin], data.iloc[origin : origin + horizon_hours]
        for name, factory in model_factories.items():
            model = factory().fit(train)
            predicted = model.predict(pd.DatetimeIndex(test.timestamp))
            row = {"origin": train.timestamp.iloc[-1], "model": name}
            row.update(metrics(test.demand_mw.to_numpy(), predicted))
            records.append(row)
    return pd.DataFrame(records)


def error_slices(actual_frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, pd.DataFrame]:
    data = actual_frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data["absolute_error"] = np.abs(data.demand_mw.to_numpy() - predicted)
    local = data.timestamp.dt.tz_convert("Europe/Berlin")
    data["hour"] = local.dt.hour
    data["weekday"] = local.dt.day_name()
    data["month"] = local.dt.month
    holidays = german_holidays(set(local.dt.year.unique()))
    data["is_public_holiday"] = local.dt.date.isin(holidays).astype(int)
    return {
        "hour": data.groupby("hour", as_index=False).absolute_error.mean(),
        "weekday": data.groupby("weekday", as_index=False).absolute_error.mean(),
        "month": data.groupby("month", as_index=False).absolute_error.mean(),
        "holiday": data.groupby("is_public_holiday", as_index=False).absolute_error.mean(),
    }
