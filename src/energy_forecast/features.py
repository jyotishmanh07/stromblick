"""Leakage-safe calendar, lag, and rolling-window features."""

from __future__ import annotations

from datetime import date

import pandas as pd

FEATURE_COLUMNS = [
    "hour", "weekday", "month", "is_weekend", "is_public_holiday", "is_dst_transition",
    "lag_1h", "lag_24h", "lag_168h", "rolling_24h_mean", "rolling_7d_mean",
]


def german_holidays(years: set[int]) -> set[date]:
    try:
        import holidays

        return set(holidays.Germany(years=years).keys())
    except ImportError:
        return set()


def _dst_transition_flags(timestamps: pd.Series) -> pd.Series:
    localized = pd.to_datetime(timestamps, utc=True).dt.tz_convert("Europe/Berlin")
    offsets = localized.map(
        lambda value: value.utcoffset().total_seconds() if value.utcoffset() else 0
    )
    return offsets.ne(offsets.shift(1)) | offsets.ne(offsets.shift(-1))


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create features using only observations at or before each row's timestamp."""
    required = {"timestamp", "demand_mw"}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame must contain {sorted(required)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values("timestamp").reset_index(drop=True)
    local = result["timestamp"].dt.tz_convert("Europe/Berlin")
    holidays = german_holidays(set(local.dt.year.tolist()))
    result["hour"] = local.dt.hour.astype(int)
    result["weekday"] = local.dt.weekday.astype(int)
    result["month"] = local.dt.month.astype(int)
    result["is_weekend"] = (local.dt.weekday >= 5).astype(int)
    result["is_public_holiday"] = local.dt.date.isin(holidays).astype(int)
    result["is_dst_transition"] = _dst_transition_flags(result["timestamp"]).astype(int)
    series = result.set_index("timestamp")["demand_mw"]
    result = result.set_index("timestamp")
    result["lag_1h"] = series.shift(1)
    result["lag_24h"] = series.shift(24)
    result["lag_168h"] = series.shift(168)
    prior = series.shift(1)
    result["rolling_24h_mean"] = prior.rolling(24, min_periods=24).mean()
    result["rolling_7d_mean"] = prior.rolling(168, min_periods=168).mean()
    return result.reset_index()


def feature_row(history: pd.Series, timestamp: pd.Timestamp) -> dict[str, float]:
    """Build one future row from timestamp-indexed observed/predicted history."""
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    timestamp = timestamp.tz_convert("UTC")
    local = timestamp.tz_convert("Europe/Berlin")
    holiday_set = german_holidays({local.year})

    def value_at(delta: pd.Timedelta) -> float:
        return float(history.get(timestamp - delta, float("nan")))

    prior_values = history[history.index < timestamp].tail(168)
    recent_offsets = [
        pd.Timestamp(ts).tz_convert("Europe/Berlin").utcoffset() for ts in history.index[-2:]
    ]
    dst_transition = int(
        bool(recent_offsets) and any(offset != local.utcoffset() for offset in recent_offsets)
    )
    has_24h, has_7d = len(prior_values) >= 24, len(prior_values) >= 168
    rolling_24h_mean = float(prior_values.tail(24).mean()) if has_24h else float("nan")
    rolling_7d_mean = float(prior_values.mean()) if has_7d else float("nan")
    return {
        "hour": float(local.hour),
        "weekday": float(local.weekday()),
        "month": float(local.month),
        "is_weekend": float(local.weekday() >= 5),
        "is_public_holiday": float(local.date() in holiday_set),
        "is_dst_transition": float(dst_transition),
        "lag_1h": value_at(pd.Timedelta(hours=1)),
        "lag_24h": value_at(pd.Timedelta(hours=24)),
        "lag_168h": value_at(pd.Timedelta(hours=168)),
        "rolling_24h_mean": rolling_24h_mean,
        "rolling_7d_mean": rolling_7d_mean,
    }
