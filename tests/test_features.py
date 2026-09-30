import sys

import numpy as np
import pandas as pd
import pytest

from energy_forecast.features import FEATURE_COLUMNS, add_features, feature_row


def test_lags_and_rolling_features_do_not_use_current_target():
    timestamps = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    data = pd.DataFrame({"timestamp": timestamps, "demand_mw": np.arange(200, dtype=float)})
    features = add_features(data)
    row = features.iloc[168]
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert row.lag_1h == 167
    assert row.lag_24h == 144
    assert row.rolling_24h_mean == np.mean(np.arange(144, 168))
    assert row.rolling_7d_mean == np.mean(np.arange(0, 168))


def _dst_window() -> pd.DataFrame:
    # Spans the 2024-10-27 fall-back change, so calendar features cross a UTC offset shift.
    timestamps = pd.date_range("2024-10-10", "2024-11-10", freq="h", tz="UTC")
    demand = 50_000 + np.random.default_rng(0).normal(0, 3_000, len(timestamps))
    return pd.DataFrame({"timestamp": timestamps, "demand_mw": demand})


def _mismatches(columns: list[str]) -> list[tuple]:
    frame = _dst_window()
    trained = add_features(frame).set_index("timestamp")
    series = frame.set_index("timestamp").demand_mw
    found = []
    for timestamp in frame.timestamp.iloc[200:]:
        served = feature_row(series[series.index < timestamp], timestamp)
        found += [
            (timestamp, column, trained.at[timestamp, column], served[column])
            for column in columns
            if not np.isclose(trained.at[timestamp, column], served[column])
        ]
    return found


def test_feature_row_matches_add_features_for_lags_rolling_and_calendar():
    # The two builders must stay in lockstep: training reads add_features, recursive
    # forecasting reads feature_row, and any disagreement is train/serve skew.
    columns = [column for column in FEATURE_COLUMNS if column != "is_dst_transition"]
    assert _mismatches(columns) == []


@pytest.mark.xfail(
    strict=True,
    reason="Known skew: add_features also looks one row ahead (shift(-1)) and always flags "
    "a frame's final row, while feature_row compares the last two history hours, so the "
    "hours either side of a clock change disagree. Fixing it changes the model.",
)
def test_feature_row_matches_add_features_for_dst_transition():
    assert _mismatches(["is_dst_transition"]) == []


def test_feature_row_treats_a_naive_timestamp_as_utc():
    frame = _dst_window()
    series = frame.set_index("timestamp").demand_mw
    aware = frame.timestamp.iloc[300]
    history = series[series.index < aware]
    assert feature_row(history, aware.tz_localize(None)) == feature_row(history, aware)


def test_add_features_rejects_a_frame_without_demand():
    with pytest.raises(ValueError):
        add_features(pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3)}))


def test_missing_holidays_package_silently_disables_the_holiday_flag(monkeypatch):
    # Documented degrade: without `holidays`, is_public_holiday is all zero, not an error.
    monkeypatch.setitem(sys.modules, "holidays", None)
    timestamps = pd.date_range("2024-12-24", "2024-12-27", freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "demand_mw": 50_000.0})
    assert add_features(frame).is_public_holiday.sum() == 0
