import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from window_study import evaluation_origins, holm, window_slice  # noqa: E402


def _hourly(days: int) -> pd.DataFrame:
    stamps = pd.date_range("2024-01-01", periods=24 * days, freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp": stamps, "demand_mw": 50_000.0})


def test_window_slice_never_includes_hours_after_the_origin():
    data = _hourly(30)
    origin = data.timestamp.iloc[400]
    assert window_slice(data, origin, None).timestamp.max() == origin


def test_window_slice_keeps_exactly_the_trailing_days():
    data = _hourly(30)
    origin = data.timestamp.iloc[-1]
    sliced = window_slice(data, origin, 7)
    assert len(sliced) == 24 * 7
    assert sliced.timestamp.min() > origin - pd.Timedelta(days=7)


def test_window_slice_is_time_based_when_rows_are_missing():
    data = _hourly(10).drop(index=range(100, 148))  # a two-day hole
    origin = data.timestamp.iloc[-1]
    assert len(window_slice(data, origin, 9)) == 24 * 9 - 48


def test_evaluation_origins_leave_a_full_horizon_after_the_last_one():
    data = _hourly(30)
    origins = evaluation_origins(data, eval_days=7, horizon_hours=24)
    assert origins[-1] + pd.Timedelta(hours=24) == data.timestamp.max()
    assert len(origins) == 8  # 7 days of daily steps, both ends included


def test_holm_scales_the_smallest_p_by_the_number_of_tests_and_stays_monotone():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)  # 1 x 0.04 lifted to the running max
