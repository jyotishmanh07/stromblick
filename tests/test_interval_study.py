import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from interval_study import (  # noqa: E402
    CAL_ORIGINS,
    SPLIT,
    build_bands,
    conformal_quantile,
    interval_score,
)


def test_conformal_quantile_uses_the_finite_sample_rank():
    scores = np.arange(1, 20, dtype=float)  # n = 19, ceil(20 * 0.95) = 19 -> the largest
    assert conformal_quantile(scores, 0.95) == 19.0
    assert conformal_quantile(scores, 0.5) == 10.0  # ceil(20 * 0.5) = 10


def test_conformal_quantile_is_infinite_when_the_sample_is_too_small():
    assert conformal_quantile([1.0, 2.0], 0.95) == float("inf")


def test_interval_score_is_the_width_when_the_actual_is_inside():
    assert interval_score([0.0], [10.0], [5.0])[0] == pytest.approx(10.0)


def test_interval_score_charges_two_over_alpha_per_unit_outside():
    assert interval_score([0.0], [10.0], [12.0], alpha=0.05)[0] == pytest.approx(10 + 40 * 2)


def _hourly(origins: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    starts = pd.date_range("2025-01-01", periods=origins, freq="24h", tz="UTC")
    rows = []
    for origin in starts:
        actual = 50_000 + rng.normal(0, 1_000, 24)
        rows.append(pd.DataFrame({
            "origin": origin, "horizon": np.arange(1, 25),
            "timestamp": pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="1h"),
            "actual": actual, "point": np.full(24, 50_000.0),
            "q_lower": np.full(24, 48_000.0), "q_upper": np.full(24, 52_000.0),
            **{f"shipped_{lv}": 1_500.0 for lv in (0.5, 0.8, 0.9, 0.95)},
        }))
    return pd.concat(rows, ignore_index=True)


def test_bands_start_after_the_warm_up_origins():
    bands = build_bands(_hourly(CAL_ORIGINS + 3))
    assert len(bands[SPLIT]) == 3 * 24


def test_a_band_never_depends_on_the_hours_it_scores():
    hourly = _hourly(CAL_ORIGINS + 2)
    last = hourly.origin.max()
    before = build_bands(hourly)[SPLIT].loc[hourly.origin == last]
    hourly.loc[hourly.origin == last, "actual"] += 1e6  # a wild outcome for the scored day
    after = build_bands(hourly)[SPLIT].loc[hourly.origin == last]
    pd.testing.assert_frame_equal(before, after)


def test_forecast_with_rows_matches_forecast_and_returns_one_row_per_step():
    from energy_forecast.models import HistGradientBoostingForecast
    from energy_forecast.service import demo_demand

    data = demo_demand(hours=24 * 30)
    model = HistGradientBoostingForecast().fit(data)
    target = pd.date_range(data.timestamp.max() + pd.Timedelta(hours=1), periods=6, freq="1h",
                           tz="UTC")
    predictions, rows = model.forecast_with_rows(target)
    np.testing.assert_allclose(predictions, model.forecast(target))
    assert len(rows) == 6
    assert rows.lag_1h.iloc[1] == pytest.approx(predictions[0])  # lags read the point path
