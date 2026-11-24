import numpy as np
import pandas as pd

from energy_forecast.calibration import (
    CALIBRATION_ORIGINS,
    build_pool,
    load_pool,
    pool_spread,
    write_pool,
)
from energy_forecast.inference import conformal_quantile
from energy_forecast.models import MODEL_VERSION
from energy_forecast.service import demo_demand

END = pd.Timestamp("2026-06-30 12:00", tz="UTC")


def _pool(origins: int, end: pd.Timestamp = END, version: str = MODEL_VERSION) -> pd.DataFrame:
    """`origins` daily origins ending at `end`; origin k's error is k, so picks are visible."""
    rows = []
    for k in range(origins, 0, -1):
        origin = end - pd.Timedelta(hours=24 * k)
        stamps = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="1h")
        rows.append(pd.DataFrame({
            "origin": origin, "timestamp": stamps, "demand_mw": 50_000.0,
            "prediction": 50_000.0, "abs_error": float(k), "model_version": version,
        }))
    return pd.concat(rows, ignore_index=True)


def test_spread_is_the_conformal_quantile_of_the_newest_calibration_origins():
    pool = _pool(35)
    newest = pool[pool.origin >= END - pd.Timedelta(hours=24 * CALIBRATION_ORIGINS)]
    assert pool_spread(pool, END) == conformal_quantile(newest.abs_error, 0.95)


def test_spread_never_uses_an_origin_whose_hours_end_after_the_cutoff():
    pool = _pool(35)
    cutoff = END - pd.Timedelta(hours=24 * 3)  # the 3 newest origins end after this
    allowed = pool[pool.origin + pd.Timedelta(hours=24) <= cutoff]
    expected = allowed[allowed.origin.isin(sorted(allowed.origin.unique())[-28:])]
    assert pool_spread(pool, cutoff) == conformal_quantile(expected.abs_error, 0.95)


def test_too_few_origins_means_fall_back():
    assert pool_spread(_pool(CALIBRATION_ORIGINS - 1), END) is None


def test_a_stale_pool_means_fall_back():
    assert pool_spread(_pool(35), END + pd.Timedelta(days=20)) is None


def test_another_model_generations_pool_is_ignored():
    assert pool_spread(_pool(35, version="v0"), END) is None


def test_build_pool_scores_whole_horizons_observed_by_the_end():
    history = demo_demand(24 * 40)
    end = history.timestamp.max()
    pool = build_pool(history, end, origins=3)
    assert pool.origin.nunique() == 3
    assert (pool.origin + pd.Timedelta(hours=24) <= end).all()
    assert not pool.abs_error.isna().any()
    assert set(pool.model_version) == {MODEL_VERSION}


def test_write_pool_is_idempotent_and_round_trips(tmp_path):
    path = tmp_path / "pool.csv"
    pool = _pool(30)
    assert write_pool(pool, path) is True
    assert write_pool(pool, path) is False
    assert pool_spread(load_pool(path), END) == pool_spread(pool, END)


def test_load_pool_returns_empty_for_a_missing_or_broken_file(tmp_path):
    assert load_pool(tmp_path / "missing.csv").empty
    (tmp_path / "bad.csv").write_text("not,a,pool\n1,2,3\n", encoding="utf-8")
    assert load_pool(tmp_path / "bad.csv").empty


def test_hindcast_band_comes_from_origins_before_the_replayed_week():
    from energy_forecast.verification import hindcast

    history = demo_demand(24 * 70)
    end = history.timestamp.max()
    pool = _pool(35, end=end)
    replay = hindcast(history, days=7, end=end, pool=pool)
    expected = pool_spread(pool, end - pd.Timedelta(hours=24 * 7))
    assert expected is not None
    np.testing.assert_allclose(replay.upper_bound - replay.prediction, expected)
    # The newest seven origins overlap the replayed week, so they must not size its band.
    assert expected != pool_spread(pool, end)
