import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecast import drift

# Winter plateau / summer plateau, the shape the real snapshot has (Jan 2855, Aug 1357).
# A plateau rather than a spike is the point: "every winter is equally bad" is what a
# season-matched reference has to forgive.
_SEASONAL_MAE = {
    12: 2800, 1: 2800, 2: 2800, 11: 2200, 3: 2200, 4: 1900,
    10: 1900, 5: 1600, 9: 1600, 6: 1400, 7: 1400, 8: 1400,
}


def _scores(days=730, end="2026-08-31", seed=0, model=drift.CHAMPION, params_hash="abc12345"):
    """Two years of per-origin champion scores with a seasonal MAE profile."""
    origins = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=days, freq="24h")
    noise = np.random.default_rng(seed).normal(0, 80, days)
    mae = np.array([_SEASONAL_MAE[stamp.month] for stamp in origins], dtype=float) + noise
    return pd.DataFrame(
        {
            "origin": origins, "model": model, "model_version": "v1",
            "params_hash": params_hash, "mae": mae, "rmse": mae * 1.2, "smape": mae / 500.0,
            "first_seen": origins[0],
        }
    )[drift.ORIGIN_COLUMNS]


def _window(scores, start, end):
    return scores[
        (scores.origin >= pd.Timestamp(start, tz="UTC"))
        & (scores.origin < pd.Timestamp(end, tz="UTC"))
    ].reset_index(drop=True)


def _reference_for(scores, window):
    return drift.seasonal_reference(
        scores, window.origin.iloc[len(window) // 2], exclude_from=window.origin.min()
    )


def _metrics(origins, mae, model=drift.CHAMPION):
    return pd.DataFrame(
        {"origin": origins, "model": model, "mae": mae, "rmse": mae * 1.2, "smape": mae / 500.0}
    )


def _run(run_at, snapshot_end, **overrides):
    record = dict.fromkeys(drift.RUN_COLUMNS, np.nan)
    record.update(
        {
            "run_at": pd.Timestamp(run_at, tz="UTC"),
            "snapshot_end": pd.Timestamp(snapshot_end, tz="UTC"),
            "params_hash": "abc12345", "champion": drift.CHAMPION, "champion_mae_mean": 1944.2,
        }
    )
    record.update(overrides)
    return record


def test_seasonal_reference_excludes_the_scored_window():
    scores = _scores()
    window = _window(scores, "2026-01-02", "2026-01-16")
    reference = _reference_for(scores, window)
    assert len(reference)
    assert not set(reference.origin).intersection(window.origin)


def test_seasonal_reference_matches_day_of_year_across_years():
    scores = _scores()
    window = _window(scores, "2026-01-02", "2026-01-16")
    months = set(_reference_for(scores, window).origin.dt.month)
    assert 1 in months  # the previous January
    assert 7 not in months  # not the previous July


def test_seasonal_reference_wraps_around_the_year_end():
    scores = _scores()
    window = _window(scores, "2025-12-20", "2026-01-03")
    months = set(_reference_for(scores, window).origin.dt.month)
    # Only reachable if day-of-year distance is measured around the year, not straight.
    assert {12, 1} <= months


def test_percentile_flags_a_step_change_in_error():
    scores = _scores()
    scores.loc[scores.index[-14:], "mae"] *= 2
    verdict = drift.regression_verdict(scores)
    assert verdict.percentile > 0.90


def test_seasonal_swing_alone_is_ok():
    # The test the whole design exists to pass: winter is genuinely worse than summer,
    # and that alone must not raise an alarm.
    scores = _scores()
    window = _window(scores, "2026-01-02", "2026-01-16")
    seasonal = drift.assess(window, _reference_for(scores, window))
    assert seasonal.level == "ok"
    # Against a pooled reference — the design we rejected — the same window looks alarming.
    pooled = scores[scores.origin < window.origin.min()]
    assert (pooled.mae < window.mae.mean()).mean() > 0.85


def test_single_crossing_is_watch_not_regressed():
    scores = _scores()
    scores.loc[scores.index[-14:], "mae"] *= 2
    window = drift.recent_window(scores)
    verdict = drift.assess(window, _reference_for(scores, window), consecutive_runs=0)
    assert verdict.percentile > 0.90
    assert verdict.level == "watch"


def test_two_consecutive_crossings_are_regressed():
    scores = _scores()
    scores.loc[scores.index[-14:], "mae"] *= 2
    window = drift.recent_window(scores)
    verdict = drift.assess(window, _reference_for(scores, window), consecutive_runs=1)
    assert verdict.level == "regressed"


def test_verdict_is_unknown_without_enough_reference_origins():
    scores = _scores(days=25)  # too little history for any comparable day-of-year window
    verdict = drift.regression_verdict(scores)
    assert verdict.reference_origins < drift.MIN_REFERENCE_ORIGINS
    assert verdict.level == "unknown"
    assert np.isnan(verdict.percentile)


def test_live_window_cannot_declare_regressed():
    scores = _scores()
    tail = scores.tail(7)
    daily = pd.DataFrame(
        {
            "origin": tail.origin.to_numpy(), "hours_scored": 24,
            "mae": tail.mae.to_numpy() * 2, "coverage": 0.93,
        }
    )
    verdict = drift.live_verdict(daily, scores)
    assert verdict.percentile > 0.90
    assert verdict.level == "watch"


def test_append_origin_history_is_idempotent(tmp_path):
    path = tmp_path / "origin_scores.csv"
    scores = _scores(days=40)
    rows = _metrics(scores.origin, scores.mae.to_numpy())
    drift.append_origin_history(path, rows, model_version="v1", params_hash="abc12345")
    before = path.read_bytes()
    _, changed = drift.append_origin_history(
        path, rows, model_version="v1", params_hash="abc12345",
        run_at=pd.Timestamp("2026-09-08", tz="UTC"),
    )
    assert changed is False
    assert path.read_bytes() == before


def test_append_origin_history_keeps_the_first_reading(tmp_path):
    path = tmp_path / "origin_scores.csv"
    scores = _scores(days=40)
    first = _metrics(scores.origin, scores.mae.to_numpy())
    drift.append_origin_history(path, first, model_version="v1", params_hash="abc12345")
    # A rerun trains on a front-truncated prefix, so it is a different model, not a
    # better reading of the same origin. The original stands.
    second = _metrics(scores.origin, scores.mae.to_numpy() * 3)
    merged, _ = drift.append_origin_history(
        path, second, model_version="v1", params_hash="abc12345"
    )
    assert len(merged) == len(first)
    assert merged.mae.max() < first.mae.max() * 1.01


def test_append_run_history_keeps_the_latest_run_per_snapshot(tmp_path):
    path = tmp_path / "benchmark_runs.csv"
    drift.append_run_history(path, _run("2026-09-01", "2026-08-31", champion_mae_mean=1944.2))
    # The two runs must differ in more than run_at: run_at is excluded from the change
    # comparison (the mirror of _log_payload dropping issued_at) so that a rerun with
    # identical results makes no git diff. Do not "fix" this by varying run_at alone.
    merged, changed = drift.append_run_history(
        path, _run("2026-09-08", "2026-08-31", champion_mae_mean=1900.5)
    )
    assert changed is True
    assert len(merged) == 1
    assert merged.run_at.iloc[0] == pd.Timestamp("2026-09-08", tz="UTC")
    assert merged.champion_mae_mean.iloc[0] == pytest.approx(1900.5)


def test_capped_run_is_not_complete():
    assert drift.is_complete_run(40, 40) is False
    assert drift.is_complete_run(150, None) is False  # uncapped, but a truncated snapshot
    assert drift.is_complete_run(329, None) is True


def test_coverage_drift_uses_the_established_baseline_not_nominal():
    runs = pd.DataFrame(
        [
            _run(f"2026-0{month}-01", f"2026-0{month}-01", coverage_mean_per_origin=0.932,
                 coverage_ci_lower=0.915, coverage_ci_upper=0.947, coverage_nominal=0.95)
            for month in (5, 6, 7, 8)
        ]
    )
    report = drift.coverage_drift(runs)
    assert report["level"] == "ok"  # below the 95% nominal, but exactly where it has sat
    assert report["established"] == pytest.approx(0.932)
    assert report["nominal"] == pytest.approx(0.95)


def test_load_history_returns_typed_empty_frame_when_missing(tmp_path):
    origins = drift.load_origin_history(tmp_path / "absent.csv")
    assert origins.empty
    assert list(origins.columns) == drift.ORIGIN_COLUMNS
    assert str(origins.origin.dtype) == "datetime64[ns, UTC]"
    runs = drift.load_run_history(tmp_path / "also_absent.csv")
    assert list(runs.columns) == drift.RUN_COLUMNS
    assert str(runs.run_at.dtype) == "datetime64[ns, UTC]"


def test_run_record_from_summary_reads_the_shipped_summary():
    path = Path(__file__).resolve().parents[1] / "reports" / "benchmark_summary.json"
    if not path.exists():  # artifacts not built in this checkout
        pytest.skip("reports/benchmark_summary.json is not present")
    record = drift.run_record_from_summary(json.loads(path.read_text()))
    assert set(drift.RUN_COLUMNS) <= set(record)
    assert np.isfinite(record["champion_mae_mean"])
