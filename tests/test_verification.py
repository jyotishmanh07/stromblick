import numpy as np
import pandas as pd

from energy_forecast.live import last_observed
from energy_forecast.service import demo_demand
from energy_forecast.verification import (
    HINDCAST_COLUMNS,
    LOG_COLUMNS,
    append_forecast_log,
    daily_scores,
    hindcast,
    join_log_to_actuals,
    load_forecast_log,
)


class RecordingModel:
    """Records what it was fitted on and asked to predict; predicts zeros."""

    fits: list[pd.Timestamp] = []
    predictions: list[pd.DatetimeIndex] = []

    def fit(self, frame):
        RecordingModel.fits.append(pd.to_datetime(frame.timestamp, utc=True).max())
        return self

    def predict(self, index):
        RecordingModel.predictions.append(pd.DatetimeIndex(index))
        return np.zeros(len(index))


def test_hindcast_fits_only_on_rows_up_to_each_origin():
    RecordingModel.fits, RecordingModel.predictions = [], []
    history = demo_demand(24 * 40)
    frame = hindcast(history, days=7, model_factory=RecordingModel)
    origins = list(frame.origin.unique())
    # The residual-band model is fitted first; the last `days` fits are the replay.
    assert RecordingModel.fits[-7:] == [pd.Timestamp(origin) for origin in origins]
    for origin, index in zip(origins, RecordingModel.predictions[-7:]):
        expected = pd.date_range(
            pd.Timestamp(origin) + pd.Timedelta(hours=1), periods=24, freq="1h", tz="UTC"
        )
        assert list(index) == list(expected)


def test_hindcast_prediction_ignores_values_after_the_origin():
    history = demo_demand(24 * 40)
    baseline = hindcast(history, days=7)
    first_origin = baseline.origin.min()
    tampered = history.copy()
    tampered.loc[tampered.timestamp > first_origin, "demand_mw"] += 20_000
    shifted = hindcast(tampered, days=7)
    assert np.allclose(
        baseline[baseline.origin == first_origin].prediction.to_numpy(),
        shifted[shifted.origin == first_origin].prediction.to_numpy(),
    )


def test_hindcast_frame_shape_and_columns():
    history = demo_demand(24 * 40)
    frame = hindcast(history, days=7)
    assert len(frame) == 7 * 24
    assert list(frame.columns) == HINDCAST_COLUMNS
    origins = pd.DatetimeIndex(frame.origin.unique())
    assert (origins.to_series().diff().dropna() == pd.Timedelta(hours=24)).all()
    assert frame.timestamp.max() == last_observed(history)
    assert (frame.lower_bound <= frame.prediction).all()
    assert (frame.prediction <= frame.upper_bound).all()


def test_hindcast_keeps_unpublished_hours_unscored():
    history = demo_demand(24 * 40)
    # Inside the final window but not at its end, so `last_observed` still anchors it.
    history.loc[history.index[-10:-5], "demand_mw"] = np.nan
    frame = hindcast(history, days=7)
    last_origin = frame.origin.max()
    window = frame[frame.origin == last_origin]
    assert window.demand_mw.isna().sum() == 5
    scores = daily_scores(frame)
    row = scores[scores.origin == last_origin].iloc[0]
    assert row.hours_scored == 19
    assert np.isfinite(row.mae)


def test_hindcast_too_short_history_returns_empty_frame():
    frame = hindcast(demo_demand(24 * 15), days=7)
    assert frame.empty
    assert list(frame.columns) == HINDCAST_COLUMNS


def test_hindcast_counts_only_history_before_the_requested_end():
    # Plenty of rows overall, but almost none before `end` — the rows after it can
    # feed no fit, so this must degrade rather than raise out of model.fit.
    history = demo_demand(24 * 40)
    frame = hindcast(history, days=7, end=history.timestamp.iloc[400])
    assert frame.empty


def _log_rows(origin: pd.Timestamp, issued_at: pd.Timestamp, prediction: float) -> pd.DataFrame:
    timestamps = pd.date_range(origin + pd.Timedelta(hours=1), periods=3, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "issued_at": issued_at, "origin": origin, "timestamp": timestamps,
            "prediction": prediction, "lower_bound": prediction - 1_000.0,
            "upper_bound": prediction + 1_000.0, "model_version": "v1",
        }
    )[LOG_COLUMNS]


def test_append_forecast_log_keeps_latest_issue_per_origin_hour(tmp_path):
    path = tmp_path / "forecast_log.csv"
    origin = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    append_forecast_log(path, _log_rows(origin, pd.Timestamp("2026-01-01 01:00", tz="UTC"), 50_000))
    merged, changed = append_forecast_log(
        path, _log_rows(origin, pd.Timestamp("2026-01-01 02:00", tz="UTC"), 52_000)
    )
    assert changed is True
    assert len(merged) == 3
    assert (merged.prediction == 52_000).all()
    assert (merged.issued_at == pd.Timestamp("2026-01-01 02:00", tz="UTC")).all()


def test_append_forecast_log_is_idempotent(tmp_path):
    path = tmp_path / "forecast_log.csv"
    origin = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    append_forecast_log(path, _log_rows(origin, pd.Timestamp("2026-01-01 01:00", tz="UTC"), 50_000))
    before = path.read_bytes()
    _, changed = append_forecast_log(
        path, _log_rows(origin, pd.Timestamp("2026-01-02 09:00", tz="UTC"), 50_000)
    )
    assert changed is False
    assert path.read_bytes() == before


def test_load_forecast_log_missing_file_is_empty_and_typed(tmp_path):
    log = load_forecast_log(tmp_path / "absent.csv")
    assert log.empty
    assert list(log.columns) == LOG_COLUMNS
    assert str(log.origin.dtype) == "datetime64[ns, UTC]"


def test_join_log_to_actuals_leaves_unpublished_hours_nan():
    origin = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    timestamps = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="1h", tz="UTC")
    log = pd.DataFrame({"timestamp": timestamps, "prediction": 50_000.0})
    history = pd.DataFrame(
        {"timestamp": timestamps[:-6], "demand_mw": np.arange(18, dtype=float) + 50_000}
    )
    joined = join_log_to_actuals(log, history)
    assert len(joined) == len(log)
    assert joined.demand_mw.tail(6).isna().all()
    assert joined.demand_mw.head(18).notna().all()
