import numpy as np
import pandas as pd
import pytest

from energy_forecast.service import ForecastService, demo_demand
from energy_forecast.verification import forecast_record


def test_service_learns_validation_residuals_for_intervals(tmp_path):
    service = ForecastService(tmp_path / "missing.csv", pool_path=tmp_path / "no_pool.csv")
    service.history = demo_demand(24 * 30)
    service._fit_validation_residuals()
    assert len(service.residuals) == 24 * 7


def test_validation_residuals_skip_unobserved_hours(tmp_path):
    service = ForecastService(tmp_path / "missing.csv", pool_path=tmp_path / "no_pool.csv")
    service.history = demo_demand(24 * 30)
    service.history.loc[service.history.index[-5:], "demand_mw"] = np.nan
    service._fit_validation_residuals()
    assert len(service.residuals) == 24 * 7 - 5
    assert not np.isnan(service.residuals).any()


def test_recent_anomalies_score_the_trailing_week(tmp_path):
    service = ForecastService(tmp_path / "missing.csv", pool_path=tmp_path / "no_pool.csv")
    service.history = demo_demand(24 * 30)
    result = service.detect_recent_anomalies()
    assert len(result) == 24 * 7
    assert result.is_anomaly.dtype == bool


def test_recent_anomalies_need_enough_history(tmp_path):
    service = ForecastService(tmp_path / "missing.csv", pool_path=tmp_path / "no_pool.csv")
    service.history = demo_demand(24 * 20)
    assert service.detect_recent_anomalies().empty


def test_forecast_record_issues_from_last_observed(tmp_path):
    service = ForecastService(tmp_path / "missing.csv", pool_path=tmp_path / "no_pool.csv")
    service.history = demo_demand(24 * 30)
    service.history.loc[service.history.index[-3:], "demand_mw"] = np.nan
    service._fit_validation_residuals()
    record = forecast_record(service)
    assert len(record) == 24
    assert (record.origin == service.last_observed).all()
    assert record.timestamp.iloc[0] == service.last_observed + pd.Timedelta(hours=1)
    assert (record.model_version == "v2").all()


def _pool_for(history: pd.DataFrame, origins: int, error: float = 1_000.0) -> pd.DataFrame:
    """A synthetic pool: `origins` daily origins ending at the history's last hour."""
    from energy_forecast.calibration import POOL_COLUMNS
    from energy_forecast.models import MODEL_VERSION

    end = history.timestamp.max()
    rows = []
    for k in range(origins, 0, -1):
        origin = end - pd.Timedelta(hours=24 * k)
        stamps = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="1h")
        rows.append(pd.DataFrame({
            "origin": origin, "timestamp": stamps, "demand_mw": 50_000.0,
            "prediction": 50_000.0 + error, "abs_error": error, "model_version": MODEL_VERSION,
        }))
    return pd.concat(rows, ignore_index=True)[POOL_COLUMNS]


def test_service_serves_the_conformal_band_when_a_current_pool_exists(tmp_path):
    from energy_forecast.service import INTERVAL_CONFORMAL

    service = ForecastService(tmp_path / "missing.csv", pool_path=tmp_path / "no_pool.csv")
    service.use_calibration_pool(_pool_for(service.history, 35, error=1_234.0))
    assert service.interval_method == INTERVAL_CONFORMAL
    row = service.forecast(service.last_observed, 3)["forecast"][0]
    assert row["upper_bound"] - row["prediction"] == pytest.approx(1_234.0)


def test_service_falls_back_and_says_so_without_a_pool(tmp_path):
    from energy_forecast.service import INTERVAL_FALLBACK

    service = ForecastService(tmp_path / "missing.csv", pool_path=tmp_path / "no_pool.csv")
    assert service.interval_method == INTERVAL_FALLBACK
    assert len(service.residuals) > 0
