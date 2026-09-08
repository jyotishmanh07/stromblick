import numpy as np
import pandas as pd

from energy_forecast.service import ForecastService, demo_demand
from energy_forecast.verification import forecast_record


def test_service_learns_validation_residuals_for_intervals(tmp_path):
    service = ForecastService(tmp_path / "missing.csv")
    service.history = demo_demand(24 * 30)
    service._fit_validation_residuals()
    assert len(service.residuals) == 24 * 7


def test_validation_residuals_skip_unobserved_hours(tmp_path):
    service = ForecastService(tmp_path / "missing.csv")
    service.history = demo_demand(24 * 30)
    service.history.loc[service.history.index[-5:], "demand_mw"] = np.nan
    service._fit_validation_residuals()
    assert len(service.residuals) == 24 * 7 - 5
    assert not np.isnan(service.residuals).any()


def test_recent_anomalies_score_the_trailing_week(tmp_path):
    service = ForecastService(tmp_path / "missing.csv")
    service.history = demo_demand(24 * 30)
    result = service.detect_recent_anomalies()
    assert len(result) == 24 * 7
    assert result.is_anomaly.dtype == bool


def test_recent_anomalies_need_enough_history(tmp_path):
    service = ForecastService(tmp_path / "missing.csv")
    service.history = demo_demand(24 * 20)
    assert service.detect_recent_anomalies().empty


def test_forecast_record_issues_from_last_observed(tmp_path):
    service = ForecastService(tmp_path / "missing.csv")
    service.history = demo_demand(24 * 30)
    service.history.loc[service.history.index[-3:], "demand_mw"] = np.nan
    service._fit_validation_residuals()
    record = forecast_record(service)
    assert len(record) == 24
    assert (record.origin == service.last_observed).all()
    assert record.timestamp.iloc[0] == service.last_observed + pd.Timedelta(hours=1)
    assert (record.model_version == "v1").all()
