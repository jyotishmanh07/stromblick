import numpy as np
import pandas as pd
import pytest

from energy_forecast import models
from energy_forecast.models import (
    HistGradientBoostingForecast,
    SARIMAXBaseline,
    SeasonalNaive,
    forecast_with_interval,
)
from energy_forecast.service import demo_demand


def _ramp(hours: int) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "demand_mw": np.arange(hours, dtype=float)})


def _after(data: pd.DataFrame, periods: int, gap_hours: int = 1) -> pd.DatetimeIndex:
    start = data.timestamp.max() + pd.Timedelta(hours=gap_hours)
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def test_seasonal_naive_uses_previous_day():
    model = SeasonalNaive().fit(_ramp(72))
    prediction = model.predict(pd.DatetimeIndex([pd.Timestamp("2024-01-04T00:00Z")]))
    assert prediction[0] == 48


def test_seasonal_naive_falls_back_to_previous_week_when_yesterday_is_missing():
    data = _ramp(24 * 10)
    target = pd.Timestamp("2024-01-10T00:00Z")
    data = data[data.timestamp != target - pd.Timedelta(days=1)]
    prediction = SeasonalNaive().fit(data).predict(pd.DatetimeIndex([target]))
    assert prediction[0] == 24 * 2  # the value one week earlier, 2024-01-03T00:00


def test_seasonal_naive_uses_last_observation_when_no_seasonal_lag_exists():
    data = _ramp(30)
    prediction = SeasonalNaive().fit(data).predict(_after(data, 1, gap_hours=24 * 30))
    assert prediction[0] == 29


def test_sarimax_forecasts_the_requested_horizon():
    data = demo_demand(24 * 10)
    prediction = SARIMAXBaseline().fit(data).predict(_after(data, 6))
    assert prediction.shape == (6,)
    assert np.isfinite(prediction).all()


def test_sarimax_falls_back_to_seasonal_naive_when_fitting_fails(monkeypatch):
    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr("statsmodels.tsa.statespace.sarimax.SARIMAX", broken)
    data = demo_demand(24 * 10)
    timestamps = _after(data, 6)
    model = SARIMAXBaseline().fit(data)
    assert model.result is None
    np.testing.assert_array_equal(
        model.predict(timestamps), SeasonalNaive().fit(data).predict(timestamps)
    )


def test_sarimax_falls_back_to_seasonal_naive_when_forecasting_fails():
    class Broken:
        def get_forecast(self, steps):
            raise ValueError("forecast failed")

    data = demo_demand(24 * 10)
    timestamps = _after(data, 6)
    model = SARIMAXBaseline().fit(data)
    model.result = Broken()
    np.testing.assert_array_equal(
        model.predict(timestamps), SeasonalNaive().fit(data).predict(timestamps)
    )


def test_gradient_boosting_forecasts_requested_horizon():
    data = demo_demand(24 * 20)
    prediction = HistGradientBoostingForecast().fit(data).predict(_after(data, 4))
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()


def test_gradient_boosting_feeds_each_prediction_into_the_next_lag():
    class LagPlusOne:
        def predict(self, row):
            return np.array([row["lag_1h"].iloc[0] + 1])

    data = demo_demand(24 * 20)
    model = HistGradientBoostingForecast().fit(data)
    model.regressor = LagPlusOne()
    last = data.demand_mw.iloc[-1]
    np.testing.assert_allclose(model.predict(_after(data, 3)), [last + 1, last + 2, last + 3])


def test_gradient_boosting_holds_the_last_value_when_lags_are_unavailable():
    data = demo_demand(24 * 20)
    prediction = HistGradientBoostingForecast().fit(data).predict(_after(data, 1, gap_hours=3))
    assert prediction[0] == data.demand_mw.iloc[-1]


def test_gradient_boosting_needs_enough_complete_history():
    with pytest.raises(ValueError):
        HistGradientBoostingForecast().fit(demo_demand(24 * 7))


def test_gradient_boosting_degrades_to_a_linear_fit_without_sklearn(monkeypatch):
    monkeypatch.setattr(models, "HistGradientBoostingRegressor", None)
    data = demo_demand(24 * 20)
    model = HistGradientBoostingForecast().fit(data)
    prediction = model.predict(_after(data, 4))
    assert model.regressor is None
    assert np.isfinite(prediction).all()
    assert data.demand_mw.min() * 0.5 < prediction.mean() < data.demand_mw.max() * 1.5


def test_interval_uses_the_residual_quantile_when_residuals_exist():
    data = demo_demand(24 * 10)
    residuals = np.array([-300.0, 100.0, 200.0, -400.0])
    output = forecast_with_interval(
        SeasonalNaive(), data, data.timestamp.max(), 4, validation_residuals=residuals
    )
    spread = np.quantile(np.abs(residuals), 0.95)
    np.testing.assert_allclose(output.upper_bound - output.prediction, spread)


def test_interval_falls_back_to_a_share_of_demand_spread_without_residuals():
    data = demo_demand(24 * 10)
    output = forecast_with_interval(SeasonalNaive(), data, data.timestamp.max(), 4)
    np.testing.assert_allclose(
        output.upper_bound - output.lower_bound, 2 * 0.15 * np.std(data.demand_mw)
    )
