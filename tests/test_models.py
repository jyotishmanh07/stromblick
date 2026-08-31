import numpy as np
import pandas as pd

from energy_forecast.models import HistGradientBoostingForecast, SeasonalNaive
from energy_forecast.service import demo_demand


def test_seasonal_naive_uses_previous_day():
    timestamps = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    data = pd.DataFrame({"timestamp": timestamps, "demand_mw": np.arange(72)})
    model = SeasonalNaive().fit(data)
    prediction = model.predict(pd.DatetimeIndex([pd.Timestamp("2024-01-04T00:00Z")]))
    assert prediction[0] == 48


def test_gradient_boosting_forecasts_requested_horizon():
    data = demo_demand(24 * 20)
    model = HistGradientBoostingForecast().fit(data)
    timestamps = pd.date_range(
        data.timestamp.max() + pd.Timedelta(hours=1), periods=4, freq="h", tz="UTC"
    )
    prediction = model.predict(timestamps)
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
