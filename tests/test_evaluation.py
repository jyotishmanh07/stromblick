import math

import numpy as np
import pandas as pd

from energy_forecast.evaluation import metrics, rolling_origin_backtest
from energy_forecast.models import SeasonalNaive


def test_metrics_include_mae_rmse_and_smape():
    result = metrics(np.array([10.0, 20.0]), np.array([8.0, 24.0]))
    assert set(result) == {"mae", "rmse", "smape"}
    assert result["mae"] == 3.0
    assert result["rmse"] ==  math.sqrt(10)


def test_backtest_origins_are_chronological():
    timestamps = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
    data = pd.DataFrame({"timestamp": timestamps, "demand_mw": np.arange(96, dtype=float)})
    result = rolling_origin_backtest(
        data, {"naive": SeasonalNaive}, initial_train_hours=48, horizon_hours=12, step_hours=12
    )
    assert result.origin.is_monotonic_increasing
    assert len(result) == 4
