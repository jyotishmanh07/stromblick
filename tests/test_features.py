import numpy as np
import pandas as pd

from energy_forecast.features import FEATURE_COLUMNS, add_features


def test_lags_and_rolling_features_do_not_use_current_target():
    timestamps = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    data = pd.DataFrame({"timestamp": timestamps, "demand_mw": np.arange(200, dtype=float)})
    features = add_features(data)
    row = features.iloc[168]
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert row.lag_1h == 167
    assert row.lag_24h == 144
    assert row.rolling_24h_mean == np.mean(np.arange(144, 168))
    assert row.rolling_7d_mean == np.mean(np.arange(0, 168))
