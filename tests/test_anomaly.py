import numpy as np
import pandas as pd
import pytest

from energy_forecast.anomaly import detect_anomalies


def test_injected_residual_is_flagged():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "demand_mw": [100, 100, 150]})
    result = detect_anomalies(
        frame, np.array([100, 100, 100]), np.array([-2, -1, 0, 1, 2]), quantile=0.8
    )
    assert result.is_anomaly.tolist() == [False, False, True]
    assert result.loc[2, "expected_demand_mw"] == 100


def test_anomaly_length_is_validated():
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01", tz="UTC")], "demand_mw": [1]})
    with pytest.raises(ValueError):
        detect_anomalies(frame, np.array([1, 2]), np.array([0]))
