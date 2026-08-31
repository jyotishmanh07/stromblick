import pandas as pd
import pytest

from energy_forecast.data import DataValidationError, canonicalize_demand, missing_hourly_timestamps


def test_canonicalize_deduplicates_and_converts_local_time():
    source = pd.DataFrame(
        {
            "time": ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"],
            "consumption": [1, 3, 2],
        }
    )
    result = canonicalize_demand(source)
    assert list(result.columns) == ["timestamp", "demand_mw"]
    assert result.demand_mw.tolist() == [3.0, 2.0]
    assert str(result.timestamp.dt.tz) == "UTC"


def test_missing_hours_are_reported():
    timestamps = pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T02:00Z"])
    data = pd.DataFrame({"timestamp": timestamps, "demand_mw": [1, 2]})
    assert len(missing_hourly_timestamps(data)) == 1


def test_invalid_schema_fails_loudly():
    with pytest.raises(DataValidationError):
        canonicalize_demand(pd.DataFrame({"foo": [1]}))
