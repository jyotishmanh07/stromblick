import pandas as pd

from energy_forecast.data import canonicalize_demand


def test_repeated_fall_back_local_hour_is_handled():
    source = pd.DataFrame(
        {
            "timestamp": [
                "2024-10-27 01:00", "2024-10-27 02:00", "2024-10-27 02:00",
                "2024-10-27 03:00", "2024-10-27 04:00",
            ],
            "demand_mw": [1, 2, 3, 4, 5],
        }
    )
    result = canonicalize_demand(source)
    assert len(result) == 5
    assert result.demand_mw.tolist() == [1, 2, 3, 4, 5]
