import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tso_study import TSO, day_cutoffs, forecast_day, per_day  # noqa: E402

from energy_forecast.service import demo_demand  # noqa: E402


def test_dst_days_have_23_and_25_hours():
    assert len(day_cutoffs("2026-03-29")["hours"]) == 23
    assert len(day_cutoffs("2026-10-25")["hours"]) == 25
    assert len(day_cutoffs("2026-06-15")["hours"]) == 24


def test_issue_time_is_ten_oclock_berlin_the_day_before_in_both_seasons():
    winter, summer = day_cutoffs("2026-01-15"), day_cutoffs("2026-07-15")
    assert winter["issue_time"] == pd.Timestamp("2026-01-14 09:00", tz="UTC")
    assert summer["issue_time"] == pd.Timestamp("2026-07-14 08:00", tz="UTC")


def test_cutoffs_get_earlier_as_the_information_shrinks():
    cut = day_cutoffs("2026-05-10")
    assert cut["lagged"] < cut["issue_time"] < cut["fresh"] < cut["hours"][0]
    assert cut["issue_time"] - cut["lagged"] == pd.Timedelta(hours=24)


def test_forecast_never_sees_data_after_its_origin():
    data = demo_demand(24 * 60)
    hours = pd.date_range(data.timestamp.iloc[-24], periods=24, freq="1h")
    origin = hours[0] - pd.Timedelta(hours=30)
    before = forecast_day(data, origin, hours)
    tampered = data.copy()
    tampered.loc[tampered.timestamp > origin, "demand_mw"] += 1e6
    np.testing.assert_allclose(forecast_day(tampered, origin, hours), before)


def test_per_day_scores_only_hours_every_series_has():
    stamps = pd.date_range("2026-05-10 22:00", periods=24, freq="1h", tz="UTC")
    hourly = pd.DataFrame({
        "day": pd.Timestamp("2026-05-11").date(), "timestamp": stamps,
        "actual": 100.0, "tso": 110.0, "fresh": 100.0, "issue_time": 100.0, "lagged": 90.0,
    })
    hourly.loc[0, "lagged"] = np.nan
    days = per_day(hourly)
    tso = days[days.method == TSO].iloc[0]
    assert tso.hours == 23
    assert tso.mae == 10.0
    blend = days[days.method.str.startswith("50/50")].iloc[0]
    assert blend.mae == 0.0  # (110 + 90) / 2 == 100
