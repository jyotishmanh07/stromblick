import numpy as np
import pandas as pd

from energy_forecast.data import missing_hourly_timestamps
from energy_forecast.evaluation import error_slices
from energy_forecast.features import german_holidays
from energy_forecast.service import demo_demand
from energy_forecast.warehouse import build_warehouse, query


def _db(tmp_path, frame, forecasts=None):
    return build_warehouse(frame, tmp_path / "test.duckdb", forecasts)


def test_daily_profile_sql_matches_the_pandas_groupby(tmp_path):
    frame = demo_demand()
    local = frame.timestamp.dt.tz_convert("Europe/Berlin")
    holidays = german_holidays(set(local.dt.year))
    day_type = np.select(
        [local.dt.date.isin(holidays), local.dt.weekday >= 5],
        ["Public holiday", "Weekend"], default="Weekday",
    )
    expected = (
        pd.DataFrame({"day_type": day_type, "hour": local.dt.hour, "demand_mw": frame.demand_mw})
        .groupby(["day_type", "hour"], as_index=False)
        .demand_mw.mean()
    )
    result = query("daily_profile_by_daytype", _db(tmp_path, frame))
    merged = result.merge(expected, on=["day_type", "hour"], suffixes=("_sql", "_pd"))
    assert len(merged) == len(expected) == len(result)
    assert np.allclose(merged.demand_mw_sql, merged.demand_mw_pd)


def test_weekly_pattern_sql_matches_pandas_and_starts_on_monday(tmp_path):
    frame = demo_demand()
    db = _db(tmp_path, frame)
    local = frame.timestamp.dt.tz_convert("Europe/Berlin")
    expected = frame.groupby(local.dt.day_name()).demand_mw.mean()
    result = query("weekly_pattern", db)
    assert result.weekday_name.iloc[0] == "Monday"
    assert np.allclose(result.set_index("weekday_name").demand_mw, expected[result.weekday_name])


def test_missing_hours_sql_agrees_with_the_pandas_gap_report(tmp_path):
    frame = demo_demand().drop(index=[100, 101]).reset_index(drop=True)
    frame.loc[200, "demand_mw"] = np.nan
    db = _db(tmp_path, frame)
    result = query("missing_hours", db)
    absent = result[result.gap_type == "absent from index"]
    unpublished = result[result.gap_type == "indexed, no value"]
    assert len(absent) == len(missing_hourly_timestamps(frame)) == 2
    assert len(unpublished) == 1


def test_error_slices_sql_matches_the_pandas_implementation(tmp_path):
    frame = demo_demand()
    rng = np.random.default_rng(0)
    scored = frame.iloc[-500:].reset_index(drop=True)
    predicted = scored.demand_mw.to_numpy() + rng.normal(0, 500, size=len(scored))
    forecasts = pd.DataFrame(
        {
            "origin": scored.timestamp.iloc[0], "timestamp": scored.timestamp,
            "model": "test", "demand_mw": scored.demand_mw, "predicted": predicted,
        }
    )
    db = _db(tmp_path, frame, forecasts)
    sql = query("error_slices", db, params=["test"])
    expected = error_slices(scored[["timestamp", "demand_mw"]], predicted)

    by_hour = sql[sql.slice == "hour"].astype({"bucket": int}).set_index("bucket").absolute_error
    assert np.allclose(by_hour[expected["hour"].hour], expected["hour"].absolute_error)
    by_weekday = sql[sql.slice == "weekday"].set_index("bucket").absolute_error
    assert np.allclose(
        by_weekday[expected["weekday"].weekday], expected["weekday"].absolute_error
    )


def test_model_summary_averages_per_origin_not_per_hour(tmp_path):
    frame = demo_demand()
    scored = frame.iloc[-48:].reset_index(drop=True)
    # Two origins: the first errs by 100 MW over 24h, the second by 300 MW.
    forecasts = pd.concat(
        [
            pd.DataFrame(
                {
                    "origin": scored.timestamp.iloc[i], "timestamp": scored.timestamp[i : i + 24],
                    "model": "test", "demand_mw": scored.demand_mw[i : i + 24],
                    "predicted": scored.demand_mw[i : i + 24] + error,
                }
            )
            for i, error in ((0, 100.0), (24, 300.0))
        ],
        ignore_index=True,
    )
    result = query("model_summary", _db(tmp_path, frame, forecasts))
    assert result.origins.iloc[0] == 2
    assert np.isclose(result.mae_mean.iloc[0], 200.0)
