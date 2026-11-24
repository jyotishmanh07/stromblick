import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from weather_study import (  # noqa: E402
    CITIES,
    WeatherForecast,
    weather_features,
    weighted_temperature,
)

from energy_forecast.service import demo_demand  # noqa: E402


def _payload(values):
    return {"hourly": {"time": ["2026-01-01T00:00", "2026-01-01T01:00"], "t": values}}


def test_weighted_temperature_uses_population_and_skips_missing_cities():
    berlin, hamburg = CITIES[0][3], CITIES[1][3]
    series = weighted_temperature([_payload([10.0, None]), _payload([0.0, 4.0])], "t")
    assert series.index.tz is not None
    assert series.iloc[0] == (10 * berlin) / (berlin + hamburg)
    assert series.iloc[1] == 4.0  # Berlin missing: Hamburg alone, nothing interpolated


def test_weather_features_read_only_the_past_and_present():
    stamps = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    temps = pd.Series(np.arange(48, dtype=float), index=stamps)
    features = weather_features(temps)
    later = temps.copy()
    later.iloc[30:] += 100
    np.testing.assert_allclose(weather_features(later).iloc[:30], features.iloc[:30])
    assert features.temp_mean_24h.iloc[23] == np.mean(np.arange(24))


def _weather_for(data):
    stamps = pd.DatetimeIndex(data.timestamp)
    temp = pd.Series(np.sin(np.arange(len(stamps)) / 24.0) * 10, index=stamps)
    full = pd.date_range(stamps[0], stamps[-1] + pd.Timedelta(hours=48), freq="1h")
    return weather_features(temp.reindex(full).fillna(0.0))


def test_weather_model_never_sees_demand_after_its_origin():
    data = demo_demand(24 * 40)
    origin = data.timestamp.iloc[-30]
    target = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="1h")
    weather = _weather_for(data)
    before = WeatherForecast(weather).fit(data[data.timestamp <= origin]).predict(target)
    tampered = data.copy()
    tampered.loc[tampered.timestamp > origin, "demand_mw"] += 1e6
    after = WeatherForecast(weather).fit(tampered[tampered.timestamp <= origin]).predict(target)
    np.testing.assert_allclose(after, before)


def test_weather_after_the_horizon_cannot_change_the_forecast():
    data = demo_demand(24 * 40)
    origin = data.timestamp.iloc[-30]
    target = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="1h")
    weather = _weather_for(data)
    train = data[data.timestamp <= origin]
    before = WeatherForecast(weather).fit(train).predict(target)
    shifted = weather.copy()
    shifted.loc[shifted.index > target[-1]] += 50
    np.testing.assert_allclose(WeatherForecast(shifted).fit(train).predict(target), before)


def test_missing_weather_does_not_trigger_the_last_value_fallback():
    data = demo_demand(24 * 40)
    origin = data.timestamp.iloc[-30]
    target = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="1h")
    weather = _weather_for(data)
    weather.loc[weather.index.isin(target)] = np.nan
    model = WeatherForecast(weather).fit(data[data.timestamp <= origin])
    predictions = model.predict(target)
    assert not np.allclose(predictions, model.history.iloc[-1])
    assert np.isfinite(predictions).all()
