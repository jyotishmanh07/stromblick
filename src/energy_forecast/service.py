"""Application service shared by FastAPI and Streamlit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_clean_demand
from .models import HistGradientBoostingForecast, forecast_with_interval


def demo_demand(hours: int = 24 * 90) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("1h")
    timestamps = pd.date_range(end=end, periods=hours, freq="1h", tz="UTC")
    local = timestamps.tz_convert("Europe/Berlin")
    hour, weekday = local.hour.to_numpy(), local.weekday.to_numpy()
    rng = np.random.default_rng(42)
    demand = (
        58_000 + 7_000 * np.cos((hour - 18) * 2 * np.pi / 24)
        + 3_500 * (weekday < 5) + 1_000 * np.sin(np.arange(hours) * 2 * np.pi / (24 * 365))
        + rng.normal(0, 350, hours)
    )
    return pd.DataFrame({"timestamp": timestamps, "demand_mw": demand})


class ForecastService:
    def __init__(self, data_path: str | Path = "data/clean/demand_hourly.csv") -> None:
        try:
            self.history = load_clean_demand(data_path)
            self.data_source = "SMARD clean export"
        except FileNotFoundError:
            self.history = demo_demand()
            self.data_source = "deterministic demo data (download SMARD data for production use)"
        self.model = HistGradientBoostingForecast()
        self.residuals = np.array([])
        self._fit_validation_residuals()

    def _fit_validation_residuals(self) -> None:
        """Learn interval width from a trailing, one-week validation window."""
        validation_hours = 24 * 7
        if len(self.history) <= validation_hours + 200:
            return
        train = self.history.iloc[:-validation_hours]
        validation = self.history.iloc[-validation_hours:]
        model = HistGradientBoostingForecast()
        model.fit(train)
        predictions = model.predict(pd.DatetimeIndex(validation.timestamp))
        self.residuals = validation.demand_mw.to_numpy() - predictions

    def forecast(self, as_of: pd.Timestamp, horizon_hours: int) -> dict[str, object]:
        output = forecast_with_interval(
            self.model, self.history, as_of, horizon_hours, validation_residuals=self.residuals
        )
        return {
            "model_version": output.model_version,
            "forecast": [
                {
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "prediction": round(float(prediction), 3),
                    "lower_bound": round(float(lower), 3),
                    "upper_bound": round(float(upper), 3),
                }
                for timestamp, prediction, lower, upper in zip(
                    output.timestamps, output.prediction, output.lower_bound, output.upper_bound
                )
            ],
        }
