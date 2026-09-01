"""Application service shared by FastAPI and Streamlit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly import detect_anomalies
from .data import DataValidationError, load_clean_demand
from .live import COMMITTED_LABEL, fetch_live_history, last_observed
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
    def __init__(
        self, data_path: str | Path = "data/clean/demand_hourly.csv", live: bool = False,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Load history and prepare the interval model.

        `live=True` additionally pulls the recent tail from SMARD and merges it onto
        the committed snapshot, so the forecast starts from the last published hour
        rather than from whenever the snapshot was collected. It defaults to off:
        `api.py` builds this at module import, and a network call there would slow
        every startup and put the API tests on the network.
        """
        self.is_live = False
        self.provenance: dict | None = None
        self.live_warning: str | None = None
        try:
            self.history = load_clean_demand(data_path)
            self.data_source = COMMITTED_LABEL
        except (FileNotFoundError, DataValidationError):
            # A corrupt snapshot degrades to demo data rather than taking the app down.
            self.history = demo_demand()
            self.data_source = "deterministic demo data (download SMARD data for production use)"
        if live and self.data_source == COMMITTED_LABEL:
            result = fetch_live_history(self.history, timeout_seconds=timeout_seconds)
            self.history = result.history
            self.data_source = result.source_label
            self.is_live = result.is_live
            self.provenance = result.provenance
            self.live_warning = result.warning
        self.model = HistGradientBoostingForecast()
        self.residuals = np.array([])
        self._fit_validation_residuals()

    @property
    def last_observed(self) -> pd.Timestamp | None:
        """Most recent hour with a published value — the correct forecast origin."""
        return last_observed(self.history)

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
        residuals = validation.demand_mw.to_numpy() - predictions
        self.residuals = residuals[~np.isnan(residuals)]  # hours with no observed demand

    def detect_recent_anomalies(
        self, window_hours: int = 24 * 7, quantile: float = 0.99
    ) -> pd.DataFrame:
        """Score the trailing window against residual bounds learned strictly before it."""
        validation_hours = 24 * 7
        if len(self.history) <= window_hours + validation_hours + 200:
            return pd.DataFrame()
        window = self.history.iloc[-window_hours:].reset_index(drop=True)
        earlier = self.history.iloc[:-window_hours]
        validation = earlier.iloc[-validation_hours:]
        residual_model = HistGradientBoostingForecast().fit(earlier.iloc[:-validation_hours])
        residuals = validation.demand_mw.to_numpy() - residual_model.predict(
            pd.DatetimeIndex(validation.timestamp)
        )
        residuals = residuals[~np.isnan(residuals)]  # hours with no observed demand
        window_model = HistGradientBoostingForecast().fit(earlier)
        predicted = window_model.predict(pd.DatetimeIndex(window.timestamp))
        return detect_anomalies(window, predicted, residuals, quantile)

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
