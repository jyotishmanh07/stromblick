"""Three deliberately interpretable forecasting levels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except (ImportError, OSError, TypeError, ValueError):
    HistGradientBoostingRegressor = None

from .features import FEATURE_COLUMNS, add_features, feature_row


def _timestamps(values: pd.Series | pd.DataFrame | pd.DatetimeIndex) -> pd.DatetimeIndex:
    if isinstance(values, pd.DataFrame):
        values = values["timestamp"]
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


class SeasonalNaive:
    name = "seasonal_naive"

    def fit(self, frame: pd.DataFrame) -> "SeasonalNaive":
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
        series = data.drop_duplicates("timestamp").set_index("timestamp")["demand_mw"]
        # Drop SMARD hours indexed but not yet published: a NaN is not an observation,
        # and it must never become the iloc[-1] last-resort fallback below.
        self.series = series.sort_index().dropna()
        return self

    def predict(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        values = []
        for timestamp in _timestamps(timestamps):
            value = self.series.get(timestamp - pd.Timedelta(days=1))
            if pd.isna(value):
                value = self.series.get(timestamp - pd.Timedelta(days=7))
            if pd.isna(value):
                value = self.series.iloc[-1]
            values.append(float(value))
        return np.asarray(values)


class SARIMAXBaseline:
    name = "sarimax"

    def fit(self, frame: pd.DataFrame) -> "SARIMAXBaseline":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
        self.series = data.set_index("timestamp")["demand_mw"].sort_index().asfreq("1h")
        self.fallback = SeasonalNaive().fit(frame)
        try:
            self.result = SARIMAX(
                self.series, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        except Exception:
            self.result = None
        return self

    def predict(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        timestamps = _timestamps(timestamps)
        if self.result is None:
            return self.fallback.predict(timestamps)
        try:
            forecast = self.result.get_forecast(steps=len(timestamps))
            return np.asarray(forecast.predicted_mean, dtype=float)
        except Exception:
            return self.fallback.predict(timestamps)


class HistGradientBoostingForecast:
    name = "hist_gradient_boosting"

    # Shipped configuration. scripts/tune.py searches around these and only adopts a
    # replacement when the outer backtest gain clears a paired-bootstrap CI; see the
    # "Model configuration" section of reports/benchmark.md for the standing verdict.
    DEFAULT_PARAMS = {
        "max_iter": 250, "learning_rate": 0.06, "max_leaf_nodes": 31, "l2_regularization": 1.0,
    }

    def __init__(self, random_state: int = 42, **params) -> None:
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.random_state = random_state
        self.regressor = HistGradientBoostingRegressor(
            random_state=random_state, **self.params
        ) if HistGradientBoostingRegressor is not None else None

    def fit(self, frame: pd.DataFrame) -> "HistGradientBoostingForecast":
        features = add_features(frame).dropna(subset=FEATURE_COLUMNS + ["demand_mw"])
        if len(features) < 200:
            raise ValueError("at least 200 complete hourly observations are required")
        if self.regressor is not None:
            self.regressor.fit(features[FEATURE_COLUMNS], features["demand_mw"])
        else:
            matrix = features[FEATURE_COLUMNS].to_numpy(dtype=float)
            self.coefficients = np.linalg.lstsq(
                np.column_stack([np.ones(len(matrix)), matrix]), features["demand_mw"], rcond=None
            )[0]
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
        # Recursive forecasting reads lags off this series; a NaN (indexed but unpublished
        # SMARD hour) would poison every downstream step through the iloc[-1] fallback.
        self.history = data.set_index("timestamp")["demand_mw"].sort_index().astype(float).dropna()
        return self

    def forecast(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        history = self.history.copy()
        predictions: list[float] = []
        for timestamp in _timestamps(timestamps):
            row = feature_row(history, timestamp)
            if any(pd.isna(row[column]) for column in FEATURE_COLUMNS):
                prediction = float(history.iloc[-1])
            else:
                row_frame = pd.DataFrame([row])[FEATURE_COLUMNS]
                matrix = row_frame.to_numpy(dtype=float)
                if self.regressor is not None:
                    prediction = float(self.regressor.predict(row_frame)[0])
                else:
                    design = np.column_stack([np.ones(len(matrix)), matrix])
                    prediction = float((design @ self.coefficients).item())
            predictions.append(prediction)
            history.loc[timestamp] = prediction
        return np.asarray(predictions)

    predict = forecast


@dataclass
class ForecastOutput:
    timestamps: pd.DatetimeIndex
    prediction: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    model_version: str


def forecast_with_interval(
    model: object, history: pd.DataFrame, as_of: pd.Timestamp, horizon_hours: int,
    residual_quantile: float = 0.95, validation_residuals: np.ndarray | None = None,
) -> ForecastOutput:
    start = pd.to_datetime(as_of, utc=True)
    timestamps = pd.date_range(
        start.ceil("1h") + pd.Timedelta(hours=1),
        periods=horizon_hours, freq="1h", tz="UTC",
    )
    model.fit(history)
    prediction = model.predict(timestamps)
    residuals = np.asarray(
        validation_residuals if validation_residuals is not None else [], dtype=float
    )
    if len(residuals):
        spread = float(np.quantile(np.abs(residuals), residual_quantile))
    else:
        spread = float(np.std(history.demand_mw) * 0.15)
    return ForecastOutput(timestamps, prediction, prediction - spread, prediction + spread, "v1")
