"""Residual-based anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def residual_bounds(residuals: np.ndarray, quantile: float = 0.99) -> tuple[float, float]:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        raise ValueError("historical residuals are required")
    return float(np.quantile(residuals, 1 - quantile)), float(np.quantile(residuals, quantile))


def detect_anomalies(
    frame: pd.DataFrame, prediction: np.ndarray, validation_residuals: np.ndarray,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """Flag actual-vs-expected deviations using validation residuals only."""
    if len(frame) != len(prediction):
        raise ValueError("frame and prediction lengths differ")
    lower, upper = residual_bounds(validation_residuals, quantile)
    result = frame[["timestamp", "demand_mw"]].copy()
    result["expected_demand_mw"] = np.asarray(prediction, dtype=float)
    result["deviation_mw"] = result["demand_mw"] - result["expected_demand_mw"]
    result["lower_residual_bound"] = lower
    result["upper_residual_bound"] = upper
    result["is_anomaly"] = (result.deviation_mw < lower) | (result.deviation_mw > upper)
    result["context"] = np.where(
        result.is_anomaly,
        "Statistical anomaly; investigate weather, calendar, and operational context",
        "",
    )
    return result
