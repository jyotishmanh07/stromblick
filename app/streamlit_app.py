"""Streamlit product interface; run with `streamlit run app/streamlit_app.py`."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from energy_forecast.evaluation import error_slices, metrics
from energy_forecast.models import HistGradientBoostingForecast, SeasonalNaive
from energy_forecast.service import ForecastService
from energy_forecast.theme import (
    ANOMALY,
    EXPECTED,
    FORECAST,
    INK,
    INTERVAL_FILL,
    MODEL_COLORS,
    MUTED,
    OBSERVED,
    plotly_layout,
)

GBM = MODEL_COLORS["HistGradientBoosting"]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
REPORTS = Path("reports")

st.set_page_config(page_title="Stromblick", page_icon="⚡", layout="wide")
st.title("Stromblick")
st.caption("Germany electricity demand forecasting and anomaly detection")


# --------------------------------------------------------------------------------------
# Cached computation
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and fitting the forecast model...")
def get_service() -> ForecastService:
    return ForecastService()


@st.cache_data(show_spinner="Forecasting the next 24 hours...")
def forecast_frame(_service: ForecastService, cache_key: str) -> pd.DataFrame:
    result = _service.forecast(_service.history.timestamp.max(), 24)
    frame = pd.DataFrame(result["forecast"])
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    return frame


@st.cache_data(show_spinner="Comparing models on the trailing week...")
def model_comparison(history: pd.DataFrame):
    train, test = history.iloc[: -24 * 7], history.iloc[-24 * 7 :]
    test = test[test.demand_mw.notna()].reset_index(drop=True)  # SMARD hours not yet published
    rows, gbm_predicted = [], None
    for name, model in [
        ("Seasonal naive", SeasonalNaive()),
        ("HistGradientBoosting", HistGradientBoostingForecast()),
    ]:
        model.fit(train)
        predicted = model.predict(pd.DatetimeIndex(test.timestamp))
        if name == "HistGradientBoosting":
            gbm_predicted = predicted
        rows.append({"model": name, **metrics(test.demand_mw.to_numpy(), predicted)})
    return pd.DataFrame(rows), test.reset_index(drop=True), gbm_predicted


@st.cache_data(show_spinner="Scoring the window for anomalies...")
def recent_anomalies(_service: ForecastService, cache_key: str, window_hours: int) -> pd.DataFrame:
    return _service.detect_recent_anomalies(window_hours=window_hours)


@st.cache_data
def load_benchmark():
    """Rolling-origin backtest artifacts from scripts/benchmark.py, if they exist."""
    summary_path = REPORTS / "benchmark_summary.json"
    metrics_path = REPORTS / "benchmark_metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    per_origin = pd.read_csv(metrics_path, parse_dates=["origin"])
    return summary, per_origin


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
service = get_service()
history = service.history
cache_key = f"{service.data_source}:{len(history)}:{history.timestamp.max()}"
forecast = forecast_frame(service, cache_key)
benchmark = load_benchmark()

col1, col2, col3 = st.columns(3)
col1.metric("Latest observed demand", f"{history.demand_mw.dropna().iloc[-1]:,.0f} MW")
col2.metric("Forecast horizon", "24 hours")
col3.metric("Data source", service.data_source)

forecast_tab, quality_tab, anomaly_tab, about_tab = st.tabs(
    ["Forecast", "Model quality", "Anomalies", "Data & methods"]
)


# --------------------------------------------------------------------------------------
# Forecast tab
# --------------------------------------------------------------------------------------
with forecast_tab:
    st.subheader("Next 24 hours")
    observed = history.tail(72)  # keep NaN rows so unpublished hours render as gaps
    last_observed_ts = history.dropna(subset=["demand_mw"]).timestamp.iloc[-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast.timestamp, y=forecast.upper_bound,
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.timestamp, y=forecast.lower_bound,
            name="Prediction interval", fill="tonexty", fillcolor=INTERVAL_FILL,
            line=dict(width=0), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=observed.timestamp, y=observed.demand_mw, name="Observed",
            line=dict(color=OBSERVED, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.timestamp, y=forecast.prediction, name="Forecast",
            line=dict(color=FORECAST, width=2.5),
        )
    )
    # epoch-ms is the form plotly reliably accepts for a vline on a datetime axis
    fig.add_vline(
        x=last_observed_ts.timestamp() * 1000,
        line=dict(color=MUTED, width=1, dash="dot"),
    )
    fig.update_layout(**plotly_layout(yaxis_title="Demand (MW)", xaxis_title="Time (UTC)"))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Observed demand for the last three days, then the 24-hour forecast (the dotted line "
        "marks the forecast start). The shaded band is the forecast ± the 95th percentile of "
        "absolute residuals the model made on a held-out validation week — its width reflects "
        "how wrong the model has recently been, not a probabilistic guarantee. Gaps in the "
        "observed line are hours SMARD has indexed but not yet published."
    )


# --------------------------------------------------------------------------------------
# Model quality tab
# --------------------------------------------------------------------------------------
with quality_tab:
    st.subheader("Rolling-origin backtest")
    if benchmark is None:
        st.info(
            "Run `PYTHONPATH=src python scripts/benchmark.py` to generate the full backtest "
            "(`reports/benchmark_summary.json` + `reports/benchmark_metrics.csv`). "
            "Showing the trailing-week comparison below only."
        )
    else:
        summary, per_origin = benchmark
        models = summary["models"]
        base_mae = models["Seasonal naive"]["mae_mean"]
        coverage = summary["interval_coverage"]
        st.caption(
            f"{summary['origins']} origins spaced {summary['step_hours']}h apart, "
            f"{summary['eval_hours']:,} forecast hours over "
            f"{summary['eval_start'][:10]} to {summary['eval_end'][:10]}. Every model is refit at "
            "every origin; no random split."
        )
        table = pd.DataFrame(
            [
                {
                    "Model": name,
                    "MAE (MW)": f"{m['mae_mean']:,.0f} (±{m['mae_std']:,.0f})",
                    "RMSE (MW)": f"{m['rmse_mean']:,.0f}",
                    "sMAPE (%)": f"{m['smape_mean']:.2f}",
                    "vs seasonal-naive": (
                        "—" if name == "Seasonal naive"
                        else f"{100 * (m['mae_mean'] - base_mae) / base_mae:+.1f}%"
                    ),
                }
                for name, m in models.items()
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.markdown(
            f"**{summary['champion']}** has the lowest error — MAE "
            f"**{summary['lift_vs_seasonal_naive_pct']:.1f}%** below the seasonal-naive baseline. "
            f"Prediction-interval coverage: **{coverage['empirical']:.1f}%** of observed values "
            f"land inside the {coverage['nominal']:.0f}% nominal band "
            f"(mean width {coverage['mean_band_mw']:,.0f} MW)."
        )

        origin_fig = go.Figure()
        for name, group in per_origin.groupby("model"):
            ordered = group.sort_values("origin")
            origin_fig.add_trace(
                go.Scatter(
                    x=ordered.origin, y=ordered.mae.rolling(14, min_periods=14).mean(),
                    name=name, line=dict(color=MODEL_COLORS.get(name, INK), width=2),
                )
            )
        origin_fig.update_layout(
            **plotly_layout(
                yaxis_title="MAE over the 24h window (MW)",
                xaxis_title="Origin — 14-origin trailing mean",
            )
        )
        st.plotly_chart(origin_fig, width="stretch")
        st.caption(
            "Each line is a 14-origin trailing mean of that model's 24-hour MAE; "
            "HistGradientBoosting stays below both baselines across the whole year, not just on "
            "average. The raw per-origin detail is in `reports/benchmark.md`."
        )

        importance_png = REPORTS / "figures" / "benchmark_feature_importance.png"
        slices_png = REPORTS / "figures" / "benchmark_error_slices.png"
        cols = st.columns(2)
        if importance_png.exists():
            cols[0].image(str(importance_png), caption="Permutation importance (validation week)")
        if slices_png.exists():
            cols[1].image(str(slices_png), caption=f"Where {summary['champion']} errs")

    st.divider()
    st.subheader("Trailing-week holdout")
    if len(history) >= 24 * 35:
        comparison, holdout, gbm_predicted = model_comparison(history)
        st.dataframe(
            comparison.round({"mae": 0, "rmse": 0, "smape": 2}), hide_index=True,
            width="stretch",
        )
        st.caption(
            "Metrics on the trailing 7-day holdout — the fast honesty check that runs live. "
            "Mean absolute error for HistGradientBoosting, sliced by local hour and weekday:"
        )
        slices = error_slices(holdout, gbm_predicted)
        left, right = st.columns(2)
        by_hour = slices["hour"].sort_values("hour")
        hour_fig = go.Figure(
            go.Bar(x=by_hour.hour, y=by_hour.absolute_error, marker_color=GBM)
        )
        hour_fig.update_layout(
            **plotly_layout(xaxis_title="Hour (Berlin)", yaxis_title="MAE (MW)", hovermode="x")
        )
        left.plotly_chart(hour_fig, width="stretch")
        by_weekday = slices["weekday"].copy()
        by_weekday["weekday"] = pd.Categorical(by_weekday.weekday, WEEKDAY_ORDER, ordered=True)
        by_weekday = by_weekday.sort_values("weekday")
        weekday_fig = go.Figure(
            go.Bar(
                x=[d[:3] for d in by_weekday.weekday], y=by_weekday.absolute_error,
                marker_color=GBM,
            )
        )
        weekday_fig.update_layout(
            **plotly_layout(xaxis_title="Weekday", yaxis_title="MAE (MW)", hovermode="x")
        )
        right.plotly_chart(weekday_fig, width="stretch")
    else:
        st.info("At least 35 days of hourly data are needed for the comparison panel.")


# --------------------------------------------------------------------------------------
# Anomalies tab
# --------------------------------------------------------------------------------------
with anomaly_tab:
    st.subheader("Historical anomaly explorer")
    st.info(
        "Anomalies are statistical flags from forecast residuals, not confirmed real-world events."
    )
    label_to_hours = {"Last 7 days": 24 * 7, "Last 14 days": 24 * 14, "Last 28 days": 24 * 28}
    choice = st.radio("Window", list(label_to_hours), horizontal=True, label_visibility="collapsed")
    anomalies = recent_anomalies(service, cache_key, label_to_hours[choice])
    if anomalies.empty:
        st.info("Not enough history to score this window; showing observed demand only.")
        st.line_chart(history.tail(label_to_hours[choice]).set_index("timestamp")["demand_mw"])
    else:
        flagged = anomalies[anomalies.is_anomaly]
        anomaly_fig = go.Figure()
        anomaly_fig.add_trace(
            go.Scatter(
                x=anomalies.timestamp,
                y=anomalies.expected_demand_mw + anomalies.upper_residual_bound,
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            )
        )
        anomaly_fig.add_trace(
            go.Scatter(
                x=anomalies.timestamp,
                y=anomalies.expected_demand_mw + anomalies.lower_residual_bound,
                name="Expected ± residual bounds", fill="tonexty", fillcolor=INTERVAL_FILL,
                line=dict(width=0), hoverinfo="skip",
            )
        )
        anomaly_fig.add_trace(
            go.Scatter(
                x=anomalies.timestamp, y=anomalies.expected_demand_mw, name="Expected",
                line=dict(color=EXPECTED, width=2, dash="dash"),
            )
        )
        anomaly_fig.add_trace(
            go.Scatter(
                x=anomalies.timestamp, y=anomalies.demand_mw, name="Observed",
                line=dict(color=OBSERVED, width=2),
            )
        )
        if not flagged.empty:
            anomaly_fig.add_trace(
                go.Scatter(
                    x=flagged.timestamp, y=flagged.demand_mw, name="Anomaly", mode="markers",
                    marker=dict(symbol="x", size=11, color=ANOMALY),
                )
            )
        anomaly_fig.update_layout(
            **plotly_layout(yaxis_title="Demand (MW)", xaxis_title="Time (UTC)")
        )
        st.plotly_chart(anomaly_fig, width="stretch")
        if flagged.empty:
            st.caption(f"No hour in {choice.lower()} left the 99% residual bounds.")
        else:
            st.caption(
                f"{len(flagged)} of {len(anomalies)} hours flagged (99% residual bounds). "
                "The bounds are learned from the validation week *before* this window, so they "
                "never see the data they score."
            )
            st.dataframe(
                flagged[["timestamp", "demand_mw", "expected_demand_mw", "deviation_mw"]].round(1),
                hide_index=True, width="stretch",
            )


# --------------------------------------------------------------------------------------
# Data & methods tab
# --------------------------------------------------------------------------------------
with about_tab:
    st.subheader("Data snapshot")
    metadata_path = Path("data/clean/metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        st.markdown(
            f"**{metadata['source']}** (CC BY 4.0), module {metadata['module_id']} — Germany "
            f"actual total grid load. Snapshot of **{metadata['rows']:,} hourly rows** from "
            f"`{metadata['first_timestamp']}` to `{metadata['last_timestamp']}`, collected "
            f"`{metadata['collected_at']}` from {len(metadata['chunk_urls'])} weekly chunks. "
            "SMARD may revise historical values; published results should cite this snapshot."
        )
    else:
        st.markdown(
            "This view uses deterministic synthetic demand because "
            "`data/clean/demand_hourly.csv` is absent. Run "
            "`PYTHONPATH=src python scripts/ingest_smard_api.py --weeks 52` to collect the "
            "official SMARD snapshot."
        )

    st.subheader("Method")
    st.markdown(
        "- **Three model levels.** Seasonal naive (same hour yesterday, previous week as "
        "fallback) → SARIMAX (interpretable statistical baseline) → HistGradientBoostingRegressor "
        "(calendar, German-holiday and DST flags, demand lags 1h/24h/168h, trailing means).\n"
        "- **Leakage-safe features.** Every lag and rolling window is shifted before "
        "aggregation, so no training row can see its own or a future value.\n"
        "- **Chronological evaluation.** `rolling_origin_backtest` walks origins forward in "
        "time and refits every model at each one. No random split is ever used.\n"
        "- **Honest baseline.** The gradient-boosting model is judged against seasonal-naive: "
        "if it cannot beat \"same hour yesterday\", it is not earning its complexity.\n"
        "- **Empirical intervals and anomalies.** Both come from validation-residual "
        "magnitudes only — not calibrated probabilistic forecasts, and anomalies are prompts "
        "to investigate weather, calendar, or grid context rather than confirmed events."
    )
    if benchmark is not None:
        st.caption(
            "The full backtest report lives in `reports/benchmark.md`; the EDA in `reports/eda.md`."
        )
