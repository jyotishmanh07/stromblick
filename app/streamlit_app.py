"""Streamlit product interface; run with `streamlit run app/streamlit_app.py`."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from energy_forecast.evaluation import error_slices, metrics
from energy_forecast.models import HistGradientBoostingForecast, SeasonalNaive
from energy_forecast.service import ForecastService

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Stromblick", layout="wide")
st.title("Stromblick")
st.caption("Germany electricity demand forecasting and anomaly detection")


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


@st.cache_data(show_spinner="Scoring the trailing week for anomalies...")
def recent_anomalies(_service: ForecastService, cache_key: str) -> pd.DataFrame:
    return _service.detect_recent_anomalies()


service = get_service()
history = service.history
cache_key = f"{service.data_source}:{len(history)}:{history.timestamp.max()}"
forecast = forecast_frame(service, cache_key)

col1, col2, col3 = st.columns(3)
col1.metric("Latest demand", f"{history.demand_mw.iloc[-1]:,.0f} MW")
col2.metric("Forecast horizon", "24 hours")
col3.metric("Data source", service.data_source)

st.subheader("Next 24 hours")
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=history.timestamp.tail(72), y=history.demand_mw.tail(72), name="Observed")
)
fig.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.prediction, name="Forecast"))
fig.add_trace(
    go.Scatter(
        x=forecast.timestamp, y=forecast.upper_bound,
        name="Upper interval", line=dict(width=0), showlegend=False,
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast.timestamp, y=forecast.lower_bound,
        name="Prediction interval", fill="tonexty", line=dict(width=0),
    )
)
fig.update_layout(yaxis_title="Demand (MW)", xaxis_title="Time", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model comparison")
if len(history) >= 24 * 35:
    comparison, holdout, gbm_predicted = model_comparison(history)
    st.dataframe(comparison.round({"mae": 0, "rmse": 0, "smape": 2}), hide_index=True)
    st.caption(
        "Metrics on the trailing 7-day holdout. Where does the model err? "
        "Mean absolute error sliced by local hour and weekday:"
    )
    slices = error_slices(holdout, gbm_predicted)
    left, right = st.columns(2)
    with left:
        by_hour = slices["hour"].set_index("hour")
        st.bar_chart(by_hour, y="absolute_error", x_label="Hour (Berlin)", y_label="MAE (MW)")
    with right:
        by_weekday = slices["weekday"].copy()
        by_weekday["weekday"] = pd.Categorical(by_weekday.weekday, WEEKDAY_ORDER, ordered=True)
        by_weekday = by_weekday.sort_values("weekday").set_index("weekday")
        st.bar_chart(by_weekday, y="absolute_error", x_label="Weekday", y_label="MAE (MW)")
else:
    st.info("At least 35 days of hourly data are needed for the comparison panel.")

st.subheader("Historical anomaly explorer")
st.info("Anomalies are statistical flags from forecast residuals, not confirmed real-world events.")
anomalies = recent_anomalies(service, cache_key)
if anomalies.empty:
    st.info("Not enough history to score the trailing week; showing observed demand only.")
    st.line_chart(history.tail(168).set_index("timestamp")["demand_mw"])
else:
    anomaly_fig = go.Figure()
    anomaly_fig.add_trace(
        go.Scatter(
            x=anomalies.timestamp, y=anomalies.expected_demand_mw + anomalies.upper_residual_bound,
            name="Upper bound", line=dict(width=0), showlegend=False,
        )
    )
    anomaly_fig.add_trace(
        go.Scatter(
            x=anomalies.timestamp, y=anomalies.expected_demand_mw + anomalies.lower_residual_bound,
            name="Expected ± residual bounds", fill="tonexty", line=dict(width=0),
        )
    )
    anomaly_fig.add_trace(
        go.Scatter(
            x=anomalies.timestamp, y=anomalies.expected_demand_mw,
            name="Expected", line=dict(dash="dash"),
        )
    )
    anomaly_fig.add_trace(
        go.Scatter(x=anomalies.timestamp, y=anomalies.demand_mw, name="Observed")
    )
    flagged = anomalies[anomalies.is_anomaly]
    if not flagged.empty:
        anomaly_fig.add_trace(
            go.Scatter(
                x=flagged.timestamp, y=flagged.demand_mw, name="Anomaly", mode="markers",
                marker=dict(symbol="x", size=11, color="crimson"),
            )
        )
    anomaly_fig.update_layout(
        yaxis_title="Demand (MW)", xaxis_title="Time", hovermode="x unified"
    )
    st.plotly_chart(anomaly_fig, use_container_width=True)
    if flagged.empty:
        st.caption("No hour in the trailing week left the 99% residual bounds.")
    else:
        st.caption(f"{len(flagged)} of {len(anomalies)} hours flagged (99% residual bounds).")
        st.dataframe(
            flagged[["timestamp", "demand_mw", "expected_demand_mw", "deviation_mw"]].round(1),
            hide_index=True,
        )

with st.expander("Data source and limitations"):
    metadata_path = Path("data/clean/metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        st.write(
            f"**{metadata['source']}** (CC BY 4.0), module {metadata['module_id']} — Germany "
            f"actual total grid load. Snapshot of {metadata['rows']:,} hourly rows from "
            f"{metadata['first_timestamp']} to {metadata['last_timestamp']}, collected "
            f"{metadata['collected_at']} from {len(metadata['chunk_urls'])} weekly chunks. "
            "SMARD may revise historical values; results should cite this snapshot."
        )
    else:
        st.write(
            "This view uses deterministic synthetic demand because data/clean/demand_hourly.csv "
            "is absent. Run `python scripts/ingest_smard_api.py --weeks 52` to collect the "
            "official SMARD snapshot."
        )
    st.write(
        "Prediction intervals and anomaly bounds are empirical, based on historical validation "
        "residual magnitudes; they are not calibrated probabilistic forecasts, and anomalies are "
        "investigation prompts rather than confirmed events."
    )
