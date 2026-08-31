"""Streamlit product interface; run with `streamlit run app/streamlit_app.py`."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from energy_forecast.evaluation import metrics
from energy_forecast.models import HistGradientBoostingForecast, SeasonalNaive
from energy_forecast.service import ForecastService

st.set_page_config(page_title="Stromblick", layout="wide")
st.title("Stromblick")
st.caption("Germany electricity demand forecasting and anomaly detection")

service = ForecastService()
history = service.history
result = service.forecast(history.timestamp.max(), 24)
forecast = pd.DataFrame(result["forecast"])
forecast["timestamp"] = pd.to_datetime(forecast.timestamp, utc=True)

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
    train, test = history.iloc[:-24 * 7], history.iloc[-24 * 7:]
    rows = []
    candidates = [
        ("Seasonal naive", SeasonalNaive()),
        ("HistGradientBoosting", HistGradientBoostingForecast()),
    ]
    for name, model in candidates:
        model.fit(train)
        row = {"model": name}
        predicted = model.predict(pd.DatetimeIndex(test.timestamp))
        row.update(metrics(test.demand_mw.to_numpy(), predicted))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
else:
    st.info("At least 35 days of hourly data are needed for the comparison panel.")

st.subheader("Historical anomaly explorer")
st.info("Anomalies are statistical flags from forecast residuals, not confirmed real-world events.")
st.line_chart(history.tail(168).set_index("timestamp")["demand_mw"])

with st.expander("Data source and limitations"):
    st.write(
        "The intended source is Germany's official SMARD electricity-market data. "
        "Run the ingestion command with a verified SMARD export URL before production use."
    )
    st.write(
        "The demo view uses deterministic synthetic demand when data/clean/demand_hourly.csv "
        "is absent. Prediction intervals are residual-based and should not be interpreted "
        "as causal uncertainty."
    )
