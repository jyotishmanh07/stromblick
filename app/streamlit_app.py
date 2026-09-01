"""Streamlit product interface; run with `streamlit run app/streamlit_app.py`."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import energy_forecast from the repo checkout rather than site-packages.
# Streamlit Community Cloud re-reads this file from git on every pull but only
# reinstalls dependencies when requirements.txt *changes* — and requirements.txt is
# just ".", which pip *copies* into site-packages. So a pull that adds a new module
# leaves the app running new UI code against a stale installed package, and the import
# fails until someone reboots. Pointing at src/ keeps the two in lockstep.
# Harmless elsewhere: a missing src/ is ignored and the installed package is used.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_forecast.evaluation import error_slices, metrics
from energy_forecast.events import HighDemandClassifier, daily_feature_frame
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
BERLIN = "Europe/Berlin"


def unpublished_spans(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of hours with no published value, as (start, end) in local time.

    Returned spans are widened by half an hour on each side so a single missing hour is
    still visible as a band rather than a hairline.
    """
    missing = frame[frame.demand_mw.isna()]
    if missing.empty:
        return []
    local = missing.local.reset_index(drop=True)
    breaks = local.diff() > pd.Timedelta(hours=1)
    spans = []
    for _, run in local.groupby(breaks.cumsum()):
        spans.append(
            (run.iloc[0] - pd.Timedelta(minutes=30), run.iloc[-1] + pd.Timedelta(minutes=30))
        )
    return spans

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
REPORTS = Path("reports")

st.set_page_config(page_title="Stromblick", page_icon="⚡", layout="wide")
st.title("Stromblick")
st.caption("Germany electricity demand forecasting and anomaly detection")


# --------------------------------------------------------------------------------------
# Cached computation
# --------------------------------------------------------------------------------------
@st.cache_resource(ttl=3600, show_spinner="Fetching the latest SMARD data and fitting the model...")
def get_service() -> ForecastService:
    # TTL, not permanence: the service holds the fetched history, so an expiry is what
    # lets a new SMARD publication reach the page at all.
    return ForecastService(live=True)


@st.cache_data(show_spinner="Forecasting the next 24 hours...")
def forecast_frame(_service: ForecastService, cache_key: str) -> pd.DataFrame:
    # Forecast from the last *published* hour. SMARD indexes hours before publishing
    # them, so timestamp.max() can sit past the end of the observed series.
    origin = _service.last_observed or _service.history.timestamp.max()
    result = _service.forecast(origin, 24)
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


@st.cache_data
def load_classification():
    """Daily event-classification artifacts from scripts/benchmark_classification.py."""
    summary_path = REPORTS / "classification_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


@st.cache_data(show_spinner="Scoring tomorrow's event risk...")
def event_risk(history: pd.DataFrame, cache_key: str) -> dict[str, object] | None:
    """Fit the daily classifiers on all history and score the most recent labelled day."""
    daily = daily_feature_frame(history)
    if len(daily) < 60:
        return None
    train, latest = daily.iloc[:-1], daily.iloc[[-1]]
    if train.label.nunique() < 2:
        return None
    return {
        "date": latest.date.iloc[0],
        "probability": float(HighDemandClassifier().fit(train).predict_proba(latest)[0]),
        "baseline_rate": float(train.label.mean()),
        "peak_mw": float(latest.peak_mw.iloc[0]),
        "threshold_mw": float(latest.threshold_mw.iloc[0]),
        "labelled_days": int(len(daily)),
    }


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
service = get_service()
history = service.history
# The non-null count matters: SMARD backfills previously unpublished hours without
# changing the row count or the range, and that must still invalidate downstream caches.
cache_key = (
    f"{service.data_source}:{len(history)}:"
    f"{int(history.demand_mw.notna().sum())}:{history.timestamp.max()}"
)
forecast = forecast_frame(service, cache_key)
benchmark = load_benchmark()

latest = service.last_observed
age_hours = (
    (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 3600 if latest is not None else None
)

col1, col2, col3 = st.columns(3)
col1.metric("Latest observed demand", f"{history.demand_mw.dropna().iloc[-1]:,.0f} MW")
col2.metric(
    "Data freshness",
    "unknown" if age_hours is None else f"{age_hours:,.0f}h ago",
    help="Hours since the most recent published SMARD observation. Actual grid load is "
    "published well after the fact — a lag from several hours up to about a day is normal, "
    "so this figure is rarely close to zero.",
)
col3.metric("Data source", service.data_source)

if service.live_warning:
    st.warning(service.live_warning)

forecast_tab, quality_tab, anomaly_tab, event_tab, about_tab = st.tabs(
    ["Forecast", "Model quality", "Anomalies", "Event risk", "Data & methods"]
)


# --------------------------------------------------------------------------------------
# Forecast tab
# --------------------------------------------------------------------------------------
with forecast_tab:
    st.subheader("Next 24 hours")
    observed = history.tail(72).copy()  # keep NaN rows so unpublished hours render as gaps
    last_observed_ts = history.dropna(subset=["demand_mw"]).timestamp.iloc[-1]
    # Demand follows the German working day, so plot in local time: a trough labelled
    # 03:00 UTC is really 05:00 in Berlin, which makes the daily shape read wrong.
    observed["local"] = observed.timestamp.dt.tz_convert(BERLIN)
    forecast_local = forecast.timestamp.dt.tz_convert(BERLIN)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_local, y=forecast.upper_bound,
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_local, y=forecast.lower_bound,
            name="Prediction interval", fill="tonexty", fillcolor=INTERVAL_FILL,
            line=dict(width=0), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=observed.local, y=observed.demand_mw, name="Observed",
            line=dict(color=OBSERVED, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_local, y=forecast.prediction, name="Forecast",
            line=dict(color=FORECAST, width=2.5),
        )
    )
    # Shade runs of unpublished hours. Without this a reporting gap looks identical to
    # demand collapsing to zero — the line simply stops and resumes at a different level.
    gaps = unpublished_spans(observed)
    for start, end in gaps:
        fig.add_vrect(
            x0=start.timestamp() * 1000, x1=end.timestamp() * 1000,
            fillcolor=MUTED, opacity=0.13, line_width=0, layer="below",
        )
    if gaps:
        mid = gaps[0][0] + (gaps[0][1] - gaps[0][0]) / 2
        fig.add_annotation(
            x=mid.timestamp() * 1000, yref="paper", y=0.5, text="not yet<br>published",
            showarrow=False, font=dict(size=10, color=MUTED),
        )
    # epoch-ms is the form plotly reliably accepts for a vline on a datetime axis
    fig.add_vline(
        x=last_observed_ts.tz_convert(BERLIN).timestamp() * 1000,
        line=dict(color=MUTED, width=1, dash="dot"),
    )
    fig.update_layout(
        **plotly_layout(yaxis_title="Demand (MW)", xaxis_title="Time (Europe/Berlin)")
    )
    st.plotly_chart(fig, width="stretch")
    gap_note = (
        f" The {len(gaps)} shaded band{'s' if len(gaps) > 1 else ''} mark hours SMARD has "
        "indexed but not yet published — missing readings, not a drop in demand. They are "
        "reported rather than imputed."
        if gaps else
        " Gaps in the observed line would mark hours SMARD has indexed but not yet published; "
        "there are none in this window."
    )
    st.caption(
        "Observed demand for the last three days, then the 24-hour forecast (the dotted line "
        "marks the forecast start). Times are Europe/Berlin, the timezone demand actually "
        "follows. The shaded band around the forecast is ± the 95th percentile of absolute "
        "residuals the model made on a held-out validation week — its width reflects how wrong "
        "the model has recently been, not a probabilistic guarantee." + gap_note
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

        tests = summary.get("significance")
        if tests:
            closest = max(tests["comparisons"].values(), key=lambda s: s["mean_diff"])
            cov_test = tests["coverage"]
            st.markdown(
                f"Paired bootstrap over {closest['n']} origins puts that advantage at "
                f"**[{-closest['ci_upper']:,.0f}, {-closest['ci_lower']:,.0f}] MW** against the "
                f"closest rival (Wilcoxon p = {closest['wilcoxon_p']:.1e}) — the gap is not "
                "sampling noise. Mean per-origin interval coverage is "
                f"{100 * cov_test['mean_coverage']:.1f}% "
                f"(95% CI [{100 * cov_test['ci_lower']:.1f}%, {100 * cov_test['ci_upper']:.1f}%]) "
                f"against the {100 * cov_test['nominal']:.0f}% target."
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
# Event risk — the classification track
# --------------------------------------------------------------------------------------
with event_tab:
    st.subheader("Is tomorrow a high-demand day?")
    st.markdown(
        "A second problem type on the same data. Instead of *how much* demand each hour, this "
        "asks whether the coming day lands in the **top 30% of the trailing month's peaks** — "
        "the shape a reserve or staffing decision takes. The threshold is computed from earlier "
        "days only and shifted, so the rule that judges a day is fixed before the day is seen."
    )
    risk = event_risk(history, cache_key)
    if risk is None:
        st.info("Not enough labelled history to score event risk.")
    else:
        left, mid, right = st.columns(3)
        left.metric("Probability (most recent day)", f"{risk['probability']:.0%}")
        mid.metric("Recent positive rate", f"{risk['baseline_rate']:.0%}")
        right.metric("Threshold to clear", f"{risk['threshold_mw']:,.0f} MW")
        st.caption(
            f"Scored for {risk['date']} against {risk['labelled_days']:,} labelled days. The "
            "probability is only meaningful next to the base rate beside it — a 30% forecast is "
            "high when the base rate is 12% and unremarkable when it is 30%."
        )

    classification = load_classification()
    if classification is None:
        st.info(
            "Run `PYTHONPATH=src python scripts/benchmark_classification.py` for the "
            "chronological backtest of both event targets."
        )
    else:
        st.divider()
        st.subheader("Chronological backtest")
        for key, block in classification.items():
            st.markdown(f"**{block['label']}** — {block['scored_days']:,} days scored")
            table = pd.DataFrame(
                [
                    {
                        "Model": name,
                        "PR-AUC": f"{m['pr_auc']:.3f}",
                        "Lift over base rate": f"{m['pr_auc_lift']:.2f}×",
                        "ROC-AUC": f"{m['roc_auc']:.3f}",
                        "Brier": f"{m['brier']:.4f}",
                    }
                    for name, m in block["models"].items()
                ]
            )
            st.dataframe(table, hide_index=True, width="stretch")
            for figure in (REPORTS / "figures" / f"classification_pr_{key}.png",):
                if figure.exists():
                    st.image(str(figure), width="stretch")
        st.caption(
            "PR-AUC is read against the base rate, not against 0.5 — the lift column is the "
            "honest version. On the anomaly target the calendar-only baseline nearly matches "
            "the gradient-boosted model, which says those days are mostly a calendar "
            "phenomenon (holidays and DST) rather than a demand-dynamics one."
        )


# --------------------------------------------------------------------------------------
# Data & methods tab
# --------------------------------------------------------------------------------------
with about_tab:
    st.subheader("Data snapshot")
    metadata_path = Path("data/clean/metadata.json")
    # Live provenance when the fetch succeeded, otherwise the committed snapshot's —
    # one render path, so the panel can never describe data the app isn't using.
    metadata = service.provenance
    if metadata is None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    if metadata is not None:
        origin = (
            f"Refreshed live from the SMARD API at `{metadata['collected_at']}` "
            f"({metadata.get('weeks_fetched', '?')} weekly chunks merged onto the committed "
            "snapshot)."
            if service.is_live else
            f"Committed snapshot, collected `{metadata['collected_at']}` from "
            f"{len(metadata['chunk_urls'])} weekly chunks."
        )
        st.markdown(
            f"**{metadata['source']}** (CC BY 4.0), module {metadata['module_id']} — Germany "
            f"actual total grid load. **{metadata['rows']:,} hourly rows** from "
            f"`{metadata['first_timestamp']}` to `{metadata['last_timestamp']}`. {origin} "
            "SMARD may revise historical values; published results should cite the snapshot "
            "recorded in `data/clean/metadata.json`, not the live feed."
        )
        st.caption(
            "SMARD indexes each hour before publishing its value, so the most recent hours "
            "usually carry no reading yet. The forecast therefore starts from the last "
            "*published* hour, which typically trails wall-clock time by several hours and "
            "sometimes by most of a day. Those unpublished hours appear as gaps in the "
            "observed line rather than being imputed."
        )
        quality = metadata.get("quality")
        if quality:
            failed = [c for c in quality["checks"] if not c["passed"]]
            if not failed:
                st.success(
                    f"Data-quality gate passed — all {len(quality['checks'])} checks clean "
                    "(schema, uniqueness, plausible range, continuity, unpublished hours). "
                    + (
                        "Live data is re-validated on every refresh; it is only adopted if the "
                        "gate passes." if service.is_live
                        else "Checked at ingest time."
                    )
                )
            else:
                st.warning(
                    "Data-quality gate flagged: "
                    + "; ".join(f"**{c['name']}** — {c['detail']}" for c in failed)
                )
    elif service.data_source.startswith("deterministic"):
        st.markdown(
            "This view uses deterministic synthetic demand because "
            "`data/clean/demand_hourly.csv` is absent or unreadable. Run "
            "`PYTHONPATH=src python scripts/ingest_smard_api.py --weeks 52` to collect the "
            "official SMARD snapshot."
        )
    else:
        st.markdown(
            "Running on the committed snapshot, but `data/clean/metadata.json` is missing, so "
            "no provenance can be shown. Re-run "
            "`PYTHONPATH=src python scripts/ingest_smard_api.py --weeks 52` to restore it."
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
        "to investigate weather, calendar, or grid context rather than confirmed events.\n"
        "- **Tested claims.** The champion's margin over each rival carries a paired bootstrap "
        "CI and a Wilcoxon/Diebold-Mariano p-value, computed per origin because hours inside "
        "one 24-hour window are correlated.\n"
        "- **SQL reporting layer.** EDA aggregations and error slices run against a DuckDB "
        "warehouse (`fact_demand`, `dim_calendar`, `fact_forecast`) built from the same "
        "snapshot; each query is tested against its pandas equivalent."
    )
    if benchmark is not None:
        st.caption(
            "The full backtest report lives in `reports/benchmark.md`; the EDA in `reports/eda.md`."
        )
