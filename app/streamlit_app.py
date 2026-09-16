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

from energy_forecast import drift
from energy_forecast.events import HighDemandClassifier, daily_feature_frame
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
    SURFACE,
    plotly_layout,
)
from energy_forecast.verification import (
    DEFAULT_LOG_PATH,
    daily_scores,
    join_log_to_actuals,
    load_forecast_log,
)

GBM = MODEL_COLORS["HistGradientBoosting"]
BERLIN = "Europe/Berlin"
# One hue per drift level, from the existing palette: the champion's own blue when it is
# behaving, the "expected" orange for watch, the reserved status red for a regression.
DRIFT_COLORS = {"ok": GBM, "watch": EXPECTED, "regressed": ANOMALY, "unknown": MUTED}


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


def record_figure(
    plot: pd.DataFrame, color: str, name: str, dash: str | None = None,
    shade_until: pd.Timestamp | None = None,
) -> go.Figure:
    """Band, actuals and forecast over a local-time axis — the Track record chart grammar.

    `plot` needs `timestamp` and `local` columns plus demand_mw/prediction/lower_bound/
    upper_bound. Trace order matters: the two invisible bound traces come first so
    `fill="tonexty"` paints the band *behind* the lines rather than over them.

    `shade_until` bounds the missing-data shading at the last published hour. Without it a
    forecast for hours that simply have not happened yet reads as a reporting gap, and the
    published-log chart — whose hours are all in the future when it is first written —
    would shade solid grey. Absent demand after that instant is the future, not a hole.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot.local, y=plot.upper_bound,
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot.local, y=plot.lower_bound,
            name="Prediction interval", fill="tonexty", fillcolor=INTERVAL_FILL,
            line=dict(width=0), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot.local, y=plot.demand_mw, name="Observed",
            line=dict(color=OBSERVED, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot.local, y=plot.prediction, name=name,
            line=dict(color=color, width=2.5, dash=dash),
        )
    )
    # Same reason as the Forecast tab: an unpublished run must not read as demand collapsing.
    observable = plot if shade_until is None else plot[plot.timestamp <= shade_until]
    for start, end in unpublished_spans(observable):
        # Shape coordinates must be passed in the SAME form as the trace's x values.
        # A tz-aware trace serialises to "…T15:00:00+02:00" and plotly.js ignores the
        # offset, reading 15:00; an epoch-ms number is UTC-based and would read 13:00,
        # putting every shape one or two hours off the data it marks.
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=MUTED, opacity=0.13, line_width=0, layer="below",
        )
    fig.update_layout(
        **plotly_layout(yaxis_title="Demand (MW)", xaxis_title="Time (Europe/Berlin)")
    )
    return fig


def score_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Render `daily_scores` for display. A window with no published hour shows "—", not nan."""
    return pd.DataFrame(
        {
            "Origin (Berlin)": scores.origin.dt.tz_convert(BERLIN).dt.strftime("%a %d %b %H:%M"),
            "Hours scored": scores.hours_scored,
            "MAE (MW)": [f"{v:,.0f}" if pd.notna(v) else "—" for v in scores.mae],
            "Coverage": [f"{v:.0%}" if pd.notna(v) else "—" for v in scores.coverage],
        }
    )


def verdict_badge(level: str) -> str:
    """A drift level as a coloured pill. The word carries the state; the colour repeats it."""
    return (
        f"<span style='background:{DRIFT_COLORS.get(level, MUTED)};color:{SURFACE};"
        "padding:3px 12px;border-radius:11px;font-weight:600;font-size:0.82rem;"
        f"letter-spacing:0.06em'>{level.upper()}</span>"
    )


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


@st.cache_data(show_spinner="Replaying the last seven days...")
def hindcast_frame(_service: ForecastService, cache_key: str, days: int = 7) -> pd.DataFrame:
    # days + 1 model fits (the extra one calibrates the residual band), so seconds locally
    # and a good deal longer on Streamlit Cloud's shared CPU — worth caching hard.
    return _service.hindcast(days=days)


def file_stamp(path: Path) -> str:
    """Cheap identity for a committed data file. Deliberately *not* cached — it is the
    thing that decides whether a cache entry is stale, so it has to be read every run."""
    try:
        stat = path.stat()
    except OSError:
        return "missing"
    return f"{stat.st_size}:{stat.st_mtime_ns}"


@st.cache_data(show_spinner="Scoring the published forecast log...")
def published_forecasts(_service: ForecastService, cache_key: str, log_key: str) -> pd.DataFrame:
    # `log_key` is never read here: it is in the signature purely so that a commit from the
    # forecast-logging workflow invalidates this join. `cache_key` alone would not — the log
    # gains a new origin without the demand snapshot changing at all.
    return join_log_to_actuals(load_forecast_log(DEFAULT_LOG_PATH), _service.history)


@st.cache_data(show_spinner="Reading the model-health history...")
def drift_history(runs_key: str, scores_key: str):
    """Retained run summaries and per-origin scores, or None when nothing is recorded yet.

    No `_service` argument, unlike `published_forecasts`: these are two plain file reads
    of artifacts the benchmark commits, and the demand snapshot cannot change them, so the
    two file stamps are the whole cache key.
    """
    scores = drift.load_origin_history(drift.ORIGIN_HISTORY_PATH)
    if scores.empty:
        return None
    return drift.load_run_history(drift.RUN_HISTORY_PATH), scores


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
    "published well after the fact. A lag from several hours up to about a day is normal, "
    "so this figure is rarely close to zero.",
)
col3.metric("Data source", service.data_source)

if service.live_warning:
    st.warning(service.live_warning)

forecast_tab, record_tab, quality_tab, anomaly_tab, event_tab, about_tab = st.tabs(
    ["Forecast", "Track record", "Model quality", "Anomalies", "Event risk", "Data & methods"]
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
            x0=start, x1=end,
            fillcolor=MUTED, opacity=0.13, line_width=0, layer="below",
        )
    if gaps:
        mid = gaps[0][0] + (gaps[0][1] - gaps[0][0]) / 2
        fig.add_annotation(
            x=mid, yref="paper", y=0.5, text="not yet<br>published",
            showarrow=False, font=dict(size=10, color=MUTED),
        )
    # Pass the same tz-aware form the traces use. plotly.js ignores the offset in a date
    # string, so a trace point reads as its Berlin wall-clock time; an epoch-ms number is
    # UTC-based and would land this marker an hour or two to the left of the data.
    fig.add_vline(
        x=last_observed_ts.tz_convert(BERLIN),
        line=dict(color=MUTED, width=1, dash="dot"),
    )
    fig.update_layout(
        **plotly_layout(yaxis_title="Demand (MW)", xaxis_title="Time (Europe/Berlin)")
    )
    st.plotly_chart(fig, width="stretch")
    gap_note = (
        f" The {len(gaps)} shaded band{'s' if len(gaps) > 1 else ''} mark hours SMARD has "
        "indexed but not yet published."
        if gaps else ""
    )
    st.caption(
        "Observed demand for the last three days, then the 24-hour forecast; the dotted line "
        "marks the forecast start and the shaded band is the model's recent error magnitude."
        + gap_note
    )


# --------------------------------------------------------------------------------------
# Track record tab
# --------------------------------------------------------------------------------------
with record_tab:
    st.subheader("Last seven days, replayed")
    replay = hindcast_frame(service, cache_key)
    if replay.empty:
        st.info(
            "Replaying a week needs roughly 30 days of hourly history: seven daily origins, a "
            "validation week before the first of them, and enough rows left over to fit a model. "
            "This snapshot is shorter than that."
        )
    else:
        replay_plot = replay.sort_values("timestamp").reset_index(drop=True).copy()
        replay_plot["local"] = replay_plot.timestamp.dt.tz_convert(BERLIN)
        replay_fig = record_figure(
            replay_plot, EXPECTED, "Hindcast", dash="dash", shade_until=service.last_observed
        )
        # The seven windows are contiguous, so the lines are continuous; the dotted marks are
        # where one forecast ended and the next model was refit. Same tz-aware form as the
        # traces — see record_figure on why an epoch-ms shape misplaces itself.
        for origin in sorted(replay.origin.unique()):
            replay_fig.add_vline(
                x=pd.Timestamp(origin).tz_convert(BERLIN),
                line=dict(color=MUTED, width=1, dash="dot"),
            )
        st.plotly_chart(replay_fig, width="stretch")

        replay_scores = daily_scores(replay)
        st.dataframe(score_table(replay_scores), hide_index=True, width="stretch")

        scored_hours = int(replay_scores.hours_scored.sum())
        if scored_hours == 0:
            st.markdown(
                "**No hour in the replayed week has been published yet**, so none of these "
                "windows can be scored."
            )
        else:
            mean_mae = float(replay_scores.mae.mean())
            bench_mae = None
            if benchmark is not None:
                bench_mae = (
                    benchmark[0].get("models", {}).get("HistGradientBoosting", {}).get("mae_mean")
                )
            line = (
                f"**Mean MAE across the seven days: {mean_mae:,.0f} MW** "
                f"({scored_hours} of {len(replay)} hours scored)."
            )
            if bench_mae:
                direction = "better" if mean_mae < bench_mae else "worse"
                line += (
                    f" This week ran {direction} than the year-long backtest average of "
                    f"{bench_mae:,.0f} MW."
                )
            st.markdown(line)

        st.caption(
            "Seven separate 24-hour forecasts, one per day, each refit on data dated at or "
            "before its own origin."
        )

    st.divider()
    st.subheader("Published forecasts vs what happened")
    log_key = file_stamp(DEFAULT_LOG_PATH)
    published = published_forecasts(service, cache_key, log_key)
    published_scores = daily_scores(published)
    scored_origins = (
        0 if published_scores.empty else int((published_scores.hours_scored > 0).sum())
    )
    if published.empty:
        st.info(
            "Nothing logged yet. A daily GitHub Action issues one 24-hour forecast from the last "
            "published hour and commits it to `data/forecasts/forecast_log.csv`. The first "
            "comparison appears once those forecast hours have themselves been published by "
            "SMARD, which lags by several hours up to about a day."
        )
    else:
        # Consecutive logged origins are not exactly 24h apart: SMARD's publication lag moves,
        # so the origin the workflow forecasts from moves with it. Two consequences.
        # (1) Windows overlap. On an overlapping hour, plot the most recent origin's value —
        #     the freshest forecast for that hour — but score every logged origin in the table
        #     below, because each one was a separate published claim.
        # (2) When the lag *grew*, no origin covers some hours at all. Reindexing onto a
        #     continuous hourly grid leaves those as NaN so the forecast line breaks there.
        #     Do not bridge it: a hole in the log is not a forecast.
        latest_per_hour = (
            published.sort_values(["timestamp", "origin"])
            .drop_duplicates("timestamp", keep="last")
        )
        grid = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    published.timestamp.min(), published.timestamp.max(), freq="1h", tz="UTC"
                )
            }
        )
        published_plot = grid.merge(
            latest_per_hour[["timestamp", "prediction", "lower_bound", "upper_bound"]],
            on="timestamp", how="left",
        ).merge(
            # Actuals off the full history, not off the join: a grid hour with no logged
            # forecast still has an observation, and the observed line should keep going.
            history[["timestamp", "demand_mw"]], on="timestamp", how="left",
        )
        published_plot["local"] = published_plot.timestamp.dt.tz_convert(BERLIN)
        st.plotly_chart(
            record_figure(
                published_plot, FORECAST, "Published forecast",
                shade_until=service.last_observed,
            ),
            width="stretch",
        )
        st.dataframe(score_table(published_scores), hide_index=True, width="stretch")

        if scored_origins == 0:
            st.caption(
                f"{len(published_scores)} forecast{'s' if len(published_scores) != 1 else ''} "
                "logged, none scorable yet. SMARD has not published any of the hours they cover."
            )
        elif scored_origins < 3:
            st.caption(
                f"Only {scored_origins} logged forecast"
                f"{'s have' if scored_origins != 1 else ' has'} a published hour to score "
                "against, so read it as a smoke test."
            )
        else:
            mean_mae = float(published_scores.mae.mean())
            st.markdown(
                f"**Mean MAE across {scored_origins} published forecasts: {mean_mae:,.0f} MW.**"
            )

        st.caption(
            "Forecasts the daily workflow committed to git before the outcome was known, "
            "against what SMARD has since published."
        )

    st.caption(
        "For how these numbers have moved run-over-run, see Model health on the Model "
        "quality tab."
    )


# --------------------------------------------------------------------------------------
# Model quality tab
# --------------------------------------------------------------------------------------
with quality_tab:
    # Model health first: this tab already owns the artifacts the history is derived from,
    # and "is it still as good?" is only worth asking above "is it good?".
    st.subheader("Model health")
    drift_state = drift_history(
        file_stamp(drift.RUN_HISTORY_PATH), file_stamp(drift.ORIGIN_HISTORY_PATH)
    )
    if drift_state is None:
        st.info(
            "No run history yet. `scripts/benchmark.py` appends one row per complete run; the "
            "weekly refresh writes the first. Seed it now from the artifacts already in "
            "`reports/` with `PYTHONPATH=src python scripts/benchmark.py --seed-history`."
        )
        # Never leave the panel blank: the replayed week is a reading the app can always
        # take, and `live_verdict` caps it at "watch" because 7 origins is too few to condemn.
        live = drift.live_verdict(
            daily_scores(hindcast_frame(service, cache_key)),
            pd.DataFrame(columns=drift.ORIGIN_COLUMNS),
        )
        st.markdown(verdict_badge(live.level), unsafe_allow_html=True)
        st.markdown(live.headline)
        st.caption(
            "Provisional, from the last 7 replayed days and capped at *watch*: a stand-in "
            "for the weekly verdict until the run history exists."
        )
    else:
        runs, scores = drift_state
        runs = runs.sort_values(["run_at", "snapshot_end"]).reset_index(drop=True)
        # Uncached and inline, unlike the file reads above: a percentile over a few thousand
        # rows is sub-millisecond, and keeping a frozen dataclass out of Streamlit's
        # return-value pickling avoids a class of hashing surprise.
        verdict = drift.regression_verdict(scores, runs)
        coverage = drift.coverage_drift(runs)
        window = drift.recent_window(scores)

        status = st.columns(3)
        status[0].markdown(verdict_badge(verdict.level), unsafe_allow_html=True)
        status[0].markdown(verdict.headline)
        status[0].caption(verdict.reason)

        window_mae = verdict.window_mae
        gap = window_mae - verdict.reference_median
        status[1].metric(
            f"Average error, last {drift.RECENT_ORIGINS} days",
            f"{window_mae:,.0f} MW" if pd.notna(window_mae) else "—",
            delta=f"{gap:+,.0f} MW vs similar days" if pd.notna(gap) else None,
            # Lower error is better, so a positive delta must not read as good news.
            delta_color="inverse",
            help=f"How far off the forecast was on average over the {verdict.window_origins} "
            "most recent days scored, next to the typical miss on similar days from "
            "previous years.",
        )

        coverage_help = (
            "How often actual demand landed inside the forecast's shaded range in the latest "
            "run. Judged against what this model has actually been achieving, not the 95% "
            "label it aims at. The range is known to be slightly too narrow."
        )
        if coverage["level"] == "unknown":
            status[2].metric("Actuals inside the range", "—", help=coverage_help)
        else:
            status[2].metric(
                "Actuals inside the range",
                f"{coverage['mean']:.1%}",
                delta=f"{-100 * coverage['drop']:+.1f} pp vs baseline",
                help=coverage_help,
            )
        status[2].caption(coverage["headline"])

        if len(window):
            centre = window.origin.iloc[len(window) // 2]
            reference = drift.seasonal_reference(
                scores, centre, exclude_from=window.origin.min()
            )
        else:
            reference = scores.iloc[:0]

        # A window scored by a different model generation is not comparable to a reference
        # scored by the old one; say so rather than letting the percentile imply otherwise.
        modal_hash = reference.params_hash.mode() if len(reference) else pd.Series(dtype=object)
        window_hashes = set(window.params_hash.dropna().unique()) if len(window) else set()
        if len(modal_hash) and window_hashes and window_hashes != {modal_hash.iloc[0]}:
            st.warning(
                "The recent origins were scored by a different model generation "
                f"(`{'`, `'.join(sorted(window_hashes))}`) than most of the reference days "
                f"(`{modal_hash.iloc[0]}`). The comparison is indicative until the new "
                "generation has a season of history of its own."
            )

        st.markdown("**How recent error compares with the same time of year**")
        champion = scores[scores.model == drift.CHAMPION].sort_values("origin")
        champion_local = champion.origin.dt.tz_convert(BERLIN)
        health_fig = go.Figure()
        health_fig.add_trace(
            go.Scatter(
                x=champion_local, y=champion.mae, name="Daily error", mode="markers",
                marker=dict(color=MUTED, size=4, opacity=0.32),
            )
        )
        health_fig.add_trace(
            go.Scatter(
                x=champion_local,
                y=champion.mae.rolling(
                    drift.RECENT_ORIGINS, min_periods=drift.RECENT_ORIGINS
                ).mean(),
                name=f"{drift.RECENT_ORIGINS}-day average",
                line=dict(color=GBM, width=2),
            )
        )
        if len(reference) >= 2:
            low, high = reference.mae.quantile([0.25, 0.75])
            # layer="below" so the per-origin marks stay readable on top of the band.
            health_fig.add_hrect(
                y0=low, y1=high, fillcolor=INTERVAL_FILL, line_width=0, layer="below",
                annotation_text="typical for this time of year "
                f"(±{reference.attrs.get('window_days', 0)} days)",
                annotation_position="top left",
                annotation_font=dict(size=11, color=MUTED),
            )
        if len(window):
            # Pass the tz-aware local Timestamps, not epoch-ms: they serialize exactly like
            # the trace's x-values above, so the band cannot land two hours off the marks.
            health_fig.add_vrect(
                x0=window.origin.min().tz_convert(BERLIN),
                x1=window.origin.max().tz_convert(BERLIN),
                fillcolor=DRIFT_COLORS.get(verdict.level, MUTED), opacity=0.13,
                line_width=0, layer="below",
            )
        health_fig.update_layout(
            **plotly_layout(
                yaxis_title="Average error (MW)",
                xaxis_title="Day (Europe/Berlin)",
            )
        )
        st.plotly_chart(health_fig, width="stretch")
        st.caption(
            "Each dot is one day's average error; the line smooths it over "
            f"{drift.RECENT_ORIGINS} days. The grey band is where similar days from previous "
            "years usually land, and the tinted column is the stretch being judged above."
        )

    st.divider()
    st.subheader("Rolling-origin backtest")
    if benchmark is None:
        st.info(
            "Run `PYTHONPATH=src python scripts/benchmark.py` to generate the full backtest "
            "(`reports/benchmark_summary.json` + `reports/benchmark_metrics.csv`)."
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
            f"**{summary['champion']}** has the lowest error, with MAE "
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
                f"closest rival (Wilcoxon p = {closest['wilcoxon_p']:.1e}). The gap is not "
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
                xaxis_title="Origin (14-origin trailing mean)",
            )
        )
        st.plotly_chart(origin_fig, width="stretch")
        st.caption(
            "Each line is a 14-origin trailing mean of that model's 24-hour MAE. "
            "HistGradientBoosting stays below both baselines across the whole year."
        )

        importance_png = REPORTS / "figures" / "benchmark_feature_importance.png"
        slices_png = REPORTS / "figures" / "benchmark_error_slices.png"
        cols = st.columns(2)
        if importance_png.exists():
            cols[0].image(str(importance_png), caption="Permutation importance (validation week)")
        if slices_png.exists():
            cols[1].image(str(slices_png), caption=f"Where {summary['champion']} errs")


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
                f"{len(flagged)} of {len(anomalies)} hours flagged (99% residual bounds)."
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
        "asks whether the coming day lands in the **top 30% of the trailing month's peaks**, "
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
            "probability is only meaningful next to the base rate beside it: a 30% forecast is "
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
        for block in classification.values():
            st.markdown(f"**{block['label']}**: {block['scored_days']:,} days scored")
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
        st.caption(
            "PR-AUC is read against the base rate, not against 0.5. The lift column is the "
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
            f"**{metadata['source']}** (CC BY 4.0), module {metadata['module_id']}: Germany "
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
                    f"Data-quality gate passed: all {len(quality['checks'])} checks clean "
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
                    + "; ".join(f"**{c['name']}**: {c['detail']}" for c in failed)
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
        "magnitudes only, not calibrated probabilistic forecasts, and anomalies are prompts "
        "to investigate weather, calendar, or grid context rather than confirmed events.\n"
        "- **Tested claims.** The champion's margin over each rival carries a paired bootstrap "
        "CI and a Wilcoxon/Diebold-Mariano p-value, computed per origin because hours inside "
        "one 24-hour window are correlated.\n"
        "- **SQL reporting layer.** EDA aggregations and error slices run against a DuckDB "
        "warehouse (`fact_demand`, `dim_calendar`, `fact_forecast`) built from the same "
        "snapshot; each query is tested against its pandas equivalent."
    )

    st.subheader("How to read these charts")
    st.markdown(
        "- **The forecast band.** ± the 95th percentile of absolute residuals the model made on "
        "a held-out validation week. Its width reflects how wrong the model has recently been. It "
        "is an empirical error magnitude, not a calibrated probability and not a coverage "
        "guarantee.\n"
        "- **Unpublished hours.** SMARD indexes each hour before publishing its value, so the "
        "most recent hours often carry no reading. They appear as gaps or shaded bands and are "
        "reported rather than imputed, and excluded from the scores rather than filled in, "
        "which is why some 24-hour windows show fewer than 24 hours scored.\n"
        "- **Replay versus published log.** The replayed week refits a model at each origin on "
        "rows dated at or before that origin, so no future value can reach it. But it is "
        "computed after the fact, with today's data, so it shows what the model *would have* "
        "said, not what it did say. The published log holds rows written before the outcome was "
        "known and never revised; a re-issued forecast for the same hour replaces the older one, "
        "and nothing else does. That is the one thing a replay cannot manufacture. Until about "
        "three origins have a published hour to score against, read the log as a smoke test: a "
        "day or two is an anecdote, not evidence.\n"
        "- **Scores can move.** Actuals are SMARD's readings *as currently published*, and SMARD "
        "does revise history, so a score already shown here can shift slightly after the fact.\n"
        "- **Model health.** The reference is matched by day of the year, starting at ±21 days "
        "and widening only until it holds enough origins, because January MAE is 2.10× August "
        "with no drift at all, and a pooled comparison would flag every winter forever. The "
        "statistic is a percentile rather than a z-score because the error distribution is "
        "right-skewed (mean 1,944 MW, median 1,548 MW, max 10,370 MW). A single window above the "
        "threshold is only ever *watch*; **regressed** needs two consecutive runs, which chance "
        "alone produces about once in 2,000. Coverage is judged against the coverage this model "
        "has established, not the 95% label, because the band already under-covers by design and "
        "a monitor that is always red is one nobody reads. Nothing here fails a workflow: the "
        "weekly refresh keeps committing, and that panel is where drift surfaces.\n"
        "- **Anomaly bounds.** The 99% residual bounds are learned from the validation week "
        "*before* the window being scored, so they never see the data they judge. A flag is a "
        "prompt to investigate weather, calendar or grid context, not a confirmed event."
    )
    if benchmark is not None:
        st.caption(
            "The full backtest report lives in `reports/benchmark.md`; the EDA in `reports/eda.md`."
        )
