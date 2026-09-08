# Dashboard guide

Every visual element in `app/streamlit_app.py`, tab by tab, top to bottom. Companion
to the shorter "Reading the dashboard" section in the README.

- Time-series charts use **Europe/Berlin** on the x-axis — the timezone demand actually follows, so the daily shape reads correctly. Hour-of-day bar charts use Berlin local time too. Everything is stored and computed in UTC; the conversion is for display only.
- The forecast model is `HistGradientBoostingForecast` (referred to below as HGB / the GBM).
- Intervals and anomaly bounds are **empirical residual magnitudes**, not calibrated probabilities.


## Header (shown above all tabs)

Three metric cards:

| Card | Meaning |
|---|---|
| **Latest observed demand** | Most recent non-missing hourly grid-load value in the snapshot (MW). |
| **Data freshness** | Hours since the most recent published SMARD observation. SMARD publishes actual grid load well after the fact, so a lag of several hours up to about a day is normal and this is rarely near zero. |
| **Data source** | `SMARD clean export` when `data/clean/demand_hourly.csv` is present, else `deterministic demo data`. |


## Tab 1 — Forecast

**Chart: "Next 24 hours"** — x = time (UTC), y = demand (MW).

| Element | What it is |
|---|---|
| **Black line — Observed** | Actual demand for the last 72 hours (3 days). |
| **Blue line — Forecast** | HGB's 24-hour prediction. Recursive: each hour's prediction becomes the lag input for the next hour. |
| **Shaded blue band — Prediction interval** | Forecast ± the 95th percentile of the model's absolute errors on a held-out validation week. Width = how wrong the model has been lately, not a probability guarantee. Falls back to ±0.15·std if no residuals exist. |
| **Dotted grey vertical line** | Forecast start (last observed hour). History to the left, prediction to the right. |
| **Gaps in the black line** | Hours SMARD has indexed but not yet published — shown as gaps, never interpolated. |


## Tab 2 — Track record

Two sections, and the difference between them is the whole point of the tab.

### A. Last seven days, replayed

A *hindcast*: `ForecastService.hindcast()` → `verification.hindcast()`. Seven daily origins ending at
the last published hour. At each one a **fresh model is fit on rows dated at or before that origin**
and asked for the next 24 hours, so no future value can reach the prediction.

| Element | What it is |
|---|---|
| **Dark line — Observed** | What demand actually was. |
| **Orange dashed line — Hindcast** | What the model, refit at that day's origin, predicted for each hour. |
| **Shaded band** | ± the 95th percentile of absolute residuals over the week *preceding* the replayed window — fit separately so the band never sees the hours it judges. |
| **Dotted grey vertical lines** | Each day's forecast origin. |
| **Shaded grey spans** | Runs of hours SMARD has indexed but not yet published. |

Below: a per-day table (origin, hours scored, MAE, interval coverage) and the seven-day mean MAE
compared against the year-long backtest average. Hours with no published value are **excluded from
the scores**, not imputed, which is why some days show fewer than 24 hours scored.

What this section is **not**: proof the forecast was published before the outcome. It is computed
now, with today's data. It shows what the model *would have* said.

### B. Published forecasts vs what happened

The *log*: `data/forecasts/forecast_log.csv`, appended daily by `scripts/log_forecast.py` under
`.github/workflows/log-forecast.yml` and committed to git. Each row was written at issue time,
before the outcome existed, and is never revised — a re-issue for the same hour replaces the older
row, nothing else does. That is the one thing a replay cannot manufacture.

Same chart grammar as section A, with the published forecast in blue (the same blue the Forecast tab
uses, because it is the same object). Empty until the workflow has run and SMARD has published the
forecast hours; the panel says so rather than showing an empty axis. Consecutive logged origins are
not exactly 24 hours apart because the publication lag varies, so windows can overlap (the chart
shows the most recent origin's value per hour, while the table scores every origin) or leave a hole
(shown as a gap). Actuals are as *currently* published — SMARD revises, so a score can shift.


## Tab 3 — Model quality

Two sections.

### A. Rolling-origin backtest

From `scripts/benchmark.py` artifacts (`reports/benchmark_summary.json`, `reports/benchmark_metrics.csv`).
~329 daily origins over the whole snapshot; every model refit at every origin; no random split.
If the artifacts are missing, this section is replaced by an info box and only section B shows.

1. **Caption line** — origin count, spacing (24 h), total forecast hours, date range.
2. **Metrics table** — one row per model (Seasonal naive, SARIMAX, HistGradientBoosting):
   - **MAE (MW)** with its standard deviation across origins
   - **RMSE (MW)** — penalizes large misses more
   - **sMAPE (%)** — scale-free percentage error
   - **vs seasonal-naive** — % better/worse than "same hour yesterday" (HGB ≈ −50%)
3. **Summary sentence** — champion name, its % lift over the baseline, and **interval coverage**:
   fraction of observed values that fell inside the nominal 95% band (≈93%), plus mean band width.
4. **Line chart: "MAE over the 24-hour window"** — one line per model. Each point is a
   **14-origin trailing average** of that model's 24-hour MAE, plotted against origin date.
   Point of the chart: HGB stays below both baselines across the whole year, not just on average.
   (First 13 origins blank — the rolling mean needs 14.)
5. **Two images side by side:**
   - **Left — Permutation importance (validation week):** how much validation error gets *worse*
     when each input feature is randomly shuffled. Tall bars = features the model leans on most
     (typically the 24 h and 168 h demand lags and the trailing means).
   - **Right — "Where HistGradientBoosting errs":** the champion's MAE broken down by slice
     (hour, weekday, month, holiday) over the full backtest — shows *when* errors concentrate.

### B. Trailing-week holdout

Computed live on every page load — the fast honesty check.

6. **Small table** — Seasonal naive vs HistGradientBoosting on the last 7 days: mae, rmse, smape.
7. **Bar chart (left): MAE by hour** — HGB's mean absolute error for each hour of day (Berlin local).
   Surfaces the hardest hours (usually the morning ramp and evening peak).
8. **Bar chart (right): MAE by weekday** — same error grouped Mon–Sun. Weekend vs workday usually differ.


## Tab 4 — Anomalies

- **Info banner** — statistical flags from forecast residuals, not confirmed real-world events.
- **Window selector (radio)** — score the last 7 / 14 / 28 days.

**Chart: "Historical anomaly explorer"** — x = time (UTC), y = demand (MW).

| Element | What it is |
|---|---|
| **Orange dashed line — Expected** | What a model trained *only on data before this window* expected, hour by hour. |
| **Black line — Observed** | What actually happened. |
| **Shaded band — Expected ± residual bounds** | 99th-percentile residual magnitude from the validation week *preceding* this window, so the bounds never see the data they judge. |
| **Red X markers — Anomaly** | Hours where observed demand fell outside that band. |

Below the chart:
- Caption with the count ("N of M hours flagged").
- **Table of flagged hours** — timestamp, observed demand, expected demand, deviation (MW).

If there isn't enough history to score the window, this falls back to a plain line chart of observed demand only.


## Tab 5 — Event risk

The classification track (`src/energy_forecast/events.py`), a daily binary target rather than an hourly quantity.

- **Tomorrow's high-demand probability**, shown beside the recent **base rate** and the trailing-month
  peak **threshold** in MW. The base rate is the comparison that makes the probability mean anything:
  a 30% forecast says something different when the base rate is 12% than when it is 30%.
- **Classification backtest tables** from `reports/classification_summary.json` — PR-AUC, ROC-AUC and
  lift over the base rate for both targets (high-demand day, anomaly day), against a majority-class
  floor and a calendar-only logistic baseline.
- **Precision-recall curves** — `reports/figures/classification_pr_*.png`.

If there are fewer than 60 labelled days, or the training window contains only one class, the panel
says so instead of scoring.


## Tab 6 — Data & methods

No charts — reference text.

- **Data snapshot** — provenance from `data/clean/metadata.json`: source (Bundesnetzagentur | SMARD.de,
  CC BY 4.0), module 410 (Germany actual total grid load), row count, first/last timestamp,
  collection time, weekly-chunk count. SMARD may revise history, so results should cite this snapshot.
- **Method** — five bullets: the three model levels; leakage-safe features (every lag/rolling window
  shifted before aggregation); chronological rolling-origin evaluation (never a random split);
  the honest seasonal-naive baseline; empirical residual-based intervals and anomalies.


## Where the numbers come from

| Shown in the dashboard | Produced by |
|---|---|
| Forecast line + interval band | `ForecastService.forecast()` → `forecast_with_interval()` |
| Rolling-origin table, MAE-by-origin line, the two PNGs | `scripts/benchmark.py` → `reports/benchmark_*` |
| Trailing-week table + hour/weekday bars | `model_comparison()` and `error_slices()` in the app, live |
| Seven-day replay chart + per-day table | `ForecastService.hindcast()` → `verification.hindcast()`, `daily_scores()` |
| Published-forecast chart + table | `data/forecasts/forecast_log.csv`, written by `scripts/log_forecast.py` (daily via `.github/workflows/log-forecast.yml`) |
| Anomaly chart + flagged table | `ForecastService.detect_recent_anomalies()` → `detect_anomalies()` |
| Data snapshot text | `data/clean/metadata.json` (written by `scripts/ingest_smard_api.py`) |
