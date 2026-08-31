# Stromblick

[![CI](https://github.com/jyotishmanh07/stromblick/actions/workflows/ci.yml/badge.svg)](https://github.com/jyotishmanh07/stromblick/actions/workflows/ci.yml)

**Germany Electricity Demand Forecasting & Anomaly Detection**

🔗 **Live dashboard:** _deploy on [Streamlit Community Cloud](https://share.streamlit.io) with `app/streamlit_app.py` and paste the URL here._

Stromblick forecasts Germany's electricity demand for the next 24 hours and highlights unusually high or low demand. It demonstrates data validation, time-series feature engineering, chronological evaluation, statistical reasoning, Python, an API, a dashboard, automated tests, and Docker.

## Product question

Given all demand observed up to a timestamp, what demand should we expect during each of the next 24 hours? Which observations are unusually far above or below the expected value?

The target is hourly demand in MW. Statistical anomalies are investigation prompts, not confirmed events or causal explanations.

## Method

The repository compares exactly three model levels:

1. **Seasonal naive:** same hour on the previous day, with a previous-week fallback.
2. **SARIMAX:** an interpretable statistical baseline with daily seasonality.
3. **HistGradientBoostingRegressor:** the main scikit-learn model using hour, weekday, month, German public-holiday and DST-transition flags, lagged demand (1h/24h/168h), and trailing means (24h/7d).

`rolling_origin_backtest` evaluates future windows chronologically. Model selection belongs on validation origins; a final holdout should be passed only after that choice is made. Metrics are MAE, RMSE, and sMAPE, with error slices by hour, weekday, month, and holiday status. Residual intervals and anomaly bounds are estimated from validation residuals.

Every lag and rolling feature is shifted first, so current or future demand cannot leak into a training row. DST-aware timestamps are normalized to UTC; naive source timestamps are interpreted as Europe/Berlin.

## Results

Full rolling-origin backtest of all three levels over the collected snapshot — every model refit at every origin, no random split — is in **[reports/benchmark.md](reports/benchmark.md)** (regenerate with `PYTHONPATH=src python scripts/benchmark.py`).

<!-- RESULTS-TABLE:START — copied from reports/benchmark.md; regenerate with scripts/benchmark.py -->
329 daily origins, first after 28 days of history, each scored on the next 24 hours (7,882 forecast hours, Oct 2025 – Aug 2026):

| Model | MAE (MW) | RMSE (MW) | sMAPE (%) | MAE vs seasonal-naive |
|---|---|---|---|---|
| Seasonal naive | 3,935 (±3,349) | 4,487 | 7.47 | — |
| SARIMAX | 3,366 (±2,883) | 3,991 | 6.30 | −14.4% |
| HistGradientBoosting | 1,944 (±1,366) | 2,350 | 3.56 | **−50.6%** |
<!-- RESULTS-TABLE:END -->

`HistGradientBoosting` is the champion, roughly halving seasonal-naive MAE. The report also carries prediction-interval coverage (empirical ≈ 93% against a 95% nominal target, walked forward with a trailing-week residual band), permutation importance on a held-out validation week (`lag_1h` dominates, then `hour`), per-origin error, and error slices by hour, weekday, month, and holiday — public holidays are the weak spot (MAE ~4,600 vs ~1,900 MW on ordinary days). The gradient-boosting hyperparameters were checked with a small rolling-origin sweep and left at their defaults — see the report's *Model configuration* section.

## Data

The source is the German Federal Network Agency's SMARD electricity-market platform, module 410 (Germany actual total grid load), read through the official chart-data API:

```bash
PYTHONPATH=src python scripts/ingest_smard_api.py --weeks 52
```

The client reads the weekly index at `https://www.smard.de/app/chart_data/410/DE/index_hour.json`, then pulls each weekly chunk (`.../410_DE_hour_<epoch_ms>.json`). Payload values are average-interval demand in MW at epoch-millisecond timestamps; they are converted to UTC hourly observations with columns `timestamp` and `demand_mw`. It defaults to the latest 52 weekly chunks (`--weeks` to change).

Every raw index/chunk response is written to `data/raw/`, canonical hourly demand to `data/clean/demand_hourly.csv`, and provenance (source URLs, collection time, row count, snapshot bounds) to `data/clean/metadata.json`. Missing hourly timestamps are reported, never imputed. `data/` is not tracked in git except for `metadata.json`; re-run the ingest command to rebuild the dataset.

Attribute the data to **Bundesnetzagentur | SMARD.de** under **CC BY 4.0**. SMARD may revise historical grid-load values, so a published result should cite the collected snapshot recorded in `metadata.json`.

The API and dashboard fall back to deterministic synthetic demo data when the clean file is absent. This keeps local smoke tests reproducible; it is not a substitute for official data in a published analysis.

## Dashboard

![Stromblick dashboard](reports/figures/dashboard.png)

### Reading the dashboard

A header metrics row — latest observed demand, the fixed 24-hour horizon, and the loaded data source ("SMARD clean export" for the real snapshot, the deterministic demo series otherwise) — sits above four tabs.

**Forecast** — the last three days of observed demand (dark line) then the 24-hour forecast (blue), with a dotted marker at the forecast start. The shaded band is the forecast ± the 95th percentile of absolute residuals the model made on a held-out validation week, so its width reflects how wrong the model has recently been, not a probabilistic guarantee. Gaps in the observed line are hours SMARD has indexed but not yet published.

**Model quality** — the full rolling-origin backtest from `reports/benchmark.md`: the three-model headline table, per-origin MAE for each model, prediction-interval coverage, permutation importance, and the champion's error slices. Below a divider, the fast trailing-7-day holdout comparison (seasonal-naive vs gradient boosting, with hour/weekday error slices) that runs live in the app. Both carry the same message: the main model has to beat "same hour yesterday" to justify its complexity.

**Anomalies** — observed demand against what the model expected for each hour (dashed), over a selectable 7/14/28-day window. The band is the expected value ± the 1st/99th percentile of validation residuals, learned from the week *before* the window so the bounds never see the data they score. Hours whose deviation leaves the band get a red ✕ and a table row with observed, expected, and deviation in MW. These are statistical flags — prompts to investigate weather, calendar, or grid events — not confirmed anomalies.

**Data & methods** — the provenance of the exact SMARD snapshot in use (collection time, row count, time range, weekly-chunk count) from `data/clean/metadata.json`, plus a summary of the method (leakage-safe features, chronological evaluation, the honest baseline) and the standing caveats about intervals and anomalies.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
uvicorn energy_forecast.api:app --reload
streamlit run app/streamlit_app.py
```

Rebuild the analysis artifacts from the clean snapshot:

```bash
PYTHONPATH=src python scripts/generate_eda.py    # reports/eda.md + figures
PYTHONPATH=src python scripts/benchmark.py        # reports/benchmark.md + figures + metrics CSV
```

API examples:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/forecast \
  -H 'content-type: application/json' \
  -d '{"as_of":"2026-08-28T00:00:00Z","horizon_hours":24}'
```

`POST /forecast` requires a timezone-aware ISO timestamp and accepts horizons from 1 through 24 hours. The response always includes UTC timestamps, point predictions, and symmetric residual-based interval bounds.

## Docker

```bash
docker compose up --build
```

The FastAPI service is at `http://localhost:8000`; the Streamlit dashboard is at `http://localhost:8501`.

## Limitations and next work

- Demand-only features are the first baseline. Renewable generation, temperature, wind, and solar forecasts should be added only after the demand-only comparison is stable.
- The bundled interval is empirical, based on historical validation residual magnitudes; it is not a calibrated probabilistic forecast.
- SMARD licensing, export shape, revision policy, and availability should be documented for the exact dataset snapshot used in a published analysis.
- A production version should add scheduled ingestion, persistent model artifacts, monitoring, weather covariates, and a final untouched test report.
