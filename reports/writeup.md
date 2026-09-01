# Forecasting German electricity demand — what the model does, and what it's worth

*A short read for a non-technical audience, with a technical appendix. Every number below is reproducible from this repository; see [Reproduce](#reproduce).*

---

## The question

Germany's grid operators and balancing-responsible parties have to answer the same question every afternoon: **how much electricity will the country draw during each of the next 24 hours?** Commit too little and you buy the shortfall at imbalance prices set after the fact. Commit too much and you have paid to generate power nobody used.

Stromblick answers that question from public data, and — just as importantly — reports how wrong it is likely to be.

## Why this matters more than it used to

The 2022 gas crisis made energy forecasting briefly famous, but the durable reason sits deeper in Germany's grid. The nuclear exit completed in April 2023 and the coal phase-out is ongoing, while renewables now supply the majority of generation. Supply has become weather-driven and intermittent, which means the *flexibility* to absorb a forecast error is thinner than it was. At the same time, electrification of heat and transport — heat pumps, EVs — is reshaping the demand curve itself, so the historical patterns a forecaster leans on keep drifting.

Both trends push the same way: short-term load forecasting matters **more** than it did a decade ago, and models need refitting and monitoring rather than a one-time build.

Three concrete places the number lands:

- **Balancing.** Parties pay for the gap between their schedule and reality at imbalance prices. Forecast error is a direct cost line.
- **Reserve sizing.** System operators hold reserve capacity proportional to expected error. A tighter forecast means less capacity held idle.
- **Investigation.** Unusual hours — cold snaps, industrial curtailment, holiday effects — are worth an analyst's attention if you can find them among 8,760 hours a year.

## What was built

A forecast of the next 24 hours of German grid load, with an uncertainty band, plus a detector that flags hours which departed sharply from expectation. Data comes from SMARD, the German Federal Network Agency's market platform (module 410, actual total grid load), pulled through its official API. The snapshot analysed here covers **8,578 hourly observations** from September 2025 to August 2026.

Three models were compared, deliberately in increasing order of complexity:

1. **Seasonal naive** — "same hour yesterday", falling back to last week. The honest yardstick.
2. **SARIMAX** — a classical statistical model with daily seasonality.
3. **Gradient boosting** — the main model, using calendar features, German public-holiday and daylight-saving flags, lagged demand, and trailing averages.

The first model matters most. Any forecasting project can produce a plausible-looking chart; the question a reviewer should ask is *compared to what?* If a gradient-boosted model cannot beat "same hour yesterday", it has not earned its complexity.

## What the results say

Evaluated over **329 rolling origins** — 7,882 forecast hours from October 2025 to August 2026, with every model refit at every origin and no random splitting:

| Model | Mean absolute error | vs. baseline |
|---|---|---|
| Seasonal naive | 3,935 MW | — |
| SARIMAX | 3,366 MW | −14.4% |
| **Gradient boosting** | **1,944 MW** | **−50.6%** |

The main model roughly **halves** the error of the naive baseline. On a typical day it is off by about 1,900 MW against demand that runs between 33 and 78 GW — roughly 3.6% in percentage terms.

That headline is the easy part. Three follow-up findings matter more:

**The win is real, not luck.** Averaged over 329 origins, the gradient-boosting model beats SARIMAX by 1,422 MW (95% confidence interval 1,100–1,746 MW) and the naive baseline by 1,991 MW (CI 1,630–2,345 MW). Both intervals sit far from zero. A Diebold-Mariano test on hourly errors agrees. This is worth stating because a single average can hide a model that wins slightly on most days and loses badly on a few.

**The uncertainty band is slightly too narrow — and we can say so precisely.** The shipped band is designed to contain the true value 95% of the time. Walked forward across the whole backtest, it actually contains it **93.2%** of the time (CI 91.5–94.7%). That gap is statistically real, not sampling noise. It is a small, honest miscalibration: users should read the band as "roughly how wrong this model has been lately", not as a probabilistic guarantee.

**Public holidays are the clear weak spot.** On holiday hours the model errs by **2,741 MW more** than on ordinary hours (CI 2,169–3,312 MW). The effect size is large. Germany's holidays vary by federal state and the model only knows national ones, so this is the first place additional work would pay.

![Where the model errs](figures/benchmark_error_slices.png)

## What decision this supports

**Use the forecast plus its band to size day-ahead reserve, and treat the band as empirical, not probabilistic.** Halving the mean error from 3,935 to 1,944 MW means roughly 2,000 MW less expected deviation per hour.

Putting an illustrative figure on that requires care, because nobody balances the entire national grid. Scaled to a **balancing party responsible for 1% of German load**, the error reduction is about 20 MW per hour, or roughly 157 GWh of avoided absolute deviation across the 7,882-hour window. *If* the relevant imbalance-price spread were €50/MWh, that is on the order of **€7.8 million** over eleven months. Halve the portfolio share and you halve the figure.

Those numbers are an **order-of-magnitude illustration, not a P&L claim**, and the assumptions do real work:

- Every megawatt of forecast error is assumed to convert to imbalance cost at one flat price. Real balancing prices vary by direction and scarcity, and are sometimes favourable.
- Portfolio errors net out against other positions; a real party also has intraday trading and flexible assets to close a gap more cheaply.
- The 1% share and the €50/MWh spread are both stipulated, not measured.

The defensible reading is narrow: the error reduction is large enough to be economically meaningful, and quantifying it properly needs imbalance-price data this project does not use. The unscaled national figure (~15.7 TWh, ~€785M at the same price) is arithmetically correct but should not be quoted — it implies a single actor balancing all of Germany, which is not a thing.

**Route flagged anomalies to a human, not to an automated action.** The anomaly detector marks hours whose deviation exceeds the model's recent residual bounds. These are prompts to investigate weather, calendar, or grid events. They are statistical flags, not confirmed incidents, and the dashboard says so.

**Do not treat the band as a probability.** Given the measured 93.2% coverage against a 95% design target, any downstream process needing calibrated probabilities should recalibrate first.

## A secondary question: which days are worth planning around?

A companion track reframes the problem as classification: rather than *how much* demand each hour, **is tomorrow one of the heavier days of the recent month?** The gradient-boosted classifier reaches a PR-AUC of 0.598 against a base rate of 19.6% — about 3× better than chance at ranking days.

The more interesting result is a negative one. On the *anomaly-day* target, a logistic regression using **nothing but the calendar** scores 0.716 — statistically level with the gradient-boosted model's 0.719. Anomalous days in this dataset are overwhelmingly holidays and daylight-saving transitions, not subtle demand dynamics. The sophisticated model adds nothing, and saying so is more useful than quietly shipping it.

## Limitations

- **No weather.** Temperature is the single largest missing driver. Demand-only features were a deliberate first baseline; weather covariates are the obvious next step and would most likely help exactly where the model is weakest.
- **One year of data.** Seasonal patterns are observed once. The classification track in particular works with a few hundred labelled days, so its rankings are indicative rather than settled.
- **The interval is empirical.** It comes from recent residual magnitudes, not from a probabilistic model, and it measurably under-covers.
- **Anomalies are flags, not explanations.** Nothing here establishes causation.
- **SMARD revises history.** Published results should cite the exact snapshot recorded in `data/clean/metadata.json`.
- **Production would need more.** Persistent model artifacts, drift monitoring, and — at real scale — a dedicated data-quality framework (Great Expectations, dbt tests) behind an orchestrator (Airflow, Dagster). This project runs an explicit validation gate and a weekly scheduled refresh instead, which is the right size for what it is.

---

## Technical appendix

**Leakage safety.** Every lag and rolling feature is shifted before aggregation, so no training row can see its own or a future value. The same rule governs the classification labels: each threshold is a trailing-window quantile computed from earlier days and shifted, so the rule judging a day is fixed before that day is observed.

**Evaluation.** `rolling_origin_backtest` walks origins forward in time and refits every model at each one. No random split is used anywhere in the project. Model selection belongs on validation origins; a final holdout is passed only after that choice is made.

**Sampling unit for inference.** All significance tests use **per-origin** values rather than per-hour. Hours inside one 24-hour window share weather and demand level, so pooling them would overstate confidence. The Diebold-Mariano test is the deliberate exception: it operates on hourly losses but corrects its long-run variance for 24-step serial dependence (Harvey-Leybourne-Newbold).

**Timestamps.** Everything crossing a module boundary is timezone-aware UTC; local time is used only for calendar features and display. Daylight-saving transitions are handled explicitly, including the repeated hour in autumn.

**Feature importance.** Permutation importance on a held-out validation week ranks `lag_1h` far first (7,064 MW), then `hour` (1,258 MW) and `weekday` (145 MW) — the model is mostly a smart persistence forecaster with calendar corrections, which is what a well-behaved short-horizon load model should look like.

**Hyperparameters.** Tuned by a nested rolling-origin search: the inner loop searches on origins drawn only from a training span, the outer loop confirms on later origins the search never saw. A tuned configuration is adopted only if its outer improvement clears a paired-bootstrap confidence interval. See [tuning.md](tuning.md) for the standing verdict.

**Full reports.** [benchmark.md](benchmark.md) (forecasting, significance, coverage, error slices) · [classification.md](classification.md) (event classification) · [eda.md](eda.md) (exploratory analysis) · [tuning.md](tuning.md) (hyperparameter search).

## Reproduce

```bash
pip install -e ".[dev]"
PYTHONPATH=src python scripts/ingest_smard_api.py --weeks 52   # collect + validate
PYTHONPATH=src python scripts/build_warehouse.py               # DuckDB reporting layer
PYTHONPATH=src python scripts/generate_eda.py                  # eda.md
PYTHONPATH=src python scripts/benchmark.py                     # benchmark.md (~20 min)
PYTHONPATH=src python scripts/benchmark_classification.py      # classification.md
```

Data: Bundesnetzagentur | SMARD.de, module 410 (Germany actual total grid load), CC BY 4.0.
