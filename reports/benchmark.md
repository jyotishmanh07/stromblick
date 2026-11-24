# Model benchmark

Snapshot: `2023-11-26 23:00:00+00:00` to `2026-11-23 09:00:00+00:00` (26,219 hourly rows). Source: Bundesnetzagentur | SMARD.de, module 410 (Germany actual total grid load), CC BY 4.0.

Rolling-origin backtest: **366 origins** spaced 24h apart over the latest 365 days. Each origin trains on data up to that point (HistGradientBoosting on the trailing 2 years, SARIMAX on the trailing 1 year) and is scored on the next 24 hours; every model is refit at every origin. Evaluation spans `2025-11-22 10:00:00+00:00` to `2026-11-23 09:00:00+00:00` (8,784 forecast hours). No random split is used, and every feature lag is shifted so no training row can see its own or a future value.

## Headline metrics

| Model | MAE (MW) | RMSE (MW) | sMAPE (%) | MAE vs seasonal-naive |
|---|---|---|---|---|
| Seasonal naive | 3,893 (±2,286) | 4,915 | 7.44 | — |
| SARIMAX | 3,704 (±2,345) | 4,642 | 7.25 | -4.8% |
| HistGradientBoosting | 1,505 (±838) | 1,826 | 2.86 | -61.3% |

MAE and RMSE are the mean across origins; the value in parentheses is the MAE standard deviation across origins, i.e. how much the error swings from window to window. Lower is better on every column.

**HistGradientBoosting** has the lowest error, cutting MAE by **61.3%** against the seasonal-naive baseline (1,505 vs 3,893 MW). The baseline is the honest yardstick: a model that cannot beat "same hour yesterday, last week as fallback" is not earning its complexity.

### Is the win statistically significant?

| Comparison | Mean per-origin MAE gap (MW) | 95% CI | Wilcoxon p | DM p |
|---|---|---|---|---|
| HistGradientBoosting − Seasonal naive | -2,387 | [-2,632, -2,142] | 9.84e-42 | 4.95e-47 |
| HistGradientBoosting − SARIMAX | -2,199 | [-2,444, -1,959] | 1.40e-42 | 1.07e-52 |

Per-origin MAE across the 366 paired origins is the sampling unit: hours inside one 24-hour window share weather and demand level, so treating them as independent would overstate confidence. The CI is a paired bootstrap (10,000 resamples) on the mean difference; negative means the champion errs less. The Diebold-Mariano column tests the same claim on hourly losses with a Harvey-Leybourne-Newbold correction for 24-step serial dependence. The gap is statistically significant against both rivals, not sampling noise.

**Interval coverage:** mean per-origin coverage is 94.3% (95% CI [93.1%, 95.5%]) against the 95% nominal target across 338 origins — statistically indistinguishable from nominal (p = 2.59e-01). Per-origin rates are the unit here for the same reason.

**Holiday penalty:** public-holiday hours cost 972 MW more absolute error than ordinary hours (95% CI [627, 1,344], permutation p = 2.00e-04, Cohen's d = 0.67) over 216 holiday hours against 8,568 ordinary ones. This is the clearest weakness in the champion and the first place to spend more feature work.

## Prediction-interval coverage

The shipped interval is HistGradientBoosting's forecast ± the split-conformal 95% quantile of absolute 24-hour errors from the previous 28 origins (the recipe the service serves, chosen in [interval_study.md](interval_study.md)). Walking that band forward across 8,112 scored hours, **94.3%** of observed values land inside it against a 95% nominal target, at a mean band width of 7,994 MW. Split conformal targets 95% *on average*, assuming recent errors resemble the next day's; it is not a full predictive distribution, and its width is the same for every hour ahead.

## What HistGradientBoosting relies on

![Permutation importance](figures/benchmark_feature_importance.png)

Permutation importance on a held-out trailing validation week: each bar is how much the validation MAE rises when that column is shuffled. The three that matter most are `lag_1h` (8,807 MW), `hour` (1,814 MW), `weekday` (150 MW).

## Error at each origin

![MAE by origin](figures/benchmark_mae_by_origin.png)

Per-origin MAE over the whole backtest. A model is only trustworthy if it wins consistently, not just on average.

## Where HistGradientBoosting errs

![Champion error slices](figures/benchmark_error_slices.png)

MAE for HistGradientBoosting broken down by local hour, weekday, calendar month, and German public-holiday status. The largest hourly error falls at hour 10 (Europe/Berlin). On public holidays the champion's MAE is 2,453 MW versus 1,481 MW on ordinary days.

## Model configuration

The gradient-boosting settings in `models.py` (`max_iter=250`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`) were checked with a **nested rolling-origin search**: 40 Optuna TPE trials scored on 12 origins drawn only from the training span, then confirmed on 60 later origins the search never saw.

The search did find a better inner score (2,573 vs 2,665 MW, -3.5%). On the held-out origins that shrank to 1,567 vs 1,587 MW (-1.3%), a mean difference of -20 MW with a 95% CI of [-82, 44] (Wilcoxon p = 0.38).

The shipped defaults are **kept**. The tuned configuration is only 1.3% better on origins the search never saw, and the confidence interval on that difference contains zero — the gain is smaller than the variation between origins. Adopting it would be fitting the search rather than improving the model.

That gap between the inner and outer numbers is the reason for the nesting. A search almost always improves the score on the data it searched; only the held-out comparison says whether anything was learned. Full detail in [tuning.md](tuning.md).

## Reproduce

```bash
PYTHONPATH=src python scripts/benchmark.py --step-hours 24
```

Per-origin metrics for every model are written to `benchmark_metrics.csv` alongside this report; the dashboard reads that file directly.
