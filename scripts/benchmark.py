#!/usr/bin/env python3
"""Rolling-origin backtest of all three model levels against the clean SMARD dataset.

Writes reports/benchmark.md plus two figures. Reuses the project's own evaluation
and model code so the report reflects exactly what ships: every model is refit at
every origin, features are leakage-safe, and no random split is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # report script: never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402

from energy_forecast.data import load_clean_demand
from energy_forecast.evaluation import error_slices, metrics
from energy_forecast.features import FEATURE_COLUMNS, add_features, german_holidays
from energy_forecast.inference import (
    coverage_test,
    diebold_mariano,
    paired_model_comparison,
    slice_difference_test,
)
from energy_forecast.models import (
    HistGradientBoostingForecast,
    SARIMAXBaseline,
    SeasonalNaive,
)
from energy_forecast.theme import INK
from energy_forecast.theme import MODEL_COLORS as PALETTE
from energy_forecast.theme import style_axes as _style_axes

FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/benchmark.md")
METRICS_PATH = Path("reports/benchmark_metrics.csv")
HOURLY_PATH = Path("reports/benchmark_champion_hourly.csv")
POOLED_PATH = Path("reports/benchmark_pooled_hourly.csv")
SUMMARY_PATH = Path("reports/benchmark_summary.json")
TUNING_SUMMARY_PATH = Path("reports/tuning_summary.json")
README_PATH = Path("README.md")
RESULTS_START = "<!-- RESULTS-TABLE:START"
RESULTS_END = "<!-- RESULTS-TABLE:END -->"

MODEL_FACTORIES = {
    "Seasonal naive": SeasonalNaive,
    "SARIMAX": SARIMAXBaseline,
    "HistGradientBoosting": HistGradientBoostingForecast,
}
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def walk_forward(
    frame: pd.DataFrame, initial_train_hours: int, horizon_hours: int, step_hours: int,
    max_origins: int | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return per-origin metrics and pooled per-hour predictions for every model."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    origins = list(range(initial_train_hours, len(data) - horizon_hours + 1, step_hours))
    if max_origins is not None:
        origins = origins[:max_origins]
    per_origin: list[dict[str, object]] = []
    per_hour: dict[str, list[pd.DataFrame]] = {name: [] for name in MODEL_FACTORIES}
    for i, origin in enumerate(origins, 1):
        train, test = data.iloc[:origin], data.iloc[origin : origin + horizon_hours]
        actual = test.demand_mw.to_numpy(dtype=float)
        observed = ~np.isnan(actual)
        if not observed.any():
            continue
        stamps = pd.DatetimeIndex(test.timestamp)
        for name, factory in MODEL_FACTORIES.items():
            predicted = factory().fit(train).predict(stamps)
            per_origin.append(
                {
                    "origin": train.timestamp.iloc[-1], "model": name,
                    **metrics(actual[observed], predicted[observed]),
                }
            )
            per_hour[name].append(
                pd.DataFrame(
                    {
                        "origin": train.timestamp.iloc[-1],
                        "timestamp": test.timestamp.to_numpy()[observed],
                        "demand_mw": actual[observed], "predicted": predicted[observed],
                    }
                )
            )
        print(f"  origin {i}/{len(origins)}  {train.timestamp.iloc[-1]:%Y-%m-%d}", flush=True)
    pooled = {name: pd.concat(parts, ignore_index=True) for name, parts in per_hour.items()}
    return pd.DataFrame(per_origin), pooled


def plot_mae_by_origin(per_origin: pd.DataFrame, out: Path, smooth_origins: int = 14) -> None:
    """Per-origin MAE: faint raw line plus a bold trailing mean so the persistent gap reads."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, group in per_origin.groupby("model"):
        ordered = group.sort_values("origin")
        ax.plot(
            ordered.origin, ordered.mae, color=PALETTE[name], linewidth=0.6, alpha=0.28, zorder=2
        )
        ax.plot(
            ordered.origin, ordered.mae.rolling(smooth_origins, min_periods=smooth_origins).mean(),
            color=PALETTE[name], linewidth=2, label=name, zorder=3,
        )
    ax.set_ylabel("MAE over the 24h window (MW)")
    ax.set_title(
        f"Forecast error at each rolling origin ({smooth_origins}-origin trailing mean)",
        color=INK, loc="left",
    )
    ax.legend(frameon=False, labelcolor=INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_error_slices(pooled_champion: pd.DataFrame, champion: str, out: Path) -> dict[str, object]:
    slices = error_slices(
        pooled_champion[["timestamp", "demand_mw"]], pooled_champion.predicted.to_numpy()
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    color = PALETTE[champion]

    by_hour = slices["hour"].sort_values("hour")
    axes[0, 0].bar(by_hour.hour, by_hour.absolute_error, color=color, width=0.7, zorder=3)
    axes[0, 0].set_xlabel("Hour of day (Europe/Berlin)")

    by_weekday = slices["weekday"].set_index("weekday").reindex(WEEKDAY_ORDER).reset_index()
    axes[0, 1].bar(
        range(7), by_weekday.absolute_error, color=color, width=0.7, zorder=3,
        tick_label=[d[:3] for d in WEEKDAY_ORDER],
    )

    by_month = slices["month"].sort_values("month")
    axes[1, 0].bar(by_month.month, by_month.absolute_error, color=color, width=0.7, zorder=3)
    axes[1, 0].set_xlabel("Calendar month")

    by_holiday = slices["holiday"]
    labels = ["Ordinary day", "Public holiday"]
    values = [
        float(by_holiday.loc[by_holiday.is_public_holiday == flag, "absolute_error"].mean())
        if (by_holiday.is_public_holiday == flag).any() else 0.0
        for flag in (0, 1)
    ]
    axes[1, 1].bar(labels, values, color=color, width=0.6, zorder=3)

    for ax, title in zip(
        axes.flat, ("By hour", "By weekday", "By month", "By public-holiday status")
    ):
        ax.set_ylabel("Mean absolute error (MW)")
        ax.set_title(title, color=INK, loc="left")
        _style_axes(ax)
    fig.suptitle(f"Where {champion} errs", color=INK, x=0.02, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    worst_hour = int(by_hour.loc[by_hour.absolute_error.idxmax(), "hour"])
    return {"worst_hour": worst_hour, "holiday_mae": values[1], "ordinary_mae": values[0]}


def interval_coverage(
    pooled_champion: pd.DataFrame, residual_quantile: float = 0.95,
    trailing_origins: int = 7, min_prior_origins: int = 3,
) -> dict[str, float]:
    """Walk origins forward; size a ±quantile band from the most recent `trailing_origins`
    earlier origins — the trailing-window recipe ForecastService._fit_validation_residuals
    uses. The band that scores an origin never sees that origin's own errors. Reports how
    often the observed value actually lands inside.
    """
    ordered = sorted(pooled_champion.origin.unique())
    inside = scored = 0
    widths: list[float] = []
    per_origin_rates: list[float] = []
    for i, origin in enumerate(ordered):
        if i < min_prior_origins:
            continue
        window = ordered[max(0, i - trailing_origins) : i]
        prior = pooled_champion[pooled_champion.origin.isin(window)]
        current = pooled_champion[pooled_champion.origin == origin]
        spread = float(np.quantile(np.abs(prior.demand_mw - prior.predicted), residual_quantile))
        actual = current.demand_mw.to_numpy()
        predicted = current.predicted.to_numpy()
        hit = (actual >= predicted - spread) & (actual <= predicted + spread)
        inside += int(np.sum(hit))
        scored += len(actual)
        widths.append(2 * spread)
        per_origin_rates.append(float(np.mean(hit)))
    return {
        "nominal": 100 * residual_quantile,
        "empirical": 100 * inside / scored if scored else float("nan"),
        "hours_scored": scored,
        "mean_band_mw": float(np.mean(widths)) if widths else float("nan"),
    }, per_origin_rates


def significance(
    per_origin: pd.DataFrame, pooled: dict[str, pd.DataFrame], champion: str,
    coverage_rates: list[float], residual_quantile: float = 0.95,
) -> dict[str, object]:
    """Test the headline claims: is the champion's win real, and is coverage on target?"""
    wide = per_origin.pivot(index="origin", columns="model", values="mae").dropna()
    comparisons: dict[str, dict[str, float]] = {}
    for rival in (name for name in MODEL_FACTORIES if name != champion):
        paired = paired_model_comparison(wide[champion].to_numpy(), wide[rival].to_numpy())
        merged = pooled[champion].merge(
            pooled[rival][["origin", "timestamp", "predicted"]],
            on=["origin", "timestamp"], suffixes=("", "_rival"),
        )
        dm = diebold_mariano(
            np.abs(merged.demand_mw - merged.predicted).to_numpy(),
            np.abs(merged.demand_mw - merged.predicted_rival).to_numpy(),
        )
        comparisons[rival] = {**paired, **{f"dm_{k}": v for k, v in dm.items()}}

    hourly = pooled[champion].copy()
    local = pd.to_datetime(hourly.timestamp, utc=True).dt.tz_convert("Europe/Berlin")
    holidays = german_holidays(set(local.dt.year.unique()))
    is_holiday = local.dt.date.isin(holidays).to_numpy()
    # A short evaluation span can contain no holidays at all; report nothing rather than guess.
    holiday_slice = (
        slice_difference_test(np.abs(hourly.demand_mw - hourly.predicted).to_numpy(), is_holiday)
        if is_holiday.any() and not is_holiday.all() else None
    )
    return {
        "comparisons": comparisons,
        "holiday_slice": holiday_slice,
        "coverage": coverage_test(coverage_rates, nominal=residual_quantile),
    }


def tuning_section() -> list[str]:
    """Describe the shipped hyperparameters, citing the search that justifies them.

    Reads reports/tuning_summary.json rather than asserting a result: a claim about a
    sweep that leaves no artifact behind is one a reader cannot check.
    """
    shipped = ", ".join(
        f"`{k}={v}`" for k, v in HistGradientBoostingForecast.DEFAULT_PARAMS.items()
    )
    if not TUNING_SUMMARY_PATH.exists():
        return [
            f"The gradient-boosting settings in `models.py` ({shipped}) are the shipped defaults. "
            "Run `PYTHONPATH=src python scripts/tune.py` to search around them; that script "
            "writes `reports/tuning.md` and this section then cites its verdict.",
            "",
        ]
    tuning = json.loads(TUNING_SUMMARY_PATH.read_text())
    comparison = tuning["comparison"]
    inner_delta = 100 * (
        tuning["inner_best_mae"] - tuning["inner_default_mae"]
    ) / tuning["inner_default_mae"]
    outer_delta = 100 * (
        tuning["outer_tuned_mae"] - tuning["outer_default_mae"]
    ) / tuning["outer_default_mae"]
    verdict = (
        "The tuned configuration was **adopted**: its advantage on unseen origins clears the "
        "paired-bootstrap interval."
        if tuning["adopt_tuned"] else
        f"The shipped defaults are **kept**. The tuned configuration is only "
        f"{abs(outer_delta):.1f}% better on origins the search never saw, and the "
        "confidence interval on that difference "
        "contains zero — the gain is smaller than the variation between origins. Adopting it "
        "would be fitting the search rather than improving the model."
    )
    return [
        f"The gradient-boosting settings in `models.py` ({shipped}) were checked with a "
        f"**nested rolling-origin search**: {tuning['trials']} Optuna TPE trials scored on "
        f"{tuning['inner_origins']} origins drawn only from the training span, then confirmed on "
        f"{tuning['outer_origins']} later origins the search never saw.",
        "",
        f"The search did find a better inner score ({tuning['inner_best_mae']:,.0f} vs "
        f"{tuning['inner_default_mae']:,.0f} MW, {inner_delta:+.1f}%). On the held-out origins "
        f"that shrank to {tuning['outer_tuned_mae']:,.0f} vs {tuning['outer_default_mae']:,.0f} MW "
        f"({outer_delta:+.1f}%), a mean difference of {comparison['mean_diff']:,.0f} MW with a "
        f"95% CI of [{comparison['ci_lower']:,.0f}, {comparison['ci_upper']:,.0f}] "
        f"(Wilcoxon p = {comparison['wilcoxon_p']:.2f}).",
        "",
        verdict,
        "",
        "That gap between the inner and outer numbers is the reason for the nesting. A search "
        "almost always improves the score on the data it searched; only the held-out comparison "
        "says whether anything was learned. Full detail in [tuning.md](tuning.md).",
        "",
    ]


def sync_readme_results(table: list[str], preamble: str, verdict: str) -> bool:
    """Rewrite the README's RESULTS-TABLE block so it cannot drift from the benchmark."""
    if not README_PATH.exists():
        return False
    text = README_PATH.read_text(encoding="utf-8")
    start, end = text.find(RESULTS_START), text.find(RESULTS_END)
    if start == -1 or end == -1:
        return False
    marker_end = text.find("-->", start) + len("-->")
    block = "\n".join(["", preamble, "", *table, "", verdict, ""])
    README_PATH.write_text(text[:marker_end] + block + text[end:], encoding="utf-8")
    return True


def plot_feature_importance(
    frame: pd.DataFrame, out: Path, validation_hours: int = 24 * 7
) -> dict[str, object] | None:
    """Permutation importance for the GBM champion on a held-out trailing validation week."""
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        return None
    model = HistGradientBoostingForecast()
    if model.regressor is None:  # lstsq fallback: not a meaningful permutation-importance target
        return None
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    model.fit(data.iloc[:-validation_hours])
    featured = add_features(data).dropna(subset=FEATURE_COLUMNS + ["demand_mw"])
    validation = featured.iloc[-validation_hours:]
    result = permutation_importance(
        model.regressor, validation[FEATURE_COLUMNS], validation["demand_mw"],
        n_repeats=20, random_state=42, scoring="neg_mean_absolute_error",
    )
    order = np.argsort(result.importances_mean)
    labels = [FEATURE_COLUMNS[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(len(labels)), result.importances_mean[order],
        xerr=result.importances_std[order], color=PALETTE["HistGradientBoosting"], zorder=3,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rise in validation MAE when the feature is shuffled (MW)")
    ax.set_title("What HistGradientBoosting relies on", color=INK, loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    top = [(FEATURE_COLUMNS[i], float(result.importances_mean[i])) for i in order[::-1][:3]]
    return {"top_features": top}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-train-days", type=int, default=28)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument(
        "--step-hours", type=int, default=24,
        help="spacing between origins; 24 = daily (default). Keep it a non-multiple of 168 "
        "so origins rotate through all weekdays instead of aliasing onto one.",
    )
    parser.add_argument("--max-origins", type=int, default=None, help="cap origins (for testing)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_clean_demand()
    last_valid = frame.demand_mw.last_valid_index()
    frame = frame.loc[:last_valid].reset_index(drop=True)
    initial_train_hours = args.initial_train_days * 24

    print(f"Backtesting {len(frame):,} rows, step {args.step_hours}h ...", flush=True)
    per_origin, pooled = walk_forward(
        frame, initial_train_hours, args.horizon_hours, args.step_hours, args.max_origins
    )

    summary = (
        per_origin.groupby("model")
        .agg(
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), smape_mean=("smape", "mean"),
        )
        .reindex(MODEL_FACTORIES.keys())
    )
    baseline_mae = summary.loc["Seasonal naive", "mae_mean"]
    champion = str(summary.mae_mean.idxmin())
    champion_mae = summary.loc[champion, "mae_mean"]
    lift = 100 * (baseline_mae - champion_mae) / baseline_mae

    per_origin.to_csv(METRICS_PATH, index=False)
    pooled[champion].to_csv(HOURLY_PATH, index=False)
    pd.concat(
        [frame_.assign(model=name) for name, frame_ in pooled.items()], ignore_index=True
    ).to_csv(POOLED_PATH, index=False)
    plot_mae_by_origin(per_origin, FIG_DIR / "benchmark_mae_by_origin.png")
    notes = plot_error_slices(pooled[champion], champion, FIG_DIR / "benchmark_error_slices.png")
    coverage, coverage_rates = interval_coverage(pooled[champion])
    tests = significance(per_origin, pooled, champion, coverage_rates)
    importance = (
        plot_feature_importance(frame, FIG_DIR / "benchmark_feature_importance.png")
        if champion == "HistGradientBoosting" else None
    )

    origins_used = per_origin.origin.nunique()
    eval_hours = len(pooled[champion])
    test_start = pooled[champion].timestamp.min()
    test_end = pooled[champion].timestamp.max()

    table = [
        "| Model | MAE (MW) | RMSE (MW) | sMAPE (%) | MAE vs seasonal-naive |",
        "|---|---|---|---|---|",
    ]
    for name, row in summary.iterrows():
        if name == "Seasonal naive":
            delta = "—"
        else:
            delta = f"{100 * (row.mae_mean - baseline_mae) / baseline_mae:+.1f}%"
        table.append(
            f"| {name} | {row.mae_mean:,.0f} (±{row.mae_std:,.0f}) | {row.rmse_mean:,.0f} "
            f"| {row.smape_mean:.2f} | {delta} |"
        )

    holiday_line = (
        f"On public holidays the champion's MAE is {notes['holiday_mae']:,.0f} MW versus "
        f"{notes['ordinary_mae']:,.0f} MW on ordinary days"
        if notes["holiday_mae"] else "The evaluation span contained no public-holiday hours"
    )

    significance_rows = [
        "| Comparison | Mean per-origin MAE gap (MW) | 95% CI | Wilcoxon p | DM p |",
        "|---|---|---|---|---|",
    ]
    for rival, stat in tests["comparisons"].items():
        significance_rows.append(
            f"| {champion} − {rival} | {stat['mean_diff']:,.0f} | "
            f"[{stat['ci_lower']:,.0f}, {stat['ci_upper']:,.0f}] | "
            f"{stat['wilcoxon_p']:.2e} | {stat['dm_p_value']:.2e} |"
        )
    beats_all = all(s["ci_upper"] < 0 for s in tests["comparisons"].values())
    verdict = (
        "The gap is statistically significant against both rivals, not sampling noise."
        if beats_all else
        "At least one comparison's confidence interval contains zero — treat that gap as unproven."
    )
    cov = tests["coverage"]
    hol = tests["holiday_slice"]
    coverage_verdict = (
        "below the nominal target by a margin the data can resolve"
        if cov["ci_upper"] < cov["nominal"] else "statistically indistinguishable from nominal"
    )
    holiday_significance = (
        [
            f"**Holiday penalty:** public-holiday hours cost {hol['mean_diff']:,.0f} MW more "
            f"absolute error than ordinary hours (95% CI [{hol['ci_lower']:,.0f}, "
            f"{hol['ci_upper']:,.0f}], permutation p = {hol['permutation_p']:.2e}, Cohen's d = "
            f"{hol['cohens_d']:.2f}) over {hol['n_slice']:,} holiday hours against "
            f"{hol['n_rest']:,} ordinary ones. This is the clearest weakness in the champion and "
            "the first place to spend more feature work.",
            "",
        ]
        if hol is not None else []
    )
    significance_section = [
        "### Is the win statistically significant?",
        "",
        *significance_rows,
        "",
        f"Per-origin MAE across the {tests['comparisons'][next(iter(tests['comparisons']))]['n']} "
        "paired origins is the sampling unit: hours inside one 24-hour window share weather and "
        "demand level, so treating them as independent would overstate confidence. The CI is a "
        "paired bootstrap (10,000 resamples) on the mean difference; negative means the champion "
        "errs less. The Diebold-Mariano column tests the same claim on hourly losses with a "
        f"Harvey-Leybourne-Newbold correction for 24-step serial dependence. {verdict}",
        "",
        f"**Interval coverage:** mean per-origin coverage is {100 * cov['mean_coverage']:.1f}% "
        f"(95% CI [{100 * cov['ci_lower']:.1f}%, {100 * cov['ci_upper']:.1f}%]) against the "
        f"{100 * cov['nominal']:.0f}% nominal target across {cov['n_origins']} origins — "
        f"{coverage_verdict} (p = {cov['p_value']:.2e}). Per-origin rates are the unit here for "
        "the same reason.",
        "",
        *holiday_significance,
    ]
    importance_section: list[str] = []
    if importance is not None:
        ranked = ", ".join(
            f"`{name}` ({value:,.0f} MW)" for name, value in importance["top_features"]
        )
        importance_section = [
            f"## What {champion} relies on",
            "",
            "![Permutation importance](figures/benchmark_feature_importance.png)",
            "",
            "Permutation importance on a held-out trailing validation week: each bar is how much "
            "the validation MAE rises when that column is shuffled. The three that matter most "
            f"are {ranked}.",
            "",
        ]
    lines = [
        "# Model benchmark",
        "",
        f"Snapshot: `{frame.timestamp.min()}` to `{frame.timestamp.max()}` "
        f"({len(frame):,} hourly rows). Source: Bundesnetzagentur | SMARD.de, module 410 "
        "(Germany actual total grid load), CC BY 4.0.",
        "",
        f"Rolling-origin backtest: **{origins_used} origins** spaced {args.step_hours}h apart, "
        f"the first after {args.initial_train_days} days of history. Each origin trains on all "
        f"data up to that point and is scored on the next {args.horizon_hours} hours; every model "
        f"is refit at every origin. Evaluation spans `{test_start}` to `{test_end}` "
        f"({eval_hours:,} forecast hours). No random split is used, and every feature lag is "
        "shifted so no training row can see its own or a future value.",
        "",
        "## Headline metrics",
        "",
        *table,
        "",
        "MAE and RMSE are the mean across origins; the value in parentheses is the MAE standard "
        "deviation across origins, i.e. how much the error swings from window to window. Lower is "
        "better on every column.",
        "",
        f"**{champion}** has the lowest error, cutting MAE by **{lift:.1f}%** against the "
        f"seasonal-naive baseline ({champion_mae:,.0f} vs {baseline_mae:,.0f} MW). The baseline is "
        "the honest yardstick: a model that cannot beat \"same hour yesterday, last week as "
        "fallback\" is not earning its complexity.",
        "",
        *significance_section,
        "## Prediction-interval coverage",
        "",
        f"The shipped interval is {champion}'s forecast ± the {coverage['nominal']:.0f}th "
        "percentile of absolute residuals from *earlier* origins (the same recipe the service "
        f"uses). Walking that band forward across {coverage['hours_scored']:,} scored hours, "
        f"**{coverage['empirical']:.1f}%** of observed values land inside it against a "
        f"{coverage['nominal']:.0f}% nominal target, at a mean band width of "
        f"{coverage['mean_band_mw']:,.0f} MW. This is an empirical magnitude band, not a "
        "calibrated probabilistic interval.",
        "",
        *importance_section,
        "## Error at each origin",
        "",
        "![MAE by origin](figures/benchmark_mae_by_origin.png)",
        "",
        "Per-origin MAE over the whole backtest. A model is only trustworthy if it wins "
        "consistently, not just on average.",
        "",
        f"## Where {champion} errs",
        "",
        "![Champion error slices](figures/benchmark_error_slices.png)",
        "",
        f"MAE for {champion} broken down by local hour, weekday, calendar month, and German "
        f"public-holiday status. The largest hourly error falls at hour {notes['worst_hour']:02d} "
        f"(Europe/Berlin). {holiday_line}.",
        "",
        "## Model configuration",
        "",
        *tuning_section(),
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=src python scripts/benchmark.py --step-hours {args.step_hours}",
        "```",
        "",
        f"Per-origin metrics for every model are written to `{METRICS_PATH.name}` alongside this "
        "report; the dashboard reads that file directly.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "snapshot_start": str(frame.timestamp.min()),
                "snapshot_end": str(frame.timestamp.max()),
                "rows": int(len(frame)),
                "origins": int(origins_used),
                "step_hours": int(args.step_hours),
                "horizon_hours": int(args.horizon_hours),
                "eval_start": str(test_start),
                "eval_end": str(test_end),
                "eval_hours": int(eval_hours),
                "champion": champion,
                "lift_vs_seasonal_naive_pct": round(float(lift), 1),
                "models": {
                    name: {
                        "mae_mean": round(float(row.mae_mean), 1),
                        "mae_std": round(float(row.mae_std), 1),
                        "rmse_mean": round(float(row.rmse_mean), 1),
                        "smape_mean": round(float(row.smape_mean), 3),
                    }
                    for name, row in summary.iterrows()
                },
                "interval_coverage": {k: round(float(v), 2) for k, v in coverage.items()},
                "significance": {
                    "comparisons": {
                        rival: {k: round(float(v), 4) for k, v in stat.items()}
                        for rival, stat in tests["comparisons"].items()
                    },
                    "holiday_slice": (
                        {k: round(float(v), 4) for k, v in tests["holiday_slice"].items()}
                        if tests["holiday_slice"] is not None else None
                    ),
                    "coverage": {k: round(float(v), 4) for k, v in tests["coverage"].items()},
                    "champion_beats_all_rivals": bool(beats_all),
                },
                "top_features": [list(pair) for pair in (importance or {}).get("top_features", [])],
                "worst_hour": int(notes["worst_hour"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # The closest rival is the one the champion beats by the smallest margin.
    worst = max(tests["comparisons"].values(), key=lambda s: s["mean_diff"])
    synced = sync_readme_results(
        table,
        f"{origins_used} daily origins, first after {args.initial_train_days} days of history, "
        f"each scored on the next {args.horizon_hours} hours ({eval_hours:,} forecast hours, "
        f"{test_start:%b %Y} – {test_end:%b %Y}):",
        f"Paired bootstrap over the {worst['n']} origins puts the champion's MAE advantage at "
        f"[{-worst['ci_upper']:,.0f}, {-worst['ci_lower']:,.0f}] MW against its closest rival "
        f"(Wilcoxon p = {worst['wilcoxon_p']:.1e}) — see "
        "[reports/benchmark.md](reports/benchmark.md) for the full significance section.",
    )

    figures = 3 + (importance is not None)
    print(
        f"\nWrote {REPORT_PATH}, {METRICS_PATH}, {HOURLY_PATH}, {POOLED_PATH}, {SUMMARY_PATH}, and "
        f"{figures} figures to {FIG_DIR}/"
    )
    print(f"README results block: {'rewritten' if synced else 'NOT found — markers missing'}")
    print(
        f"Interval coverage: {coverage['empirical']:.1f}% empirical vs "
        f"{coverage['nominal']:.0f}% nominal over {coverage['hours_scored']:,} hours"
    )
    print(summary.to_string(float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
