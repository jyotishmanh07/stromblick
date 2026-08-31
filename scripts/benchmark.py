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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_forecast.data import load_clean_demand
from energy_forecast.evaluation import error_slices, metrics
from energy_forecast.features import FEATURE_COLUMNS, add_features
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
SUMMARY_PATH = Path("reports/benchmark_summary.json")

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
    for i, origin in enumerate(ordered):
        if i < min_prior_origins:
            continue
        window = ordered[max(0, i - trailing_origins) : i]
        prior = pooled_champion[pooled_champion.origin.isin(window)]
        current = pooled_champion[pooled_champion.origin == origin]
        spread = float(np.quantile(np.abs(prior.demand_mw - prior.predicted), residual_quantile))
        actual = current.demand_mw.to_numpy()
        predicted = current.predicted.to_numpy()
        inside += int(np.sum((actual >= predicted - spread) & (actual <= predicted + spread)))
        scored += len(actual)
        widths.append(2 * spread)
    return {
        "nominal": 100 * residual_quantile,
        "empirical": 100 * inside / scored if scored else float("nan"),
        "hours_scored": scored,
        "mean_band_mw": float(np.mean(widths)) if widths else float("nan"),
    }


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
    plot_mae_by_origin(per_origin, FIG_DIR / "benchmark_mae_by_origin.png")
    notes = plot_error_slices(pooled[champion], champion, FIG_DIR / "benchmark_error_slices.png")
    coverage = interval_coverage(pooled[champion])
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
        "The gradient-boosting settings in `models.py` "
        "(`learning_rate=0.06`, `max_iter=250`, `max_leaf_nodes=31`, `l2_regularization=1.0`) "
        "were checked with a seven-config rolling-origin sweep over 30 daily origins, varying "
        "learning rate (0.03–0.10), tree count (150–500), leaf count (15–63), and L2 penalty "
        "(0.1–1.0). Mean 24-hour MAE moved by under 5% across every config and all of them sat "
        "inside one standard deviation of each other, so the defaults are kept: nothing in the "
        "neighbourhood beat them by a margin the backtest could resolve.",
        "",
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
                "top_features": [list(pair) for pair in (importance or {}).get("top_features", [])],
                "worst_hour": int(notes["worst_hour"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figures = 3 + (importance is not None)
    print(
        f"\nWrote {REPORT_PATH}, {METRICS_PATH}, {HOURLY_PATH}, {SUMMARY_PATH}, and "
        f"{figures} figures to {FIG_DIR}/"
    )
    print(
        f"Interval coverage: {coverage['empirical']:.1f}% empirical vs "
        f"{coverage['nominal']:.0f}% nominal over {coverage['hours_scored']:,} hours"
    )
    print(summary.to_string(float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
