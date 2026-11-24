#!/usr/bin/env python3
"""What does the champion rely on when it actually forecasts?

The benchmark's permutation importance shuffles features on a held-out week of rows
whose lags are *observed*. That is one-step-ahead forecasting. The shipped model
forecasts 24 hours recursively: from the second step on, `lag_1h` is the model's own
previous prediction, not a measurement. So importance measured one step ahead describes
a setting the model is never served in.

This script explains the rows the model really predicted from. At every benchmark
origin (the latest year, daily) it fits the champion on `timestamp <= origin`, runs the
24-hour recursive forecast, keeps the feature row behind each step
(`forecast_with_rows`), and computes exact TreeSHAP values for those rows. SHAP values
add up to each prediction, so they can be split by horizon, by feature and by day type.
Permutation importance is recomputed with the benchmark's recipe for comparison.

    PYTHONPATH=src python scripts/explain.py              # ~5 min
    PYTHONPATH=src python scripts/explain.py --report-only

Descriptive only: no decision depends on it, so no rule is fixed in advance.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")  # report script: never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402

from energy_forecast.data import load_clean_demand  # noqa: E402
from energy_forecast.features import FEATURE_COLUMNS, add_features, german_holidays  # noqa: E402
from energy_forecast.models import MODEL_VERSION, HistGradientBoostingForecast  # noqa: E402
from energy_forecast.theme import BLUE, GREEN, INK, MUTED, ORANGE  # noqa: E402
from energy_forecast.theme import style_axes as _style_axes  # noqa: E402

DATA_PATH = Path("data/clean/demand_hourly.csv")
METRICS_PATH = Path("reports/benchmark_metrics.csv")  # its origins are the ones explained
CHAMPION_PATH = Path("reports/benchmark_champion_hourly.csv")
CACHE = Path("data/clean/explain_shap.csv")  # gitignored by data/clean/*.csv
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/explainability.md")
SUMMARY_PATH = Path("reports/explainability_summary.json")

# Feature groups for the horizon view: lags carry the level, calendar the shape.
LAST_HOUR = "Last hour (lag_1h)"
GROUPS = {
    LAST_HOUR: ["lag_1h"],
    "Same hour yesterday / last week": ["lag_24h", "lag_168h"],
    "Rolling means": ["rolling_24h_mean", "rolling_7d_mean"],
    "Calendar": ["hour", "weekday", "month", "is_weekend", "is_public_holiday",
                 "is_dst_transition"],
}
GROUP_COLORS = dict(zip(GROUPS, [BLUE, ORANGE, GREEN, MUTED]))
SHAP_COLUMNS = [f"shap_{c}" for c in FEATURE_COLUMNS]


def explain_origin(data: pd.DataFrame, origin: pd.Timestamp, horizon: int = 24) -> pd.DataFrame:
    """Forecast rows, predictions and SHAP values for one origin, one row per step."""
    import shap

    model = HistGradientBoostingForecast().fit(data[data.timestamp <= origin])
    target = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="1h", tz="UTC")
    predictions, rows = model.forecast_with_rows(target)
    explanation = shap.TreeExplainer(model.regressor)(rows[FEATURE_COLUMNS])
    out = pd.DataFrame({"origin": origin, "timestamp": target,
                        "step": np.arange(1, horizon + 1), "predicted": predictions})
    out["base_value"] = np.asarray(explanation.base_values, dtype=float)
    out[SHAP_COLUMNS] = explanation.values
    out[FEATURE_COLUMNS] = rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    return out


def permutation_reference(data: pd.DataFrame, validation_hours: int = 24 * 7) -> pd.Series:
    """The benchmark's recipe: one-step rows of a held-out trailing week, observed lags."""
    from sklearn.inspection import permutation_importance

    model = HistGradientBoostingForecast().fit(data.iloc[:-validation_hours])
    featured = add_features(data).dropna(subset=FEATURE_COLUMNS + ["demand_mw"])
    validation = featured.iloc[-validation_hours:]
    result = permutation_importance(
        model.regressor, validation[FEATURE_COLUMNS], validation["demand_mw"],
        n_repeats=20, random_state=42, scoring="neg_mean_absolute_error",
    )
    return pd.Series(result.importances_mean, index=FEATURE_COLUMNS)


def group_share(shap_frame: pd.DataFrame) -> pd.DataFrame:
    """Share of total |SHAP| per feature group at each step (rows sum to 1)."""
    absolute = shap_frame[SHAP_COLUMNS].abs()
    grouped = pd.DataFrame({
        name: absolute[[f"shap_{c}" for c in cols]].sum(axis=1) for name, cols in GROUPS.items()
    })
    grouped["step"] = shap_frame.step.to_numpy()
    per_step = grouped.groupby("step").mean()
    return per_step.div(per_step.sum(axis=1), axis=0)


def error_by_step(path: Path = CHAMPION_PATH) -> pd.Series:
    """Champion MAE by hours ahead, from the committed benchmark."""
    hourly = pd.read_csv(path)
    origin = pd.to_datetime(hourly.origin, utc=True)
    step = ((pd.to_datetime(hourly.timestamp, utc=True) - origin) / pd.Timedelta(hours=1))
    return (hourly.predicted - hourly.demand_mw).abs().groupby(step.astype(int)).mean()


def plot_importance(shap_mean: pd.Series, perm: pd.Series, out: Path) -> None:
    order = shap_mean.sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    axes[0].barh(range(len(order)), shap_mean[order], color=BLUE, zorder=3)
    axes[0].set_title("Mean |SHAP| on forecast-time rows", loc="left", fontsize=10)
    axes[0].set_xlabel("MW")
    axes[1].barh(range(len(order)), perm[order].clip(lower=0), color=MUTED, zorder=3)
    axes[1].set_title("Permutation importance, one step ahead", loc="left", fontsize=10)
    axes[1].set_xlabel("Rise in MAE when shuffled (MW)")
    axes[0].set_yticks(range(len(order)), list(order))
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_horizon(shares: pd.DataFrame, mae: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = np.zeros(len(shares))
    for name in GROUPS:
        ax.bar(shares.index, shares[name], bottom=bottom, color=GROUP_COLORS[name], label=name,
               width=0.85, zorder=3)
        bottom += shares[name].to_numpy()
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("Share of |SHAP|")
    ax.set_ylim(0, 1)
    twin = ax.twinx()
    twin.plot(mae.index, mae.to_numpy(), color=INK, linewidth=1.8, marker="o", markersize=3,
              label="MAE (right axis)")
    twin.set_ylabel("Champion MAE (MW)")
    twin.set_ylim(0, mae.max() * 1.15)
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels, frameon=False, fontsize=8,
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("What the forecast leans on, and how error grows, by hours ahead", loc="left")
    _style_axes(ax)
    twin.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_beeswarm(shap_frame: pd.DataFrame, out: Path) -> None:
    import shap

    sample = shap_frame.sample(min(len(shap_frame), 4000), random_state=42)
    explanation = shap.Explanation(
        values=sample[SHAP_COLUMNS].to_numpy(), base_values=sample.base_value.to_numpy(),
        data=sample[FEATURE_COLUMNS].to_numpy(), feature_names=FEATURE_COLUMNS,
    )
    shap.plots.beeswarm(explanation, max_display=len(FEATURE_COLUMNS), show=False,
                        plot_size=(8, 5))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(7))
    plt.xlabel("SHAP value (MW pushed onto the prediction)")
    plt.title("SHAP values on forecast-time rows (sample of 4,000 steps)", loc="left",
              fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-origins", type=int, default=None, help="cap origins (smoke)")
    parser.add_argument("--report-only", action="store_true",
                        help=f"rebuild from {CACHE} without refitting")
    args = parser.parse_args()

    data = load_clean_demand(DATA_PATH)
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    if args.report_only:
        if not CACHE.exists():
            parser.error(f"{CACHE} not found; run once without --report-only")
        shap_frame = pd.read_csv(CACHE)
        shap_frame["origin"] = pd.to_datetime(shap_frame.origin, utc=True)
        shap_frame["timestamp"] = pd.to_datetime(shap_frame.timestamp, utc=True)
    else:
        origins = sorted(pd.to_datetime(pd.read_csv(METRICS_PATH).origin, utc=True).unique())
        if args.max_origins is not None:
            origins = origins[-args.max_origins:]
        print(f"Explaining {len(origins)} origins", flush=True)
        parts = []
        for i, origin in enumerate(origins, 1):
            parts.append(explain_origin(data, pd.Timestamp(origin)))
            if i % 50 == 0 or i == len(origins):
                print(f"  origin {i}/{len(origins)}", flush=True)
        shap_frame = pd.concat(parts, ignore_index=True)
        shap_frame.to_csv(CACHE, index=False)

    additivity = float((shap_frame[SHAP_COLUMNS].sum(axis=1) + shap_frame.base_value
                        - shap_frame.predicted).abs().max())
    shap_mean = shap_frame[SHAP_COLUMNS].abs().mean()
    shap_mean.index = FEATURE_COLUMNS
    perm = permutation_reference(data)
    shares = group_share(shap_frame)
    mae = error_by_step()
    first, last = shares.iloc[0], shares.iloc[-1]

    local = shap_frame.timestamp.dt.tz_convert("Europe/Berlin")
    on_holiday = local.dt.date.isin(german_holidays(set(local.dt.year.unique()))).to_numpy()
    actual = data.set_index("timestamp").demand_mw.reindex(pd.DatetimeIndex(shap_frame.timestamp))
    error = shap_frame.predicted.to_numpy() - actual.to_numpy()
    holiday = {
        "hours": int(on_holiday.sum()),
        "mean_holiday_shap_mw": float(shap_frame.loc[on_holiday, "shap_is_public_holiday"].mean()),
        "mean_error_mw": float(np.nanmean(error[on_holiday])),
        "mean_error_other_mw": float(np.nanmean(error[~on_holiday])),
    }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_importance(shap_mean, perm, FIG_DIR / "explain_importance.png")
    plot_horizon(shares, mae, FIG_DIR / "explain_by_horizon.png")
    plot_beeswarm(shap_frame, FIG_DIR / "explain_beeswarm.png")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    origins_n = shap_frame.origin.nunique()
    summary = {
        "generated": generated, "model_version": MODEL_VERSION, "origins": int(origins_n),
        "first_origin": str(shap_frame.origin.min()), "last_origin": str(shap_frame.origin.max()),
        "additivity_max_abs_mw": additivity,
        "mean_abs_shap_mw": shap_mean.round(1).to_dict(),
        "permutation_importance_mw": perm.round(1).to_dict(),
        "group_share_by_step": shares.round(3).to_dict(orient="index"),
        "mae_by_step_mw": mae.round(1).to_dict(), "holidays": holiday,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    rank = shap_mean.sort_values(ascending=False)
    lag_drop = first[LAST_HOUR] - last[LAST_HOUR]
    horizon_text = (
        f"At one hour ahead `lag_1h` carries {first[LAST_HOUR]:.0%} of the attribution; at 24 "
        f"hours it carries {last[LAST_HOUR]:.0%}, and the calendar features go from "
        f"{first['Calendar']:.0%} to {last['Calendar']:.0%}. "
    )
    if lag_drop < 0.10:
        horizon_text += (
            "So the model leans on its own previous output almost as hard at the end of the "
            "day as at the start. An early miss is carried forward, which is consistent with "
            f"MAE rising from {mae.iloc[0]:,.0f} MW one hour ahead to {mae.iloc[-1]:,.0f} MW at "
            "24 hours, and with the prediction band under-covering the last hours of the day "
            "(see `interval_study.md`)."
        )
    else:
        horizon_text += (
            "So the model shifts weight away from its own previous output as the horizon grows, "
            f"while MAE rises from {mae.iloc[0]:,.0f} MW one hour ahead to {mae.iloc[-1]:,.0f} "
            "MW at 24 hours."
        )
    # The flag's push as a share of the gap it would leave without it: push / (push + error).
    gap = abs(holiday["mean_holiday_shap_mw"]) + holiday["mean_error_mw"]
    holiday_share = abs(holiday["mean_holiday_shap_mw"]) / gap if gap else float("nan")
    origin_clock = f"{shap_frame.origin.dt.hour.mode().iloc[0]:02d}:00"
    report = [
        "# What the forecast relies on",
        "",
        f"Generated {generated} by `scripts/explain.py` (Python "
        f"{summary['versions']['python']}, scikit-learn {summary['versions']['scikit-learn']}, "
        f"model {MODEL_VERSION}).",
        "",
        "The benchmark's permutation importance shuffles features on rows whose lags are "
        "observed, i.e. one step ahead. The served forecast is recursive: from the second hour "
        "on, `lag_1h` is the model's own previous prediction. This report explains the rows the "
        f"model really predicted from, at {origins_n} daily origins from "
        f"{shap_frame.origin.min():%Y-%m-%d} to {shap_frame.origin.max():%Y-%m-%d} (the "
        "benchmark's origins), with exact TreeSHAP values. They add up to each prediction "
        f"(largest gap {additivity:.1e} MW).",
        "",
        "## Importance, two ways",
        "",
        "| Feature | Mean \\|SHAP\\| at forecast time (MW) | Permutation, one step ahead (MW) |",
        "|---|---|---|",
        *[f"| `{f}` | {rank[f]:,.0f} | {perm[f]:,.0f} |" for f in rank.index],
        "",
        "![Importance two ways](figures/explain_importance.png)",
        "",
        "The two methods answer different questions. SHAP says how far each feature moves a "
        "prediction away from the average one; permutation says how much accuracy is lost "
        "without it, on one held-out week. The difference that matters is what `lag_1h` *is*: "
        "in the permutation test it is a measured value; at forecast time it is a measurement "
        "only at step 1 and the model's own output afterwards.",
        "",
        "## By hours ahead",
        "",
        "Share of total |SHAP| per feature group at each step, with the champion's MAE by step "
        "from the committed benchmark:",
        "",
        "![SHAP share and error by horizon](figures/explain_by_horizon.png)",
        "",
        "| Hours ahead | " + " | ".join(GROUPS) + " | MAE (MW) |",
        "|---" * (len(GROUPS) + 2) + "|",
        *[f"| {step} | " + " | ".join(f"{shares.loc[step, g]:.0%}" for g in GROUPS)
          + f" | {mae.get(step, float('nan')):,.0f} |" for step in (1, 6, 12, 18, 24)],
        "",
        horizon_text,
        "",
        f"One caveat: every benchmark origin is at {origin_clock} UTC, so hours ahead and time "
        "of day move together here. The calendar share peaking mid-horizon is most likely the "
        "evening and night hours, where the hour-of-day shape matters most, rather than a "
        "horizon effect.",
        "",
        "## Direction of each effect",
        "",
        "Each dot is one forecast step; colour is the feature's value, position its push on the "
        "prediction.",
        "",
        "![Beeswarm](figures/explain_beeswarm.png)",
        "",
        "## Public holidays",
        "",
        f"Over {holiday['hours']:,} forecast hours that fall on a public holiday, "
        "`is_public_holiday` moves the prediction by "
        f"{holiday['mean_holiday_shap_mw']:+,.0f} MW on average, while the forecast still ends "
        f"up {holiday['mean_error_mw']:+,.0f} MW off (forecast minus actual; "
        f"{holiday['mean_error_other_mw']:+,.0f} MW on other hours).",
        "",
        *([
            "The holiday flag pushes in the right direction but closes only about "
            f"{holiday_share:.0%} of the gap it would otherwise leave. With two years of "
            "training data each holiday has been seen at most twice.",
            "",
        ] if holiday["mean_holiday_shap_mw"] < 0 < holiday["mean_error_mw"] else []),
        "## Files",
        "",
        f"- `{SUMMARY_PATH.name}`: every number in this report.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
