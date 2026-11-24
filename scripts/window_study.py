#!/usr/bin/env python3
"""How much history should the forecaster train on?

SMARD publishes German grid load back to 2015, but the shipped snapshot keeps 52 weeks.
More history means every season and every public holiday is seen several times. It also
means training across the 2022-23 step down in demand, after which the old level never
came back. This script measures which effect wins.

Design, fixed before any result was seen:

* One model, the shipped gradient-boosting champion with its default parameters. Only
  the training window changes, so any difference is the window's doing.
* Daily origins over the last two years. At each origin every window trains on the
  same end point and forecasts the same 24 hours, so errors are paired by origin.
* Training slices are time based (`timestamp <= origin`), never positional, following
  the rule in `verification.replay_origins`.
* Decision rule: a longer window replaces the one-year baseline only if its paired
  bootstrap 95% CI on mean per-origin MAE lies entirely below zero AND its Wilcoxon p
  survives a Holm correction across the four comparisons. Otherwise keep one year.

    PYTHONPATH=src python scripts/window_study.py            # reuses the cached history
    PYTHONPATH=src python scripts/window_study.py --refresh  # refetch every SMARD chunk

The long history is cached at data/clean/demand_history.csv, which is gitignored; only
the report, figures, per-origin CSV and summary JSON are meant to be committed.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")  # report script: never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402

from energy_forecast.data import load_clean_demand  # noqa: E402
from energy_forecast.features import german_holidays  # noqa: E402
from energy_forecast.inference import paired_model_comparison  # noqa: E402
from energy_forecast.models import HistGradientBoostingForecast  # noqa: E402
from energy_forecast.quality import validate_demand  # noqa: E402
from energy_forecast.smard_api import SMARDAPIClient  # noqa: E402
from energy_forecast.theme import INK, MUTED  # noqa: E402
from energy_forecast.theme import style_axes as _style_axes  # noqa: E402

HISTORY_PATH = Path("data/clean/demand_history.csv")
# Every hourly prediction of the last full run, so the report can be rebuilt without
# refitting. Gitignored by data/clean/*.csv like the history it was computed from.
HOURLY_CACHE = Path("data/clean/window_study_hourly.csv")
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/window_study.md")
ORIGINS_PATH = Path("reports/window_study_origins.csv")
SUMMARY_PATH = Path("reports/window_study_summary.json")

# Label -> training window in days; None means everything before the origin.
WINDOWS: dict[str, int | None] = {
    "1 year": 365,
    "2 years": 730,
    "3 years": 1095,
    "5 years": 1826,
    "All (since 2015)": None,
}
BASELINE = "1 year"
MIN_PAIRED = 5  # below this a paired comparison is not worth printing
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def load_history(refresh: bool) -> pd.DataFrame:
    """The full SMARD series, fetched once and cached; the quality gate runs on every fetch."""
    if refresh or not HISTORY_PATH.exists():
        print("Fetching every weekly SMARD chunk ...", flush=True)
        data, urls = SMARDAPIClient().fetch_latest(weeks=10_000)
        validate_demand(data).raise_for_errors()
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(HISTORY_PATH, index=False)
        print(f"Cached {len(data):,} hours from {len(urls)} chunks to {HISTORY_PATH}")
    return load_clean_demand(HISTORY_PATH)


def evaluation_origins(
    data: pd.DataFrame, eval_days: int, step_hours: int = 24, horizon_hours: int = 24
) -> list[pd.Timestamp]:
    """Origins spaced `step_hours` apart over the last `eval_days`, each with a full horizon."""
    observed = data.dropna(subset=["demand_mw"]).timestamp
    last_origin = observed.max() - pd.Timedelta(hours=horizon_hours)
    first_origin = last_origin - pd.Timedelta(days=eval_days)
    return list(pd.date_range(first_origin, last_origin, freq=f"{step_hours}h"))


def window_slice(data: pd.DataFrame, origin: pd.Timestamp, days: int | None) -> pd.DataFrame:
    """Rows at or before `origin`, limited to the trailing `days` (None = all of them)."""
    mask = data.timestamp <= origin
    if days is not None:
        mask &= data.timestamp > origin - pd.Timedelta(days=days)
    return data[mask]


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m, running, adjusted = len(ordered), 0.0, {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def run(data: pd.DataFrame, origins: list[pd.Timestamp], horizon_hours: int) -> pd.DataFrame:
    """Hourly predictions for every (origin, window). Unpublished hours are dropped."""
    actuals = data.set_index("timestamp").demand_mw
    parts = []
    for i, origin in enumerate(origins, 1):
        target = pd.date_range(
            origin + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h", tz="UTC"
        )
        for label, days in WINDOWS.items():
            # The study does its own slicing, so the model's shipped window is switched
            # off; otherwise every arm longer than it would be cut back to it silently.
            model = HistGradientBoostingForecast(train_window_days=None).fit(
                window_slice(data, origin, days)
            )
            parts.append(pd.DataFrame({
                "origin": origin, "window": label, "timestamp": target,
                "actual": actuals.reindex(target).to_numpy(dtype=float),
                "predicted": model.predict(target),
            }))
        if i % 25 == 0 or i == len(origins):
            print(f"  origin {i}/{len(origins)}  {origin:%Y-%m-%d}", flush=True)
    hourly = pd.concat(parts, ignore_index=True).dropna(subset=["actual"])
    hourly["error"] = hourly.predicted - hourly.actual
    hourly["abs_error"] = hourly.error.abs()
    local = hourly.timestamp.dt.tz_convert("Europe/Berlin")
    holidays = german_holidays(set(local.dt.year.unique()))
    hourly["is_holiday"] = local.dt.date.isin(holidays)
    return hourly


def compare(per_origin: pd.DataFrame, value: str) -> dict[str, dict]:
    """Paired comparison of every window against the baseline on one per-origin column."""
    wide = per_origin.pivot(index="origin", columns="window", values=value).dropna()
    if len(wide) < MIN_PAIRED or set(wide.columns) != set(WINDOWS):
        return {}  # e.g. a smoke run whose few origins contain no holiday hour
    results = {
        label: paired_model_comparison(wide[label], wide[BASELINE])
        for label in WINDOWS if label != BASELINE
    }
    adjusted = holm({label: r["wilcoxon_p"] for label, r in results.items()})
    for label, result in results.items():
        result["holm_p"] = adjusted[label]
        result["adopt"] = bool(result["ci_upper"] < 0 and adjusted[label] < 0.05)
    return results


def winter_label(origins: pd.Series) -> pd.Series:
    """'Winter 2024/25' for December to February origins (Berlin time), else None."""
    local = pd.to_datetime(origins, utc=True).dt.tz_convert("Europe/Berlin")
    start = np.where(local.dt.month == 12, local.dt.year, local.dt.year - 1)
    labels = [f"Winter {y}/{(y + 1) % 100:02d}" for y in start]
    return pd.Series(np.where(local.dt.month.isin([12, 1, 2]), labels, None), index=origins.index)


def stability(per_origin: pd.DataFrame) -> list[tuple[str, dict[str, dict]]]:
    """The headline comparison re-run on halves of the period and on each winter.

    The decision uses the pooled test. This checks whether the gain holds everywhere or
    rests on one part of the period, which the pooled number cannot show.
    """
    origins = per_origin.origin
    midpoint = origins.min() + (origins.max() - origins.min()) / 2
    winter = winter_label(origins)
    subsets = [
        ("First half", origins < midpoint),
        ("Second half (most recent)", origins >= midpoint),
        *[(label, winter == label) for label in sorted(winter.dropna().unique())],
        ("Outside winter", winter.isna()),
    ]
    rows = []
    for name, mask in subsets:
        results = compare(per_origin[mask.to_numpy()], "mae")
        if results:
            rows.append((name, results))
    return rows


def stability_table(rows: list[tuple[str, dict[str, dict]]]) -> list[str]:
    labels = [label for label in WINDOWS if label != BASELINE]
    lines = [
        "| Subset | Origins | " + " | ".join(f"{label} vs {BASELINE}" for label in labels) + " |",
        "|---" * (len(labels) + 2) + "|",
    ]
    for name, results in rows:
        n = results[labels[0]]["n"]
        cells = []
        for label in labels:
            r = results[label]
            mark = " *" if r["ci_upper"] < 0 or r["ci_lower"] > 0 else ""
            ci = f"[{r['ci_lower']:+,.0f}, {r['ci_upper']:+,.0f}]"
            cells.append(f"{r['mean_diff']:+,.0f} {ci}{mark}")
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    return lines


def plot_annual_level(data: pd.DataFrame, out: Path) -> pd.Series:
    """Mean demand per full calendar year: the level shift the study is about."""
    local = data.assign(year=data.timestamp.dt.tz_convert("Europe/Berlin").dt.year)
    counts = local.groupby("year").demand_mw.count()
    full = counts[counts >= 24 * 365].index
    level = local[local.year.isin(full)].groupby("year").demand_mw.mean() / 1000
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(level.index.astype(str), level.to_numpy(), color=MUTED, width=0.7)
    ax.set_ylim(level.min() - 3, level.max() + 1.5)
    for x, y in zip(level.index.astype(str), level.to_numpy()):
        ax.text(x, y + 0.2, f"{y:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylabel("Mean grid load (GW)")
    ax.set_title("German grid load by year (SMARD module 410)", loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return level


def window_colors() -> dict[str, tuple]:
    """Window length is ordered, so it gets a sequential ramp, darker = longer."""
    ramp = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(WINDOWS)))
    return dict(zip(WINDOWS, ramp))


def plot_effects(results: dict[str, dict], out: Path) -> None:
    colors = window_colors()
    labels = list(results)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for y, label in enumerate(labels):
        r = results[label]
        err = [[r["mean_diff"] - r["ci_lower"]], [r["ci_upper"] - r["mean_diff"]]]
        ax.errorbar(r["mean_diff"], y, xerr=err, fmt="o", color=colors[label],
                    ecolor=colors[label], capsize=4, markersize=7)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean per-origin MAE minus the {BASELINE} window (MW); left of 0 is better")
    ax.set_title("Effect of training on more history, with 95% bootstrap CIs", loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_over_time(per_origin: pd.DataFrame, out: Path, smooth: int = 30) -> None:
    colors = window_colors()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for label in WINDOWS:
        series = per_origin[per_origin.window == label].sort_values("origin")
        ax.plot(series.origin, series.mae.rolling(smooth, min_periods=smooth).mean(),
                color=colors[label], linewidth=1.6, label=label)
    ax.set_ylabel("MAE (MW)")
    ax.set_title(f"Forecast error over time ({smooth}-origin trailing mean)", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=len(WINDOWS), loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def comparison_table(results: dict[str, dict], baseline_mean: float, unit: str) -> list[str]:
    lines = [
        f"| Window | Mean {unit} (MW) | vs {BASELINE} (MW) | 95% CI | Wilcoxon p | Holm p | "
        "Clears the rule |",
        "|---|---|---|---|---|---|---|",
        f"| {BASELINE} (baseline) | {baseline_mean:,.0f} | | | | | |",
    ]
    for label, r in results.items():
        lines.append(
            f"| {label} | {baseline_mean + r['mean_diff']:,.0f} | {r['mean_diff']:+,.0f} | "
            f"[{r['ci_lower']:+,.0f}, {r['ci_upper']:+,.0f}] | {fmt_p(r['wilcoxon_p'])} | "
            f"{fmt_p(r['holm_p'])} | {'yes' if r['adopt'] else 'no'} |"
        )
    return lines


def pivot_table(frame: pd.DataFrame, rows: str, title: str, order: list | None = None) -> list[str]:
    table = frame.pivot_table(index=rows, columns="window", values="abs_error", aggfunc="mean")
    table = table[list(WINDOWS)]
    if order is not None:
        table = table.reindex([o for o in order if o in table.index])
    lines = [f"| {title} | " + " | ".join(WINDOWS) + " |", "|---" * (len(WINDOWS) + 1) + "|"]
    for index, row in table.iterrows():
        lines.append(f"| {index} | " + " | ".join(f"{v:,.0f}" for v in row) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="refetch the full SMARD history")
    parser.add_argument("--eval-days", type=int, default=730)
    parser.add_argument("--step-hours", type=int, default=24)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--max-origins", type=int, default=None, help="cap origins (smoke test)")
    parser.add_argument(
        "--report-only", action="store_true",
        help=f"rebuild the report from {HOURLY_CACHE} instead of refitting (~1 h saved)",
    )
    args = parser.parse_args()

    data = load_history(args.refresh)
    if args.report_only:
        if not HOURLY_CACHE.exists():
            parser.error(f"{HOURLY_CACHE} not found; run once without --report-only")
        hourly = pd.read_csv(HOURLY_CACHE, parse_dates=["origin", "timestamp"])
        origins = sorted(hourly.origin.unique())
        print(f"Rebuilding the report from {HOURLY_CACHE} ({len(origins)} origins)")
    else:
        origins = evaluation_origins(data, args.eval_days, args.step_hours, args.horizon_hours)
        if args.max_origins is not None:
            origins = origins[-args.max_origins:]
        longest = max(d for d in WINDOWS.values() if d is not None)
        if origins[0] - pd.Timedelta(days=longest) < data.timestamp.min():
            parser.error("history too short for the longest window at the first origin")
        print(f"{len(data):,} hours, {data.timestamp.min():%Y-%m-%d} to "
              f"{data.timestamp.max():%Y-%m-%d}; {len(origins)} origins x {len(WINDOWS)} windows")
        hourly = run(data, origins, args.horizon_hours)
        hourly.to_csv(HOURLY_CACHE, index=False)
    per_origin = (
        hourly.groupby(["origin", "window"])
        .agg(mae=("abs_error", "mean"), bias=("error", "mean"))
        .reset_index()
    )
    holiday_origin = (
        hourly[hourly.is_holiday].groupby(["origin", "window"]).abs_error.mean()
        .rename("holiday_mae").reset_index()
    )
    results = compare(per_origin, "mae")
    holiday_results = compare(holiday_origin, "holiday_mae")
    stable = stability(per_origin)
    mean_mae = per_origin.groupby("window").mae.mean()
    mean_bias = per_origin.groupby("window").bias.mean()
    holiday_mean = holiday_origin.groupby("window").holiday_mae.mean()

    winners = [label for label, r in results.items() if r["adopt"]]
    chosen = min(winners, key=lambda label: mean_mae[label]) if winners else BASELINE

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    level = plot_annual_level(data, FIG_DIR / "window_study_annual_level.png")
    plot_effects(results, FIG_DIR / "window_study_effects.png")
    plot_over_time(per_origin, FIG_DIR / "window_study_over_time.png")

    local = hourly.timestamp.dt.tz_convert("Europe/Berlin")
    hourly["season"] = local.dt.month.map(SEASONS)
    midpoint = origins[0] + (origins[-1] - origins[0]) / 2
    hourly["eval_year"] = np.where(hourly.origin < midpoint, "First half", "Second half")
    per_origin.to_csv(ORIGINS_PATH, index=False)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if holiday_results:
        holiday_n = holiday_results[next(iter(holiday_results))]["n"]
        holiday_section = [
            "Holidays are where one year of history is thinnest: the model has seen each one "
            f"at most once. Holiday-hour MAE, paired over the {holiday_n} origins whose "
            "forecast window contains a holiday hour:",
            "",
            *comparison_table(holiday_results, holiday_mean[BASELINE], "holiday-hour MAE"),
            "",
            "With this few paired origins the holiday result is indicative. It is not part of "
            "the decision rule.",
        ]
    else:
        holiday_section = ["Too few origins with a holiday hour to compare."]
    summary = {
        "generated": generated,
        "history_start": str(data.timestamp.min()), "history_end": str(data.timestamp.max()),
        "hours": int(len(data)), "origins": len(origins),
        "first_origin": str(origins[0]), "last_origin": str(origins[-1]),
        "windows_days": WINDOWS, "baseline": BASELINE,
        "mean_mae": mean_mae.round(1).to_dict(), "mean_bias": mean_bias.round(1).to_dict(),
        "holiday_mae": holiday_mean.round(1).to_dict(),
        "comparisons": results, "holiday_comparisons": holiday_results,
        "stability": dict(stable),
        "chosen": chosen, "annual_mean_gw": level.round(2).to_dict(),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    if chosen == BASELINE:
        verdict = (
            f"**Verdict: keep the {BASELINE} window.** No longer window met the rule set "
            "before the run (CI entirely below zero and Holm-adjusted Wilcoxon p < 0.05)."
        )
    else:
        r = results[chosen]
        verdict = (
            f"**Verdict: train on {chosen.lower()}.** It lowers mean per-origin MAE by "
            f"{-r['mean_diff']:,.0f} MW against {BASELINE} (95% CI [{r['ci_lower']:+,.0f}, "
            f"{r['ci_upper']:+,.0f}], Holm p = {fmt_p(r['holm_p'])}), the largest gain among "
            "the windows that met the rule set before the run."
        )

    report = [
        "# Training-window study",
        "",
        f"Generated {generated} by `scripts/window_study.py` (Python "
        f"{summary['versions']['python']}, scikit-learn {summary['versions']['scikit-learn']}).",
        "",
        "Question: does the gradient-boosting forecaster do better trained on one year of "
        "history, or on more? More history shows the model every season and holiday several "
        "times. It also spans the 2022-23 drop in German demand, so older rows describe a "
        "level the grid no longer runs at.",
        "",
        "## The data",
        "",
        f"SMARD module 410, {len(data):,} hourly observations from "
        f"{data.timestamp.min():%Y-%m-%d} to {data.timestamp.max():%Y-%m-%d}, all passing "
        "the project's ingest quality gate.",
        "",
        "![Mean grid load by year](figures/window_study_annual_level.png)",
        "",
        f"Mean load peaked at {level.max():.1f} GW in {level.idxmax()} and was lowest at "
        f"{level.min():.1f} GW in {level.idxmin()}.",
        "",
        "## Design",
        "",
        f"- {len(origins)} origins, one every {args.step_hours} hours from "
        f"{origins[0]:%Y-%m-%d} to {origins[-1]:%Y-%m-%d}, each scored on the next "
        f"{args.horizon_hours} hours.",
        "- At each origin, one model per window: " + ", ".join(WINDOWS) + ". Same model, "
        "same default parameters, same forecast hours. Only the start of the training slice "
        "moves.",
        "- Per-origin MAE is the sampling unit, as in the main benchmark, because the 24 hours "
        "of one forecast share weather and demand level.",
        "- The rule, set before the run: a window replaces the one-year baseline only if the "
        "paired bootstrap 95% CI on its MAE difference lies entirely below zero and its "
        "Wilcoxon p stays under 0.05 after a Holm correction for the four comparisons.",
        "",
        "## Result",
        "",
        *comparison_table(results, mean_mae[BASELINE], "per-origin MAE"),
        "",
        verdict,
        "",
        "![Effect of each window against one year](figures/window_study_effects.png)",
        "",
        "![Error over time by window](figures/window_study_over_time.png)",
        "",
        "## Stability",
        "",
        "The decision uses the pooled test above. This table re-runs the same paired "
        "comparison on parts of the period, to check whether the gain holds throughout or "
        "rests on one part of it. Each cell is the mean per-origin MAE difference against "
        f"{BASELINE} in MW, with its 95% bootstrap CI; * marks a CI that excludes zero. "
        "These are not part of the decision rule.",
        "",
        *stability_table(stable),
        "",
        "## Bias",
        "",
        "Mean signed error per window (forecast minus actual, MW). If the older, higher-demand "
        "years pulled forecasts up, the longer windows would show a positive bias.",
        "",
        "| Window | Mean bias (MW) |",
        "|---|---|",
        *[f"| {label} | {mean_bias[label]:+,.0f} |" for label in WINDOWS],
        "",
        "## Public holidays",
        "",
        *holiday_section,
        "",
        "## By season and half of the evaluation period",
        "",
        "Hourly MAE (MW). Descriptive only; the paired test above is the decision.",
        "",
        *pivot_table(hourly, "season", "Season", ["Winter", "Spring", "Summer", "Autumn"]),
        "",
        *pivot_table(hourly, "eval_year", "Evaluation period"),
        "",
        "## Files",
        "",
        f"- `{ORIGINS_PATH.name}`: per-origin MAE and bias for every window.",
        f"- `{SUMMARY_PATH.name}`: every number in this report.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}; chosen window: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
