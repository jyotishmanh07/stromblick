#!/usr/bin/env python3
"""Does the model beat the grid operators' own day-ahead forecast?

SMARD publishes a second load series, chart-data module 411, that is not in the
bundesAPI documentation. It is identified here from the data: it runs to the end of the
next day while actual load (module 410) lags by about a day, and it tracks actual load
hour by hour. That is the shape of the transmission operators' day-ahead total load
forecast. Before 2022 it sits 1.2 to 3.5 GW below actual load every year, which looks
like a definition change rather than skill, so only the latest year is compared.

The hard part is fairness. The operators issue their forecast for day D at about
10:00 Berlin on D-1 (when EU transparency rules require the day-ahead load forecast),
with weather forecasts and real-time metering. This project sees actual load only once
SMARD publishes it, roughly a day late, and uses no weather. So the model is scored under
three information cutoffs, fixed before the run:

* Fresh data: origin at the last hour before D. Impossible in practice; an upper bound.
* At their issue time: origin 10:00 Berlin on D-1, about 38 hours ahead by the end of D.
  What the model could do with real-time actuals.
* With SMARD's lag: origin 24 hours before that, about 62 hours ahead. What this
  project could actually issue at the moment the operators issue theirs.

Plus a fixed 50/50 blend of the operators' forecast and the lagged model (no weights are
fitted, so nothing is tuned on the scored days).

Rule, fixed before the run: per-day MAE on D's hours is the unit. For each of the four
comparisons against the operators, a paired bootstrap 95% CI and Wilcoxon p, Holm-
corrected across the four. "Beats" needs the CI entirely below zero and Holm p < 0.05,
"worse" needs it entirely above zero and Holm p < 0.05; anything else is reported as
indistinguishable. The headline claim is the lagged comparison, because it is the only
one this project could actually deliver.

    PYTHONPATH=src python scripts/tso_study.py               # ~20 min
    PYTHONPATH=src python scripts/tso_study.py --report-only # from the cache
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
from window_study import holm, load_history  # noqa: E402

from energy_forecast.data import load_clean_demand  # noqa: E402
from energy_forecast.features import german_holidays  # noqa: E402
from energy_forecast.inference import paired_model_comparison  # noqa: E402
from energy_forecast.models import MODEL_VERSION, HistGradientBoostingForecast  # noqa: E402
from energy_forecast.quality import validate_demand  # noqa: E402
from energy_forecast.smard_api import SMARDAPIClient  # noqa: E402
from energy_forecast.theme import BLUE, GREEN, INK, MUTED, ORANGE  # noqa: E402
from energy_forecast.theme import style_axes as _style_axes  # noqa: E402

TSO_MODULE = 411
TSO_PATH = Path("data/clean/tso_forecast_history.csv")  # gitignored
HOURLY_CACHE = Path("data/clean/tso_study_hourly.csv")  # gitignored
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/tso_study.md")
DAYS_PATH = Path("reports/tso_study_days.csv")
SUMMARY_PATH = Path("reports/tso_study_summary.json")

BERLIN = "Europe/Berlin"
ISSUE_HOUR = 10  # Berlin local time on D-1
SMARD_LAG = pd.Timedelta(hours=24)

TSO = "Operators' day-ahead"
FRESH = "Model, fresh data (upper bound)"
AT_ISSUE = "Model at their issue time"
LAGGED = "Model with SMARD's lag"
BLEND = "50/50 blend (operators + lagged model)"
METHODS = [TSO, FRESH, AT_ISSUE, LAGGED, BLEND]
CANDIDATES = [FRESH, AT_ISSUE, LAGGED, BLEND]
COLUMN = {TSO: "tso", FRESH: "fresh", AT_ISSUE: "issue_time", LAGGED: "lagged", BLEND: "blend"}
COLORS = dict(zip(METHODS, [INK, MUTED, BLUE, ORANGE, GREEN]))


def load_tso(refresh: bool) -> pd.DataFrame:
    """SMARD module 411, fetched once and cached; the quality gate runs on every fetch."""
    if refresh or not TSO_PATH.exists():
        print(f"Fetching every weekly chunk of SMARD module {TSO_MODULE} ...", flush=True)
        data, _ = SMARDAPIClient(module_id=TSO_MODULE).fetch_latest(weeks=10_000)
        validate_demand(data).raise_for_errors()
        TSO_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(TSO_PATH, index=False)
    return load_clean_demand(TSO_PATH)


def day_cutoffs(day) -> dict:
    """UTC origins for one Berlin calendar day, plus the day's UTC hours.

    `day` is a date or a Berlin midnight. Working in Berlin wall-clock time first keeps
    the cut-offs right on DST days, which have 23 or 25 hours.
    """
    naive = pd.Timestamp(pd.Timestamp(day).date())
    start = naive.tz_localize(BERLIN)
    end = (naive + pd.Timedelta(days=1)).tz_localize(BERLIN)
    issue = (naive - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)).tz_localize(BERLIN)
    hours = pd.date_range(
        start.tz_convert("UTC"), end.tz_convert("UTC"), freq="1h", inclusive="left"
    )
    return {
        "hours": hours,
        "fresh": start.tz_convert("UTC") - pd.Timedelta(hours=1),
        "issue_time": issue.tz_convert("UTC"),
        "lagged": issue.tz_convert("UTC") - SMARD_LAG,
    }


def forecast_day(data: pd.DataFrame, origin: pd.Timestamp, hours: pd.DatetimeIndex) -> np.ndarray:
    """Recursive forecast from `origin` through the last of `hours`, returned for `hours`."""
    steps = pd.date_range(origin + pd.Timedelta(hours=1), hours[-1], freq="1h")
    model = HistGradientBoostingForecast().fit(data[data.timestamp <= origin])
    return pd.Series(model.predict(steps), index=steps).reindex(hours).to_numpy(dtype=float)


def run(actual: pd.DataFrame, tso: pd.DataFrame, days: list) -> pd.DataFrame:
    observed = actual.set_index("timestamp").demand_mw
    forecast = tso.set_index("timestamp").demand_mw
    parts = []
    for i, day in enumerate(days, 1):
        cut = day_cutoffs(day)
        hours = cut["hours"]
        frame = pd.DataFrame({
            "day": pd.Timestamp(day).date(), "timestamp": hours,
            "actual": observed.reindex(hours).to_numpy(dtype=float),
            "tso": forecast.reindex(hours).to_numpy(dtype=float),
        })
        for name in ("fresh", "issue_time", "lagged"):
            frame[name] = forecast_day(actual, cut[name], hours)
        parts.append(frame)
        if i % 25 == 0 or i == len(days):
            print(f"  day {i}/{len(days)}  {frame.day.iloc[0]}", flush=True)
    return pd.concat(parts, ignore_index=True)


def per_day(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per-day MAE and bias per method, over hours where every series has a value."""
    data = hourly.assign(blend=(hourly.tso + hourly.lagged) / 2)
    data = data.dropna(subset=["actual", *COLUMN.values()])
    rows = []
    for method, column in COLUMN.items():
        err = data[column] - data.actual
        frame = data.assign(abs_err=err.abs(), err=err).groupby("day").agg(
            mae=("abs_err", "mean"), bias=("err", "mean"), hours=("err", "size")
        )
        rows.append(frame.reset_index().assign(method=method))
    return pd.concat(rows, ignore_index=True)


def decide(days: pd.DataFrame) -> dict[str, dict]:
    wide = days.pivot(index="day", columns="method", values="mae").dropna()
    results = {m: paired_model_comparison(wide[m], wide[TSO]) for m in CANDIDATES}
    adjusted = holm({m: r["wilcoxon_p"] for m, r in results.items()})
    for m, r in results.items():
        r["holm_p"] = adjusted[m]
        if r["ci_upper"] < 0 and adjusted[m] < 0.05:
            r["verdict"] = "beats the operators"
        elif r["ci_lower"] > 0 and adjusted[m] < 0.05:
            r["verdict"] = "worse than the operators"
        else:
            r["verdict"] = "indistinguishable from the operators"
    return results


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def plot_by_hour(hourly: pd.DataFrame, out: Path) -> None:
    data = hourly.assign(blend=(hourly.tso + hourly.lagged) / 2).dropna()
    local_hour = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert(BERLIN).dt.hour
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for method, column in COLUMN.items():
        mae = (data[column] - data.actual).abs().groupby(local_hour).mean()
        ax.plot(mae.index, mae.to_numpy(), color=COLORS[method], linewidth=1.8, label=method)
    ax.set_xlabel("Hour of the target day (Berlin)")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Error by hour of the target day", loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_over_time(days: pd.DataFrame, out: Path, smooth: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for method in METHODS:
        series = days[days.method == method].sort_values("day")
        ax.plot(pd.to_datetime(series.day), series.mae.rolling(smooth, min_periods=smooth).mean(),
                color=COLORS[method], linewidth=1.6, label=method)
    ax.set_ylabel("MAE (MW)")
    ax.set_title(f"Daily error over the year ({smooth}-day trailing mean)", loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def slice_table(days: pd.DataFrame, labels, title: str) -> list[str]:
    frame = days.assign(slice=labels)
    table = frame.pivot_table(index="slice", columns="method", values="mae", aggfunc="mean")
    table = table[METHODS]
    counts = frame[frame.method == TSO].groupby("slice").size()
    lines = [f"| {title} | Days | " + " | ".join(METHODS) + " |",
             "|---" * (len(METHODS) + 2) + "|"]
    for index, row in table.iterrows():
        cells = " | ".join(f"{v:,.0f}" for v in row)
        lines.append(f"| {index} | {counts[index]} | {cells} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="refetch both SMARD series")
    parser.add_argument("--eval-days", type=int, default=365)
    parser.add_argument("--max-days", type=int, default=None, help="cap days (smoke test)")
    parser.add_argument("--report-only", action="store_true", help=f"rebuild from {HOURLY_CACHE}")
    args = parser.parse_args()

    actual = load_history(args.refresh)
    tso = load_tso(args.refresh)
    if args.report_only:
        if not HOURLY_CACHE.exists():
            parser.error(f"{HOURLY_CACHE} not found; run once without --report-only")
        hourly = pd.read_csv(HOURLY_CACHE)
        hourly["timestamp"] = pd.to_datetime(hourly.timestamp, utc=True)
        hourly["day"] = pd.to_datetime(hourly.day).dt.date
    else:
        last_actual = actual.dropna(subset=["demand_mw"]).timestamp.max()
        # The last Berlin day whose every hour is already observed.
        last_day = (last_actual + pd.Timedelta(hours=1)).tz_convert(BERLIN).date() \
            - pd.Timedelta(days=1)
        days = list(pd.date_range(end=last_day, periods=args.eval_days, freq="D").date)
        if args.max_days is not None:
            days = days[-args.max_days:]
        print(f"{len(days)} target days, {days[0]} to {days[-1]}", flush=True)
        hourly = run(actual, tso, days)
        hourly.to_csv(HOURLY_CACHE, index=False)

    days = per_day(hourly)
    results = decide(days)
    means = days.groupby("method")[["mae", "bias"]].mean().reindex(METHODS)
    headline = results[LAGGED]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_by_hour(hourly, FIG_DIR / "tso_study_by_hour.png")
    plot_over_time(days, FIG_DIR / "tso_study_over_time.png")
    days.to_csv(DAYS_PATH, index=False)

    day_index = pd.to_datetime(days.day)
    season = np.where(day_index.dt.month.isin([12, 1, 2]), "Winter", "Rest of year")
    holidays = german_holidays(set(day_index.dt.year.unique()))
    daytype = np.where(
        days.day.isin(holidays), "Public holiday",
        np.where(day_index.dt.weekday >= 5, "Weekend", "Working day"),
    )

    merged = actual.merge(tso, on="timestamp", suffixes=("_actual", "_tso")).dropna()
    merged["err"] = merged.demand_mw_tso - merged.demand_mw_actual
    merged["year"] = merged.timestamp.dt.tz_convert(BERLIN).dt.year
    by_year = merged.groupby("year").err
    tso_by_year = pd.DataFrame(
        {"bias": by_year.mean(), "mae": by_year.apply(lambda e: e.abs().mean())}
    )
    tso_by_year = tso_by_year[merged.groupby("year").size() >= 24 * 300]  # full years only

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    n_days = int(days.day.nunique())
    SUMMARY_PATH.write_text(json.dumps({
        "generated": generated, "model_version": MODEL_VERSION, "days": n_days,
        "first_day": str(days.day.min()), "last_day": str(days.day.max()),
        "issue_hour_berlin": ISSUE_HOUR, "smard_lag_hours": SMARD_LAG / pd.Timedelta(hours=1),
        "means": means.round(1).to_dict(orient="index"), "comparisons": results,
        "tso_by_year": tso_by_year.round(1).to_dict(orient="index"), "versions": versions,
    }, indent=2, default=str) + "\n", encoding="utf-8")

    table = ["| Forecast | Mean daily MAE (MW) | Mean bias (MW) | vs operators (MW) | 95% CI "
             "| Holm p | Verdict |", "|---|---|---|---|---|---|---|"]
    for method in METHODS:
        row = means.loc[method]
        if method == TSO:
            table.append(f"| {method} | {row.mae:,.0f} | {row.bias:+,.0f} | | | | (reference) |")
            continue
        r = results[method]
        table.append(
            f"| {method} | {row.mae:,.0f} | {row.bias:+,.0f} | {r['mean_diff']:+,.0f} | "
            f"[{r['ci_lower']:+,.0f}, {r['ci_upper']:+,.0f}] | {fmt_p(r['holm_p'])} | "
            f"{r['verdict']} |"
        )

    blend = results[BLEND]
    report = [
        "# Against the operators' day-ahead forecast",
        "",
        f"Generated {generated} by `scripts/tso_study.py` (model {MODEL_VERSION}, Python "
        f"{versions['python']}, scikit-learn {versions['scikit-learn']}).",
        "",
        "Question: how does the model compare with the forecast Germany's transmission "
        "operators publish for the next day?",
        "",
        "## What module 411 is",
        "",
        "SMARD's chart-data module 411 is not in the bundesAPI documentation, so it is "
        "identified from the data: it extends to the end of the next day while actual load "
        "(module 410) lags by about a day, and it tracks actual load hour by hour. That is the "
        "shape of a day-ahead load forecast. Its timestamps line up with module 410: shifting "
        "it by an hour either way raises its error by about a third. Its yearly error against "
        "actual load:",
        "",
        "| Year | Mean bias (MW) | MAE (MW) |",
        "|---|---|---|",
        *[f"| {y} | {r.bias:+,.0f} | {r.mae:,.0f} |" for y, r in tso_by_year.iterrows()],
        "",
        "Before 2022 it ran well below actual load every year. That looks like a change in how "
        "one of the two series is defined rather than a change in skill, so only the latest "
        "year is compared below, where the bias is small.",
        "",
        "## Design",
        "",
        f"- {n_days} target days, {days.day.min()} to {days.day.max()}, each scored on its own "
        "Berlin-time hours (23 or 25 on DST days), one MAE per day as the sampling unit.",
        f"- The operators' forecast is assumed to be issued at {ISSUE_HOUR}:00 Berlin on the day "
        "before. If SMARD shows a later revision instead, this comparison is generous to them.",
        "- The model, the shipped v2 champion, is refit at each cut-off on data dated at or "
        "before it and forecasts recursively through the end of the target day:",
        "  - **fresh data**: the last hour before the day (impossible in practice; upper bound)",
        f"  - **at their issue time**: {ISSUE_HOUR}:00 on the day before (about 38 hours ahead)",
        "  - **with SMARD's lag**: 24 hours earlier still (about 62 hours ahead), which is "
        "what this project could actually issue at the operators' issue time",
        "- The blend averages the operators' forecast and the lagged model with fixed equal "
        "weights.",
        "- Rule set before the run: paired bootstrap CI and Holm-corrected Wilcoxon p for each "
        "comparison; \"beats\" or \"worse\" only when the CI excludes zero and Holm p < 0.05.",
        "",
        "## Result",
        "",
        *table,
        "",
        f"**Headline, with SMARD's lag: {headline['verdict']}** "
        f"({headline['mean_diff']:+,.0f} MW per day, 95% CI [{headline['ci_lower']:+,.0f}, "
        f"{headline['ci_upper']:+,.0f}], Holm p = {fmt_p(headline['holm_p'])}). That is the "
        "comparison this project could actually deliver at the moment the operators issue theirs.",
        "",
        f"The fixed 50/50 blend: **{blend['verdict']}** ({blend['mean_diff']:+,.0f} MW, CI "
        f"[{blend['ci_lower']:+,.0f}, {blend['ci_upper']:+,.0f}]). A blend can beat both inputs "
        "when their errors are not perfectly correlated; the weights were fixed, not fitted.",
        "",
        "![Daily error over the year](figures/tso_study_over_time.png)",
        "",
        "## Where the difference comes from",
        "",
        "![Error by hour of the target day](figures/tso_study_by_hour.png)",
        "",
        "Mean daily MAE (MW). Descriptive only; the paired test above is the decision.",
        "",
        *slice_table(days, season, "Season"),
        "",
        *slice_table(days, daytype, "Day type"),
        "",
        "## Files",
        "",
        f"- `{DAYS_PATH.name}`: per-day MAE and bias per forecast.",
        f"- `{SUMMARY_PATH.name}`: every number in this report.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}; lagged model: {headline['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
