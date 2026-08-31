#!/usr/bin/env python3
"""Generate EDA figures and a markdown report from the clean SMARD dataset.

Reuses the project's own data and feature functions rather than recomputing
calendar logic, so the report always matches what the models are trained on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_forecast.data import load_clean_demand, missing_hourly_timestamps
from energy_forecast.features import german_holidays

FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/eda.md")

# Categorical/sequential slots from the project's validated reference palette.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plot_overview(frame: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame.timestamp, frame.demand_mw, color=BLUE, linewidth=1, zorder=3)
    ax.set_ylabel("Demand (MW)")
    ax.set_title("Germany hourly electricity demand", color=INK, loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def daily_profile_by_daytype(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.timestamp.dt.tz_convert("Europe/Berlin")
    holidays = german_holidays(set(local.dt.year))
    is_holiday = local.dt.date.isin(holidays)
    is_weekend = local.dt.weekday >= 5
    day_type = np.select([is_holiday, is_weekend], ["Public holiday", "Weekend"], default="Weekday")
    return (
        pd.DataFrame({"hour": local.dt.hour, "demand_mw": frame.demand_mw, "day_type": day_type})
        .groupby(["day_type", "hour"], as_index=False)
        .demand_mw.mean()
    )


def plot_daily_profile(profile: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for day_type, color in [("Weekday", BLUE), ("Weekend", ORANGE), ("Public holiday", AQUA)]:
        subset = profile[profile.day_type == day_type].sort_values("hour")
        if subset.empty:
            continue
        ax.plot(subset.hour, subset.demand_mw, color=color, linewidth=2, label=day_type, zorder=3)
    ax.set_xlabel("Hour of day (Europe/Berlin)")
    ax.set_ylabel("Mean demand (MW)")
    ax.set_title("Average daily demand profile by day type", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_weekday_pattern(frame: pd.DataFrame, out: Path) -> pd.Series:
    local = frame.timestamp.dt.tz_convert("Europe/Berlin")
    means = frame.groupby(local.dt.day_name()).demand_mw.mean().reindex(WEEKDAY_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(means.index, means.values, color=BLUE, width=0.6, zorder=3)
    ax.set_ylabel("Mean demand (MW)")
    ax.set_title("Average demand by weekday", color=INK, loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return means


def plot_monthly_pattern(frame: pd.DataFrame, out: Path) -> pd.Series:
    local = frame.timestamp.dt.tz_convert("Europe/Berlin")
    means = frame.groupby(local.dt.strftime("%Y-%m")).demand_mw.mean().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(means.index, means.values, color=BLUE, width=0.6, zorder=3)
    ax.set_ylabel("Mean demand (MW)")
    ax.set_title("Average demand by month", color=INK, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return means


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_clean_demand()
    missing = missing_hourly_timestamps(frame)
    empty_hours = frame[frame.demand_mw.isna()]

    plot_overview(frame, FIG_DIR / "demand_overview.png")
    profile = daily_profile_by_daytype(frame)
    plot_daily_profile(profile, FIG_DIR / "daily_profile_by_daytype.png")
    plot_weekday_pattern(frame, FIG_DIR / "weekly_pattern.png")
    plot_monthly_pattern(frame, FIG_DIR / "monthly_pattern.png")

    weekday_peak = profile[profile.day_type == "Weekday"].demand_mw.max()
    weekend_peak = profile[profile.day_type == "Weekend"].demand_mw.max()
    lines = [
        "# Exploratory data analysis",
        "",
        f"Snapshot: `{frame.timestamp.min()}` to `{frame.timestamp.max()}` "
        f"({len(frame):,} hourly rows). Source: Bundesnetzagentur | SMARD.de, module 410 "
        "(Germany actual total grid load), CC BY 4.0.",
        "",
        f"- Mean demand: {frame.demand_mw.mean():,.0f} MW",
        f"- Peak: {frame.demand_mw.max():,.0f} MW; trough: {frame.demand_mw.min():,.0f} MW",
        f"- Missing hourly timestamps (gaps in the index): {len(missing)}",
        f"- Hours present in the index but with no demand value: {len(empty_hours)}",
    ]
    if len(empty_hours):
        sample = ", ".join(str(t) for t in empty_hours.timestamp.head(5))
        lines.append(f"  - First few: {sample}")
    lines += [
        "",
        "Missing hours are reported, not imputed, matching the project's data-validation policy.",
        "",
        "## Demand over the collected period",
        "",
        "![Demand overview](figures/demand_overview.png)",
        "",
        "## Daily profile by day type",
        "",
        f"Weekday demand peaks around {weekday_peak:,.0f} MW; weekend peaks are lower "
        f"(~{weekend_peak:,.0f} MW) and public holidays track the weekend shape even when "
        "they fall on a weekday.",
        "",
        "![Daily profile by day type](figures/daily_profile_by_daytype.png)",
        "",
        "## Weekly pattern",
        "",
        "![Weekly pattern](figures/weekly_pattern.png)",
        "",
        "## Monthly pattern",
        "",
        "![Monthly pattern](figures/monthly_pattern.png)",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH} and 4 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
