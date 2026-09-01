#!/usr/bin/env python3
"""Generate EDA figures and a markdown report from the clean SMARD dataset.

Every aggregation is a SQL query against the DuckDB warehouse (see sql/), which is
rebuilt here from the canonical frame so the report can never read a stale snapshot.
The calendar dimension is materialised from the same helpers the model features use,
so a SQL slice and a model feature agree by construction. Pandas is used only to
plot what SQL returns; tests/test_warehouse.py asserts the queries match the
equivalent pandas groupbys.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # report script: never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402

from energy_forecast.data import load_clean_demand
from energy_forecast.theme import BLUE, INK, ORANGE
from energy_forecast.theme import GREEN as AQUA
from energy_forecast.theme import style_axes as _style_axes
from energy_forecast.warehouse import DEFAULT_DB_PATH, build_warehouse, query

FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/eda.md")


def plot_overview(frame: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame.timestamp, frame.demand_mw, color=BLUE, linewidth=1, zorder=3)
    ax.set_ylabel("Demand (MW)")
    ax.set_title("Germany hourly electricity demand", color=INK, loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


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


def plot_bars(labels: list[str], values: list[float], title: str, out: Path, rotate: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 4) if rotate else (8, 4))
    ax.bar(labels, values, color=BLUE, width=0.6, zorder=3)
    ax.set_ylabel("Mean demand (MW)")
    ax.set_title(title, color=INK, loc="left")
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_clean_demand()
    db = build_warehouse(frame, DEFAULT_DB_PATH)

    profile = query("daily_profile_by_daytype", db)
    weekday = query("weekly_pattern", db)
    monthly = query("monthly_pattern", db)
    gaps = query("missing_hours", db)
    holiday = query("holiday_vs_ordinary", db)
    stats = query(
        "SELECT count(*) AS rows, avg(demand_mw) AS mean, max(demand_mw) AS peak, "
        "min(demand_mw) AS trough, min(timestamp) AS start, max(timestamp) AS end "
        "FROM fact_demand WHERE demand_mw IS NOT NULL",
        db,
    ).iloc[0]

    plot_overview(frame, FIG_DIR / "demand_overview.png")
    plot_daily_profile(profile, FIG_DIR / "daily_profile_by_daytype.png")
    plot_bars(
        list(weekday.weekday_name), list(weekday.demand_mw),
        "Average demand by weekday", FIG_DIR / "weekly_pattern.png", rotate=False,
    )
    plot_bars(
        list(monthly.year_month), list(monthly.demand_mw),
        "Average demand by month", FIG_DIR / "monthly_pattern.png", rotate=True,
    )

    absent = gaps[gaps.gap_type == "absent from index"]
    unpublished = gaps[gaps.gap_type == "indexed, no value"]
    weekday_peak = profile[profile.day_type == "Weekday"].demand_mw.max()
    weekend_peak = profile[profile.day_type == "Weekend"].demand_mw.max()
    holiday_row = holiday[holiday.day_class == "Public holiday"]

    lines = [
        "# Exploratory data analysis",
        "",
        f"Snapshot: `{frame.timestamp.min()}` to `{frame.timestamp.max()}` "
        f"({len(frame):,} hourly rows). Source: Bundesnetzagentur | SMARD.de, module 410 "
        "(Germany actual total grid load), CC BY 4.0.",
        "",
        "Every figure below is a SQL aggregation over the DuckDB warehouse built from that "
        "snapshot (`sql/`, rebuilt by `scripts/build_warehouse.py`).",
        "",
        f"- Mean demand: {stats['mean']:,.0f} MW",
        f"- Peak: {stats['peak']:,.0f} MW; trough: {stats['trough']:,.0f} MW",
        f"- Missing hourly timestamps (gaps in the index): {len(absent)}",
        f"- Hours present in the index but with no demand value: {len(unpublished)}",
    ]
    if len(unpublished):
        sample = ", ".join(str(t) for t in unpublished.timestamp.head(5))
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
    if not holiday_row.empty:
        row = holiday_row.iloc[0]
        ordinary = holiday[holiday.day_class == "Ordinary day"].iloc[0]
        drop = 100 * (ordinary.mean_demand_mw - row.mean_demand_mw) / ordinary.mean_demand_mw
        lines += [
            "## Public holidays",
            "",
            f"Across {int(row.hours):,} public-holiday hours mean demand is "
            f"{row.mean_demand_mw:,.0f} MW against {ordinary.mean_demand_mw:,.0f} MW on "
            f"{int(ordinary.hours):,} ordinary hours — a {drop:.0f}% drop. Holidays are also "
            "the champion model's weakest slice; see [benchmark.md](benchmark.md).",
            "",
        ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}, {db} and 4 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
