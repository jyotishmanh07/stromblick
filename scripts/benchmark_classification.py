#!/usr/bin/env python3
"""Chronological backtest of the daily event classifiers.

Two targets, both labelled from past data only:
  * high-demand day  -- peak in the top 30% of the trailing month
  * anomaly day      -- an hour departing sharply from the seasonal-naive expectation

Writes reports/classification.md, a metrics CSV, and three figures.
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
from energy_forecast.events import (
    CalendarLogisticBaseline,
    HighDemandClassifier,
    MajorityClassBaseline,
    anomaly_day_frame,
    classification_metrics,
    daily_feature_frame,
    rolling_origin_classification_backtest,
    threshold_cost_table,
)
from energy_forecast.theme import INK
from energy_forecast.theme import MODEL_COLORS as PALETTE
from energy_forecast.theme import style_axes as _style_axes

FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/classification.md")
METRICS_PATH = Path("reports/classification_metrics.csv")
SUMMARY_PATH = Path("reports/classification_summary.json")

MODEL_FACTORIES = {
    "Majority class": MajorityClassBaseline,
    "Calendar logistic": CalendarLogisticBaseline,
    "HistGradientBoosting": HighDemandClassifier,
}
COLORS = {
    "Majority class": PALETTE["Seasonal naive"],
    "Calendar logistic": PALETTE["SARIMAX"],
    "HistGradientBoosting": PALETTE["HistGradientBoosting"],
}
TARGETS = {
    "high_demand": ("High-demand day", daily_feature_frame),
    "anomaly": ("Anomaly day", anomaly_day_frame),
}


def plot_pr_curves(backtest: pd.DataFrame, title: str, out: Path) -> None:
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 5))
    base = None
    for name, group in backtest.groupby("model"):
        labels = group.label.to_numpy()
        base = labels.mean()
        if labels.min() == labels.max():
            continue
        precision, recall, _ = precision_recall_curve(labels, group.probability.to_numpy())
        ax.plot(recall, precision, color=COLORS.get(name, INK), linewidth=2, label=name, zorder=3)
    if base is not None:
        ax.axhline(base, color=INK, linestyle="--", linewidth=1, zorder=2,
                   label=f"Base rate ({base:.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_calibration(backtest: pd.DataFrame, title: str, out: Path, bins: int = 8) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1, zorder=2,
            label="Perfect calibration")
    for name, group in backtest.groupby("model"):
        probabilities = group.probability.to_numpy()
        labels = group.label.to_numpy()
        edges = np.linspace(0, 1, bins + 1)
        index = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
        xs, ys = [], []
        for b in range(bins):
            mask = index == b
            if mask.sum() >= 5:  # a bucket of one or two days says nothing
                xs.append(probabilities[mask].mean())
                ys.append(labels[mask].mean())
        if xs:
            ax.plot(xs, ys, marker="o", color=COLORS.get(name, INK), linewidth=2, label=name,
                    zorder=3)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title, color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def metrics_table(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODEL_FACTORIES:
        group = backtest[backtest.model == name]
        if group.empty:
            continue
        scores = classification_metrics(group.label.to_numpy(), group.probability.to_numpy())
        rows.append({"model": name, **scores})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def run_target(key: str, label: str, builder, frame: pd.DataFrame, args) -> dict[str, object]:
    daily = builder(frame)
    backtest = rolling_origin_classification_backtest(
        daily, MODEL_FACTORIES, initial_train_days=args.initial_train_days
    )
    table = metrics_table(backtest)
    champion = str(table.model.iloc[0])
    champion_rows = backtest[backtest.model == champion]
    costs = threshold_cost_table(
        champion_rows.label.to_numpy(), champion_rows.probability.to_numpy(),
        miss_cost=args.miss_cost, false_alarm_cost=args.false_alarm_cost,
    )
    best_threshold = costs.loc[costs.expected_cost.idxmin()]
    plot_pr_curves(backtest, f"{label}: precision-recall", FIG_DIR / f"classification_pr_{key}.png")
    plot_calibration(backtest, f"{label}: calibration", FIG_DIR / f"classification_cal_{key}.png")
    return {
        "key": key, "label": label, "daily": daily, "backtest": backtest, "table": table,
        "champion": champion, "costs": costs, "best_threshold": best_threshold,
    }


def target_section(result: dict[str, object], args) -> list[str]:
    table, best = result["table"], result["best_threshold"]
    backtest, daily = result["backtest"], result["daily"]
    base_rate = float(backtest[backtest.model == result["champion"]].label.mean())
    rows = [
        "| Model | PR-AUC | Lift over base rate | ROC-AUC | Brier |",
        "|---|---|---|---|---|",
    ]
    for row in table.itertuples():
        rows.append(
            f"| {row.model} | {row.pr_auc:.3f} | {row.pr_auc_lift:.2f}× | "
            f"{row.roc_auc:.3f} | {row.brier:.4f} |"
        )
    champion_row = table[table.model == result["champion"]].iloc[0]
    calendar_row = table[table.model == "Calendar logistic"]
    verdict = (
        f"**{result['champion']}** ranks best. "
        if result["champion"] != "Calendar logistic"
        else "**The calendar baseline wins**, which is the finding. "
    )
    if not calendar_row.empty and result["champion"] != "Calendar logistic":
        gap = champion_row.pr_auc - float(calendar_row.pr_auc.iloc[0])
        if gap < 0.05:
            verdict += (
                "It barely separates from the calendar-only baseline, so most of the signal "
                "is the almanac rather than demand dynamics. "
            )
    return [
        f"## {result['label']}",
        "",
        f"{len(daily):,} labelled days, {backtest.date.nunique():,} scored chronologically after "
        f"a {args.initial_train_days}-day warm-up; every model refit at every origin. "
        f"Positive rate over the scored window: {base_rate:.1%} "
        f"({int(backtest[backtest.model == result['champion']].label.sum())} days).",
        "",
        *rows,
        "",
        f"{verdict}PR-AUC is read against the base rate, not against 0.5 — the lift column is "
        "the honest version. The majority-class row is the floor: it predicts the training "
        "positive rate for every day, so any ranking it appears to achieve is noise.",
        "",
        f"![Precision-recall](figures/classification_pr_{result['key']}.png)",
        "",
        f"![Calibration](figures/classification_cal_{result['key']}.png)",
        "",
        "### Choosing an operating point",
        "",
        f"Assuming a missed day costs {args.miss_cost:.0f}× a false alarm, expected cost is "
        f"lowest at a threshold of **{best.threshold:.2f}** — flagging {int(best.flagged)} days "
        f"to catch {int(best.true_positive)} of "
        f"{int(best.true_positive + best.false_negative)}, at "
        f"{best.precision:.0%} precision and {best.recall:.0%} recall. That ratio is an "
        "assumption, not a measurement: change it and the operating point moves, which is "
        "the point of reporting the whole curve rather than one number.",
        "",
        *boundary_note(result, args),
    ]


def boundary_note(result: dict[str, object], args) -> list[str]:
    """Flag an operating point that sits at the edge of the searched grid.

    An edge solution is not really an optimum — it says the assumed cost ratio is
    extreme enough to dominate the model, and reporting it as a tuned threshold
    would overstate what the analysis showed.
    """
    costs, best = result["costs"], result["best_threshold"]
    if best.threshold > costs.threshold.min() + 1e-9:
        return []
    base_rate = float(result["backtest"].query("model == @result['champion']").label.mean())
    return [
        f"> **This is a boundary solution.** The minimum sits at the lowest threshold searched, "
        f"which means a {args.miss_cost:.0f}:1 cost ratio against a {base_rate:.0%} base rate is "
        "extreme enough that flagging nearly everything wins — the model is barely consulted. "
        "Read it as a statement about the assumed costs, not as a tuned threshold. A ratio "
        f"below roughly {(1 - base_rate) / base_rate:.0f}:1 is where the decision starts "
        "depending on the model's ranking at all.",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-train-days", type=int, default=120)
    parser.add_argument("--miss-cost", type=float, default=10.0)
    parser.add_argument("--false-alarm-cost", type=float, default=1.0)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_clean_demand()
    results = [run_target(key, label, builder, frame, args)
               for key, (label, builder) in TARGETS.items()]

    metrics = pd.concat(
        [result["table"].assign(target=result["key"]) for result in results], ignore_index=True
    )
    metrics.to_csv(METRICS_PATH, index=False)

    lines = [
        "# Daily event classification",
        "",
        "A second problem type on the same SMARD snapshot: instead of \"how much demand each "
        "hour\", these models answer \"is tomorrow worth planning around\" — the shape a reserve "
        "or staffing decision actually takes. The forecasting benchmark remains the headline "
        "result; this is a companion track.",
        "",
        "Both targets are labelled from **past data only**. The threshold that judges a day is "
        "computed from the days before it and shifted, so the leakage rule that governs the "
        "forecasting features governs the labels too. Evaluation walks forward one day at a "
        "time, refitting every model at every origin — the classification counterpart of "
        "`rolling_origin_backtest`.",
        "",
        "Two baselines make the comparison honest: a majority-class predictor (the floor) and a "
        "logistic regression on calendar columns alone (how far the almanac gets you without "
        "looking at demand at all).",
        "",
    ]
    for result in results:
        lines += target_section(result, args)

    lines += [
        "## Label design",
        "",
        "The high-demand threshold is a trailing 30-day 70th percentile. Two stricter framings "
        "were tried first and both collapsed on a single year of strongly seasonal data:",
        "",
        "- An **expanding all-time top decile** is set by January's peaks, so no summer day can "
        "clear it. The label degenerates into \"is it winter\", and an evaluation window "
        "starting after winter contains no positive days at all.",
        "- A **trailing 60-day top decile** still empties out during sustained seasonal decline: "
        "March, April and May produced zero positives, because a falling series never exceeds "
        "its own recent upper tail.",
        "",
        "A shorter window and a milder quantile keep every month populated. The positive rate is "
        "still not stationary — it rises when demand trends up and falls when it trends down — "
        "which is a real property of the target rather than something to smooth away.",
        "",
        "The anomaly label uses the **seasonal-naive** deviation, `demand(t) − demand(t−24h)`, "
        "which depends on no fitted model. Labelling with the champion's own residuals would "
        "ask the classifier to predict where one particular model fails, and any evaluation of "
        "that would be circular.",
        "",
        "## Limitations",
        "",
        f"- One year of data yields roughly {len(results[0]['daily']):,} labelled days, and the "
        "scored window holds a few dozen positives. Confidence intervals on these metrics would "
        "be wide; treat the ranking as indicative, not settled.",
        "- The cost ratio behind the operating point is assumed, not measured.",
        "- No weather covariates. Temperature is the obvious missing driver for both targets.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/benchmark_classification.py",
        "```",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                result["key"]: {
                    "label": result["label"],
                    "champion": result["champion"],
                    "scored_days": int(result["backtest"].date.nunique()),
                    "labelled_days": int(len(result["daily"])),
                    "models": {
                        row.model: {
                            "pr_auc": round(float(row.pr_auc), 4),
                            "pr_auc_lift": round(float(row.pr_auc_lift), 3),
                            "roc_auc": round(float(row.roc_auc), 4),
                            "brier": round(float(row.brier), 4),
                            "base_rate": round(float(row.base_rate), 4),
                        }
                        for row in result["table"].itertuples()
                    },
                    "operating_point": {
                        "threshold": round(float(result["best_threshold"].threshold), 2),
                        "precision": round(float(result["best_threshold"].precision), 4),
                        "recall": round(float(result["best_threshold"].recall), 4),
                        "flagged": int(result["best_threshold"].flagged),
                    },
                }
                for result in results
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {REPORT_PATH}, {METRICS_PATH}, {SUMMARY_PATH}, and 4 figures to {FIG_DIR}/")
    for result in results:
        print(f"\n{result['label']} (champion: {result['champion']})")
        print(result["table"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
