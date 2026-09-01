#!/usr/bin/env python3
"""Nested rolling-origin hyperparameter search for the gradient-boosting champion.

The nesting is the point. The *inner* loop searches hyperparameters on rolling
origins drawn only from a training span; the *outer* loop then scores the winner
against the shipped configuration on the held-out remainder, which the search never
saw. A single-loop search that tuned on the same origins it reported would produce a
number that cannot be trusted, and that is the usual way portfolio tuning goes wrong.

Adoption is gated: a tuned configuration replaces the default only when its outer
improvement clears a paired-bootstrap confidence interval. "Kept the defaults, the
gain was inside the noise" is a real result and is reported as one.

    PYTHONPATH=src python scripts/tune.py --trials 40
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

from energy_forecast.data import load_clean_demand  # noqa: E402
from energy_forecast.evaluation import metrics  # noqa: E402
from energy_forecast.inference import paired_model_comparison  # noqa: E402
from energy_forecast.models import HistGradientBoostingForecast  # noqa: E402
from energy_forecast.theme import BLUE, INK, MUTED  # noqa: E402
from energy_forecast.theme import style_axes as _style_axes  # noqa: E402

FIG_DIR = Path("reports/figures")
TRIALS_PATH = Path("reports/tuning_trials.csv")
SUMMARY_PATH = Path("reports/tuning_summary.json")
REPORT_PATH = Path("reports/tuning.md")


def score_config(
    frame: pd.DataFrame, params: dict, origins: list[int], horizon_hours: int
) -> np.ndarray:
    """Mean absolute error at each origin for one hyperparameter configuration."""
    errors: list[float] = []
    for origin in origins:
        train = frame.iloc[:origin]
        test = frame.iloc[origin : origin + horizon_hours]
        actual = test.demand_mw.to_numpy(dtype=float)
        observed = ~np.isnan(actual)
        if not observed.any():
            continue
        predicted = (
            HistGradientBoostingForecast(**params)
            .fit(train)
            .predict(pd.DatetimeIndex(test.timestamp))
        )
        errors.append(metrics(actual[observed], predicted[observed])["mae"])
    return np.array(errors)


def build_origins(start: int, stop: int, step: int, horizon: int, limit: int | None) -> list[int]:
    origins = list(range(start, stop - horizon + 1, step))
    if limit is not None and len(origins) > limit:
        # Thin evenly rather than truncating, so the sample still spans the whole period.
        idx = np.linspace(0, len(origins) - 1, limit).round().astype(int)
        origins = [origins[i] for i in dict.fromkeys(idx)]
    return origins


def plot_history(trials: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(trials.number, trials.value, color=MUTED, s=18, zorder=3, label="Trial")
    ax.plot(trials.number, trials.value.cummin(), color=BLUE, linewidth=2, zorder=4,
            label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Inner-CV mean MAE (MW)")
    ax.set_title("Hyperparameter search history", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--inner-origins", type=int, default=12,
                        help="origins per trial; more is steadier but linearly slower")
    parser.add_argument("--outer-origins", type=int, default=60)
    parser.add_argument("--train-fraction", type=float, default=0.7,
                        help="fraction of the snapshot the inner search may see")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_clean_demand()
    frame = frame.loc[: frame.demand_mw.last_valid_index()].reset_index(drop=True)
    split = int(len(frame) * args.train_fraction)
    warmup = 24 * 28

    inner_origins = build_origins(
        warmup, split, 24 * 7, args.horizon_hours, args.inner_origins
    )
    outer_origins = build_origins(
        split, len(frame), 24, args.horizon_hours, args.outer_origins
    )
    print(
        f"Snapshot {len(frame):,} rows | inner span ends at index {split:,} "
        f"({inner_origins and len(inner_origins)} origins) | "
        f"outer {len(outer_origins)} origins on unseen data"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True),
        }
        return float(score_config(frame, params, inner_origins, args.horizon_hours).mean())

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    print(f"Inner search: {args.trials} trials x {len(inner_origins)} origins ...", flush=True)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.columns = [c.replace("params_", "") for c in trials.columns]
    trials.to_csv(TRIALS_PATH, index=False)
    plot_history(trials, FIG_DIR / "tuning_history.png")

    default_params = dict(HistGradientBoostingForecast.DEFAULT_PARAMS)
    tuned_params = dict(study.best_params)
    default_inner = float(score_config(
        frame, default_params, inner_origins, args.horizon_hours
    ).mean())

    print("Outer confirmation on unseen origins ...", flush=True)
    tuned_outer = score_config(frame, tuned_params, outer_origins, args.horizon_hours)
    default_outer = score_config(frame, default_params, outer_origins, args.horizon_hours)
    comparison = paired_model_comparison(tuned_outer, default_outer)
    # Adopt only if the whole CI sits below zero: tuned errs less by a resolvable margin.
    adopt = bool(comparison["ci_upper"] < 0)

    verdict = (
        "**Adopt the tuned configuration.** Its outer-backtest advantage clears the paired "
        "bootstrap interval, so the gain is larger than the origin-to-origin noise."
        if adopt else
        "**Keep the shipped defaults.** The tuned configuration does not beat them by a margin "
        "the backtest can resolve — the confidence interval on the difference contains zero. "
        "Adopting it would be fitting the search, not improving the model."
    )

    rows = ["| Parameter | Shipped default | Inner-search best |", "|---|---|---|"]
    for key in sorted(set(default_params) | set(tuned_params)):
        rows.append(
            f"| `{key}` | {default_params.get(key, '—')} | {tuned_params.get(key, '—')} |"
        )

    lines = [
        "# Hyperparameter search",
        "",
        f"Optuna TPE, {args.trials} trials, seed {args.seed}. The search is **nested**: trials "
        f"are scored on {len(inner_origins)} rolling origins drawn only from the first "
        f"{args.train_fraction:.0%} of the snapshot, and the winner is then confirmed on "
        f"{len(outer_origins)} later origins the search never saw. A single-loop search that "
        "tuned and reported on the same origins would produce a number that cannot be trusted.",
        "",
        "## Search space and outcome",
        "",
        *rows,
        "",
        f"Best inner-CV mean MAE: **{study.best_value:,.0f} MW** against "
        f"**{default_inner:,.0f} MW** for the shipped defaults "
        f"({100 * (study.best_value - default_inner) / default_inner:+.1f}%).",
        "",
        "![Search history](figures/tuning_history.png)",
        "",
        "## Outer confirmation",
        "",
        f"On {len(outer_origins)} held-out origins the tuned configuration averages "
        f"**{tuned_outer.mean():,.0f} MW** MAE against **{default_outer.mean():,.0f} MW** for "
        f"the defaults — a mean difference of {comparison['mean_diff']:,.0f} MW "
        f"(95% CI [{comparison['ci_lower']:,.0f}, {comparison['ci_upper']:,.0f}], "
        f"Wilcoxon p = {comparison['wilcoxon_p']:.3f}). Negative favours the tuned settings.",
        "",
        verdict,
        "",
        "## Why the gate matters",
        "",
        "Hyperparameter search almost always produces *some* improvement on the data it "
        "searched. The question a reviewer should ask is whether that improvement survives "
        "contact with data the search never saw, and whether it is bigger than the variation "
        "between origins. Reporting the search without that gate is how tuned numbers end up "
        "unreproducible.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=src python scripts/tune.py --trials {args.trials}",
        "```",
        "",
        f"Per-trial configurations and scores are in `{TRIALS_PATH.name}`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "trials": args.trials, "seed": args.seed,
                "inner_origins": len(inner_origins), "outer_origins": len(outer_origins),
                "default_params": default_params, "tuned_params": tuned_params,
                "inner_best_mae": round(float(study.best_value), 1),
                "inner_default_mae": round(default_inner, 1),
                "outer_tuned_mae": round(float(tuned_outer.mean()), 1),
                "outer_default_mae": round(float(default_outer.mean()), 1),
                "comparison": {k: round(float(v), 4) for k, v in comparison.items()},
                "adopt_tuned": adopt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {REPORT_PATH}, {TRIALS_PATH}, {SUMMARY_PATH}, and 1 figure")
    print(f"Inner best {study.best_value:,.0f} vs default {default_inner:,.0f} MW")
    print(f"Outer tuned {tuned_outer.mean():,.0f} vs default {default_outer.mean():,.0f} MW")
    print(f"Mean diff {comparison['mean_diff']:,.0f} MW "
          f"CI [{comparison['ci_lower']:,.0f}, {comparison['ci_upper']:,.0f}]")
    print("ADOPT TUNED" if adopt else "KEEP DEFAULTS (gain within noise)")


if __name__ == "__main__":
    main()
