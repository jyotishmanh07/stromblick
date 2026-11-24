#!/usr/bin/env python3
"""Which prediction band is actually calibrated?

The shipped band is the forecast plus or minus the 95th percentile of absolute errors
from one 168-hour recursive forecast over the week before the origin
(`ForecastService._fit_validation_residuals`). The benchmark measures a different
recipe: the 24-hour errors of the previous 7 origins. This script scores both against
four alternatives on the same origins:

* Split conformal: the previous 28 origins' 24-hour absolute errors, with the
  finite-sample quantile ceil((n + 1)(1 - alpha)) / n.
* Adaptive conformal inference (Gibbs and Candes, 2021): the same pool, but the
  quantile level moves after every origin, alpha_t+1 = alpha_t + gamma(alpha - miss_t).
* Quantile gradient boosting: two HistGradientBoostingRegressor(loss="quantile") models
  at 2.5% and 97.5%, trained on the point model's rows and evaluated on the point
  forecast's own recursive path.
* Conformalized quantile regression (Romano et al., 2019): the quantile band widened by
  the conformal quantile of max(lower - y, y - upper) over the previous 28 origins.

Design, fixed before any result was seen:

* The shipped point model (v2, trailing two-year window) at daily origins over the
  latest year, after 28 warm-up origins used only as calibration history.
* Calibration never sees the origin it scores: every pool ends with the previous origin,
  whose 24 hours are all observed by the current origin.
* Per-origin values are the sampling unit, as everywhere in this project.
* Decision rule: a method replaces the shipped band only if (1) the bootstrap 95% CI
  of its mean per-origin coverage contains 0.95, and (2) the paired bootstrap 95% CI on
  its per-origin interval score minus the shipped band's lies entirely below zero, with
  Wilcoxon p < 0.05 after a Holm correction across the five candidates. Among methods
  that pass both, the lowest mean interval score wins. Otherwise keep the shipped band.

The interval (Winkler) score is width + (2 / alpha) x distance outside the band, so it
charges for a band that is too wide as well as for one that misses. For a central
interval it equals (2 / alpha) times the summed pinball loss of its two quantiles.

    PYTHONPATH=src python scripts/interval_study.py               # ~30 min
    PYTHONPATH=src python scripts/interval_study.py --report-only # from the cache
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")  # report script: never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from window_study import holm, load_history  # noqa: E402

from energy_forecast.features import FEATURE_COLUMNS, add_features  # noqa: E402
from energy_forecast.inference import (  # noqa: E402
    conformal_quantile,
    coverage_test,
    paired_model_comparison,
)
from energy_forecast.models import (  # noqa: E402
    MODEL_VERSION,
    HistGradientBoostingForecast,
    trailing_window,
)
from energy_forecast.theme import ANOMALY, BLUE, GREEN, INK, MUTED, ORANGE  # noqa: E402
from energy_forecast.theme import style_axes as _style_axes  # noqa: E402

HOURLY_CACHE = Path("data/clean/interval_study_hourly.csv")  # gitignored
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/interval_study.md")
ORIGINS_PATH = Path("reports/interval_study_origins.csv")
SUMMARY_PATH = Path("reports/interval_study_summary.json")

ALPHA = 0.05
LEVELS = [0.5, 0.8, 0.9, 0.95]  # reliability levels for the residual-pool methods
CAL_ORIGINS = 28
TRAILING_ORIGINS = 7  # the benchmark's recipe
ACI_GAMMA = 0.02
ACI_CLIP = (0.001, 0.5)
HORIZON = 24
VALIDATION_HOURS = 168  # the service's validation week

SHIPPED = "Shipped (168h validation week)"
BENCHMARK_RECIPE = "Benchmark recipe (7 origins)"
SPLIT = "Split conformal (28 origins)"
ACI = "Adaptive conformal"
QUANTILE = "Quantile GBM"
CQR = "Conformalized quantile (CQR)"
METHODS = [SHIPPED, BENCHMARK_RECIPE, SPLIT, ACI, QUANTILE, CQR]
COLORS = dict(zip(METHODS, [MUTED, INK, BLUE, GREEN, ORANGE, ANOMALY]))


# ----------------------------------------------------------------------------- fitting


def run(data: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.DataFrame:
    """Per-hour point forecast, quantile forecasts and the shipped band's spreads."""
    actuals = data.set_index("timestamp").demand_mw
    params = HistGradientBoostingForecast.DEFAULT_PARAMS
    window = HistGradientBoostingForecast.DEFAULT_TRAIN_WINDOW_DAYS
    parts = []
    for i, origin in enumerate(origins, 1):
        train = data[data.timestamp <= origin]
        target = pd.date_range(
            origin + pd.Timedelta(hours=1), periods=HORIZON, freq="1h", tz="UTC"
        )
        point, rows = HistGradientBoostingForecast().fit(train).forecast_with_rows(target)

        # The shipped recipe, by time instead of position: fit on everything up to a week
        # before the origin, forecast that week recursively, keep the absolute errors.
        cut = origin - pd.Timedelta(hours=VALIDATION_HOURS)
        week = pd.date_range(
            cut + pd.Timedelta(hours=1), periods=VALIDATION_HOURS, freq="1h", tz="UTC"
        )
        val_model = HistGradientBoostingForecast().fit(train[train.timestamp <= cut])
        val_err = np.abs(actuals.reindex(week).to_numpy() - val_model.predict(week))
        val_err = val_err[~np.isnan(val_err)]

        feats = add_features(trailing_window(train, window)).dropna(
            subset=FEATURE_COLUMNS + ["demand_mw"]
        )
        quantiles = {}
        for name, q in (("q_lower", ALPHA / 2), ("q_upper", 1 - ALPHA / 2)):
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, random_state=42, **params
            ).fit(feats[FEATURE_COLUMNS], feats.demand_mw)
            quantiles[name] = model.predict(rows[FEATURE_COLUMNS])

        frame = pd.DataFrame({
            "origin": origin, "timestamp": target, "horizon": np.arange(1, HORIZON + 1),
            "actual": actuals.reindex(target).to_numpy(dtype=float), "point": point,
            **quantiles,
        })
        for level in LEVELS:
            frame[f"shipped_{level}"] = float(np.quantile(val_err, level))
        parts.append(frame)
        if i % 25 == 0 or i == len(origins):
            print(f"  origin {i}/{len(origins)}  {origin:%Y-%m-%d}", flush=True)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- bands


def pooled_spread(pools: list[np.ndarray], level: float, conformal: bool) -> float:
    scores = np.concatenate(pools) if pools else np.array([])
    if conformal:
        return conformal_quantile(scores, level)
    return float(np.quantile(scores, level)) if len(scores) else float("nan")


def build_bands(hourly: pd.DataFrame, level: float = 1 - ALPHA) -> dict[str, pd.DataFrame]:
    """Lower/upper bounds per hour for every method, at one nominal level.

    Quantile GBM and CQR exist only at 1 - ALPHA, and adaptive conformal targets only
    ALPHA, so at other levels only the residual-pool methods are returned.
    """
    hourly = hourly.sort_values(["origin", "horizon"]).reset_index(drop=True)
    origins = list(hourly.origin.drop_duplicates())
    by_origin = {o: g for o, g in hourly.groupby("origin", sort=False)}
    observed = {o: g.dropna(subset=["actual"]) for o, g in by_origin.items()}
    abs_err = {o: np.abs(g.actual - g.point).to_numpy() for o, g in observed.items()}
    cqr_score = {
        o: np.maximum(g.q_lower - g.actual, g.actual - g.q_upper).to_numpy()
        for o, g in observed.items()
    }
    nominal = math.isclose(level, 1 - ALPHA)
    out: dict[str, list[pd.DataFrame]] = {m: [] for m in METHODS}
    alpha_t = ALPHA
    for i, origin in enumerate(origins):
        if i < CAL_ORIGINS:
            continue
        g = by_origin[origin]
        prior = origins[i - CAL_ORIGINS:i]
        recent = origins[i - TRAILING_ORIGINS:i]

        def band(lower, upper, index=g.index):
            return pd.DataFrame({"lower": lower, "upper": upper}, index=index)

        spreads = {
            SHIPPED: g[f"shipped_{level}"].to_numpy(),
            BENCHMARK_RECIPE: pooled_spread([abs_err[o] for o in recent], level, False),
            SPLIT: pooled_spread([abs_err[o] for o in prior], level, True),
        }
        for name, spread in spreads.items():
            out[name].append(band(g.point - spread, g.point + spread))
        if not nominal:
            continue

        level_t = 1 - float(np.clip(alpha_t, *ACI_CLIP))
        aci = pooled_spread([abs_err[o] for o in prior], level_t, True)
        out[ACI].append(band(g.point - aci, g.point + aci))
        seen = g.actual.notna()
        miss = ((g.actual < g.point - aci) | (g.actual > g.point + aci))[seen]
        if len(miss):
            # Known by the next origin: this origin's 24 hours end exactly there.
            alpha_t += ACI_GAMMA * (ALPHA - float(miss.mean()))

        out[QUANTILE].append(band(g.q_lower, g.q_upper))
        q = conformal_quantile(np.concatenate([cqr_score[o] for o in prior]), level)
        out[CQR].append(band(g.q_lower - q, g.q_upper + q))
    return {m: pd.concat(parts) for m, parts in out.items() if parts}


def interval_score(lower, upper, actual, alpha: float = ALPHA) -> np.ndarray:
    """Winkler score: width plus (2 / alpha) times the distance outside the band."""
    lower, upper, actual = (np.asarray(x, dtype=float) for x in (lower, upper, actual))
    return (
        (upper - lower)
        + (2 / alpha) * np.clip(lower - actual, 0, None)
        + (2 / alpha) * np.clip(actual - upper, 0, None)
    )


def score(hourly: pd.DataFrame, bands: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Hourly coverage, width and interval score per method (observed hours only)."""
    hourly = hourly.sort_values(["origin", "horizon"]).reset_index(drop=True)
    parts = []
    for method, band in bands.items():
        rows = hourly.loc[band.index, ["origin", "timestamp", "horizon", "actual"]].copy()
        rows["method"] = method
        rows["width"] = (band.upper - band.lower).to_numpy()
        inside = (rows.actual >= band.lower) & (rows.actual <= band.upper)
        rows["covered"] = inside.to_numpy()
        rows["interval_score"] = interval_score(band.lower, band.upper, rows.actual)
        parts.append(rows.dropna(subset=["actual"]))
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- decision


def decide(per_origin: pd.DataFrame) -> tuple[dict[str, dict], str]:
    """Apply the rule fixed before the run; return per-method results and the winner."""
    wide = per_origin.pivot(index="origin", columns="method", values="interval_score")
    results: dict[str, dict] = {}
    for method in METHODS:
        rates = per_origin[per_origin.method == method].coverage
        cov = coverage_test(rates, nominal=1 - ALPHA)
        results[method] = {
            "coverage": cov,
            "calibrated": bool(cov["ci_lower"] <= 1 - ALPHA <= cov["ci_upper"]),
        }
    candidates = [m for m in METHODS if m != SHIPPED]
    for m in candidates:
        pair = wide[[m, SHIPPED]].dropna()
        results[m]["interval_score_vs_shipped"] = paired_model_comparison(pair[m], pair[SHIPPED])
    adjusted = holm({m: results[m]["interval_score_vs_shipped"]["wilcoxon_p"] for m in candidates})
    mean_is = per_origin.groupby("method").interval_score.mean()
    passing = []
    for m in candidates:
        comp = results[m]["interval_score_vs_shipped"]
        comp["holm_p"] = adjusted[m]
        results[m]["sharper"] = bool(comp["ci_upper"] < 0 and adjusted[m] < 0.05)
        results[m]["adopt"] = results[m]["calibrated"] and results[m]["sharper"]
        if results[m]["adopt"]:
            passing.append(m)
    chosen = min(passing, key=lambda m: mean_is[m]) if passing else SHIPPED
    return results, chosen


# ----------------------------------------------------------------------------- report


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def plot_coverage_by_horizon(hourly_scores: pd.DataFrame, out: Path) -> None:
    table = hourly_scores.groupby(["method", "horizon"]).covered.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for method in METHODS:
        if method in table:
            ax.plot(table.index, 100 * table[method], color=COLORS[method], linewidth=1.8,
                    label=method)
    ax.axhline(100 * (1 - ALPHA), color=INK, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("Actuals inside the band (%)")
    ax.set_title("Coverage by forecast horizon (dashed: the 95% target)", loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_tradeoff(means: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, row in means.iterrows():
        ax.scatter(row.width, 100 * row.coverage, color=COLORS[method], s=60, zorder=3)
        ax.annotate(method, (row.width, 100 * row.coverage), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color=INK)
    ax.axhline(100 * (1 - ALPHA), color=INK, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean band width (MW)")
    ax.set_ylabel("Mean per-origin coverage (%)")
    ax.set_title("Coverage against width: up is safer, left is sharper", loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def horizon_table(hourly_scores: pd.DataFrame, value: str, fmt) -> list[str]:
    labels = ["1-6h", "7-12h", "13-18h", "19-24h"]
    bins = pd.cut(hourly_scores.horizon, [0, 6, 12, 18, 24], labels=labels)
    table = (
        hourly_scores.assign(bin=bins)
        .pivot_table(index="method", columns="bin", values=value, aggfunc="mean", observed=True)
        .reindex(METHODS).dropna(how="all")
    )
    lines = ["| Method | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for method, row in table.iterrows():
        lines.append(f"| {method} | " + " | ".join(fmt(v) for v in row) + " |")
    return lines


def results_table(means: pd.DataFrame, results: dict[str, dict]) -> list[str]:
    lines = [
        "| Method | Coverage | 95% CI | Mean width (MW) | Interval score | vs shipped "
        "| 95% CI | Holm p | Calibrated | Sharper | Meets the rule |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, row in means.iterrows():
        r, cov = results[method], results[method]["coverage"]
        cells = [
            method, f"{100 * cov['mean_coverage']:.1f}%",
            f"[{100 * cov['ci_lower']:.1f}, {100 * cov['ci_upper']:.1f}]",
            f"{row.width:,.0f}", f"{row.interval_score:,.0f}",
        ]
        if method == SHIPPED:
            cells += ["", "", "", "yes" if r["calibrated"] else "no", "", "(baseline)"]
        else:
            c = r["interval_score_vs_shipped"]
            cells += [
                f"{c['mean_diff']:+,.0f}", f"[{c['ci_lower']:+,.0f}, {c['ci_upper']:+,.0f}]",
                fmt_p(c["holm_p"]), "yes" if r["calibrated"] else "no",
                "yes" if r["sharper"] else "no", "yes" if r["adopt"] else "no",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="refetch the full SMARD history")
    parser.add_argument("--eval-days", type=int, default=365)
    parser.add_argument("--max-origins", type=int, default=None, help="cap origins (smoke test)")
    parser.add_argument("--report-only", action="store_true", help=f"rebuild from {HOURLY_CACHE}")
    args = parser.parse_args()

    data = load_history(args.refresh)
    if args.report_only:
        if not HOURLY_CACHE.exists():
            parser.error(f"{HOURLY_CACHE} not found; run once without --report-only")
        hourly = pd.read_csv(HOURLY_CACHE, parse_dates=["origin", "timestamp"])
    else:
        last = data.dropna(subset=["demand_mw"]).timestamp.max() - pd.Timedelta(hours=HORIZON)
        first = last - pd.Timedelta(days=args.eval_days + CAL_ORIGINS)
        origins = list(pd.date_range(first, last, freq="24h"))
        if args.max_origins is not None:
            origins = origins[-(args.max_origins + CAL_ORIGINS):]
        print(f"{len(origins)} origins ({CAL_ORIGINS} warm-up), {origins[0]:%Y-%m-%d} to "
              f"{origins[-1]:%Y-%m-%d}", flush=True)
        hourly = run(data, origins)
        hourly.to_csv(HOURLY_CACHE, index=False)

    hourly_scores = score(hourly, build_bands(hourly))
    per_origin = (
        hourly_scores.groupby(["origin", "method"])
        .agg(coverage=("covered", "mean"), width=("width", "mean"),
             interval_score=("interval_score", "mean"))
        .reset_index()
    )
    results, chosen = decide(per_origin)
    means = per_origin.groupby("method")[["coverage", "width", "interval_score"]].mean()
    means = means.reindex([m for m in METHODS if m in means.index])

    reliability = {}
    for level in LEVELS:
        level_scores = score(hourly, build_bands(hourly, level))
        reliability[level] = (
            level_scores.groupby(["method", "origin"]).covered.mean().groupby("method").mean()
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_coverage_by_horizon(hourly_scores, FIG_DIR / "interval_study_horizon.png")
    plot_tradeoff(means, FIG_DIR / "interval_study_tradeoff.png")
    per_origin.to_csv(ORIGINS_PATH, index=False)

    eval_origins = sorted(per_origin.origin.unique())
    local = pd.to_datetime(per_origin.origin, utc=True).dt.tz_convert("Europe/Berlin")
    winter = local.dt.month.isin([12, 1, 2])
    season = (
        per_origin.assign(season=np.where(winter, "Winter", "Rest of year"))
        .pivot_table(index="method", columns="season", values="coverage", aggfunc="mean")
        .reindex(means.index)
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    SUMMARY_PATH.write_text(json.dumps({
        "generated": generated, "model_version": MODEL_VERSION,
        "origins": len(eval_origins), "first_origin": str(eval_origins[0]),
        "last_origin": str(eval_origins[-1]), "alpha": ALPHA, "cal_origins": CAL_ORIGINS,
        "aci_gamma": ACI_GAMMA, "results": results, "chosen": chosen,
        "means": means.round(4).to_dict(orient="index"),
        "reliability": {str(k): v.round(4).to_dict() for k, v in reliability.items()},
        "versions": versions,
    }, indent=2, default=str) + "\n", encoding="utf-8")

    if chosen == SHIPPED:
        verdict = (
            "**Verdict: keep the shipped band.** No candidate met both parts of the rule set "
            "before the run."
        )
    else:
        c, cov = results[chosen]["interval_score_vs_shipped"], results[chosen]["coverage"]
        verdict = (
            f"**Verdict: switch to {chosen}.** Its coverage is {100 * cov['mean_coverage']:.1f}% "
            f"(CI [{100 * cov['ci_lower']:.1f}, {100 * cov['ci_upper']:.1f}] contains 95%) and its "
            f"interval score is {-c['mean_diff']:,.0f} lower than the shipped band's (CI "
            f"[{c['ci_lower']:+,.0f}, {c['ci_upper']:+,.0f}], Holm p = {fmt_p(c['holm_p'])})."
        )

    reliability_rows = [
        f"| {m} | " + " | ".join(f"{100 * reliability[lv][m]:.1f}%" for lv in LEVELS) + " |"
        for m in METHODS if all(m in reliability[lv] for lv in LEVELS)
    ]
    shipped_cov = results[SHIPPED]["coverage"]["mean_coverage"]
    bench_cov = results[BENCHMARK_RECIPE]["coverage"]["mean_coverage"]
    report = [
        "# Prediction-band study",
        "",
        f"Generated {generated} by `scripts/interval_study.py` (model {MODEL_VERSION}, Python "
        f"{versions['python']}, scikit-learn {versions['scikit-learn']}).",
        "",
        "Question: the forecast's 95% band is built from recent errors. Does 95% of demand "
        "actually land inside it, and is there a band that does while staying narrow?",
        "",
        "## Two recipes that were treated as one",
        "",
        "The dashboard and API size the band from one 168-hour recursive forecast over the week "
        "before the origin, so its error pool mixes horizons from 1 to 168 hours. The weekly "
        "benchmark has been reporting coverage for a different recipe: the 24-hour errors of "
        f"the previous {TRAILING_ORIGINS} origins. On the same origins here, the shipped band "
        f"covers {100 * shipped_cov:.1f}% and the benchmark's recipe {100 * bench_cov:.1f}%.",
        "",
        "## Design",
        "",
        f"- {len(eval_origins)} daily origins from {eval_origins[0]:%Y-%m-%d} to "
        f"{eval_origins[-1]:%Y-%m-%d}, after {CAL_ORIGINS} warm-up origins used only as "
        "calibration history. Point forecasts come from the shipped model.",
        "- Calibration pools end with the previous origin, whose 24 hours are all observed by "
        "the current one, so no band sees the hours it scores.",
        f"- Adaptive conformal uses gamma = {ACI_GAMMA}, alpha clipped to {ACI_CLIP}. Quantile "
        "GBM uses the champion's parameters with `loss=\"quantile\"` at 2.5% and 97.5%.",
        "- The rule, set before the run: switch only if the coverage CI contains 95% and the "
        "paired interval-score CI against the shipped band is entirely below zero with Holm "
        "p < 0.05 across the five candidates.",
        "",
        "## Result",
        "",
        *results_table(means, results),
        "",
        verdict,
        "",
        "![Coverage against width](figures/interval_study_tradeoff.png)",
        "",
        "## By horizon",
        "",
        "A pooled band has one width for every hour ahead, but error grows with the horizon, "
        "so a pooled band tends to be too wide early and too narrow late.",
        "",
        "![Coverage by horizon](figures/interval_study_horizon.png)",
        "",
        "Coverage:",
        "",
        *horizon_table(hourly_scores, "covered", lambda v: f"{100 * v:.1f}%"),
        "",
        "Mean width (MW):",
        "",
        *horizon_table(hourly_scores, "width", lambda v: f"{v:,.0f}"),
        "",
        "## By season",
        "",
        "Mean per-origin coverage. Descriptive only.",
        "",
        "| Method | " + " | ".join(season.columns) + " |",
        "|---" * (len(season.columns) + 1) + "|",
        *[
            f"| {m} | " + " | ".join(f"{100 * v:.1f}%" for v in row) + " |"
            for m, row in season.iterrows()
        ],
        "",
        "## Reliability across levels",
        "",
        "Mean per-origin coverage when each residual-pool method targets other levels. A "
        "calibrated method sits on the diagonal (50% at 50%, and so on). Adaptive conformal "
        "and the quantile methods were built for the 95% level only.",
        "",
        "| Method | " + " | ".join(f"{int(100 * lv)}% band" for lv in LEVELS) + " |",
        "|---" * (len(LEVELS) + 1) + "|",
        *reliability_rows,
        "",
        "## Files",
        "",
        f"- `{ORIGINS_PATH.name}`: per-origin coverage, width and interval score per method.",
        f"- `{SUMMARY_PATH.name}`: every number in this report.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}; chosen: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
