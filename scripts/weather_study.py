#!/usr/bin/env python3
"""Does temperature help the forecaster, using only forecasts that existed at the time?

The model uses no weather. Demand in Germany rises with heating in winter, and the
operators' forecast (which uses weather) beats the model mostly in winter
(reports/tso_study.md). Adding temperature is the obvious next feature, and also the
easiest place to leak: training and scoring on *observed* temperature lets the model see
the weather it is supposed to be forecasting.

So every arm trains and predicts on one weather source, and the sources are ordered by
how much they know:

* No weather: the shipped v2 model. The baseline.
* Forecast 48 h ahead: GFS 2 m temperature as it was forecast 48 hours before each hour
  (Open-Meteo's previous-runs archive, `_previous_day2`). Every value was published well
  before any origin that uses it, so this is the arm the project could really run.
* Forecast 24 h ahead (`_previous_day1`): a fresher forecast. For the last few hours of
  a 24 h horizon the run behind the value may have been published a few hours after the
  origin, so this arm can be slightly optimistic.
* Observed (oracle): ERA5 reanalysis. Not available at forecast time; it bounds what a
  perfect weather forecast would add. The gap between it and the 48 h arm is the honest
  price of forecasting the weather.

Temperature is a population-weighted mean over Germany's ten largest cities. Two features
are added: the hour's temperature and its trailing 24 h mean (thermal inertia of
buildings), both read from the same source as the arm. GFS is pinned because it is the
only model in the archive with 2 m temperature before 2024, which a two-year training
window needs.

Rule, fixed before the run: per-origin MAE is the unit. Each weather arm is compared with
no weather by paired bootstrap 95% CI and Wilcoxon p, Holm-corrected across the three.
Weather is worth shipping only if the 48 h arm's CI lies entirely below zero and its Holm
p < 0.05. The other two arms are context, never the decision.

    PYTHONPATH=src python scripts/weather_study.py               # ~20 min
    PYTHONPATH=src python scripts/weather_study.py --report-only # from the cache
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import httpx
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")  # report script: never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402
from window_study import SEASONS, evaluation_origins, holm, load_history  # noqa: E402

from energy_forecast.features import (  # noqa: E402
    FEATURE_COLUMNS,
    add_features,
    feature_row,
    german_holidays,
)
from energy_forecast.inference import paired_model_comparison  # noqa: E402
from energy_forecast.models import (  # noqa: E402
    MODEL_VERSION,
    HistGradientBoostingForecast,
    _timestamps,
    trailing_window,
)
from energy_forecast.theme import BLUE, GREEN, INK, MUTED, ORANGE  # noqa: E402
from energy_forecast.theme import style_axes as _style_axes  # noqa: E402

WEATHER_PATH = Path("data/clean/weather_history.csv")  # gitignored by data/clean/*.csv
HOURLY_CACHE = Path("data/clean/weather_study_hourly.csv")  # gitignored
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/weather_study.md")
ORIGINS_PATH = Path("reports/weather_study_origins.csv")
SUMMARY_PATH = Path("reports/weather_study_summary.json")

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_MODEL = "gfs_seamless"
WEATHER_START = "2022-01-01"

# Germany's ten largest cities, population in thousands (Destatis, 2023, rounded).
CITIES = [
    ("Berlin", 52.52, 13.41, 3755), ("Hamburg", 53.55, 9.99, 1892),
    ("Munich", 48.14, 11.58, 1512), ("Cologne", 50.94, 6.96, 1084),
    ("Frankfurt", 50.11, 8.68, 776), ("Stuttgart", 48.78, 9.18, 633),
    ("Duesseldorf", 51.23, 6.78, 631), ("Leipzig", 51.34, 12.37, 616),
    ("Dortmund", 51.51, 7.47, 595), ("Essen", 51.46, 7.01, 584),
]

NONE = "No weather (v2)"
D2 = "Forecast 48 h ahead"
D1 = "Forecast 24 h ahead"
OBS = "Observed (oracle)"
ARMS = [NONE, D2, D1, OBS]
CANDIDATES = [D2, D1, OBS]
HEADLINE = D2
SOURCE = {D2: "temp_d2", D1: "temp_d1", OBS: "temp_obs"}
COLORS = dict(zip(ARMS, [MUTED, BLUE, GREEN, ORANGE]))
WEATHER_COLUMNS = ["temp_c", "temp_mean_24h"]
MIN_PAIRED = 5


def weighted_temperature(payloads: list[dict], variable: str) -> pd.Series:
    """Population-weighted mean of one hourly variable across the CITIES payloads.

    A city missing an hour is left out of that hour's mean (weights renormalised); an
    hour missing everywhere stays NaN. Nothing is interpolated.
    """
    weights = np.array([city[3] for city in CITIES[: len(payloads)]], dtype=float)
    columns = []
    for payload in payloads:
        hourly = payload["hourly"]
        index = pd.DatetimeIndex(pd.to_datetime(hourly["time"]), name="timestamp")
        values = [np.nan if v is None else v for v in hourly[variable]]
        columns.append(pd.Series(values, index=index.tz_localize("UTC"), dtype=float))
    values = pd.concat(columns, axis=1)
    present = values.notna().to_numpy()
    total = (values.fillna(0).to_numpy() * weights).sum(axis=1)
    weight = (present * weights).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(weight > 0, total / weight, np.nan)
    return pd.Series(mean, index=values.index, name=variable)


def _fetch(url: str, variables: str, start: str, end: str, **extra) -> list[dict]:
    params = {
        "latitude": ",".join(str(c[1]) for c in CITIES),
        "longitude": ",".join(str(c[2]) for c in CITIES),
        "hourly": variables, "start_date": start, "end_date": end, "timezone": "GMT", **extra,
    }
    response = httpx.get(url, params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else [payload]


def load_weather(refresh: bool, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly weighted temperature from all three sources, fetched by year and cached."""
    if WEATHER_PATH.exists() and not refresh:
        frame = pd.read_csv(WEATHER_PATH)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        return frame
    parts = []
    last_day = end.tz_convert("UTC").tz_localize(None).normalize()
    for year in range(pd.Timestamp(WEATHER_START).year, last_day.year + 1):
        start = f"{year}-01-01"
        stop = min(pd.Timestamp(f"{year}-12-31"), last_day).strftime("%Y-%m-%d")
        print(f"  weather {start} to {stop}", flush=True)
        forecast = _fetch(
            PREVIOUS_RUNS_URL, "temperature_2m_previous_day1,temperature_2m_previous_day2",
            start, stop, models=FORECAST_MODEL,
        )
        observed = _fetch(ARCHIVE_URL, "temperature_2m", start, stop, models="era5")
        parts.append(pd.DataFrame({
            "temp_d1": weighted_temperature(forecast, "temperature_2m_previous_day1"),
            "temp_d2": weighted_temperature(forecast, "temperature_2m_previous_day2"),
            "temp_obs": weighted_temperature(observed, "temperature_2m"),
        }))
    frame = pd.concat(parts).sort_index().rename_axis("timestamp").reset_index()
    WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(WEATHER_PATH, index=False)
    return frame


def weather_features(temperature: pd.Series) -> pd.DataFrame:
    """The hour's temperature and its trailing 24 h mean, from one source only.

    Both read the source series at or before the hour; for a forecast source every value
    is the forecast made for that hour, so no observation enters.
    """
    temperature = temperature.sort_index().asfreq("1h")
    return pd.DataFrame({
        "temp_c": temperature,
        "temp_mean_24h": temperature.rolling(24, min_periods=18).mean(),
    })


class WeatherForecast(HistGradientBoostingForecast):
    """The shipped champion plus WEATHER_COLUMNS read from a fixed hourly table.

    Missing weather stays NaN and goes to the regressor, which handles it natively; it
    never triggers the last-value fallback, which only the demand features can trigger.
    """

    name = "hist_gradient_boosting_weather"

    def __init__(self, weather: pd.DataFrame, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weather = weather
        self.columns = FEATURE_COLUMNS + WEATHER_COLUMNS

    def _with_weather(self, frame: pd.DataFrame) -> pd.DataFrame:
        stamps = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True))
        values = self.weather.reindex(stamps)
        return frame.assign(**{c: values[c].to_numpy() for c in WEATHER_COLUMNS})

    def fit(self, frame: pd.DataFrame) -> "WeatherForecast":
        frame = trailing_window(frame, self.train_window_days)
        features = add_features(frame).dropna(subset=FEATURE_COLUMNS + ["demand_mw"])
        features = self._with_weather(features)
        self.regressor.fit(features[self.columns], features["demand_mw"])
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
        self.history = data.set_index("timestamp")["demand_mw"].sort_index().astype(float).dropna()
        return self

    def forecast_with_rows(self, timestamps: pd.DatetimeIndex):
        history = self.history.copy()
        predictions, rows = [], []
        for timestamp in _timestamps(timestamps):
            row = feature_row(history, timestamp)
            weather = self.weather.reindex([timestamp])
            row.update({c: float(weather[c].iloc[0]) for c in WEATHER_COLUMNS})
            rows.append(row)
            if any(pd.isna(row[column]) for column in FEATURE_COLUMNS):
                prediction = float(history.iloc[-1])
            else:
                prediction = float(self.regressor.predict(pd.DataFrame([row])[self.columns])[0])
            predictions.append(prediction)
            history.loc[timestamp] = prediction
        return np.asarray(predictions), pd.DataFrame(rows)

    def forecast(self, timestamps):
        return self.forecast_with_rows(timestamps)[0]

    predict = forecast


def forecast_origin(
    data: pd.DataFrame, tables: dict[str, pd.DataFrame | None], origin: pd.Timestamp,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Every arm's 24 h forecast from one origin, trained on `timestamp <= origin`."""
    target = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon_hours,
                           freq="1h", tz="UTC")
    train = data[data.timestamp <= origin]
    out = {"origin": origin, "timestamp": target}
    for arm, table in tables.items():
        model = (HistGradientBoostingForecast() if table is None
                 else WeatherForecast(table)).fit(train)
        out[arm] = model.predict(target)
    return pd.DataFrame(out)


def run(data: pd.DataFrame, weather: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.DataFrame:
    # Serial on purpose: a process pool forks after scikit-learn's OpenMP runtime has
    # started, and the forked workers spun at full CPU without finishing an origin.
    indexed = weather.set_index("timestamp")
    tables = {NONE: None, **{arm: weather_features(indexed[SOURCE[arm]]) for arm in CANDIDATES}}
    parts = []
    for i, origin in enumerate(origins, 1):
        parts.append(forecast_origin(data, tables, origin))
        if i % 25 == 0 or i == len(origins):
            print(f"  origin {i}/{len(origins)}", flush=True)
    wide = pd.concat(parts, ignore_index=True)
    actual = data.set_index("timestamp").demand_mw
    wide["actual"] = actual.reindex(pd.DatetimeIndex(wide.timestamp)).to_numpy(dtype=float)
    hourly = wide.melt(id_vars=["origin", "timestamp", "actual"], value_vars=ARMS,
                       var_name="arm", value_name="predicted").dropna(subset=["actual"])
    temps = indexed.temp_obs.reindex(pd.DatetimeIndex(hourly.timestamp))
    hourly["temp_obs"] = temps.to_numpy()
    return hourly


def annotate(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["error"] = hourly.predicted - hourly.actual
    hourly["abs_error"] = hourly.error.abs()
    local = pd.to_datetime(hourly.timestamp, utc=True).dt.tz_convert("Europe/Berlin")
    hourly["season"] = local.dt.month.map(SEASONS)
    hourly["is_holiday"] = local.dt.date.isin(german_holidays(set(local.dt.year.unique())))
    return hourly


def compare(per_origin: pd.DataFrame, value: str = "mae") -> dict[str, dict]:
    wide = per_origin.pivot(index="origin", columns="arm", values=value).dropna()
    if len(wide) < MIN_PAIRED or not set(ARMS) <= set(wide.columns):
        return {}
    results = {arm: paired_model_comparison(wide[arm], wide[NONE]) for arm in CANDIDATES}
    adjusted = holm({arm: r["wilcoxon_p"] for arm, r in results.items()})
    for arm, r in results.items():
        r["holm_p"] = adjusted[arm]
        r["beats"] = bool(r["ci_upper"] < 0 and adjusted[arm] < 0.05)
    return results


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def comparison_table(results: dict[str, dict], base: float) -> list[str]:
    lines = [
        "| Arm | Mean MAE (MW) | vs no weather (MW) | 95% CI | Wilcoxon p | Holm p | Beats |",
        "|---|---|---|---|---|---|---|",
        f"| {NONE} | {base:,.0f} | | | | | |",
    ]
    for arm, r in results.items():
        lines.append(
            f"| {arm} | {base + r['mean_diff']:,.0f} | {r['mean_diff']:+,.0f} | "
            f"[{r['ci_lower']:+,.0f}, {r['ci_upper']:+,.0f}] | {fmt_p(r['wilcoxon_p'])} | "
            f"{fmt_p(r['holm_p'])} | {'yes' if r['beats'] else 'no'} |"
        )
    return lines


def subset_table(per_origin: pd.DataFrame, subsets: list[tuple[str, pd.Series]]) -> list[str]:
    lines = ["| Subset | Origins | No weather MAE | " + " | ".join(f"{a} vs none" for a in
             CANDIDATES) + " |", "|---" * (len(CANDIDATES) + 3) + "|"]
    for name, mask in subsets:
        part = per_origin[per_origin.origin.isin(mask[mask].index)]
        results = compare(part)
        if not results:
            continue
        base = part[part.arm == NONE].mae.mean()
        cells = []
        for arm in CANDIDATES:
            r = results[arm]
            mark = " *" if r["ci_upper"] < 0 or r["ci_lower"] > 0 else ""
            cells.append(f"{r['mean_diff']:+,.0f} [{r['ci_lower']:+,.0f}, "
                         f"{r['ci_upper']:+,.0f}]{mark}")
        lines.append(f"| {name} | {results[D2]['n']} | {base:,.0f} | " + " | ".join(cells) + " |")
    return lines


def forecast_skill(weather: pd.DataFrame, start: pd.Timestamp) -> dict[str, float]:
    """MAE of each forecast source against ERA5 over the scored period, in deg C."""
    recent = weather[weather.timestamp >= start]
    return {arm: float((recent[SOURCE[arm]] - recent.temp_obs).abs().mean())
            for arm in (D2, D1)}


def plot_effects(results: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for y, arm in enumerate(CANDIDATES):
        r = results[arm]
        err = [[r["mean_diff"] - r["ci_lower"]], [r["ci_upper"] - r["mean_diff"]]]
        ax.errorbar(r["mean_diff"], y, xerr=err, fmt="o", color=COLORS[arm],
                    ecolor=COLORS[arm], capsize=4, markersize=7)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(range(len(CANDIDATES)), CANDIDATES)
    ax.invert_yaxis()
    ax.set_xlabel("Mean per-origin MAE minus no weather (MW); left of 0 is better")
    ax.set_title("Effect of adding temperature, with 95% bootstrap CIs", loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_by_temperature(hourly: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Hourly MAE by observed-temperature band: where the weather feature earns its keep."""
    bins = [-30, 0, 5, 10, 15, 20, 25, 45]
    labels = ["< 0", "0-5", "5-10", "10-15", "15-20", "20-25", "> 25"]
    banded = hourly.assign(band=pd.cut(hourly.temp_obs, bins, labels=labels))
    table = banded.pivot_table(index="band", columns="arm", values="abs_error",
                               aggfunc="mean", observed=True)[ARMS]
    counts = banded[banded.arm == NONE].groupby("band", observed=True).size()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    width = 0.8 / len(ARMS)
    x = np.arange(len(table))
    for i, arm in enumerate(ARMS):
        ax.bar(x + (i - 1.5) * width, table[arm], width, color=COLORS[arm], label=arm)
    ax.set_xticks(x, [str(b) for b in table.index])
    ax.set_xlabel("Observed temperature, population-weighted (deg C)")
    ax.set_ylabel("Hourly MAE (MW)")
    ax.set_title("Forecast error by temperature", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return table.assign(hours=counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="refetch SMARD and weather")
    parser.add_argument("--eval-days", type=int, default=365)
    parser.add_argument("--max-origins", type=int, default=None, help="cap origins (smoke)")
    parser.add_argument("--report-only", action="store_true",
                        help=f"rebuild the report from {HOURLY_CACHE} without refitting")
    args = parser.parse_args()

    data = load_history(args.refresh)
    weather = load_weather(args.refresh, data.timestamp.max())
    # ERA5 lags real time by about five days; score only hours every source covers.
    covered = weather.dropna(subset=list(SOURCE.values())).timestamp.max()
    if args.report_only:
        if not HOURLY_CACHE.exists():
            parser.error(f"{HOURLY_CACHE} not found; run once without --report-only")
        hourly = pd.read_csv(HOURLY_CACHE)
        hourly["origin"] = pd.to_datetime(hourly.origin, utc=True)
        hourly["timestamp"] = pd.to_datetime(hourly.timestamp, utc=True)
        origins = sorted(hourly.origin.unique())
    else:
        origins = evaluation_origins(data[data.timestamp <= covered], args.eval_days)
        if args.max_origins is not None:
            origins = origins[-args.max_origins:]
        window = pd.Timedelta(days=HistGradientBoostingForecast.DEFAULT_TRAIN_WINDOW_DAYS)
        if origins[0] - window < weather.timestamp.min():
            parser.error("weather history does not cover the first origin's training window")
        print(f"{len(origins)} origins x {len(ARMS)} arms, {origins[0]:%Y-%m-%d} to "
              f"{origins[-1]:%Y-%m-%d}", flush=True)
        hourly = run(data, weather, origins)
        hourly.to_csv(HOURLY_CACHE, index=False)
    hourly = annotate(hourly)

    per_origin = (hourly.groupby(["origin", "arm"])
                  .agg(mae=("abs_error", "mean"), bias=("error", "mean")).reset_index())
    results = compare(per_origin)
    mean_mae = per_origin.groupby("arm").mae.mean()
    holiday = (hourly[hourly.is_holiday].groupby(["origin", "arm"]).abs_error.mean()
               .rename("mae").reset_index())
    holiday_results = compare(holiday)
    first = pd.Timestamp(origins[0])
    skill = forecast_skill(weather, first)
    season_of_origin = (hourly[hourly.arm == NONE].groupby("origin").season
                        .agg(lambda s: s.mode().iloc[0]))
    subsets = [(s, season_of_origin == s) for s in ["Winter", "Spring", "Summer", "Autumn"]]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_effects(results, FIG_DIR / "weather_study_effects.png")
    by_temp = plot_by_temperature(hourly, FIG_DIR / "weather_study_by_temperature.png")
    per_origin.to_csv(ORIGINS_PATH, index=False)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    head = results[HEADLINE]
    ship = head["beats"]
    summary = {
        "generated": generated, "model_version": MODEL_VERSION,
        "origins": len(origins), "first_origin": str(origins[0]),
        "last_origin": str(origins[-1]), "forecast_model": FORECAST_MODEL,
        "cities": [c[0] for c in CITIES], "mean_mae": mean_mae.round(1).to_dict(),
        "comparisons": results, "holiday_comparisons": holiday_results,
        "temperature_forecast_mae_c": {k: round(v, 2) for k, v in skill.items()},
        "headline": HEADLINE, "ship_weather": ship,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    if ship:
        verdict = (
            f"**Verdict: temperature helps, with forecasts the project could really use.** "
            f"The {HEADLINE.lower()} arm lowers mean per-origin MAE by "
            f"{-head['mean_diff']:,.0f} MW (95% CI [{head['ci_lower']:+,.0f}, "
            f"{head['ci_upper']:+,.0f}], Holm p = {fmt_p(head['holm_p'])}), which meets the "
            "rule set before the run."
        )
    else:
        verdict = (
            f"**Verdict: not shown to help.** The {HEADLINE.lower()} arm changes mean "
            f"per-origin MAE by {head['mean_diff']:+,.0f} MW (95% CI [{head['ci_lower']:+,.0f},"
            f" {head['ci_upper']:+,.0f}], Holm p = {fmt_p(head['holm_p'])}), which does not "
            "meet the rule set before the run."
        )
    oracle_gap = results[OBS]["mean_diff"] - head["mean_diff"]
    if results[OBS]["beats"]:
        oracle_line = (
            f"Perfect weather would change mean MAE by a further {oracle_gap:+,.0f} MW "
            "relative to the 48 h forecast. That gap is the price of having to forecast the "
            "weather."
        )
    else:
        oracle_line = (
            "Even observed weather does not meet the rule "
            f"({results[OBS]['mean_diff']:+,.0f} MW, 95% CI [{results[OBS]['ci_lower']:+,.0f},"
            f" {results[OBS]['ci_upper']:+,.0f}]), so the limit is not the accuracy of the "
            "weather forecast. The likely reason: at a 24-hour horizon the lag features "
            "already carry most of what temperature says, because yesterday's load at the "
            "same hour was drawn under nearly the same weather."
        )
    holiday_lines = (comparison_table(holiday_results, holiday[holiday.arm == NONE].mae.mean())
                     if holiday_results else ["Too few origins with a holiday hour."])
    report = [
        "# Weather study",
        "",
        f"Generated {generated} by `scripts/weather_study.py` (Python "
        f"{summary['versions']['python']}, scikit-learn {summary['versions']['scikit-learn']}, "
        f"model {MODEL_VERSION}).",
        "",
        "Question: does temperature make the 24-hour forecast better, if the model may only "
        "use weather forecasts that existed when its own forecast was issued?",
        "",
        "## Design",
        "",
        f"- {len(origins)} daily origins from {first:%Y-%m-%d} to "
        f"{pd.Timestamp(origins[-1]):%Y-%m-%d}, each scored on the next 24 hours. Every arm "
        "trains on the same two years before the origin (the shipped v2 window).",
        "- Temperature: population-weighted mean of Germany's ten largest cities. Two "
        "features, the hour's temperature and its trailing 24-hour mean.",
        f"- Each arm trains *and* predicts on one source. Forecasts are {FORECAST_MODEL} from "
        "Open-Meteo's previous-runs archive; observed is ERA5 reanalysis.",
        f"  - **{D2}**: published long before any origin that uses it. The realistic arm.",
        f"  - **{D1}**: fresher; for the last hours of the horizon the run may have been "
        "published a few hours after the origin, so it can be slightly optimistic.",
        f"  - **{OBS}**: not available at forecast time. An upper bound.",
        "- Rule, set before the run: per-origin MAE, paired bootstrap 95% CI and Wilcoxon p "
        f"against no weather, Holm-corrected over three. Weather ships only if the {D2} arm's "
        "CI is entirely below zero and its Holm p < 0.05.",
        "",
        "Over the scored period the temperature forecasts miss ERA5 by "
        f"{skill[D2]:.2f} deg C (48 h) and {skill[D1]:.2f} deg C (24 h) on average.",
        "",
        "## Result",
        "",
        *comparison_table(results, mean_mae[NONE]),
        "",
        verdict,
        "",
        oracle_line,
        "",
        "![Effect of each weather source](figures/weather_study_effects.png)",
        "",
        "## By season",
        "",
        "Same paired comparison within each season of the origin; * marks a CI that excludes "
        "zero. Not part of the decision rule.",
        "",
        *subset_table(per_origin, subsets),
        "",
        "## By temperature",
        "",
        "Hourly MAE (MW) by observed temperature band. Descriptive only.",
        "",
        "| Band (deg C) | Hours | " + " | ".join(ARMS) + " |",
        "|---" * (len(ARMS) + 2) + "|",
        *[f"| {band} | {int(row.hours):,} | " + " | ".join(f"{row[a]:,.0f}" for a in ARMS)
          + " |" for band, row in by_temp.iterrows()],
        "",
        "![Error by temperature](figures/weather_study_by_temperature.png)",
        "",
        "## Public holidays",
        "",
        "Holiday-hour MAE over origins whose window contains a holiday hour. Indicative only.",
        "",
        *holiday_lines,
        "",
        "## Caveats",
        "",
        "- One weather model (GFS) at ten points. A finer model (ICON) is only archived from "
        "2024, too short for a two-year training window with a year of scoring behind it.",
        "- Temperature only. Solar radiation matters too (rooftop solar lowers measured grid "
        "load), but the archive has it only from 2024 for the same reason.",
        "- Shipping would serve fresher live forecasts than the 48 h ones the model trained "
        "on. That mismatch is in the safe direction but is a mismatch; it would need its own "
        "check once live forecasts are logged.",
        "",
        "## Files",
        "",
        f"- `{ORIGINS_PATH.name}`: per-origin MAE and bias for every arm.",
        f"- `{SUMMARY_PATH.name}`: every number in this report.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}; ship weather: {ship}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
