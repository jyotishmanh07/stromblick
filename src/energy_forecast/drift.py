"""Model drift: is the model worse than it used to be?

Everything else in this repo measures quality *within* one run. `inference.py`
compares the champion against a rival on the same origins; `coverage_test` compares
the band against a nominal constant; the reports are overwritten in place. Nothing
stores a prior value or compares against one. This module is that missing axis, and
it is separate for two concrete reasons: `inference.py` is pure numpy/scipy with no
I/O (persistence is exactly what makes it trivially testable), and `verification.py`'s
unit of work is one issued 24-hour forecast. **Drift's unit is a run.**

It may import from `inference.py`; nothing imports back from here.

Why a season-matched rank and never a z-score
---------------------------------------------
Champion per-origin MAE over the 329 backtest origins, by month::

    Jan 2855 | Feb 1758 | Mar 1645 | Apr 2498 | May 1768 | Jun 1482
    Jul 1387 | Aug 1357 | Oct 2623 | Nov 1996 | Dec 2052

January is 2.10x August *with no drift at all*, and the distribution is right-skewed
(mean 1944, median 1548, max 10370). A z-score against the pooled distribution would
flag every winter, permanently. So the reference is matched by day-of-year and the
statistic is a percentile within that reference.

Why the reference window is adaptive
------------------------------------
A window centred on a seasonal peak sits high in its own neighbourhood, so too wide a
reference re-introduces the problem the season match exists to solve. Measured on the
real 329 origins at two widths::

    mid-Jan  +/-21d  n= 42  pct 0.667   |  +/-45d  n= 90  pct 0.744
    mid-Apr  +/-21d  n= 43  pct 0.605   |  +/-45d  n= 91  pct 0.637
    mid-Jul  +/-21d  n= 43  pct 0.349   |  +/-45d  n= 91  pct 0.374
    mid-Oct  +/-21d  n= 32  pct 0.781   |  +/-45d  n= 56  pct 0.786

Tighter is uniformly better at the peaks and still clears `MIN_REFERENCE_ORIGINS` on a
single year. `seasonal_reference` therefore starts at +/-21 days and widens (21 -> 30
-> 45) only until it has enough origins, recording the width it used on the `Verdict`.
The payoff grows with retained history: once two winters exist, a +/-21-day reference
for a January window is mostly *previous Januaries*, which is the comparison we
actually want, and is why the retained history file is the primary artifact.

Honest consequence: the **first** winter will likely read "watch", because there is no
prior winter to compare it against. That is the truth about what the data supports, not
a bug. `regressed` still cannot fire from it without two consecutive runs above the
0.90 threshold.

Thresholds and their calibration
--------------------------------
Under no drift, a percentile is uniform, so P(percentile > 0.90) is the false-alarm
rate per run: **2.2% for a 14-origin window, 5.0% for a 7-origin window**. That gap is
why `live_verdict` passes `allow_regressed=False` -- a 7-day replay is too noisy to
condemn on and can reach "watch" but never "regressed" -- and why "regressed" from the
weekly path additionally requires two consecutive runs above 0.90. Keep these numbers
attached to the thresholds; bare constants erode.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .models import HistGradientBoostingForecast

CHAMPION = "HistGradientBoosting"
HISTORY_DIR = Path("reports/history")
RUN_HISTORY_PATH = HISTORY_DIR / "benchmark_runs.csv"
ORIGIN_HISTORY_PATH = HISTORY_DIR / "origin_scores.csv"
SEASON_WINDOW_DAYS = 21  # starting half-width, in days of the year
MAX_SEASON_WINDOW_DAYS = 45  # cap; widen only until the reference is big enough
RECENT_ORIGINS = 14
MIN_REFERENCE_ORIGINS = 20
MIN_COMPLETE_ORIGINS = 200
WATCH_PERCENTILE = 0.80
REGRESSED_PERCENTILE = 0.90
COVERAGE_WATCH_DROP = 0.015
DAYS_IN_YEAR = 365.25

ORIGIN_COLUMNS = [
    "origin", "model", "model_version", "params_hash", "mae", "rmse", "smape", "first_seen",
]
RUN_COLUMNS = [
    "run_at", "snapshot_start", "snapshot_end", "rows", "origins", "step_hours",
    "horizon_hours", "eval_start", "eval_end", "eval_hours", "champion", "champion_mae_mean",
    "champion_mae_std", "seasonal_naive_mae_mean", "lift_vs_seasonal_naive_pct",
    "coverage_nominal", "coverage_empirical", "coverage_mean_per_origin",
    "coverage_ci_lower", "coverage_ci_upper", "coverage_p_value", "coverage_origins",
    "mean_band_mw", "model_version", "params_hash", "drift_level", "drift_percentile",
    "drift_window_mae", "drift_reference_median", "drift_reference_origins",
]

_ORIGIN_TIMESTAMPS = ("origin", "first_seen")
_RUN_TIMESTAMPS = ("run_at", "snapshot_start", "snapshot_end", "eval_start", "eval_end")
# Rounded on write: without this a rerun yields a 329-line diff of trailing float digits.
_ORIGIN_ROUNDING = {"mae": 2, "rmse": 2, "smape": 3}
_RUN_ROUNDING = {
    "champion_mae_mean": 2, "champion_mae_std": 2, "seasonal_naive_mae_mean": 2,
    "lift_vs_seasonal_naive_pct": 2, "mean_band_mw": 2, "drift_window_mae": 2,
    "drift_reference_median": 2, "coverage_nominal": 4, "coverage_empirical": 4,
    "coverage_mean_per_origin": 4, "coverage_ci_lower": 4, "coverage_ci_upper": 4,
    "coverage_p_value": 4, "drift_percentile": 4,
}
# 8 hex characters can be all digits, which pandas would read back as an integer and
# then fail to match against the string we wrote. Pin every identifier column to str.
_ORIGIN_DTYPES = {"model": str, "model_version": str, "params_hash": str}
_RUN_DTYPES = {"champion": str, "model_version": str, "params_hash": str, "drift_level": str}


@dataclass(frozen=True)
class Verdict:
    """One drift judgement, with enough context for a dashboard panel to explain it."""

    level: str  # "ok" | "watch" | "regressed" | "unknown"
    headline: str
    reason: str
    window_mae: float = float("nan")
    window_origins: int = 0
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None
    reference_median: float = float("nan")
    reference_origins: int = 0
    reference_window_days: int = 0
    percentile: float = float("nan")
    consecutive_runs: int = 0


def params_fingerprint(
    params: dict | None = None, features: list[str] | None = None
) -> str:
    """8-char hash identifying a *model generation* (defaults plus feature contract).

    Feature order is not part of the identity: `fit` and `predict` both index by
    `FEATURE_COLUMNS`, so a reorder is the same model.
    """
    params = HistGradientBoostingForecast.DEFAULT_PARAMS if params is None else params
    features = FEATURE_COLUMNS if features is None else features
    canonical = json.dumps(
        {"params": {k: params[k] for k in sorted(params)}, "features": sorted(features)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- history


def _empty_origin_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": pd.Series(dtype="datetime64[ns, UTC]"),
            "model": pd.Series(dtype=object),
            "model_version": pd.Series(dtype=object),
            "params_hash": pd.Series(dtype=object),
            "mae": pd.Series(dtype=float),
            "rmse": pd.Series(dtype=float),
            "smape": pd.Series(dtype=float),
            "first_seen": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )[ORIGIN_COLUMNS]


def _empty_run_history() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for name in RUN_COLUMNS:
        if name in _RUN_TIMESTAMPS:
            columns[name] = pd.Series(dtype="datetime64[ns, UTC]")
        elif name in ("champion", "model_version", "params_hash", "drift_level"):
            columns[name] = pd.Series(dtype=object)
        else:
            columns[name] = pd.Series(dtype=float)
    return pd.DataFrame(columns)[RUN_COLUMNS]


def _load(path, columns, timestamps, dtypes, empty) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return empty()
    frame = pd.read_csv(path, dtype=dtypes)
    if frame.empty:
        return empty()
    for column in timestamps:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns]


def load_origin_history(path: str | Path = ORIGIN_HISTORY_PATH) -> pd.DataFrame:
    """Read the retained per-origin scores, or an empty typed frame when there are none."""
    return _load(path, ORIGIN_COLUMNS, _ORIGIN_TIMESTAMPS, _ORIGIN_DTYPES, _empty_origin_history)


def load_run_history(path: str | Path = RUN_HISTORY_PATH) -> pd.DataFrame:
    """Read the retained per-run summaries, or an empty typed frame when there are none."""
    return _load(path, RUN_COLUMNS, _RUN_TIMESTAMPS, _RUN_DTYPES, _empty_run_history)


def is_complete_run(origins: int, max_origins: int | None) -> bool:
    """Did this benchmark run cover the whole snapshot?

    The `--max-origins` flag is the obvious cap, but it misses the other one: an
    uncapped run against a truncated snapshot (a bad ingest) is partial too.
    """
    return max_origins is None and int(origins) >= MIN_COMPLETE_ORIGINS


def _round(frame: pd.DataFrame, rounding: dict[str, int]) -> pd.DataFrame:
    rounded = frame.copy()
    for column, places in rounding.items():
        if column in rounded:
            rounded[column] = pd.to_numeric(rounded[column], errors="coerce").round(places)
    return rounded


def _payload(frame: pd.DataFrame, drop: str, rounding: dict[str, int]) -> pd.DataFrame:
    """Everything but the run-stamp column, rounded to what survives the CSV round-trip.

    Mirrors `verification._log_payload`: a rerun that adds no information must leave the
    file byte-for-byte identical, so a scheduled workflow running twice makes no commit.
    Numeric columns are cast to float because a column read back from CSV as int64 would
    otherwise compare unequal to the same values built in memory.
    """
    payload = _round(frame.drop(columns=[drop]), rounding).reset_index(drop=True)
    for column in payload.columns:
        if column in rounding:
            payload[column] = payload[column].astype(float)
    return payload


def append_origin_history(
    path: str | Path, per_origin: pd.DataFrame, *, model_version: str, params_hash: str,
    run_at: pd.Timestamp | None = None, models: tuple[str, ...] = (CHAMPION,),
) -> tuple[pd.DataFrame, bool]:
    """Merge benchmark per-origin scores into the retained history at `path`.

    `per_origin` is `reports/benchmark_metrics.csv` (origin, model, mae, rmse, smape).
    Returns the merged frame and whether the file was written.
    """
    path = Path(path)
    existing = load_origin_history(path)
    run_at = pd.Timestamp(run_at) if run_at is not None else pd.Timestamp.now(tz="UTC").floor("s")

    rows = per_origin[per_origin.model.isin(models)].copy()
    rows["origin"] = pd.to_datetime(rows.origin, utc=True)
    rows["model_version"] = model_version
    rows["params_hash"] = params_hash
    rows["first_seen"] = run_at
    rows = rows.reindex(columns=ORIGIN_COLUMNS)

    frames = [frame for frame in (existing, rows) if not frame.empty]
    merged = pd.concat(frames, ignore_index=True) if frames else _empty_origin_history()
    merged["origin"] = pd.to_datetime(merged.origin, utc=True)
    merged["first_seen"] = pd.to_datetime(merged.first_seen, utc=True)
    # keep="first" is deliberate. A rerun of an already-recorded origin trains on a
    # front-truncated prefix -- `--weeks 52` makes the snapshot a rolling window -- so it
    # is a slightly different model, not a better measurement of the same one. The first
    # reading is the one taken under the conditions the row claims to describe.
    merged = (
        merged.drop_duplicates(["origin", "model", "params_hash"], keep="first")
        .sort_values(["origin", "model"], kind="mergesort")
        .reset_index(drop=True)[ORIGIN_COLUMNS]
    )
    merged = _round(merged, _ORIGIN_ROUNDING)
    if _payload(merged, "first_seen", _ORIGIN_ROUNDING).equals(
        _payload(existing, "first_seen", _ORIGIN_ROUNDING)
    ):
        return existing, False
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged, True


def append_run_history(
    path: str | Path, record: dict
) -> tuple[pd.DataFrame, bool]:
    """Merge one run summary into the retained run history at `path`.

    Dedup is on `(snapshot_end, params_hash)` keeping the latest: re-benchmarking the
    same snapshot with the same model generation supersedes the earlier attempt.
    """
    path = Path(path)
    existing = load_run_history(path)
    rows = pd.DataFrame([record]).reindex(columns=RUN_COLUMNS)

    frames = [frame for frame in (existing, rows) if not frame.empty]
    merged = pd.concat(frames, ignore_index=True) if frames else _empty_run_history()
    for column in _RUN_TIMESTAMPS:
        merged[column] = pd.to_datetime(merged[column], utc=True)
    merged = (
        # Sorted by the data each row describes, not by when it was computed: re-running
        # an older snapshot must not shuffle rows in a file whose diff is meant to be
        # one appended line a week.
        merged.drop_duplicates(["snapshot_end", "params_hash"], keep="last")
        .sort_values(["snapshot_end", "run_at"], kind="mergesort")
        .reset_index(drop=True)[RUN_COLUMNS]
    )
    merged = _round(merged, _RUN_ROUNDING)
    if _payload(merged, "run_at", _RUN_ROUNDING).equals(
        _payload(existing, "run_at", _RUN_ROUNDING)
    ):
        return existing, False
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged, True


# ------------------------------------------------------------------- summary -> row


def _utc(value) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _fraction(value) -> float:
    """Coverage arrives as either 0.9317 or 93.16 depending on the summary block."""
    number = _number(value)
    if np.isnan(number):
        return number
    return number / 100.0 if number > 1.5 else number


def _model_stat(models: dict, name: str, key: str) -> float:
    if not isinstance(models, dict):
        return float("nan")
    if name in models:
        return _number(models[name].get(key))
    for candidate, stats in models.items():  # tolerate a renamed label
        if isinstance(candidate, str) and name.lower() in candidate.lower():
            return _number(stats.get(key))
    return float("nan")


def run_record_from_summary(
    summary: dict, *, run_at: pd.Timestamp | None = None, verdict: Verdict | None = None,
    params_hash: str | None = None, model_version: str | None = None,
) -> dict:
    """Flatten `reports/benchmark_summary.json` into one row of `RUN_COLUMNS`.

    Every lookup degrades to None/nan rather than raising: the summary schema is written
    by `scripts/benchmark.py` and may change, and a history that silently loses a column
    is better than a workflow that dies. `tests/test_drift.py` reads the real shipped
    summary so a schema change is caught rather than quietly emptying the history.
    """
    summary = summary or {}
    models = summary.get("models", {})
    interval = summary.get("interval_coverage", {}) or {}
    coverage = (summary.get("significance", {}) or {}).get("coverage", {}) or {}
    champion = summary.get("champion", CHAMPION)
    nominal = coverage.get("nominal", interval.get("nominal"))

    return {
        "run_at": _utc(run_at) if run_at is not None else pd.Timestamp.now(tz="UTC").floor("s"),
        "snapshot_start": _utc(summary.get("snapshot_start")),
        "snapshot_end": _utc(summary.get("snapshot_end")),
        "rows": _number(summary.get("rows")),
        "origins": _number(summary.get("origins")),
        "step_hours": _number(summary.get("step_hours")),
        "horizon_hours": _number(summary.get("horizon_hours")),
        "eval_start": _utc(summary.get("eval_start")),
        "eval_end": _utc(summary.get("eval_end")),
        "eval_hours": _number(summary.get("eval_hours")),
        "champion": champion,
        "champion_mae_mean": _model_stat(models, champion, "mae_mean"),
        "champion_mae_std": _model_stat(models, champion, "mae_std"),
        "seasonal_naive_mae_mean": _model_stat(models, "Seasonal naive", "mae_mean"),
        "lift_vs_seasonal_naive_pct": _number(summary.get("lift_vs_seasonal_naive_pct")),
        "coverage_nominal": _fraction(nominal),
        "coverage_empirical": _fraction(interval.get("empirical")),
        "coverage_mean_per_origin": _fraction(coverage.get("mean_coverage")),
        "coverage_ci_lower": _fraction(coverage.get("ci_lower")),
        "coverage_ci_upper": _fraction(coverage.get("ci_upper")),
        "coverage_p_value": _number(coverage.get("p_value")),
        "coverage_origins": _number(coverage.get("n_origins")),
        "mean_band_mw": _number(interval.get("mean_band_mw")),
        # The summary carries no model_version of its own, so the caller supplies it;
        # without it the column is dead weight and the run history cannot be
        # cross-referenced against the forecast log, which does record it.
        "model_version": model_version if model_version is not None
        else summary.get("model_version"),
        "params_hash": params_hash if params_hash is not None else params_fingerprint(),
        "drift_level": verdict.level if verdict is not None else None,
        "drift_percentile": verdict.percentile if verdict is not None else float("nan"),
        "drift_window_mae": verdict.window_mae if verdict is not None else float("nan"),
        "drift_reference_median": (
            verdict.reference_median if verdict is not None else float("nan")
        ),
        "drift_reference_origins": (
            float(verdict.reference_origins) if verdict is not None else float("nan")
        ),
    }


# ------------------------------------------------------------------------ reference


def _day_of_year(timestamps: pd.Series) -> pd.Series:
    """Day-of-year in Europe/Berlin -- the calendar the demand series actually lives in."""
    return pd.to_datetime(timestamps, utc=True).dt.tz_convert("Europe/Berlin").dt.dayofyear


def _circular_distance(days: pd.Series, centre: int) -> pd.Series:
    """Distance around the year, so a late-December window can pull January origins."""
    straight = (days - centre).abs()
    return np.minimum(straight, DAYS_IN_YEAR - straight)


def _widths(window_days: int, max_window_days: int) -> list[int]:
    steps = [window_days, 30, max_window_days]
    return sorted({width for width in steps if window_days <= width <= max_window_days})


def seasonal_reference(
    scores: pd.DataFrame, center, *, model: str = CHAMPION,
    window_days: int = SEASON_WINDOW_DAYS, max_window_days: int = MAX_SEASON_WINDOW_DAYS,
    exclude_from=None, params_hash: str | None = None, min_reference: int = MIN_REFERENCE_ORIGINS,
) -> pd.DataFrame:
    """Historical per-origin rows for comparable days of the year.

    Starts at `window_days` and widens (21 -> 30 -> 45 by default) only until the
    reference reaches `min_reference` rows; if the widest still falls short, returns what
    it has and lets `assess` rule "unknown". The width used is recorded on the returned
    frame's `.attrs["window_days"]`.
    """
    if scores is None or len(scores) == 0:
        empty = _empty_origin_history()
        empty.attrs["window_days"] = int(window_days)
        return empty

    pool = scores[scores.model == model].copy()
    if params_hash is not None:
        pool = pool[pool.params_hash == params_hash]
    pool["origin"] = pd.to_datetime(pool.origin, utc=True)
    if exclude_from is not None:
        # Never score a window against itself.
        pool = pool[pool.origin < _utc(exclude_from)]

    centre_day = int(_day_of_year(pd.Series([_utc(center)])).iloc[0])
    distance = (
        _circular_distance(_day_of_year(pool.origin), centre_day)
        if len(pool) else pd.Series(dtype=float)
    )

    reference = pool.iloc[:0]
    width = window_days
    for width in _widths(window_days, max_window_days):
        reference = pool[distance <= width] if len(pool) else pool
        if len(reference) >= min_reference:
            break
    reference = reference.sort_values("origin").reset_index(drop=True)
    reference.attrs["window_days"] = int(width)
    return reference


def recent_window(
    scores: pd.DataFrame, *, model: str = CHAMPION, window: int = RECENT_ORIGINS
) -> pd.DataFrame:
    """The `window` most recent origins for `model`, oldest first."""
    if scores is None or len(scores) == 0:
        return _empty_origin_history()
    pool = scores[scores.model == model].copy()
    pool["origin"] = pd.to_datetime(pool.origin, utc=True)
    return pool.sort_values("origin").tail(window).reset_index(drop=True)


# --------------------------------------------------------------------------- verdict


def _ordinal(value: int) -> str:
    if 11 <= value % 100 <= 13:
        return f"{value}th"
    return f"{value}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th') }"


def assess(
    window: pd.DataFrame, reference: pd.DataFrame, *, watch: float = WATCH_PERCENTILE,
    regressed: float = REGRESSED_PERCENTILE, min_reference: int = MIN_REFERENCE_ORIGINS,
    consecutive_runs: int = 0, allow_regressed: bool = True,
    reference_window_days: int | None = None,
) -> Verdict:
    """Score one window of origins against a season-matched reference.

    The single scorer: both the weekly artifact path and the live 7-day replay call it,
    so they cannot judge the same number differently.
    """
    if reference_window_days is None:
        reference_window_days = int(getattr(reference, "attrs", {}).get("window_days", 0) or 0)

    window_mae = float(window.mae.mean()) if len(window) else float("nan")
    n_window, n_reference = len(window), len(reference)
    start = pd.Timestamp(window.origin.min()) if n_window else None
    end = pd.Timestamp(window.origin.max()) if n_window else None
    reference_median = float(reference.mae.median()) if n_reference else float("nan")

    def verdict(level: str, headline: str, reason: str, percentile: float) -> Verdict:
        return Verdict(
            level=level, headline=headline, reason=reason, window_mae=window_mae,
            window_origins=n_window, window_start=start, window_end=end,
            reference_median=reference_median, reference_origins=n_reference,
            reference_window_days=reference_window_days, percentile=percentile,
            consecutive_runs=consecutive_runs,
        )

    if n_reference < min_reference or n_window < 2:
        return verdict(
            "unknown",
            f"Not enough past data yet to judge. Only {n_reference} similar days from previous "
            f"years are available to compare against, and at least {min_reference} are needed.",
            f"{n_reference} similar days (need {min_reference}), and {n_window} day"
            f"{'' if n_window == 1 else 's'} of recent error to score (need at least 2). "
            "A verdict from this little data would be noise.",
            float("nan"),
        )

    percentile = float((reference.mae.to_numpy(dtype=float) < window_mae).mean())
    shared = (
        f"Over the last {n_window} days the model's average error was higher than "
        f"{round(percentile * 100)}% of {n_reference} similar days from previous years "
        f"(days falling within {reference_window_days} days of the same date)"
    )

    if percentile <= watch:
        return verdict(
            "ok", f"{shared}. Normal for this time of year.",
            f"Anything up to {watch:.0%} counts as normal for the season; this is "
            f"{percentile:.0%}.",
            percentile,
        )
    if percentile <= regressed:
        return verdict(
            "watch", f"{shared}. Higher than usual for this time of year, worth watching.",
            f"{percentile:.0%} is above the {watch:.0%} mark that counts as normal, but below "
            f"the {regressed:.0%} mark that would flag a real problem.",
            percentile,
        )
    if not allow_regressed:
        return verdict(
            "watch", f"{shared}. Higher than usual for this time of year, worth watching.",
            f"{percentile:.0%} is above {regressed:.0%}, but {n_window} days is too small a "
            "sample to condemn the model on. About 1 in 20 trouble-free weeks looks this bad "
            "by chance. Only the full weekly benchmark can call a regression.",
            percentile,
        )
    if consecutive_runs < 1:
        return verdict(
            "watch", f"{shared}. Higher than usual for this time of year, worth watching.",
            f"{percentile:.0%} is above {regressed:.0%}, but this is the first run that high, "
            "and about 1 in 45 runs does that by chance even with nothing wrong. A second run "
            "in a row would be called a regression.",
            percentile,
        )
    return verdict(
        "regressed",
        f"{shared}. That is the {_ordinal(consecutive_runs + 1)} run in a row this high, so the "
        "model looks worse than it used to be.",
        f"{percentile:.0%} is above {regressed:.0%} for {consecutive_runs + 1} runs in a row. "
        "Chance alone would do that about once in 2,000 runs.",
        percentile,
    )


def _consecutive_crossings(runs: pd.DataFrame, threshold: float = REGRESSED_PERCENTILE) -> int:
    """How many *immediately preceding* runs already sat above the regression threshold."""
    if runs is None or len(runs) == 0 or "drift_percentile" not in runs:
        return 0
    ordered = runs.sort_values(["run_at", "snapshot_end"]) if "run_at" in runs else runs
    values = pd.to_numeric(ordered.drift_percentile, errors="coerce").to_numpy(dtype=float)
    count = 0
    for value in values[::-1]:
        if np.isnan(value) or value <= threshold:
            break
        count += 1
    return count


def regression_verdict(
    scores: pd.DataFrame, runs: pd.DataFrame | None = None, *, model: str = CHAMPION,
    window: int = RECENT_ORIGINS, window_days: int = SEASON_WINDOW_DAYS,
    max_window_days: int = MAX_SEASON_WINDOW_DAYS, params_hash: str | None = None, **kwargs,
) -> Verdict:
    """The weekly verdict: the last `window` benchmark origins against comparable days."""
    recent = recent_window(scores, model=model, window=window)
    if len(recent) == 0:
        return assess(recent, _empty_origin_history(), **kwargs)
    centre = recent.origin.iloc[len(recent) // 2]
    reference = seasonal_reference(
        scores, centre, model=model, window_days=window_days,
        max_window_days=max_window_days, exclude_from=recent.origin.min(),
        params_hash=params_hash,
        min_reference=kwargs.get("min_reference", MIN_REFERENCE_ORIGINS),
    )
    kwargs.setdefault("consecutive_runs", _consecutive_crossings(runs))
    return assess(recent, reference, **kwargs)


def live_verdict(
    daily: pd.DataFrame, scores: pd.DataFrame, *, model: str = CHAMPION,
    window_days: int = SEASON_WINDOW_DAYS, max_window_days: int = MAX_SEASON_WINDOW_DAYS,
    params_hash: str | None = None, **kwargs,
) -> Verdict:
    """The dashboard verdict: `verification.daily_scores` against comparable days.

    Capped at "watch" -- 5.0% of no-drift 7-origin windows exceed the 0.90 threshold by
    chance, against 2.2% at 14, so a week of replay cannot condemn the model on its own.
    """
    kwargs["allow_regressed"] = False
    if daily is None or len(daily) == 0:
        return assess(_empty_origin_history(), _empty_origin_history(), **kwargs)
    recent = daily[pd.to_numeric(daily.hours_scored, errors="coerce").fillna(0) > 0].copy()
    if len(recent) == 0:
        return assess(recent, _empty_origin_history(), **kwargs)
    recent["origin"] = pd.to_datetime(recent.origin, utc=True)
    recent = recent.sort_values("origin").reset_index(drop=True)
    centre = recent.origin.iloc[len(recent) // 2]
    reference = seasonal_reference(
        scores, centre, model=model, window_days=window_days,
        max_window_days=max_window_days, exclude_from=recent.origin.min(),
        params_hash=params_hash,
        min_reference=kwargs.get("min_reference", MIN_REFERENCE_ORIGINS),
    )
    return assess(recent, reference, **kwargs)


# -------------------------------------------------------------------------- coverage


def coverage_drift(
    runs: pd.DataFrame, *, baseline_runs: int = 3, watch_drop: float = COVERAGE_WATCH_DROP
) -> dict:
    """Has interval coverage fallen below what this model has been *establishing*?

    The baseline is the established coverage, **not** the 95% nominal. The shipped band
    already under-covers significantly (93.2%, CI [91.5%, 94.7%], p=0.025), so comparing
    against nominal would report a problem forever and nobody would read the panel. The
    nominal is returned for the panel's reference line and never thresholded on.

    The CI compared here is the per-origin bootstrap from `inference.coverage_test`,
    carried through `benchmark_summary.json` into the run history -- so the sampling-unit
    rule (per-origin, never per-hour) is preserved without recomputing it here, and there
    is deliberately no per-hour binomial anywhere in this function.
    """
    nominal = float("nan")
    if runs is not None and len(runs) and "coverage_nominal" in runs:
        nominal = _number(runs.coverage_nominal.iloc[-1])

    def unknown(reason: str) -> dict:
        return {
            "level": "unknown", "established": None, "mean": float("nan"),
            "ci_lower": float("nan"), "ci_upper": float("nan"), "drop": float("nan"),
            "nominal": nominal, "baseline_runs": 0,
            "headline": "Not enough past runs yet to tell whether the forecast range is still "
            "as reliable as it was.",
            "reason": reason,
        }

    if runs is None or len(runs) < 2:
        count = 0 if runs is None else len(runs)
        return unknown(
            f"{count} run{'' if count == 1 else 's'} recorded so far; at least 2 are needed."
        )

    ordered = runs.sort_values(["run_at", "snapshot_end"]).reset_index(drop=True)
    latest = ordered.iloc[-1]
    prior = ordered.iloc[:-1].tail(baseline_runs)
    baseline = pd.to_numeric(prior.coverage_mean_per_origin, errors="coerce").dropna()
    if baseline.empty:
        return unknown("No earlier run recorded a per-origin coverage mean.")

    established = float(baseline.median())
    mean = _number(latest.coverage_mean_per_origin)
    ci_lower = _number(latest.coverage_ci_lower)
    ci_upper = _number(latest.coverage_ci_upper)
    if np.isnan(mean):
        return unknown("The latest run recorded no per-origin coverage mean.")

    drop = established - mean
    if not np.isnan(ci_upper) and ci_upper < established:
        level = "regressed"
        headline = (
            f"The forecast range caught only {mean:.1%} of actual values, clearly below the "
            f"{established:.1%} this model had been holding."
        )
        reason = (
            f"Even the optimistic end of the margin of error ({ci_upper:.1%}) sits below the "
            f"established {established:.1%}, so this is more than sampling noise."
        )
    elif drop > watch_drop:
        level = "watch"
        headline = (
            f"The forecast range caught {mean:.1%} of actual values, down from the "
            f"{established:.1%} this model had been holding. Worth watching."
        )
        reason = (
            f"That is {drop:.1%} below the established {established:.1%}, but the margin of "
            "error still overlaps it."
        )
    else:
        level = "ok"
        headline = (
            f"The forecast range caught {mean:.1%} of actual values, in line with the "
            f"{established:.1%} this model has been holding."
        )
        reason = (
            f"Within {watch_drop:.1%} of the established {established:.1%}. The {nominal:.0%} "
            "label is a design target, not the baseline drift is judged against."
        )

    return {
        "level": level, "established": established, "mean": mean, "ci_lower": ci_lower,
        "ci_upper": ci_upper, "drop": drop, "nominal": nominal,
        "baseline_runs": int(len(baseline)), "headline": headline, "reason": reason,
    }
