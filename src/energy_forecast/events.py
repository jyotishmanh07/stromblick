"""Daily event classification: will tomorrow be a high-demand day?

A second problem type on the same data, kept out of models.py because that module
is deliberately three forecasting levels. The forecasting track answers "how much
demand, each hour"; this one answers "is tomorrow worth planning around" -- the
shape a reserve-planning or staffing decision actually takes.

The leakage rule extends to labels here. A "high-demand day" is one whose peak
clears a top-decile threshold, and that threshold is computed from *past days only*
and shifted, so the label for day d never depends on day d or anything after it.
Features are restricted to what is known by the evening of day d-1 plus the target
day's calendar, which is known arbitrarily far ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import german_holidays

DAILY_FEATURE_COLUMNS = [
    "weekday", "month", "is_weekend", "is_public_holiday",
    "prev_peak_mw", "prev_mean_mw", "prev_range_mw",
    "peak_7d_mean", "peak_7d_max", "mean_7d_mean", "peak_trend",
]

try:  # keep import-time failure from taking down the module, as models.py does
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    _SKLEARN = True
except ImportError:  # pragma: no cover - exercised only on a broken install
    _SKLEARN = False


def daily_demand(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly demand to one row per local calendar day."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    local = data.timestamp.dt.tz_convert("Europe/Berlin")
    data["date"] = local.dt.date
    daily = (
        data.dropna(subset=["demand_mw"])
        .groupby("date")
        .agg(
            peak_mw=("demand_mw", "max"), mean_mw=("demand_mw", "mean"),
            trough_mw=("demand_mw", "min"), hours=("demand_mw", "size"),
        )
        .reset_index()
    )
    # Partial days (DST, truncated snapshot ends) would bias the peak downward.
    return daily[daily.hours >= 20].drop(columns="hours").reset_index(drop=True)


def label_high_demand_days(
    daily: pd.DataFrame, quantile: float = 0.7, window_days: int = 30
) -> pd.DataFrame:
    """Flag days whose peak lands in the top 30% of the trailing month.

    The threshold is a trailing-window quantile over the previous `window_days`,
    shifted by one day, so the rule that judges day d is fixed before day d is
    observed. Read the label as "tomorrow is one of the heavier days of the recent
    month" -- the shape a reserve or staffing decision takes.

    The defaults are a deliberate choice, not a tuned result. Two earlier framings
    failed on a single year of strongly seasonal data, and both failures are
    instructive:

    * An *expanding* all-time top decile is set by January's peaks, so no summer day
      can ever clear it. The label collapses into "is it winter" and an evaluation
      window starting after winter contains no positives at all.
    * A *trailing* 60-day top decile still degenerates during sustained seasonal
      decline: March, April and May produced zero positives, because a falling
      series never exceeds its own recent upper tail.

    A shorter window and a milder quantile keep every month populated (the observed
    monthly rate ranges from 3% to 71%, with no empty month). The rate is still not
    stationary -- it rises when demand trends up and falls when it trends down --
    which is a genuine property of the target, reported rather than smoothed away.
    """
    result = daily.sort_values("date").reset_index(drop=True).copy()
    result["threshold_mw"] = (
        result.peak_mw.shift(1).rolling(window_days, min_periods=window_days).quantile(quantile)
    )
    result["label"] = (result.peak_mw >= result.threshold_mw).astype(int)
    return result.dropna(subset=["threshold_mw"]).reset_index(drop=True)


def label_anomaly_days(
    frame: pd.DataFrame, quantile: float = 0.99, window_days: int = 30
) -> pd.DataFrame:
    """Flag days containing an hour that departs sharply from the seasonal-naive expectation.

    The deviation is `demand(t) - demand(t-24h)` -- the seasonal-naive error, which
    depends on no fitted model. That independence is the point: labelling anomalies
    with the champion's own residuals would ask the classifier to predict where a
    specific model happens to fail, and any evaluation of it would be circular.

    The threshold is the `quantile` of absolute deviations over the previous
    `window_days`, so like the high-demand label it is fixed before the day it judges.
    """
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    series = data.set_index("timestamp").demand_mw
    deviation = (series - series.shift(24, freq="h").reindex(series.index)).abs()
    daily = pd.DataFrame(
        {
            "date": deviation.index.tz_convert("Europe/Berlin").date,
            "deviation_mw": deviation.to_numpy(),
        }
    ).dropna()

    by_date = daily.groupby("date").deviation_mw
    peaks = by_date.max().rename("max_deviation_mw").reset_index()
    values_by_date = {date: group.to_numpy() for date, group in by_date}
    dates = list(peaks.date)
    thresholds: list[float] = []
    for i, date in enumerate(dates):
        window = dates[max(0, i - window_days) : i]
        prior = np.concatenate([values_by_date[d] for d in window]) if window else np.array([])
        thresholds.append(float(np.quantile(prior, quantile)) if len(prior) else np.nan)
    peaks["threshold_dev_mw"] = thresholds
    peaks = peaks.iloc[window_days:].reset_index(drop=True)
    peaks["label"] = (peaks.max_deviation_mw >= peaks.threshold_dev_mw).astype(int)
    return peaks.dropna(subset=["threshold_dev_mw"]).reset_index(drop=True)


def add_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Attach features known by the evening before each target day.

    Everything derived from demand is lagged at least one day; only the target day's
    calendar is used unlagged, because a calendar is known arbitrarily far ahead.
    """
    result = daily.copy()
    dates = pd.to_datetime(result.date)
    holidays = german_holidays(set(dates.dt.year.tolist()))

    # Target-day calendar is known in advance; everything else is lagged by a day.
    result["weekday"] = dates.dt.weekday.astype(int)
    result["month"] = dates.dt.month.astype(int)
    result["is_weekend"] = (dates.dt.weekday >= 5).astype(int)
    result["is_public_holiday"] = dates.dt.date.isin(holidays).astype(int)
    result["prev_peak_mw"] = result.peak_mw.shift(1)
    result["prev_mean_mw"] = result.mean_mw.shift(1)
    result["prev_range_mw"] = (result.peak_mw - result.trough_mw).shift(1)
    prior_peak = result.peak_mw.shift(1)
    prior_mean = result.mean_mw.shift(1)
    result["peak_7d_mean"] = prior_peak.rolling(7, min_periods=7).mean()
    result["peak_7d_max"] = prior_peak.rolling(7, min_periods=7).max()
    result["mean_7d_mean"] = prior_mean.rolling(7, min_periods=7).mean()
    result["peak_trend"] = prior_peak - prior_peak.rolling(7, min_periods=7).mean()
    return result.dropna(subset=DAILY_FEATURE_COLUMNS).reset_index(drop=True)


def daily_feature_frame(
    frame: pd.DataFrame, quantile: float = 0.7, window_days: int = 30
) -> pd.DataFrame:
    """High-demand-day training frame: features known by the previous evening, plus the label."""
    daily = label_high_demand_days(
        daily_demand(frame), quantile=quantile, window_days=window_days
    )
    return add_daily_features(daily)


def anomaly_day_frame(
    frame: pd.DataFrame, quantile: float = 0.99, window_days: int = 30
) -> pd.DataFrame:
    """Anomaly-day training frame, sharing the high-demand feature set."""
    labels = label_anomaly_days(frame, quantile=quantile, window_days=window_days)
    daily = daily_demand(frame).merge(
        labels[["date", "max_deviation_mw", "threshold_dev_mw", "label"]], on="date"
    )
    return add_daily_features(daily)


class MajorityClassBaseline:
    """Predicts the training-set positive rate for every day. The floor to beat."""

    name = "Majority class"

    def fit(self, frame: pd.DataFrame) -> MajorityClassBaseline:
        self.rate = float(frame.label.mean())
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.rate)


class CalendarLogisticBaseline:
    """Logistic regression on calendar columns only -- how far does the almanac get you?"""

    name = "Calendar logistic"
    columns = ["weekday", "month", "is_weekend", "is_public_holiday"]

    def fit(self, frame: pd.DataFrame) -> CalendarLogisticBaseline:
        self.rate = float(frame.label.mean())
        self.model = None
        if _SKLEARN and frame.label.nunique() > 1:
            self.scaler = StandardScaler().fit(frame[self.columns])
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
                self.scaler.transform(frame[self.columns]), frame.label
            )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), self.rate)
        return self.model.predict_proba(self.scaler.transform(frame[self.columns]))[:, 1]


class HighDemandClassifier:
    """Gradient-boosted classifier over the full daily feature set."""

    name = "HistGradientBoosting"

    def __init__(self, random_state: int = 42, **params) -> None:
        self.random_state = random_state
        self.params = {"max_iter": 200, "learning_rate": 0.06, "max_leaf_nodes": 15, **params}

    def fit(self, frame: pd.DataFrame) -> HighDemandClassifier:
        self.rate = float(frame.label.mean())
        self.model = None
        if _SKLEARN and frame.label.nunique() > 1:
            self.model = HistGradientBoostingClassifier(
                random_state=self.random_state, **self.params
            ).fit(frame[DAILY_FEATURE_COLUMNS], frame.label)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), self.rate)
        return self.model.predict_proba(frame[DAILY_FEATURE_COLUMNS])[:, 1]


def rolling_origin_classification_backtest(
    frame: pd.DataFrame, model_factories: dict[str, type], initial_train_days: int = 120,
    step_days: int = 1,
) -> pd.DataFrame:
    """Walk forward one day at a time, refitting every model at every origin.

    Mirrors rolling_origin_backtest's contract: chronological, refit per origin,
    never a random split. Returns one row per (date, model) with the predicted
    probability and the realised label.
    """
    data = frame.sort_values("date").reset_index(drop=True)
    records: list[dict[str, object]] = []
    for origin in range(initial_train_days, len(data), step_days):
        train = data.iloc[:origin]
        test = data.iloc[origin : origin + step_days]
        if train.label.nunique() < 2:
            continue
        for name, factory in model_factories.items():
            model = factory().fit(train)
            probabilities = model.predict_proba(test)
            for (_, row), probability in zip(test.iterrows(), probabilities):
                records.append(
                    {
                        "date": row.date, "model": name, "probability": float(probability),
                        "label": int(row.label), "peak_mw": float(row.peak_mw),
                    }
                )
    return pd.DataFrame(records)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Ranking, calibration, and a decision-threshold summary."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    base_rate = float(labels.mean())
    single_class = labels.min() == labels.max()
    return {
        "pr_auc": float("nan") if single_class else float(
            average_precision_score(labels, probabilities)
        ),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "base_rate": base_rate,
        # PR-AUC is only impressive relative to the positive rate a coin flip would get.
        "pr_auc_lift": float("nan") if single_class or base_rate == 0 else float(
            average_precision_score(labels, probabilities) / base_rate
        ),
        "n": int(len(labels)),
        "positives": int(labels.sum()),
    }


def threshold_cost_table(
    labels: np.ndarray, probabilities: np.ndarray, miss_cost: float = 10.0,
    false_alarm_cost: float = 1.0, thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Precision/recall and expected cost across decision thresholds.

    The default 10:1 ratio says a missed high-demand day hurts ten times as much as
    an unnecessary alert -- plausible when the response is "have an analyst look",
    but it is an assumption, and the operating point follows from it.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    grid = np.arange(0.05, 0.96, 0.05) if thresholds is None else np.asarray(thresholds)
    rows = []
    for threshold in grid:
        flagged = probabilities >= threshold
        true_positive = int(np.sum(flagged & (labels == 1)))
        false_positive = int(np.sum(flagged & (labels == 0)))
        false_negative = int(np.sum(~flagged & (labels == 1)))
        rows.append(
            {
                "threshold": float(threshold),
                "flagged": int(flagged.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": true_positive / (true_positive + false_positive)
                if true_positive + false_positive else float("nan"),
                "recall": true_positive / (true_positive + false_negative)
                if true_positive + false_negative else float("nan"),
                "expected_cost": false_negative * miss_cost + false_positive * false_alarm_cost,
            }
        )
    return pd.DataFrame(rows)
