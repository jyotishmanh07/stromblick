import numpy as np
import pandas as pd

from energy_forecast.events import (
    CalendarLogisticBaseline,
    HighDemandClassifier,
    MajorityClassBaseline,
    anomaly_day_frame,
    classification_metrics,
    daily_demand,
    daily_feature_frame,
    label_anomaly_days,
    label_high_demand_days,
    rolling_origin_classification_backtest,
    threshold_cost_table,
)
from energy_forecast.service import demo_demand


def test_high_demand_threshold_uses_only_earlier_days():
    daily = daily_demand(demo_demand())
    labelled = label_high_demand_days(daily, quantile=0.7, window_days=30)
    peaks = daily.set_index("date").peak_mw
    for row in labelled.itertuples():
        position = list(peaks.index).index(row.date)
        window = peaks.iloc[max(0, position - 30) : position]
        assert np.isclose(row.threshold_mw, np.quantile(window, 0.7))
        # The window must end strictly before the day being judged.
        assert row.date not in list(window.index)


def test_a_spike_on_the_final_day_cannot_change_its_own_label():
    daily = daily_demand(demo_demand())
    baseline = label_high_demand_days(daily).threshold_mw.to_numpy()
    inflated = daily.copy()
    inflated.loc[inflated.index[-1], "peak_mw"] *= 3
    assert np.allclose(label_high_demand_days(inflated).threshold_mw.to_numpy(), baseline)


def test_daily_features_are_lagged_by_at_least_one_day():
    frame = daily_feature_frame(demo_demand())
    daily = daily_demand(demo_demand()).set_index("date")
    for row in frame.itertuples():
        position = list(daily.index).index(row.date)
        assert np.isclose(row.prev_peak_mw, daily.peak_mw.iloc[position - 1])
        assert np.isclose(row.peak_7d_mean, daily.peak_mw.iloc[position - 7 : position].mean())


def test_classifier_beats_the_majority_baseline_on_a_learnable_pattern():
    # Fridays get a large demand bump, so the label becomes calendar-predictable.
    frame = demo_demand(hours=24 * 400).copy()
    friday = frame.timestamp.dt.tz_convert("Europe/Berlin").dt.weekday == 4
    frame.loc[friday, "demand_mw"] += 25_000
    daily = daily_feature_frame(frame)
    result = rolling_origin_classification_backtest(
        daily,
        {"Majority class": MajorityClassBaseline, "HistGradientBoosting": HighDemandClassifier},
        initial_train_days=120,
    )
    scores = {
        name: classification_metrics(group.label.to_numpy(), group.probability.to_numpy())
        for name, group in result.groupby("model")
    }
    assert scores["HistGradientBoosting"]["pr_auc"] > scores["Majority class"]["pr_auc"]
    assert scores["HistGradientBoosting"]["pr_auc_lift"] > 1.5


def test_classification_backtest_is_chronological_and_refits_per_origin():
    daily = daily_feature_frame(demo_demand(hours=24 * 300))
    result = rolling_origin_classification_backtest(
        daily, {"Calendar logistic": CalendarLogisticBaseline}, initial_train_days=150
    )
    dates = pd.to_datetime(result.date)
    assert dates.is_monotonic_increasing
    assert dates.min() > pd.to_datetime(daily.date).iloc[149]


def test_anomaly_labels_come_from_a_past_only_threshold():
    frame = demo_demand(hours=24 * 200)
    labelled = label_anomaly_days(frame, quantile=0.99, window_days=30)
    assert set(labelled.label.unique()) <= {0, 1}
    assert labelled.threshold_dev_mw.notna().all()
    # An injected spike on the last day must not move any earlier threshold.
    spiked = frame.copy()
    spiked.loc[spiked.index[-1], "demand_mw"] += 40_000
    after = label_anomaly_days(spiked, quantile=0.99, window_days=30)
    assert np.allclose(
        labelled.threshold_dev_mw.to_numpy()[:-1], after.threshold_dev_mw.to_numpy()[:-1]
    )


def test_anomaly_frame_shares_the_daily_feature_set():
    frame = anomaly_day_frame(demo_demand(hours=24 * 200))
    assert {"label", "max_deviation_mw", "prev_peak_mw", "peak_7d_mean"} <= set(frame.columns)


def test_threshold_cost_table_trades_recall_for_precision():
    rng = np.random.default_rng(0)
    labels = rng.binomial(1, 0.3, size=400)
    probabilities = np.clip(labels * 0.4 + rng.normal(0.3, 0.2, size=400), 0, 1)
    table = threshold_cost_table(labels, probabilities)
    assert table.recall.is_monotonic_decreasing
    assert table.flagged.is_monotonic_decreasing
