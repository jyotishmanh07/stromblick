import numpy as np
import pandas as pd
import pytest

from energy_forecast.data import DataValidationError
from energy_forecast.quality import validate_demand
from energy_forecast.service import demo_demand


def _named(report, name):
    return next(check for check in report.checks if check.name == name)


def test_clean_demo_series_passes_the_gate():
    report = validate_demand(demo_demand())
    assert report.passed
    assert not report.errors


def test_decimal_shifted_values_fail_the_plausible_range_check():
    frame = demo_demand()
    frame.loc[10, "demand_mw"] *= 100
    report = validate_demand(frame)
    assert not report.passed
    assert not _named(report, "plausible_range").passed
    with pytest.raises(DataValidationError, match="plausible_range"):
        report.raise_for_errors()


def test_duplicate_timestamps_are_an_error():
    frame = demo_demand()
    frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True).sort_values(
        "timestamp"
    ).reset_index(drop=True)
    report = validate_demand(frame)
    assert not _named(report, "unique_timestamps").passed
    assert _named(report, "unique_timestamps").value == 1


def test_gaps_warn_but_do_not_block():
    frame = demo_demand().drop(index=range(100, 200)).reset_index(drop=True)
    report = validate_demand(frame, max_gap_fraction=0.001)
    assert not _named(report, "hourly_continuity").passed
    # Gaps are a normal property of a SMARD snapshot: reported, never fatal.
    assert report.passed
    report.raise_for_errors()


def test_unpublished_hours_warn_once_they_exceed_the_limit():
    frame = demo_demand()
    frame.loc[: len(frame) // 2, "demand_mw"] = np.nan
    report = validate_demand(frame, max_nan_fraction=0.05)
    assert not _named(report, "nan_fraction").passed
    assert report.passed


def test_stale_snapshot_fails_only_when_freshness_is_requested():
    frame = demo_demand()
    assert validate_demand(frame).passed
    report = validate_demand(
        frame, max_staleness_hours=6, now=frame.timestamp.max() + pd.Timedelta(days=3)
    )
    assert not _named(report, "freshness").passed
    assert not report.passed


def test_unsorted_timestamps_are_rejected():
    frame = demo_demand().sample(frac=1, random_state=0).reset_index(drop=True)
    report = validate_demand(frame)
    assert not _named(report, "chronological").passed


def test_report_serialises_for_metadata():
    payload = validate_demand(demo_demand()).to_dict()
    assert payload["passed"] is True
    assert {"name", "passed", "severity", "detail"} <= set(payload["checks"][0])
