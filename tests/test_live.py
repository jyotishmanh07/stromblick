import json
import time

import numpy as np
import pandas as pd
import pytest

from energy_forecast.live import LIVE_LABEL, bridge_weeks, fetch_live_history, last_observed
from energy_forecast.service import demo_demand

INDEX_URL = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"


class Response:
    headers = {"content-type": "application/json"}

    def __init__(self, payload):
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        return None


def committed_frame(end="2026-08-31 07:00", hours=24 * 40):
    stamps = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=hours, freq="1h")
    return pd.DataFrame({"timestamp": stamps, "demand_mw": np.linspace(50_000, 55_000, hours)})


def fake_smard(monkeypatch, series_by_chunk, week_starts=(1_787_000_000_000,)):
    """Serve a SMARD index plus one payload per weekly chunk."""
    payloads = {INDEX_URL: {"timestamps": list(week_starts)}}
    for start, series in zip(week_starts, series_by_chunk):
        url = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{start}.json"
        payloads[url] = {"series": series}

    def fake_get(url, **kwargs):
        if url not in payloads:
            raise AssertionError(f"unexpected URL {url}")
        return Response(payloads[url])

    monkeypatch.setattr("energy_forecast.smard_api.httpx.get", fake_get)


def as_series(frame):
    """Frame -> SMARD [[epoch_ms, value], ...] rows, with NaN rendered as null."""
    return [
        [int(row.timestamp.timestamp() * 1000), None if pd.isna(row.demand_mw) else row.demand_mw]
        for row in frame.itertuples()
    ]


def test_bridge_weeks_scales_with_the_gap_and_gives_up_when_too_wide():
    now = pd.Timestamp("2026-09-01 12:00", tz="UTC")
    assert bridge_weeks(pd.Timestamp("2026-08-31 07:00", tz="UTC"), now) == 2
    # 22 days back needs ceil(22/7) + 1 = 5 chunks, one more than strictly spans the gap.
    assert bridge_weeks(pd.Timestamp("2026-08-10 07:00", tz="UTC"), now) == 5
    # Beyond the cap a live fetch would leave a hole; say so instead of serving one.
    assert bridge_weeks(pd.Timestamp("2026-05-01 07:00", tz="UTC"), now) is None


def test_live_values_win_on_overlapping_hours_and_history_is_kept(monkeypatch):
    committed = committed_frame()
    overlap = committed.tail(6).copy()
    overlap["demand_mw"] = 99_999.0
    fake_smard(monkeypatch, [as_series(overlap)])

    result = fetch_live_history(committed, now=committed.timestamp.max() + pd.Timedelta(hours=2))
    assert result.is_live
    assert result.source_label == LIVE_LABEL
    merged = result.history.set_index("timestamp").demand_mw
    assert (merged.tail(6) == 99_999.0).all()
    # Everything before the live window survives the merge.
    assert len(result.history) == len(committed)
    assert np.isclose(merged.iloc[0], committed.demand_mw.iloc[0])


def test_live_fetch_extends_the_series_past_the_committed_snapshot(monkeypatch):
    committed = committed_frame()
    new_hours = pd.date_range(
        start=committed.timestamp.max() + pd.Timedelta(hours=1), periods=12, freq="1h"
    )
    fresh = pd.DataFrame({"timestamp": new_hours, "demand_mw": 61_000.0})
    fake_smard(monkeypatch, [as_series(fresh)])

    result = fetch_live_history(committed, now=new_hours.max())
    assert result.is_live
    assert result.history.timestamp.max() == new_hours.max()
    assert result.provenance["weeks_fetched"] == 2
    assert result.provenance["last_observed_timestamp"].startswith("2026-08-31")


def test_unpublished_live_hour_cannot_overwrite_an_observed_value(monkeypatch):
    committed = committed_frame()
    original = float(committed.demand_mw.iloc[-1])
    blanked = committed.tail(3).copy()
    blanked["demand_mw"] = np.nan
    fake_smard(monkeypatch, [as_series(blanked)])

    result = fetch_live_history(committed, now=committed.timestamp.max() + pd.Timedelta(hours=1))
    assert np.isclose(result.history.demand_mw.iloc[-1], original)


def test_network_failure_falls_back_to_the_committed_snapshot(monkeypatch):
    committed = committed_frame()

    def boom(url, **kwargs):
        raise ConnectionError("SMARD unreachable")

    monkeypatch.setattr("energy_forecast.smard_api.httpx.get", boom)
    result = fetch_live_history(committed, now=committed.timestamp.max() + pd.Timedelta(hours=2))
    assert not result.is_live
    assert "unreachable" in result.warning
    assert result.history.timestamp.max() == committed.timestamp.max()


def test_all_null_chunk_falls_back_rather_than_raising(monkeypatch):
    # Early in a SMARD week every hour is indexed but unpublished, which makes
    # canonicalize_demand raise inside fetch_latest before any merge happens.
    committed = committed_frame()
    empty = committed.tail(5).copy()
    empty["demand_mw"] = np.nan
    fake_smard(monkeypatch, [as_series(empty)])

    result = fetch_live_history(committed, now=committed.timestamp.max() + pd.Timedelta(hours=1))
    assert not result.is_live
    assert "no numeric demand observations" in result.warning
    assert result.history.equals(committed)


def test_quality_gate_failure_falls_back(monkeypatch):
    committed = committed_frame()
    absurd = committed.tail(4).copy()
    absurd["demand_mw"] = 5_000_000.0  # far outside the plausible band
    fake_smard(monkeypatch, [as_series(absurd)])

    result = fetch_live_history(committed, now=committed.timestamp.max() + pd.Timedelta(hours=1))
    assert not result.is_live
    assert "quality gate" in result.warning
    assert result.history.demand_mw.max() < 100_000


def test_stale_snapshot_beyond_the_bridge_limit_is_reported_not_patched(monkeypatch):
    committed = committed_frame()

    def unexpected(url, **kwargs):
        raise AssertionError("should not fetch when the gap is too wide")

    monkeypatch.setattr("energy_forecast.smard_api.httpx.get", unexpected)
    result = fetch_live_history(
        committed, now=committed.timestamp.max() + pd.Timedelta(days=120)
    )
    assert not result.is_live
    assert "too far behind" in result.warning


def test_a_hanging_upstream_cannot_stall_the_page_past_the_deadline(monkeypatch):
    committed = committed_frame()

    def hang(url, **kwargs):
        time.sleep(30)  # longer than any deadline the app would use

    monkeypatch.setattr("energy_forecast.smard_api.httpx.get", hang)
    start = time.time()
    result = fetch_live_history(
        committed, now=committed.timestamp.max() + pd.Timedelta(hours=1), deadline_seconds=1.0
    )
    elapsed = time.time() - start
    assert elapsed < 5, f"deadline not enforced, took {elapsed:.1f}s"
    assert not result.is_live
    assert "exceeded" in result.warning
    assert result.history.equals(committed)


def test_last_observed_ignores_indexed_but_unpublished_hours():
    frame = demo_demand(hours=48).copy()
    frame.loc[frame.index[-3:], "demand_mw"] = np.nan
    assert last_observed(frame) == frame.timestamp.iloc[-4]
    assert last_observed(frame) < frame.timestamp.max()


def test_last_observed_is_none_when_nothing_is_published():
    frame = demo_demand(hours=5).copy()
    frame["demand_mw"] = np.nan
    assert last_observed(frame) is None


@pytest.mark.parametrize("weeks", [1, 3])
def test_bridge_never_returns_below_the_minimum(weeks):
    now = pd.Timestamp("2026-09-01 12:00", tz="UTC")
    assert bridge_weeks(now - pd.Timedelta(days=weeks), now) >= 2
