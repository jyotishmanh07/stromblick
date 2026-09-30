import json

import httpx
import pandas as pd
import pytest

from energy_forecast.data import DataValidationError
from energy_forecast.smard_api import SMARDAPIClient

INDEX_URL = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
START_MS = 1704067200000  # 2024-01-01T00:00Z
CHUNK_URL = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{START_MS}.json"


class Response:
    headers = {"content-type": "application/json"}

    def __init__(self, payload):
        self.content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def raise_for_status(self):
        return None


def _serve(monkeypatch, index=None, chunk=None):
    def fake_get(url, **kwargs):
        assert url in {INDEX_URL, CHUNK_URL}
        return Response(index if url == INDEX_URL else chunk)

    monkeypatch.setattr("energy_forecast.smard_api.httpx.get", fake_get)


def _week(demand_mw: float = 50_000.0) -> dict:
    return {"series": [[START_MS + hour * 3_600_000, demand_mw] for hour in range(168)]}


def test_fetch_latest_uses_index_and_weekly_chunks(monkeypatch):
    _serve(monkeypatch, {"timestamps": [START_MS]}, {"series": [[START_MS, 42000.0]]})
    data, urls = SMARDAPIClient().fetch_latest(weeks=1)
    assert urls == [CHUNK_URL]
    assert data.demand_mw.tolist() == [42000.0]


def test_download_writes_raw_payloads_clean_csv_and_metadata(monkeypatch, tmp_path):
    _serve(monkeypatch, {"timestamps": [START_MS]}, _week())
    data = SMARDAPIClient().download(1, tmp_path / "raw", tmp_path / "clean")
    metadata = json.loads((tmp_path / "clean" / "metadata.json").read_text())
    assert len(list((tmp_path / "raw").glob("smard_*.json"))) == 2  # index + one chunk
    assert len(pd.read_csv(tmp_path / "clean" / "demand_hourly.csv")) == len(data) == 168
    assert metadata["quality"]["passed"] is True
    # refresh-data.yml's watchdog compares this key across commits to prove data landed.
    assert metadata["last_timestamp"] == data.timestamp.max().isoformat()


def test_download_rejected_by_the_gate_keeps_raw_evidence_and_writes_no_csv(
    monkeypatch, tmp_path
):
    _serve(monkeypatch, {"timestamps": [START_MS]}, _week(demand_mw=500.0))  # decimal shift
    with pytest.raises(DataValidationError, match="plausible_range"):
        SMARDAPIClient().download(1, tmp_path / "raw", tmp_path / "clean")
    assert len(list((tmp_path / "raw").glob("smard_*.json"))) == 2
    assert not (tmp_path / "clean" / "demand_hourly.csv").exists()


def test_index_without_timestamps_is_rejected(monkeypatch):
    _serve(monkeypatch, index={"timestamps": []})
    with pytest.raises(DataValidationError, match="no timestamps"):
        SMARDAPIClient().fetch_latest(weeks=1)


def test_chunk_without_series_is_rejected(monkeypatch):
    _serve(monkeypatch, {"timestamps": [START_MS]}, {"meta_data": {}})
    with pytest.raises(DataValidationError, match="no series"):
        SMARDAPIClient().fetch_latest(weeks=1)


def test_invalid_json_is_a_validation_error(monkeypatch):
    _serve(monkeypatch, index=b"<html>maintenance</html>")
    with pytest.raises(DataValidationError, match="invalid JSON"):
        SMARDAPIClient().fetch_latest(weeks=1)


def test_http_failure_is_a_connection_error(monkeypatch):
    def unreachable(url, **kwargs):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr("energy_forecast.smard_api.httpx.get", unreachable)
    with pytest.raises(ConnectionError):
        SMARDAPIClient().fetch_latest(weeks=1)
