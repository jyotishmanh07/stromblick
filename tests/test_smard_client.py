import json

from energy_forecast.smard_api import SMARDAPIClient


def test_fetch_latest_uses_index_and_weekly_chunks(monkeypatch):
    index_url = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
    chunk_url = "https://www.smard.de/app/chart_data/410/DE/410_DE_hour_1704067200000.json"

    class Response:
        headers = {"content-type": "application/json"}

        def __init__(self, payload):
            self.content = json.dumps(payload).encode()

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        assert url in {index_url, chunk_url}
        if url == index_url:
            return Response({"timestamps": [1704067200000]})
        return Response({"series": [[1704067200000, 42000.0]]})

    monkeypatch.setattr("energy_forecast.smard_api.httpx.get", fake_get)
    data, urls = SMARDAPIClient().fetch_latest(weeks=1)
    assert urls == [chunk_url]
    assert data.demand_mw.tolist() == [42000.0]
