import json

from energy_forecast.data import SMARDClient, smard_api_chunk_url


def test_smard_series_payload_is_parsed():
    payload = {
        "meta_data": {"version": 1},
        "series": [[1704067200000, 42000.5], [1704070800000, 42100.0]],
    }
    frame = SMARDClient().parse(json.dumps(payload).encode(), "application/json")
    assert frame.demand_mw.tolist() == [42000.5, 42100.0]
    assert str(frame.timestamp.dt.tz) == "UTC"


def test_smard_chunk_url_uses_index_timestamp():
    assert smard_api_chunk_url(410, "DE", "hour", 1787522400000) == (
        "https://www.smard.de/app/chart_data/410/DE/410_DE_hour_1787522400000.json"
    )
