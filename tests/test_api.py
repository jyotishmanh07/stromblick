from fastapi.testclient import TestClient

from energy_forecast.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_contract():
    response = client.post("/forecast", json={"as_of": "2026-08-28T00:00:00Z", "horizon_hours": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v1"
    assert len(body["forecast"]) == 2
    assert {"timestamp", "prediction", "lower_bound", "upper_bound"}.issubset(body["forecast"][0])


def test_forecast_rejects_invalid_horizon_and_naive_timestamp():
    invalid_horizon = client.post(
        "/forecast", json={"as_of": "2026-08-28T00:00:00Z", "horizon_hours": 25}
    )
    naive_timestamp = client.post(
        "/forecast", json={"as_of": "2026-08-28T00:00:00", "horizon_hours": 2}
    )
    assert invalid_horizon.status_code == 422
    assert naive_timestamp.status_code == 422
