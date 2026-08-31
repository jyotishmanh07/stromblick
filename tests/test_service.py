from energy_forecast.service import ForecastService, demo_demand


def test_service_learns_validation_residuals_for_intervals(tmp_path):
    service = ForecastService(tmp_path / "missing.csv")
    service.history = demo_demand(24 * 30)
    service._fit_validation_residuals()
    assert len(service.residuals) == 24 * 7
