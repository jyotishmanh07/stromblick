import numpy as np

from energy_forecast.inference import (
    coverage_test,
    diebold_mariano,
    paired_model_comparison,
    slice_difference_test,
)


def test_identical_error_series_are_not_significantly_different():
    errors = np.random.default_rng(0).uniform(1000, 2000, size=60)
    result = paired_model_comparison(errors, errors.copy())
    assert result["ci_lower"] <= 0 <= result["ci_upper"]
    assert result["wilcoxon_p"] == 1.0


def test_consistently_better_model_is_significant():
    rng = np.random.default_rng(1)
    worse = rng.uniform(1000, 2000, size=60)
    better = worse - rng.uniform(200, 400, size=60)
    result = paired_model_comparison(better, worse)
    assert result["ci_upper"] < 0
    assert result["wilcoxon_p"] < 0.01


def test_diebold_mariano_detects_a_persistent_hourly_gap():
    rng = np.random.default_rng(2)
    worse = rng.uniform(1000, 2000, size=720)
    better = worse - rng.normal(300, 80, size=720)
    result = diebold_mariano(better, worse, h=24)
    assert result["dm_stat"] < 0
    assert result["p_value"] < 0.01
    assert diebold_mariano(worse, worse.copy(), h=24)["p_value"] == 1.0


def test_slice_difference_ci_excludes_zero_only_for_a_real_gap():
    rng = np.random.default_rng(3)
    values = rng.normal(1000, 50, size=300)
    mask = np.zeros(300, dtype=bool)
    mask[:30] = True
    values[mask] += 500
    result = slice_difference_test(values, mask)
    assert result["ci_lower"] > 0
    assert result["permutation_p"] < 0.01
    null = slice_difference_test(rng.normal(1000, 50, size=300), mask)
    assert null["ci_lower"] <= 0 <= null["ci_upper"]


def test_coverage_test_flags_undercoverage_but_accepts_on_target_rates():
    rng = np.random.default_rng(4)
    low = coverage_test(rng.uniform(0.78, 0.86, size=50), nominal=0.95)
    assert low["ci_upper"] < 0.95
    assert low["p_value"] < 0.01
    on_target = coverage_test(rng.uniform(0.93, 0.97, size=50), nominal=0.95)
    assert on_target["ci_lower"] <= 0.95 <= on_target["ci_upper"]
