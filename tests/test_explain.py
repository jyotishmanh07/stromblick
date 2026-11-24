import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from explain import SHAP_COLUMNS, explain_origin, group_share  # noqa: E402

from energy_forecast.service import demo_demand  # noqa: E402


@pytest.fixture(scope="module")
def explained():
    data = demo_demand(24 * 40)
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    return data, explain_origin(data, data.timestamp.iloc[-30])


def test_shap_values_add_up_to_each_recursive_prediction(explained):
    _, frame = explained
    total = frame[SHAP_COLUMNS].sum(axis=1) + frame.base_value
    np.testing.assert_allclose(total, frame.predicted, atol=1e-6)
    assert list(frame.step) == list(range(1, 25))


def test_lag_1h_after_step_one_is_the_models_own_prediction(explained):
    data, frame = explained
    np.testing.assert_allclose(frame.lag_1h.iloc[1:], frame.predicted.iloc[:-1])
    origin = frame.origin.iloc[0]
    observed = data.set_index("timestamp").demand_mw[origin]
    assert frame.lag_1h.iloc[0] == pytest.approx(observed)


def test_group_shares_sum_to_one_per_step(explained):
    shares = group_share(explained[1])
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    assert len(shares) == 24
