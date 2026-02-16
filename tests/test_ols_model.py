from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.ols import OLSModel


def test_ols_model_recovers_coefficients() -> None:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        rng.normal(size=(400, 2)), columns=["brent_proxy_ret", "ulsd_proxy_ret"]
    )
    target = (
        0.001 + 0.8 * features["brent_proxy_ret"] - 0.35 * features["ulsd_proxy_ret"]
    )

    model = OLSModel().fit(features, target)

    assert model.coef_ is not None
    assert model.intercept_ == pytest.approx(0.001, abs=1e-10)
    assert model.coef_[0] == pytest.approx(0.8, abs=1e-10)
    assert model.coef_[1] == pytest.approx(-0.35, abs=1e-10)
