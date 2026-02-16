from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.kalman import (
    KalmanHedgeEstimator,
    KalmanMultiProxyHedgeEstimator,
    KalmanSingleProxyHedgeEstimator,
)


def _make_panel(n_obs: int = 120, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=n_obs)
    proxy_1 = rng.normal(0.0, 0.02, size=n_obs)
    proxy_2 = rng.normal(0.0, 0.02, size=n_obs)
    true_beta_1 = np.linspace(0.4, 0.9, num=n_obs)
    true_beta_2 = np.linspace(-0.2, 0.3, num=n_obs)
    noise = rng.normal(0.0, 0.005, size=n_obs)
    target = true_beta_1 * proxy_1 + true_beta_2 * proxy_2 + noise
    return pd.DataFrame(
        {
            "jet_fuel_illiquid": target,
            "proxy_1": proxy_1,
            "proxy_2": proxy_2,
        },
        index=index,
    )


def test_kalman_no_lookahead_ratios_at_t() -> None:
    base = _make_panel(n_obs=90, seed=3)
    shock_date = base.index[45]

    model_base = KalmanSingleProxyHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_column="proxy_1",
        process_noise=1e-4,
        observation_noise=1e-3,
        max_abs_hedge_ratio=5.0,
    ).fit(base[["jet_fuel_illiquid", "proxy_1"]], calibrate=False)

    shocked = base[["jet_fuel_illiquid", "proxy_1"]].copy()
    shocked.loc[shock_date, "proxy_1"] = 5.0
    shocked.loc[shock_date, "jet_fuel_illiquid"] = -5.0

    model_shocked = KalmanSingleProxyHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_column="proxy_1",
        process_noise=1e-4,
        observation_noise=1e-3,
        max_abs_hedge_ratio=5.0,
    ).fit(shocked, calibrate=False)

    beta_base_t = float(model_base.predict_hedge_ratio(shock_date)["proxy_1"])
    beta_shocked_t = float(model_shocked.predict_hedge_ratio(shock_date)["proxy_1"])
    assert beta_base_t == pytest.approx(beta_shocked_t)

    next_date = base.index[46]
    beta_base_next = float(model_base.predict_hedge_ratio(next_date)["proxy_1"])
    beta_shocked_next = float(model_shocked.predict_hedge_ratio(next_date)["proxy_1"])
    assert beta_base_next != pytest.approx(beta_shocked_next)


def test_kalman_calibration_selects_grid_hyperparameters() -> None:
    panel = _make_panel(n_obs=140, seed=7)
    train = panel.iloc[:100]
    valid = panel.iloc[100:]

    process_grid = (1e-6, 1e-5, 1e-4)
    observation_grid = (1e-4, 1e-3)
    model = KalmanMultiProxyHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1", "proxy_2"),
        process_noise_grid=process_grid,
        observation_noise_grid=observation_grid,
        max_abs_hedge_ratio=5.0,
    ).fit(train, validation_data=valid, calibrate=True)

    assert model.process_noise in process_grid
    assert model.observation_noise in observation_grid
    assert model.calibration_results_ is not None
    assert len(model.calibration_results_) == len(process_grid) * len(observation_grid)

    hedge_ratios = model.hedge_ratio_time_series()
    assert hedge_ratios.index.equals(train.index)
    assert list(hedge_ratios.columns) == ["proxy_1", "proxy_2"]


def test_generic_kalman_predict_and_ratio_series() -> None:
    panel = _make_panel(n_obs=80, seed=11)
    model = KalmanHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1", "proxy_2"),
        process_noise=1e-4,
        observation_noise=1e-3,
        max_abs_hedge_ratio=2.0,
    ).fit(panel, calibrate=False)

    x_slice = panel.loc[panel.index[:3], ["proxy_1", "proxy_2"]]
    preds = model.predict(x_slice)
    assert len(preds) == 3

    ratios = model.hedge_ratio_time_series()
    assert ratios.index.equals(panel.index)
    assert (ratios.abs() <= 2.0 + 1e-12).all().all()
