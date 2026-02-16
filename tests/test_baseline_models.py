from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.baselines import (
    LassoHedgeEstimator,
    RidgeHedgeEstimator,
    RollingOLSHedgeEstimator,
    StaticOLSHedgeEstimator,
)


def _synthetic_train_data(n_obs: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_obs)
    proxy_1 = rng.normal(loc=0.0, scale=1.0, size=n_obs)
    proxy_2 = 0.92 * proxy_1 + rng.normal(loc=0.0, scale=0.15, size=n_obs)
    target = 0.8 * proxy_1 - 0.3 * proxy_2 + rng.normal(loc=0.0, scale=0.05, size=n_obs)
    return pd.DataFrame(
        {
            "jet_fuel_illiquid": target,
            "proxy_1": proxy_1,
            "proxy_2": proxy_2,
        },
        index=idx,
    )


def test_static_ols_applies_hedge_ratio_cap() -> None:
    idx = pd.bdate_range("2024-01-01", periods=80)
    proxy_1 = np.linspace(0.01, 1.0, num=80)
    proxy_2 = np.linspace(0.02, 0.6, num=80)
    target = 2.5 * proxy_1 - 0.2 * proxy_2
    train_data = pd.DataFrame(
        {
            "jet_fuel_illiquid": target,
            "proxy_1": proxy_1,
            "proxy_2": proxy_2,
        },
        index=idx,
    )

    model = StaticOLSHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1", "proxy_2"),
        max_abs_hedge_ratio=1.0,
    ).fit(train_data)

    ratios = model.predict_hedge_ratio()
    assert (ratios.abs() <= 1.0 + 1e-12).all()
    assert ratios["proxy_1"] == pytest.approx(1.0)

    ratio_ts = model.hedge_ratio_time_series()
    assert ratio_ts.index.equals(train_data.index)


def test_rolling_ols_uses_only_past_information_per_date() -> None:
    train_data = _synthetic_train_data(n_obs=90, seed=4)
    shock_date = train_data.index[45]

    model_base = RollingOLSHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1",),
        window=20,
        refit_frequency=1,
        max_abs_hedge_ratio=100.0,
    ).fit(train_data[["jet_fuel_illiquid", "proxy_1"]])

    shocked = train_data[["jet_fuel_illiquid", "proxy_1"]].copy()
    shocked.loc[shock_date, "proxy_1"] = 25.0
    shocked.loc[shock_date, "jet_fuel_illiquid"] = -25.0
    model_shocked = RollingOLSHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1",),
        window=20,
        refit_frequency=1,
        max_abs_hedge_ratio=100.0,
    ).fit(shocked)

    ratio_base_t = float(model_base.predict_hedge_ratio(shock_date)["proxy_1"])
    ratio_shocked_t = float(model_shocked.predict_hedge_ratio(shock_date)["proxy_1"])
    assert ratio_base_t == pytest.approx(ratio_shocked_t)

    next_date = train_data.index[46]
    ratio_base_next = float(model_base.predict_hedge_ratio(next_date)["proxy_1"])
    ratio_shocked_next = float(model_shocked.predict_hedge_ratio(next_date)["proxy_1"])
    assert ratio_base_next != pytest.approx(ratio_shocked_next)


def test_regularized_models_fit_with_time_series_cv_and_align_outputs() -> None:
    train_data = _synthetic_train_data(n_obs=140, seed=9)
    alphas = (1e-3, 1e-2, 1e-1, 1.0)

    ridge = RidgeHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1", "proxy_2"),
        alphas=alphas,
        cv_splits=5,
        max_abs_hedge_ratio=2.0,
    ).fit(train_data)
    lasso = LassoHedgeEstimator(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1", "proxy_2"),
        alphas=alphas,
        cv_splits=5,
        max_abs_hedge_ratio=2.0,
    ).fit(train_data)

    assert ridge.best_alpha_ in alphas
    assert lasso.best_alpha_ in alphas

    x_slice = train_data.loc[train_data.index[:5], ["proxy_1", "proxy_2"]]
    ridge_pred = ridge.predict(x_slice)
    lasso_pred = lasso.predict(x_slice)
    assert len(ridge_pred) == 5
    assert len(lasso_pred) == 5

    ridge_ts = ridge.hedge_ratio_time_series(index=train_data.index)
    lasso_ts = lasso.hedge_ratio_time_series(index=train_data.index)
    assert ridge_ts.index.equals(train_data.index)
    assert lasso_ts.index.equals(train_data.index)
