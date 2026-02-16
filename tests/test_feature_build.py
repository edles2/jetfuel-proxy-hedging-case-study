from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.build import FeatureConfig, build_dataset


def _prices_from_returns(returns: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    return base * (1.0 + returns).cumprod()


def test_build_dataset_respects_no_lookahead_under_same_day_shock() -> None:
    index = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.DataFrame(
        {
            "jet_fuel_illiquid": np.linspace(0.001, 0.010, num=10),
            "proxy_1": np.linspace(0.002, 0.020, num=10),
            "proxy_2": np.linspace(0.003, 0.030, num=10),
        },
        index=index,
    )
    prices = _prices_from_returns(returns)

    config = FeatureConfig(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1", "proxy_2"),
        max_lag=2,
        rolling_windows=(3,),
        include_return_spreads=True,
        include_price_spreads=True,
        include_pca=True,
        pca_components=1,
        pca_min_history=3,
        dropna=False,
    )
    features_base, _, _ = build_dataset(prices=prices, returns=returns, config=config)

    shocked_returns = returns.copy()
    shocked_returns.loc[index[6], "proxy_1"] = 9.0
    shocked_returns.loc[index[6], "proxy_2"] = -9.0
    shocked_prices = _prices_from_returns(shocked_returns)

    features_shocked, _, _ = build_dataset(
        prices=shocked_prices,
        returns=shocked_returns,
        config=config,
    )

    same_day = index[6]
    pd.testing.assert_series_equal(
        features_base.loc[same_day],
        features_shocked.loc[same_day],
        check_names=False,
    )

    next_day = index[7]
    assert not np.allclose(
        features_base.loc[next_day].to_numpy(dtype=float),
        features_shocked.loc[next_day].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_build_dataset_lag_and_rolling_corr_are_shifted() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame(
        {
            "jet_fuel_illiquid": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "proxy_1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "proxy_2": [8.0, 9.0, 8.0, 9.0, 8.0, 9.0],
        },
        index=index,
    )
    prices = _prices_from_returns(returns / 100.0)

    config = FeatureConfig(
        target_column="jet_fuel_illiquid",
        proxy_columns=("proxy_1", "proxy_2"),
        max_lag=1,
        rolling_windows=(2,),
        include_return_spreads=False,
        include_price_spreads=False,
        include_pca=False,
        dropna=False,
    )
    features, target, dataset_index = build_dataset(
        prices=prices,
        returns=returns,
        config=config,
    )

    t = index[3]
    assert target.loc[t] == returns.loc[t, "jet_fuel_illiquid"]
    assert features.loc[t, "proxy_1_ret_lag_1"] == returns.loc[index[2], "proxy_1"]

    # window=2 correlation at t uses rows (t-2, t-1) after one-step shift.
    expected_corr = np.corrcoef([2.0, 3.0], [11.0, 12.0])[0, 1]
    assert features.loc[t, "roll_corr_jet_fuel_illiquid_proxy_1_w2"] == pytest.approx(
        expected_corr
    )

    assert dataset_index.equals(features.index)
