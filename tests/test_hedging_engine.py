from __future__ import annotations

import numpy as np
import pandas as pd

from costs.transaction_costs import TransactionCostModel
from hedging.engine import HedgeConstraints, run_hedging_engine


def test_perfect_hedge_reduces_variance_to_near_zero_without_costs() -> None:
    rng = np.random.default_rng(21)
    index = pd.bdate_range("2024-01-01", periods=200)
    proxy_returns = pd.DataFrame(
        {
            "proxy_1": rng.normal(0.0, 0.01, size=len(index)),
            "proxy_2": rng.normal(0.0, 0.012, size=len(index)),
        },
        index=index,
    )

    beta_true = pd.Series({"proxy_1": 0.8, "proxy_2": -0.35})
    target_returns = (
        proxy_returns["proxy_1"] * beta_true["proxy_1"]
        + proxy_returns["proxy_2"] * beta_true["proxy_2"]
    )

    hedge_ratios = pd.DataFrame(
        np.tile(beta_true.to_numpy(dtype=float), (len(index), 1)),
        index=index,
        columns=beta_true.index,
    )

    result = run_hedging_engine(
        target_returns=target_returns,
        proxy_returns=proxy_returns,
        hedge_ratios=hedge_ratios,
        rebalance_frequency="daily",
        constraints=HedgeConstraints(max_abs_beta=5.0, leverage_cap=10.0),
        exposure_notional=1.0,
    )

    unhedged_var = float(result.unhedged_pnl.var(ddof=1))
    hedged_var = float(result.hedged_pnl.var(ddof=1))
    assert hedged_var <= unhedged_var * 1e-10


def test_weekly_rebalance_holds_positions_constant_between_rebalances() -> None:
    index = pd.bdate_range("2024-01-01", periods=15)
    proxy_returns = pd.DataFrame(
        {
            "proxy_1": np.full(len(index), 0.01),
            "proxy_2": np.full(len(index), -0.005),
        },
        index=index,
    )
    target_returns = pd.Series(np.full(len(index), 0.004), index=index)

    varying_betas = pd.DataFrame(
        {
            "proxy_1": np.linspace(-3.0, 3.0, len(index)),
            "proxy_2": np.linspace(2.0, -2.0, len(index)),
        },
        index=index,
    )

    result = run_hedging_engine(
        target_returns=target_returns,
        proxy_returns=proxy_returns,
        hedge_ratios=varying_betas,
        rebalance_frequency="weekly",
        constraints=HedgeConstraints(max_abs_beta=1.0, leverage_cap=1.2),
        exposure_notional=1.0,
    )

    weekly_bucket = index.to_period("W-FRI")
    for bucket in weekly_bucket.unique():
        bucket_index = index[weekly_bucket == bucket]
        first_day = bucket_index[0]
        week_positions = result.positions.loc[bucket_index]
        for day in bucket_index[1:]:
            pd.testing.assert_series_equal(
                week_positions.loc[day],
                week_positions.loc[first_day],
                check_names=False,
            )

    non_rebalance_days = index[1:][weekly_bucket[1:] == weekly_bucket[:-1]]
    assert np.allclose(result.turnover.loc[non_rebalance_days].to_numpy(), 0.0)

    gross_leverage = result.positions.abs().sum(axis=1)
    assert (gross_leverage <= 1.2 + 1e-12).all()


def test_engine_applies_transaction_costs_to_net_pnl() -> None:
    index = pd.bdate_range("2024-02-05", periods=3)
    target_returns = pd.Series([0.0, 0.0, 0.0], index=index)
    proxy_returns = pd.DataFrame({"proxy_1": [0.0, 0.0, 0.0]}, index=index)
    hedge_ratios = pd.Series([1.0, 1.5, 1.5], index=index, name="proxy_1")

    result = run_hedging_engine(
        target_returns=target_returns,
        proxy_returns=proxy_returns,
        hedge_ratios=hedge_ratios,
        rebalance_frequency="daily",
        transaction_cost_model=TransactionCostModel(spread_bps=10.0, fixed_fee=0.01),
        exposure_notional=1.0,
    )

    expected_turnover = pd.Series([1.0, 0.5, 0.0], index=index, name="turnover")
    pd.testing.assert_series_equal(result.turnover, expected_turnover)

    expected_cost = pd.Series(
        [0.0110, 0.0105, 0.0000], index=index, name="transaction_cost"
    )
    pd.testing.assert_series_equal(result.transaction_cost, expected_cost)
    assert np.allclose(result.hedged_pnl_gross.to_numpy(), 0.0)
    assert np.allclose(result.hedged_pnl_net.to_numpy(), -expected_cost.to_numpy())
    assert np.allclose(result.hedged_pnl.to_numpy(), result.hedged_pnl_net.to_numpy())
