from __future__ import annotations

import pandas as pd

from costs.transaction_costs import (
    TransactionCostConfig,
    TransactionCostModel,
    compute_delta_positions,
    compute_transaction_costs,
    compute_turnover,
    compute_turnover_metrics,
)


def test_transaction_cost_formula_and_turnover_metrics() -> None:
    index = pd.bdate_range("2024-03-01", periods=3)
    positions = pd.DataFrame(
        {
            "proxy_1": [-1.0, -1.5, -1.5],
            "proxy_2": [0.0, 0.0, 0.0],
        },
        index=index,
    )

    delta = compute_delta_positions(positions)
    expected_delta = pd.DataFrame(
        {
            "proxy_1": [-1.0, -0.5, 0.0],
            "proxy_2": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    pd.testing.assert_frame_equal(delta, expected_delta)

    turnover = compute_turnover(delta)
    expected_turnover = pd.Series([1.0, 0.5, 0.0], index=index, name="turnover")
    pd.testing.assert_series_equal(turnover, expected_turnover)

    model = TransactionCostModel(spread_bps=10.0, fixed_fee=0.01)
    costs = compute_transaction_costs(delta, model=model)
    expected_costs = pd.Series([0.0110, 0.0105, 0.0000], index=index)
    expected_costs.name = "transaction_cost"
    pd.testing.assert_series_equal(costs, expected_costs)

    metrics = compute_turnover_metrics(turnover)
    assert metrics["total_turnover"] == 1.5
    assert metrics["trade_count"] == 2.0


def test_transaction_cost_config_resolves_low_med_high_scenarios() -> None:
    config = TransactionCostConfig.from_dict(
        {
            "default_scenario": "med",
            "scenarios": {
                "low": {"spread_bps": 0.5, "fixed_fee": 0.0},
                "med": {"spread_bps": 2.0, "fixed_fee": 0.0001},
                "high": {"spread_bps": 5.0, "fixed_fee": 0.0010},
            },
        }
    )

    low = config.resolve("low")
    med = config.resolve("med")
    high = config.resolve("high")

    assert low.spread_bps < med.spread_bps < high.spread_bps
    assert low.fixed_fee < med.fixed_fee < high.fixed_fee
    assert config.resolve().spread_bps == med.spread_bps
