from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def estimate_static_hedge_transaction_costs(
    hedge_weights: Mapping[str, float],
    n_periods: int,
    transaction_cost_bps: float,
) -> np.ndarray:
    """Estimate per-period transaction costs for entering and exiting a static hedge."""
    if n_periods <= 0:
        raise ValueError("n_periods must be positive.")
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be non-negative.")

    gross_notional = float(sum(abs(value) for value in hedge_weights.values()))
    round_trip_cost = 2.0 * gross_notional * (transaction_cost_bps / 10_000.0)
    return np.full(shape=n_periods, fill_value=round_trip_cost / n_periods, dtype=float)


def apply_transaction_costs(
    hedged_returns_before_costs: np.ndarray,
    period_costs: np.ndarray,
) -> np.ndarray:
    if hedged_returns_before_costs.shape != period_costs.shape:
        raise ValueError(
            "hedged_returns_before_costs and period_costs must have same shape."
        )

    return hedged_returns_before_costs - period_costs
