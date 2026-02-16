from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from costs.transaction_costs import (
    TransactionCostConfig,
    TransactionCostModel,
    compute_delta_positions,
    compute_transaction_costs,
    compute_turnover_metrics,
)


@dataclass(frozen=True, slots=True)
class HedgeConstraints:
    """Trading constraints applied to hedge ratios before position sizing."""

    max_abs_beta: float | None = None
    leverage_cap: float | None = None


@dataclass(frozen=True, slots=True)
class HedgeEngineResult:
    """Outputs produced by the hedging engine."""

    unhedged_pnl: pd.Series
    hedged_pnl_gross: pd.Series
    transaction_cost: pd.Series
    hedged_pnl_net: pd.Series
    delta_positions: pd.DataFrame
    hedged_pnl: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    turnover_metrics: dict[str, float]
    transaction_cost_model: TransactionCostModel


def _validate_constraints(constraints: HedgeConstraints) -> None:
    if constraints.max_abs_beta is not None and constraints.max_abs_beta <= 0.0:
        raise ValueError("max_abs_beta must be positive when provided.")
    if constraints.leverage_cap is not None and constraints.leverage_cap <= 0.0:
        raise ValueError("leverage_cap must be positive when provided.")


def _prepare_hedge_ratio_panel(
    hedge_ratios: pd.Series | pd.DataFrame,
    proxy_columns: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    if isinstance(hedge_ratios, pd.Series):
        if len(proxy_columns) != 1:
            raise ValueError(
                "Series hedge_ratios can only be used with a single proxy column."
            )
        ratio_frame = hedge_ratios.to_frame(name=proxy_columns[0])
    else:
        ratio_frame = hedge_ratios.copy()

    missing_proxy_cols = set(proxy_columns).difference(ratio_frame.columns)
    if missing_proxy_cols:
        raise ValueError(
            f"hedge_ratios is missing proxy columns: {sorted(missing_proxy_cols)}"
        )

    ratio_panel = ratio_frame.loc[:, proxy_columns].sort_index().reindex(index).ffill()
    ratio_panel = ratio_panel.fillna(0.0)
    return ratio_panel


def _apply_constraints_to_row(
    beta_row: pd.Series,
    constraints: HedgeConstraints,
) -> pd.Series:
    constrained = beta_row.copy()

    if constraints.max_abs_beta is not None:
        constrained = constrained.clip(
            lower=-constraints.max_abs_beta,
            upper=constraints.max_abs_beta,
        )

    if constraints.leverage_cap is not None:
        gross = float(constrained.abs().sum())
        if gross > constraints.leverage_cap and gross > 0.0:
            scale = constraints.leverage_cap / gross
            constrained = constrained * scale

    return constrained


def _rebalance_flags(
    index: pd.DatetimeIndex, frequency: Literal["daily", "weekly"]
) -> pd.Series:
    if frequency == "daily":
        return pd.Series(True, index=index)
    if frequency != "weekly":
        raise ValueError("rebalance_frequency must be one of {'daily', 'weekly'}.")

    weekly_bucket = index.to_period("W-FRI")
    flags = pd.Series(False, index=index)
    flags.iloc[0] = True
    flags.iloc[1:] = weekly_bucket[1:] != weekly_bucket[:-1]
    return flags


def run_hedging_engine(
    target_returns: pd.Series,
    proxy_returns: pd.DataFrame,
    hedge_ratios: pd.Series | pd.DataFrame,
    rebalance_frequency: Literal["daily", "weekly"] = "daily",
    constraints: HedgeConstraints | None = None,
    exposure_notional: float = 1.0,
    cost_config: TransactionCostConfig | None = None,
    cost_scenario: str | None = None,
    transaction_cost_model: TransactionCostModel | None = None,
) -> HedgeEngineResult:
    """Compute unhedged/hedged P&L from returns, hedge ratios, and rebalancing rules.

    Mechanics:
    - On each rebalance date: position_t = -beta_t * exposure_notional.
    - Between rebalances: positions are held constant.
    - Gross hedged P&L_t = exposure_notional * target_return_t
      + sum(position_t * proxy_return_t).
    - Transaction cost_t = spread_bps * |delta_position_t| + fixed_fee * 1_{trade}.
      spread_bps is interpreted in basis points over gross traded notional.
    - Net hedged P&L_t = gross hedged P&L_t - transaction cost_t.
    """
    if exposure_notional == 0.0:
        raise ValueError("exposure_notional must be non-zero.")
    if transaction_cost_model is not None and (
        cost_config is not None or cost_scenario is not None
    ):
        raise ValueError(
            "Provide either transaction_cost_model or (cost_config/cost_scenario), not both."
        )

    policy = constraints or HedgeConstraints()
    _validate_constraints(policy)

    if proxy_returns.empty:
        raise ValueError("proxy_returns must not be empty.")
    if target_returns.empty:
        raise ValueError("target_returns must not be empty.")

    common_index = target_returns.index.intersection(proxy_returns.index).sort_values()
    if common_index.empty:
        raise ValueError(
            "No overlapping dates between target_returns and proxy_returns."
        )

    target_aligned = target_returns.loc[common_index].astype(float)
    proxy_aligned = proxy_returns.loc[common_index].astype(float)
    proxy_columns = list(proxy_aligned.columns)

    ratio_panel = _prepare_hedge_ratio_panel(
        hedge_ratios=hedge_ratios,
        proxy_columns=proxy_columns,
        index=common_index,
    )

    if not isinstance(common_index, pd.DatetimeIndex):
        raise ValueError("Input series must be indexed by a DatetimeIndex.")
    rebalance = _rebalance_flags(common_index, frequency=rebalance_frequency)

    positions = pd.DataFrame(0.0, index=common_index, columns=proxy_columns)
    current_position = pd.Series(0.0, index=proxy_columns)

    for timestamp in common_index:
        if bool(rebalance.loc[timestamp]):
            constrained_beta = _apply_constraints_to_row(
                ratio_panel.loc[timestamp], policy
            )
            current_position = -float(exposure_notional) * constrained_beta
        positions.loc[timestamp] = current_position

    delta_positions = compute_delta_positions(positions)
    turnover = delta_positions.abs().sum(axis=1)
    turnover.name = "turnover"

    unhedged_pnl = float(exposure_notional) * target_aligned
    unhedged_pnl.name = "unhedged_pnl"
    hedge_leg_pnl = (positions * proxy_aligned).sum(axis=1)
    hedged_pnl_gross = unhedged_pnl + hedge_leg_pnl
    hedged_pnl_gross.name = "hedged_pnl_gross"

    if transaction_cost_model is not None:
        active_cost_model = transaction_cost_model
    elif cost_config is not None or cost_scenario is not None:
        active_cost_config = cost_config or TransactionCostConfig()
        active_cost_model = active_cost_config.resolve(cost_scenario)
    else:
        active_cost_model = TransactionCostModel(spread_bps=0.0, fixed_fee=0.0)

    transaction_cost = compute_transaction_costs(
        delta_positions=delta_positions,
        model=active_cost_model,
    )
    hedged_pnl_net = hedged_pnl_gross - transaction_cost
    hedged_pnl_net.name = "hedged_pnl_net"
    turnover_metrics = compute_turnover_metrics(turnover)

    return HedgeEngineResult(
        unhedged_pnl=unhedged_pnl,
        hedged_pnl_gross=hedged_pnl_gross,
        transaction_cost=transaction_cost,
        hedged_pnl_net=hedged_pnl_net,
        delta_positions=delta_positions,
        hedged_pnl=hedged_pnl_net,
        positions=positions,
        turnover=turnover,
        turnover_metrics=turnover_metrics,
        transaction_cost_model=active_cost_model,
    )
