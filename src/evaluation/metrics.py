from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_pair(
    unhedged: pd.Series, hedged: pd.Series
) -> tuple[pd.Series, pd.Series]:
    aligned_index = unhedged.index.intersection(hedged.index).sort_values()
    if aligned_index.empty:
        raise ValueError("No overlapping dates between unhedged and hedged series.")
    x = unhedged.loc[aligned_index].astype(float)
    y = hedged.loc[aligned_index].astype(float)
    if x.isna().any() or y.isna().any():
        raise ValueError("Input P&L series must not contain NaN values.")
    return x, y


def annualized_volatility(series: pd.Series, periods_per_year: int = 252) -> float:
    return float(series.std(ddof=1) * np.sqrt(periods_per_year))


def variance_reduction(unhedged_pnl: pd.Series, hedged_pnl: pd.Series) -> float:
    """1 - Var(hedged) / Var(unhedged)."""
    unhedged, hedged = _validate_pair(unhedged_pnl, hedged_pnl)
    unhedged_var = float(unhedged.var(ddof=1))
    if unhedged_var == 0.0:
        raise ValueError(
            "Cannot compute variance reduction when unhedged variance is 0."
        )
    hedged_var = float(hedged.var(ddof=1))
    return 1.0 - (hedged_var / unhedged_var)


def hedge_effectiveness(unhedged_pnl: pd.Series, hedged_pnl: pd.Series) -> float:
    """Hedge effectiveness, defined here as variance reduction."""
    return variance_reduction(unhedged_pnl, hedged_pnl)


def tracking_error(unhedged_pnl: pd.Series, hedged_pnl: pd.Series) -> float:
    """Tracking error as std(unhedged - hedged)."""
    unhedged, hedged = _validate_pair(unhedged_pnl, hedged_pnl)
    return float((unhedged - hedged).std(ddof=1))


def sharpe_ratio_no_rf(series: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio assuming zero risk-free rate."""
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float((mean / std) * np.sqrt(periods_per_year))


def value_at_risk(series: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR as positive loss magnitude at given confidence."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    q = float(series.quantile(1.0 - confidence))
    return float(-q)


def conditional_value_at_risk(series: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR (expected shortfall) as positive loss magnitude."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    threshold = float(series.quantile(1.0 - confidence))
    tail = series[series <= threshold]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1.0 + returns).cumprod()
    running_peak = cumulative.cummax()
    drawdowns = cumulative / running_peak - 1.0
    return float(drawdowns.min())


def evaluate_hedge_performance(
    unhedged_pnl: pd.Series,
    hedged_pnl_gross: pd.Series,
    hedged_pnl_net: pd.Series | None = None,
    turnover: pd.Series | None = None,
    transaction_cost: pd.Series | None = None,
    var_confidence: float = 0.95,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute risk/performance/cost metrics for hedging results.

    Notes:
    - Sharpe is reported as `*_sharpe_annualized_no_rf` (zero risk-free assumption).
    - VaR/CVaR are reported as positive loss magnitudes.
    """
    unhedged, hedged_gross = _validate_pair(unhedged_pnl, hedged_pnl_gross)
    hedged_net = (
        hedged_gross if hedged_pnl_net is None else hedged_pnl_net.loc[unhedged.index]
    )

    if hedged_net.isna().any():
        raise ValueError("hedged_pnl_net must not contain NaN values on aligned index.")

    metrics = {
        "unhedged_mean": float(unhedged.mean()),
        "hedged_gross_mean": float(hedged_gross.mean()),
        "hedged_net_mean": float(hedged_net.mean()),
        "unhedged_vol_annualized": annualized_volatility(
            unhedged, periods_per_year=periods_per_year
        ),
        "hedged_gross_vol_annualized": annualized_volatility(
            hedged_gross, periods_per_year=periods_per_year
        ),
        "hedged_net_vol_annualized": annualized_volatility(
            hedged_net, periods_per_year=periods_per_year
        ),
        "variance_reduction": variance_reduction(unhedged, hedged_net),
        "hedge_effectiveness": hedge_effectiveness(unhedged, hedged_net),
        "tracking_error": tracking_error(unhedged, hedged_net),
        "unhedged_var_loss": value_at_risk(unhedged, confidence=var_confidence),
        "hedged_net_var_loss": value_at_risk(hedged_net, confidence=var_confidence),
        "unhedged_cvar_loss": conditional_value_at_risk(
            unhedged, confidence=var_confidence
        ),
        "hedged_net_cvar_loss": conditional_value_at_risk(
            hedged_net, confidence=var_confidence
        ),
        "hedged_net_sharpe_annualized_no_rf": sharpe_ratio_no_rf(
            hedged_net, periods_per_year=periods_per_year
        ),
        "unhedged_max_drawdown": _max_drawdown(unhedged),
        "hedged_net_max_drawdown": _max_drawdown(hedged_net),
    }

    if turnover is not None:
        aligned_turnover = turnover.loc[unhedged.index].astype(float)
        metrics["total_turnover"] = float(aligned_turnover.sum())
        metrics["average_turnover"] = float(aligned_turnover.mean())
    else:
        metrics["total_turnover"] = 0.0
        metrics["average_turnover"] = 0.0

    if transaction_cost is not None:
        aligned_costs = transaction_cost.loc[unhedged.index].astype(float)
        metrics["total_transaction_cost"] = float(aligned_costs.sum())
        metrics["average_transaction_cost"] = float(aligned_costs.mean())
    else:
        metrics["total_transaction_cost"] = 0.0
        metrics["average_transaction_cost"] = 0.0

    return metrics


def evaluate_hedge(
    unhedged_returns: pd.Series,
    hedged_returns: pd.Series,
) -> dict[str, float]:
    """Backward-compatible wrapper for legacy pipeline calls."""
    return evaluate_hedge_performance(
        unhedged_pnl=unhedged_returns,
        hedged_pnl_gross=hedged_returns,
        hedged_pnl_net=hedged_returns,
    )
