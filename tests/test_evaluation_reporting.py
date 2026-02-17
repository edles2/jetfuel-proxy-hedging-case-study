from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import evaluate_hedge_performance
from evaluation.report import MethodEvaluationData, generate_evaluation_artifacts


def test_evaluate_hedge_performance_metrics_include_risk_and_costs() -> None:
    index = pd.bdate_range("2024-01-01", periods=120)
    unhedged = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
    hedged_gross = 0.5 * unhedged
    transaction_cost = pd.Series(0.0001, index=index)
    hedged_net = hedged_gross - transaction_cost
    turnover = pd.Series(0.2, index=index)

    metrics = evaluate_hedge_performance(
        unhedged_pnl=unhedged,
        hedged_pnl_gross=hedged_gross,
        hedged_pnl_net=hedged_net,
        turnover=turnover,
        transaction_cost=transaction_cost,
    )

    assert metrics["variance_reduction"] > 0.70
    assert "hedge_effectiveness" in metrics
    assert "tracking_error" in metrics
    assert "hedged_net_sharpe_annualized_no_rf" in metrics
    assert "unhedged_var_loss" in metrics
    assert "hedged_net_cvar_loss" in metrics
    assert metrics["total_turnover"] == float(turnover.sum())
    assert metrics["total_transaction_cost"] == float(transaction_cost.sum())


def test_generate_evaluation_artifacts_writes_tables_and_figures(tmp_path) -> None:
    index = pd.bdate_range("2024-01-01", periods=140)
    rng = np.random.default_rng(123)
    unhedged = pd.Series(rng.normal(0.0, 0.01, size=len(index)), index=index)
    hedged_a = 0.7 * unhedged
    hedged_b = 0.4 * unhedged
    costs_a = pd.Series(0.00005, index=index)
    costs_b = pd.Series(0.00008, index=index)
    turnover_a = pd.Series(0.15, index=index)
    turnover_b = pd.Series(0.20, index=index)
    ratio_a = pd.DataFrame({"proxy_1": 0.8}, index=index)
    ratio_b = pd.DataFrame({"proxy_1": 0.6, "proxy_2": -0.2}, index=index)

    reports_dir = tmp_path / "reports"
    table = generate_evaluation_artifacts(
        method_results={
            "Static OLS": MethodEvaluationData(
                unhedged_pnl=unhedged,
                hedged_pnl_gross=hedged_a,
                hedged_pnl_net=hedged_a - costs_a,
                turnover=turnover_a,
                transaction_cost=costs_a,
                hedge_ratios=ratio_a,
            ),
            "Kalman Dynamic": MethodEvaluationData(
                unhedged_pnl=unhedged,
                hedged_pnl_gross=hedged_b,
                hedged_pnl_net=hedged_b - costs_b,
                turnover=turnover_b,
                transaction_cost=costs_b,
                hedge_ratios=ratio_b,
            ),
        },
        reports_dir=reports_dir,
        rolling_window=30,
    )

    assert list(table.index) == ["Kalman Dynamic", "Static OLS"]
    assert (reports_dir / "tables" / "method_comparison.csv").exists()
    assert (reports_dir / "tables" / "static_ols_pnl.csv").exists()
    assert (reports_dir / "tables" / "kalman_dynamic_pnl.csv").exists()

    assert (reports_dir / "figures" / "static_ols_cumulative_pnl.png").exists()
    assert (reports_dir / "figures" / "static_ols_pnl_histogram.png").exists()
    assert (reports_dir / "figures" / "static_ols_rolling_effectiveness.png").exists()
    assert (reports_dir / "figures" / "static_ols_hedge_ratios.png").exists()

    assert (reports_dir / "figures" / "kalman_dynamic_cumulative_pnl.png").exists()
    assert (reports_dir / "figures" / "kalman_dynamic_pnl_histogram.png").exists()
    assert (
        reports_dir / "figures" / "kalman_dynamic_rolling_effectiveness.png"
    ).exists()
    assert (reports_dir / "figures" / "kalman_dynamic_hedge_ratios.png").exists()

    # Canonical stable filenames used by CASE_STUDY.md
    assert (reports_dir / "figures" / "cum_pnl.png").exists()
    assert (reports_dir / "figures" / "hedge_ratios.png").exists()
    assert (reports_dir / "figures" / "rolling_effectiveness.png").exists()
    assert (reports_dir / "figures" / "pnl_distribution.png").exists()
