from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from evaluation.metrics import evaluate_hedge_performance
from plots.figures import (
    plot_cumulative_pnl,
    plot_hedge_ratios,
    plot_pnl_histogram,
    plot_rolling_hedge_effectiveness,
)


@dataclass(frozen=True, slots=True)
class MethodEvaluationData:
    """Container for one hedging method's evaluation series."""

    unhedged_pnl: pd.Series
    hedged_pnl_gross: pd.Series
    hedged_pnl_net: pd.Series | None = None
    turnover: pd.Series | None = None
    transaction_cost: pd.Series | None = None
    hedge_ratios: pd.Series | pd.DataFrame | None = None


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return slug.strip("_") or "method"


def _ensure_report_dirs(reports_dir: Path) -> tuple[Path, Path]:
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def generate_results_table(
    method_results: Mapping[str, MethodEvaluationData],
    reports_dir: Path,
    var_confidence: float = 0.95,
    periods_per_year: int = 252,
    filename: str = "method_comparison.csv",
) -> pd.DataFrame:
    """Generate and save a single metrics table comparing all methods."""
    if not method_results:
        raise ValueError("method_results must contain at least one method.")

    _, tables_dir = _ensure_report_dirs(reports_dir)
    rows: list[dict[str, float | str]] = []

    for method_name, payload in method_results.items():
        metrics = evaluate_hedge_performance(
            unhedged_pnl=payload.unhedged_pnl,
            hedged_pnl_gross=payload.hedged_pnl_gross,
            hedged_pnl_net=payload.hedged_pnl_net,
            turnover=payload.turnover,
            transaction_cost=payload.transaction_cost,
            var_confidence=var_confidence,
            periods_per_year=periods_per_year,
        )
        row: dict[str, float | str] = {"method": method_name}
        row.update(metrics)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("method").sort_index()
    table.to_csv(tables_dir / filename)
    return table


def generate_evaluation_artifacts(
    method_results: Mapping[str, MethodEvaluationData],
    reports_dir: Path,
    rolling_window: int = 60,
    var_confidence: float = 0.95,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Generate plots + method comparison table and save under reports/.

    Files generated:
    - CSV: `reports/tables/method_comparison.csv`
    - Per-method CSV: `reports/tables/<method>_pnl.csv`
    - PNG figures in `reports/figures/`:
      cumulative P&L, histogram, rolling hedge effectiveness, and hedge ratios.
    """
    figures_dir, tables_dir = _ensure_report_dirs(reports_dir)
    table = generate_results_table(
        method_results=method_results,
        reports_dir=reports_dir,
        var_confidence=var_confidence,
        periods_per_year=periods_per_year,
    )

    for method_name, payload in method_results.items():
        slug = _slugify(method_name)
        hedged_to_plot = (
            payload.hedged_pnl_net
            if payload.hedged_pnl_net is not None
            else payload.hedged_pnl_gross
        )

        plot_cumulative_pnl(
            unhedged_pnl=payload.unhedged_pnl,
            hedged_pnl=hedged_to_plot,
            output_path=figures_dir / f"{slug}_cumulative_pnl.png",
        )
        plot_pnl_histogram(
            unhedged_pnl=payload.unhedged_pnl,
            hedged_pnl=hedged_to_plot,
            output_path=figures_dir / f"{slug}_pnl_histogram.png",
        )
        plot_rolling_hedge_effectiveness(
            unhedged_pnl=payload.unhedged_pnl,
            hedged_pnl=hedged_to_plot,
            output_path=figures_dir / f"{slug}_rolling_effectiveness.png",
            window=rolling_window,
        )

        if payload.hedge_ratios is not None:
            plot_hedge_ratios(
                hedge_ratios=payload.hedge_ratios,
                output_path=figures_dir / f"{slug}_hedge_ratios.png",
            )

        pnl_table = pd.DataFrame(
            {
                "unhedged_pnl": payload.unhedged_pnl,
                "hedged_pnl_gross": payload.hedged_pnl_gross,
                "hedged_pnl_net": hedged_to_plot,
            }
        )
        if payload.transaction_cost is not None:
            pnl_table["transaction_cost"] = payload.transaction_cost
        if payload.turnover is not None:
            pnl_table["turnover"] = payload.turnover
        pnl_table.to_csv(tables_dir / f"{slug}_pnl.csv", index=True, index_label="date")

    return table
