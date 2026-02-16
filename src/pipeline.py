from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from config.settings import PipelineConfig
from costs.transaction import (
    apply_transaction_costs,
    estimate_static_hedge_transaction_costs,
)
from data.io import load_market_data, save_market_data
from data.synthetic import generate_synthetic_market_data
from evaluation.metrics import evaluate_hedge
from features.returns import add_returns, build_feature_target
from hedging.strategy import apply_static_proxy_hedge
from models.ols import OLSModel
from plots.figures import plot_cumulative_returns


def _load_or_generate_market_data(config: PipelineConfig) -> tuple[pd.DataFrame, str]:
    if config.raw_data_path.exists():
        return load_market_data(config.raw_data_path), "raw-csv"

    synthetic = generate_synthetic_market_data(
        n_days=config.n_synthetic_days,
        seed=config.random_seed,
    )
    save_market_data(synthetic, config.raw_data_path)
    return synthetic, "synthetic"


def _write_markdown_report(
    report_path: Path,
    data_source: str,
    train_size: int,
    test_size: int,
    hedge_weights: Mapping[str, float],
    metrics: Mapping[str, float],
) -> None:
    report_lines = [
        "# Proxy Hedging Case Study Report",
        "",
        "## Data",
        f"- Source: `{data_source}`",
        f"- Train observations: {train_size}",
        f"- Test observations: {test_size}",
        "",
        "## Estimated Static Hedge Ratios",
    ]

    for name, value in hedge_weights.items():
        report_lines.append(f"- {name}: {value:.4f}")

    report_lines.append("")
    report_lines.append("## Performance Metrics")

    for name, value in metrics.items():
        report_lines.append(f"- {name}: {value:.6f}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, float]:
    cfg = config if config is not None else PipelineConfig()
    cfg.validate()
    cfg.ensure_directories()

    market_data, data_source = _load_or_generate_market_data(cfg)
    data_with_returns = add_returns(market_data)
    save_market_data(data_with_returns, cfg.processed_data_path)

    model_table = data_with_returns.set_index("date")
    features, target = build_feature_target(model_table)

    split_index = int(len(model_table) * cfg.train_ratio)
    if split_index <= 1 or split_index >= len(model_table):
        raise ValueError("train_ratio produced an invalid train/test split.")

    x_train, x_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    model = OLSModel().fit(x_train, y_train)
    hedge_weights = model.hedge_weights(x_train.columns)

    hedge_results = apply_static_proxy_hedge(y_test, x_test, hedge_weights)

    period_costs = estimate_static_hedge_transaction_costs(
        hedge_weights=hedge_weights,
        n_periods=len(hedge_results),
        transaction_cost_bps=cfg.transaction_cost_bps,
    )
    hedge_results["transaction_cost"] = period_costs
    hedge_results["hedged_return_after_costs"] = apply_transaction_costs(
        hedge_results["hedged_return_before_costs"].to_numpy(dtype=float),
        hedge_results["transaction_cost"].to_numpy(dtype=float),
    )

    metrics = evaluate_hedge(
        unhedged_returns=hedge_results["unhedged_return"],
        hedged_returns=hedge_results["hedged_return_after_costs"],
    )

    hedge_results.to_csv(cfg.hedged_returns_path, index=True, index_label="date")

    plot_cumulative_returns(
        unhedged_returns=hedge_results["unhedged_return"],
        hedged_returns=hedge_results["hedged_return_after_costs"],
        output_path=cfg.figure_path,
    )

    _write_markdown_report(
        report_path=cfg.report_path,
        data_source=data_source,
        train_size=len(x_train),
        test_size=len(x_test),
        hedge_weights=hedge_weights,
        metrics=metrics,
    )

    return metrics


def main() -> None:
    metrics = run_pipeline()
    print("Pipeline completed. Key metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
