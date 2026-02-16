from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class TransactionCostModel:
    """Linear-plus-fixed transaction cost model.

    Assumptions:
    - `delta_position_t` is in notional units.
    - `spread_bps` is applied to gross traded notional:
      variable_cost_t = (spread_bps / 10_000) * sum_i |delta_position_{t,i}|.
    - `fixed_fee` is charged once per timestamp if any trade occurs.
    """

    spread_bps: float = 0.0
    fixed_fee: float = 0.0

    def validate(self) -> None:
        if self.spread_bps < 0.0:
            raise ValueError("spread_bps must be non-negative.")
        if self.fixed_fee < 0.0:
            raise ValueError("fixed_fee must be non-negative.")


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    """Scenario configuration for transaction-cost assumptions."""

    scenarios: dict[str, TransactionCostModel] = field(
        default_factory=lambda: {
            "low": TransactionCostModel(spread_bps=0.5, fixed_fee=0.0),
            "med": TransactionCostModel(spread_bps=2.0, fixed_fee=0.00005),
            "high": TransactionCostModel(spread_bps=5.0, fixed_fee=0.0002),
        }
    )
    default_scenario: str = "med"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TransactionCostConfig":
        scenarios_raw = raw.get("scenarios", {})
        if not isinstance(scenarios_raw, Mapping):
            raise ValueError("`scenarios` must be a mapping.")

        parsed: dict[str, TransactionCostModel] = {}
        for name, payload in scenarios_raw.items():
            if not isinstance(payload, Mapping):
                raise ValueError(f"Scenario {name} must be a mapping.")
            model = TransactionCostModel(
                spread_bps=float(payload.get("spread_bps", 0.0)),
                fixed_fee=float(payload.get("fixed_fee", 0.0)),
            )
            model.validate()
            parsed[str(name)] = model

        default_scenario = str(raw.get("default_scenario", "med"))
        config = cls(scenarios=parsed, default_scenario=default_scenario)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.scenarios:
            raise ValueError("At least one cost scenario must be provided.")
        for model in self.scenarios.values():
            model.validate()
        if self.default_scenario not in self.scenarios:
            raise ValueError(
                f"default_scenario '{self.default_scenario}' not found in scenarios."
            )

    def resolve(self, scenario: str | None = None) -> TransactionCostModel:
        scenario_name = scenario or self.default_scenario
        if scenario_name not in self.scenarios:
            raise ValueError(
                f"Unknown cost scenario '{scenario_name}'. "
                f"Available: {sorted(self.scenarios)}"
            )
        return self.scenarios[scenario_name]


def compute_delta_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """Compute delta positions, including the initial entry trade at t0."""
    if positions.empty:
        raise ValueError("positions must not be empty.")

    delta = positions.diff()
    delta.iloc[0] = positions.iloc[0]
    return delta


def compute_turnover(delta_positions: pd.DataFrame) -> pd.Series:
    """Compute gross turnover as row-wise sum of absolute traded notionals."""
    turnover = delta_positions.abs().sum(axis=1)
    turnover.name = "turnover"
    return turnover


def compute_transaction_costs(
    delta_positions: pd.DataFrame,
    model: TransactionCostModel,
) -> pd.Series:
    """Compute per-period transaction costs using spread + fixed fee policy."""
    model.validate()
    turnover = compute_turnover(delta_positions)
    has_trade = turnover > 0.0

    variable_cost = (model.spread_bps / 10_000.0) * turnover
    fixed_cost = has_trade.astype(float) * model.fixed_fee
    costs = variable_cost + fixed_cost
    costs.name = "transaction_cost"
    return costs


def compute_turnover_metrics(turnover: pd.Series) -> dict[str, float]:
    """Summarize turnover for scenario analysis and cost attribution."""
    if turnover.empty:
        raise ValueError("turnover must not be empty.")

    trade_count = int((turnover > 0.0).sum())
    metrics = {
        "total_turnover": float(turnover.sum()),
        "average_turnover": float(turnover.mean()),
        "max_turnover": float(turnover.max()),
        "trade_count": float(trade_count),
    }
    return metrics
