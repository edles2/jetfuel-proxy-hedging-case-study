from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    """Central configuration for reproducible pipeline execution."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    raw_filename: str = "jet_fuel_proxy_prices.csv"
    processed_filename: str = "market_data_with_returns.csv"
    hedged_returns_filename: str = "hedged_test_returns.csv"
    report_filename: str = "hedging_report.md"
    figure_filename: str = "cumulative_returns.png"
    n_synthetic_days: int = 756
    random_seed: int = 42
    train_ratio: float = 0.7
    transaction_cost_bps: float = 1.5

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def raw_data_path(self) -> Path:
        return self.raw_data_dir / self.raw_filename

    @property
    def processed_data_path(self) -> Path:
        return self.processed_data_dir / self.processed_filename

    @property
    def hedged_returns_path(self) -> Path:
        return self.processed_data_dir / self.hedged_returns_filename

    @property
    def report_path(self) -> Path:
        return self.reports_dir / self.report_filename

    @property
    def figure_path(self) -> Path:
        return self.figures_dir / self.figure_filename

    def validate(self) -> None:
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")
        if self.n_synthetic_days < 60:
            raise ValueError("n_synthetic_days must be at least 60 observations.")
        if self.transaction_cost_bps < 0.0:
            raise ValueError("transaction_cost_bps cannot be negative.")

    def ensure_directories(self) -> None:
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.reports_dir,
            self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
