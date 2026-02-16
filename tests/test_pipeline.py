from __future__ import annotations

from pathlib import Path

from config.settings import PipelineConfig
from pipeline import run_pipeline


def test_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    config = PipelineConfig(
        project_root=tmp_path,
        n_synthetic_days=260,
        train_ratio=0.75,
        transaction_cost_bps=1.0,
    )

    metrics = run_pipeline(config)

    assert "variance_reduction" in metrics
    assert config.raw_data_path.exists()
    assert config.processed_data_path.exists()
    assert config.hedged_returns_path.exists()
    assert config.report_path.exists()
    assert config.figure_path.exists()
