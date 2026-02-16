from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_experiments import run_experiments


def test_run_experiments_generates_manifest_tables_and_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    business_index = pd.bdate_range("2020-01-01", periods=1700)
    rng = np.random.default_rng(123)

    base_returns = {
        "DJFUELUSGULF": rng.normal(0.0002, 0.01, size=len(business_index)),
        "DDFUELUSGULF": rng.normal(0.00015, 0.011, size=len(business_index)),
        "DHOILNYH": rng.normal(0.00012, 0.012, size=len(business_index)),
        "DCOILBRENTEU": rng.normal(0.0001, 0.013, size=len(business_index)),
        "DCOILWTICO": rng.normal(0.0001, 0.013, size=len(business_index)),
        "DGASNYH": rng.normal(0.00018, 0.012, size=len(business_index)),
    }

    synthetic_prices = {
        sid: 100.0 * np.exp(np.cumsum(ret)) for sid, ret in base_returns.items()
    }

    def fake_download(
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        cache_dir=None,  # noqa: ANN001, ARG001
        refresh: bool = False,  # noqa: ARG001
        timeout_seconds: int = 30,  # noqa: ARG001
    ) -> pd.Series:
        series = pd.Series(
            synthetic_prices[series_id], index=business_index, name=series_id
        )
        if start_date is not None:
            series = series.loc[series.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            series = series.loc[series.index <= pd.Timestamp(end_date)]
        return series

    monkeypatch.setattr("data.loaders.download_fred_series", fake_download)

    config = {
        "data": {
            "frequency": "B",
            "max_ffill_gap_days": 3,
            "fred_cache_dir": "data/raw/fred",
            "fred": {
                "series_ids": {
                    "jet_fuel_benchmark": "DJFUELUSGULF",
                    "proxy_diesel_us_gulf": "DDFUELUSGULF",
                    "proxy_heating_oil_nyh": "DHOILNYH",
                    "proxy_brent_spot": "DCOILBRENTEU",
                    "proxy_wti_spot": "DCOILWTICO",
                    "proxy_gasoline_nyh": "DGASNYH",
                }
            },
        },
        "date_range": {
            "start": "2020-01-01",
            "end": "2025-12-31",
            "end_offset_days": 0,
        },
        "returns_type": "simple",
        "features": {
            "max_lag": 2,
            "rolling_windows": [10],
            "include_return_spreads": True,
            "include_price_spreads": True,
            "include_pca": False,
            "dropna": True,
        },
        "illiquid_hub": {
            "basis_ar_coeff": 0.98,
            "basis_process_sigma": 0.002,
            "regime_shift_probability": 0.01,
            "regime_shift_sigma": 0.01,
            "idiosyncratic_base_sigma": 0.002,
            "heteroskedastic_scale": 5.0,
            "missingness_probability": 0.0,
            "delayed_update_probability": 0.0,
            "delay_max_days": 2,
            "seed": 42,
        },
        "transaction_costs": {
            "default_scenario": "med",
            "scenarios": {
                "low": {"spread_bps": 0.5, "fixed_fee": 0.0},
                "med": {"spread_bps": 2.0, "fixed_fee": 0.00005},
                "high": {"spread_bps": 5.0, "fixed_fee": 0.0002},
            },
        },
        "hedging": {
            "rebalance_frequency": "daily",
            "exposure_notional": 1.0,
            "cost_scenario": "med",
            "constraints": {"max_abs_beta": 3.0, "leverage_cap": 5.0},
        },
        "models": {
            "single_static_ols": {"max_abs_hedge_ratio": 3.0},
            "multi_proxy_ridge": {"alphas": [0.001, 0.01, 0.1], "cv_splits": 3},
            "multi_proxy_kalman": {
                "calibrate": True,
                "process_noise_grid": [1e-5, 1e-4],
                "observation_noise_grid": [1e-4, 1e-3],
            },
        },
        "walk_forward": {
            "splits": [
                {
                    "name": "wf_1",
                    "train_start": "2020-01-02",
                    "train_end": "2022-12-30",
                    "val_start": "2023-01-03",
                    "val_end": "2023-12-29",
                    "test_start": "2024-01-02",
                    "test_end": "2025-12-31",
                }
            ]
        },
        "evaluation": {"rolling_window": 30, "proxy_diagnostic_window": 30},
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifest_path = run_experiments(
        config_path=config_path,
        project_root=tmp_path,
        refresh=False,
    )
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "git_commit_hash" in manifest
    assert "artifacts" in manifest

    reports_dir = tmp_path / "reports"
    assert (reports_dir / "tables" / "method_comparison.csv").exists()
    assert (reports_dir / "tables" / "walk_forward_splits.csv").exists()
    assert (reports_dir / "tables" / "feature_dataset.csv").exists()
    assert (reports_dir / "tables" / "rolling_correlations.csv").exists()
    assert (reports_dir / "tables" / "beta_stability.csv").exists()
    assert (reports_dir / "tables" / "artifacts_manifest.json").exists()
    assert any((reports_dir / "figures").glob("*.png"))
