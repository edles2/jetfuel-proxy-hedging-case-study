from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.loaders import load_proxy_hedging_prices
from data.preprocess import (
    MissingDataPolicy,
    preprocess_price_panel,
    preprocess_prices_and_returns,
)


def _write_price_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_proxy_hedging_prices_aligns_to_union_calendar(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    _write_price_csv(
        raw_dir / "jet_fuel_illiquid.csv",
        [
            {"Date": "2024-01-01", "Price": 100.0},
            {"Date": "2024-01-02", "Price": 101.0},
            {"Date": "2024-01-04", "Price": 103.0},
        ],
    )
    _write_price_csv(
        raw_dir / "proxy_1.csv",
        [
            {"Date": "2024-01-01", "Price": 50.0},
            {"Date": "2024-01-03", "Price": 51.0},
            {"Date": "2024-01-04", "Price": 52.0},
        ],
    )
    _write_price_csv(
        raw_dir / "proxy_2.csv",
        [
            {"Date": "2024-01-02", "Price": 70.0},
            {"Date": "2024-01-03", "Price": 71.0},
            {"Date": "2024-01-04", "Price": 72.0},
        ],
    )

    prices = load_proxy_hedging_prices(
        raw_dir=raw_dir,
        target_file="jet_fuel_illiquid.csv",
        proxy_files={"proxy_1": "proxy_1.csv", "proxy_2": "proxy_2.csv"},
    )

    expected_index = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    )
    assert list(prices.columns) == ["jet_fuel_illiquid", "proxy_1", "proxy_2"]
    assert prices.index.equals(expected_index)


def test_preprocessing_removes_nans_and_returns_only_first_row_nan() -> None:
    index = pd.bdate_range("2024-01-01", periods=7)
    raw_prices = pd.DataFrame(
        {
            "jet_fuel_illiquid": [100.0, 101.0, None, 103.0, None, None, 105.0],
            "proxy_1": [50.0, 50.2, 50.4, 50.6, 50.7, 50.9, 51.1],
            "proxy_2": [70.0, None, 70.5, None, None, 71.1, 71.2],
        },
        index=index,
    )

    policy = MissingDataPolicy(method="ffill_then_drop", max_forward_fill=1)
    cleaned = preprocess_price_panel(
        raw_prices,
        frequency="B",
        missing_data_policy=policy,
    )

    assert not cleaned.isna().any().any()

    _, simple_returns, log_returns = preprocess_prices_and_returns(
        raw_prices,
        frequency="B",
        missing_data_policy=policy,
    )
    assert simple_returns.iloc[0].isna().all()
    assert log_returns.iloc[0].isna().all()
    assert simple_returns.iloc[1:].notna().all().all()
    assert log_returns.iloc[1:].notna().all().all()
