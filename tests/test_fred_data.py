from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.fred import download_fred_series
from data.loaders import load_fred_market_prices
from data.preprocess import MissingDataPolicy, preprocess_prices_and_returns


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_fred_download_uses_cache_when_available(monkeypatch, tmp_path: Path) -> None:
    payload = "DATE,VALUE\n2024-01-01,1.0\n2024-01-02,1.1\n"
    call_count = {"n": 0}

    def fake_get(url: str, timeout: int):  # noqa: ARG001
        call_count["n"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr("data.fred.requests.get", fake_get)
    cache_dir = tmp_path / "fred"

    series_first = download_fred_series(
        "TESTSERIES",
        start_date="2024-01-01",
        end_date="2024-01-31",
        cache_dir=cache_dir,
        refresh=False,
    )
    assert call_count["n"] == 1
    assert (cache_dir / "TESTSERIES.csv").exists()

    def fail_get(url: str, timeout: int):  # noqa: ARG001
        raise AssertionError("Network call should not happen when cache is reused.")

    monkeypatch.setattr("data.fred.requests.get", fail_get)
    series_second = download_fred_series(
        "TESTSERIES",
        start_date="2024-01-01",
        end_date="2024-01-31",
        cache_dir=cache_dir,
        refresh=False,
    )

    pd.testing.assert_series_equal(series_first, series_second)


def test_fred_alignment_and_preprocess_no_nans_except_first_return(
    monkeypatch,
) -> None:
    def fake_download(
        series_id: str,
        start_date: str | None = None,  # noqa: ARG001
        end_date: str | None = None,  # noqa: ARG001
        cache_dir=None,  # noqa: ANN001, ARG001
        refresh: bool = False,  # noqa: ARG001
    ) -> pd.Series:
        index = pd.to_datetime(
            ["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-08", "2024-01-09"]
        )
        base = {
            "JET": [10.0, 10.2, 10.1, 10.3, 10.4],
            "P1": [20.0, 20.1, 20.3, 20.4, 20.5],
            "P2": [30.0, 30.2, 30.0, 30.3, 30.4],
        }[series_id]
        return pd.Series(base, index=index, name=series_id)

    monkeypatch.setattr("data.loaders.download_fred_series", fake_download)
    prices = load_fred_market_prices(
        series_ids={
            "jet_fuel_benchmark": "JET",
            "proxy_one": "P1",
            "proxy_two": "P2",
        },
        start_date="2024-01-01",
        end_date="2024-01-10",
        frequency="B",
        max_ffill_gap_days=2,
        refresh=False,
    )
    assert not prices.isna().any().any()

    cleaned, simple_returns, log_returns = preprocess_prices_and_returns(
        prices=prices,
        frequency="B",
        start_date="2024-01-01",
        end_date="2024-01-10",
        missing_data_policy=MissingDataPolicy(
            method="ffill_then_drop", max_forward_fill=2
        ),
    )
    assert not cleaned.isna().any().any()
    assert simple_returns.iloc[0].isna().all()
    assert log_returns.iloc[0].isna().all()
    assert simple_returns.iloc[1:].notna().all().all()
    assert log_returns.iloc[1:].notna().all().all()
