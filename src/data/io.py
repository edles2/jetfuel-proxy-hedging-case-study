from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICE_COLUMNS: tuple[str, ...] = ("jet_fuel_spot", "brent_proxy", "ulsd_proxy")
REQUIRED_COLUMNS: tuple[str, ...] = ("date", *PRICE_COLUMNS)


def validate_market_data(data: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {sorted(missing)}")


def load_market_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"])
    validate_market_data(data)
    data = (
        data.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    )
    return data


def save_market_data(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
