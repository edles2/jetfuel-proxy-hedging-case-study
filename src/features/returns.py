from __future__ import annotations

from typing import Iterable

import pandas as pd

from data.io import PRICE_COLUMNS


def add_returns(
    data: pd.DataFrame,
    price_columns: Iterable[str] = PRICE_COLUMNS,
) -> pd.DataFrame:
    """Add simple daily returns columns for each price series."""
    enriched = data.copy()

    for column in price_columns:
        enriched[f"{column}_ret"] = enriched[column].pct_change()

    enriched = enriched.dropna().reset_index(drop=True)
    return enriched


def build_feature_target(
    data_with_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = ["brent_proxy_ret", "ulsd_proxy_ret"]
    target_column = "jet_fuel_spot_ret"

    missing = set(feature_columns + [target_column]).difference(
        data_with_returns.columns
    )
    if missing:
        raise ValueError(
            f"Missing return columns required for modeling: {sorted(missing)}"
        )

    features = data_with_returns[feature_columns].copy()
    target = data_with_returns[target_column].copy()
    return features, target
