from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Configuration for proxy-hedging feature engineering.

    Assumptions:
    - `prices` and `returns` each contain one column for the target and one or more
      columns for proxies, with consistent instrument names.
    - Feature values at date t are built only from information available up to t-1.
    """

    target_column: str = "jet_fuel_illiquid"
    proxy_columns: tuple[str, ...] = ("proxy_1", "proxy_2")
    max_lag: int = 3
    rolling_windows: tuple[int, ...] = (20, 60)
    include_return_spreads: bool = True
    include_price_spreads: bool = True
    include_pca: bool = False
    pca_components: int = 2
    pca_min_history: int = 60
    dropna: bool = True


def _validate_inputs(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    target_column: str,
    proxy_columns: Sequence[str],
    max_lag: int,
    rolling_windows: Sequence[int],
) -> None:
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1.")
    if not proxy_columns:
        raise ValueError("proxy_columns must contain at least one proxy series.")
    if any(window < 2 for window in rolling_windows):
        raise ValueError("rolling_windows entries must be at least 2.")

    required_columns = {target_column, *proxy_columns}
    missing_prices = required_columns.difference(prices.columns)
    missing_returns = required_columns.difference(returns.columns)
    if missing_prices:
        raise ValueError(
            f"prices is missing required columns: {sorted(missing_prices)}"
        )
    if missing_returns:
        raise ValueError(
            f"returns is missing required columns: {sorted(missing_returns)}"
        )


def _causal_pca_features(
    proxy_returns: pd.DataFrame,
    n_components: int,
    min_history: int,
) -> pd.DataFrame:
    """Build causal PCA features.

    For row t, PCA is fit on proxy returns up to t-1 and applied to the t-1
    observation. This avoids look-ahead leakage.
    """
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")

    n_components = min(n_components, proxy_returns.shape[1])
    columns = [f"proxy_pca_{idx + 1}" for idx in range(n_components)]
    factors = pd.DataFrame(index=proxy_returns.index, columns=columns, dtype=float)

    for row_idx in range(len(proxy_returns)):
        history = proxy_returns.iloc[:row_idx].dropna()
        if len(history) < min_history:
            continue

        current_available = proxy_returns.iloc[row_idx - 1]
        if current_available.isna().any():
            continue

        history_values = history.to_numpy(dtype=float)
        mean_vector = history_values.mean(axis=0)
        centered_history = history_values - mean_vector

        covariance = np.cov(centered_history, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        order = np.argsort(eigvals)[::-1]
        principal_axes = eigvecs[:, order[:n_components]]

        projected = (
            current_available.to_numpy(dtype=float) - mean_vector
        ) @ principal_axes
        factors.iloc[row_idx] = projected

    return factors


def _build_feature_frame(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    target_column: str,
    proxy_columns: Sequence[str],
    max_lag: int,
    rolling_windows: Sequence[int],
    include_return_spreads: bool,
    include_price_spreads: bool,
    include_pca: bool,
    pca_components: int,
    pca_min_history: int,
) -> pd.DataFrame:
    feature_map: dict[str, pd.Series] = {}

    for proxy in proxy_columns:
        for lag in range(1, max_lag + 1):
            feature_map[f"{proxy}_ret_lag_{lag}"] = returns[proxy].shift(lag)

        for window in rolling_windows:
            corr = returns[target_column].rolling(window=window).corr(returns[proxy])
            feature_map[f"roll_corr_{target_column}_{proxy}_w{window}"] = corr.shift(1)

    if len(proxy_columns) > 1:
        for left_proxy, right_proxy in combinations(proxy_columns, 2):
            if include_return_spreads:
                feature_map[f"ret_spread_{left_proxy}_{right_proxy}"] = (
                    returns[left_proxy] - returns[right_proxy]
                ).shift(1)

            if include_price_spreads:
                feature_map[f"price_spread_{left_proxy}_{right_proxy}"] = (
                    prices[left_proxy] - prices[right_proxy]
                ).shift(1)

    features = pd.DataFrame(feature_map, index=returns.index)

    if include_pca:
        pca_features = _causal_pca_features(
            proxy_returns=returns[list(proxy_columns)],
            n_components=pca_components,
            min_history=pca_min_history,
        )
        features = pd.concat([features, pca_features], axis=1)

    return features


def build_dataset(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Build an aligned feature matrix X and target y for proxy hedging.

    No-lookahead rule:
    - y at t is `returns[target_column]` at t.
    - every feature at t uses data observed no later than t-1.
    """
    cfg = config or FeatureConfig()
    _validate_inputs(
        prices=prices,
        returns=returns,
        target_column=cfg.target_column,
        proxy_columns=cfg.proxy_columns,
        max_lag=cfg.max_lag,
        rolling_windows=cfg.rolling_windows,
    )

    aligned_index = prices.index.intersection(returns.index).sort_values()
    aligned_prices = prices.loc[aligned_index].sort_index()
    aligned_returns = returns.loc[aligned_index].sort_index()

    features = _build_feature_frame(
        prices=aligned_prices,
        returns=aligned_returns,
        target_column=cfg.target_column,
        proxy_columns=cfg.proxy_columns,
        max_lag=cfg.max_lag,
        rolling_windows=cfg.rolling_windows,
        include_return_spreads=cfg.include_return_spreads,
        include_price_spreads=cfg.include_price_spreads,
        include_pca=cfg.include_pca,
        pca_components=cfg.pca_components,
        pca_min_history=cfg.pca_min_history,
    )
    target = aligned_returns[cfg.target_column].copy()

    if cfg.dropna:
        valid_rows = features.notna().all(axis=1) & target.notna()
        features = features.loc[valid_rows]
        target = target.loc[valid_rows]

    return features, target, features.index
