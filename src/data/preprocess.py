from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MissingDataPolicy:
    """Policy for handling missing prices after calendar alignment.

    Supported policy:
    - method=`ffill_then_drop`: forward-fill only up to `max_forward_fill` rows
      per gap, then drop rows that still contain missing prices.
    """

    method: str = "ffill_then_drop"
    max_forward_fill: int = 2


def parse_and_sort_price_index(prices: pd.DataFrame) -> pd.DataFrame:
    """Parse index as datetime, sort, and de-duplicate dates.

    Assumptions:
    - Input has one column per instrument and date-like index labels.
    - If duplicate dates exist, the last observation is kept.
    """
    parsed = prices.copy()

    if not isinstance(parsed.index, pd.DatetimeIndex):
        parsed.index = pd.to_datetime(parsed.index, errors="coerce")

    if parsed.index.isna().any():
        raise ValueError("Price index contains invalid or unparsable dates.")

    parsed = parsed.sort_index()
    parsed = parsed[~parsed.index.duplicated(keep="last")]
    parsed = parsed.apply(pd.to_numeric, errors="coerce")
    return parsed


def align_calendars(
    prices: pd.DataFrame,
    frequency: str = "B",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Align all instruments to one common calendar index."""
    if prices.empty:
        raise ValueError("Cannot align an empty price DataFrame.")

    start = pd.Timestamp(start_date) if start_date is not None else prices.index.min()
    end = pd.Timestamp(end_date) if end_date is not None else prices.index.max()
    if start > end:
        raise ValueError("start_date must be before or equal to end_date.")

    common_calendar = pd.date_range(start=start, end=end, freq=frequency)
    return prices.reindex(common_calendar)


def apply_missing_data_policy(
    prices: pd.DataFrame,
    policy: MissingDataPolicy,
) -> pd.DataFrame:
    """Apply missing-data cleaning policy to aligned prices."""
    if policy.method != "ffill_then_drop":
        raise ValueError(f"Unsupported missing data policy method: {policy.method}")
    if policy.max_forward_fill < 0:
        raise ValueError("max_forward_fill must be non-negative.")

    filled = prices.ffill(limit=policy.max_forward_fill)
    cleaned = filled.dropna(axis=0, how="any")
    return cleaned


def preprocess_price_panel(
    prices: pd.DataFrame,
    frequency: str = "B",
    start_date: str | None = None,
    end_date: str | None = None,
    missing_data_policy: MissingDataPolicy | None = None,
) -> pd.DataFrame:
    """Run the full preprocessing workflow for price panels."""
    policy = missing_data_policy or MissingDataPolicy()

    parsed = parse_and_sort_price_index(prices)
    aligned = align_calendars(
        parsed,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
    )
    cleaned = apply_missing_data_policy(aligned, policy)
    if cleaned.empty:
        raise ValueError("No rows remain after preprocessing.")
    return cleaned


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns, preserving index and instrument columns."""
    return prices.pct_change()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns, preserving index and instrument columns."""
    return np.log(prices / prices.shift(1))


def preprocess_prices_and_returns(
    prices: pd.DataFrame,
    frequency: str = "B",
    start_date: str | None = None,
    end_date: str | None = None,
    missing_data_policy: MissingDataPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess prices and compute both simple and log return panels.

    Returns:
    - cleaned prices
    - simple returns
    - log returns
    """
    cleaned_prices = preprocess_price_panel(
        prices=prices,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        missing_data_policy=missing_data_policy,
    )
    simple_returns = compute_simple_returns(cleaned_prices)
    log_returns = compute_log_returns(cleaned_prices)
    return cleaned_prices, simple_returns, log_returns
