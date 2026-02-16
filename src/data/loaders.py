from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from data.fred import download_fred_series

LOGGER = logging.getLogger(__name__)


def load_price_series(
    csv_path: Path,
    instrument_name: str,
    date_column: str = "Date",
    price_column: str = "Price",
) -> pd.Series:
    """Load one instrument CSV into a date-indexed price series.

    This legacy helper is kept for local-file workflows and tests.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Price file not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    missing_columns = {date_column, price_column}.difference(raw.columns)
    if missing_columns:
        raise ValueError(
            f"File {csv_path} is missing columns: {sorted(missing_columns)}"
        )

    series_frame = raw[[date_column, price_column]].copy()
    series_frame[date_column] = pd.to_datetime(
        series_frame[date_column], errors="coerce"
    )
    if series_frame[date_column].isna().any():
        raise ValueError(f"File {csv_path} contains invalid dates in {date_column}.")

    series_frame[price_column] = pd.to_numeric(
        series_frame[price_column], errors="coerce"
    )

    series_frame = (
        series_frame.sort_values(date_column)
        .drop_duplicates(subset=[date_column], keep="last")
        .set_index(date_column)
    )
    series = series_frame[price_column].rename(instrument_name)
    return series


def load_proxy_hedging_prices(
    raw_dir: Path,
    target_file: str,
    proxy_files: Mapping[str, str],
    target_name: str = "jet_fuel_illiquid",
    date_column: str = "Date",
    price_column: str = "Price",
) -> pd.DataFrame:
    """Load target + proxy CSVs and return one aligned wide price DataFrame.

    This local-file loader is preserved for compatibility.
    """
    files_by_instrument: dict[str, str] = {
        target_name: target_file,
        **dict(proxy_files),
    }

    series_list: list[pd.Series] = []
    for instrument_name, filename in files_by_instrument.items():
        csv_path = raw_dir / filename
        series = load_price_series(
            csv_path=csv_path,
            instrument_name=instrument_name,
            date_column=date_column,
            price_column=price_column,
        )
        series_list.append(series)

    aligned_prices = pd.concat(series_list, axis=1, join="outer").sort_index()
    return aligned_prices


def load_fred_market_prices(
    series_ids: Mapping[str, str],
    start_date: str,
    end_date: str,
    frequency: str = "B",
    max_ffill_gap_days: int = 3,
    cache_dir: Path = Path("data/raw/fred"),
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch target benchmark + proxy prices from FRED, align, and clean.

    Output columns are the keys of `series_ids` (e.g., `jet_fuel_benchmark`,
    `proxy_diesel`, `proxy_brent`, ...).

    Alignment policy:
    1) outer join on dates,
    2) business-day reindex over [start_date, end_date],
    3) forward-fill up to `max_ffill_gap_days`,
    4) drop rows with remaining missing values.
    """
    if not series_ids:
        raise ValueError("series_ids must not be empty.")
    if max_ffill_gap_days < 0:
        raise ValueError("max_ffill_gap_days must be non-negative.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date must be greater than or equal to start_date.")

    series_list: list[pd.Series] = []
    for output_name, series_id in series_ids.items():
        fetched = download_fred_series(
            series_id=str(series_id),
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        series_list.append(fetched.rename(str(output_name)))

    merged = pd.concat(series_list, axis=1, join="outer").sort_index()
    business_calendar = pd.date_range(start=start_ts, end=end_ts, freq=frequency)
    aligned = merged.reindex(business_calendar)
    aligned = aligned.ffill(limit=max_ffill_gap_days)
    cleaned = aligned.dropna(axis=0, how="any")

    if cleaned.empty:
        raise ValueError(
            "No usable rows after alignment and missing-data handling. "
            "Increase max_ffill_gap_days or shorten date range."
        )

    LOGGER.info(
        "Loaded FRED market panel with %d rows and %d columns.",
        len(cleaned),
        cleaned.shape[1],
    )
    return cleaned
