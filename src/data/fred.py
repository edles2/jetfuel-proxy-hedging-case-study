from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _parse_fred_csv(raw_csv: str, series_id: str) -> pd.Series:
    frame = pd.read_csv(StringIO(raw_csv), na_values=["."])
    if frame.shape[1] < 2:
        raise ValueError(f"Unexpected FRED CSV format for {series_id}.")

    date_col = frame.columns[0]
    value_col = "VALUE" if "VALUE" in frame.columns else frame.columns[1]

    parsed = frame[[date_col, value_col]].copy()
    parsed[date_col] = pd.to_datetime(parsed[date_col], errors="coerce")
    parsed[value_col] = pd.to_numeric(parsed[value_col], errors="coerce")
    parsed = parsed.dropna(subset=[date_col])
    parsed = parsed.sort_values(date_col).drop_duplicates(
        subset=[date_col], keep="last"
    )
    parsed = parsed.set_index(date_col)

    series = parsed[value_col].rename(series_id)
    return series


def download_fred_series(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_dir: Path = Path("data/raw/fred"),
    refresh: bool = False,
    timeout_seconds: int = 30,
) -> pd.Series:
    """Download a FRED series via CSV endpoint, with local raw-cache reuse.

    Endpoint pattern:
    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{series_id}.csv"

    if cache_path.exists() and not refresh:
        LOGGER.info("Using cached FRED series %s from %s", series_id, cache_path)
        raw_csv = cache_path.read_text(encoding="utf-8")
    else:
        url = FRED_CSV_URL.format(series_id=series_id)
        LOGGER.info("Downloading FRED series %s from %s", series_id, url)
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            if cache_path.exists():
                LOGGER.warning(
                    "Download failed for %s (%s). Falling back to cached file.",
                    series_id,
                    exc,
                )
                raw_csv = cache_path.read_text(encoding="utf-8")
            else:
                raise RuntimeError(
                    f"Failed to download FRED series {series_id} and no cache is available."
                ) from exc
        else:
            raw_csv = response.text
            cache_path.write_text(raw_csv, encoding="utf-8")

    series = _parse_fred_csv(raw_csv=raw_csv, series_id=series_id)
    if start_date is not None:
        series = series.loc[series.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        series = series.loc[series.index <= pd.Timestamp(end_date)]
    return series
