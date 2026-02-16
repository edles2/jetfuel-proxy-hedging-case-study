"""Data loading, validation, and synthetic generation."""

from data.fred import download_fred_series
from data.illiquid_hub import IlliquidHubConfig, simulate_illiquid_hub
from data.io import PRICE_COLUMNS, load_market_data, save_market_data
from data.loaders import load_fred_market_prices, load_proxy_hedging_prices
from data.preprocess import (
    MissingDataPolicy,
    compute_log_returns,
    compute_simple_returns,
    preprocess_price_panel,
    preprocess_prices_and_returns,
)
from data.synthetic import generate_synthetic_market_data

__all__ = [
    "IlliquidHubConfig",
    "MissingDataPolicy",
    "PRICE_COLUMNS",
    "compute_log_returns",
    "compute_simple_returns",
    "download_fred_series",
    "generate_synthetic_market_data",
    "load_fred_market_prices",
    "load_market_data",
    "load_proxy_hedging_prices",
    "preprocess_price_panel",
    "preprocess_prices_and_returns",
    "save_market_data",
    "simulate_illiquid_hub",
]
