from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_market_data(
    n_days: int = 756,
    seed: int = 42,
    start_date: str = "2021-01-01",
) -> pd.DataFrame:
    """Create correlated synthetic jet fuel and proxy hub price paths."""
    rng = np.random.default_rng(seed)

    mean_returns = np.array([0.00025, 0.00022, 0.00024], dtype=float)
    covariance = np.array(
        [
            [0.00018, 0.00014, 0.00015],
            [0.00014, 0.00020, 0.00016],
            [0.00015, 0.00016, 0.00021],
        ],
        dtype=float,
    )

    shocks = rng.multivariate_normal(mean=mean_returns, cov=covariance, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))

    dates = pd.bdate_range(start=start_date, periods=n_days)
    data = pd.DataFrame(
        {
            "date": dates,
            "jet_fuel_spot": prices[:, 0],
            "brent_proxy": prices[:, 1],
            "ulsd_proxy": prices[:, 2],
        }
    )
    return data
