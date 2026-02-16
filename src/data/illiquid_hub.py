from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class IlliquidHubConfig:
    """Parameters for synthetic illiquid-hub construction.

    Model:
    - Benchmark log-price: log(P_benchmark,t)
    - Basis: b_t = phi * b_{t-1} + u_t + regime_jump_t
    - Idiosyncratic noise: e_t ~ N(0, sigma_t^2), with heteroskedastic sigma_t
    - True illiquid log-price: log(P_true,t) = log(P_benchmark,t) + b_t + e_t

    Optional microstructure effects:
    - missingness_probability: stochastic missing observations
    - delayed_update_probability: stale quotes using past observed prices
    """

    basis_ar_coeff: float = 0.98
    basis_process_sigma: float = 0.002
    regime_shift_probability: float = 0.01
    regime_shift_sigma: float = 0.02
    idiosyncratic_base_sigma: float = 0.003
    heteroskedastic_scale: float = 8.0
    missingness_probability: float = 0.0
    delayed_update_probability: float = 0.0
    delay_max_days: int = 3
    seed: int = 42

    def validate(self) -> None:
        if not 0.0 <= self.basis_ar_coeff <= 1.0:
            raise ValueError("basis_ar_coeff must be between 0 and 1.")
        if self.basis_process_sigma < 0.0:
            raise ValueError("basis_process_sigma must be non-negative.")
        if not 0.0 <= self.regime_shift_probability <= 1.0:
            raise ValueError("regime_shift_probability must be in [0, 1].")
        if self.regime_shift_sigma < 0.0:
            raise ValueError("regime_shift_sigma must be non-negative.")
        if self.idiosyncratic_base_sigma < 0.0:
            raise ValueError("idiosyncratic_base_sigma must be non-negative.")
        if self.heteroskedastic_scale < 0.0:
            raise ValueError("heteroskedastic_scale must be non-negative.")
        if not 0.0 <= self.missingness_probability <= 1.0:
            raise ValueError("missingness_probability must be in [0, 1].")
        if not 0.0 <= self.delayed_update_probability <= 1.0:
            raise ValueError("delayed_update_probability must be in [0, 1].")
        if self.delay_max_days < 1:
            raise ValueError("delay_max_days must be at least 1.")


def simulate_illiquid_hub(
    benchmark_prices: pd.Series,
    config: IlliquidHubConfig,
) -> pd.DataFrame:
    """Generate synthetic illiquid-hub price/return series from benchmark jet fuel.

    The exposure to hedge is `illiquid_return` from `illiquid_price` (observed series).
    """
    config.validate()
    if benchmark_prices.empty:
        raise ValueError("benchmark_prices must not be empty.")

    benchmark = benchmark_prices.sort_index().astype(float).copy()
    benchmark.name = benchmark.name or "jet_fuel_benchmark"
    if benchmark.isna().any():
        raise ValueError("benchmark_prices must not contain NaN values.")

    rng = np.random.default_rng(config.seed)
    n_obs = len(benchmark)

    benchmark_ret = benchmark.pct_change().fillna(0.0).to_numpy(dtype=float)
    benchmark_log = np.log(benchmark.to_numpy(dtype=float))

    basis = np.zeros(n_obs, dtype=float)
    idio_noise = np.zeros(n_obs, dtype=float)

    for t in range(1, n_obs):
        regime_jump = 0.0
        if rng.random() < config.regime_shift_probability:
            regime_jump = rng.normal(0.0, config.regime_shift_sigma)

        basis_innovation = rng.normal(0.0, config.basis_process_sigma)
        basis[t] = config.basis_ar_coeff * basis[t - 1] + basis_innovation + regime_jump

        local_sigma = config.idiosyncratic_base_sigma * (
            1.0 + config.heteroskedastic_scale * abs(benchmark_ret[t - 1])
        )
        idio_noise[t] = rng.normal(0.0, local_sigma)

    illiquid_log_true = benchmark_log + basis + idio_noise
    illiquid_true = np.exp(illiquid_log_true)

    illiquid_observed = illiquid_true.copy()
    for t in range(1, n_obs):
        if rng.random() < config.missingness_probability:
            illiquid_observed[t] = np.nan
            continue
        if rng.random() < config.delayed_update_probability:
            lag = int(rng.integers(1, min(config.delay_max_days, t) + 1))
            illiquid_observed[t] = illiquid_observed[t - lag]

    output = pd.DataFrame(
        {
            "jet_fuel_benchmark": benchmark.to_numpy(dtype=float),
            "illiquid_price_true": illiquid_true,
            "illiquid_price": illiquid_observed,
            "basis_component": basis,
            "idiosyncratic_noise": idio_noise,
        },
        index=benchmark.index,
    )
    output["illiquid_return"] = output["illiquid_price"].pct_change()
    return output
