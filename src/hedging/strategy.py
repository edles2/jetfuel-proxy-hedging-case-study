from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def apply_static_proxy_hedge(
    target_returns: pd.Series,
    proxy_returns: pd.DataFrame,
    hedge_weights: Mapping[str, float],
) -> pd.DataFrame:
    """Apply fixed hedge ratios to proxy return series."""
    weight_columns = list(hedge_weights.keys())
    aligned_proxy = proxy_returns.loc[target_returns.index, weight_columns]

    weight_vector = np.array(
        [hedge_weights[col] for col in weight_columns], dtype=float
    )
    hedge_leg = aligned_proxy.to_numpy(dtype=float) @ weight_vector
    unhedged = target_returns.to_numpy(dtype=float)
    hedged_before_costs = unhedged - hedge_leg

    return pd.DataFrame(
        {
            "unhedged_return": unhedged,
            "hedge_leg_return": hedge_leg,
            "hedged_return_before_costs": hedged_before_costs,
        },
        index=target_returns.index,
    )
