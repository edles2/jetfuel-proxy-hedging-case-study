from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class OLSModel:
    """Ordinary least squares model for estimating proxy hedge ratios."""

    intercept_: float = 0.0
    coef_: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "OLSModel":
        if features.empty:
            raise ValueError("Features must not be empty.")
        if len(features) != len(target):
            raise ValueError("Features and target must have the same number of rows.")

        x_matrix = np.column_stack(
            (np.ones(len(features)), features.to_numpy(dtype=float))
        )
        y_vector = target.to_numpy(dtype=float)

        beta, *_ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fit before prediction.")

        x_matrix = features.to_numpy(dtype=float)
        return self.intercept_ + x_matrix @ self.coef_

    def hedge_weights(self, feature_columns: Iterable[str]) -> dict[str, float]:
        if self.coef_ is None:
            raise RuntimeError("Model must be fit before extracting hedge weights.")

        names = list(feature_columns)
        if len(names) != len(self.coef_):
            raise ValueError(
                "feature_columns length does not match model coefficients."
            )

        return {name: float(weight) for name, weight in zip(names, self.coef_)}
