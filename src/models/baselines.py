from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def _validate_columns(
    train_data: pd.DataFrame,
    target_column: str,
    proxy_columns: Sequence[str],
) -> None:
    required = {target_column, *proxy_columns}
    missing = required.difference(train_data.columns)
    if missing:
        raise ValueError(f"train_data is missing required columns: {sorted(missing)}")


def _prepare_training_data(
    train_data: pd.DataFrame,
    target_column: str,
    proxy_columns: Sequence[str],
) -> pd.DataFrame:
    if train_data.empty:
        raise ValueError("train_data must not be empty.")
    if not proxy_columns:
        raise ValueError("proxy_columns must contain at least one proxy.")

    _validate_columns(
        train_data, target_column=target_column, proxy_columns=proxy_columns
    )

    prepared = (
        train_data[[target_column, *proxy_columns]]
        .copy()
        .sort_index()
        .dropna(axis=0, how="any")
    )
    if prepared.empty:
        raise ValueError("No rows remain after dropping NaNs in train_data.")
    return prepared


def _cap_series(values: pd.Series, cap_abs: float) -> pd.Series:
    if cap_abs <= 0.0:
        raise ValueError("max_abs_hedge_ratio must be positive.")
    return values.clip(lower=-cap_abs, upper=cap_abs)


def _condition_number(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")


def _select_n_splits(n_obs: int, desired_splits: int) -> int:
    if desired_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    n_splits = min(desired_splits, n_obs - 1)
    if n_splits < 2:
        raise ValueError("Not enough observations for time-series CV.")
    return n_splits


@dataclass(slots=True)
class StaticOLSHedgeEstimator:
    """Static OLS hedge estimator for single- or multi-proxy setups."""

    target_column: str = "jet_fuel_illiquid"
    proxy_columns: tuple[str, ...] = ("proxy_1",)
    max_abs_hedge_ratio: float = 3.0
    condition_number_warn_threshold: float = 1e8
    intercept_: float = field(init=False, default=0.0)
    hedge_ratios_: pd.Series | None = field(init=False, default=None)
    diagnostics_: dict[str, float] = field(init=False, default_factory=dict)
    fit_index_: pd.Index | None = field(init=False, default=None)

    def fit(self, train_data: pd.DataFrame) -> "StaticOLSHedgeEstimator":
        data = _prepare_training_data(
            train_data,
            target_column=self.target_column,
            proxy_columns=self.proxy_columns,
        )

        x = sm.add_constant(data[list(self.proxy_columns)], has_constant="add")
        y = data[self.target_column]

        cond = _condition_number(x.to_numpy(dtype=float))
        if cond > self.condition_number_warn_threshold:
            LOGGER.warning(
                "Static OLS condition number is high (%.2e); multicollinearity risk.",
                cond,
            )

        result = sm.OLS(y, x, missing="drop").fit()
        raw_ratios = result.params[list(self.proxy_columns)].astype(float)
        capped_ratios = _cap_series(raw_ratios, self.max_abs_hedge_ratio)
        if not capped_ratios.equals(raw_ratios):
            LOGGER.warning(
                "Static OLS hedge ratios were capped at +/- %.4f",
                self.max_abs_hedge_ratio,
            )

        self.intercept_ = float(result.params.get("const", 0.0))
        self.hedge_ratios_ = capped_ratios
        self.fit_index_ = data.index
        self.diagnostics_ = {
            "n_obs": float(result.nobs),
            "r_squared": float(result.rsquared),
            "adj_r_squared": float(result.rsquared_adj),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "condition_number": cond,
        }
        LOGGER.info("Static OLS fitted with diagnostics: %s", self.diagnostics_)
        return self

    def predict(self, x_t: pd.Series | pd.DataFrame) -> pd.Series:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before predict.")
        x_frame = pd.DataFrame(x_t).T if isinstance(x_t, pd.Series) else x_t.copy()
        aligned = x_frame.loc[:, list(self.proxy_columns)]
        prediction = self.intercept_ + aligned @ self.hedge_ratios_
        return prediction.rename("predicted_target_return")

    def predict_hedge_ratio(self, t: pd.Timestamp | None = None) -> pd.Series:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before predict_hedge_ratio.")
        return self.hedge_ratios_.copy()

    def hedge_ratio_time_series(self, index: pd.Index | None = None) -> pd.DataFrame:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before hedge_ratio_time_series.")
        series_index = index if index is not None else self.fit_index_
        if series_index is None:
            raise RuntimeError("No index available for hedge ratio time series.")
        values = np.tile(
            self.hedge_ratios_.to_numpy(dtype=float), (len(series_index), 1)
        )
        return pd.DataFrame(
            values, index=series_index, columns=self.hedge_ratios_.index
        )


@dataclass(slots=True)
class RollingOLSHedgeEstimator:
    """Rolling-window OLS hedge estimator with periodic re-fitting."""

    target_column: str = "jet_fuel_illiquid"
    proxy_columns: tuple[str, ...] = ("proxy_1",)
    window: int = 60
    refit_frequency: int = 1
    max_abs_hedge_ratio: float = 3.0
    condition_number_warn_threshold: float = 1e8
    hedge_ratios_: pd.DataFrame | None = field(init=False, default=None)
    intercepts_: pd.Series | None = field(init=False, default=None)
    diagnostics_: pd.DataFrame | None = field(init=False, default=None)

    def fit(self, train_data: pd.DataFrame) -> "RollingOLSHedgeEstimator":
        if self.window < 5:
            raise ValueError("window must be at least 5.")
        if self.refit_frequency < 1:
            raise ValueError("refit_frequency must be >= 1.")

        data = _prepare_training_data(
            train_data,
            target_column=self.target_column,
            proxy_columns=self.proxy_columns,
        )
        if len(data) <= self.window:
            raise ValueError("train_data must contain more rows than rolling window.")
        index = data.index
        ratios = pd.DataFrame(
            index=index, columns=list(self.proxy_columns), dtype=float
        )
        intercepts = pd.Series(index=index, dtype=float, name="intercept")
        diagnostics_rows: list[dict[str, Any]] = []

        active_ratio: pd.Series | None = None
        active_intercept = 0.0

        for i, date in enumerate(index):
            if i < self.window:
                continue

            should_refit = (
                active_ratio is None or (i - self.window) % self.refit_frequency == 0
            )
            if should_refit:
                train_slice = data.iloc[i - self.window : i]
                x_window = sm.add_constant(
                    train_slice[list(self.proxy_columns)], has_constant="add"
                )
                y_window = train_slice[self.target_column]

                cond = _condition_number(x_window.to_numpy(dtype=float))
                if cond > self.condition_number_warn_threshold:
                    LOGGER.warning(
                        "Rolling OLS high condition number at %s: %.2e",
                        date,
                        cond,
                    )

                result = sm.OLS(y_window, x_window, missing="drop").fit()
                raw_ratios = result.params[list(self.proxy_columns)].astype(float)
                active_ratio = _cap_series(raw_ratios, self.max_abs_hedge_ratio)
                active_intercept = float(result.params.get("const", 0.0))

                diagnostics_rows.append(
                    {
                        "refit_date": date,
                        "train_start": train_slice.index.min(),
                        "train_end": train_slice.index.max(),
                        "n_obs": float(result.nobs),
                        "r_squared": float(result.rsquared),
                        "adj_r_squared": float(result.rsquared_adj),
                        "condition_number": cond,
                    }
                )
                LOGGER.info(
                    "Rolling OLS refit at %s with r2=%.4f",
                    date,
                    float(result.rsquared),
                )

            ratios.loc[date] = active_ratio.to_numpy(dtype=float)
            intercepts.loc[date] = active_intercept

        self.hedge_ratios_ = ratios
        self.intercepts_ = intercepts
        diagnostics = pd.DataFrame(diagnostics_rows)
        self.diagnostics_ = (
            diagnostics.set_index("refit_date")
            if not diagnostics.empty
            else pd.DataFrame()
        )
        return self

    def predict(self, x_t: pd.Series | pd.DataFrame) -> pd.Series:
        if self.hedge_ratios_ is None or self.intercepts_ is None:
            raise RuntimeError("fit must be called before predict.")
        x_frame = pd.DataFrame(x_t).T if isinstance(x_t, pd.Series) else x_t.copy()
        aligned_x = x_frame.loc[:, list(self.proxy_columns)]

        ratio_panel = self.hedge_ratios_.reindex(aligned_x.index).ffill()
        intercept_panel = self.intercepts_.reindex(aligned_x.index).ffill()
        prediction = intercept_panel + (aligned_x * ratio_panel).sum(axis=1)
        return prediction.rename("predicted_target_return")

    def predict_hedge_ratio(self, t: pd.Timestamp) -> pd.Series:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before predict_hedge_ratio.")
        if t not in self.hedge_ratios_.index:
            raise KeyError(f"Timestamp {t} not present in hedge ratio index.")
        row = self.hedge_ratios_.loc[t]
        if row.isna().any():
            raise ValueError(f"Hedge ratio unavailable at {t} (insufficient lookback).")
        return row

    def hedge_ratio_time_series(self, index: pd.Index | None = None) -> pd.DataFrame:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before hedge_ratio_time_series.")
        if index is None:
            return self.hedge_ratios_.copy()
        return self.hedge_ratios_.reindex(index).ffill()


@dataclass(slots=True)
class _RegularizedBaseHedgeEstimator:
    """Base class for regularized multi-proxy hedge ratio estimators."""

    target_column: str = "jet_fuel_illiquid"
    proxy_columns: tuple[str, ...] = ("proxy_1", "proxy_2")
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    cv_splits: int = 5
    max_abs_hedge_ratio: float = 3.0
    condition_number_warn_threshold: float = 1e8
    intercept_: float = field(init=False, default=0.0)
    hedge_ratios_: pd.Series | None = field(init=False, default=None)
    diagnostics_: dict[str, float] = field(init=False, default_factory=dict)
    fit_index_: pd.Index | None = field(init=False, default=None)
    best_alpha_: float | None = field(init=False, default=None)

    def _model_type(self) -> Literal["ridge", "lasso"]:
        raise NotImplementedError

    def _build_pipeline(self) -> Pipeline:
        if self._model_type() == "ridge":
            model = Ridge()
        else:
            model = Lasso(max_iter=50_000)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    def fit(self, train_data: pd.DataFrame) -> "_RegularizedBaseHedgeEstimator":
        data = _prepare_training_data(
            train_data,
            target_column=self.target_column,
            proxy_columns=self.proxy_columns,
        )
        x = data[list(self.proxy_columns)].copy()
        y = data[self.target_column].copy()

        cond = _condition_number(x.to_numpy(dtype=float))
        if cond > self.condition_number_warn_threshold:
            LOGGER.warning(
                "%s condition number is high (%.2e); regularization mitigates instability.",
                self._model_type().capitalize(),
                cond,
            )

        n_splits = _select_n_splits(n_obs=len(data), desired_splits=self.cv_splits)
        pipeline = self._build_pipeline()
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid={"model__alpha": list(self.alphas)},
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring="neg_mean_squared_error",
            n_jobs=None,
        )
        grid.fit(x, y)

        best_pipeline = grid.best_estimator_
        scaler = best_pipeline.named_steps["scaler"]
        model = best_pipeline.named_steps["model"]

        coef_scaled = np.asarray(model.coef_, dtype=float)
        coef_unscaled = coef_scaled / scaler.scale_
        intercept_unscaled = float(
            model.intercept_ - np.sum(coef_scaled * scaler.mean_ / scaler.scale_)
        )

        raw_ratios = pd.Series(coef_unscaled, index=list(self.proxy_columns))
        capped_ratios = _cap_series(raw_ratios, self.max_abs_hedge_ratio)
        if not capped_ratios.equals(raw_ratios):
            LOGGER.warning(
                "%s hedge ratios were capped at +/- %.4f",
                self._model_type().capitalize(),
                self.max_abs_hedge_ratio,
            )

        self.intercept_ = intercept_unscaled
        self.hedge_ratios_ = capped_ratios
        self.fit_index_ = data.index
        self.best_alpha_ = float(grid.best_params_["model__alpha"])
        self.diagnostics_ = {
            "n_obs": float(len(data)),
            "best_alpha": self.best_alpha_,
            "best_cv_score": float(grid.best_score_),
            "condition_number": cond,
            "n_nonzero_coefs": float((np.abs(capped_ratios) > 0.0).sum()),
        }
        LOGGER.info(
            "%s fitted with diagnostics: %s",
            self._model_type().capitalize(),
            self.diagnostics_,
        )
        return self

    def predict(self, x_t: pd.Series | pd.DataFrame) -> pd.Series:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before predict.")
        x_frame = pd.DataFrame(x_t).T if isinstance(x_t, pd.Series) else x_t.copy()
        aligned = x_frame.loc[:, list(self.proxy_columns)]
        prediction = self.intercept_ + aligned @ self.hedge_ratios_
        return prediction.rename("predicted_target_return")

    def predict_hedge_ratio(self, t: pd.Timestamp | None = None) -> pd.Series:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before predict_hedge_ratio.")
        return self.hedge_ratios_.copy()

    def hedge_ratio_time_series(self, index: pd.Index | None = None) -> pd.DataFrame:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before hedge_ratio_time_series.")
        series_index = index if index is not None else self.fit_index_
        if series_index is None:
            raise RuntimeError("No index available for hedge ratio time series.")
        values = np.tile(
            self.hedge_ratios_.to_numpy(dtype=float), (len(series_index), 1)
        )
        return pd.DataFrame(
            values, index=series_index, columns=self.hedge_ratios_.index
        )


@dataclass(slots=True)
class RidgeHedgeEstimator(_RegularizedBaseHedgeEstimator):
    """Ridge regression hedge estimator with time-series cross-validation."""

    def _model_type(self) -> Literal["ridge", "lasso"]:
        return "ridge"


@dataclass(slots=True)
class LassoHedgeEstimator(_RegularizedBaseHedgeEstimator):
    """Lasso regression hedge estimator with time-series cross-validation."""

    def _model_type(self) -> Literal["ridge", "lasso"]:
        return "lasso"
