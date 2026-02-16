from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _prepare_panel(
    data: pd.DataFrame,
    target_column: str,
    proxy_columns: Sequence[str],
) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Input data must not be empty.")
    if not proxy_columns:
        raise ValueError("proxy_columns must contain at least one proxy.")

    required = {target_column, *proxy_columns}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    panel = (
        data[[target_column, *proxy_columns]]
        .copy()
        .sort_index()
        .dropna(axis=0, how="any")
    )
    if panel.empty:
        raise ValueError("No rows remain after dropping missing values.")
    return panel


def _clip_vector(vector: np.ndarray, max_abs: float) -> np.ndarray:
    if max_abs <= 0.0:
        raise ValueError("max_abs_hedge_ratio must be positive.")
    return np.clip(vector, -max_abs, max_abs)


@dataclass(slots=True)
class KalmanHedgeEstimator:
    """Kalman-filter hedge ratio estimator for state-space regression.

    Equations:
    - Observation: y_t = x_t' * beta_t + eps_t, eps_t ~ N(0, R)
    - State:       beta_t = beta_{t-1} + eta_t, eta_t ~ N(0, Q)

    Assumptions:
    - `y_t` is target return at t.
    - `x_t` are proxy returns at t.
    - `beta_t` follows a random walk (time-varying hedge ratios).
    - No-lookahead hedge ratio at t uses beta_{t|t-1} (prior before observing y_t).
    """

    target_column: str = "jet_fuel_illiquid"
    proxy_columns: tuple[str, ...] = ("proxy_1",)
    process_noise: float = 1e-4
    observation_noise: float = 1e-3
    initial_covariance_scale: float = 1.0
    max_abs_hedge_ratio: float = 3.0
    numerical_epsilon: float = 1e-10
    process_noise_grid: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3)
    observation_noise_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    hedge_ratios_: pd.DataFrame | None = field(init=False, default=None)
    filtered_hedge_ratios_: pd.DataFrame | None = field(init=False, default=None)
    predicted_target_: pd.Series | None = field(init=False, default=None)
    diagnostics_: dict[str, float] = field(init=False, default_factory=dict)
    calibration_results_: pd.DataFrame | None = field(init=False, default=None)
    fit_index_: pd.Index | None = field(init=False, default=None)
    final_state_mean_: np.ndarray | None = field(init=False, default=None)
    final_state_cov_: np.ndarray | None = field(init=False, default=None)

    def _validate_noises(self, process_noise: float, observation_noise: float) -> None:
        if process_noise <= 0.0:
            raise ValueError("process_noise must be strictly positive.")
        if observation_noise <= 0.0:
            raise ValueError("observation_noise must be strictly positive.")
        if self.initial_covariance_scale <= 0.0:
            raise ValueError("initial_covariance_scale must be strictly positive.")

    def _run_filter(
        self,
        panel: pd.DataFrame,
        process_noise: float,
        observation_noise: float,
        initial_mean: np.ndarray | None = None,
        initial_cov: np.ndarray | None = None,
    ) -> dict[str, Any]:
        self._validate_noises(
            process_noise=process_noise,
            observation_noise=observation_noise,
        )

        proxies = list(self.proxy_columns)
        x = panel[proxies].to_numpy(dtype=float)
        y = panel[self.target_column].to_numpy(dtype=float)

        n_obs, n_features = x.shape
        identity = np.eye(n_features)

        q = process_noise * identity
        r = float(observation_noise)

        beta_post = (
            np.zeros(n_features, dtype=float)
            if initial_mean is None
            else initial_mean.astype(float).copy()
        )
        cov_post = (
            self.initial_covariance_scale * identity
            if initial_cov is None
            else initial_cov.astype(float).copy()
        )

        beta_prior_path = np.zeros((n_obs, n_features), dtype=float)
        beta_post_path = np.zeros((n_obs, n_features), dtype=float)
        one_step_predictions = np.zeros(n_obs, dtype=float)
        innovations = np.zeros(n_obs, dtype=float)

        for i in range(n_obs):
            beta_prior = beta_post.copy()
            cov_prior = cov_post + q

            x_t = x[i]
            y_pred = float(x_t @ beta_prior)
            innovation = y[i] - y_pred

            s_t = float(x_t @ cov_prior @ x_t + r + self.numerical_epsilon)
            gain = (cov_prior @ x_t) / s_t

            beta_post = beta_prior + gain * innovation
            beta_post = _clip_vector(beta_post, self.max_abs_hedge_ratio)
            cov_post = (identity - np.outer(gain, x_t)) @ cov_prior
            cov_post = 0.5 * (cov_post + cov_post.T)

            beta_prior_path[i] = _clip_vector(beta_prior, self.max_abs_hedge_ratio)
            beta_post_path[i] = beta_post
            one_step_predictions[i] = y_pred
            innovations[i] = innovation

        return {
            "beta_prior_path": beta_prior_path,
            "beta_post_path": beta_post_path,
            "predictions": one_step_predictions,
            "innovations": innovations,
            "final_beta": beta_post,
            "final_cov": cov_post,
        }

    def calibrate(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
    ) -> tuple[float, float]:
        """Grid-search process/observation noise on train/validation split.

        For each (Q, R) pair:
        1. filter on train
        2. continue filtering on validation
        3. score using validation one-step-ahead MSE
        """
        train_panel = _prepare_panel(
            train_data,
            target_column=self.target_column,
            proxy_columns=self.proxy_columns,
        )
        validation_panel = _prepare_panel(
            validation_data,
            target_column=self.target_column,
            proxy_columns=self.proxy_columns,
        )

        rows: list[dict[str, float]] = []
        best_process = self.process_noise
        best_observation = self.observation_noise
        best_mse = float("inf")

        for process_noise in self.process_noise_grid:
            for observation_noise in self.observation_noise_grid:
                train_result = self._run_filter(
                    train_panel,
                    process_noise=process_noise,
                    observation_noise=observation_noise,
                )
                validation_result = self._run_filter(
                    validation_panel,
                    process_noise=process_noise,
                    observation_noise=observation_noise,
                    initial_mean=train_result["final_beta"],
                    initial_cov=train_result["final_cov"],
                )
                mse = float(np.mean(np.square(validation_result["innovations"])))
                rows.append(
                    {
                        "process_noise": float(process_noise),
                        "observation_noise": float(observation_noise),
                        "validation_mse": mse,
                    }
                )
                if mse < best_mse:
                    best_mse = mse
                    best_process = float(process_noise)
                    best_observation = float(observation_noise)

        self.calibration_results_ = pd.DataFrame(rows).sort_values(
            "validation_mse", ascending=True
        )
        LOGGER.info(
            "Kalman calibration complete: process_noise=%.2e, observation_noise=%.2e, mse=%.6f",
            best_process,
            best_observation,
            best_mse,
        )
        return best_process, best_observation

    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
        calibrate: bool = True,
    ) -> "KalmanHedgeEstimator":
        panel = _prepare_panel(
            train_data,
            target_column=self.target_column,
            proxy_columns=self.proxy_columns,
        )
        self.fit_index_ = panel.index

        if calibrate and validation_data is not None:
            best_process, best_observation = self.calibrate(
                train_data=train_data,
                validation_data=validation_data,
            )
            self.process_noise = best_process
            self.observation_noise = best_observation

        result = self._run_filter(
            panel,
            process_noise=self.process_noise,
            observation_noise=self.observation_noise,
        )

        self.hedge_ratios_ = pd.DataFrame(
            result["beta_prior_path"],
            index=panel.index,
            columns=list(self.proxy_columns),
        )
        self.filtered_hedge_ratios_ = pd.DataFrame(
            result["beta_post_path"],
            index=panel.index,
            columns=list(self.proxy_columns),
        )
        self.predicted_target_ = pd.Series(
            result["predictions"],
            index=panel.index,
            name="predicted_target_return",
        )
        self.final_state_mean_ = result["final_beta"]
        self.final_state_cov_ = result["final_cov"]

        mse = float(np.mean(np.square(result["innovations"])))
        self.diagnostics_ = {
            "n_obs": float(len(panel)),
            "process_noise": float(self.process_noise),
            "observation_noise": float(self.observation_noise),
            "in_sample_one_step_mse": mse,
        }
        LOGGER.info("Kalman fit diagnostics: %s", self.diagnostics_)
        return self

    def predict(self, x_t: pd.Series | pd.DataFrame) -> pd.Series:
        """Predict target returns using the latest filtered beta state."""
        if self.final_state_mean_ is None:
            raise RuntimeError("fit must be called before predict.")

        x_frame = pd.DataFrame(x_t).T if isinstance(x_t, pd.Series) else x_t.copy()
        x_aligned = x_frame.loc[:, list(self.proxy_columns)].to_numpy(dtype=float)
        predictions = x_aligned @ self.final_state_mean_
        return pd.Series(
            predictions, index=x_frame.index, name="predicted_target_return"
        )

    def predict_hedge_ratio(self, t: pd.Timestamp) -> pd.Series:
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before predict_hedge_ratio.")
        if t not in self.hedge_ratios_.index:
            raise KeyError(f"Timestamp {t} not present in fitted hedge ratio index.")
        return self.hedge_ratios_.loc[t].copy()

    def hedge_ratio_time_series(self, index: pd.Index | None = None) -> pd.DataFrame:
        """Return no-lookahead beta_t = beta_{t|t-1} aligned with dates."""
        if self.hedge_ratios_ is None:
            raise RuntimeError("fit must be called before hedge_ratio_time_series.")
        if index is None:
            return self.hedge_ratios_.copy()
        return self.hedge_ratios_.reindex(index).ffill()


class KalmanSingleProxyHedgeEstimator(KalmanHedgeEstimator):
    """Single-proxy Kalman hedge estimator."""

    def __init__(
        self,
        target_column: str = "jet_fuel_illiquid",
        proxy_column: str = "proxy_1",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_column=target_column,
            proxy_columns=(proxy_column,),
            **kwargs,
        )


class KalmanMultiProxyHedgeEstimator(KalmanHedgeEstimator):
    """Multi-proxy Kalman hedge estimator."""

    def __init__(
        self,
        target_column: str = "jet_fuel_illiquid",
        proxy_columns: Sequence[str] = ("proxy_1", "proxy_2"),
        **kwargs: Any,
    ) -> None:
        columns = tuple(proxy_columns)
        if len(columns) < 2:
            raise ValueError(
                "KalmanMultiProxyHedgeEstimator requires at least two proxies."
            )
        super().__init__(
            target_column=target_column,
            proxy_columns=columns,
            **kwargs,
        )
