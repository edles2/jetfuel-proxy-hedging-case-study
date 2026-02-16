from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Support `python -m src.run_experiments` while existing modules use top-level imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from costs.transaction_costs import TransactionCostConfig
from data.illiquid_hub import IlliquidHubConfig, simulate_illiquid_hub
from data.loaders import load_fred_market_prices
from data.preprocess import MissingDataPolicy, preprocess_prices_and_returns
from evaluation.metrics import evaluate_hedge_performance
from evaluation.report import MethodEvaluationData, generate_evaluation_artifacts
from features.build import FeatureConfig, build_dataset
from hedging.engine import HedgeConstraints, run_hedging_engine
from models.baselines import RidgeHedgeEstimator, StaticOLSHedgeEstimator
from models.kalman import KalmanMultiProxyHedgeEstimator

LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a YAML mapping at top-level.")
    return payload


def _resolve_date_range(date_cfg: Mapping[str, Any]) -> tuple[str, str]:
    start = date_cfg.get("start")
    if start is None:
        raise ValueError("date_range.start must be provided in config.")

    end_cfg = date_cfg.get("end")
    if isinstance(end_cfg, str) and end_cfg.strip():
        end_text = end_cfg.strip().lower()
        if end_text.startswith("today"):
            offset_days = 0
            if "-" in end_text:
                # Supports forms like "today-7d"
                tail = end_text.split("today", maxsplit=1)[1]
                if tail.startswith("-") and tail.endswith("d"):
                    offset_days = int(tail[1:-1])
            end_ts = datetime.now(timezone.utc).date() - timedelta(days=offset_days)
            end = end_ts.isoformat()
        else:
            end = end_cfg
    else:
        offset_days = int(date_cfg.get("end_offset_days", 7))
        end_ts = datetime.now(timezone.utc).date() - timedelta(days=offset_days)
        end = end_ts.isoformat()

    return str(start), str(end)


def _git_commit_hash(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _slice_by_dates(
    data: pd.DataFrame, start: str, end: str, label: str
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"{label}: end date {end} is before start date {start}.")
    out = data.loc[(data.index >= start_ts) & (data.index <= end_ts)]
    if out.empty:
        raise ValueError(f"{label}: no rows in range [{start}, {end}].")
    return out


def _constant_ratio_frame(ratio: pd.Series, index: pd.Index) -> pd.DataFrame:
    values = np.tile(ratio.to_numpy(dtype=float), (len(index), 1))
    return pd.DataFrame(values, index=index, columns=ratio.index)


def _append_method_part(
    store: dict[str, dict[str, list[pd.Series | pd.DataFrame]]],
    method: str,
    result: Any,
    hedge_ratios: pd.DataFrame,
) -> None:
    if method not in store:
        store[method] = {
            "unhedged": [],
            "hedged_gross": [],
            "hedged_net": [],
            "turnover": [],
            "cost": [],
            "ratios": [],
        }
    store[method]["unhedged"].append(result.unhedged_pnl)
    store[method]["hedged_gross"].append(result.hedged_pnl_gross)
    store[method]["hedged_net"].append(result.hedged_pnl_net)
    store[method]["turnover"].append(result.turnover)
    store[method]["cost"].append(result.transaction_cost)
    store[method]["ratios"].append(hedge_ratios)


def _concat_series(parts: list[pd.Series]) -> pd.Series:
    return pd.concat(parts).sort_index()


def _concat_frames(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parts).sort_index()


def _build_method_data(
    parts_store: dict[str, dict[str, list[pd.Series | pd.DataFrame]]]
) -> dict[str, MethodEvaluationData]:
    output: dict[str, MethodEvaluationData] = {}
    for method, parts in parts_store.items():
        output[method] = MethodEvaluationData(
            unhedged_pnl=_concat_series(parts["unhedged"]),  # type: ignore[arg-type]
            hedged_pnl_gross=_concat_series(parts["hedged_gross"]),  # type: ignore[arg-type]
            hedged_pnl_net=_concat_series(parts["hedged_net"]),  # type: ignore[arg-type]
            turnover=_concat_series(parts["turnover"]),  # type: ignore[arg-type]
            transaction_cost=_concat_series(parts["cost"]),  # type: ignore[arg-type]
            hedge_ratios=_concat_frames(parts["ratios"]),  # type: ignore[arg-type]
        )
    return output


def _write_proxy_diagnostics(
    returns_panel: pd.DataFrame,
    target_column: str,
    proxy_columns: list[str],
    reports_dir: Path,
    window: int,
) -> None:
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    corr_frame = pd.DataFrame(index=returns_panel.index)
    for proxy in proxy_columns:
        corr_frame[proxy] = (
            returns_panel[target_column]
            .rolling(window=window)
            .corr(returns_panel[proxy])
        )
    corr_frame.to_csv(
        tables_dir / "rolling_correlations.csv", index=True, index_label="date"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for proxy in proxy_columns:
        ax.plot(corr_frame.index, corr_frame[proxy], label=proxy)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title(f"Rolling Correlation with {target_column} (window={window})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "proxy_rolling_correlations.png", dpi=160)
    plt.close(fig)


def _write_beta_stability_table(
    method_data: Mapping[str, MethodEvaluationData],
    reports_dir: Path,
) -> None:
    rows: list[dict[str, float | str]] = []
    for method, payload in method_data.items():
        if payload.hedge_ratios is None:
            continue
        ratios = (
            payload.hedge_ratios.to_frame(name="beta")
            if isinstance(payload.hedge_ratios, pd.Series)
            else payload.hedge_ratios
        )
        for column in ratios.columns:
            series = ratios[column].dropna().astype(float)
            if series.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "beta_name": str(column),
                    "beta_mean": float(series.mean()),
                    "beta_std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "beta_avg_abs_change": float(series.diff().abs().mean())
                    if len(series) > 1
                    else 0.0,
                    "beta_autocorr_lag1": float(series.autocorr(lag=1))
                    if len(series) > 2
                    else float("nan"),
                }
            )

    beta_table = pd.DataFrame(rows)
    beta_table.to_csv(reports_dir / "tables" / "beta_stability.csv", index=False)


def run_experiments(
    config_path: Path,
    project_root: Path | None = None,
    refresh: bool = False,
) -> Path:
    config = _load_config(config_path)
    root = project_root or PROJECT_ROOT

    data_cfg = config["data"]
    fred_cfg = data_cfg["fred"]
    series_ids = dict(fred_cfg["series_ids"])
    if "jet_fuel_benchmark" not in series_ids:
        raise ValueError("data.fred.series_ids must include jet_fuel_benchmark.")

    start_date, end_date = _resolve_date_range(config["date_range"])
    max_ffill_gap_days = int(data_cfg.get("max_ffill_gap_days", 3))
    frequency = str(config.get("frequency", "B"))

    LOGGER.info("Loading FRED prices from %s to %s", start_date, end_date)
    market_prices = load_fred_market_prices(
        series_ids=series_ids,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        max_ffill_gap_days=max_ffill_gap_days,
        cache_dir=root / str(data_cfg.get("fred_cache_dir", "data/raw/fred")),
        refresh=refresh,
    )

    illiquid_cfg = IlliquidHubConfig(**config["illiquid_hub"])
    illiquid_panel = simulate_illiquid_hub(
        benchmark_prices=market_prices["jet_fuel_benchmark"],
        config=illiquid_cfg,
    )

    full_prices = market_prices.copy()
    full_prices["jet_fuel_illiquid"] = illiquid_panel["illiquid_price"]

    cleaned_prices, simple_returns, log_returns = preprocess_prices_and_returns(
        prices=full_prices,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        missing_data_policy=MissingDataPolicy(
            method="ffill_then_drop",
            max_forward_fill=max_ffill_gap_days,
        ),
    )

    returns_panel = (
        simple_returns
        if str(config.get("returns_type", "simple")) == "simple"
        else log_returns
    ).dropna()

    target_column = "jet_fuel_illiquid"
    benchmark_column = "jet_fuel_benchmark"
    proxy_columns = [col for col in market_prices.columns if col.startswith("proxy_")]
    if not proxy_columns:
        raise ValueError(
            "No proxy columns found. Expected names starting with 'proxy_'."
        )

    # Feature diagnostics dataset
    feature_cfg_raw: Mapping[str, Any] = config.get("features", {})
    feature_cfg = FeatureConfig(
        target_column=target_column,
        proxy_columns=tuple(proxy_columns),
        max_lag=int(feature_cfg_raw.get("max_lag", 3)),
        rolling_windows=tuple(feature_cfg_raw.get("rolling_windows", (20, 60))),
        include_return_spreads=bool(
            feature_cfg_raw.get("include_return_spreads", True)
        ),
        include_price_spreads=bool(feature_cfg_raw.get("include_price_spreads", True)),
        include_pca=bool(feature_cfg_raw.get("include_pca", False)),
        pca_components=int(feature_cfg_raw.get("pca_components", 2)),
        pca_min_history=int(feature_cfg_raw.get("pca_min_history", 60)),
        dropna=bool(feature_cfg_raw.get("dropna", True)),
    )
    x_features, y_target, _ = build_dataset(
        prices=cleaned_prices[[target_column, *proxy_columns]],
        returns=returns_panel[[target_column, *proxy_columns]],
        config=feature_cfg,
    )

    reports_dir = root / "reports"
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    feature_dataset = x_features.copy()
    feature_dataset[target_column] = y_target
    feature_dataset.to_csv(
        tables_dir / "feature_dataset.csv", index=True, index_label="date"
    )

    hedging_cfg = config["hedging"]
    constraints = HedgeConstraints(**hedging_cfg.get("constraints", {}))
    exposure_notional = float(hedging_cfg.get("exposure_notional", 1.0))
    rebalance_frequency = str(hedging_cfg.get("rebalance_frequency", "daily"))

    cost_config = TransactionCostConfig.from_dict(config["transaction_costs"])
    cost_scenario = str(hedging_cfg.get("cost_scenario", cost_config.default_scenario))

    splits = list(config["walk_forward"]["splits"])
    if not splits:
        raise ValueError("walk_forward.splits must contain at least one split.")

    methods_store: dict[str, dict[str, list[pd.Series | pd.DataFrame]]] = {}
    split_rows: list[dict[str, Any]] = []

    for split in splits:
        split_name = str(split["name"])
        train_df = _slice_by_dates(
            returns_panel,
            start=str(split["train_start"]),
            end=str(split["train_end"]),
            label=f"{split_name}:train",
        )
        val_df = _slice_by_dates(
            returns_panel,
            start=str(split["val_start"]),
            end=str(split["val_end"]),
            label=f"{split_name}:val",
        )
        test_df = _slice_by_dates(
            returns_panel,
            start=str(split["test_start"]),
            end=str(split["test_end"]),
            label=f"{split_name}:test",
        )
        train_val_df = pd.concat([train_df, val_df]).sort_index()

        # 1) Single-proxy static OLS (test all proxies, pick best on validation)
        best_proxy = proxy_columns[0]
        best_score = -np.inf
        for proxy in proxy_columns:
            candidate_model = StaticOLSHedgeEstimator(
                target_column=target_column,
                proxy_columns=(proxy,),
                max_abs_hedge_ratio=float(
                    config.get("models", {})
                    .get("single_static_ols", {})
                    .get("max_abs_hedge_ratio", 3.0)
                ),
            ).fit(train_df[[target_column, proxy]])

            candidate_ratios_val = candidate_model.hedge_ratio_time_series(
                index=val_df.index
            )
            candidate_res = run_hedging_engine(
                target_returns=val_df[target_column],
                proxy_returns=val_df[[proxy]],
                hedge_ratios=candidate_ratios_val,
                rebalance_frequency=rebalance_frequency,
                constraints=constraints,
                exposure_notional=exposure_notional,
                cost_config=cost_config,
                cost_scenario=cost_scenario,
            )
            score = evaluate_hedge_performance(
                unhedged_pnl=candidate_res.unhedged_pnl,
                hedged_pnl_gross=candidate_res.hedged_pnl_gross,
                hedged_pnl_net=candidate_res.hedged_pnl_net,
                turnover=candidate_res.turnover,
                transaction_cost=candidate_res.transaction_cost,
            )["hedge_effectiveness"]
            if score > best_score:
                best_score = score
                best_proxy = proxy

        static_model = StaticOLSHedgeEstimator(
            target_column=target_column,
            proxy_columns=(best_proxy,),
            max_abs_hedge_ratio=float(
                config.get("models", {})
                .get("single_static_ols", {})
                .get("max_abs_hedge_ratio", 3.0)
            ),
        ).fit(train_val_df[[target_column, best_proxy]])
        static_ratios_test = static_model.hedge_ratio_time_series(index=test_df.index)
        static_result = run_hedging_engine(
            target_returns=test_df[target_column],
            proxy_returns=test_df[[best_proxy]],
            hedge_ratios=static_ratios_test,
            rebalance_frequency=rebalance_frequency,
            constraints=constraints,
            exposure_notional=exposure_notional,
            cost_config=cost_config,
            cost_scenario=cost_scenario,
        )
        _append_method_part(
            methods_store,
            method="single_static_ols_best_proxy",
            result=static_result,
            hedge_ratios=static_ratios_test,
        )

        # 2) Multi-proxy ridge
        ridge_cfg = config.get("models", {}).get("multi_proxy_ridge", {})
        ridge = RidgeHedgeEstimator(
            target_column=target_column,
            proxy_columns=tuple(proxy_columns),
            alphas=tuple(
                float(v) for v in ridge_cfg.get("alphas", [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
            ),
            cv_splits=int(ridge_cfg.get("cv_splits", 5)),
            max_abs_hedge_ratio=float(ridge_cfg.get("max_abs_hedge_ratio", 3.0)),
        ).fit(train_val_df[[target_column, *proxy_columns]])
        ridge_ratios_test = ridge.hedge_ratio_time_series(index=test_df.index)
        ridge_result = run_hedging_engine(
            target_returns=test_df[target_column],
            proxy_returns=test_df[proxy_columns],
            hedge_ratios=ridge_ratios_test,
            rebalance_frequency=rebalance_frequency,
            constraints=constraints,
            exposure_notional=exposure_notional,
            cost_config=cost_config,
            cost_scenario=cost_scenario,
        )
        _append_method_part(
            methods_store,
            method="multi_proxy_ridge",
            result=ridge_result,
            hedge_ratios=ridge_ratios_test,
        )

        # 3) Multi-proxy Kalman (calibrate on train/val, then online-filter through test)
        kalman_cfg = config.get("models", {}).get("multi_proxy_kalman", {})
        kalman = KalmanMultiProxyHedgeEstimator(
            target_column=target_column,
            proxy_columns=tuple(proxy_columns),
            process_noise=float(kalman_cfg.get("process_noise", 1e-4)),
            observation_noise=float(kalman_cfg.get("observation_noise", 1e-3)),
            process_noise_grid=tuple(
                float(v)
                for v in kalman_cfg.get("process_noise_grid", [1e-6, 1e-5, 1e-4, 1e-3])
            ),
            observation_noise_grid=tuple(
                float(v)
                for v in kalman_cfg.get("observation_noise_grid", [1e-4, 1e-3, 1e-2])
            ),
            max_abs_hedge_ratio=float(kalman_cfg.get("max_abs_hedge_ratio", 3.0)),
        )
        if bool(kalman_cfg.get("calibrate", True)):
            kalman.calibrate(
                train_data=train_df[[target_column, *proxy_columns]],
                validation_data=val_df[[target_column, *proxy_columns]],
            )

        kalman_fit_panel = pd.concat([train_df, val_df, test_df]).sort_index()[
            [target_column, *proxy_columns]
        ]
        kalman.fit(train_data=kalman_fit_panel, calibrate=False)
        kalman_ratios_test = kalman.hedge_ratio_time_series().loc[
            test_df.index, proxy_columns
        ]
        kalman_result = run_hedging_engine(
            target_returns=test_df[target_column],
            proxy_returns=test_df[proxy_columns],
            hedge_ratios=kalman_ratios_test,
            rebalance_frequency=rebalance_frequency,
            constraints=constraints,
            exposure_notional=exposure_notional,
            cost_config=cost_config,
            cost_scenario=cost_scenario,
        )
        _append_method_part(
            methods_store,
            method="multi_proxy_kalman",
            result=kalman_result,
            hedge_ratios=kalman_ratios_test,
        )

        # 4) Cheap baseline: direct benchmark jet hedge (unit beta)
        cheap_ratios_test = pd.DataFrame({benchmark_column: 1.0}, index=test_df.index)
        cheap_result = run_hedging_engine(
            target_returns=test_df[target_column],
            proxy_returns=test_df[[benchmark_column]],
            hedge_ratios=cheap_ratios_test,
            rebalance_frequency=rebalance_frequency,
            constraints=constraints,
            exposure_notional=exposure_notional,
            cost_config=cost_config,
            cost_scenario=cost_scenario,
        )
        _append_method_part(
            methods_store,
            method="cheap_benchmark_direct",
            result=cheap_result,
            hedge_ratios=cheap_ratios_test,
        )

        split_rows.append(
            {
                "split": split_name,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "best_single_proxy": best_proxy,
                "best_single_proxy_validation_hedge_effectiveness": best_score,
            }
        )

    pd.DataFrame(split_rows).to_csv(tables_dir / "walk_forward_splits.csv", index=False)

    method_data = _build_method_data(methods_store)
    eval_cfg = config.get("evaluation", {})
    comparison_table = generate_evaluation_artifacts(
        method_results=method_data,
        reports_dir=reports_dir,
        rolling_window=int(eval_cfg.get("rolling_window", 60)),
        var_confidence=float(eval_cfg.get("var_confidence", 0.95)),
        periods_per_year=int(eval_cfg.get("periods_per_year", 252)),
    )
    comparison_table.to_csv(tables_dir / "method_comparison.csv")

    _write_proxy_diagnostics(
        returns_panel=returns_panel,
        target_column=target_column,
        proxy_columns=proxy_columns,
        reports_dir=reports_dir,
        window=int(eval_cfg.get("proxy_diagnostic_window", 60)),
    )
    _write_beta_stability_table(method_data=method_data, reports_dir=reports_dir)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "git_commit_hash": _git_commit_hash(root),
        "config": config,
        "data_window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "artifacts": {
            "tables": sorted(
                str(path.relative_to(root))
                for path in (reports_dir / "tables").glob("*")
            ),
            "figures": sorted(
                str(path.relative_to(root))
                for path in (reports_dir / "figures").glob("*.png")
            ),
        },
    }
    manifest_path = reports_dir / "tables" / "artifacts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full proxy hedging experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/default.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh of cached FRED raw CSV files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(level=args.log_level)

    manifest_path = run_experiments(
        config_path=args.config,
        refresh=bool(args.refresh),
    )
    print(f"Experiment run completed. Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
