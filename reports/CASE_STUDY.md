# Proxy Hedging Case Study: Airline Jet Fuel Exposure (Public Data, Reproducible)

## 1. Executive Summary
This case study demonstrates an end-to-end proxy hedging workflow for airline jet fuel exposure using only public data. Because a true illiquid jet fuel hub is not publicly available, we construct a synthetic illiquid hub from an observable jet benchmark and evaluate hedge strategies against that simulated exposure.

The pipeline is fully reproducible from scratch: it downloads FRED CSV series (no API key), preprocesses and aligns data, simulates illiquidity, estimates hedge ratios with multiple methods, applies transaction costs, and saves tables/figures for analysis.

## 2. Business Context: Airline Jet Fuel Exposure & Hedging Problem
Airlines consume jet fuel physically, but liquid hedging instruments are often concentrated in broader energy hubs (diesel, heating oil, crude, gasoline). This creates a classic mismatch:

- Exposure: local or operationally specific fuel costs (often less liquid and less directly tradable).
- Instruments: liquid proxies with imperfect co-movement.

The core objective is to reduce P&L variance and tail risk while controlling turnover and execution costs.

## 3. Why Proxy Hedging: Liquidity, Basis Risk, Tradability
Proxy hedging is used when the exact exposure instrument is not liquid enough (or not listed) for practical risk transfer.

- Liquidity: deeper markets enable tighter execution and scalable sizing.
- Basis risk: proxy and exposure diverge, so hedge quality depends on correlation stability.
- Tradability: some reference prices are informative but not directly hedgeable at scale.

This report explicitly separates what is economically informative from what is operationally tradable.

## 4. Data (Public): Sources, Series IDs, Units, Frequency
Source: FRED CSV endpoint, no key required  
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`

Default series in `src/config/default.yaml`:

| Role | Column Name | FRED Series ID | Description | Unit |
|---|---|---|---|---|
| Jet benchmark | `jet_fuel_benchmark` | `DJFUELUSGULF` | Kerosene-Type Jet Fuel Prices: U.S. Gulf Coast | USD/gal |
| Proxy | `proxy_diesel_us_gulf` | `DDFUELUSGULF` | ULSD Diesel: U.S. Gulf Coast | USD/gal |
| Proxy | `proxy_heating_oil_nyh` | `DHOILNYH` | No.2 Heating Oil: NY Harbor | USD/gal |
| Proxy | `proxy_brent_spot` | `DCOILBRENTEU` | Brent Spot Price | USD/bbl |
| Proxy | `proxy_wti_spot` | `DCOILWTICO` | WTI Spot Price | USD/bbl |
| Proxy | `proxy_gasoline_nyh` | `DGASNYH` | Conventional Gasoline: NY Harbor Regular | USD/gal |

Frequency and alignment:
- Target calendar: business days (`B`).
- Forward-fill is bounded by `max_ffill_gap_days`.
- Remaining missing rows are dropped after alignment.

Unit mismatch note:
- Crude is quoted in USD/barrel while distillates are mostly USD/gallon.
- The modeling layer works on returns (and optionally standardized features), which mitigates level-unit incompatibility.
- This is statistically valid for hedge ratio estimation, but not a substitute for structural conversion economics.

## 5. Constructing the Illiquid Hub (Key Assumption)
We do **not** observe a true illiquid airline hub publicly. Therefore, the exposure is simulated.

Let `P_b,t` be the observable benchmark jet fuel price. We build:

1. Basis process (persistent + regimes):
   `b_t = phi * b_(t-1) + u_t + j_t`  
   where `u_t ~ N(0, sigma_basis^2)` and regime jump `j_t` occurs with configured probability.
2. Idiosyncratic heteroskedastic noise:
   `e_t ~ N(0, sigma_t^2)`, with `sigma_t` increasing with recent benchmark volatility.
3. True illiquid log-price:
   `log(P_illiquid,true,t) = log(P_b,t) + b_t + e_t`
4. Optional microstructure frictions:
   - random missing observations,
   - delayed updates (stale prints).

Exposed return for hedging:
- `r_illiquid,t = pct_change(P_illiquid,observed,t)`

Reproducibility:
- A fixed random seed in config makes simulation deterministic.

Why this is reasonable in a public demo:
- It preserves realistic hedge challenges (basis drift, regime breaks, noise, stale prints) while remaining fully reproducible and transparent.

What would change with proprietary data:
- Replace simulated `P_illiquid` with observed internal hub/realized procurement series.
- Re-estimate model class and constraints under real execution and accounting policies.

## 6. Methodology
### 6.1 Baselines: single-proxy static OLS, rolling OLS
Static OLS estimates one constant beta (single proxy or multi-proxy).  
Single-proxy baseline evaluates all proxies in-sample and carries the best validated proxy out-of-sample.

Rolling OLS (implemented baseline) allows beta updates through rolling windows to adapt gradually to regime changes.

### 6.2 Multi-proxy: ridge (shrinkage), how we choose alpha (time-series split)
Ridge regression estimates multi-proxy hedge weights with `L2` shrinkage:
- reduces variance under collinearity,
- improves stability and interpretability versus unconstrained OLS.

`alpha` is selected via time-series split CV on train/validation windows (no random shuffling).

### 6.3 Dynamic: Kalman time-varying betas (state-space regression)
State-space model:
- Observation: `y_t = x_t' beta_t + eps_t`
- Transition: `beta_t = beta_(t-1) + eta_t`

where `eps_t` and `eta_t` are Gaussian noises with configurable variances.  
This yields causal, time-varying hedge ratios that can adapt faster than static models.

### 6.4 Costs: transaction cost model + turnover, and why it matters
Per-period cost:
- `cost_t = spread_bps * |delta_position_t| + fixed_fee * 1_{trade}`

Net hedged P&L is reported alongside gross P&L.  
Turnover is tracked to show whether higher responsiveness is economically justified after costs.

### 6.5 Guardrails: no-lookahead, walk-forward evaluation, constraints on hedge ratios
- Features are lagged so hedge decisions at `t` use information available through `t-1`.
- Hedge ratios are applied causally (especially in dynamic models).
- Evaluation uses explicit walk-forward splits.
- Positioning respects beta caps and leverage constraints.

## 7. Experimental Design
Walk-forward splits (from config):

| Split | Train | Validation | Test |
|---|---|---|---|
| `wf_1` | 2006-01-03 to 2014-12-31 | 2015-01-02 to 2017-12-29 | 2018-01-02 to 2020-12-31 |
| `wf_2` | 2009-01-02 to 2017-12-29 | 2018-01-02 to 2020-12-31 | 2021-01-04 to 2023-12-29 |
| `wf_3` | 2012-01-03 to 2020-12-31 | 2021-01-04 to 2023-12-29 | 2024-01-02 to 2025-12-31 |

Execution settings:
- Rebalance frequency: daily (configurable).
- Cost scenario: low/med/high (default: med).
- Notional exposure: 1.0 by default.

Primary metrics:
- variance reduction / hedge effectiveness,
- tracking error,
- VaR / CVaR (hedged vs unhedged),
- turnover and total transaction costs,
- optional annualized Sharpe on net P&L (reported as no-risk-free approximation).

## 8. Results
If the artifacts below are missing in a fresh clone, run:

```bash
python -m src.run_experiments --config src/config/default.yaml
```

### 8.1 Main comparison table
Source file: [`tables/method_comparison.csv`](tables/method_comparison.csv)

Snapshot (example from the latest generated artifacts):

| Method | Variance Reduction | Tracking Error | Net Sharpe (ann.) | Total Turnover | Total Cost |
|---|---:|---:|---:|---:|---:|
| `cheap_benchmark_direct` | 0.8152 | 0.0295 | 0.0830 | 3.0000 | 0.000750 |
| `multi_proxy_kalman` | 0.6318 | 0.0253 | 0.0593 | 30.2638 | 0.108903 |
| `multi_proxy_ridge` | 0.6451 | 0.0243 | 0.1371 | 2.9922 | 0.000748 |
| `single_static_ols_best_proxy` | 0.6308 | 0.0242 | 0.1461 | 2.8149 | 0.000713 |

### 8.2 Key figures
Canonical report figures (stable filenames emitted by pipeline):

![Cumulative P&L](figures/cum_pnl.png)

![Hedge Ratios](figures/hedge_ratios.png)

![Rolling Hedge Effectiveness](figures/rolling_effectiveness.png)

![P&L Distribution](figures/pnl_distribution.png)

Additional diagnostic (proxy selection rationale):

![Rolling Correlations: Illiquid Target vs Proxies](figures/proxy_rolling_correlations.png)

## 9. Analysis & Discussion
Proxy quality:
- Distillate proxies (diesel/heating oil) often track jet fuel economics better than crude alone because product-level dynamics are closer.
- Crude proxies can still add value in multi-proxy sets by capturing broader macro oil shocks.

Stability vs turnover:
- Dynamic models (Kalman) can adapt quickly in shifting regimes but may trade more, raising cost drag.
- Static/regularized models can be less reactive but often more cost-efficient.

When multi-proxy helps:
- Useful when complementary factors exist (product cracks, regional spreads, global crude drivers).
- Less useful when proxies are highly collinear and relationships are stable, where shrinkage may dominate full flexibility.

Overfitting risk:
- Adding many features/proxies can improve in-sample fit but degrade out-of-sample robustness.
- Walk-forward validation and no-lookahead constraints are essential controls.

## 10. Limitations & Extensions
Limitations:
- Illiquid target is simulated, not observed.
- Public proxies are approximations and may not be directly executable in all airline contexts.
- Unit differences (USD/gal vs USD/bbl) are treated statistically in returns space, not through physical conversion.
- No true listed jet fuel futures instrument is used as a direct tradable benchmark in this public setup.

Extensions:
- Replace synthetic exposure with proprietary hub/real procurement series.
- Add futures term-structure and calendar-spread hedges.
- Include crack features, seasonality, storage/inventory signals, freight bottlenecks.
- Add explicit regime-switching or nonlinear state-space dynamics.
- Model execution frictions with richer liquidity curves.

## 11. Reproducibility
Single command:

```bash
python -m src.run_experiments --config src/config/default.yaml
```

Artifacts are written to:
- `reports/tables/`
- `reports/figures/`

Key files:
- `reports/tables/method_comparison.csv`
- `reports/tables/artifacts_manifest.json`
- `reports/figures/cum_pnl.png`
- `reports/figures/hedge_ratios.png`
- `reports/figures/rolling_effectiveness.png`
- `reports/figures/pnl_distribution.png`

Safe config edits:
- update `src/config/default.yaml` for date windows, series IDs, model grids, cost scenarios.
- keep walk-forward chronology valid (`train < val < test`).
- preserve no-lookahead rules when changing features/rebalancing.

## Appendix A: Definitions / Equations (OLS, ridge objective, Kalman model equations)
OLS (multi-proxy):
- `min_beta sum_t (y_t - x_t' beta)^2`

Ridge:
- `min_beta sum_t (y_t - x_t' beta)^2 + alpha * ||beta||_2^2`

Kalman (state-space):
- Observation: `y_t = x_t' beta_t + eps_t`, `eps_t ~ N(0, R)`
- State transition: `beta_t = beta_(t-1) + eta_t`, `eta_t ~ N(0, Q)`
- Filtering updates provide causal `beta_t` estimates through time.

Hedge effectiveness:
- `HE = 1 - Var(PnL_hedged) / Var(PnL_unhedged)`

Transaction costs:
- `cost_t = spread_bps * |delta_position_t| + fixed_fee * 1_{trade}`

## Appendix B: Configuration reference (key YAML fields)
Reference: `src/config/default.yaml`

```yaml
data:
  frequency: B
  max_ffill_gap_days: 5
  fred_cache_dir: data/raw/fred
  fred:
    series_ids:
      jet_fuel_benchmark: DJFUELUSGULF
      proxy_diesel_us_gulf: DDFUELUSGULF
      proxy_heating_oil_nyh: DHOILNYH
      proxy_brent_spot: DCOILBRENTEU
      proxy_wti_spot: DCOILWTICO
      proxy_gasoline_nyh: DGASNYH

date_range:
  start: "2006-01-01"
  end_offset_days: 7

illiquid_hub:
  basis_ar_coeff: 0.985
  basis_process_sigma: 0.0018
  regime_shift_probability: 0.012
  regime_shift_sigma: 0.02
  idiosyncratic_base_sigma: 0.0025
  heteroskedastic_scale: 10.0
  missingness_probability: 0.01
  delayed_update_probability: 0.04
  delay_max_days: 3
  seed: 2026

hedging:
  rebalance_frequency: daily
  constraints:
    max_abs_beta: 3.0
    leverage_cap: 5.0

transaction_costs:
  default_scenario: med
  scenarios:
    low: {spread_bps: 0.5, fixed_fee: 0.0}
    med: {spread_bps: 2.0, fixed_fee: 0.00005}
    high: {spread_bps: 5.0, fixed_fee: 0.0002}
```
