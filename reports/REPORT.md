# Proxy Hedging of Airline Jet Fuel Exposure with Public Data

## 1. Context and Problem Framing
Airline fuel programs face a persistent mismatch between exposure and hedge instrument liquidity. Operational exposure may map to local, illiquid jet-fuel hubs, while liquid risk transfer occurs in broader distillate and crude benchmarks. This study evaluates a reproducible proxy-hedging workflow under that mismatch.

The objective is to hedge a synthetic illiquid jet-fuel exposure with observable public proxy markets, while controlling turnover and transaction costs.

## 2. Data and Assumptions
### 2.1 Public market data
All raw series are downloaded from FRED CSV endpoints (no API key):
- `DJFUELUSGULF` (jet fuel benchmark, U.S. Gulf Coast)
- `DDFUELUSGULF` (ULSD diesel, U.S. Gulf Coast)
- `DHOILNYH` (No.2 heating oil, NY Harbor)
- `DCOILBRENTEU` (Brent spot)
- `DCOILWTICO` (WTI spot)
- `DGASNYH` (gasoline, NY Harbor regular)

Units:
- Jet/diesel/heating oil/gasoline: USD per gallon
- Brent/WTI: USD per barrel

### 2.2 Illiquid hub construction
A true illiquid jet-fuel hub is not publicly observable. We therefore construct `P_illiquid` from benchmark jet fuel by adding:
- a time-varying basis process (AR(1) with regime jumps),
- heteroskedastic idiosyncratic noise,
- optional missingness and delayed updates (stale prints).

The hedge target is the return of this simulated illiquid series, not benchmark jet fuel returns.

## 3. Methodology
### 3.1 Strategy set
1. **Single-proxy static OLS**: each proxy is tested; best proxy is selected on validation performance and evaluated out-of-sample.
2. **Multi-proxy ridge**: regularized static hedge with time-series split CV for alpha.
3. **Multi-proxy Kalman**: dynamic state-space regression with random-walk betas.
4. **Cheap benchmark-direct baseline**: unit hedge with jet benchmark itself (informative lower bound, not tradability-accurate for all contexts).

### 3.2 Execution engine
Positions are set on rebalance dates by `position = -beta * notional`, held constant between rebalances, and constrained by:
- per-beta absolute caps,
- gross leverage caps.

Transaction costs follow:
`cost_t = spread_bps * |delta_position_t| + fixed_fee * 1_{trade}`
with low/med/high scenarios.

## 4. Experimental Design
- Calendar alignment: business-day index with bounded forward-fill.
- Validation: explicit walk-forward train/validation/test splits from config.
- No look-ahead guardrails:
  - feature construction uses information up to `t-1`,
  - dynamic hedge ratios are applied causally,
  - model selection uses validation only (before test).
- Outputs include method-level tables, risk/cost metrics, and diagnostic charts.

## 5. Results Summary
Result tables and figures are generated under `reports/tables` and `reports/figures`.

Interpretation priorities:
- Evaluate **net** performance first (cost-adjusted Sharpe, VaR/CVaR, turnover, total costs).
- Compare gross hedge effectiveness to net outcomes to quantify cost drag.
- Use rolling correlations and beta stability diagnostics to justify proxy selection and dynamic model complexity.

## 6. Conclusions
Public-data proxy hedging can be reproduced end-to-end without proprietary datasets. Static models provide transparent baselines, while multi-proxy regularization and Kalman dynamics improve adaptability when basis relationships evolve. Practical deployment decisions should remain net-performance-driven, with strict governance around turnover and model drift.

## 7. Limitations
- Illiquid exposure is simulated, not observed.
- Energy unit heterogeneity (gallon vs barrel) is handled statistically via returns but may obscure structural conversion economics.
- Regime behavior may differ from true regional microstructure during stress episodes.
