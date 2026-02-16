# Public-Data Proxy Hedging Case Study (Airline Jet Fuel Exposure)

## Problem Statement
Airlines are exposed to jet fuel price risk, but true illiquid exposure hubs are often not observable or tradable with deep liquidity. This repository reproduces a **proxy hedging** case study using only public data, with fully automated ingestion and no API keys.

Core idea:
- Use public FRED energy series for benchmark/proxy markets.
- Simulate an illiquid jet-fuel hub from the observable benchmark (basis dynamics + idiosyncratic noise + optional stale/missing updates).
- Hedge the simulated illiquid exposure with liquid proxies.

## Public Data Sources (FRED CSV, no key)
Data is fetched from:
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`

Default series:
- `DJFUELUSGULF` -> `jet_fuel_benchmark` (U.S. Gulf Coast jet fuel)
- `DDFUELUSGULF` -> proxy diesel (U.S. Gulf Coast)
- `DHOILNYH` -> proxy heating oil (NY Harbor)
- `DCOILBRENTEU` -> proxy Brent spot
- `DCOILWTICO` -> proxy WTI spot
- `DGASNYH` -> proxy gasoline (NY Harbor)

Units:
- Jet, diesel, heating oil, gasoline: USD per gallon
- Brent, WTI: USD per barrel

## Critical Assumption (Illiquid Hub)
A truly illiquid jet-fuel hub is not publicly available. We therefore:
1. treat `DJFUELUSGULF` as the closest observable benchmark,
2. construct synthetic illiquid price `P_illiquid` with stochastic basis/idiosyncratic effects,
3. hedge **illiquid returns** (not benchmark returns) using proxy instruments.

This assumption is explicit and configurable in `src/config/default.yaml`.

## Reproducible Run (Single Command)
```bash
python -m src.run_experiments --config src/config/default.yaml
```

Optional forced refresh of cached FRED raw files:
```bash
python -m src.run_experiments --config src/config/default.yaml --refresh
```

## What the Runner Does
1. Downloads (or reuses cached) FRED CSV series under `data/raw/fred/`.
2. Aligns series to business-day calendar with bounded forward-fill.
3. Simulates `jet_fuel_illiquid` exposure from benchmark jet fuel.
4. Builds returns/features with no-lookahead alignment.
5. Runs walk-forward train/validation/test experiments:
   - single-proxy static OLS (tests all proxies; picks best in validation)
   - multi-proxy ridge (time-series CV)
   - multi-proxy Kalman (dynamic beta)
   - cheap benchmark-direct hedge baseline
6. Applies transaction costs and constraints in hedging engine.
7. Saves tables/figures + JSON manifest in `reports/`.

## Output Artifacts
- `reports/tables/method_comparison.csv`
- `reports/tables/artifacts_manifest.json`
- `reports/tables/rolling_correlations.csv`
- `reports/tables/beta_stability.csv`
- `reports/figures/*.png`

## Reproducibility Notes
- FRED raw CSVs are cached locally for offline reruns.
- Illiquid hub simulation is deterministic given config seed.
- Walk-forward splits and hyperparameter grids are explicit in config.
