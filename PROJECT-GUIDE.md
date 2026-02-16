# Project Guide

## Goal
Provide a modular and reproducible research baseline for testing proxy hedging designs for airline jet fuel exposure.

## Design Principles
- Reproducibility first: deterministic synthetic fallback with fixed seed.
- Clear separation of concerns across `src` subpackages.
- Strict typing and small functions for easier extension.
- Output artifacts committed to predictable paths.

## Module Responsibilities
- `src/config`: runtime and path configuration.
- `src/data`: loading, validation, and synthetic data generation.
- `src/features`: return engineering and model table construction.
- `src/models`: hedge ratio estimation models (currently OLS).
- `src/hedging`: hedge application logic.
- `src/costs`: transaction cost models.
- `src/evaluation`: hedge effectiveness metrics.
- `src/plots`: reporting charts.
- `src/pipeline.py`: orchestrates end-to-end execution.

## Extension Points
1. Add richer basis features in `src/features` (lags, spreads, seasonality).
2. Add constrained/dynamic hedge models in `src/models` and `src/hedging`.
3. Replace static costs with venue-specific nonlinear cost curves in `src/costs`.
4. Add stress-test and scenario attribution in `src/evaluation`.
5. Add publication-quality figures in `src/plots`.

## Testing Strategy
- Unit test coefficient recovery for deterministic OLS behavior.
- End-to-end pipeline test with temporary project root to validate generated artifacts.

## Recommended Workflow
1. Create/refresh virtual environment via `make install`.
2. Use notebooks for exploration.
3. Promote stable logic into `src/` modules.
4. Validate using `make lint` and `make test`.
5. Execute full run with `make run-pipeline`.
