# Architecture & Technical Design

## 1. Design principles

1. **Single source of truth for inputs.** Every model prices the *same* immutable
   `OptionContract` + `MarketData` value objects, so cross-model comparisons are
   apples-to-apples by construction.
2. **Strategy pattern for models.** All pricers implement `PricingModel.price()`
   returning a `PricingResult`. Sensitivity analysis and benchmarking depend only
   on this interface — adding a new model (e.g. Heston, finite-difference PDE)
   requires no changes to downstream code (Open/Closed Principle).
3. **Separation of concerns.** Pricing, Greeks, simulation, analytics,
   benchmarking, and presentation (dashboard) are independent packages with a
   one-directional dependency graph.
4. **Numerical reproducibility.** Seeds and tolerances are centralized in
   `config.py`; Monte Carlo defaults to a fixed seed.
5. **Typed and documented.** Full type hints, dataclasses with validation, and
   docstrings carrying the relevant mathematics.

## 2. Module responsibilities

| Package | Responsibility | Key public API |
| --- | --- | --- |
| `instruments` | Domain value objects + validation | `OptionContract`, `MarketData`, `OptionType`, `ExerciseStyle` |
| `pricing` | The three models behind one interface | `BlackScholesModel`, `BinomialTreeModel`, `MonteCarloModel` |
| `simulations` | GBM path/terminal generation, variance reduction | `GBMSimulator`, `terminal_prices`, `SimulationConfig` |
| `greeks` | Analytic + model-agnostic numerical sensitivities | `black_scholes_greeks`, `numerical_greeks`, `Greeks` |
| `analytics` | 1-D/2-D sensitivity, scenario grids, profiles | `price_vs_parameter`, `two_factor_surface`, `scenario_grid` |
| `benchmarks` | Accuracy/runtime/memory/scalability harness | `benchmark_models`, `accuracy_vs_blackscholes`, `scalability_curve` |
| `config` | Numerical defaults + structured logging | `EngineConfig`, `get_logger` |
| `dashboard` | Streamlit + Plotly presentation layer | `dashboard/app.py` |

## 3. Dependency graph

```
instruments  ◄─────────────┬───────────────┬──────────────┐
   ▲                        │               │              │
   │                   simulations          │              │
pricing.base ◄── pricing.{bs,binomial,mc} ◄─┘              │
   ▲                        ▲                               │
   │                        │                               │
greeks.{analytical,numerical}                               │
   ▲                        ▲                               │
analytics ──────────────────┘                               │
benchmarks ─────────────────────────────────────────────────┘
dashboard ── depends on everything above (presentation only)
```

No cycles; presentation and benchmarking are leaves.

## 4. Error handling

- Construction-time validation in `OptionContract`/`MarketData` (`ValueError`,
  `TypeError`) fails fast on bad economic inputs.
- Models raise `NotImplementedError` for unsupported exercise styles
  (`PricingModel.supports()` lets callers check first).
- The binomial model raises if the risk-neutral probability falls outside
  $[0,1]$ (degenerate $\sigma$/step combinations).

## 5. Configuration & logging

`config.EngineConfig` centralizes lattice steps, MC path counts/seed,
variance-reduction switches, and finite-difference bumps. `get_logger()`
configures a single root handler driven by the `OPTION_ENGINE_LOGLEVEL`
environment variable, keeping library use quiet by default.

## 6. Testing strategy (summary)

Property-based invariants (put-call parity, monotonicity, intrinsic bounds),
cross-model agreement, convergence studies, and analytic-vs-numerical Greeks
checks. See `docs/` and `tests/`. Run: `pytest -q`.

## 7. Extensibility roadmap

- **New dynamics:** add a `pricing/heston.py` implementing `PricingModel`.
- **Path-dependent payoffs:** `GBMSimulator.paths()` already returns full paths;
  add Asian/barrier payoff functions and a `MonteCarloModel` payoff strategy.
- **PDE solver:** a Crank-Nicolson finite-difference model slots in behind the
  same interface and would provide a second American-exercise reference.
- **Vectorized portfolios:** batch `OptionContract`s for book-level risk.
