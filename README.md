# Option Pricing Engine

A production-quality, extensible engine for valuing European (and American, via
the lattice) options with **three independent methodologies** — closed-form
Black-Scholes, a Cox-Ross-Rubinstein binomial tree, and risk-neutral Monte Carlo
— plus analytic & numerical Greeks, sensitivity/scenario analysis, a benchmarking
harness, and an interactive Streamlit dashboard.

## Highlights

- **Three models behind one interface** (`PricingModel`) so they are
  interchangeable for analysis and benchmarking.
- **Black-Scholes-Merton** closed form with continuous dividend yield.
- **Binomial tree (CRR)** — vectorized, supports European *and* American exercise.
- **Monte Carlo** with antithetic + control variates, standard errors & 95% CIs.
- **Greeks** — analytic (Δ, Γ, Vega, Θ, ρ) and model-agnostic finite differences,
  cross-validated against each other.
- **Sensitivity analysis** — 1-D sweeps, 2-D price surfaces/heatmaps, P&L scenario
  ladders, Greeks profiles (all returned as tidy DataFrames).
- **Benchmarking** — accuracy vs. analytic reference, runtime, peak memory,
  convergence and scalability curves.
- **Interactive dashboard** — Streamlit + Plotly.
- **37 tests** covering parity invariants, convergence, and cross-model agreement.

## Project layout

```
src/option_engine/
├── instruments.py        # OptionContract, MarketData, enums (validated)
├── config.py             # numerical defaults + logging
├── pricing/              # base interface + black_scholes / binomial / monte_carlo
├── simulations/          # GBM path & terminal generation, variance reduction
├── greeks/               # analytical (BS) + numerical (finite difference)
├── analytics/            # sensitivity sweeps, surfaces, scenario grids
└── benchmarks/           # accuracy / runtime / memory / scalability harness
dashboard/app.py          # Streamlit dashboard
benchmarks/run_benchmarks.py
tests/                    # unit + property + integration tests
docs/                     # math foundations, architecture, agent swarm plan
```

## Installation

```bash
python -m pip install -e ".[all]"     # engine + viz + dashboard + dev
# or minimal:
python -m pip install -e .
```

Requires Python 3.11+.

## Quickstart

```python
from option_engine import (
    OptionContract, MarketData, OptionType,
    BlackScholesModel, BinomialTreeModel, MonteCarloModel,
    black_scholes_greeks,
)

contract = OptionContract(spot=100, strike=100, maturity=1.0, option_type=OptionType.CALL)
market = MarketData(rate=0.05, volatility=0.20, dividend_yield=0.0)

print(BlackScholesModel().value(contract, market))            # 10.4506
print(BinomialTreeModel(steps=1000).value(contract, market))  # 10.4486
mc = MonteCarloModel(n_paths=200_000).price(contract, market)
print(mc.price, mc.confidence_interval)                       # ~10.45, (lo, hi)

print(black_scholes_greeks(contract, market).scaled())        # display-scaled Greeks
```

## Run the tests

```bash
pytest -q                 # all tests
pytest -m "not slow"      # skip slow tests
```

## Run the benchmark report

```bash
python benchmarks/run_benchmarks.py          # print tables
python benchmarks/run_benchmarks.py --save   # also write CSVs to benchmarks/results/
```

## Launch the dashboard

```bash
streamlit run dashboard/app.py
```

## Documentation

- [`docs/mathematical_foundations.md`](docs/mathematical_foundations.md) — derivations & formulas.
- [`docs/architecture.md`](docs/architecture.md) — design, module map, extensibility.
- [`docs/agent_swarm_plan.md`](docs/agent_swarm_plan.md) — Ruflo multi-agent execution plan.

## Benchmark snapshot (ATM call, S=K=100, T=1, r=5%, σ=20%)

| Model | Price | Abs. error | Mean runtime |
| --- | --- | --- | --- |
| Black-Scholes | 10.4506 | — (reference) | ~0.2 ms |
| Binomial (1000 steps) | 10.4486 | 2.0e-3 | ~9 ms |
| Monte Carlo (200k, CV) | ~10.44 | ~1e-2 | ~17 ms |

Binomial error scales $O(1/N)$; Monte Carlo error scales $O(1/\sqrt{M})$.

