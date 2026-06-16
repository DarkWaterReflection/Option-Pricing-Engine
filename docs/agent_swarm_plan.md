# Ruflo Agent Swarm Execution Plan

This project is decomposed into seven specialized agents that can be orchestrated
by the Ruflo Agent Swarm. The decomposition is **interface-first**: Agent 0
freezes the shared contracts so the remaining agents can work in parallel against
stable APIs. Each agent owns a directory and a test surface.

## Topology & orchestration

```
                    ┌──────────────────────────┐
                    │ A0  Quant Research /       │  (defines interfaces + math spec)
                    │     Architecture (lead)    │
                    └──────────────┬─────────────┘
              ┌──────────────┬─────┴────┬──────────────┐
              ▼              ▼          ▼              ▼
   ┌────────────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────┐
   │ A1 Pricing     │ │ A2 Monte  │ │ A3 Greeks │ │ A6 Docs      │
   │    Engine      │ │    Carlo  │ │ & Analytics│ │ (continuous) │
   └───────┬────────┘ └─────┬─────┘ └─────┬─────┘ └──────────────┘
           └────────────────┴─────────────┘
                            ▼
                  ┌───────────────────┐     ┌────────────────────┐
                  │ A4 Dashboard      │     │ A5 Testing &       │
                  │                   │     │    Validation      │
                  └───────────────────┘     └────────────────────┘
```

**Recommended Ruflo topology:** hierarchical (A0 as coordinator) with a parallel
fan-out for A1–A3, then a join before A4/A5. Suggested MCP calls:
`swarm_init` (topology=hierarchical) → `agent_spawn` per role →
`task_create`/`task_assign` per deliverable → `coordination_orchestrate` →
`hive-mind_consensus` at the integration gate.

---

## Agent 0 — Quant Research / Architecture (Lead)

- **Responsibilities:** Own the mathematical spec (`docs/mathematical_foundations.md`),
  freeze the `instruments` + `PricingModel`/`PricingResult` interfaces, set numerical
  conventions (Greek scaling, theta sign, seeds).
- **Inputs:** Project requirements; quant references.
- **Outputs:** `instruments.py`, `pricing/base.py`, `config.py`, math spec.
- **Dependencies:** None (root).
- **Success criteria:** Interfaces import cleanly; spec reviewed; downstream
  agents can stub against the API. **Gate:** no further breaking changes to base
  interfaces after sign-off.

## Agent 1 — Pricing Engine

- **Responsibilities:** Implement Black-Scholes (closed form) and the CRR binomial
  tree (European + American), vectorized with NumPy.
- **Inputs:** A0 interfaces + math spec.
- **Outputs:** `pricing/black_scholes.py`, `pricing/binomial.py`.
- **Dependencies:** A0.
- **Success criteria:** Put-call parity holds to 1e-9; binomial converges to BS
  to <5e-3 at 2000 steps; American ≥ European.

## Agent 2 — Monte Carlo

- **Responsibilities:** GBM simulation engine with antithetic + control variates;
  the `MonteCarloModel` with standard-error / CI reporting.
- **Inputs:** A0 interfaces.
- **Outputs:** `simulations/paths.py`, `pricing/monte_carlo.py`.
- **Dependencies:** A0.
- **Success criteria:** True price inside 95% CI; control variate strictly reduces
  SE; reproducible under fixed seed.

## Agent 3 — Greeks & Analytics

- **Responsibilities:** Analytic BS Greeks; model-agnostic finite-difference Greeks
  (with common-random-numbers note for MC); sensitivity/scenario tooling.
- **Inputs:** A1/A2 models, A0 interfaces.
- **Outputs:** `greeks/*`, `analytics/sensitivity.py`.
- **Dependencies:** A0, A1 (for FD Greeks targets).
- **Success criteria:** Analytic ≈ numerical Greeks within tolerance;
  monotonicity invariants in sensitivity tables.

## Agent 4 — Dashboard

- **Responsibilities:** Streamlit + Plotly UI: model selection, parameter controls,
  real-time pricing, Greeks, sensitivity charts, heatmaps, scenario ladder,
  performance tab, CSV downloads.
- **Inputs:** All engine packages.
- **Outputs:** `dashboard/app.py`.
- **Dependencies:** A1, A2, A3, A5 (benchmarks).
- **Success criteria:** `streamlit run` works; all tabs render; downloads produce
  valid CSVs.

## Agent 5 — Testing & Validation

- **Responsibilities:** Unit, property-based, convergence, and integration tests;
  the benchmarking harness.
- **Inputs:** All engine packages.
- **Outputs:** `tests/*`, `benchmarks/runner.py`, `benchmarks/run_benchmarks.py`.
- **Dependencies:** A1, A2, A3.
- **Success criteria:** `pytest` green; coverage of every model + invariant;
  benchmark report runs.

## Agent 6 — Documentation

- **Responsibilities:** Math derivations, architecture doc, README, this plan;
  keep docs in sync as APIs evolve.
- **Inputs:** Outputs from all agents.
- **Outputs:** `docs/*`, `README.md`.
- **Dependencies:** Continuous; finalizes after integration.
- **Success criteria:** Every public module documented; formulas render; quickstart
  reproduces results.

---

## Integration gates

1. **Interface freeze (after A0):** base API + instruments locked.
2. **Model parity (after A1/A2/A3):** cross-model agreement + Greeks tests pass.
3. **System integration (after A4/A5):** dashboard runs, full `pytest` green,
   benchmark report generated. Use `hive-mind_consensus` to confirm before release.

## Parallelization summary

A1, A2, A3 (mostly), and A6 run concurrently once A0 freezes interfaces. A5
develops tests against the frozen interfaces in parallel and tightens them as
implementations land. A4 is the final consumer. This maps cleanly onto Ruflo's
parallel task queue with a hierarchical coordinator.
