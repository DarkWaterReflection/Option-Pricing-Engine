"""Benchmarking: accuracy, runtime, memory, and scalability.

The Black-Scholes closed form is treated as ground truth for European options,
so accuracy is reported as absolute error versus that reference. Runtime uses
``time.perf_counter`` with repeated trials; peak memory uses :mod:`tracemalloc`,
which captures Python-level allocations without external dependencies.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict, dataclass

import pandas as pd

from option_engine.instruments import MarketData, OptionContract
from option_engine.pricing.base import PricingModel
from option_engine.pricing.black_scholes import BlackScholesModel


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Per-model benchmark record."""

    model: str
    price: float
    abs_error: float
    rel_error: float
    mean_runtime_ms: float
    min_runtime_ms: float
    peak_memory_kb: float
    repeats: int


def _time_and_measure(
    model: PricingModel, contract: OptionContract, market: MarketData, repeats: int
) -> tuple[float, list[float], float]:
    """Return (price, list of runtimes in ms, peak memory in KB)."""
    tracemalloc.start()
    price = model.value(contract, market)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtimes_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.value(contract, market)
        runtimes_ms.append((time.perf_counter() - start) * 1e3)
    return price, runtimes_ms, peak / 1024.0


def benchmark_models(
    models: list[PricingModel],
    contract: OptionContract,
    market: MarketData,
    *,
    repeats: int = 5,
    reference: float | None = None,
) -> pd.DataFrame:
    """Benchmark several models on one contract.

    ``reference`` defaults to the analytic Black-Scholes price (European ground
    truth). Returns a DataFrame sorted by mean runtime.
    """
    if reference is None:
        reference = BlackScholesModel().value(contract, market)

    results: list[BenchmarkResult] = []
    for model in models:
        price, runtimes, peak_kb = _time_and_measure(model, contract, market, repeats)
        abs_err = abs(price - reference)
        rel_err = abs_err / abs(reference) if reference else float("nan")
        results.append(
            BenchmarkResult(
                model=model.name,
                price=price,
                abs_error=abs_err,
                rel_error=rel_err,
                mean_runtime_ms=sum(runtimes) / len(runtimes),
                min_runtime_ms=min(runtimes),
                peak_memory_kb=peak_kb,
                repeats=repeats,
            )
        )
    df = pd.DataFrame(asdict(r) for r in results)
    return df.sort_values("mean_runtime_ms").reset_index(drop=True)


def accuracy_vs_blackscholes(
    model: PricingModel,
    contract: OptionContract,
    market: MarketData,
    resolutions: list[int],
    *,
    factory,
) -> pd.DataFrame:
    """Convergence study: error vs a discretization parameter.

    ``factory(resolution) -> PricingModel`` constructs the model at each
    resolution (steps for the lattice, paths for Monte Carlo). Returns a
    DataFrame with columns ``[resolution, price, abs_error]``.
    """
    reference = BlackScholesModel().value(contract, market)
    rows = []
    for res in resolutions:
        m = factory(res)
        price = m.value(contract, market)
        rows.append(
            {"resolution": res, "price": price, "abs_error": abs(price - reference)}
        )
    return pd.DataFrame(rows)


def scalability_curve(
    contract: OptionContract,
    market: MarketData,
    factory,
    resolutions: list[int],
    *,
    repeats: int = 3,
) -> pd.DataFrame:
    """Measure runtime as a function of resolution (steps or paths).

    ``factory(resolution) -> PricingModel``. Returns a DataFrame with columns
    ``[resolution, mean_runtime_ms, peak_memory_kb]`` for trade-off charts.
    """
    rows = []
    for res in resolutions:
        model = factory(res)
        _, runtimes, peak_kb = _time_and_measure(model, contract, market, repeats)
        rows.append(
            {
                "resolution": res,
                "mean_runtime_ms": sum(runtimes) / len(runtimes),
                "peak_memory_kb": peak_kb,
            }
        )
    return pd.DataFrame(rows)
