"""Command-line benchmarking script.

Runs the full comparison (accuracy, runtime, memory) plus a convergence study and
a scalability curve, prints tidy tables, and optionally writes CSVs to
``benchmarks/results/``.

Usage::

    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from option_engine import BinomialTreeModel, BlackScholesModel, MonteCarloModel
from option_engine.benchmarks import (
    accuracy_vs_blackscholes,
    benchmark_models,
    scalability_curve,
)
from option_engine.instruments import MarketData, OptionContract, OptionType

pd.set_option("display.float_format", lambda v: f"{v:,.6f}")

CONTRACT = OptionContract(spot=100.0, strike=100.0, maturity=1.0, option_type=OptionType.CALL)
MARKET = MarketData(rate=0.05, volatility=0.2, dividend_yield=0.0)


def main(save: bool) -> None:
    print("=" * 70)
    print("OPTION PRICING ENGINE — BENCHMARK REPORT")
    print("=" * 70)
    print(f"Contract: {CONTRACT}")
    print(f"Market:   {MARKET}\n")

    models = [
        BlackScholesModel(),
        BinomialTreeModel(steps=1000),
        MonteCarloModel(n_paths=200_000, seed=12345, control_variate=True),
    ]
    summary = benchmark_models(models, CONTRACT, MARKET, repeats=10)
    print("--- Accuracy / Runtime / Memory (vs Black-Scholes reference) ---")
    print(summary.to_string(index=False))
    print()

    conv_tree = accuracy_vs_blackscholes(
        BinomialTreeModel(), CONTRACT, MARKET,
        resolutions=[10, 25, 50, 100, 250, 500, 1000, 2000],
        factory=lambda n: BinomialTreeModel(steps=n),
    )
    print("--- Binomial convergence (error vs steps) ---")
    print(conv_tree.to_string(index=False))
    print()

    conv_mc = accuracy_vs_blackscholes(
        MonteCarloModel(), CONTRACT, MARKET,
        resolutions=[1_000, 10_000, 50_000, 100_000, 500_000],
        factory=lambda n: MonteCarloModel(n_paths=n, seed=12345, control_variate=True),
    )
    print("--- Monte Carlo convergence (error vs paths) ---")
    print(conv_mc.to_string(index=False))
    print()

    scale = scalability_curve(
        CONTRACT, MARKET,
        factory=lambda n: BinomialTreeModel(steps=n),
        resolutions=[100, 500, 1000, 2000, 4000], repeats=3,
    )
    print("--- Binomial scalability (runtime vs steps) ---")
    print(scale.to_string(index=False))

    if save:
        out = Path(__file__).parent / "results"
        out.mkdir(exist_ok=True)
        summary.to_csv(out / "summary.csv", index=False)
        conv_tree.to_csv(out / "binomial_convergence.csv", index=False)
        conv_mc.to_csv(out / "monte_carlo_convergence.csv", index=False)
        scale.to_csv(out / "binomial_scalability.csv", index=False)
        print(f"\nSaved CSVs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="write CSV results")
    main(parser.parse_args().save)
