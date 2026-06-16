"""Benchmarking harness for accuracy, runtime, memory, and scalability."""

from __future__ import annotations

from option_engine.benchmarks.runner import (
    BenchmarkResult,
    accuracy_vs_blackscholes,
    benchmark_models,
    scalability_curve,
)

__all__ = [
    "BenchmarkResult",
    "benchmark_models",
    "accuracy_vs_blackscholes",
    "scalability_curve",
]
