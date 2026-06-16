"""Option Pricing Engine.

A production-quality library for valuing European (and American, via the binomial
tree) options using three independent methodologies:

* :mod:`option_engine.pricing.black_scholes` -- closed-form analytic pricing.
* :mod:`option_engine.pricing.binomial`      -- Cox-Ross-Rubinstein lattice.
* :mod:`option_engine.pricing.monte_carlo`   -- risk-neutral simulation.

The package also provides analytic and numerical Greeks, sensitivity analysis,
and a benchmarking harness. See the top-level ``README.md`` and ``docs/`` for the
mathematical derivations and architecture overview.
"""

from __future__ import annotations

from option_engine.instruments import (
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
)
from option_engine.pricing.base import PricingModel, PricingResult
from option_engine.pricing.black_scholes import BlackScholesModel
from option_engine.pricing.binomial import BinomialTreeModel
from option_engine.pricing.monte_carlo import MonteCarloModel
from option_engine.greeks.analytical import black_scholes_greeks
from option_engine.greeks.numerical import numerical_greeks

__version__ = "0.1.0"

__all__ = [
    "OptionType",
    "ExerciseStyle",
    "OptionContract",
    "MarketData",
    "PricingModel",
    "PricingResult",
    "BlackScholesModel",
    "BinomialTreeModel",
    "MonteCarloModel",
    "black_scholes_greeks",
    "numerical_greeks",
    "__version__",
]
