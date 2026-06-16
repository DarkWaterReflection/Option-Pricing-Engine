"""Pricing models.

All models implement the :class:`~option_engine.pricing.base.PricingModel`
interface so they are interchangeable for sensitivity analysis and benchmarking.
"""

from __future__ import annotations

from option_engine.pricing.base import PricingModel, PricingResult
from option_engine.pricing.black_scholes import BlackScholesModel
from option_engine.pricing.binomial import BinomialTreeModel
from option_engine.pricing.monte_carlo import MonteCarloModel

__all__ = [
    "PricingModel",
    "PricingResult",
    "BlackScholesModel",
    "BinomialTreeModel",
    "MonteCarloModel",
]
