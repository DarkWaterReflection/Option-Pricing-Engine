r"""Black-Scholes-Merton closed-form pricing.

For a European option on an asset paying a continuous dividend yield :math:`q`,
with spot :math:`S`, strike :math:`K`, maturity :math:`T`, risk-free rate
:math:`r` and volatility :math:`\sigma`:

.. math::

    d_1 &= \frac{\ln(S/K) + (r - q + \tfrac12 \sigma^2) T}{\sigma \sqrt{T}}, \\
    d_2 &= d_1 - \sigma \sqrt{T}, \\
    C &= S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \\
    P &= K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).

Assumptions / limitations: constant :math:`r, \sigma, q`; lognormal underlying;
frictionless markets; European exercise only. See ``docs/mathematical_foundations.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from option_engine.instruments import ExerciseStyle, MarketData, OptionContract
from option_engine.pricing.base import PricingModel, PricingResult


@dataclass(frozen=True, slots=True)
class BlackScholesInputs:
    """Pre-computed ``d1``/``d2`` and discount factors -- reused by the Greeks."""

    d1: float
    d2: float
    discount: float  # e^{-rT}
    carry: float  # e^{-qT}


def _d1_d2(contract: OptionContract, market: MarketData) -> BlackScholesInputs:
    s, k, t = contract.spot, contract.strike, contract.maturity
    r, sigma, q = market.rate, market.volatility, market.dividend_yield
    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return BlackScholesInputs(
        d1=d1,
        d2=d2,
        discount=math.exp(-r * t),
        carry=math.exp(-q * t),
    )


class BlackScholesModel(PricingModel):
    """Analytic Black-Scholes-Merton pricer for European options."""

    name = "Black-Scholes"
    supported_styles = frozenset({ExerciseStyle.EUROPEAN})

    def price(self, contract: OptionContract, market: MarketData) -> PricingResult:
        self._check_supported(contract)
        inp = _d1_d2(contract, market)
        s, k = contract.spot, contract.strike
        if contract.is_call:
            price = s * inp.carry * norm.cdf(inp.d1) - k * inp.discount * norm.cdf(inp.d2)
        else:
            price = k * inp.discount * norm.cdf(-inp.d2) - s * inp.carry * norm.cdf(-inp.d1)

        return PricingResult(
            price=price,
            model=self.name,
            diagnostics={"d1": inp.d1, "d2": inp.d2, "closed_form": True},
        )

    @staticmethod
    def inputs(contract: OptionContract, market: MarketData) -> BlackScholesInputs:
        """Expose the intermediate ``d1``/``d2`` terms (used by analytic Greeks)."""
        return _d1_d2(contract, market)
