r"""Closed-form Black-Scholes-Merton Greeks (with continuous dividend yield).

With :math:`\phi` the standard-normal pdf, :math:`N` its cdf, and
:math:`d_1, d_2` as in :mod:`option_engine.pricing.black_scholes`:

.. math::

    \Delta_{\text{call}} &= e^{-qT} N(d_1), &
    \Delta_{\text{put}}  &= -e^{-qT} N(-d_1), \\
    \Gamma &= \frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt T}, &
    \mathcal V &= S e^{-qT}\phi(d_1)\sqrt T, \\
    \rho_{\text{call}} &= K T e^{-rT} N(d_2), &
    \rho_{\text{put}}  &= -K T e^{-rT} N(-d_2).

Theta (per year):

.. math::

    \Theta_{\text{call}} = -\frac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt T}
        - rK e^{-rT} N(d_2) + qS e^{-qT} N(d_1).
"""

from __future__ import annotations

import math

from scipy.stats import norm

from option_engine.greeks.types import Greeks
from option_engine.instruments import MarketData, OptionContract
from option_engine.pricing.black_scholes import _d1_d2


def black_scholes_greeks(contract: OptionContract, market: MarketData) -> Greeks:
    """Return analytic Greeks for a European option (raw per-unit scaling)."""
    inp = _d1_d2(contract, market)
    s, k, t = contract.spot, contract.strike, contract.maturity
    r, sigma, q = market.rate, market.volatility, market.dividend_yield

    pdf_d1 = norm.pdf(inp.d1)
    sqrt_t = math.sqrt(t)

    gamma = inp.carry * pdf_d1 / (s * sigma * sqrt_t)
    vega = s * inp.carry * pdf_d1 * sqrt_t
    common_theta = -(s * inp.carry * pdf_d1 * sigma) / (2.0 * sqrt_t)

    if contract.is_call:
        delta = inp.carry * norm.cdf(inp.d1)
        theta = (
            common_theta
            - r * k * inp.discount * norm.cdf(inp.d2)
            + q * s * inp.carry * norm.cdf(inp.d1)
        )
        rho = k * t * inp.discount * norm.cdf(inp.d2)
    else:
        delta = -inp.carry * norm.cdf(-inp.d1)
        theta = (
            common_theta
            + r * k * inp.discount * norm.cdf(-inp.d2)
            - q * s * inp.carry * norm.cdf(-inp.d1)
        )
        rho = -k * t * inp.discount * norm.cdf(-inp.d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
