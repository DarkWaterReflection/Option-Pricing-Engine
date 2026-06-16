r"""Model-agnostic numerical (finite-difference) Greeks.

Works with **any** :class:`~option_engine.pricing.base.PricingModel` by bumping
the relevant input and re-pricing. Central differences are used throughout for
:math:`O(h^2)` accuracy:

.. math::

    \Delta \approx \frac{V(S+h) - V(S-h)}{2h}, \qquad
    \Gamma \approx \frac{V(S+h) - 2V(S) + V(S-h)}{h^2}.

Theta follows the standard sign convention :math:`\Theta = -\partial V/\partial T`
(value lost as maturity shortens, per year).

.. note::
    For Monte Carlo, pass a model constructed with a **fixed seed** so that the
    bumped re-pricings share random draws (common random numbers). This cancels
    simulation noise between the up/down evaluations and makes finite-difference
    Greeks stable; otherwise the differences are dominated by Monte Carlo error.
"""

from __future__ import annotations

from dataclasses import replace

from option_engine.config import DEFAULT_CONFIG
from option_engine.greeks.types import Greeks
from option_engine.instruments import MarketData, OptionContract
from option_engine.pricing.base import PricingModel


def numerical_greeks(
    model: PricingModel,
    contract: OptionContract,
    market: MarketData,
    *,
    rel_bump: float = DEFAULT_CONFIG.fd_rel_bump,
    vol_bump: float = DEFAULT_CONFIG.fd_vol_bump,
    time_bump: float = DEFAULT_CONFIG.fd_time_bump,
    rate_bump: float = DEFAULT_CONFIG.fd_rate_bump,
) -> Greeks:
    """Estimate Greeks for ``model`` via central finite differences."""

    def price(c: OptionContract, m: MarketData) -> float:
        return model.price(c, m).price

    base = price(contract, market)

    # --- Delta & Gamma (bump spot) ---
    h_s = max(rel_bump * contract.spot, 1e-8)
    up_s = replace(contract, spot=contract.spot + h_s)
    down_s = replace(contract, spot=contract.spot - h_s)
    v_up_s = price(up_s, market)
    v_down_s = price(down_s, market)
    delta = (v_up_s - v_down_s) / (2.0 * h_s)
    gamma = (v_up_s - 2.0 * base + v_down_s) / (h_s * h_s)

    # --- Vega (bump volatility) ---
    m_up_v = replace(market, volatility=market.volatility + vol_bump)
    m_down_v = replace(market, volatility=market.volatility - vol_bump)
    vega = (price(contract, m_up_v) - price(contract, m_down_v)) / (2.0 * vol_bump)

    # --- Rho (bump rate) ---
    m_up_r = replace(market, rate=market.rate + rate_bump)
    m_down_r = replace(market, rate=market.rate - rate_bump)
    rho = (price(contract, m_up_r) - price(contract, m_down_r)) / (2.0 * rate_bump)

    # --- Theta (bump maturity); Theta = -dV/dT ---
    h_t = min(time_bump, 0.5 * contract.maturity)
    c_up_t = replace(contract, maturity=contract.maturity + h_t)
    c_down_t = replace(contract, maturity=contract.maturity - h_t)
    theta = -(price(c_up_t, market) - price(c_down_t, market)) / (2.0 * h_t)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
