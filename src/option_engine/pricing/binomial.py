r"""Cox-Ross-Rubinstein (CRR) binomial tree pricing.

The continuous process is approximated by a recombining lattice with ``N`` steps
of size :math:`\Delta t = T/N`. With

.. math::

    u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad
    p = \frac{e^{(r-q)\Delta t} - d}{u - d},

the option is valued by backward induction, discounting expected continuation
values at the risk-free rate. For American options each node also takes the
maximum of continuation value and immediate intrinsic value (early exercise).

The implementation is fully vectorized with NumPy: each backward step operates
on a shrinking 1-D array of node values, giving :math:`O(N^2)` work with
:math:`O(N)` memory. Convergence to Black-Scholes is :math:`O(1/N)` and is
verified in the test suite.
"""

from __future__ import annotations

import math

import numpy as np

from option_engine.config import DEFAULT_CONFIG
from option_engine.instruments import ExerciseStyle, MarketData, OptionContract
from option_engine.pricing.base import PricingModel, PricingResult


class BinomialTreeModel(PricingModel):
    """CRR lattice pricer supporting European and American exercise."""

    name = "Binomial Tree (CRR)"
    supported_styles = frozenset({ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN})

    def __init__(self, steps: int = DEFAULT_CONFIG.binomial_steps) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.steps = int(steps)

    def price(self, contract: OptionContract, market: MarketData) -> PricingResult:
        self._check_supported(contract)
        n = self.steps
        s, k, t = contract.spot, contract.strike, contract.maturity
        r, sigma, q = market.rate, market.volatility, market.dividend_yield

        dt = t / n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = (math.exp((r - q) * dt) - d) / (u - d)

        if not 0.0 <= p <= 1.0:
            # Numerically possible when sigma is tiny relative to drift*sqrt(dt).
            raise ValueError(
                f"risk-neutral probability p={p:.4f} outside [0, 1]; "
                "increase steps or check volatility/rate inputs"
            )

        # Terminal asset prices: S * u^j * d^(n-j) for j = 0..n.
        j = np.arange(n + 1)
        spot_terminal = s * u**j * d ** (n - j)
        sign = contract.option_type.sign
        values = np.maximum(sign * (spot_terminal - k), 0.0)

        is_american = contract.exercise is ExerciseStyle.AMERICAN
        # Backward induction.
        for step in range(n, 0, -1):
            values = disc * (p * values[1:step + 1] + (1.0 - p) * values[:step])
            if is_american:
                j_step = np.arange(step)
                spot_step = s * u**j_step * d ** (step - 1 - j_step)
                intrinsic = np.maximum(sign * (spot_step - k), 0.0)
                values = np.maximum(values, intrinsic)

        return PricingResult(
            price=float(values[0]),
            model=self.name,
            diagnostics={
                "steps": n,
                "u": u,
                "d": d,
                "p": p,
                "exercise": contract.exercise.value,
            },
        )
