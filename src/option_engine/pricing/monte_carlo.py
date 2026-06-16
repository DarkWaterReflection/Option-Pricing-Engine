r"""Monte Carlo pricing under the risk-neutral measure.

The price of a European option is the discounted risk-neutral expectation of its
payoff:

.. math::

    V = e^{-rT}\, \mathbb{E}^{\mathbb{Q}}\!\big[\,\text{payoff}(S_T)\,\big].

We estimate the expectation by averaging the payoff over simulated terminal
prices. Two variance-reduction techniques are supported:

* **Antithetic variates** -- pairing :math:`Z` with :math:`-Z` (handled in the
  path generator).
* **Control variates** -- using the underlying's discounted terminal price,
  whose true expectation is known (:math:`S_0 e^{-qT}`), to remove correlated
  noise from the estimator.

The reported standard error is the Monte Carlo standard error of the mean, and
the 95% confidence interval is :math:`\bar V \pm 1.96\,\mathrm{SE}`.
"""

from __future__ import annotations

import math

import numpy as np

from option_engine.config import DEFAULT_CONFIG
from option_engine.instruments import ExerciseStyle, MarketData, OptionContract
from option_engine.pricing.base import PricingModel, PricingResult
from option_engine.simulations.paths import SimulationConfig, terminal_prices

_Z95 = 1.959963984540054  # norm.ppf(0.975)


class MonteCarloModel(PricingModel):
    """Risk-neutral Monte Carlo pricer for European options."""

    name = "Monte Carlo"
    supported_styles = frozenset({ExerciseStyle.EUROPEAN})

    def __init__(
        self,
        n_paths: int = DEFAULT_CONFIG.mc_paths,
        seed: int | None = DEFAULT_CONFIG.mc_seed,
        antithetic: bool = DEFAULT_CONFIG.mc_antithetic,
        control_variate: bool = DEFAULT_CONFIG.mc_control_variate,
    ) -> None:
        if n_paths < 2:
            raise ValueError("n_paths must be >= 2 for a variance estimate")
        self.n_paths = int(n_paths)
        self.seed = seed
        self.antithetic = antithetic
        self.control_variate = control_variate

    def _payoff(self, contract: OptionContract, s_t: np.ndarray) -> np.ndarray:
        sign = contract.option_type.sign
        return np.maximum(sign * (s_t - contract.strike), 0.0)

    def price(self, contract: OptionContract, market: MarketData) -> PricingResult:
        self._check_supported(contract)
        sim_cfg = SimulationConfig(
            n_paths=self.n_paths,
            n_steps=1,
            seed=self.seed,
            antithetic=self.antithetic,
        )
        s_t = terminal_prices(
            contract.spot,
            market.rate,
            market.dividend_yield,
            market.volatility,
            contract.maturity,
            sim_cfg,
        )
        discount = math.exp(-market.rate * contract.maturity)
        discounted_payoff = discount * self._payoff(contract, s_t)

        if self.control_variate:
            samples = self._apply_control_variate(
                discounted_payoff, s_t, contract, market, discount
            )
            cv_used = True
        else:
            samples = discounted_payoff
            cv_used = False

        n = samples.size
        price = float(samples.mean())
        std_error = float(samples.std(ddof=1) / math.sqrt(n))
        ci = (price - _Z95 * std_error, price + _Z95 * std_error)

        return PricingResult(
            price=price,
            model=self.name,
            std_error=std_error,
            confidence_interval=ci,
            diagnostics={
                "n_paths": n,
                "antithetic": self.antithetic,
                "control_variate": cv_used,
                "seed": self.seed,
            },
        )

    @staticmethod
    def _apply_control_variate(
        discounted_payoff: np.ndarray,
        s_t: np.ndarray,
        contract: OptionContract,
        market: MarketData,
        discount: float,
    ) -> np.ndarray:
        r"""Reduce variance using the discounted terminal price as a control.

        The control :math:`X = e^{-rT} S_T` has known mean
        :math:`\mathbb{E}[X] = S_0 e^{-qT}`. The optimal coefficient is
        :math:`\beta^* = \operatorname{Cov}(Y, X)/\operatorname{Var}(X)`, and the
        adjusted sample is :math:`Y - \beta^*(X - \mathbb{E}[X])`.
        """
        control = discount * s_t
        control_mean = contract.spot * math.exp(-market.dividend_yield * contract.maturity)
        var_x = control.var(ddof=1)
        if var_x == 0:
            return discounted_payoff
        cov = np.cov(discounted_payoff, control, ddof=1)[0, 1]
        beta = cov / var_x
        return discounted_payoff - beta * (control - control_mean)
