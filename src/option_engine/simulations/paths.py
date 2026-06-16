r"""Geometric Brownian motion path generation under the risk-neutral measure.

Under the risk-neutral measure the underlying follows

.. math::

    dS_t = (r - q) S_t\, dt + \sigma S_t\, dW_t,

whose exact solution lets us sample the terminal price in a single step:

.. math::

    S_T = S_0 \exp\!\Big[(r - q - \tfrac12\sigma^2) T + \sigma\sqrt{T}\, Z\Big],
    \quad Z \sim \mathcal N(0, 1).

Full multi-step paths use the same exact log-Euler increment per step, which is
unbiased for GBM (no discretization error). Antithetic sampling is supported by
mirroring the normal draws :math:`Z \to -Z`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parameters controlling a simulation run."""

    n_paths: int = 100_000
    n_steps: int = 1
    seed: int | None = 12345
    antithetic: bool = True

    def __post_init__(self) -> None:
        if self.n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")


def _draw_normals(
    rng: np.random.Generator, shape: tuple[int, ...], antithetic: bool
) -> np.ndarray:
    """Draw standard normals, optionally antithetic along the path axis.

    With antithetic sampling the first half of the paths are drawn i.i.d. and the
    second half are their negation, halving the variance of symmetric payoffs.
    """
    n_paths = shape[0]
    if not antithetic:
        return rng.standard_normal(shape)
    half = (n_paths + 1) // 2
    base = rng.standard_normal((half, *shape[1:]))
    z = np.concatenate([base, -base], axis=0)
    return z[:n_paths]


def terminal_prices(
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    config: SimulationConfig,
) -> np.ndarray:
    """Sample terminal underlying prices ``S_T`` (exact, single-step GBM)."""
    rng = np.random.default_rng(config.seed)
    z = _draw_normals(rng, (config.n_paths,), config.antithetic)
    drift = (rate - dividend_yield - 0.5 * volatility**2) * maturity
    diffusion = volatility * np.sqrt(maturity) * z
    return spot * np.exp(drift + diffusion)


class GBMSimulator:
    """Generate GBM sample paths under the risk-neutral measure."""

    def __init__(
        self,
        spot: float,
        rate: float,
        dividend_yield: float,
        volatility: float,
        maturity: float,
    ) -> None:
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.volatility = volatility
        self.maturity = maturity

    def terminal(self, config: SimulationConfig) -> np.ndarray:
        """Sample only the terminal price (fast path for European payoffs)."""
        return terminal_prices(
            self.spot,
            self.rate,
            self.dividend_yield,
            self.volatility,
            self.maturity,
            config,
        )

    def paths(self, config: SimulationConfig) -> np.ndarray:
        """Return a ``(n_paths, n_steps + 1)`` array of simulated prices.

        Column 0 is ``spot``; each subsequent column is one exact GBM increment.
        Useful for path-dependent payoffs (Asian, barrier) added later.
        """
        rng = np.random.default_rng(config.seed)
        dt = self.maturity / config.n_steps
        drift = (self.rate - self.dividend_yield - 0.5 * self.volatility**2) * dt
        vol_step = self.volatility * np.sqrt(dt)

        z = _draw_normals(rng, (config.n_paths, config.n_steps), config.antithetic)
        log_increments = drift + vol_step * z
        log_paths = np.cumsum(log_increments, axis=1)
        prices = self.spot * np.exp(log_paths)
        return np.column_stack([np.full(config.n_paths, self.spot), prices])
