"""Configuration management and structured logging.

Centralizing tunables (default lattice steps, Monte Carlo path counts, finite
difference bumps, seeds) keeps numerical behaviour reproducible and makes the
engine easy to configure from a dashboard, CLI, or test fixture.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

_LOG_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler once.

    The log level is taken from the ``OPTION_ENGINE_LOGLEVEL`` environment
    variable (default ``WARNING``) so that library use stays quiet by default
    but is trivially verbose for debugging.
    """
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        level = os.environ.get("OPTION_ENGINE_LOGLEVEL", "WARNING").upper()
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Numerical defaults shared across models.

    Attributes
    ----------
    binomial_steps:
        Default number of time steps in the CRR lattice.
    mc_paths:
        Default number of Monte Carlo sample paths.
    mc_seed:
        Default RNG seed for reproducible simulation.
    mc_antithetic:
        Enable antithetic variates by default.
    mc_control_variate:
        Enable the Black-Scholes control variate by default (European only).
    fd_rel_bump:
        Relative finite-difference bump for numerical Greeks (fraction of the
        parameter value). Absolute bumps are derived per-parameter.
    """

    binomial_steps: int = 500
    mc_paths: int = 100_000
    mc_seed: int | None = 12345
    mc_antithetic: bool = True
    mc_control_variate: bool = True
    fd_rel_bump: float = 1e-4
    fd_vol_bump: float = 1e-4
    fd_time_bump: float = 1.0 / 365.0
    fd_rate_bump: float = 1e-4

    extra: dict[str, object] = field(default_factory=dict)


DEFAULT_CONFIG = EngineConfig()
