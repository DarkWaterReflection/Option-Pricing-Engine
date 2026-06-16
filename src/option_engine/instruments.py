"""Domain model: option contracts and market data.

These immutable value objects are the single source of truth for the inputs to
every pricing model. Keeping them validated and model-agnostic guarantees that
Black-Scholes, the binomial tree, and Monte Carlo are all priced from exactly
the same economic assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    """Right conferred by the option."""

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        """+1 for a call, -1 for a put (the omega/phi term in many formulas)."""
        return 1 if self is OptionType.CALL else -1


class ExerciseStyle(str, Enum):
    """When the option may be exercised."""

    EUROPEAN = "european"  # at maturity only
    AMERICAN = "american"  # at any time up to maturity


@dataclass(frozen=True, slots=True)
class OptionContract:
    """A vanilla option contract.

    Parameters
    ----------
    spot:
        Current price ``S`` of the underlying asset (must be > 0).
    strike:
        Strike price ``K`` (must be > 0).
    maturity:
        Time to expiry ``T`` in years (must be > 0).
    option_type:
        :class:`OptionType.CALL` or :class:`OptionType.PUT`.
    exercise:
        :class:`ExerciseStyle`. The closed-form Black-Scholes model and the
        plain Monte Carlo estimator support European exercise only; the binomial
        tree supports both.
    """

    spot: float
    strike: float
    maturity: float
    option_type: OptionType = OptionType.CALL
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be > 0, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"maturity must be > 0, got {self.maturity}")
        if not isinstance(self.option_type, OptionType):
            raise TypeError("option_type must be an OptionType")
        if not isinstance(self.exercise, ExerciseStyle):
            raise TypeError("exercise must be an ExerciseStyle")

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    def intrinsic_value(self, spot: float | None = None) -> float:
        """Payoff if exercised immediately at ``spot`` (defaults to current spot)."""
        s = self.spot if spot is None else spot
        if self.is_call:
            return max(s - self.strike, 0.0)
        return max(self.strike - s, 0.0)

    def moneyness(self) -> float:
        """Log-moneyness ``ln(S/K)`` -- 0 is at-the-money."""
        return math.log(self.spot / self.strike)


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market environment shared by all models.

    Parameters
    ----------
    rate:
        Continuously-compounded risk-free rate ``r`` (annualized). May be
        negative (e.g. EUR rates) but is bounded for sanity.
    volatility:
        Annualized volatility ``sigma`` of the underlying's returns (> 0).
    dividend_yield:
        Continuous dividend (or carry/convenience) yield ``q`` (annualized).
        Defaults to 0.
    """

    rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.volatility <= 0:
            raise ValueError(f"volatility must be > 0, got {self.volatility}")
        if not -1.0 < self.rate < 1.0:
            raise ValueError(f"rate {self.rate} outside sane bounds (-1, 1)")
        if not -1.0 < self.dividend_yield < 1.0:
            raise ValueError(
                f"dividend_yield {self.dividend_yield} outside sane bounds (-1, 1)"
            )
