"""Shared Greeks container.

Scaling conventions are documented per-field. We store **raw** (per-unit)
sensitivities so analytic and numerical Greeks are directly comparable; the
dashboard applies display scaling (vega per 1% vol, theta per calendar day).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Greeks:
    """First- and second-order option sensitivities.

    Attributes
    ----------
    delta:
        :math:`\\partial V/\\partial S` -- change in value per unit change in spot.
    gamma:
        :math:`\\partial^2 V/\\partial S^2` -- change in delta per unit spot.
    vega:
        :math:`\\partial V/\\partial \\sigma` per **1.0** of volatility
        (i.e. per 100 vol points). Divide by 100 for per-1%-vol.
    theta:
        :math:`\\partial V/\\partial t` per **year** (negative for long options,
        typically). Divide by 365 for per-calendar-day.
    rho:
        :math:`\\partial V/\\partial r` per **1.0** of rate (per 100bp x 100).
        Divide by 100 for per-1%-rate.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }

    def scaled(self) -> dict[str, float]:
        """Return display-friendly Greeks (vega per 1% vol, theta per day, rho per 1%)."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega / 100.0,
            "theta": self.theta / 365.0,
            "rho": self.rho / 100.0,
        }
