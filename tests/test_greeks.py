"""Tests for analytic and numerical Greeks."""

from __future__ import annotations

import math

import pytest

from option_engine import (
    BinomialTreeModel,
    BlackScholesModel,
    black_scholes_greeks,
    numerical_greeks,
)
from option_engine.instruments import OptionContract, OptionType


def test_analytic_matches_numerical_bs(atm_call, market):
    analytic = black_scholes_greeks(atm_call, market)
    numeric = numerical_greeks(BlackScholesModel(), atm_call, market)
    assert numeric.delta == pytest.approx(analytic.delta, abs=1e-4)
    assert numeric.gamma == pytest.approx(analytic.gamma, abs=1e-4)
    assert numeric.vega == pytest.approx(analytic.vega, abs=1e-2)
    assert numeric.theta == pytest.approx(analytic.theta, abs=1e-2)
    assert numeric.rho == pytest.approx(analytic.rho, abs=1e-2)


def test_call_delta_in_unit_interval(atm_call, market):
    g = black_scholes_greeks(atm_call, market)
    assert 0.0 <= g.delta <= 1.0


def test_put_delta_negative(atm_put, market):
    g = black_scholes_greeks(atm_put, market)
    assert -1.0 <= g.delta <= 0.0


def test_gamma_and_vega_positive(atm_call, market):
    g = black_scholes_greeks(atm_call, market)
    assert g.gamma > 0
    assert g.vega > 0


def test_delta_put_call_relationship(atm_call, atm_put, market):
    # delta_call - delta_put = e^{-qT}  (here q=0 so = 1).
    gc = black_scholes_greeks(atm_call, market)
    gp = black_scholes_greeks(atm_put, market)
    factor = math.exp(-market.dividend_yield * atm_call.maturity)
    assert gc.delta - gp.delta == pytest.approx(factor, abs=1e-9)


def test_gamma_equal_for_call_and_put(atm_call, atm_put, market):
    assert black_scholes_greeks(atm_call, market).gamma == pytest.approx(
        black_scholes_greeks(atm_put, market).gamma, abs=1e-12
    )


def test_binomial_numerical_greeks_match_analytic(atm_call, market):
    """Finite-difference Greeks on a fine lattice should approximate BS Greeks."""
    analytic = black_scholes_greeks(atm_call, market)
    numeric = numerical_greeks(BinomialTreeModel(steps=1500), atm_call, market)
    assert numeric.delta == pytest.approx(analytic.delta, abs=1e-2)
    assert numeric.vega == pytest.approx(analytic.vega, abs=0.5)
