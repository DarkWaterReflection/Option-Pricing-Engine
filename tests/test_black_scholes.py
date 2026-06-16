"""Tests for the closed-form Black-Scholes model."""

from __future__ import annotations

import math

import pytest

from option_engine import BlackScholesModel
from option_engine.instruments import (
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
)


def test_known_atm_call_value(atm_call, market):
    # Reference value computed by hand: S=K=100, T=1, r=5%, sigma=20%, q=0.
    price = BlackScholesModel().value(atm_call, market)
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity(atm_call, atm_put, market):
    bs = BlackScholesModel()
    call = bs.value(atm_call, market)
    put = bs.value(atm_put, market)
    s, k, t, r = atm_call.spot, atm_call.strike, atm_call.maturity, market.rate
    # C - P = S - K e^{-rT}  (no dividends).
    assert call - put == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)


def test_put_call_parity_with_dividends(atm_call, atm_put, market_with_div):
    bs = BlackScholesModel()
    call = bs.value(atm_call, market_with_div)
    put = bs.value(atm_put, market_with_div)
    s, k, t = atm_call.spot, atm_call.strike, atm_call.maturity
    r, q = market_with_div.rate, market_with_div.dividend_yield
    lhs = call - put
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_price_above_intrinsic(market):
    itm_call = OptionContract(spot=120.0, strike=100.0, maturity=1.0)
    price = BlackScholesModel().value(itm_call, market)
    assert price >= itm_call.intrinsic_value()


def test_deep_itm_call_approaches_forward(market):
    deep = OptionContract(spot=1000.0, strike=100.0, maturity=1.0)
    price = BlackScholesModel().value(deep, market)
    forward = deep.spot - deep.strike * math.exp(-market.rate * deep.maturity)
    assert price == pytest.approx(forward, rel=1e-6)


def test_american_unsupported():
    bs = BlackScholesModel()
    amer = OptionContract(
        spot=100, strike=100, maturity=1, exercise=ExerciseStyle.AMERICAN
    )
    assert not bs.supports(amer)
    with pytest.raises(NotImplementedError):
        bs.price(amer, MarketData(rate=0.05, volatility=0.2))


@pytest.mark.parametrize("bad", [{"spot": -1}, {"strike": 0}, {"maturity": -0.5}])
def test_invalid_contract(bad):
    base = dict(spot=100.0, strike=100.0, maturity=1.0)
    base.update(bad)
    with pytest.raises(ValueError):
        OptionContract(**base)


def test_invalid_market():
    with pytest.raises(ValueError):
        MarketData(rate=0.05, volatility=-0.2)
