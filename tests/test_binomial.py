"""Tests for the CRR binomial tree model."""

from __future__ import annotations

import pytest

from option_engine import BinomialTreeModel, BlackScholesModel
from option_engine.instruments import ExerciseStyle, MarketData, OptionContract, OptionType


def test_converges_to_black_scholes_call(atm_call, market):
    bs = BlackScholesModel().value(atm_call, market)
    tree = BinomialTreeModel(steps=2000).value(atm_call, market)
    assert tree == pytest.approx(bs, abs=5e-3)


def test_converges_to_black_scholes_put(atm_put, market):
    bs = BlackScholesModel().value(atm_put, market)
    tree = BinomialTreeModel(steps=2000).value(atm_put, market)
    assert tree == pytest.approx(bs, abs=5e-3)


def test_convergence_improves_with_steps(atm_call, market):
    bs = BlackScholesModel().value(atm_call, market)
    err_coarse = abs(BinomialTreeModel(steps=50).value(atm_call, market) - bs)
    err_fine = abs(BinomialTreeModel(steps=1000).value(atm_call, market) - bs)
    assert err_fine < err_coarse


def test_american_put_premium_over_european(market):
    """American put should be worth at least as much as the European put."""
    euro = OptionContract(
        spot=100, strike=110, maturity=1, option_type=OptionType.PUT,
        exercise=ExerciseStyle.EUROPEAN,
    )
    amer = OptionContract(
        spot=100, strike=110, maturity=1, option_type=OptionType.PUT,
        exercise=ExerciseStyle.AMERICAN,
    )
    model = BinomialTreeModel(steps=500)
    assert model.value(amer, market) >= model.value(euro, market)


def test_american_call_no_dividends_equals_european(atm_call, market):
    """Without dividends an American call equals the European call (never exercise early)."""
    amer = OptionContract(
        spot=100, strike=100, maturity=1, option_type=OptionType.CALL,
        exercise=ExerciseStyle.AMERICAN,
    )
    model = BinomialTreeModel(steps=1000)
    assert model.value(amer, market) == pytest.approx(model.value(atm_call, market), abs=1e-6)


def test_invalid_steps():
    with pytest.raises(ValueError):
        BinomialTreeModel(steps=0)
