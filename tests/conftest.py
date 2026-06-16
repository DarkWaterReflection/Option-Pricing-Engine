"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from option_engine.instruments import MarketData, OptionContract, OptionType


@pytest.fixture
def atm_call() -> OptionContract:
    """A vanilla at-the-money European call."""
    return OptionContract(spot=100.0, strike=100.0, maturity=1.0, option_type=OptionType.CALL)


@pytest.fixture
def atm_put() -> OptionContract:
    return OptionContract(spot=100.0, strike=100.0, maturity=1.0, option_type=OptionType.PUT)


@pytest.fixture
def market() -> MarketData:
    return MarketData(rate=0.05, volatility=0.2, dividend_yield=0.0)


@pytest.fixture
def market_with_div() -> MarketData:
    return MarketData(rate=0.05, volatility=0.2, dividend_yield=0.03)
