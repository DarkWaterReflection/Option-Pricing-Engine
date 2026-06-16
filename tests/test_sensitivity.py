"""Tests for the analytics / sensitivity module."""

from __future__ import annotations

import numpy as np

from option_engine import BlackScholesModel
from option_engine.analytics import (
    Parameter,
    greeks_profile,
    price_vs_parameter,
    scenario_grid,
    two_factor_surface,
)


def test_price_increases_with_spot_for_call(atm_call, market):
    spots = np.linspace(50, 150, 11)
    df = price_vs_parameter(BlackScholesModel(), atm_call, market, Parameter.SPOT, spots)
    assert df["price"].is_monotonic_increasing


def test_price_increases_with_volatility(atm_call, market):
    vols = np.linspace(0.05, 0.6, 12)
    df = price_vs_parameter(
        BlackScholesModel(), atm_call, market, Parameter.VOLATILITY, vols
    )
    assert df["price"].is_monotonic_increasing


def test_two_factor_surface_shape(atm_call, market):
    spots = np.linspace(80, 120, 5)
    vols = np.linspace(0.1, 0.4, 4)
    surf = two_factor_surface(
        BlackScholesModel(), atm_call, market,
        Parameter.SPOT, spots, Parameter.VOLATILITY, vols,
    )
    assert surf.shape == (4, 5)


def test_greeks_profile_columns(atm_call, market):
    spots = np.linspace(80, 120, 6)
    df = greeks_profile(BlackScholesModel(), atm_call, market, Parameter.SPOT, spots)
    assert {"price", "delta", "gamma", "vega", "theta", "rho"}.issubset(df.columns)


def test_scenario_grid_zero_shock_is_zero(atm_call, market):
    spot_shocks = np.array([0.9, 1.0, 1.1])
    vol_shocks = np.array([0.8, 1.0, 1.2])
    grid = scenario_grid(BlackScholesModel(), atm_call, market, spot_shocks, vol_shocks)
    # The unshocked centre cell (1.0, 1.0) must be ~0 change.
    assert abs(grid.loc["+0%", "+0%"]) < 1e-9
