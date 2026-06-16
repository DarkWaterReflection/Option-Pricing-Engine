"""Sensitivity analysis and scenario tooling.

All functions return tidy :class:`pandas.DataFrame` objects so they can feed
straight into tables, Plotly/Matplotlib charts, or be exported to CSV. The four
risk factors the spec calls for -- spot, volatility, time to maturity, and the
risk-free rate -- are addressed by :func:`price_vs_parameter` (1-D),
:func:`two_factor_surface` (2-D heatmap), :func:`greeks_profile`, and
:func:`scenario_grid`.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import numpy as np
import pandas as pd

from option_engine.greeks.numerical import numerical_greeks
from option_engine.instruments import MarketData, OptionContract
from option_engine.pricing.base import PricingModel


class Parameter(str, Enum):
    """Risk factor that can be varied in a sensitivity sweep."""

    SPOT = "spot"
    VOLATILITY = "volatility"
    MATURITY = "maturity"
    RATE = "rate"

    @property
    def label(self) -> str:
        return {
            Parameter.SPOT: "Underlying price (S)",
            Parameter.VOLATILITY: "Volatility (sigma)",
            Parameter.MATURITY: "Time to maturity (T)",
            Parameter.RATE: "Risk-free rate (r)",
        }[self]


def _apply(
    contract: OptionContract, market: MarketData, param: Parameter, value: float
) -> tuple[OptionContract, MarketData]:
    """Return a (contract, market) pair with ``param`` set to ``value``."""
    if param is Parameter.SPOT:
        return replace(contract, spot=value), market
    if param is Parameter.MATURITY:
        return replace(contract, maturity=value), market
    if param is Parameter.VOLATILITY:
        return contract, replace(market, volatility=value)
    if param is Parameter.RATE:
        return contract, replace(market, rate=value)
    raise ValueError(f"unknown parameter {param}")


def price_vs_parameter(
    model: PricingModel,
    contract: OptionContract,
    market: MarketData,
    param: Parameter,
    values: np.ndarray,
) -> pd.DataFrame:
    """Tabulate option price as a single risk factor is swept over ``values``.

    Returns a DataFrame with columns ``[param.value, "price"]``.
    """
    prices = [
        model.value(*_apply(contract, market, param, float(v))) for v in values
    ]
    return pd.DataFrame({param.value: np.asarray(values, dtype=float), "price": prices})


def two_factor_surface(
    model: PricingModel,
    contract: OptionContract,
    market: MarketData,
    x_param: Parameter,
    x_values: np.ndarray,
    y_param: Parameter,
    y_values: np.ndarray,
) -> pd.DataFrame:
    """Compute a price surface over two risk factors (for heatmaps/3-D plots).

    Returns a DataFrame indexed by ``y_values`` with columns ``x_values`` -- a
    matrix ready for :func:`plotly.graph_objects.Heatmap` or ``seaborn.heatmap``.
    """
    if x_param is y_param:
        raise ValueError("x_param and y_param must differ")
    matrix = np.empty((len(y_values), len(x_values)), dtype=float)
    for i, yv in enumerate(y_values):
        c_y, m_y = _apply(contract, market, y_param, float(yv))
        for jx, xv in enumerate(x_values):
            c_xy, m_xy = _apply(c_y, m_y, x_param, float(xv))
            matrix[i, jx] = model.value(c_xy, m_xy)
    return pd.DataFrame(
        matrix,
        index=pd.Index(np.asarray(y_values, dtype=float), name=y_param.value),
        columns=pd.Index(np.asarray(x_values, dtype=float), name=x_param.value),
    )


def greeks_profile(
    model: PricingModel,
    contract: OptionContract,
    market: MarketData,
    param: Parameter,
    values: np.ndarray,
) -> pd.DataFrame:
    """Tabulate price and all Greeks as ``param`` is swept (numerical Greeks)."""
    rows = []
    for v in values:
        c, m = _apply(contract, market, param, float(v))
        g = numerical_greeks(model, c, m)
        rows.append({param.value: float(v), "price": model.value(c, m), **g.as_dict()})
    return pd.DataFrame(rows)


def scenario_grid(
    model: PricingModel,
    contract: OptionContract,
    market: MarketData,
    spot_shocks: np.ndarray,
    vol_shocks: np.ndarray,
) -> pd.DataFrame:
    """Profit-and-loss style scenario matrix under joint spot/vol shocks.

    ``spot_shocks`` and ``vol_shocks`` are multiplicative (e.g. ``0.9`` = -10%).
    Cells contain the option value *change* versus the base case -- the classic
    risk-desk scenario ladder.
    """
    base = model.value(contract, market)
    matrix = np.empty((len(vol_shocks), len(spot_shocks)), dtype=float)
    for i, vs in enumerate(vol_shocks):
        m = replace(market, volatility=market.volatility * float(vs))
        for jx, ss in enumerate(spot_shocks):
            c = replace(contract, spot=contract.spot * float(ss))
            matrix[i, jx] = model.value(c, m) - base
    return pd.DataFrame(
        matrix,
        index=pd.Index([f"{(v - 1) * 100:+.0f}%" for v in vol_shocks], name="vol_shock"),
        columns=pd.Index([f"{(s - 1) * 100:+.0f}%" for s in spot_shocks], name="spot_shock"),
    )
