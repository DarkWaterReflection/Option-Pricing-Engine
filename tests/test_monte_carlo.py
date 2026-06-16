"""Tests for the Monte Carlo model."""

from __future__ import annotations

import pytest

from option_engine import BlackScholesModel, MonteCarloModel


def test_within_confidence_interval_of_bs(atm_call, market):
    bs = BlackScholesModel().value(atm_call, market)
    result = MonteCarloModel(n_paths=200_000, seed=7, control_variate=False).price(
        atm_call, market
    )
    lo, hi = result.confidence_interval
    # The true value should lie inside the 95% CI (allow a tiny tolerance).
    assert lo - 1e-6 <= bs <= hi + 1e-6


def test_control_variate_reduces_std_error(atm_call, market):
    plain = MonteCarloModel(n_paths=50_000, seed=1, antithetic=False, control_variate=False)
    cv = MonteCarloModel(n_paths=50_000, seed=1, antithetic=False, control_variate=True)
    se_plain = plain.price(atm_call, market).std_error
    se_cv = cv.price(atm_call, market).std_error
    assert se_cv < se_plain


def test_antithetic_is_reproducible(atm_call, market):
    m = MonteCarloModel(n_paths=10_000, seed=42)
    assert m.value(atm_call, market) == m.value(atm_call, market)


def test_accuracy_close_to_bs(atm_put, market):
    bs = BlackScholesModel().value(atm_put, market)
    mc = MonteCarloModel(n_paths=500_000, seed=3, control_variate=True).value(atm_put, market)
    assert mc == pytest.approx(bs, abs=1e-2)


def test_requires_minimum_paths():
    with pytest.raises(ValueError):
        MonteCarloModel(n_paths=1)
