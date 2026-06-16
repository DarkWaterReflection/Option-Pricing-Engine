"""End-to-end integration tests across all three models and the benchmark harness."""

from __future__ import annotations

import pytest

from option_engine import BinomialTreeModel, BlackScholesModel, MonteCarloModel
from option_engine.benchmarks import accuracy_vs_blackscholes, benchmark_models
from option_engine.instruments import MarketData, OptionContract, OptionType


@pytest.mark.integration
@pytest.mark.parametrize("opt_type", [OptionType.CALL, OptionType.PUT])
def test_all_models_agree(opt_type, market):
    contract = OptionContract(spot=100, strike=95, maturity=0.75, option_type=opt_type)
    bs = BlackScholesModel().value(contract, market)
    tree = BinomialTreeModel(steps=2000).value(contract, market)
    mc = MonteCarloModel(n_paths=400_000, seed=11, control_variate=True).value(
        contract, market
    )
    assert tree == pytest.approx(bs, abs=1e-2)
    assert mc == pytest.approx(bs, abs=2e-2)


@pytest.mark.integration
def test_benchmark_runner_columns(atm_call, market):
    df = benchmark_models(
        [BlackScholesModel(), BinomialTreeModel(steps=200), MonteCarloModel(n_paths=20_000)],
        atm_call,
        market,
        repeats=2,
    )
    assert {"model", "price", "abs_error", "mean_runtime_ms", "peak_memory_kb"}.issubset(
        df.columns
    )
    assert len(df) == 3
    # Black-Scholes should be the reference, hence ~zero error.
    bs_row = df[df["model"] == "Black-Scholes"].iloc[0]
    assert bs_row["abs_error"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.integration
def test_binomial_convergence_study(atm_call, market):
    df = accuracy_vs_blackscholes(
        BinomialTreeModel(),
        atm_call,
        market,
        resolutions=[25, 100, 400],
        factory=lambda n: BinomialTreeModel(steps=n),
    )
    # Error should shrink as steps increase.
    assert df["abs_error"].iloc[-1] < df["abs_error"].iloc[0]
