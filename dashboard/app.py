"""Interactive Streamlit dashboard for the Option Pricing Engine.

Run with::

    streamlit run dashboard/app.py

Features: model selection, parameter controls, real-time pricing across all three
models, Greeks display, 1-D sensitivity charts, a 2-D price heatmap, a P&L
scenario ladder, a performance comparison table, and CSV/HTML downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from option_engine import (
    BinomialTreeModel,
    BlackScholesModel,
    MonteCarloModel,
    black_scholes_greeks,
    numerical_greeks,
)
from option_engine.analytics import (
    Parameter,
    price_vs_parameter,
    scenario_grid,
    two_factor_surface,
)
from option_engine.benchmarks import benchmark_models
from option_engine.instruments import (
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
)

st.set_page_config(page_title="Option Pricing Engine", layout="wide", page_icon="📈")
st.title("📈 Option Pricing Engine")
st.caption(
    "Black-Scholes · Binomial Tree (CRR) · Monte Carlo — with Greeks, "
    "sensitivity analysis, and performance benchmarking."
)

# --------------------------------------------------------------------------- #
# Sidebar: contract & market inputs
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Contract")
    spot = st.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0)
    strike = st.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0)
    maturity = st.number_input("Maturity (years, T)", value=1.0, min_value=0.01, step=0.25)
    opt_type = st.radio("Type", [OptionType.CALL, OptionType.PUT], format_func=lambda o: o.value.title())
    exercise = st.radio(
        "Exercise", [ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN],
        format_func=lambda e: e.value.title(),
    )

    st.header("Market")
    rate = st.slider("Risk-free rate (r)", -0.05, 0.20, 0.05, 0.005)
    vol = st.slider("Volatility (σ)", 0.01, 1.00, 0.20, 0.01)
    div = st.slider("Dividend yield (q)", 0.0, 0.15, 0.0, 0.005)

    st.header("Numerical settings")
    steps = st.slider("Binomial steps", 10, 3000, 500, 10)
    paths = st.select_slider(
        "Monte Carlo paths", options=[10_000, 50_000, 100_000, 250_000, 500_000],
        value=100_000,
    )

contract = OptionContract(
    spot=spot, strike=strike, maturity=maturity, option_type=opt_type, exercise=exercise
)
market = MarketData(rate=rate, volatility=vol, dividend_yield=div)

bs_model = BlackScholesModel()
tree_model = BinomialTreeModel(steps=steps)
mc_model = MonteCarloModel(n_paths=int(paths), seed=12345, control_variate=True)

is_european = exercise is ExerciseStyle.EUROPEAN

# --------------------------------------------------------------------------- #
# Pricing summary
# --------------------------------------------------------------------------- #
st.subheader("Pricing")
cols = st.columns(3)

tree_res = tree_model.price(contract, market)
cols[1].metric("Binomial Tree (CRR)", f"{tree_res.price:,.4f}", help=f"{steps} steps")

if is_european:
    bs_res = bs_model.price(contract, market)
    mc_res = mc_model.price(contract, market)
    cols[0].metric("Black-Scholes", f"{bs_res.price:,.4f}", help="Closed form (reference)")
    cols[2].metric(
        "Monte Carlo", f"{mc_res.price:,.4f}",
        delta=f"± {1.96 * mc_res.std_error:,.4f} (95% CI)", delta_color="off",
    )
else:
    cols[0].metric("Black-Scholes", "N/A", help="European exercise only")
    cols[2].metric("Monte Carlo", "N/A", help="European exercise only")
    st.info("American exercise selected — only the binomial tree supports early exercise.")

# --------------------------------------------------------------------------- #
# Greeks
# --------------------------------------------------------------------------- #
st.subheader("Greeks")
if is_european:
    analytic = black_scholes_greeks(contract, market).scaled()
    numeric = numerical_greeks(tree_model, contract, market).scaled()
    greeks_df = pd.DataFrame(
        {"Analytic (Black-Scholes)": analytic, "Numerical (Binomial)": numeric}
    )
    greeks_df.index = ["Delta", "Gamma", "Vega (per 1%)", "Theta (per day)", "Rho (per 1%)"]
    st.dataframe(greeks_df.style.format("{:.4f}"), use_container_width=True)
    st.caption("Vega/Rho shown per 1% move; Theta per calendar day.")
else:
    numeric = numerical_greeks(tree_model, contract, market).scaled()
    st.dataframe(
        pd.DataFrame({"Numerical (Binomial)": numeric}).style.format("{:.4f}"),
        use_container_width=True,
    )

# --------------------------------------------------------------------------- #
# Tabs: sensitivity / heatmap / scenarios / performance
# --------------------------------------------------------------------------- #
tab_sens, tab_heat, tab_scen, tab_perf = st.tabs(
    ["📉 Sensitivity", "🔥 Heatmap", "🎯 Scenarios", "⚡ Performance"]
)

active_model = bs_model if is_european else tree_model

with tab_sens:
    param = st.selectbox(
        "Vary parameter", list(Parameter), format_func=lambda p: p.label, key="sens_param"
    )
    ranges = {
        Parameter.SPOT: np.linspace(0.5 * spot, 1.5 * spot, 60),
        Parameter.VOLATILITY: np.linspace(0.01, 1.0, 60),
        Parameter.MATURITY: np.linspace(0.01, max(2.0, maturity * 2), 60),
        Parameter.RATE: np.linspace(-0.05, 0.20, 60),
    }
    df = price_vs_parameter(active_model, contract, market, param, ranges[param])
    fig = px.line(df, x=param.value, y="price", labels={param.value: param.label, "price": "Option price"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "⬇ Download sensitivity CSV", df.to_csv(index=False),
        file_name=f"sensitivity_{param.value}.csv", mime="text/csv",
    )

with tab_heat:
    spots = np.linspace(0.7 * spot, 1.3 * spot, 40)
    vols = np.linspace(0.05, 0.6, 40)
    surf = two_factor_surface(
        active_model, contract, market, Parameter.SPOT, spots, Parameter.VOLATILITY, vols
    )
    fig = go.Figure(
        go.Heatmap(z=surf.values, x=surf.columns, y=surf.index, colorscale="Viridis",
                   colorbar=dict(title="Price"))
    )
    fig.update_layout(
        height=480, xaxis_title="Spot (S)", yaxis_title="Volatility (σ)",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_scen:
    st.caption("Change in option value under joint spot / volatility shocks.")
    spot_shocks = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
    vol_shocks = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    grid = scenario_grid(active_model, contract, market, spot_shocks, vol_shocks)
    fig = go.Figure(
        go.Heatmap(
            z=grid.values, x=grid.columns, y=grid.index, colorscale="RdYlGn",
            zmid=0, colorbar=dict(title="ΔValue"), text=np.round(grid.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        height=420, xaxis_title="Spot shock", yaxis_title="Vol shock",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "⬇ Download scenario grid CSV", grid.to_csv(),
        file_name="scenario_grid.csv", mime="text/csv",
    )

with tab_perf:
    if is_european:
        models = [bs_model, tree_model, mc_model]
    else:
        models = [tree_model]
    bench = benchmark_models(models, contract, market, repeats=5)
    st.dataframe(
        bench.style.format(
            {
                "price": "{:.4f}", "abs_error": "{:.2e}", "rel_error": "{:.2e}",
                "mean_runtime_ms": "{:.3f}", "min_runtime_ms": "{:.3f}",
                "peak_memory_kb": "{:.1f}",
            }
        ),
        use_container_width=True,
    )
    fig = px.bar(bench, x="model", y="mean_runtime_ms", color="model",
                 labels={"mean_runtime_ms": "Mean runtime (ms)", "model": ""})
    fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "⬇ Download benchmark CSV", bench.to_csv(index=False),
        file_name="benchmark.csv", mime="text/csv",
    )
