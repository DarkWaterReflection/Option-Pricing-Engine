# Mathematical Foundations

This document derives the three pricing methodologies and the Greeks. Notation:

| Symbol | Meaning |
| --- | --- |
| $S$ | Spot price of the underlying |
| $K$ | Strike price |
| $T$ | Time to maturity (years) |
| $r$ | Continuously-compounded risk-free rate |
| $q$ | Continuous dividend / carry yield |
| $\sigma$ | Annualized volatility |
| $N(\cdot)$ | Standard normal CDF; $\phi(\cdot)$ its PDF |
| $\omega$ | $+1$ for a call, $-1$ for a put |

---

## 1. Black-Scholes-Merton

### 1.1 The model

Under the risk-neutral measure $\mathbb{Q}$, the underlying follows geometric
Brownian motion with drift equal to the cost of carry $r-q$:

$$ dS_t = (r-q)\,S_t\,dt + \sigma\,S_t\,dW_t. $$

### 1.2 Derivation (PDE route)

Form a self-financing portfolio $\Pi = V - \Delta S$ long one option and short
$\Delta$ units of the underlying. Applying Itô's lemma to $V(S,t)$:

$$ dV = \frac{\partial V}{\partial t}dt + \frac{\partial V}{\partial S}dS + \tfrac12 \frac{\partial^2 V}{\partial S^2}\sigma^2 S^2\,dt. $$

Choosing $\Delta = \partial V/\partial S$ eliminates the stochastic $dW$ term, so
the portfolio is instantaneously riskless and must earn $r$. This yields the
**Black-Scholes PDE**:

$$ \frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \tfrac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} - rV = 0. $$

### 1.3 Closed-form solution

Solving the PDE with the terminal payoff $V(S,T)=\max(\omega(S-K),0)$ gives

$$ d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}, $$

$$ \boxed{\,C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)\,}, \qquad \boxed{\,P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)\,}. $$

### 1.4 Put-call parity

$$ C - P = S e^{-qT} - K e^{-rT}. $$

This identity is used as a unit-test invariant and as the Monte Carlo control
variate's theoretical anchor.

### 1.5 Assumptions & limitations

Constant $r,\sigma,q$; continuous frictionless trading; no arbitrage; lognormal
returns; European exercise. Real markets exhibit volatility smiles, jumps, and
transaction costs, which motivate the lattice and simulation models for
extensions (American exercise, path dependence, alternative dynamics).

---

## 2. Binomial Tree (Cox-Ross-Rubinstein)

### 2.1 Lattice construction

Partition $[0,T]$ into $N$ steps of size $\Delta t = T/N$. Over each step the
price moves up by factor $u$ or down by $d$:

$$ u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac1u = e^{-\sigma\sqrt{\Delta t}}. $$

The CRR choice matches the variance of log-returns to $\sigma^2\Delta t$.

### 2.2 Risk-neutral probability

Enforcing that the discounted expected price equals the forward gives

$$ p = \frac{e^{(r-q)\Delta t} - d}{u - d}, \qquad 1-p = \frac{u - e^{(r-q)\Delta t}}{u-d}. $$

### 2.3 Backward induction

Terminal payoffs at node $j$ (with $j$ up-moves): $V_{N,j} = \max(\omega(Su^j d^{N-j} - K),0)$.
Roll back:

$$ V_{n,j} = e^{-r\Delta t}\big(p\,V_{n+1,j+1} + (1-p)\,V_{n+1,j}\big). $$

For **American** options, override each node with the early-exercise value:

$$ V_{n,j} = \max\!\Big(e^{-r\Delta t}\big(p V_{n+1,j+1} + (1-p)V_{n+1,j}\big),\ \max(\omega(S_{n,j}-K),0)\Big). $$

### 2.4 Convergence

The lattice converges to Black-Scholes at rate $O(1/N)$ (with characteristic
even/odd oscillation). The benchmark report confirms the absolute error roughly
halves when $N$ doubles: ~0.020 at 100 steps → ~0.001 at 2000 steps.

---

## 3. Monte Carlo Simulation

### 3.1 Risk-neutral valuation

$$ V = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\big[\max(\omega(S_T - K),0)\big]. $$

### 3.2 Exact terminal sampling

The GBM SDE has the exact solution

$$ S_T = S_0 \exp\!\Big[(r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}\,Z\Big], \quad Z\sim\mathcal N(0,1), $$

so European payoffs need **no time discretization** — the only error is
statistical, decaying as $O(1/\sqrt{M})$ in the number of paths $M$.

### 3.3 Variance reduction

**Antithetic variates.** Pair each draw $Z$ with $-Z$. For symmetric integrands
this cancels odd-order error and reduces variance at no extra path cost.

**Control variates.** The discounted terminal price $X = e^{-rT}S_T$ has known
mean $\mathbb{E}[X] = S_0 e^{-qT}$. With payoff estimator $Y$, the adjusted
estimator

$$ Y^\star = Y - \beta^\star\big(X - \mathbb{E}[X]\big), \qquad \beta^\star = \frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)} $$

is unbiased and has minimal variance. Empirically this cuts the standard error
by a large factor for near-the-money options (verified in `test_monte_carlo.py`).

### 3.4 Uncertainty quantification

Report the standard error $\mathrm{SE} = s/\sqrt{M}$ and the 95% confidence
interval $\bar V \pm 1.96\,\mathrm{SE}$.

---

## 4. Greeks

Raw (per-unit) analytic Greeks under Black-Scholes-Merton:

| Greek | Call | Put |
| --- | --- | --- |
| Delta $\partial V/\partial S$ | $e^{-qT}N(d_1)$ | $-e^{-qT}N(-d_1)$ |
| Gamma $\partial^2 V/\partial S^2$ | $\dfrac{e^{-qT}\phi(d_1)}{S\sigma\sqrt T}$ | same |
| Vega $\partial V/\partial \sigma$ | $S e^{-qT}\phi(d_1)\sqrt T$ | same |
| Rho $\partial V/\partial r$ | $KTe^{-rT}N(d_2)$ | $-KTe^{-rT}N(-d_2)$ |

Theta $\partial V/\partial t$ (per year):

$$ \Theta_{\text{call}} = -\frac{Se^{-qT}\phi(d_1)\sigma}{2\sqrt T} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1), $$
$$ \Theta_{\text{put}} = -\frac{Se^{-qT}\phi(d_1)\sigma}{2\sqrt T} + rKe^{-rT}N(-d_2) - qSe^{-qT}N(-d_1). $$

### 4.1 Numerical approximations

For models without closed forms (lattice, simulation), use central differences:

$$ \Delta \approx \frac{V(S+h)-V(S-h)}{2h}, \quad \Gamma \approx \frac{V(S+h)-2V(S)+V(S-h)}{h^2}, $$

with $O(h^2)$ truncation error and analogous bumps for $\sigma, r, T$. We use the
sign convention $\Theta = -\partial V/\partial T$.

**Monte Carlo Greeks** use *common random numbers* (a fixed seed across bumped
re-pricings) so simulation noise cancels between the up/down evaluations —
without this, finite-difference Greeks are swamped by Monte Carlo error.

`tests/test_greeks.py` verifies analytic and numerical Greeks agree to tight
tolerances, validating both implementations against each other.
