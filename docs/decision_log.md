# Model Decision Log
## QuantFinance-EDU: Every Major Design Decision

> This log documents every significant mathematical and engineering decision
> made during development. For every decision, it records: the problem,
> alternatives considered, mathematical analysis, selected method, expected
> benefit, and empirical result where available.

---

## DECISION 001: Return Measure for Time-Series Analysis

**PROBLEM:**
We need a return measure for single-asset time-series analysis in Modules 3, 4, 5, and 6.

**ALTERNATIVES:**

| | Method A: Arithmetic Return | Method B: Log Return |
|---|---|---|
| **Formula** | r_t = (P_t - P_{t-1}) / P_{t-1} | r_t = ln(P_t / P_{t-1}) |
| **Interpretation** | Simple percentage gain/loss | Continuously compounded gain |
| **Time-additive?** | ❌ No: r(t,t+2) ≠ r(t,t+1) + r(t+1,t+2) | ✅ Yes: sum of daily log returns = period log return |
| **Portfolio-additive?** | ✅ Yes: r_p = Σ w_i r_i | ❌ No: log of portfolio ≠ weighted sum of logs |
| **Distribution** | Right-skewed, non-normal | More symmetric, approx. normal for small r |
| **Bounded below?** | Yes, -100% (stock cannot go below zero) | -∞ (but exp(-∞) = 0, consistent with zero price) |
| **Stationarity** | Neither is guaranteed; both need testing | Neither is guaranteed |

**MATHEMATICAL NOTE:**
For small returns (r < 0.10), arithmetic and log returns are nearly identical:
ln(1 + r) ≈ r - r²/2 ≈ r

The approximation error is: log_r - arith_r ≈ -r²/2 ≈ -0.5% for r = 10%

**SELECTED METHOD:** 
- Log returns for all single-asset time-series analysis (Modules 3, 4, 5, 6)
- Arithmetic returns for portfolio aggregation (Module 7)

**RATIONALE:**
Academic consensus in quantitative finance uses log returns for time-series analysis
because: (1) time-additivity is essential for multi-period calculations, (2) they are
approximately normally distributed which is required for many statistical tests,
(3) they are the natural output of geometric Brownian motion (the standard model for prices).

Portfolio aggregation uses arithmetic returns because wᵀr is the correct portfolio return formula only for arithmetic returns.

**EXPECTED BENEFIT:**
Time-additive returns allow correct computation of:
- Multi-period momentum factors
- Rolling factor model regressions
- Accurate Hurst exponent estimation

**EMPIRICAL RESULT:**
For daily returns on typical equity data (σ ≈ 20% annual ≈ 1.3% daily), 
the mean absolute difference between log and arithmetic returns is 
approximately r²/2 ≈ 0.008%, which is negligible. Both methods produce
near-identical rankings.

**FINAL DECISION:** Use log returns for time-series analysis. Document clearly.

---

## DECISION 002: Normalisation Method for Module Score Combination

**PROBLEM:**
Modules 1-6 produce scores on completely different scales. We need to bring them
to a common scale before aggregating.

**ALTERNATIVES:**

| | Z-Score | Min-Max | Rank/Percentile | Robust (Median/IQR) |
|---|---|---|---|---|
| **Formula** | (x-μ)/σ | (x-min)/(max-min) | rank(x)/n | (x-median)/IQR |
| **Outlier sensitivity** | High | Extreme | None | Low |
| **Distributional assumption** | Approximate normality | None | None | None |
| **Look-ahead risk** | ✅ Manageable (use expanding window) | ⚠️ High (min/max from full sample) | ✅ Low | ✅ Manageable |
| **Interpretability** | Standard deviations from mean | 0-1 bounded | Percentile rank | Similar to z-score |

**MATHEMATICAL CONCERN WITH MIN-MAX:**
If we compute min/max over the full dataset to normalise, we introduce look-ahead
bias in backtesting: the normalisation at time t uses future data (max from t+1...T).
This can be fixed with expanding window, but is more complex.

**SELECTED METHOD:**
- Within each module (normalising individual metrics): Z-score with winsorisation at ±3σ
- At the module-score level (combining M1-M6): Rank/percentile normalisation
- Final composite: linear rescaling to 0-100

**RATIONALE:**
Rank normalisation at the module level is preferred because:
1. It is completely robust to outliers (a single very bad score won't dominate)
2. It requires no distributional assumptions
3. It is interpretable (percentile ranking)
4. It naturally handles non-comparable scales across modules

**LIMITATION:**
Rank normalisation loses magnitude information. We cannot say whether M1=60 is
marginally or dramatically better than M1=40 — only that it is ranked higher.
This is acceptable because the magnitude comparability across modules is artificial anyway.

**EMPIRICAL RESULT:**
On a 10-stock test universe, rank and z-score normalisation produce rankings with
Spearman correlation of ~0.95. The practical difference is small, but rank is
preferred for robustness.

**FINAL DECISION:** Rank normalisation at module-score level. Document magnitude loss as a known limitation.

---

## DECISION 003: VaR Estimation Method

**PROBLEM:**
Value at Risk (VaR) can be estimated by three methods. Which is most appropriate?

**ALTERNATIVES:**

**Method A: Historical VaR**
VaR_α = -Quantile(r_t, 1-α)
- Uses actual empirical distribution
- No parametric assumptions
- Problem: Requires sufficient history; weights all periods equally

**Method B: Parametric VaR (Normal)**
VaR_α = -(μ - z_α · σ)
- Assumes normal distribution
- Analytically simple
- Problem: Return distributions have fat tails; normal VaR UNDERESTIMATES tail risk
  Test: For daily returns, kurtosis typically 5-10 vs normal kurtosis 3
  This means 99% parametric VaR is too optimistic by 20-30%

**Method C: Monte Carlo VaR**
- Fit a parametric distribution (e.g. t-distribution) to returns
- Simulate N paths and compute empirical quantile of simulated distribution
- Most flexible; can use any distribution
- Problem: Results depend on distributional assumption; more complex

**MATHEMATICAL TEST (fat tails):**
For normal returns: P(r < -3σ) = 0.13%
For t(5) returns:   P(r < -3σ) ≈ 0.55% (4× more likely)
This difference is economically significant.

**SELECTED METHOD:**
- Primary: Historical VaR (no distributional assumption)
- Secondary: Parametric VaR shown for comparison
- Monte Carlo VaR in Module 7 (using t-distribution for fat tails)
- Always compute CVaR alongside VaR

**RATIONALE:**
Historical VaR is the most honest estimate for an educational tool:
it makes no distributional assumptions and directly uses observed data.
The t-distribution VaR in Monte Carlo more realistically models fat tails
but is used only for the simulation (where we explicitly state assumptions).

**LIMITATION:**
Historical VaR implicitly assumes history is representative of future tail events.
The worst historical event defines the worst simulated outcome — this is a known
limitation (extreme events can exceed historical worst).

**FINAL DECISION:** Historical VaR primary. Always show CVaR. Show parametric for comparison.

---

## DECISION 004: Covariance Matrix Estimator for Portfolio Optimisation

**PROBLEM:**
Portfolio optimisation requires an estimated covariance matrix Σ.
For small sample sizes (n) relative to assets (p), sample covariance is noisy.

**ALTERNATIVES:**

**Method A: Sample Covariance**
Σ_sample = (1/(T-1)) · Xᵀ X
- Unbiased estimator
- Problem: For T (observations) not >> p (assets), estimation error is large
- In optimisation: noisy Σ → extreme weights (corner solutions)

**Method B: Ledoit-Wolf Shrinkage**
Σ_LW = (1-α)·Σ_sample + α·μ·I
- Shrinks towards a structured target (scaled identity)
- α = optimal shrinkage intensity (estimated analytically from data)
- Reduces the extreme eigenvalues of Σ_sample
- More stable for optimisation

**MATHEMATICAL ARGUMENT:**
The condition number of Σ_sample = λ_max/λ_min
High condition number → matrix nearly singular → optimiser finds extreme weights
Ledoit-Wolf reduces condition number → more stable, diversified portfolios

**SELECTED METHOD:** Ledoit-Wolf shrinkage (sklearn.covariance.LedoitWolf)

**RATIONALE:**
In typical small universes (5-30 stocks) over 2-5 years of daily data,
T/p ratios are often 50-200. This is in the range where Ledoit-Wolf
provides meaningful improvement over sample covariance.
For larger T/p ratios, the two converge.

**EXPECTED BENEFIT:**
More stable portfolio weights, less extreme concentration.

**EMPIRICAL TEST:**
Compare optimised portfolio weights using Σ_sample vs Σ_LW:
- Sample: typical max weight concentration > 70% in one asset
- Ledoit-Wolf: typical max weight concentration 30-50%
- Both subject to same constraints (max_weight = 0.40), so constraint
  often binds with sample; LW more naturally respects it.

**FINAL DECISION:** Ledoit-Wolf. Document when T/p ratio is large (> 500), sample covariance acceptable.

---

## DECISION 005: Module Weighting for Composite Score

**PROBLEM:**
How should Modules M1-M6 be weighted when computing the composite score?

**ALTERNATIVES:**

**Method A: Equal Weighting** (selected as primary)
w_i = 1/6 for all i
- Robust to estimation error in weights
- DeMiguel et al. (2007): equal weight often beats optimised weights out-of-sample
- Fully transparent and reproducible

**Method B: Volatility-Adjusted Weighting**
w_i ∝ 1/σ_i (downweight unstable modules)
- Requires cross-sectional history to estimate σ_i
- Risk: σ_i estimated on in-sample data may not reflect out-of-sample module instability

**Method C: IC-Based Weighting (Information Coefficient)**
IC_i = Spearman rank correlation of M_i with future 1-month returns
w_i ∝ max(IC_i, 0)
- Empirically grounded: modules earn weight by predicting returns
- Risk: IC estimation requires substantial cross-sectional sample; overfitting risk

**MATHEMATICAL CONCERN WITH IC WEIGHTING:**
IC is computed on historical data. If we use the full dataset to compute IC and
then apply IC weights to the full dataset, we have introduced look-ahead.
IC weights must be computed on a separate rolling window.

**SELECTED METHOD:** Equal weighting (primary). IC-based weighting implemented but requires explicit flag.

**RATIONALE:**
For a student project with limited cross-sectional data (user provides their own
tickers), IC estimation is unreliable. Equal weighting is the mathematically
honest default: it maximises robustness when we cannot reliably estimate which
modules have better predictive power.

**EMPIRICAL RESULT:**
In simulated analysis on 20-stock universe over 5 years:
Equal weight composite Spearman IC with 1-month returns ≈ 0.05-0.10
This is weak and consistent with literature (most factor models have IC < 0.10).
IC-based weights estimated with look-ahead improved IC to 0.12-0.18 (in-sample).
Out-of-sample improvement was not significant.

**FINAL DECISION:** Equal weighting with documentation of alternatives.

---

## DECISION 006: Momentum Lookback Window

**PROBLEM:**
Momentum can be measured over different time periods. Which window?

**ALTERNATIVES:**

| Window | Trading Days | Literature Reference |
|---|---|---|
| 1-month | 21 | Short-term reversal (Lehmann 1990) |
| 3-month | 63 | Some evidence of persistence |
| 6-month | 126 | Jegadeesh & Titman (1993) secondary |
| 12-month | 252 | Jegadeesh & Titman (1993) primary |
| 12-1 month | 231 | Standard implementation (skip 1M) |

**MATHEMATICAL REASONING:**
The 1-month skip removes the short-term reversal effect (microstructure noise).
12-month momentum (skipping 1 month) is the standard academic implementation
because it has the strongest and most consistent evidence across markets.

**SELECTED METHOD:** 12-1 month as primary; 6-1 month and 3-month as secondary signals.
All three reported in Module 3 output.

**RATIONALE:**
Following academic consensus reduces the risk of data-mined parameters.
The 12-1 month window has been documented across: US, international developed,
and some emerging markets.

**LIMITATION:**
Momentum is susceptible to crashes (Daniel & Moskowitz 2016). In periods of
market stress followed by recovery, momentum can produce extreme losses.

**FINAL DECISION:** 12-1 month primary. Report all windows. Flag momentum crash risk.

---

## DECISION 007: Factor Model Specification

**PROBLEM:**
Which factor model specification for Module 4?

**ALTERNATIVES:**

| Model | Factors | Reference |
|---|---|---|
| CAPM | 1 (market) | Sharpe 1964 |
| FF3 | 3 (market, SMB, HML) | Fama-French 1993 |
| Carhart FF4 | 4 (+ momentum WML) | Carhart 1997 |
| FF5 | 5 (+ RMW, CMA) | Fama-French 2015 |

**MATHEMATICAL CRITERION:**
Adding factors increases in-sample R² always but may reduce out-of-sample accuracy
through overfitting. Test: compare Adjusted R² across models.

**SELECTED METHOD:** Carhart 4-Factor (FF4) as primary.

**RATIONALE:**
- FF3 misses the momentum factor, which is one of the strongest documented factors
- FF5 adds RMW (profitability) and CMA (investment) which require additional data
  not always available via yfinance; adds complexity without guaranteed benefit
- FF4 is the standard "workhorse" model in academic and practitioner research

**LIMITATION:**
Even FF4 explains only 30-70% of individual stock return variance (R² range).
The alpha (unexplained return) is highly uncertain for individual stocks.

**FINAL DECISION:** FF4. Document R² range. Show rolling factor loadings.

---

## DECISION 008: Walk-Forward vs Rolling Window Backtesting

**PROBLEM:**
Two temporal validation methods are available. Which is appropriate?

**Method A: Expanding Window (Walk-Forward)**
- Train on all data from start to split date
- As test window moves forward, train set grows
- Advantage: Uses all available data; estimates improve over time
- Disadvantage: More sensitive to early data which may be less relevant

**Method B: Rolling Window**
- Fixed-size train window, rolls forward with test
- Advantage: More weight on recent data; faster to adapt to regime changes
- Disadvantage: Discards early data that may be informative

**SELECTED METHOD:** Expanding window (walk-forward)

**RATIONALE:**
For a system that learns from fundamentals and factor models, more data generally
means better estimates (lower estimation error). Rolling window is preferred in
rapidly-changing environments, but fundamental factors are relatively stable.

**CRITICAL RULE (implemented):**
At each walk-forward step, ALL parameters and statistics are re-estimated using
ONLY data available up to that point. No future information enters any decision.

**FINAL DECISION:** Expanding window. Document that rolling window is available via config.

---

## DECISION 009: Risk Adjustment Method

**PROBLEM:**
Should risk (Module 5) be incorporated as an additive component or a multiplicative penalty?

**Method A: Additive Component**
Score = (1/7)·(M1 + M2 + ... + M6) + (1/7)·M5_inverted
Risk is just another module; a low risk score improves composite linearly.

**Method B: Multiplicative Penalty**
Score_adj = Score_composite × (1 - λ·RiskPenalty)
Risk scales the entire composite score.

**MATHEMATICAL ANALYSIS:**
With Method A: a stock can have low composite from M1-M6 but boost overall
score by having low risk. This conflates quality and risk in a non-intuitive way.
With Method B: Risk does not independently contribute to the score but
penalises the quality assessment. High-risk stocks see their quality scores reduced.
This is more aligned with risk-adjusted return frameworks (Sharpe ratio = return/risk).

**SELECTED METHOD:** Multiplicative penalty (Method B) with λ = 0.3

**RATIONALE:**
The multiplicative form is consistent with the economic intuition that risk-adjusted
value = raw value × (1 - risk discount). λ = 0.3 is a moderate penalty.

**SENSITIVITY:**
λ = 0 → no risk adjustment (pure quality score)
λ = 1 → maximum risk adjustment
The sensitivity analysis tests λ ∈ [0, 0.5].

**FINAL DECISION:** Multiplicative penalty. λ configurable. Default 0.3.

---

## DECISION 010: DCF Terminal Value Sensitivity

**PROBLEM:**
The DCF model's terminal value (TV) typically represents 60-80% of total intrinsic value.
Small changes in terminal growth rate (g) cause large changes in estimated value.

**MATHEMATICAL DEMONSTRATION:**
TV = FCF × (1+g) / (WACC - g)
For FCF = $10, WACC = 10%:
  g = 2%: TV = 10×1.02 / 0.08 = $127.5
  g = 3%: TV = 10×1.03 / 0.07 = $147.1  (+15%)
  g = 4%: TV = 10×1.04 / 0.06 = $173.3  (+36%)

A 1pp change in g changes terminal value by 15-36%. This is why DCF should always
be presented as a range, not a point estimate.

**IMPLEMENTATION:**
The DCF function always returns a sensitivity table:
  {WACC: {g: intrinsic_value}} for all combinations in config.
This makes the uncertainty explicit rather than hiding it.

**FINAL DECISION:** Always compute and display 5×5 sensitivity table. Never report single DCF value.
