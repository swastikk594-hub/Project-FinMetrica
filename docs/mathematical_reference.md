# Mathematical Reference
## QuantFinance-EDU: Definitions, Formulas, and Intuitions

> This document defines every mathematical technique used in the system.
> For every technique: definition, equation, intuition, assumptions, limitations, and alternatives.

---

## Part 1: Return Measures

### 1.1 Arithmetic Return

**Definition**: The percentage change in price over one period.

**Equation**:
```
r_t = (P_t - P_{t-1}) / P_{t-1} = P_t/P_{t-1} - 1
```

**Intuition**: If you bought a stock at £100 and it rose to £110, your arithmetic return is 10%.

**Assumptions**: Returns are i.i.d. (independent and identically distributed).

**Advantages**:
- Portfolio-additive: r_p = Σ w_i · r_i (exact)
- Intuitive and widely understood

**Disadvantages**:
- Not time-additive: r(t,t+2) ≠ r(t,t+1) + r(t+1,t+2) (approximately, but not exactly)
- Cannot fall below -100% for a stock, but can be very right-skewed

**Where used**: Module 7 (portfolio aggregation), performance metrics

---

### 1.2 Logarithmic (Log) Return

**Definition**: The natural logarithm of the price ratio.

**Equation**:
```
r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
```

**Intuition**: Log returns measure the continuously compounded rate of return.
A log return of 0.095 means the same as earning 9.5% continuously compounded.

**Key property — Time additivity**:
```
r(t, t+2) = r(t, t+1) + r(t+1, t+2)

Proof:
ln(P_{t+2}/P_t) = ln(P_{t+2}/P_{t+1}) + ln(P_{t+1}/P_t)
```

**Assumptions**: Prices follow geometric Brownian motion (GBM).

**Relationship to arithmetic return**:
```
For small r: ln(1+r) ≈ r - r²/2
The approximation error is: log_r ≈ arith_r - arith_r²/2
```

**Where used**: Modules 3, 4, 5, 6 (all time-series analysis)

---

## Part 2: Moments and Statistics

### 2.1 Mean (Expected Value)

**Equation**:
```
μ = E[r] = (1/T) Σ_{t=1}^{T} r_t
```

**Intuition**: The average daily return over the sample period.

**Limitation**: The mean is notoriously imprecise for financial returns.
Standard error of mean ≈ σ/√T. For 5 years of daily data (T=1260), σ=1%:
SE ≈ 1%/√1260 ≈ 0.028% per day → ~7% annual uncertainty on a 5-year mean estimate.
This makes expected return estimation unreliable.

---

### 2.2 Variance and Standard Deviation

**Equation**:
```
σ² = Var[r] = (1/(T-1)) Σ_{t=1}^{T} (r_t - μ)²
σ = √(σ²)
```

**Annualised volatility**:
```
σ_annual = σ_daily × √252
```
Assuming 252 trading days per year. The √252 comes from the variance of a sum
of independent random variables: Var(Σr_t) = T × Var(r_t), so SD(annual) = √T × σ_daily.

**Intuition**: Volatility measures the "spread" of returns around the mean.
Higher σ → returns are more unpredictable.

**Limitation**: σ treats upside and downside equally. Investors care more about downside.

---

### 2.3 Z-Score

**Equation**:
```
z = (x - μ) / σ
```

**Intuition**: How many standard deviations is x from the mean?
- |z| < 1: within 1σ of mean (68% of normal distribution)
- |z| < 2: within 2σ (95%)
- |z| > 3: "outlier" for normal distributions

**Use in this system**:
- Normalising individual metrics within modules
- Identifying outliers in data quality checks
- Comparing P/E ratios vs historical mean (relative valuation)

---

### 2.4 Covariance

**Equation**:
```
Cov(X, Y) = (1/(T-1)) Σ_{t=1}^{T} (X_t - μ_X)(Y_t - μ_Y)
```

**Intuition**: Measures whether X and Y move together (positive) or oppositely (negative).

**Use in this system**:
- Computing beta: β = Cov(R_i, R_m) / Var(R_m)
- Building the covariance matrix Σ for portfolio optimisation

---

### 2.5 Correlation

**Equation**:
```
ρ(X, Y) = Cov(X, Y) / (σ_X × σ_Y)

ρ ∈ [-1, +1]
```

**Intuition**: Standardised covariance. +1 = perfect co-movement, 0 = independent, -1 = perfect opposition.

**Covariance vs Correlation decision**:
For portfolio optimisation, we need Cov (not correlation) because:
σ_portfolio² = wᵀΣw requires actual covariances (not correlations).
For analysing module redundancy, we use correlation (it's scale-independent).

---

## Part 3: Factor Models

### 3.1 CAPM (Capital Asset Pricing Model)

**Equation**:
```
E[R_i] - R_f = β_i × (E[R_m] - R_f)
```

**Intuition**: The expected excess return of any asset equals its beta (market sensitivity) times the market excess return.

**Limitation**: CAPM is routinely rejected empirically. It ignores size, value, momentum factors.

---

### 3.2 Fama-French 3-Factor Model

**Equation**:
```
R_i - R_f = α + β₁(R_m - R_f) + β₂·SMB + β₃·HML + ε
```

**Factor definitions**:
- **MKT (R_m - R_f)**: Market excess return. Systematic market risk.
- **SMB (Small Minus Big)**: Return of small-cap minus large-cap stocks. Captures size effect.
- **HML (High Minus Low)**: Return of high book-to-market minus low BtM. Captures value effect.
- **α**: Jensen's alpha — return unexplained by the three factors. Should be ~0 in efficient markets.

---

### 3.3 Carhart 4-Factor Model (FF4)

**Equation**:
```
R_i - R_f = α + β₁·MKT + β₂·SMB + β₃·HML + β₄·WML + ε
```

**Additional factor**:
- **WML (Winners Minus Losers)**: Return of past 12-month winners minus losers. Momentum factor.

**Why we chose FF4**: Adds momentum (one of the strongest documented factor premia) without requiring additional data sources (WML downloaded from Kenneth French's library alongside FF3 factors).

---

### 3.4 OLS Factor Loading Estimation

**Equation**:
```
β = (XᵀX)⁻¹ Xᵀy
```

Where X = matrix of factor returns, y = asset excess returns.

**Standard errors**:
```
SE(β) = √(σ²_ε × (XᵀX)⁻¹)
```

**Interpretation**:
- β_MKT = 1.2 → when market rises 1%, this stock rises 1.2% on average
- β_SMB = 0.5 → stock has positive small-cap tilt
- β_HML = -0.3 → stock has growth (not value) characteristics
- α = +0.003/month → 3bp/month unexplained excess return (after factor adjustment)

---

## Part 4: Risk Measures

### 4.1 Maximum Drawdown (MDD)

**Equation**:
```
DD_t = (V_t - max_{s≤t} V_s) / max_{s≤t} V_s

MDD = min_t DD_t
```

**Intuition**: The worst peak-to-trough decline in portfolio value. MDD = -30% means
the portfolio fell by 30% from its peak (at some point in history).

**Why it matters**: Volatility does not capture sequence risk.
A portfolio can have low volatility but still experience a severe drawdown.

---

### 4.2 Value at Risk (VaR)

**Equation** (Historical):
```
VaR_α = -Quantile(r_t, 1-α)
```

**Interpretation**: With probability α, daily losses will NOT exceed VaR_α.
VaR at 95% confidence = the 5th percentile of the loss distribution (negated).

**Example**: VaR_95 = 2.5% means:
- On 95% of days, the loss is less than 2.5%
- On 5% of days, the loss exceeds 2.5%

**Critical limitation**: VaR tells us the threshold but NOT how bad losses get beyond it.

---

### 4.3 Conditional Value at Risk (CVaR / Expected Shortfall)

**Equation**:
```
CVaR_α = -E[r_t | r_t ≤ -VaR_α]
```

**Interpretation**: The AVERAGE loss given that we are in the worst (1-α)% of outcomes.
CVaR is always ≥ VaR (mathematically provable: CVaR is the conditional expectation below VaR).

**Why CVaR is preferred over VaR**:
1. CVaR is coherent (satisfies subadditivity: CVaR(A+B) ≤ CVaR(A) + CVaR(B))
2. VaR is NOT coherent — it can violate subadditivity
3. CVaR tells us something about the tail beyond VaR

**We always report both** to show what VaR misses.

---

### 4.4 Downside Deviation

**Equation**:
```
σ_d = √[(1/(T-1)) Σ_{t=1}^{T} min(r_t - MAR, 0)²] × √252
```

Where MAR = Minimum Acceptable Return (default 0).

**Intuition**: Like standard deviation, but only counting negative deviations.
A stock that is volatile only to the upside will have low downside deviation but high σ.

**Use in Sortino Ratio**:
```
Sortino = (E[r] - r_f) / σ_d × √252
```

Sortino is preferred over Sharpe when returns are positively skewed (as many assets are).

---

### 4.5 Beta

**Equation**:
```
β = Cov(R_i, R_m) / Var(R_m)
```

**Intuition**:
- β = 1: moves exactly with market
- β > 1: amplifies market moves (higher systematic risk)
- β < 1: dampens market moves (lower systematic risk)
- β < 0: moves against market (rare; useful for hedging)

**Estimation**: Via OLS regression of asset returns on market returns.

**Rolling beta**: Estimated over a rolling 252-day window to capture changes over time.
High time-variation in β suggests the asset's risk profile is changing.

---

## Part 5: Portfolio Theory

### 5.1 Portfolio Expected Return

**Equation**:
```
E(R_p) = wᵀμ = Σ_{i=1}^{n} w_i × μ_i
```

**Intuition**: Portfolio return is the weighted average of individual expected returns.
This is exact for arithmetic returns.

---

### 5.2 Portfolio Variance

**Equation**:
```
σ_p² = wᵀΣw = Σ_i Σ_j w_i × w_j × Cov(R_i, R_j)
```

Where Σ = n×n covariance matrix.

**Intuition**: Portfolio risk is NOT the weighted average of individual risks (unless ρ=1 everywhere).
Diversification reduces portfolio variance when ρ < 1.

**Why matrix form matters**:
For n=2: σ_p² = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρ₁₂σ₁σ₂
For n=10: 10 variances + 45 covariances = 55 terms
Matrix algebra handles this efficiently.

---

### 5.3 Sharpe Ratio

**Equation**:
```
SR = (E[R_p] - R_f) / σ_p
```

**Intuition**: Return per unit of risk (volatility). Higher is better.
Named after William Sharpe (Nobel Prize 1990).

**Common values**:
- SR < 0.5: Poor
- SR 0.5-1.0: Acceptable
- SR > 1.0: Good
- SR > 2.0: Excellent (rare in practice)

**Limitation**: Assumes risk = standard deviation (penalises upside volatility equally).
Sortino ratio addresses this.

---

### 5.4 Efficient Frontier

**Definition**: The set of portfolios that maximise expected return for a given level of risk
(or equivalently, minimise risk for a given return target).

**Mathematical formulation**:
```
min_w wᵀΣw
subject to:
  wᵀμ = target_return
  Σw_i = 1
  w_i ≥ 0
```

Solved for a range of target_return values → traces the efficient frontier.

**Capital Market Line**: Line from risk-free rate tangent to the efficient frontier.
The tangent point = Maximum Sharpe Ratio portfolio.

---

## Part 6: Fundamental Analysis Mathematics

### 6.1 DuPont ROE Decomposition

**Equation**:
```
ROE = Net Income / Equity
    = (Net Income / Sales) × (Sales / Assets) × (Assets / Equity)
    = Net Margin × Asset Turnover × Equity Multiplier
```

**Intuition**: Decomposes ROE into three economically distinct drivers:
1. **Net Margin**: Profitability per pound of sales
2. **Asset Turnover**: Efficiency of asset utilisation
3. **Equity Multiplier**: Financial leverage

A company can have high ROE from any combination. DuPont reveals WHICH driver is responsible.

---

### 6.2 Piotroski F-Score

9 binary signals grouped into three areas:

**Profitability (4 signals)**:
1. ROA > 0 (positive return on assets)
2. CFO > 0 (positive cash flow from operations)
3. ΔROA > 0 (ROA improving year-over-year)
4. Accruals: CFO/Assets > ROA (cash earnings > reported earnings → quality signal)

**Leverage & Liquidity (3 signals)**:
5. ΔLong-term debt/Assets < 0 (leverage declining)
6. ΔCurrent Ratio > 0 (liquidity improving)
7. No new share issuance (no dilution)

**Operating Efficiency (2 signals)**:
8. ΔGross Margin > 0 (margins improving)
9. ΔAsset Turnover > 0 (efficiency improving)

F = Σ signals ∈ [0, 9]
Piotroski (2000): F ≥ 8 = strong; F ≤ 2 = weak

---

### 6.3 Altman Z-Score

**Equation** (for public companies):
```
Z = 1.2·X₁ + 1.4·X₂ + 3.3·X₃ + 0.6·X₄ + 1.0·X₅

X₁ = Working Capital / Total Assets
X₂ = Retained Earnings / Total Assets
X₃ = EBIT / Total Assets
X₄ = Market Capitalisation / Total Liabilities
X₅ = Revenue / Total Assets
```

**Zones**:
- Z > 2.99: Safe zone (low bankruptcy risk)
- 1.81 ≤ Z ≤ 2.99: Grey zone
- Z < 1.81: Distress zone

**Limitation**: Calibrated on 1960s US manufacturing firms. May not be appropriate for:
- Financial companies (different capital structure)
- Technology companies (intangible-heavy)
- Non-US companies

---

## Part 7: Discounted Cash Flow

### 7.1 DCF Valuation

**Equation**:
```
Intrinsic Value = Σ_{t=1}^{n} [FCF_t / (1+WACC)^t] + [TV / (1+WACC)^n]

Terminal Value = FCF_n × (1+g) / (WACC - g)
```

**WACC**:
```
WACC = (E/V)·Ke + (D/V)·Kd·(1-t)

where:
  E = market value of equity
  D = market value of debt
  V = E + D
  Ke = cost of equity (from CAPM: Rf + β·ERP)
  Kd = cost of debt
  t = corporate tax rate
```

**CAPM cost of equity**:
```
Ke = Rf + β × ERP
```

Where:
- Rf = risk-free rate (10Y Treasury yield)
- β = asset beta (from Module 4)
- ERP = Equity Risk Premium (~5.5% for US; Damodaran estimate)

**Critical sensitivity**: The system always generates a sensitivity table showing
intrinsic value for combinations of WACC and g. This makes the model's
uncertainty explicit.

---

## Part 8: Uncertainty Quantification

### 8.1 Bootstrap Resampling

**Method**:
1. Have T observations of data
2. Draw T samples WITH REPLACEMENT from the data (bootstrap sample)
3. Compute statistic on bootstrap sample
4. Repeat N times (N = 1000+)
5. The distribution of bootstrap statistics approximates the sampling distribution

**Use in this system**:
- Confidence intervals on composite scores
- Confidence intervals on portfolio weights (Module 7)
- Stability of factor loadings

**Why not standard confidence intervals?**
Standard CIs assume normality. Bootstrap is non-parametric and works for any statistic.

---

### 8.2 Monte Carlo Simulation

**For portfolio value simulation**:
1. Fit return distribution (t-distribution for fat tails)
2. Sample n × T values from the distribution
3. Compute portfolio path for each simulation
4. Report percentile outcomes

**IMPORTANT DISCLAIMER (built into system output)**:
Monte Carlo simulation is NOT a prediction of the future.
It shows what range of outcomes is mathematically consistent with
the historical statistical properties of the asset.
The real future may be completely outside this range.

---

## Part 9: Normalisation

### The Four Methods Compared

| Method | Formula | Outlier Robust | Bounded | Look-ahead Risk |
|--------|---------|-----------------|---------|-----------------|
| Z-Score | (x-μ)/σ | Low | No | Manageable |
| Min-Max | (x-min)/(max-min) | None | [0,1] | High |
| Rank | rank(x)/n | Complete | [0,1] | None |
| Robust | (x-median)/IQR | High | No | Manageable |

**Selected for this system**:
- Individual metrics within modules: Z-score (winsorised at ±3σ)
- Module-level scores: Rank percentile normalisation
- Final output: Scaled to 0-100

**Look-ahead prevention**:
In backtesting, all normalisation uses EXPANDING window statistics:
at time t, μ and σ are computed using only data from [0, t].
Never use full-sample statistics to normalise in backtesting.
