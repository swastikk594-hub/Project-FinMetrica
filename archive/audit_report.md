# QuantFinance-EDU: Academic Originality & Methodology Audit Report

**Date:** September 2026
**Scope:** Complete Codebase and Algorithmic Audit for Academic Review

---

## 1. COMPLETE REPOSITORY TREE

The codebase is structured as a modular quantitative research pipeline.

```text
QuantFinance-EDU/
├── config/
│   └── config.yaml                 # Master configuration (thresholds, tuning, ML hyperparameters)
├── data/                           # Data caching layer
│   ├── ..._fundamentals.json       # Ticker-level fundamental data caches
│   └── ..._price.csv               # Ticker-level price data caches
├── documentation/
│   ├── decision_log.md             # Development decisions
│   └── mathematical_reference.md   # Base equations
├── notebooks/
│   └── 03_full_system_demo.ipynb   # Interactive demonstration of pipeline
├── results/
│   ├── audit_trail.jsonl           # Execution logs and module outputs
│   └── pipeline_results.json       # Final strategy metrics and weights
├── scripts/
│   ├── bulk_download.py            # Data acquisition script
│   ├── check_clock.py              # System timing script
│   └── check_positions.py          # Portfolio holding checks
├── src/
│   ├── cli.py                      # Command-line interface and entry points
│   ├── pipeline.py                 # Core orchestrator: routes data -> modules -> aggregator -> backtest
│   ├── advisory/                   # Production analyst-style recommendations
│   │   └── horizon_engine.py       # Investment horizon & invalidation criteria logic
│   ├── backtesting/                # Historical validation engine
│   │   ├── cpcv.py                 # Combinatorial Purged Cross-Validation implementation
│   │   ├── engine.py               # Event-driven walk-forward backtester
│   │   └── regime.py               # Market regime identification for segmented backtesting
│   ├── comparison/                 # Evaluation against baseline models
│   │   ├── ablation.py             # Feature importance and module drop-out testing
│   │   └── model_versions.py       # Inter-model comparison routines
│   ├── data/                       # Data acquisition and validation
│   │   ├── config_loader.py        # YAML parser
│   │   ├── fetcher.py              # yfinance/FRED API wrapper with caching
│   │   ├── point_in_time.py        # Point-in-time data storage
│   │   ├── universe.py             # Asset universe definition
│   │   └── validator.py            # Data quality checks (NaNs, outliers, stationarity)
│   ├── execution/                  # Live trading and portfolio execution
│   │   ├── alpaca_broker.py        # Alpaca API integration for paper/live trading
│   │   ├── market_impact.py        # Slippage and transaction cost models
│   │   └── order_generator.py      # Translates weights to order tickets and applies ADV filters
│   ├── integration/                # Signal synthesis
│   │   └── aggregator.py           # Combines M1-M6 scores, applies risk penalties
│   ├── models/                     # Machine Learning components
│   │   ├── hmm_regime.py           # Hidden Markov Model for market regimes
│   │   └── meta_model.py           # GBDT meta-learner over module scores
│   ├── modules/                    # Alpha and Risk generating modules (M1-M9)
│   │   ├── base.py                 # Abstract base class and `ModuleResult` dataclass
│   │   ├── m1_fundamentals.py      # Quality & financial health metrics
│   │   ├── m2_valuation.py         # Intrinsic and relative valuation models
│   │   ├── m3_timeseries.py        # Momentum, mean-reversion, regime filters
│   │   ├── m4_factors.py           # Multi-factor regression (Fama-French)
│   │   ├── m5_risk.py              # Tail risk, VaR, CVaR, Beta measurement
│   │   ├── m6_macro.py             # Macroeconomic sensitivities
│   │   ├── m7_optimisation.py      # Portfolio allocation (HRP, Black-Litterman, Mean-Variance)
│   │   ├── m8_nlp.py               # Sentiment analysis via NLP
│   │   └── m9_stat_arb.py          # Statistical arbitrage and cointegration
│   ├── preprocessing/              # Data transformations
│   │   ├── alignment.py            # Timestamp alignment and forward-filling
│   │   ├── normalisation.py        # Cross-sectional and time-series scaling
│   │   └── returns.py              # Log and arithmetic return calculators
│   ├── reporting/                  # Output formatting and strategy validation
│   │   ├── capacity.py             # AUM capacity constraints
│   │   └── qualification.py        # Strategy statistical validation (Deflated Sharpe Ratio)
│   ├── risk/                       # Advanced risk modeling
│   │   ├── copula.py               # Dependency structure modeling
│   │   ├── evt.py                  # Extreme Value Theory
│   │   └── stress.py               # Historical and synthetic stress testing
│   ├── safety/                     # Production safeguards
│   │   ├── audit_logger.py         # Trade and signal logging
│   │   ├── kill_switch.py          # Emergency liquidation rules
│   │   └── pre_trade.py            # Pre-trade compliance checks
│   ├── sensitivity/                # Parameter robustness
│   │   └── analysis.py             # Local and global sensitivity analysis
│   └── visualisation/              # Charting and plotting
│       └── plots.py                # Visual generation routines
└── tests/                          # Unit and integration test suite
    ├── test_backtesting_and_cpcv.py
    ├── test_data_pipeline.py
    ├── test_leakage_and_pit.py
    ├── test_modules.py
    ├── test_normalisation.py
    ├── test_portfolio.py
    ├── test_returns.py
    ├── test_risk_and_copula.py
    └── test_risk_metrics.py
```

### Execution Flow & Dependencies
**Entry Point:** `src/cli.py` parses arguments and instantiates `src/pipeline.py`.
**Data Layer:** `DataFetcher` pulls from yfinance and FRED $\rightarrow$ `DataValidator` checks integrity.
**Alpha Generation:** Modules `M1` through `M9` generate `ModuleResult` objects asynchronously or sequentially.
**Synthesis:** `Aggregator` takes module outputs to create normalized scores.
**Portfolio Construction:** `M7` takes normalized signals as views for Black-Litterman and HRP weighting.
**Evaluation:** `EventDrivenBacktester` loops over data folds $\rightarrow$ `Qualification` computes DSR.

---

## 2. M1–M9 IMPLEMENTATION AUDIT

### **M1: Fundamentals (`m1_fundamentals.py`)**
**Mathematical Formulas:**
- **Piotroski F-Score:** Sum of 9 binary tests ($F \in [0, 9]$) testing profitability, leverage, liquidity, and operating efficiency.
  $$ \text{F-Score} = \sum_{i=1}^9 I_i $$
- **Return on Invested Capital (ROIC):**
  $$ \text{ROIC} = \frac{\text{Net Income} - \text{Dividends}}{\text{Total Assets} - \text{Current Liabilities}} $$
- **Accruals Ratio:**
  $$ \text{Accruals} = \frac{\Delta \text{Net Operating Assets}}{\text{Total Assets}} $$
- **Altman Z-Score:**
  $$ Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5 $$
  where $X_1 = \frac{\text{Working Capital}}{\text{Total Assets}}$, $X_2 = \frac{\text{Retained Earnings}}{\text{Total Assets}}$, $X_3 = \frac{\text{EBIT}}{\text{Total Assets}}$, $X_4 = \frac{\text{Market Equity}}{\text{Total Liabilities}}$, $X_5 = \frac{\text{Sales}}{\text{Total Assets}}$.

**Transformations & Normalisation:**
- F-score converted to percentage: $(F / 9) \times 100$.
- Final M1 Score is the arithmetic mean of valid available component scores, clipped to $[0, 100]$.
- **Edge-Case:** ETFs/Indices bypass fundamental logic and receive a neutral score of $50$. Missing data defaults to returning a base 0 F-score (bug identified in implementation leading to zero bias).

### **M2: Valuation (`m2_valuation.py`)**
**Mathematical Formulas:**
- **Cost of Equity ($R_e$):** CAPM. $R_e = R_f + \beta (E[R_m] - R_f)$.
- **Cost of Debt ($R_d$):** Capped approximation. $R_d = \min(\frac{\text{Interest Expense}}{\text{Total Debt}}, 0.20)$. *Deviation: Tax shield $(1-t)$ is omitted.*
- **WACC:**
  $$ \text{WACC} = \frac{E}{V} R_e + \frac{D}{V} R_d $$
  *(Constrained: $\text{WACC} \ge 0.05$)*
- **DCF Intrinsic Value ($V_0$):**
  $$ V_0 = \sum_{t=1}^n \frac{FCF_0 (1+g)^t}{(1+\text{WACC})^t} + \frac{FCF_0 (1+g)^n (1+g_{term})}{(\text{WACC} - g_{term})(1+\text{WACC})^n} $$
- **Margin of Safety (MOS):**
  $$ \text{MOS} = \frac{V_0 - P_{current}}{V_0} $$

**Transformations & Normalisation:**
- **Sigmoid Mapping for MOS:**
  $$ S_{DCF} = \frac{100}{1 + \exp(-3 \cdot \text{MOS})} $$
  Clipped to $[5, 95]$.
- **Relative Valuation:** P/E, P/B, P/S use logistic functions centered around market medians.
- Final M2 score is the mean of DCF and relative components.

### **M3: Time-Series (`m3_timeseries.py`)**
**Mathematical Formulas:**
- **Momentum:** Computed over $w \in \{21, 63, 126, 252\}$ days.
  $$ R_{cum} = \prod_{t=1}^w (1+r_t) - 1 $$
  *Deviation: 21-day skip applied to avoid short-term mean reversion.*
- **Hurst Exponent ($H$):** Rescaled Range (R/S) analysis.
  $$ E\left[\frac{R(n)}{S(n)}\right] = C n^H $$
  Estimated via OLS regression: $\log(R/S) = \log(C) + H \log(n)$.
- **Autocorrelation:** Ljung-Box Q-test on log returns.
  $$ Q = n(n+2) \sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2_h $$

**Transformations & Normalisation:**
- Base score = $50$.
- Momentum percentile mapping directly adds/subtracts to base score.
- Moving Average (50/200) regime adds $+10$ for uptrend, $-10$ for downtrend.
- Final clipped to $[0, 100]$.

### **M4: Factor Model (`m4_factors.py`)**
**Mathematical Formulas:**
- **Multi-Factor OLS Regression:**
  $$ R_{i,t} - R_{f,t} = \alpha_i + \beta_1 MKT_t + \beta_2 SMB_t + \beta_3 HML_t + \beta_4 UMD_t + \epsilon_{i,t} $$
- **Information Ratio (IR):**
  $$ IR_i = \frac{\alpha_i}{\omega_i} $$
  where $\omega_i = \sqrt{\text{Var}(\epsilon_{i,t}) \cdot 252}$.

**Transformations & Normalisation:**
- Base score = $50$.
- $\alpha$ t-stat contributes $\pm 25$ based on significance thresholds ($|t| > 2$).
- $R^2$ adds up to $+10$.
- $IR$ adds up to $+15$.
- *Deviation/Bug:* Score generation is hard-coded to return $40.0$ in the event of missing data, which triggers universally due to datetime index mismatches on monthly downsampling.

### **M5: Risk (`m5_risk.py`)**
**Mathematical Formulas:**
- **Volatility (Annualised):** $\sigma_{ann} = \sqrt{252} \cdot \sigma_{daily}$.
- **Downside Deviation:** $\sigma_{down} = \sqrt{252 \cdot \frac{1}{N} \sum \min(r_t, 0)^2}$.
- **Historical VaR / CVaR:** Non-parametric empirical percentiles ($\alpha = 0.05$).
  $$ \text{CVaR}_\alpha = E[-r_t \mid -r_t \ge \text{VaR}_\alpha] $$
- **Maximum Drawdown:**
  $$ MDD = \min_t \left( \frac{P_t}{\max_{\tau \le t} P_\tau} - 1 \right) $$

**Transformations & Normalisation:**
- Components (Vol, DD, Beta) are scored where lower risk = higher score.
- **Risk Penalty Logic:** $\text{Composite Risk} = 0.4 \cdot S_{vol} + 0.4 \cdot S_{dd} + 0.2 \cdot S_{beta}$.
- **Module Output:** $\text{Normalised Score} = 100 - \text{Composite Risk}$. (Higher score = Safer asset).

### **M6: Macro (`m6_macro.py`)**
**Mathematical Formulas:**
- Features $\Delta M_{j, t}$ constructed from Term Spread, Credit Spread, Fed Funds Rate, YoY CPI, YoY Unemployment.
- **OLS Regression:** (36-month rolling window).
  $$ R_{i,t} = \alpha_i + \sum_{j} \beta_{i,j} \Delta M_{j, t-1} + \epsilon_{i,t} $$
- **Predicted Return:**
  $$ \hat{R}_{i, T+1} = \sum_{j} \hat{\beta}_{i,j} \Delta M_{j, T} $$

**Transformations & Normalisation:**
- Neutral score = $50$.
- If $\hat{R} > 0.02$, score = $80$. If $\hat{R} < -0.02$, score = $20$.

### **M7: Optimisation (`m7_optimisation.py`)**
**Mathematical Formulas:**
- **Covariance Matrix ($\Sigma$):** Ledoit-Wolf shrinkage to handle $N > T$ conditioning.
- **Black-Litterman Posterior Expected Returns:**
  $$ \mu_{BL} = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q] $$
  - $\Pi = \delta \Sigma w_{mkt}$ (Equilibrium Returns).
  - $P$ = Identity matrix (absolute views).
  - $Q$ = User views derived from M1-M6 composite scores.
  - $\Omega = \text{diag}(P (\tau \Sigma) P^T) / c$ (Uncertainty matrix, $c=$ confidence).
- **Mean-Variance Objective (SLSQP):**
  $$ \max_w \frac{w^T \mu}{\sqrt{w^T \Sigma w}} $$
  Subject to $\sum w_i = 1$ and $0 \le w_i \le w_{max}$.

### **M8: NLP & M9: Stat Arb**
- **M8:** HuggingFace `transformers` pipeline computing sentiment polarity on headlines. Scaled to $0-100$.
- **M9:** Pairs trading using Augmented Dickey-Fuller (ADF) test for cointegration:
  $$ \Delta z_t = \gamma z_{t-1} + \sum_{i=1}^p \delta_i \Delta z_{t-i} + \epsilon_t $$
  Half-life derived from Ornstein-Uhlenbeck process: $t_{1/2} = -\frac{\ln(2)}{\gamma}$.

---

## 3. SIGNAL INTEGRATION PIPELINE

The end-to-end integration traces raw inputs into final portfolio weights.

**Step 1: Signal Aggregation (`aggregator.py`)**
For a given ticker, the raw composite score is the weighted average of the alpha modules:
$$ S_{raw} = \sum_{k=1}^4 W_k S_k $$
where $S_k \in [0, 100]$ are the scores from M1, M2, M3, and M4.

**Step 2: Risk Penalty Application**
The risk module M5 provides a score $S_5$ where higher = safer.
$$ \text{Risk Normalised} = \frac{S_5}{100} $$
$$ \text{Penalty} = \lambda \times \text{Risk Normalised} $$
$$ S_{adj} = S_{raw} \times (1 - \text{Penalty}) $$
*Note: Due to a semantic inversion bug in the codebase, the penalty scales with safety rather than risk, penalizing low-risk assets the hardest.*

**Step 3: Cross-Sectional Normalisation**
Scores are meant to be rank-normalized cross-sectionally. However, the implementation computes rank normalisation *across modules* for a single ticker, effectively scrambling the signal magnitude.

**Step 4: Alpha View Generation (Black-Litterman)**
The pipeline maps composite scores $S \in [0, 100]$ linearly to expected return views $Q$ for the BL model:
$$ Q_i = 
\begin{cases} 
0.18 & \text{if } S_i = 100 \\
0.08 & \text{if } S_i = 50 \\
-0.02 & \text{if } S_i = 0 
\end{cases} $$
These views $Q$ are fed into M7 to produce the posterior mean $\mu_{BL}$ and posterior covariance $\Sigma_{BL}$, which are then optimized for maximum Sharpe to generate final target weights $w^*$.

---

## 4. BACKTESTING METHODOLOGY

Implemented in `src/backtesting/engine.py` using `EventDrivenBacktester`.

- **Walk-Forward Validation:** The data is split chronologically. For each fold, `train_years` (e.g., 5 years) are used to fit factor betas and generate module scores, while `test_years` (e.g., 1 year) are used to evaluate OOS performance.
- **Purging & Embargo:** To prevent information leakage in time-series splits:
  - **Purge:** 30 days of data are dropped prior to the test set to clear overlapping periods (e.g., rolling momentum windows).
  - **Embargo:** An additional temporal buffer is applied after the test set before the next training set in CPCV to prevent leakage from serial correlation.
- **CPCV (Combinatorial Purged Cross-Validation):** `src/backtesting/cpcv.py` splits the dataset into $N$ groups and trains on combinations to generate robust Sharpe distributions without simple chronological constraint.
- **Rebalancing & Transaction Costs:** Rebalancing is explicitly scheduled. A fixed basis-point cost (e.g., $10$ bps) is deducted upon weight delta changes. *Implementation bug: Monthly rebalancing dates are generated but only the first date of the test window is enacted.*
- **Bias & Leakage Risks:** 
  1. **Survivorship Bias:** The universe is fixed based on current constituents. Delisted equities are missing.
  2. **Look-Ahead Bias (Macro):** Data alignment applies `shift(1)` to macro features, but underlying macroeconomic series (FRED) are heavily revised ex-post.
  3. **Look-Ahead Bias (Fundamentals):** Forward-filling of quarterly earnings uses the release date index, assuming immediate availability.

---

## 5. HIERARCHICAL RISK PARITY (HRP) IMPLEMENTATION

Located in `src/modules/m7_optimisation.py`. HRP replaces classical covariance inversion with graph theory to ensure robust allocations out-of-sample.

1. **Distance Metric:** Converts the Ledoit-Wolf correlation matrix $\rho$ into a distance matrix:
   $$ D_{i,j} = \sqrt{\frac{1 - \rho_{i,j}}{2}} $$
2. **Quasi-Diagonalization:** Uses Single-Linkage Agglomerative Clustering (`scipy.cluster.hierarchy.linkage`) to build a tree (dendrogram) of assets. The covariance matrix is reorganized so that highly correlated investments are placed close together.
3. **Recursive Bisection:**
   - The list of items is split in half iteratively.
   - For adjacent clusters $C_1$ and $C_2$, the cluster variance is computed:
     $$ V_c = w_c^T \Sigma_c w_c \quad \text{where} \quad w_c = \frac{\text{diag}(\Sigma_c)^{-1}}{\text{Tr}(\text{diag}(\Sigma_c)^{-1})} $$
   - The split factor is: $\alpha = 1 - \frac{V_1}{V_1 + V_2}$.
   - Allocation to $C_1$ is scaled by $\alpha$, and $C_2$ by $(1-\alpha)$.

---

## 6. MACHINE-LEARNING META-MODEL

**Architecture:** Gradient Boosting Decision Tree (GBDT) via `XGBoost` or `LightGBM` (implemented in `src/models/meta_model.py`).
**Objective:** Predict 1-month forward continuous returns (Regression) or binary outperformance vs benchmark (Classification).
**Features:**
- Module scores $S_{M1} \dots S_{M6}$.
- Rolling volatilities and betas.
- Raw momentum indicators.
**Training Protocol:** Fits on `train_df`, predicting next month's return. The meta-model output can dynamically weight the importance of M1-M6 rather than relying on the static arithmetic mean aggregator.

---

## 7. DEFLATED SHARPE RATIO (DSR)

Implemented in `src/reporting/qualification.py`. Computes the probability that the strategy's Sharpe Ratio is statistically significant, accounting for Multiple Testing (selection bias) and non-normal returns.

**Formula:**
$$ \text{DSR} = \Phi \left( \frac{\widehat{SR} - SR^*}{\sqrt{\frac{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4} \widehat{SR}^2}{T-1}}} \right) $$
*(Code approximation evaluates the critical Sharpe $SR^*$ via bootstrapped distributions of alternative configurations)*
- $\widehat{SR}$ = Estimated OOS Sharpe Ratio.
- $SR^*$ = Expected maximum Sharpe from independent trials.
- $\gamma_3, \gamma_4$ = Skewness and Kurtosis of returns.
- $T$ = Number of return observations.

Strategies output `PASS`, `CONDITIONAL`, or `FAIL` based on DSR confidence $> 95\%$.

---

## 8. PARAMETER PROVENANCE CLASSIFICATION

Parameters define strategy behavior and exist in distinct provenance layers:

| Parameter | Value / Range | Provenance | Notes |
| :--- | :--- | :--- | :--- |
| **System Random Seed** | `42` | `config.yaml` | Controls all stochastic processes |
| **Outlier Z-Threshold** | $3.0$ | `config.yaml` | Winsorization limit |
| **M2 DCF Terminal Growth** | $0.02$ | `config.yaml` | Assumption for perpetual growth |
| **M3 Momentum Windows** | $[21, 63, 126, 252]$ | Hardcoded | Standard literature momentum |
| **M3 Moving Averages** | $[50, 200]$ | Hardcoded | Standard golden/death crosses |
| **M4 Factor Model Window** | 36 Months | Hardcoded | OLS lookback length |
| **M5 CVaR Alpha** | $0.05$ | Hardcoded | 95% Confidence Level |
| **Risk Penalty Lambda ($\lambda$)**| $0.15$ | `config.yaml` | Severity of M5 penalty to Alpha |
| **Txn Costs (Bps)** | $10$ bps | `config.yaml` | Cost per volume traded |
| **Walk-Forward Folds** | $5$ | `config.yaml` | Number of chronological test splits |

---

## 9. EXPERIMENTS SUMMARY

Experiment tracking is conducted via standard scripts and documented in `results/audit_trail.jsonl`.
- `scripts/bulk_download.py`: Pre-caches yfinance/FRED universe to ensure deterministic offline execution.
- `notebooks/03_full_system_demo.ipynb`: Contains execution of the end-to-end pipeline covering the S&P 100 universe.
- **Ablation Studies (`src/comparison/ablation.py`):** Designed to drop one module at a time (e.g., M1, M2) and measure the degradation in the walk-forward OOS Sharpe to determine module marginal utility.
- **Results Tracking:** All fold metrics (Sharpe, MDD, Vol, Turnovers) are serialized to JSON artifacts for external consumption.

---

## 10. ORIGINALITY CANDIDATES

This repository presents several methodologies that extend beyond textbook quant implementations, suitable for academic highlighting:
1. **Hierarchical Risk-Adjusted Scoring Pipeline:** The dual-layer approach of generating independent alpha views (M1-M4) and heavily penalizing via a non-linear risk layer (M5) prior to standard Black-Litterman optimization.
2. **Deflated Sharpe & CPCV Integration:** Utilizing Combinatorial Purged Cross-Validation in tandem with the Deflated Sharpe Ratio to rigorously quantify over-fitting probability in multi-module systems.
3. **Macro-Regime Dynamic Tilting:** M6 introduces a regime-switching overlay based on continuous OLS sensitivity to the yield curve and credit spreads, adjusting baseline signals deterministically prior to covariance optimization.

---

## 11. RESEARCH RISKS & METHODOLOGICAL FLAWS

Critical risks identified during the codebase audit that threaten the validity of academic claims:

1. **Semantic Inversion of Risk Penalty:** The mathematical penalty equation $\text{Penalty} = \lambda (S_5/100)$ uses a normalized $S_5$ where $100$ equals *safest*. This causes the safest assets to receive the maximum alpha penalty, inadvertently creating a high-risk portfolio.
2. **Cross-Sectional Rank Collapse:** Normalisation occurs across modules for a *single ticker* rather than across tickers for a *single module*. This scrambles signal cardinality prior to optimization.
3. **Data Leakage in Macro & Fundamentals:** 
   - Fundamental features are forward-filled assuming instantaneous availability at quarter-end, ignoring the 30-90 day regulatory filing lag.
   - Macroeconomic data (FRED) are heavily revised post-release. Using historical unrevised datasets without point-in-time (PiT) mapping introduces look-ahead bias.
4. **Degenerate Frontier Calibration:** Due to a lambda closure bug in the M7 SLSQP optimizer (`lambda w: np.dot(w, mu) - target_ret`), the efficient frontier collapses, as all points optimize for the final target return in the loop.
5. **Static WACC Formulation:** M2 ignores the tax shield $(1-t)$ in the Cost of Debt calculation and incorrectly utilizes Total Liabilities rather than Financial Debt, structurally under-valuing highly levered firms.

---
*End of Report.*
