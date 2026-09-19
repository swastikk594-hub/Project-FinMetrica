# FinMetrica

An open-source quantitative finance research engine — and a case study in why your first exciting result is usually wrong.

FinMetrica implements a full stack of classical and modern portfolio theory (Markowitz optimization, Black-Litterman, Ledoit-Wolf covariance shrinkage, Hierarchical Risk Parity) alongside a leakage-free backtesting framework (Combinatorial Purged Cross-Validation) and a battery of statistical tests designed specifically to catch the ways backtests lie. It was used to run a controlled empirical study asking: does "optimal" portfolio construction actually beat naive diversification once you account for estimation error, multiple testing, and survivorship bias?

The honest answer, reproduced independently here, is **no** — a result consistent with DeMiguel, Garlappi & Uppal's well-known "1/N puzzle" (2009).

## Key Finding

Across two independently constructed asset universes (a hindsight-biased "mega-cap winners" set and a diversified alternative), no tested portfolio construction method — including Hierarchical Risk Parity — produced a statistically significant, replicable improvement over naive equal-weighting, once transaction costs, multiple-comparison correction, and pipeline determinism were properly controlled for.

What did replicate robustly: structured allocation methods produce dramatically more stable portfolios over time. Hierarchical Risk Parity's average turnover (~0.009) was roughly 6x lower than classical Markowitz's (~0.056), with no corresponding cost in risk-adjusted return.

Full results, methodology, and the complete debugging history are in [`research/results/FINDINGS.md`](research/results/FINDINGS.md) and the alternate-universe replication at [`research/results/alt_universe/FINDINGS.md`](research/results/alt_universe/FINDINGS.md).

> **A note on how this finding was arrived at:** An earlier version of this pipeline appeared to show Hierarchical Risk Parity significantly beating classical optimization. That result disappeared entirely once a hidden source of unseeded randomness in the pipeline was found and fixed — it was an artifact of a lucky random seed, not a real effect. The debugging process that uncovered this — and the decision to report the less exciting but true result instead of the more exciting but wrong one — is documented in full in the methodology paper and is, honestly, the part of this project I'm proudest of. See **Methodology & Debugging History** below.

## Why This Exists

Most quant-finance side projects either (a) implement a textbook model and stop, or (b) claim a strategy "beats the market" without the statistical rigor to back it up. This project tries to do neither: it implements real, citable methodology correctly, then subjects its own results to the same skepticism a professional researcher would — controlling for multiple testing, checking reproducibility, testing on more than one dataset, and reporting null results honestly when that's what the data shows.

## Core Research: Two Tracks

### Track 1 — Estimator Instability
Isolates a single question: does regularizing the covariance matrix (Ledoit-Wolf shrinkage) actually make portfolio optimization more stable, holding everything else fixed? Measures condition number, effective rank, and out-of-sample weight turnover across expanding walk-forward windows, comparing sample covariance vs. Ledoit-Wolf-shrunk covariance vs. Hierarchical Risk Parity.

### Track 2 — Full-System Benchmark
Compares five complete allocation methods — Equal Weight, Naive Markowitz, Ledoit-Wolf-Shrinkage-Only Markowitz, Black-Litterman + Ledoit-Wolf ("Regularized Markowitz"), and Hierarchical Risk Parity — across 15 Combinatorial Purged Cross-Validation paths, with realistic transaction costs and market impact. Every pairwise comparison (10 total) is tested for statistical significance via the Jobson-Korkie/Memmel test, corrected for multiple comparisons (Bonferroni and Benjamini-Hochberg), and the best-performing method is checked against a Deflated Sharpe Ratio to correct for the selection bias of having tried five methods and picked the best one.

Both tracks are re-run on a second, deliberately less hindsight-curated ticker universe to check whether findings are real or an artifact of universe selection.

## Repository Structure

```text
src/
  modules/          M1-M9: fundamentals, valuation, technicals, factor regression,
                    risk (VaR/CVaR/EVT/copulas), macro regime, portfolio optimization
                    (Black-Litterman/Ledoit-Wolf/HRP/Max-Sharpe QP), NLP sentiment,
                    statistical arbitrage
  backtesting/      Event-driven walk-forward engine; Combinatorial Purged
                    Cross-Validation (CPCV) with calendar-correct purge/embargo
  risk/             Matrix diagnostics (condition number, effective rank),
                    EVT, copula tail-dependence modelling
  stats/            Jobson-Korkie/Memmel significance testing, multiple-comparison
                    correction, Deflated Sharpe Ratio
  data/             Three-tier data fetcher (cache -> yfinance -> calibrated
                    synthetic fallback), universe construction, config loading
  execution/        Broker integration (paper/live) — not used in the research
                    pipeline; see note below
research/
  experiments/      Track 1 & Track 2 study code, auto-generated findings report
  results/          FINDINGS.md, alt_universe/FINDINGS.md, universe_comparison.md,
                    raw CSVs, figures
docs/
  math_appendix.md               Full worked derivations for every model used
  student_methodology_paper/     Formal write-up (.docx) + build script
scripts/
  verify_sharpe_by_hand.py       Independent hand-check of the Sharpe ratio calculation
  check_paper_consistency.py     Flags drift between the written paper and the latest findings
tests/                           104 tests covering every module above
```

## Methodology & Debugging History

This project went through several rounds of finding and fixing real methodological bugs before arriving at its final results. This is documented transparently rather than scrubbed out, because the debugging *is* the research process:

*   **Risk-free-rate handling:** An early version of the Sharpe ratio calculation understated (and later, briefly, double-subtracted) the risk-free rate, causing reported Sharpe ratios to swing substantially across otherwise-identical runs. Caught via an independent hand-verification script (`scripts/verify_sharpe_by_hand.py`) built specifically to compute Sharpe ratios outside the main pipeline and cross-check.
*   **Survivorship/hindsight bias:** The initial ticker universe (AAPL, MSFT, GOOG, AMZN, META, etc.) is today's list of mega-cap winners, applied backward to a 2005 start date. This is a form of hindsight bias that likely inflates absolute performance for every method tested. Addressed by building and running a second, deliberately less hindsight-curated universe and reporting whether findings replicate.
*   **Headline/conclusion logic bug:** An early version of the auto-generated report's summary contradicted its own significance table (claiming "no significant differences" while the table showed several). Fixed by deriving the summary programmatically from the actual p-value table at render time rather than a fixed template.
*   **Deflated Sharpe Ratio sample-size inflation:** An early DSR calculation used a naively pooled observation count across overlapping CPCV paths, producing a numerically saturated (and misleading) 100.00% result. Fixed by using a defensible, non-overlapping observation count.
*   **Determinism:** Identical CLI commands with identical inputs produced different results across runs, traced to unseeded randomness in Monte Carlo valuation simulations and MLE-based statistical fits feeding into the Black-Litterman alpha views. Once every random number generator in the pipeline was properly seeded from a single global config value, two consecutive runs were confirmed to produce bit-for-bit identical output — and the earlier "HRP significantly beats classical optimization" finding disappeared, revealing it had been an artifact of a specific random seed rather than a real effect.

The full findings report, including this debugging history, is generated automatically by `research/experiments/generate_report.py` and reconciled against the written methodology paper by `scripts/check_paper_consistency.py`.

## The Mathematics

Every model implemented here is derived from first principles in `docs/math_appendix.md`, including:

*   The **Black-Litterman posterior**, derived via Bayes' rule and completing the square on the normal-normal conjugate update.
*   The closed-form **maximum-Sharpe-ratio ("tangency") portfolio**, derived via the Charnes-Cooper scale-invariance trick, and the full KKT system for the box-constrained (long-only, position-capped) version actually solved numerically in code.
*   The bias-variance argument for why **Ledoit-Wolf shrinkage** improves covariance matrix conditioning.
*   The leakage mechanism **CPCV's purge/embargo logic** is designed to prevent.
*   The **Jobson-Korkie/Memmel test** and **Deflated Sharpe Ratio** formulas used for significance testing and overfitting correction.

## Key References

*   Markowitz, H. (1952), "Portfolio Selection," *Journal of Finance*.
*   Ledoit, O. and Wolf, M. (2004), "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices," *Journal of Multivariate Analysis*.
*   López de Prado, M. (2018), *Advances in Financial Machine Learning* (CPCV methodology).
*   Bailey, D.H. and López de Prado, M. (2014), "The Deflated Sharpe Ratio," *Journal of Portfolio Management*.
*   Chopra, V.K. and Ziemba, W.T. (1993), "The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice," *Journal of Portfolio Management*.
*   DeMiguel, V., Garlappi, L., and Uppal, R. (2009), "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?," *Review of Financial Studies*.

## How to Use It (CLI Commands)

FinMetrica is driven by a powerful Command Line Interface (`src.cli`) designed to handle everything from point-in-time portfolio recommendations to massive scientific backtests. 

First, install dependencies:
```bash
pip install -r requirements.txt
```

### 1. The Quantitative Advisor (Fast Portfolio Allocation)
Want to know exactly what stocks to buy today from a specific universe? This command scores the assets using Modules 1-6, skips the heavy historical backtesting, and instantly generates a Black-Litterman and HRP optimized portfolio allocation.
```bash
python -m src.cli run --tickers AAPL MSFT NVDA JNJ PG XOM --mode advise --no-backtest --no-stress
```

### 2. Deep Portfolio Simulation & Stress Testing
If you want the full institutional report—including 10-year historical backtests of the strategy and Extreme Value Theory (EVT) Copula crash simulations (like a 2008 housing crash scenario).
```bash
python -m src.cli run --tickers AAPL MSFT NVDA JNJ PG XOM --mode advise
```

### 3. The Scientific Research Engine (Track 1 & 2)
This is the core laboratory used for the empirical findings documented above. It runs the Combinatorial Purged Cross-Validation (CPCV) engine to scientifically test whether the complex models actually beat an equal-weight portfolio out-of-sample.
```bash
# Run the full research pipeline on the primary universe
python -m src.cli research --track both \
  --tickers AAPL MSFT GOOG AMZN META JNJ PFE UNH JPM BAC GS WFC XOM CVX COP PG KO HD MCD V \
  --start 2005-01-01

# Run the strict alternate-universe replication check
python -m src.cli research --track both --universe-alt --start 2005-01-01
```

### 4. Live / Paper Trade Execution
If you have configured your Alpaca broker API keys, you can pipe the algorithm's optimized weights directly into live execution. (Note: Only use this with Paper keys unless you fully understand the risks).
```bash
python -m src.cli run --tickers AAPL MSFT NVDA JNJ PG XOM --mode advise --live-trade
```

### 5. Verification & Testing
Maintain the strict scientific integrity of the codebase with these utility scripts:
```bash
# Verify the Sharpe ratio calculation independently to check for math bugs
python scripts/verify_sharpe_by_hand.py

# Check the written paper against the latest FINDINGS.md to ensure no drift
python scripts/check_paper_consistency.py

# Run the full 104-test suite (pytest)
python -m pytest tests/ -v
```

See all granular options and module-specific arguments via `python -m src.cli --help`.

## The Quantitative Pipeline (M1-M9)

While the core research tracks intentionally isolate the portfolio optimization problem, the full FinMetrica engine implements a complete, 9-module alpha-generation and risk-management pipeline. Every module is built to be modular, statistically rigorous, and fully testable:

*   **M1: Fundamental Quality** - Evaluates balance sheet strength, cash flow stability, and profitability metrics.
*   **M2: Valuation** - Computes intrinsic value using Discounted Cash Flow (DCF) simulations and relative multiples.
*   **M3: Time Series & Momentum** - Analyzes price trends, moving average crossovers, and momentum regimes.
*   **M4: Factor Beta** - Regresses asset returns against broader macroeconomic and Fama-French style factors.
*   **M5: Tail Risk & EVT** - Models extreme market crashes using Extreme Value Theory (EVT) and calculates 99% Conditional Value at Risk (CVaR) via Student-t Copulas.
*   **M6: Macro Regime Detection** - Uses Gaussian Hidden Markov Models (HMM) to classify the current market environment (e.g., high-volatility bear vs. low-volatility bull).
*   **M7: Portfolio Optimization** - The core allocator. Implements Black-Litterman, Hierarchical Risk Parity (HRP), and Ledoit-Wolf shrinkage to convert M1-M6 scores into strict capital weights.
*   **M8: NLP Sentiment** - Integrates text-based sentiment analysis from financial news and SEC filings.
*   **M9: Statistical Arbitrage** - Scans the universe for highly cointegrated pairs using Augmented Dickey-Fuller (ADF) tests for mean-reversion trading.

*(Note: The system also includes a broker execution layer for paper/live trading via Alpaca, used for applying the system's outputs to a real portfolio decision context).*

## The Philosophy: Extreme Experimental Rigor

The defining feature of FinMetrica is not just the complexity of the modules, but **how violently they are tested**. The quantitative finance industry is plagued by backtests that look brilliant on paper but fail instantly in live markets due to overfitting. 

To combat this, FinMetrica treats every backtest as a highly skeptical scientific experiment:
1.  **Combinatorial Purged Cross-Validation (CPCV):** We do not just run a single historical backtest. We slice history into blocks, scramble them into thousands of alternate market realities, and test the algorithm across all of them. Crucially, we "purge" the boundaries between training and testing data so the algorithms absolutely cannot peek into the future.
2.  **The Deflated Sharpe Ratio (DSR):** If you test enough strategies, one will look profitable purely by accident. FinMetrica mathematically penalizes the final Sharpe ratio based on how many combinations were tested, destroying the illusion of "p-hacked" results.
3.  **Out-of-Sample Truth:** As highlighted in our **Key Findings**, the system is honest enough to admit when a complex algorithm fails to beat a simple 1/N equal-weight portfolio out-of-sample. 

The entire architecture is designed to protect the researcher from their own biases, ensuring that any alpha discovered is mathematically real, not just a historical mirage.

## Testing

104 tests across every module, including regression tests for each bug described in the methodology section above (embargo/leakage correctness, no risk-free double-subtraction, headline-logic correctness, DSR sample-size sanity, and full-pipeline determinism).

```bash
python -m pytest tests/ -v
```

## License

FinMetrica License. See `LICENSE`.

## Disclaimer

This is a research and educational project. Nothing in this repository constitutes financial advice, and past backtested performance (real or simulated) is not indicative of future results.
