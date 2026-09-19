# FinMetrica: Master Navigation Guide

Welcome to the FinMetrica Quantitative Research & Risk System! This map explains exactly what every folder and core file does so you can easily navigate the codebase.

## 📂 Root Directory
* **`src/`** — **The Core System Codebase.** All 9 quantitative modules, the backtesting engine, and data pipelines live here.
* **`research/`** — **The Research Extension.** Contains the rigorous experiments (like the Covariance Regularization Study) that use the `src/` modules to test falsifiable hypotheses.
* **`tests/`** — **Unit & Integration Tests.** Over 90 tests ensuring the mathematics (like normalisation and covariance shrinkage) are mathematically pure. Run with `pytest tests/`.
* **`docs/`** — **Documentation.** Mathematical proofs, LaTeX derivations (`math_appendix.md`), and system architecture decisions.
* **`data/`** — **Raw & Processed Data Storage.** Caches downloaded CSVs, Fama-French factors, and cleaned datasets to prevent re-downloading.
* **`notebooks/`** — **Jupyter Notebooks.** Interactive demos and scratchpads for visualising pipeline outputs.
* **`scripts/`** — **Standalone Utilities.** Helpful automation scripts (e.g., bulk downloading data, checking broker clocks).
* **`results/`** — **Execution Results.** Outputs from the live-trading/execution pipeline (saved JSON portfolios and allocations).
* **`archive/`** — **Old Audits & Logs.** Previous execution logs and text dumps moved here for cleanliness.

---

## 🧠 Inside `src/` (The Core Modules)

### Modules
* **`src/modules/m1_fundamentals.py`** — Parses Balance Sheets, Income Statements, and Cash Flows.
* **`src/modules/m2_valuation.py`** — Computes DCF models, WACC, and historical valuation ratios (P/E, P/B).
* **`src/modules/m3_timeseries.py`** — Calculates moving averages, momentum, Hurst exponents, and RSI.
* **`src/modules/m4_factors.py`** — Fama-French multi-factor regressions and Alpha computations.
* **`src/modules/m5_risk.py`** — Quantifies Risk: VaR, CVaR, Maximum Drawdown, and Downside Deviation.
* **`src/modules/m6_macro.py`** — Ingests macroeconomic indicators (yield curve, inflation) to generate regime flags.
* **`src/modules/m7_optimisation.py`** — **Portfolio Construction.** Markowitz, Ledoit-Wolf Shrinkage, Black-Litterman, and Hierarchical Risk Parity (HRP).
* **`src/modules/m8_nlp.py`** (Stub) — Natural Language Processing for sentiment analysis.
* **`src/modules/m9_statarb.py`** (Stub) — Statistical Arbitrage and cointegration testing.

### Sub-systems
* **`src/data/`** — The Data Fetcher. Multi-tier fallback architecture (Alpaca -> yfinance -> FRED).
* **`src/backtesting/`** — The Event-Driven Backtester. Contains the core loop (`engine.py`) and the advanced Combinatorial Purged Cross-Validator (`cpcv.py`).
* **`src/integration/`** — Combines raw scores from M1-M9 into a single unified signal (`aggregator.py`).
* **`src/stats/`** — Pure mathematical functions (e.g., `performance_tests.py` for Jobson-Korkie significance, `deflated_sharpe.py`).
* **`src/cli.py`** — The Command Line Interface. Run `python -m src.cli run` or `python -m src.cli research`.

---

## 🔬 Inside `research/` (The Lab)

* **`research/experiments/`**
  * `covariance_regularization_study.py` — Track 1: Measures if Ledoit-Wolf shrinkage reduces portfolio turnover and matrix condition numbers.
  * `full_system_benchmark.py` — Track 2: Pits HRP against Markowitz using Combinatorial Purged Cross-Validation (CPCV).
  * `generate_report.py` — Compiles the CSV outputs from Tracks 1 & 2 into the beautiful markdown report.
* **`research/results/`** — Where the experimental outputs go! Look here for `FINDINGS.md`, significance testing CSVs, and the `figures/` folder containing generated plots.
* **`research/normalization/`** — Mathematics for cross-sectional normalization (e.g., MinMax, Plotting-Position Rank, Robust Z-Score) to ensure signals are comparable without look-ahead bias.

---
*Tip: Start by reading `research/results/FINDINGS.md` to see the output of the system, then dive into `src/modules/m7_optimisation.py` to see how the mathematical allocations are actually computed!*
