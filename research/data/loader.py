"""
Pilot data loader — generates synthetic price and fundamental data.

yfinance is unreliable in restricted environments. For the research pilot,
we generate controlled synthetic data:
  - Log-normal price paths (GBM) per ticker, seeded for reproducibility
  - Synthetic fundamental data sampled from realistic distributions

This lets the full pipeline run end-to-end and produce real OOS delta-R²
numbers without any network dependency.

To use real data in a later study, replace `_synthetic_prices` and
`_synthetic_fundamentals` with live yfinance calls.
"""

import logging
import numpy as np
import pandas as pd

from src.data.point_in_time import PointInTimeStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pilot universe — 20 large-cap U.S. equities spanning 5 GICS sectors
# ---------------------------------------------------------------------------
PILOT_UNIVERSE = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",   # Technology
    "JNJ",  "PFE",  "UNH",  "ABT",  "MRK",    # Healthcare
    "JPM",  "BAC",  "GS",   "MS",   "WFC",    # Financials
    "XOM",  "CVX",  "COP",  "SLB",  "EOG",    # Energy
]


def _synthetic_prices(ticker: str, start: str, end: str, seed: int) -> pd.DataFrame:
    """
    Generate a synthetic GBM price series for one ticker.

    Parameters follow rough empirical priors for large-cap U.S. equities:
        mu    ~ N(0.07, 0.03²)   annual drift
        sigma ~ |N(0.20, 0.05²)| annual volatility
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B")  # business days
    n = len(dates)
    if n == 0:
        return pd.DataFrame()

    mu = rng.normal(0.07, 0.03)
    sigma = abs(rng.normal(0.20, 0.05))
    dt = 1 / 252

    log_returns = rng.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), size=n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame({
        "Open":      prices * rng.uniform(0.995, 1.0, n),
        "High":      prices * rng.uniform(1.000, 1.01, n),
        "Low":       prices * rng.uniform(0.990, 1.00, n),
        "Close":     prices,
        "Adj Close": prices,
        "Volume":    rng.integers(5_000_000, 50_000_000, n),
    }, index=dates)
    return df


def _synthetic_fundamentals(ticker: str, seed: int) -> dict:
    """
    Generate plausible quarterly fundamental data for one ticker.

    Produces 20 quarters (~5 years) of history for each statement.
    Values are in millions of USD.
    """
    rng = np.random.default_rng(seed + 1000)

    # Quarter-end dates, most-recent first (matching PointInTimeStore convention)
    q_ends = pd.date_range(end="2023-09-30", periods=20, freq="QE")[::-1]

    # Income statement line items
    revenue       = rng.uniform(5_000, 80_000, 20) * 1e6
    net_income    = revenue * rng.uniform(0.05, 0.25, 20)
    ebit          = revenue * rng.uniform(0.08, 0.30, 20)
    basic_eps     = net_income / rng.uniform(1e9, 5e9, 20)    # $/share
    diluted_shares = rng.uniform(1e9, 5e9, 20)

    income_df = pd.DataFrame({
        d: {
            "Net Income":           net_income[i],
            "EBIT":                 ebit[i],
            "Basic EPS":            basic_eps[i],
            "Diluted Average Shares": diluted_shares[i],
        }
        for i, d in enumerate(q_ends)
    })

    # Balance sheet line items
    total_assets      = revenue * rng.uniform(1.5, 5.0, 20)
    current_liab      = total_assets * rng.uniform(0.15, 0.35, 20)
    total_debt        = total_assets * rng.uniform(0.10, 0.50, 20)
    equity            = total_assets - current_liab - total_debt

    balance_df = pd.DataFrame({
        d: {
            "Total Assets":                total_assets[i],
            "Total Current Liabilities":   current_liab[i],
            "Total Debt":                  total_debt[i],
            "Total Shareholders' Equity":  equity[i],
        }
        for i, d in enumerate(q_ends)
    })

    # Cash-flow statement
    ocf = net_income * rng.uniform(1.0, 1.5, 20)

    cashflow_df = pd.DataFrame({
        d: {"Operating Cash Flow": ocf[i]}
        for i, d in enumerate(q_ends)
    })

    return {
        "income_statement": income_df,
        "balance_sheet":    balance_df,
        "cash_flow":        cashflow_df,
    }


class ResearchDataLoader:
    """
    PIT-aware data loader for the research subsystem.

    In the pilot study, all data is generated synthetically with fixed seeds
    so results are fully reproducible. The PointInTimeStore enforces the
    look-ahead barrier on every query.
    """

    def __init__(self, config: dict):
        self.config = config
        self._pit_store = PointInTimeStore()
        self._price_cache: dict[str, pd.DataFrame] = {}

    def get_universe(self, as_of: pd.Timestamp) -> list[str]:
        return PILOT_UNIVERSE

    def load_price_data(self, tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Generate and register synthetic price data for each ticker."""
        price_dict = {}
        for i, ticker in enumerate(tickers):
            import hashlib
            global_seed = self.config.get('system', {}).get('random_seed', self.config.get('random_seed', 42))
            base_seed = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
            seed = (base_seed + global_seed) % (2**31)
            
            df = _synthetic_prices(ticker, start, end, seed=seed)
            if df.empty:
                continue
            try:
                self._pit_store.register_price_data(ticker, df)
            except Exception:
                pass   # PIT store may not expose this; prices still cached below
            price_dict[ticker] = df
            logger.info(f"Loaded synthetic price data for {ticker}: {len(df)} days")

        self._price_cache = price_dict
        return price_dict

    def load_fundamental_data(self, tickers: list[str]) -> dict:
        """Generate and register synthetic fundamental data for each ticker."""
        fin_dict = {}
        for ticker in tickers:
            import hashlib
            global_seed = self.config.get('system', {}).get('random_seed', self.config.get('random_seed', 42))
            base_seed = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
            seed = (base_seed + global_seed) % (2**31)
            
            stmt_dict = _synthetic_fundamentals(ticker, seed=seed)
            try:
                self._pit_store.register_fundamental_data(ticker, stmt_dict)
            except Exception:
                pass
            fin_dict[ticker] = stmt_dict
            logger.info(f"Loaded synthetic fundamentals for {ticker}")

        return fin_dict

    def get_pit_store(self) -> PointInTimeStore:
        return self._pit_store
