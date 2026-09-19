"""
Quality factor computation using PointInTimeStore API.
"""

import pandas as pd
import numpy as np

def _get_line_item(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Try multiple row name variants to find a line item."""
    if df.empty:
        return pd.Series(dtype=float)
    for name in names:
        if name in df.index:
            return df.loc[name]
    return pd.Series(dtype=float)

def _ttm(series: pd.Series, n_quarters: int = 4) -> float:
    """Sum the n most recent quarterly values for trailing twelve months."""
    if series.empty:
        return np.nan
    valid = series.dropna()
    if len(valid) < n_quarters:
        return np.nan
    return float(valid.iloc[:n_quarters].sum())

def _latest(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.iloc[0])

def _avg(series: pd.Series, periods: int = 2) -> float:
    if series.empty:
        return np.nan
    valid = series.dropna()
    if len(valid) < periods:
        return np.nan
    return float(valid.iloc[:periods].mean())

def compute(price_data: dict, pit_store, ticker: str, as_of: pd.Timestamp) -> dict[str, float]:
    """
    Compute Quality factors for a given ticker as of a specific date.
    
    Args:
        price_data: (Ignored, use pit_store)
        pit_store: PointInTimeStore instance
        ticker: Ticker symbol
        as_of: Date for computation
        
    Returns:
        dict: Mapping of factor names to computed values
    """
    fundas = pit_store.get_point_in_time_fundamentals(ticker, as_of)
    income_stmt = fundas.get('income_statement', pd.DataFrame())
    balance_sheet = fundas.get('balance_sheet', pd.DataFrame())
    
    # ROE
    net_income = _get_line_item(income_stmt, ['Net Income', 'NetIncome', 'net_income'])
    net_income_ttm = _ttm(net_income)
    equity = _get_line_item(balance_sheet, ["Total Shareholders' Equity", 'Total Equity', 'Book Value', 'Total Stockholder Equity'])
    avg_equity = _avg(equity, 2)
    roe = np.nan
    if not pd.isna(net_income_ttm) and not pd.isna(avg_equity) and avg_equity != 0:
        roe = net_income_ttm / avg_equity
        
    # ROCE
    ebit = _get_line_item(income_stmt, ['EBIT', 'Operating Income', 'ebit'])
    ebit_ttm = _ttm(ebit)
    total_assets = _get_line_item(balance_sheet, ['Total Assets', 'total_assets'])
    current_liab = _get_line_item(balance_sheet, ['Total Current Liabilities', 'Current Liabilities', 'current_liabilities'])
    roce = np.nan
    ta_val = _latest(total_assets)
    cl_val = _latest(current_liab)
    if not pd.isna(ebit_ttm) and not pd.isna(ta_val) and not pd.isna(cl_val):
        capital_employed = ta_val - cl_val
        if capital_employed != 0:
            roce = ebit_ttm / capital_employed
            
    # Earnings Stability
    eps = _get_line_item(income_stmt, ['Basic EPS', 'EPS', 'eps', 'Diluted EPS'])
    if eps.empty:
        # fallback: net income
        eps = net_income
    
    earnings_stability = np.nan
    valid_eps = eps.dropna()
    if len(valid_eps) >= 8:
        eps_8q = valid_eps.iloc[:8]
        mean_eps = eps_8q.mean()
        if mean_eps != 0:
            earnings_stability = 1 - (eps_8q.std() / abs(mean_eps))
            
    # Leverage
    total_debt = _get_line_item(balance_sheet, ['Total Debt', 'Long Term Debt', 'total_debt'])
    eq_val = _latest(equity)
    td_val = _latest(total_debt)
    leverage = np.nan
    if not pd.isna(td_val) and not pd.isna(eq_val) and eq_val != 0:
        leverage = -(td_val / eq_val)
        
    return {
        'roe': roe,
        'roce': roce,
        'earnings_stability': earnings_stability,
        'leverage': leverage
    }

def compute_cross_section(price_data: dict, pit_store, tickers: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """Compute quality factors for multiple tickers."""
    results = {}
    for ticker in tickers:
        results[ticker] = compute(price_data, pit_store, ticker, as_of)
    return pd.DataFrame.from_dict(results, orient='index')
