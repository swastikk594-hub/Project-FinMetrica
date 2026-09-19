"""
Value factor computation using PointInTimeStore API.
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
    # Assumes columns are most-recent-first
    return float(valid.iloc[:n_quarters].sum())

def _latest(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.iloc[0])

def compute(price_data: dict, pit_store, ticker: str, as_of: pd.Timestamp) -> dict[str, float]:
    """
    Compute Value factors for a given ticker as of a specific date.
    
    Args:
        price_data: (Ignored, use pit_store)
        pit_store: PointInTimeStore instance
        ticker: Ticker symbol
        as_of: Date for computation
        
    Returns:
        dict: Mapping of factor names to computed values
    """
    hist = pit_store.get_price_history(ticker, as_of)
    if hist.empty:
        return {'earnings_yield': np.nan, 'book_to_market': np.nan, 'cash_flow_yield': np.nan}
    
    # Try multiple variants for Close
    close_price = np.nan
    for col in ['Close', 'close', 'Adj Close']:
        if col in hist.columns:
            close_price = hist[col].iloc[-1]
            break
            
    if pd.isna(close_price):
         return {'earnings_yield': np.nan, 'book_to_market': np.nan, 'cash_flow_yield': np.nan}

    fundas = pit_store.get_point_in_time_fundamentals(ticker, as_of)
    income_stmt = fundas.get('income_statement', pd.DataFrame())
    balance_sheet = fundas.get('balance_sheet', pd.DataFrame())
    cash_flow = fundas.get('cash_flow', pd.DataFrame())
    
    shares = _get_line_item(income_stmt, ['Diluted Average Shares', 'Shares Outstanding', 'shares_outstanding', 'Common Stock', 'Basic Average Shares'])
    shares_out = _latest(shares)
    
    if pd.isna(shares_out):
        shares_out = 1.0  # Fallback
    
    market_cap = close_price * shares_out
    if market_cap == 0:
        return {'earnings_yield': np.nan, 'book_to_market': np.nan, 'cash_flow_yield': np.nan}
        
    net_income = _get_line_item(income_stmt, ['Net Income', 'NetIncome', 'net_income'])
    earnings_ttm = _ttm(net_income)
    earnings_yield = earnings_ttm / market_cap if not pd.isna(earnings_ttm) else np.nan
    
    book_value = _get_line_item(balance_sheet, ["Total Shareholders' Equity", 'Total Equity', 'Book Value', 'book_value'])
    bv_latest = _latest(book_value)
    book_to_market = bv_latest / market_cap if not pd.isna(bv_latest) else np.nan
    
    ocf = _get_line_item(cash_flow, ['Operating Cash Flow', 'Cash Flow From Operations', 'operating_cash_flow', 'Net Cash Provided By Operating Activities'])
    ocf_ttm = _ttm(ocf)
    cash_flow_yield = ocf_ttm / market_cap if not pd.isna(ocf_ttm) else np.nan
    
    return {
        'earnings_yield': earnings_yield,
        'book_to_market': book_to_market,
        'cash_flow_yield': cash_flow_yield
    }

def compute_cross_section(price_data: dict, pit_store, tickers: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """Compute value factors for multiple tickers."""
    results = {}
    for ticker in tickers:
        results[ticker] = compute(price_data, pit_store, ticker, as_of)
    return pd.DataFrame.from_dict(results, orient='index')
