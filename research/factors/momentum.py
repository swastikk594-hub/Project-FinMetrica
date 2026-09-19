import pandas as pd
import numpy as np

def compute(price_data: pd.DataFrame, pit_store: 'PointInTimeStore', ticker: str, as_of: pd.Timestamp) -> dict[str, float]:
    """
    Compute raw momentum characteristics for a single security at a given point in time.

    Characteristics:
    - 12m_momentum_skip1: (P_t-21 / P_t-252) - 1. Skips most recent month.
    - 6m_momentum: (P_t / P_t-126) - 1.

    Args:
        price_data: DataFrame with daily pricing, including 'close'.
        pit_store: PointInTimeStore instance (unused directly here).
        ticker: Ticker symbol.
        as_of: Target calculation date.

    Returns:
        Dict mapping characteristic names to raw values.
    """
    res = {
        '12m_momentum_skip1': np.nan,
        '6m_momentum': np.nan
    }
    
    if price_data is None or price_data.empty:
        return res
        
    prices = price_data.loc[:as_of]
    if len(prices) < 252:
        return res
        
    try:
        close_col = next((c for c in ('Close', 'Adj Close', 'close') if c in prices.columns), None)
        if close_col is None:
            return res
        closes = prices[close_col].values
        
        # 12m momentum skip 1 month (approx 21 trading days)
        if len(closes) >= 252:
            p_t_21 = closes[-21]
            p_t_252 = closes[-252]
            if p_t_252 > 0:
                res['12m_momentum_skip1'] = (p_t_21 / p_t_252) - 1.0
                
        # 6m momentum (approx 126 trading days)
        if len(closes) >= 126:
            p_t = closes[-1]
            p_t_126 = closes[-126]
            if p_t_126 > 0:
                res['6m_momentum'] = (p_t / p_t_126) - 1.0
                
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error computing momentum for {ticker} on {as_of}: {e}")
        
    return res

def compute_cross_section(price_data: dict[str, pd.DataFrame], pit_store: 'PointInTimeStore', tickers: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Compute raw momentum characteristics for all securities in the cross-section.

    Args:
        price_data: Dictionary mapping ticker to its price DataFrame.
        pit_store: PointInTimeStore instance.
        tickers: List of ticker symbols.
        as_of: Target calculation date.

    Returns:
        DataFrame with rows=tickers, columns=characteristics.
    """
    results = []
    for ticker in tickers:
        p_data = price_data.get(ticker, pd.DataFrame())
        vals = compute(p_data, pit_store, ticker, as_of)
        vals['ticker'] = ticker
        results.append(vals)
        
    df = pd.DataFrame(results).set_index('ticker')
    return df
