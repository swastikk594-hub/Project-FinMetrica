import pandas as pd
import numpy as np

def compute(price_data: pd.DataFrame, pit_store: 'PointInTimeStore', ticker: str, as_of: pd.Timestamp, market_returns: pd.Series = None) -> dict[str, float]:
    """
    Compute raw low-risk characteristics for a single security at a given point in time.

    Characteristics:
    - volatility_30d: Inverted annualized std dev of daily log returns over last 30 days.
    - beta: Inverted CAPM beta = Cov(r_i, r_m) / Var(r_m) over last 252 days.
    - downside_volatility: Inverted annualized std dev using only negative returns over last 30 days.

    Args:
        price_data: DataFrame with daily pricing, including 'close'.
        pit_store: PointInTimeStore instance.
        ticker: Ticker symbol.
        as_of: Target calculation date.
        market_returns: Series of daily market returns (required for beta).

    Returns:
        Dict mapping characteristic names to raw values.
    """
    res = {
        'volatility_30d': np.nan,
        'beta': np.nan,
        'downside_volatility': np.nan
    }
    
    if price_data is None or price_data.empty:
        return res
        
    prices = price_data.loc[:as_of]
    if len(prices) < 2:
        return res
        
    try:
        close_col = next((c for c in ('Close', 'Adj Close', 'close') if c in prices.columns), None)
        if close_col is None:
            return res
        closes = prices[close_col]
        returns = np.log(closes / closes.shift(1)).dropna()
        
        # Volatility 30d
        returns_30d = returns.iloc[-30:]
        if len(returns_30d) > 5:
            vol = returns_30d.std() * np.sqrt(252)
            if not pd.isna(vol):
                res['volatility_30d'] = -vol
                
        # Downside Volatility 30d
        neg_returns = returns_30d[returns_30d < 0]
        if len(neg_returns) > 3:
            downside_vol = neg_returns.std() * np.sqrt(252)
            if not pd.isna(downside_vol):
                res['downside_volatility'] = -downside_vol
                
        # Beta
        if market_returns is not None and len(returns) >= 252:
            returns_252 = returns.iloc[-252:]
            aligned_mkt = market_returns.reindex(returns_252.index).dropna()
            returns_252, aligned_mkt = returns_252.align(aligned_mkt, join='inner')
            
            if len(returns_252) > 100:
                cov_matrix = np.cov(returns_252.values, aligned_mkt.values)
                if cov_matrix.shape == (2, 2) and cov_matrix[1, 1] > 0:
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                    res['beta'] = -beta
                    
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error computing low_risk for {ticker} on {as_of}: {e}")
        
    return res

def compute_cross_section(price_data: dict[str, pd.DataFrame], pit_store: 'PointInTimeStore', tickers: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Compute raw low-risk characteristics for all securities in the cross-section.
    Calculates equal-weighted market returns dynamically.

    Args:
        price_data: Dictionary mapping ticker to its price DataFrame.
        pit_store: PointInTimeStore instance.
        tickers: List of ticker symbols.
        as_of: Target calculation date.

    Returns:
        DataFrame with rows=tickers, columns=characteristics.
    """
    # Calculate equal-weighted market return
    all_returns = []
    for t in tickers:
        p_df = price_data.get(t)
        if p_df is not None and not p_df.empty:
            close_col = next((c for c in ('Close', 'Adj Close', 'close') if c in p_df.columns), None)
            if close_col is None:
                continue
            closes = p_df.loc[:as_of, close_col]
            if len(closes) > 1:
                rets = np.log(closes / closes.shift(1))
                rets.name = t
                all_returns.append(rets)
                
    market_returns = None
    if all_returns:
        returns_df = pd.concat(all_returns, axis=1)
        market_returns = returns_df.mean(axis=1)
        
    results = []
    for ticker in tickers:
        p_data = price_data.get(ticker, pd.DataFrame())
        vals = compute(p_data, pit_store, ticker, as_of, market_returns=market_returns)
        vals['ticker'] = ticker
        results.append(vals)
        
    df = pd.DataFrame(results).set_index('ticker')
    return df
