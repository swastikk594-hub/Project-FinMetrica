"""
Returns Calculation Module
==========================
IMPORTANT DESIGN DECISION: Log Returns vs Arithmetic Returns

PROBLEM: We need a return measure for time-series analysis.

METHOD A — Arithmetic Return:
  r_t = (P_t - P_{t-1}) / P_{t-1}
  Assumptions: Returns are i.i.d. and can compound linearly
  Advantages: Intuitive, correct for portfolio aggregation (cross-sectional)
  Disadvantages: Not time-additive, can exceed -100%, not normally distributed
  
METHOD B — Logarithmic Return:
  r_t = ln(P_t / P_{t-1})
  Assumptions: Prices follow geometric Brownian motion
  Advantages: Time-additive, symmetrical, approximately normally distributed for small r
  Disadvantages: Not portfolio-additive (r_portfolio ≠ sum of log returns of components)
  
SELECTED: Log returns for time-series analysis (Modules 3, 4, 5, 6)
          Arithmetic returns for portfolio aggregation (Module 7)
          
RATIONALE: In quantitative finance literature, log returns are standard for
           single-asset time-series analysis because they satisfy the requirement
           of time-additivity: r(t1,t3) = r(t1,t2) + r(t2,t3).
           However, portfolio weights apply to arithmetic returns.
           We implement BOTH and use each where mathematically appropriate.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def log_returns(prices: pd.Series, dropna: bool = True) -> pd.Series:
    """
    Calculate logarithmic returns from a series of prices.
    
    The logarithmic return is defined as:
        r_t = ln(P_t / P_{t-1})
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices.
    dropna : bool, optional
        Whether to drop NaN values that result from calculation, by default True
        
    Returns
    -------
    pd.Series
        Logarithmic returns series.
    """
    try:
        # Avoid log of zero or negative numbers
        if (prices <= 0).any():
            logger.warning("Prices containing zero or negative values detected. Setting to NaN before log.")
            prices = prices.where(prices > 0)
        
        returns = np.log(prices / prices.shift(1))
        
        if dropna:
            returns = returns.dropna()
            
        return returns
    except Exception as e:
        logger.error(f"Error calculating log returns: {e}")
        raise

def arithmetic_returns(prices: pd.Series, dropna: bool = True) -> pd.Series:
    """
    Calculate arithmetic (simple) returns from a series of prices.
    
    The arithmetic return is defined as:
        r_t = (P_t - P_{t-1}) / P_{t-1}
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices.
    dropna : bool, optional
        Whether to drop NaN values that result from calculation, by default True
        
    Returns
    -------
    pd.Series
        Arithmetic returns series.
    """
    try:
        returns = prices.pct_change()
        
        if dropna:
            returns = returns.dropna()
            
        return returns
    except Exception as e:
        logger.error(f"Error calculating arithmetic returns: {e}")
        raise

def cumulative_return(returns: pd.Series, method: str = 'arithmetic') -> pd.Series:
    """
    Compute cumulative return series from a series of returns.
    
    For log returns:
        C_t = exp(sum_{i=1}^t r_i) - 1
    For arithmetic returns:
        C_t = prod_{i=1}^t (1 + r_i) - 1
        
    Parameters
    ----------
    returns : pd.Series
        Time series of returns.
    method : str, optional
        Method used for returns calculation ('arithmetic' or 'log'), by default 'arithmetic'
        
    Returns
    -------
    pd.Series
        Cumulative returns series.
    """
    try:
        if method == 'log':
            cum_ret = np.exp(returns.cumsum()) - 1
        elif method == 'arithmetic':
            cum_ret = (1 + returns).cumprod() - 1
        else:
            raise ValueError("Method must be 'arithmetic' or 'log'.")
            
        return cum_ret
    except Exception as e:
        logger.error(f"Error calculating cumulative return: {e}")
        raise

def rolling_return(prices: pd.Series, window: int, method: str = 'log') -> pd.Series:
    """
    Calculate rolling N-period return.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices.
    window : int
        Number of periods for the rolling window.
    method : str, optional
        Method used for returns ('arithmetic' or 'log'), by default 'log'
        
    Returns
    -------
    pd.Series
        Rolling N-period returns series.
    """
    try:
        if method == 'log':
            # r(t-w, t) = ln(P_t / P_{t-w})
            ret = np.log(prices / prices.shift(window))
        elif method == 'arithmetic':
            ret = (prices - prices.shift(window)) / prices.shift(window)
        else:
            raise ValueError("Method must be 'arithmetic' or 'log'.")
            
        return ret
    except Exception as e:
        logger.error(f"Error calculating rolling return: {e}")
        raise

def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualised return from a series of returns.
    
    Geometric annualisation is computed as:
        (1 + total_return)^(periods_per_year/n) - 1
        
    Parameters
    ----------
    returns : pd.Series
        Time series of arithmetic returns.
    periods_per_year : int, optional
        Number of periods in a year, by default 252
        
    Returns
    -------
    float
        Annualised return.
    """
    try:
        n_periods = len(returns.dropna())
        if n_periods == 0:
            return np.nan
            
        total_return = (1 + returns).prod() - 1
        ann_ret = (1 + total_return) ** (periods_per_year / n_periods) - 1
        return float(ann_ret)
    except Exception as e:
        logger.error(f"Error calculating annualised return: {e}")
        raise

def compare_return_methods(prices: pd.Series) -> pd.DataFrame:
    """
    Compute both log and arithmetic returns and return a comparison DataFrame.
    
    Parameters
    ----------
    prices : pd.Series
        Time series of prices.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with comparison metrics.
    """
    try:
        arith_ret = arithmetic_returns(prices, dropna=True)
        log_ret = log_returns(prices, dropna=True)
        
        comparison = pd.DataFrame(index=['Arithmetic', 'Logarithmic'])
        
        for name, ret_series in zip(['Arithmetic', 'Logarithmic'], [arith_ret, log_ret]):
            comparison.loc[name, 'Mean'] = ret_series.mean()
            comparison.loc[name, 'Std'] = ret_series.std()
            comparison.loc[name, 'Skewness'] = ret_series.skew()
            comparison.loc[name, 'Kurtosis'] = ret_series.kurtosis()
            
        # Add correlation between the two series
        corr = arith_ret.corr(log_ret)
        comparison['Correlation with Other'] = corr
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparing return methods: {e}")
        raise
