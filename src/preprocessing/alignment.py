"""
Alignment Module
================
Provides functions to align time series data of varying frequencies
and properly handle look-ahead bias through strict lagging and 
point-in-time considerations.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def align_to_business_days(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Reindex a DataFrame to a business day calendar, filling missing days.
    Forward fills up to a maximum of 5 days to cover weekends and short holidays.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with DatetimeIndex.
    start : str
        Start date string (e.g., 'YYYY-MM-DD').
    end : str
        End date string (e.g., 'YYYY-MM-DD').
        
    Returns
    -------
    pd.DataFrame
        Business day aligned DataFrame.
    """
    try:
        bday_index = pd.bdate_range(start=start, end=end)
        # Reindex to new business day index and forward fill up to 5 days
        aligned = df.reindex(bday_index).ffill(limit=5)
        return aligned
    except Exception as e:
        logger.error(f"Error in align_to_business_days: {e}")
        raise

def align_monthly_to_daily(monthly_df: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align monthly data (e.g., macro or Fama-French) to a daily price index.
    
    CRITICAL: Ensures no look-ahead bias by reindexing to daily and then 
    forward-filling the previous month's data up to 31 days.
    
    Parameters
    ----------
    monthly_df : pd.DataFrame
        DataFrame with monthly frequency and DatetimeIndex.
    daily_index : pd.DatetimeIndex
        The daily index to align to.
        
    Returns
    -------
    pd.DataFrame
        Daily frequency DataFrame of the monthly data.
    """
    try:
        # First ensure the monthly data is sorted
        monthly_df = monthly_df.sort_index()
        
        # Reindex to the daily index, which puts the monthly values on their exact dates
        aligned = monthly_df.reindex(daily_index)
        
        # Forward fill up to 31 days to carry the previous month's value forward
        aligned = aligned.ffill(limit=31)
        
        return aligned
    except Exception as e:
        logger.error(f"Error in align_monthly_to_daily: {e}")
        raise

def align_quarterly_to_daily(quarterly_df: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align quarterly fundamental data to a daily price index.
    
    CRITICAL: Prevents look-ahead bias by forward filling the previously 
    available quarterly data point up to 92 days.
    
    Parameters
    ----------
    quarterly_df : pd.DataFrame
        DataFrame with quarterly frequency and DatetimeIndex.
    daily_index : pd.DatetimeIndex
        The daily index to align to.
        
    Returns
    -------
    pd.DataFrame
        Daily frequency DataFrame of the quarterly data.
    """
    try:
        quarterly_df = quarterly_df.sort_index()
        
        aligned = quarterly_df.reindex(daily_index)
        # Forward fill up to 92 days (roughly one quarter)
        aligned = aligned.ffill(limit=92)
        
        return aligned
    except Exception as e:
        logger.error(f"Error in align_quarterly_to_daily: {e}")
        raise

def lag_series(df: pd.DataFrame, lag_periods: int) -> pd.DataFrame:
    """
    Shift data forward in time to prevent look-ahead bias.
    
    Parameters
    ----------
    df : pd.DataFrame
        Time series DataFrame.
    lag_periods : int
        Number of periods to shift data forward.
        
    Returns
    -------
    pd.DataFrame
        Lagged DataFrame.
    """
    try:
        if lag_periods < 0:
            logger.warning(f"Negative lag_periods ({lag_periods}) will introduce look-ahead bias!")
            
        return df.shift(lag_periods)
    except Exception as e:
        logger.error(f"Error in lag_series: {e}")
        raise

def point_in_time_fundamentals(fundamental_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.Series:
    """
    Return the most recently available fundamental data exactly as of a given date.
    
    Parameters
    ----------
    fundamental_df : pd.DataFrame
        DataFrame containing fundamental data, indexed by publication/filing date.
    as_of_date : pd.Timestamp
        The exact point in time to evaluate.
        
    Returns
    -------
    pd.Series
        The fundamental data row available at 'as_of_date'.
    """
    try:
        # Filter for all data that was available on or before the given date
        available_data = fundamental_df.loc[fundamental_df.index <= as_of_date]
        
        if available_data.empty:
            logger.warning(f"No fundamental data available prior to {as_of_date}")
            return pd.Series(dtype=float)
            
        # Return the most recent available data
        return available_data.iloc[-1]
    except Exception as e:
        logger.error(f"Error in point_in_time_fundamentals: {e}")
        raise
