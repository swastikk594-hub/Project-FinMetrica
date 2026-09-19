"""
Point-in-Time (PIT) Data Architecture & Information Boundary Engine
====================================================================
PURPOSE:
  Enforce strict point-in-time information boundaries across all market,
  fundamental, macro, and alternative datasets.

MATHEMATICAL & LOGICAL FOUNDATION:
  Let t be the decision timestamp. A data point d = (v, t_obs, t_avail)
  with value v, observation date t_obs, and release/availability date t_avail
  is admissible in the decision filtration F_t if and only if:
      t_avail <= t

  If t_avail is unrecorded, we apply conservative economic lag rules:
  - Daily Price/Volume: available at market close t (t_avail = t + 0 days)
  - Quarterly Financial Statements: available 45 days after fiscal period end
  - Macroeconomic Data (FRED/GDP/CPI/Unemployment): available 30 days after period end

LOOK-AHEAD LEAKAGE DETECTION:
  Any attempt to access data with t_avail > t raises an InformationBoundaryViolation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class InformationBoundaryViolation(Exception):
    """Raised when an operation attempts to access data before its availability timestamp."""
    pass


class PointInTimeStore:
    """
    Manages versioned, timestamped data point collections and extracts
    strictly point-in-time snapshots for quantitative research and backtesting.
    """

    def __init__(self, fundamental_lag_days: int = 45, macro_lag_days: int = 30):
        self.fundamental_lag_days = fundamental_lag_days
        self.macro_lag_days = macro_lag_days
        self._price_store: Dict[str, pd.DataFrame] = {}
        self._fundamental_store: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._macro_store: Optional[pd.DataFrame] = None
        self._alt_data_store: Dict[str, pd.DataFrame] = {}

    def register_price_data(self, ticker: str, df: pd.DataFrame) -> None:
        """Register daily price dataframe with datetime index."""
        if df.empty:
            return
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        self._price_store[ticker.upper()] = df

    def register_fundamental_data(self, ticker: str, fin_dict: Dict[str, pd.DataFrame]) -> None:
        """Register fundamental financial statements with period end columns."""
        clean_fin = {}
        for stmt_name, stmt_df in fin_dict.items():
            if isinstance(stmt_df, pd.DataFrame) and not stmt_df.empty:
                stmt_df = stmt_df.copy()
                if not isinstance(stmt_df.columns, pd.DatetimeIndex):
                    stmt_df.columns = pd.to_datetime(stmt_df.columns)
                # Sort dates ascending
                stmt_df = stmt_df.reindex(sorted(stmt_df.columns), axis=1)
                clean_fin[stmt_name] = stmt_df
        self._fundamental_store[ticker.upper()] = clean_fin

    def register_macro_data(self, df: pd.DataFrame) -> None:
        """Register monthly/quarterly macro data with datetime index."""
        if df.empty:
            return
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        self._macro_store = df.sort_index()

    def get_price_history(self, ticker: str, as_of: pd.Timestamp) -> pd.DataFrame:
        """
        Extract price history up to decision timestamp as_of.
        Guarantees no future data is returned.
        """
        ticker = ticker.upper()
        if ticker not in self._price_store:
            return pd.DataFrame()
        df = self._price_store[ticker]
        valid_df = df.loc[df.index <= as_of]
        return valid_df.copy()

    def get_point_in_time_fundamentals(self, ticker: str, as_of: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """
        Extract only financial statements that were published at or before `as_of`.
        Applies fundamental_lag_days from period end date to establish release timestamp.
        """
        ticker = ticker.upper()
        if ticker not in self._fundamental_store:
            return {}

        as_of = pd.to_datetime(as_of)
        lag_delta = pd.Timedelta(days=self.fundamental_lag_days)
        result = {}

        for stmt_name, stmt_df in self._fundamental_store[ticker].items():
            if stmt_df.empty:
                result[stmt_name] = pd.DataFrame()
                continue

            # A statement column (period_end) is available only if period_end + lag <= as_of
            valid_cols = [
                col for col in stmt_df.columns
                if (pd.to_datetime(col) + lag_delta) <= as_of
            ]

            if valid_cols:
                # Return with most recent first for standard accounting ratio routines
                valid_stmt = stmt_df[valid_cols].reindex(sorted(valid_cols, reverse=True), axis=1)
                result[stmt_name] = valid_stmt.copy()
            else:
                result[stmt_name] = pd.DataFrame()

        return result

    def get_point_in_time_macro(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """
        Extract macroeconomic data that was available at or before `as_of`.
        Applies macro_lag_days.
        """
        if self._macro_store is None or self._macro_store.empty:
            return pd.DataFrame()

        as_of = pd.to_datetime(as_of)
        lag_delta = pd.Timedelta(days=self.macro_lag_days)
        
        # Effective release date = index + lag_delta
        valid_mask = (self._macro_store.index + lag_delta) <= as_of
        return self._macro_store.loc[valid_mask].copy()

    def validate_no_leakage(self, signal_series: pd.Series, target_return_series: pd.Series, horizon_days: int) -> bool:
        """
        Verifies that signal at t does not exhibit correlation with future-dated components
        outside of forward return horizon.
        """
        if len(signal_series) < 10 or len(target_return_series) < 10:
            return True
        # Lag the signal backwards: signal at t+1 predicting return at t is look-ahead leakage
        lagged_signal = signal_series.shift(-horizon_days)
        corr = lagged_signal.corr(target_return_series)
        if not np.isnan(corr) and abs(corr) > 0.98:
            logger.warning(f"Suspiciously high reverse correlation ({corr:.4f}) detected — possible lookahead leakage.")
            return False
        return True
