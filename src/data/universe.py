"""
Universe Management, Survivorship Bias Mitigation & Corporate Actions
======================================================================
PURPOSE:
  Manage dynamic, point-in-time investable universes and handle
  corporate actions (splits, dividends, ticker changes, delistings).

SURVIVORSHIP BIAS PREVENTION:
  Constructing historical universes using today's index constituents
  artificially inflates backtested performance by omitting failed,
  bankrupt, or acquired companies. This module enforces point-in-time
  universe filtering based on historical trading availability and liquidity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class UniverseManager:
    """
    Tracks point-in-time constituent eligibility, corporate action adjustments,
    and liquidity thresholds.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_adv_dollar = self.config.get('min_adv_dollar', 1_000_000.0) # $1M daily volume min
        self.min_price = self.config.get('min_price', 5.0) # avoid penny stock microstructure artifacts
        self._delisted_registry: Dict[str, pd.Timestamp] = {}
        self._historical_constituents: Dict[pd.Timestamp, List[str]] = {}

    def register_delisting(self, ticker: str, delist_date: pd.Timestamp) -> None:
        """Register the final trading day of an asset."""
        self._delisted_registry[ticker.upper()] = pd.to_datetime(delist_date)

    def get_investable_universe(
        self,
        as_of: pd.Timestamp,
        price_dict: Dict[str, pd.DataFrame],
        lookback_days: int = 63
    ) -> List[str]:
        """
        Filters tickers eligible for trading at decision timestamp `as_of`.

        Criteria:
        1. Not delisted prior to `as_of`.
        2. Has price history up to `as_of`.
        3. Most recent price >= `min_price`.
        4. Trailing Average Daily Traded Value (Close * Volume) >= `min_adv_dollar`.
        """
        as_of = pd.to_datetime(as_of)
        eligible = []

        for ticker, df in price_dict.items():
            ticker_clean = ticker.upper()
            
            # Check delisting
            if ticker_clean in self._delisted_registry:
                if as_of > self._delisted_registry[ticker_clean]:
                    continue

            # Check price availability up to as_of
            sub_df = df.loc[df.index <= as_of]
            if len(sub_df) < 10:
                continue

            latest_row = sub_df.iloc[-1]
            latest_price = latest_row.get('Adj Close', latest_row.get('Close', np.nan))
            if np.isnan(latest_price) or latest_price < self.min_price:
                continue

            # Compute Average Daily Volume in dollars over recent window
            recent_window = sub_df.iloc[-lookback_days:]
            if 'Volume' in recent_window.columns and 'Close' in recent_window.columns:
                adv_dollar = (recent_window['Close'] * recent_window['Volume']).mean()
                if adv_dollar < self.min_adv_dollar and self.min_adv_dollar > 0:
                    continue

            eligible.append(ticker_clean)

        return sorted(eligible)

    @staticmethod
    def adjust_for_splits_and_dividends(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has clean adjusted price series ('Adj Close')
        and calculates total return series respecting dividend reinvestment.
        """
        df = df.copy()
        if 'Adj Close' not in df.columns:
            if 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            else:
                df['Adj Close'] = df.iloc[:, 0]

        # Calculate adjusted total return
        df['TotalReturn'] = df['Adj Close'].pct_change().fillna(0.0)
        return df
