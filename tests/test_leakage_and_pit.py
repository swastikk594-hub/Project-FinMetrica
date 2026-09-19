"""
Point-in-Time Boundary & Lookahead Leakage Prevention Unit Tests
"""

import pytest
import numpy as np
import pandas as pd

from src.data.point_in_time import PointInTimeStore
from src.data.universe import UniverseManager


def test_point_in_time_fundamental_lag():
    pit = PointInTimeStore(fundamental_lag_days=45)
    
    # Financial report for FY ending 2024-12-31
    period_end = pd.Timestamp("2024-12-31")
    fin_df = pd.DataFrame({
        period_end: {'Net Income': 1000.0, 'Total Assets': 5000.0}
    })
    pit.register_fundamental_data("TEST", {'income_stmt': fin_df})

    # Test query on 2025-01-15 (15 days after period end - NOT available yet)
    as_of_early = pd.Timestamp("2025-01-15")
    early_data = pit.get_point_in_time_fundamentals("TEST", as_of=as_of_early)
    assert early_data['income_stmt'].empty

    # Test query on 2025-02-20 (51 days after period end - available)
    as_of_valid = pd.Timestamp("2025-02-20")
    valid_data = pit.get_point_in_time_fundamentals("TEST", as_of=as_of_valid)
    assert not valid_data['income_stmt'].empty
    assert period_end in valid_data['income_stmt'].columns


def test_universe_survivorship_and_liquidity():
    mgr = UniverseManager(config={'min_adv_dollar': 500_000.0, 'min_price': 5.0})
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    price_dict = {
        'LIQUID': pd.DataFrame({'Close': [50.0]*100, 'Volume': [100_000]*100}, index=dates),
        'PENNY': pd.DataFrame({'Close': [2.0]*100, 'Volume': [100_000]*100}, index=dates),
        'ILLIQUID': pd.DataFrame({'Close': [50.0]*100, 'Volume': [100]*100}, index=dates)
    }

    as_of = dates[-1]
    eligible = mgr.get_investable_universe(as_of, price_dict)
    assert 'LIQUID' in eligible
    assert 'PENNY' not in eligible
    assert 'ILLIQUID' not in eligible
