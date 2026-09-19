"""
Test suite for src/preprocessing/returns.py

All tests are self-contained — they do not require API calls.
They verify mathematical correctness against hand-computed expected values.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.returns import (
    log_returns,
    arithmetic_returns,
    cumulative_return,
    rolling_return,
    annualised_return,
    compare_return_methods,
)


class TestLogReturns:
    """Tests for log_returns() function."""

    def test_log_return_known_value(self):
        """
        MATHEMATICAL VERIFICATION:
        log_return(110/100) = ln(1.1) = 0.09531...
        """
        prices = pd.Series([100.0, 110.0])
        returns = log_returns(prices)
        expected = np.log(1.1)  # = 0.09531017980...
        assert np.isclose(returns.iloc[-1], expected, atol=1e-8), (
            f"Expected {expected:.8f}, got {returns.iloc[-1]:.8f}"
        )

    def test_log_return_time_additivity(self):
        """
        MATHEMATICAL PROPERTY:
        r(t1, t3) = r(t1, t2) + r(t2, t3)
        i.e. ln(P3/P1) = ln(P2/P1) + ln(P3/P2)
        
        This is the key advantage of log returns.
        """
        prices = pd.Series([100.0, 110.0, 121.0])
        daily_rets = log_returns(prices)
        
        total_via_sum = daily_rets.sum()
        total_direct = np.log(121.0 / 100.0)
        
        assert np.isclose(total_via_sum, total_direct, atol=1e-10), (
            f"Time-additivity violated: sum={total_via_sum:.10f}, direct={total_direct:.10f}"
        )

    def test_log_returns_for_positive_prices(self):
        """Log returns should be finite for all positive prices."""
        prices = pd.Series([100.0, 50.0, 200.0, 75.0, 0.01])
        returns = log_returns(prices)
        assert np.all(np.isfinite(returns)), "Log returns must be finite for positive prices"

    def test_log_return_of_zero_price_handled(self):
        """Zero or negative prices should not produce -inf unhandled."""
        prices = pd.Series([100.0, 0.0, 110.0])
        # Should produce NaN (not crash) for the zero-price observation
        returns = log_returns(prices, dropna=False)
        # At least one value should be NaN (the zero-price period)
        # and none should be -inf
        assert not np.any(np.isinf(returns.values)), (
            "Zero price should result in NaN, not uncaught -inf"
        )

    def test_log_returns_drop_first_nan(self):
        """First observation should be dropped (no previous price)."""
        prices = pd.Series([100.0, 110.0, 121.0])
        returns = log_returns(prices, dropna=True)
        assert len(returns) == 2, f"Expected 2 observations, got {len(returns)}"


class TestArithmeticReturns:
    """Tests for arithmetic_returns() function."""

    def test_arithmetic_return_known_value(self):
        """
        MATHEMATICAL VERIFICATION:
        arithmetic_return(110, 100) = (110 - 100) / 100 = 0.10
        """
        prices = pd.Series([100.0, 110.0])
        returns = arithmetic_returns(prices)
        assert np.isclose(returns.iloc[-1], 0.10, atol=1e-8), (
            f"Expected 0.10, got {returns.iloc[-1]:.8f}"
        )

    def test_arithmetic_portfolio_aggregation(self):
        """
        MATHEMATICAL PROPERTY:
        Portfolio arithmetic return = weighted average of individual arithmetic returns.
        This is the key advantage of arithmetic returns for portfolio aggregation.
        
        r_p = Σ w_i * r_i
        """
        prices_A = pd.Series([100.0, 110.0])
        prices_B = pd.Series([100.0, 90.0])
        
        ret_A = arithmetic_returns(prices_A).iloc[-1]  # = 0.10
        ret_B = arithmetic_returns(prices_B).iloc[-1]  # = -0.10
        
        weights = [0.6, 0.4]
        portfolio_return = weights[0] * ret_A + weights[1] * ret_B
        expected = 0.6 * 0.10 + 0.4 * (-0.10)  # = 0.02
        
        assert np.isclose(portfolio_return, expected, atol=1e-8)

    def test_arithmetic_NOT_time_additive(self):
        """
        MATHEMATICAL PROPERTY:
        Arithmetic returns are NOT time-additive.
        (1 + r1)(1 + r2) - 1 ≠ r1 + r2
        
        This confirms why log returns are used for time-series analysis.
        """
        prices = pd.Series([100.0, 110.0, 121.0])
        daily_rets = arithmetic_returns(prices)
        
        # Incorrect (simple sum)
        simple_sum = daily_rets.sum()  # r1 + r2 = 0.1 + 0.1 = 0.2
        
        # Correct (geometric compounding)
        correct = (1 + daily_rets).prod() - 1  # = 1.1 * 1.1 - 1 = 0.21
        
        # They should differ
        assert not np.isclose(simple_sum, correct, atol=1e-6), (
            "This test should FAIL if arithmetic returns were time-additive "
            "(they should NOT be)"
        )


class TestCumulativeReturn:
    """Tests for cumulative_return() function."""

    def test_cumulative_log_return_consistency(self):
        """
        MATHEMATICAL VERIFICATION:
        Cumulative log return = exp(Σ log_returns) - 1
        Should equal overall price change.
        """
        prices = pd.Series([100.0, 105.0, 102.0, 110.0])
        log_rets = log_returns(prices)
        
        # Cumulative return via cumulative_return function
        cum = cumulative_return(log_rets, method='log')
        
        # Direct calculation
        expected = (110.0 / 100.0) - 1.0
        
        assert np.isclose(cum.iloc[-1], expected, atol=1e-8)

    def test_cumulative_arithmetic_return(self):
        """
        Cumulative arithmetic return = Π(1 + r_t) - 1
        """
        prices = pd.Series([100.0, 110.0, 99.0])
        arith_rets = arithmetic_returns(prices)
        cum = cumulative_return(arith_rets, method='arithmetic')
        expected = (99.0 / 100.0) - 1.0  # = -0.01
        assert np.isclose(cum.iloc[-1], expected, atol=1e-8)


class TestAnnualisedReturn:
    """Tests for annualised_return() function."""

    def test_annualised_return_over_one_year(self):
        """
        If a stock earns exactly 10% total over 252 trading days,
        the annualised return should be 10%.
        """
        # Construct 252 daily returns that compound to exactly 10%
        daily_rate = (1.10 ** (1.0 / 252)) - 1.0
        returns = pd.Series([daily_rate] * 252)
        
        ann_ret = annualised_return(returns, periods_per_year=252)
        assert np.isclose(ann_ret, 0.10, atol=1e-6), (
            f"Expected 10%, got {ann_ret:.6%}"
        )

    def test_empty_series_returns_nan(self):
        """Empty series should return NaN gracefully."""
        returns = pd.Series(dtype=float)
        ann_ret = annualised_return(returns)
        assert np.isnan(ann_ret)


class TestCompareReturnMethods:
    """Tests for compare_return_methods() function."""

    def test_comparison_returns_dataframe(self):
        """compare_return_methods should return a non-empty DataFrame."""
        prices = pd.Series([100.0, 105.0, 102.0, 110.0, 108.0])
        result = compare_return_methods(prices)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_comparison_includes_both_methods(self):
        """Comparison should include data for both return methods."""
        prices = pd.Series([100.0, 105.0, 102.0, 110.0, 108.0])
        result = compare_return_methods(prices)
        # Should have at least 2 rows (one per method) or 2 columns
        assert result.shape[0] >= 2 or result.shape[1] >= 2
