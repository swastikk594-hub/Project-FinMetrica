"""
Test to ensure Sharpe ratio calculations explicitly subtract the risk-free rate 
before computation (excess returns).
"""

import pytest
import numpy as np
import pandas as pd
from src.backtesting.engine import EventDrivenBacktester
from src.modules.m5_risk import RiskModule

def test_sharpe_uses_excess_returns():
    # Synthetic returns: mean 0.001 daily
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    returns_series = pd.Series(returns)
    
    # 0.05 annualized approx 0.05 / 252 daily
    rf_daily = 0.05 / 252
    
    excess_returns = returns_series - rf_daily
    
    # Hand-calculate
    expected_excess_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    raw_sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
    
    # Test Backtester Engine
    engine = EventDrivenBacktester()
    # We can test compute_performance_metrics directly
    metrics = engine.compute_performance_metrics(returns_series, returns_series, risk_free_daily=rf_daily)
    
    # Assert engine calculation matches expected excess sharpe
    # (Allow small floating point diff)
    assert np.isclose(metrics['sharpe_ratio'], expected_excess_sharpe), "Engine did not compute excess Sharpe correctly."
    # Assert it DOES NOT match raw sharpe
    assert not np.isclose(metrics['sharpe_ratio'], raw_sharpe), "Engine used raw returns for Sharpe, not excess."
    
    # Test m5_risk
    risk_mod = RiskModule()
    m5_metrics = risk_mod.compute_sharpe_ratio(returns_series, risk_free_daily=rf_daily)
    
    assert np.isclose(m5_metrics['sharpe_ratio'], expected_excess_sharpe), "m5_risk did not compute excess Sharpe correctly."
    assert not np.isclose(m5_metrics['sharpe_ratio'], raw_sharpe), "m5_risk used raw returns for Sharpe, not excess."
