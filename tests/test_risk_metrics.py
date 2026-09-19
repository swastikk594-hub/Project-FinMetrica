import pytest
import numpy as np
import pandas as pd
from src.modules.m5_risk import RiskModule


def test_cvar_gte_var():
    """Test that CVaR is always greater than or equal to VaR for the same confidence level."""
    m5 = RiskModule({})
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 1000))
    var_dict = m5.compute_var(returns)
    
    assert var_dict[0.95]['cvar'] >= var_dict[0.95]['var']
    assert var_dict[0.99]['cvar'] >= var_dict[0.99]['var']


def test_max_drawdown_known_value():
    """Test maximum drawdown calculation."""
    m5 = RiskModule({})
    # Prices: 100 -> 110 -> 60 -> 80
    prices = pd.Series([100.0, 110.0, 60.0, 80.0])
    mdd_res = m5.compute_max_drawdown(prices)
    # Expected: (60 - 110) / 110 = -0.4545
    assert np.isclose(mdd_res['max_drawdown'], -0.454545, atol=1e-4)


def test_sharpe_and_sortino_ratio():
    m5 = RiskModule({})
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
    sharpe_res = m5.compute_sharpe_ratio(returns, risk_free_daily=0.0)
    sortino_res = m5.compute_sortino_ratio(returns, risk_free_daily=0.0)
    
    assert 'sharpe_ratio' in sharpe_res
    assert 'sortino_ratio' in sortino_res
    assert not np.isnan(sharpe_res['sharpe_ratio'])
