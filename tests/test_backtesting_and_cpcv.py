"""
Unit Tests for CPCV, Embargo, Market Impact, and Strategy Qualification
"""

import pytest
import numpy as np
import pandas as pd

from src.backtesting.cpcv import PurgedCrossValidator
from src.execution.market_impact import MarketImpactModel
from src.reporting.qualification import StrategyQualificationEngine


def test_purged_cross_validation_purging():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    cv = PurgedCrossValidator(n_splits=5, horizon_days=10, embargo_pct=0.02)
    splits = list(cv.split(dates))

    assert len(splits) == 5
    for train_idx, test_idx in splits:
        # Verify no exact overlap
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_square_root_market_impact():
    impact_model = MarketImpactModel(y_coeff=0.30)
    
    # Small trade: $10,000 on $50M ADV
    small_cost = impact_model.calculate_cost(
        trade_dollar_value=10_000.0,
        adv_dollar=50_000_000.0,
        daily_volatility=0.02
    )
    assert small_cost['market_impact_bps'] < 2.0
    assert not small_cost['liquidity_breached']

    # Large institutional trade: $10,000,000 on $50M ADV (20% participation -> breach)
    large_cost = impact_model.calculate_cost(
        trade_dollar_value=10_000_000.0,
        adv_dollar=50_000_000.0,
        daily_volatility=0.02
    )
    assert large_cost['market_impact_bps'] > small_cost['market_impact_bps']
    assert large_cost['liquidity_breached']


def test_strategy_qualification_deflated_sharpe():
    qual = StrategyQualificationEngine(n_trials_conducted=1)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    
    # Strong strategy returns (mean 0.002, std 0.01 -> annualized Sharpe ~ 3.1)
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.002, 0.01, 300), index=dates)
    bm_rets = pd.Series(np.random.normal(0.0003, 0.01, 300), index=dates)

    verdict = qual.evaluate_strategy(rets, bm_rets, max_drawdown=-0.10)
    assert verdict['qualification_status'] in ['PASS', 'CONDITIONAL PASS']
    assert verdict['metrics']['deflated_sharpe_ratio'] > 0.80

    # Overfit / failing strategy (negative returns)
    fail_rets = pd.Series(np.random.normal(-0.001, 0.02, 300), index=dates)
    fail_verdict = qual.evaluate_strategy(fail_rets, bm_rets, max_drawdown=-0.45)
    assert fail_verdict['qualification_status'] in ['FAILED', 'OVERFIT']
