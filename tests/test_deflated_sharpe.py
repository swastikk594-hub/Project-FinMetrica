"""
Tests for the Deflated Sharpe Ratio (DSR) implementation.
"""

import pytest
import numpy as np

from src.stats.deflated_sharpe import (
    expected_max_sharpe_under_multiple_trials,
    deflated_sharpe_ratio
)


def test_expected_max_sharpe():
    """Verify expected maximum Sharpe ratio increases with trials and variance."""
    # N=1 -> SR* = 0
    assert expected_max_sharpe_under_multiple_trials(1, 0.5) == 0.0
    
    # N=10, var=0.1
    sr10 = expected_max_sharpe_under_multiple_trials(10, 0.1)
    
    # N=100, var=0.1 -> should be greater than N=10
    sr100 = expected_max_sharpe_under_multiple_trials(100, 0.1)
    assert sr100 > sr10
    
    # N=100, var=0.2 -> should be greater than var=0.1
    sr100_v2 = expected_max_sharpe_under_multiple_trials(100, 0.2)
    assert sr100_v2 > sr100


def test_deflated_sharpe_ratio():
    """Verify DSR behaves as a probability and penalizes multiple trials."""
    n_obs = 1000
    observed_sr = 0.1
    var = 0.05
    
    # With 1 trial, DSR = Probabilistic Sharpe Ratio
    dsr_1 = deflated_sharpe_ratio(observed_sr, 1, var, n_obs)
    assert 0 < dsr_1 <= 1.0
    
    # With 100 trials, the threshold increases, so DSR should decrease
    dsr_100 = deflated_sharpe_ratio(observed_sr, 100, var, n_obs)
    assert dsr_100 < dsr_1
    
    # With 1000 trials, DSR should decrease further
    dsr_1000 = deflated_sharpe_ratio(observed_sr, 1000, var, n_obs)
    assert dsr_1000 < dsr_100
    
    # High kurtosis increases uncertainty (fattens tails), so DSR should drop
    dsr_kurt = deflated_sharpe_ratio(observed_sr, 1, var, n_obs, skew=0.0, kurtosis=6.0)
    assert dsr_kurt < dsr_1
