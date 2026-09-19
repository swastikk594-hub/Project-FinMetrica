"""
Tests for performance significance tests.
"""

import pytest
import numpy as np
from src.stats.performance_tests import jobson_korkie_memmel, stationary_bootstrap_sharpe_diff


def test_jobson_korkie_identical():
    """Identical series should have p-value ~1.0."""
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01 + 0.001
    
    z_stat, p_val = jobson_korkie_memmel(returns, returns)
    assert np.isclose(z_stat, 0.0)
    assert np.isclose(p_val, 1.0)


def test_jobson_korkie_different():
    """Obviously different series should have low p-value."""
    np.random.seed(42)
    # Series A: positive drift, low noise
    returns_a = np.random.randn(500) * 0.005 + 0.002
    # Series B: negative drift, high noise
    returns_b = np.random.randn(500) * 0.02 - 0.001
    
    z_stat, p_val = jobson_korkie_memmel(returns_a, returns_b)
    # A should be significantly better than B
    assert z_stat > 2.0
    assert p_val < 0.05


def test_stationary_bootstrap_identical():
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01 + 0.001
    
    p_val, diff = stationary_bootstrap_sharpe_diff(returns, returns, n_bootstraps=100)
    assert np.isclose(diff, 0.0)
    assert np.isclose(p_val, 1.0)


def test_stationary_bootstrap_different():
    np.random.seed(42)
    returns_a = np.random.randn(500) * 0.005 + 0.002
    returns_b = np.random.randn(500) * 0.02 - 0.001
    
    p_val, diff = stationary_bootstrap_sharpe_diff(returns_a, returns_b, n_bootstraps=200)
    assert diff > 0.0
    assert p_val < 0.05
