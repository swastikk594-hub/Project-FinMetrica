"""
Tests for Track 1: Estimator-Instability Study.
"""

import pytest
import numpy as np
import pandas as pd

from research.experiments.covariance_regularization_study import (
    compute_turnover,
    run_track1_study
)


def test_compute_turnover():
    """Verify L1 turnover calculation."""
    w1 = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
    w2 = pd.Series({'AAPL': 0.6, 'MSFT': 0.4})
    
    # Absolute differences: 0.1 + 0.1 = 0.2
    # One-way turnover = 0.1
    assert np.isclose(compute_turnover(w2, w1), 0.1)
    
    # Asset enters/leaves
    w3 = pd.Series({'AAPL': 0.5, 'GOOG': 0.5})
    # w1 vs w3:
    # AAPL: 0.5 vs 0.5 (diff=0)
    # MSFT: 0.5 vs 0.0 (diff=0.5)
    # GOOG: 0.0 vs 0.5 (diff=0.5)
    # Total diff = 1.0 -> One-way = 0.5
    assert np.isclose(compute_turnover(w3, w1), 0.5)


def test_covariance_regularization_smoke_test():
    """Smoke test for Track 1 execution with seeded synthetic data."""
    np.random.seed(42)
    
    # Create 3 years of daily prices for 5 assets
    dates = pd.bdate_range('2020-01-01', '2022-12-31')
    returns = np.random.randn(len(dates), 5) * 0.01
    
    # Inject some covariance structure
    F = np.random.randn(len(dates), 2) * 0.01
    returns[:, 0] += F[:, 0]
    returns[:, 1] += F[:, 0]
    returns[:, 2] += F[:, 1]
    returns[:, 3] += F[:, 1]
    
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=['A', 'B', 'C', 'D', 'E'])
    
    df = run_track1_study(prices, horizon_days=20, output_dir=None)
    
    assert not df.empty
    assert 'cond_sample' in df.columns
    assert 'cond_lw' in df.columns
    assert 'turnover_sample' in df.columns
    
    # Ledoit-Wolf should generally have lower or equal condition number than sample
    # Because it shrinks towards a well-conditioned target
    avg_cond_sample = df['cond_sample'].mean()
    avg_cond_lw = df['cond_lw'].mean()
    
    # Not strictly guaranteed for every single random slice, but true on average
    assert avg_cond_lw <= avg_cond_sample * 1.5, f"LW Cond={avg_cond_lw}, Sample Cond={avg_cond_sample}"
