import pytest
import numpy as np
import pandas as pd
from src.modules.m7_optimisation import InstitutionalPortfolioOptimiser


def test_portfolio_covariance_and_sharpe():
    m7 = InstitutionalPortfolioOptimiser()
    np.random.seed(42)
    rets = pd.DataFrame(np.random.randn(100, 3) * 0.02, columns=['A', 'B', 'C'])
    sigma = m7.estimate_covariance(rets)
    
    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(sigma.values)
    assert np.all(eigenvalues >= -1e-6)

    mu = pd.Series({'A': 0.12, 'B': 0.08, 'C': 0.06})
    opt = m7.optimise_max_sharpe(mu, sigma, risk_free_rate=0.04)
    
    weights = opt['weights']
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-4)
    assert all(w >= 0.0 for w in weights.values())
    assert all(w <= 0.40 + 1e-4 for w in weights.values())


def test_hierarchical_risk_parity_weights():
    m7 = InstitutionalPortfolioOptimiser()
    np.random.seed(42)
    rets = pd.DataFrame(np.random.randn(100, 4) * 0.02, columns=['A', 'B', 'C', 'D'])
    sigma = m7.estimate_covariance(rets)
    
    hrp_w = m7.optimise_hierarchical_risk_parity(rets, sigma)
    assert np.isclose(hrp_w.sum(), 1.0, atol=1e-4)
    assert (hrp_w >= 0.0).all()
