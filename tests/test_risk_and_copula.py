"""
Unit & Integration Tests for Copula Tail-Risk, EVT, and Crisis Stress Testing
"""

import pytest
import numpy as np
import pandas as pd

from src.risk.copula import CopulaTailRiskSimulator
from src.risk.evt import EVTTailEstimator
from src.risk.stress import StressTestingEngine


def test_copula_tail_risk_simulation():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=150, freq="B")
    rets = pd.DataFrame({
        'A': np.random.standard_t(df=3, size=150) * 0.02,
        'B': np.random.standard_t(df=3, size=150) * 0.025
    }, index=dates)

    copula = CopulaTailRiskSimulator(degrees_of_freedom=3.0)
    fit_res = copula.fit(rets)
    assert fit_res['status'] == 'fitted'

    w = pd.Series({'A': 0.5, 'B': 0.5})
    risk_eval = copula.compute_portfolio_tail_risk(w, n_simulations=2000)
    assert 'student_t_copula' in risk_eval
    assert risk_eval['student_t_copula']['cvar_99'] > 0.0


def test_evt_gpd_tail_estimation():
    np.random.seed(42)
    # Heavy tailed Pareto losses
    losses = pd.Series(np.random.pareto(a=2.5, size=200) * 0.02)
    evt = EVTTailEstimator(threshold_quantile=0.90)
    res = evt.fit_and_estimate_tail(losses, confidence_level=0.99)
    assert res['evt_var'] > 0.0
    assert res['evt_cvar'] >= res['evt_var']


def test_crisis_stress_testing():
    stress = StressTestingEngine()
    w = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
    betas = {'AAPL': 1.2, 'MSFT': 1.1}

    scenarios = stress.run_predefined_scenarios(w, betas)
    assert '2008_global_financial_crisis' in scenarios
    assert scenarios['2008_global_financial_crisis']['projected_portfolio_loss'] < 0.0

    rev_res = stress.reverse_stress_test(w, betas, max_acceptable_loss=-0.20)
    assert rev_res['portfolio_weighted_beta'] == 1.15
