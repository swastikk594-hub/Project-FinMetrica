"""
Comprehensive Institutional Unit Tests for Quantitative Alpha & Risk Modules
"""

import pytest
import numpy as np
import pandas as pd

from src.modules.m1_fundamentals import FundamentalsModule
from src.modules.m2_valuation import ValuationModule
from src.modules.m3_timeseries import TimeSeriesModule
from src.modules.m4_factors import FactorModel
from src.modules.m5_risk import RiskModule
from src.modules.m6_macro import MacroModule
from src.modules.m7_optimisation import InstitutionalPortfolioOptimiser
from src.modules.m8_nlp import FinancialNLPEngine
from src.modules.m9_stat_arb import StatisticalArbitrageEngine


def test_m1_roic_and_accruals():
    m1 = FundamentalsModule()
    dates = [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    
    fin_data = {
        'income_stmt': pd.DataFrame({
            dates[1]: {'EBIT': 100.0, 'Net Income': 80.0, 'Total Revenue': 1000.0, 'Tax Provision': 20.0, 'Pretax Income': 100.0},
            dates[0]: {'EBIT': 90.0, 'Net Income': 70.0, 'Total Revenue': 900.0, 'Tax Provision': 18.0, 'Pretax Income': 90.0}
        }),
        'balance_sheet': pd.DataFrame({
            dates[1]: {'Total Assets': 500.0, 'Total Debt': 100.0, 'Stockholders Equity': 300.0, 'Cash And Cash Equivalents': 50.0},
            dates[0]: {'Total Assets': 450.0, 'Total Debt': 100.0, 'Stockholders Equity': 280.0, 'Cash And Cash Equivalents': 40.0}
        }),
        'cash_flow': pd.DataFrame({
            dates[1]: {'Operating Cash Flow': 120.0, 'Free Cash Flow': 90.0},
            dates[0]: {'Operating Cash Flow': 100.0, 'Free Cash Flow': 75.0}
        })
    }

    roic_res = m1.compute_roic(fin_data)
    assert not np.isnan(roic_res['roic'])
    assert roic_res['roic'] > 0.0

    accruals = m1.compute_accruals_ratio(fin_data)
    # Accruals = (80 - 120) / 475 = -40 / 475 = -0.0842 (negative accruals = high cash flow quality)
    assert accruals < 0.0


def test_m2_dcf_wacc_constraint():
    m2 = ValuationModule()
    dates = [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    fin_data = {
        'cash_flow': pd.DataFrame({
            dates[1]: {'Free Cash Flow': 100.0},
            dates[0]: {'Free Cash Flow': 90.0}
        }),
        'balance_sheet': pd.DataFrame({
            dates[1]: {'Cash And Cash Equivalents': 50.0, 'Total Debt': 20.0},
            dates[0]: {'Cash And Cash Equivalents': 40.0, 'Total Debt': 20.0}
        })
    }

    dcf_res = m2.compute_probabilistic_dcf(fin_data, current_price=150.0, wacc=0.08, shares_outstanding=10.0)
    assert dcf_res['status'] == 'success'
    assert not np.isnan(dcf_res['intrinsic_value'])
    assert dcf_res['intrinsic_value'] > 0.0


def test_m7_black_litterman_and_hrp():
    m7 = InstitutionalPortfolioOptimiser()
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    
    # Deterministic diagonal covariance to strictly verify view transmission
    sigma = pd.DataFrame(np.diag([0.04, 0.04, 0.04]), index=['AAPL', 'MSFT', 'GOOGL'], columns=['AAPL', 'MSFT', 'GOOGL'])
    rets = pd.DataFrame(np.random.normal(0.001, 0.02, (100, 3)), index=dates, columns=['AAPL', 'MSFT', 'GOOGL'])

    views = pd.Series({'AAPL': 0.18, 'MSFT': 0.10, 'GOOGL': 0.05})
    mkt_w = pd.Series({'AAPL': 0.33, 'MSFT': 0.33, 'GOOGL': 0.34})
    bl_mu = m7.compute_black_litterman_returns(sigma, mkt_w, views, view_confidence=0.80)

    assert len(bl_mu) == 3
    assert bl_mu['AAPL'] > bl_mu['GOOGL']

    hrp_w = m7.optimise_hierarchical_risk_parity(rets, sigma)
    assert np.isclose(hrp_w.sum(), 1.0)
    assert (hrp_w >= 0.0).all()


def test_m8_nlp_sentiment_and_events():
    nlp = FinancialNLPEngine()
    text = "Company reported record quarterly profit and exceeded revenue estimates. Board approved share repurchase."
    scores = nlp.score_text(text)
    assert scores['sentiment_score'] > 0.0

    events = nlp.detect_events(text)
    assert 'capital_return' in events or 'earnings_beat' in events


def test_m9_stat_arb_ornstein_uhlenbeck():
    np.random.seed(42)
    # Synthetic mean reverting series
    n = 200
    x = np.zeros(n)
    theta_true = 0.10
    mu_true = 0.0
    sigma_true = 0.05
    for t in range(1, n):
        x[t] = x[t-1] + theta_true * (mu_true - x[t-1]) + sigma_true * np.random.normal()

    series = pd.Series(x)
    ou_res = StatisticalArbitrageEngine.estimate_ornstein_uhlenbeck(series)
    assert ou_res['half_life_days'] > 0.0
    assert ou_res['half_life_days'] < 50.0
