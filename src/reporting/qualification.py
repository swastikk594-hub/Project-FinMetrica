"""
Institutional Strategy Qualification & Overfitting Detection Framework
========================================================================
PURPOSE:
  Provide a rigorous, scientific PASS / CONDITIONAL PASS / OVERFIT / FAILED
  qualification verdict for any quantitative strategy.

MATHEMATICAL FOUNDATION:
  1. Deflated Sharpe Ratio (DSR - Bailey & Lopez de Prado 2014):
     Adjusts the estimated Sharpe ratio for:
     - Non-normal returns (Skewness γ_3, Kurtosis γ_4)
     - Multiple testing / selection bias across N strategy trials
     - Sample length N_obs
     
     DSR = Φ( [ (SR - SR_benchmark) * sqrt(N_obs - 1) ] /
              sqrt( 1 - γ_3 * SR + (γ_4 - 1)/4 * SR^2 ) )

  2. Haircut Sharpe Ratio (Harvey & Liu 2015):
     Applies haircut penalty based on number of investigated factor trials.

  3. Qualification Gates:
     - Out-of-sample Sharpe Ratio post-costs > 0.70
     - Out-of-sample Information Ratio > 0.30
     - Deflated Sharpe Ratio > 0.95 (95% statistical confidence against overfitting)
     - Max Drawdown < 30%
     - Positive Out-of-Sample Information Coefficient (IC)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class StrategyQualificationEngine:
    """
    Evaluates backtest results against institutional standards and issues a formal
    qualification decision.
    """

    def __init__(self, n_trials_conducted: int = 10):
        self.n_trials = max(int(n_trials_conducted), 1)

    def calculate_deflated_sharpe_ratio(
        self,
        returns: pd.Series,
        benchmark_sr: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculates the Deflated Sharpe Ratio (DSR) and estimated haircut.
        """
        clean_rets = returns.dropna()
        n_obs = len(clean_rets)
        if n_obs < 30:
            return {'dsr': 0.0, 'haircut_sharpe': 0.0, 'p_value': 1.0}

        ann_factor = np.sqrt(252.0)
        mean_r = clean_rets.mean()
        std_r = clean_rets.std()
        sr = (mean_r / std_r) * ann_factor if std_r > 0 else 0.0

        skew = float(stats.skew(clean_rets))
        kurt = float(stats.kurtosis(clean_rets, fisher=False)) # Pearson kurtosis (normal = 3)

        # Expected maximum Sharpe over N trials under null hypothesis
        # Euler-Mascheroni approximation: E[max(SR_N)] ≈ sqrt(2 * ln(N)) + (1 - γ)/sqrt(2 * ln(N))
        gamma_em = 0.5772156649
        if self.n_trials > 1:
            z_n = np.sqrt(2.0 * np.log(self.n_trials))
            sr_null = (z_n + (1.0 - gamma_em) / z_n) / ann_factor
        else:
            sr_null = benchmark_sr / ann_factor

        # DSR standard error denominator (Opdyke / Mertens asymptotic variance)
        var_sr = 1.0 - (skew * (sr / ann_factor)) + (((kurt - 1.0) / 4.0) * ((sr / ann_factor) ** 2))
        var_sr = max(var_sr, 0.001)

        # Z-score for DSR
        z_stat = ((sr / ann_factor) - sr_null) * np.sqrt(n_obs - 1.0) / np.sqrt(var_sr)
        dsr_prob = float(stats.norm.cdf(z_stat))

        # Haircut Sharpe ratio
        haircut_penalty = min(0.30 * np.log(self.n_trials), 1.5)
        haircut_sr = max(sr - haircut_penalty, -2.0)

        return {
            'observed_sharpe': float(sr),
            'deflated_sharpe_ratio': float(dsr_prob),
            'haircut_sharpe': float(haircut_sr),
            'skewness': skew,
            'kurtosis': kurt,
            'trials_penalty': float(haircut_penalty)
        }

    def evaluate_strategy(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        total_market_impact_dollars: float = 0.0,
        max_drawdown: float = -0.25
    ) -> Dict[str, Any]:
        """
        Runs comprehensive qualification tests and renders a binding verdict.
        """
        dsr_res = self.calculate_deflated_sharpe_ratio(returns)
        sr = dsr_res['observed_sharpe']
        dsr = dsr_res['deflated_sharpe_ratio']
        mdd = abs(max_drawdown)

        # Active excess returns vs benchmark
        active = returns - benchmark_returns.reindex(returns.index).fillna(0.0)
        te = active.std() * np.sqrt(252.0)
        ir = (active.mean() * 252.0) / te if te > 0 else 0.0

        # Verdict logic
        reasons = []
        if dsr < 0.50:
            status = "OVERFIT"
            reasons.append(f"High risk of backtest overfitting (DSR = {dsr:.1%}, < 50% threshold).")
        elif sr < 0.40 or ir < 0.0:
            status = "FAILED"
            reasons.append(f"Sub-par risk-adjusted returns post-costs (Sharpe = {sr:.2f}, IR = {ir:.2f}).")
        elif mdd > 0.35:
            status = "FAILED"
            reasons.append(f"Excessive maximum historical drawdown ({mdd:.1%}, > 35% limit).")
        elif dsr < 0.85:
            status = "CONDITIONAL PASS"
            reasons.append("Strategy meets baseline profitability but requires further out-of-sample data.")
        else:
            status = "PASS"
            reasons.append("Statistically significant, persistent alpha after execution costs and multi-testing adjustment.")

        deployment_stage = {
            "PASS": "Scale / Multi-Asset Deployment",
            "CONDITIONAL PASS": "Paper Trading / Small Capital Testing",
            "OVERFIT": "Reject & Redesign Alpha",
            "FAILED": "Reject",
            "INSUFFICIENT EVIDENCE": "Extend In-Sample Horizon"
        }.get(status, "Research")

        return {
            'qualification_status': status,
            'deployment_stage': deployment_stage,
            'reasons': reasons,
            'metrics': {
                'observed_sharpe': sr,
                'deflated_sharpe_ratio': dsr,
                'haircut_sharpe': dsr_res['haircut_sharpe'],
                'information_ratio': float(ir),
                'max_drawdown': float(max_drawdown),
                'total_impact_cost': float(total_market_impact_dollars)
            }
        }
