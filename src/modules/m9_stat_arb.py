"""
Module 9: Statistical Arbitrage, Cointegration & Ornstein-Uhlenbeck Engine
==========================================================================
PURPOSE:
  Detect stationary mean-reverting spreads across asset pairs using the
  Engle-Granger cointegration framework and model mean-reversion half-life
  via the continuous-time Ornstein-Uhlenbeck (OU) process.

MATHEMATICAL FOUNDATION:
  1. Engle-Granger Two-Step Cointegration:
     Step 1: OLS regression P_A = α + β * P_B + ε
     Step 2: Augmented Dickey-Fuller (ADF) test on residuals ε.
             If p-value < 0.05, the pair is cointegrated of order CI(1,1).

  2. Continuous-time Ornstein-Uhlenbeck (OU) Process:
     dX_t = θ * (μ - X_t) dt + σ * dW_t
     Discrete AR(1) estimation: X_t = a + b * X_{t-1} + η
       θ = -ln(b) / Δt
       Half-Life t_{1/2} = ln(2) / θ

  3. Stat-Arb Spread Z-Score:
     Z_t = (Spread_t - μ_Spread) / σ_Spread
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import logging

from src.modules.base import ModuleResult

logger = logging.getLogger(__name__)


class StatisticalArbitrageEngine:
    """
    Identifies cointegrated pairs, estimates Ornstein-Uhlenbeck parameters,
    and generates mean-reversion alpha signals.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.adf_alpha = self.config.get('stat_arb', {}).get('adf_alpha', 0.05)
        self.min_half_life_days = 2.0
        self.max_half_life_days = 63.0

    def test_cointegration(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> Dict[str, Any]:
        """
        Executes Engle-Granger two-step cointegration test between two price series.
        """
        aligned = pd.DataFrame({'a': prices_a, 'b': prices_b}).dropna()
        if len(aligned) < 60:
            return {'cointegrated': False, 'p_value': 1.0, 'reason': 'insufficient_data'}

        y = aligned['a']
        X = sm.add_constant(aligned['b'])

        try:
            model = sm.OLS(y, X).fit()
            residuals = model.resid
            alpha = float(model.params.iloc[0])
            beta = float(model.params.iloc[1])

            # ADF test on residuals without constant (residuals mean ~ 0)
            # result_object=False silences the statsmodels FutureWarning
            adf_res = adfuller(residuals, autolag='AIC', result_object=False)
            p_val = float(adf_res[1])
            is_coint = bool(p_val < self.adf_alpha)

            # Fit Ornstein-Uhlenbeck parameters on residuals
            ou_params = self.estimate_ornstein_uhlenbeck(residuals)

            spread = residuals
            z_score = float((spread.iloc[-1] - spread.mean()) / spread.std()) if spread.std() > 0 else 0.0

            return {
                'cointegrated': is_coint,
                'p_value': p_val,
                'hedge_ratio_beta': beta,
                'intercept_alpha': alpha,
                'current_spread_zscore': z_score,
                'half_life_days': ou_params['half_life_days'],
                'mean_reversion_speed_theta': ou_params['theta']
            }
        except Exception as e:
            logger.debug(f"Cointegration test failed: {e}")
            return {'cointegrated': False, 'p_value': 1.0, 'error': str(e)}

    @staticmethod
    def estimate_ornstein_uhlenbeck(series: pd.Series) -> Dict[str, float]:
        """
        Estimates continuous-time OU parameters θ, μ, σ and half-life t_{1/2}.
        """
        x = series.values
        x_lag = x[:-1]
        x_curr = x[1:]

        X = sm.add_constant(x_lag)
        try:
            reg = sm.OLS(x_curr, X).fit()
            a = float(reg.params[0])
            b = float(reg.params[1])

            if 0 < b < 1.0:
                theta = float(-np.log(b) * 252.0) # annualized mean reversion rate
                half_life_days = float(np.log(2.0) / (-np.log(b)))
                mu = float(a / (1.0 - b))
                sigma = float(np.std(reg.resid) * np.sqrt(252.0))
            else:
                theta = 0.0
                half_life_days = 999.0
                mu = float(np.mean(x))
                sigma = float(np.std(x) * np.sqrt(252.0))

            return {
                'theta': theta,
                'mu': mu,
                'sigma': sigma,
                'half_life_days': half_life_days
            }
        except Exception:
            return {'theta': 0.0, 'mu': 0.0, 'sigma': 0.0, 'half_life_days': 999.0}

    def scan_universe_pairs(
        self,
        price_dict: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Scans all pairs in universe to find statistically significant cointegrated relationships.
        """
        tickers = sorted(list(price_dict.keys()))
        results = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t_a, t_b = tickers[i], tickers[j]
                df_a = price_dict[t_a]
                df_b = price_dict[t_b]
                if df_a.empty or df_b.empty:
                    continue

                col_a = 'Adj Close' if 'Adj Close' in df_a.columns else 'Close'
                col_b = 'Adj Close' if 'Adj Close' in df_b.columns else 'Close'

                coint_res = self.test_cointegration(df_a[col_a], df_b[col_b])
                if coint_res.get('cointegrated', False):
                    results.append({
                        'ticker_a': t_a,
                        'ticker_b': t_b,
                        **coint_res
                    })

        return sorted(results, key=lambda x: x['p_value'])
