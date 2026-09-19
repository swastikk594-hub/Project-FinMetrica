"""
Module 7: Institutional Portfolio Optimisation & Decision Engine
=================================================================
PURPOSE:
  Synthesize alpha forecasts, Ledoit-Wolf covariance shrinkage, and
  risk parameters into optimal portfolio allocations using Black-Litterman,
  Hierarchical Risk Parity (HRP), Maximum Sharpe, and CVaR objectives.

MATHEMATICAL FRAMEWORK:
  1. Black-Litterman Model:
     Equilibrium Implied Returns: Π = δ * Σ * w_mkt
     Posterior Combined Expected Returns:
     μ_BL = [ (τ Σ)^{-1} + P^T Ω^{-1} P ]^{-1} * [ (τ Σ)^{-1} Π + P^T Ω^{-1} Q ]
     where Q represents the ML alpha view vector and Ω is the view uncertainty matrix.

  2. Hierarchical Risk Parity (HRP - Marcos Lopez de Prado):
     Quasi-diagonalizes the covariance matrix using tree clustering,
     allocating capital hierarchically to eliminate matrix inversion instability.

  3. Convex Portfolio Optimizers:
     - Maximum Sharpe Ratio: max_w (w^T μ - r_f) / sqrt(w^T Σ w)
     - Minimum Variance: min_w w^T Σ w
     - Risk Parity: w_i * (Σ w)_i / σ_p = σ_p / N for all i
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.modules.base import ModuleResult

logger = logging.getLogger(__name__)


class InstitutionalPortfolioOptimiser:
    """
    Constructs mathematically robust allocations via Black-Litterman, HRP,
    Max Sharpe, and Risk Parity.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.risk_aversion_delta = self.config.get('module7', {}).get('risk_aversion', 2.5)
        self.tau = self.config.get('module7', {}).get('tau', 0.05)

    def estimate_covariance(
        self,
        returns_df: pd.DataFrame,
        method: str = 'ledoit_wolf'
    ) -> pd.DataFrame:
        """
        Estimates robust covariance matrix with Ledoit-Wolf shrinkage.
        """
        clean_df = returns_df.dropna()
        if clean_df.empty:
            return pd.DataFrame()

        if method == 'ledoit_wolf':
            lw = LedoitWolf()
            lw.fit(clean_df.values)
            cov_annual = lw.covariance_ * 252.0
            cov_df = pd.DataFrame(cov_annual, index=clean_df.columns, columns=clean_df.columns)
        else:
            cov_df = clean_df.cov() * 252.0

        # Enforce positive semi-definiteness
        evals, evecs = np.linalg.eigh(cov_df.values)
        evals = np.maximum(evals, 1e-6)
        clean_cov = evecs @ np.diag(evals) @ evecs.T
        return pd.DataFrame(clean_cov, index=clean_df.columns, columns=clean_df.columns)

    def compute_black_litterman_returns(
        self,
        sigma: pd.DataFrame,
        market_weights: pd.Series,
        alpha_views: pd.Series,
        view_confidence: float = 0.50,
        risk_free_rate: float = 0.04
    ) -> pd.Series:
        """
        Computes Black-Litterman posterior combined return vector.
        """
        tickers = list(sigma.columns)
        n = len(tickers)
        w_mkt = market_weights.reindex(tickers).fillna(1.0 / n).values
        w_mkt = w_mkt / w_mkt.sum()

        sigma_mat = sigma.values
        # 1. Reverse-engineer equilibrium implied returns: Π = δ * Σ * w_mkt
        delta = self.risk_aversion_delta
        pi = delta * (sigma_mat @ w_mkt)

        # 2. Setup views matrix P and views vector Q
        # Each view is absolute expected return for asset i
        P = np.eye(n)
        q = alpha_views.reindex(tickers).fillna(pd.Series(pi, index=tickers)).values

        # 3. View uncertainty matrix Ω = diag(diag(P (τ Σ) P^T)) / confidence
        tau_sigma = self.tau * sigma_mat
        omega = np.diag(np.diag(P @ tau_sigma @ P.T)) / max(view_confidence, 0.01)

        # 4. Master Black-Litterman formula
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega)

        post_cov = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
        post_mu = post_cov @ (inv_tau_sigma @ pi + P.T @ inv_omega @ q)

        return pd.Series(post_mu, index=tickers)

    def optimise_max_sharpe(
        self,
        mu: pd.Series,
        sigma: pd.DataFrame,
        risk_free_rate: float = 0.04,
        max_weight: float = 0.40
    ) -> Dict[str, Any]:
        """Maximizes Sharpe Ratio under long-only concentration constraints."""
        tickers = list(mu.index)
        n = len(tickers)
        
        mu_vec = mu.values
        sigma_mat = sigma.values

        if n == 1:
            exp_ret = float(mu_vec[0])
            exp_vol = float(np.sqrt(sigma_mat[0, 0]))
            sr = float((exp_ret - risk_free_rate) / exp_vol) if exp_vol > 0 else 0.0
            return {
                'weights': {tickers[0]: 1.0},
                'expected_return': exp_ret,
                'expected_volatility': exp_vol,
                'sharpe_ratio': sr
            }

        bounds = tuple((0.0, max_weight) for _ in range(n))
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        init_guess = np.ones(n) / n

        def neg_sharpe(w):
            r = float(np.dot(w, mu_vec))
            v = float(np.sqrt(np.dot(w.T, np.dot(sigma_mat, w))))
            return -(r - risk_free_rate) / v if v > 0 else 0.0

        res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        w_opt = np.maximum(res.x, 0.0)
        w_opt = w_opt / w_opt.sum()

        exp_ret = float(np.dot(w_opt, mu_vec))
        exp_vol = float(np.sqrt(np.dot(w_opt.T, np.dot(sigma_mat, w_opt))))
        sr = float((exp_ret - risk_free_rate) / exp_vol) if exp_vol > 0 else 0.0

        return {
            'weights': {tickers[i]: float(w_opt[i]) for i in range(n)},
            'expected_return': exp_ret,
            'expected_volatility': exp_vol,
            'sharpe_ratio': sr
        }

    def optimise_hierarchical_risk_parity(
        self,
        returns_df: pd.DataFrame,
        sigma: pd.DataFrame
    ) -> pd.Series:
        """
        Computes Hierarchical Risk Parity (HRP) allocations.
        """
        clean_df = returns_df.dropna()
        tickers = list(sigma.columns)
        n = len(tickers)
        if clean_df.empty or n < 2:
            return pd.Series(1.0 / max(n, 1), index=tickers)

        # 1. Correlation distance matrix
        corr = clean_df.corr().fillna(0.0)
        dist = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0.0, 1.0))

        # 2. Hierarchical clustering
        dist_condensed = squareform(dist, checks=False)
        link = linkage(dist_condensed, method='single')
        sort_ix = leaves_list(link)
        sorted_tickers = [tickers[i] for i in sort_ix]

        # 3. Recursive bisection
        def get_cluster_var(sub_cov):
            w = 1.0 / np.diag(sub_cov)
            w = w / w.sum()
            return np.dot(w.T, np.dot(sub_cov, w))

        def bisect(items):
            if len(items) == 1:
                return {items[0]: 1.0}
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]

            left_cov = sigma.loc[left, left].values
            right_cov = sigma.loc[right, right].values

            left_var = get_cluster_var(left_cov)
            right_var = get_cluster_var(right_cov)

            alpha = 1.0 - (left_var / (left_var + right_var)) if (left_var + right_var) > 0 else 0.5
            
            w_left = bisect(left)
            w_right = bisect(right)

            res = {}
            for k, v in w_left.items(): res[k] = v * alpha
            for k, v in w_right.items(): res[k] = v * (1.0 - alpha)
            return res

        w_hrp_dict = bisect(sorted_tickers)
        return pd.Series(w_hrp_dict).reindex(tickers).fillna(0.0)

    def run(
        self,
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        module_scores: Dict[str, Any],
        risk_free_rate: float = 0.04
    ) -> Dict[str, Any]:
        """Executes complete institutional portfolio construction pipeline."""
        logger.info(f"Running Portfolio Optimisation Module for {len(tickers)} assets")
        returns_dict = {}
        for t in tickers:
            if t in price_data and not price_data[t].empty:
                col = 'Adj Close' if 'Adj Close' in price_data[t].columns else 'Close'
                returns_dict[t] = price_data[t][col].pct_change().dropna()

        returns_df = pd.DataFrame(returns_dict).dropna()
        valid_tickers = list(returns_df.columns)
        n = len(valid_tickers)
        if n == 0:
            return {'max_sharpe': {'weights': {t: 1.0 / len(tickers) for t in tickers}}}

        sigma = self.estimate_covariance(returns_df, method='ledoit_wolf')

        # Extract ML alpha forecasts from composite scores
        alpha_views = pd.Series(0.08, index=valid_tickers)
        for t in valid_tickers:
            score = module_scores.get(t, 50.0)
            score_val = getattr(score, 'adjusted_score', getattr(score, 'normalised_score', float(score) if isinstance(score, (int, float)) else 50.0))
            # Tilt expected return: 50 -> 8%, 100 -> 18%, 0 -> -2%
            alpha_views[t] = 0.08 + (score_val - 50.0) / 50.0 * 0.10

        market_weights = pd.Series(1.0 / n, index=valid_tickers)
        bl_mu = self.compute_black_litterman_returns(sigma, market_weights, alpha_views, risk_free_rate=risk_free_rate)

        opt_sharpe = self.optimise_max_sharpe(bl_mu, sigma, risk_free_rate=risk_free_rate)
        hrp_weights = self.optimise_hierarchical_risk_parity(returns_df, sigma)

        return {
            'expected_returns_bl': bl_mu.to_dict(),
            'covariance': sigma.to_dict(),
            'max_sharpe': opt_sharpe,
            'hierarchical_risk_parity_weights': hrp_weights.to_dict(),
            'status': 'optimised'
        }


# Backward compatibility alias
PortfolioOptimiser = InstitutionalPortfolioOptimiser
