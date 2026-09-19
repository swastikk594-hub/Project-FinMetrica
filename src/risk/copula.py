"""
Student-t & Gaussian Copula Non-Linear Tail-Risk Simulation Engine
==================================================================
PURPOSE:
  Model joint non-linear tail dependence across portfolio assets.
  In financial market panics, standard linear correlation fails because
  diversification breaks down as assets crash simultaneously (tail dependence).

MATHEMATICAL FOUNDATION:
  1. Sklar's Theorem:
     Any multivariate joint distribution F(x_1, ..., x_d) can be decomposed into:
     F(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d))
     where C is the Copula and F_i are the marginal CDFs.

  2. Probability Integral Transform (PIT):
     u_i = F_i(r_i) = (Rank(r_i) - 0.5) / N  in (0, 1)

  3. Student-t Copula:
     C_nu,R(u_1, ..., u_d) = t_nu,R(t_nu^{-1}(u_1), ..., t_nu^{-1}(u_d))
     Generates joint tail dependence:
     λ_L = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))) > 0

COMPARISON BASELINE:
  Gaussian Copula exhibits asymptotic tail independence (λ_L = 0),
  which catastrophically underestimates simultaneous joint crashes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CopulaTailRiskSimulator:
    """
    Fits empirical marginals and Student-t / Gaussian copulas to simulate
    joint portfolio loss distributions and extreme tail risk.
    """

    def __init__(self, degrees_of_freedom: float = 4.0, random_state: int = 42):
        self.nu = max(float(degrees_of_freedom), 2.1)
        self.random_state = random_state
        self.corr_matrix: Optional[np.ndarray] = None
        self.marginals: Dict[str, np.ndarray] = {}

    def fit(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits marginal empirical distributions and copula correlation matrix.
        """
        clean_df = returns_df.dropna()
        if len(clean_df) < 30:
            return {'status': 'insufficient_data'}

        n_assets = clean_df.shape[1]
        tickers = list(clean_df.columns)
        uniform_matrix = np.zeros(clean_df.shape)

        # 1. Transform each asset to uniform via empirical CDF (ranks)
        for i, col in enumerate(tickers):
            arr = clean_df[col].values
            self.marginals[col] = np.sort(arr)
            # PIT transformation
            ranks = stats.rankdata(arr)
            uniform_matrix[:, i] = (ranks - 0.5) / len(arr)

        # 2. Map uniforms to Student-t quantiles
        t_quantiles = stats.t.ppf(uniform_matrix, df=self.nu)
        # Handle numerical boundaries
        t_quantiles = np.nan_to_num(t_quantiles, nan=0.0, posinf=4.0, neginf=-4.0)

        # 3. Estimate copula correlation matrix
        if n_assets == 1:
            self.corr_matrix = np.array([[1.0]])
        else:
            cov = np.corrcoef(t_quantiles, rowvar=False)
            # Ensure positive semi-definite
            evals, evecs = np.linalg.eigh(cov)
            evals = np.maximum(evals, 1e-6)
            self.corr_matrix = evecs @ np.diag(evals) @ evecs.T

        # Normalize diagonal to 1.0
        d = np.sqrt(np.diag(self.corr_matrix))
        self.corr_matrix = self.corr_matrix / np.outer(d, d)

        return {
            'n_assets': n_assets,
            'degrees_of_freedom': self.nu,
            'status': 'fitted'
        }

    def simulate_joint_returns(
        self,
        n_simulations: int = 10_000,
        copula_type: str = "student_t"
    ) -> np.ndarray:
        """
        Generates synthetic joint return scenarios across all assets.
        """
        if self.corr_matrix is None:
            raise ValueError("CopulaTailRiskSimulator must be fitted before simulation.")

        np.random.seed(self.random_state)
        n_assets = len(self.marginals)
        tickers = list(self.marginals.keys())

        # Cholesky factor
        L = np.linalg.cholesky(self.corr_matrix)

        if copula_type == "student_t":
            # Generate multivariate t variates
            z = np.random.normal(size=(n_simulations, n_assets))
            w = np.random.chisquare(df=self.nu, size=(n_simulations, 1)) / self.nu
            t_samples = (z @ L.T) / np.sqrt(w)
            u_samples = stats.t.cdf(t_samples, df=self.nu)
        else: # Gaussian copula baseline
            z = np.random.normal(size=(n_simulations, n_assets))
            g_samples = z @ L.T
            u_samples = stats.norm.cdf(g_samples)

        # Invert uniforms back to empirical return distribution
        simulated_returns = np.zeros((n_simulations, n_assets))
        for i, col in enumerate(tickers):
            sorted_arr = self.marginals[col]
            indices = np.clip(np.floor(u_samples[:, i] * len(sorted_arr)).astype(int), 0, len(sorted_arr) - 1)
            simulated_returns[:, i] = sorted_arr[indices]

        return simulated_returns

    def compute_portfolio_tail_risk(
        self,
        weights: pd.Series,
        n_simulations: int = 10_000
    ) -> Dict[str, Any]:
        """
        Compares Student-t Copula tail risk against Gaussian Copula baseline.
        """
        tickers = list(self.marginals.keys())
        w_vec = weights.reindex(tickers).fillna(0.0).values
        if w_vec.sum() > 0:
            w_vec = w_vec / w_vec.sum()

        sim_t = self.simulate_joint_returns(n_simulations, copula_type="student_t")
        sim_g = self.simulate_joint_returns(n_simulations, copula_type="gaussian")

        port_returns_t = sim_t @ w_vec
        port_returns_g = sim_g @ w_vec

        # 99% VaR and CVaR
        var_99_t = float(-np.percentile(port_returns_t, 1))
        cvar_99_t = float(-np.mean(port_returns_t[port_returns_t <= -var_99_t]))

        var_99_g = float(-np.percentile(port_returns_g, 1))
        cvar_99_g = float(-np.mean(port_returns_g[port_returns_g <= -var_99_g]))

        # Joint crash probability (fraction of paths where > 50% of assets experience > 3-sigma decline)
        joint_crash_t = float(np.mean(np.sum(sim_t < -0.05, axis=1) >= (len(tickers) / 2)))
        joint_crash_g = float(np.mean(np.sum(sim_g < -0.05, axis=1) >= (len(tickers) / 2)))

        return {
            'student_t_copula': {
                'var_99': var_99_t,
                'cvar_99': cvar_99_t,
                'joint_crash_probability': joint_crash_t
            },
            'gaussian_copula_baseline': {
                'var_99': var_99_g,
                'cvar_99': cvar_99_g,
                'joint_crash_probability': joint_crash_g
            },
            'tail_risk_delta_cvar': float(cvar_99_t - cvar_99_g)
        }
