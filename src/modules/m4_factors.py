import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any, Optional

from src.modules.base import ModuleResult

logger = logging.getLogger(__name__)

"""
Module 4: Statistical / Factor Model
=====================================
PURPOSE:
  Decompose the asset's returns into systematic factor exposures
  using the Fama-French multi-factor framework. This reveals:
  - How much return is explained by known risk factors
  - What the asset's 'alpha' (unexplained excess return) is
  - The asset's specific style exposures (size, value, momentum)

MATHEMATICAL FRAMEWORK:
  Carhart 4-Factor Model (extends Fama-French 3-Factor):
  
  Rᵢ - Rf = αᵢ + β₁·(Rm-Rf) + β₂·SMB + β₃·HML + β₄·WML + εᵢ
  
  where:
    Rᵢ    = asset return
    Rf    = risk-free rate
    Rm-Rf = market excess return
    SMB   = Small Minus Big (size factor)
    HML   = High Minus Low (value factor)
    WML   = Winners Minus Losers (momentum factor)
    αᵢ    = Jensen's alpha (risk-adjusted excess return)
    βⱼ    = factor loading (sensitivity to factor j)
    εᵢ    = idiosyncratic (unexplained) return
    
  Estimation via OLS (rolling window of 36 months):
    β = (XᵀX)⁻¹Xᵀy
    
  R² = 1 - SS_res/SS_tot  (explanatory power)
  
METHOD COMPARISON — OLS vs Robust Regression:
  OLS: Minimises sum of squared residuals; optimal under normality
       Sensitive to outliers in return data
  Huber Robust: Minimises combination of L1 and L2 loss
                Less sensitive to outliers; more robust for fat-tailed distributions
  SELECTED: OLS primary (interpretable, standard), with Robust as comparison
            If estimates differ substantially, flag as robustness concern

METHOD COMPARISON — Rolling 36M vs Rolling 60M window:
  36M: Faster to react to changing factor exposures; higher estimation variance
  60M: Smoother estimates; may miss structural changes
  SELECTED: 36M primary (documented by Fama-French in their methodology)
            60M shown for comparison

ASSUMPTIONS:
  - Factor returns are the correct systematic factors for this asset
  - Factor loadings are stable over the estimation window
  - Residuals are approximately i.i.d.
  
LIMITATIONS:
  - OLS beta estimates have substantial uncertainty for short series
  - Alpha estimates are notoriously noisy and not reliable forecasts
  - Factors may not span the full systematic risk of all assets
"""

class FactorModel:
    """
    Factor Model Module to decompose asset returns into systematic
    factor exposures using multi-factor framework.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def load_ff_factors(self, ff_data: pd.DataFrame, monthly: bool = True) -> pd.DataFrame:
        """
        Align Fama-French factor data format and verify columns.

        Parameters
        ----------
        ff_data : pd.DataFrame
            Fama-French factors data.
        monthly : bool, optional
            Whether data is monthly, by default True.

        Returns
        -------
        pd.DataFrame
            Cleaned factor data.
        """
        # Ensure we have common factors
        expected_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
        
        # Check if WML (momentum) is present, add to expected if so
        if 'WML' in ff_data.columns:
            expected_cols.append('WML')
        elif 'Mom' in ff_data.columns:
            ff_data = ff_data.rename(columns={'Mom': 'WML'})
            expected_cols.append('WML')
            
        missing_cols = [col for col in expected_cols if col not in ff_data.columns]
        if missing_cols:
            raise ValueError(f"Fama-French data missing columns: {missing_cols}")
            
        return ff_data[expected_cols].copy()

    def estimate_factor_loadings(self, excess_returns: pd.Series, factor_data: pd.DataFrame, method: str = 'ols') -> Dict[str, Any]:
        """
        Estimate factor loadings using OLS or Robust regression.

        Parameters
        ----------
        excess_returns : pd.Series
            Asset excess returns (R - Rf).
        factor_data : pd.DataFrame
            Factor data (Mkt-RF, SMB, HML, etc.).
        method : str, optional
            Regression method ('ols' or 'robust'), by default 'ols'.

        Returns
        -------
        Dict[str, Any]
            Regression results including alpha, betas, r_squared, etc.
        """
        # Align data
        df = pd.concat([excess_returns.rename('ExcessReturn'), factor_data], axis=1).dropna()
        if len(df) < 12:  # Minimum 12 periods
            return {'error': 'Insufficient data for factor estimation'}
            
        y = df['ExcessReturn']
        X = df.drop(columns=['ExcessReturn', 'RF'], errors='ignore')
        X = sm.add_constant(X)
        
        if method.lower() == 'robust':
            model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            res = model.fit()
            # R-squared proxy for robust regression
            # Pseudo R^2 = (var(y) - var(res)) / var(y)
            r_squared = 1 - (np.var(res.resid) / np.var(y))
        else:
            model = sm.OLS(y, X)
            res = model.fit()
            r_squared = res.rsquared
            
        alpha = float(res.params['const'])
        betas = {col: float(res.params[col]) for col in X.columns if col != 'const'}
        t_stats = {col: float(res.tvalues[col]) for col in X.columns}
        p_values = {col: float(res.pvalues[col]) for col in X.columns}
        
        return {
            'alpha': alpha,
            'betas': betas,
            'r_squared': float(r_squared),
            't_stats': t_stats,
            'p_values': p_values,
            'residuals': res.resid,
            'method': method
        }

    def rolling_factor_loadings(self, excess_returns: pd.Series, factor_data: pd.DataFrame, window: int = 36) -> pd.DataFrame:
        """
        Estimate factor loadings on a rolling window.

        Parameters
        ----------
        excess_returns : pd.Series
            Asset excess returns.
        factor_data : pd.DataFrame
            Factor data.
        window : int, optional
            Rolling window size, by default 36.

        Returns
        -------
        pd.DataFrame
            Time-varying alphas and betas.
        """
        df = pd.concat([excess_returns.rename('ExcessReturn'), factor_data], axis=1).dropna()
        if len(df) < window:
            return pd.DataFrame()
            
        y = df['ExcessReturn']
        X = df.drop(columns=['ExcessReturn', 'RF'], errors='ignore')
        X = sm.add_constant(X)
        
        results = []
        dates = []
        
        for i in range(window, len(df) + 1):
            y_window = y.iloc[i-window:i]
            X_window = X.iloc[i-window:i]
            
            model = sm.OLS(y_window, X_window)
            res = model.fit()
            
            params = res.params.to_dict()
            results.append(params)
            dates.append(df.index[i-1])
            
        res_df = pd.DataFrame(results, index=dates)
        res_df.rename(columns={'const': 'alpha'}, inplace=True)
        return res_df

    def compute_information_ratio(self, residuals: pd.Series) -> Dict[str, Any]:
        """
        Compute Information Ratio (IR).

        Parameters
        ----------
        residuals : pd.Series
            Residuals from factor regression.

        Returns
        -------
        Dict[str, Any]
            Information ratio and related metrics.
        """
        if len(residuals) < 12:
            return {'error': 'Insufficient residuals'}
            
        alpha = residuals.mean()
        idiosyncratic_vol = residuals.std()
        
        if idiosyncratic_vol == 0:
            return {'error': 'Zero idiosyncratic volatility'}
            
        # Assuming monthly data for annualisation
        annual_alpha = alpha * 12
        annual_idiosyncratic_vol = idiosyncratic_vol * np.sqrt(12)
        
        ir = annual_alpha / annual_idiosyncratic_vol
        
        # Fix HIGH-1b: check thresholds in descending order so each band is reachable
        if ir > 1.0:
            interp = 'Excellent'
        elif ir > 0.5:
            interp = 'Good'
        elif ir > 0.0:
            interp = 'Average'
        else:
            interp = 'Negative'
            
        return {
            'ir': float(ir),
            'annualised_alpha': float(annual_alpha),
            'idiosyncratic_vol': float(annual_idiosyncratic_vol),
            'interpretation': interp
        }

    def variance_decomposition(self, returns: pd.Series, factor_data: pd.DataFrame, betas: Dict[str, float]) -> Dict[str, float]:
        """
        Decompose total variance into systematic and idiosyncratic components.

        Parameters
        ----------
        returns : pd.Series
            Asset returns.
        factor_data : pd.DataFrame
            Factor data matching the estimation period.
        betas : Dict[str, float]
            Estimated factor betas.

        Returns
        -------
        Dict[str, float]
            Variance decomposition metrics.
        """
        total_var = returns.var()
        
        # Calculate systematic variance: sum of β_i * β_j * Cov(f_i, f_j)
        valid_betas = {k: v for k, v in betas.items() if k in factor_data.columns}
        factors = factor_data[list(valid_betas.keys())]
        cov_matrix = factors.cov()
        
        beta_vector = pd.Series(valid_betas)
        systematic_var = beta_vector.T @ cov_matrix @ beta_vector
        
        idiosyncratic_var = total_var - systematic_var
        
        r_squared = systematic_var / total_var if total_var > 0 else 0
        
        return {
            'total_var': float(total_var),
            'systematic_var': float(systematic_var),
            'idiosyncratic_var': float(idiosyncratic_var),
            'r_squared': float(r_squared)
        }

    def run(self, ticker: str, price_data: pd.DataFrame, ff_factors: pd.DataFrame, risk_free_rate: Optional[pd.Series] = None) -> ModuleResult:
        """
        Execute Factor Model analysis.

        Parameters
        ----------
        ticker : str
            Asset ticker.
        price_data : pd.DataFrame
            Price/Return data.
        ff_factors : pd.DataFrame
            Fama-French factors data.
        risk_free_rate : Optional[pd.Series], optional
            Risk-free rate series, by default None.

        Returns
        -------
        ModuleResult
            Factor model results.
        """
        logger.info(f"Running Factor Model Module for {ticker}")
        
        try:
            factors = self.load_ff_factors(ff_factors)
        except Exception as e:
            return ModuleResult(
                module_name="Factors",
                ticker=ticker,
                normalised_score=0.0,
                warnings=[f"Factor data error: {str(e)}"],
            )

        # Ensure we have monthly returns for factor analysis
        close_col = 'Adj Close' if 'Adj Close' in price_data.columns else 'Close'
        prices = price_data[close_col]
        monthly_prices = prices.resample('ME').last()
        monthly_returns = monthly_prices.pct_change().dropna()
        # Align index to first of month to match Fama-French factors
        monthly_returns.index = monthly_returns.index.to_period('M').to_timestamp()

        # Align indices
        df = pd.concat([monthly_returns.rename('Return'), factors], axis=1).dropna()
        if df.empty:
            return ModuleResult(
                module_name="Factors",
                ticker=ticker,
                normalised_score=50.0,
                warnings=["No overlapping data between returns and factors"],
            )

            
        returns_align = df['Return']
        rf = df['RF']
        excess_returns = returns_align - rf
        
        factor_subset = df.drop(columns=['Return', 'RF'])
        
        # OLS
        ols_res = self.estimate_factor_loadings(excess_returns, factor_subset, method='ols')
        
        if 'error' in ols_res:
            return ModuleResult(module_name="Factors", ticker=ticker, normalised_score=50.0, warnings=[str(ols_res.get('error', ''))])
            
        # Robust
        robust_res = self.estimate_factor_loadings(excess_returns, factor_subset, method='robust')
        
        # Rolling
        rolling_betas = self.rolling_factor_loadings(excess_returns, factor_subset, window=36)
        
        # IR
        ir_res = self.compute_information_ratio(ols_res['residuals'])
        
        # Variance decomposition
        var_decomp = self.variance_decomposition(returns_align, factor_subset, ols_res['betas'])
        
        # Check robustness flag
        robustness_concern = False
        if not ('error' in robust_res):
            ols_alpha = ols_res['alpha']
            rob_alpha = robust_res['alpha']
            if abs(ols_alpha - rob_alpha) > 0.01: # Large difference in monthly alpha
                robustness_concern = True
                
        metrics = {
            'ols_results': {k: v for k, v in ols_res.items() if k != 'residuals'},
            'robust_results': {k: v for k, v in robust_res.items() if k != 'residuals'},
            'information_ratio': ir_res,
            'variance_decomposition': var_decomp,
            'robustness_concern': robustness_concern,
            # Rolling betas not stored fully to keep metrics light, but could store latest
            'latest_rolling': rolling_betas.iloc[-1].to_dict() if not rolling_betas.empty else {}
        }
        
        # --- Graduated scoring ---
        # Base: 50 (neutral).
        # Primary driver: alpha t-statistic (maps to ±25 points).
        #   t < -2 → -25, t = 0 → 0, t > 2 → +25
        # Secondary: R² quality bonus (maps to ±10 points).
        #   R² < 0.1 → -10 (model explains little), R² > 0.7 → +10
        # Tertiary: Information Ratio (maps to ±15 points).
        score = 50.0

        alpha_t = ols_res['t_stats'].get('const', 0.0)
        r2 = float(ols_res.get('r_squared', 0.0))

        # Alpha contribution: clamp t-stat to ±3, scale to ±25
        alpha_contribution = float(np.clip(alpha_t, -3.0, 3.0)) / 3.0 * 25.0
        score += alpha_contribution

        # R² contribution: deviation from 0.4 "typical" benchmark
        r2_contribution = (r2 - 0.40) / 0.30 * 10.0  # ±10 points
        r2_contribution = float(np.clip(r2_contribution, -10.0, 10.0))
        score += r2_contribution

        # IR contribution
        if 'ir' in ir_res:
            ir_val = float(ir_res['ir'])
            ir_contribution = float(np.clip(ir_val, -2.0, 2.0)) / 2.0 * 15.0
            score += ir_contribution

        score = max(0.0, min(100.0, score))
        
        return ModuleResult(
            module_name="Factors",
            ticker=ticker,
            normalised_score=score,
            components=metrics,
            metadata={"description": "Systematic factor exposures and alpha decomposition"},
        )

    def explain(self, result: ModuleResult) -> str:
        """
        Provide human-readable explanation of the Factor results.

        Parameters
        ----------
        result : ModuleResult
            Result from run().

        Returns
        -------
        str
            Explanation text.
        """
        lines = [f"--- Factor Model Analysis for {result.ticker} ---",
                 f"Score: {result.score:.2f}/100", ""]
        
        metrics = result.metrics
        
        if 'error' in result.details:
            lines.append(f"Error: {result.details['error']}")
            return "\n".join(lines)
            
        ols = metrics.get('ols_results', {})
        if 'alpha' in ols:
            lines.append("Factor Loadings (OLS):")
            lines.append(f"Alpha (monthly): {ols['alpha']:.4f}")
            for factor, beta in ols.get('betas', {}).items():
                pval = ols.get('p_values', {}).get(factor, 1.0)
                sig = "*" if pval < 0.05 else ""
                lines.append(f"{factor} Beta: {beta:.3f}{sig}")
            lines.append(f"R-squared: {ols.get('r_squared', 0):.3f}")
            lines.append("")
            
        ir = metrics.get('information_ratio', {})
        if 'ir' in ir:
            lines.append(f"Information Ratio: {ir['ir']:.2f} ({ir.get('interpretation', 'Unknown')})")
            lines.append(f"Annualised Alpha: {ir.get('annualised_alpha', 0)*100:.2f}%")
            
        if metrics.get('robustness_concern'):
            lines.append("")
            lines.append("WARNING: Robustness concern flagged! OLS and Robust regression estimates differ substantially.")
            
        return "\n".join(lines)
