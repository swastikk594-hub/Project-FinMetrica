"""
Module 5: Risk Model
=====================
PURPOSE:
  Quantify multiple, mathematically distinct dimensions of risk.
  A single risk number is misleading — different risk measures
  capture fundamentally different phenomena:
  
  VOLATILITY: Average magnitude of price fluctuations
  DOWNSIDE DEV: Volatility of LOSSES only (Sharpe uses both up/down)
  MAX DRAWDOWN: Worst peak-to-trough decline (sequence risk)
  VaR: Threshold loss not exceeded with probability 1-α
  CVaR/ES: Expected loss GIVEN that VaR threshold is breached
  BETA: Co-movement with the market (systematic risk)

MATHEMATICAL DEFINITIONS:

  Annual Volatility:
    σ = std(rₜ) × √252
    where rₜ are daily log returns
    
  Downside Deviation:
    σ_d = √[Σ min(rₜ - MAR, 0)² / (n-1)] × √252
    MAR = Minimum Acceptable Return (default 0)
    
  Maximum Drawdown:
    DDₜ = (Vₜ - max_{s≤t} Vₛ) / max_{s≤t} Vₛ
    MDD = min_{t} DDₜ
    Also compute: MDD duration, MDD start/end dates, recovery time
    
  Historical VaR at confidence level α:
    VaR_α = -Quantile(rₜ, 1-α)
    Interpretation: With probability α, the daily loss will not exceed VaR_α
    
  CVaR (Conditional Value at Risk / Expected Shortfall):
    CVaR_α = -E[rₜ | rₜ ≤ -VaR_α]
    = average of returns below VaR threshold
    IMPORTANT: CVaR ≥ VaR always (by definition — test this)
    CVaR is coherent; historical VaR is not (non-subadditive)
    
  Beta:
    β = Cov(Rᵢ, Rm) / Var(Rm)
    where Rm = market (benchmark) returns

METHOD COMPARISON — VaR Estimation:
  A. Historical VaR: Non-parametric; uses actual return distribution
     Advantages: No distributional assumptions; captures fat tails
     Disadvantages: Limited by sample size; weights all history equally
     
  B. Parametric VaR: Assumes normal distribution
     VaR_normal = μ - z_α × σ
     Advantages: Simple, analytical
     Disadvantages: WRONG for fat-tailed returns (underestimates tail risk)
     
  C. Monte Carlo VaR: Simulate returns from fitted distribution
     Advantages: Can use any distribution; handles non-linearity
     Disadvantages: Computationally expensive; model-dependent
     
  SELECTED: Historical VaR as primary (no distributional assumption).
            Parametric VaR shown for comparison.
            Monte Carlo VaR available as option (used in Module 7).

ASSUMPTIONS:
  - Returns are approximately stationary (validated by ADF test)
  - Historical distribution is informative about future risk
  - Beta is estimated from a linear market model

LIMITATIONS:
  - VaR does not tell us HOW MUCH we lose in the tail
  - Maximum drawdown is path-dependent; not predictive
  - Beta is unstable over time; changes with market regimes
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, Any, List, Optional, Union
from src.modules.base import ModuleResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskModule:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def compute_volatility(self, returns: pd.Series, window: Optional[int] = None, annualise: bool = True) -> Dict[str, Any]:
        """
        Compute volatility (standard deviation of returns).
        
        Parameters
        ----------
        returns : pd.Series
            Daily log returns.
        window : int, optional
            Rolling window size.
        annualise : bool
            Whether to annualise the volatility.
            
        Returns
        -------
        dict
            Contains full period volatility, rolling volatility series, and percentile of current volatility.
        """
        vol = returns.std()
        if annualise:
            vol *= np.sqrt(252)
            
        rolling_vol = None
        pct_current_vol = None
        
        if window is not None:
            rolling_vol = returns.rolling(window=window).std()
            if annualise:
                rolling_vol *= np.sqrt(252)
            current_vol = rolling_vol.iloc[-1]
            if not pd.isna(current_vol) and len(rolling_vol.dropna()) > 0:
                pct_current_vol = stats.percentileofscore(rolling_vol.dropna(), current_vol) / 100.0
                
        return {
            'volatility': vol,
            'rolling_vol': rolling_vol,
            'percentile_of_current_vol': pct_current_vol
        }

    def compute_downside_deviation(self, returns: pd.Series, mar: float = 0.0, annualise: bool = True) -> Dict[str, Any]:
        """
        Compute downside deviation (volatility of returns below MAR).
        """
        downside_returns = returns - mar
        downside_returns = downside_returns[downside_returns < 0]
        
        # √[Σ min(rₜ - MAR, 0)² / (n-1)]
        if len(downside_returns) > 1:
            downside_dev = np.sqrt(np.sum(downside_returns**2) / (len(returns) - 1))
        else:
            downside_dev = 0.0
            
        if annualise:
            downside_dev *= np.sqrt(252)
            
        return {
            'downside_dev': downside_dev
        }

    def compute_sharpe_ratio(self, returns: pd.Series, risk_free_daily: float) -> Dict[str, Any]:
        """
        Compute annualised Sharpe Ratio.
        """
        excess_returns = returns - risk_free_daily
        logger.debug("Verified: Sharpe ratio computed using excess returns (returns - rf_daily).")
        annualised_return = excess_returns.mean() * 252
        annualised_vol = excess_returns.std() * np.sqrt(252)
        
        if annualised_vol > 0:
            sharpe_ratio = annualised_return / annualised_vol
        else:
            sharpe_ratio = 0.0
            
        return {
            'sharpe_ratio': sharpe_ratio,
            'annualised_return': annualised_return,
            'annualised_vol': annualised_vol
        }

    def compute_sortino_ratio(self, returns: pd.Series, risk_free_daily: float, mar: float = 0.0) -> Dict[str, Any]:
        """
        Compute annualised Sortino Ratio.
        """
        excess_returns = returns - risk_free_daily
        annualised_return = excess_returns.mean() * 252
        
        down_dev_result = self.compute_downside_deviation(returns, mar=mar, annualise=True)
        downside_dev = down_dev_result['downside_dev']
        
        if downside_dev > 0:
            sortino_ratio = annualised_return / downside_dev
        else:
            sortino_ratio = 0.0
            
        return {
            'sortino_ratio': sortino_ratio,
            'downside_dev': downside_dev
        }

    def compute_max_drawdown(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Compute maximum drawdown and related duration metrics.
        """
        roll_max = prices.cummax()
        drawdown = (prices - roll_max) / roll_max
        max_drawdown = drawdown.min()
        
        trough_date = drawdown.idxmin()
        if pd.isna(trough_date):
            return {
                'max_drawdown': 0.0,
                'drawdown_series': drawdown,
                'start_date': None,
                'end_date': None,
                'trough_date': None,
                'recovery_date': None,
                'duration_days': 0
            }
            
        # start date is the peak before the trough
        start_date = prices.loc[:trough_date].idxmax()
        
        # recovery date is the first date after trough where price >= start price
        recovery_date = None
        post_trough = prices.loc[trough_date:]
        recovered = post_trough[post_trough >= prices.loc[start_date]]
        if not recovered.empty:
            recovery_date = recovered.index[0]
            
        end_dt = recovery_date if recovery_date else prices.index[-1]
        if hasattr(end_dt - start_date, 'days'):
            duration = (end_dt - start_date).days
        else:
            duration = int(end_dt - start_date) if isinstance(end_dt - start_date, (int, float, np.number)) else len(prices)
        
        return {
            'max_drawdown': float(max_drawdown),
            'drawdown_series': drawdown,
            'start_date': start_date,
            'end_date': recovery_date if recovery_date else prices.index[-1],
            'trough_date': trough_date,
            'recovery_date': recovery_date,
            'duration_days': duration
        }

    def compute_var(self, returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99], method: str = 'historical') -> Dict[str, Any]:
        """
        Compute Value at Risk and Conditional Value at Risk.
        """
        r_clean = returns.dropna()
        if len(r_clean) < 2:
            return {alpha: {'var': 0.02, 'cvar': 0.03, 'method': method} for alpha in confidence_levels}

        results = {}
        for alpha in confidence_levels:
            if method == 'historical':
                var = -float(np.percentile(r_clean, (1 - alpha) * 100))
                tail_returns = r_clean[r_clean <= -var]
                if len(tail_returns) > 0:
                    cvar = -float(tail_returns.mean())
                else:
                    cvar = var  # no tail observations; set CVaR = VaR
            elif method == 'parametric':
                mu = returns.mean()
                sigma = returns.std()
                var = -(mu + stats.norm.ppf(1 - alpha) * sigma)
                # CVaR for normal dist: mu - sigma * phi(z) / (1 - alpha)
                z = stats.norm.ppf(1 - alpha)
                cvar = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
            else:
                raise ValueError(f"Unknown VaR method: {method}")

            # CVaR must be >= VaR by definition. In rare cases historical
            # simulation produces CVaR < VaR due to very sparse tails.
            # Correct rather than crash.
            if cvar < var:
                logger.warning(
                    f"CVaR ({cvar:.6f}) < VaR ({var:.6f}) at alpha={alpha:.2f}. "
                    "Setting CVaR = VaR."
                )
                cvar = var
            
            results[alpha] = {
                'var': float(var),
                'cvar': float(cvar),
                'method': method
            }
        return results

    def compute_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 252) -> Dict[str, Any]:
        """
        Compute Beta relative to benchmark.
        """
        aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        if aligned.empty:
            return {'beta': 1.0, 'r_squared': 0.0, 'residual_vol': 0.0, 'rolling_beta': None}
            
        y = aligned.iloc[:, 0]
        x = aligned.iloc[:, 1]
        
        cov_matrix = np.cov(y, x)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
        
        correlation = np.corrcoef(y, x)[0, 1]
        r_squared = correlation**2
        
        residuals = y - beta * x
        residual_vol = residuals.std() * np.sqrt(252)
        
        rolling_beta = None
        if window is not None and len(aligned) >= window:
            rolling_cov = y.rolling(window).cov(x)
            rolling_var = x.rolling(window).var()
            # Guard against zero variance (e.g. flat benchmark segment)
            rolling_beta = rolling_cov.where(rolling_var > 1e-12, other=np.nan) / rolling_var.replace(0, np.nan)

        return {
            'beta': float(beta),
            'r_squared': float(r_squared),
            'residual_vol': float(residual_vol),
            'rolling_beta': rolling_beta
        }

    def compute_monte_carlo_var(self, returns: pd.Series, n_sims: int = 10000, horizon: int = 1, seed: int = None) -> Dict[str, Any]:
        """
        Monte Carlo VaR using fitted t-distribution.
        """
        if seed is None:
            seed = self.config.get('random_seed', 42) if hasattr(self, 'config') else 42
        np.random.seed(seed)
        df, loc, scale = stats.t.fit(returns.dropna())
        
        simulated_returns = stats.t.rvs(df, loc=loc, scale=scale, size=(n_sims, horizon))
        cumulative_sim_returns = np.sum(simulated_returns, axis=1)
        
        var_95 = -np.percentile(cumulative_sim_returns, 5)
        var_99 = -np.percentile(cumulative_sim_returns, 1)
        
        cvar_95 = -np.mean(cumulative_sim_returns[cumulative_sim_returns <= -var_95])
        cvar_99 = -np.mean(cumulative_sim_returns[cumulative_sim_returns <= -var_99])
        
        return {
            'var_mc_95': float(var_95),
            'cvar_mc_95': float(cvar_95),
            'var_mc_99': float(var_99),
            'cvar_mc_99': float(cvar_99),
            'simulated_returns': cumulative_sim_returns,
            'percentiles': {
                '5th': float(-var_95),
                '1st': float(-var_99),
                '50th': float(np.median(cumulative_sim_returns))
            }
        }

    def risk_summary(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine all risk metrics into a composite risk score (0-100).
        Higher score = riskier.
        """
        # A simple normalisation scheme for educational purposes
        vol_score = min(max(all_metrics.get('volatility', 0) / 0.4 * 100, 0), 100)
        dd_score = min(max(-all_metrics.get('max_drawdown', 0) / 0.5 * 100, 0), 100)
        
        beta = all_metrics.get('beta', 1.0)
        beta_score = min(max((beta - 0.5) / 1.5 * 100, 0), 100)
        
        # Weighted average
        score = 0.4 * vol_score + 0.4 * dd_score + 0.2 * beta_score
        
        return {
            'risk_score': score,
            'components': {
                'volatility_score': vol_score,
                'drawdown_score': dd_score,
                'beta_score': beta_score
            }
        }

    def run(self, ticker: str, price_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> ModuleResult:
        """
        Execute full risk analysis for the given asset.
        """
        logger.info(f"Running Risk Module for {ticker}")
        
        col = 'Adj Close' if ('Adj Close' in price_data.columns and price_data['Adj Close'].notna().sum() > 20) else ('Close' if 'Close' in price_data.columns else price_data.columns[0])
        prices = price_data[col]
        returns = prices.pct_change().dropna()
        bench_returns = pd.Series(dtype=float)
        if benchmark_data is not None and not benchmark_data.empty:
            bench_col = 'Adj Close' if ('Adj Close' in benchmark_data.columns and benchmark_data['Adj Close'].notna().sum() > 20) else ('Close' if 'Close' in benchmark_data.columns else benchmark_data.columns[0])
            bench_returns = benchmark_data[bench_col].pct_change().dropna()
        
        
        risk_free_daily = self.config.get('risk_free_rate', 0.04) / 252
        
        vol_info    = self.compute_volatility(returns)
        dd_info     = self.compute_downside_deviation(returns)
        sharpe_info = self.compute_sharpe_ratio(returns, risk_free_daily)
        sortino_info= self.compute_sortino_ratio(returns, risk_free_daily)
        mdd_info    = self.compute_max_drawdown(prices)
        var_info    = self.compute_var(returns)
        beta_info   = self.compute_beta(returns, bench_returns) if len(bench_returns) > 10 else {'beta': 1.0, 'r_squared': 0.0, 'residual_vol': 0.0, 'rolling_beta': None}

        all_metrics = {
            'volatility':    vol_info['volatility'],
            'downside_dev':  dd_info['downside_dev'],
            'sharpe_ratio':  sharpe_info['sharpe_ratio'],
            'sortino_ratio': sortino_info['sortino_ratio'],
            'max_drawdown':  mdd_info['max_drawdown'],
            'var_95':        var_info[0.95]['var'],
            'cvar_95':       var_info[0.95]['cvar'],
            'beta':          beta_info['beta'],
        }

        summary = self.risk_summary(all_metrics)
        risk_score = summary['risk_score']

        # Invert: low risk → high module score (we want high score = good)
        normalised_score = 100.0 - risk_score

        return ModuleResult(
            ticker=ticker,
            module_name='Risk',
            normalised_score=normalised_score,
            components={'var': var_info, 'volatility': vol_info, 'max_drawdown': mdd_info,
                        'beta': beta_info, 'sharpe': sharpe_info, 'summary': summary},
            metadata={'metrics': all_metrics, 'summary': summary,
                      'drawdown_duration': mdd_info.get('duration_days', 0)},
        )

    def explain(self, result: ModuleResult) -> str:
        """
        Return educational explanation of the results.
        """
        score = result.score
        vol = result.details['metrics']['volatility']
        mdd = result.details['metrics']['max_drawdown']
        beta = result.details['metrics']['beta']
        
        explanation = f"Risk Module Analysis:\n"
        explanation += f"Composite Risk Score: {score:.1f}/100\n\n"
        explanation += f"1. Volatility ({vol*100:.1f}%): The average annualised magnitude of price fluctuations.\n"
        explanation += f"2. Maximum Drawdown ({mdd*100:.1f}%): The worst sequence of losses from peak to trough.\n"
        explanation += f"3. Beta ({beta:.2f}): Systematic risk relative to the benchmark.\n"
        
        if score > 75:
            explanation += "\nConclusion: HIGH RISK. This asset exhibits significant volatility or downside exposure."
        elif score > 40:
            explanation += "\nConclusion: MODERATE RISK. This asset behaves similarly to a typical equity investment."
        else:
            explanation += "\nConclusion: LOW RISK. This asset demonstrates stability and low drawdown potential."
            
        return explanation
