"""
Statistical Significance Tests for Performance
==============================================
PURPOSE:
  Evaluate whether outperformance of one strategy over another is
  statistically significant, accounting for non-normality and serial
  correlation in returns.

MATHEMATICAL FRAMEWORK:
  1. Jobson-Korkie (1981) with Memmel (2003) Correction:
     Asymptotic test for equality of Sharpe ratios of two correlated
     return series. Corrects for the covariance between the two strategies.
     
  2. Stationary Bootstrap (Politis & Romano, 1994):
     Non-parametric block bootstrap that preserves serial dependence
     by using random block lengths drawn from a geometric distribution.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Tuple


def jobson_korkie_memmel(returns_a: np.ndarray, returns_b: np.ndarray) -> Tuple[float, float]:
    """
    Jobson-Korkie (1981) test for equality of Sharpe Ratios, 
    incorporating Memmel's (2003) correction.

    Null Hypothesis (H0): SR_a = SR_b
    Alternative (H1): SR_a != SR_b

    Parameters
    ----------
    returns_a : np.ndarray
        Returns of strategy A.
    returns_b : np.ndarray
        Returns of strategy B.

    Returns
    -------
    Tuple[float, float]
        (z_stat, p_value)
    """
    # Ensure 1D arrays
    returns_a = np.asarray(returns_a).ravel()
    returns_b = np.asarray(returns_b).ravel()
    
    if len(returns_a) != len(returns_b):
        raise ValueError("Return series must be of the same length.")
        
    T = len(returns_a)
    if T < 10:
        return 0.0, 1.0

    mu_a = np.mean(returns_a)
    mu_b = np.mean(returns_b)
    
    # Degrees of freedom = T-1 for unbiased variance
    var_a = np.var(returns_a, ddof=1)
    var_b = np.var(returns_b, ddof=1)
    
    if var_a == 0 or var_b == 0:
        return 0.0, 1.0
        
    std_a = np.sqrt(var_a)
    std_b = np.sqrt(var_b)
    
    cov_ab = np.cov(returns_a, returns_b)[0, 1]
    
    # Memmel's (2003) correction variance term: theta
    # theta = (1/T) * [ 2(σ_a^2)(σ_b^2) - 2(σ_a)(σ_b)(cov_ab) + 0.5(μ_a^2)(σ_b^2) + 0.5(μ_b^2)(σ_a^2) - (μ_a*μ_b)/(σ_a*σ_b) * (cov_ab^2) ]
    # Wait, the spec has a specific formula:
    # theta = 1/T [ 2(σ_a^2 * σ_b^2) - 2(σ_a * σ_b * cov_ab) + 0.5(μ_a^2 * σ_b^2) + 0.5(μ_b^2 * σ_a^2) - (μ_a * μ_b)/(σ_a * σ_b) * cov_ab^2 ]
    
    term1 = 2 * var_a * var_b
    term2 = 2 * std_a * std_b * cov_ab
    term3 = 0.5 * (mu_a**2) * var_b
    term4 = 0.5 * (mu_b**2) * var_a
    term5 = (mu_a * mu_b) / (std_a * std_b) * (cov_ab**2)
    
    theta = (1.0 / T) * (term1 - term2 + term3 + term4 - term5)
    
    if theta <= 0:
        return 0.0, 1.0
        
    # The test statistic is z = (σ_b * μ_a - σ_a * μ_b) / sqrt(theta)
    numerator = std_b * mu_a - std_a * mu_b
    z_stat = numerator / np.sqrt(theta)
    
    # Two-sided p-value
    p_value = 2 * (1.0 - stats.norm.cdf(abs(z_stat)))
    
    return float(z_stat), float(p_value)


def stationary_bootstrap(data: np.ndarray, block_length: float = 10.0, n_bootstraps: int = 1000, seed: int = None) -> np.ndarray:
    """
    Politis & Romano (1994) Stationary Bootstrap.
    Uses random block lengths drawn from a geometric distribution with mean `block_length`.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    block_length : float
        Expected length of the blocks.
    n_bootstraps : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_bootstraps, len(data)) containing bootstrap resamples.
    """
    data = np.asarray(data).ravel()
    T = len(data)
    
    if seed is None:
        try:
            from src.data.config_loader import get_config
            seed = get_config().get('random_seed', 42)
        except Exception:
            seed = 42
            
    rng = np.random.default_rng(seed)
    prob = 1.0 / block_length
    
    resamples = np.zeros((n_bootstraps, T))
    for i in range(n_bootstraps):
        indices = []
        while len(indices) < T:
            idx = rng.integers(0, T)
            length = rng.geometric(prob)
            block_idx = [(idx + j) % T for j in range(length)]
            indices.extend(block_idx)
        indices = np.array(indices[:T])
        resamples[i] = data[indices]
        
    return resamples


def stationary_bootstrap_sharpe_diff(
    returns_a: np.ndarray, 
    returns_b: np.ndarray, 
    block_length: float = 10.0, 
    n_bootstraps: int = 1000, 
    seed: int = None
) -> Tuple[float, float]:
    """
    Politis & Romano (1994) Stationary Bootstrap for the difference in Sharpe ratios.
    Uses random block lengths drawn from a geometric distribution with mean `block_length`.

    Parameters
    ----------
    returns_a : np.ndarray
    returns_b : np.ndarray
    block_length : float
        Expected length of the blocks.
    n_bootstraps : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        (bootstrap_p_value, empirical_sr_diff)
    """
    returns_a = np.asarray(returns_a).ravel()
    returns_b = np.asarray(returns_b).ravel()
    T = len(returns_a)
    
    if T < 10:
        return 1.0, 0.0
        
    if seed is None:
        try:
            from src.data.config_loader import get_config
            seed = get_config().get('random_seed', 42)
        except Exception:
            seed = 42
            
    rng = np.random.default_rng(seed)
    prob = 1.0 / block_length
    
    # Empirical Sharpe Ratios
    std_a = np.std(returns_a, ddof=1)
    std_b = np.std(returns_b, ddof=1)
    
    sr_a = np.mean(returns_a) / std_a if std_a > 0 else 0
    sr_b = np.mean(returns_b) / std_b if std_b > 0 else 0
    diff_emp = sr_a - sr_b
    
    diffs_boot = np.zeros(n_bootstraps)
    
    for i in range(n_bootstraps):
        # Generate stationary bootstrap indices
        indices = []
        while len(indices) < T:
            # Start index uniform
            idx = rng.integers(0, T)
            # Block length geometric
            length = rng.geometric(prob)
            
            # Wrap around
            block_idx = [(idx + j) % T for j in range(length)]
            indices.extend(block_idx)
            
        indices = np.array(indices[:T])
        
        boot_a = returns_a[indices]
        boot_b = returns_b[indices]
        
        std_boot_a = np.std(boot_a, ddof=1)
        std_boot_b = np.std(boot_b, ddof=1)
        
        sr_boot_a = np.mean(boot_a) / std_boot_a if std_boot_a > 0 else 0
        sr_boot_b = np.mean(boot_b) / std_boot_b if std_boot_b > 0 else 0
        
        # Center the bootstrap distribution around 0 to test the null hypothesis (SR_a == SR_b)
        # Shift bootstrap difference by the empirical difference
        diffs_boot[i] = (sr_boot_a - sr_boot_b) - diff_emp
        
    # Two-sided p-value: proportion of bootstrap differences more extreme than empirical diff
    p_value = np.mean(np.abs(diffs_boot) >= np.abs(diff_emp))
    
    return float(p_value), float(diff_emp)

def apply_multiple_comparison_correction(p_values: list[float], method: str = 'benjamini_hochberg') -> list[tuple[float, bool]]:
    """
    Apply multiple-comparison correction to a list of p-values.
    
    Parameters
    ----------
    p_values : List[float]
        List of raw p-values.
    method : str
        'bonferroni' or 'benjamini_hochberg'
        
    Returns
    -------
    List[Tuple[float, bool]]
        List of (adjusted_p_value, is_significant_at_5_percent)
    """
    n = len(p_values)
    if n == 0:
        return []
        
    if method.lower() == 'bonferroni':
        adjusted = [min(1.0, p * n) for p in p_values]
        return [(adj_p, adj_p < 0.05) for adj_p in adjusted]
        
    elif method.lower() in ['benjamini_hochberg', 'bh']:
        # Sort p-values, keeping track of original indices
        indexed_p = list(enumerate(p_values))
        indexed_p.sort(key=lambda x: x[1])
        
        adjusted = [0.0] * n
        is_sig = [False] * n
        
        # BH adjustment: p_adj = min(1.0, p_raw * n / rank)
        # and enforce monotonicity (p_adj must be non-decreasing with rank)
        prev_adj_p = 1.0
        for rank in range(n, 0, -1):
            idx, p_raw = indexed_p[rank - 1]
            adj_p = p_raw * n / rank
            # Ensure monotonicity from right to left
            adj_p = min(prev_adj_p, min(1.0, adj_p))
            prev_adj_p = adj_p
            adjusted[idx] = adj_p
            is_sig[idx] = adj_p < 0.05
            
        return list(zip(adjusted, is_sig))
        
    else:
        raise ValueError(f"Unknown correction method: {method}")
