"""
Deflated Sharpe Ratio (DSR)
===========================
PURPOSE:
  Correct the upward bias in the maximum Sharpe ratio when selecting
  the best performing strategy out of N trials (multiple testing).

MATHEMATICAL FRAMEWORK (Bailey & Lopez de Prado, 2014):
  Expected Maximum Sharpe under the null:
  E[max(SR)] = sqrt(V) * [ (1-gamma)*Phi^{-1}(1-1/N) + gamma*Phi^{-1}(1-1/(N*e)) ]
  where V is the variance of the trials' Sharpe ratios, and gamma is the
  Euler-Mascheroni constant (~0.5772).

  Probabilistic Sharpe Ratio (PSR) accounts for non-normal returns (skew, kurtosis).
  DSR = PSR(E[max(SR)])
"""

import numpy as np
import scipy.stats as stats
import warnings

# Euler-Mascheroni constant
GAMMA = 0.57721566490153286


def expected_max_sharpe_under_multiple_trials(
    n_trials: int, 
    variance: float
) -> float:
    """
    Computes the expected maximum Sharpe ratio under the null hypothesis,
    assuming trials are drawn from a normal distribution with the given variance.

    Parameters
    ----------
    n_trials : int
        Number of independent strategy trials.
    variance : float
        Variance of the Sharpe ratios across the trials.

    Returns
    -------
    float
        The expected maximum Sharpe ratio (SR*).
    """
    if n_trials <= 1:
        return 0.0
        
    if variance <= 0.0:
        return 0.0
        
    std = np.sqrt(variance)
    
    # Extreme value theory approximation for the expected max of N normal variables
    p1 = 1.0 - (1.0 / n_trials)
    p2 = 1.0 - (1.0 / (n_trials * np.e))
    
    # Cap probabilities slightly below 1 to prevent inf
    p1 = min(p1, 1.0 - 1e-15)
    p2 = min(p2, 1.0 - 1e-15)
    
    term1 = (1.0 - GAMMA) * stats.norm.ppf(p1)
    term2 = GAMMA * stats.norm.ppf(p2)
    
    return std * (term1 + term2)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    variance: float,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Computes the Deflated Sharpe Ratio (DSR), which is the Probabilistic
    Sharpe Ratio evaluated against the expected maximum Sharpe from multiple trials.

    Parameters
    ----------
    observed_sharpe : float
        The best observed Sharpe ratio among the trials.
    n_trials : int
        The total number of trials attempted.
    variance : float
        The variance of Sharpe ratios across all trials.
    n_observations : int
        The number of return observations (e.g., trading days).
    skew : float
        Skewness of the returns.
    kurtosis : float
        Kurtosis of the returns (Pearson's definition, where Normal = 3).

    Returns
    -------
    float
        The Deflated Sharpe Ratio (probability between 0 and 1).
    """
    if n_observations < 3:
        return 0.0
        
    # Calculate the hurdle rate SR*
    sr_star = expected_max_sharpe_under_multiple_trials(n_trials, variance)
    
    # Calculate the denominator of the PSR z-statistic
    # denominator = sqrt( 1 - skew*SR + ((kurtosis - 1)/4)*SR^2 )
    term1 = 1.0
    term2 = skew * observed_sharpe
    term3 = ((kurtosis - 1.0) / 4.0) * (observed_sharpe ** 2)
    
    denom_sq = term1 - term2 + term3
    
    if denom_sq <= 0:
        # Extreme non-normality causing negative variance estimate
        warnings.warn("Negative denominator in DSR calculation. Defaulting to 0.")
        return 0.0
        
    denominator = np.sqrt(denom_sq)
    
    # Calculate the z-statistic
    z = (observed_sharpe - sr_star) * np.sqrt(n_observations - 1) / denominator
    
    return float(stats.norm.cdf(z))
