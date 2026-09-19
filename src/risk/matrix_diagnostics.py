"""
Matrix Diagnostics
==================
PURPOSE:
  Evaluate the stability and conditioning of covariance matrices.
  Ill-conditioned matrices amplify estimation noise during Markowitz
  optimization, leading to unstable weights and poor out-of-sample
  performance.

MATHEMATICAL FRAMEWORK:
  Let Σ be an N x N covariance matrix with eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_N ≥ 0.

  1. Condition Number:
     κ(Σ) = λ_1 / λ_N
     Measures sensitivity of the solution (Σ^{-1}) to perturbations in the input.
     A high condition number indicates multicollinearity or near-singular matrices.

  2. Eigenvalue Spectrum:
     The set {λ_1, ..., λ_N}. A steep spectrum (few large eigenvalues, many near 0)
     suggests the risk is concentrated in a few dominant factors.

  3. Effective Rank (Roy & Vetterli, 2007):
     Let p_i = λ_i / sum(λ). The Shannon entropy is H(p) = -sum(p_i * log(p_i)).
     Effective Rank = exp(H(p))
     Provides a continuous measure of dimensionality. A matrix with N equal
     eigenvalues has effective rank N. If one eigenvalue dominates, it approaches 1.

  4. Participation Ratio:
     PR = (sum(λ))^2 / sum(λ^2)
     Another measure of effective dimensionality. Bounds are [1, N].
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List


def condition_number(sigma: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Computes the condition number (ratio of largest to smallest eigenvalue).

    Parameters
    ----------
    sigma : np.ndarray or pd.DataFrame
        Covariance matrix.

    Returns
    -------
    float
        Condition number. High values (>1000) indicate instability.
    """
    mat = sigma.values if isinstance(sigma, pd.DataFrame) else sigma
    # For a symmetric positive semi-definite matrix, cond is ratio of max to min eigenvalue
    # Cond handles singular matrices by returning inf or very large numbers
    return np.linalg.cond(mat)


def eigenvalue_spectrum(sigma: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Returns the eigenvalues sorted in descending order.

    Parameters
    ----------
    sigma : np.ndarray or pd.DataFrame
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Sorted eigenvalues (largest to smallest).
    """
    mat = sigma.values if isinstance(sigma, pd.DataFrame) else sigma
    # eigvalsh is more stable and faster for symmetric/Hermitian matrices
    evals = np.linalg.eigvalsh(mat)
    # Sort descending
    return evals[::-1]


def effective_rank(sigma: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Computes the effective rank using the spectral entropy (Roy & Vetterli formula).
    
    Effective Rank = exp( -sum( p_i * log(p_i) ) )
    where p_i = λ_i / sum(λ).

    Parameters
    ----------
    sigma : np.ndarray or pd.DataFrame
        Covariance matrix.

    Returns
    -------
    float
        Effective rank, bounded between 1 and N.
    """
    evals = eigenvalue_spectrum(sigma)
    # Filter out negative or zero eigenvalues due to numerical noise
    evals = evals[evals > 1e-10]
    
    if len(evals) == 0:
        return 0.0
        
    p = evals / np.sum(evals)
    # Shannon entropy (base e)
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def participation_ratio(sigma: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Computes the participation ratio of the eigenvalues.
    
    PR = (sum(λ))^2 / sum(λ^2)

    Parameters
    ----------
    sigma : np.ndarray or pd.DataFrame
        Covariance matrix.

    Returns
    -------
    float
        Participation ratio, bounded between 1 and N.
    """
    evals = eigenvalue_spectrum(sigma)
    
    # Filter negatives (numerical noise)
    evals = np.clip(evals, 0, None)
    
    sum_evals = np.sum(evals)
    if sum_evals == 0:
        return 0.0
        
    sum_sq_evals = np.sum(evals**2)
    return (sum_evals**2) / sum_sq_evals
