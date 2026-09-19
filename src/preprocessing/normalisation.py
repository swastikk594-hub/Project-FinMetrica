"""
Normalisation Method Comparison
================================
PROBLEM: Seven modules produce scores on different scales.
We need a normalisation framework that:
  1. Does not introduce look-ahead bias
  2. Is robust to outliers
  3. Is interpretable

METHOD A — Z-Score:
  z = (x - μ) / σ
  Assumptions: Approximately normal distribution
  Advantages: Interpretable (standard deviations from mean), simple
  Disadvantages: Sensitive to outliers; assumes normality

METHOD B — Min-Max Scaling:
  x' = (x - min) / (max - min)
  Advantages: Bounded [0,1], preserves relationships
  Disadvantages: Extremely sensitive to outliers; look-ahead risk if
                 min/max computed on full dataset

METHOD C — Rank/Percentile Transform:
  x' = rank(x) / n
  Advantages: Completely outlier-robust; no distributional assumptions
  Disadvantages: Loses magnitude information; non-parametric

METHOD D — Robust Scaling:
  x' = (x - median) / IQR
  Advantages: Outlier-robust; no distributional assumptions
  Disadvantages: Less interpretable than z-score

SELECTED:
  - Individual metrics within modules: z-score (with winsorisation at ±3σ)
  - Cross-sectional module scores: rank transform (outlier-robust)
  - Final composite score: linear rescaling to 0-100
  
RATIONALE: z-score is appropriate within modules where metrics are approximately
           comparable. Rank transform is used at the module-score level because
           module scores will have different distributions and outlier behaviour.
           
LOOK-AHEAD PREVENTION: All normalisations in the backtesting engine use
                        expanding-window statistics (not full-sample statistics).
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def zscore_normalise(series: pd.Series, winsorise: bool = True, winsor_limits: tuple = (-3, 3)) -> pd.Series:
    """
    Normalize a series using Z-score methodology.
    
    z = (x - μ) / σ
    
    Parameters
    ----------
    series : pd.Series
        Input series to normalise.
    winsorise : bool, optional
        Whether to clip outliers to limits, by default True
    winsor_limits : tuple, optional
        Lower and upper limits in standard deviations for clipping, by default (-3, 3)
        
    Returns
    -------
    pd.Series
        Z-score normalised series.
    """
    try:
        mean = series.mean()
        std = series.std()
        
        if std == 0 or pd.isna(std):
            logger.warning("Standard deviation is zero or NaN. Returning zero series.")
            return pd.Series(0, index=series.index)
            
        z_scores = (series - mean) / std
        
        if winsorise:
            z_scores = z_scores.clip(lower=winsor_limits[0], upper=winsor_limits[1])
            
        return z_scores
    except Exception as e:
        logger.error(f"Error in zscore_normalise: {e}")
        raise

def minmax_normalise(series: pd.Series, feature_range: tuple = (0, 1)) -> pd.Series:
    """
    Normalize a series using min-max scaling.
    
    x' = (x - min) / (max - min) * (max_range - min_range) + min_range
    
    Parameters
    ----------
    series : pd.Series
        Input series to normalise.
    feature_range : tuple, optional
        Desired range of transformed data, by default (0, 1)
        
    Returns
    -------
    pd.Series
        Min-max scaled series.
    """
    try:
        s_min = series.min()
        s_max = series.max()
        
        if s_max == s_min:
            logger.warning("Max and min are equal. Returning uniform series.")
            return pd.Series(feature_range[0], index=series.index)
            
        scaled = (series - s_min) / (s_max - s_min)
        scaled = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        return scaled
    except Exception as e:
        logger.error(f"Error in minmax_normalise: {e}")
        raise

def rank_normalise(series: pd.Series) -> pd.Series:
    """
    Normalize a series using rank percentile transform (0 to 100).
    
    x' = rank(x) / n * 100
    
    Parameters
    ----------
    series : pd.Series
        Input series to normalise.
        
    Returns
    -------
    pd.Series
        Percentile rank normalised series.
    """
    try:
        ranks = series.rank(pct=True) * 100
        return ranks
    except Exception as e:
        logger.error(f"Error in rank_normalise: {e}")
        raise

def robust_normalise(series: pd.Series) -> pd.Series:
    """
    Normalize a series using robust scaling based on median and IQR.
    
    x' = (x - median) / IQR
    
    Parameters
    ----------
    series : pd.Series
        Input series to normalise.
        
    Returns
    -------
    pd.Series
        Robust scaled series.
    """
    try:
        median = series.median()
        q75, q25 = series.quantile(0.75), series.quantile(0.25)
        iqr = q75 - q25
        
        if iqr == 0:
            logger.warning("IQR is zero. Using standard deviation instead or returning zeros.")
            std = series.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=series.index)
            return (series - median) / std
            
        robust_scaled = (series - median) / iqr
        return robust_scaled
    except Exception as e:
        logger.error(f"Error in robust_normalise: {e}")
        raise

def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each column across rows (cross-sectional normalisation).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
        
    Returns
    -------
    pd.DataFrame
        Cross-sectionally normalised dataframe.
    """
    try:
        # Compute mean and std for each row across all columns
        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1)
        
        # Prevent division by zero
        row_std = row_std.replace(0, np.nan)
        
        return df.sub(row_mean, axis=0).div(row_std, axis=0)
    except Exception as e:
        logger.error(f"Error in cross_sectional_zscore: {e}")
        raise

def expanding_zscore(series: pd.Series) -> pd.Series:
    """
    Compute Z-score using expanding window statistics to prevent look-ahead bias.
    
    Parameters
    ----------
    series : pd.Series
        Input series to normalise.
        
    Returns
    -------
    pd.Series
        Expanding Z-score series.
    """
    try:
        expanding_mean = series.expanding(min_periods=2).mean()
        expanding_std = series.expanding(min_periods=2).std()
        
        # Prevent division by zero
        expanding_std = expanding_std.replace(0, np.nan)
        
        z_scores = (series - expanding_mean) / expanding_std
        return z_scores
    except Exception as e:
        logger.error(f"Error in expanding_zscore: {e}")
        raise

def normalise_to_score(series: pd.Series, min_score: float = 0, max_score: float = 100) -> pd.Series:
    """
    Linearly rescale a series to a given score range [min_score, max_score].
    
    Parameters
    ----------
    series : pd.Series
        Input series.
    min_score : float, optional
        Minimum score bound, by default 0
    max_score : float, optional
        Maximum score bound, by default 100
        
    Returns
    -------
    pd.Series
        Rescaled series.
    """
    try:
        return minmax_normalise(series, feature_range=(min_score, max_score))
    except Exception as e:
        logger.error(f"Error in normalise_to_score: {e}")
        raise

def compare_normalisation_methods(series: pd.Series) -> pd.DataFrame:
    """
    Compare different normalisation methods applied to a series.
    
    Parameters
    ----------
    series : pd.Series
        Input series.
        
    Returns
    -------
    pd.DataFrame
        Summary statistics of the various normalisations.
    """
    try:
        z_scaled = zscore_normalise(series, winsorise=False)
        z_winsorised = zscore_normalise(series, winsorise=True)
        minmax = minmax_normalise(series)
        ranked = rank_normalise(series)
        robust = robust_normalise(series)
        
        df_comp = pd.DataFrame({
            'Original': series,
            'Z-Score': z_scaled,
            'Z-Score Winsorised': z_winsorised,
            'MinMax': minmax,
            'Rank': ranked,
            'Robust': robust
        })
        
        desc = df_comp.describe().T
        desc['skew'] = df_comp.skew()
        desc['kurtosis'] = df_comp.kurtosis()
        
        return desc
    except Exception as e:
        logger.error(f"Error comparing normalisation methods: {e}")
        raise
