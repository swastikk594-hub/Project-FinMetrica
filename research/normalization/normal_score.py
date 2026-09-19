"""
Standard Normal Deviate of the Z-score (SNDZ).
"""
import pandas as pd
import numpy as np
from scipy.stats import norm

def normalize(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Apply Standard Normal Deviate (inverse-normal transform) normalization.
    
    Formula:
        1. r_i = rank(x_i) / (n + 1)
        2. z_i = \\Phi^{-1}(r_i)
        
    Using (n + 1) in the denominator avoids producing exact 0 or 1 percentiles,
    which would result in infinite values when passed to the inverse normal CDF.
    
    Args:
        df: DataFrame with rows=securities, columns=characteristics for ONE date.
        **params: Additional parameters (ignored).
        
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    n = df.count()
    ranks = df.rank(method='average', na_option='keep')
    
    # Compute percentiles, strictly inside (0, 1)
    percentiles = ranks / (n + 1)
    
    # Apply inverse normal CDF
    # scipy.stats.norm.ppf handles NaN by returning NaN
    result_data = norm.ppf(percentiles.values)
    
    result = pd.DataFrame(result_data, index=df.index, columns=df.columns)
    
    return result
