"""
Cross-sectional Z-score normalization.
"""
import pandas as pd
import numpy as np

def normalize(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Apply cross-sectional Z-score normalization.
    
    Formula:
        z_i = (x_i - \\mu) / \\sigma
    
    where \\mu is the mean and \\sigma is the standard deviation.
    
    Args:
        df: DataFrame with rows=securities, columns=characteristics for ONE date.
        **params: Additional parameters (ignored).
        
    Returns:
        pd.DataFrame: Normalized DataFrame. Returns 0 for columns with 0 standard deviation.
    """
    mean = df.mean(skipna=True)
    std = df.std(skipna=True)
    
    # To handle std = 0 gracefully, replace 0 with NaN for division, then handle explicitly
    std_safe = std.replace(0.0, np.nan)
    
    result = (df - mean) / std_safe
    
    # Where std is exactly 0, the result would be NaN. 
    # We want to return zeros for those columns (but keep original NaNs as NaN)
    for col in df.columns:
        if std[col] == 0:
            result[col] = np.where(df[col].isna(), np.nan, 0.0)
            
    return result
