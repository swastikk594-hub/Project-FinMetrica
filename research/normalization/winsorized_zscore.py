"""
Winsorized cross-sectional Z-score normalization.
"""
import pandas as pd
import numpy as np

def normalize(df: pd.DataFrame, lower_pct: float = 0.01, upper_pct: float = 0.99, **params) -> pd.DataFrame:
    """
    Apply Winsorized cross-sectional Z-score normalization.
    
    Values are clipped at percentile bounds BEFORE computing the Z-score.
    
    Formula:
        1. q_low = quantile(lower_pct), q_high = quantile(upper_pct)
        2. x_clip = clip(x, q_low, q_high)
        3. z_i = (x_clip_i - mean(x_clip)) / std(x_clip)
        
    Args:
        df: DataFrame with rows=securities, columns=characteristics for ONE date.
        lower_pct: Lower percentile bound (default 0.01).
        upper_pct: Upper percentile bound (default 0.99).
        **params: Additional parameters.
        
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    q_low = df.quantile(lower_pct)
    q_high = df.quantile(upper_pct)
    
    # Clip the dataframe column by column
    df_clipped = df.clip(lower=q_low, upper=q_high, axis=1)
    
    mean = df_clipped.mean(skipna=True)
    std = df_clipped.std(skipna=True)
    
    std_safe = std.replace(0.0, np.nan)
    result = (df_clipped - mean) / std_safe
    
    for col in df.columns:
        if std[col] == 0:
            result[col] = np.where(df[col].isna(), np.nan, 0.0)
            
    return result
