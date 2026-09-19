"""
Robust cross-sectional Z-score normalization using median and MAD.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def normalize(df: pd.DataFrame, scale_factor: float = 1.4826, **params) -> pd.DataFrame:
    r"""
    Apply robust cross-sectional Z-score normalization using median and MAD.
    
    Formula:
        MAD = median(|x_i - median(x)|)
        z_i = (x_i - median(x)) / (scale_factor * MAD(x))
        
    The scale factor 1.4826 is used to make the MAD consistent with the standard deviation
    for a normal distribution (since 1 / \Phi^{-1}(0.75) \approx 1.4826).
    
    If MAD is exactly 0 for a characteristic, the resulting Z-score
    for that characteristic will be entirely NaN, rather than falling back to standard deviation.
    A warning is logged.
    
    Args:
        df: DataFrame with rows=securities, columns=characteristics for ONE date.
        scale_factor: Constant to ensure consistency with standard deviation (default 1.4826).
        **params: Additional parameters.
        
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    median = df.median(skipna=True)
    
    # Calculate MAD manually as median of absolute deviations
    abs_dev = (df - median).abs()
    mad = abs_dev.median(skipna=True)
    
    # Check for MAD = 0
    zero_mad_mask = (mad == 0)
    if zero_mad_mask.any():
        logger.warning(f"MAD is zero for {zero_mad_mask.sum()} characteristics. Returning NaN for these columns.")
        
    # Replace 0 MAD with NaN to ensure division results in NaN
    mad = mad.replace(0.0, np.nan)
    
    denom = scale_factor * mad
    result = (df - median) / denom
            
    return result
