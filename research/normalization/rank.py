"""
Cross-sectional rank/percentile normalization.
"""
import pandas as pd
import numpy as np

def normalize(df: pd.DataFrame, **params) -> pd.DataFrame:
    r"""
    Apply cross-sectional rank/percentile normalization.
    
    Produces values strictly in (0, 100).
    
    Formula:
        p_{i,t,k} = (r_{i,t,k} - 0.5) / n_t
        result = p_{i,t,k} * 100
        
    where:
    - r_{i,t,k} is the rank of security i at date t for characteristic k.
    - Ties are resolved using the 'average' method.
    - n_t is the number of non-NaN observations for characteristic k.
    
    Excluding 0 and 100 is required if an inverse normal transformation will subsequently be applied.
    
    Args:
        df: DataFrame with rows=securities, columns=characteristics for ONE date.
        **params: Additional parameters (ignored).
        
    Returns:
        pd.DataFrame: Normalized DataFrame with values strictly in (0, 100).
    """
    n_t = df.count()
    ranks = df.rank(method='average', na_option='keep')
    
    p = (ranks - 0.5) / n_t
    result = p * 100
    return result
