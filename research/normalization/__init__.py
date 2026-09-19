"""
Normalization operators for cross-sectional data.
"""
from typing import Callable
import pandas as pd

from . import zscore
from . import rank
from . import winsorized_zscore
from . import robust_zscore
from . import normal_score

NORMALIZATION_REGISTRY: dict[str, Callable] = {
    'cross_sectional_zscore': zscore.normalize,
    'cross_sectional_rank': rank.normalize,
    'winsorized_zscore': winsorized_zscore.normalize,
    'robust_zscore': robust_zscore.normalize,
    'sndz': normal_score.normalize,
}

def get_normalizer(name: str) -> Callable:
    """
    Return the normalization function by name.
    
    Args:
        name: The name of the normalizer.
        
    Returns:
        Callable: The normalization function.
        
    Raises:
        KeyError: If the normalizer name is not found in the registry.
    """
    if name not in NORMALIZATION_REGISTRY:
        raise KeyError(f"Normalizer '{name}' not found. Available normalizers: {list(NORMALIZATION_REGISTRY.keys())}")
    return NORMALIZATION_REGISTRY[name]

def apply_normalization(df: pd.DataFrame, name: str, **params) -> pd.DataFrame:
    """
    Convenience function: look up normalizer by name and apply it.
    
    Args:
        df: DataFrame with rows=securities, columns=characteristics for ONE date.
        name: The name of the normalizer to apply.
        **params: Additional parameters for the normalizer.
        
    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    normalizer = get_normalizer(name)
    return normalizer(df, **params)
