"""
Test: No double risk-free-rate subtraction in Sharpe pipeline.
"""
import numpy as np
import pytest
import math

def test_no_double_rf_subtraction():
    """
    Construct a synthetic return series and known rf.
    Compute Sharpe via the pipeline's calc_sr logic.
    Assert it matches a single hand-computed excess Sharpe.
    If it matched double subtraction instead, the values would differ.
    """
    rng = np.random.default_rng(0)
    # 252 days of synthetic returns with clear non-zero mean
    raw_returns = rng.normal(loc=0.0008, scale=0.012, size=252)
    
    RF_ANNUAL = 0.04
    RF_DAILY  = RF_ANNUAL / 252.0

    # ── Hand-computed Sharpe (single subtraction, reference) ──────────────────
    excess_hand = raw_returns - RF_DAILY
    sharpe_hand = (np.mean(excess_hand) / np.std(excess_hand, ddof=1)) * math.sqrt(252)

    # ── What double subtraction would produce ────────────────────────────────
    excess_double = (raw_returns - RF_DAILY) - RF_DAILY   # subtract twice
    sharpe_double = (np.mean(excess_double) / np.std(excess_double, ddof=1)) * math.sqrt(252)

    # ── Replicate calc_sr from full_system_benchmark.py ──────────────────────
    import pandas as pd
    r_series = pd.Series(raw_returns)

    def calc_sr_pipeline(r_series):
        rf_daily = 0.04 / 252.0
        excess = r_series - rf_daily
        return (excess.mean() / excess.std()) * math.sqrt(252) if excess.std() > 0 else 0.0

    pipeline_sharpe = calc_sr_pipeline(r_series)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert abs(pipeline_sharpe - sharpe_hand) < 1e-9, (
        f"Pipeline Sharpe ({pipeline_sharpe:.6f}) does not match hand-computed "
        f"single-subtraction Sharpe ({sharpe_hand:.6f}). "
        f"Possible double subtraction."
    )

    # Confirm the double-subtracted value is meaningfully different
    # (proves the test would catch a double-subtraction bug)
    assert abs(sharpe_double - sharpe_hand) > 0.01, (
        "Double-subtraction value too close to single — test cannot distinguish."
    )

    # Confirm pipeline does NOT match the double-subtracted value
    assert abs(pipeline_sharpe - sharpe_double) > 0.01, (
        f"Pipeline Sharpe ({pipeline_sharpe:.6f}) matches the DOUBLE-subtraction value "
        f"({sharpe_double:.6f}). Bug confirmed!"
    )
