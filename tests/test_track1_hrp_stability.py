import numpy as np
import pandas as pd
from research.experiments.covariance_regularization_study import run_track1_study

def test_track1_hrp_stability_smoke_test(tmp_path):
    """
    Smoke test to verify that run_track1_study now returns HRP turnover and OOS Sharpe.
    """
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    returns = np.random.normal(0.0005, 0.015, size=(200, 5))
    prices = pd.DataFrame(np.exp(np.cumsum(returns, axis=0)), index=dates, columns=['A', 'B', 'C', 'D', 'E'])

    # Run the study
    df = run_track1_study(prices, horizon_days=10, output_dir=str(tmp_path))
    
    # Assert HRP columns exist
    assert 'turnover_hrp' in df.columns
    assert 'oos_sharpe_hrp' in df.columns
    assert 'weight_diff_l2_hrp_sample' in df.columns
    assert 'weight_diff_l2_hrp_lw' in df.columns
    
    # Assert they contain valid numbers
    assert df['turnover_hrp'].notna().all()
    assert df['oos_sharpe_hrp'].notna().all()
