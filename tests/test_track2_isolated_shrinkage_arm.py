import pandas as pd
import numpy as np
import os
from research.experiments.full_system_benchmark import run_track2_study

def test_isolated_shrinkage_arm_runs_and_outputs(tmp_path):
    # Setup test data
    dates = pd.date_range("2020-01-01", "2023-01-01", freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    
    np.random.seed(42)
    prices = pd.DataFrame(
        np.exp(np.random.normal(0.0005, 0.015, size=(len(dates), len(tickers))).cumsum(axis=0)),
        index=dates,
        columns=tickers
    )
    
    out_dir = str(tmp_path)
    
    # Run the system benchmark
    df = run_track2_study(prices, output_dir=out_dir, fast_mode=True)
    
    assert df is not None
    assert not df.empty
    
    # Check that 'sr_lw_shrinkage_only' is present in the output
    assert 'sr_lw_shrinkage_only' in df.columns
    
    # Check that 5 methods are present in the significance test results
    sig_path = os.path.join(out_dir, "track2_significance_tests.csv")
    assert os.path.exists(sig_path)
    
    sig_df = pd.read_csv(sig_path)
    methods = pd.concat([sig_df["Method A"], sig_df["Method B"]]).unique()
    assert "LW-Shrinkage-Only Markowitz" in methods
    assert len(methods) == 5
    
    # 5 methods -> 10 pairwise comparisons
    assert len(sig_df) == 10
    
    # Check DSR uses 5 trials
    dsr_path = os.path.join(out_dir, "track2_dsr.csv")
    assert os.path.exists(dsr_path)
    
    dsr_df = pd.read_csv(dsr_path)
    assert dsr_df["n_trials"].iloc[0] == 5
