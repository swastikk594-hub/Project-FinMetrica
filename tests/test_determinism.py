import os
import numpy as np
import pandas as pd
from research.experiments.covariance_regularization_study import run_track1_study
from src.data.config_loader import get_config

def test_track1_determinism(tmp_path):
    # Set a fixed seed
    seed = 42
    np.random.seed(seed)
    
    # Mock config to make sure global seed is respected
    config = get_config()
    if 'system' not in config:
        config['system'] = {}
    config['system']['random_seed'] = seed
    config['random_seed'] = seed
    
    # Generate synthetic fixture data
    dates = pd.date_range("2020-01-01", "2023-01-01", freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    
    prices = pd.DataFrame(
        np.exp(np.random.normal(0.0005, 0.015, size=(len(dates), len(tickers))).cumsum(axis=0)),
        index=dates,
        columns=tickers
    )
    
    # First run
    dir1 = tmp_path / "run1"
    dir1.mkdir()
    
    # reset seed before run to ensure identical generation internally if any
    np.random.seed(seed)
    run_track1_study(prices, output_dir=str(dir1))
    df1 = pd.read_csv(dir1 / "track1_estimator_instability.csv")
    
    # Second run
    dir2 = tmp_path / "run2"
    dir2.mkdir()
    
    np.random.seed(seed)
    run_track1_study(prices, output_dir=str(dir2))
    df2 = pd.read_csv(dir2 / "track1_estimator_instability.csv")
    
    # Assert dataframes are exactly equal
    pd.testing.assert_frame_equal(df1, df2)
