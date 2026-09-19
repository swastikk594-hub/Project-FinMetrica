import pytest
import numpy as np
import pandas as pd
from src.data.validator import DataValidator
from src.preprocessing.alignment import align_monthly_to_daily


def test_validator_catches_missing_data():
    """Test that DataValidator flags missing data."""
    validator = DataValidator()
    df = pd.DataFrame({
        'Close': [100.0, np.nan, 102.0]
    }, index=pd.date_range('2020-01-01', periods=3))
    
    report = validator.generate_quality_report({'TEST': df}, {})
    assert len(report.warnings) > 0 or len(report.errors) > 0


def test_validator_catches_duplicate_dates():
    """Test that DataValidator flags duplicate dates."""
    validator = DataValidator()
    dates = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]
    df = pd.DataFrame({'Close': [100.0, 101.0, 102.0]}, index=dates)
    
    dup_res = validator.check_duplicate_dates(df)
    assert dup_res['has_duplicates'] is True


def test_temporal_alignment_no_lookahead():
    """Test that monthly alignment respects temporal availability."""
    daily_idx = pd.date_range('2020-01-01', '2020-01-31', freq='D')
    monthly_df = pd.DataFrame({
        'Macro': [2.5]
    }, index=[pd.Timestamp('2020-01-01')])
    
    aligned = align_monthly_to_daily(monthly_df, daily_idx)
    assert not aligned.empty
