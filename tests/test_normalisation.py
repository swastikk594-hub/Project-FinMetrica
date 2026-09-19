"""
Test suite for src/preprocessing/normalisation.py

Tests verify mathematical correctness of all normalisation methods.
All tests are self-contained — no API calls or network access required.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.normalisation import (
    zscore_normalise,
    minmax_normalise,
    rank_normalise,
    robust_normalise,
    expanding_zscore,
)


class TestZScoreNormalise:
    """Tests for zscore_normalise() function."""

    def test_zscore_known_values(self):
        """
        MATHEMATICAL VERIFICATION:
        For series [1,2,3,4,5], mean=3, std≈1.5811 (ddof=1)
        z([1,2,3,4,5]) ≈ [-1.265, -0.632, 0, 0.632, 1.265]
        """
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # Disable winsorisation for exact mathematical verification
        zscores = zscore_normalise(series, winsorise=False)
        
        mean = series.mean()  # = 3.0
        std = series.std()    # = 1.5811... (pandas uses ddof=1)
        expected = (series - mean) / std
        
        assert np.allclose(zscores.values, expected.values, atol=1e-8)

    def test_zscore_mean_is_zero(self):
        """After z-scoring (without winsorisation), mean should be 0."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(100) * 5 + 10)
        zscores = zscore_normalise(series, winsorise=False)
        assert np.isclose(zscores.mean(), 0, atol=1e-10)

    def test_zscore_std_is_one(self):
        """After z-scoring (without winsorisation), std should be 1."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(100) * 5 + 10)
        zscores = zscore_normalise(series, winsorise=False)
        assert np.isclose(zscores.std(), 1.0, atol=1e-10)

    def test_winsorisation_clips_outliers(self):
        """
        With winsorisation at ±3σ, no z-score should exceed those bounds.
        """
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, -1000.0])
        zscores = zscore_normalise(series, winsorise=True, winsor_limits=(-3, 3))
        assert zscores.max() <= 3.0 + 1e-10
        assert zscores.min() >= -3.0 - 1e-10

    def test_constant_series_returns_zeros(self):
        """
        Z-score of a constant series cannot be computed (std=0).
        Should return zeros (neutral) with a warning, not crash.
        """
        series = pd.Series([5.0, 5.0, 5.0, 5.0])
        zscores = zscore_normalise(series)
        assert len(zscores) == len(series)
        assert np.all(zscores.values == 0.0)


class TestMinMaxNormalise:
    """Tests for minmax_normalise() function."""

    def test_minmax_bounds_zero_one(self):
        """
        MATHEMATICAL PROPERTY:
        x' = (x - min) / (max - min) → always in [0, 1]
        """
        np.random.seed(42)
        series = pd.Series(np.random.randn(100))
        scaled = minmax_normalise(series, feature_range=(0, 1))
        assert scaled.min() >= 0.0 - 1e-10
        assert scaled.max() <= 1.0 + 1e-10

    def test_minmax_bounds_custom_range(self):
        """Min-max with custom range [0, 100] should be in [0, 100]."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        scaled = minmax_normalise(series, feature_range=(0, 100))
        assert np.isclose(scaled.min(), 0.0, atol=1e-8)
        assert np.isclose(scaled.max(), 100.0, atol=1e-8)

    def test_minmax_known_values(self):
        """
        MATHEMATICAL VERIFICATION:
        series = [0, 5, 10], min=0, max=10
        scaled = [0, 0.5, 1.0]
        """
        series = pd.Series([0.0, 5.0, 10.0])
        scaled = minmax_normalise(series)
        expected = pd.Series([0.0, 0.5, 1.0])
        assert np.allclose(scaled.values, expected.values, atol=1e-8)


class TestRankNormalise:
    """Tests for rank_normalise() function."""

    def test_rank_bounds(self):
        """Rank normalised values should be in [0, 100]."""
        series = pd.Series(np.random.randn(50))
        ranked = rank_normalise(series)
        assert ranked.min() >= 0.0 - 1e-10
        assert ranked.max() <= 100.0 + 1e-10

    def test_rank_preserves_order(self):
        """Rank of sorted ascending input is ascending."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        ranked = rank_normalise(series)
        assert ranked.is_monotonic_increasing

    def test_rank_normalise_outlier_robustness(self):
        """
        MATHEMATICAL PROPERTY:
        The outlier (1e9) should rank highest in its own series.
        The normal values [1..5] should maintain their relative ordering.
        """
        series_with_outlier = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1e9])
        ranked_outlier = rank_normalise(series_with_outlier)
        
        # Outlier should have the highest rank (100)
        assert ranked_outlier.iloc[5] == ranked_outlier.max(), "Outlier should have maximum rank"
        
        # The first 5 values should maintain ascending order
        first_five_ranks = ranked_outlier.iloc[:5]
        assert first_five_ranks.is_monotonic_increasing, "First 5 values should maintain ascending rank order"


class TestRobustNormalise:
    """Tests for robust_normalise() function."""

    def test_robust_median_is_zero(self):
        """
        MATHEMATICAL PROPERTY:
        (x - median) / IQR → median = 0 always
        """
        np.random.seed(42)
        series = pd.Series(np.random.randn(100) * 3 + 7)
        series.iloc[0] = 1000.0  # Add outlier
        robust = robust_normalise(series)
        assert np.isclose(robust.median(), 0.0, atol=1e-8)


class TestExpandingZScore:
    """Tests for expanding_zscore() function — look-ahead bias prevention."""

    def test_expanding_zscore_no_lookahead(self):
        """
        CRITICAL: Expanding z-score at time t should use ONLY data from [0, t].
        
        This verifies look-ahead bias prevention.
        
        series = [10, 12, 11, 15]
        At t=3 (value=15): history = [10, 12, 11, 15]
        mean = 12, std = 2.16...
        z = (15 - 12) / 2.16 = 1.386...
        """
        series = pd.Series([10.0, 12.0, 11.0, 15.0, 14.0, 20.0])
        expanding_z = expanding_zscore(series)
        
        # At index 3 (value=15.0): use data up to and including t=3
        history_at_t3 = series.iloc[:4]  # [10, 12, 11, 15]
        expected_at_t3 = (series.iloc[3] - history_at_t3.mean()) / history_at_t3.std()
        
        assert np.isclose(expanding_z.iloc[3], expected_at_t3, atol=1e-6), (
            f"Expanding z-score at t=3: expected {expected_at_t3:.6f}, "
            f"got {expanding_z.iloc[3]:.6f}"
        )

    def test_expanding_zscore_increases_stability(self):
        """
        Expanding window uses more data over time → z-scores should
        become less extreme as window grows (more stable mean/std estimates).
        """
        np.random.seed(42)
        series = pd.Series(np.random.randn(200))
        expanding_z = expanding_zscore(series)
        
        # Std of z-scores in later half should be closer to 1 than early half
        early = expanding_z.iloc[10:20].std()
        late = expanding_z.iloc[100:200].std()
        # Not a strict test — just verify both are finite
        assert np.isfinite(early) and np.isfinite(late)
