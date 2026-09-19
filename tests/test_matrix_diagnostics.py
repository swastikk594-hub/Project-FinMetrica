"""
Tests for the matrix diagnostics utility (Task B).
"""

import pytest
import numpy as np
import pandas as pd

from src.risk.matrix_diagnostics import (
    condition_number,
    eigenvalue_spectrum,
    effective_rank,
    participation_ratio
)


class TestMatrixDiagnostics:
    def test_identity_matrix(self):
        """Identity matrix should have condition number 1, rank N, and PR N."""
        N = 10
        I = np.eye(N)
        
        # Condition number should be 1.0 (max eig / min eig = 1/1)
        np.testing.assert_almost_equal(condition_number(I), 1.0)
        
        # All eigenvalues should be 1.0
        evals = eigenvalue_spectrum(I)
        assert len(evals) == N
        np.testing.assert_array_almost_equal(evals, np.ones(N))
        
        # Effective rank should be exactly N
        np.testing.assert_almost_equal(effective_rank(I), float(N))
        
        # Participation ratio should be N
        # PR = (sum(1))^2 / sum(1^2) = N^2 / N = N
        np.testing.assert_almost_equal(participation_ratio(I), float(N))

    def test_ill_conditioned_matrix(self):
        """Matrix with near-collinear rows/cols should have high cond number and low rank."""
        N = 5
        # Create a matrix where rank is effectively 2
        # F1 and F2 are two dominant factors
        F = np.random.randn(N, 2)
        # Add tiny noise so it's not perfectly singular
        noise = np.eye(N) * 1e-6
        Sigma = F @ F.T + noise
        
        # High condition number due to noise
        cond = condition_number(Sigma)
        assert cond > 1e4
        
        # Effective rank should be close to 2
        erank = effective_rank(Sigma)
        assert erank < 3.0
        
        # PR should be around 2
        pr = participation_ratio(Sigma)
        assert pr < 3.0

    def test_random_well_conditioned_matrix(self):
        """Random well-conditioned matrix should have sensible values."""
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        N = 5
        # A diagonally dominant matrix is well conditioned
        A = np.random.randn(N, N) * 0.1
        Sigma = A @ A.T + np.eye(N)
        
        cond = condition_number(Sigma)
        assert cond > 1.0
        assert cond < 100.0  # Well conditioned
        
        evals = eigenvalue_spectrum(Sigma)
        assert len(evals) == N
        # Should be sorted descending
        assert np.all(np.diff(evals) <= 0)
        
        erank = effective_rank(Sigma)
        assert 1.0 <= erank <= N
        
        pr = participation_ratio(Sigma)
        assert 1.0 <= pr <= N

    def test_dataframe_input(self):
        """Utility should accept pandas DataFrames transparently."""
        N = 3
        I = pd.DataFrame(np.eye(N), columns=['A', 'B', 'C'], index=['A', 'B', 'C'])
        
        np.testing.assert_almost_equal(condition_number(I), 1.0)
        np.testing.assert_almost_equal(effective_rank(I), float(N))
        np.testing.assert_almost_equal(participation_ratio(I), float(N))

    def test_zero_matrix(self):
        """Edge case: zero matrix or very small values."""
        Z = np.zeros((3, 3))
        # rank and PR should handle it gracefully without division by zero
        assert effective_rank(Z) == 0.0
        assert participation_ratio(Z) == 0.0
