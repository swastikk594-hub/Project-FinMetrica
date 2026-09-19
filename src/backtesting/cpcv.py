"""
Combinatorial Purged Cross-Validation (CPCV), Purging & Embargo Engine
======================================================================
PURPOSE:
  Prevent data leakage and backtest overfitting across overlapping label horizons.

MATHEMATICAL FOUNDATION (Marcos Lopez de Prado, 2018):
  Standard k-fold cross-validation fails in finance because:
  1. Financial time series exhibit serial correlation.
  2. Forward-looking labels (e.g., 20-day return y_t = r_{t -> t+20}) span
     multiple future days. If sample t is in training and sample t+5 is in test,
     information from the test set leaks into the training set (leakage).

PURGING:
  Drop all training observations whose label evaluation window [t, t + horizon]
  overlaps with any test observation's window [t_test, t_test + horizon].

EMBARGO:
  After the end of a test window, discard an additional embargo window
  (e.g., 5 to 20 days) from training to account for auto-regressive memory.

  CALENDAR-TIME EMBARGO (Bug fix, September 2026):
    The original implementation computed the embargo window as an integer
    index-position offset: `embargo_samples = int(n_samples * embargo_pct)`,
    then checked `idx_val <= test_idx[-1] + embargo_samples`. This compares
    raw integer positions, silently assuming the underlying DatetimeIndex has
    uniform spacing with no gaps. On real daily market data, weekends,
    holidays, and trading halts create irregular gaps in the index. An
    integer offset of "20 samples" corresponds to different true calendar
    durations depending on how many non-trading days fall inside the window,
    causing the embargo to under-protect (leakage through gaps) or
    over-protect (wasted training data) inconsistently across the dataset.

    The fix converts all embargo comparisons to calendar time using
    pd.Timestamp / pd.Timedelta arithmetic, consistent with how purging
    already operates. A new `embargo_days` parameter allows specifying the
    embargo directly in calendar days. When not provided, `embargo_pct` is
    converted to calendar time using the actual median spacing of the index.

COMBINATORIAL PATH GENERATION:
  Given N groups, choose k test groups per split, yielding (N choose k) splits
  and generating multiple unique backtest paths for robust distribution estimation.
"""

import itertools
import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PurgedCrossValidator:
    """
    Implements Purged and Embargoed Time-Series Cross Validation.

    Parameters
    ----------
    n_splits : int
        Number of groups to partition the data into.
    horizon_days : int
        Forward label horizon in calendar days (used for purging).
    embargo_pct : float
        Fraction of total samples to embargo after each test window.
        Converted to calendar time at split-time using the actual median
        spacing of the datetime index.
    embargo_days : int, optional
        If provided, overrides ``embargo_pct`` and defines the embargo
        as a fixed calendar-time window in days. This is the recommended
        parameter for production use because it is not sensitive to index
        frequency or irregularity.
    combinatorial : bool
        If True, use Combinatorial Purged K-Fold (CPCV).
    n_test_groups : int
        Number of groups to hold out as test in each combinatorial split.
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon_days: int = 20,
        embargo_pct: float = 0.01,
        embargo_days: Optional[int] = None,
        combinatorial: bool = False,
        n_test_groups: int = 2
    ):
        self.n_splits = n_splits
        self.horizon_days = horizon_days
        self.embargo_pct = embargo_pct
        self.embargo_days = embargo_days
        self.combinatorial = combinatorial
        self.n_test_groups = n_test_groups

    def _compute_embargo_delta(self, index: pd.DatetimeIndex) -> pd.Timedelta:
        """
        Compute the embargo window as a calendar-time Timedelta.

        If ``embargo_days`` was provided at construction, use it directly.
        Otherwise, convert ``embargo_pct`` to calendar time by estimating
        the median inter-observation spacing of the actual index, so the
        result is consistent regardless of data frequency or gaps.
        """
        if self.embargo_days is not None:
            return pd.Timedelta(days=self.embargo_days)

        n_samples = len(index)
        embargo_n_samples = int(n_samples * self.embargo_pct)

        if n_samples < 2 or embargo_n_samples < 1:
            return pd.Timedelta(days=0)

        # Use the actual median spacing to convert sample count to calendar time
        diffs = np.diff(index.values)
        median_gap = pd.Timedelta(np.median(diffs))
        return median_gap * embargo_n_samples

    def split(
        self,
        index: pd.DatetimeIndex
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) with purging and embargoing applied.

        All embargo checks use calendar-time comparisons (pd.Timestamp /
        pd.Timedelta) rather than integer index-position offsets, ensuring
        consistent protection regardless of trading-calendar gaps.
        """
        n_samples = len(index)
        if n_samples < 20:
            yield np.arange(n_samples), np.arange(n_samples)
            return

        embargo_delta = self._compute_embargo_delta(index)
        group_size = n_samples // self.n_splits
        groups = [
            np.arange(i * group_size, (i + 1) * group_size if i < self.n_splits - 1 else n_samples)
            for i in range(self.n_splits)
        ]

        if not self.combinatorial:
            # Standard Purged K-Fold
            for i in range(self.n_splits):
                test_idx = groups[i]
                t_test_start = index[test_idx[0]]
                t_test_end = index[test_idx[-1]]
                t_embargo_end = t_test_end + embargo_delta

                # Purge training: remove samples whose label window overlaps test
                # or whose timestamp falls in the post-test embargo zone
                train_idx_list = []
                for j in range(self.n_splits):
                    if i == j:
                        continue
                    grp_idx = groups[j]
                    for idx_val in grp_idx:
                        t_obs = index[idx_val]
                        # Label spans [t_obs, t_obs + horizon_days]
                        t_obs_end = t_obs + pd.Timedelta(days=self.horizon_days)

                        # Check purge: label window overlaps test window
                        overlaps = not (t_obs_end < t_test_start or t_obs > t_test_end)

                        # Check embargo: observation falls in the post-test
                        # embargo zone [t_test_end, t_test_end + embargo_delta]
                        # (calendar-time comparison, not integer offset)
                        is_embargoed = (t_obs > t_test_end) and (t_obs <= t_embargo_end)

                        if not overlaps and not is_embargoed:
                            train_idx_list.append(idx_val)

                yield np.array(train_idx_list, dtype=int), test_idx
        else:
            # Combinatorial Purged K-Fold (CPCV)
            group_indices = list(range(self.n_splits))
            for test_comb in itertools.combinations(group_indices, self.n_test_groups):
                test_idx = np.concatenate([groups[g] for g in test_comb])
                test_ranges = [
                    (index[groups[g][0]], index[groups[g][-1]])
                    for g in test_comb
                ]

                train_idx_list = []
                for g in group_indices:
                    if g in test_comb:
                        continue
                    for idx_val in groups[g]:
                        t_obs = index[idx_val]
                        t_obs_end = t_obs + pd.Timedelta(days=self.horizon_days)

                        # Purge & Embargo check against all active test groups
                        purged = False
                        for (t_start, t_end) in test_ranges:
                            t_embargo_end = t_end + embargo_delta

                            # Purge: label window overlaps test window
                            overlaps = not (t_obs_end < t_start or t_obs > t_end)

                            # Embargo: calendar-time check
                            is_embargoed = (t_obs > t_end) and (t_obs <= t_embargo_end)

                            if overlaps or is_embargoed:
                                purged = True
                                break

                        if not purged:
                            train_idx_list.append(idx_val)

                yield np.array(train_idx_list, dtype=int), test_idx
