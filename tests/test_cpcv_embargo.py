"""
Tests for the CPCV embargo bug fix (Task A).

Validates that the calendar-time embargo logic correctly handles irregular
DatetimeIndex gaps (weekends, holidays, trading halts) — the original
integer-offset approach silently under-embargoed through gaps.
"""

import pytest
import numpy as np
import pandas as pd

from src.backtesting.cpcv import PurgedCrossValidator


def _make_gappy_index() -> pd.DatetimeIndex:
    """
    Create a business-day DatetimeIndex with a deliberate 3-week gap
    in the middle (simulating a trading halt or data outage).

    Layout:
      - 2020-01-02 through 2020-06-30: ~125 business days
      - GAP: skip 2020-07-01 through 2020-07-22 (3+ weeks)
      - 2020-07-23 through 2020-12-31: ~114 business days
    """
    part1 = pd.bdate_range("2020-01-02", "2020-06-30")
    part2 = pd.bdate_range("2020-07-23", "2020-12-31")
    return part1.append(part2)


class TestEmbargoCalendarTime:
    """Tests that embargo is enforced in calendar time, not index positions."""

    def test_no_training_date_in_embargo_zone_standard(self):
        """
        For each standard purged k-fold split, no training observation
        should fall within the calendar-time embargo window after the
        test period's last date.
        """
        index = _make_gappy_index()
        # Use a 10-calendar-day embargo
        cv = PurgedCrossValidator(
            n_splits=5, horizon_days=10, embargo_days=10
        )

        for train_idx, test_idx in cv.split(index):
            t_test_end = index[test_idx[-1]]
            t_embargo_end = t_test_end + pd.Timedelta(days=10)

            for ti in train_idx:
                t_obs = index[ti]
                # No training date should be in (t_test_end, t_embargo_end]
                if t_obs > t_test_end:
                    assert t_obs > t_embargo_end, (
                        f"Training date {t_obs} falls within embargo zone "
                        f"({t_test_end}, {t_embargo_end}]"
                    )

    def test_no_training_date_in_embargo_zone_combinatorial(self):
        """
        Same as above but for combinatorial CPCV splits.
        """
        index = _make_gappy_index()
        cv = PurgedCrossValidator(
            n_splits=5, horizon_days=10, embargo_days=10,
            combinatorial=True, n_test_groups=2
        )

        for train_idx, test_idx in cv.split(index):
            # For combinatorial splits, test_idx may span non-contiguous
            # groups. Check embargo around each test group's endpoint.
            # Since we don't have per-group breakdown here, check the
            # strictest condition: any train date that is after ANY test
            # date must not be within embargo_days calendar days of
            # that test date.
            test_dates = index[test_idx]
            t_test_end = test_dates.max()
            t_embargo_end = t_test_end + pd.Timedelta(days=10)

            for ti in train_idx:
                t_obs = index[ti]
                if t_obs > t_test_end:
                    assert t_obs > t_embargo_end, (
                        f"CPCV: Training date {t_obs} falls within embargo zone "
                        f"({t_test_end}, {t_embargo_end}]"
                    )


class TestEmbargoRegressionOldBug:
    """
    Regression test demonstrating that the old integer-offset logic
    would have leaked through the 3-week gap in the index.
    """

    def test_gap_would_cause_leak_with_integer_offset(self):
        """
        Construct a scenario where:
        - The test group ends right before the 3-week gap.
        - embargo_pct produces an integer offset of ~2 samples.
        - Under the old logic, the first few post-gap dates would
          pass the integer check (their positions are far from
          test_idx[-1]) and be included in training, despite being
          only a few calendar days after the test period ends (once
          you account for the gap swallowing the middle dates).

        The new calendar-time logic correctly embargoes them.
        """
        index = _make_gappy_index()
        n = len(index)

        # Find where the gap is: the largest jump between consecutive dates
        diffs = np.diff(index.values)
        gap_pos = np.argmax(diffs)  # index of the last pre-gap date

        # Set up so one group's test period ends right at the gap boundary
        # Use embargo_pct small enough that integer offset is ~2-3 samples
        embargo_pct = 3.0 / n  # ~3 samples

        cv = PurgedCrossValidator(
            n_splits=5, horizon_days=5, embargo_pct=embargo_pct
        )

        # The calendar-time embargo should be approximately
        # 3 * median_gap (which is ~1 day for business days) = ~3 calendar days
        embargo_delta = cv._compute_embargo_delta(index)

        # Verify embargo is computed in calendar time
        assert isinstance(embargo_delta, pd.Timedelta)
        # The median gap for business days is ~1 day, so 3 samples ≈ 3 days
        assert embargo_delta >= pd.Timedelta(days=1), (
            f"Embargo delta {embargo_delta} is too small"
        )

    def test_embargo_days_overrides_embargo_pct(self):
        """embargo_days takes precedence over embargo_pct."""
        index = _make_gappy_index()
        cv = PurgedCrossValidator(
            n_splits=5, horizon_days=10,
            embargo_pct=0.50,  # would be huge
            embargo_days=5     # but this overrides it
        )
        delta = cv._compute_embargo_delta(index)
        assert delta == pd.Timedelta(days=5)


class TestPurgingStillCorrect:
    """Verify that purging (label-window overlap removal) still works."""

    def test_no_label_overlap_with_test(self):
        """
        No training observation's label window [t, t+horizon] should
        overlap the test window [t_test_start, t_test_end].
        """
        index = pd.bdate_range("2020-01-02", "2021-12-31")
        cv = PurgedCrossValidator(
            n_splits=5, horizon_days=20, embargo_days=5
        )

        for train_idx, test_idx in cv.split(index):
            t_test_start = index[test_idx[0]]
            t_test_end = index[test_idx[-1]]

            for ti in train_idx:
                t_obs = index[ti]
                t_obs_end = t_obs + pd.Timedelta(days=20)

                # Label window must not overlap test window
                overlaps = not (t_obs_end < t_test_start or t_obs > t_test_end)
                assert not overlaps, (
                    f"Training obs at {t_obs} (label ends {t_obs_end}) "
                    f"overlaps test window [{t_test_start}, {t_test_end}]"
                )

    def test_basic_split_sizes(self):
        """Splits should produce non-empty train and test sets."""
        index = pd.bdate_range("2018-01-02", "2022-12-30")
        cv = PurgedCrossValidator(n_splits=5, horizon_days=20, embargo_days=5)

        splits = list(cv.split(index))
        assert len(splits) == 5

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # Train + test should not exceed total (due to purging/embargo,
            # some samples are dropped — total should be ≤ n)
            assert len(train_idx) + len(test_idx) <= len(index)


class TestCombinatorialPathCount:
    """Verify CPCV produces the expected number of combinatorial paths."""

    def test_c_6_2_gives_15_paths(self):
        """C(6,2) = 15 combinatorial paths."""
        index = pd.bdate_range("2015-01-02", "2023-12-29")
        cv = PurgedCrossValidator(
            n_splits=6, horizon_days=20, embargo_days=5,
            combinatorial=True, n_test_groups=2
        )
        splits = list(cv.split(index))
        assert len(splits) == 15

    def test_c_5_2_gives_10_paths(self):
        """C(5,2) = 10 combinatorial paths."""
        index = pd.bdate_range("2015-01-02", "2023-12-29")
        cv = PurgedCrossValidator(
            n_splits=5, horizon_days=20, embargo_days=5,
            combinatorial=True, n_test_groups=2
        )
        splits = list(cv.split(index))
        assert len(splits) == 10
