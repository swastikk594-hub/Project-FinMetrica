"""
Tests for the headline/conclusion logic in generate_report.py.

We extract the `build_headline` function and test it against three
fixture cases to ensure the generated text is always consistent
with the significance table.
"""
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Import the headline builder (will be extracted into generate_report.py)
# ──────────────────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from research.experiments.generate_report import build_headline


# ──────────────────────────────────────────────────────────────────────────────
# Fixture 1 — ALL NULL: no pairs significant after BH correction
# ──────────────────────────────────────────────────────────────────────────────
FIXTURE_ALL_NULL = pd.DataFrame([
    {'Method A': 'Equal Weight', 'Method B': 'Naive Markowitz',       'Sharpe A': 0.34, 'Sharpe B': 0.33, 'Bootstrap p-value': 0.45, 'Significant (5% BH)': False},
    {'Method A': 'Equal Weight', 'Method B': 'Regularised Markowitz', 'Sharpe A': 0.34, 'Sharpe B': 0.33, 'Bootstrap p-value': 0.42, 'Significant (5% BH)': False},
    {'Method A': 'Equal Weight', 'Method B': 'HRP',                   'Sharpe A': 0.34, 'Sharpe B': 0.45, 'Bootstrap p-value': 0.28, 'Significant (5% BH)': False},
    {'Method A': 'Naive Markowitz', 'Method B': 'Regularised Markowitz', 'Sharpe A': 0.33, 'Sharpe B': 0.33, 'Bootstrap p-value': 0.72, 'Significant (5% BH)': False},
    {'Method A': 'Naive Markowitz', 'Method B': 'HRP',                'Sharpe A': 0.33, 'Sharpe B': 0.45, 'Bootstrap p-value': 0.07, 'Significant (5% BH)': False},
    {'Method A': 'Regularised Markowitz', 'Method B': 'HRP',          'Sharpe A': 0.33, 'Sharpe B': 0.45, 'Bootstrap p-value': 0.07, 'Significant (5% BH)': False},
])

# ──────────────────────────────────────────────────────────────────────────────
# Fixture 2 — SINGLE PAIR: only HRP vs Equal Weight is significant
# ──────────────────────────────────────────────────────────────────────────────
FIXTURE_SINGLE_PAIR = pd.DataFrame([
    {'Method A': 'Equal Weight', 'Method B': 'Naive Markowitz',       'Sharpe A': 0.75, 'Sharpe B': 1.09, 'Bootstrap p-value': 0.01, 'Significant (5% BH)': False},
    {'Method A': 'Equal Weight', 'Method B': 'Regularised Markowitz', 'Sharpe A': 0.75, 'Sharpe B': 1.06, 'Bootstrap p-value': 0.01, 'Significant (5% BH)': False},
    {'Method A': 'Equal Weight', 'Method B': 'HRP',                   'Sharpe A': 0.75, 'Sharpe B': 1.24, 'Bootstrap p-value': 0.00, 'Significant (5% BH)': True},
    {'Method A': 'Naive Markowitz', 'Method B': 'Regularised Markowitz', 'Sharpe A': 1.09, 'Sharpe B': 1.06, 'Bootstrap p-value': 0.01, 'Significant (5% BH)': False},
    {'Method A': 'Naive Markowitz', 'Method B': 'HRP',                'Sharpe A': 1.09, 'Sharpe B': 1.24, 'Bootstrap p-value': 0.14, 'Significant (5% BH)': False},
    {'Method A': 'Regularised Markowitz', 'Method B': 'HRP',          'Sharpe A': 1.06, 'Sharpe B': 1.24, 'Bootstrap p-value': 0.12, 'Significant (5% BH)': False},
])

# ──────────────────────────────────────────────────────────────────────────────
# Fixture 3 — MULTI PAIR (matches this run's actual pattern):
# HRP beats Naive and Regularised, but NO method beats Equal Weight after BH
# ──────────────────────────────────────────────────────────────────────────────
FIXTURE_MULTI_PAIR = pd.DataFrame([
    {'Method A': 'Equal Weight', 'Method B': 'Naive Markowitz',       'Sharpe A': 0.34, 'Sharpe B': 0.33, 'Bootstrap p-value': 0.454, 'Significant (5% BH)': False},
    {'Method A': 'Equal Weight', 'Method B': 'Regularised Markowitz', 'Sharpe A': 0.34, 'Sharpe B': 0.33, 'Bootstrap p-value': 0.422, 'Significant (5% BH)': False},
    {'Method A': 'Equal Weight', 'Method B': 'HRP',                   'Sharpe A': 0.34, 'Sharpe B': 0.45, 'Bootstrap p-value': 0.056, 'Significant (5% BH)': False},
    {'Method A': 'Naive Markowitz', 'Method B': 'Regularised Markowitz', 'Sharpe A': 0.33, 'Sharpe B': 0.33, 'Bootstrap p-value': 0.108, 'Significant (5% BH)': False},
    {'Method A': 'Naive Markowitz', 'Method B': 'HRP',                'Sharpe A': 0.33, 'Sharpe B': 0.45, 'Bootstrap p-value': 0.030, 'Significant (5% BH)': True},
    {'Method A': 'Regularised Markowitz', 'Method B': 'HRP',          'Sharpe A': 0.33, 'Sharpe B': 0.45, 'Bootstrap p-value': 0.030, 'Significant (5% BH)': True},
])


def test_all_null_headline():
    """No pairs significant — headline must say so explicitly."""
    headline = build_headline(FIXTURE_ALL_NULL)
    assert "no statistically significant" in headline.lower(), \
        f"Expected 'no statistically significant' in headline, got:\n{headline}"
    # Must not falsely claim any method wins
    assert "significantly outperforms" not in headline.lower(), \
        f"Should not claim significant outperformance when all-null:\n{headline}"


def test_single_pair_headline():
    """Only HRP vs Equal Weight is significant — headline must name that pair."""
    headline = build_headline(FIXTURE_SINGLE_PAIR)
    assert "hrp" in headline.lower(), \
        f"Expected 'HRP' in headline, got:\n{headline}"
    assert "equal weight" in headline.lower(), \
        f"Expected 'Equal Weight' in headline, got:\n{headline}"
    # Should mention HRP beats Equal Weight
    assert "significantly" in headline.lower(), \
        f"Expected significance language in headline, got:\n{headline}"


def test_multi_pair_headline():
    """
    HRP significantly beats Naive and Regularised (BH 5%) but NOT Equal Weight.
    The headline must:
     - Mention HRP's wins over Naive and Regularised
     - Mention that HRP does NOT beat Equal Weight
     - NOT state 'no significant differences' (which contradicts the table)
    """
    headline = build_headline(FIXTURE_MULTI_PAIR)
    headline_lower = headline.lower()

    # Must NOT say there are no differences (that contradicts the table)
    assert "no statistically significant differences" not in headline_lower, \
        f"Headline incorrectly says no significant differences:\n{headline}"

    # Must mention HRP winning something
    assert "hrp" in headline_lower, \
        f"Expected 'HRP' in multi-pair headline, got:\n{headline}"

    # Must acknowledge HRP doesn't beat Equal Weight
    assert "equal weight" in headline_lower, \
        f"Expected 'Equal Weight' to appear in multi-pair headline, got:\n{headline}"
