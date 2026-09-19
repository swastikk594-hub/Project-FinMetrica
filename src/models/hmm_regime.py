"""
Gaussian Hidden Markov Model (HMM) & Macro Regime Switching Engine
===================================================================
PURPOSE:
  Model changing macroeconomic and market volatility regimes through a
  probabilistic Hidden Markov Model, outputting continuous posterior state
  probabilities rather than simplistic hard thresholds.

MATHEMATICAL FOUNDATION:
  Let S_t in {0, 1, ..., K-1} represent the unobserved market regime.
  Transition dynamics: P(S_t = j | S_{t-1} = i) = A_{ij}
  Emission distribution: Y_t | S_t = k ~ Normal(mu_k, Sigma_k)

  Features:
  - S&P 500 benchmark returns
  - Realized / Implied Volatility
  - 10Y-2Y Term Spread
  - High Yield Credit Spread

REGIME CLASSIFICATION:
  State 0: Low-Volatility Growth / Expansion
  State 1: Normal / Transition
  State 2: High-Volatility Stress / Contraction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)


class MacroHMMRegimeEngine:
    """
    Gaussian Hidden Markov Model for Macroeconomic Regime Detection.
    """

    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.gmm = GaussianMixture(
            n_components=n_states,
            covariance_type='full',
            random_state=random_state,
            max_iter=100
        )
        self.is_fitted = False
        self.transition_matrix: Optional[np.ndarray] = None
        self.state_order: List[int] = [] # Maps cluster index to sorted risk levels (0=Growth, 1=Normal, 2=Stress)

    def prepare_features(self, benchmark_returns: pd.Series, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Aligns benchmark returns and macro features for regime modeling."""
        df = pd.DataFrame({'mkt_return': benchmark_returns}).dropna()
        df['realized_vol'] = df['mkt_return'].rolling(21).std() * np.sqrt(252.0)

        if not macro_data.empty:
            macro_aligned = macro_data.reindex(df.index).ffill().bfill()
            for col in macro_aligned.columns:
                df[col] = macro_aligned[col]

        return df.dropna()

    def fit(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits the Gaussian regime model and estimates transition matrix.
        """
        if len(feature_df) < 60:
            logger.warning("Insufficient samples to fit Macro HMM.")
            return {'status': 'insufficient_data'}

        X = feature_df.values
        self.gmm.fit(X)
        self.is_fitted = True

        # Sort states by volatility / negative return to establish canonical ordering:
        # State 0 = Low Vol / Growth, State (K-1) = High Vol / Stress
        vol_col_idx = 1 if feature_df.shape[1] > 1 else 0
        state_vols = [self.gmm.means_[k][vol_col_idx] for k in range(self.n_states)]
        self.state_order = list(np.argsort(state_vols)) # lowest vol to highest vol

        # Compute empirical transition matrix from posterior state path
        posteriors = self.gmm.predict_proba(X)
        state_sequence = np.argmax(posteriors, axis=1)
        # Remap sequence to sorted state order
        remapped_seq = np.array([self.state_order.index(s) for s in state_sequence])

        trans_counts = np.zeros((self.n_states, self.n_states))
        for t in range(len(remapped_seq) - 1):
            i = remapped_seq[t]
            j = remapped_seq[t + 1]
            trans_counts[i, j] += 1.0

        # Normalize rows to create transition matrix
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transition_matrix = trans_counts / row_sums

        return {
            'n_states': self.n_states,
            'transition_matrix': self.transition_matrix.tolist(),
            'state_means': self.gmm.means_.tolist(),
            'status': 'fitted'
        }

    def predict_regime_probabilities(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes posterior state probabilities for each historical timestamp.
        """
        if not self.is_fitted or feature_df.empty:
            # Fallback: uniform probabilities
            cols = [f'prob_state_{i}' for i in range(self.n_states)]
            return pd.DataFrame(1.0 / self.n_states, index=feature_df.index, columns=cols)

        X = feature_df.values
        raw_posteriors = self.gmm.predict_proba(X)

        # Reorder columns to canonical order: [Growth, Normal, Stress]
        ordered_posteriors = raw_posteriors[:, self.state_order]
        state_names = ['prob_growth', 'prob_normal', 'prob_stress'] if self.n_states == 3 else [f'prob_state_{i}' for i in range(self.n_states)]

        prob_df = pd.DataFrame(ordered_posteriors, index=feature_df.index, columns=state_names[:self.n_states])
        prob_df['dominant_regime'] = prob_df.iloc[:, :self.n_states].idxmax(axis=1)
        return prob_df
