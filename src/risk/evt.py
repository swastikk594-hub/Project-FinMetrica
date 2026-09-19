"""
Extreme Value Theory (EVT) & Generalized Pareto Distribution (GPD) Engine
===========================================================================
PURPOSE:
  Model extreme negative return tails beyond the maximum historical drawdown
  using the Peaks Over Threshold (POT) approach.

MATHEMATICAL FOUNDATION:
  Balkema-de Haan-Pickands Theorem:
  For a high threshold u, the conditional excess loss Y = X - u given X > u
  asymptotically follows a Generalized Pareto Distribution:
  G_{xi, beta}(y) = 1 - (1 + xi * y / beta)^{-1/xi}

  - xi > 0: Heavy-tailed (Frechet-type, standard in finance)
  - xi = 0: Exponential tail
  - xi < 0: Bounded tail
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EVTTailEstimator:
    """
    Fits Generalized Pareto Distribution to threshold exceedances to estimate
    extreme out-of-sample Value-at-Risk and Expected Shortfall.
    """

    def __init__(self, threshold_quantile: float = 0.95):
        self.threshold_quantile = threshold_quantile

    def fit_and_estimate_tail(
        self,
        losses: pd.Series,
        confidence_level: float = 0.99
    ) -> Dict[str, Any]:
        """
        Fits GPD to positive losses (L = -r) exceeding threshold u.
        """
        clean_losses = losses.dropna()
        if len(clean_losses) < 100:
            return {
                'status': 'insufficient_data',
                'evt_var': float(clean_losses.quantile(confidence_level)) if len(clean_losses) > 0 else 0.0,
                'evt_cvar': float(clean_losses.quantile(confidence_level) * 1.25) if len(clean_losses) > 0 else 0.0
            }

        u = float(np.percentile(clean_losses, self.threshold_quantile * 100.0))
        exceedances = clean_losses[clean_losses > u] - u
        n_total = len(clean_losses)
        n_exceed = len(exceedances)

        if n_exceed < 10:
            return {
                'status': 'insufficient_exceedances',
                'evt_var': float(clean_losses.quantile(confidence_level)),
                'evt_cvar': float(clean_losses.quantile(confidence_level) * 1.2)
            }

        try:
            # Fit GPD: scipy genpareto (c = xi, scale = beta)
            c_shape, loc, scale = stats.genpareto.fit(exceedances, floc=0.0)
            xi = float(c_shape)
            beta = float(scale)

            # EVT VaR formula: VaR_alpha = u + (beta / xi) * [ ((n / n_u) * (1 - alpha))^(-xi) - 1 ]
            p = 1.0 - confidence_level
            ratio = (n_total / n_exceed) * p
            if xi != 0 and ratio > 0:
                evt_var = u + (beta / xi) * ((ratio ** (-xi)) - 1.0)
                # EVT Expected Shortfall formula:
                # ES_alpha = (VaR_alpha + beta - xi * u) / (1 - xi)
                if xi < 1.0:
                    evt_cvar = (evt_var + beta - xi * u) / (1.0 - xi)
                else:
                    evt_cvar = evt_var * 1.5
            else:
                evt_var = float(clean_losses.quantile(confidence_level))
                evt_cvar = float(clean_losses[clean_losses >= evt_var].mean())

            return {
                'status': 'success',
                'threshold_u': u,
                'shape_xi': xi,
                'scale_beta': beta,
                'evt_var': float(evt_var),
                'evt_cvar': float(evt_cvar),
                'n_exceedances': n_exceed
            }
        except Exception as e:
            logger.debug(f"EVT fit failed: {e}")
            return {
                'status': 'fit_error',
                'evt_var': float(clean_losses.quantile(confidence_level)),
                'evt_cvar': float(clean_losses.quantile(confidence_level) * 1.2)
            }
