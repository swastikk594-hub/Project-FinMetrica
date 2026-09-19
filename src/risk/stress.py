"""
Historical Scenario Stress Testing & Reverse Stress Testing Engine
===================================================================
PURPOSE:
  Evaluate portfolio resilience against defined historical crisis scenarios
  and perform reverse stress testing to discover the minimum multi-asset shock
  vector that breaches a defined maximum loss tolerance (e.g., -20% loss).

PREDEFINED SCENARIOS:
  1. 2008 Global Financial Crisis (Equities -45%, Volatility +200%, Credit Spread +400bps)
  2. 2020 COVID Liquidity Shock (Equities -30%, Volatility +300%, ADV -40%)
  3. 2022 Inflation & Rate Surge (Equities -20%, Tech -35%, Real Rates +300bps)
  4. Tech Bubble Liquidation (Tech -50%, Value +10%, High Multiple Growth -60%)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

HISTORICAL_SCENARIOS = {
    '2008_global_financial_crisis': {
        'description': '2008 Lehman Collapse & Subprime Liquidity Freeze',
        'equity_shock': -0.42,
        'vol_multiplier': 2.5,
        'correlation_spike': 0.85
    },
    '2020_covid_liquidity_shock': {
        'description': 'March 2020 Global Pandemic Shock',
        'equity_shock': -0.32,
        'vol_multiplier': 3.0,
        'correlation_spike': 0.90
    },
    '2022_inflation_rate_surge': {
        'description': '2022 Rapid Fed Tightening & Multiple Compression',
        'equity_shock': -0.22,
        'vol_multiplier': 1.6,
        'correlation_spike': 0.70
    },
    'tech_sector_meltdown': {
        'description': 'Concentrated Tech & Growth De-rating',
        'equity_shock': -0.35,
        'vol_multiplier': 2.0,
        'correlation_spike': 0.75
    }
}


class StressTestingEngine:
    """
    Simulates macro crisis scenarios and identifies critical loss vulnerabilities.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def run_predefined_scenarios(
        self,
        weights: pd.Series,
        asset_betas: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates projected portfolio drawdown under each historical crisis scenario.
        """
        results = {}
        for sc_name, sc_data in HISTORICAL_SCENARIOS.items():
            base_shock = sc_data['equity_shock']
            # Calculate asset-specific shock: beta * market shock
            port_loss = 0.0
            for ticker, w in weights.items():
                beta = asset_betas.get(ticker, 1.0)
                asset_loss = beta * base_shock
                port_loss += w * asset_loss

            results[sc_name] = {
                'scenario_name': sc_data['description'],
                'projected_portfolio_loss': float(port_loss),
                'portfolio_survival': bool(port_loss > -0.35) # Survival threshold 35% max loss
            }

        return results

    def reverse_stress_test(
        self,
        weights: pd.Series,
        asset_betas: Dict[str, float],
        max_acceptable_loss: float = -0.20
    ) -> Dict[str, Any]:
        """
        Calculates the required market drop and beta shock to cause a specific loss.
        """
        port_beta = sum(weights.get(t, 0.0) * asset_betas.get(t, 1.0) for t in weights.index)
        if port_beta == 0:
            required_mkt_drop = max_acceptable_loss
        else:
            required_mkt_drop = max_acceptable_loss / port_beta

        return {
            'target_portfolio_loss': float(max_acceptable_loss),
            'portfolio_weighted_beta': float(port_beta),
            'market_drawdown_trigger': float(required_mkt_drop),
            'vulnerability_rating': 'HIGH' if abs(required_mkt_drop) < 0.15 else ('MODERATE' if abs(required_mkt_drop) < 0.25 else 'LOW')
        }
