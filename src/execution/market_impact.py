"""
Market Microstructure, Non-Linear Market Impact & Execution Cost Engine
========================================================================
PURPOSE:
  Model institutional transaction costs, bid-ask spread, non-linear slippage,
  and liquidity participation constraints.

MATHEMATICAL FOUNDATION:
  1. Square-Root Law of Market Impact (Almgren-Chriss, Bouchaud et al.):
     ΔP / P = Y * σ_daily * sqrt(Q / ADV)
     where:
       Q = order traded value ($)
       ADV = Average Daily Traded Value ($) = Close * Volume
       σ_daily = daily price volatility
       Y = empirical calibration coefficient (typically 0.1 to 0.7)

  2. Total Transaction Cost (TC):
     TC = Commission + (0.5 * BidAskSpread) + MarketImpact + BorrowCost
     
  3. Liquidity Participation Constraint:
     Max Traded Volume Q_max = α * ADV (e.g., α = 0.05 max 5% of daily volume)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MarketImpactModel:
    """
    Calculates execution slippage, square-root market impact, and transaction costs.
    """

    def __init__(
        self,
        y_coeff: float = 0.30,
        fixed_commission_bps: float = 2.0,
        half_spread_bps: float = 3.0,
        max_participation_rate: float = 0.05
    ):
        self.y_coeff = y_coeff
        self.fixed_commission_bps = fixed_commission_bps
        self.half_spread_bps = half_spread_bps
        self.max_participation_rate = max_participation_rate

    def calculate_cost(
        self,
        trade_dollar_value: float,
        adv_dollar: float,
        daily_volatility: float = 0.02
    ) -> Dict[str, float]:
        """
        Computes the total transaction cost for an individual trade.

        Parameters
        ----------
        trade_dollar_value : float
            Dollar value of the rebalance trade Q = |w_new - w_old| * Portfolio_AUM.
        adv_dollar : float
            Average daily dollar volume.
        daily_volatility : float
            Daily standard deviation of returns.

        Returns
        -------
        Dict with total cost, impact bps, and liquidity limit violation flags.
        """
        trade_val = abs(float(trade_dollar_value))
        if trade_val <= 0:
            return {
                'total_cost_dollar': 0.0,
                'total_cost_bps': 0.0,
                'market_impact_bps': 0.0,
                'participation_pct': 0.0,
                'liquidity_breached': False
            }

        adv = max(float(adv_dollar), 10_000.0)
        participation = trade_val / adv
        liquidity_breached = participation > self.max_participation_rate

        # Fixed commission and spread in bps
        linear_cost_bps = self.fixed_commission_bps + self.half_spread_bps

        # Non-linear square root market impact
        # Impact as a decimal = Y * sigma * sqrt(Q / ADV)
        impact_decimal = self.y_coeff * daily_volatility * np.sqrt(min(participation, 1.0))
        market_impact_bps = impact_decimal * 10_000.0

        total_cost_bps = linear_cost_bps + market_impact_bps
        total_cost_dollar = trade_val * (total_cost_bps / 10_000.0)

        return {
            'total_cost_dollar': float(total_cost_dollar),
            'total_cost_bps': float(total_cost_bps),
            'market_impact_bps': float(market_impact_bps),
            'participation_pct': float(participation),
            'liquidity_breached': bool(liquidity_breached)
        }
