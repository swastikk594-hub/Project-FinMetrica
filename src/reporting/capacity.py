"""
AUM Capacity Analysis & Alpha Decay Curve Engine
=================================================
PURPOSE:
  Evaluate strategy performance, slippage, and net alpha decay across
  scaling capital tiers ($100K, $1M, $10M, $50M, $100M, $500M, $1B).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

from src.execution.market_impact import MarketImpactModel

logger = logging.getLogger(__name__)

DEFAULT_CAPITAL_TIERS = [
    100_000.0,
    1_000_000.0,
    10_000_000.0,
    50_000_000.0,
    100_000_000.0,
    500_000_000.0,
    1_000_000_000.0
]


class CapacityAnalysisEngine:
    """
    Simulates the degradation of alpha and Sharpe ratio as AUM scales.
    """

    def __init__(self, impact_model: Optional[MarketImpactModel] = None):
        self.impact_model = impact_model or MarketImpactModel()

    def generate_capacity_curve(
        self,
        gross_cagr: float,
        gross_vol: float,
        annual_turnover: float,
        universe_median_adv: float = 50_000_000.0,
        capital_tiers: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Computes net CAGR, market impact bps, and net Sharpe ratio across AUM tiers.
        """
        tiers = capital_tiers or DEFAULT_CAPITAL_TIERS
        rows = []

        for aum in tiers:
            # Average rebalance trade size in dollars
            annual_trade_volume = aum * annual_turnover
            # Assuming monthly rebalancing (12 chunks)
            chunk_size = annual_trade_volume / 12.0

            # Calculate market impact for this chunk
            daily_vol = gross_vol / np.sqrt(252.0)
            cost_res = self.impact_model.calculate_cost(chunk_size, universe_median_adv, daily_vol)

            impact_bps = cost_res['total_cost_bps']
            annual_drag_dollars = annual_trade_volume * (impact_bps / 10_000.0)
            annual_drag_pct = annual_drag_dollars / aum

            net_cagr = gross_cagr - annual_drag_pct
            net_sharpe = net_cagr / gross_vol if gross_vol > 0 else 0.0

            rows.append({
                'aum': aum,
                'aum_formatted': f"${aum:,.0f}",
                'gross_cagr': gross_cagr,
                'annual_drag_pct': annual_drag_pct,
                'net_cagr': net_cagr,
                'net_sharpe': net_sharpe,
                'market_impact_bps': impact_bps,
                'liquidity_breached': cost_res['liquidity_breached']
            })

        return pd.DataFrame(rows)
