import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AdvisoryEngine:
    """
    Translates raw alpha scores and risk metrics into actionable 
    investment horizons, allocations, and exit triggers for human analysts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def _determine_dominant_factor(self, scores: Dict[str, Any]) -> str:
        """Determines which alpha model is driving the thesis."""
        m_scores = {
            'Quality (M1)': scores.get('m1_fundamentals', 50),
            'Valuation (M2)': scores.get('m2_valuation', 50),
            'Momentum (M3)': scores.get('m3_timeseries', 50),
            'Factor Beta (M4)': scores.get('m4_factors', 50),
            'Stat-Arb Reversion (M9)': scores.get('m9_stat_arb', 50)
        }
        best_model = max(m_scores, key=m_scores.get)
        if m_scores[best_model] < 60:
            return "Mixed / Passive Beta"
        return best_model

    def _estimate_horizon(self, dominant_factor: str) -> str:
        """Assigns an investment horizon based on the thesis driver."""
        if 'Stat-Arb' in dominant_factor:
            return "1 to 4 Weeks (Mean Reversion)"
        elif 'Momentum' in dominant_factor or 'Factor Beta' in dominant_factor:
            return "3 to 6 Months (Regime & Trend)"
        elif 'Quality' in dominant_factor or 'Valuation' in dominant_factor:
            return "12 to 36 Months (Fundamental Value Convergence)"
        else:
            return "6 to 12 Months (General Beta)"

    def _generate_invalidation_criteria(self, dominant_factor: str, risk_metrics: Dict[str, Any]) -> str:
        """Generates stop-loss and thesis invalidation logic."""
        cvar = risk_metrics.get('cvar_95', 0.05)
        stop_pct = min(max(cvar * 2.5, 0.05), 0.25) * 100
        
        base_rule = f"Hard Stop-Loss: Exit if position drops > {stop_pct:.1f}% from entry."
        
        if 'Stat-Arb' in dominant_factor:
            return f"{base_rule} Thesis Invalidation: Z-score reverts to 0 or spread breaks structural cointegration."
        elif 'Momentum' in dominant_factor:
            return f"{base_rule} Thesis Invalidation: 50-day moving average crosses below 200-day moving average."
        elif 'Valuation' in dominant_factor:
            return f"{base_rule} Thesis Invalidation: Operating margins degrade structurally, breaking DCF assumptions."
        else:
            return base_rule

    def generate_advisory_report(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        capital: float,
        latest_prices: Dict[str, float],
        multi_factor_scores: Dict[str, Dict[str, float]],
        risk_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds the comprehensive advisory report for a custom basket.
        """
        logger.info(f"Generating Quantitative Advisory Report for {len(tickers)} assets.")
        
        report = {
            'total_capital': capital,
            'assets': {}
        }
        
        for t in tickers:
            weight = weights.get(t, 0.0)
            if weight <= 0.001:
                continue
                
            alloc_dollars = capital * weight
            price = latest_prices.get(t, 1.0)
            shares = int(alloc_dollars / price) if price > 0 else 0
            
            scores = multi_factor_scores.get(t, {})
            rm = risk_metrics.get(t, {})
            
            dominant_factor = self._determine_dominant_factor(scores)
            horizon = self._estimate_horizon(dominant_factor)
            invalidation = self._generate_invalidation_criteria(dominant_factor, rm)
            
            report['assets'][t] = {
                'allocation_pct': weight * 100,
                'allocation_usd': alloc_dollars,
                'target_shares': shares,
                'current_price': price,
                'dominant_factor': dominant_factor,
                'recommended_horizon': horizon,
                'invalidation_criteria': invalidation,
                'cvar_95': rm.get('cvar_95', 0.0) * 100
            }
            
        return report
