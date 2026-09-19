"""
Pre-Trade Risk Management & Safety Checks Engine
=================================================
PURPOSE:
  Validate every target order before submission against institutional
  hard risk limits.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PreTradeRiskEngine:
    """
    Validates portfolio allocations and orders against concentration, leverage,
    and liquidity limits.
    """

    def __init__(
        self,
        max_position_weight: float = 0.40,
        max_leverage: float = 1.0,
        max_turnover_per_rebalance: float = 0.60
    ):
        self.max_position_weight = max_position_weight
        self.max_leverage = max_leverage
        self.max_turnover = max_turnover_per_rebalance

    def validate_target_weights(
        self,
        target_weights: pd.Series,
        current_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Validates target weights against pre-trade risk rules.
        """
        w = target_weights.fillna(0.0)
        violations = []

        # 1. Position limit check
        max_w = float(w.max()) if len(w) > 0 else 0.0
        if max_w > self.max_position_weight + 1e-4:
            violations.append(f"Position concentration breached: Max weight is {max_w:.1%}, limit is {self.max_position_weight:.1%}")

        # 2. Leverage check (gross exposure)
        gross_exposure = float(w.abs().sum())
        if gross_exposure > self.max_leverage + 1e-4:
            violations.append(f"Leverage limit breached: Gross exposure is {gross_exposure:.1%}, limit is {self.max_leverage:.1%}")

        # 3. Turnover check if current weights provided
        if current_weights is not None:
            c = current_weights.reindex(w.index).fillna(0.0)
            turnover = float((w - c).abs().sum() / 2.0)
            if turnover > self.max_turnover:
                violations.append(f"Turnover threshold breached: Rebalance turnover is {turnover:.1%}, limit is {self.max_turnover:.1%}")

        is_passed = len(violations) == 0
        return {
            'passed': is_passed,
            'violations': violations,
            'gross_exposure': gross_exposure,
            'max_weight': max_w
        }
