"""
Multi-Tier Emergency Kill Switch Architecture
==============================================
PURPOSE:
  Halt trading and force portfolio liquidation / risk reduction upon
  breaching catastrophic daily loss or drawdown limits.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class KillSwitchManager:
    """
    Monitors intraday and portfolio-level risk limits to trigger emergency kill switches.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,      # 5% daily loss limit
        max_portfolio_drawdown_pct: float = 0.25 # 25% max cumulative drawdown limit
    ):
        self.max_daily_loss = max_daily_loss_pct
        self.max_drawdown = max_portfolio_drawdown_pct
        self.is_tripped = False
        self.trip_reason = ""

    def check_status(
        self,
        daily_pnl_pct: float,
        cumulative_drawdown_pct: float
    ) -> Dict[str, Any]:
        """
        Evaluates current risk state against emergency shutdown criteria.
        """
        if self.is_tripped:
            return {'tripped': True, 'action': 'HALT_TRADING_LIQUIDATE', 'reason': self.trip_reason}

        if daily_pnl_pct < -self.max_daily_loss:
            self.is_tripped = True
            self.trip_reason = f"MAX_DAILY_LOSS breached: {daily_pnl_pct:.2%} < -{self.max_daily_loss:.2%}"
            logger.critical(f"EMERGENCY KILL SWITCH TRIPPED: {self.trip_reason}")
            return {'tripped': True, 'action': 'HALT_TRADING', 'reason': self.trip_reason}

        if abs(cumulative_drawdown_pct) > self.max_drawdown:
            self.is_tripped = True
            self.trip_reason = f"MAX_PORTFOLIO_DRAWDOWN breached: {cumulative_drawdown_pct:.2%} > {self.max_drawdown:.2%}"
            logger.critical(f"EMERGENCY KILL SWITCH TRIPPED: {self.trip_reason}")
            return {'tripped': True, 'action': 'HALT_TRADING_LIQUIDATE', 'reason': self.trip_reason}

        return {'tripped': False, 'action': 'PROCEED', 'reason': 'All risk limits normal'}

    def manual_override_kill(self, reason: str = "Manual Admin Override") -> None:
        """Manually trigger global kill switch."""
        self.is_tripped = True
        self.trip_reason = f"GLOBAL_KILL_SWITCH: {reason}"
        logger.critical(f"MANUAL KILL SWITCH ACTIVATED: {reason}")
